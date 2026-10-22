# MLflow, Neo4j and LangGraph/LangChain — how to use them in FORGE

This is a usage guide to go with `README_agentic_architecture_questions.md`. It covers what each tool does, the minimal commands and code, and where each would plug into the current repo.

> No real keys or passwords appear here. `<password>` and similar values are placeholders, so keep the real ones in `.env`.

## What FORGE uses today

| Need | Today in the repo | Candidate |
|---|---|---|
| LLM / agent traces | Langfuse (optional, `src/forge/agents/langfuse_tracer.py`), plus `.run/llm_calls.sqlite3` | **MLflow Tracing** |
| Prompt and model versions | Prompts live in code | **MLflow Prompt / Model Registry** |
| Agent conversation state | LangGraph SQLite checkpointer (`src/forge/agents/checkpointer.py`, `.checkpoint.db`) | **PostgresSaver**, or Neo4jSaver |
| Long-term memory | none | **LangGraph Store** (Postgres, with semantic search) |
| Knowledge graph | Tables in Tiled's catalog DB, served over `/api/graphql` (ADR-002) | **Neo4j** (benchmark it first) |
| Data | Tiled | keep Tiled |

---

## 1. MLflow

### 1.1 Run a tracking server

```bash
pip install "mlflow>=3.6"
mlflow server --host 127.0.0.1 --port 5000
```

- Since 3.7 the backend defaults to SQLite (`mlflow.db`). To use Postgres, pass `--backend-store-uri postgresql://user:<password>@host/mlflow`.
- To store artifacts elsewhere, add `--artifacts-destination s3://bucket/path`. A local folder is the default.
- To expose the server beyond localhost, add `--host 0.0.0.0 --allowed-hosts <host> --cors-allowed-origins http://127.0.0.1:5173`, and put auth in front of it (`--app-name basic-auth` or SSO).
- For a shared deployment, the MLflow docs provide a Docker Compose stack (MLflow, PostgreSQL and S3-compatible storage) and a Helm chart.

The UI is at http://127.0.0.1:5000.

### 1.2 Trace LangGraph / LangChain agents (one line)

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("forge-agent")
mlflow.langchain.autolog()          # traces every graph/chain/LLM/tool call
# async graphs: mlflow.langchain.autolog(run_tracer_inline=True)

result = graph.invoke(
    {"messages": [("user", "integrate sample X")]},
    config={"configurable": {"thread_id": "session-123"}},
)
```

- Each `invoke` becomes one trace. It shows the span tree (node → LLM → tool), inputs and outputs, latency and **token usage per span**.
- In MLflow 3.6+, traces that share a `thread_id` are grouped into a **session**, so one chat appears as one conversation.
- If you call the vLLM endpoint through the raw `openai` client rather than LangChain, also add `mlflow.openai.autolog()`.

Custom spans for non-LLM work such as reduction or Tiled reads:

```python
with mlflow.start_span(name="tiled_read") as span:
    span.set_inputs({"uri": uri})
    arr = client[uri].read()
    span.set_outputs({"shape": list(arr.shape)})
```

Or decorate a function:

```python
@mlflow.trace(span_type="TOOL")
def reduce_saxs(uri: str): ...
```

Tags make traces searchable by beamline, sample or user:

```python
mlflow.update_current_trace(tags={"beamline": "11.0.1.2", "sample": "OPV-03"})
```

### 1.3 Prompt Registry (versioned system prompts)

```python
p = mlflow.genai.register_prompt(
    name="forge-system",
    template="You are the FORGE agent for {{beamline}}. ...",
    commit_message="add tool-use rules",
)
prompt = mlflow.genai.load_prompt("prompts:/forge-system/3")   # or @alias
text = prompt.format(beamline="11.0.1.2")
# LangChain templates use single braces:
from langchain_core.prompts import ChatPromptTemplate
tmpl = ChatPromptTemplate.from_template(prompt.to_single_brace_format())
```

Traces link back to the prompt version that produced them, which makes A/B comparison straightforward.

### 1.4 Log the agent itself (models-from-code)

```python
# agent_app.py
from mlflow.models import set_model
graph = build_graph()          # your compiled LangGraph
set_model(graph)
```

```python
info = mlflow.langchain.log_model(lc_model="agent_app.py", name="forge-agent",
                                  prompts=["prompts:/forge-system/3"])
agent = mlflow.langchain.load_model(info.model_uri)
```

- Register the logged model with `mlflow.register_model(info.model_uri, "forge-agent")`.
- This also works for classic ML models such as the VAE: `mlflow.pytorch.log_model(...)`.

### 1.5 Evaluate

`mlflow.genai.evaluate(data=..., predict_fn=..., scorers=[...])` runs a fixed question set against the agent and scores the results, with LLM-judge or custom scorers. Use it to compare models (for example glm-5.3-flash against others) or prompt versions.

### 1.6 Where it fits FORGE

1. Start `mlflow server` next to `start_all.sh`, and add `MLFLOW_TRACKING_URI` to `.env`.
2. Call `mlflow.langchain.autolog()` once, at agentic-gateway startup.
3. Reuse the existing `thread_id` from the checkpointer, so MLflow sessions match chat sessions.
4. Keep Langfuse optional, or replace it. Running both doubles the overhead.

---

## 2. Neo4j

### 2.1 Run it locally

```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/<password> \
  -v $HOME/neo4j/data:/data neo4j:5
pip install neo4j langchain-neo4j
```

- The browser UI is at http://localhost:7474.
- Bolt is at `bolt://localhost:7687`.
- With podman, replace `docker` with `podman`.

### 2.2 Python driver (plain Cypher)

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "<password>"))
driver.verify_connectivity()

driver.execute_query(
    """MERGE (s:Sample {id:$sid})
       MERGE (m:Measurement {uri:$uri})
       MERGE (s)-[:MEASURED_IN]->(m)""",
    sid="OPV-03", uri="tiled://raw/…/scan_0042", database_="neo4j",
)

records, summary, keys = driver.execute_query(
    "MATCH (s:Sample)-[:MEASURED_IN]->(m) WHERE s.id=$sid RETURN m.uri AS uri",
    sid="OPV-03", database_="neo4j",
)
```

- Store **Tiled URIs** in graph nodes, not the arrays themselves. Neo4j holds the relationships, and Tiled stays the data store.
- Add constraints once, for example `CREATE CONSTRAINT FOR (s:Sample) REQUIRE s.id IS UNIQUE`.

### 2.3 LangChain integration

```python
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain, Neo4jVector

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="<password>")
graph.refresh_schema(); print(graph.schema)

qa = GraphCypherQAChain.from_llm(
    llm, graph=graph, verbose=True,
    allow_dangerous_requests=True,   # required flag: the LLM writes Cypher
    validate_cypher=True, top_k=20,
)
qa.invoke({"query": "Which samples share the same donor polymer?"})
```

- **Safety:** the LLM-generated Cypher runs with whatever rights the account has. Connect this chain with a **read-only Neo4j user**, and never use the admin account.
- `Neo4jVector` stores embeddings in a Neo4j vector index, for hybrid graph + vector search over papers or notes.
- `Neo4jChatMessageHistory` stores chat history as nodes.
- A safer alternative to free Cypher is to wrap a few fixed, parameterized Cypher queries as LangGraph **tools**. This matches how FORGE's tool registry already works.

### 2.4 Neo4j as a LangGraph checkpointer (optional)

```python
from langgraph.checkpoint.neo4j import Neo4jSaver   # pip install langgraph-checkpoint-neo4j
with Neo4jSaver.from_conn_string(uri="bolt://localhost:7687",
                                 user="neo4j", password="<password>") as cp:
    cp.setup()                      # once
    app = builder.compile(checkpointer=cp)
```

The async variant is `AsyncNeo4jSaver`. This is only worth doing if Neo4j is adopted anyway; otherwise Postgres (section 3) is simpler.

### 2.5 Where it fits FORGE

- Today the KG is tables in Tiled's catalog DB, served over GraphQL. Neo4j would mainly add **multi-hop queries** (sample → measurement → reduction → paper) and graph algorithms.
- Suggested test: export the current KG entities into Neo4j, then time the same five to ten real agent questions on each backend before deciding.

---

## 3. LangGraph / LangChain persistence

### 3.1 Checkpointers (short-term, per-thread memory)

Every run is keyed by a thread ID:

```python
config = {"configurable": {"thread_id": "session-123"}}
```

| Saver | Import | Use |
|---|---|---|
| `InMemorySaver` | `langgraph.checkpoint.memory` | tests |
| `SqliteSaver` / `AsyncSqliteSaver` | `langgraph.checkpoint.sqlite` | single user (what FORGE uses now) |
| `PostgresSaver` / `AsyncPostgresSaver` | `langgraph.checkpoint.postgres` (`pip install langgraph-checkpoint-postgres`) | shared / production |

```python
from langgraph.checkpoint.postgres import PostgresSaver
DB = "postgresql://forge:<password>@localhost:5432/forge?sslmode=disable"
with PostgresSaver.from_conn_string(DB) as cp:
    cp.setup()                      # creates tables, once
    app = builder.compile(checkpointer=cp)
    app.invoke({"messages": [("user", "hi")]}, config)
```

Keep `thread_id` under 255 characters on Postgres, and prune old threads periodically.

### 3.2 Inspect, replay and fork (time travel)

```python
state = app.get_state(config)                 # latest snapshot: .values, .next
for snap in app.get_state_history(config):    # newest → oldest
    print(snap.config["configurable"]["checkpoint_id"], snap.next)

# replay from an earlier checkpoint
old = {"configurable": {"thread_id": "session-123", "checkpoint_id": "<id>"}}
app.invoke(None, old)

# fork: edit state, then continue
new_cfg = app.update_state(old, {"messages": [("user", "use q-range 0.01–0.3")]})
app.invoke(None, new_cfg)
```

This is useful for debugging a bad tool call or re-running an analysis with different parameters. It also supports human-in-the-loop approval with `interrupt()`.

### 3.3 Stores (long-term memory across threads)

```python
from langgraph.store.postgres import PostgresStore
from langchain_openai import OpenAIEmbeddings   # or any embeddings endpoint

with PostgresStore.from_conn_string(
    DB, index={"dims": 1024, "embed": embeddings, "fields": ["text"]}
) as store:
    store.setup()
    app = builder.compile(checkpointer=cp, store=store)

    ns = ("user", "xchong", "notes")
    store.put(ns, "opv-q-range", {"text": "OPV films: use q 0.005–0.3 Å⁻¹"})
    hits = store.search(ns, query="q range for OPV", limit=3)
```

- Inside a node, add `store: BaseStore` to the node's signature and LangGraph injects it.
- Namespaces work well per user, per beamline or per project.
- `InMemoryStore` is available for tests.

### 3.4 Where it fits FORGE

1. Swap `SqliteSaver` for `PostgresSaver` in `checkpointer.py` when sessions must be shared or survive restarts on a server. The graph code does not change.
2. Add a `PostgresStore` for user and beamline preferences, and for "lessons learned" notes.
3. Use the same `thread_id` across the checkpointer, MLflow sessions and the UI chat, so one ID links the state, the trace and the conversation.

---

## 4. Suggested minimal stack

```
Tiled            → data (unchanged)
Postgres         → LangGraph checkpoints + store, MLflow backend
MLflow           → traces, prompts, models, evals
Neo4j (optional) → only if the KG benchmark shows a clear win
```

Postgres is the one new service that serves three roles. Neo4j is added only if the knowledge-graph benchmark justifies it.

## Sources

- MLflow LangGraph tracing: https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langgraph/
- MLflow LangChain flavor / models-from-code: https://mlflow.org/docs/latest/genai/flavors/langchain/
- MLflow Prompt Registry: https://mlflow.org/docs/latest/genai/prompt-registry/
- MLflow tracking server: https://mlflow.org/docs/latest/self-hosting/
- Neo4j Python driver: https://neo4j.com/docs/python-manual/current/
- LangChain Neo4j: https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/
- Neo4j LangGraph checkpointer: https://pypi.org/project/langgraph-checkpoint-neo4j/
- LangGraph persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph PostgresStore: https://reference.langchain.com/python/langgraph.store.postgres/base/PostgresStore
