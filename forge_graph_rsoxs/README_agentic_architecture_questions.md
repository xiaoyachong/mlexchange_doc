# Agentic AI platform: questions to explore

From the Agentic AI demo meeting, 16 Oct 2026. Each answer is a **proposed starting point**, not a team decision. "Today" refers to the shared SPLASH code (`splash-forge`, commit `ebf2ba2e`).

**Agreed:** keep the current prototype running on one machine; research alternatives to storing everything in Tiled in parallel. **Not decided:** the final storage architecture (not expected before February).

---

## 1. Knowledge graph

**Q1. Where should scientific knowledge live: in Tiled, or in a dedicated graph database such as Neo4j?**
Today: entities and links are tables in Tiled's catalog database, queried over Tiled's `/api/graphql` (ADR-002; Alex's graph covers ~1,600 ALS scattering papers).
Answer: benchmark Neo4j (Cypher, graph algorithms, mature tooling) against the current Tiled graph on real agent queries (multi-hop paper → material → sample → simulation). Move only if Neo4j is clearly faster or simpler; keep Tiled node IDs as the link to datasets either way.

**Q2. How do we model concepts, datasets, publications and analysis outputs?**
Answer: one small shared schema: node types (Paper, Material, Sample, Dataset, Simulation, Fit, Analysis) and typed edges (`measured_on`, `cites`, `derived_from`, `simulates`). Datasets are referenced by Tiled path or URI, never copied into the graph. Version the schema; `packages/scattering_schema` is a natural home.

**Q3. How do agents retrieve knowledge efficiently?**
Answer: graph lookups for exact relationships, plus vector search over paper text for fuzzy questions, both behind a few stable tools (`kg_find_entities`, `kg_get_sample_context`, …). Measure with a fixed question set.

**Q4. Should scientific knowledge and agent execution traces share one graph?**
Answer: no. Keep them separate: knowledge changes slowly and is curated; traces are high-volume logs. Link them by ID (a trace step that produced a Fit points to the Fit node).

---

## 2. Storage: what goes where

**Q5. What should stay in Tiled?**
Answer: scientific data and its metadata: raw frames, reduced data (Q-maps, I(q)), catalog structure and search. Tiled is built for arrays, tables and metadata.

**Q6. What should move out of Tiled?**
Answer: agent reasoning history, generated code, run artifacts and project documents. The prototype uses Tiled for some of these; they belong in a trace/artifact store (Q8) and project storage (Q14).

**Q7. Which system owns which information?**

| Information | Proposed owner |
|---|---|
| Datasets, metadata, reduced data | Tiled |
| Scientific knowledge and relationships | Graph database (Tiled graph today; maybe Neo4j) |
| Agent and tool versions, traces, generated code, run artifacts | MLflow |
| Project documents, notes, shared context | Project storage (Q14) |
| Live conversation state | LangGraph checkpointer (SQLite today) |

Every record carries the same **project ID** and stable references (Tiled URI, MLflow run ID, graph node ID).

---

## 3. MLflow

**Q8. Can MLflow register and version agents and tools?**
Today: no MLflow code in the repo; Alex said agents were prepared for registration but not integrated. Tools are Python functions in a code registry (106 tools, 19 skills).
Answer: prototype one agent and two or three tools in the MLflow Model Registry. Register versioned "agent bundles" (prompt + tool list + model ID) rather than every function.

**Q9. Can MLflow capture reasoning traces and tool calls?**
Today: optional Langfuse tracing (`langfuse_tracer.py`), LangGraph checkpoints (`.checkpoint.db`) and LLM call logs (`.run/llm_calls.sqlite3`).
Answer: test MLflow Tracing with LangGraph autologging on one real run and compare it with Langfuse on detail, UI and self-hosting. Pick **one** tracing backend.

**Q10. Can MLflow store generated code and outputs, linked to projects and datasets?**
Answer: yes, as run artifacts. Log each executed code block, its output and figures; tag the run with project ID and Tiled dataset URIs. This is the most direct win from the meeting.

**Q11. Does that make analyses reproducible?**
Answer: only together with versions. Record code commit, agent bundle version, model ID, tool versions and input dataset URIs on every run. Test by re-running a logged analysis on a second machine.

---

## 4. Agent tools and execution

**Q12. How should tools be registered, discovered and reused?**
Answer: keep tool code in the repo (tested, reviewed), but publish a **tool manifest** (name, version, schema, description) to a registry (MLflow or a simple service) so agents and other projects can discover and pin versions. Avoid storing executable code in a database.

**Q13. Is code execution safe enough?**
Today: generated Python runs in a Podman sandbox, with admin-key approval for code runs and writes.
Answer: add per-project resource limits, no network by default, read-only Tiled access with scoped keys, and an audit log of every execution in the trace store (Q9). Writes to shared data only through approved tools.

---

## 5. Project sharing

**Q14. How can two researchers open the same project and continue with its context?**
Today: projects are workspaces inside one local install; sharing means copying the whole folder (the 44 GB `splash-share` bundle).
Answer: a project is a record (ID, members, permissions) that **points to** data in Tiled, knowledge in the graph, runs in MLflow and documents in project storage. Nothing is copied. Prototype: two users, one shared server, same project ID, both see the datasets, history and artifacts.

**Q15. Where do project files and permissions live?**
Answer: evaluate an existing system before building one: a shared filesystem or object store (S3/MinIO) for files, plus the lab's identity system for permissions, mirrored into Tiled access policies, MLflow experiments and graph access.

---

## 6. Suggested order

1. **MLflow:** Q8–Q11, one agent end to end (register, trace, log code and artifacts, retrieve, re-run).
2. **Knowledge store:** Q1–Q4, benchmark the current Tiled graph against Neo4j on real queries.
3. **Integration:** Q5–Q7, one project ID and one reference format across Tiled, MLflow and the graph.
4. **Sharing:** Q14–Q15, two users on one project without copying folders.

Throughout: keep the current demo running unchanged (one machine) while these are tested.
