# Use the vLLM model on Shelley from forge_rsoxs_analysis on your Mac

Shelley (`shelley.als.lbl.gov`) runs **vLLM**, not Ollama:

- Model: **`glm-5.3-flash`** (`--served-model-name`), really **GLM-5.3-Flash-NVFP4**: ~320B-parameter mixture-of-experts (~15–20B active per token), 4-bit, multimodal, 181 GB on disk — on 4 × H200 with tensor + expert parallelism (details below)
- Port: **8000**, OpenAI-compatible API at `/v1`
- No `--api-key`, so no key is needed
- Tool calling enabled (`--enable-auto-tool-choice --tool-call-parser glm47`), context up to 750,000 tokens
- Started by `ahexemer`; default reasoning effort is `max`, so answers can be slow

## 1. Open the tunnel

In its own terminal on your Mac, and leave it running:

```bash
ssh -N -o ServerAliveInterval=60 -L 8000:localhost:8000 xchong@shelley.als.lbl.gov
```

Password: LBNL password + token. You may need the LBL network or VPN. Your local Ollama can keep running (it uses 11434).

## 2. Check it works

```bash
curl -s http://127.0.0.1:8000/v1/models
```

You should see `glm-5.3-flash`.

## 3. Restart the app

Ctrl+C, then `./start_all.sh`. No `.env` change is needed: the app's default `VLLM_BASE_URL` is already `http://localhost:8000/v1`.

## 4. Pick the model

On the **Config** page: **Agent provider → vllm**, press **Refresh models**, choose `glm-5.3-flash`.

## Notes

- The vision panels and the knowledge-graph pipeline keep using your local Ollama (`qwen3-vl:8b`, `qwen3-embedding:0.6b`).
- If the tunnel drops (e.g. your Mac sleeps), the Agent stops responding. Rerun the `ssh` command.
- This is a shared server; check with ahexemer before running heavy batch jobs against it.

## First-time login: find the running model

```bash
ssh xchong@shelley.als.lbl.gov          # LBNL password + token; "Could not chdir to home directory" is harmless

nvidia-smi                               # Memory-Usage per GPU + Processes table (PIDs)
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

GPU_PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -1)
ps -o pid,user,etime,cmd -p "$GPU_PID"                        # the GPU worker
ps -o pid,user,etime,cmd -p $(ps -o ppid= -p "$GPU_PID")      # its parent

ps aux | grep -E '[v]llm serve|[o]llama serve|[l]lama-server|[t]ext-generation|[s]glang'   # launch command: model + port
ss -ltn                                  # listening ports
curl -s http://127.0.0.1:8000/v1/models  # vLLM / OpenAI-style; Ollama: curl -s http://127.0.0.1:11434/api/tags
exit
```

## Worked example (real session, 16 Oct 2026)

### Step 1: `nvidia-smi` — is a model loaded?

```text
xchong@shelley:/$ nvidia-smi
|   0  NVIDIA H200 NVL                Off |   00000000:0B:00.0 Off |                    0 |
| N/A   51C    P0            111W /  600W |  131569MiB / 143771MiB |      0%      Default |
|   1  NVIDIA H200 NVL                Off |   00000000:33:00.0 Off |                    0 |
| N/A   58C    P0            112W /  600W |  131083MiB / 143771MiB |      0%      Default |
|   2  NVIDIA H200 NVL                Off |   00000000:8B:00.0 Off |                    0 |
| N/A   51C    P0            105W /  600W |  131083MiB / 143771MiB |      0%      Default |
|   3  NVIDIA H200 NVL                Off |   00000000:B4:00.0 Off |                    0 |
| N/A   48C    P0            101W /  600W |  131083MiB / 143771MiB |      0%      Default |
...
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|    0   N/A  N/A         2296067      C   VLLM::Worker_TP0_EP0                  13155... |
|    1   N/A  N/A         2296068      C   VLLM::Worker_TP1_EP1                  13107... |
|    2   N/A  N/A         2296069      C   VLLM::Worker_TP2_EP2                  13107... |
|    3   N/A  N/A         2296070      C   VLLM::Worker_TP3_EP3                  13107... |
```

How to read it:

- **Memory-Usage** ~131 GB of 143 GB on each of the 4 GPUs → something big is loaded across all four. (vLLM also reserves memory for requests, `--gpu-memory-utilization 0.90`, so this is not the model's exact size.)
- **GPU-Util 0%** → idle at that moment; the model stays loaded between requests.
- **Processes** → four `VLLM::Worker` processes, one per GPU: it's **vLLM**. `TP0…TP3` = model split across 4 GPUs (tensor parallel); `EP` = expert parallel (mixture-of-experts model).
- Any worker PID will do for the next step; here **2296067** (GPU 0).

### Step 2: the worker's parent — who started it?

```text
xchong@shelley:/$ ps -o pid,user,etime,cmd -p $(ps -o ppid= -p 2296067)
    PID USER         ELAPSED CMD
2295805 ahexemer  9-15:20:38 VLLM::EngineCore
```

Started by **ahexemer**, running for 9 days 15 hours. Workers don't show the model or port, so find the launch command next.

### Step 3: the launch command — model and port

```text
xchong@shelley:/$ ps aux | grep -E '[v]llm serve|[v]llm.entrypoints'
ahexemer 2295145 ... /usr/local/bin/vllm serve /model --served-model-name glm-5.3-flash
  --tensor-parallel-size 4 --reasoning-parser glm45 --tool-call-parser glm47
  --enable-auto-tool-choice --max-model-len 750000 --max-num-seqs 16
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 --trust-remote-code
  --kv-cache-dtype bfloat16 --enable-expert-parallel
  --default-chat-template-kwargs {"reasoning_effort":"max"} --host 0.0.0.0 --port 8000
```

(Wrapped here for reading; it is one line.)

| Flag | Meaning |
|---|---|
| `--served-model-name glm-5.3-flash` | the name to use in requests and in the app |
| `--port 8000` | the port to tunnel |
| no `--api-key` | no key needed |
| `--enable-auto-tool-choice --tool-call-parser glm47` | supports tool calling (needed by the Agent) |
| `--max-model-len 750000` | very long context — the Agent's 106 tools fit easily |
| `--reasoning-parser glm45`, `reasoning_effort: max` | thinks before answering, returned separately as `reasoning_content`; slow but careful |
| `--max-num-seqs 16` | at most 16 requests at once — it's shared |

### Step 4: listening ports

```text
xchong@shelley:/$ ss -ltn
LISTEN  0  4096   127.0.0.1:8080   0.0.0.0:*
LISTEN  0  4096   127.0.0.1:8000   0.0.0.0:*     <- vLLM
LISTEN  0  64       0.0.0.0:8006   0.0.0.0:*
LISTEN  0  4096     0.0.0.0:22     0.0.0.0:*     <- SSH
LISTEN  0  4096           *:9100         *:*     <- node_exporter (monitoring)
...
```

**8000** is listening, matching the launch command.

### Step 5: ask it a question

```text
xchong@shelley:/$ curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"In one sentence, what is GIWAXS?"}],"max_tokens":2000}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
GIWAXS (grazing-incidence wide-angle X-ray scattering) is an X-ray diffraction technique in which a beam strikes a thin-film sample at a very shallow angle to probe its crystal structure, molecular packing, and orientation with enhanced surface sensitivity.
```

The model answers. Next: open the tunnel from your Mac (section 1) and pick `glm-5.3-flash` in the app.

## The model in detail

### Where it lives

```text
xchong@shelley:/$ grep -E ' /model ' /proc/2295145/mountinfo
644 706 0:50 /GLM-5.3-Flash-NVFP4 /model ro,noatime - zfs llm-data rw,...
xchong@shelley:/$ cat /proc/2295145/cgroup
0::/user.slice/user-44518.slice/user@44518.service/user.slice/libpod-9b144f53....scope/container
xchong@shelley:/$ findmnt -t zfs | grep llm-data
/llm-data llm-data zfs    rw,noatime,xattr,posixacl,casesensitive
```

- vLLM runs inside a **rootless Podman container** (`libpod`) owned by ahexemer (UID 44518) — that's why `/model` doesn't exist on the host and `docker ps` shows nothing.
- `/model` in the container is **`/llm-data/GLM-5.3-Flash-NVFP4`** on Shelley (ZFS pool `llm-data`, mounted read-only).

### Size on disk

```text
xchong@shelley:/$ du -sh /llm-data/GLM-5.3-Flash-NVFP4
181G    /llm-data/GLM-5.3-Flash-NVFP4
```

33 `model-000NN-of-00033.safetensors` shards plus `config.json`, `hf_quant_config.json`, tokenizer files, `chat_template.jinja`, `processor_config.json` and a `README.md` (check it for the official numbers: `head -60 /llm-data/GLM-5.3-Flash-NVFP4/README.md`).

### What `config.json` says

| Field | Value | Meaning |
|---|---|---|
| `architectures` | `Glm5NextForConditionalGeneration` | GLM-5 family; "ConditionalGeneration" + `vision_config` = **multimodal** (text, images, video) |
| `num_hidden_layers` | 45 | 3 dense layers, then 42 mixture-of-experts layers |
| `hidden_size` | 4096 | |
| `n_routed_experts` / `num_experts_per_tok` | 288 / 8 | each token uses 8 of 288 experts (+ 1 shared expert) |
| `moe_intermediate_size` | 2048 | size of each expert |
| `layer_types` | 34 × `linear_attention`, 11 × `deepseek_sparse_attention` | hybrid attention: cheap linear layers with a full sparse-attention layer every 4th — why a 750k context fits |
| `max_position_embeddings` | 1,048,576 | trained up to ~1M tokens; served with `--max-model-len 750000` |
| `num_nextn_predict_layers` | 1 | extra multi-token-prediction layer (speculative decoding) |
| `vision_config` | 24-layer, 448-px, patch 14 | built-in image/video encoder |
| `quant_algo` | `NVFP4` (modelopt) | experts in 4-bit float (group size 16); attention, shared experts, embeddings, output head and vision kept in bf16 |

### Estimated size (worked out from config.json — not an official figure)

- **Total parameters ≈ 320 billion**, of which ~305B are the routed experts (288 experts × 42 layers × 3 × 4096 × 2048).
- **Active per token ≈ 15–20 billion** (8 + 1 experts per layer, plus attention and embeddings) — so it runs at roughly the speed of a ~20B model while having the knowledge capacity of a much larger one.
- **Cross-check against disk:** ~306B expert weights at NVFP4 (~4.5 bits incl. scales) ≈ 172 GB, plus ~10B bf16 parameters ≈ 20 GB → ~190 GB ≈ the 181 GiB `du` reports.
- **GPU memory:** ~181 GB of weights spread over 4 × H200 (~45 GB each); the rest of the ~131 GB per GPU is vLLM's reserved KV cache for long contexts (`--gpu-memory-utilization 0.90`).

NVFP4 is a native format on NVIDIA Blackwell GPUs; on these H200s (Hopper) vLLM runs it with software kernels — it works, but isn't as fast as on Blackwell.

## Tutorial: talk to the model directly

Works on Shelley itself, or on your Mac while the tunnel is open (same URL: `http://127.0.0.1:8000`).

### 1. One question, one answer (curl)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "glm-5.3-flash",
        "messages": [{"role": "user", "content": "In one sentence, what is GIWAXS?"}],
        "max_tokens": 2000
      }' | python3 -m json.tool
```

The answer is in `choices[0].message.content`. Because the server uses a reasoning parser, the model's thinking appears separately in `choices[0].message.reasoning_content`. To print only the answer:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"In one sentence, what is GIWAXS?"}],"max_tokens":2000}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

Reasoning effort defaults to `max`, so a reply can take a while. If `content` comes back empty, raise `max_tokens` — the thinking used up the budget.

### 2. Watch the answer stream in

```bash
curl -N -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.3-flash","stream":true,"messages":[{"role":"user","content":"List three uses of RSoXS."}],"max_tokens":3000}'
```

Text arrives as `data: {...}` lines while the model generates.

### 3. A small interactive chat (Python)

On your Mac: `pip install openai`, then save as `chat_shelley.py` and run `python chat_shelley.py`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
history = []

while True:
    question = input("you> ").strip()
    if question in ("", "exit", "quit"):
        break
    history.append({"role": "user", "content": question})
    reply = client.chat.completions.create(
        model="glm-5.3-flash", messages=history, max_tokens=4000
    )
    answer = reply.choices[0].message.content
    print("glm>", answer, "\n")
    history.append({"role": "assistant", "content": answer})
```

Type a question, press Enter, read the answer; it remembers the conversation until you type `exit`.
