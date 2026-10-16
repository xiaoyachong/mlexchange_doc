# Use Shelley's LLM as the backend for the forge_rsoxs_analysis app

Shelley (`shelley.als.lbl.gov`) serves **`glm-5.3-flash`** with vLLM on port **8000** (OpenAI-compatible, no API key, tool calling on, 750k context). This guide: check it on Shelley, test it from your Mac, then point the rsoxs app at it.

For how the model was found (`nvidia-smi`, `ps`, `ss`) see `README_shelley_vllm.md`.

---

## Part 1 — Check the model on Shelley

```bash
ssh xchong@shelley.als.lbl.gov
```

LBNL password + token. You may need the LBL network or VPN. "Could not chdir to home directory" is harmless.

On Shelley:

```bash
nvidia-smi
ps aux | grep -E '[v]llm serve'
curl -s http://127.0.0.1:8000/v1/models
```

- `nvidia-smi`: four `VLLM::Worker` processes holding ~131 GB per GPU → the model is loaded.
- `ps`: the launch command shows `--served-model-name glm-5.3-flash ... --port 8000`.
- `curl`: lists `glm-5.3-flash`.

If any of these comes back empty, the server isn't running — ask ahexemer (who started it). Then `exit`.

---

## Part 2 — Small test from your Mac

### 2.1 Open the tunnel

In its own terminal, leave it running:

```bash
ssh -N -o ServerAliveInterval=60 -L 8000:localhost:8000 xchong@shelley.als.lbl.gov
```

This makes Shelley's port 8000 appear as `127.0.0.1:8000` on your Mac. Your local Ollama (port 11434) can keep running.

### 2.2 Is it reachable?

In a second terminal:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

Expect JSON containing `"id": "glm-5.3-flash"`.

### 2.3 Ask a question

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"In one sentence, what is GIWAXS?"}],"max_tokens":2000}' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

Expected (real answer, 16 Oct 2026):

> GIWAXS (grazing-incidence wide-angle X-ray scattering) is an X-ray diffraction technique in which a beam strikes a thin-film sample at a very shallow angle to probe its crystal structure, molecular packing, and orientation with enhanced surface sensitivity.

- Replies can take a while: reasoning effort defaults to `max`.
- Empty answer → raise `max_tokens`; the thinking used up the budget.

### 2.4 Tool-calling test (what the app's Agent needs)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"glm-5.3-flash","max_tokens":2000,"messages":[{"role":"user","content":"What is the weather in Berkeley?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}' | python3 -c "import sys,json; m=json.load(sys.stdin)['choices'][0]['message']; print(m.get('tool_calls') or m.get('content'))"
```

Pass: it prints a `tool_calls` entry naming `get_weather` with `"city": "Berkeley"`.

### 2.5 Image test (decides whether it can replace the vision model)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"glm-5.3-flash","max_tokens":1000,"messages":[{"role":"user","content":[{"type":"text","text":"Describe this image."},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="}}]}]}'
```

- An answer → it accepts images.
- An error about images/multimodal → text only; keep `qwen3-vl:8b` on local Ollama for vision.

---

## Part 3 — Use it in the rsoxs app

Keep the tunnel from 2.1 open whenever the app uses Shelley.

### What goes where

| Part of the app | Where to set it | Value |
|---|---|---|
| **Agent / Scattering Agent** | Config page → LLM → Agent provider **vllm** → Refresh models | `glm-5.3-flash` |
| **Knowledge graph** (Build / Ingest / Paper · KG) | model dropdown on those pages | `vllm:glm-5.3-flash` |
| **Default for other KG steps** | `.env`: `OLLAMA_MODEL` | `vllm:glm-5.3-flash` |
| **Paper → simulation extraction** | `.env`: `EXTRACTION_PROVIDER` | `vllm` |
| **Vision panels** | `.env`: `VISION_MODEL` | `vllm:glm-5.3-flash` if 2.5 passed (its config.json has a vision encoder, so it likely will); else `qwen3-vl:8b` |
| **Text embeddings** | `.env`: `VISION_EMBEDDING_MODEL` | stays local: `qwen3-embedding:0.6b` |

- The `vllm:` prefix tells those parts of the app to call vLLM instead of Ollama.
- No `VLLM_BASE_URL` needed: the app's default is already `http://localhost:8000/v1`, where the tunnel puts it. No `VLLM_API_KEY` needed: the server has none.
- Embeddings go only to Ollama, so keep a small local Ollama for `qwen3-embedding:0.6b` (skip it if you don't use "embed analysis text").

### Steps

1. Back up `.env`: `cp .env .env.bak`
2. Edit `.env` in `forge_rsoxs_analysis`:
   ```bash
   OLLAMA_MODEL=vllm:glm-5.3-flash
   EXTRACTION_PROVIDER=vllm
   # VISION_MODEL=vllm:glm-5.3-flash      # only if 2.5 passed
   ```
3. Restart: Ctrl+C, then `./start_all.sh`.
4. Config page → LLM → **Agent provider: vllm** → **Refresh models** → `glm-5.3-flash`.
5. Agent page: the footer should read `Model: vllm / glm-5.3-flash`. Test with:
   > Simulate 5 nm gold spheres hexagonally packed with 12 nm spacing on silicon
   
   Pass: the Results panel shows a tool call (e.g. `sim_simulate_structure`), not `0`.

To go back to local models: `cp .env.bak .env`, restart, set Agent provider back to ollama.

---

## Troubleshooting

| Symptom | Check |
|---|---|
| `curl` to `127.0.0.1:8000` hangs or "connection refused" | Tunnel terminal still open? Rerun the `ssh -N -L ...` command (it drops when the Mac sleeps). |
| `ssh` asks again / fails | LBL network or VPN; password + token. |
| Refresh models shows nothing under vllm | Tunnel down, or app started before the tunnel — restart the app. |
| Agent answers but Results stays `0` | Footer still says `ollama`? Re-select vllm in Config. |
| Very slow answers | Expected with `reasoning_effort: max`; the server also takes at most 16 requests at once. |
| Vision panels error after switching | The server isn't accepting images (vLLM can be started with images disabled); set `VISION_MODEL=qwen3-vl:8b` again. |

Shelley is shared (started by ahexemer). Mention large knowledge-graph batch runs before starting them.
