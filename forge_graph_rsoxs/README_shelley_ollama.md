# Use the Ollama model on Shelley from forge_rsoxs_analysis on your Mac

An SSH tunnel makes Shelley's Ollama appear on your Mac as if it were running locally.

## 1. Stop your local Ollama

Quit the menu-bar app, or stop `ollama serve`. The tunnel uses the same port, 11434.

## 2. Open the tunnel

In its own terminal, and leave it running:

```bash
ssh -N -o ServerAliveInterval=60 -L 11434:localhost:11434 xchong@shelley.als.lbl.gov
```

- Password: LBNL password + token. `-N` opens no shell; it only forwards traffic.
- If Ollama on Shelley uses a port other than 11434, change the second number. Check on Shelley with `ss -ltnp | grep ollama` or `ps aux | grep ollama`.
- You may need to be on the LBL network or VPN.

## 3. Check it works

From another terminal on your Mac:

```bash
curl http://127.0.0.1:11434/api/tags
```

You should see Shelley's models listed.

## 4. Restart the app

Ctrl+C, then `./start_all.sh`. No `.env` change is needed: the launcher turns `OLLAMA_BASE_URL` into `127.0.0.1:11434`, and the Agent's Ollama client connects there by default.

## 5. Pick the model

On the **Config** page: **Agent provider → ollama**, press **Refresh models**, choose Shelley's large model.

## Notes

- **Why port 11434:** the Agent's Ollama client doesn't read `OLLAMA_BASE_URL`; it uses `127.0.0.1:11434` unless `OLLAMA_HOST` is set. To keep your local Ollama running too, forward to another port (e.g. `-L 11435:localhost:11434`) and set both `OLLAMA_BASE_URL=http://127.0.0.1:11435` and `OLLAMA_HOST=127.0.0.1:11435` in the shell before `./start_all.sh`.
- **Vision panels and the knowledge-graph pipeline** also use Shelley now. Make sure `qwen3-vl:8b` and `qwen3-embedding:0.6b` exist there, or set `VISION_MODEL` / `VISION_EMBEDDING_MODEL` in `.env` to models Shelley has.
- **Context length is set on Shelley's side.** Ask whoever runs Ollama there whether it was started with a large `OLLAMA_CONTEXT_LENGTH`, or the Agent's tool list may get cut off.
- **If the tunnel drops** (e.g. your Mac sleeps), the Agent stops responding. Rerun the `ssh` command.
