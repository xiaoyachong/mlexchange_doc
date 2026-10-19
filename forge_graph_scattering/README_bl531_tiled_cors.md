# Connecting forge_Agentic_AI_Scattering to the BL531 Tiled server

The BL531 Tiled server is at **`http://131.243.80.226:8000`** (LBL network or VPN). This explains why the app is opened at `http://tiled-test:5173` and how to set up the repo to run against that server. Companion to `README.md` in this folder.

---

## Why `/etc/hosts` and CORS

### What CORS restricts

The server answers anyone. CORS restricts only one thing: **JavaScript on a web page from a different site** calling the server.

- **The Tiled UI works without any setup.** Opening `http://131.243.80.226:8000/ui/browse/...` loads a page *served by Tiled itself*, which then calls its own API on the same address. Browsers call that "same-origin"; CORS doesn't apply.
- **The forge app is a different origin.** Its page comes from your Mac (`http://localhost:5173` or `http://tiled-test:5173`). When JavaScript on that page calls `http://131.243.80.226:8000/api/v1/...`, the browser first asks the Tiled server whether pages from that origin are allowed. If the origin isn't on the server's allow-list, the browser blocks the response — the server still got the request; the browser refuses to hand the answer to the page.

| Who's asking | CORS applies? | Works? |
|---|---|---|
| You, in Tiled's own UI at `131.243.80.226:8000/ui/...` | No (same origin) | Yes |
| `curl`, Python, the forge backend containers | No (not a browser) | Yes |
| The forge page at `localhost:5173` calling Tiled from the browser | Yes | Only if `localhost:5173` is on the allow-list |
| The forge page at `tiled-test:5173` calling Tiled from the browser | Yes | Yes, if that's what the allow-list contains |

Most of the app goes through the forge backend (no CORS). Only the parts that call Tiled straight from the browser — e.g. the Data page — need the allowed origin.

### Why `/etc/hosts` maps `tiled-test` to 127.0.0.1

The server's allow-list contains `http://tiled-test:5173` (per the BL531 team). `tiled-test` isn't a real machine, so you give that name to **your own Mac**:

```
127.0.0.1   tiled-test
```

Then `http://tiled-test:5173` opens the app running on your Mac, and the browser reports its origin as `tiled-test:5173`, which the server accepts.

**Don't** map `131.243.80.226 tiled-test`: `http://tiled-test:5173` would then try to load the app from the beamline machine, which doesn't host it. The server's IP goes in `.env`, not in `/etc/hosts`.

### Check the allow-list yourself

```bash
curl -s -I -H "Origin: http://tiled-test:5173" http://131.243.80.226:8000/api/v1/ | grep -i access-control
curl -s -I -H "Origin: http://localhost:5173"  http://131.243.80.226:8000/api/v1/ | grep -i access-control
```

An `access-control-allow-origin: ...` line for an origin means that origin is allowed. If `localhost:5173` is allowed too, you don't need the `/etc/hosts` entry.

### Access

The Tiled UI opens without logging in, so the server allows anonymous **reading**. An API key is needed only to **write** processed results back (`processed/<sample_id>`).

---

## Set up the repo to run the app

All in `~/Documents/3agentic_scattering/forge_Agentic_AI_Scattering`. Nothing tracked in git needs editing — only `.env` (git-ignored) and `/etc/hosts`.

### 1. Switch to the BL531 branch

```bash
cd ~/Documents/3agentic_scattering/forge_Agentic_AI_Scattering
docker compose down
git fetch upstream
git switch -c process-bl531-tiled upstream/process-bl531-tiled
```

(Already have the branch? `git switch process-bl531-tiled && git pull`.) This branch has no local Tiled or ingest service — it uses the remote server.

### 2. Check the server is reachable

```bash
curl -s http://131.243.80.226:8000/api/v1/
```

JSON back = reachable. No answer → connect to the LBL network or VPN.

### 3. Add the hosts entry (one time)

```bash
echo "127.0.0.1 tiled-test" | sudo tee -a /etc/hosts
```

### 4. Edit `.env`

```bash
cp .env .env.bak
```

Then set:

```bash
TILED_URI=http://131.243.80.226:8000
TILED_API_KEY=<write key, if they gave you one; otherwise leave empty for read-only>
VITE_API_TILED_URL=http://131.243.80.226:8000
__VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS=tiled-test
```

- `TILED_URI` / `TILED_API_KEY`: used by every backend container.
- `VITE_API_TILED_URL`: used by the UI parts that call Tiled from the browser.
- `__VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS`: the Vite 6 dev server rejects unknown hostnames ("Blocked request. This host is not allowed"); this allows `tiled-test`.

### 5. Start the app

```bash
docker compose up -d --build
```

Open **http://tiled-test:5173** (not `localhost:5173`).

### 6. Check it works

- Browse / Select tabs list samples from the BL531 server.
- Browser console (Developer Tools → Console): no `CORS` errors.

---

## Notes

- **Port 8000 twice is fine.** The Shelley vLLM tunnel uses port 8000 on *your Mac* (`127.0.0.1:8000`); the Tiled server is port 8000 on *another machine* (`131.243.80.226`). No clash.
- **Open question — GIWAXS API CORS.** `gixsgui_api/app/main.py` only allows `localhost` / `127.0.0.1` origins. With the page at `tiled-test:5173`, the browser may block GIWAXS tools (Process, Viewer…) on port 8003. If you see a CORS error mentioning `:8003`, raise it with Bowen — the fix is a one-line regex change in that file.
- **Go back to the local-catalog setup:** `cp .env.bak .env`, `git switch main`, `docker compose up -d --build`.
