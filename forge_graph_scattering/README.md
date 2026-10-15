# forge_Agentic_AI_Scattering — running the `process-bl531-tiled` branch

This branch does **not** start a local Tiled. It connects to the ALS beamline 5.3.1 Tiled server instead. It is `upstream/main` plus 17 commits (BL531 work by Bowen Zheng, PR #6).

## What the branch expects

- **Tiled server address:** `TILED_URI` in `.env`, with `TILED_API_KEY` for the key. The branch removes the local `tiled` and `ingest` services from `docker-compose.yml`, so every backend container reads these from `.env`.
- **Several servers (optional):** `TILED_SERVER_1_NAME`, `TILED_SERVER_1_URI`, `TILED_SERVER_1_API_KEY`, then `_2_`, and so on, for the UI's server picker.
- **Writing processed data:** Q-maps and features are written back to the BL531 server under `processed/<sample_id>`, so a **write API key** is needed. Search hides those `processed` entries.
- **No ingest step:** the data already lives on the 531 server.

## Before Monday (no IP needed yet)

```bash
cd ~/Documents/3agentic_scattering/forge_Agentic_AI_Scattering
docker compose down
git fetch upstream
git switch -c process-bl531-tiled upstream/process-bl531-tiled

# CORS alias: makes "tiled-test" mean your own machine
echo "127.0.0.1 tiled-test" | sudo tee -a /etc/hosts
```

Later updates: `git switch process-bl531-tiled && git pull`. Back to the local-catalog setup: `git switch main`.

## On Monday, once you have the IP and key

Edit `.env`:

```bash
TILED_URI=http://<531-IP>:<port>
TILED_API_KEY=<write key>
VITE_API_TILED_URL=http://<531-IP>:<port>
__VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS=tiled-test
```

- `VITE_API_TILED_URL` is for the few UI parts that call Tiled directly from the browser (e.g. the Data page).
- `__VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS` stops Vite 6.4 rejecting the `tiled-test` hostname ("Blocked request. This host is not allowed").

Check the server is reachable (probably needs the LBL network or VPN):

```bash
curl -H "Authorization: Apikey <key>" http://<531-IP>:<port>/api/v1/
```

Start the app:

```bash
docker compose up -d --build
```

Open **http://tiled-test:5173**, not localhost. Tiled only accepts browser requests from that address.

## Open question for Monday

The GIWAXS API's CORS rule (`gixsgui_api/app/main.py`) only allows `localhost` and `127.0.0.1`. With the page opened at `tiled-test:5173`, the browser may block the GIWAXS tools (Process, Viewer, …) on port 8003. The other two backends accept any origin. Ask whether anyone runs it this way successfully; the fix is a one-line regex change adding `tiled-test`.

## Monday checklist

- [ ] 531 Tiled server IP and port
- [ ] Write API key
- [ ] Server's CORS list includes `http://tiled-test:5173`
- [ ] Answer on the GIWAXS API CORS question
