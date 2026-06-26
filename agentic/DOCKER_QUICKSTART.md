# FORGE · Docker Quickstart

How to run the full stack with Docker Compose (replaces `start_all.sh`).

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running
- A `.env` file in the repo root (copy from `env.example`)

---

## First-time setup

### 1. Configure environment

```bash
cp env.example .env
# Edit .env — at minimum set TILED_API_KEY to any secret string
```

### 2. Build images

```bash
docker compose build
```

### 3. Initialize the Tiled catalog database

This only needs to be run once (or after wiping `catalog.db`).

```bash
docker compose run --rm tiled tiled catalog init /catalog/catalog.db
```

### 4. Start the stack

```bash
docker compose up -d
```

Verify all services are healthy:

```bash
docker compose ps
```

Tiled must show `healthy` before you can ingest data. It may take ~30 seconds on first start.

### 5. Ingest data

Run from the repo root with your virtual environment active:

```bash
source .venv/bin/activate   # or: source agent_venv/bin/activate
source .env                  # load TILED_API_KEY into shell

python scripts/ingest_ten_k_h5_to_tiled.py \
  --source /path/to/ten_k_Samples \
  --tiled-uri "http://127.0.0.1:8010" \
  --api-key "${TILED_API_KEY}" \
  --workers 4
```

---

## Daily use

| Task | Command |
|------|---------|
| Start stack | `docker compose up -d` |
| Stop stack | `docker compose down` |
| View logs | `docker compose logs -f` |
| View one service | `docker compose logs -f tiled` |
| Rebuild after code changes | `docker compose build && docker compose up -d` |

---

## Services

| Service | URL | Description |
|---------|-----|-------------|
| Tiled | http://127.0.0.1:8010 | Scientific data catalog |
| API | http://127.0.0.1:8002 | FastAPI browse/agentic backend |
| GIWAXS | http://127.0.0.1:8003 | GIWAXS/SAXS reduction API |
| Agent | http://127.0.0.1:8004 | ALS Knowledge Agent |
| Finch | http://127.0.0.1:5173 | React frontend (AI Data Assistant) |

The MCP server is off by default. Start it with:

```bash
docker compose --profile mcp up -d
```

---

## Reset and re-ingest

To wipe the catalog and start over:

```bash
docker compose down
rm -f catalog.db catalog.db-wal catalog.db-shm
rm -rf data/tiled_write && mkdir -p data/tiled_write

docker compose run --rm tiled tiled catalog init /catalog/catalog.db
docker compose up -d
# then re-run the ingest command above
```

---

## Persistent data

| Path (host) | Mounted as | What lives there |
|-------------|------------|-----------------|
| `./catalog.db` | `/catalog/catalog.db` | Tiled catalog index |
| `./data/` | `/catalog/data` | Raw HDF5 data registered by Tiled |
| `./gixsgui_api/data/` | `/app/gixsgui_api/data/` | GIWAXS database + thumbnails + cache |
| `./als_knowledge_agent/data/` | `/app/als_knowledge_agent/data/` | Knowledge graph JSON |

---

## Troubleshooting

**Tiled stays unhealthy**
Check logs: `docker compose logs tiled --tail=40`
Common cause: `catalog.db` missing → run the `tiled catalog init` command in step 3.

**GIWAXS fails with `unable to open database file`**
The `giwaxs.db` must be inside `gixsgui_api/data/` (not the repo root).
Copy it: `cp gixsgui_api/giwaxs.db gixsgui_api/data/giwaxs.db`

**Connection refused on port 8010 during ingest**
Tiled isn't ready yet. Run `docker compose ps` and wait for `healthy`, then retry.

**`502` errors from a service**
An upstream service isn't ready. Check `docker compose logs` for the failing container.
