# Docker Networking Notes

When running under Docker Compose, services communicate via Docker's internal
network using **service names** as hostnames — not `localhost`. `localhost`
inside a container refers to that container itself, not the host machine or
another service.

## Environment variables overridden in docker-compose.yml

| Variable | Dev default (local) | Docker value | Why |
|----------|---------------------|--------------|-----|
| `TILED_URI` | `http://localhost:8010` | `http://tiled:8010` | All services reach Tiled by service name |
| `GIWAXS_API_URL` | `http://localhost:8003` | `http://giwaxs:8003` | `api` calls GIWAXS by service name |
| `API_SERVER_URL` | `http://localhost:8002` | `http://api:8002` | Internal service-to-service calls |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | `http://host.docker.internal:11434` | Ollama runs on the host, not in Docker |
| `SPLASH_LINKS_URL` | `http://localhost:8080` | `http://host.docker.internal:8080` | splash_links runs on the host |

## Where the localhost defaults live in the code

The Python services fall back to `localhost` when env vars are not set — correct
for local dev but broken inside Docker:

- `src/tiled_agentic/_config.py` — `GIWAXS_API_URL` defaults to `http://localhost:8003`
- `src/tiled_agentic/routers/companion.py` — `GIWAXS_API_URL` and `API_BASE_URL` default to `localhost`
- `src/tiled_agentic/tiled_config.py` — `TILED_URI` / `TILED_API_KEY` defaults

The `docker-compose.yml` overrides all of these via the `environment:` block on
each service, so no source code changes are needed when switching between local
and Docker.

## Resetting the catalog

If you delete `catalog.db` (or the `data/` folder), re-initialize before starting:

```bash
# 1. Init the catalog schema
docker compose run --rm tiled tiled catalog init sqlite+aiosqlite:////catalog/catalog.db

# 2. Start the stack
docker compose up -d

# 3. Re-ingest data
INGEST_SOURCE=<your-data-path> \
  docker compose run --rm ingest
```

Also clear the GIWAXS DB if you want a full reset (otherwise the samples list
in the UI will still show stale entries from the old DB):

```bash
rm gixsgui_api/data/giwaxs.db
rm -rf gixsgui_api/data/qmaps gixsgui_api/data/thumbnails gixsgui_api/data/tiled_cache
docker compose restart giwaxs
```
