# prefect_worker_refactor

A refactor of `mlex_prefect_worker` along the lines Tanny suggested: **keep
Prefect for orchestration, replace the parent worker with a FastAPI endpoint,
and run inference as its own service** so PyTorch stays out of app backends and
Prefect flows. The design rationale and open questions are in
[docs/DESIGN.md](docs/DESIGN.md).

**Old vs. new architecture, with diagrams:** [ARCHITECTURE.md](ARCHITECTURE.md)

![New architecture](docs/images/architecture_new.png)

```
app backend ──HTTP──► job_api (FastAPI) ──► Prefect ──► worker/ child flows ──► docker / conda / slurm / NERSC (SFAPI · IRI)
app backend ──HTTP──► inference_service (FastAPI + PyTorch) ──► MLflow registry
```

## Layout

| Path | What | PyTorch? |
|---|---|---|
| `job_api/` | FastAPI **Job Submission Endpoint**. Replaces `parent_flow.py` and `start_parent_worker.sh` | no |
| `worker/` | Prefect child workers (docker, podman, conda, slurm, **sfapi**), `prefect.yaml`, start scripts | no |
| `worker/flows/chain.py` | Runs multi-step jobs without a parent flow | no |
| `inference_service/` | **Inference Endpoint** (models from the MLflow registry, LRU cache) plus an MLflow-server option | yes |
| `examples/annotation_app_client.py` | Fine-tune on NERSC → inference, then interactive inference | no |
| `tests/` | Routing, chaining, Job API and inference tests (Prefect and MLflow mocked) | – |

### What changed compared to `mlex_prefect_worker`

* **Removed:** `flows/parent_flow.py`, `start_parent_worker*.sh`, the `parent_pool` deployment.
* **Moved:** the parent flow's routing went to `job_api/routing.py`, and `config.yml` went to `job_api/config.yml`.
* **Child flows** take a new `next_steps` argument and share `prepare_io_parameters()`. They also no longer crash when `io_parameters` is missing.
* **Added:** the `sfapi` worker (SFAPI or IRI API login, from the `sfapi_worker` package).
* **MLflow lookup** uses `search_model_versions` and an optional `model_version` instead of the deprecated `get_latest_versions`.
* **Dependencies** are split into extras (`api`, `worker`, `inference`), so only the inference service installs PyTorch.

## Quick start

```bash
conda create -n mlex python=3.11 && conda activate mlex
pip install -e ".[api,worker,dev]"        # inference: pip install -e ".[inference]" (or use the Dockerfile)
```

**1. Child workers** (each in its own terminal, or the `*_background.sh` variants):

```bash
cd worker
cp .env.example .env                      # Prefect URL, credentials, NERSC login
./start_docker_child_worker.sh            # ALS
./start_sfapi_child_worker.sh             # NERSC (SFAPI or IRI API, see job_api/config.yml sfapi.login_method)
```

**2. Job API** (instead of `start_parent_worker.sh`):

```bash
cp job_api/.env.example job_api/.env      # PREFECT_API_URL, MLFLOW_*, JOB_API_KEYS
./job_api/start_job_api.sh                # http://localhost:8090/docs
```

**3. Submit a job:**

```bash
curl -X POST localhost:8090/api/v1/jobs -H "X-API-Key: $KEY" -H 'Content-Type: application/json' -d '{
  "target": "nersc",
  "steps": [
    {"model_name": "DLSIA MSDNet", "task_name": "train", "params": {"io_parameters": {"data_tiled_uri": "..."}}},
    {"model_name": "DLSIA MSDNet", "task_name": "inference", "params": {"io_parameters": {}}}
  ]}'
# -> {"job_id": "3f9c...", "first_flow_run_id": "...", "steps": [...], "status_url": ".../api/v1/jobs/3f9c..."}

curl -H "X-API-Key: $KEY" localhost:8090/api/v1/jobs/3f9c...     # queued | running | completed | failed | cancelled
curl -X DELETE -H "X-API-Key: $KEY" localhost:8090/api/v1/jobs/3f9c...
```

The old parent-flow payload `{"params_list": [...]}` is also accepted.

**4. Inference service** (GPU host):

```bash
docker build -f inference_service/Dockerfile -t mlex-inference .
docker run --gpus all --env-file inference_service/.env -p 8091:8091 mlex-inference
# or a single model with MLflow's server:
./inference_service/serve_with_mlflow.sh "DLSIA MSDNet" 3 5001
```

## Job API reference

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/jobs` | Body: `steps[]` (`model_name`, `task_name`, optional `model_version`, `resources`, `params`), optional `target` (one per job, so all steps use one child work pool), `job_name`. Returns 201 with `job_id`. 404: unknown model. 422: invalid step or target. 502: Prefect unreachable. |
| GET | `/api/v1/jobs/{job_id}` | Overall status, per-step Prefect state, and `result_uid` (the last completed step's `uid_save`) |
| DELETE | `/api/v1/jobs/{job_id}` | Cancels the job's unfinished flow runs |
| GET | `/health`, `/ready` | Liveness; `/ready` also checks the Prefect API |

Auth uses the `X-API-Key` header, with keys set in `JOB_API_KEYS="name:key,..."`.
The key's name is attached to every run as the tag `mlex-client:<name>`.

## Tests

```bash
pytest            # 29 tests; Prefect, MLflow and NERSC are mocked
```

## Known limitations

* Cancelling a job doesn't cancel a NERSC job that was already submitted (see DESIGN.md, open question 4).
* `job_api/config.yml` values such as `container.volumes` are expanded on the Job API host, but they describe paths on the worker host.
* The SFAPI and IRI paths have only been tested against mocked NERSC responses. Run a `debug` QOS job before relying on them.
