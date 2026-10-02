# Design proposal: ML job submission for the Computing Hub

*Draft for discussion with Tanny, Dylan and Wiebke. Author: Xiaoya Chong.*

## Problem

Today MLExchange apps start work through a Prefect **parent flow**
(`launch_parent_flow`), which runs on its own worker, looks up the algorithm in
MLflow, picks a compute environment and then runs the child flows one by one.
Separately, some backends and flows import PyTorch directly to run inference,
which makes them heavy to build, deploy and scale.

## Proposal

Keep Prefect for orchestration. Move the request handling and the heavy compute
out of it, behind two dedicated endpoints:

```
 Frontend / app backend (annotation app, ...)      no PyTorch, no Prefect, no MLflow
        │                          │
        │ 1. jobs (train, batch    │ 2. interactive inference
        │    inference)            │    (one slice, low latency)
        ▼                          ▼
 ┌───────────────┐          ┌────────────────────┐
 │  Job API      │          │ Inference service  │  ◄── only component with PyTorch
 │  (FastAPI)    │          │ (FastAPI or MLflow │      (GPU host)
 │  replaces the │          │  model server)     │
 │  parent worker│          └─────────┬──────────┘
 └──────┬────────┘                    │ load model
        │ MLflow metadata             ▼
        │ + create flow run       MLflow registry
        ▼
   Prefect server ──► child workers (unchanged design)
                       docker · podman · conda · slurm · sfapi
                                                 │
                                                 ▼ SFAPI / IRI API
                                          NERSC Perlmutter (fine-tuning)
```

### 1. Training / Job Submission Endpoint (Job API)

* `POST /api/v1/jobs`, `GET /api/v1/jobs/{id}`, `DELETE /api/v1/jobs/{id}`.
* Does what the parent flow did (MLflow lookup, target → flow type, building and
  validating child parameters), but **inside the HTTP request**. A bad model
  name, a missing conda env or a target the client may not use comes back as a
  4xx immediately, instead of as a failed parent run later.
* Submits the first step to Prefect and returns a `job_id`. It does not wait
  for the job and holds no state: Prefect is the source of truth, and the API
  finds a job's runs by tag (`mlex-job:<id>`).
* Child parameters are validated with the **same schema classes** the workers
  use (`worker/flows/*/schema.py`), so the two cannot drift apart.
* The client chooses **one target per job** (`als`, `nersc`, ...), limited by
  `allowed_targets`. Every step of the job goes to that target's single child
  work pool, as in the old design. It can also send a few whitelisted resource overrides (`max_time`,
  `num_nodes`). Credentials never come from clients. Child flows add them on
  the worker, exactly as before.
* The old payload (`params_list`) is still accepted, so apps can switch from
  `run_deployment("Parent flow/...")` to one HTTP call without changing what they send.

**Multi-step jobs without a parent worker.** The Job API resolves *all* steps up
front and passes the remaining ones to the first child flow as `next_steps`.
When a child flow succeeds, it submits the next step with
`prev_flow_run_id = <its own run id>` (`worker/flows/chain.py`). That keeps the
existing `uid_save` → `uid_retrieve` contract between steps. A failed step
stops the chain. Submission uses an idempotency key, so a retried flow does not
submit the next step twice.

**Concrete example: SFAPI fine-tuning from the annotation app.** The annotation
app sends one request with `"target": "nersc"`: *fine-tune, then run inference*.
The Job API routes the job to `SFAPI flow/launch_sfapi`, so only `sfapi_pool`
is used and the docker pool is never involved. The sfapi worker can run anywhere
with HTTPS access to NERSC, logs in with SFAPI or the IRI API, runs the container
with `podman-hpc` and waits. If the job has a second step, that step runs on
`sfapi_pool` too. See `examples/annotation_app_client.py`.

### 2. Inference Endpoint

Two options, which can coexist:

| | A. `inference_service/` (FastAPI) | B. MLflow model server (`mlflow models serve`) |
|---|---|---|
| Models per process | many (LRU cache, `models:/<name>/<version>`) | one |
| Switch model / version per request | yes | no (one server per model) |
| Auth | API keys (same scheme as the Job API) | none, needs a proxy |
| Input format | JSON list or base64 `.npy` | MLflow scoring JSON (`/invocations`) |
| Custom code to maintain | small (about 200 lines) | none |
| Uses the model's logged environment | no (one shared image) | yes (`--env-manager`) |
| Good for | the annotation app trying several models | one production model with a fixed API |

Both load the model from the MLflow registry, so "fine-tune on NERSC → register
in MLflow → serve" needs no copying. Only interactive, small requests should go
here. Whole-volume inference stays a **batch job** through the Job API (for
example on NERSC), because an HTTP request should not run for minutes.

> MLflow's "AI Gateway / deployments server" is for proxying LLM providers. It
> is not a fit for serving our own PyTorch models; option B is `mlflow models serve`.

## Why this fits the Computing Hub

* **One standard mechanism.** Any app that can make an HTTP call can submit ML
  jobs. The facility-specific parts are config (targets) plus one child worker
  per compute resource. Adding a resource (ALCF, a new cluster) means adding a
  child flow and a target, with no app changes.
* **Lighter backends.** The app backend, the Job API and the Prefect workers
  contain no PyTorch. PyTorch lives in the algorithm images (run as jobs) and
  in the inference service.
* **Same Prefect and workers.** Child flows are unchanged apart from
  `next_steps`. Deployments, pools and start scripts work as before, minus the
  parent pool.

## Open questions for the discussion

1. **Ownership.** Should this live under ALS Computing / the Computing Hub
   instead of MLExchange? (The code no longer depends on MLExchange apps.)
2. **Auth.** API keys are a stopgap. Should we use the Hub's identity
   (OIDC / Globus Auth) and map users to allowed targets and NERSC accounts?
3. **NERSC identity.** Should jobs run under a collaboration account
   (`asldev`, as splash_flows does) or per user? This decides SFAPI vs. IRI and
   who owns the keys.
4. **Cancellation.** `DELETE /jobs/{id}` cancels the Prefect runs, but a NERSC
   job that has already been submitted keeps running. Should the sfapi flow add
   an `on_cancellation` hook that runs `scancel`?
5. **Chaining.** Is chaining in the workers enough, or do we want Prefect
   Automations / a DAG service for more complex pipelines (fan-out, retries per
   step)?
6. **Model serving.** Should option A or B be the default, and where does the
   GPU for interactive inference come from (ALS GPU node vs. NERSC Spin)?
7. **Results.** The API returns `result_uid` (the last step's `uid_save`).
   Should it also return the Tiled URIs of the outputs?
