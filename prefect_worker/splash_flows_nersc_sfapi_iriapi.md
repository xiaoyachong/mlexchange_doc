# How splash_flows runs jobs at NERSC: SFAPI and IRI API

Summary of the NERSC job-submission code in `splash_flows` (BL 8.3.2), as of
commit `79a6d4f` (2026-08-25). It covers how the two login methods work, how a
job goes from a Prefect flow to Perlmutter and back, and the gaps worth knowing
about before reusing the approach (e.g. in the MLExchange SFAPI worker).

## 1. The big picture

splash_flows does not run a worker at NERSC. A Prefect flow runs on an ALS
machine and drives Perlmutter **remotely over HTTPS**, using one of two NERSC
APIs:

| | **SFAPI** (Superfacility API) | **IRI API** (Integrated Research Infrastructure) |
|---|---|---|
| Base URL | `https://api.nersc.gov/api/v1.2` | `https://api.iri.nersc.gov` |
| Client | `sfapi_client.Client` (official NERSC library) | plain `httpx.Client` with a Bearer token |
| Credential | Iris client ID and private key (JWK), created for user **`alsdev`** | Globus Auth token with the `iri_api` scope |
| Env vars | `PATH_NERSC_CLIENT_ID`, `PATH_NERSC_PRI_KEY` | `PATH_GLOBUS_TOKEN_FILE` (default `~/.globus/auth_tokens.json`), `NERSC_USERNAME` |
| Status in splash_flows | **Default**, used in production | Optional; used by `nersc_recon_test_iriapi_flow` and manual runs |

The choice between them is the enum `NERSCLoginMethod` (`SFAPI` / `IRIAPI`) in
`orchestration/flows/bl832/job_controller.py`.

```
Prefect flow (ALS)                      NERSC
──────────────────                      ─────
nersc_recon_flow(login_method)
   │
   ├─ get_controller(HPC.NERSC, login_method)
   │     └─ create_nersc_client() ──► SFAPI client  or  IRI httpx client
   │
   ├─ controller.reconstruct()
   │     ├─ build Slurm script (f-string)
   │     ├─ _submit_job(script)  ─────────────► Slurm on Perlmutter
   │     ├─ _wait_for_job(id)    ◄── poll ────  job state
   │     └─ _read_remote_file()  ◄────────────  timing file on $PSCRATCH
   │
   ├─ globus_transfer_task()  pscratch ──► CFS / data832
   ├─ register_file_to_tiled()
   └─ schedule_pruning()
```

## 2. Where the code lives

| File | Role |
|---|---|
| `orchestration/flows/bl832/job_controller.py` | `NERSCLoginMethod` enum, `HPC` enum, abstract `TomographyHPCController`, and the `get_controller()` factory |
| `orchestration/flows/bl832/nersc.py` | `NERSCTomographyHPCController`: client creation plus the four API "primitives" (submit, wait, mkdir, read file) implemented for **both** methods; the Slurm scripts; and the Prefect flows (`nersc_recon_flow`, `nersc_petiole_segment_flow`, `nersc_moon_segment_flow`, `nersc_recon_test_iriapi_flow`, …) |
| `orchestration/globus/get_globus_token.py` | Globus login and token refresh for the IRI API (CLI and the `get_iri_access_token()` function) |
| `orchestration/flows/bl832/streaming_mixin.py` | `nersc_streaming_flow`: a long-running GPU streaming job, **SFAPI only**, with a cancellation hook |
| `orchestration/flows/bl832/dispatcher.py` | Starts the NERSC flows after a new file arrives (does not pass `login_method`, so SFAPI is used) |
| `scripts/cancel_sfapi_job.py` | CLI to list and cancel Slurm jobs through SFAPI |
| `config.yml` → `nersc_resources` | API base URLs and IRI resource UUIDs |
| `config.yml` → `hpc_submission_settings832` | QOS, account, reservation, nodes, CPUs and walltime per job type |
| `docs/mkdocs/docs/nersc832.md` | How to create the SFAPI client in Iris |

## 3. How the login method is chosen

1. **Flow parameter.** Each NERSC flow has `login_method: NERSCLoginMethod = NERSCLoginMethod.SFAPI`.
2. **Controller factory.** `get_controller(HPC.NERSC, config, login_method)` falls back to SFAPI if
   the value isn't a `NERSCLoginMethod`. It then calls
   `NERSCTomographyHPCController.create_nersc_client(config, login_method)` and builds the
   controller with that client.
3. **Constructor.** The controller stores `self.login_method` and selects
   `config.nersc_resources["sfapi"]` or `["iri"]`.
4. **Every primitive.** Each primitive (`_submit_job`, `_wait_for_job`, `_mkdir_remote`,
   `_read_remote_file`, `_get_nersc_username`, the Shifter image check) branches on
   `self.login_method`. The job logic above them (recon, multires, SAM3, DINOv3,
   combine) is written once and works with either method.

## 4. SFAPI path

### Authentication

`_create_sfapi_client()`:

1. Reads `PATH_NERSC_CLIENT_ID` (a text file) and `PATH_NERSC_PRI_KEY` (a JWK JSON file). It raises
   `ValueError` or `FileNotFoundError` if either is missing.
2. Calls `JsonWebKey.import_key(...)` (authlib), then `sfapi_client.Client(client_id, key)`.
   The library handles the OAuth2 client-credentials flow (a signed JWT assertion) and token refresh.

How the client is created in Iris (from `docs/mkdocs/docs/nersc832.md`):

- **Where:** Profile → *Superfacility API Clients* → *New Client*.
- **User:** **`alsdev`**, the collaboration account. Without it, the key can't access the data.
- **Security level:** **Red**, which is required to submit jobs.
- **IP ranges:** the ALS host's IP and the Perlmutter nodes.

### Operations

| Primitive | SFAPI call |
|---|---|
| username | `client.user().name` |
| submit | `client.compute(Machine.perlmutter).submit_job(script)` → `job.jobid` |
| wait | `perlmutter.job(jobid=…)` then `job.complete()` (polls until a terminal state) |
| mkdir | `perlmutter.run("mkdir -p …")` (synchronous utility command) |
| read file | `perlmutter.run("cat …")` |
| Shifter image check | `perlmutter.run("shifterimg images | grep …")` |
| cancel | `perlmutter.job(jobid=…).cancel()` (streaming hook and `cancel_sfapi_job.py`) |

## 5. IRI API path

### Authentication (`get_globus_token.py`)

1. **Login type:** a Globus **native app** login with a public client ID baked into the script.
   The requested scopes are `openid profile email view_identities` plus the IRI scope
   `https://auth.globus.org/scopes/ed3e577d-…/iri_api`.
2. **One-time interactive login:** a person runs
   `python orchestration/globus/get_globus_token.py --validate-iri`, opens the URL, logs in and
   pastes the code. Tokens, including **refresh tokens**, are saved to `~/.globus/auth_tokens.json`
   with mode `0600` (atomic write, safe when several processes write at once).
3. **At runtime:** `_create_iriapi_client()` calls
   `get_iri_access_token(token_file, force_login=False, prompt_login=False)`, which:
   - returns the cached access token if it expires in more than 60 s;
   - otherwise refreshes it with the stored refresh token and saves the new one;
   - if refreshing fails, **falls back to interactive login** (`input()`).
4. **Client:** `httpx.Client(base_url=api_base_url, headers={"Authorization": "Bearer <token>"})`,
   with a 120 s read timeout.
5. **Username:** IRI has no user lookup in this code, so `NERSC_USERNAME` must be set to build the
   `/pscratch` paths.

### Operations

Calls go to resource UUIDs from `config.yml → nersc_resources.iri`:

| Primitive | IRI call |
|---|---|
| submit | `POST /api/v1/compute/job/{perlmutter_job_submit}` with a PSI/J-style job spec (below) → `id` |
| wait | `GET /api/v1/compute/status/{compute_resource}/{job_id}` every 60 s. `completed` → True; `failed`, `canceled` or `timeout` → False |
| mkdir | `POST /api/v1/filesystem/mkdir/{perlmutter_login}` with `{"path", "parents": true}` |
| read file | `GET /api/v1/filesystem/view/{perlmutter_login}?path=…` returns a `task_id`. The code then polls `GET /api/v1/task/{task_id}` (3 s × up to 40) and reads `result.output.content` |
| Shifter image check | No synchronous command is available, so it **submits a 5-minute debug job** that writes `shifterimg images` to a file, waits, then reads the file |

**From `#SBATCH` script to IRI job spec.** The job builders always write a normal Slurm script.
For IRI, `_submit_job` parses it:

- **Lines it understands:** `-q`, `-A`, `--time`, `-N`, `-C`, `--output`, `--error` and
  `--reservation`.
- **Translation:** `queue_name`, `account`, `duration` (seconds), `node_count`, `constraint`,
  `stdout_path`, `stderr_path` and `reservation_id`.
- **The script body:** it removes the `#SBATCH` lines and the shebang and sends the rest as
  `pre_launch`, with `executable: /bin/bash` and `arguments: ["-s"]`.
- **Fixed resources:** `processes_per_node: 1`, `exclusive_node_use: true`, and either
  `cpu_cores_per_process: 128` or `gpu_cores_per_process: 4` (GPU when the constraint contains `gpu`).

## 6. What runs on Perlmutter

The job scripts are f-strings in `nersc.py`. The reconstruction job, for example:

1. **Slurm header:** `#SBATCH -q realtime -A als -C cpu -N <n> --exclusive --image=<recon image>`.
   The values come from `hpc_submission_settings832`.
2. **Timing:** writes a timing file `timing_$SLURM_JOB_ID.txt` (JOB_START, COPY_*, METADATA_*,
   RECON_*, JOB_STATUS).
3. **Copy input:** copies the raw `.h5` from CFS (`alsdev` raw path) to `$PSCRATCH`, unless it's
   already there.
4. **Plan the split:** reads the number of slices with `shifter python -c "import h5py…"`.
5. **Run:** starts one `srun … shifter --image=… python /opt/scripts/sfapi_reconstruction_multinode.py`
   per node, each on its own range of sinogram slices, then `wait`s for all of them.

Afterwards the controller reads the timing file back (`_read_remote_file`) and the flow logs a
timing breakdown.

**Containers:** batch jobs use **Shifter** (`--image=`, pre-staged by `pull_shifter_image_flow`).
The streaming job uses **podman-hpc**.

**Job options have three layers** (`_load_job_options`; later layers win):

1. **Defaults:** `config.yml` (`hpc_submission_settings832`).
2. **MLflow registry** (segmentation only): `inference_params` tags and `nersc_path`, which gives the
   checkpoint location on NERSC.
3. **Prefect Variable** (for example `nersc-reconstruction-options`): runtime overrides, applied only
   when it contains `"defaults": false`.

## 7. End to end: `nersc_recon_flow`

1. The **dispatcher** starts `nersc_recon_flow/nersc_recon_flow` with `file_path`, `num_nodes` and
   `config`. This uses SFAPI by default.
2. `get_controller(...)` creates the client (§3).
3. `controller.reconstruct()` builds the script, submits it, sleeps 60 s, then waits and reads the
   timing data.
4. `globus_transfer_task` copies the TIFFs from pscratch to CFS and data832.
5. `controller.build_multi_resolution()` submits a second Slurm job (Zarr pyramid), followed by
   Globus transfers of the Zarr.
6. `register_file_to_tiled()` registers the TIFF and Zarr outputs in Tiled.
7. `schedule_pruning()` schedules cleanup of the scratch copies.

Data moves with **Globus**, not through the API. The API only submits and monitors jobs and does
small file operations.

## 8. The streaming flow (SFAPI only)

`streaming_mixin.py → nersc_streaming_flow`:

- **Job:** submits a realtime GPU job (`podman-hpc run --gpu --network=host als-computing/streamtomocupy-recon`).
- **Monitoring:** checks the job every `monitor_interval` seconds (`JobSacct.update()`), retries
  "Job not found" up to 5 times, and stores `{job_id, state, elapsed, timelimit}` in a Prefect JSON
  block `<flow_run_id>-metadata`.
- **Cancellation:** `on_cancellation=[cancellation_hook]` reads the job ID from that block and calls
  `job.cancel()` through SFAPI. This is the only place splash_flows cancels the NERSC job when a
  Prefect run is cancelled.

## 9. Setup checklist

**SFAPI**

1. In Iris, create a **Red** SFAPI client for user **`alsdev`**, restricted to the worker host's IP.
2. Save the client ID and private key on the host (`chmod 600`), and set `PATH_NERSC_CLIENT_ID` and
   `PATH_NERSC_PRI_KEY` in `.env`.
3. Watch for key expiry and the IP allowlist when hosts change.

**IRI API**

1. Run `python orchestration/globus/get_globus_token.py --validate-iri` once on the worker host,
   logging in as the NERSC-linked identity.
2. Set `NERSC_USERNAME`, and `PATH_GLOBUS_TOKEN_FILE` if the token isn't in the default location.
3. Check that the resource UUIDs in `config.yml → nersc_resources.iri` are still valid.

**Both**

- Pre-stage the container images with `pull_shifter_image_flow`.
- Make sure the account and QOS in `hpc_submission_settings832` are allowed (`realtime` needs approval).

## 10. Observations and gaps

These are worth fixing in splash_flows, and avoiding when reusing the code:

1. **SFAPI `_wait_for_job` always returns True.** `job.complete()` returns the final state and does
   not raise on FAILED or TIMEOUT. The return value is ignored, so a failed reconstruction counts as
   success. Downstream, Globus transfers and Tiled registration then run on missing outputs. Fix:
   `return job.complete() == JobState.COMPLETED`. The IRI branch checks the state correctly.
2. **The IRI token can trigger an interactive login inside a flow.** If refreshing fails,
   `get_iri_access_token()` calls `input()`, which hangs or crashes a worker. The worker should
   refresh only and fail with a clear "run `get_globus_token.py`" message. The MLExchange SFAPI
   worker does this.
3. **The IRI token is fetched once per controller.** Long jobs (hours of polling) can outlive the
   access token, and a 401 isn't handled. The worker should refresh on 401.
4. **The docstring doesn't match the code.** `_create_iriapi_client` says it needs
   `GLOBUS_CLIENT_ID` and `GLOBUS_CLIENT_SECRET` (client credentials), but the code uses the native
   app refresh-token flow. Update the docstring or switch to a confidential client, which would
   avoid the human login.
5. **The streaming flow reads credentials differently.** `NerscStreamingSettings` uses the `CLIENT_ID`
   env var plus `PATH_NERSC_PRI_KEY`, while the controller uses the `PATH_NERSC_CLIENT_ID` file. If
   only the file is set, `CLIENT_ID` is `None` and the client fails. The `except RuntimeError` there
   only logs, so `client` is then undefined.
6. **IRI resources are fixed.** The job spec always requests 128 CPU cores or 4 GPUs per process and
   exclusive nodes, ignoring `--cpus-per-task` and `--gpus-per-node`.
7. **IRI has no synchronous command,** so the Shifter image check costs a debug job. Consider skipping
   the check for IRI or caching its result.
8. **Production never uses IRI.** The dispatcher doesn't pass `login_method`, so production always uses
   SFAPI. IRI is reached only through `nersc_recon_test_iriapi_flow` or manual calls. Moving to IRI
   needs a deployment parameter or config switch.
9. **Cancelling** a Prefect run cancels the NERSC job only in the streaming flow. Batch
   recon and segmentation jobs keep running.

## 11. How this maps to the MLExchange SFAPI worker

`prefect_worker_refactor/worker/flows/sfapi/sfapi_flows.py` ports the same design:

- **Same structure:** one `login_method` (`sfapi` / `iriapi`), the same four primitives in two
  backend classes, the same IRI endpoints and resource UUIDs, and the same token helper (copied from
  `get_globus_token.py`).
- **Gaps 1–3 fixed:** it checks the final state, never prompts for login, and refreshes the token on 401.
- **Other differences:** it uses **podman-hpc** instead of Shifter, so it can run the existing MLExchange
  Docker images; it creates the log directory before submitting; and the params file, which holds API
  keys, is written with mode `0600` and deleted when the job exits.
- **Still missing:** like splash_flows' batch flows, it doesn't yet cancel the NERSC job when the
  Prefect run is cancelled (gap 9). The streaming flow's `on_cancellation` hook is the pattern to copy.
