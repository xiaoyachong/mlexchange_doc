"""Prefect child flow that runs an MLExchange algorithm on NERSC Perlmutter.

Two login methods are supported (``SFAPIParams.login_method``), mirroring
splash_flows/orchestration/flows/bl832/nersc.py:

- ``sfapi``  : NERSC Superfacility API. Iris client id + private key
               (``PATH_NERSC_CLIENT_ID`` / ``PATH_NERSC_PRI_KEY``).
- ``iriapi`` : NERSC IRI API. Globus bearer token cached in a token file
               (``PATH_GLOBUS_TOKEN_FILE``, default ~/.globus/auth_tokens.json),
               created once by ``python flows/sfapi/globus_token.py``.

The worker can run anywhere with HTTPS access to NERSC. Each flow run:
  1. creates the log / params directories on NERSC (Slurm needs the log
     directory to exist *before* the job starts);
  2. submits a job that writes the params YAML (mode 600), runs the container
     with podman-hpc, and removes the params file on exit;
  3. waits for a terminal state and fails the flow run unless it completed.
"""

import json
import logging
import os
import shlex
import time
from pathlib import Path
from typing import Optional

import httpx
import yaml
from authlib.jose import JsonWebKey
from prefect import context, flow
from prefect.states import Failed
from sfapi_client import Client
from sfapi_client.compute import Machine
from sfapi_client.jobs import JobState

from flows.chain import chain_or_return, prepare_io_parameters
from flows.credentials import add_credentials_to_io_parameters
from flows.logger import setup_logger
from flows.sfapi.schema import SFAPIParams

logger = logging.getLogger(__name__)

# Path of the params file inside the container
CONTAINER_PARAMS_PATH = "/app/work/config/params.yaml"

IRI_POLL_SECONDS = 60
IRI_FAILED_STATES = ("failed", "canceled", "cancelled", "timeout")


# --------------------------------------------------------------------------- #
# Script building (shared by both login methods)
# --------------------------------------------------------------------------- #
def resolve_paths(sfapi_params: SFAPIParams, nersc_user: str) -> dict:
    """Resolve working/log/params directories on NERSC for ``nersc_user``."""
    scratch = f"/pscratch/sd/{nersc_user[0]}/{nersc_user}"
    return {
        "working_dir": sfapi_params.working_dir or scratch,
        "output_dir": sfapi_params.output_dir or f"{scratch}/mlex_job_logs",
        "error_dir": sfapi_params.error_dir or f"{scratch}/mlex_job_logs",
        "params_dir": f"{scratch}/mlex_temp",
    }


def build_job_body(
    sfapi_params: SFAPIParams,
    paths: dict,
    params_file_path: str,
    params_yaml: str,
) -> str:
    """Build the shell body of the job (everything after the #SBATCH lines)."""
    podman_args = []
    if sfapi_params.gpus_per_node > 0:
        podman_args.append("--gpu")
    for volume in sfapi_params.volumes or []:
        podman_args.append(f"--volume {volume}")
    podman_args.append(f"--volume {params_file_path}:{CONTAINER_PARAMS_PATH}:ro")

    image = f"{sfapi_params.image_name}:{sfapi_params.image_tag}"
    container_command = f"{sfapi_params.command} {CONTAINER_PARAMS_PATH}"
    podman_line = (
        "srun podman-hpc run --rm "
        + " ".join(podman_args)
        + f" {image} bash -c {shlex.quote(container_command)}"
    )

    return f"""set -o pipefail
date

# Params contain API keys: keep them private and always remove them on exit
umask 077
mkdir -p {paths['params_dir']}
trap 'rm -f {params_file_path}' EXIT
cat > {params_file_path} << 'PARAMS_EOF'
{params_yaml}
PARAMS_EOF
chmod 600 {params_file_path}

cd {paths['working_dir']}
echo "Running {image} with podman-hpc..."
{podman_line}
exit_code=$?

date
echo "Container exit code: $exit_code"
exit $exit_code
"""


def build_slurm_script(sfapi_params: SFAPIParams, paths: dict, body: str) -> str:
    """Prepend #SBATCH directives to ``body`` (used by the SFAPI backend)."""
    sbatch = [
        "#!/bin/bash",
        f"#SBATCH -q {sfapi_params.queue}",
        f"#SBATCH -A {sfapi_params.account}",
        f"#SBATCH -C {sfapi_params.constraint}",
        f"#SBATCH --job-name={sfapi_params.job_name}",
        f"#SBATCH --output={paths['output_dir']}/%x_%j.out",
        f"#SBATCH --error={paths['error_dir']}/%x_%j.err",
        f"#SBATCH -N {sfapi_params.num_nodes}",
        f"#SBATCH --ntasks-per-node={sfapi_params.ntasks_per_node}",
        f"#SBATCH --cpus-per-task={sfapi_params.cpus_per_task}",
        f"#SBATCH --time={sfapi_params.max_time}",
    ]
    if sfapi_params.gpus_per_node > 0:
        sbatch.append(f"#SBATCH --gpus-per-node={sfapi_params.gpus_per_node}")
    if sfapi_params.exclusive:
        sbatch.append("#SBATCH --exclusive")
    return "\n".join(sbatch) + "\n\n" + body


def walltime_to_seconds(max_time: str) -> int:
    """Convert ``[H]H:MM:SS`` or ``MM:SS`` to seconds."""
    parts = [int(p) for p in max_time.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class SfapiBackend:
    """Submit and monitor jobs through the NERSC Superfacility API."""

    name = "sfapi"

    def __init__(self, sfapi_params: SFAPIParams):
        self.params = sfapi_params
        self.client = self._create_client()
        self.compute = self.client.compute(Machine(sfapi_params.machine.lower()))

    @staticmethod
    def _create_client() -> Client:
        client_id_path = os.getenv("PATH_NERSC_CLIENT_ID")
        client_secret_path = os.getenv("PATH_NERSC_PRI_KEY")
        if not client_id_path or not client_secret_path:
            raise ValueError(
                "PATH_NERSC_CLIENT_ID and PATH_NERSC_PRI_KEY must be set for login_method=sfapi."
            )
        if not os.path.isfile(client_id_path) or not os.path.isfile(client_secret_path):
            raise FileNotFoundError("NERSC SFAPI credential files are missing.")
        with open(client_id_path, "r") as f:
            client_id = f.read().strip()
        with open(client_secret_path, "r") as f:
            client_secret = JsonWebKey.import_key(json.loads(f.read()))
        return Client(client_id, client_secret)

    def close(self) -> None:
        self.client.close()

    def username(self) -> str:
        return self.client.user().name

    def mkdir(self, dirs: list[str]) -> None:
        self.compute.run(f"mkdir -p {' '.join(dirs)}")

    def submit(self, paths: dict, body: str, run_id: str) -> str:
        job = self.compute.submit_job(build_slurm_script(self.params, paths, body))
        return str(job.jobid)

    def log_hint(self, paths: dict, job_id: str, run_id: str) -> str:
        return f"{paths['output_dir']}/{self.params.job_name}_{job_id}.out"

    def wait(self, job_id: str) -> tuple[bool, str]:
        """Wait for a terminal state. complete() does NOT raise on FAILED."""
        for attempt in range(5):
            try:
                job = self.compute.job(jobid=job_id)
                state = job.complete()
                return state == JobState.COMPLETED, str(state)
            except Exception as e:
                # SFAPI sometimes reports "Job not found" right after submission
                if "Job not found" not in str(e) or attempt == 4:
                    raise
                logger.warning(f"Job {job_id} not found yet, retrying ({attempt + 1}/5)")
                time.sleep(30)
        return False, "UNKNOWN"


class IriBackend:
    """Submit and monitor jobs through the NERSC IRI API (Globus token)."""

    name = "iriapi"

    def __init__(self, sfapi_params: SFAPIParams):
        self.params = sfapi_params
        self.client = httpx.Client(
            base_url=sfapi_params.iri_api_base_url,
            headers={"Authorization": f"Bearer {self._access_token()}"},
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
        )

    @staticmethod
    def _access_token() -> str:
        """Return a valid IRI access token without ever prompting for login.

        Uses the cached token if still valid, otherwise refreshes it with the
        stored refresh token. A worker cannot answer a browser login, so if
        refreshing fails the run fails with instructions instead of hanging.
        """
        from flows.sfapi import globus_token as gt

        token_file_env = os.getenv("PATH_GLOBUS_TOKEN_FILE")
        token_file = Path(token_file_env) if token_file_env else gt.DEFAULT_TOKEN_FILE
        login_hint = (
            f"Log in once on the worker host: python flows/sfapi/globus_token.py "
            f"--token-file {token_file} --validate-iri"
        )

        stored = gt.load_tokens(token_file)
        if not stored:
            raise FileNotFoundError(f"No Globus token file at {token_file}. {login_hint}")

        try:
            iri_token = gt.get_iri_token(stored)
            if time.time() < iri_token.get("expires_at_seconds", 0) - 60:
                return iri_token["access_token"]
        except RuntimeError:
            pass

        client = gt.globus_sdk.NativeAppAuthClient(gt.CLIENT_ID)
        refreshed, _ = gt.refresh_stored_tokens(client, stored)
        if refreshed is None:
            raise RuntimeError(f"Globus token refresh failed. {login_hint}")
        iri_token = gt.validate_auth_data(refreshed)
        gt.save_tokens(token_file, refreshed)
        return iri_token["access_token"]

    def close(self) -> None:
        self.client.close()

    def username(self) -> str:
        username = os.getenv("NERSC_USERNAME")
        if not username:
            raise ValueError("NERSC_USERNAME must be set for login_method=iriapi.")
        return username

    def mkdir(self, dirs: list[str]) -> None:
        for path in dirs:
            response = self.client.post(
                f"/api/v1/filesystem/mkdir/{self.params.iri_login_resource}",
                json={"path": path, "parents": True},
            )
            response.raise_for_status()

    def _log_paths(self, paths: dict, run_id: str) -> tuple[str, str]:
        stem = f"{self.params.job_name}_{run_id}"
        return f"{paths['output_dir']}/{stem}.out", f"{paths['error_dir']}/{stem}.err"

    def submit(self, paths: dict, body: str, run_id: str) -> str:
        p = self.params
        resources = {
            "node_count": p.num_nodes,
            "processes_per_node": p.ntasks_per_node,
            "exclusive_node_use": p.exclusive,
        }
        if p.gpus_per_node > 0:
            resources["gpu_cores_per_process"] = p.gpus_per_node
        else:
            resources["cpu_cores_per_process"] = p.cpus_per_task

        stdout_path, stderr_path = self._log_paths(paths, run_id)
        # Same job-spec shape as splash_flows' IRIAPI branch of _submit_job
        job_spec = {
            "executable": "/bin/bash",
            "arguments": ["-s"],
            "pre_launch": body,
            "resources": resources,
            "attributes": {
                "duration": walltime_to_seconds(p.max_time),
                "queue_name": p.queue,
                "account": p.account,
                "custom_attributes": {"constraint": p.constraint},
            },
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
        }
        response = self.client.post(
            f"/api/v1/compute/job/{p.iri_job_resource}", json=job_spec
        )
        if not response.is_success:
            # Do not log job_spec: pre_launch contains API keys
            logger.error(f"IRI job submission failed: {response.status_code} {response.text}")
        response.raise_for_status()
        return str(response.json()["id"])

    def log_hint(self, paths: dict, job_id: str, run_id: str) -> str:
        return self._log_paths(paths, run_id)[0]

    def wait(self, job_id: str) -> tuple[bool, str]:
        while True:
            url = f"/api/v1/compute/status/{self.params.iri_status_resource}/{job_id}"
            response = self.client.get(url)
            if response.status_code == 401:
                # Access tokens are short-lived; long jobs outlive them. Refresh and retry.
                self.client.headers["Authorization"] = f"Bearer {self._access_token()}"
                response = self.client.get(url)
            response.raise_for_status()
            state = str(response.json().get("status", {}).get("state", "")).lower()
            logger.info(f"IRI job {job_id} state: {state}")
            if state == "completed":
                return True, state
            if state in IRI_FAILED_STATES:
                return False, state
            time.sleep(IRI_POLL_SECONDS)


BACKENDS = {SfapiBackend.name: SfapiBackend, IriBackend.name: IriBackend}


def create_backend(sfapi_params: SFAPIParams):
    """Instantiate the backend selected by ``sfapi_params.login_method``."""
    try:
        backend_cls = BACKENDS[sfapi_params.login_method]
    except KeyError:
        raise ValueError(
            f"Unknown login_method {sfapi_params.login_method!r}; use one of {list(BACKENDS)}"
        )
    return backend_cls(sfapi_params)


# --------------------------------------------------------------------------- #
# Flow
# --------------------------------------------------------------------------- #
@flow(name="SFAPI flow")
def launch_sfapi(
    sfapi_params: SFAPIParams,
    prev_flow_run_id: str = "",
    next_steps: Optional[list[dict]] = None,
):
    """Launch a job on NERSC via SFAPI or the IRI API.

    Synchronous on purpose: both clients block while polling.

    Args:
        sfapi_params: Job parameters (``login_method`` selects the API).
        prev_flow_run_id: Previous flow run ID for chaining jobs.
        next_steps: Remaining steps of the job (see flows/chain.py).

    Returns:
        Current flow run ID on success, a Failed state otherwise.
    """
    flow_logger = setup_logger()
    flow_logger.info(
        f"NERSC job {sfapi_params.job_name} via {sfapi_params.login_method} on "
        f"{sfapi_params.machine}: {sfapi_params.image_name}:{sfapi_params.image_tag}"
    )

    current_flow_run_id = str(context.get_run_context().flow_run.id)
    prepare_io_parameters(sfapi_params.params, prev_flow_run_id, current_flow_run_id)

    sfapi_params.params = add_credentials_to_io_parameters(sfapi_params.params)

    backend = None
    try:
        backend = create_backend(sfapi_params)
        paths = resolve_paths(sfapi_params, backend.username())
        params_file_path = f"{paths['params_dir']}/params_{current_flow_run_id}.yaml"

        backend.mkdir(sorted({paths["output_dir"], paths["error_dir"], paths["params_dir"]}))

        body = build_job_body(
            sfapi_params, paths, params_file_path, yaml.safe_dump(sfapi_params.params)
        )
        job_id = backend.submit(paths, body, current_flow_run_id)
        flow_logger.info(f"Submitted NERSC job {job_id}")
        flow_logger.info(f"Logs: {backend.log_hint(paths, job_id, current_flow_run_id)}")

        ok, final_state = backend.wait(job_id)
    except Exception as e:
        flow_logger.error(f"NERSC job failed: {e}")
        return Failed(message=f"NERSC job failed: {e}")
    finally:
        if backend is not None:
            backend.close()

    if not ok:
        msg = f"NERSC job {job_id} ended in state {final_state}"
        flow_logger.error(msg)
        return Failed(message=msg)

    flow_logger.info(f"NERSC job {job_id} completed")
    return chain_or_return(current_flow_run_id, next_steps, current_flow_run_id)


if __name__ == "__main__":
    test_params = SFAPIParams(
        job_name="test_mlex_job",
        login_method="sfapi",
        queue="debug",
        account="als",
        num_nodes=1,
        max_time="0:05:00",
        image_name="ghcr.io/mlexchange/mlex_dlsia_segmentation_prototype",
        image_tag="latest",
        command="python src/train.py",
        params={"test": "data"},
    )
    launch_sfapi(test_params)
