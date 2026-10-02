"""Prefect child flow that runs an MLExchange algorithm on NERSC via SFAPI.

The worker can run anywhere with HTTPS access to api.nersc.gov. Each flow run:
  1. builds a Slurm script that writes the params YAML on NERSC (mode 600),
     runs the container with podman-hpc, and removes the params file on exit;
  2. creates the log directory remotely (Slurm needs it to exist *before* the
     job starts, otherwise the job dies without any output);
  3. submits the script and waits for a terminal state;
  4. fails the flow run unless the final state is COMPLETED.
"""

import json
import logging
import os
import re
import shlex
import time

import yaml
from authlib.jose import JsonWebKey
from prefect import context, flow
from prefect.states import Failed
from sfapi_client import Client
from sfapi_client.compute import Compute, Machine
from sfapi_client.jobs import JobState

from flows.credentials import add_credentials_to_io_parameters
from flows.logger import setup_logger
from flows.sfapi.schema import SFAPIParams

logger = logging.getLogger(__name__)

# Path of the params file inside the container
CONTAINER_PARAMS_PATH = "/app/work/config/params.yaml"


def create_sfapi_client() -> Client:
    """Create an authenticated NERSC SFAPI client.

    Reads ``PATH_NERSC_CLIENT_ID`` (client id text file) and
    ``PATH_NERSC_PRI_KEY`` (private key, JWK JSON) from the environment.

    Returns:
        Authenticated :class:`sfapi_client.Client`.

    Raises:
        ValueError: If the environment variables are unset.
        FileNotFoundError: If the credential files do not exist.
    """
    client_id_path = os.getenv("PATH_NERSC_CLIENT_ID")
    client_secret_path = os.getenv("PATH_NERSC_PRI_KEY")

    if not client_id_path or not client_secret_path:
        raise ValueError(
            "PATH_NERSC_CLIENT_ID and PATH_NERSC_PRI_KEY must be set in the environment."
        )
    if not os.path.isfile(client_id_path) or not os.path.isfile(client_secret_path):
        raise FileNotFoundError("NERSC SFAPI credential files are missing.")

    with open(client_id_path, "r") as f:
        client_id = f.read().strip()
    with open(client_secret_path, "r") as f:
        client_secret = JsonWebKey.import_key(json.loads(f.read()))

    return Client(client_id, client_secret)


def resolve_paths(sfapi_params: SFAPIParams, nersc_user: str) -> dict:
    """Resolve working/log/params directories on NERSC.

    Defaults are built from the *NERSC* username (from SFAPI), not the local
    ``$USER`` of the machine the worker runs on.
    """
    scratch = f"/pscratch/sd/{nersc_user[0]}/{nersc_user}"
    return {
        "working_dir": sfapi_params.working_dir or scratch,
        "output_dir": sfapi_params.output_dir or f"{scratch}/mlex_job_logs",
        "error_dir": sfapi_params.error_dir or f"{scratch}/mlex_job_logs",
        "params_dir": f"{scratch}/mlex_temp",
    }


def build_slurm_script(
    sfapi_params: SFAPIParams,
    paths: dict,
    params_file_path: str,
    params_yaml: str,
) -> str:
    """Build the Slurm batch script submitted through SFAPI.

    Args:
        sfapi_params: Validated job parameters.
        paths: Output of :func:`resolve_paths`.
        params_file_path: Where the params YAML is written on NERSC.
        params_yaml: YAML content of the algorithm parameters.

    Returns:
        The complete batch script.
    """
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

    body = f"""
set -o pipefail
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
    return "\n".join(sbatch) + "\n" + body


def wait_for_job(compute: Compute, job, poll_interval: int = 30):
    """Wait for a job to reach a terminal state, tolerating transient lookups.

    SFAPI sometimes reports "Job not found" right after submission; in that
    case re-fetch the job by id and keep waiting.

    Returns:
        The final :class:`JobState`.
    """
    job_id = job.jobid
    for attempt in range(5):
        try:
            # complete() returns the terminal state; it does NOT raise on FAILED
            return job.complete()
        except Exception as e:
            if "Job not found" not in str(e) or attempt == 4:
                raise
            logger.warning(f"Job {job_id} not found yet, retrying lookup ({attempt + 1}/5)")
            time.sleep(poll_interval)
            job = compute.job(jobid=job_id)


@flow(name="SFAPI flow")
def launch_sfapi(
    sfapi_params: SFAPIParams,
    prev_flow_run_id: str = "",
):
    """Launch a job on NERSC using the Superfacility API (SFAPI).

    This is a synchronous flow on purpose: sfapi_client's sync Client blocks
    while polling, which must not run inside an async event loop.

    Args:
        sfapi_params: SFAPI job parameters.
        prev_flow_run_id: Previous flow run ID for chaining jobs.

    Returns:
        Current flow run ID on success, a Failed state otherwise.
    """
    flow_logger = setup_logger()
    flow_logger.info(
        f"SFAPI job {sfapi_params.job_name} on {sfapi_params.machine}: "
        f"{sfapi_params.image_name}:{sfapi_params.image_tag}"
    )

    io_params = sfapi_params.params.setdefault("io_parameters", {})
    if prev_flow_run_id and not io_params.get("uid_retrieve"):
        io_params["uid_retrieve"] = prev_flow_run_id

    current_flow_run_id = str(context.get_run_context().flow_run.id)
    io_params["uid_save"] = current_flow_run_id

    sfapi_params.params = add_credentials_to_io_parameters(sfapi_params.params)

    try:
        machine = Machine(sfapi_params.machine.lower())
    except ValueError:
        return Failed(message=f"Unknown NERSC machine: {sfapi_params.machine}")

    try:
        with create_sfapi_client() as client:
            compute = client.compute(machine)
            nersc_user = client.user().name
            paths = resolve_paths(sfapi_params, nersc_user)
            params_file_path = f"{paths['params_dir']}/params_{current_flow_run_id}.yaml"

            # Slurm opens --output/--error at job start, so the dirs must exist now
            dirs = {paths["output_dir"], paths["error_dir"], paths["params_dir"]}
            compute.run(f"mkdir -p {' '.join(sorted(dirs))}")

            job_script = build_slurm_script(
                sfapi_params,
                paths,
                params_file_path,
                yaml.safe_dump(sfapi_params.params),
            )
            job = compute.submit_job(job_script)
            flow_logger.info(f"Submitted NERSC job {job.jobid}")
            flow_logger.info(f"Logs: {paths['output_dir']}/{sfapi_params.job_name}_{job.jobid}.out")

            final_state = wait_for_job(compute, job)
    except Exception as e:
        flow_logger.error(f"SFAPI job failed: {e}")
        return Failed(message=f"SFAPI job failed: {e}")

    if final_state != JobState.COMPLETED:
        msg = f"NERSC job {job.jobid} ended in state {final_state}"
        flow_logger.error(msg)
        return Failed(message=msg)

    flow_logger.info(f"NERSC job {job.jobid} completed")
    return current_flow_run_id


if __name__ == "__main__":
    test_params = SFAPIParams(
        job_name="test_mlex_job",
        machine="perlmutter",
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
