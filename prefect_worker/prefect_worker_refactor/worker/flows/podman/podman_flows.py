import tempfile

import yaml
from typing import Optional

from prefect import context, flow
from prefect.states import Failed
from prefect.utilities.processutils import run_process

from flows.logger import setup_logger
from flows.podman.schema import PodmanParams
from flows.chain import chain_or_return, prepare_io_parameters
from flows.credentials import add_credentials_to_io_parameters


@flow(name="Podman flow")
async def launch_podman(
    podman_params: PodmanParams,
    prev_flow_run_id: str = "",
    next_steps: Optional[list[dict]] = None,
):
    logger = setup_logger()

    current_flow_run_id = str(context.get_run_context().flow_run.id)
    prepare_io_parameters(podman_params.params, prev_flow_run_id, current_flow_run_id)

    # Add credentials to io_parameters at the child flow level
    podman_params.params = add_credentials_to_io_parameters(podman_params.params)

    # Create temporary file for parameters
    with tempfile.NamedTemporaryFile(mode="w+t") as temp_file:
        yaml.dump(podman_params.params, temp_file)
        logger.info(f"Parameters file: {temp_file.name}")

        # Mount extra volume with parameters yaml file
        volumes = podman_params.volumes + [
            f"{temp_file.name}:/app/work/config/params.yaml"
        ]
        command = f"{podman_params.command} /app/work/config/params.yaml"

        # Define podman command
        cmd = [
            "flows/podman/bash_run_podman.sh",
            f"{podman_params.image_name}:{podman_params.image_tag}",
            command,
            " ".join(volumes),
            podman_params.network,
            " ".join(f"{k}={v}" for k, v in podman_params.env_vars.items()),
        ]
        logger.info(f"Launching with command: {cmd}")
        process = await run_process(cmd, stream_output=True)

    if process.returncode != 0:
        return Failed(message="Podman command failed")

    return chain_or_return(current_flow_run_id, next_steps, current_flow_run_id)
