import tempfile
import os
import yaml
from prefect import context, flow
from prefect.states import Failed
from prefect.utilities.processutils import run_process

from flows.logger import setup_logger
from flows.podman.schema import PodmanParams


@flow(name="Podman flow")
async def launch_podman(
    podman_params: PodmanParams,
    prev_flow_run_id: str = "",
):
    logger = setup_logger()

    if (
        prev_flow_run_id != ""
        and podman_params.params["io_parameters"]["uid_retrieve"] == ""
    ):
        # Append the previous flow run id to parameters if provided
        podman_params.params["io_parameters"]["uid_retrieve"] = prev_flow_run_id

    current_flow_run_id = str(context.get_run_context().flow_run.id)

    # Append current flow run id
    podman_params.params["io_parameters"]["uid_save"] = current_flow_run_id

    # Get paths from environment variables
    container_work_dir = os.getenv("CONTAINER_WORK_DIR", "/mlex_prefect_worker")
    host_work_dir = os.getenv("PREFECT_WORK_DIR", os.getcwd())
    
    # Create temp directory if it doesn't exist
    temp_dir = os.path.join(container_work_dir, "tmp")
    os.makedirs(temp_dir, exist_ok=True)

    # Create temporary file for parameters in the mounted directory
    with tempfile.NamedTemporaryFile(mode="w+t", dir=temp_dir, delete=False) as temp_file:
        yaml.dump(podman_params.params, temp_file)
        temp_file.flush()  # Ensure data is written
        temp_path = temp_file.name
    
    try:
        logger.info(f"Parameters file (container path): {temp_path}")
        
        # Convert container path to host path for Podman volume mounting
        host_temp_path = temp_path.replace(container_work_dir, host_work_dir)
        logger.info(f"Parameters file (host path): {host_temp_path}")

        # Mount extra volume with parameters yaml file using HOST path
        volumes = podman_params.volumes + [
            f"{host_temp_path}:/app/work/config/params.yaml"
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

        return current_flow_run_id
        
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Failed to clean up temp file: {e}")