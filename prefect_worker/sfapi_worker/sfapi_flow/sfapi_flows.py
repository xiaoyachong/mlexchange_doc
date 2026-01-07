import json
import logging
import os
import re
import tempfile
import time

import yaml
from authlib.jose import JsonWebKey
from prefect import context, flow
from prefect.states import Failed
from sfapi_client import Client
from sfapi_client.compute import Machine

from flows.credentials import add_credentials_to_io_parameters
from flows.logger import setup_logger
from flows.sfapi.schema import SFAPIParams

logger = logging.getLogger(__name__)


def create_sfapi_client() -> Client:
    """
    Create and return an NERSC SFAPI client instance.
    
    Requires environment variables:
    - PATH_NERSC_CLIENT_ID: Path to file containing SFAPI client ID
    - PATH_NERSC_PRI_KEY: Path to file containing SFAPI private key (JSON)
    
    Returns:
        Authenticated SFAPI Client instance
    """
    client_id_path = os.getenv("PATH_NERSC_CLIENT_ID")
    client_secret_path = os.getenv("PATH_NERSC_PRI_KEY")

    if not client_id_path or not client_secret_path:
        logger.error("NERSC credentials paths are missing.")
        raise ValueError("Missing NERSC credentials paths in environment variables.")
    
    if not os.path.isfile(client_id_path) or not os.path.isfile(client_secret_path):
        logger.error("NERSC credential files are missing.")
        raise FileNotFoundError("NERSC credential files are missing.")

    with open(client_id_path, "r") as f:
        client_id = f.read().strip()

    with open(client_secret_path, "r") as f:
        client_secret = JsonWebKey.import_key(json.loads(f.read()))

    try:
        client = Client(client_id, client_secret)
        logger.info("NERSC SFAPI client created successfully.")
        return client
    except Exception as e:
        logger.error(f"Failed to create NERSC SFAPI client: {e}")
        raise e


def build_slurm_script(sfapi_params: SFAPIParams, params_file_path: str) -> str:
    """
    Build a SLURM batch script for NERSC execution.
    
    Args:
        sfapi_params: SFAPI job parameters
        params_file_path: Path to temporary parameters file on NERSC
        
    Returns:
        Complete SLURM batch script as a string
    """
    user = os.getenv("USER", "unknown")
    
    # Determine output/error directories
    output_dir = sfapi_params.output_dir or f"/pscratch/sd/{user[0]}/{user}/job_logs"
    error_dir = sfapi_params.error_dir or f"/pscratch/sd/{user[0]}/{user}/job_logs"
    
    # Build volume mount arguments
    volume_args = ""
    if sfapi_params.volumes:
        for volume in sfapi_params.volumes:
            volume_args += f"--volume {volume} \\\n"
    
    # Add params file mount
    volume_args += f"--volume {params_file_path}:/app/work/config/params.yaml \\\n"
    
    # Build the command with params file
    container_command = f"{sfapi_params.command} /app/work/config/params.yaml"
    
    # Determine working directory
    working_dir = sfapi_params.working_dir or f"/pscratch/sd/{user[0]}/{user}"
    
    # Build SLURM script - IMPORTANT: must be left-aligned
    script = f"""#!/bin/bash
#SBATCH -q {sfapi_params.queue}
#SBATCH -A {sfapi_params.account}
#SBATCH -C {sfapi_params.constraint}
#SBATCH --job-name={sfapi_params.job_name}
#SBATCH --output={output_dir}/%x_%j.out
#SBATCH --error={error_dir}/%x_%j.err
#SBATCH -N {sfapi_params.num_nodes}
#SBATCH --ntasks-per-node {sfapi_params.ntasks_per_node}
#SBATCH --cpus-per-task {sfapi_params.cpus_per_task}
#SBATCH --time={sfapi_params.max_time}
"""
    
    if sfapi_params.exclusive:
        script += "#SBATCH --exclusive\n"
    
    script += f"""
date
echo "Working directory: {working_dir}"
cd {working_dir}

echo "Creating log directories..."
mkdir -p {output_dir}
mkdir -p {error_dir}

echo "Running container with podman-hpc..."
srun podman-hpc run \\
{volume_args}{sfapi_params.image_name}:{sfapi_params.image_tag} \\
bash -c "{container_command}"

exit_code=$?
date
echo "Container exit code: $exit_code"
exit $exit_code
"""
    
    return script


@flow(name="SFAPI flow")
async def launch_sfapi(
    sfapi_params: SFAPIParams,
    prev_flow_run_id: str = "",
):
    """
    Launch a job on NERSC using the Superfacility API (SFAPI).
    
    Args:
        sfapi_params: SFAPI job parameters
        prev_flow_run_id: Previous flow run ID for chaining jobs
        
    Returns:
        Current flow run ID on success, Failed state on error
    """
    logger = setup_logger()
    
    logger.info(f"Starting SFAPI job submission: {sfapi_params.job_name}")
    logger.info(f"Target machine: {sfapi_params.machine}")
    logger.info(f"Image: {sfapi_params.image_name}:{sfapi_params.image_tag}")
    
    # Handle previous flow run ID
    if (
        prev_flow_run_id != ""
        and sfapi_params.params.get("io_parameters", {}).get("uid_retrieve") == ""
    ):
        if "io_parameters" not in sfapi_params.params:
            sfapi_params.params["io_parameters"] = {}
        sfapi_params.params["io_parameters"]["uid_retrieve"] = prev_flow_run_id
    
    # Get current flow run ID
    current_flow_run_id = str(context.get_run_context().flow_run.id)
    
    # Append current flow run ID
    if "io_parameters" not in sfapi_params.params:
        sfapi_params.params["io_parameters"] = {}
    sfapi_params.params["io_parameters"]["uid_save"] = current_flow_run_id
    
    # Add credentials to io_parameters at the child flow level
    sfapi_params.params = add_credentials_to_io_parameters(sfapi_params.params)
    
    # Create SFAPI client
    try:
        client = create_sfapi_client()
    except Exception as e:
        logger.error(f"Failed to create SFAPI client: {e}")
        return Failed(message=f"SFAPI client creation failed: {e}")
    
    # Get the target machine
    try:
        machine_map = {
            "perlmutter": Machine.perlmutter,
            "cori": Machine.cori,
        }
        machine = machine_map.get(sfapi_params.machine.lower())
        if not machine:
            raise ValueError(f"Unknown machine: {sfapi_params.machine}")
        
        compute = client.compute(machine)
        logger.info(f"Connected to {sfapi_params.machine}")
    except Exception as e:
        logger.error(f"Failed to connect to machine: {e}")
        return Failed(message=f"Machine connection failed: {e}")
    
    # Create temporary file for parameters on NERSC filesystem
    user = client.user()
    temp_dir = f"/pscratch/sd/{user.name[0]}/{user.name}/mlex_temp"
    params_filename = f"params_{current_flow_run_id}.yaml"
    params_file_path = f"{temp_dir}/{params_filename}"
    
    # Create temporary local file first
    with tempfile.NamedTemporaryFile(mode="w+t", suffix=".yaml", delete=False) as temp_file:
        yaml.dump(sfapi_params.params, temp_file)
        local_params_path = temp_file.name
    
    try:
        # TODO: Upload params file to NERSC using SFAPI or assume it's accessible
        # For now, we'll assume the params are small enough to include in the script
        # In production, you might want to use Globus or scp to transfer the file
        
        logger.info(f"Parameters file would be at: {params_file_path}")
        logger.info(f"Local params file: {local_params_path}")
        
        # Build SLURM script
        job_script = build_slurm_script(sfapi_params, params_file_path)
        logger.info("Generated SLURM batch script")
        
        # Note: Since we can't easily upload the params file via SFAPI,
        # we'll embed the params in the job script as a workaround
        with open(local_params_path, 'r') as f:
            params_content = f.read()
        
        # Modify script to create params file inline
        setup_commands = f"""
echo "Creating temporary directory..."
mkdir -p {temp_dir}

echo "Creating parameters file..."
cat > {params_file_path} << 'PARAMS_EOF'
{params_content}
PARAMS_EOF

chmod 644 {params_file_path}
"""
        
        # Insert setup commands after the SBATCH directives
        lines = job_script.split('\n')
        sbatch_end = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#SBATCH') and not line.strip().startswith('#!'):
                sbatch_end = i
                break
        
        lines.insert(sbatch_end, setup_commands)
        job_script = '\n'.join(lines)
        
        logger.info("Submitting job to NERSC...")
        job = compute.submit_job(job_script)
        job_id = job.jobid
        logger.info(f"Job submitted successfully with ID: {job_id}")
        
        # Initial update to get job state
        try:
            job.update()
            logger.info(f"Job {job_id} initial state: {job.state}")
        except Exception as update_err:
            logger.warning(f"Initial job update failed, continuing: {update_err}")
        
        # Wait a bit before checking status
        time.sleep(10)
        
        # Monitor and wait for job completion
        logger.info(f"Waiting for job {job_id} to complete...")
        try:
            job.complete()  # This blocks until job completes
            logger.info(f"Job {job_id} completed successfully")
            
            # Clean up temporary params file
            cleanup_script = f"rm -f {params_file_path}"
            try:
                # Note: SFAPI doesn't have a direct way to run cleanup commands
                # In production, you might schedule a cleanup job or use ssh
                logger.info(f"Cleanup command (manual): {cleanup_script}")
            except Exception as cleanup_err:
                logger.warning(f"Cleanup warning: {cleanup_err}")
            
            return current_flow_run_id
            
        except Exception as e:
            logger.error(f"Error during job execution: {e}")
            
            # Try to recover job if it's a "Job not found" error
            match = re.search(r"Job not found:\s*(\d+)", str(e))
            if match:
                recovered_job_id = match.group(1)
                logger.info(f"Attempting to recover job {recovered_job_id}...")
                try:
                    job = compute.job(jobid=recovered_job_id)
                    time.sleep(30)
                    job.complete()
                    logger.info(f"Job {recovered_job_id} completed after recovery")
                    return current_flow_run_id
                except Exception as recovery_err:
                    logger.error(f"Failed to recover job {recovered_job_id}: {recovery_err}")
                    return Failed(message=f"Job recovery failed: {recovery_err}")
            else:
                return Failed(message=f"Job execution failed: {e}")
    
    finally:
        # Clean up local temporary file
        try:
            os.unlink(local_params_path)
        except Exception as e:
            logger.warning(f"Failed to clean up local temp file: {e}")


if __name__ == "__main__":
    import asyncio
    
    # Example usage
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
        params={"test": "data"}
    )
    
    asyncio.run(launch_sfapi(test_params))
