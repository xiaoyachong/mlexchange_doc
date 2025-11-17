import asyncio  # ADDED
import logging
import os
import json

from prefect import flow, task, get_run_logger
from prefect.deployments import run_deployment
from prefect.flow_runs import wait_for_flow_run
from prefect.states import Failed
from httpx import HTTPStatusError  # ADDED

# Import the Prefect client to check flow run states
from prefect.client import get_client

# Import schema classes for validation
from flows.conda.schema import CondaParams
from flows.docker.schema import DockerParams
from flows.podman.schema import PodmanParams
from flows.slurm.schema import SlurmParams

# Import utility functions and constants
from flows.utils import (
    FlowType,
    load_config,
    determine_best_environment,
    get_algorithm_details_from_mlflow,
)

logger = logging.getLogger(__name__)


async def safe_wait_for_flow_run(flow_run_id, prefect_logger, poll_interval=30):
    """
    Wait indefinitely for flow run to complete, with tolerance for transient 500 errors.
    Suitable for long-running ML training jobs that may take hours or days.
    
    Args:
        flow_run_id: ID of the flow run to monitor
        prefect_logger: Logger instance
        poll_interval: How often to poll (seconds) - longer intervals reduce server load
    """
    await asyncio.sleep(3)  # Initial delay for registration
    
    start_time = asyncio.get_event_loop().time()
    error_count = 0
    max_consecutive_errors = 10  # Only fail after many consecutive errors
    
    prefect_logger.info(f"Waiting for child flow {flow_run_id} to complete (no timeout)...")
    
    while True:
        try:
            # This will poll internally until the flow completes (no matter how long)
            result = await wait_for_flow_run(flow_run_id, poll_interval=poll_interval)
            
            elapsed = asyncio.get_event_loop().time() - start_time
            prefect_logger.info(
                f"Child flow {flow_run_id} completed after {elapsed/60:.1f} minutes "
                f"({error_count} transient errors encountered)"
            )
            return result
            
        except HTTPStatusError as e:
            if e.response.status_code == 500:
                error_count += 1
                
                # Only fail if we've had many consecutive errors (likely a real problem)
                if error_count >= max_consecutive_errors:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    prefect_logger.error(
                        f"Flow {flow_run_id} failed after {max_consecutive_errors} consecutive 500 errors "
                        f"(elapsed: {elapsed/60:.1f} minutes)"
                    )
                    raise
                
                # Calculate backoff with cap at 2 minutes
                wait_time = min(10 + (error_count * 10), 120)  # 20s, 30s, 40s... up to 120s
                
                elapsed = asyncio.get_event_loop().time() - start_time
                prefect_logger.warning(
                    f"Transient 500 error #{error_count} for flow {flow_run_id} "
                    f"(elapsed: {elapsed/60:.1f} min), retrying in {wait_time}s..."
                )
                
                await asyncio.sleep(wait_time)
                continue
            else:
                # Non-500 error, re-raise immediately
                raise
                
        except Exception as e:
            # Unexpected error - log and re-raise
            elapsed = asyncio.get_event_loop().time() - start_time
            prefect_logger.error(
                f"Unexpected error waiting for flow {flow_run_id} after {elapsed/60:.1f} min: "
                f"{type(e).__name__}: {e}"
            )
            raise

@task
def determine_best_environment_task(hpc_type: str) -> FlowType:
    """
    Determine the best execution environment based on hpc_type.
    
    Args:
        hpc_type: Type of HPC to execute on
    
    Returns:
        Best flow type to use
    """
    return determine_best_environment(hpc_type)


@task
def get_algorithm_details_from_mlflow_task(model_name: str, config: dict):
    """
    Retrieve algorithm details from MLflow using the model name.
    
    Args:
        model_name: The name of the model in MLflow
        config: Configuration dictionary from config.yml
    
    Returns:
        Tuple containing (algorithm_details, job_details)
    """
    return get_algorithm_details_from_mlflow(model_name, config)


@flow(name="Parent flow")
async def launch_parent_flow(params_list: list[dict]):
    """
    Smart job router that automatically selects the best execution environment
    based on the HPC type and loads algorithm details from MLflow.
    
    Args:
        params_list: List of parameters for the job, each containing model_name and task_name
    """
    prefect_logger = get_run_logger()
    client = get_client()
    
    # Load configuration from file (with env vars expanded)
    config = load_config()
    
    # Get HPC type from config, default to "als" if not specified
    hpc_type = config.get("hpc_type", "als")
    prefect_logger.info(f"Starting job router (parent flow) for HPC: {hpc_type}")
    
    # Auto-select environment based on hpc_type
    target_env = determine_best_environment_task(hpc_type)
    prefect_logger.info(f"Selected target environment: {target_env}")
    
    # Execute each step in sequence based on the selected environment
    flow_run_id = ""
    
    for i, child_job_params in enumerate(params_list):
        prefect_logger.info(f"Running step {i+1} of {len(params_list)}")
        
        try:
            # Get model name and task
            model_name = child_job_params.get("model_name", "")
            task_name = child_job_params.get("task_name", "")
            params = child_job_params.get("params", {})
            
            # NOTE: Credentials are NO LONGER added here - they will be added in child flows
            
            # Get algorithm details and job details from MLflow
            algorithm_details, job_details = get_algorithm_details_from_mlflow_task(model_name, config)
            
            # Get the appropriate python file name based on the task name
            if task_name == "execute":
                python_file = algorithm_details.get("python_file", "")
            elif task_name == "train":
                python_file = algorithm_details.get("python_file_train", "")
            elif task_name == "inference":
                python_file = algorithm_details.get("python_file_inference", "")
            elif task_name == "tune":
                python_file = algorithm_details.get("python_file_tune", "")
            else:
                # For any other task, default to python_file
                python_file = algorithm_details.get("python_file", "")
            
            if not python_file:
                prefect_logger.error(f"No Python file found for task {task_name}")
                raise ValueError(f"No Python file found for task {task_name}")
            
            if target_env == FlowType.conda:
                # Prepare conda parameters - use job_details for conda_env
                conda_relevant_params = {
                    "conda_env_name": job_details["conda_env"],
                    "python_file_name": python_file,
                    "params": params
                }
                # Validate parameters with the schema
                conda_params = CondaParams(**conda_relevant_params)
                # If there's a previous flow run ID, set it in the parameters
                if flow_run_id:
                    if "io_parameters" not in conda_params.params:
                        conda_params.params["io_parameters"] = {}
                    conda_params.params["io_parameters"]["uid_retrieve"] = flow_run_id
                
                # Run the conda deployment with parameters
                deployment_data = {
                    "conda_params": conda_params.dict(),
                    "prev_flow_run_id": flow_run_id
                }
                flow_run = await run_deployment(
                    name="launch_conda/launch_conda",
                    parameters=deployment_data
                )
                
                # CHANGED: Use safe_wait_for_flow_run and check .state
                prefect_logger.info(f"Waiting for child flow {flow_run.id} to complete...")
                flow_run_state = await safe_wait_for_flow_run(flow_run.id, prefect_logger)
                
                if flow_run_state.state.is_failed():
                    prefect_logger.error(f"Step {i+1} failed with state: {flow_run_state.state.type}")
                    raise RuntimeError(f"Child flow failed at step {i+1} with state: {flow_run_state.state.type}")
                    
                flow_run_id = str(flow_run.id)
                
            elif target_env == FlowType.docker:
                # Prepare docker parameters - use algorithm_details for image info and job_details for environment
                docker_relevant_params = {
                    "image_name": algorithm_details["image_name"],
                    "image_tag": algorithm_details["image_tag"],
                    "command": f"python {python_file}",
                    "volumes": job_details["volumes"],
                    "network": job_details["network"],
                    "env_vars": {},
                    "params": params
                }
                # Validate parameters with the schema
                docker_params = DockerParams(**docker_relevant_params)
                # If there's a previous flow run ID, set it in the parameters
                if flow_run_id:
                    if "io_parameters" not in docker_params.params:
                        docker_params.params["io_parameters"] = {}
                    docker_params.params["io_parameters"]["uid_retrieve"] = flow_run_id
                
                # Run the docker deployment with parameters
                deployment_data = {
                    "docker_params": docker_params.dict(),
                    "prev_flow_run_id": flow_run_id
                }
                flow_run = await run_deployment(
                    name="Docker flow/launch_docker",
                    parameters=deployment_data
                )
                
                # CHANGED: Use safe_wait_for_flow_run and check .state
                prefect_logger.info(f"Waiting for child flow {flow_run.id} to complete...")
                flow_run_state = await safe_wait_for_flow_run(flow_run.id, prefect_logger)
                
                if flow_run_state.state.is_failed():
                    prefect_logger.error(f"Step {i+1} failed with state: {flow_run_state.state.type}")
                    raise RuntimeError(f"Child flow failed at step {i+1} with state: {flow_run_state.state.type}")
                    
                flow_run_id = str(flow_run.id)
                
            elif target_env == FlowType.podman:
                # Prepare podman parameters - use algorithm_details for image info and job_details for environment
                podman_relevant_params = {
                    "image_name": algorithm_details["image_name"],
                    "image_tag": algorithm_details["image_tag"],
                    "command": f"python {python_file}",
                    "volumes": job_details["volumes"],
                    "network": job_details["network"],
                    "env_vars": {},
                    "params": params
                }
                # Validate parameters with the schema
                podman_params = PodmanParams(**podman_relevant_params)
                # If there's a previous flow run ID, set it in the parameters
                if flow_run_id:
                    if "io_parameters" not in podman_params.params:
                        podman_params.params["io_parameters"] = {}
                    podman_params.params["io_parameters"]["uid_retrieve"] = flow_run_id
                
                # Run the podman deployment with parameters
                deployment_data = {
                    "podman_params": podman_params.dict(),
                    "prev_flow_run_id": flow_run_id
                }
                flow_run = await run_deployment(
                    name="Podman flow/launch_podman", 
                    parameters=deployment_data
                )
                
                # CHANGED: Use safe_wait_for_flow_run and check .state
                prefect_logger.info(f"Waiting for child flow {flow_run.id} to complete...")
                flow_run_state = await safe_wait_for_flow_run(flow_run.id, prefect_logger)
                
                if flow_run_state.state.is_failed():
                    prefect_logger.error(f"Step {i+1} failed with state: {flow_run_state.state.type}")
                    raise RuntimeError(f"Child flow failed at step {i+1} with state: {flow_run_state.state.type}")
                    
                flow_run_id = str(flow_run.id)
                
            elif target_env == FlowType.slurm:
                # Parse string JSON values if needed
                partitions = job_details["partitions"]
                if isinstance(partitions, str):
                    partitions = json.loads(partitions)
                
                reservations = job_details["reservations"]
                if isinstance(reservations, str):
                    reservations = json.loads(reservations)
                
                forward_ports = job_details["forward_ports"]
                if isinstance(forward_ports, str):
                    forward_ports = json.loads(forward_ports)
                
                # Prepare slurm parameters - use job_details for slurm configuration
                slurm_relevant_params = {
                    "job_name": f"{model_name}_{task_name}",
                    "num_nodes": job_details["num_nodes"],
                    "partitions": partitions,
                    "reservations": reservations,
                    "max_time": job_details["max_time"],
                    "conda_env_name": job_details["conda_env"],
                    "forward_ports": forward_ports,
                    "submission_ssh_key": job_details["submission_ssh_key"],
                    "python_file_name": python_file,
                    "params": params
                }

                # Validate parameters with the schema
                slurm_params = SlurmParams(**slurm_relevant_params)
                
                # If there's a previous flow run ID, set it in the parameters
                if flow_run_id:
                    if "io_parameters" not in slurm_params.params:
                        slurm_params.params["io_parameters"] = {}
                    slurm_params.params["io_parameters"]["uid_retrieve"] = flow_run_id
                
                # Run the slurm deployment with parameters
                deployment_data = {
                    "slurm_params": slurm_params.dict(),
                    "prev_flow_run_id": flow_run_id
                }
                flow_run = await run_deployment(
                    name="launch_slurm/launch_slurm",
                    parameters=deployment_data
                )
                
                # CHANGED: Use safe_wait_for_flow_run and check .state
                prefect_logger.info(f"Waiting for child flow {flow_run.id} to complete...")
                flow_run_state = await safe_wait_for_flow_run(flow_run.id, prefect_logger)
                
                if flow_run_state.state.is_failed():
                    prefect_logger.error(f"Step {i+1} failed with state: {flow_run_state.state.type}")
                    raise RuntimeError(f"Child flow failed at step {i+1} with state: {flow_run_state.state.type}")
                    
                flow_run_id = str(flow_run.id)
                
            else:
                prefect_logger.error("Flow type not supported")
                raise ValueError("Flow type not supported")

            prefect_logger.info(f"Step {i+1} completed with flow run ID: {flow_run_id}")
            
        except Exception as e:
            prefect_logger.error(f"Error in step {i+1}: {str(e)}")
            return Failed(message=f"Error in step {i+1}: {str(e)}")
    
    prefect_logger.info(f"All steps completed successfully. Final flow run ID: {flow_run_id}")
    return flow_run_id