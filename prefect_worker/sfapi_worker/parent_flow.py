import logging
import os
import json

from prefect import flow, task, get_run_logger
from prefect.deployments import run_deployment

# Import the Prefect client to check flow run states
from prefect.client import get_client

# Import schema classes for validation
from flows.conda.schema import CondaParams
from flows.docker.schema import DockerParams
from flows.podman.schema import PodmanParams
from flows.sfapi.schema import SFAPIParams
from flows.slurm.schema import SlurmParams

# Import utility functions and constants
from flows.utils import (
    FlowType,
    load_config,
    determine_best_environment,
    get_algorithm_details_from_mlflow,
    extract_folder_name_from_image,
)

logger = logging.getLogger(__name__)


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
    based on the worker configuration and loads algorithm details from MLflow.
    
    Args:
        params_list: List of parameters for the job, each containing model_name and task_name
    """
    prefect_logger = get_run_logger()
    client = get_client()
    
    # Load configuration from file (with env vars expanded)
    config = load_config()
    
    # Get worker configuration from config, default to "als" if not specified
    worker_config = config.get("worker", {})
    worker_name = worker_config.get("name", "als")
    prefect_logger.info(f"Starting job router (parent flow) for worker: {worker_name}")
    
    # Auto-select environment based on worker_type if specified, otherwise use worker_name
    target_env = determine_best_environment_task(worker_name)
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
            
            # Extract folder name from image_name
            folder_name = extract_folder_name_from_image(algorithm_details.get("image_name", ""))
            
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
                # Check if conda_env is available before proceeding
                conda_env = job_details["conda_env"]
                if not conda_env:
                    prefect_logger.error(f"No conda environment found for model {model_name}. Please update config.yml with the appropriate conda environment mapping.")
                    raise ValueError(f"No conda environment configured for model {model_name}")
                
                # Prepare conda parameters - use job_details for conda_env
                conda_relevant_params = {
                    "conda_env_name": conda_env,
                    "python_file_name": python_file,
                    "folder_name": folder_name,
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
                    parameters=deployment_data,
                    poll_interval=60
                )
                
                if flow_run.state.is_failed():
                    raise RuntimeError(f"Child flow failed at step {i+1}")
                    
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
                    parameters=deployment_data,
                    poll_interval=60
                )
                
                if flow_run.state.is_failed():
                    raise RuntimeError(f"Child flow failed at step {i+1}")
                    
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
                    parameters=deployment_data,
                    poll_interval=60
                )
                
                if flow_run.state.is_failed():
                    raise RuntimeError(f"Child flow failed at step {i+1}")
                    
                flow_run_id = str(flow_run.id)
                
            elif target_env == FlowType.slurm:
                # Check if conda_env is available before proceeding (Slurm also uses conda)
                conda_env = job_details["conda_env"]
                if not conda_env:
                    prefect_logger.error(f"No conda environment found for model {model_name}. Please update config.yml with the appropriate conda environment mapping.")
                    raise ValueError(f"No conda environment configured for model {model_name}")
                
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
                    "conda_env_name": conda_env,
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
                    parameters=deployment_data,
                    poll_interval=60
                )
                
                if flow_run.state.is_failed():
                    raise RuntimeError(f"Child flow failed at step {i+1}")
                    
                flow_run_id = str(flow_run.id)
                
            elif target_env == FlowType.sfapi:
                # Prepare SFAPI parameters for NERSC execution
                prefect_logger.info("Preparing SFAPI job for NERSC")
                
                # Build job name
                job_name = f"{model_name.replace(' ', '_')}_{task_name}_{folder_name}"[:50]  # SLURM job name limit
                
                # Get SFAPI configuration from job_details
                sfapi_relevant_params = {
                    "job_name": job_name,
                    "machine": job_details.get("sfapi_machine", "perlmutter"),
                    "queue": job_details.get("sfapi_queue", "realtime"),
                    "account": job_details.get("sfapi_account", "als"),
                    "constraint": job_details.get("sfapi_constraint", "cpu"),
                    "num_nodes": job_details.get("num_nodes", 1),
                    "ntasks_per_node": 1,
                    "cpus_per_task": 64,
                    "max_time": job_details.get("max_time", "0:15:00"),
                    "exclusive": job_details.get("sfapi_exclusive", True),
                    "image_name": algorithm_details["image_name"],
                    "image_tag": algorithm_details["image_tag"],
                    "command": f"python {python_file}",
                    "volumes": job_details.get("volumes", []),
                    "working_dir": job_details.get("sfapi_working_dir", ""),
                    "output_dir": job_details.get("sfapi_output_dir", ""),
                    "error_dir": job_details.get("sfapi_error_dir", ""),
                    "params": params
                }
                
                # Validate parameters with the schema
                sfapi_params = SFAPIParams(**sfapi_relevant_params)
                
                # If there's a previous flow run ID, set it in the parameters
                if flow_run_id:
                    if "io_parameters" not in sfapi_params.params:
                        sfapi_params.params["io_parameters"] = {}
                    sfapi_params.params["io_parameters"]["uid_retrieve"] = flow_run_id
                
                prefect_logger.info(f"Submitting SFAPI job: {job_name} to {sfapi_params.machine}")
                
                # Run the SFAPI deployment with parameters
                deployment_data = {
                    "sfapi_params": sfapi_params.dict(),
                    "prev_flow_run_id": flow_run_id
                }
                flow_run = await run_deployment(
                    name="SFAPI flow/launch_sfapi",
                    parameters=deployment_data,
                    poll_interval=60
                )
                
                if flow_run.state.is_failed():
                    raise RuntimeError(f"Child flow failed at step {i+1}")
                    
                flow_run_id = str(flow_run.id)
                prefect_logger.info(f"SFAPI job completed successfully: {flow_run_id}")
                
            else:
                raise ValueError(f"Flow type not supported: {target_env}")

            prefect_logger.info(f"Step {i+1} completed with flow run ID: {flow_run_id}")
            
        except Exception as e:
            prefect_logger.error(f"Error in step {i+1}: {str(e)}")
            raise
    
    prefect_logger.info(f"All steps completed successfully. Final flow run ID: {flow_run_id}")
    return flow_run_id
