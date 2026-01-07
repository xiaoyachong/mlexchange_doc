import logging
import os
import json
from enum import Enum

import yaml
import mlflow
from mlflow.tracking import MlflowClient
from prefect import get_run_logger
from dotenv import load_dotenv

# Load .env file at module import
load_dotenv()

logger = logging.getLogger(__name__)

# MLflow connection parameters - load from environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

# Path to configuration file
CONFIG_PATH = "config.yml"


class FlowType(str, Enum):
    podman = "podman"
    conda = "conda"
    slurm = "slurm"
    docker = "docker"
    sfapi = "sfapi"  # Added SFAPI support


def expand_env_vars(obj):
    """Recursively expand environment variables in nested structures"""
    if isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [expand_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        # Expand ${VAR} and $VAR patterns
        return os.path.expandvars(obj)
    else:
        return obj


def load_config():
    """
    Load the configuration from config.yml file and expand environment variables.
    
    Returns:
        Dictionary containing configuration with expanded env vars
    """
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        
        # Expand all environment variables
        config = expand_env_vars(config)
        
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {CONFIG_PATH}: {str(e)}")
        return {}


def extract_folder_name_from_image(image_name: str) -> str:
    """
    Extract folder name from image_name.
    For example: ghcr.io/mlexchange/mlex_dlsia_segmentation_prototype -> mlex_dlsia_segmentation_prototype
    
    Args:
        image_name: Full image name from MLflow
    
    Returns:
        Folder name extracted from image_name
    """
    if not image_name:
        return ""
    
    # Split by '/' and get the last part
    parts = image_name.split('/')
    if len(parts) > 0:
        return parts[-1]
    
    return ""


def determine_best_environment(hpc_type: str):
    """
    Determine the best execution environment based on hpc_type.
    
    Args:
        hpc_type: Type of HPC to execute on (can be worker name or flow type)
    
    Returns:
        Best flow type to use
    """
    logger = get_run_logger()
    
    # Map HPC type to flow type
    hpc_type = hpc_type.lower()
    if hpc_type == "nersc":
        logger.info(f"Worker type is NERSC, selecting SFAPI")
        return FlowType.sfapi
    elif hpc_type == "nersc-slurm":
        # Legacy NERSC support via SLURM
        logger.info(f"Worker type is NERSC-SLURM, selecting SLURM")
        return FlowType.slurm
    elif hpc_type == "nsls-ii":
        logger.info(f"Worker type is NSLS-II, selecting PODMAN")
        return FlowType.podman
    elif hpc_type == "als":
        logger.info(f"Worker type is ALS cluster-ball, selecting DOCKER")
        return FlowType.docker
    elif hpc_type in [ft.value for ft in FlowType]:
        # If the hpc_type is actually a flow type, use it directly
        return FlowType(hpc_type)
    else:
        # Default to conda for unknown types
        logger.info(f"Unknown worker type: {hpc_type}, defaulting to CONDA environment")
        return FlowType.conda


def _get_conda_env_for_model(model_name: str, config: dict, model_version: str = "") -> str:
    """
    Simple helper function to determine the appropriate conda environment for a model.
    
    Args:
        model_name: The name of the model
        config: The configuration dictionary from config.yml
        model_version: The version of the model from MLflow tags (optional)
        
    Returns:
        The appropriate conda environment name
    """
    conda_envs = config.get("conda", {}).get("conda_env_name", {})
    
    # Try direct lookup by model_name_version first if version is provided
    if model_version:
        versioned_key = f"{model_name}_{model_version}"
        if versioned_key in conda_envs:
            return conda_envs[versioned_key]
    
    # Try direct lookup by model name
    if model_name in conda_envs:
        return conda_envs[model_name]
    
    # Return empty string if no match found
    return ""


def get_algorithm_details_from_mlflow(model_name: str, config: dict):
    """
    Retrieve algorithm details from MLflow using the model name.
    
    Args:
        model_name: The name of the model in MLflow
        config: Configuration dictionary from config.yml
    
    Returns:
        Tuple containing (algorithm_details, job_details)
    """
    logger = get_run_logger()
    logger.info(f"Retrieving details for model {model_name} from MLflow")
    
    # Log MLflow connection parameters for debugging
    logger.info(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow Username: {'Set' if MLFLOW_TRACKING_USERNAME else 'Not set'}")
    logger.info(f"MLflow Password: {'Set' if MLFLOW_TRACKING_PASSWORD else 'Not set'}")
    
    # Set MLflow connection
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        client = MlflowClient()
        
        # Get the latest version of the model
        logger.info(f"Attempting to get latest versions for model: {model_name}")
        versions = client.get_latest_versions(model_name)
        if not versions:
            logger.error(f"No versions found for model {model_name}")
            raise ValueError(f"Model {model_name} not found in MLflow")
            
        version = versions[0]
        logger.info(f"Found version {version.version} for model {model_name}")
        
        # Get the run to access parameters and tags
        run = client.get_run(version.run_id)
        logger.info(f"Retrieved run with ID: {run.info.run_id}")
        
        # Extract the relevant parameters
        params = run.data.params
        
        # Extract algorithm version from tags
        tags = run.data.tags
        algorithm_version = tags.get("version", "")
        logger.info(f"Algorithm version from tags: {algorithm_version}")
        
        # Get algorithm details from MLflow - only the core information
        algorithm_details = {
            "model_name": model_name,
            # Core Algorithm Information
            "image_name": params.get("image_name", ""),
            "image_tag": params.get("image_tag", ""),
            "source": params.get("source", ""),
            "is_gpu_enabled": params.get("is_gpu_enabled", "False").lower() == "true"
        }
        
        # Handle Python file paths
        if "python_file_train" in params:
            algorithm_details["python_file_train"] = params.get("python_file_train", "")
        if "python_file_inference" in params:
            algorithm_details["python_file_inference"] = params.get("python_file_inference", "")
        if "python_file_tune" in params:
            algorithm_details["python_file_tune"] = params.get("python_file_tune", "")
        if "python_file" in params:
            algorithm_details["python_file"] = params.get("python_file", "")
        
        # Create job details from config.yml (already expanded by load_config)
        job_details = {
            # Container settings
            "volumes": config.get("container", {}).get("volumes", []),
            "network": config.get("container", {}).get("network", ""),
            # Slurm settings
            "num_nodes": config.get("slurm", {}).get("num_nodes", 1),
            "partitions": config.get("slurm", {}).get("partitions", "[]"),
            "reservations": config.get("slurm", {}).get("reservations", "[]"),
            "max_time": config.get("slurm", {}).get("max_time", "1:00:00"),
            "submission_ssh_key": config.get("slurm", {}).get("submission_ssh_key", ""),
            "forward_ports": config.get("slurm", {}).get("forward_ports", "[]"),
            # SFAPI settings
            "sfapi_machine": config.get("sfapi", {}).get("machine", "perlmutter"),
            "sfapi_queue": config.get("sfapi", {}).get("queue", "realtime"),
            "sfapi_account": config.get("sfapi", {}).get("account", "als"),
            "sfapi_constraint": config.get("sfapi", {}).get("constraint", "cpu"),
            "sfapi_working_dir": config.get("sfapi", {}).get("working_dir", ""),
            "sfapi_output_dir": config.get("sfapi", {}).get("output_dir", ""),
            "sfapi_error_dir": config.get("sfapi", {}).get("error_dir", ""),
            "sfapi_exclusive": config.get("sfapi", {}).get("exclusive", True),
            # Get conda environment based on the model type and version from tags
            "conda_env": _get_conda_env_for_model(model_name, config, algorithm_version)
        }
        
        logger.info(f"Successfully retrieved details for model {model_name}")
        return algorithm_details, job_details
        
    except Exception as e:
        logger.error(f"Error retrieving algorithm details from MLflow: {str(e)}")
        
        # Print the full exception traceback for better debugging
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
