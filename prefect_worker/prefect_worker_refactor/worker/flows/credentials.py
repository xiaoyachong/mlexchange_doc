"""
Credential management utilities for child flows.
This module has minimal dependencies and can be imported by all flow types.
"""
import os
from dotenv import load_dotenv

# Load .env file at module import
load_dotenv()

# MLflow connection parameters - load from environment variables
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

# Tiled API keys - load from environment variables
DATA_TILED_API_KEY = os.getenv("DATA_TILED_API_KEY", "")
MASK_TILED_API_KEY = os.getenv("MASK_TILED_API_KEY", "")
SEG_TILED_API_KEY = os.getenv("SEG_TILED_API_KEY", "")
RESULTS_TILED_API_KEY = os.getenv("RESULTS_TILED_API_KEY", "")


def add_credentials_to_io_parameters(params: dict) -> dict:
    """
    Add credentials to io_parameters that were removed from the application.
    This function intelligently adds only the credentials that are needed based on
    which URIs are present in the io_parameters.
    
    Args:
        params: Parameters dictionary containing io_parameters
        
    Returns:
        Updated parameters dictionary with credentials added
    """
    if "io_parameters" not in params:
        params["io_parameters"] = {}
    
    io_params = params["io_parameters"]
    
    # Mapping of URI keys to their corresponding API key environment variables
    tiled_uri_to_key_mapping = {
        "data_tiled_uri": ("data_tiled_api_key", DATA_TILED_API_KEY),
        "mask_tiled_uri": ("mask_tiled_api_key", MASK_TILED_API_KEY),
        "seg_tiled_uri": ("seg_tiled_api_key", SEG_TILED_API_KEY),
        "results_tiled_uri": ("results_tiled_api_key", RESULTS_TILED_API_KEY),
    }
    
    # Add Tiled API keys only if corresponding URI exists and key is not already set
    for uri_key, (api_key_name, api_key_value) in tiled_uri_to_key_mapping.items():
        if uri_key in io_params and api_key_name not in io_params:
            io_params[api_key_name] = api_key_value
    
    # Add MLflow credentials only if not already present
    if "mlflow_tracking_username" not in io_params:
        io_params["mlflow_tracking_username"] = MLFLOW_TRACKING_USERNAME
    if "mlflow_tracking_password" not in io_params:
        io_params["mlflow_tracking_password"] = MLFLOW_TRACKING_PASSWORD
    
    return params