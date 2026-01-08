"""
MLflow Model Saving Utility for Segmentation Models

This script saves segmentation models to MLflow with PyFunc wrappers.
The model types are determined by the configuration.
"""

import os
import sys
import traceback
import yaml
from pathlib import Path

# Fix transformers compatibility BEFORE any imports
os.environ["TRANSFORMERS_USE_TORCH_EXPORT"] = "0"
os.environ['MLFLOW_ARTIFACT_ROOT'] = os.path.expanduser('~/mlflow_artifacts')

import mlflow
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv(dotenv_path="../.env")

# Load MLflow configuration from YAML
CONFIG_PATH = Path(__file__).parent / "segmentation_models_config.yaml"
config = {}
try:
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
    print("✅ Loaded MLflow configuration from YAML")
except Exception as e:
    print(f"⚠️ Error loading configuration: {e}")
    sys.exit(1)

# Import wrapper
from segmentation_wrapper import save_segmentation_model_with_wrapper

# Get dataset and network type from config
DATASET = config.get("common", {}).get("dataset", "default").lower()
NETWORK_TYPE = config.get("common", {}).get("network_type", "DLSIA TUNet").upper()

# MLflow Configuration from environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI_OUTSIDE", "http://localhost:5000")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
MLFLOW_TRACKING_INSECURE_TLS = os.getenv("MLFLOW_TRACKING_INSECURE_TLS", "")

# Set MLflow authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD
os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = MLFLOW_TRACKING_INSECURE_TLS

# Load model-specific configuration based on dataset and network type
model_config_key = f"{DATASET}_{NETWORK_TYPE.replace(' ', '_').lower()}"
model_config = config.get("models", {}).get(model_config_key, {})

if model_config:
    # For the selected model, use values from YAML config
    WEIGHTS_PATH = model_config.get("weights_path")
    QLTY_WINDOW = model_config.get("qlty_window", 64)
    IMAGE_SIZE = tuple(model_config.get("image_size", [QLTY_WINDOW, QLTY_WINDOW]))
    NUM_CLASSES = model_config.get("num_classes", 2)
    IN_CHANNELS = model_config.get("in_channels", 1)
    MLFLOW_EXPERIMENT_NAME = model_config.get("experiment_name")
    MLFLOW_MODEL_NAME = model_config.get("model_name")
    NETWORK_PARAMS = model_config.get("network_params", {})
else:
    print(f"❌ Error: No configuration found for {model_config_key} in the YAML file.")
    sys.exit(1)

# Configure segmentation model
MODEL_CONFIG = {
    "name": "Segmentation_Model",
    "weights_path": WEIGHTS_PATH,
    "network_type": NETWORK_TYPE,
    "num_classes": NUM_CLASSES,
    "in_channels": IN_CHANNELS,
    "image_shape": IMAGE_SIZE,
    "network_params": NETWORK_PARAMS,
}

# Print configuration for verification
print("----------------------------------------------")
print("MLFLOW_TRACKING_URI:", MLFLOW_TRACKING_URI)
print("MLFLOW_TRACKING_USERNAME:", MLFLOW_TRACKING_USERNAME)
print("DATASET:", DATASET)
print("NETWORK_TYPE:", NETWORK_TYPE)
print("MLFLOW_EXPERIMENT_NAME:", MLFLOW_EXPERIMENT_NAME)
print("MLFLOW_MODEL_NAME:", MLFLOW_MODEL_NAME)
print("WEIGHTS_PATH:", WEIGHTS_PATH)
print("NUM_CLASSES:", NUM_CLASSES)
print("IN_CHANNELS:", IN_CHANNELS)
print("IMAGE_SIZE:", IMAGE_SIZE)
print("NETWORK_PARAMS:", NETWORK_PARAMS)
print("----------------------------------------------")

if __name__ == "__main__":
    try:
        # Check if MLflow server is accessible
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.search_experiments()
            print(f"✅ MLflow server accessible at {MLFLOW_TRACKING_URI}")
        except Exception as e:
            print(f"⚠️  Cannot connect to MLflow server at {MLFLOW_TRACKING_URI}: {e}")
            sys.exit(1)

        print(f"\nSaving segmentation model: {MLFLOW_MODEL_NAME}")

        # Initialize success tracker
        model_success = False

        # Save segmentation model with PyFunc wrapper
        if WEIGHTS_PATH:
            model_name, model_run_id = save_segmentation_model_with_wrapper(
                MODEL_CONFIG,
                MLFLOW_TRACKING_URI,
                MLFLOW_EXPERIMENT_NAME,
                MLFLOW_MODEL_NAME,
            )
            model_success = bool(model_name)
        else:
            print(f"\n⚠️ Skipping segmentation model: missing weights path")

        # Report results
        print("\n---------- SUMMARY ----------")
        
        if model_success:
            print(f"\n✅ Segmentation model saved successfully!")
        else:
            print(f"\n❌ Failed to save segmentation model.")

    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
