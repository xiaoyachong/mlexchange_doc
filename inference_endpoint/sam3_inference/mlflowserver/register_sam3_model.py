"""
Script to register SAM3 model to MLflow Model Registry

Usage:
    python scripts/register_sam3_model.py

Place this file in: scripts/register_sam3_model.py
"""
import mlflow
import os
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
SAM_MODEL_NAME = os.getenv("SAM_MODEL_NAME", "facebook/sam3")
HF_TOKEN = os.getenv("HF_TOKEN")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("=" * 80)
print("SAM3 Model Registration to MLflow")
print("=" * 80)
print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"SAM Model Name: {SAM_MODEL_NAME}")
print("=" * 80)

# Log the model
with mlflow.start_run(run_name="sam3-inference-service") as run:
    print("\n[1/4] Creating SAM3 model instance...")
    from utils.sam3_mlflow_model import SAM3MLflowModel
    
    sam3_model = SAM3MLflowModel()
    print("✓ Model instance created")
    
    print("\n[2/4] Defining conda environment...")
    # Define dependencies
    conda_env = {
        "channels": ["defaults", "conda-forge", "pytorch"],
        "dependencies": [
            "python=3.10",
            "pytorch>=2.0.0",
            "torchvision>=0.15.0",
            "pillow>=10.0.0",
            {
                "pip": [
                    "mlflow>=2.0.0",
                    "transformers @ git+https://github.com/huggingface/transformers.git",
                    "huggingface_hub",
                    "numpy<2.0.0",
                ]
            }
        ],
        "name": "sam3_env"
    }
    print("✓ Environment defined")
    
    print("\n[3/4] Logging model to MLflow...")
    # Log model with artifacts
    artifacts = {
        "model_name": SAM_MODEL_NAME,
    }
    
    if HF_TOKEN:
        artifacts["hf_token"] = HF_TOKEN
    
    mlflow.pyfunc.log_model(
        artifact_path="sam3_model",
        python_model=sam3_model,
        artifacts=artifacts,
        conda_env=conda_env,
        registered_model_name="sam3-inference",
        input_example={
            "image": "base64_encoded_image_string",
            "boxes": [[100, 100, 200, 200]],
            "threshold": 0.5,
            "mask_threshold": 0.5
        }
    )
    print("✓ Model logged")
    
    print("\n[4/4] Registration complete!")
    print("=" * 80)
    print(f"Run ID: {run.info.run_id}")
    print(f"Model URI: runs:/{run.info.run_id}/sam3_model")
    print(f"Registered Model: sam3-inference")
    print("=" * 80)
    
    print("\nTo serve this model, run:")
    print(f"mlflow models serve -m 'models:/sam3-inference/latest' -p 5001 --host 0.0.0.0")
    print("\nOr use the run URI directly:")
    print(f"mlflow models serve -m 'runs:/{run.info.run_id}/sam3_model' -p 5001 --host 0.0.0.0")
