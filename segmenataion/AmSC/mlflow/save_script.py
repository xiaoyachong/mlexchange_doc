"""
Register fine-tuned SAM3 and DINOv3 models to MLflow Model Registry.

Usage:
    python register_sam3_dino_mlflow.py

Env vars required (or set in .env):
    MLFLOW_TRACKING_URI_OUTSIDE
    MLFLOW_TRACKING_USERNAME
    MLFLOW_TRACKING_PASSWORD
    SAM3_ORIGINAL_CHECKPOINT   — path to facebook/sam3 base .pt
    SAM3_FINETUNED_CHECKPOINT  — path to your finetuned checkpoint.pt
    SAM3_BPE_PATH              — path to bpe_simple_vocab_16e6.txt.gz
    DINO_FINETUNED_CHECKPOINT  — path to your best.ckpt
"""

import os
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv

load_dotenv()

# ── MLflow connection ────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI      = os.getenv("MLFLOW_TRACKING_URI_OUTSIDE", "http://localhost:5000")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

# ── Checkpoint paths ─────────────────────────────────────────────────────────
SAM3_ORIGINAL_CHECKPOINT  = os.getenv("SAM3_ORIGINAL_CHECKPOINT")
SAM3_FINETUNED_CHECKPOINT = os.getenv("SAM3_FINETUNED_CHECKPOINT")
SAM3_BPE_PATH             = os.getenv("SAM3_BPE_PATH")
DINO_FINETUNED_CHECKPOINT = os.getenv("DINO_FINETUNED_CHECKPOINT")

# ── Skip model loading during registration ───────────────────────────────────
os.environ["MLFLOW_DISABLE_MODEL_LOADING"] = "true"

print("=" * 70)
print("SAM3 + DINOv3 Model Registration")
print("=" * 70)
print(f"MLflow URI          : {MLFLOW_TRACKING_URI}")
print(f"SAM3 original ckpt  : {SAM3_ORIGINAL_CHECKPOINT}")
print(f"SAM3 finetuned ckpt : {SAM3_FINETUNED_CHECKPOINT}")
print(f"SAM3 BPE path       : {SAM3_BPE_PATH}")
print(f"DINO finetuned ckpt : {DINO_FINETUNED_CHECKPOINT}")
print("=" * 70)

# ── Validate paths ───────────────────────────────────────────────────────────
missing = []
for name, path in [
    ("SAM3_ORIGINAL_CHECKPOINT",  SAM3_ORIGINAL_CHECKPOINT),
    ("SAM3_FINETUNED_CHECKPOINT", SAM3_FINETUNED_CHECKPOINT),
    ("SAM3_BPE_PATH",             SAM3_BPE_PATH),
    ("DINO_FINETUNED_CHECKPOINT", DINO_FINETUNED_CHECKPOINT),
]:
    if not path or not Path(path).exists():
        missing.append(name)

if missing:
    print(f"❌ Missing or non-existent paths: {missing}")
    sys.exit(1)

# ── Shared pip requirements ───────────────────────────────────────────────────
BASE_REQUIREMENTS = [
    "mlflow==2.22.0",
    "torch==2.2.2",
    "torchvision==0.17.2",
    "numpy<2.0.0",
    "Pillow",
]

SAM3_REQUIREMENTS = BASE_REQUIREMENTS + [
    "sam3",           # adjust to your actual package name / git URL
]

DINO_REQUIREMENTS = BASE_REQUIREMENTS + [
    "lightly-train",  # adjust to your actual package name / git URL
]

# ── Register SAM3 ─────────────────────────────────────────────────────────────
print("\n[1/2] Registering SAM3 model...")

from utils.sam3_wrapper import SAM3Wrapper

with mlflow.start_run(run_name="sam3-finetuned-registration") as run:
    mlflow.log_params({
        "original_checkpoint":  SAM3_ORIGINAL_CHECKPOINT,
        "finetuned_checkpoint": SAM3_FINETUNED_CHECKPOINT,
        "bpe_path":             SAM3_BPE_PATH,
    })
    mlflow.set_tags({
        "exp_type":   "live_mode",
        "model_type": "segmentation",
    })

    mlflow.pyfunc.log_model(
        artifact_path="sam3_model",
        python_model=SAM3Wrapper(),
        artifacts={
            "original_checkpoint":  SAM3_ORIGINAL_CHECKPOINT,
            "finetuned_checkpoint": SAM3_FINETUNED_CHECKPOINT,
            "bpe_path":             SAM3_BPE_PATH,
        },
        pip_requirements=SAM3_REQUIREMENTS,
        conda_env=None,
        registered_model_name="sam3-finetuned",
        code_paths=["utils"],
        signature=None,
    )

    print(f"✓ SAM3 registered  |  run_id: {run.info.run_id}")

# ── Register DINO ─────────────────────────────────────────────────────────────
print("\n[2/2] Registering DINOv3 model...")

from utils.dino_wrapper import DinoWrapper

with mlflow.start_run(run_name="dino-finetuned-registration") as run:
    mlflow.log_params({
        "finetuned_checkpoint": DINO_FINETUNED_CHECKPOINT,
    })
    mlflow.set_tags({
        "exp_type":   "live_mode",
        "model_type": "segmentation",
    })

    mlflow.pyfunc.log_model(
        artifact_path="dino_model",
        python_model=DinoWrapper(),
        artifacts={
            "checkpoint": DINO_FINETUNED_CHECKPOINT,
        },
        pip_requirements=DINO_REQUIREMENTS,
        conda_env=None,
        registered_model_name="dino-finetuned",
        code_paths=["utils"],
        signature=None,
    )

    print(f"✓ DINO registered  |  run_id: {run.info.run_id}")

print("\n" + "=" * 70)
print("Registration complete.")
print("=" * 70)
print("\nTo serve SAM3:")
print("  mlflow models serve -m 'models:/sam3-finetuned/latest' -p 5001 --host 0.0.0.0 --env-manager local --no-conda")
print("\nTo serve DINO:")
print("  mlflow models serve -m 'models:/dino-finetuned/latest' -p 5002 --host 0.0.0.0 --env-manager local --no-conda")
