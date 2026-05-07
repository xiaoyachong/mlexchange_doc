"""
save_mlflow_wrapper.py
======================
Reads config.yaml, wraps the lightly_train checkpoint in LightlySegWrapper,
and registers it to the MLflow Model Registry.

Usage:
    python save_mlflow_wrapper.py
    python save_mlflow_wrapper.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import mlflow
import yaml
from dotenv import load_dotenv

from lightly_mlflow_wrapper import LightlySegWrapper

load_dotenv(dotenv_path="../.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config_register.yaml") -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Allow env-var overrides for sensitive fields
    cfg["mlflow"]["tracking_uri"] = os.getenv(
        "MLFLOW_TRACKING_URI_OUTSIDE", cfg["mlflow"]["tracking_uri"]
    )
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    return cfg


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_to_mlflow(cfg: dict) -> tuple[str | None, str | None]:
    """
    Wraps the checkpoint in LightlySegWrapper and registers it to MLflow.

    Returns:
        (model_name, run_id) on success, (None, None) on failure.
    """
    tracking_uri     = cfg["mlflow"]["tracking_uri"]
    experiment_name  = cfg["mlflow"]["experiment_name"]
    model_name       = cfg["mlflow"]["base_model_name"]
    base_model       = cfg["checkpoint"]["base_model"]
    pip_requirements = cfg["pip_requirements"]

    # Resolve final out_dir = out_dir / base_model  (same logic as finetune.py)
    out_dir         = Path(cfg["finetune"]["out_dir"]) / base_model
    checkpoint_path = out_dir / "checkpoints" / "best.ckpt"

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Run finetune.py first to generate the checkpoint.")
        return None, None

    ckpt_size_mb = checkpoint_path.stat().st_size / 1024 / 1024
    logger.info(f"Checkpoint: {checkpoint_path} ({ckpt_size_mb:.1f} MB)")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_name = f"register_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        try:
            mlflow.log_params({
                "model_name":      model_name,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_mb":   round(ckpt_size_mb, 2),
                "task":            "semantic_segmentation",
                "base_model":      base_model,
            })
            mlflow.set_tags({
                "task":      "semantic_segmentation",
                "framework": "lightly_train",
            })

            t0 = time.time()
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=LightlySegWrapper(),
                artifacts={"checkpoint": str(checkpoint_path)},
                registered_model_name=model_name,
                pip_requirements=pip_requirements,
                code_path=["lightly_mlflow_wrapper.py"],
            )
            mlflow.log_metric("registration_time_s", time.time() - t0)

            logger.info(f"✅ Registered '{model_name}' (run={run.info.run_id})")
            return model_name, run.info.run_id

        except Exception:
            logger.error("Registration failed:")
            traceback.print_exc()
            return None, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_register.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    name, run_id = register_to_mlflow(cfg)

    if name:
        print(f"\n✅ Model '{name}' registered (run_id={run_id})")
        print(f"   MLflow UI: {cfg['mlflow']['tracking_uri']}")
    else:
        print("\n❌ Registration failed — check logs above.")
