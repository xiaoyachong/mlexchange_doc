"""
finetune.py
===========
1. Reads config.yaml
2. Downloads the registered lightly_train checkpoint from MLflow
3. Reads images + masks from a Tiled server, writes them as PNG to scratch
4. Calls lightly_train.train_semantic_segmentation() — fully official API,
   supports multi-node / multi-GPU via Lightning Fabric + DDP
5. Registers the finetuned checkpoint back to MLflow

Usage — single GPU:
    python finetune.py

Usage — SLURM multi-node (e.g. 2 nodes × 4 GPUs):
    srun --nodes=2 --ntasks-per-node=4 python finetune.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import lightly_train
import mlflow
import numpy as np
import yaml
from dotenv import load_dotenv
from PIL import Image
from tiled.client import from_uri

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

def load_config(config_path: str = "config_finetune.yaml") -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Env-var overrides for sensitive / cluster-specific fields
    cfg["mlflow"]["tracking_uri"] = os.getenv(
        "MLFLOW_TRACKING_URI_OUTSIDE", cfg["mlflow"]["tracking_uri"]
    )
    tiled = cfg["tiled"]
    tiled["images_uri"] = os.getenv("DATA_TILED_URI_IMAGES", tiled.get("images_uri", ""))
    tiled["masks_uri"]  = os.getenv("DATA_TILED_URI_MASKS",  tiled.get("masks_uri",  ""))
    tiled["api_key"]    = os.getenv("TILED_API_KEY",         tiled.get("api_key"))

    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

    # Resolve scratch paths — expand ${SCRATCH} if present
    scratch_default = os.getenv("SCRATCH", tempfile.gettempdir())
    for key in ("data_dir",):
        raw = cfg["scratch"][key]
        cfg["scratch"][key] = raw.replace("${SCRATCH}", scratch_default)

    # num_nodes: prefer SLURM_NNODES at runtime
    cfg["finetune"]["num_nodes"] = int(
        os.environ.get("SLURM_NNODES", cfg["finetune"]["num_nodes"])
    )

    # Set lightly_train / torch cache directories
    cache = cfg["cache"]
    os.environ["LIGHTLY_TRAIN_CACHE_DIR"]       = cache["lightly_train_cache_dir"]
    os.environ["LIGHTLY_TRAIN_MODEL_CACHE_DIR"] = cache["lightly_train_model_cache_dir"]
    os.environ["TORCH_HOME"]                    = cache["torch_home"]

    return cfg


# ---------------------------------------------------------------------------
# Step 1 — resolve base model
# ---------------------------------------------------------------------------

def get_base_model(cfg: dict) -> str:
    """
    Returns the lightly_train model name from config.
    lightly_train will download the pretrained weights automatically to
    LIGHTLY_TRAIN_MODEL_CACHE_DIR if not already cached.
    """
    base_model = cfg["checkpoint"]["base_model"]
    logger.info(f"Base model: {base_model}")
    logger.info(f"Cache dir : {cfg['cache']['lightly_train_model_cache_dir']}")
    return base_model


# ---------------------------------------------------------------------------
# Step 2 — read from Tiled, write PNG files to shared scratch
# ---------------------------------------------------------------------------

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.squeeze(arr)
    if arr.dtype == np.uint8:
        return arr
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        return ((arr.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
    return np.zeros_like(arr, dtype=np.uint8)


def _save_split(tiled_images, tiled_masks, indices: list[int],
                img_dir: Path, msk_dir: Path) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    for i, idx in enumerate(indices):
        fname = f"{i:06d}.png"
        # image
        img = _to_uint8(np.array(tiled_images[idx]))
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.concatenate([img] * 3, axis=-1)
        Image.fromarray(img).save(img_dir / fname)
        # mask — preserve class indices as uint8
        msk = np.squeeze(np.array(tiled_masks[idx])).astype(np.uint8)
        Image.fromarray(msk).save(msk_dir / fname)
        if (i + 1) % 50 == 0:
            logger.info(f"  {i + 1}/{len(indices)} written ...")


def prepare_data(cfg: dict) -> tuple[str, str, str, str]:
    """
    Connects to Tiled, writes train + val splits to disk.
    Returns (train_img_dir, train_msk_dir, val_img_dir, val_msk_dir).
    """
    tiled_cfg = cfg["tiled"]
    api_key   = tiled_cfg["api_key"]
    logger.info(f"Connecting to images container: {tiled_cfg['images_uri']}")
    tiled_images = from_uri(tiled_cfg["images_uri"], api_key=api_key)
    logger.info(f"Connecting to masks container : {tiled_cfg['masks_uri']}")
    tiled_masks  = from_uri(tiled_cfg["masks_uri"],  api_key=api_key)

    tmp_root = Path(cfg["scratch"]["data_dir"])

    train_idx = list(range(
        cfg["tiled"]["train_indices"]["start"],
        cfg["tiled"]["train_indices"]["end"],
    ))
    val_idx = list(range(
        cfg["tiled"]["val_indices"]["start"],
        cfg["tiled"]["val_indices"]["end"],
    ))

    logger.info(f"Saving {len(train_idx)} train samples ...")
    tr_img = tmp_root / "train" / "images"
    tr_msk = tmp_root / "train" / "masks"
    _save_split(tiled_images, tiled_masks, train_idx, tr_img, tr_msk)

    logger.info(f"Saving {len(val_idx)} val samples ...")
    va_img = tmp_root / "val" / "images"
    va_msk = tmp_root / "val" / "masks"
    _save_split(tiled_images, tiled_masks, val_idx, va_img, va_msk)

    # Sentinel so other SLURM ranks know data is ready
    (tmp_root / ".data_ready").touch()
    logger.info("Data written to disk.")
    return str(tr_img), str(tr_msk), str(va_img), str(va_msk)


def wait_for_data(cfg: dict) -> tuple[str, str, str, str]:
    """Non-rank-0 ranks wait for the sentinel then return the same paths."""
    tmp_root = Path(cfg["scratch"]["data_dir"])
    sentinel = tmp_root / ".data_ready"
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 1)))
    logger.info(f"Rank {rank} waiting for data ...")
    while not sentinel.exists():
        time.sleep(2)
    return (
        str(tmp_root / "train" / "images"),
        str(tmp_root / "train" / "masks"),
        str(tmp_root / "val"   / "images"),
        str(tmp_root / "val"   / "masks"),
    )


# ---------------------------------------------------------------------------
# Step 3 — finetune with the official lightly_train API
# ---------------------------------------------------------------------------

def finetune(
    cfg: dict,
    base_model: str,
    tr_img: str,
    tr_msk: str,
    va_img: str,
    va_msk: str,
) -> Path:
    """
    Calls lightly_train.train_semantic_segmentation() and returns the path
    to the best checkpoint saved under cfg["finetune"]["out_dir"].
    lightly_train downloads the pretrained weights automatically if not cached.
    """
    ft      = cfg["finetune"]
    dataset = cfg["dataset"]

    logger.info("=" * 60)
    logger.info("Starting finetuning")
    logger.info(f"  base_model : {base_model}")
    logger.info(f"  out        : {ft['out_dir']}")
    logger.info(f"  steps      : {ft['steps']}")
    logger.info(f"  batch_size : {ft['batch_size']}")
    logger.info(f"  num_nodes  : {ft['num_nodes']}")
    logger.info(f"  devices    : {ft['devices']}")
    logger.info("=" * 60)

    lightly_train.train_semantic_segmentation(
        out=ft["out_dir"],
        # lightly_train downloads pretrained weights to LIGHTLY_TRAIN_MODEL_CACHE_DIR
        # if not already present, then finetunes from them.
        model=base_model,
        steps=ft["steps"],
        devices=ft["devices"],
        num_nodes=ft["num_nodes"],
        batch_size=ft["batch_size"],
        data={
            "train":          {"images": tr_img, "masks": tr_msk},
            "val":            {"images": va_img, "masks": va_msk},
            "classes":        {int(k): v for k, v in dataset["classes"].items()},
            "ignore_classes": dataset["ignore_classes"],
        },
        logger_args={
            "log_every_num_steps":     ft["log_every_num_steps"],
            "val_every_num_steps":     ft["val_every_num_steps"],
            "val_log_every_num_steps": ft["val_log_every_num_steps"],
        },
        save_checkpoint_args={
            "save_every_num_steps": ft["save_every_num_steps"],
            "save_last":            ft["save_last"],
            "save_best":            ft["save_best"],
        },
    )

    best_ckpt = Path(ft["out_dir"]) / "checkpoints" / "best.ckpt"
    logger.info(f"Finetuning complete.")
    logger.info(f"Best checkpoint saved to: {best_ckpt}")
    return best_ckpt


# ---------------------------------------------------------------------------
# Step 4 — register finetuned model back to MLflow
# ---------------------------------------------------------------------------

def register_finetuned(cfg: dict, best_ckpt: Path) -> None:
    tracking_uri    = cfg["mlflow"]["tracking_uri"]
    experiment_name = cfg["mlflow"]["experiment_name"]
    model_name       = cfg["mlflow"]["finetuned_model_name"]
    base_model_arch  = cfg["checkpoint"]["base_model"]
    pip_requirements = cfg["pip_requirements"]
    ft               = cfg["finetune"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_name = f"finetune_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        try:
            mlflow.log_params({
                "base_model_arch":  base_model_arch,
                "finetuned_model":  model_name,
                "steps":            ft["steps"],
                "batch_size":       ft["batch_size"],
                "num_nodes":        ft["num_nodes"],
            })
            mlflow.set_tags({
                "task":            "semantic_segmentation",
                "framework":       "lightly_train",
                "finetune":        "true",
                "base_model_arch": base_model_arch,
            })

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=LightlySegWrapper(),
                artifacts={"checkpoint": str(best_ckpt)},
                registered_model_name=model_name,
                pip_requirements=pip_requirements,
                code_path=["lightly_mlflow_wrapper.py"],
            )

            logger.info(
                f"✅ Registered '{model_name}' (run={run.info.run_id})"
            )
        except Exception:
            logger.error("Registration of finetuned model failed:")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_finetune.yaml")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))

    # ------------------------------------------------------------------
    # Rank 0: download checkpoint + dump Tiled data to shared scratch
    # Other ranks: wait for sentinel, then pick up the same paths
    # ------------------------------------------------------------------
    # base_model is just a string — same on all ranks, no download needed here
    base_model = get_base_model(cfg)

    if rank == 0:
        tr_img, tr_msk, va_img, va_msk = prepare_data(cfg)
    else:
        tr_img, tr_msk, va_img, va_msk = wait_for_data(cfg)

    # ------------------------------------------------------------------
    # All ranks finetune together (DDP across nodes / GPUs)
    # lightly_train downloads pretrained weights on rank 0 and caches them
    # ------------------------------------------------------------------
    best_ckpt = finetune(cfg, base_model, tr_img, tr_msk, va_img, va_msk)

    # ------------------------------------------------------------------
    # Rank 0: register the finetuned model back to MLflow
    # ------------------------------------------------------------------
    if rank == 0:
        register_finetuned(cfg, best_ckpt)
