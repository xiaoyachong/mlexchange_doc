# Setup & Usage
# =============
# Files in this directory:
#   config_finetune.yaml       — configuration for finetune.py
#   config_register.yaml       — configuration for save_mlflow_wrapper.py
#   lightly_mlflow_wrapper.py  — MLflow pyfunc wrapper class (inference)
#   save_mlflow_wrapper.py     — registers initial checkpoint to MLflow
#   finetune.py                — downloads model, reads Tiled, finetunes, re-registers
#   requirements.txt           — deps for save_mlflow_wrapper.py  (mlflow only)
#   requirements_finetune.txt  — deps for finetune.py (torch, lightly-train, tiled, ...)

# ---------------------------------------------------------------------------
# 1. Set up your .env file
# ---------------------------------------------------------------------------
# Create a .env file in the project root with the following variables:

# TILED_API_KEY=your_tiled_api_key
# DATA_TILED_URI_IMAGES=http://tiled:8000/api/v1/metadata/images
# DATA_TILED_URI_MASKS=http://tiled:8000/api/v1/metadata/masks
# MLFLOW_TRACKING_URI_OUTSIDE=http://localhost:5000
# MLFLOW_TRACKING_USERNAME=your_username
# MLFLOW_TRACKING_PASSWORD=your_password

# ---------------------------------------------------------------------------
# 2. Ingest images and masks into local Tiled
# ---------------------------------------------------------------------------

conda activate seg
source .env
docker compose exec tiled \
    env TILED_API_KEY=$TILED_API_KEY \
    tiled register \
    http://localhost:8000 \
    /tiled_storage

# Verify the keys are registered correctly
python -c "
from tiled.client import from_uri
import os
images = from_uri(os.getenv('DATA_TILED_URI_IMAGES'), api_key=os.getenv('TILED_API_KEY'))
masks  = from_uri(os.getenv('DATA_TILED_URI_MASKS'),  api_key=os.getenv('TILED_API_KEY'))
print('images:', list(images))
print('masks: ', list(masks))
"

# ---------------------------------------------------------------------------
# 3. Edit config_finetune.yaml
# ---------------------------------------------------------------------------
# Key fields to set:
#   checkpoint.base_model                  — lightly_train pretrained model name
#                                            downloaded automatically to cache dir
#   cache.lightly_train_cache_dir          — path to cache lightly_train internals
#   cache.lightly_train_model_cache_dir    — path to cache pretrained model weights
#   cache.torch_home                       — path to cache torch weights
#   tiled.images_uri                       — images container URI
#                                            (or set DATA_TILED_URI_IMAGES in .env)
#   tiled.masks_uri                        — masks container URI
#                                            (or set DATA_TILED_URI_MASKS in .env)
#   tiled.api_key                          — Tiled API key (or set TILED_API_KEY in .env)
#   finetune.out_dir                       — output dir; best.ckpt → out_dir/checkpoints/best.ckpt
#   mlflow.tracking_uri                    — MLflow server URL
#                                            (or set MLFLOW_TRACKING_URI_OUTSIDE in .env)
#   scratch.data_dir                       — shared filesystem path visible to all SLURM nodes
#
# Edit config_register.yaml
# ---------------------------------------------------------------------------
# Key fields to set:
#   mlflow.tracking_uri                    — MLflow server URL
#                                            (or set MLFLOW_TRACKING_URI_OUTSIDE in .env)
#   mlflow.base_model_name                 — name to register the model under
#   checkpoint.base_model                  — lightly_train model name (for traceability)
#   finetune.out_dir                       — must match finetune config out_dir

# ---------------------------------------------------------------------------
# 4. Run finetuning  (downloads pretrained weights automatically to cache)
#    Env: lightly_finetune  (heavy — torch, lightly-train, tiled, etc.)
# ---------------------------------------------------------------------------

conda create -n lightly_finetune python=3.11 -y
conda activate lightly_finetune
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_finetune.txt

source .env

# --- Single GPU ---
python finetune.py --config config_finetune.yaml

# --- SLURM single node, multiple GPUs (e.g. 4 GPUs) ---
srun --nodes=1 --ntasks-per-node=4 --gpus-per-task=1 \
    python finetune.py --config config_finetune.yaml

# --- SLURM multi-node (e.g. 2 nodes x 4 GPUs each) ---
srun --nodes=2 --ntasks-per-node=4 --gpus-per-task=1 \
    python finetune.py --config config_finetune.yaml

# After finetuning, the best checkpoint is saved to:
#   <finetune.out_dir>/checkpoints/best.ckpt

# ---------------------------------------------------------------------------
# 5. Register the finetuned checkpoint to MLflow
#    Env: mlflow_client  (lightweight — mlflow only, no torch needed)
# ---------------------------------------------------------------------------

conda create -n mlflow_client python=3.11 -y
conda activate mlflow_client
pip install -r requirements.txt

source .env
python save_mlflow_wrapper.py --config config_register.yaml
