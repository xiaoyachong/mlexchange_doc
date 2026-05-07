# Setup & Usage
# =============
# Files in this directory:
#   config.yaml                — all configuration (edit this first)
#   lightly_mlflow_wrapper.py  — MLflow pyfunc wrapper class (inference)
#   save_mlflow_wrapper.py     — registers initial checkpoint to MLflow
#   finetune.py                — downloads model, reads Tiled, finetunes, re-registers
#   requirements.txt           — deps for save_mlflow_wrapper.py  (mlflow only)
#   requirements_finetune.txt  — deps for finetune.py (torch, lightly-train, tiled, ...)

# ---------------------------------------------------------------------------
# 1. Edit config.yaml before running anything
# ---------------------------------------------------------------------------
# Key fields to set:
#   mlflow.tracking_uri       — your MLflow server URL
#   checkpoint.path           — path to best.ckpt from lightly_train
#   checkpoint.base_model     — lightly_train model name (e.g. dinov3/vitl16-eomt-cityscapes)
#   tiled.uri                 — your Tiled server URL
#   scratch.data_dir          — shared filesystem path visible to all SLURM nodes

# ---------------------------------------------------------------------------
# 2. Register the initial lightly_train checkpoint to MLflow
#    Env: mlflow_client  (lightweight — mlflow only, no torch needed)
# ---------------------------------------------------------------------------

conda create -n mlflow_client python=3.11 -y
conda activate mlflow_client
pip install -r requirements.txt

python save_mlflow_wrapper.py --config config.yaml

# ---------------------------------------------------------------------------
# 3. Run finetuning
#    Env: lightly_finetune  (heavy — torch, lightly-train, tiled, etc.)
# ---------------------------------------------------------------------------

conda create -n lightly_finetune python=3.11 -y
conda activate lightly_finetune
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_finetune.txt

# --- Single GPU ---
python finetune.py --config config.yaml

# --- SLURM single node, multiple GPUs (e.g. 4 GPUs) ---
srun --nodes=1 --ntasks-per-node=4 --gpus-per-task=1 \
    python finetune.py --config config.yaml

# --- SLURM multi-node (e.g. 2 nodes x 4 GPUs each) ---
srun --nodes=2 --ntasks-per-node=4 --gpus-per-task=1 \
    python finetune.py --config config.yaml
