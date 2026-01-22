#!/bin/bash
#SBATCH -C gpu
#SBATCH -A m4880_g
#SBATCH -q regular
#SBATCH --nodes=2                    # CHANGE THIS: Number of nodes (1, 2, 4, 8, etc.)
#SBATCH --gpus-per-node=4            # Perlmutter has 4 GPUs per node
#SBATCH --cpus-per-task=128          # 128 CPUs per node on Perlmutter
#SBATCH --time=24:00:00              # CHANGE THIS: Max runtime
#SBATCH -o logs/inference_%j.out
#SBATCH -e logs/inference_%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=xchong@lbl.gov

# =============================================================================
# NERSC Perlmutter Configuration
# =============================================================================

# Conda environment
CONDA_ENV="/pscratch/sd/x/xchong/sam3_finetune/sam3_seg"

# Data paths
INPUT_DIR="/pscratch/sd/x/xchong/sam3_finetune/sample_data"
OUTPUT_DIR="/pscratch/sd/x/xchong/sam3_finetune/results"

# Inference parameters
PATCH_SIZE=512
BATCH_SIZE=1                         # Batch size per GPU
NUM_WORKERS=4
CONFIDENCE=0.5

# Prompts (space-separated, in quotes)
PROMPTS="cortex Phloem_Fibers Xylem_vessels Pith_cells outer_cells"

# =============================================================================
# DO NOT EDIT BELOW THIS LINE
# =============================================================================

# Print job info
echo "=================================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Node list: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "Node: $(hostname)"
echo "=================================================="

# Create logs directory
mkdir -p logs

# Load modules (NERSC specific)
module load python

# Activate conda environment
source /global/common/software/nersc/pe/conda/24.10.0/Miniforge3-24.7.1-0/bin/activate $CONDA_ENV

# IMPORTANT: Unset these to prevent SLURM conflicts with PyTorch DDP
unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE

# Display GPU info
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --list-gpus

# Get master node address
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

# Number of GPUs per node (always 4 on Perlmutter GPU nodes)
GPUS_PER_NODE=4

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $(($SLURM_NNODES * $GPUS_PER_NODE))"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Effective batch size: $(($BATCH_SIZE * $SLURM_NNODES * $GPUS_PER_NODE))"
echo "=================================================="

# Run distributed inference
srun torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m src.inference \
    --input-dir $INPUT_DIR \
    --output-dir $OUTPUT_DIR \
    --patch-size $PATCH_SIZE \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --confidence $CONFIDENCE \
    --prompts $PROMPTS

# Check exit status
EXIT_CODE=$?

echo "=================================================="
echo "Job finished at: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS"
else
    echo "Status: FAILED (exit code: $EXIT_CODE)"
fi
echo "=================================================="

exit $EXIT_CODE
