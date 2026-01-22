#!/bin/bash
# run_single_node.sh
# Single node with 4 GPUs
# Usage: bash run_single_node.sh

module load conda
conda activate /pscratch/sd/t/tachavez/SYNAPS/forge_feb_seg_model_demo/sam3_seg

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    -m src.inference \
    --input-dir /pscratch/sd/t/tachavez/SYNAPS/forge_feb_seg_model_demo/sample_data \
    --output-dir /pscratch/sd/t/tachavez/SYNAPS/forge_feb_seg_model_demo/results \
    --patch-size 512 \
    --batch-size 1 \
    --num-workers 4 \
    --confidence 0.5 \
    --prompts "background" "cell"
