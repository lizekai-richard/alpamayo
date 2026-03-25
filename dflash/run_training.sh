#!/bin/bash
# Train DFlash from scratch for Alpamayo VLM
#
# This script creates a NEW DFlash model with Alpamayo's vocabulary (155,698 tokens)
# and trains it using pre-computed distillation data.
#
# Key features:
# - Creates DFlash from scratch (not loading pre-trained weights)
# - Uses Alpamayo's vocab_size (155,697 + 1 MASK token)
# - Loads embed_tokens and lm_head from Alpamayo
# - Supports trainable or frozen embeddings

set -e

# ============== Configuration ==============
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPUS_PER_NODE=4

# Model paths
TARGET_MODEL="nvidia/Alpamayo-1.5-10B"
DATA_DIR="/mnt/moosefs-1/users/zekail/dflash_train/b8"
OUTPUT_DIR="/mnt/moosefs-1/users/zekail/exp"

# Training hyperparameters
NUM_EPOCHS=15
BATCH_SIZE=64           # Per GPU
LEARNING_RATE=3e-4      # Will be scaled by world size
BLOCK_SIZE=8            # Match your distillation data
NUM_DRAFT_LAYERS=2      # Number of draft decoder layers

# Layer selection (must be subset of stored layers during data generation)
# Stored layers: 0,16,20,24,28,30,31,32,34,35
# Default training layers: 24,30,31,32,34
TARGET_LAYERS="24,30,31,32,34"

# Loss configuration
PREFIX_WEIGHT_GAMMA=3  # Geometric decay for prefix-weighted CE

# Performance
USE_COMPILE=true       # Use torch.compile for faster training

# ============== Run Training ==============
echo "=============================================="
echo "Training DFlash from Scratch for Alpamayo"
echo "=============================================="
echo "Target model: $TARGET_MODEL"
echo "Data dir: $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Target layers: [$TARGET_LAYERS]"
echo "Use compile: $USE_COMPILE"
echo ""

TRAIN_ARGS=(
    --target-model "$TARGET_MODEL"
    --data-dir "$DATA_DIR"
    --output-dir "$OUTPUT_DIR"
    --num-epochs $NUM_EPOCHS
    --batch-size $BATCH_SIZE
    --learning-rate $LEARNING_RATE
    --block-size $BLOCK_SIZE
    --num-draft-layers $NUM_DRAFT_LAYERS
    --target-layers "$TARGET_LAYERS"
    --prefix-weight-gamma $PREFIX_WEIGHT_GAMMA
    --num-workers 0
)

if [ "$USE_COMPILE" = true ]; then
    TRAIN_ARGS+=(--compile)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    "$SCRIPT_DIR/train_dflash.py" \
    "${TRAIN_ARGS[@]}"

echo ""
echo "Training complete!"
