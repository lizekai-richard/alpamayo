nani#!/bin/bash
# Train Alpamayo-R1 VLM stage with 8-GPU FSDP.
#
# Usage:
#   bash scripts/train_vlm.sh
#   bash scripts/train_vlm.sh num_epochs=3 lr=5e-5   # override any config key

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------- Configurable via env vars ----------
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/train_expert.yaml}"
DATA_DIR="${DATA_DIR:-/mnt/moosefs-1/users/zekail/dumped_inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/checkpoints_expert}"

# ---------- Setup ----------
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Alpamayo-R1 Expert Training (DDP, ${NUM_GPUS} GPUs)"
echo "=============================================="
echo "Config:       $CONFIG"
echo "Data dir:     $DATA_DIR"
echo "Output dir:   $OUTPUT_DIR"
echo "Extra args:   $*"
echo "=============================================="
echo ""

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    -m alpamayo_r1.train.train \
    --config "$CONFIG" \
    data_dir="$DATA_DIR" \
    output_dir="$OUTPUT_DIR" \
    "$@"
