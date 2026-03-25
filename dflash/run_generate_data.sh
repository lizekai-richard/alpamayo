#!/bin/bash
# Generate distillation data on 8 GPUs in parallel
#
# Usage: ./run_generate_data.sh [start_chunk] [end_chunk] [output_dir] [target_layers] [stride] [block_sizes] [samples_per_clip]
#
# Arguments:
#   start_chunk      - First chunk to process (default: 0)
#   end_chunk        - Last chunk (exclusive) (default: 400)
#   output_dir       - Output directory (default: /data/dflash_train_1)
#   target_layers    - Comma-separated layer IDs (default: 24,30,31,32,34)
#   stride           - Sliding window stride (default: 1)
#   block_sizes      - Comma-separated block sizes (default: 8,16)
#   samples_per_clip - Number of t0 samples per clip (default: 3)
#   context_len      - Number of consecutive hidden states per block (default: 1)
#
# Examples:
#   # Generate at full resolution (default)
#   ./run_generate_data.sh
#
#   # Generate chunks 0-99 on machine 1
#   ./run_generate_data.sh 0 100 /data/dflash_train
#
# All outputs go to the same directory, shards are named with rank suffix to avoid conflicts.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_CHUNK="${1:-0}"
END_CHUNK="${2:-600}"
OUTPUT_DIR="${3:-/mnt/moosefs-1/users/zekail/dflash_train_1}"
TARGET_LAYERS="${4:-24,30,31,32,34}"
STRIDE="${5:-1}"
BLOCK_SIZES="${6:-8,16}"  # Support multiple: "8,16" or single: "16"
SAMPLES_PER_CLIP="${7:-3}"  # Number of t0 samples per clip
CONTEXT_LEN="${8:-8}"  # Number of consecutive hidden states per block
NUM_GPUS=8

TOTAL_CHUNKS=$((END_CHUNK - START_CHUNK))
CHUNKS_PER_GPU=$((TOTAL_CHUNKS / NUM_GPUS))

# Handle case where chunks don't divide evenly
if [ $((TOTAL_CHUNKS % NUM_GPUS)) -ne 0 ]; then
    echo "Warning: ${TOTAL_CHUNKS} chunks doesn't divide evenly by ${NUM_GPUS} GPUs"
    echo "Some GPUs will process fewer chunks"
fi

echo "========================================"
echo "DFlash Distillation Data Generation"
echo "========================================"
echo "Chunk range: ${START_CHUNK} - ${END_CHUNK} (${TOTAL_CHUNKS} chunks)"
echo "GPUs: ${NUM_GPUS}"
echo "Chunks per GPU: ~${CHUNKS_PER_GPU}"
echo "Output: ${OUTPUT_DIR}"
echo "Target layers: [${TARGET_LAYERS}]"
echo "Stride: ${STRIDE}"
echo "Block sizes: [${BLOCK_SIZES}]"
echo "Samples per clip: ${SAMPLES_PER_CLIP}"
echo "Context len: ${CONTEXT_LEN}"
echo "========================================"

mkdir -p "${OUTPUT_DIR}"

for rank in {0..7}; do
    # Calculate chunk range for this GPU
    GPU_START=$((START_CHUNK + rank * CHUNKS_PER_GPU))
    GPU_END=$((START_CHUNK + (rank + 1) * CHUNKS_PER_GPU))

    # Last GPU takes remaining chunks
    if [ $rank -eq 7 ]; then
        GPU_END=${END_CHUNK}
    fi

    # Skip if no chunks for this GPU
    if [ $GPU_START -ge $END_CHUNK ]; then
        echo "Rank ${rank}: No chunks to process, skipping"
        continue
    fi

    echo "Starting rank ${rank} (chunks ${GPU_START}-$((GPU_END - 1)))..."

    CUDA_VISIBLE_DEVICES=${rank} python "${SCRIPT_DIR}/generate_data.py" \
        --rank ${rank} \
        --start-chunk ${GPU_START} \
        --end-chunk ${GPU_END} \
        --output-dir "${OUTPUT_DIR}" \
        --target-layers "${TARGET_LAYERS}" \
        --stride ${STRIDE} \
        --block-sizes "${BLOCK_SIZES}" \
        --num-samples-per-clip ${SAMPLES_PER_CLIP} \
        --context-len ${CONTEXT_LEN} \
        --full-vocab \
        2>&1 | tee "${OUTPUT_DIR}/rank${rank}.log" &
done

wait

echo ""
echo "All processes complete!"
echo "Chunk range: ${START_CHUNK} - ${END_CHUNK}"
echo "Output directories:"
for bs in ${BLOCK_SIZES//,/ }; do
    echo "  - ${OUTPUT_DIR}/b${bs}"
done
echo ""
echo "To generate more chunks on another machine, run:"
echo "  ./run_generate_data.sh <start> <end> ${OUTPUT_DIR}"
