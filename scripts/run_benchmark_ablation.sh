#!/bin/bash
# Ablation study benchmark script for Alpamayo-R1
# This script runs all combinations of streaming/non-streaming and fusion options

set -e

# Configuration
MODEL_PATH="${MODEL_PATH:-./Alpamayo-R1-10B}"
OUTPUT_DIR="${OUTPUT_DIR:-./benchmark_results}"
NUM_STEPS="${NUM_STEPS:-15}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
CLIP_ID="${CLIP_ID:-030c760c-ae38-49aa-9ad8-f5650a545d26}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Alpamayo-R1 Benchmark Ablation Study"
echo "=============================================="
echo "Model path: $MODEL_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Num steps: $NUM_STEPS"
echo "Warmup steps: $WARMUP_STEPS"
echo "=============================================="
echo ""

# Common args
COMMON_ARGS="--model_path $MODEL_PATH --output_dir $OUTPUT_DIR --num_steps $NUM_STEPS --warmup_steps $WARMUP_STEPS --clip_id $CLIP_ID"

# Function to check if benchmark result already exists
check_exists() {
    local pattern="$1"
    if ls "$OUTPUT_DIR"/$pattern 1> /dev/null 2>&1; then
        return 0  # exists
    else
        return 1  # not exists
    fi
}

# =============================================================================
# Streaming mode ablations (supports fuse options)
# =============================================================================

echo ">>> [1/6] Streaming mode - baseline (no fusion)"
if check_exists "benchmark_streaming_max-autotune_2*.json"; then
    echo "    [SKIP] Result already exists"
else
    python -m alpamayo_r1.benchmark_compile $COMMON_ARGS
fi

echo ""
echo ">>> [2/6] Streaming mode - fuse_qkv only"
if check_exists "benchmark_streaming_max-autotune_fuse_qkv_2*.json"; then
    echo "    [SKIP] Result already exists"
else
    python -m alpamayo_r1.benchmark_compile $COMMON_ARGS --fuse_qkv
fi

echo ""
echo ">>> [3/6] Streaming mode - fuse_gate_up only"
if check_exists "benchmark_streaming_max-autotune_fuse_gate_up_2*.json"; then
    echo "    [SKIP] Result already exists"
else
    python -m alpamayo_r1.benchmark_compile $COMMON_ARGS --fuse_gate_up
fi

echo ""
echo ">>> [4/6] Streaming mode - fuse_qkv + fuse_gate_up"
if check_exists "benchmark_streaming_max-autotune_fuse_qkv_fuse_gate_up_2*.json"; then
    echo "    [SKIP] Result already exists"
else
    python -m alpamayo_r1.benchmark_compile $COMMON_ARGS --fuse_qkv --fuse_gate_up
fi

# =============================================================================
# Non-streaming mode (fuse options not supported)
# =============================================================================

echo ""
echo ">>> [5/6] Non-streaming mode - baseline"
if check_exists "benchmark_non_streaming_max-autotune_2*.json"; then
    echo "    [SKIP] Result already exists"
else
    python -m alpamayo_r1.benchmark_compile $COMMON_ARGS --non_streaming
fi

# =============================================================================
# Comparison: compiled vs non-compiled (streaming + full fusion)
# =============================================================================

echo ""
echo ">>> [6/6] Comparison: compiled vs non-compiled (streaming + full fusion)"
if check_exists "comparison_2*.png"; then
    echo "    [SKIP] Result already exists"
else
    python -m alpamayo_r1.benchmark_compile $COMMON_ARGS --compare --fuse_qkv --fuse_gate_up
fi

echo ""
echo "=============================================="
echo "All benchmarks completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
