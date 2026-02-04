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

# =============================================================================
# Streaming mode ablations (supports fuse options)
# =============================================================================

echo ">>> [1/6] Streaming mode - baseline (no fusion)"
python -m alpamayo_r1.benchmark_compile $COMMON_ARGS

echo ""
echo ">>> [2/6] Streaming mode - fuse_qkv only"
python -m alpamayo_r1.benchmark_compile $COMMON_ARGS --fuse_qkv

echo ""
echo ">>> [3/6] Streaming mode - fuse_gate_up only"
python -m alpamayo_r1.benchmark_compile $COMMON_ARGS --fuse_gate_up

echo ""
echo ">>> [4/6] Streaming mode - fuse_qkv + fuse_gate_up"
python -m alpamayo_r1.benchmark_compile $COMMON_ARGS --fuse_qkv --fuse_gate_up

# =============================================================================
# Non-streaming mode (fuse options not supported)
# =============================================================================

echo ""
echo ">>> [5/6] Non-streaming mode - baseline"
python -m alpamayo_r1.benchmark_compile $COMMON_ARGS --non_streaming

# =============================================================================
# Comparison: compiled vs non-compiled (streaming + full fusion)
# =============================================================================

echo ""
echo ">>> [6/6] Comparison: compiled vs non-compiled (streaming + full fusion)"
python -m alpamayo_r1.benchmark_compile $COMMON_ARGS --compare --fuse_qkv --fuse_gate_up

echo ""
echo "=============================================="
echo "All benchmarks completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
