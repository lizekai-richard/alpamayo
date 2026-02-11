#!/bin/bash
# Benchmark script for Alpamayo-R1
# Runs 5 settings: original, sysopt, sysopt+pruning, streaming, streaming+pruning

set -e

# Configuration
MODEL_PATH="${MODEL_PATH:-./Alpamayo-R1-10B}"
OUTPUT_DIR="${OUTPUT_DIR:-./benchmark_results}"
NUM_STEPS="${NUM_STEPS:-50}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SPARSITY_RATIO="${SPARSITY_RATIO:-0.5}"
CLIP_ID="${CLIP_ID:-030c760c-ae38-49aa-9ad8-f5650a545d26}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Alpamayo-R1 Benchmark"
echo "=============================================="
echo "Model path: $MODEL_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Num steps: $NUM_STEPS"
echo "Warmup steps: $WARMUP_STEPS"
echo "Num samples: $NUM_SAMPLES"
echo "Sparsity ratio: $SPARSITY_RATIO"
echo "=============================================="
echo ""

COMMON_ARGS="--model_path $MODEL_PATH --output_dir $OUTPUT_DIR --num_steps $NUM_STEPS --warmup_steps $WARMUP_STEPS --num_samples $NUM_SAMPLES --clip_id $CLIP_ID"

# =============================================================================
# 1. Original model (HF generate, no compile)
# =============================================================================

echo ">>> [1/5] alpamayo (original)"
python -m alpamayo_r1.benchmark_compile $COMMON_ARGS --setting alpamayo

# =============================================================================
# 2. Non-streaming + torch.compile + QKV/MLP fusion
# =============================================================================

echo ""
echo ">>> [2/5] alpamayo_sysopt (non-streaming + compile + fusion)"
python -m alpamayo_r1.benchmark_compile $COMMON_ARGS --setting alpamayo_sysopt

# =============================================================================
# 3. Non-streaming + torch.compile + QKV/MLP fusion + token pruning
# =============================================================================

echo ""
echo ">>> [3/5] alpamayo_sysopt_pruning (non-streaming + compile + fusion + pruning)"
python -m alpamayo_r1.benchmark_compile $COMMON_ARGS --setting alpamayo_sysopt_pruning --sparsity_ratio $SPARSITY_RATIO

# =============================================================================
# 4. Streaming + torch.compile + QKV/MLP fusion
# =============================================================================

echo ""
echo ">>> [4/5] alpamayo_sysopt_streaming (streaming + compile + fusion)"
python -m alpamayo_r1.benchmark_compile $COMMON_ARGS --setting alpamayo_sysopt_streaming

# =============================================================================
# 5. Streaming + torch.compile + QKV/MLP fusion + token pruning
# =============================================================================

echo ""
echo ">>> [5/5] alpamayo_sysopt_streaming_pruning (streaming + compile + fusion + pruning)"
python -m alpamayo_r1.benchmark_compile $COMMON_ARGS --setting alpamayo_sysopt_streaming_pruning --sparsity_ratio $SPARSITY_RATIO

echo ""
echo "=============================================="
echo "All benchmarks completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
