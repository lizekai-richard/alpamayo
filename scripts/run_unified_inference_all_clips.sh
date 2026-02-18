#!/bin/bash
# Run test_unified_inference over all clip IDs in clip_ids.json.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/Alpamayo-R1-10B}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/test_results}"
CLIP_IDS_FILE="${CLIP_IDS_FILE:-$REPO_ROOT/clip_ids.json}"

NUM_STEPS="${NUM_STEPS:-120}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
NUM_TRAJ_SAMPLES="${NUM_TRAJ_SAMPLES:-6}"
SEED="${SEED:-42}"
T0_US="${T0_US:-1700000}"
TIME_STEP_US="${TIME_STEP_US:-100000}"
SPARSITY_RATIO="${SPARSITY_RATIO:-0.5}"

# Modes to run: "streaming", "non_streaming", or both.
MODES="${MODES:-streaming}"

# Skip clips that already have result files.
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Limit to first N clips (0 = all).
MAX_CLIPS="${MAX_CLIPS:-0}"

if [[ ! -f "$CLIP_IDS_FILE" ]]; then
  echo "Clip IDs file not found: $CLIP_IDS_FILE" >&2
  exit 1
fi

OUTPUT_DIR="$OUTPUT_DIR/${NUM_TRAJ_SAMPLES}samples_sparsity${SPARSITY_RATIO}"
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=============================================="
echo "Alpamayo-R1 Unified Inference - All Clips"
echo "=============================================="
echo "Repo root:      $REPO_ROOT"
echo "Model path:     $MODEL_PATH"
echo "Output dir:     $OUTPUT_DIR"
echo "Clip IDs file:  $CLIP_IDS_FILE"
echo "Num steps:      $NUM_STEPS"
echo "Warmup steps:   $WARMUP_STEPS"
echo "Traj samples:   $NUM_TRAJ_SAMPLES"
echo "Seed:           $SEED"
echo "t0_us:          $T0_US"
echo "time_step_us:   $TIME_STEP_US"
echo "Modes:          $MODES"
echo "Skip existing:  $SKIP_EXISTING"
echo "Max clips:      $MAX_CLIPS"
echo "Sparsity ratio: $SPARSITY_RATIO"
echo "=============================================="
echo ""

mapfile -t CLIP_IDS < <("$PYTHON_BIN" - <<PY
import json
with open("${CLIP_IDS_FILE}", "r", encoding="utf-8") as f:
    data = json.load(f)
for clip_id in data:
    print(clip_id)
PY
)

if [[ ${#CLIP_IDS[@]} -eq 0 ]]; then
  echo "No clip IDs found in $CLIP_IDS_FILE" >&2
  exit 1
fi

for mode in $MODES; do
  case "$mode" in
    non_streaming|streaming) ;;
    *)
      echo "Unsupported mode: $mode (use streaming or non_streaming)" >&2
      exit 1
      ;;
  esac

  echo ">>> Mode: $mode"
  clip_count=0
  for clip_id in "${CLIP_IDS[@]}"; do
    if [[ "$MAX_CLIPS" -gt 0 && "$clip_count" -ge "$MAX_CLIPS" ]]; then
      echo "  [DONE] Reached max clips ($MAX_CLIPS)"
      break
    fi
    clip_count=$((clip_count + 1))
    if [[ "$mode" == "streaming" ]]; then
      result_file="$OUTPUT_DIR/streaming_${clip_id}.json"
    else
      result_file="$OUTPUT_DIR/original_${clip_id}.json"
    fi

    if [[ "$SKIP_EXISTING" == "1" && -f "$result_file" ]]; then
      echo "  [SKIP] $clip_id (found ${NUM_TRAJ_SAMPLES}samples/$(basename "$result_file"))"
      continue
    fi

    echo "  [RUN]  $clip_id"
    "$PYTHON_BIN" -m alpamayo_r1.test_unified_inference \
      --model_path "$MODEL_PATH" \
      --mode "$mode" \
      --warmup_steps "$WARMUP_STEPS" \
      --num_steps "$NUM_STEPS" \
      --seed "$SEED" \
      --num_traj_samples "$NUM_TRAJ_SAMPLES" \
      --clip_id "$clip_id" \
      --t0_us "$T0_US" \
      --time_step_us "$TIME_STEP_US" \
      --sparsity_ratio "$SPARSITY_RATIO" \
      --output_dir "$OUTPUT_DIR"
  done
done

echo ""
echo "=============================================="
echo "All clips completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
