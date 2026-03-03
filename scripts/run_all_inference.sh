#!/bin/bash
# Run test_streaming_inference.py for selected clip IDs.
# Clip IDs are read from clips.json in the repo root.
#
# Supports multi-GPU parallelism: clips are evenly distributed across GPUs,
# each running as a separate background process with its own CUDA_VISIBLE_DEVICES.
#
# Environment variables:
#   NUM_GPUS          Number of GPUs to use (default: 8)
#   PYTHON_BIN        Python binary (default: python)
#   MODEL_PATH        Path to model weights
#   NUM_TRAJ_SAMPLES  Trajectory samples per clip (default: 6)
#   SPARSITY_RATIO    Sparsity ratio (default: 0)
#   ROPE_MODE         RoPE mode (default: contiguous)
#   OUTPUT_DIR        Where to write results
#   CLIP_IDS_FILE     Path to clips.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INFERENCE_SCRIPT="$REPO_ROOT/src/alpamayo_r1/test_inference_all.py"
CLIP_IDS_FILE="${CLIP_IDS_FILE:-$REPO_ROOT/clips.json}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/checkpoints_expert/checkpoint-6446}"
DRAFT_MODEL="${DRAFT_MODEL:-$REPO_ROOT/Alpamayo-DFlash}"
NUM_TRAJ_SAMPLES="${NUM_TRAJ_SAMPLES:-6}"
CACHE_STEPS="${CACHE_STEPS:-[1, 3, 5, 7, 9]}"
NUM_GPUS="${NUM_GPUS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/test_results/6samples_all_checkpoint6446}"
DUMPED_DATA_DIR="${DUMPED_DATA_DIR:-$REPO_ROOT/dumped_eval_data}"

# -----------------------------------------------------------------------------
# Load clip IDs from clips.json, or fall back to default list
# -----------------------------------------------------------------------------
if [[ -f "$CLIP_IDS_FILE" ]]; then
  if command -v jq &>/dev/null; then
    mapfile -t CLIP_IDS < <(jq -r '.[]' "$CLIP_IDS_FILE")
  else
    mapfile -t CLIP_IDS < <("$PYTHON_BIN" -c "import json,sys; [print(x) for x in json.load(open(sys.argv[1]))]" "$CLIP_IDS_FILE")
  fi
else
  echo "Clip IDs file not found: $CLIP_IDS_FILE (using default clip)" >&2
  CLIP_IDS=("b80a15fc-d540-4c8f-81d1-5db83216b2e0")
fi

TOTAL_CLIPS=${#CLIP_IDS[@]}

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Run from repo root so model path "./Alpamayo-R1-10B" works
cd "$REPO_ROOT"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Alpamayo-R1 Streaming Inference (Multi-GPU)"
echo "=============================================="
echo "Repo root:    $REPO_ROOT"
echo "Model path:   $MODEL_PATH"
echo "Output dir:   $OUTPUT_DIR"
echo "Total clips:  $TOTAL_CLIPS"
echo "Num GPUs:     $NUM_GPUS"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Worker function: runs a subset of clips on a single GPU
# -----------------------------------------------------------------------------
run_worker() {
  local gpu_id=$1
  shift
  local clips=("$@")

  echo "[GPU $gpu_id] Processing ${#clips[@]} clip(s): ${clips[0]} ... ${clips[-1]}"

  for clip_id in "${clips[@]}"; do
    RESULT_FILE="$OUTPUT_DIR/streaming_${clip_id}.json"
    if [[ -f "$RESULT_FILE" ]]; then
      echo "[GPU $gpu_id] >>> Skipping $clip_id (result exists: $RESULT_FILE)"
      continue
    fi
    echo "[GPU $gpu_id] >>> Clip: $clip_id"
    WORKER_RANK="$gpu_id" CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "$INFERENCE_SCRIPT" \
      --clip-id "$clip_id" \
      --model_path "$MODEL_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --cache_steps "$CACHE_STEPS" \
      --num_traj_samples "$NUM_TRAJ_SAMPLES" \
      --draft-model "$DRAFT_MODEL" \
      ${DUMPED_DATA_DIR:+--dumped_data_dir "$DUMPED_DATA_DIR"}
  done

  echo "[GPU $gpu_id] Finished all ${#clips[@]} clip(s)."
}

# -----------------------------------------------------------------------------
# Distribute clips across GPUs (round-robin) and launch workers
# -----------------------------------------------------------------------------
declare -a PIDS=()

for (( gpu=0; gpu<NUM_GPUS; gpu++ )); do
  # Collect clips for this GPU via round-robin assignment
  worker_clips=()
  for (( i=gpu; i<TOTAL_CLIPS; i+=NUM_GPUS )); do
    worker_clips+=("${CLIP_IDS[$i]}")
  done

  # Skip if this GPU got no clips (more GPUs than clips)
  if [[ ${#worker_clips[@]} -eq 0 ]]; then
    continue
  fi

  # Launch worker in background
  run_worker "$gpu" "${worker_clips[@]}" &
  PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} workers (PIDs: ${PIDS[*]})"
echo "Waiting for all workers to finish..."
echo ""

# -----------------------------------------------------------------------------
# Wait for all workers; track failures
# -----------------------------------------------------------------------------
FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "ERROR: Worker PID $pid failed." >&2
    FAILED=$((FAILED + 1))
  fi
done

echo ""
if [[ $FAILED -gt 0 ]]; then
  echo "WARNING: $FAILED / ${#PIDS[@]} worker(s) failed." >&2
  exit 1
else
  echo "Done. All $TOTAL_CLIPS clip(s) completed across $NUM_GPUS GPU(s)."
  echo "Results (streaming_<clip_id>.json) saved under $OUTPUT_DIR"
fi
