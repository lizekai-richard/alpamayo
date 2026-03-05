#!/bin/bash
# Run test_inference.py (AlpamayoR1 DIT Cache model) for selected clip IDs.
# Clip IDs are read from clip_ids.json in the repo root.
#
# Supports multi-GPU parallelism: clips are evenly distributed across GPUs,
# each running as a separate background process with its own CUDA_VISIBLE_DEVICES.
#
# Environment variables:
#   NUM_GPUS          Number of GPUs to use (default: 2, ignored if GPU_IDS is set)
#   GPU_IDS           Comma-separated GPU IDs to use (e.g. "2,5"); overrides NUM_GPUS
#   PYTHON_BIN        Python binary (default: python)
#   MODEL_PATH        Path to model weights
#   CACHE_STEPS       Cache steps list (default: [1, 3, 5, 7, 9])
#   OUTPUT_DIR        Where to write results
#   CLIP_IDS_FILE     Path to clip_ids.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INFERENCE_SCRIPT="$REPO_ROOT/src/alpamayo_r1/test_inference_dit_cache.py"
CLIP_IDS_FILE="${CLIP_IDS_FILE:-$REPO_ROOT/clips.json}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/data/scratch/zekaili/Alpamayo-R1-10B}"
NUM_GPUS="${NUM_GPUS:-2}"

# Build GPU ID array: honour GPU_IDS if set, otherwise use 0..NUM_GPUS-1
if [[ -n "${GPU_IDS:-}" ]]; then
  IFS=',' read -ra GPU_ID_ARRAY <<< "$GPU_IDS"
  NUM_GPUS=${#GPU_ID_ARRAY[@]}
else
  GPU_ID_ARRAY=()
  for (( i=0; i<NUM_GPUS; i++ )); do
    GPU_ID_ARRAY+=("$i")
  done
fi
DIFFUSION_STEPS="${DIFFUSION_STEPS:-8}"
CACHE_STEPS="${CACHE_STEPS:-[3, 4, 5, 6]}"
DUMPED_DATA_DIR="${DUMPED_DATA_DIR:-/data/scratch/zekaili/dumped_eval_data}"
NUM_TRAJ_SAMPLES="${NUM_TRAJ_SAMPLES:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-./test_results/${NUM_TRAJ_SAMPLES}samples_8steps_cache3456}"

# -----------------------------------------------------------------------------
# Load clip IDs from clip_ids.json, or fall back to default list
# -----------------------------------------------------------------------------
if [[ -f "$CLIP_IDS_FILE" ]]; then
  if command -v jq &>/dev/null; then
    mapfile -t CLIP_IDS < <(jq -r '.[]' "$CLIP_IDS_FILE")
  else
    mapfile -t CLIP_IDS < <("$PYTHON_BIN" -c "import json,sys; [print(x) for x in json.load(open(sys.argv[1]))]" "$CLIP_IDS_FILE")
  fi
else
  echo "Clip IDs file not found: $CLIP_IDS_FILE (using default clip)" >&2
  CLIP_IDS=("87147a1b-3eef-4c25-94d2-ec7718a49a7a")
fi

TOTAL_CLIPS=${#CLIP_IDS[@]}

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Run from repo root so model path "./Alpamayo-R1-10B" works
cd "$REPO_ROOT"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Alpamayo-R1 DIT Cache Inference (Multi-GPU)"
echo "=============================================="
echo "Repo root:    $REPO_ROOT"
echo "Model path:   $MODEL_PATH"
echo "Output dir:   $OUTPUT_DIR"
echo "Cache steps:  $CACHE_STEPS"
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
    echo "[GPU $gpu_id] >>> Clip: $clip_id"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "$INFERENCE_SCRIPT" \
      --clip-id "$clip_id" \
      --model_path "$MODEL_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --cache_steps "$CACHE_STEPS" \
      --num_traj_samples "$NUM_TRAJ_SAMPLES" \
      --diffusion_steps "$DIFFUSION_STEPS" \
      ${DUMPED_DATA_DIR:+--dumped_data_dir "$DUMPED_DATA_DIR"}
  done

  echo "[GPU $gpu_id] Finished all ${#clips[@]} clip(s)."
}

# -----------------------------------------------------------------------------
# Distribute clips across GPUs (round-robin) and launch workers
# -----------------------------------------------------------------------------
declare -a PIDS=()

for (( idx=0; idx<NUM_GPUS; idx++ )); do
  gpu_id="${GPU_ID_ARRAY[$idx]}"

  # Collect clips for this GPU via round-robin assignment
  worker_clips=()
  for (( i=idx; i<TOTAL_CLIPS; i+=NUM_GPUS )); do
    worker_clips+=("${CLIP_IDS[$i]}")
  done

  # Skip if this GPU got no clips (more GPUs than clips)
  if [[ ${#worker_clips[@]} -eq 0 ]]; then
    continue
  fi

  # Launch worker in background
  run_worker "$gpu_id" "${worker_clips[@]}" &
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
  echo "Results (dit_cache_<clip_id>.json) saved under $OUTPUT_DIR"
fi
