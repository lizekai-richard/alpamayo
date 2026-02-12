#!/bin/bash
# Run test_streaming_inference.py for selected clip IDs.
# Edit the CLIP_IDS array below to add or change clip IDs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INFERENCE_SCRIPT="$REPO_ROOT/src/alpamayo_r1/test_streaming_inference.py"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/Alpamayo-R1-10B}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/test_results/sys_streaming_pruning_logs}"

# -----------------------------------------------------------------------------
# Clip IDs to run (edit this list manually)
# -----------------------------------------------------------------------------
CLIP_IDS=(
  "b80a15fc-d540-4c8f-81d1-5db83216b2e0"
  # "ee76a44e-0087-4afd-be52-401eab2205ae"
  # "b252537e-58a6-48a4-b5a1-04929c81c88b"
  # Add more clip IDs here, one per line:
  # "your-clip-id-1"
  # "your-clip-id-2"
)

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Optional: run from repo root so model path "./Alpamayo-R1-10B" works
cd "$REPO_ROOT"

mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Alpamayo-R1 Streaming Inference"
echo "=============================================="
echo "Repo root:   $REPO_ROOT"
echo "Model path:  $MODEL_PATH"
echo "Log dir:     $LOG_DIR"
echo "Clips:       ${#CLIP_IDS[@]} clip(s)"
echo "=============================================="
echo ""

for clip_id in "${CLIP_IDS[@]}"; do
  log_file="$LOG_DIR/${clip_id}.log"
  echo "=============================================="
  echo ">>> Clip: $clip_id"
  echo ">>> Log:  $log_file"
  echo "=============================================="
  "$PYTHON_BIN" "$INFERENCE_SCRIPT" --clip-id "$clip_id" 2>&1 | tee "$log_file"
  echo ""
done

echo "Done. Logs saved under $LOG_DIR"
