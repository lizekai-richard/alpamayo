#!/bin/bash
# Run test_streaming_inference.py for selected clip IDs.
# Clip IDs are read from clip_ids.json in the repo root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INFERENCE_SCRIPT="$REPO_ROOT/src/alpamayo_r1/test_streaming_inference.py"
CLIP_IDS_FILE="${CLIP_IDS_FILE:-$REPO_ROOT/clip_ids.json}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/Alpamayo-R1-10B}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/test_results/6samples_pruning}"

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
  CLIP_IDS=("b80a15fc-d540-4c8f-81d1-5db83216b2e0")
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Optional: run from repo root so model path "./Alpamayo-R1-10B" works
cd "$REPO_ROOT"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Alpamayo-R1 Streaming Inference"
echo "=============================================="
echo "Repo root:    $REPO_ROOT"
echo "Model path:   $MODEL_PATH"
echo "Output dir:   $OUTPUT_DIR"
echo "Clips:        ${#CLIP_IDS[@]} clip(s)"
echo "=============================================="
echo ""

for clip_id in "${CLIP_IDS[@]}"; do
  echo "=============================================="
  echo ">>> Clip: $clip_id"
  echo "=============================================="
  "$PYTHON_BIN" "$INFERENCE_SCRIPT" \
    --clip-id "$clip_id" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR"
  echo ""
done

echo "Done. Results (streaming_<clip_id>.json) saved under $OUTPUT_DIR"
