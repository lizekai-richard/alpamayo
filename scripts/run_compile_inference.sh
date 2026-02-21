#!/bin/bash
# Run test_inference.py (AlpamayoR1 compile model) for selected clip IDs.
# Clip IDs are read from clip_ids.json in the repo root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INFERENCE_SCRIPT="$REPO_ROOT/src/alpamayo_r1/test_inference.py"
CLIP_IDS_FILE="${CLIP_IDS_FILE:-$REPO_ROOT/clip_ids.json}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/Alpamayo-R1-10B}"

SPARSITY_RATIO="${SPARSITY_RATIO:-0.5}"
ROPE_MODE="${ROPE_MODE:-direct}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/test_results/6samples_sparsity${SPARSITY_RATIO}_${ROPE_MODE}}"
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

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Run from repo root so model path "./Alpamayo-R1-10B" works
cd "$REPO_ROOT"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Alpamayo-R1 Compile Inference"
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
    --output_dir "$OUTPUT_DIR" \
    --sparsity_ratio "$SPARSITY_RATIO" \
    --rope_mode "$ROPE_MODE"
  echo ""
done

echo "Done. Results (compile_<clip_id>.json) saved under $OUTPUT_DIR"
