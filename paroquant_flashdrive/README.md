# ParoQuant for AlpamayoR1

AlpamayoR1-specific scripts for [ParoQuant](https://github.com/z-lab/paroquant) INT4 weight quantization.

This directory contains only our model-specific optimization and conversion scripts.

## Prerequisites

Install the ParoQuant library and CUDA kernels:

```bash
git clone https://github.com/z-lab/paroquant
cd paroquant

# install dependencies
conda env create -f environment.yml
conda activate paroquant
# or: pip install -r requirements.txt

# install CUDA kernels
pip install ./kernels --no-build-isolation

# add paroquant to PYTHONPATH (no setup.py at repo root)
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## End-to-End Workflow

### 0. Generate multimodal calibration data (optional)

Generate calibration data with image features pre-fused, so ParoQuant
optimization sees activations representative of the actual multimodal
workload (images + prompt) instead of text-only data.

```bash
python paroquant/generate_calib_data.py \
    --model-path /data/scratch/zekaili/Alpamayo-R1-10B \
    --output /data/scratch/zekaili/calib_data_multimodal.pt \
    --start-chunk 0 --end-chunk 5 \
    --seed 42
```

This pre-downloads the dataset chunks, then for each clip loads 4 cameras
at t0=5.1s, runs the vision encoder, and saves the fused `inputs_embeds`,
`position_ids`, and `attention_mask` to a single `.pt` file.

### 1. Optimize VLM layers

With text-only calibration data:

```bash
python paroquant/optimize_alpamayo.py \
    --model /data/scratch/zekaili/Alpamayo-R1-10B \
    --target vlm \
    --datasets wikitext2 c4 \
    --val-dataset wikitext2 \
    --output-dir /data/scratch/zekaili/paroquant_results \
    --params "channel_scales:0.05,angles:0.05" "weight:1e-5,quantizer:1e-6" \
    --epochs 10 10 \
    --group-size 128 --n-bit 4 --num-rotations 8 \
    --train-size 128 --validation-size 32 \
    --batch-size 4 --seqlen 2048 --seed 42
```

With multimodal calibration data (from step 0):

```bash
python paroquant/optimize_alpamayo.py \
    --model /data/scratch/zekaili/Alpamayo-R1-10B \
    --target vlm \
    --calib-data /data/scratch/zekaili/calib_data_multimodal.pt \
    --output-dir /data/scratch/zekaili/paroquant_results_multimodal \
    --params "channel_scales:0.05,angles:0.05" "weight:1e-5,quantizer:1e-6" \
    --epochs 10 10 \
    --group-size 128 --n-bit 4 --num-rotations 8 \
    --train-size 256 --validation-size 64 \
    --batch-size 4 --seed 42
```

### 2. Optimize expert layers

```bash
python paroquant/optimize_alpamayo.py \
    --model /data/scratch/zekaili/Alpamayo-R1-10B \
    --target expert \
    --datasets wikitext2 c4 \
    --val-dataset wikitext2 \
    --output-dir /data/scratch/zekaili/paroquant_results \
    --params "channel_scales:0.05,angles:0.05" "weight:1e-5,quantizer:1e-6" \
    --epochs 10 10 \
    --group-size 128 --n-bit 4 --num-rotations 8 \
    --train-size 128 --validation-size 32 \
    --batch-size 4 --seqlen 2048 --seed 42
```

### 3. Convert to checkpoint

```bash
# VLM + Expert
python paroquant/real_quant_alpamayo.py \
    --model /data/scratch/zekaili/Alpamayo-R1-10B \
    --vlm-result-dir /data/scratch/zekaili/paroquant_results/Alpamayo-R1-10B/vlm \
    --expert-result-dir /data/scratch/zekaili/paroquant_results/Alpamayo-R1-10B/expert \
    --output-path /data/scratch/zekaili/quant_cache/alpamayo-paro-w4.pt

# VLM only (skip --expert-result-dir if expert layers not optimized)
python paroquant/real_quant_alpamayo.py \
    --model /data/scratch/zekaili/Alpamayo-R1-10B \
    --vlm-result-dir /data/scratch/zekaili/paroquant_results/Alpamayo-R1-10B/vlm \
    --output-path /data/scratch/zekaili/quant_cache/alpamayo-paro-w4-vlm.pt
```

### 4. Run evaluation

```bash
# From PhysicalAI-AV dataset (streams data)
python eval/eval_dflash_clips_paro.py \
    --model-path /data/scratch/zekaili/Alpamayo-R1-10B \
    --paro-checkpoint /data/scratch/zekaili/quant_cache/alpamayo-paro-w4.pt \
    --num-clips 100

# From pre-dumped data (faster, no dataset download needed)
python eval/eval_dflash_clips_paro.py \
    --model-path /data/scratch/zekaili/Alpamayo-R1-10B \
    --paro-checkpoint /data/scratch/zekaili/quant_cache/alpamayo-paro-w4-vlm.pt \
    --data-dir /data/scratch/zekaili/dumped_eval_data \
    --num-clips 100 \
    --quantize-VLM-only
```

## Directory Layout

```
paroquant/                      # This directory — AlpamayoR1 scripts
    generate_calib_data.py      # Generate multimodal calibration data
    optimize_alpamayo.py        # Layerwise optimization (VLM or expert)
    real_quant_alpamayo.py      # Convert .pt results → single checkpoint
    README.md

../../paroquant/                 # Upstream ParoQuant library (sibling repo, do not modify)
    paroquant/                  # Core optimization library
    inference_engine/           # Inference runtime + RotateLinearInt4
    kernels/                    # CUDA kernels (pip install -e .)
    ...
```
