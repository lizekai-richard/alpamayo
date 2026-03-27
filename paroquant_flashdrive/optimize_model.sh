#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

python paroquant_flashdrive/optimize_alpamayo_v1p5.py \
    --model nvidia/Alpamayo-1.5-10B \
    --target vlm \
    --calib-data paroquant_flashdrive/calib_data_multimodal_v1p5.pt \
    --output-dir /data/scratch/zekaili/paroquant_results_v1p5 \
    --params "channel_scales:0.05,angles:0.05" "weight:1e-5,quantizer:1e-6" \
    --epochs 10 10 \
    --group-size 128 --n-bit 4 --num-rotations 8 \
    --train-size 128 --validation-size 32 \
    --batch-size 4 --seed 42