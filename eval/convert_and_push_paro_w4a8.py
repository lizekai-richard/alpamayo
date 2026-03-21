#!/usr/bin/env python3
"""Convert ParoQuant WQLinear checkpoint → Marlin W4A8, save locally, and push to HF Hub.

Usage:
    # Baseline
    CUDA_VISIBLE_DEVICES=6 python eval/convert_and_push_paro_w4a8.py \
        --base-model-path /data/scratch/zekaili/Alpamayo-R1-10B \
        --paro-checkpoint /data/scratch/zekaili/quant_cache/alpamayo-paro-w4-vlm-mm.pt \
        --save-path /data/scratch/zekaili/quant_cache/alpamayo-paro-w4a8-marlin-baseline \
        --hub-repo FlashDriveVLA/Alpamayo-R1-10B-PARO

    # Finetuned
    CUDA_VISIBLE_DEVICES=7 python eval/convert_and_push_paro_w4a8.py \
        --base-model-path /data/scratch/zekaili/train_expert_ckpts_deepspeed/checkpoint-6446 \
        --paro-checkpoint /data/scratch/zekaili/quant_cache/ckpt-paro-w4-vlm-mm.pt \
        --save-path /data/scratch/zekaili/quant_cache/alpamayo-paro-w4a8-marlin-finetuned \
        --hub-repo FlashDriveVLA/Alpamayo-R1-10B-finetuned-PARO

    # Convert only (no push)
    CUDA_VISIBLE_DEVICES=0 python eval/convert_and_push_paro_w4a8.py \
        --base-model-path /path/to/base \
        --paro-checkpoint /path/to/paro.pt \
        --save-path /path/to/output
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "paroquant"))

import torch

from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import load_paroquant_model
from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import convert_wqlinear_layer, save_w4a8_pretrained
from alpamayo_r1.utils.quantization.rotation_linear import RotateLinearInt4
from alpamayo_r1.utils.quantization.qmodule import WQLinear

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description="Convert ParoQuant → Marlin W4A8 and push to Hub")
    ap.add_argument("--base-model-path", required=True,
                    help="Path to base AlpamayoR1 pretrained model (for architecture)")
    ap.add_argument("--paro-checkpoint", required=True,
                    help="Path to ParoQuant .pt checkpoint (WQLinear format)")
    ap.add_argument("--save-path", required=True,
                    help="Local directory to save converted model (safetensors)")
    ap.add_argument("--hub-repo", default=None,
                    help="HF Hub repo ID to push to (e.g. 'FlashDriveVLA/Alpamayo-R1-10B-PARO')")
    ap.add_argument("--commit-message", default="Upload ParoQuant W4A8 Marlin model")
    args = ap.parse_args()

    # ── Load & convert ──
    log.info(f"Loading model from {args.base_model_path} with checkpoint {args.paro_checkpoint}")
    model = load_paroquant_model(
        model_path=args.base_model_path,
        paro_checkpoint=args.paro_checkpoint,
        mode="streaming",
        quantize_expert=False,
    )

    log.info("Converting WQLinear → MarlinW4A8Linear...")
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, RotateLinearInt4) and isinstance(mod.qlinear, WQLinear):
            wq = mod.qlinear
            marlin, _ = convert_wqlinear_layer(
                wq.qweight.data, wq.scales.data, wq.scaled_zeros.data,
                wq.out_features, wq.in_features, wq.group_size,
                bias=wq.bias, device=wq.qweight.device,
            )
            mod.qlinear = marlin
            n += 1
    log.info(f"Converted {n} layers")

    # ── Save locally ──
    save_w4a8_pretrained(model, args.save_path)

    # ── Push to Hub ──
    if args.hub_repo:
        log.info(f"Pushing to HF Hub: {args.hub_repo}...")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.hub_repo, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=args.save_path,
            repo_id=args.hub_repo,
            commit_message=args.commit_message,
        )
        log.info(f"Pushed to https://huggingface.co/{args.hub_repo}")

    log.info("Done.")


if __name__ == "__main__":
    main()
