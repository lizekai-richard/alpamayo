#!/usr/bin/env python3
"""Example: load a saved ParoQuant W4A8 Marlin model and run one clip.

Usage:
    CUDA_VISIBLE_DEVICES=7 python eval/load_w4a8_example.py
    CUDA_VISIBLE_DEVICES=7 python eval/load_w4a8_example.py --save-path /path/to/w4a8_model
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "paroquant"))

import torch

from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import load_w4a8_pretrained
from alpamayo_r1 import helper
from alpamayo_r1.utils.dflash.dflash_integration import setup_dflash_for_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


def main():
    ap = argparse.ArgumentParser(description="Load saved W4A8 model and run one clip")
    ap.add_argument("--save-path",
                    default="/data/scratch/zekaili/quant_cache/alpamayo-paro-w4a8-marlin")
    ap.add_argument("--base-model-path",
                    default="/data/scratch/zekaili/train_expert_ckpts_deepspeed/checkpoint-6446")
    ap.add_argument("--draft-model",
                    default="/data/scratch/zekaili/Alpamayo-DFlash")
    ap.add_argument("--data-dir",
                    default="/data/scratch/zekaili/dumped_eval_data")
    ap.add_argument("--push-to-hub", type=str, default=None,
                    help="Push saved model to HF Hub (e.g. 'zekaili/alpamayo-paro-w4a8-marlin')")
    ap.add_argument("--skip-inference", action="store_true",
                    help="Only load and optionally push, skip inference test")
    args = ap.parse_args()

    # ── Load W4A8 model from saved safetensors ──
    log.info(f"Loading W4A8 model from {args.save_path}...")
    model = load_w4a8_pretrained(
        save_path=args.save_path,
        base_model_path=args.base_model_path,
        mode="streaming",
        device="cuda",
    )

    # ── Push to Hub ──
    if args.push_to_hub:
        log.info(f"Pushing to HF Hub: {args.push_to_hub}...")
        model.push_to_hub(args.push_to_hub, safe_serialization=True)
        log.info(f"Pushed to https://huggingface.co/{args.push_to_hub}")

    if args.skip_inference:
        log.info("Skipping inference (--skip-inference). Done.")
        return

    # ── Setup DFlash ──
    setup_dflash_for_model(model, args.draft_model)
    log.info("DFlash enabled")

    # ── Load one clip ──
    data_dir = Path(args.data_dir)
    clip_id = sorted(
        d.name for d in data_dir.iterdir()
        if d.is_dir() and (d / "sliding_window_inputs.pt").exists()
    )[0]
    log.info(f"Running clip: {clip_id}")

    streaming_inputs = torch.load(
        data_dir / clip_id / "sliding_window_inputs.pt",
        map_location="cpu", weights_only=False,
    )

    # ── Run a few steps ──
    for si, inputs in enumerate(streaming_inputs[:5]):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = model.sample_trajectories(
                data=helper.to_device(inputs, "cuda"),
                streaming=True,
                dflash=True,
                torch_compile="max-autotune",
                num_traj_samples=1,
                max_generation_length=128,
                return_extra=True,
                fuse_qkv=False,
                fuse_gate_up=False,
                diffusion_kwargs={
                    "inference_step": 8,
                    "cache_steps": [3, 4, 5, 6],
                    "int_method": "euler_with_cache",
                },
            )

        if result is None or result[0] is None:
            log.info(f"  Step {si}: prefill")
            continue

        pred_xyz, _, extra = result
        timing = extra.get("timing", {})
        log.info(
            f"  Step {si}: {timing.get('total_time_ms', 0):.1f}ms "
            f"pred_xyz shape={tuple(pred_xyz.shape)}"
        )

    log.info("Done.")


if __name__ == "__main__":
    main()
