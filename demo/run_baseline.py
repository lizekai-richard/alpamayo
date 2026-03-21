#!/usr/bin/env python3
"""Baseline inference using NVIDIA's original Alpamayo-R1 (no optimizations).

Uses the original alpamayo_r1 repo (demo/alpamayo/) with vlm.generate().
Produces per-frame JSON data compatible with the video rendering pipeline.

Usage:
    python demo/run_baseline.py --data-dir ~/exp/demo/clip_data
    python demo/run_baseline.py --data-dir ~/exp/demo/clip_data --num-frames 20
    python demo/run_baseline.py --clip-id <id> --cache-dir ~/data/physicalai_av/hf_cache
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Use NVIDIA's original alpamayo_r1 (not the dev version)
_ORIGINAL_SRC = str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, _ORIGINAL_SRC)

import numpy as np
import torch
from tqdm import tqdm

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

STEP_US = 100_000


# ─── Data loading ─────────────────────────────────────────────

def load_cached_data(data_dir, t0_us, num_frames):
    pt_path = Path(data_dir) / f"{t0_us}.pt"
    cached = torch.load(pt_path, map_location="cpu", weights_only=True)
    image_frames = cached["image_frames_nf4"] if num_frames == 4 else cached["image_frames_nf1"]
    return {
        "image_frames": image_frames,
        "ego_history_xyz": cached["ego_history_xyz"],
        "ego_history_rot": cached["ego_history_rot"],
        "ego_future_xyz": cached["ego_future_xyz"],
        "ego_future_rot": cached["ego_future_rot"],
    }


def prepare_inputs(data, processor):
    frames = data["image_frames"].flatten(0, 1)
    messages = helper.create_message(frames)
    tok = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    return {
        "tokenized_data": tok,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
        "ego_future_xyz": data["ego_future_xyz"],
        "ego_future_rot": data["ego_future_rot"],
    }


def calc_min_ade(gt_future_xy, pred_xyz):
    try:
        gt_xy = gt_future_xy.cpu()[0, 0, :, :2].T.numpy()
        pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
        diff = np.linalg.norm(pred_xy - gt_xy[None], axis=1)
        ade = diff.mean(axis=-1)
        return float(ade.min()), float(ade[0])
    except Exception:
        return None, None


def extract_coc_text(extra):
    if extra is None:
        return ""
    cot = extra.get("cot")
    if cot is None:
        return ""
    if isinstance(cot, str):
        return cot
    try:
        if hasattr(cot, "ndim"):
            if cot.ndim >= 3:
                return str(cot[0, 0, 0])
            elif cot.ndim == 2:
                return str(cot[0, 0])
            elif cot.ndim == 1:
                return str(cot[0])
            return str(cot.item()) if cot.ndim == 0 else str(cot)
        if isinstance(cot, (list, tuple)):
            inner = cot[0]
            while isinstance(inner, (list, tuple)):
                inner = inner[0]
            return str(inner)
    except (IndexError, TypeError):
        pass
    return str(cot)


# ─── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline inference for demo video")
    parser.add_argument("--model-path", default="nvidia/Alpamayo-R1-10B")
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--clip-file", default="~/data/physicalai_av/clip_ids.json")
    parser.add_argument("--data-dir", default=None,
                        help="Pre-cached data directory (from run_all_methods.py --save-data)")
    parser.add_argument("--start-us", type=int, default=None)
    parser.add_argument("--end-us", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--output-dir", default="~/exp/demo")
    parser.add_argument("--cache-dir", default="~/data/physicalai_av/hf_cache")
    args = parser.parse_args()

    for attr in ("model_path", "clip_file", "output_dir", "cache_dir"):
        setattr(args, attr, os.path.expanduser(getattr(args, attr)))
    if args.data_dir:
        args.data_dir = os.path.expanduser(args.data_dir)

    # ── Determine timesteps ──
    avdi = None
    gt_cutoff_us = None
    if args.data_dir:
        manifest_path = Path(args.data_dir) / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        if args.clip_id is None:
            args.clip_id = manifest["clip_id"]
        all_t0s = manifest["t0_us_list"]
        if args.start_us:
            all_t0s = [t for t in all_t0s if t >= args.start_us]
        if args.end_us:
            all_t0s = [t for t in all_t0s if t <= args.end_us]
        if args.num_frames is not None:
            all_t0s = all_t0s[:args.num_frames]
    else:
        import physical_ai_av
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=args.cache_dir)
        if args.clip_id is None:
            with open(args.clip_file) as f:
                clip_ids = json.load(f)
            args.clip_id = clip_ids[args.clip_index]

        from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

        egomotion = avdi.get_clip_feature(
            args.clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True)
        ego_end_us = int(egomotion.timestamps[-1])
        gt_cutoff_us = ego_end_us - 6_400_000
        camera = avdi.get_clip_feature(
            args.clip_id, avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV, maybe_stream=True)
        clip_end_us = min(ego_end_us, int(camera.timestamps[-1]))
        start_us = args.start_us or 1_700_000
        end_us = args.end_us or (clip_end_us - STEP_US)
        all_t0s = list(range(start_us, end_us + 1, STEP_US))
        if args.num_frames is not None:
            all_t0s = all_t0s[:args.num_frames]

    # ── Output dir ──
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    output_root = Path(args.output_dir) / f"baseline_{timestamp}"
    frame_dir = output_root / "1_baseline"
    frame_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Baseline Inference (NVIDIA original, no optimizations)")
    log.info("=" * 60)
    log.info(f"Clip:        {args.clip_id}")
    log.info(f"Frames:      {len(all_t0s)}")
    log.info(f"K:           {args.num_samples}")
    log.info(f"Diff steps:  {args.diffusion_steps}")
    log.info(f"Output:      {output_root}")
    log.info(f"Source:      {_ORIGINAL_SRC}")

    config = {
        "clip_id": args.clip_id,
        "num_frames": len(all_t0s),
        "num_samples": args.num_samples,
        "diffusion_steps": args.diffusion_steps,
        "method": "baseline",
    }
    with open(output_root / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Load model (NVIDIA original) ──
    log.info(f"Loading model from {args.model_path}...")
    model = AlpamayoR1.from_pretrained(args.model_path, dtype=torch.bfloat16).to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    # ── Pre-load data ──
    log.info(f"Pre-loading {len(all_t0s)} timesteps...")
    all_inputs = []
    all_gt = []
    for t0_us in all_t0s:
        has_gt = gt_cutoff_us is None or t0_us <= gt_cutoff_us
        if args.data_dir:
            data = load_cached_data(args.data_dir, t0_us, num_frames=4)
        else:
            nfs = 64 if has_gt else 1
            data = load_physical_aiavdataset(args.clip_id, t0_us=t0_us, num_frames=4,
                                             num_future_steps=nfs, avdi=avdi)
        all_inputs.append(prepare_inputs(data, processor))
        all_gt.append((has_gt, data["ego_future_xyz"] if has_gt else None))

    # ── Inference (uses vlm.generate, same as NVIDIA's test_inference.py) ──
    log.info("Starting inference...")
    stats = []
    pbar = tqdm(enumerate(all_t0s), total=len(all_t0s), desc="Baseline")

    for frame_idx, t0_us in pbar:
        try:
            inputs = all_inputs[frame_idx]
            has_gt, ego_future_xyz = all_gt[frame_idx]

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                result = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=helper.to_device(inputs, "cuda"),
                    num_traj_samples=args.num_samples,
                    max_generation_length=args.max_tokens,
                    return_extra=True,
                    diffusion_kwargs={"inference_step": args.diffusion_steps},
                )

            pred_xyz, pred_rot, extra = result
            if pred_xyz is None:
                continue

            timing = extra.get("timing", {}) if extra else {}
            is_warmup = frame_idx < args.warmup_steps

            min_ade_k, min_ade_1 = (None, None)
            if has_gt:
                min_ade_k, min_ade_1 = calc_min_ade(ego_future_xyz, pred_xyz)

            ntok = timing.get("num_decode_tokens", 0)
            dec_ms = timing.get("decode_time_ms", 0)

            frame_data = {
                "frame": frame_idx,
                "t0_us": t0_us,
                "is_prefill": False,
                "is_warmup": is_warmup,
                "total_ms": timing.get("total_time_ms", 0),
                "encode_ms": timing.get("encode_time_ms", 0),
                "prefill_ms": timing.get("prefill_time_ms", 0),
                "decode_ms": dec_ms,
                "diffusion_ms": timing.get("action_time_ms", 0),
                "tokens": ntok,
                "tokens_per_sec": ntok / (dec_ms / 1000) if dec_ms > 0 else 0,
                "coc": extract_coc_text(extra),
                "pred_xyz": pred_xyz.cpu().numpy()[0, 0].tolist(),
            }
            if min_ade_k is not None:
                frame_data["min_ade_k"] = min_ade_k
                frame_data["min_ade_1"] = min_ade_1

            with open(frame_dir / f"frame_{frame_idx:04d}.json", "w") as f:
                json.dump(frame_data, f, indent=2)
            stats.append(frame_data)

            pbar.set_postfix({
                "tot": f"{frame_data['total_ms']:.0f}",
                "enc": f"{frame_data['encode_ms']:.0f}",
                "dec": f"{frame_data['decode_ms']:.0f}",
                "tok": ntok,
            })

        except Exception as e:
            log.warning(f"Frame {frame_idx} error: {e}")
            import traceback
            log.warning(traceback.format_exc())
            if "CUDA" in str(e):
                raise

    # ── Summary ──
    valid = [s for s in stats if not s.get("is_warmup")]
    if valid:
        def avg(key):
            vals = [s[key] for s in valid if key in s]
            return float(np.mean(vals)) if vals else 0

        summary = {
            "method": "baseline",
            "label": "Baseline",
            "num_frames": len(valid),
            "avg_total_ms": avg("total_ms"),
            "avg_encode_ms": avg("encode_ms"),
            "avg_prefill_ms": avg("prefill_ms"),
            "avg_decode_ms": avg("decode_ms"),
            "avg_diffusion_ms": avg("diffusion_ms"),
            "avg_tokens": avg("tokens"),
            "avg_tokens_per_sec": avg("tokens_per_sec"),
        }
        if any("min_ade_k" in s for s in valid):
            summary["avg_min_ade_k"] = avg("min_ade_k")
            summary["avg_min_ade_1"] = avg("min_ade_1")

        with open(output_root / "summary.json", "w") as f:
            json.dump({"config": config, "methods": {"1_baseline": summary}}, f, indent=2)

        log.info(f"\n{'='*60}")
        log.info(f"Baseline Results ({len(valid)} frames)")
        log.info(f"{'='*60}")
        log.info(f"  Avg total:     {summary['avg_total_ms']:.1f} ms")
        log.info(f"  Avg encode:    {summary['avg_encode_ms']:.1f} ms")
        log.info(f"  Avg prefill:   {summary['avg_prefill_ms']:.1f} ms")
        log.info(f"  Avg decode:    {summary['avg_decode_ms']:.1f} ms")
        log.info(f"  Avg diffusion: {summary['avg_diffusion_ms']:.1f} ms")
        if "avg_min_ade_1" in summary:
            log.info(f"  Avg ADE_1:     {summary['avg_min_ade_1']:.3f} m")

    log.info(f"\nOutput saved to {output_root}")


if __name__ == "__main__":
    main()
