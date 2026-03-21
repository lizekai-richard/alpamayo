#!/usr/bin/env python3
"""System-optimization-only evaluation: non-streaming mode with torch.compile.

Uses the dev branch AlpamayoR1 with streaming=False — torch.compile + static
cache + manual decode loop, but NO KV cache reuse across frames.
Runs through clips from 1.7s to 13.6s at 10Hz (120 timesteps).
Reports minADE_K (default K=6) and per-step timing.

Usage:
    python eval/eval_system_opt.py
    python eval/eval_system_opt.py --num-clips 10 --diffusion-steps 5
    python eval/eval_system_opt.py --num-traj-samples 1
"""

import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import torch

from alpamayo_r1.models.alpamayo_r1_flashdrive import AlpamayoR1FlashDrive
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")

# 10 Hz data
STEP_US = 100_000
T0_START_US = 1_700_000   # 1.7 s
T0_END_US = 13_600_000    # 13.6 s  → 120 timesteps


def load_clip_ids(path, num_clips):
    with open(path) as f:
        ids = json.load(f)
    return list(set(ids))[:num_clips]


def prepare_inputs(data, processor):
    frames = data["image_frames"].flatten(0, 1)
    messages = helper.create_message(frames)
    tok = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {
        "tokenized_data": tok,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
        "ego_future_xyz": data["ego_future_xyz"],
        "ego_future_rot": data["ego_future_rot"],
    }


def calc_min_ade(gt_future_xy, pred_xyz):
    """minADE across trajectory samples and single-sample ADE.

    Args:
        gt_future_xy: [B, groups, T, 2+]
        pred_xyz:     [B, sets, samples, T, 3]

    Returns:
        (minADE_K, minADE_1) where K = number of trajectory samples.
        minADE_1 is the ADE of a single sample (sample index 0).
    """
    gt_xy = gt_future_xy.cpu()[0, 0, :, :2].T.numpy()          # (2, T)
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2]            # (samples, T, 2)
    pred_xy = pred_xy.transpose(0, 2, 1)                        # (samples, 2, T)
    diff = np.linalg.norm(pred_xy - gt_xy[None], axis=1)        # (samples, T)
    ade_per_sample = diff.mean(axis=-1)                          # (samples,)
    return float(ade_per_sample.min()), float(ade_per_sample[0])  # minADE_K, minADE_1


def validate_clip(clip_id, avdi):
    """Check clip has enough egomotion data for the eval range."""
    ego = avdi.get_clip_feature(
        clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True,
    )
    ego_end = int(ego.timestamps[-1])
    # Need 6.4s of future after T0_END
    if ego_end <= T0_END_US + 6_400_000:
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description="System-opt eval (non-streaming, torch.compile)")
    ap.add_argument("--model-path", default="/data/scratch/zekaili/Alpamayo-R1-10B")
    ap.add_argument("--clip-ids-file", default="/data/scratch/zekaili/physicalai_av/clip_ids.json")
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--num-traj-samples", type=int, default=6,
                     help="K for minADE_K (default 6)")
    ap.add_argument("--diffusion-steps", type=int, default=10)
    ap.add_argument("--warmup-steps", type=int, default=3,
                     help="First N steps excluded from metrics")
    ap.add_argument("--output-dir", default="~/exp/eval_results")
    ap.add_argument("--cache-dir", default="/data/scratch/zekaili/physicalai_av/hf_cache")
    args = ap.parse_args()

    for attr in ("model_path", "clip_ids_file", "output_dir", "cache_dir"):
        setattr(args, attr, os.path.expanduser(getattr(args, attr)))

    run_dir = args.output_dir
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- dataset ---
    import physical_ai_av
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=args.cache_dir)

    clip_ids = load_clip_ids(args.clip_ids_file, args.num_clips)
    log.info(f"Loaded {len(clip_ids)} clips, K={args.num_traj_samples}, "
             f"diffusion_steps={args.diffusion_steps}")

    # --- model (dev branch: torch.compile + static cache, non-streaming) ---
    model = AlpamayoR1FlashDrive.from_pretrained(
        args.model_path, dtype=torch.bfloat16,
    ).to("cuda")
    processor = helper.get_processor(model.tokenizer)

    # --- eval loop ---
    t0s = list(range(T0_START_US, T0_END_US + 1, STEP_US))
    log.info(f"Per-clip: {len(t0s)} timesteps ({T0_START_US/1e6:.1f}s – {T0_END_US/1e6:.1f}s)")

    all_timing, all_ade_k, all_ade_1, all_results = [], [], [], []
    warmup_left = args.warmup_steps

    for ci, clip_id in enumerate(clip_ids):
        if not validate_clip(clip_id, avdi):
            log.warning(f"Clip {clip_id}: too short, skipping")
            continue

        log.info(f"\nClip {ci+1}/{len(clip_ids)}: {clip_id}")
        clip_timing, clip_ade_k, clip_ade_1 = [], [], []

        # Pre-create all inputs for this clip
        log.info(f"  Creating inputs for {len(t0s)} timesteps...")
        clip_inputs = []
        try:
            for t0 in t0s:
                data = load_physical_aiavdataset(
                    clip_id, t0_us=t0, num_frames=4, avdi=avdi,
                )
                clip_inputs.append(prepare_inputs(data, processor))
        except Exception as e:
            log.warning(f"  Error creating inputs: {e}")
            continue
        log.info(f"  Created {len(clip_inputs)} inputs, starting inference...")

        for si, (t0, inputs) in enumerate(zip(t0s, clip_inputs)):
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model.sample_trajectories_from_flashdrive(
                        data=helper.to_device(inputs, "cuda"),
                        streaming=False,
                        dflash=False,
                        num_traj_samples=args.num_traj_samples,
                        max_new_tokens=128,
                        return_extra=True,
                        fuse_qkv=True,
                        fuse_gate_up=True,
                        diffusion_kwargs={"inference_step": args.diffusion_steps},
                    )

                pred_xyz, pred_rot, extra = result
                if pred_xyz is None:
                    continue

                min_ade_k, min_ade_1 = calc_min_ade(inputs["ego_future_xyz"], pred_xyz)
                timing = extra.get("timing") if extra else None

                is_warmup = warmup_left > 0
                if is_warmup:
                    warmup_left -= 1
                tag = " [W]" if is_warmup else ""

                if timing:
                    enc = timing.get("encode_time_ms", 0)
                    pf = timing.get("prefill_time_ms", 0)
                    dec = timing.get("decode_time_ms", 0)
                    act = timing.get("action_time_ms", 0)
                    total = timing.get("total_time_ms", 0)
                    ntok = timing.get("num_decode_tokens", 0)
                    tps = ntok / (dec / 1000) if dec > 0 else 0
                    log.info(
                        f"  Step {si}{tag}: {total:.1f}ms "
                        f"(enc={enc:.1f} pf={pf:.1f} dec={dec:.1f} act={act:.1f}) "
                        f"{ntok}tok {tps:.1f}tok/s, "
                        f"minADE_{args.num_traj_samples}={min_ade_k:.3f}m, "
                        f"minADE_1={min_ade_1:.3f}m"
                    )

                if not is_warmup:
                    clip_ade_k.append(min_ade_k)
                    clip_ade_1.append(min_ade_1)
                    if timing:
                        clip_timing.append(timing)
                    all_results.append({
                        "clip_id": clip_id,
                        "t0_us": t0,
                        f"min_ade_{args.num_traj_samples}": min_ade_k,
                        "min_ade_1": min_ade_1,
                        **(timing or {}),
                    })

            except Exception as e:
                log.warning(f"  Step {si} error: {e}")
                if "CUDA" in str(e):
                    raise SystemExit(1)

        if clip_ade_k:
            ct = clip_timing
            avg_total = np.mean([t["total_time_ms"] for t in ct]) if ct else 0
            avg_dec = np.mean([t.get("decode_time_ms", 0) for t in ct]) if ct else 0
            avg_tok = np.mean([t.get("num_decode_tokens", 0) for t in ct]) if ct else 0
            avg_tps = avg_tok / (avg_dec / 1000) if avg_dec > 0 else 0
            log.info(
                f"  Clip avg: {avg_total:.1f}ms total, {avg_dec:.1f}ms decode, "
                f"{avg_tok:.0f}tok, {avg_tps:.1f}tok/s, "
                f"minADE_{args.num_traj_samples}={np.mean(clip_ade_k):.3f}m, "
                f"minADE_1={np.mean(clip_ade_1):.3f}m "
                f"({len(clip_ade_k)} steps)"
            )
            all_ade_k.extend(clip_ade_k)
            all_ade_1.extend(clip_ade_1)
            all_timing.extend(clip_timing)

    # --- summary ---
    n = len(all_timing)
    log.info(f"\n{'='*60}")
    log.info(f"AGGREGATE RESULTS — System-opt (non-streaming, torch.compile) ({n} steps)")
    log.info("=" * 60)
    if all_timing:
        def avg(key):
            vals = [t.get(key, 0) for t in all_timing]
            return np.mean(vals) if vals else 0

        avg_decode = avg("decode_time_ms")
        avg_tokens = avg("num_decode_tokens")
        avg_tps = avg_tokens / (avg_decode / 1000) if avg_decode > 0 else 0

        log.info(f"  Avg total time:      {avg('total_time_ms'):.1f} ms")
        log.info(f"  Avg encode (ViT):    {avg('encode_time_ms'):.1f} ms")
        log.info(f"  Avg prefill (LLM):   {avg('prefill_time_ms'):.1f} ms")
        log.info(f"  Avg decode:          {avg_decode:.1f} ms")
        log.info(f"  Avg tokens:          {avg_tokens:.1f}")
        log.info(f"  Avg tokens/sec:      {avg_tps:.1f}")
        log.info(f"  Avg action (diff):   {avg('action_time_ms'):.1f} ms")
    if all_ade_k:
        log.info(f"  Avg minADE_{args.num_traj_samples}:       {np.mean(all_ade_k):.3f} m")
        log.info(f"  Avg minADE_1:        {np.mean(all_ade_1):.3f} m")

    # --- save ---
    out_file = os.path.join(
        run_dir,
        f"system_opt_K{args.num_traj_samples}_d{args.diffusion_steps}_{len(clip_ids)}clips.json",
    )
    with open(out_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": vars(args),
            "summary": {
                f"min_ade_{args.num_traj_samples}": float(np.mean(all_ade_k)) if all_ade_k else None,
                "min_ade_1": float(np.mean(all_ade_1)) if all_ade_1 else None,
                "num_steps": len(all_ade_k),
                "num_clips": len(clip_ids),
                **({"avg_total_ms": float(np.mean([t["total_time_ms"] for t in all_timing])),
                    "avg_encode_ms": float(np.mean([t.get("encode_time_ms", 0) for t in all_timing])),
                    "avg_prefill_ms": float(np.mean([t.get("prefill_time_ms", 0) for t in all_timing])),
                    "avg_decode_ms": float(np.mean([t.get("decode_time_ms", 0) for t in all_timing])),
                    "avg_action_ms": float(np.mean([t.get("action_time_ms", 0) for t in all_timing])),
                    "avg_num_tokens": float(np.mean([t.get("num_decode_tokens", 0) for t in all_timing])),
                   } if all_timing else {}),
            },
            "samples": all_results,
        }, f, indent=2)
    log.info(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
