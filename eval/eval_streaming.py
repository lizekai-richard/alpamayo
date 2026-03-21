#!/usr/bin/env python3
"""Streaming evaluation (no DFlash): torch.compile + KV cache reuse across frames.

Uses streaming=True, dflash=False — torch.compile + static cache + autoregressive
decode + KV cache reuse across frames. No speculative decoding.
First frame per clip is a prefill (returns None), subsequent frames stream incrementally.
Runs through clips from 1.7s to 13.6s at 10Hz (120 timesteps).
Reports minADE_K (default K=6) and per-step timing.

Usage:
    python eval/eval_streaming.py
    python eval/eval_streaming.py --num-clips 10 --diffusion-steps 5
    python eval/eval_streaming.py --num-traj-samples 1
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
    return list(dict.fromkeys(ids))[:num_clips]


def prepare_inputs(data, processor, is_prefill=False):
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
        "is_prefill": is_prefill,
    }


def create_streaming_inputs(processor, clip_id, avdi):
    """Create sliding-window streaming inputs for one clip.

    Window 0 (prefill): 4 cameras x 4 frames = 16 frames
    Window 1+ (streaming): 4 cameras x 1 new frame = 4 frames
    """
    all_t0s = list(range(T0_START_US, T0_END_US + 1, STEP_US))
    streaming_inputs = []

    for i, t0 in enumerate(all_t0s):
        if i == 0:
            data = load_physical_aiavdataset(clip_id, t0_us=t0, num_frames=4, avdi=avdi)
            inputs = prepare_inputs(data, processor, is_prefill=True)
            seq_len = inputs["tokenized_data"]["input_ids"].shape[-1]
            log.info(f"  Prefill input: {seq_len} tokens")
        else:
            data = load_physical_aiavdataset(clip_id, t0_us=t0, num_frames=1, avdi=avdi)
            inputs = prepare_inputs(data, processor, is_prefill=False)
        streaming_inputs.append(inputs)

    return streaming_inputs


def calc_min_ade(gt_future_xy, pred_xyz):
    """minADE across trajectory samples and single-sample ADE.

    Args:
        gt_future_xy: [B, groups, T, 2+]
        pred_xyz:     [B, sets, samples, T, 3]

    Returns:
        (minADE_K, minADE_1) where K = number of trajectory samples.
    """
    try:
        gt_xy = gt_future_xy.cpu()[0, 0, :, :2].T.numpy()
        pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2]
        pred_xy = pred_xy.transpose(0, 2, 1)
        diff = np.linalg.norm(pred_xy - gt_xy[None], axis=1)
        ade_per_sample = diff.mean(axis=-1)
        return float(ade_per_sample.min()), float(ade_per_sample[0])
    except Exception as e:
        log.warning(f"calc_min_ade error: {e}")
        return float("inf"), float("inf")


def validate_clip(clip_id, avdi):
    """Check clip has enough egomotion data for the eval range."""
    ego = avdi.get_clip_feature(
        clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True,
    )
    ego_end = int(ego.timestamps[-1])
    if ego_end <= T0_END_US + 6_400_000:
        return False
    return True


def reset_clip_state(model):
    """Reset all streaming and compiled state between clips."""
    with torch.inference_mode():
        model.reset_streaming_state()
        if model._past_key_values is not None:
            model._past_key_values.reset()

    # Clear VLM internal cached embeddings (shape mismatch between clips)
    for attr in (
        "_cached_pos_embeds", "_cached_position_embeddings", "_cached_cu_seqlens",
    ):
        if hasattr(model.vlm.model.visual, attr):
            delattr(model.vlm.model.visual, attr)
    if hasattr(model.vlm.model.language_model, "_cached_deepstack_indices"):
        delattr(model.vlm.model.language_model, "_cached_deepstack_indices")

    # Clear compiled functions and buffers to avoid CUDAGraph tensor reference issues
    attrs_to_clear = [
        attr for attr in list(model.__dict__.keys())
        if attr.startswith("_") and any(pattern in attr for pattern in [
            "_encode_", "_prefill_", "_dp_",
            "_compiled_", "_action_", "_decode_",
        ]) and attr not in [
            "_dflash_refs_initialized", "_dflash_embed_tokens", "_dflash_lm_head",
            "_dflash_language_model", "_dflash_target_layer_ids", "_dflash_block_size",
            "_dflash_mask_token_id", "_dflash_logits_processor",
        ]
    ]
    for attr in attrs_to_clear:
        delattr(model, attr)
    torch._dynamo.reset()


def main():
    ap = argparse.ArgumentParser(description="Streaming eval (torch.compile + KV reuse, no DFlash)")
    ap.add_argument("--model-path", default="/data/scratch/zekaili/Alpamayo-R1-10B")
    ap.add_argument("--clip-ids-file", default="/data/scratch/zekaili/physicalai_av/clip_ids.json")
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--num-traj-samples", type=int, default=6,
                     help="K for minADE_K (default 6)")
    ap.add_argument("--max-tokens", type=int, default=128,
                     help="Max CoC tokens to generate")
    ap.add_argument("--diffusion-steps", type=int, default=10)
    ap.add_argument("--warmup-steps", type=int, default=3,
                     help="First N streaming steps per clip excluded from metrics (on top of prefill)")
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

    # --- model ---
    model = AlpamayoR1FlashDrive.from_pretrained(
        args.model_path, dtype=torch.bfloat16,
    ).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    log.info("Streaming eval (no DFlash)")

    # --- eval loop ---
    num_timesteps = len(list(range(T0_START_US, T0_END_US + 1, STEP_US)))
    log.info(f"Per-clip: {num_timesteps} timesteps ({T0_START_US/1e6:.1f}s – {T0_END_US/1e6:.1f}s)")

    all_timing, all_ade_k, all_ade_1, all_results = [], [], [], []

    for ci, clip_id in enumerate(clip_ids):
        if not validate_clip(clip_id, avdi):
            log.warning(f"Clip {clip_id}: too short, skipping")
            continue

        log.info(f"\nClip {ci+1}/{len(clip_ids)}: {clip_id}")

        # Reset streaming state for each clip
        reset_clip_state(model)

        clip_timing, clip_ade_k, clip_ade_1 = [], [], []

        # Create streaming inputs (prefill + streaming windows)
        log.info(f"  Creating streaming inputs...")
        try:
            streaming_inputs = create_streaming_inputs(processor, clip_id, avdi)
        except Exception as e:
            log.warning(f"  Error creating inputs: {e}")
            continue
        log.info(f"  Created {len(streaming_inputs)} streaming inputs, starting inference...")

        for si, inputs in enumerate(streaming_inputs):
            try:
                torch.compiler.cudagraph_mark_step_begin()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model.sample_trajectories_from_flashdrive(
                        data=helper.to_device(inputs, "cuda"),
                        torch_compile="max-autotune",
                        streaming=True,
                        dflash=False,
                        top_p=0.98,
                        temperature=0.6,
                        max_new_tokens=128,
                        num_traj_samples=args.num_traj_samples,
                        return_extra=True,
                        fuse_qkv=True,
                        fuse_gate_up=True,
                        diffusion_kwargs={"inference_step": args.diffusion_steps},
                    )

                if result is None or result[0] is None:
                    # Prefill returns None
                    log.info(f"  Step {si}: prefill (no output)")
                    continue

                pred_xyz, pred_rot, extra = result
                min_ade_k, min_ade_1 = calc_min_ade(inputs["ego_future_xyz"], pred_xyz)
                timing = extra.get("timing") if extra else None

                # First N streaming steps per clip are warmup (step 0 is prefill → None)
                is_warmup = si <= args.warmup_steps
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
                        "step": si,
                        f"min_ade_{args.num_traj_samples}": min_ade_k,
                        "min_ade_1": min_ade_1,
                        **(timing or {}),
                    })

            except Exception as e:
                log.warning(f"  Step {si} error: {e}")
                if "CUDA" in str(e) or "device-side assert" in str(e):
                    log.error("CUDA error detected, exiting to avoid GPU hang")
                    raise SystemExit(1)
                continue

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
    log.info(f"AGGREGATE RESULTS — Streaming (torch.compile + KV reuse, no DFlash) ({n} steps)")
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
        f"streaming_K{args.num_traj_samples}_d{args.diffusion_steps}_{len(clip_ids)}clips.json",
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
                **({
                    "avg_total_ms": float(np.mean([t["total_time_ms"] for t in all_timing])),
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
