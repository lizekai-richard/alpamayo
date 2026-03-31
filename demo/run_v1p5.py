#!/usr/bin/env python3
"""Demo script for Alpamayo v1.5: Baseline vs Full-Stack Optimized.

Runs two inference modes on demo clips, saving per-frame JSON for
video generation. Inference logic matches the eval scripts exactly.

Modes:
    baseline  - Alpamayo1_5, vlm.generate(), no optimizations
                (matches eval/eval_baseline_v1p5.py)
    optimized - ParoQuant W4A8 + DFlash + Streaming + Action Cache
                (matches eval/eval_all_paro_w4a8_v1p5.py)

Output:
    <output-dir>/run_<timestamp>/<clip_id>/baseline/frame_XXXX.json
    <output-dir>/run_<timestamp>/<clip_id>/optimized/frame_XXXX.json
    <output-dir>/run_<timestamp>/<clip_id>/summary.json

Usage:
    python demo/run_v1p5.py                             # both modes, all clips
    python demo/run_v1p5.py --only baseline             # baseline only
    python demo/run_v1p5.py --only optimized            # optimized only
    python demo/run_v1p5.py --clip-index 0 --num-frames 30
    python demo/run_v1p5.py --clip-id <uuid>
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import alpamayo_r1
sys.modules["alpamayo1_5"] = alpamayo_r1

import numpy as np
import torch
from tqdm import tqdm

from alpamayo_r1 import helper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")

STEP_US = 100_000
T0_START_US = 1_700_000
T0_END_US = 13_600_000


# ─── Utilities ───────────────────────────────────────────────────────────


def calc_min_ade(gt_future_xy, pred_xyz):
    try:
        gt_xy = gt_future_xy.cpu()[0, 0, :, :2].T.numpy()
        pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
        diff = np.linalg.norm(pred_xy - gt_xy[None], axis=1)
        ade = diff.mean(axis=-1)
        return float(ade.min()), float(ade[0])
    except Exception as e:
        log.warning(f"calc_min_ade error: {e}")
        return float("inf"), float("inf")


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


def save_frame(output_dir, frame_idx, frame_data):
    with open(output_dir / f"frame_{frame_idx:04d}.json", "w") as f:
        json.dump(frame_data, f, indent=2)


def make_frame_data(frame_idx, timing, extra, pred_xyz,
                    *, t0_us=0, is_prefill=False, is_warmup=False,
                    min_ade_k=None, min_ade_1=None):
    ntok = timing.get("num_decode_tokens", 0)
    dec_ms = timing.get("decode_time_ms", 0)

    frame = {
        "frame": frame_idx,
        "t0_us": t0_us,
        "is_prefill": is_prefill,
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
        frame["min_ade_k"] = min_ade_k
    if min_ade_1 is not None:
        frame["min_ade_1"] = min_ade_1

    ds = extra.get("dflash_stats") if extra else None
    if ds:
        frame["acceptance_rate"] = ds.get("acceptance_rate", 0)
        frame["mean_acceptance_length"] = ds.get("mean_acceptance_length", 0)
        frame["total_iterations"] = ds.get("total_iterations", 0)
        frame["acceptance_lengths"] = ds.get("acceptance_lengths", [])

    return frame


def summarize_stats(stats):
    valid = [s for s in stats if not s.get("is_prefill") and not s.get("is_warmup")]
    if not valid:
        return {"num_frames": 0}

    def avg(key):
        vals = [s[key] for s in valid if key in s]
        return float(np.mean(vals)) if vals else 0

    summary = {
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
    if any("acceptance_rate" in s for s in valid):
        summary["avg_acceptance_rate"] = avg("acceptance_rate")
        summary["avg_mean_acceptance_length"] = avg("mean_acceptance_length")
        summary["avg_iterations"] = avg("total_iterations")
    return summary


def unload_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ─── Baseline (matches eval/eval_baseline_v1p5.py) ──────────────────────


def run_baseline(args, clip_id, windows, output_dir, warmup_steps):
    """Baseline: Alpamayo1_5.sample_trajectories_from_data_with_vlm_rollout.

    Exactly mirrors eval/eval_baseline_v1p5.py inference loop.
    """
    from alpamayo_r1.models.alpamayo_r1p5 import Alpamayo1_5

    log.info(f"Loading baseline model from {args.baseline_model_path}...")
    model = Alpamayo1_5.from_pretrained(
        args.baseline_model_path, dtype=torch.bfloat16,
    ).to("cuda")

    stats = []
    warmup_left = warmup_steps
    pbar = tqdm(enumerate(windows), total=len(windows), desc="Baseline")

    for si, inputs in pbar:
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                result = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=helper.to_device(inputs, "cuda"),
                    num_traj_samples=args.num_samples,
                    max_generation_length=128,
                    return_extra=True,
                )

            pred_xyz, pred_rot, extra = result
            if pred_xyz is None:
                continue

            timing = extra.get("timing", {}) if extra else {}
            min_ade_k, min_ade_1 = calc_min_ade(inputs["ego_future_xyz"], pred_xyz)

            is_warmup = warmup_left > 0
            if is_warmup:
                warmup_left -= 1

            frame_data = make_frame_data(
                si, timing, extra, pred_xyz,
                t0_us=inputs.get("timestamp", 0),
                is_warmup=is_warmup,
                min_ade_k=min_ade_k, min_ade_1=min_ade_1,
            )
            save_frame(output_dir, si, frame_data)
            stats.append(frame_data)
            pbar.set_postfix({"ms": f"{frame_data['total_ms']:.0f}"})

        except Exception as e:
            log.warning(f"  Step {si} error: {e}")
            traceback.print_exc()
            if "CUDA" in str(e):
                raise

    unload_model(model)
    return stats


# ─── Optimized (matches eval/eval_all_paro_w4a8_v1p5.py) ───────────────


def _convert_model_to_marlin_w4a8(model):
    from alpamayo_r1.utils.quantization.rotation_linear import RotateLinearInt4
    from alpamayo_r1.utils.quantization.qmodule import WQLinear
    from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import convert_wqlinear_layer

    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, RotateLinearInt4):
            continue
        wq = mod.qlinear
        if not isinstance(wq, WQLinear):
            continue
        marlin, _ = convert_wqlinear_layer(
            wq.qweight.data, wq.scales.data, wq.scaled_zeros.data,
            wq.out_features, wq.in_features, wq.group_size,
            bias=wq.bias, device=wq.qweight.device,
        )
        mod.qlinear = marlin
        n += 1
        if n <= 3:
            log.info(f"  Converted {name}.qlinear -> MarlinW4A8Linear "
                     f"({wq.in_features}->{wq.out_features})")
    log.info(f"Converted {n} WQLinear -> MarlinW4A8Linear")
    return n


def _reset_clip_state(model, keep_frame_labels=False, kv_shift_mode="vision_only"):
    """Reset streaming and compiled state between clips.

    Mirrors eval/eval_all_paro_w4a8_v1p5.py reset_clip_state.
    """
    with torch.inference_mode():
        model.reset_streaming_state(
            keep_frame_labels=keep_frame_labels,
            kv_shift_mode=kv_shift_mode,
        )
        if model._past_key_values is not None:
            model._past_key_values.reset()

    for attr in (
        "_cached_pos_embeds", "_cached_position_embeddings", "_cached_cu_seqlens",
    ):
        if hasattr(model.vlm.model.visual, attr):
            delattr(model.vlm.model.visual, attr)
    if hasattr(model.vlm.model.language_model, "_cached_deepstack_indices"):
        delattr(model.vlm.model.language_model, "_cached_deepstack_indices")

    attrs_to_clear = [
        attr for attr in list(model.__dict__.keys())
        if attr.startswith("_") and any(pattern in attr for pattern in [
            "_encode_", "_prefill_", "_dp_", "_dflash_draft", "_dflash_prefill",
            "_compiled_", "_action_", "_decode_", "_traj_fwd_",
        ]) and attr not in [
            "_dflash_refs_initialized", "_dflash_embed_tokens", "_dflash_lm_head",
            "_dflash_language_model", "_dflash_target_layer_ids", "_dflash_block_size",
            "_dflash_mask_token_id", "_dflash_logits_processor",
        ]
    ]
    for attr in attrs_to_clear:
        if hasattr(model, attr):
            delattr(model, attr)
    torch._dynamo.reset()


def _build_streaming_inputs(windows, tokenizer, keep_frame_labels=False):
    """Convert dumped windows to streaming format.

    Mirrors eval/eval_all_paro_w4a8_v1p5.py create_or_load_streaming_inputs.
    """
    vs_id = tokenizer.encode("<|vision_start|>")[0]
    ve_id = tokenizer.encode("<|vision_end|>")[0]

    streaming_inputs = []
    for i, w in enumerate(windows):
        if i == 0:
            data = {
                "tokenized_data": w["tokenized_data"],
                "ego_history_xyz": w["ego_history_xyz"],
                "ego_history_rot": w["ego_history_rot"],
                "is_prefill": True,
            }
        else:
            data = helper.convert_to_streaming_window_v1p5(
                w, ve_id,
                keep_frame_labels=keep_frame_labels,
                vision_start_id=vs_id if not keep_frame_labels else None,
            )
        streaming_inputs.append(data)
    return streaming_inputs


def run_optimized(args, clip_id, windows, output_dir, warmup_steps, model):
    """Full-stack optimized: ParoQuant W4A8 + DFlash + Streaming + Action Cache.

    Exactly mirrors eval/eval_all_paro_w4a8_v1p5.py inference loop.
    """
    _reset_clip_state(model, args.keep_frame_labels, args.kv_shift_mode)

    streaming_inputs = _build_streaming_inputs(
        windows, model.tokenizer, args.keep_frame_labels,
    )
    log.info(f"  Built {len(streaming_inputs)} streaming inputs")

    stats = []
    pbar = tqdm(enumerate(streaming_inputs), total=len(streaming_inputs), desc="Optimized")

    for si, inputs in pbar:
        try:
            torch.compiler.cudagraph_mark_step_begin()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                result = model.sample_trajectories_from_flashdrive(
                    data=helper.to_device(inputs, "cuda"),
                    streaming=True,
                    dflash=True,
                    num_traj_samples=args.num_samples,
                    max_new_tokens=128,
                    torch_compile="max-autotune",
                    return_extra=True,
                    fuse_qkv=False,
                    fuse_gate_up=False,
                    diffusion_kwargs={
                        "inference_step": args.diffusion_steps,
                        "cache_steps": args.cache_steps,
                        "int_method": "euler_with_cache",
                    },
                )

            if result is None or result[0] is None:
                prefill_frame = {
                    "frame": si, "t0_us": windows[si].get("timestamp", 0),
                    "is_prefill": True, "is_warmup": True,
                }
                save_frame(output_dir, si, prefill_frame)
                stats.append(prefill_frame)
                log.info(f"  Step {si}: prefill")
                continue

            pred_xyz, _, extra = result
            timing = extra.get("timing", {}) if extra else {}
            min_ade_k, min_ade_1 = calc_min_ade(
                windows[si]["ego_future_xyz"], pred_xyz,
            )

            is_warmup = si <= warmup_steps
            frame_data = make_frame_data(
                si, timing, extra, pred_xyz,
                t0_us=windows[si].get("timestamp", 0),
                is_warmup=is_warmup,
                min_ade_k=min_ade_k, min_ade_1=min_ade_1,
            )
            save_frame(output_dir, si, frame_data)
            stats.append(frame_data)
            pbar.set_postfix({"ms": f"{frame_data['total_ms']:.0f}"})

        except Exception as e:
            log.warning(f"  Step {si} error: {e}")
            traceback.print_exc()
            if "CUDA" in str(e):
                raise

    return stats


def load_optimized_model(args):
    """Load and prepare the full-stack optimized model.

    Mirrors eval/eval_all_paro_w4a8_v1p5.py model loading.
    """
    from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import load_paroquant_model_v1p5
    from alpamayo_r1.utils.system.patches import fuse_expert_projections
    from alpamayo_r1.utils.dflash.dflash_integration import setup_dflash_for_model

    log.info(f"Loading ParoQuant model from {args.model_path}...")
    log.info(f"  Checkpoint: {args.paro_checkpoint}")
    model = load_paroquant_model_v1p5(
        model_path=args.model_path,
        paro_checkpoint=args.paro_checkpoint,
        mode="streaming",
    )

    log.info("Converting to Marlin W4A8...")
    n = _convert_model_to_marlin_w4a8(model)
    if n == 0:
        raise SystemExit("No layers converted!")

    log.info("Fusing expert projections...")
    fuse_expert_projections(model)

    log.info(f"Setting up DFlash from {args.draft_model}...")
    setup_dflash_for_model(model, args.draft_model)
    log.info("Model ready: ParoQuant W4A8 + DFlash + Streaming + Action Cache")

    return model


# ─── Main ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Demo: Baseline vs Full-Stack Optimized (Alpamayo v1.5)",
    )
    # Model paths
    parser.add_argument("--baseline-model-path", default="/root/Alpamayo-1.5-10B")
    parser.add_argument("--model-path", default="/root/Alpamayo1_5-Finetuned",
                        help="Model path for optimized mode")
    parser.add_argument("--paro-checkpoint",
                        default="/root/alpamayo-1.5-finetuned-paro-w4-vlm-mm.pt")
    parser.add_argument("--draft-model", default="/root/Alpamayo1_5-DFlash")

    # Clip selection
    parser.add_argument("--clip-id", default=None, help="Run a single clip by ID")
    parser.add_argument("--clip-index", type=int, default=None,
                        help="Run a single clip by index")
    parser.add_argument("--clip-file", default="./demo_clips.json")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Limit number of frames per clip")

    # Inference params (match eval defaults)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--cache-steps", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--keep-frame-labels", action="store_true", default=False)
    parser.add_argument("--kv-shift-mode", type=str, default="vision_only",
                        choices=["block", "vision_only"])

    # I/O
    parser.add_argument("--dumped-data-dir", default="/root/dumped_eval_data_v1p5")
    parser.add_argument("--output-dir", default="./demo_v1p5")
    parser.add_argument("--cache-dir", default="~/.cache/huggingface")
    parser.add_argument("--only", default=None, choices=["baseline", "optimized"],
                        help="Run only one mode")
    args = parser.parse_args()

    for attr in ("baseline_model_path", "model_path", "paro_checkpoint",
                 "draft_model", "clip_file", "output_dir", "cache_dir",
                 "dumped_data_dir"):
        setattr(args, attr, os.path.expanduser(getattr(args, attr)))

    run_baseline_mode = args.only in (None, "baseline")
    run_optimized_mode = args.only in (None, "optimized")

    # ── Clip list ──
    with open(args.clip_file) as f:
        all_clip_ids = json.load(f)

    if args.clip_id is not None:
        clips_to_run = [args.clip_id]
    elif args.clip_index is not None:
        clips_to_run = [all_clip_ids[args.clip_index]]
    else:
        clips_to_run = all_clip_ids

    # ── Output ──
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    output_root = Path(args.output_dir) / f"run_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Demo: Alpamayo v1.5  —  Baseline vs Full-Stack Optimized")
    log.info("=" * 60)
    log.info(f"Clips:    {len(clips_to_run)}")
    log.info(f"Modes:    {['baseline'] * run_baseline_mode + ['optimized'] * run_optimized_mode}")
    log.info(f"Output:   {output_root}")

    num_frames = args.num_frames or 120

    all_clip_summaries = {}

    # ── Baseline ─────────────────────────────────────────────────
    if run_baseline_mode:
        for clip_id in clips_to_run:
            log.info(f"\n{'=' * 60}")
            log.info(f"[Baseline] Clip: {clip_id}")
            log.info(f"{'=' * 60}")

            windows = helper.load_dumped_inputs(args.dumped_data_dir, clip_id)
            windows = windows[:num_frames]
            log.info(f"  Loaded {len(windows)} windows")

            clip_dir = output_root / clip_id
            baseline_dir = clip_dir / "baseline"
            baseline_dir.mkdir(parents=True, exist_ok=True)

            stats = run_baseline(args, clip_id, windows, baseline_dir, args.warmup_steps)
            summary = summarize_stats(stats)
            all_clip_summaries.setdefault(clip_id, {})["baseline"] = summary
            log.info(f"  Baseline done: {summary.get('num_frames', 0)} frames, "
                     f"avg {summary.get('avg_total_ms', 0):.1f}ms")

    # ── Optimized ────────────────────────────────────────────────
    if run_optimized_mode:
        model = load_optimized_model(args)

        for clip_id in clips_to_run:
            log.info(f"\n{'=' * 60}")
            log.info(f"[Optimized] Clip: {clip_id}")
            log.info(f"{'=' * 60}")

            windows = helper.load_dumped_inputs(args.dumped_data_dir, clip_id)
            windows = windows[:num_frames]
            log.info(f"  Loaded {len(windows)} windows")

            clip_dir = output_root / clip_id
            opt_dir = clip_dir / "optimized"
            opt_dir.mkdir(parents=True, exist_ok=True)

            stats = run_optimized(
                args, clip_id, windows, opt_dir, args.warmup_steps, model,
            )
            summary = summarize_stats(stats)
            all_clip_summaries.setdefault(clip_id, {})["optimized"] = summary
            log.info(f"  Optimized done: {summary.get('num_frames', 0)} frames, "
                     f"avg {summary.get('avg_total_ms', 0):.1f}ms")

        unload_model(model)

    # ── Save summaries & print ───────────────────────────────────
    for clip_id in clips_to_run:
        clip_dir = output_root / clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        clip_data = all_clip_summaries.get(clip_id, {})

        summary_json = {
            "clip_id": clip_id,
            "num_samples": args.num_samples,
            "diffusion_steps": args.diffusion_steps,
            "cache_steps": args.cache_steps,
            "warmup_steps": args.warmup_steps,
            "methods": clip_data,
        }

        bl = clip_data.get("baseline", {})
        opt = clip_data.get("optimized", {})
        if bl.get("avg_total_ms", 0) > 0 and opt.get("avg_total_ms", 0) > 0:
            summary_json["speedup"] = bl["avg_total_ms"] / opt["avg_total_ms"]

        with open(clip_dir / "summary.json", "w") as f:
            json.dump(summary_json, f, indent=2)

        # Print table
        log.info(f"\n{'=' * 80}")
        log.info(f"SUMMARY  clip={clip_id}")
        log.info(f"{'=' * 80}")
        log.info(f"{'Method':<15} {'Total':>8} {'Encode':>8} {'Prefill':>8} "
                 f"{'Decode':>8} {'Diff':>8} {'Tok/s':>7} {'ADE_K':>7} {'ADE_1':>7}")
        log.info("-" * 90)
        for name, s in clip_data.items():
            if s.get("num_frames", 0) == 0:
                continue
            ade_k = f"{s['avg_min_ade_k']:.3f}m" if "avg_min_ade_k" in s else "  N/A"
            ade_1 = f"{s['avg_min_ade_1']:.3f}m" if "avg_min_ade_1" in s else "  N/A"
            log.info(
                f"{name:<15} {s['avg_total_ms']:>7.1f}ms {s['avg_encode_ms']:>7.1f}ms "
                f"{s['avg_prefill_ms']:>7.1f}ms {s['avg_decode_ms']:>7.1f}ms "
                f"{s['avg_diffusion_ms']:>7.1f}ms {s['avg_tokens_per_sec']:>6.1f} "
                f"{ade_k:>7} {ade_1:>7}"
            )
        if "speedup" in summary_json:
            log.info(f"  Speedup: {summary_json['speedup']:.2f}x")

    log.info(f"\nOutput: {output_root}")


if __name__ == "__main__":
    main()
