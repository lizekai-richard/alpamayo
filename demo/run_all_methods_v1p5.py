#!/usr/bin/env python3
"""Run progressive inference methods on a single clip (Alpamayo v1.5).

Methods (each builds on the previous):
    1. Baseline                 - vanilla vlm.generate(), no optimizations
    2. + System optimizations   - torch.compile + StaticCache + fused projections
    3. + DFlash                 - speculative decoding (non-streaming)
    4. + Streaming              - KV cache reuse across frames
    5. + Action Cache           - diffusion step caching (euler_with_cache)
    6. + ParoQuant INT4         - 4-bit quantization with pairwise rotation

Each method produces per-frame JSON data for video generation (same format
as run_all_methods.py — compatible with generate_frame.py / generate_videos.py).

Usage:
    python demo/run_all_methods_v1p5.py
    python demo/run_all_methods_v1p5.py --clip-index 2 --num-frames 50
    python demo/run_all_methods_v1p5.py --only 4,5,6
    python demo/run_all_methods_v1p5.py --only 6 --paro-checkpoint /path/to/paro.pt
"""

import argparse
import gc
import json
import logging
import os
import sys
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import alpamayo_r1
sys.modules["alpamayo1_5"] = alpamayo_r1

import numpy as np
import torch
from tqdm import tqdm

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")

STEP_US = 100_000  # 100ms = 10Hz

METHODS = OrderedDict([
    (1, {"name": "1_baseline",      "label": "Baseline"}),
    (2, {"name": "2_sys_opt",       "label": "+ System Opt"}),
    (3, {"name": "3_dflash",        "label": "+ DFlash"}),
    (4, {"name": "4_streaming",     "label": "+ Streaming"}),
    (5, {"name": "5_action_cache",  "label": "+ Action Cache"}),
    (6, {"name": "6_paroquant",     "label": "+ ParoQuant INT4"}),
])


# ─── Utilities ────────────────────────────────────────────────────────────


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


def load_clip_ids(clip_file):
    with open(os.path.expanduser(clip_file)) as f:
        return json.load(f)


def prepare_inputs(data, processor, is_prefill=None):
    frames = data["image_frames"].flatten(0, 1)
    messages = helper.create_message(frames, camera_indices=data["camera_indices"])
    tok = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    result = {
        "tokenized_data": tok,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
        "ego_future_xyz": data["ego_future_xyz"],
        "ego_future_rot": data["ego_future_rot"],
    }
    if is_prefill is not None:
        result["is_prefill"] = is_prefill
    return result


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


def make_frame_data(frame_idx, t0_us, timing, extra, pred_xyz,
                    *, is_prefill=False, is_warmup=False,
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


# ─── State management ─────────────────────────────────────────────────────


def reset_streaming_state(model, next_mode="streaming"):
    with torch.inference_mode():
        model.reset_streaming_state()
        model._past_key_values = None

    for attr in ("_cached_pos_embeds", "_cached_position_embeddings", "_cached_cu_seqlens"):
        if hasattr(model.vlm.model.visual, attr):
            delattr(model.vlm.model.visual, attr)
    if hasattr(model.vlm.model.language_model, "_cached_deepstack_indices"):
        delattr(model.vlm.model.language_model, "_cached_deepstack_indices")

    for module in model.modules():
        if hasattr(module, "mode") and type(module).__name__ == "Qwen3VLTextAttention":
            module.mode = next_mode

    attrs_to_clear = [
        attr for attr in list(model.__dict__.keys())
        if attr.startswith("_") and any(p in attr for p in [
            "_encode_", "_prefill_", "_dp_", "_dflash_draft", "_dflash_prefill",
            "_compiled_", "_action_", "_decode_", "_traj_fwd_",
            "_patched_for_compile",
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


# ─── Model loading ────────────────────────────────────────────────────────


def load_baseline_model(model_path):
    """Load Alpamayo1_5 for baseline (vlm.generate, no torch.compile)."""
    from alpamayo_r1.models.alpamayo_r1p5 import Alpamayo1_5

    log.info(f"Loading v1.5 baseline model from {model_path}...")
    model = Alpamayo1_5.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda")
    model.eval()
    return model


def load_flashdrive_model(model_path):
    """Load Alpamayo1_5FlashDrive (torch.compile + streaming + DFlash capable)."""
    from alpamayo_r1.models.alpamayo_r1p5_flashdrive import Alpamayo1_5FlashDrive

    log.info(f"Loading v1.5 FlashDrive model from {model_path}...")
    model = Alpamayo1_5FlashDrive.from_pretrained(
        model_path, dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()
    return model


def load_paro_model(model_path, paro_checkpoint):
    """Load ParoQuant v1.5 model with Marlin W4A8 backend."""
    from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import (
        load_paroquant_model_v1p5,
        convert_wqlinear_layer,
    )
    from alpamayo_r1.utils.quantization.rotation_linear import RotateLinearInt4
    from alpamayo_r1.utils.quantization.qmodule import WQLinear
    from alpamayo_r1.utils.system.patches import fuse_expert_projections

    log.info(f"Loading v1.5 ParoQuant model from {model_path}...")
    log.info(f"  Checkpoint: {paro_checkpoint}")
    model = load_paroquant_model_v1p5(
        model_path=model_path,
        paro_checkpoint=paro_checkpoint,
        mode="streaming",
    )

    # Convert WQLinear → MarlinW4A8Linear
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
    log.info(f"  Converted {n} WQLinear -> MarlinW4A8Linear")

    fuse_expert_projections(model)
    return model


def setup_dflash(model, draft_model_path):
    from alpamayo_r1.utils.dflash.dflash_integration import setup_dflash_for_model
    log.info(f"Setting up DFlash from {draft_model_path}...")
    setup_dflash_for_model(model, draft_model_path)


def unload_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ─── Method runners ───────────────────────────────────────────────────────


def run_baseline(model, processor, all_t0s, clip_id, avdi, output_dir,
                 *, method_name, num_samples, max_tokens, diffusion_steps,
                 warmup_steps, gt_cutoff_us=None):
    """Method 1: Baseline with vlm.generate(), no optimizations."""
    log.info(f"\n{'=' * 60}")
    log.info(f"Running: {method_name}")
    log.info(f"  non-streaming, vlm.generate, diffusion_steps={diffusion_steps}")
    log.info(f"{'=' * 60}")

    # Pre-load all data
    log.info(f"  Pre-loading data for {len(all_t0s)} timesteps...")
    all_inputs, all_gt = [], []
    for t0_us in all_t0s:
        has_gt = gt_cutoff_us is None or t0_us <= gt_cutoff_us
        nfs = 64 if has_gt else 1
        data = load_physical_aiavdataset(
            clip_id, t0_us=t0_us, num_frames=4, num_future_steps=nfs, avdi=avdi,
        )
        all_inputs.append(prepare_inputs(data, processor))
        all_gt.append((has_gt, data["ego_future_xyz"] if has_gt else None))

    stats = []
    pbar = tqdm(enumerate(all_t0s), total=len(all_t0s), desc=method_name[:25])

    for frame_idx, t0_us in pbar:
        try:
            inputs = all_inputs[frame_idx]
            has_gt, ego_future_xyz = all_gt[frame_idx]

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                result = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=helper.to_device(inputs, "cuda"),
                    num_traj_samples=num_samples,
                    diffusion_kwargs={"inference_step": diffusion_steps},
                    return_extra=True,
                )

            pred_xyz, pred_rot, extra = result
            if pred_xyz is None:
                continue

            timing = extra.get("timing", {}) if extra else {}
            is_warmup = frame_idx < warmup_steps
            min_ade_k, min_ade_1 = calc_min_ade(ego_future_xyz, pred_xyz) if has_gt else (None, None)
            frame_data = make_frame_data(
                frame_idx, t0_us, timing, extra, pred_xyz,
                is_warmup=is_warmup, min_ade_k=min_ade_k, min_ade_1=min_ade_1,
            )
            save_frame(output_dir, frame_idx, frame_data)
            stats.append(frame_data)
            pbar.set_postfix({"ms": f"{frame_data['total_ms']:.0f}"})

        except Exception as e:
            log.warning(f"  Frame {frame_idx} error: {e}")
            log.warning(traceback.format_exc())
            if "CUDA" in str(e):
                raise

    return stats


def run_nonstreaming(model, processor, all_t0s, clip_id, avdi, output_dir,
                     *, method_name, dflash, num_samples, max_tokens,
                     diffusion_steps, warmup_steps, gt_cutoff_us=None,
                     fuse_qkv=True, fuse_gate_up=True):
    """Methods 2-3: Non-streaming FlashDrive inference."""
    log.info(f"\n{'=' * 60}")
    log.info(f"Running: {method_name}")
    log.info(f"  streaming=False, dflash={dflash}, diffusion_steps={diffusion_steps}")
    log.info(f"{'=' * 60}")

    log.info(f"  Pre-loading data for {len(all_t0s)} timesteps...")
    all_inputs, all_gt = [], []
    for t0_us in all_t0s:
        has_gt = gt_cutoff_us is None or t0_us <= gt_cutoff_us
        nfs = 64 if has_gt else 1
        data = load_physical_aiavdataset(
            clip_id, t0_us=t0_us, num_frames=4, num_future_steps=nfs, avdi=avdi,
        )
        all_inputs.append(prepare_inputs(data, processor))
        all_gt.append((has_gt, data["ego_future_xyz"] if has_gt else None))

    stats = []
    pbar = tqdm(enumerate(all_t0s), total=len(all_t0s), desc=method_name[:25])

    for frame_idx, t0_us in pbar:
        try:
            inputs = all_inputs[frame_idx]
            has_gt, ego_future_xyz = all_gt[frame_idx]

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                result = model.sample_trajectories_from_flashdrive(
                    data=helper.to_device(inputs, "cuda"),
                    streaming=False,
                    dflash=dflash,
                    torch_compile="max-autotune",
                    num_traj_samples=num_samples,
                    max_new_tokens=max_tokens,
                    return_extra=True,
                    fuse_qkv=fuse_qkv,
                    fuse_gate_up=fuse_gate_up,
                    diffusion_kwargs={"inference_step": diffusion_steps},
                )

            pred_xyz, pred_rot, extra = result
            if pred_xyz is None:
                continue

            timing = extra.get("timing", {}) if extra else {}
            is_warmup = frame_idx < warmup_steps
            min_ade_k, min_ade_1 = calc_min_ade(ego_future_xyz, pred_xyz) if has_gt else (None, None)
            frame_data = make_frame_data(
                frame_idx, t0_us, timing, extra, pred_xyz,
                is_warmup=is_warmup, min_ade_k=min_ade_k, min_ade_1=min_ade_1,
            )
            save_frame(output_dir, frame_idx, frame_data)
            stats.append(frame_data)
            pbar.set_postfix({"ms": f"{frame_data['total_ms']:.0f}"})

        except Exception as e:
            log.warning(f"  Frame {frame_idx} error: {e}")
            log.warning(traceback.format_exc())
            if "CUDA" in str(e):
                raise

    return stats


def run_streaming(model, processor, all_t0s, clip_id, avdi, output_dir,
                  *, method_name, dflash, num_samples, max_tokens,
                  diffusion_steps, warmup_steps, gt_cutoff_us=None,
                  fuse_qkv=True, fuse_gate_up=True,
                  diffusion_kwargs_extra=None):
    """Methods 4-6: Streaming FlashDrive inference."""
    log.info(f"\n{'=' * 60}")
    log.info(f"Running: {method_name}")
    log.info(f"  streaming=True, dflash={dflash}, diffusion_steps={diffusion_steps}")
    if diffusion_kwargs_extra:
        log.info(f"  diffusion_kwargs_extra={diffusion_kwargs_extra}")
    log.info(f"{'=' * 60}")

    diff_kwargs = {"inference_step": diffusion_steps}
    if diffusion_kwargs_extra:
        diff_kwargs.update(diffusion_kwargs_extra)

    stats = []
    pbar = tqdm(enumerate(all_t0s), total=len(all_t0s), desc=method_name[:25])

    for frame_idx, t0_us in pbar:
        try:
            has_gt = gt_cutoff_us is None or t0_us <= gt_cutoff_us
            nfs = 64 if has_gt else 1
            if frame_idx == 0:
                data = load_physical_aiavdataset(
                    clip_id, t0_us=t0_us, num_frames=4, num_future_steps=nfs, avdi=avdi,
                )
                inputs = prepare_inputs(data, processor, is_prefill=True)
            else:
                data = load_physical_aiavdataset(
                    clip_id, t0_us=t0_us, num_frames=1, num_future_steps=nfs, avdi=avdi,
                )
                inputs = prepare_inputs(data, processor, is_prefill=False)

            torch.compiler.cudagraph_mark_step_begin()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                result = model.sample_trajectories_from_flashdrive(
                    data=helper.to_device(inputs, "cuda"),
                    streaming=True,
                    dflash=dflash,
                    torch_compile="max-autotune",
                    num_traj_samples=num_samples,
                    max_new_tokens=max_tokens,
                    return_extra=True,
                    fuse_qkv=fuse_qkv,
                    fuse_gate_up=fuse_gate_up,
                    diffusion_kwargs=diff_kwargs,
                )

            if result is None or result[0] is None:
                prefill_frame = {
                    "frame": frame_idx, "t0_us": t0_us,
                    "is_prefill": True, "is_warmup": True,
                }
                save_frame(output_dir, frame_idx, prefill_frame)
                stats.append(prefill_frame)
                log.info(f"  Frame {frame_idx}: prefill")
                continue

            pred_xyz, pred_rot, extra = result
            timing = extra.get("timing", {}) if extra else {}
            is_warmup = frame_idx <= warmup_steps
            min_ade_k, min_ade_1 = calc_min_ade(data["ego_future_xyz"], pred_xyz) if has_gt else (None, None)
            frame_data = make_frame_data(
                frame_idx, t0_us, timing, extra, pred_xyz,
                is_warmup=is_warmup, min_ade_k=min_ade_k, min_ade_1=min_ade_1,
            )
            save_frame(output_dir, frame_idx, frame_data)
            stats.append(frame_data)
            pbar.set_postfix({"ms": f"{frame_data['total_ms']:.0f}"})

        except Exception as e:
            log.warning(f"  Frame {frame_idx} error: {e}")
            log.warning(traceback.format_exc())
            if "CUDA" in str(e):
                raise

    return stats


# ─── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run progressive inference methods on a single clip (Alpamayo v1.5)",
    )
    parser.add_argument("--model-path", default="/data/scratch/zekaili/Alpamayo1_5-Finetuned")
    parser.add_argument("--draft-model", default="/data/scratch/zekaili/Alpamayo1_5-DFlash")
    parser.add_argument("--paro-checkpoint",
                        default="/data/scratch/zekaili/quant_cache/alpamayo-1.5-finetuned-paro-w4-vlm-mm.pt")
    parser.add_argument("--clip-id", default=None, help="Clip ID (overrides --clip-index)")
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--clip-file", default="./clips.json")
    parser.add_argument("--start-us", type=int, default=None)
    parser.add_argument("--end-us", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output-dir", default="~/exp/demo_v1p5")
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--cache-steps", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--only", default="", help="Comma-separated method numbers (e.g. '4,5,6')")
    parser.add_argument("--list-clips", action="store_true")
    parser.add_argument("--cache-dir", default="/data/scratch/zekaili/physicalai_av/hf_cache")
    args = parser.parse_args()

    for attr in ("model_path", "draft_model", "paro_checkpoint",
                 "clip_file", "output_dir", "cache_dir"):
        setattr(args, attr, os.path.expanduser(getattr(args, attr)))

    if args.list_clips:
        clip_ids = load_clip_ids(args.clip_file)
        for i, cid in enumerate(clip_ids):
            print(f"  [{i}] {cid}")
        return

    if args.only:
        methods_to_run = set(int(x) for x in args.only.split(","))
    else:
        methods_to_run = set(range(1, 7))

    # ── Dataset ──
    import physical_ai_av
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=args.cache_dir)

    if args.clip_id is None:
        clip_ids = load_clip_ids(args.clip_file)
        args.clip_id = clip_ids[args.clip_index]

    egomotion = avdi.get_clip_feature(
        args.clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True,
    )
    ego_end_us = int(egomotion.timestamps[-1])
    gt_cutoff_us = ego_end_us - 6_400_000

    start_us = args.start_us or 1_700_000
    end_us = args.end_us or (ego_end_us - STEP_US)
    all_t0s = list(range(start_us, end_us + 1, STEP_US))
    if args.num_frames is not None and args.num_frames < len(all_t0s):
        all_t0s = all_t0s[:args.num_frames]

    # ── Output ──
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    output_root = Path(args.output_dir) / f"run_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    for mid in methods_to_run:
        (output_root / METHODS[mid]["name"]).mkdir(exist_ok=True)

    log.info("=" * 60)
    log.info("Demo Inference: Alpamayo v1.5")
    log.info("=" * 60)
    log.info(f"Clip:          {args.clip_id}")
    log.info(f"Time range:    {start_us/1e6:.2f}s - {end_us/1e6:.2f}s")
    log.info(f"Frames:        {len(all_t0s)}")
    log.info(f"K (samples):   {args.num_samples}")
    log.info(f"Methods:       {sorted(methods_to_run)}")
    log.info(f"Output:        {output_root}")

    config = {
        "clip_id": args.clip_id,
        "start_us": start_us,
        "end_us": end_us,
        "num_frames": len(all_t0s),
        "num_samples": args.num_samples,
        "max_tokens": args.max_tokens,
        "diffusion_steps": args.diffusion_steps,
        "cache_steps": args.cache_steps,
        "warmup_steps": args.warmup_steps,
        "methods": sorted(methods_to_run),
    }
    with open(output_root / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_summaries = {}

    def save_summary():
        summary = {"config": config, "methods": {}}
        for mid in sorted(all_summaries):
            summary["methods"][METHODS[mid]["name"]] = {
                "label": METHODS[mid]["label"],
                **all_summaries[mid],
            }
        if 1 in all_summaries and all_summaries[1].get("avg_total_ms", 0) > 0:
            baseline_total = all_summaries[1]["avg_total_ms"]
            speedups = {}
            for mid in sorted(all_summaries):
                if mid == 1:
                    continue
                s = all_summaries[mid]
                if s.get("avg_total_ms", 0) > 0:
                    speedups[METHODS[mid]["name"]] = {
                        "total_speedup": baseline_total / s["avg_total_ms"],
                    }
            summary["speedups_vs_baseline"] = speedups
        with open(output_root / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # ─── Method 1: Baseline ───────────────────────────────────────────
    if 1 in methods_to_run:
        model = load_baseline_model(args.model_path)
        processor = helper.get_processor(model.tokenizer)
        stats = run_baseline(
            model, processor, all_t0s, args.clip_id, avdi,
            output_root / METHODS[1]["name"],
            method_name=METHODS[1]["label"],
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
            diffusion_steps=args.diffusion_steps,
            warmup_steps=args.warmup_steps,
            gt_cutoff_us=gt_cutoff_us,
        )
        all_summaries[1] = summarize_stats(stats)
        save_summary()
        unload_model(model)

    # ─── Methods 2-5: FlashDrive (shared model) ──────────────────────
    fd_methods = sorted(methods_to_run & {2, 3, 4, 5})
    if fd_methods:
        model = load_flashdrive_model(args.model_path)
        processor = helper.get_processor(model.tokenizer)

        needs_dflash = bool(methods_to_run & {3, 4, 5})
        if needs_dflash:
            setup_dflash(model, args.draft_model)

        # Method 2: System Opt (non-streaming, no DFlash)
        if 2 in fd_methods:
            stats = run_nonstreaming(
                model, processor, all_t0s, args.clip_id, avdi,
                output_root / METHODS[2]["name"],
                method_name=METHODS[2]["label"],
                dflash=False,
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
                diffusion_steps=args.diffusion_steps,
                warmup_steps=args.warmup_steps,
                gt_cutoff_us=gt_cutoff_us,
            )
            all_summaries[2] = summarize_stats(stats)
            save_summary()

        # Method 3: DFlash (non-streaming)
        if 3 in fd_methods:
            reset_streaming_state(model, next_mode="non-streaming")
            stats = run_nonstreaming(
                model, processor, all_t0s, args.clip_id, avdi,
                output_root / METHODS[3]["name"],
                method_name=METHODS[3]["label"],
                dflash=True,
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
                diffusion_steps=args.diffusion_steps,
                warmup_steps=args.warmup_steps,
                gt_cutoff_us=gt_cutoff_us,
            )
            all_summaries[3] = summarize_stats(stats)
            save_summary()

        # Method 4: Streaming + DFlash
        if 4 in fd_methods:
            reset_streaming_state(model)
            stats = run_streaming(
                model, processor, all_t0s, args.clip_id, avdi,
                output_root / METHODS[4]["name"],
                method_name=METHODS[4]["label"],
                dflash=True,
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
                diffusion_steps=args.diffusion_steps,
                warmup_steps=args.warmup_steps,
                gt_cutoff_us=gt_cutoff_us,
            )
            all_summaries[4] = summarize_stats(stats)
            save_summary()

        # Method 5: Streaming + DFlash + Action Cache
        if 5 in fd_methods:
            reset_streaming_state(model)
            stats = run_streaming(
                model, processor, all_t0s, args.clip_id, avdi,
                output_root / METHODS[5]["name"],
                method_name=METHODS[5]["label"],
                dflash=True,
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
                diffusion_steps=8,
                warmup_steps=args.warmup_steps,
                gt_cutoff_us=gt_cutoff_us,
                diffusion_kwargs_extra={
                    "cache_steps": args.cache_steps,
                    "int_method": "euler_with_cache",
                },
            )
            all_summaries[5] = summarize_stats(stats)
            save_summary()

        unload_model(model)

    # ─── Method 6: ParoQuant ─────────────────────────────────────────
    if 6 in methods_to_run:
        model = load_paro_model(args.model_path, args.paro_checkpoint)
        processor = helper.get_processor(model.tokenizer)
        setup_dflash(model, args.draft_model)

        stats = run_streaming(
            model, processor, all_t0s, args.clip_id, avdi,
            output_root / METHODS[6]["name"],
            method_name=METHODS[6]["label"],
            dflash=True,
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
            diffusion_steps=8,
            warmup_steps=args.warmup_steps,
            gt_cutoff_us=gt_cutoff_us,
            fuse_qkv=False,
            fuse_gate_up=False,
            diffusion_kwargs_extra={
                "cache_steps": args.cache_steps,
                "int_method": "euler_with_cache",
            },
        )
        all_summaries[6] = summarize_stats(stats)
        save_summary()
        unload_model(model)

    # ─── Summary ──────────────────────────────────────────────────────
    save_summary()

    log.info(f"\n{'=' * 80}")
    log.info("SUMMARY")
    log.info(f"{'=' * 80}")
    log.info(f"{'Method':<30} {'Total':>8} {'Decode':>8} {'Diff':>6} {'Tok/s':>7} {'ADE_K':>7} {'ADE_1':>7}")
    log.info(f"{'-' * 30} {'-' * 8} {'-' * 8} {'-' * 6} {'-' * 7} {'-' * 7} {'-' * 7}")

    for mid in sorted(all_summaries):
        s = all_summaries[mid]
        if s["num_frames"] == 0:
            continue
        label = METHODS[mid]["label"]
        ade_str = ""
        if "avg_min_ade_k" in s:
            ade_str = f" {s['avg_min_ade_k']:>6.3f}m {s['avg_min_ade_1']:>6.3f}m"
        accept_str = ""
        if "avg_acceptance_rate" in s:
            accept_str = f"  accept={s['avg_acceptance_rate']:.0%}"
        log.info(
            f"{label:<30} {s['avg_total_ms']:>7.1f}ms {s['avg_decode_ms']:>7.1f}ms "
            f"{s['avg_diffusion_ms']:>5.1f}ms {s['avg_tokens_per_sec']:>6.1f}{ade_str}{accept_str}"
        )

    log.info(f"\nOutput: {output_root}")


if __name__ == "__main__":
    main()
