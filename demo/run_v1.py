#!/usr/bin/env python3
"""Demo script for Alpamayo v1: Baseline vs Full-Stack Optimized.

Runs two inference modes on demo clips, saving per-frame JSON for
video generation. Inference logic matches the eval scripts exactly.

Modes:
    baseline  - AlpamayoR1, vlm.generate(), no optimizations
                (matches eval/eval_system_opt.py baseline path)
    optimized - ParoQuant W4A8 + DFlash + Streaming + Action Cache
                (matches eval/eval_all_paro_w4a8.py)

Output:
    <output-dir>/run_<timestamp>/<clip_id>/baseline/frame_XXXX.json
    <output-dir>/run_<timestamp>/<clip_id>/optimized/frame_XXXX.json
    <output-dir>/run_<timestamp>/<clip_id>/summary.json

Usage:
    python demo/run_v1.py                               # both modes, all clips
    python demo/run_v1.py --only baseline               # baseline only
    python demo/run_v1.py --only optimized              # optimized only
    python demo/run_v1.py --clip-index 0 --num-frames 30
    python demo/run_v1.py --clip-id <uuid>
"""

import argparse
import gc
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from tqdm import tqdm

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

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
                    *, is_prefill=False, is_warmup=False,
                    min_ade_k=None, min_ade_1=None):
    ntok = timing.get("num_decode_tokens", 0)
    dec_ms = timing.get("decode_time_ms", 0)

    frame = {
        "frame": frame_idx,
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


def prepare_inputs(data, processor):
    """Prepare tokenized inputs. Matches eval/eval_system_opt.py prepare_inputs."""
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


# ─── Baseline (matches eval/eval_system_opt.py with base model) ─────────


def run_baseline(args, clip_id, all_t0s, avdi, output_dir, warmup_steps,
                 *, data_dir=None, gt_cutoff_us=None):
    """Baseline: AlpamayoR1.sample_trajectories_from_data_with_vlm_rollout.

    Exactly mirrors the baseline inference path.
    """
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    log.info(f"Loading baseline model from {args.baseline_model_path}...")
    model = AlpamayoR1.from_pretrained(
        args.baseline_model_path, dtype=torch.bfloat16,
    ).to("cuda")
    processor = helper.get_processor(model.tokenizer)

    # Pre-load all data
    log.info(f"  Pre-loading data for {len(all_t0s)} timesteps...")
    all_inputs = []
    all_gt = []
    for t0_us in all_t0s:
        has_gt = gt_cutoff_us is None or t0_us <= gt_cutoff_us
        if data_dir:
            from demo.run_all_methods import load_cached_data
            data = load_cached_data(data_dir, t0_us, num_frames=4)
        else:
            data = load_physical_aiavdataset(
                clip_id, t0_us=t0_us, num_frames=4, avdi=avdi,
            )
        all_inputs.append(prepare_inputs(data, processor))
        all_gt.append((has_gt, data["ego_future_xyz"] if has_gt else None))

    stats = []
    warmup_left = warmup_steps
    pbar = tqdm(enumerate(all_t0s), total=len(all_t0s), desc="Baseline")

    for si, t0_us in pbar:
        try:
            inputs = all_inputs[si]
            has_gt, ego_future_xyz = all_gt[si]

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                result = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=helper.to_device(inputs, "cuda"),
                    num_traj_samples=args.num_samples,
                    max_generation_length=128,
                    return_extra=True,
                    diffusion_kwargs={"inference_step": args.diffusion_steps},
                )

            pred_xyz, pred_rot, extra = result
            if pred_xyz is None:
                continue

            timing = extra.get("timing", {}) if extra else {}

            is_warmup = warmup_left > 0
            if is_warmup:
                warmup_left -= 1

            min_ade_k, min_ade_1 = calc_min_ade(ego_future_xyz, pred_xyz) if has_gt else (None, None)
            frame_data = make_frame_data(
                si, timing, extra, pred_xyz,
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


# ─── Optimized (matches eval/eval_all_paro_w4a8.py) ─────────────────────


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


def _fuse_expert_projections(model, mode="streaming"):
    """Fuse expert QKV and gate/up projections.

    Mirrors eval/eval_all_paro_w4a8.py fuse_expert_projections.
    """
    from alpamayo_r1.utils.system.patches import (
        Qwen3VLTextAttention as PatchedAttn,
        Qwen3VLTextMLP as PatchedMLP,
    )
    n_attn = n_mlp = 0
    for layer in model.expert.layers:
        attn = layer.self_attn
        if hasattr(attn, "q_proj") and not getattr(attn, "fuse_qkv", False):
            config = attn.config if hasattr(attn, "config") else layer.self_attn.config
            layer_idx = getattr(attn, "layer_idx", 0)
            device, dtype = attn.q_proj.weight.device, attn.q_proj.weight.dtype
            new_attn = PatchedAttn(config, layer_idx, mode=mode, fuse_qkv=True)
            new_attn.o_proj = attn.o_proj
            if hasattr(attn, "q_norm"):
                new_attn.q_norm = attn.q_norm
            if hasattr(attn, "k_norm"):
                new_attn.k_norm = attn.k_norm
            with torch.no_grad():
                q_s = attn.q_proj.weight.shape[0]
                k_s = attn.k_proj.weight.shape[0]
                new_attn.qkv_proj.weight[:q_s].copy_(attn.q_proj.weight)
                new_attn.qkv_proj.weight[q_s:q_s + k_s].copy_(attn.k_proj.weight)
                new_attn.qkv_proj.weight[q_s + k_s:].copy_(attn.v_proj.weight)
                if new_attn.qkv_proj.bias is not None:
                    new_attn.qkv_proj.bias[:q_s].copy_(attn.q_proj.bias)
                    new_attn.qkv_proj.bias[q_s:q_s + k_s].copy_(attn.k_proj.bias)
                    new_attn.qkv_proj.bias[q_s + k_s:].copy_(attn.v_proj.bias)
            layer.self_attn = new_attn.to(device=device, dtype=dtype)
            n_attn += 1

        mlp = layer.mlp
        if hasattr(mlp, "gate_proj") and not getattr(mlp, "fuse_gate_up", False):
            config = mlp.config if hasattr(mlp, "config") else getattr(layer, "config", None)
            if config is None:
                from types import SimpleNamespace
                config = SimpleNamespace(
                    hidden_size=mlp.gate_proj.in_features,
                    intermediate_size=mlp.gate_proj.out_features,
                    hidden_act=getattr(mlp, "hidden_act", "silu"),
                )
            device, dtype = mlp.gate_proj.weight.device, mlp.gate_proj.weight.dtype
            new_mlp = PatchedMLP(config, fuse_gate_up=True)
            new_mlp.down_proj = mlp.down_proj
            new_mlp.act_fn = mlp.act_fn
            with torch.no_grad():
                g_s = mlp.gate_proj.weight.shape[0]
                new_mlp.gate_up_proj.weight[:g_s].copy_(mlp.gate_proj.weight)
                new_mlp.gate_up_proj.weight[g_s:].copy_(mlp.up_proj.weight)
                if new_mlp.gate_up_proj.bias is not None:
                    new_mlp.gate_up_proj.bias[:g_s].copy_(mlp.gate_proj.bias)
                    new_mlp.gate_up_proj.bias[g_s:].copy_(mlp.up_proj.bias)
            layer.mlp = new_mlp.to(device=device, dtype=dtype)
            n_mlp += 1
    log.info(f"Fused expert projections: {n_attn} QKV + {n_mlp} gate/up")


def _reset_clip_state(model):
    """Reset streaming and compiled state between clips.

    Mirrors eval/eval_all_paro_w4a8.py reset_clip_state.
    """
    with torch.inference_mode():
        model.reset_streaming_state()
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
        a for a in list(model.__dict__.keys())
        if a.startswith("_") and any(p in a for p in [
            "_encode_", "_prefill_", "_dp_", "_dflash_draft", "_dflash_prefill",
            "_compiled_", "_action_", "_decode_", "_traj_fwd_",
        ]) and a not in [
            "_dflash_refs_initialized", "_dflash_embed_tokens", "_dflash_lm_head",
            "_dflash_language_model", "_dflash_target_layer_ids", "_dflash_block_size",
            "_dflash_mask_token_id", "_dflash_logits_processor",
        ]
    ]
    for a in attrs_to_clear:
        if hasattr(model, a):
            delattr(model, a)
    torch._dynamo.reset()


def _build_streaming_inputs(windows, tokenizer):
    """Convert dumped 16-frame windows to streaming format.

    Window 0 is kept as prefill. Windows 1+ are converted from 16-frame
    to 4-frame streaming format via helper.convert_to_streaming_window.
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
            data = helper.convert_to_streaming_window(w, vs_id, ve_id)
        streaming_inputs.append(data)
    return streaming_inputs


def run_optimized(args, clip_id, output_dir, warmup_steps, model):
    """Full-stack optimized: ParoQuant W4A8 + DFlash + Streaming + Action Cache.

    Exactly mirrors eval/eval_all_paro_w4a8.py inference loop.
    """
    _reset_clip_state(model)

    windows = helper.load_dumped_inputs(args.dumped_data_dir, clip_id)
    num_frames = args.num_frames or len(windows)
    windows = windows[:num_frames]

    streaming_inputs = _build_streaming_inputs(windows, model.tokenizer)
    log.info(f"  Built {len(streaming_inputs)} streaming inputs from {len(windows)} windows")

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
                    "frame": si, "is_prefill": True, "is_warmup": True,
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

    Mirrors eval/eval_all_paro_w4a8.py model loading.
    """
    from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import load_paroquant_model
    from alpamayo_r1.utils.dflash.dflash_integration import setup_dflash_for_model

    log.info(f"Loading ParoQuant model from {args.model_path}...")
    log.info(f"  Checkpoint: {args.paro_checkpoint}")
    model = load_paroquant_model(
        model_path=args.model_path,
        paro_checkpoint=args.paro_checkpoint,
        mode="streaming",
        quantize_expert=not args.quantize_vlm_only,
    )

    log.info("Converting to Marlin W4A8...")
    n = _convert_model_to_marlin_w4a8(model)
    if n == 0:
        raise SystemExit("No layers converted!")

    log.info("Fusing expert projections...")
    _fuse_expert_projections(model)

    log.info(f"Setting up DFlash from {args.draft_model}...")
    setup_dflash_for_model(model, args.draft_model)
    log.info("Model ready: ParoQuant W4A8 + DFlash + Streaming + Action Cache")

    return model


# ─── Main ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Demo: Baseline vs Full-Stack Optimized (Alpamayo v1)",
    )
    # Model paths
    parser.add_argument("--baseline-model-path", default="/data/scratch/zekaili/Alpamayo-R1-10B")
    parser.add_argument("--model-path", default="/data/scratch/zekaili/train_expert_ckpts_deepspeed/checkpoint-6446",
                        help="Model path for optimized mode")
    parser.add_argument("--paro-checkpoint",
                        default="/data/scratch/zekaili/quant_cache/ckpt-paro-w4-vlm-mm.pt")
    parser.add_argument("--draft-model", default="/data/scratch/zekaili/Alpamayo-DFlash")

    # Clip selection
    parser.add_argument("--clip-id", default=None, help="Run a single clip by ID")
    parser.add_argument("--clip-index", type=int, default=None,
                        help="Run a single clip by index")
    parser.add_argument("--clip-file", default="~/data/physicalai_av/clip_ids.json")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Limit number of frames per clip")

    # Inference params (match eval defaults)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--cache-steps", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--quantize-vlm-only", action="store_true")

    # I/O
    parser.add_argument("--dumped-data-dir", default="/data/scratch/zekaili/dumped_eval_data",
                        help="Root dir of pre-dumped sliding_window_inputs.pt files")
    parser.add_argument("--data-dir", default=None,
                        help="Pre-cached per-timestep .pt files for baseline (optional)")
    parser.add_argument("--output-dir", default="~/exp/demo")
    parser.add_argument("--cache-dir", default="~/data/physicalai_av/hf_cache")
    parser.add_argument("--only", default=None, choices=["baseline", "optimized"],
                        help="Run only one mode")
    args = parser.parse_args()

    for attr in ("baseline_model_path", "model_path", "paro_checkpoint",
                 "draft_model", "clip_file", "output_dir", "cache_dir",
                 "dumped_data_dir"):
        setattr(args, attr, os.path.expanduser(getattr(args, attr)))
    if args.data_dir:
        args.data_dir = os.path.expanduser(args.data_dir)

    run_baseline_mode = args.only in (None, "baseline")
    run_optimized_mode = args.only in (None, "optimized")

    # ── Clip list ──
    with open(os.path.expanduser(args.clip_file)) as f:
        all_clip_ids = json.load(f)

    if args.clip_id is not None:
        clips_to_run = [args.clip_id]
    elif args.clip_index is not None:
        clips_to_run = [all_clip_ids[args.clip_index]]
    else:
        clips_to_run = all_clip_ids

    # ── Dataset (for baseline) ──
    avdi = None
    if run_baseline_mode and not args.data_dir:
        import physical_ai_av
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface(
            cache_dir=args.cache_dir,
        )

    # ── Output ──
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    output_root = Path(args.output_dir) / f"run_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Demo: Alpamayo v1  —  Baseline vs Full-Stack Optimized")
    log.info("=" * 60)
    log.info(f"Clips:    {len(clips_to_run)}")
    log.info(f"Modes:    {['baseline'] * run_baseline_mode + ['optimized'] * run_optimized_mode}")
    log.info(f"Output:   {output_root}")

    # Compute timestep range for baseline
    t0s = list(range(T0_START_US, T0_END_US + 1, STEP_US))
    if args.num_frames is not None and args.num_frames < len(t0s):
        t0s = t0s[:args.num_frames]

    all_clip_summaries = {}

    # ── Baseline ─────────────────────────────────────────────────
    if run_baseline_mode:
        for clip_id in clips_to_run:
            log.info(f"\n{'=' * 60}")
            log.info(f"[Baseline] Clip: {clip_id}")
            log.info(f"{'=' * 60}")

            clip_dir = output_root / clip_id
            baseline_dir = clip_dir / "baseline"
            baseline_dir.mkdir(parents=True, exist_ok=True)

            # Determine GT cutoff if we have avdi
            gt_cutoff_us = None
            if avdi is not None:
                ego = avdi.get_clip_feature(
                    clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True,
                )
                ego_end_us = int(ego.timestamps[-1])
                gt_cutoff_us = ego_end_us - 6_400_000

            stats = run_baseline(
                args, clip_id, t0s, avdi, baseline_dir, args.warmup_steps,
                data_dir=args.data_dir, gt_cutoff_us=gt_cutoff_us,
            )
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

            clip_dir = output_root / clip_id
            opt_dir = clip_dir / "optimized"
            opt_dir.mkdir(parents=True, exist_ok=True)

            stats = run_optimized(
                args, clip_id, opt_dir, args.warmup_steps, model,
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
