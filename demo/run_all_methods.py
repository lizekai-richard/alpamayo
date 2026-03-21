#!/usr/bin/env python3
"""Run 7 inference methods on a single clip for demo video generation.

Methods (each builds on the previous):
    1. Baseline                 - vanilla vlm.generate(), no optimizations
    2. + System optimizations   - torch.compile + StaticCache + fused projections
    3. + Streaming              - KV cache reuse across frames
    4. + DFlash                 - speculative decoding (block-parallel)
    5. + 4-step diffusion       - reduced diffusion steps (10 -> 4)
    6. + AWQ INT4               - 4-bit weight quantization
    7. + ParoQuant INT4         - 4-bit quantization with pairwise rotation

NOTE: Methods 6 (AWQ) and 7 (ParoQuant) cannot run in the same process —
      they register conflicting ``awq::`` CUDA ops.  Use ``--only 7`` or
      ``--only 6`` to select one.

Each method produces per-frame JSON data for video generation.

Output structure:
    ~/exp/demo/run_MMDD_HHMMSS/
    +-- config.json
    +-- 1_baseline/frame_XXXX.json
    +-- 2_sys_opt/frame_XXXX.json
    +-- 3_streaming/frame_XXXX.json
    +-- 4_dflash/frame_XXXX.json
    +-- 5_dflash_4step/frame_XXXX.json
    +-- 6_awq/frame_XXXX.json
    +-- 7_paroquant/frame_XXXX.json
    +-- summary.json

Per-frame JSON contains:
    frame, t0_us, is_prefill, is_warmup,
    total_ms, encode_ms, prefill_ms, decode_ms, diffusion_ms,
    tokens, tokens_per_sec,
    coc (Chain-of-Condition text),
    pred_xyz,
    (DFlash only) acceptance_rate, mean_acceptance_length, total_iterations

Usage:
    # List available clips
    python demo/run_all_methods.py --list-clips

    # Run all 6 methods on default clip
    python demo/run_all_methods.py

    # Run specific clip, limit frames
    python demo/run_all_methods.py --clip-index 2 --num-frames 50

    # Run only specific methods
    python demo/run_all_methods.py --only 1,4,6

    # Run ParoQuant (cannot coexist with AWQ in same process)
    python demo/run_all_methods.py --only 7 --paro-checkpoint ~/models/quant_cache/alpamayo-paro-w4.pt
"""

import argparse
import gc
import importlib.util
import json
import logging
import os
import sys
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from tqdm import tqdm

from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper
from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import load_paroquant_model
from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import convert_wqlinear_layer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")

STEP_US = 100_000  # 100ms = 10Hz

METHODS = OrderedDict([
    (1, {"name": "1_baseline",     "label": "Baseline"}),
    (2, {"name": "2_sys_opt",      "label": "Baseline + Sys Opt"}),
    (3, {"name": "3_dflash",       "label": "Baseline + Sys Opt + DFlash"}),
    (4, {"name": "4_dflash_stream","label": "Baseline + Sys Opt + DFlash + Streaming"}),
    (5, {"name": "5_dflash_4step", "label": "Baseline + Sys Opt + DFlash + Streaming + 4-step"}),
    (6, {"name": "6_awq",          "label": "Baseline + Sys Opt + DFlash + Streaming + 4-step + AWQ"}),
    (7, {"name": "7_paroquant",    "label": "Baseline + Sys Opt + DFlash + Streaming + 4-step + ParoQuant"}),
])


def convert_model_to_marlin_w4a8(model):
    from alpamayo_r1.utils.quantization.rotation_linear import RotateLinearInt4
    from alpamayo_r1.utils.quantization.qmodule import WQLinear
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
    log.info(f"Converted {n} WQLinear -> MarlinW4A8Linear (Marlin W4A8)")
    return n


# ─── Post-load expert fusion ──────────────────────────────────

def fuse_expert_projections(model, mode: str = "streaming"):
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
            if hasattr(attn, "q_norm"): new_attn.q_norm = attn.q_norm
            if hasattr(attn, "k_norm"): new_attn.k_norm = attn.k_norm
            with torch.no_grad():
                q_s, k_s = attn.q_proj.weight.shape[0], attn.k_proj.weight.shape[0]
                new_attn.qkv_proj.weight[:q_s].copy_(attn.q_proj.weight)
                new_attn.qkv_proj.weight[q_s:q_s+k_s].copy_(attn.k_proj.weight)
                new_attn.qkv_proj.weight[q_s+k_s:].copy_(attn.v_proj.weight)
                if new_attn.qkv_proj.bias is not None:
                    new_attn.qkv_proj.bias[:q_s].copy_(attn.q_proj.bias)
                    new_attn.qkv_proj.bias[q_s:q_s+k_s].copy_(attn.k_proj.bias)
                    new_attn.qkv_proj.bias[q_s+k_s:].copy_(attn.v_proj.bias)
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

# ─── Utilities ─────────────────────────────────────────────────────────────


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


def load_clip_ids(clip_file):
    clip_file = os.path.expanduser(clip_file)
    with open(clip_file) as f:
        return json.load(f)


def save_clip_data(clip_id, all_t0s, avdi, output_dir, num_future_steps=64):
    """Pre-download and cache clip data so subsequent runs can use --data-dir.

    Saves one .pt file per timestep (with both num_frames=1 and num_frames=4
    image variants) plus a manifest.json.  The format is compatible with
    load_cached_data().
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving {len(all_t0s)} timesteps to {output_dir} ...")

    for t0_us in tqdm(all_t0s, desc="Caching data"):
        data_nf4 = load_physical_aiavdataset(
            clip_id, t0_us=t0_us, num_frames=4,
            num_future_steps=num_future_steps, avdi=avdi,
        )
        data_nf1 = load_physical_aiavdataset(
            clip_id, t0_us=t0_us, num_frames=1,
            num_future_steps=num_future_steps, avdi=avdi,
        )
        cached = {
            "image_frames_nf4": data_nf4["image_frames"],
            "image_frames_nf1": data_nf1["image_frames"],
            "ego_history_xyz": data_nf4["ego_history_xyz"],
            "ego_history_rot": data_nf4["ego_history_rot"],
            "ego_future_xyz": data_nf4["ego_future_xyz"],
            "ego_future_rot": data_nf4["ego_future_rot"],
        }
        torch.save(cached, output_dir / f"{t0_us}.pt")

    manifest = {
        "clip_id": clip_id,
        "t0_us_list": all_t0s,
        "num_future_steps": num_future_steps,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Saved {len(all_t0s)} timesteps + manifest.json to {output_dir}")


def load_cached_data(data_dir, t0_us, num_frames):
    """Load pre-cached data from a .pt file (see demo/prepare_data.py).

    Returns a dict matching the format of load_physical_aiavdataset().
    """
    pt_path = Path(data_dir) / f"{t0_us}.pt"
    cached = torch.load(pt_path, map_location="cpu", weights_only=True)

    if num_frames == 1:
        image_frames = cached["image_frames_nf1"]
    else:
        image_frames = cached["image_frames_nf4"]

    return {
        "image_frames": image_frames,
        "ego_history_xyz": cached["ego_history_xyz"],
        "ego_history_rot": cached["ego_history_rot"],
        "ego_future_xyz": cached["ego_future_xyz"],
        "ego_future_rot": cached["ego_future_rot"],
    }


def prepare_inputs(data, processor, is_prefill=None):
    """Prepare tokenized inputs from dataset sample."""
    frames = data["image_frames"].flatten(0, 1)
    messages = helper.create_message(frames)
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
    """Extract Chain-of-Condition text from model output."""
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
    """Build standardized per-frame dict from model outputs."""
    ntok = timing.get("num_decode_tokens", timing.get("num_decode_steps", 0))
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
        "diffusion_ms": timing.get("action_time_ms", timing.get("diffusion_time_ms", 0)),
        "tokens": ntok,
        "tokens_per_sec": ntok / (dec_ms / 1000) if dec_ms > 0 else 0,
        "coc": extract_coc_text(extra),
        "pred_xyz": pred_xyz.cpu().numpy()[0, 0].tolist(),
    }

    if min_ade_k is not None:
        frame["min_ade_k"] = min_ade_k
    if min_ade_1 is not None:
        frame["min_ade_1"] = min_ade_1

    # DFlash stats
    ds = extra.get("dflash_stats") if extra else None
    if ds:
        frame["acceptance_rate"] = ds.get("acceptance_rate", 0)
        frame["mean_acceptance_length"] = ds.get("mean_acceptance_length", 0)
        frame["total_iterations"] = ds.get("total_iterations", 0)
        frame["acceptance_lengths"] = ds.get("acceptance_lengths", [])

    return frame


def summarize_stats(stats):
    """Aggregate per-frame stats into summary dict."""
    # Filter out prefill and warmup frames
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

    # minADE
    if any("min_ade_k" in s for s in valid):
        summary["avg_min_ade_k"] = avg("min_ade_k")
        summary["avg_min_ade_1"] = avg("min_ade_1")

    # DFlash stats
    if any("acceptance_rate" in s for s in valid):
        summary["avg_acceptance_rate"] = avg("acceptance_rate")
        summary["avg_mean_acceptance_length"] = avg("mean_acceptance_length")
        summary["avg_iterations"] = avg("total_iterations")

    return summary


def reset_streaming_state(model, next_mode="streaming"):
    """Reset all streaming and compiled state between methods/clips.

    Args:
        next_mode: "streaming" or "non-streaming". Updates the attention mode
                   on patched Qwen3VLTextAttention modules so they use the
                   correct RoPE application pattern.
    """
    with torch.inference_mode():
        model.reset_streaming_state()
        # Destroy the old StaticCache so the next method creates a fresh one
        # with the correct config (text_config for streaming, vlm.config for non-streaming).
        model._past_key_values = None

    # Clear visual model caches (position embeddings, cu_seqlens)
    for attr in ("_cached_pos_embeds", "_cached_position_embeddings", "_cached_cu_seqlens"):
        if hasattr(model.vlm.model.visual, attr):
            delattr(model.vlm.model.visual, attr)
    if hasattr(model.vlm.model.language_model, "_cached_deepstack_indices"):
        delattr(model.vlm.model.language_model, "_cached_deepstack_indices")

    # Update attention mode on all patched text attention modules.
    # patch_for_torch_compile skips already-patched modules, so we must
    # update the mode attribute directly.
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
        delattr(model, attr)
    torch._dynamo.reset()


# ─── Model loading ─────────────────────────────────────────────────────────


def load_base_model(model_path):
    """Load baseline AlpamayoR1 (main branch, uses vlm.generate)."""
    eval_dir = Path(__file__).parent.parent / "eval"
    spec = importlib.util.spec_from_file_location(
        "alpamayo_r1_base", str(eval_dir / "alpamayo_r1_base.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    AlpamayoR1Base = mod.AlpamayoR1

    log.info(f"Loading baseline model from {model_path}...")
    model = AlpamayoR1Base.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda")
    model.eval()

    return model


def load_dev_model(model_path):
    """Load dev-branch AlpamayoR1 (torch.compile + static cache)."""
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    log.info(f"Loading dev model from {model_path}...")
    model = AlpamayoR1.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda")
    model.eval()
    return model


def load_awq_model(model_path, awq_checkpoint):
    """Load AWQ-quantized AlpamayoR1 for streaming + DFlash."""
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.utils.system.patches import patch_for_torch_compile
    from awq.quantize.quantizer import real_quantize_model_weight

    log.info(f"Loading AWQ model from {model_path}...")
    log.info(f"  AWQ checkpoint: {awq_checkpoint}")
    model = AlpamayoR1.from_pretrained(model_path, dtype=torch.bfloat16)

    patch_for_torch_compile(model, mode="streaming", fuse_qkv=True, fuse_gate_up=True)

    q_config = {"zero_point": True, "q_group_size": 128}
    real_quantize_model_weight(model, 4, q_config, init_only=True, target="vlm")

    expert_q_config = {"zero_point": True, "q_group_size": 64}
    real_quantize_model_weight(model, 4, expert_q_config, init_only=True, target="expert")

    awq_sd = torch.load(awq_checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(awq_sd, strict=False)
    if missing:
        log.warning(f"AWQ missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        log.warning(f"AWQ unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    del awq_sd

    model._patched_for_compile = True
    model = model.to("cuda")
    model.eval()
    return model


def load_paro_model(model_path, paro_checkpoint):
    """Load ParoQuant-quantized AlpamayoR1 for streaming + DFlash."""
    from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import load_paroquant_model

    log.info(f"Loading ParoQuant model from {model_path}...")
    log.info(f"  ParoQuant checkpoint: {paro_checkpoint}")
    return load_paroquant_model(
        model_path=model_path,
        paro_checkpoint=paro_checkpoint,
        mode="streaming",
        quantize_expert=True,
    )


def setup_dflash(model, draft_model_path):
    """Set up DFlash speculative decoding on a model."""
    from alpamayo_r1.utils.dflash.dflash_integration import setup_dflash_for_model
    log.info(f"Setting up DFlash from {draft_model_path}...")
    setup_dflash_for_model(model, draft_model_path)


def unload_model(model):
    """Delete model and free GPU memory."""
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ─── Method runners ────────────────────────────────────────────────────────


def run_nonstreaming(model, processor, all_t0s, clip_id, avdi, output_dir,
                     *, method_name, use_sample_traj_api,
                     num_samples, max_tokens, diffusion_steps, warmup_steps,
                     data_dir=None, gt_cutoff_us=None, dflash=False):
    """Run non-streaming inference (Methods 1, 2, and 3).

    Args:
        use_sample_traj_api: If True, use model.sample_trajectories (dev, Methods 2-3).
                             If False, use model.sample_trajectories_from_data_with_vlm_rollout (baseline, Method 1).
        dflash: If True, use DFlash speculative decoding (Method 3).
    """
    log.info(f"\n{'=' * 60}")
    log.info(f"Running: {method_name}")
    log.info(f"  streaming=False, dflash={dflash}, diffusion_steps={diffusion_steps}")
    log.info(f"  frames={len(all_t0s)}, K={num_samples}")
    log.info(f"{'=' * 60}")

    # Pre-load all data before inference (avoid I/O stalling GPU pipeline)
    log.info(f"  Pre-loading data for {len(all_t0s)} timesteps...")
    all_inputs = []
    all_gt_data = []  # (has_gt, ego_future_xyz or None)
    for t0_us in all_t0s:
        has_gt = gt_cutoff_us is None or t0_us <= gt_cutoff_us
        nfs = 64 if has_gt else 1
        if data_dir:
            data = load_cached_data(data_dir, t0_us, num_frames=4)
        else:
            data = load_physical_aiavdataset(clip_id, t0_us=t0_us, num_frames=4, num_future_steps=nfs, avdi=avdi)
        all_inputs.append(prepare_inputs(data, processor))
        all_gt_data.append((has_gt, data["ego_future_xyz"] if has_gt else None))
    log.info(f"  Pre-loaded {len(all_inputs)} inputs, starting inference...")

    stats = []
    pbar = tqdm(enumerate(all_t0s), total=len(all_t0s), desc=method_name[:25])

    for frame_idx, t0_us in pbar:
        try:
            inputs = all_inputs[frame_idx]
            has_gt, ego_future_xyz = all_gt_data[frame_idx]

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                if use_sample_traj_api:
                    result = model.sample_trajectories(
                        data=helper.to_device(inputs, "cuda"),
                        streaming=False,
                        dflash=dflash,
                        num_traj_samples=num_samples,
                        max_generation_length=max_tokens,
                        return_extra=True,
                        fuse_qkv=True,
                        fuse_gate_up=True,
                        diffusion_kwargs={"inference_step": diffusion_steps},
                    )
                else:
                    result = model.sample_trajectories_from_data_with_vlm_rollout(
                        data=helper.to_device(inputs, "cuda"),
                        num_traj_samples=num_samples,
                        max_generation_length=max_tokens,
                        return_extra=True,
                        diffusion_kwargs={"inference_step": diffusion_steps},
                    )

            pred_xyz, pred_rot, extra = result
            if pred_xyz is None:
                continue

            timing = extra.get("timing", {}) if extra else {}
            is_warmup = frame_idx < warmup_steps
            if has_gt:
                min_ade_k, min_ade_1 = calc_min_ade(ego_future_xyz, pred_xyz)
            else:
                min_ade_k, min_ade_1 = None, None
            frame_data = make_frame_data(
                frame_idx, t0_us, timing, extra, pred_xyz,
                is_warmup=is_warmup,
                min_ade_k=min_ade_k, min_ade_1=min_ade_1,
            )
            save_frame(output_dir, frame_idx, frame_data)
            stats.append(frame_data)

            pbar.set_postfix({
                "tot": f"{frame_data['total_ms']:.0f}",
                "enc": f"{frame_data['encode_ms']:.0f}",
                "pf": f"{frame_data['prefill_ms']:.0f}",
                "dec": f"{frame_data['decode_ms']:.0f}",
                "diff": f"{frame_data['diffusion_ms']:.0f}",
                "tok": frame_data['tokens'],
            })

        except Exception as e:
            log.warning(f"  Frame {frame_idx} error: {e}")
            log.warning(traceback.format_exc())
            if "CUDA" in str(e) or "device-side assert" in str(e):
                raise

    return stats


def run_streaming(model, processor, all_t0s, clip_id, avdi, output_dir,
                  *, method_name, dflash,
                  num_samples, max_tokens, diffusion_steps, warmup_steps,
                  data_dir=None, gt_cutoff_us=None,
                  fuse_qkv=True, fuse_gate_up=True):
    """Run streaming inference (Methods 3-7).

    Uses model.sample_trajectories(streaming=True, dflash=...) with default
    temperature=0.6 and top_p=0.98, matching the eval scripts.
    """
    log.info(f"\n{'=' * 60}")
    log.info(f"Running: {method_name}")
    log.info(f"  streaming=True, dflash={dflash}, diffusion_steps={diffusion_steps}")
    log.info(f"  frames={len(all_t0s)}, K={num_samples}")
    log.info(f"{'=' * 60}")

    stats = []
    pbar = tqdm(enumerate(all_t0s), total=len(all_t0s), desc=method_name[:25])

    for frame_idx, t0_us in pbar:
        try:
            # Prefill uses 4 cameras x 4 frames; streaming uses 4 cameras x 1 frame
            has_gt = gt_cutoff_us is None or t0_us <= gt_cutoff_us
            nfs = 64 if has_gt else 1
            if frame_idx == 0:
                if data_dir:
                    data = load_cached_data(data_dir, t0_us, num_frames=4)
                else:
                    data = load_physical_aiavdataset(clip_id, t0_us=t0_us, num_frames=4, num_future_steps=nfs, avdi=avdi)
                inputs = prepare_inputs(data, processor, is_prefill=True)
            else:
                if data_dir:
                    data = load_cached_data(data_dir, t0_us, num_frames=1)
                else:
                    data = load_physical_aiavdataset(clip_id, t0_us=t0_us, num_frames=1, num_future_steps=nfs, avdi=avdi)
                inputs = prepare_inputs(data, processor, is_prefill=False)

            torch.compiler.cudagraph_mark_step_begin()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                result = model.sample_trajectories(
                    data=helper.to_device(inputs, "cuda"),
                    streaming=True,
                    dflash=dflash,
                    torch_compile="max-autotune",
                    num_traj_samples=num_samples,
                    max_generation_length=max_tokens,
                    return_extra=True,
                    fuse_qkv=fuse_qkv,
                    fuse_gate_up=fuse_gate_up,
                    diffusion_kwargs={"inference_step": diffusion_steps},
                )

            # Prefill returns None
            if result is None or result[0] is None:
                prefill_frame = {
                    "frame": frame_idx,
                    "t0_us": t0_us,
                    "is_prefill": True,
                    "is_warmup": True,
                }
                save_frame(output_dir, frame_idx, prefill_frame)
                stats.append(prefill_frame)
                log.info(f"  Frame {frame_idx}: prefill (no output)")
                continue

            pred_xyz, pred_rot, extra = result
            timing = extra.get("timing", {}) if extra else {}

            # Warmup: prefill (frame 0) + first N streaming steps
            is_warmup = frame_idx <= warmup_steps

            if has_gt:
                min_ade_k, min_ade_1 = calc_min_ade(data["ego_future_xyz"], pred_xyz)
            else:
                min_ade_k, min_ade_1 = None, None

            frame_data = make_frame_data(
                frame_idx, t0_us, timing, extra, pred_xyz,
                is_warmup=is_warmup,
                min_ade_k=min_ade_k, min_ade_1=min_ade_1,
            )
            save_frame(output_dir, frame_idx, frame_data)
            stats.append(frame_data)

            pbar.set_postfix({
                "ms": f"{frame_data['total_ms']:.0f}",
            })

        except Exception as e:
            log.warning(f"  Frame {frame_idx} error: {e}")
            log.warning(traceback.format_exc())
            if "CUDA" in str(e) or "device-side assert" in str(e):
                raise

    return stats


# ─── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run 6 inference methods on a single clip for demo video generation",
    )
    parser.add_argument("--model-path", default="/data/scratch/zekaili/Alpamayo-R1-10B")
    parser.add_argument("--draft-model", default="/data/scratch/zekaili/Alpamayo-DFlash")
    parser.add_argument("--awq-checkpoint",
                        default="~/models/quant_cache/alpamayo-all-w4-g128-v2.pt")
    parser.add_argument("--paro-checkpoint",
                        default="~/models/quant_cache/alpamayo-paro-w4.pt")
    parser.add_argument("--clip-id", default="9a249c59-2b25-43a4-9658-c3e20829f80e", help="Clip ID")
    parser.add_argument("--clip-index", type=int, default=0,
                        help="Index into clip_ids.json (default: 0)")
    parser.add_argument("--clip-file", default="~/data/physicalai_av/clip_ids.json")
    parser.add_argument("--start-us", type=int, default=None)
    parser.add_argument("--end-us", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Max frames to process (default: all)")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of trajectory samples K (default: 1)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output-dir", default="~/exp/demo")
    parser.add_argument("--diffusion-steps", type=int, default=10,
                        help="Diffusion steps for methods 1-4 (default: 10)")
    parser.add_argument("--diffusion-steps-fast", type=int, default=4,
                        help="Diffusion steps for methods 5-6 (default: 4)")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--only", default="",
                        help="Comma-separated method numbers to run (e.g. '4,5,6')")
    parser.add_argument("--list-clips", action="store_true")
    parser.add_argument("--camera-range", action="store_true",
                        help="Restrict to camera data time range")
    parser.add_argument("--cache-dir", default="~/data/physicalai_av/hf_cache")
    parser.add_argument("--data-dir", default=None,
                        help="Pre-cached data directory (from --save-data). "
                             "If set, loads from .pt files instead of API calls.")
    parser.add_argument("--save-data", default=None,
                        help="Save clip data to this directory and exit (skip inference). "
                             "Use --data-dir to load it in subsequent runs.")
    args = parser.parse_args()

    # Expand paths
    for attr in ("model_path", "draft_model", "awq_checkpoint", "paro_checkpoint",
                 "clip_file", "output_dir", "cache_dir"):
        setattr(args, attr, os.path.expanduser(getattr(args, attr)))
    if args.data_dir:
        args.data_dir = os.path.expanduser(args.data_dir)
    if args.save_data:
        args.save_data = os.path.expanduser(args.save_data)

    # List clips mode
    if args.list_clips:
        clip_ids = load_clip_ids(args.clip_file)
        print(f"Available clips in {args.clip_file}:")
        for i, cid in enumerate(clip_ids):
            print(f"  [{i}] {cid}")
        return

    # Determine which methods to run
    if args.only:
        methods_to_run = set(int(x) for x in args.only.split(","))
    else:
        methods_to_run = set(range(1, 8))

    # When using cached data, read manifest for clip_id and timesteps
    avdi = None
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
        if args.num_frames is not None and args.num_frames < len(all_t0s):
            all_t0s = all_t0s[:args.num_frames]
        start_us = all_t0s[0]
        end_us = all_t0s[-1]
        gt_cutoff_us = None  # cached data: GT availability unknown, skip check
        log.info(f"Using cached data from {args.data_dir} ({len(all_t0s)} timesteps)")
    else:
        # Get clip ID
        if args.clip_id is None:
            clip_ids = load_clip_ids(args.clip_file)
            if args.clip_index >= len(clip_ids):
                raise ValueError(f"Clip index {args.clip_index} out of range (0-{len(clip_ids)-1})")
            args.clip_id = clip_ids[args.clip_index]

        # Dataset interface
        import physical_ai_av
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=args.cache_dir)

        # Detect clip time range
        log.info("Detecting clip time range...")
        egomotion = avdi.get_clip_feature(
            args.clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True,
        )
        ego_end_us = int(egomotion.timestamps[-1])
        # GT future trajectory needs 6.4s (64 steps at 10Hz) ahead of t0.
        # Only compute minADE for timesteps where full GT is available.
        gt_cutoff_us = ego_end_us - 6_400_000

        # Always fetch camera range — every timestep needs camera frames.
        camera = avdi.get_clip_feature(
            args.clip_id,
            avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            maybe_stream=True,
        )
        cam_start_us = int(camera.timestamps[0])
        cam_end_us = int(camera.timestamps[-1])
        clip_end_us = min(ego_end_us, cam_end_us)
        if args.camera_range:
            clip_start_us = max(1_700_000, cam_start_us)
        else:
            clip_start_us = 1_700_000  # 1.7s minimum for 16 history steps

        start_us = args.start_us or clip_start_us
        end_us = args.end_us or (clip_end_us - STEP_US)

        all_t0s = list(range(start_us, end_us + 1, STEP_US))
        if args.num_frames is not None and args.num_frames < len(all_t0s):
            all_t0s = all_t0s[:args.num_frames]

    # Save data and exit (no model loading / inference)
    if args.save_data:
        if avdi is None:
            import physical_ai_av
            avdi = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=args.cache_dir)
        save_clip_data(args.clip_id, all_t0s, avdi, args.save_data)
        return

    # Output folder
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    output_root = Path(args.output_dir) / f"run_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    for mid in methods_to_run:
        (output_root / METHODS[mid]["name"]).mkdir(exist_ok=True)

    # Print config
    log.info("=" * 60)
    log.info("Demo Inference: 6 Methods Comparison")
    log.info("=" * 60)
    log.info(f"Clip:          {args.clip_id}")
    log.info(f"Time range:    {start_us/1e6:.2f}s - {end_us/1e6:.2f}s")
    if gt_cutoff_us is not None:
        log.info(f"GT cutoff:     {gt_cutoff_us/1e6:.2f}s (minADE only for t0 <= this)")
    log.info(f"Frames:        {len(all_t0s)}")
    log.info(f"K (samples):   {args.num_samples}")
    log.info(f"Methods:       {sorted(methods_to_run)}")
    log.info(f"Output:        {output_root}")

    # Save config
    config = {
        "clip_id": args.clip_id,
        "start_us": start_us,
        "end_us": end_us,
        "num_frames": len(all_t0s),
        "num_samples": args.num_samples,
        "max_tokens": args.max_tokens,
        "diffusion_steps": args.diffusion_steps,
        "diffusion_steps_fast": args.diffusion_steps_fast,
        "warmup_steps": args.warmup_steps,
        "methods": sorted(methods_to_run),
    }
    with open(output_root / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_summaries = {}

    def save_summary():
        """Save summary.json with all results so far."""
        summary = {"config": config, "methods": {}}
        for mid in sorted(all_summaries):
            summary["methods"][METHODS[mid]["name"]] = {
                "label": METHODS[mid]["label"],
                **all_summaries[mid],
            }
        if 1 in all_summaries and all_summaries[1].get("avg_total_ms", 0) > 0:
            baseline_total = all_summaries[1]["avg_total_ms"]
            baseline_decode = all_summaries[1]["avg_decode_ms"]
            speedups = {}
            for mid in sorted(all_summaries):
                if mid == 1:
                    continue
                s = all_summaries[mid]
                if s.get("avg_total_ms", 0) > 0:
                    speedups[METHODS[mid]["name"]] = {
                        "total_speedup": baseline_total / s["avg_total_ms"],
                        "decode_speedup": baseline_decode / s["avg_decode_ms"] if s.get("avg_decode_ms", 0) > 0 else 0,
                    }
            summary["speedups_vs_baseline"] = speedups
        with open(output_root / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"  Saved summary.json ({len(all_summaries)} methods so far)")

    # ─── Phase 1: Baseline (method 1) ──────────────────────────────────
    if 1 in methods_to_run:
        model = load_base_model(args.model_path)
        processor = helper.get_processor(model.tokenizer)
        stats = run_nonstreaming(
            model, processor, all_t0s, args.clip_id, avdi,
            output_root / METHODS[1]["name"],
            method_name=METHODS[1]["label"],
            use_sample_traj_api=False,
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
            diffusion_steps=args.diffusion_steps,
            warmup_steps=args.warmup_steps,
            data_dir=args.data_dir,
            gt_cutoff_us=gt_cutoff_us,
        )
        all_summaries[1] = summarize_stats(stats)
        save_summary()
        unload_model(model)

    # ─── Phase 2a: Non-streaming (method 2) ─────────────────────────────
    # Following eval/eval_system_opt.py: fresh model, streaming=False only.
    if 2 in methods_to_run:
        model = load_dev_model(args.model_path)
        processor = helper.get_processor(model.tokenizer)

        stats = run_nonstreaming(
            model, processor, all_t0s, args.clip_id, avdi,
            output_root / METHODS[2]["name"],
            method_name=METHODS[2]["label"],
            use_sample_traj_api=True,
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
            diffusion_steps=args.diffusion_steps,
            warmup_steps=args.warmup_steps,
            data_dir=args.data_dir,
            gt_cutoff_us=gt_cutoff_us,
        )
        all_summaries[2] = summarize_stats(stats)
        save_summary()
        unload_model(model)

    # ─── Phase 2b: DFlash methods (methods 3-5) ────────────────────────
    # Methods 3-5 all need DFlash, so they share a model with DFlash setup.
    # Method 3: DFlash non-streaming, Methods 4-5: DFlash + streaming.
    dflash_methods = sorted(methods_to_run & {3, 4, 5})
    if dflash_methods:
        model = load_dev_model(args.model_path)
        processor = helper.get_processor(model.tokenizer)
        setup_dflash(model, args.draft_model)

        # Method 3: DFlash non-streaming (10-step diffusion)
        if 3 in dflash_methods:
            stats = run_nonstreaming(
                model, processor, all_t0s, args.clip_id, avdi,
                output_root / METHODS[3]["name"],
                method_name=METHODS[3]["label"],
                use_sample_traj_api=True,
                dflash=True,
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
                diffusion_steps=args.diffusion_steps,
                warmup_steps=args.warmup_steps,
                data_dir=args.data_dir,
                gt_cutoff_us=gt_cutoff_us,
            )
            all_summaries[3] = summarize_stats(stats)
            save_summary()

        # Method 4: DFlash + Streaming (10-step diffusion)
        if 4 in dflash_methods:
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
                data_dir=args.data_dir,
                gt_cutoff_us=gt_cutoff_us,
            )
            all_summaries[4] = summarize_stats(stats)
            save_summary()

        # Method 5: DFlash + Streaming (4-step diffusion)
        if 5 in dflash_methods:
            reset_streaming_state(model)
            stats = run_streaming(
                model, processor, all_t0s, args.clip_id, avdi,
                output_root / METHODS[5]["name"],
                method_name=METHODS[5]["label"],
                dflash=True,
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
                diffusion_steps=args.diffusion_steps_fast,
                warmup_steps=args.warmup_steps,
                data_dir=args.data_dir,
                gt_cutoff_us=gt_cutoff_us,
            )
            all_summaries[5] = summarize_stats(stats)
            save_summary()

        unload_model(model)

    # ─── Phase 3: AWQ (method 6) ───────────────────────────────────────
    if 6 in methods_to_run:
        model = load_awq_model(args.model_path, args.awq_checkpoint)
        processor = helper.get_processor(model.tokenizer)
        setup_dflash(model, args.draft_model)

        stats = run_streaming(
            model, processor, all_t0s, args.clip_id, avdi,
            output_root / METHODS[6]["name"],
            method_name=METHODS[6]["label"],
            dflash=True,
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
            diffusion_steps=args.diffusion_steps_fast,
            warmup_steps=args.warmup_steps,
            data_dir=args.data_dir,
            gt_cutoff_us=gt_cutoff_us,
        )
        all_summaries[6] = summarize_stats(stats)
        save_summary()
        unload_model(model)

    # ─── Phase 4: ParoQuant (method 7) ────────────────────────────────
    # NOTE: Methods 6 and 7 cannot coexist — they register conflicting CUDA ops.
    if 7 in methods_to_run:
        model = load_paroquant_model(
            model_path=args.model_path,
            paro_checkpoint=args.paro_checkpoint,
            mode="streaming",
            quantize_expert=False,
        )
        processor = helper.get_processor(model.tokenizer)
        n = convert_model_to_marlin_w4a8(model)
        fuse_expert_projections(model)
        setup_dflash(model, args.draft_model)

        stats = run_streaming(
            model, processor, all_t0s, args.clip_id, avdi,
            output_root / METHODS[7]["name"],
            method_name=METHODS[7]["label"],
            dflash=True,
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
            diffusion_steps=args.diffusion_steps_fast,
            warmup_steps=args.warmup_steps,
            data_dir=args.data_dir,
            gt_cutoff_us=gt_cutoff_us,
            fuse_qkv=False,
            fuse_gate_up=False,
        )
        all_summaries[7] = summarize_stats(stats)
        save_summary()
        unload_model(model)

    # ─── Summary ───────────────────────────────────────────────────────
    save_summary()

    # Load final summary for printing
    with open(output_root / "summary.json") as f:
        summary = json.load(f)

    # Print summary table
    log.info(f"\n{'=' * 80}")
    log.info("SUMMARY")
    log.info(f"{'=' * 80}")
    log.info(f"{'Method':<50} {'Total':>8} {'Decode':>8} {'Diff':>6} {'Tok/s':>7} {'ADE_K':>7} {'ADE_1':>7}")
    log.info(f"{'-' * 50} {'-' * 8} {'-' * 8} {'-' * 6} {'-' * 7} {'-' * 7} {'-' * 7}")

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
            accept_str = f"  accept={s['avg_acceptance_rate']:.0%} iter={s.get('avg_iterations', 0):.1f}"
        log.info(
            f"{label:<50} {s['avg_total_ms']:>7.1f}ms {s['avg_decode_ms']:>7.1f}ms "
            f"{s['avg_diffusion_ms']:>5.1f}ms {s['avg_tokens_per_sec']:>6.1f}{ade_str}{accept_str}"
        )

    if "speedups_vs_baseline" in summary:
        log.info(f"\nSpeedup vs Baseline:")
        for name, sp in summary["speedups_vs_baseline"].items():
            log.info(f"  {name:<25} total={sp['total_speedup']:.2f}x  decode={sp['decode_speedup']:.2f}x")

    log.info(f"\nOutput: {output_root}")


if __name__ == "__main__":
    main()
