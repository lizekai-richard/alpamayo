#!/usr/bin/env python3
"""ParoQuant W4A8 (Marlin) evaluation: DFlash + streaming + action caching.

Same as eval_all_paro.py but replaces WQLinear (FP16 GEMM) with
MarlinW4A8Linear (INT4 weight x INT8 activation via vLLM Marlin kernel).

Supports multi-GPU parallelism: clips are split across ranks via env vars.

Usage:
    # Single GPU
    python eval/eval_all_paro_w4a8.py
    python eval/eval_all_paro_w4a8.py --num-clips 10
    python eval/eval_all_paro_w4a8.py --quantize-VLM-only

    # Multi-GPU
    torchrun --nproc_per_node=4 eval/eval_all_paro_w4a8.py --num-clips 100

    # Manual multi-GPU
    RANK=0 WORLD_SIZE=4 LOCAL_RANK=0 python eval/eval_all_paro_w4a8.py --num-clips 100 &
    RANK=1 WORLD_SIZE=4 LOCAL_RANK=1 python eval/eval_all_paro_w4a8.py --num-clips 100 &
    ...
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import alpamayo_r1
sys.modules["alpamayo1_5"] = alpamayo_r1

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "paroquant"))

import numpy as np
import torch

from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import load_paroquant_model_v1p5
from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import convert_wqlinear_layer
from alpamayo_r1 import helper
from alpamayo_r1.utils.dflash.dflash_integration import setup_dflash_for_model
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")

STEP_US = 100_000
T0_START_US = 1_700_000
T0_END_US = 13_600_000


# ─── Distributed ──────────────────────────────────────────────

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, device
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", num_gpus))
        local_rank = rank % num_gpus
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, device
    return 0, 0, 1, "cuda:0"


def split_clips_for_rank(clip_ids, rank, world_size):
    return [clip_ids[i] for i in range(len(clip_ids)) if i % world_size == rank]


# ─── W4A8 conversion ─────────────────────────────────────────

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


# ─── Eval helpers ─────────────────────────────────────────────

def load_clip_ids(path, num_clips):
    with open(path) as f:
        ids = json.load(f)
    return list(dict.fromkeys(ids))[:num_clips]


def validate_clip(clip_id, avdi):
    """Check clip has enough egomotion data for the eval range."""
    ego = avdi.get_clip_feature(
        clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True,
    )
    ego_end = int(ego.timestamps[-1])
    if ego_end <= T0_END_US + 6_400_000:
        return False
    return True


def prepare_inputs(data, processor, is_prefill=False):
    frames = data["image_frames"].flatten(0, 1)
    messages = helper.create_message(frames, camera_indices=data["camera_indices"])
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

def create_or_load_streaming_inputs(args, tokenizer, processor, clip_id, avdi):
    """Create sliding-window streaming inputs for one clip.

    Window 0 (prefill): 4 cameras x 4 frames = 16 frames
    Window 1+ (streaming): 4 cameras x 1 new frame = 4 frames
    """
    if args.dumped_data_dir:
        windows = helper.load_dumped_inputs(args.dumped_data_dir, clip_id)
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
                    keep_frame_labels=args.keep_frame_labels,
                    vision_start_id=vs_id if not args.keep_frame_labels else None,
                )
            streaming_inputs.append(data)
        log.info(f"Streaming input data shape: {streaming_inputs[1]['tokenized_data']['input_ids'].shape}")

    else:
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
    try:
        gt_xy = gt_future_xy.cpu()[0, 0, :, :2].T.numpy()
        pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
        diff = np.linalg.norm(pred_xy - gt_xy[None], axis=1)
        ade = diff.mean(axis=-1)
        return float(ade.min()), float(ade[0])
    except Exception as e:
        log.warning(f"calc_min_ade error: {e}")
        return float("inf"), float("inf")


def reset_clip_state(model, keep_frame_labels: bool = True, kv_shift_mode: str = "block"):
    """Reset all streaming and compiled state between clips."""
    with torch.inference_mode():
        model.reset_streaming_state(keep_frame_labels=keep_frame_labels, kv_shift_mode=kv_shift_mode)
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

# ─── Cross-rank aggregation ───────────────────────────────────

def aggregate_results_across_ranks(
    run_dir, num_traj_samples, diffusion_steps, cache_str,
    world_size, expected_rank_clip_counts, run_started_at_s,
    max_wait_seconds=300,
):
    """Aggregate W4A8 Marlin results from all rank files into a single summary."""
    if world_size <= 1:
        return None

    prefix = f"w4a8_marlin_dflash_stream_acache{cache_str}_K{num_traj_samples}_d{diffusion_steps}_"
    expected_rank_files = {
        r: Path(run_dir) / f"{prefix}{expected_rank_clip_counts[r]}clips_rank{r}.json"
        for r in range(world_size)
    }

    start_time = time.time()
    rank_files = {}
    while len(rank_files) < world_size and (time.time() - start_time) < max_wait_seconds:
        rank_files = {}
        for r, rf in expected_rank_files.items():
            if not rf.exists():
                continue
            if rf.stat().st_mtime + 1 < run_started_at_s:
                continue
            rank_files[r] = rf
        if len(rank_files) < world_size:
            time.sleep(1)

    if len(rank_files) < world_size:
        missing = [str(r) for r in range(world_size) if r not in rank_files]
        log.warning(
            "Only found %d/%d rank files for this run; missing ranks: %s; proceeding with available files",
            len(rank_files), world_size, ",".join(missing),
        )

    if not rank_files:
        log.warning("No rank files found for aggregation")
        return None

    all_samples = []
    all_ade_k = []
    all_ade_1 = []
    all_timing = []
    all_dflash_stats = []
    total_steps = 0

    for _, rf in sorted(rank_files.items()):
        try:
            data = json.loads(rf.read_text())
            samples = data.get("samples", [])
            all_samples.extend(samples)

            for s in samples:
                k_key = f"min_ade_{num_traj_samples}"
                if k_key in s and s[k_key] is not None:
                    all_ade_k.append(s[k_key])
                if "min_ade_1" in s and s["min_ade_1"] is not None:
                    all_ade_1.append(s["min_ade_1"])
                if "total_time_ms" in s:
                    all_timing.append(s)
                ds = s.get("dflash_stats")
                if ds:
                    all_dflash_stats.append(ds)

            total_steps += len(samples)
            log.info(f"Loaded results from {rf.name}: {len(samples)} samples")
        except Exception as e:
            log.warning(f"Error loading {rf}: {e}")
            continue

    if not all_samples:
        log.warning("No samples found in rank files")
        return None

    summary = {
        f"min_ade_{num_traj_samples}": float(np.mean(all_ade_k)) if all_ade_k else None,
        "min_ade_1": float(np.mean(all_ade_1)) if all_ade_1 else None,
        "num_steps": total_steps,
        "num_clips": len(set(s.get("clip_id", "") for s in all_samples)),
    }

    if all_timing:
        summary.update({
            "avg_total_ms": float(np.mean([t.get("total_time_ms", 0) for t in all_timing])),
            "avg_encode_ms": float(np.mean([t.get("encode_time_ms", 0) for t in all_timing])),
            "avg_prefill_ms": float(np.mean([t.get("prefill_time_ms", 0) for t in all_timing])),
            "avg_decode_ms": float(np.mean([t.get("decode_time_ms", 0) for t in all_timing])),
            "avg_action_ms": float(np.mean([t.get("action_time_ms", 0) for t in all_timing])),
            "avg_num_tokens": float(np.mean([t.get("num_decode_tokens", 0) for t in all_timing])),
            "avg_first_token_sample_ms": float(np.mean([t.get("first_token_sample_time_ms", 0) for t in all_timing])),
            "avg_dflash_loop_ms": float(np.mean([t.get("dflash_loop_time_ms", 0) for t in all_timing])),
            "avg_traj_forward_ms": float(np.mean([t.get("traj_forward_time_ms", 0) for t in all_timing])),
        })

    if all_dflash_stats:
        summary.update({
            "avg_acceptance_rate": float(np.mean([d["acceptance_rate"] for d in all_dflash_stats])),
            "avg_acceptance_length": float(np.mean([d["mean_acceptance_length"] for d in all_dflash_stats])),
            "avg_match_rate": float(np.mean([d["match_rate"] for d in all_dflash_stats])),
            "avg_iterations": float(np.mean([d["total_iterations"] for d in all_dflash_stats])),
        })

    agg_file = Path(run_dir) / (
        f"w4a8_marlin_dflash_stream_acache{cache_str}"
        f"_K{num_traj_samples}_d{diffusion_steps}_{total_steps}steps_aggregated.json"
    )
    agg_file.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "aggregated_from_ranks": len(rank_files),
        "expected_world_size": world_size,
        "summary": summary,
        "samples": all_samples,
    }, indent=2))

    log.info(
        f"Aggregated results from {len(rank_files)} ranks: "
        f"{summary['num_clips']} clips, {total_steps} steps"
    )
    log.info(f"  Aggregated minADE_{num_traj_samples}: {summary.get(f'min_ade_{num_traj_samples}', 'N/A')}")
    log.info(f"  Aggregated minADE_1: {summary.get('min_ade_1', 'N/A')}")
    if all_timing:
        log.info(f"  Aggregated avg total time: {summary.get('avg_total_ms', 0):.1f} ms")
    if all_dflash_stats:
        log.info(f"  Aggregated avg acceptance rate: {summary.get('avg_acceptance_rate', 0):.1%}")
    log.info(f"Saved aggregated to {agg_file}")
    return str(agg_file)


# ─── Main ─────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="ParoQuant W4A8 Marlin eval")
    ap.add_argument("--model-path", default="/data/scratch/zekaili/Alpamayo1_5-Finetuned")
    ap.add_argument("--paro-checkpoint", default="/data/scratch/zekaili/quant_cache/alpamayo-1.5-finetuned-paro-w4-vlm-mm.pt")
    ap.add_argument("--draft-model", default="/data/scratch/zekaili/Alpamayo1_5-DFlash")
    ap.add_argument("--clip-ids-file", default="./clips.json")
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--num-traj-samples", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--diffusion-steps", type=int, default=8)
    ap.add_argument("--cache-steps", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--output-dir", default="./eval_all_paro_w4a8_v1p5")
    ap.add_argument("--cache-dir", default="/data/scratch/zekaili/physicalai_av/hf_cache")
    ap.add_argument("--keep-frame-labels", action="store_true", default=False)
    ap.add_argument("--kv-shift-mode", type=str, default="vision_only", choices=["block", "vision_only"])
    ap.add_argument("--dumped-data-dir", default="/data/scratch/zekaili/dumped_eval_data_v1p5")
    args = ap.parse_args()

    for attr in ("model_path", "paro_checkpoint", "draft_model", "clip_ids_file", "output_dir", "cache_dir", "dumped_data_dir"):
        setattr(args, attr, os.path.expanduser(getattr(args, attr)))

    run_started_at_s = time.time()
    rank, local_rank, world_size, device = setup_distributed()
    if rank != 0:
        logging.getLogger().setLevel(logging.CRITICAL)
        log.setLevel(logging.CRITICAL)

    log.info(f"Distributed: rank={rank}, local_rank={local_rank}, "
             f"world_size={world_size}, device={device}")

    cache_str = "-".join(str(s) for s in args.cache_steps)
    run_dir = args.output_dir
    os.makedirs(run_dir, exist_ok=True)

    config = vars(args).copy()
    config.update({"rank": rank, "local_rank": local_rank,
                   "world_size": world_size, "device": device})
    with open(os.path.join(run_dir, f"config_rank{rank}.json"), "w") as f:
        json.dump(config, f, indent=2)

    all_clip_ids = load_clip_ids(args.clip_ids_file, args.num_clips)
    clip_ids = split_clips_for_rank(all_clip_ids, rank, world_size)
    expected_rank_clip_counts = {
        r: len(split_clips_for_rank(all_clip_ids, r, world_size))
        for r in range(world_size)
    }
    log.info(f"Rank {rank}/{world_size-1}: {len(clip_ids)}/{len(all_clip_ids)} clips")

    if not clip_ids:
        log.warning(f"Rank {rank}: no clips assigned")
        return
    
    import physical_ai_av
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=args.cache_dir)

    # ── Load model ──
    model = load_paroquant_model_v1p5(
        model_path=args.model_path,
        paro_checkpoint=args.paro_checkpoint,
        mode="streaming",
    )
    tokenizer = model.tokenizer
    processor = helper.get_processor(model.tokenizer)

    log.info("Converting to Marlin W4A8 backend...")
    n = convert_model_to_marlin_w4a8(model)
    if n == 0:
        raise SystemExit("No layers converted!")

    log.info("Fusing expert projections...")
    fuse_expert_projections(model)

    setup_dflash_for_model(model, args.draft_model)
    log.info("DFlash enabled (ParoQuant W4A8 Marlin, streaming, action_cache)")

    # ── Eval loop ──
    all_timing, all_ade_k, all_ade_1, all_dflash_stats, all_results = [], [], [], [], []

    for ci, clip_id in enumerate(clip_ids):
        if not validate_clip(clip_id, avdi):
            log.warning(f"Clip {clip_id}: too short, skipping")
            continue

        log.info(f"\n[Rank {rank}] Clip {ci+1}/{len(clip_ids)}: {clip_id}")
        reset_clip_state(model, args.keep_frame_labels, args.kv_shift_mode)

        try:
            streaming_inputs = create_or_load_streaming_inputs(args, tokenizer, processor, clip_id, avdi)
        except Exception as e:
            log.warning(f"  Error loading inputs: {e}")
            continue
        log.info(f"  Loaded {len(streaming_inputs)} inputs")

        clip_timing, clip_ade_k, clip_ade_1, clip_dflash_stats = [], [], [], []

        for si, inputs in enumerate(streaming_inputs):
            try:
                torch.compiler.cudagraph_mark_step_begin()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model.sample_trajectories_from_flashdrive(
                        data=helper.to_device(inputs, device),
                        streaming=True,
                        dflash=True,
                        num_traj_samples=args.num_traj_samples,
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
                    log.info(f"  Step {si}: prefill (no output)")
                    continue

                pred_xyz, _, extra = result
                min_ade_k, min_ade_1 = calc_min_ade(inputs["ego_future_xyz"], pred_xyz)
                timing = extra.get("timing") if extra else None
                ds = extra.get("dflash_stats") if extra else None

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
                    dflash_str = ""
                    if ds:
                        loop_ms = timing.get("dflash_loop_time_ms", 0)
                        traj_fwd_ms = timing.get("traj_forward_time_ms", 0)
                        dflash_str = (
                            f", dflash: loop={loop_ms:.1f}ms traj_fwd={traj_fwd_ms:.1f}ms "
                            f"accept={ds['acceptance_rate']:.0%} "
                            f"len={ds['mean_acceptance_length']:.2f} "
                            f"iters={ds['total_iterations']}"
                        )
                    log.info(
                        f"  Step {si}{tag}: {total:.1f}ms "
                        f"(enc={enc:.1f} pf={pf:.1f} dec={dec:.1f} act={act:.1f}) "
                        f"{ntok}tok {tps:.1f}tok/s, "
                        f"minADE_{args.num_traj_samples}={min_ade_k:.3f}m, "
                        f"minADE_1={min_ade_1:.3f}m"
                        f"{dflash_str}"
                    )

                if not is_warmup:
                    clip_ade_k.append(min_ade_k)
                    clip_ade_1.append(min_ade_1)
                    if timing: clip_timing.append(timing)
                    if ds: clip_dflash_stats.append(ds)
                    all_results.append({
                        "clip_id": clip_id,
                        "step": si,
                        f"min_ade_{args.num_traj_samples}": min_ade_k,
                        "min_ade_1": min_ade_1,
                        **(timing or {}),
                        **({"dflash_stats": ds} if ds else {}),
                    })

            except Exception as e:
                log.warning(f"  Step {si} error: {e}", exc_info=True)
                if "CUDA" in str(e):
                    raise SystemExit(1)
                continue

        if clip_ade_k:
            ct = clip_timing
            avg_total = np.mean([t["total_time_ms"] for t in ct]) if ct else 0
            avg_dec = np.mean([t.get("decode_time_ms", 0) for t in ct]) if ct else 0
            avg_tok = np.mean([t.get("num_decode_tokens", 0) for t in ct]) if ct else 0
            avg_tps = avg_tok / (avg_dec / 1000) if avg_dec > 0 else 0
            dflash_str = ""
            if clip_dflash_stats:
                avg_accept = np.mean([d["acceptance_rate"] for d in clip_dflash_stats])
                avg_accept_len = np.mean([d["mean_acceptance_length"] for d in clip_dflash_stats])
                avg_iters = np.mean([d["total_iterations"] for d in clip_dflash_stats])
                dflash_str = f", accept={avg_accept:.0%}, len={avg_accept_len:.2f}, iters={avg_iters:.1f}"
            log.info(
                f"  Clip avg: {avg_total:.1f}ms total, {avg_dec:.1f}ms decode, "
                f"{avg_tok:.0f}tok, {avg_tps:.1f}tok/s, "
                f"minADE_{args.num_traj_samples}={np.mean(clip_ade_k):.3f}m, "
                f"minADE_1={np.mean(clip_ade_1):.3f}m{dflash_str} "
                f"({len(clip_ade_k)} steps)"
            )
            all_ade_k.extend(clip_ade_k)
            all_ade_1.extend(clip_ade_1)
            all_timing.extend(clip_timing)
            all_dflash_stats.extend(clip_dflash_stats)

    # ── Per-rank summary ──
    n = len(all_timing)
    log.info(f"\n{'='*60}")
    log.info(
        f"[Rank {rank}] RESULTS — ParoQuant W4A8 Marlin "
        f"(DFlash + streaming + action_cache) ({n} steps)"
    )
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

        log.info(f"\n  DFlash decode breakdown:")
        log.info(f"    Avg first_token_sample: {avg('first_token_sample_time_ms'):.1f} ms")
        log.info(f"    Avg dflash_loop:    {avg('dflash_loop_time_ms'):.1f} ms")
        log.info(f"    Avg traj_forward:   {avg('traj_forward_time_ms'):.1f} ms")

    if all_dflash_stats:
        def ds_avg(key):
            vals = [d[key] for d in all_dflash_stats if d.get(key) is not None]
            return np.mean(vals) if vals else 0

        log.info(f"  Avg acceptance rate:   {ds_avg('acceptance_rate'):.1%}")
        log.info(f"  Avg acceptance length: {ds_avg('mean_acceptance_length'):.2f}")
        log.info(f"  Avg match rate:        {ds_avg('match_rate'):.1%}")
        log.info(f"  Avg iterations:        {ds_avg('total_iterations'):.1f}")

    if all_ade_k:
        log.info(f"  Avg minADE_{args.num_traj_samples}:       {np.mean(all_ade_k):.3f} m")
        log.info(f"  Avg minADE_1:        {np.mean(all_ade_1):.3f} m")

    # ── Save per-rank ──
    rank_suffix = f"_rank{rank}" if world_size > 1 else ""
    out_file = os.path.join(
        run_dir,
        f"w4a8_marlin_dflash_stream_acache{cache_str}"
        f"_K{args.num_traj_samples}_d{args.diffusion_steps}"
        f"_{len(clip_ids)}clips{rank_suffix}.json",
    )
    with open(out_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "distributed": {"rank": rank, "local_rank": local_rank,
                            "world_size": world_size, "device": device},
            "summary": {
                f"min_ade_{args.num_traj_samples}": float(np.mean(all_ade_k)) if all_ade_k else None,
                "min_ade_1": float(np.mean(all_ade_1)) if all_ade_1 else None,
                "num_steps": len(all_ade_k), "num_clips": len(clip_ids),
                **({
                    "avg_total_ms": float(np.mean([t["total_time_ms"] for t in all_timing])),
                    "avg_encode_ms": float(np.mean([t.get("encode_time_ms", 0) for t in all_timing])),
                    "avg_prefill_ms": float(np.mean([t.get("prefill_time_ms", 0) for t in all_timing])),
                    "avg_decode_ms": float(np.mean([t.get("decode_time_ms", 0) for t in all_timing])),
                    "avg_action_ms": float(np.mean([t.get("action_time_ms", 0) for t in all_timing])),
                    "avg_num_tokens": float(np.mean([t.get("num_decode_tokens", 0) for t in all_timing])),
                    "avg_first_token_sample_ms": float(np.mean([t.get("first_token_sample_time_ms", 0) for t in all_timing])),
                    "avg_dflash_loop_ms": float(np.mean([t.get("dflash_loop_time_ms", 0) for t in all_timing])),
                    "avg_traj_forward_ms": float(np.mean([t.get("traj_forward_time_ms", 0) for t in all_timing])),
                } if all_timing else {}),
                **({
                    "avg_acceptance_rate": float(np.mean([d["acceptance_rate"] for d in all_dflash_stats])),
                    "avg_acceptance_length": float(np.mean([d["mean_acceptance_length"] for d in all_dflash_stats])),
                    "avg_match_rate": float(np.mean([d["match_rate"] for d in all_dflash_stats])),
                    "avg_iterations": float(np.mean([d["total_iterations"] for d in all_dflash_stats])),
                } if all_dflash_stats else {}),
            },
            "samples": all_results,
        }, f, indent=2)
    log.info(f"Saved to {out_file}")

    # ── Aggregate (rank 0 only) ──
    if rank == 0 and world_size > 1:
        log.info(f"\n{'='*60}")
        log.info(f"Aggregating results across {world_size} ranks...")
        log.info("=" * 60)
        aggregated_file = aggregate_results_across_ranks(
            run_dir, args.num_traj_samples, args.diffusion_steps, cache_str,
            world_size, expected_rank_clip_counts, run_started_at_s,
        )
        if aggregated_file:
            log.info(f"Aggregated results saved to {aggregated_file}")


if __name__ == "__main__":
    main()
