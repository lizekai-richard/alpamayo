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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "paroquant"))

import numpy as np
import torch

from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import load_paroquant_model
from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import convert_wqlinear_layer
from alpamayo_r1 import helper
from alpamayo_r1.utils.dflash.dflash_integration import setup_dflash_for_model

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

def load_clip_ids_from_data_dir(data_dir, num_clips):
    clip_ids = sorted(
        d.name for d in Path(data_dir).iterdir()
        if d.is_dir() and (d / "sliding_window_inputs.pt").exists()
    )
    return clip_ids[:num_clips]


def load_streaming_inputs(data_dir, clip_id):
    return torch.load(Path(data_dir) / clip_id / "sliding_window_inputs.pt",
                      map_location="cpu", weights_only=False)


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


def reset_clip_state(model):
    with torch.inference_mode():
        model.reset_streaming_state()
        if model._past_key_values is not None:
            model._past_key_values.reset()
    for attr in ("_cached_pos_embeds", "_cached_position_embeddings", "_cached_cu_seqlens"):
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
        delattr(model, a)
    torch._dynamo.reset()


# ─── Cross-rank aggregation ───────────────────────────────────

def aggregate_results_across_ranks(
    run_dir, num_traj_samples, diffusion_steps, cache_str,
    world_size, expected_rank_clip_counts, run_started_at_s,
    max_wait_seconds=300,
):
    if world_size <= 1:
        return None
    expected_rank_files = {
        r: Path(run_dir) / (
            f"w4a8_marlin_dflash_stream_acache{cache_str}"
            f"_K{num_traj_samples}_d{diffusion_steps}"
            f"_{expected_rank_clip_counts[r]}clips_rank{r}.json"
        )
        for r in range(world_size)
    }
    start = time.time()
    rank_files = {}
    while len(rank_files) < world_size and (time.time() - start) < max_wait_seconds:
        rank_files = {}
        for r, rf in expected_rank_files.items():
            if rf.exists() and rf.stat().st_mtime + 1 >= run_started_at_s:
                rank_files[r] = rf
        if len(rank_files) < world_size:
            time.sleep(1)
    if len(rank_files) < world_size:
        missing = [str(r) for r in range(world_size) if r not in rank_files]
        log.warning("Found %d/%d rank files; missing: %s", len(rank_files), world_size, ",".join(missing))
    if not rank_files:
        return None

    all_clips, all_ade_k, all_ade_1 = [], [], []
    all_timing, all_dflash, total_steps = [], [], 0
    for _, rf in sorted(rank_files.items()):
        try:
            data = json.loads(rf.read_text())
            clips = data.get("clips", [])
            all_clips.extend(clips)
            for c in clips:
                k_key = f"min_ade_{num_traj_samples}"
                if k_key in c and c[k_key] is not None: all_ade_k.append(c[k_key])
                if "min_ade_1" in c and c["min_ade_1"] is not None: all_ade_1.append(c["min_ade_1"])
                if "avg_total_ms" in c: all_timing.append(c)
                if "avg_acceptance_rate" in c: all_dflash.append(c)
                total_steps += int(c.get("num_steps", 0))
            log.info(f"Loaded {rf.name}: {len(clips)} clips")
        except Exception as e:
            log.warning(f"Error loading {rf}: {e}")
    if not all_clips:
        return None

    summary = {
        f"min_ade_{num_traj_samples}": float(np.mean(all_ade_k)) if all_ade_k else None,
        "min_ade_1": float(np.mean(all_ade_1)) if all_ade_1 else None,
        "num_steps": total_steps, "num_clips": len(all_clips),
    }
    if all_timing:
        for k in ("avg_total_ms", "avg_encode_ms", "avg_prefill_ms", "avg_decode_ms",
                   "avg_action_ms", "avg_num_tokens", "avg_first_token_sample_ms",
                   "avg_dflash_loop_ms", "avg_traj_forward_ms"):
            summary[k] = float(np.mean([t.get(k, 0) for t in all_timing]))
    if all_dflash:
        for k in ("avg_acceptance_rate", "avg_acceptance_length", "avg_match_rate", "avg_iterations"):
            summary[k] = float(np.mean([d.get(k, 0) for d in all_dflash]))

    log.info(f"\n{'='*60}")
    log.info(f"AGGREGATED ({len(rank_files)} ranks, {len(all_clips)} clips, {total_steps} steps)")
    log.info("=" * 60)
    for k, v in summary.items():
        log.info(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    agg_file = Path(run_dir) / (
        f"w4a8_marlin_dflash_stream_acache{cache_str}"
        f"_K{num_traj_samples}_d{diffusion_steps}_{len(all_clips)}clips_aggregated.json"
    )
    agg_file.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "aggregated_from_ranks": len(rank_files),
        "expected_world_size": world_size,
        "summary": summary, "clips": all_clips,
    }, indent=2))
    log.info(f"Saved aggregated to {agg_file}")
    return str(agg_file)


# ─── Main ─────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="ParoQuant W4A8 Marlin eval")
    ap.add_argument("--model-path", default="/data/scratch/zekaili/train_expert_ckpts_deepspeed/checkpoint-6446")
    ap.add_argument("--paro-checkpoint", default="/data/scratch/zekaili/quant_cache/ckpt-paro-w4-vlm-mm.pt")
    ap.add_argument("--draft-model", default="/data/scratch/zekaili/Alpamayo-DFlash")
    ap.add_argument("--data-dir", default="/data/scratch/zekaili/dumped_eval_data")
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--num-traj-samples", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--diffusion-steps", type=int, default=8)
    ap.add_argument("--cache-steps", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--quantize-VLM-only", action="store_true")
    ap.add_argument("--output-dir", default="~/exp_paro_w4a8")
    args = ap.parse_args()

    for attr in ("model_path", "paro_checkpoint", "draft_model", "data_dir", "output_dir"):
        setattr(args, attr, os.path.expanduser(getattr(args, attr)))

    run_started_at_s = time.time()
    rank, local_rank, world_size, device = setup_distributed()
    if rank != 0:
        logging.getLogger().setLevel(logging.CRITICAL)
        log.setLevel(logging.CRITICAL)

    log.info(f"Distributed: rank={rank}, local_rank={local_rank}, "
             f"world_size={world_size}, device={device}")

    quantize_expert = not args.quantize_VLM_only
    cache_str = "-".join(str(s) for s in args.cache_steps)
    run_dir = args.output_dir
    os.makedirs(run_dir, exist_ok=True)

    config = vars(args).copy()
    config.update({"rank": rank, "local_rank": local_rank,
                   "world_size": world_size, "device": device})
    with open(os.path.join(run_dir, f"config_rank{rank}.json"), "w") as f:
        json.dump(config, f, indent=2)

    all_clip_ids = load_clip_ids_from_data_dir(args.data_dir, args.num_clips)
    clip_ids = split_clips_for_rank(all_clip_ids, rank, world_size)
    expected_rank_clip_counts = {
        r: len(split_clips_for_rank(all_clip_ids, r, world_size))
        for r in range(world_size)
    }
    log.info(f"Rank {rank}/{world_size-1}: {len(clip_ids)}/{len(all_clip_ids)} clips")

    if not clip_ids:
        log.warning(f"Rank {rank}: no clips assigned")
        return

    # ── Load model ──
    model = load_paroquant_model(
        model_path=args.model_path,
        paro_checkpoint=args.paro_checkpoint,
        mode="streaming",
        quantize_expert=quantize_expert,
    )

    log.info("Converting to Marlin W4A8 backend...")
    n = convert_model_to_marlin_w4a8(model)
    if n == 0:
        raise SystemExit("No layers converted!")

    log.info("Fusing expert projections...")
    fuse_expert_projections(model)

    setup_dflash_for_model(model, args.draft_model)
    log.info("DFlash enabled (ParoQuant W4A8 Marlin, streaming, action_cache)")

    # ── Eval loop ──
    all_clip_results, all_clip_ade_k, all_clip_ade_1 = [], [], []
    all_clip_timing, all_clip_dflash = [], []
    total_eval_steps = 0

    for ci, clip_id in enumerate(clip_ids):
        log.info(f"\n[Rank {rank}] Clip {ci+1}/{len(clip_ids)}: {clip_id}")
        reset_clip_state(model)

        try:
            streaming_inputs = load_streaming_inputs(args.data_dir, clip_id)
        except Exception as e:
            log.warning(f"  Error loading inputs: {e}")
            continue
        log.info(f"  Loaded {len(streaming_inputs)} inputs")

        clip_timing, clip_ade_k, clip_ade_1, clip_dflash = [], [], [], []

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
                    log.info(f"  Step {si}: prefill")
                    continue

                pred_xyz, _, extra = result
                min_ade_k, min_ade_1 = calc_min_ade(inputs["ego_future_xyz"], pred_xyz)
                timing = extra.get("timing") if extra else None
                ds = extra.get("dflash_stats") if extra else None

                is_warmup = si <= args.warmup_steps
                tag = " [W]" if is_warmup else ""
                if timing:
                    t = timing
                    tps = t.get("num_decode_tokens", 0) / (t.get("decode_time_ms", 1) / 1000)
                    ds_str = ""
                    if ds:
                        ds_str = (f", accept={ds['acceptance_rate']:.0%} "
                                  f"len={ds['mean_acceptance_length']:.2f}")
                    log.info(
                        f"  Step {si}{tag}: {t.get('total_time_ms',0):.1f}ms "
                        f"(enc={t.get('encode_time_ms',0):.1f} pf={t.get('prefill_time_ms',0):.1f} "
                        f"dec={t.get('decode_time_ms',0):.1f} act={t.get('action_time_ms',0):.1f}) "
                        f"{t.get('num_decode_tokens',0)}tok {tps:.1f}tok/s "
                        f"ADE={min_ade_1:.3f}m{ds_str}"
                    )

                if not is_warmup:
                    clip_ade_k.append(min_ade_k)
                    clip_ade_1.append(min_ade_1)
                    if timing: clip_timing.append(timing)
                    if ds: clip_dflash.append(ds)

            except Exception as e:
                log.warning(f"  Step {si} error: {e}")
                if "CUDA" in str(e):
                    raise SystemExit(1)
                continue

        if clip_ade_k:
            clip_result = {
                "clip_id": clip_id,
                "num_steps": len(clip_ade_k),
                f"min_ade_{args.num_traj_samples}": float(np.mean(clip_ade_k)),
                "min_ade_1": float(np.mean(clip_ade_1)),
            }
            if clip_timing:
                ct = clip_timing
                clip_result.update({
                    "avg_total_ms": float(np.mean([t["total_time_ms"] for t in ct])),
                    "avg_encode_ms": float(np.mean([t.get("encode_time_ms", 0) for t in ct])),
                    "avg_prefill_ms": float(np.mean([t.get("prefill_time_ms", 0) for t in ct])),
                    "avg_decode_ms": float(np.mean([t.get("decode_time_ms", 0) for t in ct])),
                    "avg_action_ms": float(np.mean([t.get("action_time_ms", 0) for t in ct])),
                    "avg_num_tokens": float(np.mean([t.get("num_decode_tokens", 0) for t in ct])),
                    "avg_first_token_sample_ms": float(np.mean([t.get("first_token_sample_time_ms", 0) for t in ct])),
                    "avg_dflash_loop_ms": float(np.mean([t.get("dflash_loop_time_ms", 0) for t in ct])),
                    "avg_traj_forward_ms": float(np.mean([t.get("traj_forward_time_ms", 0) for t in ct])),
                })
            if clip_dflash:
                clip_result.update({
                    "avg_acceptance_rate": float(np.mean([d["acceptance_rate"] for d in clip_dflash])),
                    "avg_acceptance_length": float(np.mean([d["mean_acceptance_length"] for d in clip_dflash])),
                    "avg_match_rate": float(np.mean([d["match_rate"] for d in clip_dflash])),
                    "avg_iterations": float(np.mean([d["total_iterations"] for d in clip_dflash])),
                })
            all_clip_results.append(clip_result)
            all_clip_ade_k.append(clip_result[f"min_ade_{args.num_traj_samples}"])
            all_clip_ade_1.append(clip_result["min_ade_1"])
            all_clip_timing.append(clip_result)
            if clip_dflash: all_clip_dflash.append(clip_result)
            total_eval_steps += len(clip_ade_k)

            avg_t = clip_result.get("avg_total_ms", 0)
            log.info(f"  Clip avg: {avg_t:.1f}ms, ADE_1={clip_result['min_ade_1']:.3f}m "
                     f"({len(clip_ade_k)} steps)")

    # ── Per-rank summary ──
    n = len(all_clip_results)
    log.info(f"\n{'='*60}")
    log.info(f"[Rank {rank}] RESULTS — ParoQuant W4A8 Marlin ({n} clips, {total_eval_steps} steps)")
    log.info("=" * 60)
    if all_clip_timing:
        def avg(k): return float(np.mean([t.get(k, 0) for t in all_clip_timing]))
        log.info(f"  Avg total:   {avg('avg_total_ms'):.1f} ms")
        log.info(f"  Avg decode:  {avg('avg_decode_ms'):.1f} ms")
        log.info(f"  Avg action:  {avg('avg_action_ms'):.1f} ms")
    if all_clip_ade_k:
        log.info(f"  Avg minADE_{args.num_traj_samples}: {np.mean(all_clip_ade_k):.3f} m")
        log.info(f"  Avg minADE_1: {np.mean(all_clip_ade_1):.3f} m")

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
                f"min_ade_{args.num_traj_samples}": float(np.mean(all_clip_ade_k)) if all_clip_ade_k else None,
                "min_ade_1": float(np.mean(all_clip_ade_1)) if all_clip_ade_1 else None,
                "num_steps": total_eval_steps, "num_clips": n,
                **({k: float(np.mean([t.get(k, 0) for t in all_clip_timing]))
                    for k in ("avg_total_ms", "avg_encode_ms", "avg_prefill_ms",
                              "avg_decode_ms", "avg_action_ms", "avg_num_tokens",
                              "avg_first_token_sample_ms", "avg_dflash_loop_ms",
                              "avg_traj_forward_ms")} if all_clip_timing else {}),
                **({k: float(np.mean([d.get(k, 0) for d in all_clip_dflash]))
                    for k in ("avg_acceptance_rate", "avg_acceptance_length",
                              "avg_match_rate", "avg_iterations")} if all_clip_dflash else {}),
            },
            "clips": all_clip_results,
        }, f, indent=2)
    log.info(f"Saved to {out_file}")

    # ── Aggregate (rank 0 only) ──
    if rank == 0 and world_size > 1:
        log.info(f"\nAggregating across {world_size} ranks...")
        aggregate_results_across_ranks(
            run_dir, args.num_traj_samples, args.diffusion_steps, cache_str,
            world_size, expected_rank_clip_counts, run_started_at_s,
        )


if __name__ == "__main__":
    main()
