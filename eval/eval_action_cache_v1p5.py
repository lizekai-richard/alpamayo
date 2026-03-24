#!/usr/bin/env python3
"""System-optimization-only evaluation: non-streaming mode with torch.compile.

Uses the dev branch AlpamayoR1 with streaming=False — torch.compile + static
cache + manual decode loop, but NO KV cache reuse across frames.
Runs through clips from 1.7s to 13.6s at 10Hz (120 timesteps).
Reports minADE_K (default K=6) and per-step timing.

Usage:
    # Single GPU
    python eval/eval_system_opt.py
    python eval/eval_system_opt.py --num-clips 10 --diffusion-steps 5
    python eval/eval_system_opt.py --num-traj-samples 1

    # Multi-GPU (using torchrun or similar)
    torchrun --nproc_per_node=4 eval/eval_system_opt.py --num-clips 100

    # Manual multi-GPU (set environment variables)
    RANK=0 WORLD_SIZE=2 LOCAL_RANK=0 python eval/eval_system_opt.py --num-clips 100 &
    RANK=1 WORLD_SIZE=2 LOCAL_RANK=1 python eval/eval_system_opt.py --num-clips 100 &
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import alpamayo_r1
sys.modules["alpamayo1_5"] = alpamayo_r1

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpamayo_r1.models.alpamayo_r1p5_dit_cache import Alpamayo1_5
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


def setup_distributed():
    """Setup distributed evaluation environment.

    Returns:
        (rank, local_rank, world_size, device)
        If not distributed, returns (0, 0, 1, "cuda:0")
    """
    # Check for torch.distributed environment variables
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, device

    # Fallback: use number of GPUs and manual rank/world_size
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", num_gpus))
        local_rank = rank % num_gpus
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, device

    # Single GPU
    return 0, 0, 1, "cuda:0"


def split_clips_for_rank(clip_ids, rank, world_size):
    """Split clips across ranks using round-robin assignment.

    Args:
        clip_ids: List of all clip IDs
        rank: Current rank (0 to world_size-1)
        world_size: Total number of ranks

    Returns:
        List of clip IDs assigned to this rank
    """
    return [clip_ids[i] for i in range(len(clip_ids)) if i % world_size == rank]


def load_inputs(clip_id, dumped_data_dir):
    """Load dumped inputs or create them from scratch."""
    num_steps = 120
    log.info("Loading dumped inputs from %s for clip %s", dumped_data_dir, clip_id)
    windows = helper.load_dumped_inputs(dumped_data_dir, clip_id)
    windows = windows[:num_steps]
    return windows


def aggregate_results_across_ranks(
    run_dir,
    num_traj_samples,
    diffusion_steps,
    world_size,
    expected_rank_clip_counts,
    run_started_at_s,
    max_wait_seconds=300,
):
    """Aggregate results from all rank files into a single summary.

    Args:
        run_dir: Output directory containing rank result files
        num_traj_samples: Number of trajectory samples (for filename)
        diffusion_steps: Number of diffusion steps (for filename)
        world_size: Total number of ranks
        expected_rank_clip_counts: Dict rank -> number of clips assigned to that rank
        run_started_at_s: Unix timestamp when this run started (used to ignore stale files)
        max_wait_seconds: Maximum time to wait for all rank files to appear

    Returns:
        Path to aggregated results file, or None if aggregation failed
    """
    if world_size <= 1:
        return None

    # Wait for expected rank files from this run.
    expected_rank_files = {
        r: Path(run_dir) / (
            f"system_opt_K{num_traj_samples}_d{diffusion_steps}_"
            f"{expected_rank_clip_counts[r]}clips_rank{r}.json"
        )
        for r in range(world_size)
    }

    start_time = time.time()
    rank_files = {}
    while len(rank_files) < world_size and (time.time() - start_time) < max_wait_seconds:
        rank_files = {}
        for r, rank_file in expected_rank_files.items():
            if not rank_file.exists():
                continue
            # Ignore stale files from previous runs.
            if rank_file.stat().st_mtime + 1 < run_started_at_s:
                continue
            rank_files[r] = rank_file
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

    # Load all rank results.
    all_samples = []
    all_ade_k = []
    all_ade_1 = []
    all_timing = []
    total_steps = 0

    for _, rank_file in sorted(rank_files.items()):
        try:
            with open(rank_file, "r") as f:
                data = json.load(f)

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

            total_steps += len(samples)
            log.info(f"Loaded results from {rank_file.name}: {len(samples)} samples")
        except Exception as e:
            log.warning(f"Error loading {rank_file}: {e}")
            continue

    if not all_samples:
        log.warning("No samples found in rank files")
        return None

    # Compute aggregated summary
    aggregated_summary = {
        f"min_ade_{num_traj_samples}": float(np.mean(all_ade_k)) if all_ade_k else None,
        "min_ade_1": float(np.mean(all_ade_1)) if all_ade_1 else None,
        "num_steps": total_steps,
        "num_clips": len(set(s.get("clip_id", "") for s in all_samples)),
    }

    if all_timing:
        aggregated_summary.update({
            "avg_total_ms": float(np.mean([t.get("total_time_ms", 0) for t in all_timing])),
            "avg_encode_ms": float(np.mean([t.get("encode_time_ms", 0) for t in all_timing])),
            "avg_prefill_ms": float(np.mean([t.get("prefill_time_ms", 0) for t in all_timing])),
            "avg_decode_ms": float(np.mean([t.get("decode_time_ms", 0) for t in all_timing])),
            "avg_action_ms": float(np.mean([t.get("action_time_ms", 0) for t in all_timing])),
            "avg_num_tokens": float(np.mean([t.get("num_decode_tokens", 0) for t in all_timing])),
        })

    # Save aggregated results
    aggregated_file = os.path.join(
        run_dir,
        f"system_opt_K{num_traj_samples}_d{diffusion_steps}_{total_steps}steps_aggregated.json",
    )

    with open(aggregated_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "aggregated_from_ranks": len(rank_files),
            "expected_world_size": world_size,
            "summary": aggregated_summary,
            "samples": all_samples,
        }, f, indent=2)

    log.info(
        f"Aggregated results from {len(rank_files)} ranks: "
        f"{aggregated_summary['num_clips']} clips, {total_steps} steps"
    )
    log.info(f"  Aggregated minADE_{num_traj_samples}: {aggregated_summary.get(f'min_ade_{num_traj_samples}', 'N/A')}")
    log.info(f"  Aggregated minADE_1: {aggregated_summary.get('min_ade_1', 'N/A')}")
    if all_timing:
        log.info(f"  Aggregated avg total time: {aggregated_summary.get('avg_total_ms', 0):.1f} ms")
        log.info(f"  Aggregated avg decode time: {aggregated_summary.get('avg_decode_ms', 0):.1f} ms")

    return aggregated_file


def main():
    ap = argparse.ArgumentParser(description="System-opt eval (non-streaming, torch.compile)")
    ap.add_argument("--model-path", default="nvidia/Alpamayo-1.5-10B")
    ap.add_argument("--clip-ids-file", default="./clips.json")
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--num-traj-samples", type=int, default=1,
                     help="K for minADE_K (default 6)")
    ap.add_argument("--diffusion-steps", type=int, default=8)
    ap.add_argument("--cache-steps", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--warmup-steps", type=int, default=3,
                     help="First N steps excluded from metrics")
    ap.add_argument("--output-dir", default="./action_cache_results_v1.5")
    ap.add_argument("--cache-dir", default="/data/scratch/zekaili/hf_cache")
    ap.add_argument("--dumped-data-dir", default="/data/scratch/zekaili/dumped_eval_data_v1.5")
    args = ap.parse_args()

    for attr in ("model_path", "clip_ids_file", "output_dir", "cache_dir", "dumped_data_dir"):
        setattr(args, attr, os.path.expanduser(getattr(args, attr)))

    run_started_at_s = time.time()

    # --- setup distributed ---
    rank, local_rank, world_size, device = setup_distributed()

    # Only rank 0 emits logs to avoid duplicated multi-GPU output.
    if rank != 0:
        logging.getLogger().setLevel(logging.CRITICAL)
        log.setLevel(logging.CRITICAL)

    log.info(f"Distributed setup: rank={rank}, local_rank={local_rank}, world_size={world_size}, device={device}")

    run_dir = args.output_dir
    os.makedirs(run_dir, exist_ok=True)

    # Save config with distributed info
    config = vars(args).copy()
    config.update({
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
    })
    with open(os.path.join(run_dir, f"config_rank{rank}.json"), "w") as f:
        json.dump(config, f, indent=2)

    # --- dataset ---
    import physical_ai_av
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=args.cache_dir)

    # Load all clips and split across ranks
    all_clip_ids = load_clip_ids(args.clip_ids_file, args.num_clips)
    clip_ids = split_clips_for_rank(all_clip_ids, rank, world_size)
    expected_rank_clip_counts = {
        r: len(split_clips_for_rank(all_clip_ids, r, world_size))
        for r in range(world_size)
    }

    log.info(f"Rank {rank}/{world_size-1}: Processing {len(clip_ids)}/{len(all_clip_ids)} clips "
             f"(K={args.num_traj_samples}, diffusion_steps={args.diffusion_steps})")

    if not clip_ids:
        log.warning(f"Rank {rank}: No clips assigned, exiting")
        return

    # --- model (dev branch: torch.compile + static cache, non-streaming) ---
    model = Alpamayo1_5.from_pretrained(
        args.model_path, dtype=torch.bfloat16,
    ).to(device)
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

        log.info(f"\n[Rank {rank}] Clip {ci+1}/{len(clip_ids)}: {clip_id}")
        clip_timing, clip_ade_k, clip_ade_1 = [], [], []

        # Pre-create all inputs for this clip
        log.info(f"  Creating inputs for {len(t0s)} timesteps...")
        clip_inputs = []
        try:
            # for t0 in t0s:
            #     data = load_physical_aiavdataset(
            #         clip_id, t0_us=t0, num_frames=4, avdi=avdi,
            #     )
            #     clip_inputs.append(prepare_inputs(data, processor))
            clip_inputs = load_inputs(clip_id, args.dumped_data_dir)
        except Exception as e:
            log.warning(f"  Error creating inputs: {e}")
            continue
        log.info(f"  Created {len(clip_inputs)} inputs, starting inference...")

        for si, (t0, inputs) in enumerate(zip(t0s, clip_inputs)):
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model.sample_trajectories_from_data_with_vlm_rollout(
                        data=helper.to_device(inputs, device),
                        num_traj_samples=args.num_traj_samples,
                        max_new_tokens=128,
                        return_extra=True,
                        fuse_qkv=True,
                        fuse_gate_up=True,
                        diffusion_kwargs={
                            "inference_step": args.diffusion_steps,
                            "cache_steps": args.cache_steps,
                            "int_method": "euler_with_cache"
                        },
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
    log.info(f"[Rank {rank}] AGGREGATE RESULTS — System-opt (non-streaming, torch.compile) ({n} steps)")
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
    # Include rank in filename for distributed runs
    rank_suffix = f"_rank{rank}" if world_size > 1 else ""
    out_file = os.path.join(
        run_dir,
        f"system_opt_K{args.num_traj_samples}_d{args.diffusion_steps}_{len(clip_ids)}clips{rank_suffix}.json",
    )
    with open(out_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "distributed": {
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "device": device,
            },
            "summary": {
                f"min_ade_{args.num_traj_samples}": float(np.mean(all_ade_k)) if all_ade_k else None,
                "min_ade_1": float(np.mean(all_ade_1)) if all_ade_1 else None,
                "num_steps": len(all_ade_k),
                "num_clips": len(all_results),
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

    # --- aggregate results across all ranks (rank 0 only) ---
    if rank == 0 and world_size > 1:
        log.info(f"\n{'='*60}")
        log.info(f"Aggregating results across {world_size} ranks...")
        log.info("=" * 60)
        aggregated_file = aggregate_results_across_ranks(
            run_dir,
            args.num_traj_samples,
            args.diffusion_steps,
            world_size,
            expected_rank_clip_counts,
            run_started_at_s,
        )
        if aggregated_file:
            log.info(f"Aggregated results saved to {aggregated_file}")


if __name__ == "__main__":
    main()
