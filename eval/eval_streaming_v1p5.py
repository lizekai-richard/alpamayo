#!/usr/bin/env python3
"""Streaming evaluation for Alpamayo 1.5.

Compares three streaming configurations:
  A) keep_frame_labels=True  + kv_shift_mode=block
  B) keep_frame_labels=True  + kv_shift_mode=vision_only
  C) keep_frame_labels=False + kv_shift_mode=vision_only

Uses pre-dumped sliding window inputs (v1.5 format with camera/frame labels).
Supports DDP: each GPU evaluates a subset of clips.

Usage:
    # Single GPU
    python eval/eval_streaming_v1p5.py --data-dir /path/to/dumped_eval_data_v1.5 --clip-list eval_clips.json

    # Multi-GPU (8 GPUs)
    torchrun --nproc_per_node=8 eval/eval_streaming_v1p5.py --data-dir /path/to/dumped_eval_data_v1.5 --clip-list eval_clips.json
"""

import argparse
import json
import logging
import os
import sys
import datetime
from datetime import datetime as dt_now

import numpy as np
import torch
import torch.distributed as dist

import alpamayo_r1
sys.modules["alpamayo1_5"] = alpamayo_r1

from alpamayo_r1.train.alpamayo1_5 import Alpamayo1_5
from alpamayo_r1.train.patches import patch_for_training
from alpamayo_r1.helper import convert_to_streaming_window_v1p5, to_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)


CONFIGS = [
    {"name": "labels+block",       "keep_frame_labels": True,  "kv_shift_mode": "block"},
    {"name": "labels+vision_only", "keep_frame_labels": True,  "kv_shift_mode": "vision_only"},
    {"name": "no_labels+vision_only", "keep_frame_labels": False, "kv_shift_mode": "vision_only"},
]


def setup_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(hours=2),
        )
        return rank, local_rank, world_size
    return 0, 0, 1


def calc_min_ade(gt_future_xyz, pred_xyz):
    """Compute minADE_1 and minADE_K.

    Args:
        gt_future_xyz: [1, 1, T, 3+] ground truth.
        pred_xyz: [1, 1, K, T, 3] predictions.

    Returns:
        (minADE_K, minADE_1) as floats.
    """
    gt_xy = gt_future_xyz[0, 0, :, :2]  # [T, 2]
    pred_xy = pred_xyz[0, 0, :, :, :2].cpu()  # [K, T, 2]
    ade_per_sample = (pred_xy - gt_xy.unsqueeze(0)).norm(dim=-1).mean(dim=-1)  # [K]
    return ade_per_sample.min().item(), ade_per_sample[0].item()


def eval_clip(model, windows, ve_id, vs_id, config, num_traj_samples, max_gen_len, diffusion_steps):
    """Run streaming eval on one clip with given config. Returns list of (ade_k, ade_1)."""
    model.keep_frame_labels = config["keep_frame_labels"]
    model.kv_shift_mode = config["kv_shift_mode"]
    model.reset_streaming_state()

    results = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, w in enumerate(windows):
            if i == 0:
                data = {
                    "tokenized_data": w["tokenized_data"],
                    "ego_history_xyz": w["ego_history_xyz"],
                    "ego_history_rot": w["ego_history_rot"],
                    "is_prefill": True,
                }
            else:
                data = convert_to_streaming_window_v1p5(
                    w, ve_id,
                    keep_frame_labels=config["keep_frame_labels"],
                    vision_start_id=vs_id if not config["keep_frame_labels"] else None,
                )

            result = model.sample_trajectories_from_data_with_streaming_vlm_rollout(
                data=to_device(data, "cuda"),
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=num_traj_samples,
                return_extra=False,
                max_generation_length=max_gen_len,
                diffusion_kwargs={"inference_step": diffusion_steps},
            )

            if result is None:
                continue

            pred_xyz, pred_rot = result[0], result[1]
            ade_k, ade_1 = calc_min_ade(w["ego_future_xyz"], pred_xyz)
            results.append((ade_k, ade_1))

    return results


def main():
    ap = argparse.ArgumentParser(description="Streaming eval for Alpamayo 1.5 (3 configs)")
    ap.add_argument("--model-path", default="nvidia/Alpamayo-1.5-10B")
    ap.add_argument("--data-dir", required=True, help="Path to dumped v1.5 eval data")
    ap.add_argument("--clip-list", required=True, help="JSON file with clip IDs")
    ap.add_argument("--num-clips", type=int, default=None, help="Limit number of clips")
    ap.add_argument("--num-traj-samples", type=int, default=6, help="K for minADE_K")
    ap.add_argument("--max-gen-len", type=int, default=128, help="Max CoT tokens")
    ap.add_argument("--diffusion-steps", type=int, default=10)
    ap.add_argument("--output-dir", default="eval_results_v1p5")
    args = ap.parse_args()

    rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0

    # Load clip list
    with open(args.clip_list) as f:
        all_clip_ids = json.load(f)
    if args.num_clips is not None:
        all_clip_ids = all_clip_ids[:args.num_clips]

    # Distribute clips across ranks
    my_clip_ids = all_clip_ids[rank::world_size]
    if is_main:
        log.info(f"Total clips: {len(all_clip_ids)}, world_size: {world_size}, "
                 f"clips per rank: ~{len(my_clip_ids)}, K={args.num_traj_samples}")

    # Load model
    if is_main:
        log.info(f"Loading model from {args.model_path}...")
    model = Alpamayo1_5.from_pretrained(args.model_path, dtype=torch.bfloat16).to("cuda")
    patch_for_training(model)
    tokenizer = model.tokenizer
    vs_id = tokenizer.encode("<|vision_start|>")[0]
    ve_id = tokenizer.encode("<|vision_end|>")[0]
    if is_main:
        log.info("Model loaded")

    # Eval each config
    all_config_results = {}
    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        if is_main:
            log.info(f"\n{'='*60}")
            log.info(f"Config: {cfg_name}")
            log.info(f"  keep_frame_labels={cfg['keep_frame_labels']}, kv_shift_mode={cfg['kv_shift_mode']}")
            log.info(f"{'='*60}")

        local_ade_k, local_ade_1 = [], []

        for ci, clip_id in enumerate(my_clip_ids):
            clip_path = os.path.join(args.data_dir, clip_id, "sliding_window_inputs.pt")
            if not os.path.isfile(clip_path):
                log.warning(f"[rank {rank}] Clip {clip_id}: file not found, skipping")
                continue

            windows = torch.load(clip_path, map_location="cpu", weights_only=False)
            results = eval_clip(
                model, windows, ve_id, vs_id, cfg,
                args.num_traj_samples, args.max_gen_len, args.diffusion_steps,
            )

            if results:
                clip_ade_k = np.mean([r[0] for r in results])
                clip_ade_1 = np.mean([r[1] for r in results])
                local_ade_k.append(clip_ade_k)
                local_ade_1.append(clip_ade_1)
                log.info(f"[rank {rank}] {cfg_name} clip {ci+1}/{len(my_clip_ids)} "
                         f"{clip_id[:8]}...: minADE_K={clip_ade_k:.3f}m, "
                         f"minADE_1={clip_ade_1:.3f}m ({len(results)} windows)")

        # Sync all ranks before gathering
        if world_size > 1:
            dist.barrier()

        # Gather results across ranks
        if world_size > 1:
            # Gather list lengths
            local_count = torch.tensor([len(local_ade_k)], dtype=torch.long, device="cuda")
            all_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
            dist.all_gather(all_counts, local_count)

            # Gather ADE values
            local_ade_k_t = torch.tensor(local_ade_k, dtype=torch.float64, device="cuda")
            local_ade_1_t = torch.tensor(local_ade_1, dtype=torch.float64, device="cuda")

            max_count = max(c.item() for c in all_counts)
            # Pad to same length for all_gather
            padded_k = torch.zeros(max_count, dtype=torch.float64, device="cuda")
            padded_1 = torch.zeros(max_count, dtype=torch.float64, device="cuda")
            padded_k[:len(local_ade_k)] = local_ade_k_t
            padded_1[:len(local_ade_1)] = local_ade_1_t

            gathered_k = [torch.zeros_like(padded_k) for _ in range(world_size)]
            gathered_1 = [torch.zeros_like(padded_1) for _ in range(world_size)]
            dist.all_gather(gathered_k, padded_k)
            dist.all_gather(gathered_1, padded_1)

            # Unpad and concatenate
            global_ade_k, global_ade_1 = [], []
            for r in range(world_size):
                n = all_counts[r].item()
                global_ade_k.extend(gathered_k[r][:n].cpu().tolist())
                global_ade_1.extend(gathered_1[r][:n].cpu().tolist())
        else:
            global_ade_k = local_ade_k
            global_ade_1 = local_ade_1

        all_config_results[cfg_name] = {
            "ade_k": global_ade_k,
            "ade_1": global_ade_1,
        }

        if is_main and global_ade_k:
            log.info(f"\n  {cfg_name}: Avg minADE_K={np.mean(global_ade_k):.3f}m, "
                     f"Avg minADE_1={np.mean(global_ade_1):.3f}m "
                     f"({len(global_ade_k)} clips)")

    # Summary
    if is_main:
        log.info(f"\n{'='*60}")
        log.info("SUMMARY")
        log.info(f"{'='*60}")
        log.info(f"{'Config':<30} {'minADE_K':>10} {'minADE_1':>10} {'clips':>6}")
        log.info("-" * 58)
        for cfg_name, res in all_config_results.items():
            if res["ade_k"]:
                log.info(f"{cfg_name:<30} {np.mean(res['ade_k']):>10.3f} "
                         f"{np.mean(res['ade_1']):>10.3f} {len(res['ade_k']):>6}")

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        out_file = os.path.join(args.output_dir, f"streaming_v1p5_comparison_{len(all_clip_ids)}clips.json")
        summary = {
            "timestamp": dt_now.now().isoformat(),
            "config": vars(args),
            "results": {
                name: {
                    f"avg_min_ade_{args.num_traj_samples}": float(np.mean(res["ade_k"])) if res["ade_k"] else None,
                    "avg_min_ade_1": float(np.mean(res["ade_1"])) if res["ade_1"] else None,
                    "num_clips": len(res["ade_k"]),
                    "per_clip_ade_k": res["ade_k"],
                    "per_clip_ade_1": res["ade_1"],
                }
                for name, res in all_config_results.items()
            },
        }
        with open(out_file, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"Saved to {out_file}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
