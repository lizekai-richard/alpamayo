#!/usr/bin/env python3
"""Streaming evaluation for Alpamayo 1.5.

Compares two streaming configurations:
  A) keep_frame_labels=True  + kv_shift_mode=block
  B) keep_frame_labels=False + kv_shift_mode=vision_only

Uses pre-dumped sliding window inputs (v1.5 format with camera/frame labels).
Supports DDP: each GPU evaluates a subset of clips independently.
Each rank saves its own results — no cross-rank communication needed.
After all ranks finish, run with --aggregate to merge results and print summary.

Usage:
    # Multi-GPU eval (8 GPUs)
    torchrun --nproc_per_node=8 eval/eval_streaming_v1p5.py \
        --data-dir /path/to/dumped_eval_data_v1p5 --clip-list eval_clips.json

    # Aggregate after all ranks finish
    python eval/eval_streaming_v1p5.py --aggregate --output-dir eval_results_v1p5
"""

import argparse
import glob
import json
import logging
import os
import sys

import numpy as np
import torch

import alpamayo_r1
sys.modules["alpamayo1_5"] = alpamayo_r1

from alpamayo_r1.models.alpamayo_r1p5_streaming import StreamingAlpamayo1_5
from alpamayo_r1.helper import convert_to_streaming_window_v1p5, to_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)


CONFIGS = [
    {"name": "vision_only+no_labels", "keep_frame_labels": False, "kv_shift_mode": "vision_only"},
    {"name": "block+with_labels", "keep_frame_labels": True, "kv_shift_mode": "block"},
]


def calc_min_ade(gt_future_xyz, pred_xyz):
    """Compute minADE_1 and minADE_K.

    Returns:
        (minADE_K, minADE_1) as floats.
    """
    gt_xy = gt_future_xyz[0, 0, :, :2]  # [T, 2]
    pred_xy = pred_xyz[0, 0, :, :, :2].cpu()  # [K, T, 2]
    ade_per_sample = (pred_xy - gt_xy.unsqueeze(0)).norm(dim=-1).mean(dim=-1)  # [K]
    return ade_per_sample.min().item(), ade_per_sample[0].item()


def eval_clip(model, windows, ve_id, vs_id, config, num_traj_samples, max_gen_len, diffusion_steps):
    """Run streaming eval on one clip. Returns list of per-step dicts."""
    model.reset_streaming_state(keep_frame_labels=config["keep_frame_labels"], kv_shift_mode=config["kv_shift_mode"])

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
            log.info(f"Input data shape: {data['tokenized_data']['input_ids'].shape}")
            result = model.sample_trajectories_from_data_with_streaming_vlm_rollout(
                data=to_device(data, "cuda"),
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=num_traj_samples,
                return_extra=True,
                max_generation_length=max_gen_len,
                diffusion_kwargs={"inference_step": diffusion_steps},
                fuse_qkv=True,
                fuse_gate_up=True,
                torch_compile="max-autotune",
            )

            if result is None:
                continue

            pred_xyz, pred_rot = result[0], result[1]
            extra = result[2] if len(result) > 2 else None
            ade_k, ade_1 = calc_min_ade(w["ego_future_xyz"], pred_xyz)

            cot_texts = []
            if extra and "cot" in extra:
                cot_arr = extra["cot"][0][0]
                cot_texts = cot_arr.tolist() if hasattr(cot_arr, "tolist") else list(cot_arr)

            results.append({"step": i, "ade_k": ade_k, "ade_1": ade_1, "cot": cot_texts})

    return results


def save_clip_result(output_dir, cfg_name, clip_id, step_results):
    """Save one clip's results as a JSON file with per-step data and summary."""
    cfg_dir = os.path.join(output_dir, cfg_name)
    os.makedirs(cfg_dir, exist_ok=True)

    ade_k_list = [r["ade_k"] for r in step_results]
    ade_1_list = [r["ade_1"] for r in step_results]

    clip_json = {
        "clip_id": clip_id,
        "num_steps": len(step_results),
        "avg_minade_k": float(np.mean(ade_k_list)),
        "avg_minade_1": float(np.mean(ade_1_list)),
        "steps": step_results,
    }

    out_path = os.path.join(cfg_dir, f"{clip_id}.json")
    with open(out_path, "w") as f:
        json.dump(clip_json, f, indent=2)


def aggregate(output_dir):
    """Read all per-clip JSONs and print summary table."""
    log.info(f"Aggregating results from {output_dir}")
    log.info(f"{'Config':<30} {'minADE_K':>10} {'minADE_1':>10} {'clips':>6}")
    log.info("-" * 58)

    summary = {}
    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        cfg_dir = os.path.join(output_dir, cfg_name)
        if not os.path.isdir(cfg_dir):
            continue

        clip_files = sorted(glob.glob(os.path.join(cfg_dir, "*.json")))
        all_ade_k, all_ade_1 = [], []
        for fpath in clip_files:
            with open(fpath) as f:
                data = json.load(f)
            all_ade_k.append(data["avg_minade_k"])
            all_ade_1.append(data["avg_minade_1"])

        if all_ade_k:
            avg_k = float(np.mean(all_ade_k))
            avg_1 = float(np.mean(all_ade_1))
            log.info(f"{cfg_name:<30} {avg_k:>10.3f} {avg_1:>10.3f} {len(all_ade_k):>6}")
            summary[cfg_name] = {
                "avg_minade_k": avg_k,
                "avg_minade_1": avg_1,
                "num_clips": len(all_ade_k),
                "per_clip_minade_k": all_ade_k,
                "per_clip_minade_1": all_ade_1,
            }

    out_path = os.path.join(output_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Streaming eval for Alpamayo 1.5 (A/B configs)")
    ap.add_argument("--model-path", default="nvidia/Alpamayo-1.5-10B")
    ap.add_argument("--data-dir", default="/mnt/moosefs/users/zekail/dumped_eval_data_v1p5", help="Path to dumped v1.5 eval data")
    ap.add_argument("--clip-list", default="./clips.json", help="JSON file with clip IDs")
    ap.add_argument("--num-clips", type=int, default=None, help="Limit number of clips")
    ap.add_argument("--num-traj-samples", type=int, default=6, help="K for minADE_K")
    ap.add_argument("--max-gen-len", type=int, default=128, help="Max CoT tokens")
    ap.add_argument("--diffusion-steps", type=int, default=10)
    ap.add_argument("--output-dir", default="eval_results_v1p5")
    ap.add_argument("--aggregate", action="store_true", help="Aggregate results and print summary (no GPU needed)")
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.output_dir)
        return

    # --- Distributed setup ---
    rank, world_size = 0, 1
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
    is_main = rank == 0

    # --- Clip list ---
    with open(args.clip_list) as f:
        all_clip_ids = json.load(f)
    if args.num_clips is not None:
        all_clip_ids = all_clip_ids[:args.num_clips]

    my_clip_ids = all_clip_ids[rank::world_size]
    if is_main:
        log.info(f"Total clips: {len(all_clip_ids)}, world_size: {world_size}, "
                 f"clips per rank: ~{len(my_clip_ids)}, K={args.num_traj_samples}")

    # --- Model ---
    if is_main:
        log.info(f"Loading model from {args.model_path}...")
    model = StreamingAlpamayo1_5.from_pretrained(args.model_path, dtype=torch.bfloat16).to("cuda")
    tokenizer = model.tokenizer
    vs_id = tokenizer.encode("<|vision_start|>")[0]
    ve_id = tokenizer.encode("<|vision_end|>")[0]
    if is_main:
        log.info("Model loaded")

    # --- Eval each config ---
    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        if is_main:
            log.info(f"\n{'='*60}")
            log.info(f"Config: {cfg_name}")
            log.info(f"{'='*60}")

        for ci, clip_id in enumerate(my_clip_ids):
            clip_path = os.path.join(args.data_dir, clip_id, "sliding_window_inputs.pt")
            if not os.path.isfile(clip_path):
                log.warning(f"[rank {rank}] Clip {clip_id}: file not found, skipping")
                continue

            # Skip if already done
            out_path = os.path.join(args.output_dir, cfg_name, f"{clip_id}.json")
            if os.path.isfile(out_path):
                log.info(f"[rank {rank}] {cfg_name} clip {ci+1}/{len(my_clip_ids)} {clip_id[:8]}...: already done, skipping")
                continue

            windows = torch.load(clip_path, map_location="cpu", weights_only=False)
            results = eval_clip(
                model, windows, ve_id, vs_id, cfg,
                args.num_traj_samples, args.max_gen_len, args.diffusion_steps,
            )

            if results:
                save_clip_result(args.output_dir, cfg_name, clip_id, results)
                avg_k = np.mean([r["ade_k"] for r in results])
                avg_1 = np.mean([r["ade_1"] for r in results])
                log.info(f"[rank {rank}] {cfg_name} clip {ci+1}/{len(my_clip_ids)} "
                         f"{clip_id[:8]}...: minADE_K={avg_k:.3f}m, "
                         f"minADE_1={avg_1:.3f}m ({len(results)} windows)")

    log.info(f"[rank {rank}] All configs done.")
    log.info(f"[rank {rank}] Run with --aggregate to see summary after all ranks finish.")


if __name__ == "__main__":
    main()
