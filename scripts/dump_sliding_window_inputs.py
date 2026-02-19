#!/usr/bin/env python3
"""Dump sliding_window_inputs for each clip to disk. dump_data.py can then use --dumped_inputs_dir to read from disk.

Each clip is saved to <output_dir>/<clip_id>/sliding_window_inputs.pt (list of model_inputs dicts).
Run this once (e.g. with distributed or many workers) to populate the dir; then run dump_data.py with
--dumped_inputs_dir <output_dir> to skip load_physical_aiavdataset and tokenization.

Usage:
  # Single process (from repo root, PYTHONPATH=src)
  python scripts/dump_sliding_window_inputs.py --clip_ids_file clips_for_train.json --output_dir ./dumped_inputs

  # Distributed (e.g. 8 GPUs): each rank dumps its subset of clips (no GPU needed for this script)
  torchrun --nproc_per_node=8 scripts/dump_sliding_window_inputs.py --clip_ids_file clips_for_train.json --output_dir ./dumped_inputs
"""

import argparse
import json
import logging
import os
import sys

import torch
import torch.distributed as dist

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _to_cpu(obj):
    """Move all tensors in nested dict/list to CPU (for saving)."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    return obj


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="gloo")
        return rank, world_size
    return 0, 1


def main():
    parser = argparse.ArgumentParser(description="Dump sliding_window_inputs for each clip to disk.")
    parser.add_argument("--clip_ids_file", type=str, required=True, help="JSON file with list of clip IDs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Root dir for dumps: output_dir/<clip_id>/sliding_window_inputs.pt")
    parser.add_argument("--model_path", type=str, default="./Alpamayo-R1-10B", help="Model path (for processor/tokenizer).")
    parser.add_argument("--num_steps", type=int, default=120)
    parser.add_argument("--t0_us", type=int, default=1_700_000)
    parser.add_argument("--time_step_us", type=int, default=100_000)
    args = parser.parse_args()

    rank, world_size = setup_distributed()

    with open(args.clip_ids_file) as f:
        all_clip_ids = json.load(f)
    if not isinstance(all_clip_ids, list):
        all_clip_ids = list(all_clip_ids)
    my_clip_ids = all_clip_ids[rank::world_size]

    logger.info("Rank %s/%s: dumping inputs for %s clips (total %s)", rank, world_size, len(my_clip_ids), len(all_clip_ids))

    # Import here so we don't load model when only parsing args
    from alpamayo_r1.dump_data import create_sliding_window_inputs, load_model
    from alpamayo_r1 import helper

    # Load model on CPU to get tokenizer (AutoTokenizer.from_pretrained fails with custom AlpamayoR1Config)
    _model, processor = load_model(args, device=torch.device("cpu"))
    del _model  # free memory; we only need processor for create_sliding_window_inputs

    for i, clip_id in enumerate(my_clip_ids):
        out_subdir = os.path.join(args.output_dir, clip_id)
        out_file = os.path.join(out_subdir, "sliding_window_inputs.pt")
        if os.path.isfile(out_file):
            logger.info("Rank %s: skip (exists) %s/%s %s", rank, i + 1, len(my_clip_ids), clip_id)
            continue
        try:
            sliding_window_inputs = create_sliding_window_inputs(
                processor=processor,
                num_windows=args.num_steps,
                clip_id=clip_id,
                t0_us=args.t0_us,
                time_step_us=args.time_step_us,
            )
            # Move all tensors to CPU and save
            sliding_window_inputs_cpu = _to_cpu(sliding_window_inputs)
            os.makedirs(out_subdir, exist_ok=True)
            torch.save(sliding_window_inputs_cpu, out_file)
            logger.info("Rank %s: saved %s/%s %s (%s windows)", rank, i + 1, len(my_clip_ids), clip_id, len(sliding_window_inputs))
        except Exception as e:
            logger.warning("Rank %s: failed %s: %s", rank, clip_id, e)

    if dist.is_initialized():
        dist.destroy_process_group()
    logger.info("Rank %s: done.", rank)


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(repo_root, "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)
    main()
