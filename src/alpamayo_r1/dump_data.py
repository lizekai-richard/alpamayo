# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# End-to-end example script for the inference pipeline:
# This script loads a dataset, runs inference, and computes the minADE.
# It can be used to test the inference pipeline.
#
# Distributed (8 GPUs): clip list is split across ranks (rank 0 gets 0,8,16,..., rank 1 gets 1,9,17,...).
#   From repo root: torchrun --nproc_per_node=8 src/alpamayo_r1/dump_data.py --clip_ids_file clips_for_train.json --output_dir ./out
# Single clip:
#   python src/alpamayo_r1/dump_data.py --clip_id <uuid> --output_dir ./out

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(args, device=None):
    if device is None:
        device = torch.device("cuda")
    model = AlpamayoR1.from_pretrained(args.model_path, dtype=torch.bfloat16).to(device)
    processor = helper.get_processor(model.tokenizer)
    return model, processor


def calc_minADE(gt_future_xyz, pred_xyz):
    gt_xy = gt_future_xyz.cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    min_ade_idx = diff.argmin()
    return min_ade, min_ade_idx

def create_sliding_window_inputs(
    processor,
    num_windows: int,
    clip_id: str,
    t0_us: int = 2_000_000,
    time_step_us: int = 100_000,
):
    """
    Create sliding window inputs for streaming inference.

    For streaming:
    - Window 0 (prefill): 4 cameras × 4 frames = 16 frames
    - Window 1+: 4 cameras × 1 new frame = 4 frames (to be appended)

    Args:
        num_windows: Total number of windows (1 prefill + (num_windows-1) streaming steps)
        clip_id: Clip ID to load data from
        t0_us: Initial timestamp in microseconds
        time_step_us: Time step between frames (100_000 us = 0.1s)

    Returns:
        List of (model_inputs, is_prefill) tuples
    """
    streaming_inputs = []

    for window_idx in range(num_windows):
        # Calculate t0 for this window
        # Window 0: t0_us (frames at t0-0.3s, t0-0.2s, t0-0.1s, t0)
        # Window 1: t0_us + 100_000 (need frame at t0+0.1s)
        # Window 2: t0_us + 200_000 (need frame at t0+0.2s)
        current_t0 = t0_us + window_idx * time_step_us

        # Prefill: load full 4 frames per camera
        data = load_physical_aiavdataset(clip_id, t0_us=current_t0, num_frames=4)
        frames = data["image_frames"].flatten(0, 1)  # (4, 4, C, H, W) -> (16, C, H, W)
        is_prefill = True

        messages = helper.create_message(frames)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
            "ego_future_xyz": data["ego_future_xyz"],
            "ego_future_rot": data["ego_future_rot"],
            "is_prefill": is_prefill,
            "timestamp": current_t0,
        }
        streaming_inputs.append(model_inputs)

    return streaming_inputs


@torch.inference_mode()
def run_inference(args, model, sliding_window_inputs, device=None):
    if device is None:
        device = torch.device("cuda")
    clip_data_list = []

    for i in range(len(sliding_window_inputs)):
        model_inputs = sliding_window_inputs[i]
        with torch.autocast(device.type, dtype=torch.bfloat16):
            pred_xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=helper.to_device(model_inputs, device),
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=6,
                max_generation_length=256,
                return_extra=True
            )
        all_cots = extra["cot"][0][0].tolist()
        min_ade, min_ade_idx = calc_minADE(model_inputs["ego_future_xyz"], pred_xyz)
        clip_data_list.append({
            "timestamp": model_inputs["timestamp"],
            "all_cots": all_cots,
            "min_ade": float(min_ade),
            "best_cot": all_cots[min_ade_idx],
        })

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.clip_id}.json")
    with open(out_path, "w") as f:
        json.dump(clip_data_list, f, indent=2)
    logger.info("Results saved to %s", out_path)


def load_sliding_window_inputs_from_dump(dumped_inputs_dir: str, clip_id: str):
    """Load pre-dumped sliding_window_inputs from dumped_inputs_dir/<clip_id>/sliding_window_inputs.pt."""
    path = os.path.join(dumped_inputs_dir, clip_id, "sliding_window_inputs.pt")
    if not os.path.isfile(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


def dump_data(args, model, processor, device=None):
    if device is None:
        device = torch.device("cuda")
    clip_id = args.clip_id
    sliding_window_inputs = None
    if getattr(args, "dumped_inputs_dir", None):
        sliding_window_inputs = load_sliding_window_inputs_from_dump(args.dumped_inputs_dir, clip_id)
        if sliding_window_inputs is not None:
            logger.info("Loaded %s windows from dump for clip_id: %s", len(sliding_window_inputs), clip_id)
    if sliding_window_inputs is None:
        logger.info("Creating sliding window inputs for clip_id: %s", clip_id)
        sliding_window_inputs = create_sliding_window_inputs(
            processor=processor,
            num_windows=args.num_steps,
            clip_id=clip_id,
            t0_us=args.t0_us,
            time_step_us=args.time_step_us,
        )
        logger.info("Created %s windows", len(sliding_window_inputs))
    run_inference(args, model, sliding_window_inputs, device=device)
    logger.info("Completed compile inference for %s", clip_id)


def setup_distributed():
    """Initialize torch.distributed. Returns (rank, world_size, local_rank, device) or (0, 1, 0, cuda:0)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        return rank, world_size, local_rank, device
    return 0, 1, 0, torch.device("cuda")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run compile-model inference on a clip or distributed over clips.")
    parser.add_argument("--model_path", type=str, default="./Alpamayo-R1-10B")
    parser.add_argument("--num_steps", type=int, default=120)
    parser.add_argument("--t0_us", type=int, default=1_700_000)
    parser.add_argument("--time_step_us", type=int, default=100_000)
    parser.add_argument("--output_dir", type=str, default="./test_results/sys_pruning_logs")
    parser.add_argument("--clip_id", type=str, default=None, help="Single clip ID (for non-distributed run).")
    parser.add_argument(
        "--clip_ids_file",
        type=str,
        default=None,
        help="JSON file with list of clip IDs; use with torchrun for distributed (each rank runs a subset).",
    )
    parser.add_argument(
        "--dumped_inputs_dir",
        type=str,
        default=None,
        help="If set, load sliding_window_inputs from this dir (output of dump_sliding_window_inputs.py) instead of creating from dataset.",
    )
    args = parser.parse_args()

    rank, world_size, local_rank, device = setup_distributed()

    if args.clip_ids_file:
        # Distributed: each rank processes clip_ids[rank], clip_ids[rank+world_size], ...
        with open(args.clip_ids_file) as f:
            all_clip_ids = json.load(f)
        my_clip_ids = all_clip_ids[rank::world_size]
        logger.info(
            "Rank %s/%s: processing %s clips (total %s)",
            rank,
            world_size,
            len(my_clip_ids),
            len(all_clip_ids),
        )
        model, processor = load_model(args, device=device)
        for i, clip_id in enumerate(my_clip_ids):
            args.clip_id = clip_id
            logger.info("Rank %s: clip %s/%s %s", rank, i + 1, len(my_clip_ids), clip_id)
            dump_data(args, model, processor, device=device)
        if dist.is_initialized():
            dist.destroy_process_group()
    elif args.clip_id:
        # Single-clip run
        model, processor = load_model(args, device=device)
        dump_data(args, model, processor, device=device)
    else:
        raise SystemExit("Provide either --clip_id (single) or --clip_ids_file (distributed with torchrun).")