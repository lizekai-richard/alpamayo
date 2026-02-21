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
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _is_rank0():
    """True only on rank 0 (or when not in distributed)."""
    return int(os.environ.get("RANK", 0)) == 0


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
def run_inference(args, model, tokenizer, sliding_window_inputs, device=None, save_path=None):
    if device is None:
        device = torch.device("cuda")

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
        best_cot = all_cots[min_ade_idx]

        if _is_rank0():
            logger.info(f"MinADE: {min_ade}, Best cot: {best_cot}")

        # Tokenize best_cot and store separately (concatenation deferred to dataset/forward)
        cot_token_ids = tokenizer.encode(best_cot, add_special_tokens=False)
        cot_end_id = tokenizer.encode("<|cot_end|>", add_special_tokens=False)
        eos_id = tokenizer.encode("<|endoftext|>", add_special_tokens=False)
        cot_tokens = torch.tensor(cot_token_ids + cot_end_id + eos_id, dtype=torch.long)
        sliding_window_inputs[i]["output_token_ids"] = cot_tokens

    # Save sliding_window_inputs (with output_token_ids/cot) to save_path or output_dir/clip_id
    if save_path is not None:
        out_path = save_path
    else:
        out_dir = os.path.join(args.output_dir, args.clip_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "sliding_window_inputs.pt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(sliding_window_inputs, out_path)
    if _is_rank0():
        logger.info("Saved sliding_window_inputs (with cot) to %s", out_path)


def load_sliding_window_inputs_from_dump(dumped_inputs_dir: str, clip_id: str):
    """Load pre-dumped sliding_window_inputs from dumped_inputs_dir/<clip_id>/sliding_window_inputs.pt.
    Returns None if file missing or load fails (e.g. corrupt/truncated on MooseFS).
    """
    path = os.path.join(dumped_inputs_dir, clip_id, "sliding_window_inputs.pt")
    if not os.path.isfile(path):
        return None
    try:
        # weights_only=False: dump contains BatchFeature (tokenized_data), trusted source
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        rank = int(os.environ.get("RANK", 0))
        logger.warning("Rank %s: Failed to load %s: %s (skip clip, deleting)", rank, path, e)
        clip_dir = os.path.join(dumped_inputs_dir, clip_id)
        # Safety: only delete if clip_dir is strictly inside dumped_inputs_dir (never the root)
        try:
            real_dump = os.path.realpath(dumped_inputs_dir)
            real_clip = os.path.realpath(clip_dir)
            if real_clip != real_dump and real_clip.startswith(real_dump + os.sep) and os.path.isdir(clip_dir):
                shutil.rmtree(clip_dir)
        except OSError as err:
            logger.warning("Rank %s: Failed to delete %s: %s", rank, clip_dir, err)
        return None


def discover_clip_ids_from_dump(dumped_inputs_dir: str):
    """Discover clip IDs by listing subdirs that contain sliding_window_inputs.pt."""
    if not os.path.isdir(dumped_inputs_dir):
        return []
    out = []
    for name in os.listdir(dumped_inputs_dir):
        subdir = os.path.join(dumped_inputs_dir, name)
        if os.path.isdir(subdir) and os.path.isfile(os.path.join(subdir, "sliding_window_inputs.pt")):
            out.append(name)
    return sorted(out)


def has_complete_output_token_ids(sliding_window_inputs):
    """Check if all windows have non-empty output_token_ids.
    
    Returns True if all windows have 'output_token_ids' key and each tensor has length > 0.
    Returns False otherwise.
    """
    if not sliding_window_inputs:
        return False
    for window_input in sliding_window_inputs:
        if "output_token_ids" not in window_input:
            return False
        output_tokens = window_input["output_token_ids"]
        # Check if it's a tensor and has non-zero length
        if not isinstance(output_tokens, torch.Tensor):
            return False
        if output_tokens.numel() == 0:
            return False
    return True


def dump_data(args, model, processor, device=None):
    """Process and dump data for a clip. Returns True if successfully dumped, False otherwise."""
    if device is None:
        device = torch.device("cuda")
    clip_id = args.clip_id
    sliding_window_inputs = None
    loaded_from_dump = False
    if getattr(args, "dumped_inputs_dir", None):
        sliding_window_inputs = load_sliding_window_inputs_from_dump(args.dumped_inputs_dir, clip_id)
        if sliding_window_inputs is not None:
            if _is_rank0():
                logger.info("Loaded %s windows from dump for clip_id: %s", len(sliding_window_inputs), clip_id)
            loaded_from_dump = True
            # Check if already has complete output_token_ids
            if has_complete_output_token_ids(sliding_window_inputs):
                if _is_rank0():
                    logger.info("Clip %s already has complete output_token_ids, skipping inference", clip_id)
                return True  # Already complete, count as successful
        elif sliding_window_inputs is None:
            # Load failed (missing or corrupt): already logged in load_sliding_window_inputs_from_dump
            return False
    if sliding_window_inputs is None:
        if _is_rank0():
            logger.info("Creating sliding window inputs for clip_id: %s", clip_id)
        sliding_window_inputs = create_sliding_window_inputs(
            processor=processor,
            num_windows=args.num_steps,
            clip_id=clip_id,
            t0_us=args.t0_us,
            time_step_us=args.time_step_us,
        )
        if _is_rank0():
            logger.info("Created %s windows", len(sliding_window_inputs))
    # Default: when loaded from dump, save back to same path (overwrite with cot added)
    save_path = (
        os.path.join(args.dumped_inputs_dir, clip_id, "sliding_window_inputs.pt")
        if loaded_from_dump
        else None
    )
    try:
        run_inference(args, model, model.tokenizer, sliding_window_inputs, device=device, save_path=save_path)
        if _is_rank0():
            logger.info("Completed compile inference for %s", clip_id)
        return True
    except Exception as e:
        if _is_rank0():
            logger.error("Failed to dump clip %s: %s", clip_id, e)
        return False


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
        help="JSON file with list of clip IDs (optional). If omitted, clip IDs are discovered from --dumped_inputs_dir.",
    )
    parser.add_argument(
        "--dumped_inputs_dir",
        type=str,
        default="/mnt/moosefs-1/users/zekail/dumped_inputs",
        help="Dir to load sliding_window_inputs from (output of dump_sliding_window_inputs.py). Default: MooseFS path.",
    )
    args = parser.parse_args()

    rank, world_size, local_rank, device = setup_distributed()

    if args.clip_id:
        # Single-clip run
        model, processor = load_model(args, device=device)
        success = dump_data(args, model, processor, device=device)
        if _is_rank0():
            if success:
                logger.info("Successfully dumped 1 clip")
            else:
                logger.info("Failed to dump clip")
    else:
        # Distributed: need list of clip IDs from file or from dumped_inputs_dir
        if args.clip_ids_file:
            with open(args.clip_ids_file) as f:
                all_clip_ids = json.load(f)
        else:
            all_clip_ids = discover_clip_ids_from_dump(args.dumped_inputs_dir)
            if not all_clip_ids:
                raise SystemExit(
                    f"No clips found in {args.dumped_inputs_dir} "
                    "(expected subdirs with sliding_window_inputs.pt). "
                    "Or provide --clip_ids_file or --clip_id."
                )
            logger.info("Discovered %s clips from %s", len(all_clip_ids), args.dumped_inputs_dir)
        my_clip_ids = all_clip_ids[rank::world_size]
        if _is_rank0():
            logger.info(
                "Rank %s/%s: processing %s clips (total %s)",
                rank,
                world_size,
                len(my_clip_ids),
                len(all_clip_ids),
            )
        model, processor = load_model(args, device=device)
        successful_dumps = 0
        for clip_id in tqdm(my_clip_ids, desc="Clips", disable=not _is_rank0()):
            args.clip_id = clip_id
            if dump_data(args, model, processor, device=device):
                successful_dumps += 1
        if _is_rank0():
            logger.info("Rank %s: Successfully dumped %s/%s clips", rank, successful_dumps, len(my_clip_ids))
        if dist.is_initialized():
            # Aggregate counts across all ranks
            count_tensor = torch.tensor(successful_dumps, dtype=torch.int32, device=device)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            if _is_rank0():
                logger.info("Total successfully dumped across all ranks: %s/%s clips", count_tensor.item(), len(all_clip_ids))
            dist.destroy_process_group()