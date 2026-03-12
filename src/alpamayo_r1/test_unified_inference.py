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
import os
import json
import argparse
import logging
import torch
import time
import numpy as np
from alpamayo_r1.models.alpamayo_r1_unified import AlpamayoR1FlashDrive
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper
from alpamayo_r1.utils import setup_dflash_for_model
from alpamayo_r1.utils import load_paroquant_model, convert_model_to_marlin_w4a8
from alpamayo_r1.utils import fuse_expert_projections

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable TF32 matmul for better perf on Ampere+ GPUs.
torch.set_float32_matmul_precision("high")


def create_streaming_inputs(
    processor,
    num_windows: int,
    clip_id: str,
    t0_us: int = 5_100_000,
    time_step_us: int = 100_000,  # 0.1s = 100_000 us
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

        if window_idx == 0:
            # Prefill: load full 4 frames per camera
            data = load_physical_aiavdataset(clip_id, t0_us=current_t0, num_frames=4)
            frames = data["image_frames"].flatten(0, 1)  # (4, 4, C, H, W) -> (16, C, H, W)
            is_prefill = True
        else:
            # Streaming: load only the newest frame per camera
            # num_frames=1 means only load the frame at current_t0
            data = load_physical_aiavdataset(clip_id, t0_us=current_t0, num_frames=1)
            frames = data["image_frames"].flatten(0, 1)  # (4, 1, C, H, W) -> (4, C, H, W)
            is_prefill = False

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
        }
        # model_inputs = helper.to_device(model_inputs, "cuda")
        streaming_inputs.append(model_inputs)

    return streaming_inputs


def create_non_streaming_inputs(
    processor,
    num_windows: int,
    clip_id: str,
    t0_us: int = 5_100_000,
    time_step_us: int = 100_000,  # 0.1s = 100_000 us
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
    non_streaming_inputs = []

    for window_idx in range(num_windows):
        # Calculate t0 for this window
        # Window 0: t0_us (frames at t0-0.3s, t0-0.2s, t0-0.1s, t0)
        # Window 1: t0_us + 100_000 (need frame at t0+0.1s)
        # Window 2: t0_us + 200_000 (need frame at t0+0.2s)
        current_t0 = t0_us + window_idx * time_step_us

        data = load_physical_aiavdataset(clip_id, t0_us=current_t0, num_frames=4)
        frames = data["image_frames"].flatten(0, 1)  # (4, 4, C, H, W) -> (16, C, H, W)

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
        }
        # model_inputs = helper.to_device(model_inputs, "cuda")
        non_streaming_inputs.append(model_inputs)

    return non_streaming_inputs


def calc_minADE(gt_future_xyz, pred_xyz):
    gt_xy = gt_future_xyz.cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    min_ade_idx = diff.argmin()
    return min_ade, min_ade_idx


@torch.inference_mode()
def run_streaming_inference(args, model, model_inputs, _logging: bool = True):
    """
    Run streaming inference with sliding window inputs.

    The model internally checks `past_key_values is None` to determine
    if this is the first prefill or a streaming step.

    Args:
        streaming_inputs: List of model_inputs dicts, each containing:
            - tokenized_data: processor output (input_ids, pixel_values, etc.)
            - ego_history_xyz: history trajectory positions
            - ego_history_rot: history trajectory rotations
    """
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_flashdrive(
            data=helper.to_device(model_inputs, "cuda"),
            streaming=True,
            dflash=not args.disable_dflash,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=args.num_traj_samples,
            max_new_tokens=128,
            return_extra=True,
            diffusion_kwargs={
                "inference_step": args.diffusion_steps,
                "cache_steps": args.cache_steps,
                "int_method": "euler" if args.cache_steps is None else "euler_with_cache"
            },
        )
        if pred_xyz is not None:
            min_ade, min_ade_idx = calc_minADE(model_inputs["ego_future_xyz"], pred_xyz)
            cot = extra["cot"][0][0][min_ade_idx]
    
    if _logging:
        logger.info("Chain-of-Causation:\n%s", cot)
        logger.info(f"MinADE: {min_ade}")
    
    if pred_xyz is not None:
        return float(min_ade), cot
    else:
        return float('inf'), None


@torch.inference_mode()
def run_non_streaming_inference(args, model, model_inputs, _logging: bool = True):
    """
    Run non-streaming inference with sliding window inputs.
    """
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_flashdrive(
            data=helper.to_device(model_inputs, "cuda"),
            streaming=False,
            dflash=not args.disable_dflash,
            top_p=0.98,
            temperature=0.6,
            max_new_tokens=128,
            num_traj_samples=args.num_traj_samples,
            num_traj_sets=1,
            return_extra=True,
            diffusion_kwargs={
                "inference_step": args.diffusion_steps,
                "cache_steps": args.cache_steps,
                "int_method": "euler" if args.cache_steps is None else "euler_with_cache"
            },
        )
        min_ade, min_ade_idx = calc_minADE(model_inputs["ego_future_xyz"], pred_xyz)
        cot = extra["cot"][0][0][min_ade_idx]

    if _logging:
        logger.info("Chain-of-Causation:\n%s", cot)
        logger.info(f"MinADE: {min_ade}")
    
    return float(min_ade), cot


def warmup_model(args, model, processor, all_inputs):
    """Warmup the model."""
    warmup_steps = args.warmup_steps
    for i in range(warmup_steps):
        model_inputs = all_inputs[i]
        if not args.disable_streaming:
            run_streaming_inference(args, model, model_inputs, _logging=False)
        else:
            run_non_streaming_inference(args, model, model_inputs, _logging=False)
    logger.info("Warmup completed")


def create_inputs(args, clip_id, processor, streaming=False):
    """Load dumped inputs or create them from scratch."""
    num_steps = args.num_steps
    if streaming:
        return create_streaming_inputs(
            processor=processor,
            num_windows=num_steps,
            clip_id=clip_id,
            t0_us=args.t0_us,
            time_step_us=args.time_step_us,
        )
    else:
        return create_non_streaming_inputs(
            processor=processor,
            num_windows=num_steps,
            clip_id=clip_id,
            t0_us=args.t0_us,
            time_step_us=args.time_step_us,
        )


def flashdrive_inference(args, model, processor):
    if not args.disable_streaming:
        all_inputs = create_inputs(
            args=args,
            clip_id=args.clip_id,
            processor=processor,
            streaming=True,
        )
    else:
        all_inputs = create_inputs(
            args=args,
            clip_id=args.clip_id,
            processor=processor,
            streaming=False,
        )
    
    warmup_model(args, model, processor, all_inputs)

    minADE_list = []
    for i, model_inputs in enumerate(all_inputs):
        if not args.disable_streaming:
            minADE, _ = run_streaming_inference(args, model, model_inputs, _logging=True)
            minADE_list.append(minADE)
        else:
            minADE, _ = run_non_streaming_inference(args, model, model_inputs, _logging=True)
            minADE_list.append(minADE)
        logger.info(f"Step {i+1}/{len(all_inputs)} completed")
    logger.info(f"Average MinADE: {np.mean(minADE_list)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data/scratch/zekaili/train_expert_ckpts_deepspeed/checkpoint-6446")
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--num_steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_traj_samples", type=int, default=1)
    parser.add_argument("--clip_id", type=str, default="87147a1b-3eef-4c25-94d2-ec7718a49a7a")
    parser.add_argument("--t0_us", type=int, default=1_700_000)
    parser.add_argument("--time_step_us", type=int, default=100_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--draft_model_path", type=str, default="/data/scratch/zekaili/Alpamayo-DFlash")
    parser.add_argument("--quantized_model_path", type=str, default="/data/scratch/zekaili/quant_cache/ckpt-paro-w4-vlm-mm.pt")
    parser.add_argument("--disable_compile", action="store_true")
    parser.add_argument("--disable_streaming", action="store_true")
    parser.add_argument("--disable_dflash", action="store_true")
    parser.add_argument("--disable_w4a8", action="store_true")
    parser.add_argument("--torch_compile", type=str, default="max-autotune")
    parser.add_argument("--diffusion_steps", type=int, default=8)
    parser.add_argument("--cache_steps", type=int, nargs="*", default=[3, 4, 5, 6])
    
    args = parser.parse_args()
    if args.disable_w4a8:
        model = AlpamayoR1FlashDrive.from_pretrained(args.model_path, dtype=torch.bfloat16)
        model = model.to(args.device)
        model.eval()
        processor = helper.get_processor(model.tokenizer)
    else:
        model = load_paroquant_model(
            model_path=args.model_path,
            paro_checkpoint=args.quantized_model_path,
            mode="streaming" if not args.disable_streaming else "non-streaming",
            device=args.device,
        )
        convert_model_to_marlin_w4a8(model)
        processor = helper.get_processor(model.tokenizer)
        fuse_expert_projections(model)

    if not args.disable_dflash:
        setup_dflash_for_model(model, args.draft_model_path)

    flashdrive_inference(args, model, processor)