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
from alpamayo_r1.models.alpamayo_r1_unified import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable TF32 matmul for better perf on Ampere+ GPUs.
torch.set_float32_matmul_precision("high")


def load_model(args):
    model = AlpamayoR1.from_pretrained(args.model_path, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    return model, processor

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


def calc_minADE(gt_future_xy, pred_xyz):
    gt_xy = gt_future_xy.cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    return min_ade


@torch.inference_mode()
def run_streaming_inference(model, model_inputs, _logging: bool = True):
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
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_streaming_vlm_rollout(
            data=helper.to_device(model_inputs, "cuda"),
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )
        min_ade = calc_minADE(model_inputs["ego_future_xyz"], pred_xyz)
    
    if _logging:
        logger.info("Chain-of-Causation:\n%s", extra["cot"][0])
        logger.info(f"MinADE: {min_ade}")
    
    return float(min_ade), extra["cot"][0]


@torch.inference_mode()
def run_non_streaming_inference(model, model_inputs, _logging: bool = True):
    """
    Run non-streaming inference with sliding window inputs.
    """
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=helper.to_device(model_inputs, "cuda"),
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )
        min_ade = calc_minADE(model_inputs["ego_future_xyz"], pred_xyz)

    if _logging:
        logger.info("Chain-of-Causation:\n%s", extra["cot"][0])
        logger.info(f"MinADE: {min_ade}")
    
    return float(min_ade), extra["cot"][0]

@torch.inference_mode()
def test_non_streaming_inference(args, model, processor):
    """Test non-streaming inference."""
    warmup_steps = args.warmup_steps
    non_streaming_inputs = create_non_streaming_inputs(
        processor=processor,
        num_windows=args.num_steps,
        clip_id=args.clip_id,
        t0_us=args.t0_us,
        time_step_us=args.time_step_us,
    )

    logger.info("Warming up model...")
    for i in range(warmup_steps):
        model_inputs = non_streaming_inputs[i]
        run_non_streaming_inference(model, model_inputs, _logging=False)
    logger.info("Warmup completed")

    logger.info(f"Running non-streaming inference for {len(non_streaming_inputs)} windows:")
    time_list = []
    min_ade_list = []
    cot_list = []
    for i in range(warmup_steps, len(non_streaming_inputs)):
        model_inputs = non_streaming_inputs[i]
        start_time = time.perf_counter()
        min_ade, cot = run_non_streaming_inference(model, model_inputs, _logging=True)
        min_ade_list.append(min_ade)
        cot_list.append(cot)
        end_time = time.perf_counter()
        time_list.append(end_time - start_time)
        logger.info(f"Time taken for step {i}: {end_time - start_time} seconds")
    logger.info(f"Total time taken: {sum(time_list)} seconds")
    logger.info(f"Average time per step: {sum(time_list) / len(time_list)} seconds")
    logger.info(f"\nCompleted non-streaming inference")

    results = {}
    for i in range(len(time_list)):
        results[i] = {
            "latency": time_list[i],
            "min_ade": min_ade_list[i],
            "cot": cot_list[i],
        }
    with open(os.path.join(args.output_dir, f"non_streaming_results.json"), "w") as f:
        json.dump(results, f)


@torch.inference_mode()
def test_streaming_inference(args, model, processor):
    """Test streaming inference."""
    warmup_steps = args.warmup_steps
    streaming_inputs = create_streaming_inputs(
        processor=processor,
        num_windows=args.num_steps,
        clip_id=args.clip_id,
        t0_us=args.t0_us,
        time_step_us=args.time_step_us,
    )
        
    logger.info("Warming up model...")
    # warmup the model
    for i in range(warmup_steps):
        model_inputs = streaming_inputs[i]
        run_streaming_inference(model, model_inputs, _logging=False)
    logger.info("Warmup completed")

    logger.info(f"Running streaming inference for {len(streaming_inputs)} windows:")
    time_list = []
    min_ade_list = []
    cot_list = []
    for i in range(warmup_steps, len(streaming_inputs)):
        model_inputs = streaming_inputs[i]
        start_time = time.perf_counter()
        min_ade, cot = run_streaming_inference(model, model_inputs, _logging=True)
        min_ade_list.append(min_ade)
        cot_list.append(cot)
        end_time = time.perf_counter()
        time_list.append(end_time - start_time)
        logger.info(f"Time taken for step {i}: {end_time - start_time} seconds")
    logger.info(f"Total time taken: {sum(time_list)} seconds")
    logger.info(f"Average time per step: {sum(time_list) / len(time_list)} seconds")
    logger.info(f"\nCompleted streaming inference")

    results = {}
    for i in range(len(time_list)):
        results[i] = {
            "latency": time_list[i],
            "min_ade": min_ade_list[i],
            "cot": cot_list[i],
        }
    with open(os.path.join(args.output_dir, f"streaming_results.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./Alpamayo-R1-10B")
    parser.add_argument("--mode", type=str, default="streaming", choices=["streaming", "non_streaming"])
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--num_steps", type=int, default=15)
    parser.add_argument("--clip_id", type=str, default="030c760c-ae38-49aa-9ad8-f5650a545d26")
    parser.add_argument("--t0_us", type=int, default=5_100_000)
    parser.add_argument("--time_step_us", type=int, default=100_000)
    parser.add_argument("--output_dir", type=str, default="./test_results")

    args = parser.parse_args()
    model, processor = load_model(args)
    if args.mode == "streaming":
        test_streaming_inference(args, model, processor)
    elif args.mode == "non_streaming":
        test_non_streaming_inference(args, model, processor)