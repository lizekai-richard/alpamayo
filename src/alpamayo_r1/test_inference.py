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

import argparse
import json
import logging
import os
import time

import numpy as np
import torch

from alpamayo_r1.models.alpamayo_r1_compile import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def load_model(args):
    model = AlpamayoR1.from_pretrained(args.model_path, dtype=torch.bfloat16).to("cuda")
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
        }
        # model_inputs = helper.to_device(model_inputs, "cuda")
        streaming_inputs.append(model_inputs)

    return streaming_inputs


@torch.inference_mode()
def run_inference(args, model, processor, sliding_window_inputs):
    warmup_steps = args.warmup_steps
    logger.info("Warming up model...")
    for i in range(warmup_steps):
        model_inputs = sliding_window_inputs[i]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.sample_trajectories_from_data_with_vlm_rollout(
                data=helper.to_device(model_inputs, "cuda"),
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=args.num_traj_samples,
                max_generation_length=256,
                return_extra=True,
                torch_compile="max-autotune",
                fuse_qkv=True,
                fuse_gate_up=True,
                sparsity_ratio=args.sparsity_ratio,
            )
    logger.info("Warmup completed")

    time_list = []
    min_ade_list = []
    cot_list = []
    for i in range(warmup_steps, len(sliding_window_inputs)):
        model_inputs = sliding_window_inputs[i]
        start_time = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=helper.to_device(model_inputs, "cuda"),
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=args.num_traj_samples,
                max_generation_length=256,
                torch_compile="max-autotune",
                return_extra=True,
                fuse_qkv=True,
                fuse_gate_up=True,
                sparsity_ratio=args.sparsity_ratio,
            )
        end_time = time.perf_counter()
        min_ade, min_ade_idx = calc_minADE(model_inputs["ego_future_xyz"], pred_xyz)
        cot = extra["cot"][0][0][min_ade_idx]
        time_list.append(end_time - start_time)
        min_ade_list.append(float(min_ade))
        cot_list.append(cot)
        logger.info("Step %s: latency=%.3fs, MinADE=%.4f", i, end_time - start_time, min_ade)
        logger.info("Chain-of-Causation:\n%s", cot)

    logger.info(
        "Total time: %.2fs, avg latency: %.3fs, avg MinADE: %.4f",
        sum(time_list),
        sum(time_list) / len(time_list),
        sum(min_ade_list) / len(min_ade_list),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    for j in range(len(min_ade_list)):
        results[j] = {
            "latency": time_list[j],
            "min_ade": min_ade_list[j],
            "cot": cot_list[j],
        }
    out_path = os.path.join(args.output_dir, f"compile_{args.clip_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)


def test_inference(args, model, processor):
    logger.info("Creating sliding window inputs for clip_id: %s", args.clip_id)
    sliding_window_inputs = create_sliding_window_inputs(
        processor=processor,
        num_windows=args.num_steps,
        clip_id=args.clip_id,
        t0_us=args.t0_us,
        time_step_us=args.time_step_us,
    )
    logger.info("Created %s windows", len(sliding_window_inputs))
    run_inference(args, model, processor, sliding_window_inputs)
    logger.info("Completed compile inference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run compile-model inference on a clip.")
    parser.add_argument("--model_path", type=str, default="./Alpamayo-R1-10B")
    parser.add_argument("--clip-id", "--clip_id", dest="clip_id", type=str, default="b80a15fc-d540-4c8f-81d1-5db83216b2e0")
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--num_steps", type=int, default=103)
    parser.add_argument("--t0_us", type=int, default=2_000_000)
    parser.add_argument("--time_step_us", type=int, default=100_000)
    parser.add_argument("--output_dir", type=str, default="./test_results/sys_pruning_logs")
    parser.add_argument("--num_traj_samples", type=int, default=6)
    parser.add_argument("--sparsity_ratio", type=float, default=0.5)
    args = parser.parse_args()

    model, processor = load_model(args)
    test_inference(args, model, processor)