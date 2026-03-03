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
from alpamayo_r1.models.alpamayo_r1_all import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper
from alpamayo_r1.dflash_integration import setup_dflash_for_model

def _get_rank() -> int:
    for key in ("WORKER_RANK", "RANK", "LOCAL_RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


RANK = _get_rank()
logging.basicConfig(
    level=logging.INFO if RANK == 0 else logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,
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
        streaming_inputs.append(model_inputs)

    return streaming_inputs


def calc_minADE(gt_future_xy, pred_xyz):
    gt_xy = gt_future_xy.cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    min_ade_idx = diff.argmin()
    return min_ade, min_ade_idx


def load_or_create_streaming_inputs(args, processor):
    """Load dumped inputs or create them from scratch."""
    if args.dumped_data_dir:
        logger.info("Loading dumped inputs from %s for clip %s", args.dumped_data_dir, args.clip_id)
        windows = helper.load_dumped_inputs(args.dumped_data_dir, args.clip_id)
        windows = windows[:args.num_steps]
        vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        streaming_inputs = [windows[0]]
        for w in windows[1:]:
            streaming_inputs.append(helper.convert_to_streaming_window(w, vision_start_id, vision_end_id))
        return streaming_inputs
    else:
        return create_streaming_inputs(
            processor=processor,
            num_windows=args.num_steps,
            clip_id=args.clip_id,
            t0_us=args.t0_us,
            time_step_us=args.time_step_us,
        )


@torch.inference_mode()
def test_dflash_streaming_inference(args, model, processor):
    """Test DFlash speculative decoding in streaming mode (integrated path)."""
    setup_dflash_for_model(model, args.draft_model)

    streaming_inputs = load_or_create_streaming_inputs(args, processor)
    logger.info(f"Loaded {len(streaming_inputs)} streaming inputs")
    min_ade_list = []
    cot_list = []
    time_list = []

    warmup_steps = args.warmup_steps
    logger.info("Warming up DFlash streaming...")
    for i in range(warmup_steps):
        start_time = time.perf_counter()
        model_inputs = streaming_inputs[i]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_with_dflash_streaming_vlm_rollout(
                data=helper.to_device(model_inputs, "cuda"),
                torch_compile="max-autotune",
                num_traj_samples=args.num_traj_samples,
                max_generation_length=256,
                return_extra=True,
                fuse_qkv=True,
                fuse_gate_up=True,
                diffusion_kwargs={"cache_steps": args.cache_steps, "int_method": "euler_with_cache"},
            )
        end_time = time.perf_counter()
        if i > 0:
            min_ade, min_ade_idx = calc_minADE(model_inputs["ego_future_xyz"], pred_xyz)
            min_ade_list.append(float(min_ade))
            cot_list.append(extra["cot"][0][0][min_ade_idx])
            time_list.append(end_time - start_time)
            logger.info("Step %s: latency=%.3fs, MinADE=%.4f", i, end_time - start_time, min_ade)
            logger.info("Chain-of-Causation:\n%s", extra["cot"][0][0][min_ade_idx])
    logger.info("Warmup completed")

    logger.info(f"Running DFlash streaming inference for {len(streaming_inputs) - warmup_steps} windows:")

    for i in range(warmup_steps, len(streaming_inputs)):
        model_inputs = streaming_inputs[i]
        start_time = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_with_dflash_streaming_vlm_rollout(
                data=helper.to_device(model_inputs, "cuda"),
                torch_compile="max-autotune",
                num_traj_samples=args.num_traj_samples,
                max_generation_length=256,
                return_extra=True,
                fuse_qkv=True,
                fuse_gate_up=True,
                diffusion_kwargs={"cache_steps": args.cache_steps, "int_method": "euler_with_cache"}
            )
        end_time = time.perf_counter()

        min_ade, min_ade_idx = calc_minADE(model_inputs["ego_future_xyz"], pred_xyz)
        min_ade_list.append(float(min_ade))
        cot_list.append(extra["cot"][0][0][min_ade_idx])
        time_list.append(end_time - start_time)
        logger.info("Step %s: latency=%.3fs, MinADE=%.4f", i, end_time - start_time, min_ade)
        logger.info("Chain-of-Causation:\n%s", extra["cot"][0][0][min_ade_idx])

    logger.info(
        "Total time: %.2fs, avg latency: %.3fs, avg MinADE: %.4f",
        sum(time_list[1:]),
        sum(time_list[1:]) / len(time_list[1:]),
        sum(min_ade_list[1:]) / len(min_ade_list[1:]),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./Alpamayo-R1-10B")
    parser.add_argument("--draft-model", type=str, default="./Alpamayo-DFlash")
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--num_steps", type=int, default=120)
    parser.add_argument("--clip-id", type=str, default="599eb73d-373a-4c86-afc4-020005056a6c")
    parser.add_argument("--t0_us", type=int, default=1_700_000)
    parser.add_argument("--time_step_us", type=int, default=100_000)
    parser.add_argument("--output_dir", type=str, default="./test_results/6samples_all")
    parser.add_argument("--num_traj_samples", type=int, default=6)
    parser.add_argument("--cache_steps", type=list, default=[])
    parser.add_argument("--dumped_data_dir", type=str, default="./dumped_eval_data")

    args = parser.parse_args()
    model, processor = load_model(args)
    test_dflash_streaming_inference(args, model, processor)
