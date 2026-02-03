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

import torch
import numpy as np

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper


def create_sliding_window_inputs(
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


@torch.inference_mode()
def run_inference(model, processor, sliding_window_inputs):
    for step_idx, model_inputs in enumerate(sliding_window_inputs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=helper.to_device(model_inputs, "cuda"),
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,  # Feel free to raise this for more output trajectories and CoC traces.
                max_generation_length=256,
                return_extra=True,
            )

        # the size is [batch_size, num_traj_sets, num_traj_samples]
        print(f"\n=== Step {step_idx} ===")
        print("Chain-of-Causation:\n", extra["cot"][0])

        gt_xy = model_inputs["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
        pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
        diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
        min_ade = diff.min()
        print(f"MinADE: {min_ade}")


def test_inference():
    # Example clip ID
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    print(f"Loading dataset for clip_id: {clip_id}...")

    model = AlpamayoR1.from_pretrained("./Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    sliding_window_inputs = create_sliding_window_inputs(processor, 10, clip_id)
    run_inference(model, processor, sliding_window_inputs)


if __name__ == "__main__":
    test_inference()