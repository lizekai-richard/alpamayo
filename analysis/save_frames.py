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

import logging
import os

import PIL.Image as Image

from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def save_frames(
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
        data = load_physical_aiavdataset(clip_id, t0_us=current_t0, num_frames=1)
        frames = data["image_frames"]  # (4, C, H, W)
        
        os.makedirs(f"frames/{clip_id}/{current_t0}", exist_ok=True)
        for i, frame in enumerate(frames):
            pil_frame = Image.fromarray(frame[0].permute(1, 2, 0).numpy())
            pil_frame = pil_frame.resize((480, 320))
            pil_frame.save(f"frames/{clip_id}/{current_t0}/view{i}.png")


if __name__ == "__main__":
    save_frames(num_windows=120, clip_id="87147a1b-3eef-4c25-94d2-ec7718a49a7a", t0_us=1_700_000, time_step_us=100_000)