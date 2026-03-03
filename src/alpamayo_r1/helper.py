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

from transformers import AutoProcessor, AutoTokenizer

from typing import Any

import os
import torch
import collections.abc

MIN_PIXELS = 163840
MAX_PIXELS = 196608
BASE_PROCESSOR_NAME = "Qwen/Qwen3-VL-2B-Instruct"


def create_message(frames: torch.Tensor):
    """Construct the message using images and cot."""
    assert frames.ndim == 4, f"{frames.ndim=}, expected (N, C, H, W)"

    # NOTE: we expand the padding tokens to match training, so we can directly apply native processor from VLM.
    num_traj_token = 48
    hist_traj_placeholder = (
        f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
    )

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a driving assistant that generates safe and accurate actions.",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "image", "image": frame} for frame in frames]
            + [
                {
                    "type": "text",
                    "text": f"{hist_traj_placeholder}output the chain-of-thought reasoning of the driving process, then output the future trajectory.",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "<|cot_start|>",
                }
            ],
        },
    ]


def get_processor(tokenizer: AutoTokenizer) -> AutoProcessor:
    """Get the processor for the Qwen3-VL-2B-Instruct model."""
    processor_kwargs = {
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
    }

    processor = AutoProcessor.from_pretrained(BASE_PROCESSOR_NAME, **processor_kwargs)
    processor.tokenizer = tokenizer
    return processor


def load_dumped_inputs(dumped_data_dir: str, clip_id: str) -> list[dict]:
    """Load pre-dumped sliding window inputs from disk.

    Args:
        dumped_data_dir: Root directory containing dumped data.
        clip_id: Clip ID to load.

    Returns:
        List of window dicts, each containing tokenized_data, ego trajectories, etc.
    """
    path = os.path.join(dumped_data_dir, clip_id, "sliding_window_inputs.pt")
    return torch.load(path, weights_only=False)


def convert_to_streaming_window(
    window: dict,
    vision_start_id: int,
    vision_end_id: int,
) -> dict:
    """Convert a 16-frame prefill window to 4-frame streaming format.

    Extracts the last frame per camera (indices 3, 7, 11, 15) from the
    16-frame tokenized data and sets is_prefill=False.

    Args:
        window: A window dict with 16 images in tokenized_data.
        vision_start_id: Token ID for <|vision_start|>.
        vision_end_id: Token ID for <|vision_end|>.

    Returns:
        New window dict with only 4 images and is_prefill=False.
    """
    tokenized = window["tokenized_data"]
    input_ids = tokenized["input_ids"][0]  # [seq_len]
    pixel_values = tokenized["pixel_values"]  # [total_patches, hidden]
    image_grid_thw = tokenized["image_grid_thw"]  # [16, 3]

    # Find all image block boundaries
    vs_positions = (input_ids == vision_start_id).nonzero(as_tuple=True)[0].tolist()
    ve_positions = (input_ids == vision_end_id).nonzero(as_tuple=True)[0].tolist()
    assert len(vs_positions) == 16 and len(ve_positions) == 16

    # Keep last frame per camera: indices [3, 7, 11, 15]
    keep_indices = [3, 7, 11, 15]

    # Build new input_ids: prefix + kept image blocks + suffix
    prefix = input_ids[:vs_positions[0]]
    suffix = input_ids[ve_positions[-1] + 1:]
    kept_blocks = [input_ids[vs_positions[i]:ve_positions[i] + 1] for i in keep_indices]
    new_input_ids = torch.cat([prefix] + kept_blocks + [suffix]).unsqueeze(0)

    # Extract pixel_values for kept images
    patches_per_image = [int(image_grid_thw[i].prod()) for i in range(16)]
    patch_cumsum = [0]
    for p in patches_per_image:
        patch_cumsum.append(patch_cumsum[-1] + p)
    kept_pixel_values = torch.cat([
        pixel_values[patch_cumsum[i]:patch_cumsum[i + 1]] for i in keep_indices
    ])

    # Extract image_grid_thw for kept images
    kept_grid = image_grid_thw[keep_indices]

    new_tokenized = {
        "input_ids": new_input_ids,
        "attention_mask": torch.ones_like(new_input_ids),
        "pixel_values": kept_pixel_values,
        "image_grid_thw": kept_grid,
    }

    return {
        "tokenized_data": new_tokenized,
        "ego_history_xyz": window["ego_history_xyz"],
        "ego_history_rot": window["ego_history_rot"],
        "ego_future_xyz": window["ego_future_xyz"],
        "ego_future_rot": window["ego_future_rot"],
        "is_prefill": False,
    }


def to_device(
    data: Any,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    """Recursively cast data into the specified device, dtype."""
    if isinstance(data, torch.Tensor):
        data = data.to(
            device=device,
            dtype=dtype,
        )
        return data
    elif isinstance(data, collections.abc.Mapping):
        return {key: to_device(data[key], device=device, dtype=dtype) for key in data}
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        return [to_device(elem, device=device, dtype=dtype) for elem in data]
    else:
        return data
