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

import logging
logger = logging.getLogger(__name__)

MIN_PIXELS = 163840
MAX_PIXELS = 196608
BASE_PROCESSOR_NAME = "Qwen/Qwen3-VL-2B-Instruct"

CAMERA_DISPLAY_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    4: "Rear camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}


def _build_image_content(
    frames: torch.Tensor,
    camera_indices: torch.Tensor | None = None,
    num_frames_per_camera: int = 4,
) -> list[dict[str, Any]]:
    """Build the image portion of the user message content.

    When ``camera_indices`` is provided, each image is annotated with
    a camera display name (on the first frame of each camera group) and
    a frame index, matching the format used during model training.

    Args:
        frames: Flattened camera frames, shape ``(N_total, C, H, W)``.
        camera_indices: Per-camera indices, shape ``(N_cameras,)``.
            When provided, ``N_total`` must equal
            ``N_cameras * num_frames_per_camera``.
        num_frames_per_camera: Number of temporal frames per camera.
    """
    if camera_indices is None:
        return [{"type": "image", "image": frame} for frame in frames]

    expanded_cam_ids = camera_indices.repeat_interleave(num_frames_per_camera)
    content: list[dict[str, Any]] = []
    prev_cam_id = None
    frame_idx = 0
    for i, frame in enumerate(frames):
        cam_id = expanded_cam_ids[i].item()
        if prev_cam_id is not None and cam_id != prev_cam_id:
            frame_idx = 0
        if frame_idx == 0:
            cam_name = CAMERA_DISPLAY_NAMES.get(cam_id, f"Camera {cam_id}")
            content.append({"type": "text", "text": f"{cam_name}: "})
        content.append({"type": "text", "text": f"frame {frame_idx} "})
        content.append({"type": "image", "image": frame})
        prev_cam_id = cam_id
        frame_idx += 1
    return content


def create_message(
    frames: torch.Tensor,
    camera_indices: torch.Tensor | None = None,
    num_frames_per_camera: int = 4,
    nav_text: str | None = None,
    use_nav_prompt: bool = False,
):
    """Construct the chat message for model inference.

    Args:
        frames: Camera image tensors, shape ``(N_total, C, H, W)``
            (typically ``data["image_frames"].flatten(0, 1)``).
        camera_indices: Per-camera indices from the dataset, shape
            ``(N_cameras,)``. When provided, camera display names and
            frame numbers are included before each image to match the
            training format. Pass ``data["camera_indices"]``.
        num_frames_per_camera: Number of temporal frames per camera
            (default 4, matching the dataset loader).
        nav_text: Optional navigation instruction string, e.g.
            ``"Turn left onto De La Cruz Boulevard in 40m"``.
            When provided, the model conditions its trajectory prediction
            on this instruction.
        use_nav_prompt: When ``True``, use the nav-style prompt
            (``"output the future trajectory."``) even if ``nav_text``
            is ``None``. Useful for constructing a fair no-nav baseline
            in nav-conditioned comparisons.
    """
    assert frames.ndim == 4, f"{frames.ndim=}, expected (N, C, H, W)"

    num_traj_token = 48
    hist_traj_placeholder = (
        f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
    )

    route_section = ""
    if nav_text is not None:
        route_section = f"<|route_start|>{nav_text}<|route_end|>"

    prompt_text = "output the chain-of-thought reasoning of the driving process, then output the future trajectory."

    user_text = f"{hist_traj_placeholder}{route_section}{prompt_text}"

    image_content = _build_image_content(frames, camera_indices, num_frames_per_camera)

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
            "content": image_content + [{"type": "text", "text": user_text}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<|cot_start|>"}],
        },
    ]


def create_vqa_message(
    frames: torch.Tensor,
    question: str,
    camera_indices: torch.Tensor | None = None,
    num_frames_per_camera: int = 4,
):
    """Construct the chat message for model inference.

    Args:
        frames: Camera image tensors, shape ``(N_total, C, H, W)``
            (typically ``data["image_frames"].flatten(0, 1)``).
        question: The question string.
        camera_indices: Per-camera indices from the dataset, shape
            ``(N_cameras,)``. When provided, camera display names and
            frame numbers are included before each image to match the
            training format. Pass ``data["camera_indices"]``.
        num_frames_per_camera: Number of temporal frames per camera
            (default 4, matching the dataset loader).
    """
    assert frames.ndim == 4, f"{frames.ndim=}, expected (N, C, H, W)"
    user_text = f"<|question_start|>{question}<|question_end|>"

    image_content = _build_image_content(frames, camera_indices, num_frames_per_camera)

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
            "content": image_content + [{"type": "text", "text": user_text}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<|answer_start|>"}],
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
    # no prefix = input_ids[:vs_positions[0]]
    suffix = input_ids[ve_positions[-1] + 1:]
    kept_blocks = [input_ids[vs_positions[i]:ve_positions[i] + 1] for i in keep_indices]
    # new_input_ids = torch.cat([prefix] + kept_blocks + [suffix]).unsqueeze(0)
    new_input_ids = torch.cat(kept_blocks + [suffix]).unsqueeze(0)

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


def convert_to_streaming_window_v1p5(
    window: dict,
    vision_end_id: int,
    num_views: int = 4,
    num_frames_per_view: int = 4,
    keep_frame_labels: bool = True,
    vision_start_id: int | None = None,
) -> dict:
    """Convert a 16-frame prefill window to streaming format for v1.5.

    For each view, keeps the last frame's vision block plus (optionally)
    the frame label text before it. Also keeps everything after the final
    [VE] (traj tokens, user prompt).

    When keep_frame_labels=True (default): keeps from sec-last [VE]+1 to
    last [VE], which includes "frame 3 " + [VS]...[VE].
    When keep_frame_labels=False: keeps only [VS]...[VE] of the last frame.

    Pixel values and image_grid_thw are extracted for the last frame per view
    (indices 3, 7, 11, 15).

    Args:
        window: A window dict with 16 images in tokenized_data.
        vision_end_id: Token ID for <|vision_end|>.
        num_views: Number of camera views (default 4).
        num_frames_per_view: Frames per view (default 4).
        keep_frame_labels: If True, keep frame label text (e.g. "frame 3 ")
            before each last frame's [VS]. If False, keep only [VS]...[VE].
            Requires vision_start_id when False.
        vision_start_id: Token ID for <|vision_start|>. Required when
            keep_frame_labels=False.

    Returns:
        New window dict with streaming format and is_prefill=False.
    """
    tokenized = window["tokenized_data"]
    input_ids = tokenized["input_ids"][0]  # [seq_len]
    pixel_values = tokenized["pixel_values"]  # [total_patches, hidden]
    image_grid_thw = tokenized["image_grid_thw"]  # [16, 3]

    ve_positions = (input_ids == vision_end_id).nonzero(as_tuple=True)[0].tolist()
    assert len(ve_positions) == num_views * num_frames_per_view

    keep_mask = torch.zeros(input_ids.shape[0], dtype=torch.bool)
    if keep_frame_labels:
        # Keep from sec-last VE+1 to last VE (includes frame label + [VS]...[VE])
        for view_idx in range(num_views):
            sec_last_gidx = view_idx * num_frames_per_view + (num_frames_per_view - 2)
            last_gidx = view_idx * num_frames_per_view + (num_frames_per_view - 1)
            start = ve_positions[sec_last_gidx] + 1
            end = ve_positions[last_gidx]
            keep_mask[start : end + 1] = True
    else:
        # Keep only last frame's [VS]...[VE] per view (no frame labels)
        assert vision_start_id is not None, "vision_start_id required when keep_frame_labels=False"
        vs_positions = (input_ids == vision_start_id).nonzero(as_tuple=True)[0].tolist()
        for view_idx in range(num_views):
            last_gidx = view_idx * num_frames_per_view + (num_frames_per_view - 1)
            start = vs_positions[last_gidx]
            end = ve_positions[last_gidx]
            keep_mask[start : end + 1] = True

    # Keep everything after the last VE (traj tokens, user prompt, etc.)
    keep_mask[ve_positions[-1] + 1 :] = True

    new_input_ids = input_ids[keep_mask].unsqueeze(0)

    # Extract pixel_values for last frame per view (indices 3, 7, 11, 15)
    keep_indices = [
        view_idx * num_frames_per_view + (num_frames_per_view - 1)
        for view_idx in range(num_views)
    ]
    patches_per_image = [int(image_grid_thw[i].prod()) for i in range(len(image_grid_thw))]
    patch_cumsum = [0]
    for p in patches_per_image:
        patch_cumsum.append(patch_cumsum[-1] + p)
    kept_pixel_values = torch.cat([
        pixel_values[patch_cumsum[i]:patch_cumsum[i + 1]] for i in keep_indices
    ])
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


def _patch_sensor_presence_rename():
    """Monkey-patch physical_ai_av for dataset rename: sensor_presence.parquet -> feature_presence.parquet."""
    from physical_ai_av.utils import hf_interface
    _orig = hf_interface.HfRepoInterface.download_file

    def _patched(self, filename, **kwargs):
        if filename == "metadata/sensor_presence.parquet":
            filename = "metadata/feature_presence.parquet"
        return _orig(self, filename, **kwargs)

    hf_interface.HfRepoInterface.download_file = _patched


def create_avdi(cache_dir=None):
    """Create PhysicalAIAVDatasetInterface; apply sensor_presence->feature_presence workaround if needed."""
    import physical_ai_av
    try:
        return physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=cache_dir)
    except IndexError:
        logger.warning("Dataset renamed sensor_presence -> feature_presence; applying workaround...")
        _patch_sensor_presence_rename()
        return physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=cache_dir)


def _build_v1p5_label_cache(
    tokenizer: AutoTokenizer,
    camera_indices: list[int],
    num_frames_per_camera: int,
) -> tuple[list[list[list[int]]], int, int]:
    """Pre-compute all label token ids and special token ids once."""
    vs_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    ve_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    # labels[cam_i][frame_i] = list of token ids
    labels: list[list[list[int]]] = []
    for cam_id in camera_indices:
        cam_display = CAMERA_DISPLAY_NAMES.get(cam_id, f"Camera {cam_id}")
        cam_labels: list[list[int]] = []
        for frame_i in range(num_frames_per_camera):
            if frame_i == 0:
                ids = tokenizer.encode(
                    f"{cam_display}: frame {frame_i} ",
                    add_special_tokens=False,
                )
            else:
                ids = tokenizer.encode(
                    f"frame {frame_i} ", add_special_tokens=False,
                )
            cam_labels.append(ids)
        labels.append(cam_labels)
    return labels, vs_id, ve_id


# Module-level cache: tokenizer id(obj) → (labels, vs_id, ve_id)
_v1p5_label_cache: dict[int, tuple[list[list[list[int]]], int, int]] = {}


def upgrade_window_to_v1p5(
    window: dict,
    tokenizer: AutoTokenizer,
    camera_indices: list[int] | None = None,
    num_frames_per_camera: int = 4,
) -> dict:
    """Upgrade a v1 dumped window to v1.5 format in-place on input_ids.

    v1 format has images concatenated with no camera labels.
    v1.5 format inserts a camera display name before the first frame of each
    camera and a ``"frame N "`` prefix before every frame.

    ``pixel_values`` and ``image_grid_thw`` are unchanged.

    Args:
        window: A window dict as returned by ``load_dumped_inputs``.
        tokenizer: The model tokenizer (used to encode label strings).
        camera_indices: Ordered list of camera index integers, one per camera
            group. Defaults to ``[0, 1, 2, 6]`` (the four cameras used by
            ``load_physical_aiavdataset``'s default, sorted by index:
            cross-left / front-wide / cross-right / front-tele).
        num_frames_per_camera: Frames per camera in the dump (default 4).

    Returns:
        New window dict with upgraded ``tokenized_data``.
    """
    if camera_indices is None:
        camera_indices = [0, 1, 2, 6]

    cache_key = id(tokenizer)
    if cache_key not in _v1p5_label_cache:
        _v1p5_label_cache[cache_key] = _build_v1p5_label_cache(
            tokenizer, camera_indices, num_frames_per_camera,
        )
    labels, vs_id, ve_id = _v1p5_label_cache[cache_key]

    tokenized = window["tokenized_data"]
    input_ids = tokenized["input_ids"][0]  # [seq_len]

    vs_pos = (input_ids == vs_id).nonzero(as_tuple=True)[0].tolist()
    ve_pos = (input_ids == ve_id).nonzero(as_tuple=True)[0].tolist()

    expected = len(camera_indices) * num_frames_per_camera
    assert len(vs_pos) == expected, (
        f"Expected {expected} image blocks, got {len(vs_pos)}"
    )

    prefix = input_ids[: vs_pos[0]].tolist()
    suffix = input_ids[ve_pos[-1] + 1 :].tolist()

    new_ids: list[int] = list(prefix)
    for cam_i in range(len(camera_indices)):
        for frame_i in range(num_frames_per_camera):
            new_ids += labels[cam_i][frame_i]
            img_idx = cam_i * num_frames_per_camera + frame_i
            new_ids += input_ids[vs_pos[img_idx] : ve_pos[img_idx] + 1].tolist()
    new_ids += suffix

    new_input_ids = torch.tensor(new_ids, dtype=input_ids.dtype).unsqueeze(0)

    new_tokenized = dict(tokenized)
    new_tokenized["input_ids"] = new_input_ids
    new_tokenized["attention_mask"] = torch.ones_like(new_input_ids)

    return {
        "tokenized_data": new_tokenized,
        "ego_history_xyz": window["ego_history_xyz"],
        "ego_history_rot": window["ego_history_rot"],
        "ego_future_xyz": window["ego_future_xyz"],
        "ego_future_rot": window["ego_future_rot"],
        "is_prefill": window["is_prefill"],
        "timestamp": window["timestamp"],
    }
