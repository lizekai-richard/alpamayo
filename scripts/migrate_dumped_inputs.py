# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Migrate v1 dumped sliding_window_inputs.pt to v1.5 format.

v1 format: images are concatenated without any camera label tokens.
v1.5 format: each camera's first frame is preceded by a camera display name,
             and every frame is preceded by a "frame N " token.

Only input_ids and attention_mask are modified; pixel_values and
image_grid_thw are untouched.

Default cameras (from load_physical_aiavdataset defaults, sorted by index):
  [0, 1, 2, 6] → Front left / Front / Front right / Front telephoto
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import AutoTokenizer

CAMERA_DISPLAY_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    4: "Rear camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}

# Default camera_indices used during dump (sorted order from load_physical_aiavdataset)
DEFAULT_CAMERA_INDICES = [0, 1, 2, 6]
NUM_FRAMES_PER_CAMERA = 4


def upgrade_window(
    window: dict,
    tokenizer: AutoTokenizer,
    camera_indices: list[int],
    vs_id: int,
    ve_id: int,
) -> dict:
    """Insert camera-name and frame-index tokens into a single window's input_ids."""
    tokenized = window["tokenized_data"]
    input_ids = tokenized["input_ids"][0]  # [seq_len]

    vs_pos = (input_ids == vs_id).nonzero(as_tuple=True)[0].tolist()
    ve_pos = (input_ids == ve_id).nonzero(as_tuple=True)[0].tolist()

    n_cameras = len(camera_indices)
    n_frames = NUM_FRAMES_PER_CAMERA
    expected_images = n_cameras * n_frames
    assert len(vs_pos) == expected_images, (
        f"Expected {expected_images} image blocks, got {len(vs_pos)}"
    )

    prefix = input_ids[: vs_pos[0]].tolist()
    suffix = input_ids[ve_pos[-1] + 1 :].tolist()

    new_ids: list[int] = list(prefix)
    for cam_i, cam_id in enumerate(camera_indices):
        cam_name_ids = tokenizer.encode(
            f"{CAMERA_DISPLAY_NAMES[cam_id]}: ", add_special_tokens=False
        )
        for frame_i in range(n_frames):
            frame_ids = tokenizer.encode(f"frame {frame_i} ", add_special_tokens=False)
            if frame_i == 0:
                new_ids += cam_name_ids
            new_ids += frame_ids
            img_idx = cam_i * n_frames + frame_i
            new_ids += input_ids[vs_pos[img_idx] : ve_pos[img_idx] + 1].tolist()
    new_ids += suffix

    new_input_ids = torch.tensor(new_ids, dtype=input_ids.dtype).unsqueeze(0)
    new_attention_mask = torch.ones_like(new_input_ids)

    new_tokenized = dict(tokenized)
    new_tokenized["input_ids"] = new_input_ids
    new_tokenized["attention_mask"] = new_attention_mask

    result = {
        "tokenized_data": new_tokenized,
        "ego_history_xyz": window["ego_history_xyz"],
        "ego_history_rot": window["ego_history_rot"],
        "ego_future_xyz": window["ego_future_xyz"],
        "ego_future_rot": window["ego_future_rot"],
        "is_prefill": window["is_prefill"],
        "timestamp": window["timestamp"],
    }
    if "output_token_ids" in window:
        result["output_token_ids"] = window["output_token_ids"]
    return result


def migrate_clip(
    src_path: str,
    dst_path: str,
    tokenizer: AutoTokenizer,
    camera_indices: list[int],
    vs_id: int,
    ve_id: int,
) -> None:
    windows = torch.load(src_path, weights_only=False)
    upgraded = [
        upgrade_window(w, tokenizer, camera_indices, vs_id, ve_id) for w in windows
    ]
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    torch.save(upgraded, dst_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate v1 → v1.5 dumped inputs")
    parser.add_argument(
        "--src_dir",
        default="/mnt/moosefs-1/users/zekail/dumped_eval_data",
        help="Source directory containing per-clip subdirectories",
    )
    parser.add_argument(
        "--dst_dir",
        default="/mnt/moosefs-1/users/zekail/dumped_eval_data_v1.5",
        help="Destination directory for migrated data",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Tokenizer name or path",
    )
    parser.add_argument(
        "--clip_id",
        default=None,
        help="Process a single clip ID (default: all clips)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    args = parser.parse_args()

    assert os.path.realpath(args.src_dir) != os.path.realpath(args.dst_dir), (
        "src_dir and dst_dir must be different — refusing to overwrite source data"
    )

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    vs_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    ve_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    if args.clip_id:
        clip_ids = [args.clip_id]
    else:
        clip_ids = sorted(os.listdir(args.src_dir))

    # Filter to only clips that need processing
    tasks = []
    skipped = 0
    for clip_id in clip_ids:
        src_path = os.path.join(args.src_dir, clip_id, "sliding_window_inputs.pt")
        dst_path = os.path.join(args.dst_dir, clip_id, "sliding_window_inputs.pt")
        if not os.path.exists(src_path) or os.path.exists(dst_path):
            skipped += 1
            continue
        tasks.append((src_path, dst_path, clip_id))

    print(
        f"Processing {len(tasks)} clip(s) with {args.workers} worker(s) "
        f"(skipped {skipped}): {args.src_dir} → {args.dst_dir}"
    )

    ok, failed = 0, 0
    if args.workers <= 1:
        for i, (src_path, dst_path, clip_id) in enumerate(tasks):
            try:
                migrate_clip(src_path, dst_path, tokenizer, DEFAULT_CAMERA_INDICES, vs_id, ve_id)
                ok += 1
                if ok % 50 == 0:
                    print(f"  [{ok}/{len(tasks)}] OK={ok} FAIL={failed}", flush=True)
            except Exception as e:
                failed += 1
                print(f"  FAIL {clip_id}: {e}", flush=True)
    else:
        def _process(item):
            src_path, dst_path, clip_id = item
            migrate_clip(src_path, dst_path, tokenizer, DEFAULT_CAMERA_INDICES, vs_id, ve_id)
            return clip_id

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process, t): t[2] for t in tasks}
            for future in as_completed(futures):
                clip_id = futures[future]
                try:
                    future.result()
                    ok += 1
                    if ok % 50 == 0:
                        print(f"  OK={ok} FAIL={failed} / {len(tasks)}", flush=True)
                except Exception as e:
                    failed += 1
                    print(f"  FAIL {clip_id}: {e}", flush=True)

    print(f"Done. OK={ok} SKIP={skipped} FAIL={failed}")


if __name__ == "__main__":
    main()
