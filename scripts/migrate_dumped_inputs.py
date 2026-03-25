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
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from alpamayo_r1.helper import upgrade_window_to_v1p5


_worker_tokenizer = None


def _init_worker(tokenizer_name: str) -> None:
    """Initializer for each worker process — loads its own tokenizer."""
    global _worker_tokenizer
    _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)


def migrate_clip(
    src_path: str,
    dst_path: str,
    tokenizer: AutoTokenizer | None = None,
) -> None:
    if tokenizer is None:
        tokenizer = _worker_tokenizer
    windows = torch.load(src_path, weights_only=False)
    upgraded = [upgrade_window_to_v1p5(w, tokenizer) for w in windows]
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    torch.save(upgraded, dst_path)


def _process_item(item):
    """Top-level function so ProcessPoolExecutor can pickle it."""
    src_path, dst_path, _clip_id = item
    migrate_clip(src_path, dst_path)
    return _clip_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate v1 → v1.5 dumped inputs")
    parser.add_argument(
        "--src_dir",
        default="/mnt/moosefs/users/zekail/dumped_inputs/",
        help="Source directory containing per-clip subdirectories",
    )
    parser.add_argument(
        "--dst_dir",
        default="/mnt/moosefs-1/users/zekail/dumped_inputs_v1_5/",
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
        if not os.path.exists(src_path):
            skipped += 1
            continue
        tasks.append((src_path, dst_path, clip_id))

    print(
        f"Processing {len(tasks)} clip(s) with {args.workers} worker(s) "
        f"(skipped {skipped}): {args.src_dir} → {args.dst_dir}"
    )

    ok, failed = 0, 0
    if args.workers <= 1:
        for src_path, dst_path, clip_id in tasks:
            try:
                migrate_clip(src_path, dst_path, tokenizer)
                ok += 1
                if ok % 50 == 0:
                    print(f"  [{ok}/{len(tasks)}] OK={ok} FAIL={failed}", flush=True)
            except Exception as e:
                failed += 1
                print(f"  FAIL {clip_id}: {e}", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args.tokenizer,),
        ) as pool:
            futures = {pool.submit(_process_item, t): t[2] for t in tasks}
            for future in as_completed(futures):
                clip_id = futures[future]
                try:
                    future.result()
                    ok += 1
                    if ok % 10 == 0:
                        print(f"  OK={ok} FAIL={failed} / {len(tasks)}", flush=True)
                except Exception as e:
                    failed += 1
                    print(f"  FAIL {clip_id}: {e}", flush=True)

    print(f"Done. OK={ok} SKIP={skipped} FAIL={failed}")


if __name__ == "__main__":
    main()
