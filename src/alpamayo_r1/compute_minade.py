# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compute average minADE across all clips in a results directory.

Each JSON file represents one clip. Supports files with prefix "streaming"
or "compile" (e.g. streaming_*.json, compile_*.json). The per-clip minADE
is the mean of all step-level min_ade values in that file. The overall
minADE is the mean across all per-clip minADE values.

Usage:
    python -m alpamayo_r1.compute_minade [results_dir]

    results_dir defaults to "test_results" if not provided.
"""

import glob
import json
import os
import sys


def compute_avg_minade(results_dir):
    json_files = sorted(
        glob.glob(os.path.join(results_dir, "streaming_*.json"))
        + glob.glob(os.path.join(results_dir, "compile_*.json"))
    )
    if not json_files:
        print(f"No streaming_*.json or compile_*.json files found in {results_dir}")
        return None, []

    per_clip = []
    for fpath in json_files:
        clip_id = (
            os.path.basename(fpath)
            .replace("streaming_", "")
            .replace("compile_", "")
            .replace(".json", "")
        )
        with open(fpath) as f:
            data = json.load(f)

        ade_values = [v["min_ade"] for v in data.values() if "min_ade" in v]
        if ade_values:
            clip_avg = sum(ade_values) / len(ade_values)
            per_clip.append((clip_id, len(ade_values), clip_avg))

    if not per_clip:
        print("No min_ade data found.")
        return None, []

    overall_avg = sum(v for _, _, v in per_clip) / len(per_clip)

    # Print results
    print(f"{'Clip ID':<40s} {'Steps':>6s} {'Avg minADE':>12s}")
    print("-" * 60)
    for clip_id, n_steps, avg in per_clip:
        print(f"{clip_id:<40s} {n_steps:>6d} {avg:>12.4f}")
    total_steps = sum(n for _, n, _ in per_clip)
    print("-" * 60)
    print(f"{'Overall Average':<40s} {'':>6s} {overall_avg:>12.4f}")
    print(f"Total clips: {len(per_clip)}, Total steps: {total_steps}")

    return overall_avg, per_clip


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "test_results"
    compute_avg_minade(results_dir)
