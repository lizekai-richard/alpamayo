# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot benchmark comparison between original and optimized models.

Usage:
    python -m alpamayo_r1.plot_benchmark
"""

import json
import matplotlib.pyplot as plt
import numpy as np


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_comparison():
    # Load results
    original = load_results("benchmark_results/alpamayo_n1.json")
    optimized = load_results("benchmark_results/alpamayo_sysopt_streaming_n1.json")

    # Extract breakdown data
    # For decode, we need to convert s/token to total seconds using avg token count

    # Use fixed 20 tokens for fair comparison
    num_decode_tokens = 20

    # Get mean values (convert to ms)
    orig_encode = original["breakdown"]["encode_s"]["mean"] * 1000
    orig_prefill = original["breakdown"]["prefill_s"]["mean"] * 1000
    orig_decode = original["breakdown"]["decode_s_per_token"]["mean"] * num_decode_tokens * 1000
    orig_action = original["breakdown"]["action_s"]["mean"] * 1000

    opt_encode = optimized["breakdown"]["encode_s"]["mean"] * 1000
    opt_prefill = optimized["breakdown"]["prefill_s"]["mean"] * 1000
    opt_decode = optimized["breakdown"]["decode_s_per_token"]["mean"] * num_decode_tokens * 1000
    opt_action = optimized["breakdown"]["action_s"]["mean"] * 1000

    # Data for plotting
    phases = ["Encode", "Prefill", "Decode", "Action", "Total"]
    original_vals = [orig_encode, orig_prefill, orig_decode, orig_action,
                     original["total_inference_s"]["mean"] * 1000]
    optimized_vals = [opt_encode, opt_prefill, opt_decode, opt_action,
                      optimized["total_inference_s"]["mean"] * 1000]

    x = np.arange(len(phases))
    width = 0.3

    fig, ax = plt.subplots(figsize=(6, 6))
    bars1 = ax.bar(x - width/2, original_vals, width, label="Alpamayo-R1")
    bars2 = ax.bar(x + width/2, optimized_vals, width, label="FlashDriveVLA")

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    add_labels(bars1)
    add_labels(bars2)

    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(loc="upper left", fontsize=12)
    ax.set_ylim(0, max(max(original_vals), max(optimized_vals)) * 1.15)

    plt.tight_layout()
    plt.savefig("benchmark_results/latency_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig("benchmark_results/latency_comparison.pdf", bbox_inches='tight')
    print("Saved to benchmark_results/latency_comparison.png and .pdf")
    plt.show()


if __name__ == "__main__":
    plot_comparison()
