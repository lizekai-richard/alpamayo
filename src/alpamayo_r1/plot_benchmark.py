# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot benchmark comparison between original and optimized models.

Usage:
    python -m alpamayo_r1.plot_benchmark
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_comparison():
    # Latency values in ms: [Encode, Prefill, Decode, Action, Total]
    phases = ["Encode", "Prefill", "Decode", "Action", "Total"]
    original_vals = [90.1, 165.6, 290.8, 222.7, 769.2]      # Alpamayo-R1
    optimized_vals = [12.7, 60.2, 84.3, 25.3, 182.5]        # FlashDriveVLA

    y = np.arange(len(phases))
    height = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.barh(y + height/2, original_vals, height, label="Alpamayo1", alpha=0.7)
    bars2 = ax.barh(y - height/2, optimized_vals, height, label="FlashDriveVLA", alpha=0.7)

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{width:.1f}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0),
                        textcoords="offset points",
                        ha='left', va='center', fontsize=12)

    add_labels(bars1)
    add_labels(bars2)

    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.set_xlabel("Latency (ms)", fontsize=14)
    ax.set_yticks(y)
    ax.set_yticklabels(phases, fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_xlim(0, max(max(original_vals), max(optimized_vals)) * 1.2)

    plt.tight_layout()
    plt.savefig("benchmark_results/latency_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig("benchmark_results/latency_comparison.pdf", bbox_inches='tight')
    print("Saved to benchmark_results/latency_comparison.png and .pdf")
    plt.show()


def plot_minade():
    # minADE values (n_samples=6)
    models = ["Alpamayo1", "FlashDriveVLA"]
    values = [0.798, 0.924]  # TODO: Replace with actual minADE values

    x = np.arange(len(models)) * 0.6
    width = 0.2

    fig, ax = plt.subplots(figsize=(4, 5))
    bars = ax.bar(x, values, width, color=["#1f77b4", "#ff7f0e"], alpha=0.7)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)

    ax.set_ylabel(r"$\mathrm{minADE}_{6}$ @ 6.4s (m)", fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_yticks([0, 1.0])
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=14)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("benchmark_results/minade_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig("benchmark_results/minade_comparison.pdf", bbox_inches='tight')
    print("Saved to benchmark_results/minade_comparison.png and .pdf")
    plt.show()


if __name__ == "__main__":
    plot_comparison()
    plot_minade()