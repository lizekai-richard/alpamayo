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

"""
Benchmark script to compare streaming vs original model performance.

Compares:
1. End-to-end inference time
2. Image encoding time (visual encoder forward pass)
3. VLM prefill time (language model prefill phase)

Tests multiple step counts (5, 10, 15, 20) for comprehensive comparison.

Key insights:
- Streaming model: First step (prefill) processes 16 frames, subsequent steps only 4 frames
- Original model: Every step processes 16 frames

The speedup from streaming comes from:
1. Image encoding: 4x fewer frames to encode in streaming steps (4 vs 16)
2. VLM prefill: Shorter sequence to prefill in streaming steps (~800 vs ~3000 tokens)

================================================================================
HOW TO RUN
================================================================================

1. Default configuration (test 5, 10, 15, 20 steps):

    cd /raid/home/zekail/alpamayo
    python -m alpamayo_r1.benchmark_streaming

2. Custom parameters:

    python -m alpamayo_r1.benchmark_streaming \\
        --model_path ./Alpamayo-R1-10B \\
        --clip_id 030c760c-ae38-49aa-9ad8-f5650a545d26 \\
        --steps 5 10 15 20 \\
        --warmup 2 \\
        --output_dir ./benchmark_results

3. Quick test (fewer steps):

    python -m alpamayo_r1.benchmark_streaming --steps 3 5

================================================================================
ARGUMENTS
================================================================================

--model_path    : Path to model checkpoint (default: ./Alpamayo-R1-10B)
--clip_id       : Clip ID for dataset (default: 030c760c-ae38-49aa-9ad8-f5650a545d26)
--steps         : List of step counts to benchmark (default: 5 10 15 20)
--warmup        : Number of warmup steps before timing (default: 2)
--output_dir    : Output directory for results (default: ./benchmark_results)

================================================================================
OUTPUT FILES
================================================================================

All files are saved to --output_dir with timestamp suffix:

1. benchmark_results_YYYYMMDD_HHMMSS.json
   - Complete benchmark data in JSON format
   - Contains all step-by-step results and summaries

2. benchmark_comparison_YYYYMMDD_HHMMSS.png
   - 2x3 grid comparing streaming vs original model
   - Shows: avg total time, avg encode time, avg prefill time
   - Bottom row: streaming steps only (excluding first prefill)

3. benchmark_cumulative_YYYYMMDD_HHMMSS.png
   - Cumulative time comparison over increasing steps
   - Shows total, image encode, and VLM prefill cumulative times

4. benchmark_speedup_YYYYMMDD_HHMMSS.png
   - Speedup ratios (original_time / streaming_time)
   - Shows speedup for total time, image encode, and VLM prefill

5. benchmark_breakdown_YYYYMMDD_HHMMSS.png
   - Stacked bar chart showing time breakdown by component
   - Left: streaming steps, Right: first step (full prefill)
   - Components: Image Encode, VLM Prefill, VLM Decode, Action/Diffusion

================================================================================
"""

import argparse
import copy
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

torch.set_float32_matmul_precision("high")

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimingHook:
    """Hook to time module forward passes."""

    def __init__(self, sync_cuda: bool = True):
        self.sync_cuda = sync_cuda
        self.timings = []
        self._start_time = None

    def __call__(self, module, input, output):
        if self.sync_cuda:
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - self._start_time) * 1000
        self.timings.append(elapsed)

    def pre_hook(self, module, input):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()

    def reset(self):
        self.timings = []
        self._start_time = None

    @property
    def total_time_ms(self):
        return sum(self.timings)

    @property
    def last_time_ms(self):
        return self.timings[-1] if self.timings else 0


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_type: str  # "streaming" or "original"
    num_steps: int
    step_idx: int

    # Timing results (in milliseconds)
    total_time_ms: float
    image_encode_time_ms: float
    vlm_prefill_time_ms: float
    vlm_decode_time_ms: float
    action_time_ms: float


@dataclass
class BenchmarkSummary:
    """Summary statistics for a benchmark run."""
    model_type: str
    num_steps: int

    # Average times (in milliseconds)
    avg_total_time_ms: float
    avg_image_encode_time_ms: float
    avg_vlm_prefill_time_ms: float
    avg_vlm_decode_time_ms: float
    avg_action_time_ms: float

    # First step (prefill) times
    first_total_time_ms: float
    first_image_encode_time_ms: float
    first_vlm_prefill_time_ms: float

    # Streaming step average times (excluding first step)
    streaming_avg_total_time_ms: float
    streaming_avg_image_encode_time_ms: float
    streaming_avg_vlm_prefill_time_ms: float


class TimingContext:
    """Context manager for timing code blocks."""
    def __init__(self, sync_cuda: bool = True):
        self.sync_cuda = sync_cuda
        self.elapsed_ms = 0.0

    def __enter__(self):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000


def create_sliding_window_inputs(
    num_windows: int,
    clip_id: str,
    processor,
    t0_us: int = 5_100_000,
    time_step_us: int = 100_000,
):
    """Create sliding window inputs for streaming inference."""
    streaming_inputs = []

    for window_idx in range(num_windows):
        current_t0 = t0_us + window_idx * time_step_us

        if window_idx == 0:
            data = load_physical_aiavdataset(clip_id, t0_us=current_t0, num_frames=4)
            frames = data["image_frames"].flatten(0, 1)
            is_prefill = True
        else:
            data = load_physical_aiavdataset(clip_id, t0_us=current_t0, num_frames=1)
            frames = data["image_frames"].flatten(0, 1)
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


def create_original_inputs(
    num_windows: int,
    clip_id: str,
    processor,
    t0_us: int = 5_100_000,
    time_step_us: int = 100_000,
):
    """Create inputs for original (non-streaming) inference."""
    original_inputs = []

    for window_idx in range(num_windows):
        current_t0 = t0_us + window_idx * time_step_us

        # Original model always processes all 4 frames per view
        data = load_physical_aiavdataset(clip_id, t0_us=current_t0, num_frames=4)
        frames = data["image_frames"].flatten(0, 1)

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
        }
        original_inputs.append(model_inputs)

    return original_inputs


class ModelBenchmarkBase:
    """Base class for model benchmarking with timing hooks."""

    def __init__(self):
        self.visual_hook = TimingHook()
        self.lm_hook = TimingHook()
        self._hooks = []

    def _register_hooks(self, visual_encoder: nn.Module, language_model: nn.Module):
        """Register timing hooks on visual encoder and language model."""
        # Visual encoder hook
        h1 = visual_encoder.register_forward_pre_hook(self.visual_hook.pre_hook)
        h2 = visual_encoder.register_forward_hook(self.visual_hook)
        self._hooks.extend([h1, h2])

        # Language model hook
        h3 = language_model.register_forward_pre_hook(self.lm_hook.pre_hook)
        h4 = language_model.register_forward_hook(self.lm_hook)
        self._hooks.extend([h3, h4])

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _reset_hooks(self):
        """Reset timing data in hooks."""
        self.visual_hook.reset()
        self.lm_hook.reset()


class StreamingModelBenchmark(ModelBenchmarkBase):
    """Benchmark wrapper for streaming model with detailed timing."""

    def __init__(self, model_path: str, dtype=torch.bfloat16):
        super().__init__()
        from alpamayo_r1.models.alpamayo_r1_unified import AlpamayoR1

        logger.info("Loading unified model (streaming mode)...")
        self.model = AlpamayoR1.from_pretrained(model_path, dtype=dtype).to("cuda")
        self.processor = helper.get_processor(self.model.tokenizer)
        logger.info("Model loaded.")

        # Register timing hooks
        self._register_hooks(
            self.model.vlm.model.visual,
            self.model.vlm.model.language_model,
        )

    def reset(self):
        """Reset model state for new benchmark run."""
        self.model.reset_streaming_state()
        self.model._past_key_values = None

    @torch.inference_mode()
    def run_step_with_timing(self, model_inputs: dict) -> BenchmarkResult:
        """Run one inference step with detailed timing."""
        is_prefill = model_inputs.pop("is_prefill", self.model.is_first_prefill)
        step_idx = 0 if is_prefill else -1  # Will be set by caller

        # Prepare data
        data = helper.to_device(model_inputs, "cuda")

        # Reset hooks for this step
        self._reset_hooks()

        total_timer = TimingContext()

        with total_timer:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = self.model.sample_trajectories(
                    data=data,
                    streaming=True,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1,
                    max_generation_length=256,
                    return_extra=True,
                )

        # Extract timing from hooks
        # Visual encoder is called once per step
        image_encode_time = self.visual_hook.total_time_ms

        # Language model is called multiple times:
        # - 1 prefill call
        # - N decode calls (one per generated token)
        # - M action/diffusion calls
        # For simplicity, we report total LM time and let caller analyze
        lm_total_time = self.lm_hook.total_time_ms

        # Estimate: First LM call is prefill, rest are decode/action
        lm_timings = self.lm_hook.timings
        vlm_prefill_time = lm_timings[0] if lm_timings else 0
        # Decode time = sum of all calls except first and last few (action calls)
        # Action typically has 20 calls (diffusion steps)
        num_action_calls = 20  # Estimate based on diffusion steps
        if len(lm_timings) > 1 + num_action_calls:
            vlm_decode_time = sum(lm_timings[1:-num_action_calls])
            action_time = sum(lm_timings[-num_action_calls:])
        else:
            vlm_decode_time = sum(lm_timings[1:]) if len(lm_timings) > 1 else 0
            action_time = 0

        return BenchmarkResult(
            model_type="streaming",
            num_steps=-1,  # Set by caller
            step_idx=step_idx,
            total_time_ms=total_timer.elapsed_ms,
            image_encode_time_ms=image_encode_time,
            vlm_prefill_time_ms=vlm_prefill_time,
            vlm_decode_time_ms=vlm_decode_time,
            action_time_ms=action_time,
        )


class OriginalModelBenchmark(ModelBenchmarkBase):
    """Benchmark wrapper for original (non-streaming) model with detailed timing."""

    def __init__(self, model_path: str, dtype=torch.bfloat16, use_compile: bool = False):
        super().__init__()
        from alpamayo_r1.models.alpamayo_r1_unified import AlpamayoR1

        logger.info("Loading unified model (non-streaming mode)...")
        self.model = AlpamayoR1.from_pretrained(model_path, dtype=dtype).to("cuda")
        self.processor = helper.get_processor(self.model.tokenizer)
        self.use_compile = use_compile
        logger.info("Model loaded.")

        # Register timing hooks
        self._register_hooks(
            self.model.vlm.model.visual,
            self.model.vlm.model.language_model,
        )

    @torch.inference_mode()
    def run_step_with_timing(self, model_inputs: dict) -> BenchmarkResult:
        """Run one inference step with detailed timing."""
        data = helper.to_device(model_inputs, "cuda")

        # Reset hooks for this step
        self._reset_hooks()

        total_timer = TimingContext()

        torch_compile = "max-autotune" if self.use_compile else None
        with total_timer:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = self.model.sample_trajectories(
                    data=data,
                    streaming=False,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1,
                    max_generation_length=256,
                    torch_compile=torch_compile,
                    return_extra=True,
                )

        # Extract timing from hooks
        image_encode_time = self.visual_hook.total_time_ms
        lm_total_time = self.lm_hook.total_time_ms

        lm_timings = self.lm_hook.timings
        vlm_prefill_time = lm_timings[0] if lm_timings else 0

        # Estimate action calls (diffusion steps)
        num_action_calls = 20
        if len(lm_timings) > 1 + num_action_calls:
            vlm_decode_time = sum(lm_timings[1:-num_action_calls])
            action_time = sum(lm_timings[-num_action_calls:])
        else:
            vlm_decode_time = sum(lm_timings[1:]) if len(lm_timings) > 1 else 0
            action_time = 0

        return BenchmarkResult(
            model_type="original",
            num_steps=-1,
            step_idx=-1,
            total_time_ms=total_timer.elapsed_ms,
            image_encode_time_ms=image_encode_time,
            vlm_prefill_time_ms=vlm_prefill_time,
            vlm_decode_time_ms=vlm_decode_time,
            action_time_ms=action_time,
        )


def run_benchmark(
    model_path: str,
    clip_id: str,
    step_counts: list[int],
    warmup_steps: int = 2,
    output_dir: str = "./benchmark_results",
) -> dict[str, list[BenchmarkSummary]]:
    """Run complete benchmark comparison."""

    os.makedirs(output_dir, exist_ok=True)

    results = {
        "streaming": [],
        "original": [],
    }
    all_step_results = {
        "streaming": [],
        "original": [],
    }

    # Load models
    streaming_bench = StreamingModelBenchmark(model_path)
    original_bench = OriginalModelBenchmark(model_path, use_compile=False)

    for num_steps in step_counts:
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking with {num_steps} steps")
        logger.info(f"{'='*60}")

        # Create inputs
        streaming_inputs = create_sliding_window_inputs(
            num_windows=num_steps + warmup_steps,
            clip_id=clip_id,
            processor=streaming_bench.processor,
        )
        original_inputs = create_original_inputs(
            num_windows=num_steps + warmup_steps,
            clip_id=clip_id,
            processor=original_bench.processor,
        )

        # Benchmark streaming model
        logger.info(f"\nBenchmarking streaming model ({num_steps} steps)...")
        streaming_bench.reset()
        streaming_step_results = []

        for step_idx, inputs in enumerate(streaming_inputs):
            # Deep copy to avoid modifying original inputs
            inputs_copy = copy.deepcopy(inputs)
            result = streaming_bench.run_step_with_timing(inputs_copy)
            result.step_idx = step_idx
            result.num_steps = num_steps

            # Skip warmup steps
            if step_idx >= warmup_steps:
                streaming_step_results.append(result)
                logger.info(f"  Step {step_idx - warmup_steps}: total={result.total_time_ms:.2f}ms, "
                          f"encode={result.image_encode_time_ms:.2f}ms, prefill={result.vlm_prefill_time_ms:.2f}ms")

        all_step_results["streaming"].extend(streaming_step_results)

        # Benchmark original model
        logger.info(f"\nBenchmarking original model ({num_steps} steps)...")
        original_step_results = []

        for step_idx, inputs in enumerate(original_inputs):
            # Deep copy to avoid modifying original inputs
            inputs_copy = copy.deepcopy(inputs)
            result = original_bench.run_step_with_timing(inputs_copy)
            result.step_idx = step_idx
            result.num_steps = num_steps

            # Skip warmup steps
            if step_idx >= warmup_steps:
                original_step_results.append(result)
                logger.info(f"  Step {step_idx - warmup_steps}: total={result.total_time_ms:.2f}ms, "
                          f"encode={result.image_encode_time_ms:.2f}ms, prefill={result.vlm_prefill_time_ms:.2f}ms")

        all_step_results["original"].extend(original_step_results)

        # Compute summaries
        streaming_summary = compute_summary("streaming", num_steps, streaming_step_results)
        original_summary = compute_summary("original", num_steps, original_step_results)

        results["streaming"].append(streaming_summary)
        results["original"].append(original_summary)

        # Print comparison
        print_comparison(streaming_summary, original_summary)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")

    save_results = {
        "timestamp": timestamp,
        "model_path": model_path,
        "clip_id": clip_id,
        "step_counts": step_counts,
        "warmup_steps": warmup_steps,
        "streaming_summaries": [asdict(s) for s in results["streaming"]],
        "original_summaries": [asdict(s) for s in results["original"]],
        "all_step_results": {
            "streaming": [asdict(r) for r in all_step_results["streaming"]],
            "original": [asdict(r) for r in all_step_results["original"]],
        }
    }

    with open(results_file, "w") as f:
        json.dump(save_results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    # Generate visualizations
    plot_results(results, step_counts, output_dir, timestamp)

    return results


def compute_summary(model_type: str, num_steps: int, step_results: list[BenchmarkResult]) -> BenchmarkSummary:
    """Compute summary statistics from step results."""
    if not step_results:
        return BenchmarkSummary(
            model_type=model_type,
            num_steps=num_steps,
            avg_total_time_ms=0,
            avg_image_encode_time_ms=0,
            avg_vlm_prefill_time_ms=0,
            avg_vlm_decode_time_ms=0,
            avg_action_time_ms=0,
            first_total_time_ms=0,
            first_image_encode_time_ms=0,
            first_vlm_prefill_time_ms=0,
            streaming_avg_total_time_ms=0,
            streaming_avg_image_encode_time_ms=0,
            streaming_avg_vlm_prefill_time_ms=0,
        )

    total_times = [r.total_time_ms for r in step_results]
    encode_times = [r.image_encode_time_ms for r in step_results]
    prefill_times = [r.vlm_prefill_time_ms for r in step_results]
    decode_times = [r.vlm_decode_time_ms for r in step_results]
    action_times = [r.action_time_ms for r in step_results]

    # First step
    first_result = step_results[0]

    # Streaming steps (all steps after first)
    streaming_results = step_results[1:] if len(step_results) > 1 else []
    streaming_total = [r.total_time_ms for r in streaming_results] if streaming_results else [0]
    streaming_encode = [r.image_encode_time_ms for r in streaming_results] if streaming_results else [0]
    streaming_prefill = [r.vlm_prefill_time_ms for r in streaming_results] if streaming_results else [0]

    return BenchmarkSummary(
        model_type=model_type,
        num_steps=num_steps,
        avg_total_time_ms=np.mean(total_times),
        avg_image_encode_time_ms=np.mean(encode_times),
        avg_vlm_prefill_time_ms=np.mean(prefill_times),
        avg_vlm_decode_time_ms=np.mean(decode_times),
        avg_action_time_ms=np.mean(action_times),
        first_total_time_ms=first_result.total_time_ms,
        first_image_encode_time_ms=first_result.image_encode_time_ms,
        first_vlm_prefill_time_ms=first_result.vlm_prefill_time_ms,
        streaming_avg_total_time_ms=np.mean(streaming_total),
        streaming_avg_image_encode_time_ms=np.mean(streaming_encode),
        streaming_avg_vlm_prefill_time_ms=np.mean(streaming_prefill),
    )


def print_comparison(streaming: BenchmarkSummary, original: BenchmarkSummary):
    """Print comparison between streaming and original model."""
    print(f"\n{'='*80}")
    print(f"COMPARISON ({streaming.num_steps} steps)")
    print(f"{'='*80}")

    def safe_speedup(streaming_val, original_val):
        return original_val / streaming_val if streaming_val > 0 else 0

    print(f"\n{'METRIC':<45} {'STREAMING':>12} {'ORIGINAL':>12} {'SPEEDUP':>10}")
    print("-" * 80)

    # Overall averages
    print(f"\n{'=== OVERALL AVERAGES ==='}")
    speedup = safe_speedup(streaming.avg_total_time_ms, original.avg_total_time_ms)
    print(f"{'Avg Total Time (ms)':<45} {streaming.avg_total_time_ms:>12.2f} {original.avg_total_time_ms:>12.2f} {speedup:>9.2f}x")

    speedup = safe_speedup(streaming.avg_image_encode_time_ms, original.avg_image_encode_time_ms)
    print(f"{'Avg Image Encode Time (ms)':<45} {streaming.avg_image_encode_time_ms:>12.2f} {original.avg_image_encode_time_ms:>12.2f} {speedup:>9.2f}x")

    speedup = safe_speedup(streaming.avg_vlm_prefill_time_ms, original.avg_vlm_prefill_time_ms)
    print(f"{'Avg VLM Prefill Time (ms)':<45} {streaming.avg_vlm_prefill_time_ms:>12.2f} {original.avg_vlm_prefill_time_ms:>12.2f} {speedup:>9.2f}x")

    speedup = safe_speedup(streaming.avg_vlm_decode_time_ms, original.avg_vlm_decode_time_ms)
    print(f"{'Avg VLM Decode Time (ms)':<45} {streaming.avg_vlm_decode_time_ms:>12.2f} {original.avg_vlm_decode_time_ms:>12.2f} {speedup:>9.2f}x")

    speedup = safe_speedup(streaming.avg_action_time_ms, original.avg_action_time_ms)
    print(f"{'Avg Action/Diffusion Time (ms)':<45} {streaming.avg_action_time_ms:>12.2f} {original.avg_action_time_ms:>12.2f} {speedup:>9.2f}x")

    # First step (prefill)
    print(f"\n{'=== FIRST STEP (FULL PREFILL) ==='}")
    speedup = safe_speedup(streaming.first_total_time_ms, original.first_total_time_ms)
    print(f"{'Total Time (ms)':<45} {streaming.first_total_time_ms:>12.2f} {original.first_total_time_ms:>12.2f} {speedup:>9.2f}x")

    speedup = safe_speedup(streaming.first_image_encode_time_ms, original.first_image_encode_time_ms)
    print(f"{'Image Encode Time (ms)':<45} {streaming.first_image_encode_time_ms:>12.2f} {original.first_image_encode_time_ms:>12.2f} {speedup:>9.2f}x")

    speedup = safe_speedup(streaming.first_vlm_prefill_time_ms, original.first_vlm_prefill_time_ms)
    print(f"{'VLM Prefill Time (ms)':<45} {streaming.first_vlm_prefill_time_ms:>12.2f} {original.first_vlm_prefill_time_ms:>12.2f} {speedup:>9.2f}x")

    # Streaming steps
    print(f"\n{'=== STREAMING STEPS (AVG, EXCLUDING FIRST) ==='}")
    speedup = safe_speedup(streaming.streaming_avg_total_time_ms, original.streaming_avg_total_time_ms)
    print(f"{'Total Time (ms)':<45} {streaming.streaming_avg_total_time_ms:>12.2f} {original.streaming_avg_total_time_ms:>12.2f} {speedup:>9.2f}x")

    speedup = safe_speedup(streaming.streaming_avg_image_encode_time_ms, original.streaming_avg_image_encode_time_ms)
    print(f"{'Image Encode Time (ms)':<45} {streaming.streaming_avg_image_encode_time_ms:>12.2f} {original.streaming_avg_image_encode_time_ms:>12.2f} {speedup:>9.2f}x")

    speedup = safe_speedup(streaming.streaming_avg_vlm_prefill_time_ms, original.streaming_avg_vlm_prefill_time_ms)
    print(f"{'VLM Prefill Time (ms)':<45} {streaming.streaming_avg_vlm_prefill_time_ms:>12.2f} {original.streaming_avg_vlm_prefill_time_ms:>12.2f} {speedup:>9.2f}x")

    # Cumulative time comparison
    streaming_cumulative = streaming.first_total_time_ms + streaming.streaming_avg_total_time_ms * max(0, streaming.num_steps - 1)
    original_cumulative = original.first_total_time_ms + original.streaming_avg_total_time_ms * max(0, original.num_steps - 1)
    speedup = safe_speedup(streaming_cumulative, original_cumulative)
    print(f"\n{'=== CUMULATIVE ==='}")
    print(f"{'Total Cumulative Time (ms)':<45} {streaming_cumulative:>12.2f} {original_cumulative:>12.2f} {speedup:>9.2f}x")

    # Image encode savings
    streaming_encode_cumulative = streaming.first_image_encode_time_ms + streaming.streaming_avg_image_encode_time_ms * max(0, streaming.num_steps - 1)
    original_encode_cumulative = original.first_image_encode_time_ms + original.streaming_avg_image_encode_time_ms * max(0, original.num_steps - 1)
    speedup = safe_speedup(streaming_encode_cumulative, original_encode_cumulative)
    print(f"{'Cumulative Image Encode Time (ms)':<45} {streaming_encode_cumulative:>12.2f} {original_encode_cumulative:>12.2f} {speedup:>9.2f}x")

    # VLM prefill savings
    streaming_prefill_cumulative = streaming.first_vlm_prefill_time_ms + streaming.streaming_avg_vlm_prefill_time_ms * max(0, streaming.num_steps - 1)
    original_prefill_cumulative = original.first_vlm_prefill_time_ms + original.streaming_avg_vlm_prefill_time_ms * max(0, original.num_steps - 1)
    speedup = safe_speedup(streaming_prefill_cumulative, original_prefill_cumulative)
    print(f"{'Cumulative VLM Prefill Time (ms)':<45} {streaming_prefill_cumulative:>12.2f} {original_prefill_cumulative:>12.2f} {speedup:>9.2f}x")


def plot_results(
    results: dict[str, list[BenchmarkSummary]],
    step_counts: list[int],
    output_dir: str,
    timestamp: str,
):
    """Generate visualization plots."""
    # Extract data
    streaming_summaries = results["streaming"]
    original_summaries = results["original"]

    # ============ Main comparison plot (2x3 grid) ============
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Streaming vs Original Model Benchmark', fontsize=14, fontweight='bold')

    x = np.arange(len(step_counts))
    width = 0.35

    # 1. Average Total Time per Step
    ax = axes[0, 0]
    streaming_avg_total = [s.avg_total_time_ms for s in streaming_summaries]
    original_avg_total = [s.avg_total_time_ms for s in original_summaries]

    ax.bar(x - width/2, streaming_avg_total, width, label='Streaming', color='steelblue')
    ax.bar(x + width/2, original_avg_total, width, label='Original', color='coral')
    ax.set_xlabel('Number of Steps')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Average Total Time per Step')
    ax.set_xticks(x)
    ax.set_xticklabels(step_counts)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Average Image Encode Time
    ax = axes[0, 1]
    streaming_encode = [s.avg_image_encode_time_ms for s in streaming_summaries]
    original_encode = [s.avg_image_encode_time_ms for s in original_summaries]

    ax.bar(x - width/2, streaming_encode, width, label='Streaming', color='steelblue')
    ax.bar(x + width/2, original_encode, width, label='Original', color='coral')
    ax.set_xlabel('Number of Steps')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Average Image Encode Time per Step')
    ax.set_xticks(x)
    ax.set_xticklabels(step_counts)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 3. Average VLM Prefill Time
    ax = axes[0, 2]
    streaming_prefill = [s.avg_vlm_prefill_time_ms for s in streaming_summaries]
    original_prefill = [s.avg_vlm_prefill_time_ms for s in original_summaries]

    ax.bar(x - width/2, streaming_prefill, width, label='Streaming', color='steelblue')
    ax.bar(x + width/2, original_prefill, width, label='Original', color='coral')
    ax.set_xlabel('Number of Steps')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Average VLM Prefill Time per Step')
    ax.set_xticks(x)
    ax.set_xticklabels(step_counts)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 4. Streaming Steps Average Time (Total)
    ax = axes[1, 0]
    streaming_streaming_total = [s.streaming_avg_total_time_ms for s in streaming_summaries]
    original_streaming_total = [s.streaming_avg_total_time_ms for s in original_summaries]

    ax.bar(x - width/2, streaming_streaming_total, width, label='Streaming', color='steelblue')
    ax.bar(x + width/2, original_streaming_total, width, label='Original', color='coral')
    ax.set_xlabel('Number of Steps')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Streaming Steps: Average Total Time')
    ax.set_xticks(x)
    ax.set_xticklabels(step_counts)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 5. Streaming Steps Average Image Encode Time
    ax = axes[1, 1]
    streaming_streaming_encode = [s.streaming_avg_image_encode_time_ms for s in streaming_summaries]
    original_streaming_encode = [s.streaming_avg_image_encode_time_ms for s in original_summaries]

    ax.bar(x - width/2, streaming_streaming_encode, width, label='Streaming', color='steelblue')
    ax.bar(x + width/2, original_streaming_encode, width, label='Original', color='coral')
    ax.set_xlabel('Number of Steps')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Streaming Steps: Average Image Encode Time')
    ax.set_xticks(x)
    ax.set_xticklabels(step_counts)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 6. Streaming Steps Average VLM Prefill Time
    ax = axes[1, 2]
    streaming_streaming_prefill = [s.streaming_avg_vlm_prefill_time_ms for s in streaming_summaries]
    original_streaming_prefill = [s.streaming_avg_vlm_prefill_time_ms for s in original_summaries]

    ax.bar(x - width/2, streaming_streaming_prefill, width, label='Streaming', color='steelblue')
    ax.bar(x + width/2, original_streaming_prefill, width, label='Original', color='coral')
    ax.set_xlabel('Number of Steps')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Streaming Steps: Average VLM Prefill Time')
    ax.set_xticks(x)
    ax.set_xticklabels(step_counts)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"benchmark_comparison_{timestamp}.png")
    plt.savefig(plot_file, dpi=150)
    logger.info(f"Comparison plot saved to {plot_file}")
    plt.close()

    # ============ Cumulative Time Plot ============
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Cumulative Time Comparison', fontsize=14, fontweight='bold')

    # Cumulative Total Time
    ax = axes[0]
    streaming_cumulative = [
        s.first_total_time_ms + s.streaming_avg_total_time_ms * max(0, s.num_steps - 1)
        for s in streaming_summaries
    ]
    original_cumulative = [
        s.first_total_time_ms + s.streaming_avg_total_time_ms * max(0, s.num_steps - 1)
        for s in original_summaries
    ]

    ax.plot(step_counts, streaming_cumulative, 'o-', label='Streaming', color='steelblue', linewidth=2, markersize=8)
    ax.plot(step_counts, original_cumulative, 's-', label='Original', color='coral', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Steps')
    ax.set_ylabel('Cumulative Time (ms)')
    ax.set_title('Total Cumulative Inference Time')
    ax.legend()
    ax.grid(alpha=0.3)

    # Cumulative Image Encode Time
    ax = axes[1]
    streaming_encode_cumulative = [
        s.first_image_encode_time_ms + s.streaming_avg_image_encode_time_ms * max(0, s.num_steps - 1)
        for s in streaming_summaries
    ]
    original_encode_cumulative = [
        s.first_image_encode_time_ms + s.streaming_avg_image_encode_time_ms * max(0, s.num_steps - 1)
        for s in original_summaries
    ]

    ax.plot(step_counts, streaming_encode_cumulative, 'o-', label='Streaming', color='steelblue', linewidth=2, markersize=8)
    ax.plot(step_counts, original_encode_cumulative, 's-', label='Original', color='coral', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Steps')
    ax.set_ylabel('Cumulative Time (ms)')
    ax.set_title('Cumulative Image Encode Time')
    ax.legend()
    ax.grid(alpha=0.3)

    # Cumulative VLM Prefill Time
    ax = axes[2]
    streaming_prefill_cumulative = [
        s.first_vlm_prefill_time_ms + s.streaming_avg_vlm_prefill_time_ms * max(0, s.num_steps - 1)
        for s in streaming_summaries
    ]
    original_prefill_cumulative = [
        s.first_vlm_prefill_time_ms + s.streaming_avg_vlm_prefill_time_ms * max(0, s.num_steps - 1)
        for s in original_summaries
    ]

    ax.plot(step_counts, streaming_prefill_cumulative, 'o-', label='Streaming', color='steelblue', linewidth=2, markersize=8)
    ax.plot(step_counts, original_prefill_cumulative, 's-', label='Original', color='coral', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Steps')
    ax.set_ylabel('Cumulative Time (ms)')
    ax.set_title('Cumulative VLM Prefill Time')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    cumulative_file = os.path.join(output_dir, f"benchmark_cumulative_{timestamp}.png")
    plt.savefig(cumulative_file, dpi=150)
    logger.info(f"Cumulative plot saved to {cumulative_file}")
    plt.close()

    # ============ Speedup Plot ============
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Streaming Model Speedup over Original Model', fontsize=14, fontweight='bold')

    def safe_div(a, b):
        return [bi / ai if ai > 0 else 0 for ai, bi in zip(a, b)]

    # Total Time Speedup
    ax = axes[0]
    speedups_avg = safe_div(streaming_avg_total, original_avg_total)
    speedups_streaming = safe_div(streaming_streaming_total, original_streaming_total)
    speedups_cumulative = safe_div(streaming_cumulative, original_cumulative)

    ax.plot(step_counts, speedups_avg, 'o-', label='Avg per Step', linewidth=2, markersize=8)
    ax.plot(step_counts, speedups_streaming, 's-', label='Streaming Steps Only', linewidth=2, markersize=8)
    ax.plot(step_counts, speedups_cumulative, '^-', label='Cumulative', linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No speedup')
    ax.set_xlabel('Number of Steps')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Total Time Speedup')
    ax.legend()
    ax.grid(alpha=0.3)

    # Image Encode Speedup
    ax = axes[1]
    speedups_encode_avg = safe_div(streaming_encode, original_encode)
    speedups_encode_streaming = safe_div(streaming_streaming_encode, original_streaming_encode)
    speedups_encode_cumulative = safe_div(streaming_encode_cumulative, original_encode_cumulative)

    ax.plot(step_counts, speedups_encode_avg, 'o-', label='Avg per Step', linewidth=2, markersize=8)
    ax.plot(step_counts, speedups_encode_streaming, 's-', label='Streaming Steps Only', linewidth=2, markersize=8)
    ax.plot(step_counts, speedups_encode_cumulative, '^-', label='Cumulative', linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No speedup')
    ax.set_xlabel('Number of Steps')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Image Encode Speedup')
    ax.legend()
    ax.grid(alpha=0.3)

    # VLM Prefill Speedup
    ax = axes[2]
    speedups_prefill_avg = safe_div(streaming_prefill, original_prefill)
    speedups_prefill_streaming = safe_div(streaming_streaming_prefill, original_streaming_prefill)
    speedups_prefill_cumulative = safe_div(streaming_prefill_cumulative, original_prefill_cumulative)

    ax.plot(step_counts, speedups_prefill_avg, 'o-', label='Avg per Step', linewidth=2, markersize=8)
    ax.plot(step_counts, speedups_prefill_streaming, 's-', label='Streaming Steps Only', linewidth=2, markersize=8)
    ax.plot(step_counts, speedups_prefill_cumulative, '^-', label='Cumulative', linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No speedup')
    ax.set_xlabel('Number of Steps')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('VLM Prefill Speedup')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    speedup_file = os.path.join(output_dir, f"benchmark_speedup_{timestamp}.png")
    plt.savefig(speedup_file, dpi=150)
    logger.info(f"Speedup plot saved to {speedup_file}")
    plt.close()

    # ============ Time Breakdown Stacked Bar Chart ============
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Time Breakdown by Component', fontsize=14, fontweight='bold')

    # Use the last step count for detailed breakdown
    last_idx = -1
    streaming_s = streaming_summaries[last_idx]
    original_s = original_summaries[last_idx]

    # Streaming steps breakdown
    ax = axes[0]
    categories = ['Streaming Model', 'Original Model']
    encode_times = [streaming_s.streaming_avg_image_encode_time_ms, original_s.streaming_avg_image_encode_time_ms]
    prefill_times = [streaming_s.streaming_avg_vlm_prefill_time_ms, original_s.streaming_avg_vlm_prefill_time_ms]
    decode_times = [streaming_s.avg_vlm_decode_time_ms, original_s.avg_vlm_decode_time_ms]
    action_times = [streaming_s.avg_action_time_ms, original_s.avg_action_time_ms]

    bar_width = 0.5
    x_pos = np.arange(len(categories))

    p1 = ax.bar(x_pos, encode_times, bar_width, label='Image Encode', color='#2ecc71')
    p2 = ax.bar(x_pos, prefill_times, bar_width, bottom=encode_times, label='VLM Prefill', color='#3498db')
    p3 = ax.bar(x_pos, decode_times, bar_width, bottom=np.array(encode_times) + np.array(prefill_times), label='VLM Decode', color='#9b59b6')
    p4 = ax.bar(x_pos, action_times, bar_width, bottom=np.array(encode_times) + np.array(prefill_times) + np.array(decode_times), label='Action/Diffusion', color='#e74c3c')

    ax.set_xlabel('Model')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Streaming Step Time Breakdown ({step_counts[last_idx]} steps)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value annotations
    for i, (e, p, d, a) in enumerate(zip(encode_times, prefill_times, decode_times, action_times)):
        total = e + p + d + a
        ax.annotate(f'{total:.1f}ms', xy=(i, total), ha='center', va='bottom', fontsize=10, fontweight='bold')

    # First step breakdown
    ax = axes[1]
    encode_times_first = [streaming_s.first_image_encode_time_ms, original_s.first_image_encode_time_ms]
    prefill_times_first = [streaming_s.first_vlm_prefill_time_ms, original_s.first_vlm_prefill_time_ms]
    # For first step, use first step totals to estimate decode and action
    other_times_first = [
        streaming_s.first_total_time_ms - streaming_s.first_image_encode_time_ms - streaming_s.first_vlm_prefill_time_ms,
        original_s.first_total_time_ms - original_s.first_image_encode_time_ms - original_s.first_vlm_prefill_time_ms,
    ]

    p1 = ax.bar(x_pos, encode_times_first, bar_width, label='Image Encode', color='#2ecc71')
    p2 = ax.bar(x_pos, prefill_times_first, bar_width, bottom=encode_times_first, label='VLM Prefill', color='#3498db')
    p3 = ax.bar(x_pos, other_times_first, bar_width, bottom=np.array(encode_times_first) + np.array(prefill_times_first), label='Other (Decode+Action)', color='#95a5a6')

    ax.set_xlabel('Model')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'First Step (Full Prefill) Time Breakdown')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value annotations
    for i, (e, p, o) in enumerate(zip(encode_times_first, prefill_times_first, other_times_first)):
        total = e + p + o
        ax.annotate(f'{total:.1f}ms', xy=(i, total), ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    breakdown_file = os.path.join(output_dir, f"benchmark_breakdown_{timestamp}.png")
    plt.savefig(breakdown_file, dpi=150)
    logger.info(f"Breakdown plot saved to {breakdown_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark streaming vs original model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./Alpamayo-R1-10B",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--clip_id",
        type=str,
        default="030c760c-ae38-49aa-9ad8-f5650a545d26",
        help="Clip ID for dataset",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20],
        help="Number of steps to benchmark",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./benchmark_results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    logger.info(f"Starting benchmark with steps: {args.steps}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Clip ID: {args.clip_id}")

    results = run_benchmark(
        model_path=args.model_path,
        clip_id=args.clip_id,
        step_counts=args.steps,
        warmup_steps=args.warmup,
        output_dir=args.output_dir,
    )

    logger.info("\nBenchmark completed!")


if __name__ == "__main__":
    main()
