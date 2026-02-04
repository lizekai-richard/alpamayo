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
Comprehensive Benchmark for Unified Model with Latency Breakdown.

This script benchmarks the unified AlpamayoR1 model with torch.compile support,
providing detailed latency breakdown for each inference phase:
  - Encode: Visual encoder forward pass
  - Prefill: Language model prefill phase
  - Decode: Autoregressive token generation
  - Action: Diffusion sampling for trajectory prediction

Features:
  - Supports both streaming and non-streaming modes
  - Compare compiled vs non-compiled performance
  - Detailed per-phase timing with CUDA synchronization
  - Warmup phase for torch.compile graph capture
  - Statistical analysis (mean, std, min, max, percentiles)
  - Visualization with breakdown charts

================================================================================
HOW TO RUN
================================================================================

1. Default configuration (streaming mode, compiled):

    cd /home/user/zekai/alpamayo
    python -m alpamayo_r1.benchmark_compile

2. Non-streaming mode:

    python -m alpamayo_r1.benchmark_compile --non_streaming

3. Compare compiled vs non-compiled:

    python -m alpamayo_r1.benchmark_compile --compare

4. Custom parameters:

    python -m alpamayo_r1.benchmark_compile \\
        --model_path ./Alpamayo-R1-10B \\
        --num_steps 20 \\
        --warmup_steps 3 \\
        --non_streaming \\
        --output_dir ./benchmark_results

================================================================================
"""

import argparse
import copy
import gc
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_float32_matmul_precision("high")

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Timing Utilities
# =============================================================================

class CUDATimer:
    """High-precision CUDA timer using events."""

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.elapsed_ms = 0.0

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_event.record()
        return self

    def __exit__(self, *args):
        self.end_event.record()
        torch.cuda.synchronize()
        self.elapsed_ms = self.start_event.elapsed_time(self.end_event)


class PerfTimer:
    """Simple perf_counter timer with CUDA sync."""

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


@dataclass
class StepTiming:
    """Timing results for a single inference step."""
    step_idx: int
    is_first_prefill: bool

    # Phase timings (ms)
    encode_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    decode_token_count: int = 0
    action_ms: float = 0.0

    # Total time
    total_ms: float = 0.0

    # Overhead (total - sum of phases)
    overhead_ms: float = 0.0

    @property
    def decode_per_token_ms(self) -> float:
        """Average decode time per token."""
        if self.decode_token_count > 0:
            return self.decode_ms / self.decode_token_count
        return 0.0

    def compute_overhead(self):
        """Compute timing overhead."""
        self.overhead_ms = self.total_ms - (self.encode_ms + self.prefill_ms + self.decode_ms + self.action_ms)


@dataclass
class BenchmarkStats:
    """Statistical summary of benchmark results."""
    name: str
    num_samples: int

    # Per-phase statistics
    encode_mean_ms: float = 0.0
    encode_std_ms: float = 0.0
    encode_min_ms: float = 0.0
    encode_max_ms: float = 0.0

    prefill_mean_ms: float = 0.0
    prefill_std_ms: float = 0.0
    prefill_min_ms: float = 0.0
    prefill_max_ms: float = 0.0

    decode_mean_ms: float = 0.0
    decode_std_ms: float = 0.0
    decode_min_ms: float = 0.0
    decode_max_ms: float = 0.0
    decode_per_token_mean_ms: float = 0.0

    action_mean_ms: float = 0.0
    action_std_ms: float = 0.0
    action_min_ms: float = 0.0
    action_max_ms: float = 0.0

    total_mean_ms: float = 0.0
    total_std_ms: float = 0.0
    total_min_ms: float = 0.0
    total_max_ms: float = 0.0

    overhead_mean_ms: float = 0.0

    # Percentiles
    total_p50_ms: float = 0.0
    total_p90_ms: float = 0.0
    total_p99_ms: float = 0.0


def compute_stats(name: str, timings: list[StepTiming]) -> BenchmarkStats:
    """Compute statistics from a list of step timings."""
    if not timings:
        return BenchmarkStats(name=name, num_samples=0)

    encode = [t.encode_ms for t in timings]
    prefill = [t.prefill_ms for t in timings]
    decode = [t.decode_ms for t in timings]
    decode_per_token = [t.decode_per_token_ms for t in timings]
    action = [t.action_ms for t in timings]
    total = [t.total_ms for t in timings]
    overhead = [t.overhead_ms for t in timings]

    return BenchmarkStats(
        name=name,
        num_samples=len(timings),
        encode_mean_ms=np.mean(encode),
        encode_std_ms=np.std(encode),
        encode_min_ms=np.min(encode),
        encode_max_ms=np.max(encode),
        prefill_mean_ms=np.mean(prefill),
        prefill_std_ms=np.std(prefill),
        prefill_min_ms=np.min(prefill),
        prefill_max_ms=np.max(prefill),
        decode_mean_ms=np.mean(decode),
        decode_std_ms=np.std(decode),
        decode_min_ms=np.min(decode),
        decode_max_ms=np.max(decode),
        decode_per_token_mean_ms=np.mean(decode_per_token),
        action_mean_ms=np.mean(action),
        action_std_ms=np.std(action),
        action_min_ms=np.min(action),
        action_max_ms=np.max(action),
        total_mean_ms=np.mean(total),
        total_std_ms=np.std(total),
        total_min_ms=np.min(total),
        total_max_ms=np.max(total),
        overhead_mean_ms=np.mean(overhead),
        total_p50_ms=np.percentile(total, 50),
        total_p90_ms=np.percentile(total, 90),
        total_p99_ms=np.percentile(total, 99),
    )


# =============================================================================
# Instrumented Model Wrapper
# =============================================================================

class InstrumentedStreamingModel:
    """
    Wrapper that instruments the compiled streaming model for detailed timing.

    This class wraps the model's inference method and inserts timing code
    around each major phase (encode, prefill, decode, action).
    """

    def __init__(self, model_path: str, dtype=torch.bfloat16):
        from alpamayo_r1.models.alpamayo_r1_unified import AlpamayoR1

        logger.info("Loading unified model...")
        self.model = AlpamayoR1.from_pretrained(model_path, dtype=dtype).to("cuda")
        self.processor = helper.get_processor(self.model.tokenizer)
        logger.info("Model loaded.")

        # Timing storage for current step
        self._current_timing = None

        # Instrument the model's internal methods
        self._instrument_model()

    def _instrument_model(self):
        """Wrap model methods with timing instrumentation."""
        model = self.model

        # Store original methods
        self._original_encode = model._encode
        self._original_prefill = model._prefill
        self._original_decode = model._decode
        self._original_action = model._action

        # Create instrumented versions
        def instrumented_encode(*args, **kwargs):
            with CUDATimer() as timer:
                result = self._original_encode(*args, **kwargs)
            if self._current_timing is not None:
                self._current_timing.encode_ms += timer.elapsed_ms
            return result

        def instrumented_prefill(*args, **kwargs):
            with CUDATimer() as timer:
                result = self._original_prefill(*args, **kwargs)
            if self._current_timing is not None:
                self._current_timing.prefill_ms += timer.elapsed_ms
            return result

        def instrumented_decode(*args, **kwargs):
            with CUDATimer() as timer:
                result = self._original_decode(*args, **kwargs)
            if self._current_timing is not None:
                self._current_timing.decode_ms += timer.elapsed_ms
                self._current_timing.decode_token_count += 1
            return result

        def instrumented_action(*args, **kwargs):
            with CUDATimer() as timer:
                result = self._original_action(*args, **kwargs)
            if self._current_timing is not None:
                self._current_timing.action_ms += timer.elapsed_ms
            return result

        # Replace methods
        model._encode = instrumented_encode
        model._prefill = instrumented_prefill
        model._decode = instrumented_decode
        model._action = instrumented_action

    def reset(self):
        """Reset model state for a new benchmark run."""
        self.model.reset_streaming_state()
        self.model._past_key_values = None

        # Reset compiled function caches (force recompilation)
        for attr in ['_compiled_encode_fn', '_compiled_prefill_fn',
                     '_compiled_decode_fn_streaming', '_compiled_decode_fn_non_streaming',
                     '_compiled_action_fn_streaming', '_compiled_action_fn_non_streaming']:
            if hasattr(self.model, attr):
                delattr(self.model, attr)

        # Reset static buffers
        for attr in ['_encode_fn', '_prefill_fn',
                     '_decode_fn_streaming', '_decode_fn_non_streaming',
                     '_action_fn_streaming', '_action_fn_non_streaming']:
            if hasattr(self.model, attr):
                delattr(self.model, attr)

    @torch.inference_mode()
    def run_step(
        self,
        model_inputs: dict,
        step_idx: int,
        torch_compile: str | None = "max-autotune",
        streaming: bool = True,
        fuse_qkv: bool = False,
        fuse_gate_up: bool = False,
    ) -> tuple[StepTiming, Optional[tuple]]:
        """
        Run one inference step with detailed timing.

        Args:
            model_inputs: Input data dict
            step_idx: Current step index
            torch_compile: Compile mode or None to disable
            streaming: If True, use streaming mode; if False, use non-streaming mode
            fuse_qkv: If True, fuse q/k/v projections into single QKVLinear
            fuse_gate_up: If True, fuse gate/up projections into single MergedColumnLinear

        Returns:
            timing: StepTiming with per-phase latencies
            outputs: (pred_xyz, pred_rot, extra) or None for first prefill
        """
        is_first_prefill = self.model.is_first_prefill if streaming else False

        # Initialize timing for this step
        self._current_timing = StepTiming(
            step_idx=step_idx,
            is_first_prefill=is_first_prefill,
        )

        # Prepare data
        data = helper.to_device(model_inputs, "cuda")

        # Run inference with total timing
        with CUDATimer() as total_timer:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = self.model.sample_trajectories(
                    data=data,
                    streaming=streaming,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1,
                    max_generation_length=256,
                    torch_compile=torch_compile,
                    fuse_qkv=fuse_qkv,
                    fuse_gate_up=fuse_gate_up,
                    return_extra=True,
                )

        self._current_timing.total_ms = total_timer.elapsed_ms
        self._current_timing.compute_overhead()

        timing = self._current_timing
        self._current_timing = None

        return timing, result


# =============================================================================
# Data Preparation
# =============================================================================

def create_streaming_inputs(
    processor,
    num_windows: int,
    clip_id: str,
    t0_us: int = 5_100_000,
    time_step_us: int = 100_000,
) -> list[dict]:
    """Create sliding window inputs for streaming inference.

    For streaming:
    - Window 0 (prefill): 4 cameras × 4 frames = 16 frames
    - Window 1+: 4 cameras × 1 new frame = 4 frames
    """
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


def create_non_streaming_inputs(
    processor,
    num_windows: int,
    clip_id: str,
    t0_us: int = 5_100_000,
    time_step_us: int = 100_000,
) -> list[dict]:
    """Create sliding window inputs for non-streaming inference.

    For non-streaming:
    - Every window: 4 cameras × 4 frames = 16 frames
    """
    non_streaming_inputs = []

    for window_idx in range(num_windows):
        current_t0 = t0_us + window_idx * time_step_us

        data = load_physical_aiavdataset(clip_id, t0_us=current_t0, num_frames=4)
        frames = data["image_frames"].flatten(0, 1)  # (4, 4, C, H, W) -> (16, C, H, W)

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
        non_streaming_inputs.append(model_inputs)

    return non_streaming_inputs


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_compiled_benchmark(
    model_path: str,
    clip_id: str,
    num_steps: int = 20,
    warmup_steps: int = 3,
    torch_compile: str | None = "max-autotune",
    streaming: bool = True,
    fuse_qkv: bool = False,
    fuse_gate_up: bool = False,
    output_dir: str = "./benchmark_results",
) -> dict:
    """
    Run benchmark on the unified model.

    Args:
        model_path: Path to model checkpoint
        clip_id: Clip ID for dataset
        num_steps: Number of steps to benchmark (excluding warmup)
        warmup_steps: Number of warmup steps for torch.compile
        torch_compile: Compile mode or None to disable
        streaming: If True, use streaming mode; if False, use non-streaming mode
        fuse_qkv: If True, fuse q/k/v projections into single QKVLinear
        fuse_gate_up: If True, fuse gate/up projections into single MergedColumnLinear
        output_dir: Output directory for results

    Returns:
        Dictionary with benchmark results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model_wrapper = InstrumentedStreamingModel(model_path)

    # Create inputs
    total_steps = warmup_steps + num_steps + (1 if streaming else 0)  # +1 for first prefill in streaming
    if streaming:
        benchmark_inputs = create_streaming_inputs(
            processor=model_wrapper.processor,
            num_windows=total_steps,
            clip_id=clip_id,
        )
    else:
        benchmark_inputs = create_non_streaming_inputs(
            processor=model_wrapper.processor,
            num_windows=total_steps,
            clip_id=clip_id,
        )

    all_timings: list[StepTiming] = []
    warmup_timings: list[StepTiming] = []
    first_prefill_timing: Optional[StepTiming] = None

    mode_str = "STREAMING" if streaming else "NON-STREAMING"
    fuse_str = f"fuse_qkv={fuse_qkv}, fuse_gate_up={fuse_gate_up}"
    logger.info(f"\n{'='*70}")
    logger.info(f"UNIFIED MODEL BENCHMARK ({mode_str})")
    logger.info(f"{'='*70}")
    logger.info(f"torch.compile mode: {torch_compile}")
    logger.info(f"Streaming mode: {streaming}")
    logger.info(f"Fusion options: {fuse_str}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Benchmark steps: {num_steps}")
    logger.info(f"{'='*70}\n")

    # Reset model state
    model_wrapper.reset()

    for step_idx, inputs in enumerate(benchmark_inputs):
        inputs_copy = copy.deepcopy(inputs)
        inputs_copy.pop("is_prefill", None)

        timing, result = model_wrapper.run_step(
            model_inputs=inputs_copy,
            step_idx=step_idx,
            torch_compile=torch_compile,
            streaming=streaming,
            fuse_qkv=fuse_qkv,
            fuse_gate_up=fuse_gate_up,
        )

        # Categorize timing
        if step_idx == 0:
            first_prefill_timing = timing
            phase = "FIRST PREFILL"
        elif step_idx <= warmup_steps:
            warmup_timings.append(timing)
            phase = f"WARMUP {step_idx}"
        else:
            all_timings.append(timing)
            phase = f"STEP {step_idx - warmup_steps}"

        # Log progress
        if timing.is_first_prefill:
            logger.info(f"[{phase}] First prefill - no output")
        else:
            logger.info(
                f"[{phase}] "
                f"Total: {timing.total_ms:7.2f}ms | "
                f"Encode: {timing.encode_ms:6.2f}ms | "
                f"Prefill: {timing.prefill_ms:6.2f}ms | "
                f"Decode: {timing.decode_ms:6.2f}ms ({timing.decode_token_count} tokens) | "
                f"Action: {timing.action_ms:6.2f}ms"
            )

    # Compute statistics
    stats_name = "Streaming Steps" if streaming else "Non-Streaming Steps"
    step_stats = compute_stats(stats_name, all_timings)
    warmup_stats = compute_stats("Warmup Steps", warmup_timings)

    # Print summary
    print_summary(step_stats, warmup_stats, first_prefill_timing)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    compile_mode = torch_compile if torch_compile else "no_compile"
    mode_suffix = "streaming" if streaming else "non_streaming"
    fuse_suffix = ""
    if fuse_qkv:
        fuse_suffix += "_fuse_qkv"
    if fuse_gate_up:
        fuse_suffix += "_fuse_gate_up"

    results = {
        "timestamp": timestamp,
        "model_path": model_path,
        "clip_id": clip_id,
        "torch_compile": torch_compile,
        "streaming": streaming,
        "fuse_qkv": fuse_qkv,
        "fuse_gate_up": fuse_gate_up,
        "warmup_steps": warmup_steps,
        "num_steps": num_steps,
        "first_prefill": asdict(first_prefill_timing) if first_prefill_timing else None,
        "warmup_timings": [asdict(t) for t in warmup_timings],
        "streaming_timings": [asdict(t) for t in all_timings],
        "streaming_stats": asdict(step_stats),
        "warmup_stats": asdict(warmup_stats),
    }

    results_file = os.path.join(output_dir, f"benchmark_{mode_suffix}_{compile_mode}{fuse_suffix}_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    # Generate plots
    plot_suffix = f"{mode_suffix}_{compile_mode}{fuse_suffix}"
    plot_latency_breakdown(all_timings, step_stats, output_dir, timestamp, plot_suffix)
    plot_timing_trace(all_timings, output_dir, timestamp, plot_suffix)

    # Clean up model to free GPU memory
    del model_wrapper
    gc.collect()
    torch.cuda.empty_cache()

    return results


def run_comparison_benchmark(
    model_path: str,
    clip_id: str,
    num_steps: int = 15,
    warmup_steps: int = 3,
    streaming: bool = True,
    fuse_qkv: bool = False,
    fuse_gate_up: bool = False,
    output_dir: str = "./benchmark_results",
) -> dict:
    """
    Run comparison benchmark: compiled vs non-compiled.

    Args:
        model_path: Path to model checkpoint
        clip_id: Clip ID for dataset
        num_steps: Number of steps to benchmark
        warmup_steps: Number of warmup steps
        streaming: If True, use streaming mode; if False, use non-streaming mode
        fuse_qkv: If True, fuse q/k/v projections into single QKVLinear
        fuse_gate_up: If True, fuse gate/up projections into single MergedColumnLinear
        output_dir: Output directory for results
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    mode_str = "STREAMING" if streaming else "NON-STREAMING"
    fuse_str = f"fuse_qkv={fuse_qkv}, fuse_gate_up={fuse_gate_up}"

    # Run compiled benchmark
    logger.info("\n" + "="*70)
    logger.info(f"RUNNING COMPILED {mode_str} BENCHMARK (max-autotune)")
    logger.info(f"Fusion options: {fuse_str}")
    logger.info("="*70 + "\n")

    results["compiled"] = run_compiled_benchmark(
        model_path=model_path,
        clip_id=clip_id,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        torch_compile="max-autotune",
        streaming=streaming,
        fuse_qkv=fuse_qkv,
        fuse_gate_up=fuse_gate_up,
        output_dir=output_dir,
    )

    # Clear CUDA cache between runs
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Run non-compiled benchmark
    logger.info("\n" + "="*70)
    logger.info(f"RUNNING NON-COMPILED {mode_str} BENCHMARK")
    logger.info(f"Fusion options: {fuse_str}")
    logger.info("="*70 + "\n")

    results["non_compiled"] = run_compiled_benchmark(
        model_path=model_path,
        clip_id=clip_id,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        torch_compile=None,
        streaming=streaming,
        fuse_qkv=fuse_qkv,
        fuse_gate_up=fuse_gate_up,
        output_dir=output_dir,
    )

    # Generate comparison plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_comparison(results, output_dir, timestamp)

    # Print comparison summary
    print_comparison_summary(results)

    return results


# =============================================================================
# Output and Visualization
# =============================================================================

def print_summary(
    streaming_stats: BenchmarkStats,
    warmup_stats: BenchmarkStats,
    first_prefill: Optional[StepTiming],
):
    """Print benchmark summary."""
    print(f"\n{'='*80}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*80}")

    if first_prefill:
        print(f"\n--- FIRST PREFILL ---")
        print(f"  Total: {first_prefill.total_ms:.2f}ms (no phase breakdown for first prefill)")

    print(f"\n--- WARMUP STATISTICS ({warmup_stats.num_samples} steps) ---")
    print(f"  Total:   {warmup_stats.total_mean_ms:7.2f} ± {warmup_stats.total_std_ms:6.2f}ms")
    print(f"  Encode:  {warmup_stats.encode_mean_ms:7.2f} ± {warmup_stats.encode_std_ms:6.2f}ms")
    print(f"  Prefill: {warmup_stats.prefill_mean_ms:7.2f} ± {warmup_stats.prefill_std_ms:6.2f}ms")
    print(f"  Decode:  {warmup_stats.decode_mean_ms:7.2f} ± {warmup_stats.decode_std_ms:6.2f}ms")
    print(f"  Action:  {warmup_stats.action_mean_ms:7.2f} ± {warmup_stats.action_std_ms:6.2f}ms")

    print(f"\n--- STREAMING STATISTICS ({streaming_stats.num_samples} steps) ---")
    print(f"  Total:   {streaming_stats.total_mean_ms:7.2f} ± {streaming_stats.total_std_ms:6.2f}ms  "
          f"[min: {streaming_stats.total_min_ms:.2f}, max: {streaming_stats.total_max_ms:.2f}]")
    print(f"  Encode:  {streaming_stats.encode_mean_ms:7.2f} ± {streaming_stats.encode_std_ms:6.2f}ms  "
          f"[min: {streaming_stats.encode_min_ms:.2f}, max: {streaming_stats.encode_max_ms:.2f}]")
    print(f"  Prefill: {streaming_stats.prefill_mean_ms:7.2f} ± {streaming_stats.prefill_std_ms:6.2f}ms  "
          f"[min: {streaming_stats.prefill_min_ms:.2f}, max: {streaming_stats.prefill_max_ms:.2f}]")
    print(f"  Decode:  {streaming_stats.decode_mean_ms:7.2f} ± {streaming_stats.decode_std_ms:6.2f}ms  "
          f"[min: {streaming_stats.decode_min_ms:.2f}, max: {streaming_stats.decode_max_ms:.2f}]")
    print(f"    Per-token: {streaming_stats.decode_per_token_mean_ms:.2f}ms")
    print(f"  Action:  {streaming_stats.action_mean_ms:7.2f} ± {streaming_stats.action_std_ms:6.2f}ms  "
          f"[min: {streaming_stats.action_min_ms:.2f}, max: {streaming_stats.action_max_ms:.2f}]")
    print(f"  Overhead: {streaming_stats.overhead_mean_ms:6.2f}ms")

    print(f"\n--- PERCENTILES ---")
    print(f"  P50 (median): {streaming_stats.total_p50_ms:.2f}ms")
    print(f"  P90:          {streaming_stats.total_p90_ms:.2f}ms")
    print(f"  P99:          {streaming_stats.total_p99_ms:.2f}ms")

    # Latency breakdown percentage
    total = streaming_stats.total_mean_ms
    if total > 0:
        print(f"\n--- LATENCY BREAKDOWN (% of total) ---")
        print(f"  Encode:  {streaming_stats.encode_mean_ms/total*100:5.1f}%")
        print(f"  Prefill: {streaming_stats.prefill_mean_ms/total*100:5.1f}%")
        print(f"  Decode:  {streaming_stats.decode_mean_ms/total*100:5.1f}%")
        print(f"  Action:  {streaming_stats.action_mean_ms/total*100:5.1f}%")
        print(f"  Overhead:{streaming_stats.overhead_mean_ms/total*100:5.1f}%")

    print(f"{'='*80}\n")


def print_comparison_summary(results: dict):
    """Print comparison between compiled and non-compiled."""
    compiled = results["compiled"]["streaming_stats"]
    non_compiled = results["non_compiled"]["streaming_stats"]

    print(f"\n{'='*80}")
    print(f"COMPARISON: COMPILED vs NON-COMPILED")
    print(f"{'='*80}")

    def speedup(c, nc):
        return nc / c if c > 0 else 0

    print(f"\n{'METRIC':<25} {'COMPILED':>12} {'NON-COMPILED':>14} {'SPEEDUP':>10}")
    print("-" * 65)

    print(f"{'Total (mean)':<25} {compiled['total_mean_ms']:>10.2f}ms {non_compiled['total_mean_ms']:>12.2f}ms "
          f"{speedup(compiled['total_mean_ms'], non_compiled['total_mean_ms']):>9.2f}x")
    print(f"{'Encode (mean)':<25} {compiled['encode_mean_ms']:>10.2f}ms {non_compiled['encode_mean_ms']:>12.2f}ms "
          f"{speedup(compiled['encode_mean_ms'], non_compiled['encode_mean_ms']):>9.2f}x")
    print(f"{'Prefill (mean)':<25} {compiled['prefill_mean_ms']:>10.2f}ms {non_compiled['prefill_mean_ms']:>12.2f}ms "
          f"{speedup(compiled['prefill_mean_ms'], non_compiled['prefill_mean_ms']):>9.2f}x")
    print(f"{'Decode (mean)':<25} {compiled['decode_mean_ms']:>10.2f}ms {non_compiled['decode_mean_ms']:>12.2f}ms "
          f"{speedup(compiled['decode_mean_ms'], non_compiled['decode_mean_ms']):>9.2f}x")
    print(f"{'Action (mean)':<25} {compiled['action_mean_ms']:>10.2f}ms {non_compiled['action_mean_ms']:>12.2f}ms "
          f"{speedup(compiled['action_mean_ms'], non_compiled['action_mean_ms']):>9.2f}x")

    print(f"\n{'Decode per-token':<25} {compiled['decode_per_token_mean_ms']:>10.2f}ms {non_compiled['decode_per_token_mean_ms']:>12.2f}ms "
          f"{speedup(compiled['decode_per_token_mean_ms'], non_compiled['decode_per_token_mean_ms']):>9.2f}x")

    print(f"{'='*80}\n")


def plot_latency_breakdown(
    timings: list[StepTiming],
    stats: BenchmarkStats,
    output_dir: str,
    timestamp: str,
    compile_mode: str,
):
    """Generate latency breakdown visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Latency Breakdown - {compile_mode}', fontsize=14, fontweight='bold')

    # 1. Stacked bar chart - mean breakdown
    ax = axes[0]
    phases = ['Encode', 'Prefill', 'Decode', 'Action', 'Overhead']
    values = [
        stats.encode_mean_ms,
        stats.prefill_mean_ms,
        stats.decode_mean_ms,
        stats.action_mean_ms,
        stats.overhead_mean_ms,
    ]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#95a5a6']

    bottom = 0
    bars = []
    for val, color, phase in zip(values, colors, phases):
        bar = ax.bar(['Mean'], [val], bottom=[bottom], color=color, label=phase)
        bars.append(bar)
        bottom += val

    ax.set_ylabel('Time (ms)')
    ax.set_title('Mean Latency Breakdown')
    ax.legend(loc='upper right')
    ax.set_ylim(0, bottom * 1.1)

    # Add value annotations
    bottom = 0
    for val, phase in zip(values, phases):
        if val > stats.total_mean_ms * 0.05:  # Only show if > 5%
            ax.text(0, bottom + val/2, f'{val:.1f}ms\n({val/stats.total_mean_ms*100:.0f}%)',
                   ha='center', va='center', fontsize=9)
        bottom += val

    # 2. Per-phase box plot
    ax = axes[1]
    encode_data = [t.encode_ms for t in timings]
    prefill_data = [t.prefill_ms for t in timings]
    decode_data = [t.decode_ms for t in timings]
    action_data = [t.action_ms for t in timings]

    bp = ax.boxplot(
        [encode_data, prefill_data, decode_data, action_data],
        labels=['Encode', 'Prefill', 'Decode', 'Action'],
        patch_artist=True,
    )

    for patch, color in zip(bp['boxes'], colors[:4]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Time (ms)')
    ax.set_title('Latency Distribution by Phase')
    ax.grid(axis='y', alpha=0.3)

    # 3. Pie chart
    ax = axes[2]
    non_zero_phases = [(p, v) for p, v in zip(phases, values) if v > 0]
    if non_zero_phases:
        labels, sizes = zip(*non_zero_phases)
        colors_filtered = [colors[phases.index(p)] for p in labels]

        ax.pie(sizes, labels=labels, colors=colors_filtered, autopct='%1.1f%%',
               startangle=90, explode=[0.02]*len(sizes))
        ax.set_title('Time Distribution')

    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"latency_breakdown_{compile_mode}_{timestamp}.png")
    plt.savefig(plot_file, dpi=150)
    logger.info(f"Latency breakdown plot saved to {plot_file}")
    plt.close()


def plot_timing_trace(
    timings: list[StepTiming],
    output_dir: str,
    timestamp: str,
    compile_mode: str,
):
    """Generate timing trace over steps."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Timing Trace - {compile_mode}', fontsize=14, fontweight='bold')

    steps = list(range(len(timings)))

    # 1. Total time trace
    ax = axes[0, 0]
    total_times = [t.total_ms for t in timings]
    ax.plot(steps, total_times, 'b-o', markersize=4)
    ax.axhline(y=np.mean(total_times), color='r', linestyle='--', label=f'Mean: {np.mean(total_times):.2f}ms')
    ax.fill_between(steps,
                    [np.mean(total_times) - np.std(total_times)] * len(steps),
                    [np.mean(total_times) + np.std(total_times)] * len(steps),
                    alpha=0.2, color='red')
    ax.set_xlabel('Step')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Total Inference Time per Step')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Phase breakdown over steps (stacked area)
    ax = axes[0, 1]
    encode_times = [t.encode_ms for t in timings]
    prefill_times = [t.prefill_ms for t in timings]
    decode_times = [t.decode_ms for t in timings]
    action_times = [t.action_ms for t in timings]

    ax.stackplot(steps, encode_times, prefill_times, decode_times, action_times,
                 labels=['Encode', 'Prefill', 'Decode', 'Action'],
                 colors=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'],
                 alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Phase Breakdown over Steps')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    # 3. Decode tokens per step
    ax = axes[1, 0]
    decode_tokens = [t.decode_token_count for t in timings]
    ax.bar(steps, decode_tokens, color='#9b59b6', alpha=0.7)
    ax.axhline(y=np.mean(decode_tokens), color='r', linestyle='--',
               label=f'Mean: {np.mean(decode_tokens):.1f} tokens')
    ax.set_xlabel('Step')
    ax.set_ylabel('Token Count')
    ax.set_title('Decode Tokens per Step')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 4. Per-token decode time
    ax = axes[1, 1]
    per_token_times = [t.decode_per_token_ms for t in timings]
    ax.plot(steps, per_token_times, 'g-o', markersize=4)
    ax.axhline(y=np.mean(per_token_times), color='r', linestyle='--',
               label=f'Mean: {np.mean(per_token_times):.2f}ms/token')
    ax.set_xlabel('Step')
    ax.set_ylabel('Time (ms/token)')
    ax.set_title('Decode Time per Token')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"timing_trace_{compile_mode}_{timestamp}.png")
    plt.savefig(plot_file, dpi=150)
    logger.info(f"Timing trace plot saved to {plot_file}")
    plt.close()


def plot_comparison(results: dict, output_dir: str, timestamp: str):
    """Generate comparison plots between compiled and non-compiled."""
    compiled_stats = results["compiled"]["streaming_stats"]
    non_compiled_stats = results["non_compiled"]["streaming_stats"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Compiled vs Non-Compiled Performance Comparison', fontsize=14, fontweight='bold')

    # 1. Side-by-side bar chart
    ax = axes[0]
    phases = ['Total', 'Encode', 'Prefill', 'Decode', 'Action']
    compiled_vals = [
        compiled_stats['total_mean_ms'],
        compiled_stats['encode_mean_ms'],
        compiled_stats['prefill_mean_ms'],
        compiled_stats['decode_mean_ms'],
        compiled_stats['action_mean_ms'],
    ]
    non_compiled_vals = [
        non_compiled_stats['total_mean_ms'],
        non_compiled_stats['encode_mean_ms'],
        non_compiled_stats['prefill_mean_ms'],
        non_compiled_stats['decode_mean_ms'],
        non_compiled_stats['action_mean_ms'],
    ]

    x = np.arange(len(phases))
    width = 0.35

    ax.bar(x - width/2, compiled_vals, width, label='Compiled', color='steelblue')
    ax.bar(x + width/2, non_compiled_vals, width, label='Non-Compiled', color='coral')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Mean Latency Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Speedup bar chart
    ax = axes[1]
    speedups = [nc / c if c > 0 else 0 for c, nc in zip(compiled_vals, non_compiled_vals)]
    colors = ['green' if s > 1 else 'red' for s in speedups]

    bars = ax.bar(phases, speedups, color=colors, alpha=0.7)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2)
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Speedup from Compilation')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=10)

    # 3. Stacked comparison
    ax = axes[2]
    categories = ['Compiled', 'Non-Compiled']

    phases_data = ['Encode', 'Prefill', 'Decode', 'Action']
    compiled_breakdown = [
        compiled_stats['encode_mean_ms'],
        compiled_stats['prefill_mean_ms'],
        compiled_stats['decode_mean_ms'],
        compiled_stats['action_mean_ms'],
    ]
    non_compiled_breakdown = [
        non_compiled_stats['encode_mean_ms'],
        non_compiled_stats['prefill_mean_ms'],
        non_compiled_stats['decode_mean_ms'],
        non_compiled_stats['action_mean_ms'],
    ]

    x = np.arange(len(categories))
    width = 0.5
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

    bottom_compiled = 0
    bottom_non_compiled = 0
    for i, (phase, color) in enumerate(zip(phases_data, colors)):
        ax.bar(x[0], compiled_breakdown[i], width, bottom=bottom_compiled,
               color=color, label=phase if i == 0 else "")
        ax.bar(x[1], non_compiled_breakdown[i], width, bottom=bottom_non_compiled,
               color=color)
        bottom_compiled += compiled_breakdown[i]
        bottom_non_compiled += non_compiled_breakdown[i]

    ax.set_ylabel('Time (ms)')
    ax.set_title('Time Breakdown Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(phases_data, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Add total annotations
    ax.text(0, bottom_compiled + 5, f'{bottom_compiled:.1f}ms', ha='center', fontweight='bold')
    ax.text(1, bottom_non_compiled + 5, f'{bottom_non_compiled:.1f}ms', ha='center', fontweight='bold')

    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"comparison_{timestamp}.png")
    plt.savefig(plot_file, dpi=150)
    logger.info(f"Comparison plot saved to {plot_file}")
    plt.close()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark unified model with latency breakdown (supports streaming/non-streaming modes)"
    )
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
        "--num_steps",
        type=int,
        default=15,
        help="Number of streaming steps to benchmark (excluding warmup)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=3,
        help="Number of warmup steps for torch.compile",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison between compiled and non-compiled",
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile (for baseline comparison)",
    )
    parser.add_argument(
        "--non_streaming",
        action="store_true",
        help="Use non-streaming mode (default is streaming mode)",
    )
    parser.add_argument(
        "--fuse_qkv",
        action="store_true",
        help="Fuse q/k/v projections into single QKVLinear",
    )
    parser.add_argument(
        "--fuse_gate_up",
        action="store_true",
        help="Fuse gate/up projections into single MergedColumnLinear",
    )

    args = parser.parse_args()
    streaming = not args.non_streaming

    if args.compare:
        results = run_comparison_benchmark(
            model_path=args.model_path,
            clip_id=args.clip_id,
            num_steps=args.num_steps,
            warmup_steps=args.warmup_steps,
            streaming=streaming,
            fuse_qkv=args.fuse_qkv,
            fuse_gate_up=args.fuse_gate_up,
            output_dir=args.output_dir,
        )
    else:
        torch_compile = None if args.no_compile else "max-autotune"
        results = run_compiled_benchmark(
            model_path=args.model_path,
            clip_id=args.clip_id,
            num_steps=args.num_steps,
            warmup_steps=args.warmup_steps,
            torch_compile=torch_compile,
            streaming=streaming,
            fuse_qkv=args.fuse_qkv,
            fuse_gate_up=args.fuse_gate_up,
            output_dir=args.output_dir,
        )

    logger.info("Benchmark completed!")


if __name__ == "__main__":
    main()
