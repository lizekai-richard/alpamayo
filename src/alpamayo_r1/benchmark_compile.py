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
Unified benchmark for AlpamayoR1 inference with per-phase timing breakdown.

Settings:
  - alpamayo: Original model (HF generate, no compile)
  - alpamayo_sysopt: Non-streaming + torch.compile + QKV/MLP fusion
  - alpamayo_sysopt_streaming: Streaming + torch.compile + QKV/MLP fusion
  - alpamayo_sysopt_pruning: Non-streaming + torch.compile + QKV/MLP fusion + token pruning
  - alpamayo_sysopt_streaming_pruning: Streaming + torch.compile + QKV/MLP fusion + token pruning

Usage:
  python -m alpamayo_r1.benchmark_compile --setting alpamayo --clip_file clips.json --num_samples 1
  torchrun --nproc_per_node=8 -m alpamayo_r1.benchmark_compile \
      --setting alpamayo --clip_file clips.json --num_samples 1
"""

import argparse
import copy
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.distributed as dist

torch.set_float32_matmul_precision("high")

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

_rank = int(os.environ.get("RANK", 0))
logging.basicConfig(
    level=logging.INFO if _rank == 0 else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _is_torchrun() -> bool:
    """Auto-detect distributed launch (torchrun sets WORLD_SIZE)."""
    return int(os.environ.get("WORLD_SIZE", 1)) > 1


# =============================================================================
# Timing
# =============================================================================

class PerfTimer:
    """perf_counter timer with CUDA synchronization."""

    def __init__(self):
        self.elapsed_s = 0.0

    def __enter__(self):
        torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize()
        self.elapsed_s = time.perf_counter() - self._start


@dataclass
class StepResult:
    """Timing for one inference step."""
    step_idx: int
    encode_s: float = 0.0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    decode_token_count: int = 0
    action_s: float = 0.0
    total_s: float = 0.0

    @property
    def decode_s_per_token(self):
        return self.decode_s / self.decode_token_count if self.decode_token_count > 0 else 0.0


# =============================================================================
# Data Loading
# =============================================================================

def create_non_streaming_inputs(processor, num_windows, clip_id,
                                t0_us=2_000_000, time_step_us=100_000):
    """Create inputs for non-streaming inference (16 frames per step)."""
    inputs = []
    for i in range(num_windows):
        current_t0 = t0_us + i * time_step_us
        data = load_physical_aiavdataset(clip_id, t0_us=current_t0, num_frames=4)
        frames = data["image_frames"].flatten(0, 1)  # (4,4,C,H,W) -> (16,C,H,W)
        messages = helper.create_message(frames)
        tokenized = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt",
        )
        inputs.append({
            "tokenized_data": tokenized,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
            "ego_future_xyz": data["ego_future_xyz"],
            "ego_future_rot": data["ego_future_rot"],
        })
    return inputs


def create_streaming_inputs(processor, num_windows, clip_id,
                            t0_us=2_000_000, time_step_us=100_000):
    """Create inputs for streaming inference (16 frames first, 4 frames after)."""
    inputs = []
    for i in range(num_windows):
        current_t0 = t0_us + i * time_step_us
        if i == 0:
            data = load_physical_aiavdataset(clip_id, t0_us=current_t0, num_frames=4)
        else:
            data = load_physical_aiavdataset(clip_id, t0_us=current_t0, num_frames=1)
        frames = data["image_frames"].flatten(0, 1)
        messages = helper.create_message(frames)
        tokenized = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt",
        )
        inputs.append({
            "tokenized_data": tokenized,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
            "ego_future_xyz": data["ego_future_xyz"],
            "ego_future_rot": data["ego_future_rot"],
            "is_prefill": (i == 0),
        })
    return inputs


# =============================================================================
# Setting: alpamayo (original model with forward hooks)
# =============================================================================

class InstrumentedAlpamayo:
    """Original AlpamayoR1 instrumented via forward hooks for timing breakdown.

    Hooks:
      - vlm.model.visual: encode time
      - vlm forward: first call = encode+prefill, subsequent = decode
      - diffusion.sample wrapper: action time
    """

    def __init__(self, model_path, dtype=torch.bfloat16):
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
        logger.info("Loading original AlpamayoR1 model...")
        self.model = AlpamayoR1.from_pretrained(model_path, dtype=dtype).to("cuda")
        self.processor = helper.get_processor(self.model.tokenizer)
        self._setup_hooks()
        logger.info("Original model loaded with hooks.")

    def _setup_hooks(self):
        self._encode_time = 0.0
        self._vlm_call_times = []
        self._action_time = 0.0
        wrapper = self

        def visual_pre_hook(module, args):
            torch.cuda.synchronize()
            wrapper._encode_start = time.perf_counter()

        def visual_post_hook(module, args, output):
            torch.cuda.synchronize()
            wrapper._encode_time = time.perf_counter() - wrapper._encode_start

        self.model.vlm.model.visual.register_forward_pre_hook(visual_pre_hook)
        self.model.vlm.model.visual.register_forward_hook(visual_post_hook)

        def vlm_pre_hook(module, args, kwargs):
            torch.cuda.synchronize()
            wrapper._vlm_call_start = time.perf_counter()

        def vlm_post_hook(module, args, kwargs, output):
            torch.cuda.synchronize()
            wrapper._vlm_call_times.append(time.perf_counter() - wrapper._vlm_call_start)

        self.model.vlm.register_forward_pre_hook(vlm_pre_hook, with_kwargs=True)
        self.model.vlm.register_forward_hook(vlm_post_hook, with_kwargs=True)

        orig_sample = self.model.diffusion.sample

        def instrumented_sample(*args, **kwargs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = orig_sample(*args, **kwargs)
            torch.cuda.synchronize()
            wrapper._action_time = time.perf_counter() - t0
            return result

        self.model.diffusion.sample = instrumented_sample

    def _reset_timing(self):
        self._encode_time = 0.0
        self._vlm_call_times = []
        self._action_time = 0.0

    @torch.inference_mode()
    def run_step(self, model_inputs, step_idx, num_traj_samples):
        self._reset_timing()
        data = helper.to_device(copy.deepcopy(model_inputs), "cuda")

        with PerfTimer() as total_timer:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = self.model.sample_trajectories_from_data_with_vlm_rollout(
                    data=data,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=num_traj_samples,
                    max_generation_length=256,
                    return_extra=True,
                )

        encode_s = self._encode_time
        prefill_s = (self._vlm_call_times[0] - encode_s) if self._vlm_call_times else 0.0
        decode_s = sum(self._vlm_call_times[1:]) if len(self._vlm_call_times) > 1 else 0.0
        decode_tokens = max(len(self._vlm_call_times) - 1, 0)

        return StepResult(
            step_idx=step_idx,
            encode_s=encode_s,
            prefill_s=prefill_s,
            decode_s=decode_s,
            decode_token_count=decode_tokens,
            action_s=self._action_time,
            total_s=total_timer.elapsed_s,
        ), result


# =============================================================================
# Setting: alpamayo_sysopt (compile model, monkey-patched)
# =============================================================================

class InstrumentedAlpamayoSysopt:
    """Non-streaming compile model instrumented via monkey-patching _encode/_prefill/_decode/_action."""

    def __init__(self, model_path, sparsity_ratio=0.0, dtype=torch.bfloat16):
        from alpamayo_r1.models.alpamayo_r1_compile import AlpamayoR1
        logger.info("Loading AlpamayoR1 (compile) model...")
        self.model = AlpamayoR1.from_pretrained(model_path, dtype=dtype).to("cuda")
        self.processor = helper.get_processor(self.model.tokenizer)
        self.sparsity_ratio = sparsity_ratio
        self._instrument()
        logger.info(f"Compile model loaded and instrumented (sparsity_ratio={sparsity_ratio}).")

    def _instrument(self):
        self._timing = {
            "encode_s": 0.0, "prefill_s": 0.0,
            "decode_s": 0.0, "decode_count": 0,
            "action_s": 0.0,
        }
        wrapper = self
        model = self.model
        orig_encode = model._encode
        orig_prefill = model._prefill
        orig_decode = model._decode
        orig_action = model._action

        def instr_encode(*a, **kw):
            with PerfTimer() as t:
                r = orig_encode(*a, **kw)
            wrapper._timing["encode_s"] += t.elapsed_s
            return r

        def instr_prefill(*a, **kw):
            with PerfTimer() as t:
                r = orig_prefill(*a, **kw)
            wrapper._timing["prefill_s"] += t.elapsed_s
            return r

        def instr_decode(*a, **kw):
            with PerfTimer() as t:
                r = orig_decode(*a, **kw)
            wrapper._timing["decode_s"] += t.elapsed_s
            wrapper._timing["decode_count"] += 1
            return r

        def instr_action(*a, **kw):
            with PerfTimer() as t:
                r = orig_action(*a, **kw)
            wrapper._timing["action_s"] += t.elapsed_s
            return r

        model._encode = instr_encode
        model._prefill = instr_prefill
        model._decode = instr_decode
        model._action = instr_action

    def _reset_timing(self):
        self._timing = {
            "encode_s": 0.0, "prefill_s": 0.0,
            "decode_s": 0.0, "decode_count": 0,
            "action_s": 0.0,
        }

    @torch.inference_mode()
    def run_step(self, model_inputs, step_idx, num_traj_samples):
        self._reset_timing()
        data = helper.to_device(copy.deepcopy(model_inputs), "cuda")

        with PerfTimer() as total_timer:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = self.model.sample_trajectories_from_data_with_vlm_rollout(
                    data=data,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=num_traj_samples,
                    max_generation_length=256,
                    return_extra=True,
                    torch_compile="max-autotune",
                    fuse_qkv=True,
                    fuse_gate_up=True,
                    sparsity_ratio=self.sparsity_ratio,
                )

        return StepResult(
            step_idx=step_idx,
            encode_s=self._timing["encode_s"],
            prefill_s=self._timing["prefill_s"],
            decode_s=self._timing["decode_s"],
            decode_token_count=self._timing["decode_count"],
            action_s=self._timing["action_s"],
            total_s=total_timer.elapsed_s,
        ), result


# =============================================================================
# Setting: alpamayo_sysopt_streaming (streaming compile model, monkey-patched)
# =============================================================================

class InstrumentedAlpamayoSysoptStreaming:
    """Streaming compile model instrumented via monkey-patching _encode/_prefill/_decode/_action."""

    def __init__(self, model_path, sparsity_ratio=0.0, dtype=torch.bfloat16):
        from alpamayo_r1.models.alpamayo_r1_streaming_compile import StreamingAlpamayoR1
        logger.info("Loading StreamingAlpamayoR1 model...")
        self.model = StreamingAlpamayoR1.from_pretrained(model_path, dtype=dtype).to("cuda")
        self.processor = helper.get_processor(self.model.tokenizer)
        self.sparsity_ratio = sparsity_ratio
        self._instrument()
        logger.info(f"Streaming model loaded and instrumented (sparsity_ratio={sparsity_ratio}).")

    def _instrument(self):
        self._timing = {
            "encode_s": 0.0, "prefill_s": 0.0,
            "decode_s": 0.0, "decode_count": 0,
            "action_s": 0.0,
        }
        wrapper = self
        model = self.model
        orig_encode = model._encode
        orig_prefill = model._prefill
        orig_decode = model._decode
        orig_action = model._action

        def instr_encode(*a, **kw):
            with PerfTimer() as t:
                r = orig_encode(*a, **kw)
            wrapper._timing["encode_s"] += t.elapsed_s
            return r

        def instr_prefill(*a, **kw):
            with PerfTimer() as t:
                r = orig_prefill(*a, **kw)
            wrapper._timing["prefill_s"] += t.elapsed_s
            return r

        def instr_decode(*a, **kw):
            with PerfTimer() as t:
                r = orig_decode(*a, **kw)
            wrapper._timing["decode_s"] += t.elapsed_s
            wrapper._timing["decode_count"] += 1
            return r

        def instr_action(*a, **kw):
            with PerfTimer() as t:
                r = orig_action(*a, **kw)
            wrapper._timing["action_s"] += t.elapsed_s
            return r

        model._encode = instr_encode
        model._prefill = instr_prefill
        model._decode = instr_decode
        model._action = instr_action

    def _reset_timing(self):
        self._timing = {
            "encode_s": 0.0, "prefill_s": 0.0,
            "decode_s": 0.0, "decode_count": 0,
            "action_s": 0.0,
        }

    @torch.inference_mode()
    def run_step(self, model_inputs, step_idx, num_traj_samples):
        self._reset_timing()
        data = helper.to_device(copy.deepcopy(model_inputs), "cuda")
        data.pop("is_prefill", None)

        with PerfTimer() as total_timer:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = self.model.sample_trajectories_from_data_with_streaming_vlm_rollout(
                    data=data,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=num_traj_samples,
                    max_generation_length=256,
                    return_extra=True,
                    torch_compile="max-autotune",
                    fuse_qkv=True,
                    fuse_gate_up=True,
                    sparsity_ratio=self.sparsity_ratio,
                )

        # First prefill returns None (no phase breakdown)
        if result is None:
            return StepResult(step_idx=step_idx, total_s=total_timer.elapsed_s), None

        return StepResult(
            step_idx=step_idx,
            encode_s=self._timing["encode_s"],
            prefill_s=self._timing["prefill_s"],
            decode_s=self._timing["decode_s"],
            decode_token_count=self._timing["decode_count"],
            action_s=self._timing["action_s"],
            total_s=total_timer.elapsed_s,
        ), result


# =============================================================================
# Benchmark Runner
# =============================================================================

def _benchmark_one_clip(wrapper, processor, clip_id, args, is_streaming):
    """Run warmup + benchmark for a single clip. Returns per-step results list."""
    num_samples = args.num_samples

    if args.dumped_data_dir:
        # Load pre-dumped sliding window inputs (non-streaming, 16 frames each)
        total_windows = args.warmup_steps + args.num_steps
        dumped = helper.load_dumped_inputs(args.dumped_data_dir, clip_id)
        all_inputs = dumped[:total_windows]
    elif is_streaming:
        total_windows = 1 + args.warmup_steps + args.num_steps
        all_inputs = create_streaming_inputs(
            processor, total_windows, clip_id, args.t0_us, args.time_step_us,
        )
    else:
        total_windows = args.warmup_steps + args.num_steps
        all_inputs = create_non_streaming_inputs(
            processor, total_windows, clip_id, args.t0_us, args.time_step_us,
        )

    logger.info(f"Clip {clip_id}: warmup={args.warmup_steps}, steps={args.num_steps}")

    first_prefill_s = None
    step_offset = 0

    # Streaming: first prefill (no output)
    if is_streaming:
        timing, _ = wrapper.run_step(all_inputs[0], 0, num_samples)
        first_prefill_s = timing.total_s
        logger.info(f"  [FIRST PREFILL] {timing.total_s:.3f}s")
        step_offset = 1

    # Warmup
    for i in range(args.warmup_steps):
        idx = step_offset + i
        timing, _ = wrapper.run_step(all_inputs[idx], idx, num_samples)
        logger.info(
            f"  [WARMUP {i}] Total: {timing.total_s:.3f}s | "
            f"E: {timing.encode_s:.3f}s P: {timing.prefill_s:.3f}s "
            f"D: {timing.decode_s:.3f}s ({timing.decode_token_count}tok) "
            f"A: {timing.action_s:.3f}s"
        )

    # Benchmark
    benchmark_results = []
    for i in range(args.num_steps):
        idx = step_offset + args.warmup_steps + i
        timing, result = wrapper.run_step(all_inputs[idx], idx, num_samples)
        benchmark_results.append(timing)
        logger.info(
            f"  [STEP {i:2d}] Total: {timing.total_s:.3f}s | "
            f"E: {timing.encode_s:.3f}s P: {timing.prefill_s:.3f}s "
            f"D: {timing.decode_s:.3f}s ({timing.decode_token_count}tok, "
            f"{timing.decode_s_per_token*1000:.1f}ms/tok) "
            f"A: {timing.action_s:.3f}s"
        )

    return {
        "clip_id": clip_id,
        "first_prefill_s": first_prefill_s,
        "per_step": [asdict(r) for r in benchmark_results],
    }


def run_benchmark(args):
    setting = args.setting
    num_samples = args.num_samples

    # --- DDP setup ---
    distributed = _is_torchrun()
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        rank = 0
        world_size = 1

    # --- Load clip list ---
    with open(args.clip_file, "r") as f:
        all_clip_ids = json.load(f)
    my_clip_ids = all_clip_ids[rank::world_size]
    if rank == 0:
        logger.info(f"Total clips: {len(all_clip_ids)}, world_size: {world_size}")
    logger.info(f"Rank {rank}: assigned {len(my_clip_ids)} clips")

    # --- Load model ---
    if setting == "alpamayo":
        wrapper = InstrumentedAlpamayo(args.model_path)
    elif setting in ("alpamayo_sysopt", "alpamayo_sysopt_pruning"):
        sparsity = args.sparsity_ratio if setting == "alpamayo_sysopt_pruning" else 0.0
        wrapper = InstrumentedAlpamayoSysopt(args.model_path, sparsity_ratio=sparsity)
    elif setting in ("alpamayo_sysopt_streaming", "alpamayo_sysopt_streaming_pruning"):
        sparsity = args.sparsity_ratio if setting == "alpamayo_sysopt_streaming_pruning" else 0.0
        wrapper = InstrumentedAlpamayoSysoptStreaming(args.model_path, sparsity_ratio=sparsity)
    else:
        raise ValueError(f"Unknown setting: {setting}")

    processor = wrapper.processor
    is_streaming = setting in ("alpamayo_sysopt_streaming", "alpamayo_sysopt_streaming_pruning")
    sparsity_ratio = getattr(wrapper, "sparsity_ratio", 0.0)

    if rank == 0:
        logger.info(f"{'='*70}")
        logger.info(f"BENCHMARK: {setting} | num_traj_samples={num_samples} | sparsity_ratio={sparsity_ratio}")
        logger.info(f"{'='*70}")

    # --- Run benchmark for each assigned clip ---
    local_results = []
    for clip_id in my_clip_ids:
        clip_result = _benchmark_one_clip(wrapper, processor, clip_id, args, is_streaming)
        local_results.append(clip_result)

    # --- Gather results to rank 0 ---
    if distributed:
        dist.barrier()
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(local_results, gathered, dst=0)
    else:
        gathered = [local_results]

    # --- Rank 0: merge and save ---
    if rank == 0:
        all_clip_results = []
        for rank_results in gathered:
            all_clip_results.extend(rank_results)

        # Aggregate stats across all clips
        encode_vals, prefill_vals, decode_per_tok, action_vals, total_vals = [], [], [], [], []
        for clip_res in all_clip_results:
            for step in clip_res["per_step"]:
                total_vals.append(step["total_s"])
                encode_vals.append(step["encode_s"])
                prefill_vals.append(step["prefill_s"])
                action_vals.append(step["action_s"])
                if step["decode_token_count"] > 0:
                    decode_per_tok.append(step["decode_s"] / step["decode_token_count"])

        logger.info(f"\n{'='*70}")
        logger.info(f"SUMMARY: {setting} | num_traj_samples={num_samples} | sparsity_ratio={sparsity_ratio}")
        logger.info(f"{'='*70}")
        logger.info(f"Total clips: {len(all_clip_results)}")
        logger.info(f"Encode  (s):     {np.mean(encode_vals):.4f} +/- {np.std(encode_vals):.4f}")
        logger.info(f"Prefill (s):     {np.mean(prefill_vals):.4f} +/- {np.std(prefill_vals):.4f}")
        if decode_per_tok:
            logger.info(f"Decode  (s/tok): {np.mean(decode_per_tok):.5f} +/- {np.std(decode_per_tok):.5f}")
        logger.info(f"Action  (s):     {np.mean(action_vals):.4f} +/- {np.std(action_vals):.4f}")
        logger.info(f"Total   (s):     {np.mean(total_vals):.4f} +/- {np.std(total_vals):.4f}")

        os.makedirs(args.output_dir, exist_ok=True)
        output = {
            "setting": setting,
            "num_traj_samples": num_samples,
            "sparsity_ratio": sparsity_ratio,
            "torch_compile": "max-autotune" if setting != "alpamayo" else None,
            "fuse_qkv": setting != "alpamayo",
            "fuse_gate_up": setting != "alpamayo",
            "warmup_steps": args.warmup_steps,
            "num_steps": args.num_steps,
            "num_clips": len(all_clip_results),
            "breakdown": {
                "encode_s": {"mean": float(np.mean(encode_vals)),
                             "std": float(np.std(encode_vals))},
                "prefill_s": {"mean": float(np.mean(prefill_vals)),
                              "std": float(np.std(prefill_vals))},
                "decode_s_per_token": {"mean": float(np.mean(decode_per_tok)) if decode_per_tok else 0.0,
                                       "std": float(np.std(decode_per_tok)) if decode_per_tok else 0.0},
                "action_s": {"mean": float(np.mean(action_vals)),
                             "std": float(np.std(action_vals))},
            },
            "total_inference_s": {
                "mean": float(np.mean(total_vals)),
                "std": float(np.std(total_vals)),
            },
            "per_clip": all_clip_results,
        }

        if sparsity_ratio > 0:
            filename = f"{setting}_n{num_samples}_s{sparsity_ratio}.json"
        else:
            filename = f"{setting}_n{num_samples}.json"
        filepath = os.path.join(args.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"\nResults saved to {filepath}")

    if distributed:
        dist.destroy_process_group()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark AlpamayoR1 inference across settings"
    )
    parser.add_argument(
        "--setting", type=str, required=True,
        choices=[
            "alpamayo", "alpamayo_sysopt", "alpamayo_sysopt_streaming",
            "alpamayo_sysopt_pruning", "alpamayo_sysopt_streaming_pruning",
        ],
    )
    parser.add_argument("--num_samples", type=int, default=1,
                        help="num_traj_samples")
    parser.add_argument("--sparsity_ratio", type=float, default=0.5,
                        help="Fraction of image tokens to prune (only for *_pruning settings)")
    parser.add_argument("--model_path", type=str, default="./Alpamayo-R1-10B")
    parser.add_argument("--clip_file", type=str, default="clips.json",
                        help="JSON file containing a list of clip IDs")
    parser.add_argument("--dumped_data_dir", type=str, default=None,
                        help="Directory with pre-dumped sliding window inputs (skips live data loading)")
    parser.add_argument("--t0_us", type=int, default=2_000_000)
    parser.add_argument("--time_step_us", type=int, default=100_000)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--num_steps", type=int, default=12)
    parser.add_argument("--output_dir", type=str, default="./benchmark_results")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
