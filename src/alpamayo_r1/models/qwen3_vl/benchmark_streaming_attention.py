# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark script to compare SDPA vs Flex Attention for streaming VLM.

Usage:
    python benchmark_streaming_attention.py
"""

import time
import torch
import torch.nn.functional as F
from typing import Callable

# Check Flex Attention availability
FLEX_AVAILABLE = False
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_AVAILABLE = True
except ImportError:
    print("Flex Attention not available (requires PyTorch >= 2.5)")

from modeling_qwen3_vl_streaming import (
    create_streaming_attention_mask_sdpa,
    create_streaming_attention_mask_sdpa_optimized,
)

if FLEX_AVAILABLE:
    from modeling_qwen3_vl_streaming import (
        create_streaming_attention_mask_flex,
        create_streaming_mask_mod_factory,
        _build_streaming_position_mappings,
    )


def create_dummy_ranges(
    num_views: int = 4,
    num_frames_per_view: int = 4,
    tokens_per_frame: int = 180,
    system_tokens: int = 50,
    traj_text_tokens: int = 200,
):
    """Create dummy vision_start_end_ids_ranges and traj_and_text_ids_range for testing."""
    vision_start_end_ids_ranges = []
    current_pos = system_tokens

    for view_idx in range(num_views):
        view_frames = []
        for frame_idx in range(num_frames_per_view):
            # Each frame: <vision_start> + image_tokens + <vision_end>
            # Simplified: just use tokens_per_frame for the whole thing
            start = current_pos
            end = current_pos + tokens_per_frame
            view_frames.append((start, end))
            current_pos = end
        vision_start_end_ids_ranges.append(view_frames)

    traj_and_text_ids_range = (current_pos, current_pos + traj_text_tokens)

    return vision_start_end_ids_ranges, traj_and_text_ids_range


def compute_query_length(vision_start_end_ids_ranges, traj_and_text_ids_range):
    """Compute query length (last frame of each view + traj_text)."""
    query_length = 0
    for view_frames in vision_start_end_ids_ranges:
        last_frame_start, last_frame_end = view_frames[-1]
        query_length += last_frame_end - last_frame_start
    traj_start, traj_end = traj_and_text_ids_range
    query_length += traj_end - traj_start
    return query_length


def benchmark_mask_creation(
    num_views: int = 4,
    num_frames_per_view: int = 4,
    tokens_per_frame: int = 180,
    system_tokens: int = 50,
    traj_text_tokens: int = 200,
    batch_size: int = 1,
    num_warmup: int = 5,
    num_iterations: int = 20,
    device: str = "cuda",
):
    """Benchmark mask creation time for SDPA and Flex Attention."""
    print("=" * 70)
    print("MASK CREATION BENCHMARK")
    print("=" * 70)

    # Create dummy ranges
    vision_start_end_ids_ranges, traj_and_text_ids_range = create_dummy_ranges(
        num_views, num_frames_per_view, tokens_per_frame, system_tokens, traj_text_tokens
    )

    # Compute dimensions
    kv_length = traj_and_text_ids_range[1]  # Total sequence length
    query_length = compute_query_length(vision_start_end_ids_ranges, traj_and_text_ids_range)
    cache_position = torch.arange(query_length, device=device)

    print(f"\nConfiguration:")
    print(f"  num_views: {num_views}")
    print(f"  num_frames_per_view: {num_frames_per_view}")
    print(f"  tokens_per_frame: {tokens_per_frame}")
    print(f"  system_tokens: {system_tokens}")
    print(f"  traj_text_tokens: {traj_text_tokens}")
    print(f"  kv_length: {kv_length}")
    print(f"  query_length: {query_length}")
    print(f"  batch_size: {batch_size}")
    print()

    results = {}

    # Benchmark SDPA (basic)
    print("Benchmarking SDPA (basic)...")
    torch.cuda.synchronize()
    for _ in range(num_warmup):
        _ = create_streaming_attention_mask_sdpa(
            batch_size=batch_size,
            cache_position=cache_position,
            kv_length=kv_length,
            vision_start_end_ids_ranges=vision_start_end_ids_ranges,
            traj_and_text_ids_range=traj_and_text_ids_range,
            device=device,
            dtype=torch.float32,
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iterations):
        mask_sdpa = create_streaming_attention_mask_sdpa(
            batch_size=batch_size,
            cache_position=cache_position,
            kv_length=kv_length,
            vision_start_end_ids_ranges=vision_start_end_ids_ranges,
            traj_and_text_ids_range=traj_and_text_ids_range,
            device=device,
            dtype=torch.float32,
        )
    torch.cuda.synchronize()
    sdpa_time = (time.perf_counter() - start) / num_iterations * 1000
    results["SDPA (basic)"] = sdpa_time
    print(f"  Average time: {sdpa_time:.3f} ms")

    # Benchmark SDPA (optimized)
    print("Benchmarking SDPA (optimized)...")
    torch.cuda.synchronize()
    for _ in range(num_warmup):
        _ = create_streaming_attention_mask_sdpa_optimized(
            batch_size=batch_size,
            cache_position=cache_position,
            kv_length=kv_length,
            vision_start_end_ids_ranges=vision_start_end_ids_ranges,
            traj_and_text_ids_range=traj_and_text_ids_range,
            valid_length=kv_length,
            device=device,
            dtype=torch.float32,
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iterations):
        mask_sdpa_opt = create_streaming_attention_mask_sdpa_optimized(
            batch_size=batch_size,
            cache_position=cache_position,
            kv_length=kv_length,
            vision_start_end_ids_ranges=vision_start_end_ids_ranges,
            traj_and_text_ids_range=traj_and_text_ids_range,
            valid_length=kv_length,
            device=device,
            dtype=torch.float32,
        )
    torch.cuda.synchronize()
    sdpa_opt_time = (time.perf_counter() - start) / num_iterations * 1000
    results["SDPA (optimized)"] = sdpa_opt_time
    print(f"  Average time: {sdpa_opt_time:.3f} ms")

    # Benchmark Flex Attention
    if FLEX_AVAILABLE:
        print("Benchmarking Flex Attention BlockMask creation...")
        torch.cuda.synchronize()
        for _ in range(num_warmup):
            _ = create_streaming_attention_mask_flex(
                batch_size=batch_size,
                cache_position=cache_position,
                kv_length=kv_length,
                vision_start_end_ids_ranges=vision_start_end_ids_ranges,
                traj_and_text_ids_range=traj_and_text_ids_range,
                device=device,
            )
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iterations):
            block_mask = create_streaming_attention_mask_flex(
                batch_size=batch_size,
                cache_position=cache_position,
                kv_length=kv_length,
                vision_start_end_ids_ranges=vision_start_end_ids_ranges,
                traj_and_text_ids_range=traj_and_text_ids_range,
                device=device,
            )
        torch.cuda.synchronize()
        flex_time = (time.perf_counter() - start) / num_iterations * 1000
        results["Flex Attention"] = flex_time
        print(f"  Average time: {flex_time:.3f} ms")
    else:
        print("Flex Attention not available, skipping...")

    # Memory comparison
    print("\nMemory usage:")
    mask_sdpa = create_streaming_attention_mask_sdpa(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        vision_start_end_ids_ranges=vision_start_end_ids_ranges,
        traj_and_text_ids_range=traj_and_text_ids_range,
        device=device,
        dtype=torch.float32,
    )
    sdpa_memory = mask_sdpa.numel() * mask_sdpa.element_size() / 1024 / 1024
    print(f"  SDPA 4D mask: {sdpa_memory:.2f} MB ({mask_sdpa.shape})")

    if FLEX_AVAILABLE:
        block_mask = create_streaming_attention_mask_flex(
            batch_size=batch_size,
            cache_position=cache_position,
            kv_length=kv_length,
            vision_start_end_ids_ranges=vision_start_end_ids_ranges,
            traj_and_text_ids_range=traj_and_text_ids_range,
            device=device,
        )
        # BlockMask is a compressed representation, estimate its size
        print(f"  Flex BlockMask: (compressed representation)")

    return results, mask_sdpa, block_mask if FLEX_AVAILABLE else None


def benchmark_attention_computation(
    num_views: int = 4,
    num_frames_per_view: int = 4,
    tokens_per_frame: int = 180,
    system_tokens: int = 50,
    traj_text_tokens: int = 200,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 128,
    num_warmup: int = 5,
    num_iterations: int = 20,
    device: str = "cuda",
):
    """Benchmark actual attention computation time."""
    print("\n" + "=" * 70)
    print("ATTENTION COMPUTATION BENCHMARK")
    print("=" * 70)

    # Create dummy ranges
    vision_start_end_ids_ranges, traj_and_text_ids_range = create_dummy_ranges(
        num_views, num_frames_per_view, tokens_per_frame, system_tokens, traj_text_tokens
    )

    # Compute dimensions
    kv_length = traj_and_text_ids_range[1]
    query_length = compute_query_length(vision_start_end_ids_ranges, traj_and_text_ids_range)
    cache_position = torch.arange(query_length, device=device)

    print(f"\nConfiguration:")
    print(f"  query_length: {query_length}")
    print(f"  kv_length: {kv_length}")
    print(f"  batch_size: {batch_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {head_dim}")
    print()

    # Create Q, K, V tensors
    q = torch.randn(batch_size, num_heads, query_length, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, num_heads, kv_length, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch_size, num_heads, kv_length, head_dim, device=device, dtype=torch.bfloat16)

    results = {}

    # Benchmark SDPA with mask
    print("Benchmarking SDPA with 4D mask...")
    mask_sdpa = create_streaming_attention_mask_sdpa(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        vision_start_end_ids_ranges=vision_start_end_ids_ranges,
        traj_and_text_ids_range=traj_and_text_ids_range,
        device=device,
        dtype=torch.bfloat16,
    )

    torch.cuda.synchronize()
    for _ in range(num_warmup):
        _ = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_sdpa)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iterations):
        out_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_sdpa)
    torch.cuda.synchronize()
    sdpa_time = (time.perf_counter() - start) / num_iterations * 1000
    results["SDPA with mask"] = sdpa_time
    print(f"  Average time: {sdpa_time:.3f} ms")

    # Benchmark SDPA without mask (causal only, for reference)
    print("Benchmarking SDPA causal (no custom mask, reference)...")
    torch.cuda.synchronize()
    for _ in range(num_warmup):
        _ = F.scaled_dot_product_attention(q, k[:, :, :query_length, :], v[:, :, :query_length, :], is_causal=True)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iterations):
        out_causal = F.scaled_dot_product_attention(q, k[:, :, :query_length, :], v[:, :, :query_length, :], is_causal=True)
    torch.cuda.synchronize()
    causal_time = (time.perf_counter() - start) / num_iterations * 1000
    results["SDPA causal (reference)"] = causal_time
    print(f"  Average time: {causal_time:.3f} ms")

    # Benchmark Flex Attention
    if FLEX_AVAILABLE:
        print("Benchmarking Flex Attention with BlockMask...")
        block_mask = create_streaming_attention_mask_flex(
            batch_size=batch_size,
            cache_position=cache_position,
            kv_length=kv_length,
            vision_start_end_ids_ranges=vision_start_end_ids_ranges,
            traj_and_text_ids_range=traj_and_text_ids_range,
            device=device,
        )

        # Flex attention expects different tensor layout
        # q: [batch, heads, q_len, head_dim]
        # k, v: [batch, heads, kv_len, head_dim]
        torch.cuda.synchronize()
        for _ in range(num_warmup):
            _ = flex_attention(q, k, v, block_mask=block_mask)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iterations):
            out_flex = flex_attention(q, k, v, block_mask=block_mask)
        torch.cuda.synchronize()
        flex_time = (time.perf_counter() - start) / num_iterations * 1000
        results["Flex Attention"] = flex_time
        print(f"  Average time: {flex_time:.3f} ms")

        # Benchmark Flex Attention compiled
        print("Benchmarking Flex Attention compiled...")
        flex_attention_compiled = torch.compile(flex_attention)

        torch.cuda.synchronize()
        for _ in range(num_warmup):
            _ = flex_attention_compiled(q, k, v, block_mask=block_mask)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iterations):
            out_flex_compiled = flex_attention_compiled(q, k, v, block_mask=block_mask)
        torch.cuda.synchronize()
        flex_compiled_time = (time.perf_counter() - start) / num_iterations * 1000
        results["Flex Attention (compiled)"] = flex_compiled_time
        print(f"  Average time: {flex_compiled_time:.3f} ms")
    else:
        print("Flex Attention not available, skipping...")

    return results


def verify_output_consistency(device: str = "cuda"):
    """Verify that SDPA and Flex Attention produce the same output."""
    print("\n" + "=" * 70)
    print("OUTPUT CONSISTENCY VERIFICATION")
    print("=" * 70)

    if not FLEX_AVAILABLE:
        print("Flex Attention not available, skipping verification...")
        return False

    # Use smaller dimensions for verification
    vision_start_end_ids_ranges, traj_and_text_ids_range = create_dummy_ranges(
        num_views=2,
        num_frames_per_view=2,
        tokens_per_frame=20,
        system_tokens=10,
        traj_text_tokens=30,
    )

    kv_length = traj_and_text_ids_range[1]
    query_length = compute_query_length(vision_start_end_ids_ranges, traj_and_text_ids_range)
    cache_position = torch.arange(query_length, device=device)

    batch_size = 1
    num_heads = 4
    head_dim = 64

    print(f"\nTest configuration:")
    print(f"  query_length: {query_length}")
    print(f"  kv_length: {kv_length}")
    print(f"  batch_size: {batch_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {head_dim}")

    # Create identical Q, K, V tensors
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, query_length, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, kv_length, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, kv_length, head_dim, device=device, dtype=torch.float32)

    # SDPA with mask
    mask_sdpa = create_streaming_attention_mask_sdpa(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        vision_start_end_ids_ranges=vision_start_end_ids_ranges,
        traj_and_text_ids_range=traj_and_text_ids_range,
        device=device,
        dtype=torch.float32,
    )
    out_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_sdpa)

    # Flex Attention with BlockMask
    block_mask = create_streaming_attention_mask_flex(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        vision_start_end_ids_ranges=vision_start_end_ids_ranges,
        traj_and_text_ids_range=traj_and_text_ids_range,
        device=device,
    )
    out_flex = flex_attention(q, k, v, block_mask=block_mask)

    # Compare outputs
    max_diff = (out_sdpa - out_flex).abs().max().item()
    mean_diff = (out_sdpa - out_flex).abs().mean().item()

    # Check relative error
    rel_error = ((out_sdpa - out_flex).abs() / (out_sdpa.abs() + 1e-8)).mean().item()

    print(f"\nOutput comparison:")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Mean relative error: {rel_error:.2e}")

    # Tolerance for floating point comparison
    tolerance = 1e-4
    is_consistent = max_diff < tolerance

    if is_consistent:
        print(f"\n✓ SDPA and Flex Attention outputs are consistent (max diff < {tolerance})")
    else:
        print(f"\n✗ Outputs differ significantly (max diff = {max_diff:.2e} >= {tolerance})")

        # Debug: check a few positions
        print("\nSample values at first few positions:")
        for i in range(min(3, query_length)):
            for j in range(min(3, head_dim)):
                sdpa_val = out_sdpa[0, 0, i, j].item()
                flex_val = out_flex[0, 0, i, j].item()
                diff = abs(sdpa_val - flex_val)
                print(f"  [{i},{j}] SDPA={sdpa_val:.6f}, Flex={flex_val:.6f}, diff={diff:.2e}")

    return is_consistent


def verify_mask_correctness(device: str = "cuda"):
    """Verify that SDPA and Flex Attention masks produce the same pattern."""
    print("\n" + "=" * 70)
    print("MASK CORRECTNESS VERIFICATION")
    print("=" * 70)

    if not FLEX_AVAILABLE:
        print("Flex Attention not available, skipping verification...")
        return

    # Small test case for easy verification
    vision_start_end_ids_ranges, traj_and_text_ids_range = create_dummy_ranges(
        num_views=2,
        num_frames_per_view=2,
        tokens_per_frame=10,
        system_tokens=5,
        traj_text_tokens=15,
    )

    kv_length = traj_and_text_ids_range[1]
    query_length = compute_query_length(vision_start_end_ids_ranges, traj_and_text_ids_range)
    cache_position = torch.arange(query_length, device=device)

    print(f"\nTest configuration:")
    print(f"  kv_length: {kv_length}")
    print(f"  query_length: {query_length}")

    # Create SDPA mask
    mask_sdpa = create_streaming_attention_mask_sdpa(
        batch_size=1,
        cache_position=cache_position,
        kv_length=kv_length,
        vision_start_end_ids_ranges=vision_start_end_ids_ranges,
        traj_and_text_ids_range=traj_and_text_ids_range,
        device=device,
        dtype=torch.float32,
    )

    # Convert SDPA mask to boolean (0 -> True, -inf -> False)
    sdpa_bool = mask_sdpa[0, 0] == 0  # [query_length, kv_length]

    # Create Flex mask and convert to dense for comparison
    (
        kv_region,
        kv_frame,
        kv_local_pos,
        query_region,
        query_frame,
        query_local_pos,
    ) = _build_streaming_position_mappings(
        kv_length=kv_length,
        query_length=query_length,
        vision_start_end_ids_ranges=vision_start_end_ids_ranges,
        traj_and_text_ids_range=traj_and_text_ids_range,
        device=device,
    )

    num_views = len(vision_start_end_ids_ranges)

    # Manually compute Flex mask
    flex_bool = torch.zeros(query_length, kv_length, dtype=torch.bool, device=device)
    for q_idx in range(query_length):
        for kv_idx in range(kv_length):
            q_view = query_region[q_idx].item()
            q_frame_idx = query_frame[q_idx].item()
            q_local = query_local_pos[q_idx].item()

            kv_view = kv_region[kv_idx].item()
            kv_frame_idx = kv_frame[kv_idx].item()
            kv_local = kv_local_pos[kv_idx].item()

            # System always visible
            is_system = kv_view == -1

            is_image_query = q_view < num_views
            is_traj_query = q_view == num_views

            # Image query rules
            earlier_view = is_image_query and (kv_view >= 0) and (kv_view < q_view)
            same_view_earlier_frame = is_image_query and (kv_view == q_view) and (kv_frame_idx < q_frame_idx)
            same_frame_causal = is_image_query and (kv_view == q_view) and (kv_frame_idx == q_frame_idx) and (kv_local <= q_local)

            image_can_attend = earlier_view or same_view_earlier_frame or same_frame_causal

            # Traj query rules
            can_see_all_images = is_traj_query and (kv_view >= 0) and (kv_view < num_views)
            traj_causal = is_traj_query and (kv_view == num_views) and (kv_local <= q_local)

            traj_can_attend = can_see_all_images or traj_causal

            flex_bool[q_idx, kv_idx] = is_system or image_can_attend or traj_can_attend

    # Compare
    match = torch.all(sdpa_bool == flex_bool)

    if match:
        print("\n✓ SDPA and Flex Attention masks are identical!")
    else:
        print("\n✗ Masks differ!")
        diff = sdpa_bool != flex_bool
        diff_count = diff.sum().item()
        print(f"  Number of differing positions: {diff_count}")

        # Show first few differences
        diff_positions = torch.nonzero(diff)
        print("  First few differences (q_idx, kv_idx):")
        for i, (q_idx, kv_idx) in enumerate(diff_positions[:5]):
            print(f"    ({q_idx.item()}, {kv_idx.item()}): SDPA={sdpa_bool[q_idx, kv_idx].item()}, Flex={flex_bool[q_idx, kv_idx].item()}")

    # Visualize a small portion of the mask
    print("\nSDPA mask visualization (first 30x30):")
    visualize_mask(sdpa_bool[:30, :30].cpu())

    return match


def visualize_mask(mask: torch.Tensor, max_size: int = 50):
    """Visualize a boolean mask using ASCII characters."""
    h, w = mask.shape
    h = min(h, max_size)
    w = min(w, max_size)

    # Header
    print("     ", end="")
    for j in range(0, w, 5):
        print(f"{j:<5}", end="")
    print()

    for i in range(h):
        print(f"{i:3d}: ", end="")
        for j in range(w):
            if mask[i, j]:
                print("■", end="")
            else:
                print("□", end="")
        print()


def benchmark_flex_attention_optimizations(
    num_views: int = 4,
    num_frames_per_view: int = 4,
    tokens_per_frame: int = 180,
    system_tokens: int = 50,
    traj_text_tokens: int = 200,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 128,
    num_warmup: int = 3,
    num_iterations: int = 10,
    device: str = "cuda",
):
    """Benchmark Flex Attention optimizations."""
    if not FLEX_AVAILABLE:
        print("Flex Attention not available, skipping optimization benchmark...")
        return {}

    print("\n" + "=" * 70)
    print("FLEX ATTENTION OPTIMIZATION BENCHMARK")
    print("=" * 70)

    vision_start_end_ids_ranges, traj_and_text_ids_range = create_dummy_ranges(
        num_views, num_frames_per_view, tokens_per_frame, system_tokens, traj_text_tokens
    )

    kv_length = traj_and_text_ids_range[1]
    query_length = compute_query_length(vision_start_end_ids_ranges, traj_and_text_ids_range)
    cache_position = torch.arange(query_length, device=device)

    print(f"\nConfiguration:")
    print(f"  query_length: {query_length}")
    print(f"  kv_length: {kv_length}")

    results = {}

    # Test 1: Original create_block_mask (with _compile=True, default)
    print("\n1. Original create_block_mask (_compile=True)...")
    torch.cuda.synchronize()

    times = []
    for i in range(num_warmup + num_iterations):
        start = time.perf_counter()
        block_mask = create_streaming_attention_mask_flex(
            batch_size=batch_size,
            cache_position=cache_position,
            kv_length=kv_length,
            vision_start_end_ids_ranges=vision_start_end_ids_ranges,
            traj_and_text_ids_range=traj_and_text_ids_range,
            device=device,
        )
        torch.cuda.synchronize()
        if i >= num_warmup:
            times.append((time.perf_counter() - start) * 1000)

    avg_time = sum(times) / len(times)
    results["Original (_compile=True)"] = avg_time
    print(f"   Average time: {avg_time:.3f} ms")
    print(f"   First call: {times[0]:.3f} ms, Last call: {times[-1]:.3f} ms")

    # Test 2: create_block_mask with _compile=False
    print("\n2. create_block_mask with _compile=False...")

    # Need to create mask_mod manually for _compile=False test
    (
        kv_region, kv_frame, kv_local_pos,
        query_region, query_frame, query_local_pos
    ) = _build_streaming_position_mappings(
        kv_length=kv_length,
        query_length=query_length,
        vision_start_end_ids_ranges=vision_start_end_ids_ranges,
        traj_and_text_ids_range=traj_and_text_ids_range,
        device=device,
    )

    num_views_val = len(vision_start_end_ids_ranges)

    def streaming_mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        q_view = query_region[q_idx]
        q_frame_val = query_frame[q_idx]
        q_local = query_local_pos[q_idx]
        kv_view = kv_region[kv_idx]
        kv_frame_val = kv_frame[kv_idx]
        kv_local = kv_local_pos[kv_idx]

        is_system = kv_view == -1
        is_image_query = q_view < num_views_val
        is_traj_query = q_view == num_views_val

        earlier_view = is_image_query & (kv_view >= 0) & (kv_view < q_view)
        same_view_earlier_frame = is_image_query & (kv_view == q_view) & (kv_frame_val < q_frame_val)
        same_frame_causal = is_image_query & (kv_view == q_view) & (kv_frame_val == q_frame_val) & (kv_local <= q_local)
        can_see_all_images = is_traj_query & (kv_view >= 0) & (kv_view < num_views_val)
        traj_causal = is_traj_query & (kv_view == num_views_val) & (kv_local <= q_local)

        return is_system | earlier_view | same_view_earlier_frame | same_frame_causal | can_see_all_images | traj_causal

    torch.cuda.synchronize()
    times = []
    for i in range(num_warmup + num_iterations):
        start = time.perf_counter()
        block_mask_no_compile = create_block_mask(
            mask_mod=streaming_mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=query_length,
            KV_LEN=kv_length,
            device=device,
            _compile=False,
        )
        torch.cuda.synchronize()
        if i >= num_warmup:
            times.append((time.perf_counter() - start) * 1000)

    avg_time = sum(times) / len(times)
    results["_compile=False"] = avg_time
    print(f"   Average time: {avg_time:.3f} ms")
    print(f"   First call: {times[0]:.3f} ms, Last call: {times[-1]:.3f} ms")

    # Test 3: Cached BlockMask (simulate cache hit)
    print("\n3. Cached BlockMask (cache hit simulation)...")

    # Create once
    cached_block_mask = create_streaming_attention_mask_flex(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        vision_start_end_ids_ranges=vision_start_end_ids_ranges,
        traj_and_text_ids_range=traj_and_text_ids_range,
        device=device,
    )

    torch.cuda.synchronize()
    times = []
    for i in range(num_warmup + num_iterations):
        start = time.perf_counter()
        # Simulate cache hit - just return cached mask
        _ = cached_block_mask
        torch.cuda.synchronize()
        if i >= num_warmup:
            times.append((time.perf_counter() - start) * 1000)

    avg_time = sum(times) / len(times)
    results["Cached (hit)"] = avg_time
    print(f"   Average time: {avg_time:.6f} ms (essentially 0)")

    # Test 4: Pre-computed position mappings time
    print("\n4. Position mappings creation time...")

    torch.cuda.synchronize()
    times = []
    for i in range(num_warmup + num_iterations):
        start = time.perf_counter()
        _ = _build_streaming_position_mappings(
            kv_length=kv_length,
            query_length=query_length,
            vision_start_end_ids_ranges=vision_start_end_ids_ranges,
            traj_and_text_ids_range=traj_and_text_ids_range,
            device=device,
        )
        torch.cuda.synchronize()
        if i >= num_warmup:
            times.append((time.perf_counter() - start) * 1000)

    avg_time = sum(times) / len(times)
    results["Position mappings only"] = avg_time
    print(f"   Average time: {avg_time:.3f} ms")

    # Test 5: Full attention with cached vs uncached BlockMask
    print("\n5. Full attention computation comparison...")

    q = torch.randn(batch_size, num_heads, query_length, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, num_heads, kv_length, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch_size, num_heads, kv_length, head_dim, device=device, dtype=torch.bfloat16)

    flex_attention_compiled = torch.compile(flex_attention)

    # Warmup compiled version with cached mask
    for _ in range(num_warmup):
        _ = flex_attention_compiled(q, k, v, block_mask=cached_block_mask)
    torch.cuda.synchronize()

    # Test with cached mask
    times = []
    for i in range(num_iterations):
        start = time.perf_counter()
        _ = flex_attention_compiled(q, k, v, block_mask=cached_block_mask)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    avg_time = sum(times) / len(times)
    results["Flex attn (cached mask, compiled)"] = avg_time
    print(f"   Cached mask + compiled flex_attention: {avg_time:.3f} ms")

    # Test with uncached mask (create each time)
    times = []
    for i in range(num_iterations):
        start = time.perf_counter()
        new_block_mask = create_block_mask(
            mask_mod=streaming_mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=query_length,
            KV_LEN=kv_length,
            device=device,
            _compile=False,
        )
        _ = flex_attention_compiled(q, k, v, block_mask=new_block_mask)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    avg_time = sum(times) / len(times)
    results["Flex attn (uncached, _compile=False)"] = avg_time
    print(f"   Uncached mask (_compile=False) + compiled flex_attention: {avg_time:.3f} ms")

    # Summary
    print("\n" + "-" * 50)
    print("OPTIMIZATION SUMMARY:")
    print("-" * 50)
    for name, time_ms in results.items():
        print(f"  {name}: {time_ms:.3f} ms")

    return results


def benchmark_both_cached(
    num_views: int = 4,
    num_frames_per_view: int = 4,
    tokens_per_frame: int = 180,
    system_tokens: int = 50,
    traj_text_tokens: int = 200,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 128,
    num_warmup: int = 5,
    num_iterations: int = 20,
    device: str = "cuda",
):
    """
    Fair comparison: both SDPA and Flex Attention with cached masks.

    This simulates the streaming scenario where the mask is created once
    during prefill and reused for all subsequent streaming steps.
    """
    print("\n" + "=" * 70)
    print("FAIR COMPARISON: BOTH WITH CACHED MASKS")
    print("=" * 70)

    vision_start_end_ids_ranges, traj_and_text_ids_range = create_dummy_ranges(
        num_views, num_frames_per_view, tokens_per_frame, system_tokens, traj_text_tokens
    )

    kv_length = traj_and_text_ids_range[1]
    query_length = compute_query_length(vision_start_end_ids_ranges, traj_and_text_ids_range)
    cache_position = torch.arange(query_length, device=device)

    print(f"\nConfiguration:")
    print(f"  query_length: {query_length}")
    print(f"  kv_length: {kv_length}")
    print(f"  batch_size: {batch_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {head_dim}")

    # Create Q, K, V tensors
    q = torch.randn(batch_size, num_heads, query_length, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, num_heads, kv_length, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch_size, num_heads, kv_length, head_dim, device=device, dtype=torch.bfloat16)

    results = {}

    # ========== SDPA with cached mask ==========
    print("\n--- SDPA with cached 4D mask ---")

    # Create mask once (this is the "prefill" cost)
    mask_start = time.perf_counter()
    cached_sdpa_mask = create_streaming_attention_mask_sdpa_optimized(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        vision_start_end_ids_ranges=vision_start_end_ids_ranges,
        traj_and_text_ids_range=traj_and_text_ids_range,
        valid_length=kv_length,
        device=device,
        dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    sdpa_mask_creation_time = (time.perf_counter() - mask_start) * 1000
    print(f"  Mask creation (one-time): {sdpa_mask_creation_time:.3f} ms")

    # Warmup
    for _ in range(num_warmup):
        _ = F.scaled_dot_product_attention(q, k, v, attn_mask=cached_sdpa_mask)
    torch.cuda.synchronize()

    # Benchmark attention only (with cached mask)
    start = time.perf_counter()
    for _ in range(num_iterations):
        out_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=cached_sdpa_mask)
    torch.cuda.synchronize()
    sdpa_attn_time = (time.perf_counter() - start) / num_iterations * 1000
    results["SDPA (cached mask)"] = sdpa_attn_time
    print(f"  Attention only (cached mask): {sdpa_attn_time:.3f} ms")

    # ========== Flex Attention with cached BlockMask ==========
    if FLEX_AVAILABLE:
        print("\n--- Flex Attention with cached BlockMask ---")

        # Create BlockMask once (this is the "prefill" cost)
        mask_start = time.perf_counter()
        cached_block_mask = create_streaming_attention_mask_flex(
            batch_size=batch_size,
            cache_position=cache_position,
            kv_length=kv_length,
            vision_start_end_ids_ranges=vision_start_end_ids_ranges,
            traj_and_text_ids_range=traj_and_text_ids_range,
            device=device,
        )
        torch.cuda.synchronize()
        flex_mask_creation_time = (time.perf_counter() - mask_start) * 1000
        print(f"  BlockMask creation (one-time): {flex_mask_creation_time:.3f} ms")

        # Compile flex_attention
        flex_attention_compiled = torch.compile(flex_attention)

        # Warmup
        for _ in range(num_warmup):
            _ = flex_attention_compiled(q, k, v, block_mask=cached_block_mask)
        torch.cuda.synchronize()

        # Benchmark attention only (with cached mask)
        start = time.perf_counter()
        for _ in range(num_iterations):
            out_flex = flex_attention_compiled(q, k, v, block_mask=cached_block_mask)
        torch.cuda.synchronize()
        flex_attn_time = (time.perf_counter() - start) / num_iterations * 1000
        results["Flex (cached mask, compiled)"] = flex_attn_time
        print(f"  Attention only (cached mask, compiled): {flex_attn_time:.3f} ms")

        # Also test non-compiled flex_attention
        for _ in range(num_warmup):
            _ = flex_attention(q, k, v, block_mask=cached_block_mask)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iterations):
            out_flex_nc = flex_attention(q, k, v, block_mask=cached_block_mask)
        torch.cuda.synchronize()
        flex_attn_time_nc = (time.perf_counter() - start) / num_iterations * 1000
        results["Flex (cached mask, not compiled)"] = flex_attn_time_nc
        print(f"  Attention only (cached mask, not compiled): {flex_attn_time_nc:.3f} ms")

    # ========== Summary ==========
    print("\n" + "-" * 50)
    print("STREAMING SCENARIO SUMMARY (both masks cached):")
    print("-" * 50)
    print("\nOne-time mask creation cost (prefill):")
    print(f"  SDPA mask: {sdpa_mask_creation_time:.3f} ms")
    if FLEX_AVAILABLE:
        print(f"  Flex BlockMask: {flex_mask_creation_time:.3f} ms")

    print("\nPer-streaming-step attention cost (mask reused):")
    for name, time_ms in results.items():
        print(f"  {name}: {time_ms:.3f} ms")

    if FLEX_AVAILABLE and "SDPA (cached mask)" in results and "Flex (cached mask, compiled)" in results:
        sdpa_time = results["SDPA (cached mask)"]
        flex_time = results["Flex (cached mask, compiled)"]
        if flex_time < sdpa_time:
            print(f"\n  → Flex Attention is {sdpa_time/flex_time:.2f}x faster than SDPA")
        else:
            print(f"\n  → SDPA is {flex_time/sdpa_time:.2f}x faster than Flex Attention")

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Flex Attention available: {FLEX_AVAILABLE}")

    # Verify correctness first
    verify_mask_correctness(device)

    # Verify output consistency
    verify_output_consistency(device)

    # Fair comparison with both masks cached
    cached_results = benchmark_both_cached(device=device)

    # Also run detailed benchmarks for reference
    print("\n\n" + "=" * 70)
    print("DETAILED BENCHMARKS (for reference)")
    print("=" * 70)

    # Benchmark mask creation
    mask_results, _, _ = benchmark_mask_creation(
        num_views=4,
        num_frames_per_view=4,
        tokens_per_frame=180,
        system_tokens=50,
        traj_text_tokens=200,
        batch_size=1,
        device=device,
    )

    # Benchmark attention computation
    attn_results = benchmark_attention_computation(
        num_views=4,
        num_frames_per_view=4,
        tokens_per_frame=180,
        system_tokens=50,
        traj_text_tokens=200,
        batch_size=1,
        num_heads=8,
        head_dim=128,
        device=device,
    )


if __name__ == "__main__":
    main()
