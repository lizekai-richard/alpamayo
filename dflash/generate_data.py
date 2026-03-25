#!/usr/bin/env python3
"""Generate offline distillation data for DFlash training.

Extracts (hidden_state, future_tokens) from Alpamayo VLM and saves to disk.
This allows fast training without loading Alpamayo during training.

For each clip, the valid time range is determined by querying the egomotion data:
  - min_t0 = clip_start + 1.6s (need history)
  - max_t0 = clip_end - 6.4s (need future for trajectory)

The valid range is then divided into N areas (--num-samples-per-clip), and one random
timestamp is sampled from each area. This provides diverse temporal coverage per clip.

Usage:
    # Single GPU
    python generate_data.py \
        --cache-dir /data/physicalai_av/hf_cache \
        --output-dir /data/dflash_train \
        --start-chunk 0 --end-chunk 50

    # Multi-GPU (recommended: use run_generate_data.sh)
    ./run_generate_data.sh 0 400 /data/dflash_train
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

# Disable GPU core dumps before any CUDA import
os.environ["CUDA_ENABLE_COREDUMP_ON_EXCEPTION"] = "0"
os.environ["CUDA_ENABLE_GPU_COREDUMP"] = "0"

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from patches import patch_for_torch_compile
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper
from alpamayo_r1.models.token_utils import to_special_token
import physical_ai_av
import alpamayo_r1
sys.modules["alpamayo1_5"] = alpamayo_r1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Standard Qwen3 vocabulary size (before Alpamayo's extensions)
# Tokens >= this value are trajectory tokens and should be masked in training
STANDARD_VOCAB_SIZE = 151936
IGNORE_INDEX = -100  # Standard PyTorch ignore index for CrossEntropyLoss


# For Alpamayo (36 layers) + DFlash (5 draft layers)
NUM_TARGET_LAYERS = 36
NUM_DRAFT_LAYERS = 5

# Default target layers: store a superset for flexible layer selection during training
# Includes early (0), middle (16, 20, 24, 28), and late layers (30, 31, 32, 34, 35)
DEFAULT_TARGET_LAYER_IDS = [24, 30, 31, 32, 34]

# Timing constants (in microseconds)
HISTORY_DURATION_US = 1_600_000  # 1.6s history required
FUTURE_DURATION_US = 6_400_000   # 6.4s future required


def get_valid_t0_range(
    clip_id: str = None,
    avdi = None,
    maybe_stream: bool = False,
) -> tuple[int, int]:
    """Get valid t0 sampling range for clips.

    Most PhysicalAI-AV clips have data from ~0s to ~20s.
    We use a conservative fixed range that works for most clips:
    - min_t0 = 2.0s (provides 1.6s history margin)
    - max_t0 = 13.6s (provides 6.4s future margin, assuming ~20s clips)

    Args:
        clip_id: Clip identifier (unused, kept for API compatibility)
        avdi: PhysicalAIAVDatasetInterface instance (unused)
        maybe_stream: Whether to stream data (unused)

    Returns:
        (min_t0_us, max_t0_us) tuple
    """
    # Conservative fixed range that works for most clips
    # Clips typically have ~20s of data, so this gives good coverage
    min_t0_us = 2_000_000   # 2.0s - ensures 1.6s history available
    max_t0_us = 13_600_000  # 13.6s - ensures 6.4s future available (up to 20s)

    return (min_t0_us, max_t0_us)


def sample_t0_from_areas(
    min_t0_us: int,
    max_t0_us: int,
    num_areas: int,
    seed: int | None = None,
) -> list[int]:
    """Divide valid range into areas and sample one t0 from each area.

    Args:
        min_t0_us: Minimum valid t0 (microseconds)
        max_t0_us: Maximum valid t0 (microseconds)
        num_areas: Number of areas to divide the range into
        seed: Random seed for reproducibility

    Returns:
        List of sampled t0 values (one per area)
    """
    if seed is not None:
        random.seed(seed)

    range_us = max_t0_us - min_t0_us
    area_size = range_us // num_areas

    sampled_t0s = []
    for i in range(num_areas):
        area_start = min_t0_us + i * area_size
        area_end = min_t0_us + (i + 1) * area_size if i < num_areas - 1 else max_t0_us
        t0 = random.randint(area_start, area_end)
        sampled_t0s.append(t0)

    return sampled_t0s


def extract_hidden_states_and_tokens(
    model: AlpamayoR1,
    processor,
    data: dict,
    target_layer_ids: list[int],
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Run Alpamayo forward pass and extract hidden states + generated tokens.

    Args:
        model: Alpamayo model
        processor: Tokenizer processor (with resolution settings from helper.get_processor)
        data: Input data dict
        target_layer_ids: List of layer indices to extract hidden states from
        device: Compute device

    Returns:
        hidden_states: (seq_len, num_layers * hidden_dim) - concatenated hidden states
        input_ids: (seq_len,) - full sequence (input + generated)
        generation_start_idx: int - where generation starts
    """
    messages = helper.create_message(data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"])

    # Get resolution settings from processor (set by helper.get_processor)
    min_pixels = getattr(processor, '_min_pixels', helper.MIN_PIXELS)
    max_pixels = getattr(processor, '_max_pixels', helper.MAX_PIXELS)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, device)

    input_len = inputs["input_ids"].shape[-1]

    # Run inference to get full sequence
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )

    # Get the generated token IDs from extra
    # We need to run a forward pass to get hidden states
    vlm = model.vlm

    # Get full input_ids (input + generated CoC + <|cot_end|>)
    # The CoC text is in extra["cot"]
    cot_text = extra["cot"][0][0][0] if extra.get("cot") is not None else ""
    cot_tokens = model.tokenizer(cot_text, add_special_tokens=False, return_tensors="pt")["input_ids"] 

    # Get <|cot_end|> token ID to append after CoC text
    # This teaches the drafter when to stop generating CoC
    cot_end_token_id = model.tokenizer.convert_tokens_to_ids("<|cot_end|>")
    cot_end_token = torch.tensor([[cot_end_token_id]], dtype=cot_tokens.dtype)

    traj_future_start_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    traj_future_start_token = torch.tensor([[traj_future_start_token_id]], dtype=cot_tokens.dtype)

    # Combine input + CoC tokens + <|cot_end|> + <|traj_future_start|>
    full_input_ids = torch.cat([
        inputs["input_ids"].to(device),
        cot_tokens.to(device),
        cot_end_token.to(device),
        traj_future_start_token.to(device),
    ], dim=-1)

    # Use forward hooks to capture hidden states from target layers.
    # The model is already patched (vision + TextModel) by patch_for_torch_compile
    # so hidden states match inference-time eval behavior.
    layers = vlm.model.language_model.layers
    captured_hidden = {}
    hooks = []
    for lid in target_layer_ids:
        def _make_hook(layer_id):
            def hook_fn(module, args, output):
                hs = output[0] if isinstance(output, tuple) else output
                captured_hidden[layer_id] = hs.detach()
            return hook_fn
        hooks.append(layers[lid].register_forward_hook(_make_hook(lid)))

    # Forward pass to capture hidden states (single pass, no KV cache)
    # Patched TextModel needs explicit cache_position and attention_mask.
    # Without attention_mask, HF's masking_utils detects mrope position_id jumps
    # (at vision token boundaries) as packed sequences, corrupting the causal mask.
    seq_len = full_input_ids.shape[1]
    cache_position = torch.arange(seq_len, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        vlm(
            input_ids=full_input_ids,
            attention_mask=torch.ones(1, seq_len, dtype=torch.bool, device=device),
            pixel_values=inputs.get("pixel_values", None).to(device) if inputs.get("pixel_values") is not None else None,
            image_grid_thw=inputs.get("image_grid_thw", None).to(device) if inputs.get("image_grid_thw") is not None else None,
            cache_position=cache_position,
            return_dict=True,
        )

    for h in hooks:
        h.remove()

    # Concatenate hidden states from target layers (hooks capture layer output directly)
    target_hidden_list = [captured_hidden[lid] for lid in target_layer_ids]
    # (batch, seq, num_layers * hidden)
    target_hidden = torch.cat(target_hidden_list, dim=-1)

    return target_hidden[0], full_input_ids[0], input_len


def mask_extended_vocab_tokens(
    tokens: torch.Tensor,
    mask_extended: bool = True,
    keep_token_ids: list[int] | None = None,
) -> torch.Tensor:
    """Optionally mask tokens outside standard Qwen3 vocabulary with IGNORE_INDEX.

    Alpamayo extends the vocabulary with trajectory tokens (>= 151936).
    Special tokens like <|cot_end|> should NOT be masked so the model learns them.

    Args:
        tokens: Token IDs tensor
        mask_extended: If True, mask extended vocab tokens. If False, keep all tokens.
        keep_token_ids: List of token IDs to NOT mask (e.g., special tokens like <|cot_end|>)

    Returns:
        Labels tensor (with extended tokens replaced by IGNORE_INDEX if mask_extended=True)
    """
    if not mask_extended:
        return tokens.clone()

    labels = tokens.clone()
    mask = labels >= STANDARD_VOCAB_SIZE

    # Don't mask special tokens that we want to train on
    if keep_token_ids is not None:
        for tok_id in keep_token_ids:
            mask = mask & (labels != tok_id)

    if mask.any():
        labels[mask] = IGNORE_INDEX
    return labels


def create_training_blocks(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    generation_start_idx: int,
    block_size: int,
    stride: int = 1,
    context_len: int = 1,
    mask_extended_vocab: bool = True,
    keep_token_ids: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create (hidden_state, future_tokens, labels) tuples with sliding window.

    Args:
        hidden_states: (seq_len, hidden_dim)
        input_ids: (seq_len,)
        generation_start_idx: where generation starts (we want blocks that predict generated tokens)
        block_size: number of future tokens per block
        stride: step size for sliding window
        context_len: number of consecutive hidden states per block (1 = original behavior)
        mask_extended_vocab: If True, mask trajectory tokens. If False, train on full vocab.
        keep_token_ids: List of token IDs to NOT mask (e.g., special tokens like <|cot_end|>)

    Returns:
        block_hidden: (num_blocks, context_len, hidden_dim) - target model hidden states
        block_tokens: (num_blocks, block_size) - actual token IDs (for noise embedding)
        block_labels: (num_blocks, block_size) - labels (optionally masked)
    """
    seq_len = hidden_states.shape[0]
    gen_len = seq_len - generation_start_idx

    if gen_len < 1:
        # No generated tokens at all
        return None, None, None

    # Start at generation boundary: future_tokens[0] = first generated token
    # This ensures we only train on generated tokens, not prompt tokens
    # At pos = generation_start_idx - 1:
    #   - hidden_states[pos-context_len+1 : pos+1] = context hidden states
    #   - future_tokens = input_ids[pos+1 : pos+1+block_size] = generated tokens
    start_pos = max(generation_start_idx - 1, context_len - 1)
    end_pos = seq_len - block_size

    # Pad if sequence too short to create any blocks
    if end_pos <= start_pos:
        # Need at least block_size generated tokens
        pad_len = block_size - gen_len + 1
        # Repeat last hidden state for padding
        hidden_states = torch.cat([
            hidden_states,
            hidden_states[-1:].expand(pad_len, -1)
        ], dim=0)
        # Pad input_ids with last token (likely <|cot_end|>)
        pad_token = input_ids[-1].item()
        input_ids = torch.cat([
            input_ids,
            torch.full((pad_len,), pad_token, dtype=input_ids.dtype)
        ], dim=0)
        # Update seq_len and end_pos
        seq_len = hidden_states.shape[0]
        end_pos = seq_len - block_size

    positions = list(range(start_pos, end_pos, stride))

    block_hidden = []
    block_tokens = []
    block_labels = []

    for pos in positions:
        block_hidden.append(hidden_states[pos - context_len + 1 : pos + 1])
        tokens = input_ids[pos + 1 : pos + 1 + block_size]
        block_tokens.append(tokens)
        # Optionally mask extended vocab tokens (trajectory tokens > 151936)
        # But keep special tokens like <|cot_end|> so the model learns when to stop
        block_labels.append(mask_extended_vocab_tokens(
            tokens, mask_extended=mask_extended_vocab, keep_token_ids=keep_token_ids
        ))

    block_hidden = torch.stack(block_hidden)
    block_tokens = torch.stack(block_tokens)
    block_labels = torch.stack(block_labels)

    return block_hidden, block_tokens, block_labels


def main():
    parser = argparse.ArgumentParser(description="Generate offline distillation data")
    parser.add_argument(
        "--model-path",
        type=str,
        default="nvidia/Alpamayo-1.5-10B",
        help="Path to Alpamayo model",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/mnt/moosefs-1/users/zekail/physicalai_av/hf_cache",
        help="HuggingFace cache directory with downloaded chunks",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/moosefs-1/users/zekail/dflash_train",
        help="Output directory for distillation data",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=20,
        help="Number of chunks to process (used if --start-chunk/--end-chunk not set)",
    )
    parser.add_argument(
        "--start-chunk",
        type=int,
        default=None,
        help="Start chunk index (inclusive). For parallel processing.",
    )
    parser.add_argument(
        "--end-chunk",
        type=int,
        default=None,
        help="End chunk index (exclusive). For parallel processing.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="GPU rank for parallel processing. Uses CUDA device and adds suffix to output files.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for sliding window (default: 4)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10000,
        help="Number of blocks per shard file",
    )
    parser.add_argument(
        "--target-layers",
        type=str,
        default=None,
        help="Comma-separated target layer indices (e.g., '24,30,31,32,34'). Default: [24,30,31,32,34]",
    )
    parser.add_argument(
        "--full-vocab",
        action="store_true",
        help="Train on full vocabulary including trajectory tokens (don't mask extended vocab)",
    )
    parser.add_argument(
        "--block-sizes",
        type=str,
        default="8,16",
        help="Comma-separated block sizes to generate (e.g., '8,16')",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=1,
        help="Number of consecutive hidden states per training block (default: 1)",
    )
    parser.add_argument(
        "--num-samples-per-clip",
        type=int,
        default=5,
        help="Number of t0 samples per clip. Clip is divided into N areas, one random sample per area. (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for t0 sampling (default: 42)",
    )
    args = parser.parse_args()

    # Expand ~ in paths
    args.model_path = os.path.expanduser(args.model_path)
    args.cache_dir = os.path.expanduser(args.cache_dir)
    args.output_dir = os.path.expanduser(args.output_dir)

    # Parse target layers
    if args.target_layers:
        target_layer_ids = [int(x.strip()) for x in args.target_layers.split(",")]
    else:
        target_layer_ids = DEFAULT_TARGET_LAYER_IDS

    # Determine chunk range
    if args.start_chunk is not None and args.end_chunk is not None:
        chunk_range = range(args.start_chunk, args.end_chunk)
    else:
        chunk_range = range(args.num_chunks)

    # Determine device
    if args.rank is not None:
        # When CUDA_VISIBLE_DEVICES is set, only 1 GPU is visible as device 0
        # rank is only used for file naming, not device selection
        device = "cuda:0"
        torch.cuda.set_device(0)
        rank_suffix = f"_rank{args.rank}"
    else:
        device = "cuda"
        rank_suffix = ""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"[Rank {args.rank}] Loading Alpamayo model from {args.model_path}...")
    model = AlpamayoR1.from_pretrained(args.model_path, dtype=torch.bfloat16).to(device)
    model.eval()

    # Patch model to match inference-time torch.compile setup.
    # This replaces vision modules (Conv3d→Linear, FlashAttn→SDPA) and the
    # TextModel wrapper so hidden states match eval scripts (sys opt / streaming).
    # TextAttention is NOT patched (non-streaming mode) because the decode loop
    # in Pass 1 requires original roped-KV attention.  Patched vs original
    # TextAttention produces identical hidden states in a single forward pass
    # (no KV cache), so training data matches both sys opt and streaming modes.
    patch_for_torch_compile(model.vlm, mode="non-streaming", fuse_qkv=True, fuse_gate_up=True)
    model._patched_for_compile = True  # Prevent _non_streaming_rollout from re-patching
    logger.info(f"[Rank {args.rank}] Applied patch_for_torch_compile (non-streaming) for consistent hidden states")

    processor = helper.get_processor(model.tokenizer)

    # Get special token IDs to keep (not mask) during training
    # <|cot_end|> must be trained so the model learns when to stop CoC generation
    cot_end_id = model.tokenizer.convert_tokens_to_ids("<|cot_end|>")
    traj_future_start_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    keep_token_ids = [cot_end_id, traj_future_start_id]
    logger.info(f"[Rank {args.rank}] Keep token IDs (not masked): {keep_token_ids} (<|cot_end|>={cot_end_id}, <|traj_future_start|>={traj_future_start_id})")

    # Load dataset interface
    # Use specific revision to avoid issues when HuggingFace dataset is updated
    DATASET_REVISION = "37a7cc2c868d684d0456b5412a7ec5d18597a96a"
    logger.info(f"[Rank {args.rank}] Loading dataset interface (revision: {DATASET_REVISION})...")
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(
        cache_dir=args.cache_dir,
        revision=DATASET_REVISION,
    )

    # Get clip IDs from chunks
    logger.info(f"[Rank {args.rank}] Getting clip IDs for chunks {list(chunk_range)}...")
    index_path = hf_hub_download(
        "nvidia/PhysicalAI-Autonomous-Vehicles",
        "clip_index.parquet",
        repo_type="dataset",
        cache_dir=args.cache_dir,
    )
    clip_index = pd.read_parquet(index_path)
    clip_ids = clip_index[clip_index["chunk"].isin(chunk_range)].index.tolist()

    # Parse block sizes
    block_sizes = [int(x.strip()) for x in args.block_sizes.split(",")]

    logger.info(f"[Rank {args.rank}] Processing {len(clip_ids)} clips from chunks {list(chunk_range)}")
    logger.info(f"[Rank {args.rank}] Samples per clip: {args.num_samples_per_clip}, Seed: {args.seed}")
    logger.info(f"[Rank {args.rank}] Block sizes: {block_sizes}, Stride: {args.stride}, Context len: {args.context_len}")
    logger.info(f"[Rank {args.rank}] Target layers: {target_layer_ids}")
    logger.info(f"[Rank {args.rank}] Full vocab (no masking): {args.full_vocab}")

    # Create output directories and accumulators for each block size
    # Each block size gets its own subdirectory: {output_dir}/b{block_size}
    output_dirs = {}
    accumulators = {}
    for bs in block_sizes:
        if len(block_sizes) > 1:
            # Multiple block sizes: separate subdirectories
            out_dir = output_dir / f"b{bs}"
        else:
            out_dir = output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[bs] = out_dir
        accumulators[bs] = {
            "all_hidden": [],
            "all_tokens": [],
            "all_labels": [],
            "total_blocks": 0,
            "masked_tokens_count": 0,
            "shard_idx": 0,
        }
        logger.info(f"[Rank {args.rank}] Block size {bs} → {out_dir}")

    failed_clips = []
    total_samples = 0

    pbar = tqdm(clip_ids, desc=f"[Rank {args.rank}] Processing clips")
    for clip_idx, clip_id in enumerate(pbar):
        try:
            # Get valid time range (fixed conservative range for all clips)
            min_t0_us, max_t0_us = get_valid_t0_range()

            # Sample t0 values from areas (use clip_idx for reproducible per-clip seeds)
            clip_seed = args.seed + clip_idx if args.seed is not None else None
            sampled_t0s = sample_t0_from_areas(
                min_t0_us, max_t0_us,
                num_areas=args.num_samples_per_clip,
                seed=clip_seed,
            )

            # Process each sampled t0
            for sample_idx, t0_us in enumerate(sampled_t0s):
                try:
                    # Load data at this t0
                    data = load_physical_aiavdataset(
                        clip_id,
                        t0_us=t0_us,
                        avdi=avdi,
                        maybe_stream=False,
                    )

                    # Extract hidden states and tokens
                    hidden_states, input_ids, gen_start = extract_hidden_states_and_tokens(
                        model, processor, data, target_layer_ids, device=device
                    )

                    total_samples += 1

                    # Create training blocks for EACH block size
                    for bs in block_sizes:
                        acc = accumulators[bs]
                        block_hidden, block_tokens, block_labels = create_training_blocks(
                            hidden_states.cpu(),
                            input_ids.cpu(),
                            gen_start,
                            block_size=bs,
                            stride=args.stride,
                            context_len=args.context_len,
                            mask_extended_vocab=not args.full_vocab,
                            keep_token_ids=keep_token_ids,
                        )

                        if block_hidden is not None:
                            acc["all_hidden"].append(block_hidden.half())
                            acc["all_tokens"].append(block_tokens.int())
                            acc["all_labels"].append(block_labels.int())
                            acc["total_blocks"] += block_hidden.shape[0]
                            acc["masked_tokens_count"] += (block_labels == IGNORE_INDEX).sum().item()

                        # Save shard when enough blocks accumulated
                        if acc["total_blocks"] >= args.shard_size * (acc["shard_idx"] + 1):
                            save_shard(
                                acc["all_hidden"], acc["all_tokens"], acc["all_labels"],
                                output_dirs[bs], acc["shard_idx"], args, rank_suffix
                            )
                            acc["all_hidden"] = []
                            acc["all_tokens"] = []
                            acc["all_labels"] = []
                            acc["shard_idx"] += 1

                except Exception as e:
                    failed_clips.append({"clip_id": clip_id, "t0_us": t0_us, "sample_idx": sample_idx, "error": str(e)})
                    logger.warning(f"[Rank {args.rank}] Failed {clip_id} t0={t0_us/1e6:.2f}s: {e}", exc_info=False)

            # Update progress bar with total blocks across all configs
            total = sum(acc["total_blocks"] for acc in accumulators.values())
            pbar.set_postfix({"samples": total_samples, "blocks": total})

        except Exception as e:
            failed_clips.append({"clip_id": clip_id, "error": str(e)})
            logger.warning(f"[Rank {args.rank}] Failed to process {clip_id}: {e}")

    # Save remaining blocks and metadata for each block size
    for bs in block_sizes:
        acc = accumulators[bs]
        out_dir = output_dirs[bs]

        if acc["all_hidden"]:
            save_shard(
                acc["all_hidden"], acc["all_tokens"], acc["all_labels"],
                out_dir, acc["shard_idx"], args, rank_suffix
            )
            acc["shard_idx"] += 1

        # Save metadata
        total_tokens = acc["total_blocks"] * bs
        metadata = {
            "rank": args.rank,
            "chunk_range": list(chunk_range),
            "num_shards": acc["shard_idx"],
            "total_blocks": acc["total_blocks"],
            "block_size": bs,
            "context_len": args.context_len,
            "stride": args.stride,
            "target_layer_ids": target_layer_ids,
            "num_target_layers": NUM_TARGET_LAYERS,
            "num_draft_layers": NUM_DRAFT_LAYERS,
            "full_vocab": args.full_vocab,
            "num_clips": len(clip_ids),
            "total_samples": total_samples,
            "num_samples_per_clip": args.num_samples_per_clip,
            "seed": args.seed,
            "failed_samples": len(failed_clips),
            "standard_vocab_size": STANDARD_VOCAB_SIZE,
            "masked_tokens": acc["masked_tokens_count"],
            "masked_token_ratio": acc["masked_tokens_count"] / total_tokens if total_tokens > 0 else 0,
        }

        metadata_filename = f"metadata{rank_suffix}.json"
        with open(out_dir / metadata_filename, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"[Rank {args.rank}] Block size {bs}: {acc['total_blocks']:,} blocks, {acc['shard_idx']} shards → {out_dir}")

    if failed_clips:
        failed_filename = f"failed_clips{rank_suffix}.json"
        with open(output_dir / failed_filename, "w") as f:
            json.dump(failed_clips, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"[Rank {args.rank}] Generation complete!")
    logger.info(f"  Chunks: {list(chunk_range)}")
    logger.info(f"  Clips: {len(clip_ids)}")
    logger.info(f"  Samples: {total_samples} ({args.num_samples_per_clip} per clip)")
    for bs in block_sizes:
        logger.info(f"  Block size {bs}: {accumulators[bs]['total_blocks']:,} blocks")
    logger.info(f"  Failed samples: {len(failed_clips)}")
    logger.info(f"  Output: {output_dir}")


def save_shard(all_hidden, all_tokens, all_labels, output_dir, shard_idx, args, rank_suffix=""):
    """Save accumulated blocks to a shard file.

    Each shard contains:
        - target_hidden: Hidden states from Alpamayo (for conditioning drafter)
        - future_tokens: Actual token IDs (for creating noise embeddings)
        - labels: Masked labels for loss computation (extended vocab tokens → -100)
    """
    hidden = torch.cat(all_hidden, dim=0)
    tokens = torch.cat(all_tokens, dim=0)
    labels = torch.cat(all_labels, dim=0)

    shard_path = output_dir / f"shard{rank_suffix}_{shard_idx:04d}.pt"
    torch.save({
        "target_hidden": hidden,      # (num_blocks, hidden_dim) fp16
        "future_tokens": tokens,      # (num_blocks, block_size) int32
        "labels": labels,             # (num_blocks, block_size) int32, with -100 for masked
    }, shard_path)

    # Count masked tokens in this shard
    masked_in_shard = (labels == IGNORE_INDEX).sum().item()
    total_in_shard = labels.numel()
    logger.info(
        f"[Rank {args.rank}] Saved shard {shard_idx}: {hidden.shape[0]} blocks, "
        f"{masked_in_shard}/{total_in_shard} masked tokens, {shard_path}"
    )


if __name__ == "__main__":
    main()
