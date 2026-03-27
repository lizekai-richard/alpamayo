#!/usr/bin/env python3
"""Generate multimodal calibration data for ParoQuant optimization.

Pre-computes inputs_embeds with image features already fused, so that
ParoQuant's layerwise optimization sees activations representative of
the actual multimodal workload (images + prompt -> CoC), not just text.

The output .pt file contains pre-computed:
  - inputs_embeds: (n_samples, seq_len, hidden_dim) - fused text+image embeddings
  - position_ids: (n_samples, 3, seq_len) - M-RoPE 3D positions
  - attention_mask: (n_samples, seq_len) - attention masks
  - metadata: dict with clip_ids, n_samples, etc.

Usage:
    python paroquant/generate_calib_data.py \
        --model-path /data/scratch/zekaili/Alpamayo-R1-10B \
        --output /data/scratch/zekaili/calib_data_multimodal.pt \
        --start-chunk 0 --end-chunk 5 \
        --seed 42
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

os.environ["CUDA_ENABLE_COREDUMP_ON_EXCEPTION"] = "0"
os.environ["CUDA_ENABLE_GPU_COREDUMP"] = "0"

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
import alpamayo_r1

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.modules["alpamayo1_5"] = alpamayo_r1
from alpamayo_r1.models.alpamayo_r1p5 import Alpamayo1_5
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper
import physical_ai_av

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
DATASET_REVISION = "37a7cc2c868d684d0456b5412a7ec5d18597a96a"
DOWNLOAD_FEATURES = [
    "labels/egomotion",
    "camera/camera_cross_left_120fov",
    "camera/camera_front_wide_120fov",
    "camera/camera_cross_right_120fov",
    "camera/camera_front_tele_30fov",
]

# Fixed t0 timestamp (5.1s into clip, the default for load_physical_aiavdataset)
DEFAULT_T0_US = 5_100_000


def download_chunks(chunks: list[int], cache_dir: str, workers: int = 4):
    """Pre-download dataset chunks to local cache."""
    api = HfApi()
    files = list(api.list_repo_tree(
        DATASET_REPO_ID, repo_type="dataset", revision=DATASET_REVISION, recursive=True,
    ))

    chunks_set = set(chunks)
    to_download = []
    for f in files:
        if not any(feat in f.path for feat in DOWNLOAD_FEATURES):
            continue
        match = re.search(r"chunk_(\d+)", f.path)
        if match and int(match.group(1)) in chunks_set:
            to_download.append(f.path)

    logger.info(f"  {len(to_download)} files to download for {len(chunks)} chunks")

    def _download_one(path):
        try:
            hf_hub_download(
                DATASET_REPO_ID, path,
                repo_type="dataset", revision=DATASET_REVISION, cache_dir=cache_dir,
            )
            return (path, True)
        except Exception as e:
            return (path, False)

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_download_one, p): p for p in to_download}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            path, ok = future.result()
            if not ok:
                failed.append(path)

    if failed:
        logger.warning(f"  {len(failed)} files failed to download")


@torch.inference_mode()
def compute_fused_embeddings(
    model: Alpamayo1_5,
    processor,
    data: dict,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Compute fused text+image embeddings for a single sample.

    Follows the same embedding computation pattern as the model's prefill:
    1. Tokenize message with chat template
    2. Fuse trajectory history tokens into input_ids
    3. Compute M-RoPE position_ids
    4. Get text embeddings from input_ids
    5. Encode images via visual encoder
    6. Scatter image embeddings into text embeddings

    Returns dict with inputs_embeds, position_ids, attention_mask (all on CPU, bfloat16).
    """
    # 4 cameras × N frames → flatten to (4*N, C, H, W)
    frames = data["image_frames"].flatten(0, 1)
    messages = helper.create_message(frames, camera_indices=data["camera_indices"])

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    logger.info(f"Inputs length: {inputs['input_ids'].shape[1]}")

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(device)
    image_grid_thw = inputs["image_grid_thw"].to(device)

    # Fuse trajectory history tokens
    input_ids = model.fuse_traj_tokens(
        input_ids,
        {
            "ego_history_xyz": data["ego_history_xyz"].to(device),
            "ego_history_rot": data["ego_history_rot"].to(device),
        },
    )

    # Compute M-RoPE position_ids
    position_ids, rope_deltas = model.vlm.model.get_rope_index(
        input_ids, image_grid_thw
    )

    # Compute text embeddings
    inputs_embeds = model.vlm.model.get_input_embeddings()(input_ids)

    # Encode images via visual encoder
    with torch.autocast("cuda", dtype=torch.bfloat16):
        image_embeds, _deepstack = model.vlm.model.visual(
            pixel_values, grid_thw=image_grid_thw
        )

    # Scatter image embeddings into text embeddings
    image_mask = (
        (input_ids == model.vlm.config.image_token_id)
        .unsqueeze(-1)
        .expand_as(inputs_embeds)
    )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    return {
        "inputs_embeds": inputs_embeds.cpu().to(torch.bfloat16),
        "position_ids": position_ids.cpu(),
        "attention_mask": attention_mask.cpu(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate multimodal calibration data for ParoQuant"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="nvidia/Alpamayo-1.5-10B",
        help="Path to Alpamayo model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./calib_data_multimodal_v1p5.pt",
        help="Output .pt file path",
    )
    parser.add_argument(
        "--start-chunk",
        type=int,
        default=0,
        help="Start chunk index (inclusive)",
    )
    parser.add_argument(
        "--end-chunk",
        type=int,
        default=5,
        help="End chunk index (exclusive)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=4,
        help="Number of frames per camera (default: 1, use 4 for full temporal context)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/mnt/moosefs-1/users/zekail/physicalai_av/hf_cache",
        help="PhysicalAI-AV HuggingFace cache directory",
    )
    args = parser.parse_args()

    args.model_path = os.path.expanduser(args.model_path)
    args.cache_dir = os.path.expanduser(args.cache_dir)
    args.output = os.path.expanduser(args.output)

    device = "cuda"

    # Load model
    logger.info(f"Loading AlpamayoR1 from {args.model_path}...")
    model = Alpamayo1_5.from_pretrained(args.model_path, dtype=torch.bfloat16).to(device)
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    # Load dataset interface
    logger.info("Loading PhysicalAI-AV dataset interface...")
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(
        cache_dir=args.cache_dir,
    )

    # Pre-download chunks to local cache
    chunk_range = range(args.start_chunk, args.end_chunk)
    logger.info(f"Pre-downloading chunks {list(chunk_range)} to {args.cache_dir}...")
    download_chunks(list(chunk_range), args.cache_dir)
    logger.info("Chunks downloaded.")

    # Load clip IDs from chunk index
    logger.info(f"Loading clip IDs for chunks {list(chunk_range)}...")
    index_path = hf_hub_download(
        "nvidia/PhysicalAI-Autonomous-Vehicles",
        "clip_index.parquet",
        repo_type="dataset",
        cache_dir=args.cache_dir,
    )
    clip_index = pd.read_parquet(index_path)
    clip_ids = clip_index[clip_index["chunk"].isin(chunk_range)].index.tolist()
    random.seed(args.seed)
    random.shuffle(clip_ids)
    logger.info(f"Found {len(clip_ids)} clips in chunks {list(chunk_range)}")

    inputs_embeds_list = []
    position_ids_list = []
    attention_mask_list = []
    collected_metadata = []
    n_collected = 0
    n_failed = 0

    logger.info(
        f"Generating 1 sample per clip (t0=5.1s) from {len(clip_ids)} clips"
    )

    pbar = tqdm(total=len(clip_ids), desc="Generating calibration data")
    for clip_id in clip_ids:
        try:
            data = load_physical_aiavdataset(
                clip_id,
                t0_us=DEFAULT_T0_US,
                avdi=avdi,
                maybe_stream=False,
                num_frames=args.num_frames,
            )

            result = compute_fused_embeddings(model, processor, data, device)

            inputs_embeds_list.append(result["inputs_embeds"])
            position_ids_list.append(result["position_ids"])
            attention_mask_list.append(result["attention_mask"])
            collected_metadata.append(
                {"clip_id": clip_id, "t0_us": DEFAULT_T0_US}
            )

            n_collected += 1
            pbar.update(1)

        except Exception as e:
            n_failed += 1
            pbar.update(1)
            logger.warning(
                f"Failed clip={clip_id}: {e}"
            )

    pbar.close()

    if n_collected == 0:
        logger.error("No samples collected. Exiting.")
        sys.exit(1)

    logger.info(f"Collected {n_collected} samples ({n_failed} failed)")

    # Pad to uniform sequence length and stack
    max_seq_len = max(e.shape[1] for e in inputs_embeds_list)
    hidden_dim = inputs_embeds_list[0].shape[2]
    logger.info(
        f"Max seq_len={max_seq_len}, hidden_dim={hidden_dim}, "
        f"padding shorter sequences"
    )

    padded_embeds = torch.zeros(
        n_collected, max_seq_len, hidden_dim, dtype=torch.bfloat16
    )
    # position_ids shape: (batch, 3, seq_len) for M-RoPE
    padded_pos_ids = torch.zeros(
        n_collected, 3, max_seq_len, dtype=torch.long
    )
    padded_attn_mask = torch.zeros(
        n_collected, max_seq_len, dtype=torch.long
    )

    for i in range(n_collected):
        seq_len = inputs_embeds_list[i].shape[1]
        padded_embeds[i, :seq_len, :] = inputs_embeds_list[i][0]
        padded_pos_ids[i, :, :seq_len] = position_ids_list[i][0]
        padded_attn_mask[i, :seq_len] = attention_mask_list[i][0]

    # Free per-sample lists
    del inputs_embeds_list, position_ids_list, attention_mask_list

    save_data = {
        "inputs_embeds": padded_embeds,
        "position_ids": padded_pos_ids,
        "attention_mask": padded_attn_mask,
        "metadata": {
            "clip_ids": [m["clip_id"] for m in collected_metadata],
            "t0s_us": [m["t0_us"] for m in collected_metadata],
            "n_samples": n_collected,
            "n_failed": n_failed,
            "max_seq_len": max_seq_len,
            "hidden_dim": hidden_dim,
            "model_path": args.model_path,
            "seed": args.seed,
            "t0_us": DEFAULT_T0_US,
            "num_frames": args.num_frames,
            "chunks": list(chunk_range),
            "total_clips_in_chunks": len(clip_ids),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving to {output_path}...")
    torch.save(save_data, output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"Done! Saved {n_collected} samples to {output_path} "
        f"({file_size_mb:.1f} MB)"
    )
    logger.info(
        f"  inputs_embeds: {padded_embeds.shape} ({padded_embeds.dtype})"
    )
    logger.info(f"  position_ids:  {padded_pos_ids.shape}")
    logger.info(f"  attention_mask: {padded_attn_mask.shape}")


if __name__ == "__main__":
    main()
