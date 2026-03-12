#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Push Alpamayo R1 model weights to Hugging Face Hub.
#
# Usage:
#   # Load from local checkpoint and push (uses GPU to load model)
#   python scripts/push_model_to_huggingface.py \
#     --model_path /path/to/checkpoint \
#     --repo_id username/alpamayo-r1-10b
#
#   # Only upload an already-saved directory (no model load, no GPU needed)
#   python scripts/push_model_to_huggingface.py \
#     --upload_dir /path/to/saved_model \
#     --repo_id username/alpamayo-r1-10b
#
#   # With token and private repo
#   python scripts/push_model_to_huggingface.py \
#     --model_path /path/to/checkpoint \
#     --repo_id username/alpamayo-r1-10b \
#     --token hf_xxx \
#     --private

from __future__ import annotations

import argparse
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Push Alpamayo R1 model to Hugging Face Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Local checkpoint directory. Model will be loaded and pushed via push_to_hub.",
    )
    group.add_argument(
        "--upload_dir",
        type=str,
        default=None,
        help="Local directory already saved with save_pretrained. Upload this folder as-is (no load, no GPU).",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face repo id, e.g. username/model-name.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token. If not set, uses HF_TOKEN or huggingface-cli login.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private.",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload Alpamayo R1 model weights",
        help="Commit message for the push.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("float32", "float16", "bfloat16"),
        help="Model dtype when loading from --model_path (ignored for --upload_dir).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")

    if args.upload_dir:
        # Upload existing directory without loading the model
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        if not os.path.isdir(args.upload_dir):
            raise FileNotFoundError(f"Not a directory: {args.upload_dir}")
        logger.info("Creating repo (if needed) and uploading folder %s -> %s", args.upload_dir, args.repo_id)
        api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=args.upload_dir,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )
        logger.info("Done. Model available at https://huggingface.co/%s", args.repo_id)
        return

    # Load model and push via push_to_hub
    import torch
    from alpamayo_r1.models import AlpamayoR1FlashDrive

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    logger.info("Loading model from %s (dtype=%s)", args.model_path, args.dtype)
    model = AlpamayoR1FlashDrive.from_pretrained(args.model_path, dtype=dtype)
    logger.info("Pushing to Hub: %s", args.repo_id)
    model.push_to_hub(
        args.repo_id,
        token=token,
        private=args.private,
        commit_message=args.commit_message,
    )
    logger.info("Done. Model available at https://huggingface.co/%s", args.repo_id)


if __name__ == "__main__":
    main()
