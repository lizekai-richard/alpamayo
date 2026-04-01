#!/usr/bin/env python3
"""Push Alpamayo v1.5 model weights to Hugging Face Hub.

Three models:
    finetuned  - Alpamayo1_5 finetuned base model
    dflash     - DFlash draft model for speculative decoding
    paro       - ParoQuant W4A8 Marlin quantized model

Usage:
    # Push all three models
    python scripts/push_model_to_huggingface.py --all \
        --org nvidia

    # Push a specific model
    python scripts/push_model_to_huggingface.py --model finetuned \
        --repo-id nvidia/Alpamayo1_5-Finetuned

    # Push from custom directory
    python scripts/push_model_to_huggingface.py --upload-dir /path/to/saved_model \
        --repo-id username/model-name

    # Dry run (show what would be uploaded)
    python scripts/push_model_to_huggingface.py --all --org nvidia --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
log = logging.getLogger(__name__)

# Default checkpoint paths and repo names
MODELS = {
    "finetuned": {
        "path": "/data/scratch/zekaili/Alpamayo1_5-Finetuned-new",
        "repo_name": "Alpamayo-1.5-10B-finetuned",
        "description": "Alpamayo1.5 10B FlashDrive finetuned model (bfloat16)",
    },
    "dflash": {
        "path": "/data/scratch/zekaili/Alpamayo1_5-DFlash",
        "repo_name": "Alpamayo-1.5-DFlash",
        "description": "DFlash draft model for Alpamayo1.5 10B speculative decoding",
    },
    "paro": {
        "path": "/data/scratch/zekaili/Alpamayo1_5-finetuned-PARO",
        "repo_name": "Alpamayo-1.5-10B-finetuned-PARO",
        "description": "Alpamayo1.5 10B with ParoQuant quantization and Marlin backend",
    },
}


def upload_directory(
    folder_path: str,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = False,
    commit_message: str = "Upload model weights",
    dry_run: bool = False,
):
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder_path}")

    safetensors = list(folder.glob("*.safetensors"))
    config = folder / "config.json"

    log.info(f"Directory:  {folder}")
    log.info(f"  config.json: {'yes' if config.exists() else 'MISSING'}")
    log.info(f"  safetensors: {len(safetensors)} files")
    for f in sorted(safetensors):
        size_mb = f.stat().st_size / 1024 / 1024
        log.info(f"    {f.name} ({size_mb:.0f} MB)")

    extra_files = [
        f.name for f in folder.iterdir()
        if f.suffix not in (".safetensors",) and f.name != "config.json"
        and not f.name.startswith(".")
    ]
    if extra_files:
        log.info(f"  extra files: {extra_files}")

    if not config.exists():
        log.warning("config.json not found — model may not load correctly")
    if not safetensors:
        log.warning("No .safetensors files found")

    if dry_run:
        log.info(f"  [DRY RUN] Would upload to {repo_id}")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    log.info(f"Creating repo {repo_id} ...")
    api.create_repo(
        repo_id=repo_id,
        private=private,
        exist_ok=True,
        repo_type="model",
    )

    log.info(f"Uploading {folder} -> {repo_id} ...")
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    log.info(f"Done: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Push Alpamayo v1.5 models to Hugging Face Hub",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model", choices=list(MODELS.keys()),
        help="Which model to push (finetuned / dflash / paro)",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Push all three models",
    )
    group.add_argument(
        "--upload-dir", type=str,
        help="Upload a custom directory (requires --repo-id)",
    )

    parser.add_argument("--repo-id", type=str, default=None,
                        help="HF repo id (e.g. nvidia/Alpamayo1_5-Finetuned). "
                             "Required for --upload-dir, optional for --model/--all.")
    parser.add_argument("--org", type=str, default="FlashDriveVLA",
                        help="HF organization (used with --model/--all to form repo_id)")
    parser.add_argument("--token", type=str, default=None,
                        help="HF token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be uploaded without uploading")
    parser.add_argument("--commit-message", type=str, default=None)
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    if args.upload_dir:
        if not args.repo_id:
            parser.error("--repo-id is required with --upload-dir")
        upload_directory(
            args.upload_dir,
            args.repo_id,
            token=token,
            private=args.private,
            commit_message=args.commit_message or "Upload model weights",
            dry_run=args.dry_run,
        )
        return

    # --model or --all
    models_to_push = list(MODELS.keys()) if args.all else [args.model]

    for name in models_to_push:
        info = MODELS[name]
        folder = info["path"]

        if args.repo_id and not args.all:
            repo_id = args.repo_id
        elif args.org:
            repo_id = f"{args.org}/{info['repo_name']}"
        else:
            parser.error("Specify --org or --repo-id")

        log.info(f"\n{'=' * 60}")
        log.info(f"Pushing: {name} ({info['description']})")
        log.info(f"  {folder} -> {repo_id}")
        log.info(f"{'=' * 60}")

        upload_directory(
            folder,
            repo_id,
            token=token,
            private=args.private,
            commit_message=args.commit_message or f"Upload {info['description']}",
            dry_run=args.dry_run,
        )

    log.info("\nAll done.")


if __name__ == "__main__":
    main()
