#!/usr/bin/env python3
"""Pre-download PhysicalAI-AV clip data so later load_physical_aiavdataset() reads from cache.

The dataset streams from HuggingFace when data is not local. This script touches each clip
once (minimal load) so that the HuggingFace cache is populated; subsequent runs of
dump_data.py or test_inference.py will then read from cache and be much faster.

Usage:
  # From repo root, with PYTHONPATH=src
  python scripts/predownload_clips.py --clip_ids_file clips_for_train.json
  python scripts/predownload_clips.py --clip_ids_file clips_for_train.json --workers 8

Environment (optional):
  HF_HOME or HF_DATASETS_CACHE: set to a fast local disk path for cache (e.g. SSD).
  HUGGING_FACE_HUB_TOKEN: if the dataset is gated, login first with huggingface-cli login.
"""

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _download_one(args_tuple):
    clip_id, t0_us = args_tuple
    try:
        from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

        load_physical_aiavdataset(
            clip_id,
            t0_us=t0_us,
            num_frames=1,
            maybe_stream=True,
        )
        return clip_id, None
    except Exception as e:
        return clip_id, e


def main():
    parser = argparse.ArgumentParser(description="Pre-download PhysicalAI-AV clips to HuggingFace cache.")
    parser.add_argument("--clip_ids_file", type=str, required=True, help="JSON file with list of clip IDs.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default 4).")
    parser.add_argument("--t0_us", type=int, default=2_000_000, help="t0 used for the minimal load (default 2s).")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N clips (for testing).")
    args = parser.parse_args()

    with open(args.clip_ids_file) as f:
        clip_ids = json.load(f)
    if not isinstance(clip_ids, list):
        clip_ids = list(clip_ids)
    if args.limit is not None:
        clip_ids = clip_ids[: args.limit]
        logger.info("Limited to first %s clips", len(clip_ids))

    cache_dir = os.environ.get("HF_HOME") or os.environ.get("HF_DATASETS_CACHE")
    if cache_dir:
        logger.info("Using cache dir: %s", cache_dir)
    else:
        logger.info("Using default HuggingFace cache (set HF_HOME or HF_DATASETS_CACHE for a custom path).")

    logger.info("Pre-downloading %s clips (workers=%s) ...", len(clip_ids), args.workers)

    from multiprocessing import Pool

    work = [(cid, args.t0_us) for cid in clip_ids]
    failed = []
    with Pool(processes=args.workers) as pool:
        for i, (clip_id, err) in enumerate(pool.imap_unordered(_download_one, work, chunksize=1)):
            if err is not None:
                failed.append((clip_id, err))
                logger.warning("Failed %s: %s", clip_id, err)
            if (i + 1) % 100 == 0:
                logger.info("Progress %s/%s", i + 1, len(clip_ids))

    if failed:
        logger.warning("Failed %s clips (of %s)", len(failed), len(clip_ids))
        sys.exit(1)
    logger.info("Done. %s clips cached.", len(clip_ids))


if __name__ == "__main__":
    # Ensure repo src is on path when run as script
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(repo_root, "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)
    main()
