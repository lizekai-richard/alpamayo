#!/usr/bin/env python3
"""Move dumped clip dirs to MooseFS. Poll for new dumps every 5 minutes.

One-time move: move all existing clip dirs from source to destination.
Loop mode: after optional one-time move, check for new dumps every 5 minutes
and move any completed clip dirs to destination.

Usage:
  # One-time move only (existing dumps)
  python scripts/move_dumped_inputs_to_moosefs.py --once

  # One-time move + poll every 5 min for new dumps (default)
  python scripts/move_dumped_inputs_to_moosefs.py

  # Poll only (no one-time move)
  python scripts/move_dumped_inputs_to_moosefs.py --watch-only

  # Custom interval (e.g. every 10 minutes)
  python scripts/move_dumped_inputs_to_moosefs.py --interval 600

  # Custom paths
  python scripts/move_dumped_inputs_to_moosefs.py --source ./dumped_inputs --dest /mnt/moosefs-1/users/zekail/dumped_inputs
"""

import argparse
import logging
import os
import shutil
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SOURCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dumped_inputs")
DEFAULT_DEST = "/mnt/moosefs-1/users/zekail/dumped_inputs"
DUMP_FILENAME = "sliding_window_inputs.pt"


def move_clip(source_dir: str, dest_dir: str, clip_id: str) -> bool:
    """Move source_dir/clip_id to dest_dir/clip_id. Returns True if moved, False if skipped."""
    src = os.path.join(source_dir, clip_id)
    dst = os.path.join(dest_dir, clip_id)
    pt_path = os.path.join(src, DUMP_FILENAME)
    if not os.path.isfile(pt_path):
        return False
    if os.path.exists(dst):
        logger.info("Already exists at destination, skipping: %s", clip_id)
        return False
    os.makedirs(dest_dir, exist_ok=True)
    try:
        shutil.move(src, dst)
        logger.info("Moved %s -> %s", src, dst)
        return True
    except Exception as e:
        logger.error("Failed to move %s: %s", clip_id, e)
        return False


def move_existing(source_dir: str, dest_dir: str) -> int:
    """Move all existing clip dirs that have sliding_window_inputs.pt. Returns count moved."""
    if not os.path.isdir(source_dir):
        logger.warning("Source dir does not exist: %s", source_dir)
        return 0
    os.makedirs(dest_dir, exist_ok=True)
    moved = 0
    for name in os.listdir(source_dir):
        src_sub = os.path.join(source_dir, name)
        if os.path.isdir(src_sub) and os.path.isfile(os.path.join(src_sub, DUMP_FILENAME)):
            if move_clip(source_dir, dest_dir, name):
                moved += 1
    return moved


def poll_and_move(source_dir: str, dest_dir: str, interval_sec: float) -> None:
    """Every interval_sec seconds, scan source_dir for completed dumps and move them."""
    os.makedirs(dest_dir, exist_ok=True)
    logger.info("Checking %s for %s every %s seconds", source_dir, DUMP_FILENAME, interval_sec)
    while True:
        time.sleep(interval_sec)
        n = move_existing(source_dir, dest_dir)
        if n > 0:
            logger.info("Moved %s clip(s)", n)


def main():
    parser = argparse.ArgumentParser(description="Move dumped inputs to MooseFS and optionally watch for new dumps.")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="Source dumped_inputs root")
    parser.add_argument("--dest", type=str, default=DEFAULT_DEST, help="Destination root (e.g. MooseFS path)")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Only run one-time move of existing dumps, then exit",
    )
    parser.add_argument(
        "--watch-only",
        action="store_true",
        help="Only poll for new dumps; do not move existing ones first",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=300,
        help="Seconds between checks for new dumps (default: 300 = 5 minutes)",
    )
    args = parser.parse_args()
    source = os.path.abspath(args.source)
    dest = os.path.abspath(args.dest)
    if source == dest:
        logger.error("Source and destination must differ")
        sys.exit(1)
    if not args.watch_only:
        n = move_existing(source, dest)
        logger.info("One-time move: %s clip(s) moved", n)
    if args.once:
        return
    poll_and_move(source, dest, args.interval)


if __name__ == "__main__":
    main()
