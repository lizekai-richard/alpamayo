#!/usr/bin/env python3
"""Generate demo videos from per-clip inference results (demo_results/ format).

Each video shows:
- Camera image with trajectory overlay (predicted)
- Streaming CoC text with typewriter effect
- Summary metric card at the end

Dwell-time mapping:
    Each frame's total pipeline time determines how long it stays on screen.
    N_video_frames = Round(total_ms / 1000 * fps * slowdown)
    This naturally visualizes the inference speed.

Input format (demo_results/):
    demo_results/
    +-- <clip_id>/
        +-- frame_0000.json   # {frame, t0_us, is_prefill, is_warmup,
        +-- frame_0001.json   #  total_ms, coc, pred_xyz, min_ade_k, min_ade_1}
        +-- ...

Usage:
    # Generate videos for all clips in demo_results/
    python demo/generate_videos_clips.py --log-dir demo/demo_results

    # Generate video for a single clip
    python demo/generate_videos_clips.py --log-dir demo/demo_results/<clip_id>

    # Fast 480p preview
    python demo/generate_videos_clips.py --log-dir demo/demo_results --resolution 480

    # Slower playback for analysis
    python demo/generate_videos_clips.py --log-dir demo/demo_results --slowdown 2
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

# Add src and video to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from camera_projection import load_calibration
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from utils import (
    TRAJ_COLOR_BASELINE,
    get_font, draw_glass_panel, draw_trajectory,
)


# ─── Frame loading ─────────────────────────────────────────────────────────


def load_frames(clip_dir, clip_id, target_height=720, max_frames=None):
    """Load per-frame data from a clip directory.

    Args:
        clip_dir: Path to directory containing frame_XXXX.json files.
        clip_id: Clip ID for loading camera images from the dataset.
        target_height: Target image height for resizing.
        max_frames: Optional limit on number of frames to load.

    Returns list of frame dicts with camera images and metadata.
    """
    clip_dir = Path(clip_dir)
    frame_files = sorted(clip_dir.glob("frame_*.json"))
    if not frame_files:
        return []
    if max_frames:
        frame_files = frame_files[:max_frames]

    frames = []
    for ff in tqdm(frame_files, desc=f"  Loading {clip_id[:12]}..."):
        with open(ff) as f:
            entry = json.load(f)

        t0_us = entry.get("t0_us", 0)

        # Load camera image from dataset
        try:
            data = load_physical_aiavdataset(clip_id, t0_us=t0_us, num_future_steps=1)
            image_frames = data["image_frames"]
            # Camera order: cross_left(0), front_wide(1), cross_right(2), front_tele(3)
            cam_frame = image_frames[1, -1].numpy()  # front_wide, last frame
            cam_frame = cam_frame.transpose(1, 2, 0)  # (3,H,W) -> (H,W,3)
            img = cv2.cvtColor(cam_frame, cv2.COLOR_RGB2BGR)

            orig_h, orig_w = img.shape[:2]
            if orig_h != target_height:
                scale = target_height / orig_h
                img = cv2.resize(
                    img, (int(orig_w * scale), target_height),
                    interpolation=cv2.INTER_AREA,
                )
        except Exception:
            target_width = int(target_height * 16 / 9)
            img = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        # Parse trajectories — pred_xyz may be [samples, T, 3] or [T, 3]
        pred_xyz = np.array(entry.get("pred_xyz", []))
        if pred_xyz.ndim == 3:
            pred_xyz = pred_xyz[0]  # First sample

        # Parse CoC tokens
        coc_text = entry.get("coc", "")
        for tag in ["<|cot_start|>", "<|cot_end|>", "<|traj_future_start|>"]:
            coc_text = coc_text.replace(tag, "")
        coc_text = coc_text.strip()
        tokens = coc_text.split() if coc_text else []

        frames.append({
            "image": img,
            "frame_idx": entry.get("frame", len(frames)),
            "t0_us": t0_us,
            "total_ms": entry.get("total_ms", 100),
            "is_warmup": entry.get("is_warmup", False),
            "coc_tokens": tokens,
            "pred_xyz": pred_xyz,
        })

    return frames


# ─── Video generation ──────────────────────────────────────────────────────


def generate_video(frames, output_path, clip_id, calib, *, fps=60, slowdown=1.0):
    """Generate a single MP4 video for one clip.

    Dwell time: each frame stays on screen proportional to its total_ms.
    During dwell, CoC tokens are progressively revealed (typewriter effect).
    """
    if not frames:
        print(f"  No frames for {clip_id}, skipping")
        return

    traj_color = TRAJ_COLOR_BASELINE

    img_h, img_w = frames[0]["image"].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (img_w, img_h))

    total_video_frames = sum(
        max(1, int(round(f["total_ms"] / 1000 * fps * slowdown)))
        for f in frames
    )

    print(f"  Generating: {clip_id}")
    print(f"    Data frames: {len(frames)}")
    print(f"    Video frames: {total_video_frames} @ {fps} FPS")
    print(f"    Duration: {total_video_frames / fps:.2f}s")

    # Preload fonts
    font_text = get_font(24, weight="medium")
    font_small = get_font(22, weight="bold")

    frame_count = 0

    for frame in tqdm(frames, desc=f"  {clip_id[:12]}..."):
        num_output_frames = max(1, int(round(frame["total_ms"] / 1000 * fps * slowdown)))
        total_tokens = len(frame["coc_tokens"])

        for k in range(1, num_output_frames + 1):
            canvas = frame["image"].copy()

            # Draw trajectories on camera image
            if len(frame["pred_xyz"]) > 0:
                draw_trajectory(canvas, frame["pred_xyz"], calib, traj_color, width=9)

            # ─── PIL overlay for text and panels ───
            img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb).convert("RGBA")
            ui = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(ui)

            # --- CoC text panel (top center, "Reasoning" style) ---
            # Linear token reveal (typewriter effect)
            if total_tokens > 0:
                num_tokens_to_show = max(1, min(total_tokens, int(k / num_output_frames * total_tokens)))
            else:
                num_tokens_to_show = 0
            visible_tokens = frame["coc_tokens"][:num_tokens_to_show]

            if visible_tokens:
                coc_w = int(img_w * 0.5)
                max_text_w = coc_w - 30
                lines = []
                cur = ""
                for word in visible_tokens:
                    test = cur + (" " if cur else "") + word
                    bbox = draw.textbbox((0, 0), test, font=font_text)
                    if (bbox[2] - bbox[0]) < max_text_w:
                        cur = test
                    else:
                        if cur:
                            lines.append(cur)
                        cur = word
                if cur:
                    lines.append(cur)
                lines = lines[-5:]  # Show last 5 lines

                coc_h = 55 + len(lines) * 36 + 20
                cx = (img_w - coc_w) // 2
                cy = int(img_h * 0.15)

                # Blur camera region behind panel
                region = pil_img.crop((cx, cy, cx + coc_w, cy + coc_h))
                region = region.filter(ImageFilter.GaussianBlur(radius=4))
                pil_img.paste(region, (cx, cy))

                draw_glass_panel(draw, (cx, cy, cx + coc_w, cy + coc_h), radius=12)
                draw.text((cx + 15, cy + 10), "Reasoning", font=font_small, fill=(180, 180, 180))
                for i, line in enumerate(lines):
                    draw.text(
                        (cx + 15, cy + 40 + i * 36), line,
                        font=font_text, fill=(255, 255, 255),
                    )

                # Blinking cursor while generating
                if num_tokens_to_show < total_tokens and (frame_count // 10) % 2 == 0:
                    last = lines[-1] if lines else ""
                    bbox = draw.textbbox((0, 0), last, font=font_text)
                    cursor_x = cx + 15 + bbox[2] - bbox[0] + 3
                    cursor_y = cy + 40 + (len(lines) - 1) * 36
                    draw.text((cursor_x, cursor_y), "|", font=font_text, fill=(200, 200, 255))

            # Composite and write
            final = Image.alpha_composite(pil_img, ui)
            canvas = cv2.cvtColor(np.array(final.convert("RGB")), cv2.COLOR_RGB2BGR)
            writer.write(canvas)
            frame_count += 1

    writer.release()
    print(f"    Saved: {output_path} ({frame_count} frames, {frame_count / fps:.2f}s)")


# ─── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate demo videos from per-clip inference results",
    )
    parser.add_argument(
        "--log-dir", required=True,
        help="Path to demo_results/ dir or a single clip dir",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for videos (default: same as log-dir)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")
    parser.add_argument(
        "--slowdown", type=float, default=1.0,
        help="Slowdown factor (2.0 = 2x slower for analysis)",
    )
    parser.add_argument(
        "--resolution", type=int, default=720, choices=[480, 720, 1080],
        help="Video resolution height (default: 720p)",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    log_dir = Path(os.path.expanduser(args.log_dir))
    output_dir = Path(os.path.expanduser(args.output_dir)) if args.output_dir else log_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect clip directories.
    # If log_dir itself contains frame_*.json, treat it as a single clip dir.
    # Otherwise, look for subdirectories that contain frame_*.json files.
    if list(log_dir.glob("frame_*.json")):
        clip_dirs = [(log_dir.name, log_dir)]
    else:
        clip_dirs = sorted([
            (d.name, d)
            for d in log_dir.iterdir()
            if d.is_dir() and list(d.glob("frame_*.json"))
        ])

    if not clip_dirs:
        print(f"No frame data found in {log_dir}")
        print("  Expected: frame_*.json files in the directory or its subdirectories")
        return

    print("=" * 60)
    print("Demo Video Generation")
    print("=" * 60)
    print(f"Log dir:     {log_dir}")
    print(f"Clips:       {[cid for cid, _ in clip_dirs]}")
    print(f"Resolution:  {args.resolution}p")
    print(f"FPS:         {args.fps}")
    print(f"Slowdown:    {args.slowdown}x")
    print(f"Output dir:  {output_dir}")

    for clip_id, clip_dir in clip_dirs:
        print(f"\n{'=' * 60}")
        print(f"Clip: {clip_id}")
        print(f"{'=' * 60}")

        # Load camera calibration per clip
        print("  Loading camera calibration...")
        calib = load_calibration(clip_id, "camera_front_wide_120fov")

        if args.resolution != calib.height:
            scale = args.resolution / calib.height
            print(f"    Scaling: {calib.width}x{calib.height} -> "
                  f"{int(calib.width * scale)}x{args.resolution}")
            calib.width = int(calib.width * scale)
            calib.height = args.resolution
            calib.cx *= scale
            calib.cy *= scale
            calib.fw_poly = calib.fw_poly * scale

        frames = load_frames(
            clip_dir, clip_id,
            target_height=args.resolution,
            max_frames=args.max_frames,
        )

        if not frames:
            print("  No valid frames, skipping")
            continue

        # Skip warmup frames at the start
        non_warmup_start = 0
        for i, f in enumerate(frames):
            if not f.get("is_warmup", False):
                non_warmup_start = i
                break
        if non_warmup_start > 0:
            print(f"  Skipping {non_warmup_start} warmup frames")
            frames = frames[non_warmup_start:]

        output_path = str(output_dir / f"{clip_id}.mp4")
        generate_video(
            frames, output_path, clip_id, calib,
            fps=args.fps, slowdown=args.slowdown,
        )

        del frames

    # Print summary
    print(f"\n{'=' * 60}")
    print("Done! Videos saved:")
    for clip_id, _ in clip_dirs:
        mp4 = output_dir / f"{clip_id}.mp4"
        if mp4.exists():
            size_mb = mp4.stat().st_size / (1024 * 1024)
            print(f"  {mp4.name:<45} ({size_mb:.1f} MB)")
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
