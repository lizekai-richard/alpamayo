#!/usr/bin/env python3
"""Generate 6 separate videos from demo/run_all_methods.py output.

Each video shows:
- Camera image with trajectory overlay (predicted + ground truth)
- Method label with color coding
- Timing breakdown panel (encode, prefill, decode, diffusion)
- Streaming CoC text with typewriter effect
- Metrics: tokens/s, acceptance rate (DFlash)

Dwell-time mapping:
    Each frame's total pipeline time determines how long it stays on screen.
    N_video_frames = Round(total_ms / 1000 * fps * slowdown)
    This naturally visualizes the speedup: faster methods produce shorter videos.

Usage:
    # Generate all 6 videos
    python demo/generate_videos.py --log-dir ~/exp/demo/run_MMDD_HHMMSS

    # Fast 480p preview
    python demo/generate_videos.py --log-dir ~/exp/demo/run_MMDD_HHMMSS --resolution 480

    # Only specific methods
    python demo/generate_videos.py --log-dir ~/exp/demo/run_MMDD_HHMMSS --only 1,4,6

    # Slower playback for analysis
    python demo/generate_videos.py --log-dir ~/exp/demo/run_MMDD_HHMMSS --slowdown 2
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
    METHODS, METHOD_COLORS, TRAJ_COLOR_BASELINE, TRAJ_COLOR_OPTIMIZED,
    get_font, draw_glass_panel, draw_trajectory,
)


# ─── Frame loading ─────────────────────────────────────────────────────────


def load_frames(log_dir, method_name, clip_id, target_height=720, max_frames=None):
    """Load per-frame data from a method's output folder.

    Skips prefill frames (is_prefill=True) since they have no output.
    Returns list of frame dicts with camera images and metadata.
    """
    method_dir = Path(log_dir) / method_name
    if not method_dir.exists():
        return []

    frame_files = sorted(method_dir.glob("frame_*.json"))
    if not frame_files:
        return []
    if max_frames:
        frame_files = frame_files[:max_frames]

    frames = []
    for ff in tqdm(frame_files, desc=f"  Loading {method_name}"):
        with open(ff) as f:
            entry = json.load(f)

        # Skip prefill frames
        if entry.get("is_prefill", False):
            continue

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

        # Parse trajectories
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
            "encode_ms": entry.get("encode_ms", 0),
            "prefill_ms": entry.get("prefill_ms", 0),
            "decode_ms": entry.get("decode_ms", 0),
            "diffusion_ms": entry.get("diffusion_ms", 0),
            "tokens": entry.get("tokens", 0),
            "tokens_per_sec": entry.get("tokens_per_sec", 0),
            "is_warmup": entry.get("is_warmup", False),
            "coc_tokens": tokens,
            "pred_xyz": pred_xyz,
            "acceptance_rate": entry.get("acceptance_rate"),
            "mean_acceptance_length": entry.get("mean_acceptance_length"),
            "acceptance_lengths": entry.get("acceptance_lengths", []),
        })

    return frames


# ─── Video generation ──────────────────────────────────────────────────────


def generate_video(frames, output_path, method_name, calib, *, fps=60, slowdown=1.0):
    """Generate a single MP4 video for one method.

    Dwell time: each frame stays on screen proportional to its total_ms.
    During dwell, CoC tokens are progressively revealed (typewriter effect).
    """
    if not frames:
        print(f"  No frames for {method_name}, skipping")
        return

    method_info = METHODS.get(method_name, {"label": method_name, "full": method_name})
    method_color = METHOD_COLORS.get(method_name, (255, 255, 255))
    is_baseline = method_name == "1_baseline"
    traj_color = TRAJ_COLOR_BASELINE if is_baseline else TRAJ_COLOR_OPTIMIZED

    img_h, img_w = frames[0]["image"].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (img_w, img_h))

    total_video_frames = sum(
        max(1, int(round(f["total_ms"] / 1000 * fps * slowdown)))
        for f in frames
    )

    print(f"  Generating: {method_info['full']}")
    print(f"    Data frames: {len(frames)}")
    print(f"    Video frames: {total_video_frames} @ {fps} FPS")
    print(f"    Duration: {total_video_frames / fps:.2f}s")

    # Preload fonts
    font_label = get_font(32, weight="bold")
    font_text = get_font(24, weight="medium")
    font_small = get_font(22, weight="bold")

    frame_count = 0

    for frame in tqdm(frames, desc=f"  {method_name}"):
        num_output_frames = max(1, int(round(frame["total_ms"] / 1000 * fps * slowdown)))
        total_tokens = len(frame["coc_tokens"])

        # Build per-iteration token reveal schedule for DFlash.
        # acceptance_lengths tells us how many raw tokens each DFlash iteration
        # accepted.  We distribute the visible CoC *words* across iterations
        # proportionally, then map iterations to video frames evenly.
        acc_lens = frame.get("acceptance_lengths", [])
        if acc_lens and total_tokens > 0:
            num_iters = len(acc_lens)
            total_raw = sum(acc_lens)
            # Cumulative fraction of tokens after each iteration
            cum_frac = []
            running = 0
            for al in acc_lens:
                running += al
                cum_frac.append(running / total_raw)
            # Map each video sub-frame k (1-based) to tokens to show
            frames_per_iter = num_output_frames / num_iters

            def _tokens_for_subframe(k):
                """Number of CoC words to show at video sub-frame k."""
                iter_idx = min(int((k - 1) / frames_per_iter), num_iters - 1)
                return max(1, int(round(cum_frac[iter_idx] * total_tokens)))
        else:
            # Baseline / non-DFlash: linear reveal
            def _tokens_for_subframe(k):
                return max(1, min(total_tokens, int(k / num_output_frames * total_tokens)))

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
            num_tokens_to_show = _tokens_for_subframe(k)
            num_tokens_to_show = min(num_tokens_to_show, total_tokens)
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

    # ─── Summary metric card (5 seconds after last frame) ───
    summary_frames = fps * 5
    # Use last frame's camera image as background
    last_img = frames[-1]["image"].copy()
    # Darken the background
    dark_bg = (last_img * 0.3).astype(np.uint8)

    # Compute aggregate metrics (exclude warmup)
    valid = [f for f in frames if not f.get("is_warmup")]
    if valid:
        avg_total = np.mean([f["total_ms"] for f in valid])
        avg_encode = np.mean([f["encode_ms"] for f in valid])
        avg_prefill = np.mean([f["prefill_ms"] for f in valid])
        avg_decode = np.mean([f["decode_ms"] for f in valid])
        avg_action = np.mean([f["diffusion_ms"] for f in valid])
    else:
        avg_total = avg_encode = avg_prefill = avg_decode = avg_action = 0

    # Build metric lines
    font_title = get_font(36, weight="bold")
    font_metric = get_font(24, weight="medium")

    metric_lines = [
        f"Avg Total:    {avg_total:.1f} ms",
        f"  Encode:     {avg_encode:.1f} ms",
        f"  Prefill:    {avg_prefill:.1f} ms",
        f"  Decode:     {avg_decode:.1f} ms",
        f"  Action:     {avg_action:.1f} ms",
    ]

    # Render the summary card for 5 seconds
    for _ in range(summary_frames):
        canvas = dark_bg.copy()
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).convert("RGBA")
        ui = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(ui)

        # Center panel
        panel_w = 500
        panel_h = 60 + len(metric_lines) * 35 + 20
        px = (img_w - panel_w) // 2
        py = (img_h - panel_h) // 2
        draw_glass_panel(draw, (px, py, px + panel_w, py + panel_h), radius=15)

        # Title
        title = method_info["full"]
        title_bbox = draw.textbbox((0, 0), title, font=font_title)
        title_w = title_bbox[2] - title_bbox[0]
        r, g, b = traj_color
        draw.text(((img_w - title_w) // 2, py + 15), title, font=font_title, fill=(r, g, b))

        # Metric lines
        y = py + 65
        for line in metric_lines:
            draw.text((px + 30, y), line, font=font_metric, fill=(255, 255, 255))
            y += 35

        final = Image.alpha_composite(pil_img, ui)
        canvas = cv2.cvtColor(np.array(final.convert("RGB")), cv2.COLOR_RGB2BGR)
        writer.write(canvas)
        frame_count += 1

    writer.release()
    print(f"    Saved: {output_path} ({frame_count} frames, {frame_count / fps:.2f}s)")


# ─── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate 6 separate videos from demo/run_all_methods.py output",
    )
    parser.add_argument(
        "--log-dir", required=True,
        help="Path to run_all_methods.py output (e.g. ~/exp/demo/run_MMDD_HHMMSS)",
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
    parser.add_argument(
        "--only", default="",
        help="Comma-separated method numbers to generate (e.g. '1,4,6')",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    log_dir = Path(os.path.expanduser(args.log_dir))
    output_dir = Path(os.path.expanduser(args.output_dir)) if args.output_dir else log_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_file = log_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {log_dir}")
    with open(config_file) as f:
        config = json.load(f)
    clip_id = config["clip_id"]

    # Determine which methods to generate
    available = [m for m in METHODS if (log_dir / m).exists()
                 and list((log_dir / m).glob("frame_*.json"))]
    if args.only:
        selected = set(int(x) for x in args.only.split(","))
        methods = [m for m in available if int(m[0]) in selected]
    else:
        methods = available

    if not methods:
        print(f"No method data found in {log_dir}")
        print(f"  Checked: {list(METHODS.keys())}")
        return

    print("=" * 60)
    print("Demo Video Generation")
    print("=" * 60)
    print(f"Log dir:     {log_dir}")
    print(f"Clip:        {clip_id}")
    print(f"Methods:     {methods}")
    print(f"Resolution:  {args.resolution}p")
    print(f"FPS:         {args.fps}")
    print(f"Slowdown:    {args.slowdown}x")
    print(f"Output dir:  {output_dir}")

    # Load camera calibration
    print("\nLoading camera calibration...")
    calib = load_calibration(clip_id, "camera_front_wide_120fov")

    # Scale calibration to target resolution
    if args.resolution != calib.height:
        scale = args.resolution / calib.height
        print(f"  Scaling: {calib.width}x{calib.height} -> "
              f"{int(calib.width * scale)}x{args.resolution}")
        calib.width = int(calib.width * scale)
        calib.height = args.resolution
        calib.cx *= scale
        calib.cy *= scale
        calib.fw_poly = calib.fw_poly * scale

    # Generate one video per method
    for method_name in methods:
        print(f"\n{'=' * 60}")
        print(f"Method: {METHODS[method_name]['full']}")
        print(f"{'=' * 60}")

        frames = load_frames(
            log_dir, method_name, clip_id,
            target_height=args.resolution,
            max_frames=args.max_frames,
        )

        if not frames:
            print(f"  No valid frames, skipping")
            continue

        # Skip first 3 warmup frames
        if len(frames) > 3:
            frames = frames[3:]

        output_path = str(output_dir / f"{method_name}.mp4")
        generate_video(
            frames, output_path, method_name, calib,
            fps=args.fps, slowdown=args.slowdown,
        )

        # Free memory between methods
        del frames

    # Print summary
    print(f"\n{'=' * 60}")
    print("Done! Videos saved:")
    for method_name in methods:
        mp4 = output_dir / f"{method_name}.mp4"
        if mp4.exists():
            size_mb = mp4.stat().st_size / (1024 * 1024)
            print(f"  {mp4.name:<25} ({size_mb:.1f} MB)")
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
