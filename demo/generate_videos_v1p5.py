#!/usr/bin/env python3
"""Generate demo videos from v1.5 inference results.

Reads per-frame JSON from run_v1p5.py output and produces one MP4 per
clip+method combination. Each video shows:
  - Camera image with projected trajectory overlay
  - Streaming CoC text with typewriter effect
  - Blinking cursor during token generation

Dwell-time mapping:
    Each frame's total pipeline time determines how long it stays on screen.
    N_video_frames = Round(total_ms / 1000 * fps * slowdown)
    Faster methods → shorter videos, directly visualizing the speedup.

Input format (from run_v1p5.py):
    <run_dir>/<clip_id>/baseline/frame_XXXX.json
    <run_dir>/<clip_id>/optimized/frame_XXXX.json
    <run_dir>/<clip_id>/summary.json

Usage:
    # All clips and methods
    python demo/generate_videos_v1p5.py --log-dir demo_v1p5/run_MMDD_HHMMSS

    # Single clip
    python demo/generate_videos_v1p5.py --log-dir demo_v1p5/run_MMDD_HHMMSS --clip <uuid>

    # Only one method
    python demo/generate_videos_v1p5.py --log-dir demo_v1p5/run_MMDD_HHMMSS --only optimized

    # 480p preview, 2x slowdown
    python demo/generate_videos_v1p5.py --log-dir demo_v1p5/run_MMDD_HHMMSS --resolution 480 --slowdown 2
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from camera_projection import load_calibration
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from utils import (
    TRAJ_COLOR_BASELINE, TRAJ_COLOR_OPTIMIZED,
    get_font, draw_glass_panel, draw_trajectory,
)

V1P5_METHODS = {
    "baseline":  {"label": "Baseline",  "color": TRAJ_COLOR_BASELINE},
    "optimized": {"label": "Optimized", "color": TRAJ_COLOR_OPTIMIZED},
}


# ─── Frame loading ───────────────────────────────────────────────────────


def load_frames(method_dir, clip_id, target_height=720, max_frames=None):
    """Load per-frame data from a method's output folder.

    Camera images are fetched from the physical_ai_av dataset API
    (uses local HuggingFace cache when available).

    Returns list of frame dicts with camera images and metadata.
    """
    method_dir = Path(method_dir)
    frame_files = sorted(method_dir.glob("frame_*.json"))
    if not frame_files:
        return []
    if max_frames:
        frame_files = frame_files[:max_frames]

    frames = []
    for ff in tqdm(frame_files, desc=f"  Loading {method_dir.name}", leave=False):
        with open(ff) as f:
            entry = json.load(f)

        if entry.get("is_prefill", False):
            continue

        t0_us = entry.get("t0_us", 0)

        # Load camera image from dataset
        try:
            data = load_physical_aiavdataset(
                clip_id, t0_us=t0_us, num_future_steps=1,
            )
            cam_frame = data["image_frames"][1, -1].numpy()
            cam_frame = cam_frame.transpose(1, 2, 0)
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

        # Parse trajectory
        pred_xyz = np.array(entry.get("pred_xyz", []))
        if pred_xyz.ndim == 3:
            pred_xyz = pred_xyz[0]

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
            "acceptance_lengths": entry.get("acceptance_lengths", []),
        })

    return frames


# ─── Video generation ────────────────────────────────────────────────────


def generate_video(frames, output_path, method_name, calib, *, fps=30,
                   slowdown=1.0):
    """Generate one MP4 video for a single clip+method.

    Dwell time per frame is proportional to total_ms.
    CoC tokens are revealed progressively (typewriter effect).
    """
    if not frames:
        print(f"  No frames for {method_name}, skipping")
        return

    method_info = V1P5_METHODS.get(method_name, {"label": method_name,
                                                   "color": (255, 255, 255)})
    traj_color = method_info["color"]

    img_h, img_w = frames[0]["image"].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (img_w, img_h))

    total_video_frames = sum(
        max(1, int(round(f["total_ms"] / 1000 * fps * slowdown)))
        for f in frames
    )

    print(f"  Generating: {method_info['label']}")
    print(f"    Data frames: {len(frames)}")
    print(f"    Video frames: {total_video_frames} @ {fps} FPS")
    print(f"    Duration: {total_video_frames / fps:.2f}s")

    font_text = get_font(30, weight="medium")
    font_small = get_font(28, weight="bold")

    frame_count = 0

    for frame in tqdm(frames, desc=f"  {method_name}"):
        num_output_frames = max(
            1, int(round(frame["total_ms"] / 1000 * fps * slowdown))
        )
        total_tokens = len(frame["coc_tokens"])

        # Build token reveal schedule
        acc_lens = frame.get("acceptance_lengths", [])
        if acc_lens and total_tokens > 0:
            num_iters = len(acc_lens)
            total_raw = sum(acc_lens)
            cum_frac = []
            running = 0
            for al in acc_lens:
                running += al
                cum_frac.append(running / total_raw)
            frames_per_iter = num_output_frames / num_iters

            def _tokens_for_subframe(k):
                iter_idx = min(int((k - 1) / frames_per_iter), num_iters - 1)
                return max(1, int(round(cum_frac[iter_idx] * total_tokens)))
        else:
            def _tokens_for_subframe(k):
                return max(1, min(total_tokens,
                                  int(k / num_output_frames * total_tokens)))

        for k in range(1, num_output_frames + 1):
            canvas = frame["image"].copy()

            # Draw trajectory on camera image
            if len(frame["pred_xyz"]) > 0:
                draw_trajectory(canvas, frame["pred_xyz"], calib, traj_color,
                                width=9)

            # PIL overlay for text panels
            img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb).convert("RGBA")
            ui = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(ui)

            # CoC text panel (top center)
            num_tokens_to_show = _tokens_for_subframe(k) if total_tokens > 0 else 0
            num_tokens_to_show = min(num_tokens_to_show, total_tokens)
            visible_tokens = frame["coc_tokens"][:num_tokens_to_show]

            if visible_tokens:
                coc_w = int(img_w * 0.6)
                max_text_w = coc_w - 40
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
                lines = lines[-5:]

                line_h = 44
                coc_h = 65 + len(lines) * line_h + 25
                cx = (img_w - coc_w) // 2
                cy = int(img_h * 0.13)

                # Blur behind panel
                region = pil_img.crop((cx, cy, cx + coc_w, cy + coc_h))
                region = region.filter(ImageFilter.GaussianBlur(radius=4))
                pil_img.paste(region, (cx, cy))

                draw_glass_panel(draw, (cx, cy, cx + coc_w, cy + coc_h),
                                 radius=14)
                draw.text((cx + 18, cy + 12), "Reasoning", font=font_small,
                          fill=(180, 180, 180))
                for i, line in enumerate(lines):
                    draw.text((cx + 18, cy + 50 + i * line_h), line,
                              font=font_text, fill=(255, 255, 255))

                # Blinking cursor
                if num_tokens_to_show < total_tokens and (frame_count // 10) % 2 == 0:
                    last = lines[-1] if lines else ""
                    bbox = draw.textbbox((0, 0), last, font=font_text)
                    cursor_x = cx + 18 + bbox[2] - bbox[0] + 4
                    cursor_y = cy + 50 + (len(lines) - 1) * line_h
                    draw.text((cursor_x, cursor_y), "|", font=font_text,
                              fill=(200, 200, 255))

            # Composite and write
            final = Image.alpha_composite(pil_img, ui)
            canvas = cv2.cvtColor(np.array(final.convert("RGB")),
                                  cv2.COLOR_RGB2BGR)
            writer.write(canvas)
            frame_count += 1

    writer.release()
    print(f"    Saved: {output_path} ({frame_count} frames, "
          f"{frame_count / fps:.2f}s)")


# ─── Main ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate demo videos from v1.5 run_v1p5.py output",
    )
    parser.add_argument("--log-dir", required=True,
                        help="Path to run_v1p5.py output (run_<timestamp> dir)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as log-dir)")
    parser.add_argument("--clip", default=None,
                        help="Generate for a single clip ID only")
    parser.add_argument("--only", default=None,
                        choices=["baseline", "optimized"],
                        help="Generate only one method")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--slowdown", type=float, default=1.0,
                        help="Slowdown factor (2.0 = 2x slower)")
    parser.add_argument("--resolution", type=int, default=720,
                        choices=[480, 720, 1080])
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--cache-dir", default="~/.cache/huggingface")
    args = parser.parse_args()

    log_dir = Path(os.path.expanduser(args.log_dir))
    output_dir = Path(os.path.expanduser(args.output_dir)) if args.output_dir else log_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover clips: subdirs that contain baseline/ or optimized/
    if args.clip:
        clip_dirs = [(args.clip, log_dir / args.clip)]
    else:
        clip_dirs = sorted([
            (d.name, d)
            for d in log_dir.iterdir()
            if d.is_dir() and any(
                (d / m).is_dir() and list((d / m).glob("frame_*.json"))
                for m in V1P5_METHODS
            )
        ])

    if not clip_dirs:
        print(f"No clip data found in {log_dir}")
        return

    # Discover methods
    if args.only:
        methods_to_gen = [args.only]
    else:
        methods_to_gen = list(V1P5_METHODS.keys())

    print("=" * 60)
    print("Video Generation: Alpamayo v1.5")
    print("=" * 60)
    print(f"Log dir:     {log_dir}")
    print(f"Clips:       {[cid for cid, _ in clip_dirs]}")
    print(f"Methods:     {methods_to_gen}")
    print(f"Resolution:  {args.resolution}p")
    print(f"FPS:         {args.fps}")
    print(f"Slowdown:    {args.slowdown}x")
    print(f"Output:      {output_dir}")

    for clip_id, clip_dir in clip_dirs:
        print(f"\n{'=' * 60}")
        print(f"Clip: {clip_id}")
        print(f"{'=' * 60}")

        # Load camera calibration
        print("  Loading camera calibration...")
        calib = load_calibration(clip_id, "camera_front_wide_120fov")
        if args.resolution != calib.height:
            scale = args.resolution / calib.height
            calib.width = int(calib.width * scale)
            calib.height = args.resolution
            calib.cx *= scale
            calib.cy *= scale
            calib.fw_poly = calib.fw_poly * scale

        for method_name in methods_to_gen:
            method_dir = clip_dir / method_name
            if not method_dir.is_dir():
                continue
            if not list(method_dir.glob("frame_*.json")):
                continue

            print(f"\n  Method: {method_name}")
            frames = load_frames(
                method_dir, clip_id,
                target_height=args.resolution,
                max_frames=args.max_frames,
            )
            if not frames:
                print("    No valid frames, skipping")
                continue

            # Skip warmup frames
            non_warmup_start = 0
            for i, f in enumerate(frames):
                if not f.get("is_warmup", False):
                    non_warmup_start = i
                    break
            if non_warmup_start > 0:
                print(f"    Skipping {non_warmup_start} warmup frames")
                frames = frames[non_warmup_start:]

            video_path = str(output_dir / f"{clip_id}_{method_name}.mp4")
            generate_video(
                frames, video_path, method_name, calib,
                fps=args.fps, slowdown=args.slowdown,
            )
            del frames

    # Summary
    print(f"\n{'=' * 60}")
    print("Done! Videos saved:")
    for clip_id, _ in clip_dirs:
        for method_name in methods_to_gen:
            mp4 = output_dir / f"{clip_id}_{method_name}.mp4"
            if mp4.exists():
                size_mb = mp4.stat().st_size / (1024 * 1024)
                print(f"  {mp4.name:<60} ({size_mb:.1f} MB)")
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
