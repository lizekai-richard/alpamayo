#!/usr/bin/env python3
"""Generate a single debug frame (PNG) from demo/run_all_methods.py output.

Quick alternative to generate_videos.py for iterating on the visual layout.

Usage:
    # Render frame 10 from method 6_awq
    python demo/generate_frame.py --log-dir ~/exp/demo/run_MMDD_HHMMSS --method 6_awq --frame 10

    # Render all methods side by side for frame 10
    python demo/generate_frame.py --log-dir ~/exp/demo/run_MMDD_HHMMSS --frame 10 --all

    # Custom output path
    python demo/generate_frame.py --log-dir ~/exp/demo/run_MMDD_HHMMSS --method 4_dflash_stream -o debug.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from camera_projection import load_calibration
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from utils import (
    METHODS, METHOD_COLORS, TRAJ_COLOR_BASELINE, TRAJ_COLOR_OPTIMIZED,
    get_font, draw_glass_panel, draw_trajectory, overlay_traj_plot,
)


# ─── Single frame loading ────────────────────────────────────────────────


def load_single_frame(log_dir, method_name, frame_idx, clip_id, target_height=720, data_dir=None):
    """Load one frame's data and camera image."""
    frame_path = Path(log_dir) / method_name / f"frame_{frame_idx:04d}.json"
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    with open(frame_path) as f:
        entry = json.load(f)

    if entry.get("is_prefill", False):
        raise ValueError(f"Frame {frame_idx} is a prefill frame (no output)")

    t0_us = entry.get("t0_us", 0)

    # Load camera image and ego history
    if data_dir:
        import torch
        pt_path = Path(data_dir) / f"{t0_us}.pt"
        cached = torch.load(pt_path, map_location="cpu", weights_only=True)
        cam_frame = cached["image_frames_nf1"][1, -1].numpy()
        cam_frame = cam_frame.transpose(1, 2, 0)
        img = cv2.cvtColor(cam_frame, cv2.COLOR_RGB2BGR)
        ego_history_xyz = cached["ego_history_xyz"].numpy()
    else:
        data = load_physical_aiavdataset(clip_id, t0_us=t0_us, num_future_steps=1)
        cam_frame = data["image_frames"][1, -1].numpy()
        cam_frame = cam_frame.transpose(1, 2, 0)
        img = cv2.cvtColor(cam_frame, cv2.COLOR_RGB2BGR)
        ego_history_xyz = data["ego_history_xyz"].numpy()

    orig_h, orig_w = img.shape[:2]
    if orig_h != target_height:
        scale = target_height / orig_h
        img = cv2.resize(img, (int(orig_w * scale), target_height), interpolation=cv2.INTER_AREA)

    # Parse trajectories
    pred_xyz = np.array(entry.get("pred_xyz", []))
    if pred_xyz.ndim == 3:
        pred_xyz = pred_xyz[0]

    # Parse ego history: shape [B, groups, T, 3] -> [T, 3]
    if ego_history_xyz.ndim == 4:
        history_xyz = ego_history_xyz[0, 0]  # [T, 3]
    elif ego_history_xyz.ndim == 3:
        history_xyz = ego_history_xyz[0]
    else:
        history_xyz = ego_history_xyz

    # Parse CoC tokens
    coc_text = entry.get("coc", "")
    for tag in ["<|cot_start|>", "<|cot_end|>", "<|traj_future_start|>"]:
        coc_text = coc_text.replace(tag, "")
    coc_text = coc_text.strip()
    tokens = coc_text.split() if coc_text else []

    return {
        "image": img,
        "frame_idx": entry.get("frame", frame_idx),
        "t0_us": t0_us,
        "total_ms": entry.get("total_ms", 0),
        "encode_ms": entry.get("encode_ms", 0),
        "prefill_ms": entry.get("prefill_ms", 0),
        "decode_ms": entry.get("decode_ms", 0),
        "diffusion_ms": entry.get("diffusion_ms", 0),
        "tokens": entry.get("tokens", 0),
        "tokens_per_sec": entry.get("tokens_per_sec", 0),
        "coc_tokens": tokens,
        "pred_xyz": pred_xyz,
        "history_xyz": history_xyz,
        "acceptance_rate": entry.get("acceptance_rate"),
        "mean_acceptance_length": entry.get("mean_acceptance_length"),
    }


# ─── Global trajectory bounds ─────────────────────────────────────────────


def compute_traj_bounds(log_dir, method_name):
    """Scan all frame JSONs for a method and return global (x_min, x_max, y_min, y_max).

    Uses the raw min/max of X (forward) and Y (lateral) across all frames.
    """
    method_dir = Path(log_dir) / method_name
    frame_files = sorted(method_dir.glob("frame_*.json"))
    all_x, all_y = [], []
    for ff in frame_files:
        with open(ff) as f:
            entry = json.load(f)
        if entry.get("is_prefill", False):
            continue
        pred = entry.get("pred_xyz", [])
        if not pred:
            continue
        pts = np.array(pred)
        if pts.ndim == 3:
            pts = pts[0]
        if pts.ndim == 2 and pts.shape[1] >= 2:
            all_x.append(pts[:, 0])
            all_y.append(pts[:, 1])
    if not all_x:
        return None
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    y_abs = max(abs(all_y.min()), abs(all_y.max()))
    return (float(all_x.min()), float(all_x.max()), float(-y_abs), float(y_abs))


# ─── Frame rendering ─────────────────────────────────────────────────────


def render_frame(frame, method_name, calib, traj_bounds=None, show_traj_plot=True):
    """Render a single frame with all overlays. Returns BGR numpy array."""
    method_info = METHODS.get(method_name, {"label": method_name, "full": method_name})
    method_color = METHOD_COLORS.get(method_name, (255, 255, 255))
    is_baseline = method_name == "1_baseline"
    traj_color = TRAJ_COLOR_BASELINE if is_baseline else TRAJ_COLOR_OPTIMIZED

    canvas = frame["image"].copy()
    img_h, img_w = canvas.shape[:2]

    # Draw trajectory
    if len(frame["pred_xyz"]) > 0:
        draw_trajectory(canvas, frame["pred_xyz"], calib, traj_color, width=9)

    # PIL overlay for text panels
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGBA")
    ui = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(ui)

    font_label = get_font(28, weight="bold")
    font_text = get_font(18, weight="medium")
    font_small = get_font(18, weight="bold")
    font_metric = get_font(20, weight="bold")

    # --- CoC text panel (top center, like "Reasoning" box) ---
    visible_tokens = frame["coc_tokens"]
    if visible_tokens:
        max_text_w = int(img_w * 0.5)
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
        lines = lines[-3:]

        from PIL import ImageFilter
        coc_w = int(img_w * 0.4)
        coc_h = 50 + len(lines) * 30 + 20
        cx = (img_w - coc_w) // 2
        cy = int(img_h * 0.20)

        # Blur camera region behind CoC panel
        region = pil_img.crop((cx, cy, cx + coc_w, cy + coc_h))
        region = region.filter(ImageFilter.GaussianBlur(radius=4))
        pil_img.paste(region, (cx, cy))

        draw_glass_panel(draw, (cx, cy, cx + coc_w, cy + coc_h), radius=12)
        draw.text((cx + 12, cy + 8), "Reasoning", font=font_small, fill=(180, 180, 180))
        for i, line in enumerate(lines):
            draw.text((cx + 12, cy + 34 + i * 28), line, font=font_text, fill=(255, 255, 255))

    # --- 2D trajectory plot (bottom right, blurred rounded rectangle) ---
    if show_traj_plot and len(frame["pred_xyz"]) > 0:
        overlay_traj_plot(pil_img, ui, frame, img_w, img_h, traj_bounds)

    # Composite
    final = Image.alpha_composite(pil_img, ui)
    return cv2.cvtColor(np.array(final.convert("RGB")), cv2.COLOR_RGB2BGR)


# ─── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate a single debug frame (PNG)")
    parser.add_argument("--log-dir", required=True, help="Path to run_all_methods.py output")
    parser.add_argument("--method", default=None, help="Method folder name (e.g. 6_awq)")
    parser.add_argument("--frame", type=int, default=10, help="Frame index to render")
    parser.add_argument("--resolution", type=int, default=720, choices=[480, 720, 1080])
    parser.add_argument("--all", action="store_true", help="Render all available methods side by side")
    parser.add_argument("--data-dir", default=None, help="Pre-cached data directory")
    parser.add_argument("--no-traj-plot", action="store_true", help="Disable 2D trajectory plot overlay")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path")
    args = parser.parse_args()

    log_dir = Path(os.path.expanduser(args.log_dir))

    # Load config
    config_file = log_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {log_dir}")
    with open(config_file) as f:
        config = json.load(f)
    clip_id = config["clip_id"]

    if args.data_dir:
        args.data_dir = os.path.expanduser(args.data_dir)

    # Load camera calibration
    print("Loading camera calibration...")
    calib = load_calibration(clip_id, "camera_front_wide_120fov")
    if args.resolution != calib.height:
        scale = args.resolution / calib.height
        calib.width = int(calib.width * scale)
        calib.height = args.resolution
        calib.cx *= scale
        calib.cy *= scale
        calib.fw_poly = calib.fw_poly * scale

    # Determine methods
    available = sorted([m for m in METHODS if (log_dir / m).exists()
                        and list((log_dir / m).glob("frame_*.json"))])

    if args.all:
        methods = available
    elif args.method:
        if args.method not in METHODS:
            print(f"Unknown method '{args.method}'. Available: {available}")
            return
        methods = [args.method]
    else:
        # Default to last available method
        methods = [available[-1]] if available else []

    if not methods:
        print(f"No method data found in {log_dir}")
        return

    # Compute global trajectory bounds across all frames for consistent axes
    print("Computing global trajectory bounds...")
    all_bounds = [compute_traj_bounds(log_dir, m) for m in methods]
    valid_bounds = [b for b in all_bounds if b is not None]
    if valid_bounds:
        traj_bounds = (
            min(b[0] for b in valid_bounds),
            max(b[1] for b in valid_bounds),
            min(b[2] for b in valid_bounds),
            max(b[3] for b in valid_bounds),
        )
        print(f"  X: [{traj_bounds[0]:.1f}, {traj_bounds[1]:.1f}]  Y: [{traj_bounds[2]:.1f}, {traj_bounds[3]:.1f}]")
    else:
        traj_bounds = None

    rendered = []
    for method_name in methods:
        print(f"Rendering {method_name} frame {args.frame}...")
        try:
            frame = load_single_frame(
                log_dir, method_name, args.frame, clip_id,
                target_height=args.resolution, data_dir=args.data_dir,
            )
            img = render_frame(frame, method_name, calib, traj_bounds=traj_bounds, show_traj_plot=not args.no_traj_plot)
            rendered.append((method_name, img))
        except (FileNotFoundError, ValueError) as e:
            print(f"  Skipping {method_name}: {e}")

    if not rendered:
        print("No frames rendered.")
        return

    # Compose output
    if len(rendered) == 1:
        output_img = rendered[0][1]
    else:
        # Stack vertically for multi-method view
        output_img = np.vstack([img for _, img in rendered])

    # Save
    if args.output:
        output_path = args.output
    else:
        suffix = "all" if args.all else rendered[0][0]
        output_path = str(log_dir / f"debug_frame{args.frame:04d}_{suffix}.png")

    cv2.imwrite(output_path, output_img)
    print(f"Saved: {output_path} ({output_img.shape[1]}x{output_img.shape[0]})")


if __name__ == "__main__":
    main()
