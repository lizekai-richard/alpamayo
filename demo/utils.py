"""Shared constants and rendering utilities for demo scripts."""

import os

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


# ─── Constants ───────────────────────────────────────────────────────────

METHODS = {
    "1_baseline":      {"label": "Baseline",    "full": "Baseline",
                        "rows": ["Baseline"]},
    "2_sys_opt":       {"label": "+Sys Opt",    "full": "Baseline + Sys Opt",
                        "rows": ["Baseline", "+ Sys Opt"]},
    "3_dflash":        {"label": "+DFlash",     "full": "Baseline + Sys Opt + DFlash",
                        "rows": ["Baseline", "+ Sys Opt", "+ DFlash"]},
    "4_dflash_stream": {"label": "+Streaming",  "full": "Baseline + Sys Opt + DFlash + Streaming",
                        "rows": ["Baseline", "+ Sys Opt", "+ DFlash", "+ Streaming"]},
    "5_dflash_4step":  {"label": "+4-step",     "full": "Baseline + Sys Opt + DFlash + Streaming + 4-step",
                        "rows": ["Baseline", "+ Sys Opt", "+ DFlash", "+ Streaming", "+ 4-step"]},
    "6_awq":           {"label": "+AWQ INT4",   "full": "Baseline + Sys Opt + DFlash + Streaming + 4-step + AWQ",
                        "rows": ["Baseline", "+ Sys Opt", "+ DFlash", "+ Streaming", "+ 4-step", "+ AWQ"]},
    "7_paroquant":     {"label": "+ParoQuant", "full": "Baseline + Sys Opt + DFlash + Streaming + 4-step + ParoQuant",
                        "rows": ["Baseline", "+ Sys Opt", "+ DFlash", "+ Streaming", "+ 4-step", "+ ParoQuant"]},
}

# Label colors per method (RGB)
METHOD_COLORS = {
    "1_baseline":      (255, 100, 100),
    "2_sys_opt":       (255, 180, 80),
    "3_dflash":        (240, 240, 100),
    "4_dflash_stream": (100, 255, 100),
    "5_dflash_4step":  (100, 255, 255),
    "6_awq":           (100, 180, 255),
    "7_paroquant":     (200, 130, 255),
}

# Trajectory colors (RGB): baseline = red, optimized methods = lime green
TRAJ_COLOR_BASELINE = (255, 80, 60)
TRAJ_COLOR_OPTIMIZED = (160, 255, 50)

SF_PRO_DIR = os.path.expanduser("~/.local/share/fonts/SFPro")


# ─── Rendering utilities ────────────────────────────────────────────────


def get_font(size, monospace=False, weight="bold", variant="auto"):
    """Load font with SF Pro preferred, DejaVu as fallback.

    Args:
        size: Font size in points.
        monospace: Use monospace font (falls back to DejaVu Mono).
        weight: One of 'regular', 'medium', 'bold', 'heavy', 'black'.
        variant: 'display' for titles, 'text' for body, 'auto' picks by size.
    """
    weight_map = {
        "regular": "Regular",
        "medium": "Medium",
        "bold": "Bold",
        "heavy": "Heavy",
        "black": "Black",
    }
    sf_weight = weight_map.get(weight, "Bold")

    # SF Pro Text is designed for <=19pt, Display for >=20pt
    if variant == "auto":
        sf_family = "Text" if size <= 19 else "Display"
    else:
        sf_family = "Text" if variant == "text" else "Display"

    if monospace:
        paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
    else:
        paths = [
            os.path.join(SF_PRO_DIR, f"SF-Pro-{sf_family}-{sf_weight}.otf"),
            os.path.join(SF_PRO_DIR, f"SF-Pro-{sf_family}-Regular.otf"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_glass_panel(draw, box, radius=15):
    """Draw a semi-transparent glassmorphism panel."""
    draw.rounded_rectangle(box, radius=radius, fill=(20, 20, 20, 160))
    draw.rounded_rectangle(box, radius=radius, outline=(255, 255, 255, 30), width=1)


def draw_rotated_text(target, pos, text, font, fill=(200, 200, 200), angle=90,
                      center_in=None):
    """Render text into a tight RGBA image, rotate it, and paste onto target.

    Args:
        target: PIL RGBA Image to paste onto.
        pos: (x, y) position on the target. If center_in is given, x is used
            as-is and y is ignored (computed from the region).
        text: The string to draw.
        font: PIL ImageFont.
        fill: Text color (R, G, B) or (R, G, B, A).
        angle: Counter-clockwise rotation in degrees (default 90 = vertical).
        center_in: Optional (y_start, height) region to vertically center the
            rotated text within.
    """
    tmp = Image.new("RGBA", (400, 40), (0, 0, 0, 0))
    tmp_draw = ImageDraw.Draw(tmp)
    tmp_draw.text((0, 0), text, font=font, fill=fill)
    bbox = tmp_draw.textbbox((0, 0), text, font=font)
    tmp = tmp.crop((0, 0, bbox[2] + 2, bbox[3] + 2))
    tmp = tmp.rotate(angle, expand=True)
    x = pos[0]
    if center_in is not None:
        region_y, region_h = center_in
        y = region_y + (region_h - tmp.height) // 2
    else:
        y = pos[1]
    target.paste(tmp, (x, y), tmp)


def smooth_trajectory_3d(points_3d, num_points=50):
    """B-spline interpolation in 3D ego frame."""
    from scipy.interpolate import make_interp_spline

    if len(points_3d) < 3:
        return points_3d
    t = np.linspace(0, 1, len(points_3d))
    t_new = np.linspace(0, 1, num_points)
    try:
        sx = make_interp_spline(t, points_3d[:, 0], k=2)
        sy = make_interp_spline(t, points_3d[:, 1], k=2)
        sz = make_interp_spline(t, points_3d[:, 2], k=2)
        return np.stack([sx(t_new), sy(t_new), sz(t_new)], axis=1)
    except Exception:
        return points_3d


def draw_trajectory(canvas, trajectory_xyz, calib, color_rgb, width=9):
    """Draw smoothed 3D trajectory projected onto camera image."""
    from camera_projection import project_points_to_camera

    if calib is None or len(trajectory_xyz) == 0:
        return

    if len(trajectory_xyz) >= 3:
        smooth = smooth_trajectory_3d(trajectory_xyz, num_points=60)
    else:
        smooth = trajectory_xyz

    pixels, valid = project_points_to_camera(smooth, calib)
    pixels_int = pixels.astype(np.int32)

    segments = []
    seg = []
    for p, v in zip(pixels_int, valid):
        if v:
            seg.append(tuple(p))
        else:
            if len(seg) > 1:
                segments.append(seg)
            seg = []
    if len(seg) > 1:
        segments.append(seg)
    if not segments:
        return

    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    r, g, b = color_rgb
    color = (r, g, b, 255)
    for seg in segments:
        if len(seg) < 2:
            continue
        cap_r = width / 2 - 0.5
        for pt in [seg[0], seg[-1]]:
            draw.ellipse([pt[0] - cap_r, pt[1] - cap_r, pt[0] + cap_r, pt[1] + cap_r], fill=color)
        draw.line(seg, fill=color, width=width, joint="curve")

    combined = Image.alpha_composite(pil_img, overlay)
    res = cv2.cvtColor(np.array(combined.convert("RGB")), cv2.COLOR_RGB2BGR)
    np.copyto(canvas, res)


# ─── 2D trajectory plot ─────────────────────────────────────────────────


def render_traj_plot(pred_xyz, gt_xyz=None, history_xyz=None, height=300, traj_bounds=None):
    """Render a bird's-eye trajectory plot matching the reference style.

    Portrait layout: vertical = Longitudinal (forward, X in ego), horizontal = Lateral (Y in ego).
    Lateral axis fixed to [-20, 20]. Longitudinal uses traj_bounds or auto range.
    Rendered at 2x resolution then downscaled with LANCZOS for crisp text.

    Args:
        pred_xyz: Predicted trajectory (T, 3) or (T, 2+).
        gt_xyz: Optional ground truth trajectory, same shape.
        height: Pixel height of the plot.
        traj_bounds: Optional (x_min, x_max, y_min, y_max) for consistent longitudinal axis.

    Returns:
        PIL RGBA Image of the plot (at requested height).
    """
    from scipy.interpolate import make_interp_spline

    # Supersample factor for crisp text
    SS = 2

    # Fixed axis ranges
    lat_min, lat_max = -20.0, 20.0
    lon_min, lon_max = -10.0, 80.0

    # Aspect ratio: width based on lateral/longitudinal ratio
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    width = max(int(height * lat_range / lon_range), int(height * 0.55))

    # Work at SS× resolution
    W, H = width * SS, height * SS

    # Fonts (scaled up)
    font_title = get_font(14 * SS, weight="bold")
    font_tick = get_font(11 * SS, weight="medium")
    font_label = get_font(12 * SS, weight="medium")

    # Margins for axis labels and ticks (scaled up)
    margin_l, margin_r, margin_t, margin_b = 42 * SS, 12 * SS, 34 * SS, 48 * SS
    pw = W - margin_l - margin_r
    ph = H - margin_t - margin_b

    plot = Image.new("RGBA", (W, H), (8, 12, 28, 220))
    draw = ImageDraw.Draw(plot)

    # Title (centered)
    title = "Trajectory Prediction"
    title_bbox = draw.textbbox((0, 0), title, font=font_title)
    title_w = title_bbox[2] - title_bbox[0]
    draw.text(((W - title_w) // 2, 8 * SS), title, font=font_title, fill=(255, 255, 255))

    def to_pixel(x_fwd, y_lat):
        # Horizontal axis shows -Y_ego: left in ego (Y>0) → left on screen (negative lateral)
        neg_y = -y_lat
        px = margin_l + (neg_y - lat_min) / (lat_max - lat_min) * pw
        # Vertical: larger X (forward) at top
        py = margin_t + (lon_max - x_fwd) / (lon_max - lon_min) * ph
        return (int(px), int(py))

    # Grid lines
    # Longitudinal ticks (every 20m)
    lon_step = 20
    lon_tick = int(lon_min / lon_step) * lon_step
    while lon_tick <= lon_max:
        if lon_tick >= lon_min:
            _, py = to_pixel(lon_tick, 0)
            draw.line([(margin_l, py), (margin_l + pw, py)], fill=(50, 55, 75, 180), width=SS)
            draw.text((20 * SS, py - 6 * SS), f"{lon_tick:.0f}", font=font_tick, fill=(200, 200, 200))
        lon_tick += lon_step

    # Lateral ticks (every 10m): labels show -Y_ego values
    lat_step = 10
    lat_tick = int(lat_min / lat_step) * lat_step
    while lat_tick <= lat_max:
        if lat_tick >= lat_min:
            px = margin_l + int((lat_tick - lat_min) / (lat_max - lat_min) * pw)
            draw.line([(px, margin_t), (px, margin_t + ph)], fill=(50, 55, 75, 180), width=SS)
            txt = f"{lat_tick:.0f}"
            txt_bbox = draw.textbbox((0, 0), txt, font=font_tick)
            txt_w = txt_bbox[2] - txt_bbox[0]
            draw.text((px - txt_w // 2, H - margin_b + 6 * SS), txt, font=font_tick, fill=(200, 200, 200))
        lat_tick += lat_step

    # Vertical axis label (rotated "Longitudinal (m)")
    draw_rotated_text(plot, (1 * SS, 0), "Longitudinal (m)", font=font_label,
                      center_in=(margin_t, ph))

    # Horizontal axis label
    lat_label = "Lateral (m)"
    lat_bbox = draw.textbbox((0, 0), lat_label, font=font_label)
    lat_w = lat_bbox[2] - lat_bbox[0]
    draw.text((margin_l + (pw - lat_w) // 2, H - 26 * SS), lat_label, font=font_label, fill=(200, 200, 200))

    def in_range(x_fwd, y_lat):
        return lon_min <= x_fwd <= lon_max and lat_min <= -y_lat <= lat_max

    def clip_and_draw(points, color, line_width):
        """Draw contiguous in-range segments as smooth polylines with interpolation."""
        w = line_width * SS  # Scale line width

        # Filter to in-range points
        in_pts = np.array([p for p in points if in_range(p[0], p[1])])
        if len(in_pts) < 2:
            return

        # Interpolate for smoothness
        if len(in_pts) >= 3:
            t = np.linspace(0, 1, len(in_pts))
            t_new = np.linspace(0, 1, max(len(in_pts) * 3, 100))
            try:
                k = min(3, len(in_pts) - 1)
                sx = make_interp_spline(t, in_pts[:, 0], k=k)
                sy = make_interp_spline(t, in_pts[:, 1], k=k)
                smooth = np.stack([sx(t_new), sy(t_new)], axis=1)
            except Exception:
                smooth = in_pts[:, :2]
        else:
            smooth = in_pts[:, :2]

        # Convert to pixel coordinates
        px_pts = [to_pixel(p[0], p[1]) for p in smooth]

        # Draw as filled circles along the path for seamless appearance
        r = w / 2
        for pt in px_pts:
            draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=color)
        # Also draw the line for connectivity
        if len(px_pts) > 1:
            draw.line(px_pts, fill=color, width=w)

    # Draw history trajectory (gray line)
    if history_xyz is not None and len(history_xyz) > 1:
        hist_arr = np.array(history_xyz)
        clip_and_draw(hist_arr, (160, 160, 160, 220), 4)

    # Draw predicted trajectory (cyan line)
    if len(pred_xyz) > 1:
        clip_and_draw(pred_xyz, (0, 210, 235, 255), 5)

    # Draw GT trajectory (gray)
    if gt_xyz is not None and len(gt_xyz) > 1:
        gt_arr = np.array(gt_xyz)
        clip_and_draw(gt_arr, (160, 160, 160, 200), 4)

    # Ego vehicle (yellow rectangle at origin)
    ego_px = to_pixel(0, 0)
    car_w, car_h = 8 * SS, 14 * SS
    draw.rectangle(
        [ego_px[0] - car_w // 2, ego_px[1] - car_h // 2,
         ego_px[0] + car_w // 2, ego_px[1] + car_h // 2],
        fill=(255, 220, 50, 230),
    )

    # Plot area border (axis frame)
    draw.rectangle(
        [margin_l, margin_t, margin_l + pw, margin_t + ph],
        outline=(50, 55, 75, 180), width=SS,
    )

    # Legend box (top right)
    lx = (width - 105) * SS
    ly = margin_t + 4 * SS
    lh = 56 * SS
    draw.rounded_rectangle([lx - 4 * SS, ly - 2 * SS, lx + 98 * SS, ly + lh], radius=4 * SS, fill=(18, 22, 40, 200))
    # Prediction
    draw.line([(lx, ly + 8 * SS), (lx + 16 * SS, ly + 8 * SS)], fill=(0, 210, 235), width=3 * SS)
    draw.text((lx + 20 * SS, ly), "Prediction", font=font_tick, fill=(200, 200, 200))
    # History
    draw.line([(lx, ly + 24 * SS), (lx + 16 * SS, ly + 24 * SS)], fill=(160, 160, 160), width=3 * SS)
    draw.text((lx + 20 * SS, ly + 16 * SS), "History", font=font_tick, fill=(200, 200, 200))
    # Ego
    draw.rectangle([lx, ly + 37 * SS, lx + 16 * SS, ly + 45 * SS], fill=(255, 220, 50))
    draw.text((lx + 20 * SS, ly + 34 * SS), "Ego Vehicle", font=font_tick, fill=(200, 200, 200))

    # Border
    draw.rounded_rectangle([0, 0, W - 1, H - 1], radius=15 * SS, outline=(70, 75, 95, 220), width=SS)

    # Downscale to target resolution with LANCZOS for crisp result
    plot = plot.resize((width, height), Image.LANCZOS)

    return plot


def overlay_traj_plot(pil_img, ui, frame, img_w, img_h, traj_bounds):
    """Overlay 2D bird's-eye trajectory plot on bottom right with blurred background."""
    plot_h = int(img_h * 0.55)
    traj_plot = render_traj_plot(
        frame["pred_xyz"], gt_xyz=frame.get("gt_xyz"),
        history_xyz=frame.get("history_xyz"), height=plot_h, traj_bounds=traj_bounds,
    )
    plot_x = img_w - traj_plot.width - 15
    plot_y = img_h - traj_plot.height - 15
    corner_radius = 15

    # Create rounded rectangle mask
    mask = Image.new("L", traj_plot.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle(
        [0, 0, traj_plot.width - 1, traj_plot.height - 1],
        radius=corner_radius, fill=255,
    )

    # Blur the camera region behind the plot with rounded mask
    region = pil_img.crop((plot_x, plot_y, plot_x + traj_plot.width, plot_y + traj_plot.height))
    blurred = region.filter(ImageFilter.GaussianBlur(radius=4))
    region.paste(blurred, mask=mask)
    pil_img.paste(region, (plot_x, plot_y))

    # Apply rounded mask while preserving existing alpha
    orig_alpha = traj_plot.split()[3]
    combined_alpha = ImageChops.multiply(orig_alpha, mask)
    traj_plot.putalpha(combined_alpha)
    ui.paste(traj_plot, (plot_x, plot_y), traj_plot)
