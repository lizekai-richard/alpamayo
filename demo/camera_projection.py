#!/usr/bin/env python3
"""Camera projection utilities for overlaying trajectories on images.

The physical_ai_av dataset uses a polynomial camera model (for fisheye/wide-angle lenses):
- Forward projection: 3D point → 2D pixel using fw_poly coefficients
- Backward projection: 2D pixel → 3D ray using bw_poly coefficients

Coordinate frames:
- Ego frame: X=forward, Y=left, Z=up (at t0)
- Camera frame: X=right, Y=down, Z=forward (OpenCV convention)
"""

import numpy as np
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from typing import Tuple, Optional
import cv2


@dataclass
class CameraCalibration:
    """Camera calibration data."""
    name: str
    width: int
    height: int
    cx: float  # Principal point x
    cy: float  # Principal point y
    fw_poly: np.ndarray  # Forward polynomial coefficients [0, 1, 2, 3, 4]
    bw_poly: np.ndarray  # Backward polynomial coefficients
    # Extrinsics (camera pose in vehicle frame)
    rotation: Rotation  # Vehicle → Camera rotation
    translation: np.ndarray  # Camera position in vehicle frame (3,)


def load_calibration(clip_id: str, camera_name: str = "camera_front_wide_120fov") -> CameraCalibration:
    """Load camera calibration from physical_ai_av dataset.

    Args:
        clip_id: Clip ID
        camera_name: Camera name (e.g., 'camera_front_wide_120fov')

    Returns:
        CameraCalibration object
    """
    import physical_ai_av

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    # Get intrinsics (CameraIntrinsics with .camera_models dict)
    intrinsics = avdi.get_clip_feature(
        clip_id, avdi.features.CALIBRATION.CAMERA_INTRINSICS, maybe_stream=True
    )
    cam_model = intrinsics.camera_models[camera_name]

    # th2r is forward polynomial (theta → radius in pixels)
    # Extract coefficients from the polynomial object
    fw_poly = np.array(cam_model.th2r.coef)
    bw_poly = np.array(cam_model.r2th.coef)

    # Get extrinsics (SensorExtrinsics with .sensor_poses dict)
    extrinsics = avdi.get_clip_feature(
        clip_id, avdi.features.CALIBRATION.SENSOR_EXTRINSICS, maybe_stream=True
    )
    pose = extrinsics.sensor_poses[camera_name]  # RigidTransform

    return CameraCalibration(
        name=camera_name,
        width=cam_model.width,
        height=cam_model.height,
        cx=cam_model.principal_point[0],
        cy=cam_model.principal_point[1],
        fw_poly=fw_poly,
        bw_poly=bw_poly,
        rotation=pose.rotation,
        translation=pose.translation,
    )


def project_points_to_camera(
    points_ego: np.ndarray,
    calib: CameraCalibration,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points from ego frame to camera pixel coordinates.

    Args:
        points_ego: Points in ego frame (N, 3) - X=forward, Y=left, Z=up
        calib: Camera calibration

    Returns:
        pixels: Pixel coordinates (N, 2) - (u, v)
        valid: Boolean mask (N,) - True if point is in front of camera and in image
    """
    points_ego = np.atleast_2d(points_ego)
    N = points_ego.shape[0]

    # Transform from ego frame to camera frame
    # The extrinsics give camera pose in vehicle frame (camera-to-vehicle transform)
    # So we need the inverse: vehicle-to-camera transform
    # points_cam = R_cam_from_ego @ (points_ego - t_cam_in_ego)
    # where R_cam_from_ego = R_ego_from_cam.inv()

    points_cam = calib.rotation.inv().apply(points_ego - calib.translation)

    # Check if points are in front of camera (Z > 0 in camera frame)
    in_front = points_cam[:, 2] > 0.1  # Small threshold to avoid division issues

    # Apply polynomial projection model
    # r = sqrt(x^2 + y^2) / z (angle from optical axis)
    # r_dist = fw_poly[0] + fw_poly[1]*r + fw_poly[2]*r^2 + ...
    # u = cx + r_dist * x / sqrt(x^2 + y^2)
    # v = cy + r_dist * y / sqrt(x^2 + y^2)

    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

    # Avoid division by zero
    xy_norm = np.sqrt(x**2 + y**2)
    xy_norm = np.maximum(xy_norm, 1e-6)

    # Angle from optical axis (in radians, approximately)
    theta = np.arctan2(xy_norm, z)

    # Apply forward polynomial
    r_dist = np.zeros(N)
    for i, coef in enumerate(calib.fw_poly):
        r_dist += coef * (theta ** i)

    # Project to pixel coordinates
    u = calib.cx + r_dist * (x / xy_norm)
    v = calib.cy + r_dist * (y / xy_norm)

    pixels = np.stack([u, v], axis=1)

    # Check if in image bounds
    in_image = (
        (u >= 0) & (u < calib.width) &
        (v >= 0) & (v < calib.height)
    )

    valid = in_front & in_image

    return pixels, valid


def draw_trajectory_on_image(
    image: np.ndarray,
    trajectory_ego: np.ndarray,
    calib: CameraCalibration,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
    draw_points: bool = True,
    point_radius: int = 5,
) -> np.ndarray:
    """Draw trajectory overlay on camera image.

    Args:
        image: Camera image (H, W, 3) BGR
        trajectory_ego: Trajectory points in ego frame (T, 3)
        calib: Camera calibration
        color: BGR color for trajectory
        thickness: Line thickness
        draw_points: Whether to draw points at each trajectory step
        point_radius: Radius of trajectory points

    Returns:
        Image with trajectory overlay
    """
    image = image.copy()

    # Project trajectory to pixels
    pixels, valid = project_points_to_camera(trajectory_ego, calib)

    # Draw lines between consecutive valid points
    pixels_int = pixels.astype(np.int32)

    for i in range(len(trajectory_ego) - 1):
        if valid[i] and valid[i + 1]:
            pt1 = tuple(pixels_int[i])
            pt2 = tuple(pixels_int[i + 1])
            cv2.line(image, pt1, pt2, color, thickness, cv2.LINE_AA)

    # Draw points
    if draw_points:
        for i, (pix, v) in enumerate(zip(pixels_int, valid)):
            if v:
                # Fade color based on time (darker = earlier)
                alpha = i / max(len(trajectory_ego) - 1, 1)
                pt_color = tuple(int(c * (0.5 + 0.5 * alpha)) for c in color)
                cv2.circle(image, tuple(pix), point_radius, pt_color, -1, cv2.LINE_AA)

    return image


def draw_ego_marker(
    image: np.ndarray,
    calib: CameraCalibration,
    color: Tuple[int, int, int] = (0, 0, 255),
    size: int = 20,
) -> np.ndarray:
    """Draw ego vehicle marker at origin.

    Args:
        image: Camera image
        calib: Camera calibration
        color: BGR color
        size: Marker size

    Returns:
        Image with ego marker
    """
    # Ego is at origin in ego frame
    ego_pos = np.array([[0.0, 0.0, 0.0]])
    pixels, valid = project_points_to_camera(ego_pos, calib)

    if valid[0]:
        pt = tuple(pixels[0].astype(int))
        # Draw a crosshair
        cv2.drawMarker(image, pt, color, cv2.MARKER_CROSS, size, 2, cv2.LINE_AA)

    return image


if __name__ == "__main__":
    import json
    import os

    # Test projection
    clip_file = os.path.expanduser("~/data/physicalai_av/clip_ids.json")
    with open(clip_file) as f:
        clip_id = json.load(f)[0]

    print(f"Clip: {clip_id}")

    # Load calibration for front wide camera
    calib = load_calibration(clip_id, "camera_front_wide_120fov")
    print(f"\nCamera: {calib.name}")
    print(f"Resolution: {calib.width}x{calib.height}")
    print(f"Principal point: ({calib.cx:.1f}, {calib.cy:.1f})")
    print(f"Translation: {calib.translation}")
    print(f"Rotation (euler XYZ): {calib.rotation.as_euler('xyz', degrees=True)}")
    print(f"Rotation matrix:\n{calib.rotation.as_matrix()}")

    # Debug: transform a point and see what happens
    test_pt = np.array([[10, 0, 0]])  # 10m forward
    pt_cam = calib.rotation.apply(test_pt - calib.translation)
    print(f"\nDebug: [10,0,0] in ego -> {pt_cam[0]} in camera frame")

    # Try inverse rotation
    pt_cam_inv = calib.rotation.inv().apply(test_pt - calib.translation)
    print(f"Debug: [10,0,0] with inv rotation -> {pt_cam_inv[0]} in camera frame")

    # Test projection of some points
    test_points = np.array([
        [10, 0, 0],    # 10m forward
        [20, 0, 0],    # 20m forward
        [30, 0, 0],    # 30m forward
        [10, 5, 0],    # 10m forward, 5m left
        [10, -5, 0],   # 10m forward, 5m right
    ])

    pixels, valid = project_points_to_camera(test_points, calib)

    print("\nTest projections:")
    for pt, pix, v in zip(test_points, pixels, valid):
        status = "valid" if v else "invalid"
        print(f"  {pt} -> ({pix[0]:.1f}, {pix[1]:.1f}) [{status}]")
