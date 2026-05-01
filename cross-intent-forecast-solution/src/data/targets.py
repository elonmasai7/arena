from __future__ import annotations

import math

import numpy as np


def normalize_points(points: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    norm = points.astype(np.float32).copy()
    norm[:, 0] /= max(image_width, 1)
    norm[:, 1] /= max(image_height, 1)
    return norm


def denormalize_points(points: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    out = points.astype(np.float32).copy()
    out[:, 0] *= image_width
    out[:, 1] *= image_height
    return out


def generate_future_waypoints(
    centers: list[tuple[float, float]],
    future_steps: int,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    points = np.zeros((future_steps, 2), dtype=np.float32)
    usable = min(len(centers), future_steps)
    if usable > 0:
        points[:usable] = np.asarray(centers[:usable], dtype=np.float32)
        if usable < future_steps:
            points[usable:] = points[usable - 1]
    return normalize_points(points, image_width=image_width, image_height=image_height)


def gaussian_heatmap(
    center_x: float,
    center_y: float,
    width: int,
    height: int,
    sigma: float = 1.5,
) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    exponent = ((xs - center_x) ** 2 + (ys - center_y) ** 2) / (2.0 * sigma * sigma)
    heatmap = np.exp(-exponent)
    total = float(heatmap.sum())
    return (heatmap / total) if total > 0 else heatmap


def generate_future_heatmaps(
    waypoints_norm: np.ndarray,
    heatmap_size: tuple[int, int],
) -> np.ndarray:
    height, width = heatmap_size
    heatmaps = np.zeros((waypoints_norm.shape[0], height, width), dtype=np.float32)
    for index, point in enumerate(waypoints_norm):
        x = float(point[0]) * (width - 1)
        y = float(point[1]) * (height - 1)
        heatmaps[index] = gaussian_heatmap(x, y, width=width, height=height)
    return heatmaps


def compute_heading_velocity(
    history_points: np.ndarray,
    dt: float,
) -> tuple[float, float, float]:
    if history_points.shape[0] < 2:
        return 0.0, 0.0, 0.0
    delta = history_points[-1] - history_points[-2]
    vx = float(delta[0] / max(dt, 1e-6))
    vy = float(delta[1] / max(dt, 1e-6))
    heading = math.atan2(vy, vx) if (vx or vy) else 0.0
    return vx, vy, heading
