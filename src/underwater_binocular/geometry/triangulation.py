"""Rectified correspondence and ray triangulation primitives."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StereoCorrespondence:
    """One corresponding rectified pixel pair in pixel coordinates."""

    left_uv: tuple[float, float]
    right_uv: tuple[float, float]

    @property
    def disparity_px(self) -> float:
        return float(self.left_uv[0] - self.right_uv[0])


def triangulate_rectified(
    correspondence: StereoCorrespondence,
    *,
    focal_length_px: float,
    baseline_m: float,
    principal_point_px: Sequence[float],
) -> np.ndarray:
    """Triangulate a rectified pair as ``[X_m, Y_m, Z_m]``.

    ``Z_m = f_rectified_px * baseline_m / disparity_px`` is the optical-axis
    depth. A negative or zero disparity is rejected rather than silently
    yielding a physically ambiguous result.
    """

    if focal_length_px <= 0.0 or baseline_m <= 0.0:
        raise ValueError("focal_length_px and baseline_m must be positive")
    cx, cy = (float(value) for value in principal_point_px)
    disparity = correspondence.disparity_px
    if not np.isfinite(disparity) or disparity <= 0.0:
        raise ValueError("rectified disparity must be positive")
    depth_z_m = focal_length_px * baseline_m / disparity
    x_m = (float(correspondence.left_uv[0]) - cx) * depth_z_m / focal_length_px
    y_m = (float(correspondence.left_uv[1]) - cy) * depth_z_m / focal_length_px
    return np.asarray([x_m, y_m, depth_z_m], dtype=np.float64)


def q_reprojection(point_xyz_m: Sequence[float], q_matrix: np.ndarray) -> tuple[float, float, float]:
    """Apply a 4x4 Q matrix and return ``(u, v, depth_z_m)``."""

    point = np.asarray([*point_xyz_m, 1.0], dtype=np.float64)
    q = np.asarray(q_matrix, dtype=np.float64).reshape(4, 4)
    homogeneous = q @ point
    if abs(float(homogeneous[3])) <= 1.0e-15:
        raise ValueError("Q reprojection has zero homogeneous scale")
    result = homogeneous[:3] / homogeneous[3]
    return float(result[0]), float(result[1]), float(result[2])


def triangulate_two_rays(
    origin_left_m: Sequence[float],
    direction_left: Sequence[float],
    origin_right_m: Sequence[float],
    direction_right: Sequence[float],
) -> dict[str, float | np.ndarray]:
    """Find closest points on two rays in one common metre frame."""

    left_origin = np.asarray(origin_left_m, dtype=np.float64).reshape(3)
    right_origin = np.asarray(origin_right_m, dtype=np.float64).reshape(3)
    left_direction = np.asarray(direction_left, dtype=np.float64).reshape(3)
    right_direction = np.asarray(direction_right, dtype=np.float64).reshape(3)
    left_direction /= np.linalg.norm(left_direction)
    right_direction /= np.linalg.norm(right_direction)
    matrix = np.column_stack((left_direction, -right_direction))
    solution, _, rank, singular = np.linalg.lstsq(matrix, right_origin - left_origin, rcond=None)
    t_left, t_right = float(solution[0]), float(solution[1])
    point_left = left_origin + t_left * left_direction
    point_right = right_origin + t_right * right_direction
    return {
        "point_left_m": point_left,
        "point_right_m": point_right,
        "midpoint_m": 0.5 * (point_left + point_right),
        "ray_gap_m": float(np.linalg.norm(point_left - point_right)),
        "t_left_m": t_left,
        "t_right_m": t_right,
        "rank": int(rank),
        "condition_number": float(np.inf if len(singular) < 2 or singular[-1] <= 1.0e-15 else singular[0] / singular[-1]),
    }
