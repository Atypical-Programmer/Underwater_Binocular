"""Shared COLMAP/ORB-SLAM3 pose conversions for Metashape exports."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ColmapPose:
    """COLMAP world-to-camera pose and derived camera-to-world pose."""

    image_id: int
    label: str
    qvec_wxyz: tuple[float, float, float, float]
    t_world_to_camera: np.ndarray
    rotation_world_to_camera: np.ndarray

    @property
    def rotation_camera_to_world(self) -> np.ndarray:
        return self.rotation_world_to_camera.T

    @property
    def center_world(self) -> np.ndarray:
        return -self.rotation_camera_to_world @ self.t_world_to_camera


def quaternion_wxyz_to_rotation(qvec_wxyz: Iterable[float]) -> np.ndarray:
    """Convert and normalize a COLMAP ``qw,qx,qy,qz`` quaternion."""

    q = np.asarray(tuple(float(value) for value in qvec_wxyz), dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) <= 1.0e-12:
        raise ValueError("pose quaternion must be finite and non-zero")
    qw, qx, qy, qz = q / np.linalg.norm(q)
    return np.asarray([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)], [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)], [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]], dtype=np.float64)


def colmap_to_metashape_pose(pose: ColmapPose) -> dict[str, object]:
    """Convert a pose and label it as local, non-georeferenced coordinates."""

    return {"label": pose.label, "center_world": pose.center_world.tolist(), "rotation_camera_to_world": pose.rotation_camera_to_world.tolist(), "coordinate_frame": "local coordinate frame", "georeferenced": False, "scale_source": "COLMAP arbitrary reconstruction scale unless externally constrained"}


def rotation_to_ypr(rotation_camera_to_world: np.ndarray) -> tuple[float, float, float]:
    """Extract Metashape-compatible yaw/pitch/roll in degrees."""

    rotation = np.asarray(rotation_camera_to_world, dtype=np.float64).reshape(3, 3)
    sine_pitch = float(np.clip(rotation[2, 1], -1.0, 1.0))
    pitch = math.asin(sine_pitch)
    if abs(math.cos(pitch)) > 1.0e-10:
        yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
        roll = math.atan2(-rotation[2, 0], rotation[2, 2])
    else:
        yaw, roll = math.atan2(rotation[1, 0], rotation[0, 0]), 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def write_reference_csv(path: Path, poses: Iterable[ColmapPose]) -> None:
    """Write a local-frame Metashape reference CSV with provenance headers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write("# coordinate_frame=local coordinate frame\n# georeferenced=false\n# scale_source=COLMAP arbitrary reconstruction scale unless externally constrained\n")
        writer = csv.writer(handle)
        writer.writerow(["label", "x", "y", "z", "yaw", "pitch", "roll"])
        for pose in poses:
            ypr = rotation_to_ypr(pose.rotation_camera_to_world)
            writer.writerow([pose.label, *[f"{value:.12g}" for value in pose.center_world], *[f"{value:.12g}" for value in ypr]])
