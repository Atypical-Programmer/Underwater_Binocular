"""COLMAP camera-parameter generation facade."""

from __future__ import annotations

from pathlib import Path

from .conversion import write_colmap_camera_config
from .loaders import load_calibration_profile


def generate_colmap_camera_config(profile_path: Path, output_path: Path, *, camera_model: str = "OPENCV") -> Path:
    """Generate COLMAP camera parameters from the canonical profile."""

    profile_path = profile_path.expanduser().resolve()
    return write_colmap_camera_config(load_calibration_profile(profile_path), profile_path, output_path, camera_model=camera_model)
