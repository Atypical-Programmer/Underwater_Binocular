"""ORB-SLAM3 calibration generation facade."""

from __future__ import annotations

from pathlib import Path

from .conversion import write_orbslam3_config
from .loaders import load_calibration_profile


def generate_orbslam3_config(profile_path: Path, output_path: Path, *, image_scale: float = 0.5) -> Path:
    """Generate one ORB-SLAM3 settings file from the canonical profile."""

    profile_path = profile_path.expanduser().resolve()
    return write_orbslam3_config(load_calibration_profile(profile_path), profile_path, output_path, image_scale=image_scale)
