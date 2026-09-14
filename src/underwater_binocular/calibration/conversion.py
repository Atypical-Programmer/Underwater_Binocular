"""Generate ZED, ORB-SLAM3, and COLMAP calibration files from one profile."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from .loaders import load_calibration_profile, sha256_file
from .models import StereoCalibration
from .validation import validate_calibration


def _source_label(profile_path: Path) -> str:
    parts = profile_path.as_posix().split("/")
    if "calibration" in parts:
        return "/".join(parts[parts.index("calibration") :])
    return profile_path.name


def _header(profile_path: Path, source_hash: str) -> list[str]:
    return [
        "# GENERATED FILE - DO NOT EDIT",
        f"# canonical source path: {_source_label(profile_path)}",
        f"# source hash: {source_hash}",
        f"# generator version: {__version__}",
    ]


def _fmt(value: float) -> str:
    return f"{float(value):.16g}"


def write_zed_opencv_calibration(
    profile: StereoCalibration,
    profile_path: Path,
    output_path: Path,
    *,
    source_hash: str | None = None,
) -> Path:
    """Write the OpenCV FileStorage representation expected by ZED.

    The source profile stores metres and a rotation matrix. This boundary
    converts translation to millimetres and rotation to a Rodrigues vector,
    matching the parent ZED calibration file semantics.
    """

    import cv2

    validate_calibration(profile)
    source_hash = source_hash or sha256_file(profile_path)
    rotation_vector = cv2.Rodrigues(np.asarray(profile.rotation_left_to_right, dtype=np.float64))[0].reshape(3)
    translation_mm = np.asarray(profile.translation_left_to_right_m, dtype=np.float64).reshape(3) * 1000.0

    def data(matrix: np.ndarray) -> str:
        return ", ".join(_fmt(value) for value in np.asarray(matrix).reshape(-1))

    d_left = np.pad(np.asarray(profile.left.distortion, dtype=np.float64)[:5], (0, 9))
    d_right = np.pad(np.asarray(profile.right.distortion, dtype=np.float64)[:5], (0, 9))
    width, height = profile.resolution
    lines = ["%YAML:1.0", "---", *_header(profile_path, source_hash), f"Size: [ {width}, {height} ]"]
    for name, matrix in (("K_LEFT", profile.left.matrix), ("K_RIGHT", profile.right.matrix), ("D_LEFT", d_left.reshape(1, -1)), ("D_RIGHT", d_right.reshape(1, -1)), ("R", rotation_vector.reshape(3, 1)), ("T", translation_mm.reshape(3, 1))):
        rows, cols = matrix.shape
        lines.extend([f"{name}: !!opencv-matrix", f"   rows: {rows}", f"   cols: {cols}", "   dt: d", f"   data: [ {data(matrix)} ]"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_orbslam3_config(
    profile: StereoCalibration,
    profile_path: Path,
    output_path: Path,
    *,
    image_scale: float = 0.5,
    fps: float = 30.0,
    source_hash: str | None = None,
) -> Path:
    """Write the ORB-SLAM3 stereo settings with the explicit inverse transform."""

    if not 0.0 < image_scale <= 1.0:
        raise ValueError("ORB-SLAM3 image_scale must be in (0, 1]")
    source_hash = source_hash or sha256_file(profile_path)
    left, right = profile.left.scaled(image_scale), profile.right.scaled(image_scale)
    rotation, translation = profile.inverse_transform()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    width, height = profile.resolution
    lines = [
        "%YAML:1.0",
        *[_line for _line in _header(profile_path, source_hash)],
        "# T_c1_c2 is the inverse of package left-to-right OpenCV extrinsics.",
        "File.version: \"1.0\"",
        "Camera.type: \"PinHole\"",
        "",
        f"Camera1.fx: {_fmt(left.fx_px)}",
        f"Camera1.fy: {_fmt(left.fy_px)}",
        f"Camera1.cx: {_fmt(left.cx_px)}",
        f"Camera1.cy: {_fmt(left.cy_px)}",
        f"Camera1.k1: {_fmt(left.distortion[0])}",
        f"Camera1.k2: {_fmt(left.distortion[1])}",
        f"Camera1.p1: {_fmt(left.distortion[2])}",
        f"Camera1.p2: {_fmt(left.distortion[3])}",
        f"Camera1.k3: {_fmt(left.distortion[4])}",
        "",
        f"Camera2.fx: {_fmt(right.fx_px)}",
        f"Camera2.fy: {_fmt(right.fy_px)}",
        f"Camera2.cx: {_fmt(right.cx_px)}",
        f"Camera2.cy: {_fmt(right.cy_px)}",
        f"Camera2.k1: {_fmt(right.distortion[0])}",
        f"Camera2.k2: {_fmt(right.distortion[1])}",
        f"Camera2.p1: {_fmt(right.distortion[2])}",
        f"Camera2.p2: {_fmt(right.distortion[3])}",
        f"Camera2.k3: {_fmt(right.distortion[4])}",
        "",
        f"Camera.width: {round(width * image_scale)}",
        f"Camera.height: {round(height * image_scale)}",
        f"Camera.fps: {_fmt(fps)}",
        "Camera.RGB: 0",
        "",
        "Stereo.ThDepth: 60.0",
        "Stereo.T_c1_c2: !!opencv-matrix",
        "  rows: 4",
        "  cols: 4",
        "  dt: f",
        "  data: [" + ",".join(_fmt(value) for value in transform.reshape(-1)) + "]",
        "",
        "ORBextractor.nFeatures: 1500",
        "ORBextractor.scaleFactor: 1.2",
        "ORBextractor.nLevels: 8",
        "ORBextractor.iniThFAST: 20",
        "ORBextractor.minThFAST: 7",
        "",
        "Viewer.KeyFrameSize: 0.05",
        "Viewer.KeyFrameLineWidth: 1.0",
        "Viewer.GraphLineWidth: 0.9",
        "Viewer.PointSize: 2.0",
        "Viewer.CameraSize: 0.08",
        "Viewer.CameraLineWidth: 3.0",
        "Viewer.ViewpointX: 0.0",
        "Viewer.ViewpointY: -0.7",
        "Viewer.ViewpointZ: -1.8",
        "Viewer.ViewpointF: 500.0",
        "Viewer.imageViewScale: 1.0",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_colmap_camera_config(
    profile: StereoCalibration,
    profile_path: Path,
    output_path: Path,
    *,
    camera_model: str = "OPENCV",
    source_hash: str | None = None,
) -> Path:
    """Write explicit COLMAP camera parameters for frozen/refined modes."""

    camera_model = camera_model.upper()
    if camera_model not in {"PINHOLE", "OPENCV", "FULL_OPENCV"}:
        raise ValueError(f"unsupported COLMAP camera model: {camera_model}")
    source_hash = source_hash or sha256_file(profile_path)

    def params(camera: Any) -> list[float]:
        base = [camera.fx_px, camera.fy_px, camera.cx_px, camera.cy_px]
        if camera_model == "PINHOLE":
            return base
        if camera_model == "OPENCV":
            return [*base, *camera.distortion[:4]]
        return [*base, *camera.distortion[:8]]

    width, height = profile.resolution
    value = {
        "schema_version": 1,
        "generated_file": True,
        "do_not_edit": True,
        "canonical_source_path": _source_label(profile_path),
        "source_hash": source_hash,
        "generator_version": __version__,
        "camera_model": camera_model,
        "freeze_calibration_default": True,
        "left": {"width": width, "height": height, "params": params(profile.left)},
        "right": {"width": width, "height": height, "params": params(profile.right)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return output_path


def generate_derived_calibrations(
    profile_path: Path,
    output_dir: Path,
    *,
    orb_scale: float = 0.5,
) -> tuple[Path, Path, Path]:
    """Generate all derived files from the canonical profile in one operation."""

    profile_path = profile_path.expanduser().resolve()
    profile = load_calibration_profile(profile_path)
    output_dir = output_dir.expanduser().resolve()
    source_hash = sha256_file(profile_path)
    return (
        write_zed_opencv_calibration(profile, profile_path, output_dir / "zed_custom_opencv.yml", source_hash=source_hash),
        write_orbslam3_config(profile, profile_path, output_dir / "orbslam3_stereo.yaml", image_scale=orb_scale, source_hash=source_hash),
        write_colmap_camera_config(profile, profile_path, output_dir / "colmap_camera.json", source_hash=source_hash),
    )
