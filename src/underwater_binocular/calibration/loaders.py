"""Load canonical YAML and legacy OpenCV calibration representations."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .models import STEREO_CONVENTION, CameraIntrinsics, StereoCalibration, matrix_from_values


def sha256_file(path: Path) -> str:
    """Return a file SHA256 used for run provenance and generated headers."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"calibration root must be a mapping: {path}")
    return value


def load_calibration_profile(path: Path) -> StereoCalibration:
    """Load and validate the canonical computational calibration profile."""

    path = path.expanduser().resolve()
    values = _mapping(path)
    camera = values.get("camera")
    stereo = values.get("stereo")
    provenance = values.get("provenance")
    if not isinstance(camera, dict) or not isinstance(stereo, dict) or not isinstance(provenance, dict):
        raise ValueError("canonical profile requires camera, stereo, and provenance mappings")
    resolution = camera.get("resolution")
    if not isinstance(resolution, dict):
        raise ValueError("canonical profile camera.resolution must be a mapping")
    result = StereoCalibration(
        resolution=(int(resolution["width"]), int(resolution["height"])),
        left=CameraIntrinsics.from_mapping(values["left"]),
        right=CameraIntrinsics.from_mapping(values["right"]),
        rotation_left_to_right=matrix_from_values(stereo["rotation_matrix"]),
        translation_left_to_right_m=np.asarray(stereo["translation_m"], dtype=np.float64),
        convention=str(stereo.get("convention", "")),
        provenance={str(key): value for key, value in provenance.items()},
        source_path=path,
    )
    from .validation import validate_calibration

    validate_calibration(result)
    return result


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _section(text: str, name: str, next_name: str | None = None) -> str:
    end = rf"(?=^\s*{re.escape(next_name)}\s*:|\Z)" if next_name else r"(?=\Z)"
    match = re.search(rf"(?ms)^\s*{re.escape(name)}\s*:\s*(.*?){end}", text)
    if match is None:
        raise ValueError(f"missing calibration section {name!r}")
    return match.group(1)


def _scalar(text: str, name: str) -> float:
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*:\s*({_NUMBER})", text)
    if match is None:
        raise ValueError(f"missing calibration value {name!r}")
    return float(match.group(1))


def _vector(text: str, name: str) -> list[float]:
    match = re.search(rf"(?ms)^\s*{re.escape(name)}\s*:\s*\[([^\]]+)\]", text)
    if match is None:
        raise ValueError(f"missing calibration vector {name!r}")
    return [float(item) for item in re.findall(_NUMBER, match.group(1))]


def _rotation(text: str) -> np.ndarray:
    match = re.search(r"(?ms)^\s*R\s*:\s*(.*?)(?=^\s*T\s*:)", text)
    if match is None:
        raise ValueError("missing stereo R matrix")
    rows = [
        [float(item) for item in re.findall(_NUMBER, row)]
        for row in re.findall(r"\[([^\]]+)\]", match.group(1))
    ]
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("stereo R must be a 3x3 matrix")
    return np.asarray(rows, dtype=np.float64)


def load_legacy_source_calibration(
    intrinsics_path: Path,
    extrinsics_path: Path,
    *,
    resolution: tuple[int, int] = (1920, 1080),
) -> StereoCalibration:
    """Read the original YAML evidence, converting its millimetres once to metres."""

    intrinsics_text = intrinsics_path.read_text(encoding="utf-8")
    extrinsics_text = extrinsics_path.read_text(encoding="utf-8")
    cameras: dict[str, CameraIntrinsics] = {}
    for name in ("left", "right"):
        section = _section(intrinsics_text, name, "right" if name == "left" else None)
        cameras[name] = CameraIntrinsics(
            fx_px=_scalar(section, "fx"),
            fy_px=_scalar(section, "fy"),
            cx_px=_scalar(section, "cx"),
            cy_px=_scalar(section, "cy"),
            distortion_model="opencv_rad_tan",
            distortion=tuple(_vector(section, "distortion_coefficients")),
        )
    translation_mm = np.asarray(_vector(extrinsics_text, "T"), dtype=np.float64)
    if translation_mm.shape != (3,):
        raise ValueError("legacy stereo T must contain three millimetre values")
    provenance = {
        "source": f"{intrinsics_path.parent.as_posix()}/",
        "date": "unknown",
        "medium": "unknown",
        "housing": "unknown",
        "target_type": "unknown",
        "target_square_size": "unknown",
        "notes": "Loaded from original calibration evidence; translation converted from millimetres to metres.",
    }
    return StereoCalibration(
        resolution=resolution,
        left=cameras["left"],
        right=cameras["right"],
        rotation_left_to_right=_rotation(extrinsics_text),
        translation_left_to_right_m=translation_mm / 1000.0,
        convention=STEREO_CONVENTION,
        provenance=provenance,
        source_path=intrinsics_path,
    )


def load_opencv_calibration(path: Path) -> StereoCalibration:
    """Load a ZED/OpenCV ``FileStorage`` calibration at the external boundary."""

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - dependency declared by package
        raise RuntimeError("OpenCV is required to read an OpenCV calibration file") from error
    path = path.expanduser().resolve()
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(f"cannot open OpenCV calibration: {path}")
    try:
        size_node = storage.getNode("Size")
        if not size_node.isSeq() or size_node.size() != 2:
            raise ValueError("OpenCV calibration Size must contain width and height")
        resolution = (int(round(size_node.at(0).real())), int(round(size_node.at(1).real())))

        def matrix(name: str) -> np.ndarray:
            value = storage.getNode(name).mat()
            if value is None:
                raise ValueError(f"OpenCV calibration is missing {name}")
            result = np.asarray(value, dtype=np.float64)
            if not np.isfinite(result).all():
                raise ValueError(f"OpenCV calibration {name} contains non-finite values")
            return result

        k_left, k_right = matrix("K_LEFT"), matrix("K_RIGHT")
        d_left, d_right = matrix("D_LEFT").reshape(-1), matrix("D_RIGHT").reshape(-1)
        rotation_vector, translation_mm = matrix("R").reshape(-1), matrix("T").reshape(-1)
    finally:
        storage.release()
    if k_left.shape != (3, 3) or k_right.shape != (3, 3):
        raise ValueError("K_LEFT and K_RIGHT must be 3x3")
    if rotation_vector.size != 3 or translation_mm.size != 3:
        raise ValueError("OpenCV R and T must be 3-vectors")
    rotation = cv2.Rodrigues(rotation_vector.reshape(3, 1))[0]
    return StereoCalibration(
        resolution=resolution,
        left=CameraIntrinsics(float(k_left[0, 0]), float(k_left[1, 1]), float(k_left[0, 2]), float(k_left[1, 2]), "opencv_rad_tan", tuple(d_left[:5])),
        right=CameraIntrinsics(float(k_right[0, 0]), float(k_right[1, 1]), float(k_right[0, 2]), float(k_right[1, 2]), "opencv_rad_tan", tuple(d_right[:5])),
        rotation_left_to_right=rotation,
        translation_left_to_right_m=translation_mm / 1000.0,
        convention=STEREO_CONVENTION,
        provenance={"source": str(path), "date": "unknown", "medium": "unknown", "housing": "unknown", "target_type": "unknown", "target_square_size": "unknown", "notes": "OpenCV FileStorage boundary representation."},
        source_path=path,
    )
