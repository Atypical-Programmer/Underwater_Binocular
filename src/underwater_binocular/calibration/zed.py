"""ZED calibration metadata extraction and runtime comparison helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import StereoCalibration


def _float_list(value: Any) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=np.float64).reshape(-1)]


def _transform_matrix(value: Any) -> np.ndarray:
    """Convert a ZED Transform or a plain 4x4 value to a numeric matrix."""

    matrix = np.asarray(getattr(value, "m", value), dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"ZED stereo transform must be 4x4, got {matrix.shape}")
    return matrix


def camera_parameters_metadata(parameters: Any) -> dict[str, Any]:
    """Convert a ZED SDK camera-parameter object to JSON-safe metadata."""

    def camera(camera: Any) -> dict[str, Any]:
        return {
            "fx_px": float(camera.fx),
            "fy_px": float(camera.fy),
            "cx_px": float(camera.cx),
            "cy_px": float(camera.cy),
            "distortion": _float_list(camera.disto),
            "lens_distortion_model": str(camera.lens_distortion_model),
        }

    transform = _transform_matrix(parameters.stereo_transform)
    return {
        "left": camera(parameters.left_cam),
        "right": camera(parameters.right_cam),
        "stereo_transform_m": transform.tolist(),
    }


def runtime_calibration_metadata(camera_information: Any) -> dict[str, Any]:
    """Extract raw and rectified SDK calibration from ``CameraInformation``."""

    configuration = camera_information.camera_configuration
    resolution = configuration.resolution
    return {
        "resolution": {"width": int(resolution.width), "height": int(resolution.height)},
        "raw": camera_parameters_metadata(configuration.calibration_parameters_raw),
        "rectified": camera_parameters_metadata(configuration.calibration_parameters),
    }


def _matrix_from_camera(camera: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [[camera["fx_px"], 0.0, camera["cx_px"]], [0.0, camera["fy_px"], camera["cy_px"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _expected_sdk_stereo_transform(expected: StereoCalibration) -> np.ndarray:
    """Return the ZED ``Transform.m`` representation of the custom profile.

    The OpenCV calibration file uses the package's camera-frame convention.
    ZED exposes the loaded custom transform in its right-handed Y-up frame,
    which applies the 180-degree X-axis basis change to the rotation and
    reports the negated translation in metres.
    """

    rotation = np.asarray(expected.rotation_left_to_right, dtype=np.float64)
    translation = np.asarray(expected.translation_left_to_right_m, dtype=np.float64).reshape(3)
    basis_change = np.diag([1.0, -1.0, -1.0])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = basis_change @ rotation @ basis_change
    transform[:3, 3] = -translation
    return transform


def compare_runtime_calibration(
    runtime: dict[str, Any],
    expected: StereoCalibration,
    *,
    tolerance_px: float = 1.0e-3,
    tolerance_m: float = 1.0e-6,
) -> dict[str, Any]:
    """Compare actual SDK values to the custom profile and raise on mismatch."""

    resolution = runtime["resolution"]
    expected_resolution = {"width": expected.resolution[0], "height": expected.resolution[1]}
    if resolution != expected_resolution:
        raise RuntimeError(f"calibration mismatch: runtime resolution {resolution} != {expected_resolution}")
    raw = runtime["raw"]
    errors: dict[str, float] = {}
    for side, camera in (("left", expected.left), ("right", expected.right)):
        if side not in raw or not isinstance(raw[side], dict):
            raise RuntimeError(f"calibration mismatch: runtime raw calibration is missing {side}")
        actual_k = _matrix_from_camera(raw[side])
        errors[f"{side}_K_max_abs"] = float(np.max(np.abs(actual_k - camera.matrix)))
        actual_d = np.asarray(raw[side].get("distortion", []), dtype=np.float64).reshape(-1)
        expected_d = np.asarray(camera.distortion[:5], dtype=np.float64)
        if actual_d.size < expected_d.size:
            errors[f"{side}_D_first5_max_abs"] = float("inf")
        else:
            errors[f"{side}_D_first5_max_abs"] = float(np.max(np.abs(actual_d[:5] - expected_d)))
    actual_transform = np.asarray(raw["stereo_transform_m"], dtype=np.float64).reshape(4, 4)
    expected_sdk_transform = _expected_sdk_stereo_transform(expected)
    errors["stereo_transform_max_abs"] = float(np.max(np.abs(actual_transform - expected_sdk_transform)))
    failing = [name for name, value in errors.items() if ("_K_" in name or "_D_" in name) and value > tolerance_px]
    if errors["stereo_transform_max_abs"] > tolerance_m:
        failing.append("stereo_transform_max_abs")
    if failing:
        raise RuntimeError(f"calibration mismatch after SDK open: {failing}; errors={errors}")
    return {"status": "PASS", "errors": errors, "raw": runtime["raw"], "rectified": runtime["rectified"]}
