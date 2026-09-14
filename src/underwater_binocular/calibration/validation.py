"""Calibration invariants and machine-readable validation reports."""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import StereoCalibration


class CalibrationValidationError(ValueError):
    """Raised when a calibration cannot safely enter a computation."""


def validate_calibration(
    calibration: StereoCalibration,
    *,
    rotation_tolerance: float = 1.0e-6,
    translation_tolerance_m: float = 1.0e-12,
) -> dict[str, Any]:
    """Validate shape, units, rotation, intrinsics, and transform invariants.

    The tight rotation tolerance is appropriate for a calibration matrix; the
    translation tolerance only rejects a numerically zero rig baseline. No
    algorithmic parameter is changed by this validation.
    """

    width, height = calibration.resolution
    if width <= 0 or height <= 0:
        raise CalibrationValidationError("calibration resolution must be positive")
    for side, camera in (("left", calibration.left), ("right", calibration.right)):
        matrix = camera.matrix
        if camera.cx_px < 0.0 or camera.cx_px > width or camera.cy_px < 0.0 or camera.cy_px > height:
            raise CalibrationValidationError(f"{side} principal point is outside the image")
        if not np.isfinite(matrix).all():
            raise CalibrationValidationError(f"{side} intrinsic matrix is not finite")
    rotation = np.asarray(calibration.rotation_left_to_right, dtype=np.float64)
    translation = np.asarray(calibration.translation_left_to_right_m, dtype=np.float64).reshape(3)
    orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > rotation_tolerance or abs(determinant - 1.0) > rotation_tolerance:
        raise CalibrationValidationError(
            f"invalid stereo rotation: orthogonality_error={orthogonality_error}, determinant={determinant}"
        )
    baseline = float(np.linalg.norm(translation))
    if not np.isfinite(translation).all() or baseline <= translation_tolerance_m:
        raise CalibrationValidationError("stereo translation must be a finite non-zero vector in metres")
    return {
        "resolution": {"width": width, "height": height},
        "baseline_norm_m": baseline,
        "tx_abs_m": abs(float(translation[0])),
        "rotation_determinant": determinant,
        "rotation_orthogonality_error_max_abs": orthogonality_error,
        "units": "metre for translation; pixel for intrinsics",
        "convention": calibration.convention,
    }
