"""Pure stereo-frame transforms and invariant checks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def validate_rotation(rotation: np.ndarray, *, tolerance: float = 1.0e-6) -> dict[str, float]:
    """Validate a finite proper 3x3 rotation matrix."""

    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    error = float(np.max(np.abs(value.T @ value - np.eye(3))))
    determinant = float(np.linalg.det(value))
    if error > tolerance or abs(determinant - 1.0) > tolerance:
        raise ValueError(f"invalid rotation: orthogonality_error={error}, determinant={determinant}")
    return {"orthogonality_error_max_abs": error, "determinant": determinant}


def baseline_m(translation_left_to_right_m: Sequence[float] | np.ndarray) -> float:
    """Return Euclidean stereo baseline in metres, distinct from ``abs(Tx)``."""

    translation = np.asarray(translation_left_to_right_m, dtype=np.float64).reshape(-1)
    if translation.size != 3 or not np.isfinite(translation).all():
        raise ValueError("translation_left_to_right_m must be a finite 3-vector")
    value = float(np.linalg.norm(translation))
    if value <= 0.0:
        raise ValueError("stereo baseline must be positive")
    return value


def transform_points(
    points_left: np.ndarray,
    rotation_left_to_right: np.ndarray,
    translation_left_to_right_m: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Apply ``X_right = R_left_to_right @ X_left + t_left_to_right``."""

    points = np.asarray(points_left, dtype=np.float64)
    rotation = np.asarray(rotation_left_to_right, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation_left_to_right_m, dtype=np.float64).reshape(3)
    validate_rotation(rotation)
    if points.shape[-1] != 3:
        raise ValueError("points must have a final dimension of 3")
    return np.asarray(points @ rotation.T + translation, dtype=np.float64)


def inverse_transform(
    rotation_left_to_right: np.ndarray,
    translation_left_to_right_m: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the right-to-left inverse of the package transform."""

    rotation = np.asarray(rotation_left_to_right, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation_left_to_right_m, dtype=np.float64).reshape(3)
    validate_rotation(rotation)
    return rotation.T, -rotation.T @ translation


def camera_center_right_in_left(
    rotation_left_to_right: np.ndarray,
    translation_left_to_right_m: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return the right camera center expressed in the left frame."""

    rotation = np.asarray(rotation_left_to_right, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation_left_to_right_m, dtype=np.float64).reshape(3)
    validate_rotation(rotation)
    return -rotation.T @ translation
