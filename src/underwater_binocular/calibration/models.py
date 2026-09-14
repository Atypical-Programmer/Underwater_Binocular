"""Typed camera and stereo calibration models.

All package-internal translation values are metres. Source files using
millimetres are converted by loaders and generated ZED files convert back at
that explicit boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

STEREO_CONVENTION = "X_right = R_left_to_right @ X_left + t_left_to_right"


@dataclass(frozen=True)
class CameraIntrinsics:
    """One pinhole camera's raw calibration in pixel units."""

    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    distortion_model: str
    distortion: tuple[float, ...]

    def __post_init__(self) -> None:
        values = (self.fx_px, self.fy_px, self.cx_px, self.cy_px, *self.distortion)
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("camera intrinsics must be finite")
        if self.fx_px <= 0.0 or self.fy_px <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        if len(self.distortion) < 5:
            raise ValueError("camera distortion must contain at least k1,k2,p1,p2,k3")
        if not self.distortion_model:
            raise ValueError("distortion_model must be explicit")

    @property
    def matrix(self) -> np.ndarray:
        """Return the 3x3 raw-camera intrinsic matrix in pixels."""

        return np.asarray(
            [[self.fx_px, 0.0, self.cx_px], [0.0, self.fy_px, self.cy_px], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def scaled(self, scale: float) -> CameraIntrinsics:
        """Scale pixel coordinates for uniformly resized images."""

        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("image scale must be positive and finite")
        return CameraIntrinsics(
            self.fx_px * scale,
            self.fy_px * scale,
            self.cx_px * scale,
            self.cy_px * scale,
            self.distortion_model,
            self.distortion,
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> CameraIntrinsics:
        distortion = values.get("distortion", values.get("distortion_coefficients"))
        if distortion is None:
            raise ValueError("camera mapping requires distortion")
        return cls(
            fx_px=float(values["fx"] if "fx" in values else values["fx_px"]),
            fy_px=float(values["fy"] if "fy" in values else values["fy_px"]),
            cx_px=float(values["cx"] if "cx" in values else values["cx_px"]),
            cy_px=float(values["cy"] if "cy" in values else values["cy_px"]),
            distortion_model=str(values.get("distortion_model", "opencv_rad_tan")),
            distortion=tuple(float(item) for item in distortion),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "fx": float(self.fx_px),
            "fy": float(self.fy_px),
            "cx": float(self.cx_px),
            "cy": float(self.cy_px),
            "distortion_model": self.distortion_model,
            "distortion": [float(item) for item in self.distortion],
        }


@dataclass(frozen=True)
class StereoCalibration:
    """Stereo calibration using the package-wide left-to-right convention."""

    resolution: tuple[int, int]
    left: CameraIntrinsics
    right: CameraIntrinsics
    rotation_left_to_right: np.ndarray
    translation_left_to_right_m: np.ndarray
    convention: str = STEREO_CONVENTION
    provenance: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    def __post_init__(self) -> None:
        width, height = self.resolution
        rotation = np.asarray(self.rotation_left_to_right, dtype=np.float64)
        translation = np.asarray(self.translation_left_to_right_m, dtype=np.float64).reshape(-1)
        if width <= 0 or height <= 0:
            raise ValueError("calibration resolution must be positive")
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("stereo calibration requires a 3x3 rotation and 3-vector translation")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("stereo calibration transform must be finite")
        if self.convention != STEREO_CONVENTION:
            raise ValueError(f"unsupported stereo convention: {self.convention}")

    @property
    def baseline_m(self) -> float:
        """Return the Euclidean norm of the left-to-right translation in metres."""

        return float(np.linalg.norm(np.asarray(self.translation_left_to_right_m, dtype=np.float64)))

    @property
    def tx_abs_m(self) -> float:
        """Return the absolute x component, kept distinct from ``baseline_m``."""

        return abs(float(np.asarray(self.translation_left_to_right_m).reshape(3)[0]))

    @property
    def rotation_determinant(self) -> float:
        return float(np.linalg.det(np.asarray(self.rotation_left_to_right, dtype=np.float64)))

    @property
    def rotation_orthogonality_error(self) -> float:
        rotation = np.asarray(self.rotation_left_to_right, dtype=np.float64)
        return float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))

    def inverse_transform(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the external right-to-left transform ``R.T, -R.T @ t``."""

        rotation = np.asarray(self.rotation_left_to_right, dtype=np.float64)
        translation = np.asarray(self.translation_left_to_right_m, dtype=np.float64).reshape(3)
        return rotation.T, -rotation.T @ translation

    def to_mapping(self) -> dict[str, Any]:
        """Serialize the canonical schema with explicit metre units."""

        width, height = self.resolution
        return {
            "schema_version": 1,
            "camera": {"resolution": {"width": width, "height": height}},
            "left": self.left.to_mapping(),
            "right": self.right.to_mapping(),
            "stereo": {
                "convention": self.convention,
                "rotation_matrix": np.asarray(self.rotation_left_to_right).tolist(),
                "translation_m": np.asarray(self.translation_left_to_right_m).tolist(),
            },
            "provenance": dict(self.provenance),
        }


def matrix_from_values(values: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert a nested sequence into a finite 3x3 matrix."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("expected a finite 3x3 matrix")
    return matrix
