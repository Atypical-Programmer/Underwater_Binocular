"""Feature data models and the explicit lightweight ORB test backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io.images import as_gray, read_image


@dataclass(frozen=True)
class ImageRecord:
    """A deterministic image identity used by feature and pair stages."""

    path: Path
    name: str
    side: str
    frame: int
    timestamp_ns: int = 0


@dataclass(frozen=True)
class FeatureSet:
    """Keypoints and descriptors for one image.

    ``keypoints_xy`` are always in the original image pixel coordinate system.
    The ``coordinate_space`` field is deliberately explicit because feature
    extractors may internally resize their input.
    """

    image_path: Path
    keypoints_xy: np.ndarray
    descriptors: np.ndarray
    model: str
    scores: np.ndarray | None = None
    image_size_hw: tuple[int, int] | None = None
    coordinate_space: str = "original_image_pixels"
    resize: int | None = None
    image_name: str | None = None
    camera_side: str = "left"

    def __post_init__(self) -> None:
        keypoints = np.asarray(self.keypoints_xy)
        descriptors = np.asarray(self.descriptors)
        if keypoints.ndim != 2 or keypoints.shape[1] != 2:
            raise ValueError("feature keypoints must have shape (N, 2)")
        if descriptors.ndim != 2 or len(descriptors) != len(keypoints):
            raise ValueError("feature descriptors must have shape (N, D) matching keypoints")
        if self.scores is not None:
            scores = np.asarray(self.scores)
            if scores.ndim != 1 or len(scores) != len(keypoints):
                raise ValueError("feature scores must have shape (N,)")
        if self.image_size_hw is not None:
            height, width = (int(value) for value in self.image_size_hw)
            if height <= 0 or width <= 0:
                raise ValueError("feature image_size_hw must be positive")
        if self.coordinate_space != "original_image_pixels":
            raise ValueError("FeatureSet coordinates must be restored to original_image_pixels")
        if not self.model:
            raise ValueError("feature model must be explicit")
        if self.camera_side not in {"left", "right"}:
            raise ValueError("camera_side must be left or right")


def extract_orb(image_path: Path, *, max_keypoints: int = 5000) -> FeatureSet:
    """Extract an explicit CPU ORB feature set for tiny/integration fixtures.

    ORB is intentionally a separate, named test backend. The production
    ``aliked-colmap`` workflow never calls this function as a fallback.
    """

    import cv2

    image = read_image(image_path)
    detector = cv2.ORB_create(nfeatures=max_keypoints)
    keypoints, descriptors = detector.detectAndCompute(as_gray(image), None)
    points = np.asarray([item.pt for item in keypoints], dtype=np.float32).reshape(-1, 2)
    if descriptors is None:
        descriptors = np.empty((0, 32), dtype=np.uint8)
    return FeatureSet(
        image_path=image_path,
        keypoints_xy=points,
        descriptors=np.asarray(descriptors),
        model="ORB",
        scores=None,
        image_size_hw=(int(image.shape[0]), int(image.shape[1])),
        coordinate_space="original_image_pixels",
        camera_side="left",
    )


def restore_keypoints_to_original(
    keypoints_xy: np.ndarray,
    *,
    original_size_hw: tuple[int, int],
    extracted_size_hw: tuple[int, int],
    coordinate_space: str,
) -> np.ndarray:
    """Restore keypoints from an explicitly named image coordinate system.

    This helper is used by the ALIKED adapter after controlled resizing and is
    also unit-testable without torch. It rejects unknown coordinate systems so
    a resized-image coordinate array cannot silently be written to COLMAP as
    original-resolution coordinates.
    """

    points = np.asarray(keypoints_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("keypoints must have shape (N, 2)")
    original_height, original_width = (int(value) for value in original_size_hw)
    extracted_height, extracted_width = (int(value) for value in extracted_size_hw)
    if min(original_height, original_width, extracted_height, extracted_width) <= 0:
        raise ValueError("image sizes must be positive")
    if coordinate_space == "original_image_pixels":
        restored = points.copy()
    elif coordinate_space == "resized_image_pixels":
        restored = points * np.asarray(
            [original_width / extracted_width, original_height / extracted_height],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"unsupported keypoint coordinate space: {coordinate_space}")
    if len(restored):
        tolerance = 1.0
        if (
            np.min(restored[:, 0]) < -tolerance
            or np.min(restored[:, 1]) < -tolerance
            or np.max(restored[:, 0]) > original_width + tolerance
            or np.max(restored[:, 1]) > original_height + tolerance
        ):
            raise ValueError("restored keypoints fall outside the original image bounds")
    return np.ascontiguousarray(restored, dtype=np.float32)
