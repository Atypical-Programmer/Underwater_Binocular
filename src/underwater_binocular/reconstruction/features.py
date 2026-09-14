"""Feature extraction boundary for research SfM pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io.images import as_gray, read_image


@dataclass(frozen=True)
class FeatureSet:
    """Keypoints and descriptors for one image."""

    image_path: Path
    keypoints_xy: np.ndarray
    descriptors: np.ndarray
    model: str


def extract_orb(image_path: Path, *, max_keypoints: int = 5000) -> FeatureSet:
    """Extract an explicit CPU ORB feature set for tiny/integration fixtures."""

    import cv2

    image = read_image(image_path)
    detector = cv2.ORB_create(nfeatures=max_keypoints)
    keypoints, descriptors = detector.detectAndCompute(as_gray(image), None)
    points = np.asarray([item.pt for item in keypoints], dtype=np.float32).reshape(-1, 2)
    if descriptors is None:
        descriptors = np.empty((0, 32), dtype=np.uint8)
    return FeatureSet(image_path, points, np.asarray(descriptors), "ORB")
