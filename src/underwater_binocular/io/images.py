"""Image conversion helpers with explicit raw/rectified semantics."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np


class ImageSemantic(str, Enum):
    """Whether an image is raw distorted data or rectified data."""

    RAW_UNRECTIFIED = "RAW_UNRECTIFIED"
    RECTIFIED = "RECTIFIED"


def as_gray(image: np.ndarray) -> np.ndarray:
    """Convert a raw or rectified image to contiguous uint8 grayscale."""

    import cv2

    value = np.asarray(image)
    if value.ndim == 3 and value.shape[2] == 4:
        value = cv2.cvtColor(value, cv2.COLOR_BGRA2GRAY)
    elif value.ndim == 3 and value.shape[2] == 3:
        value = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)
    if value.ndim != 2:
        raise ValueError(f"expected grayscale-compatible image, got {value.shape}")
    if value.dtype != np.uint8:
        value = np.clip(value, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(value)


def as_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an image to contiguous BGR uint8 without changing coordinates."""

    import cv2

    value = np.asarray(image)
    if value.ndim == 2:
        value = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
    elif value.ndim == 3 and value.shape[2] == 4:
        value = cv2.cvtColor(value, cv2.COLOR_BGRA2BGR)
    elif value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"expected a 2-D, 3-channel, or 4-channel image, got {value.shape}")
    if value.dtype != np.uint8:
        value = np.clip(value, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(value)


def read_image(path: Path) -> np.ndarray:
    """Read an image and fail loudly when OpenCV cannot decode it."""

    import cv2

    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise OSError(f"could not decode image: {path}")
    return value


def write_image(path: Path, image: np.ndarray) -> None:
    """Write one image, preserving the caller's explicit pixel semantics."""

    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.asarray(image)):
        raise OSError(f"could not write image: {path}")
