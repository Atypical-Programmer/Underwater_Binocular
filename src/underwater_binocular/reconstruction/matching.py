"""Descriptor matching independent of COLMAP database persistence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FeatureSet


@dataclass(frozen=True)
class DescriptorMatches:
    """Pairs of descriptor indices and distances."""

    query_indices: np.ndarray
    train_indices: np.ndarray
    distances: np.ndarray


def match_descriptors(left: FeatureSet, right: FeatureSet, *, ratio: float = 0.8) -> DescriptorMatches:
    """Perform a ratio-test match, leaving geometric verification to the caller."""

    import cv2

    if left.descriptors.ndim != 2 or right.descriptors.ndim != 2:
        raise ValueError("feature descriptors must be two-dimensional")
    if len(left.descriptors) == 0 or len(right.descriptors) == 0:
        return DescriptorMatches(np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32))
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(left.descriptors, right.descriptors, k=2)
    kept = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    return DescriptorMatches(np.asarray([item.queryIdx for item in kept], dtype=np.int32), np.asarray([item.trainIdx for item in kept], dtype=np.int32), np.asarray([item.distance for item in kept], dtype=np.float32))
