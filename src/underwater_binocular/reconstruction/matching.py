"""Explicit AdaLAM matching and bounded image-pair selection."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Any

import numpy as np

from .features import FeatureSet


@dataclass(frozen=True)
class DescriptorMatches:
    """Pairs of descriptor indices and distances."""

    query_indices: np.ndarray
    train_indices: np.ndarray
    distances: np.ndarray
    matcher: str = "unknown"
    invalid_matches_removed: int = 0
    duplicate_matches_removed: int = 0

    def __post_init__(self) -> None:
        query = np.asarray(self.query_indices)
        train = np.asarray(self.train_indices)
        distances = np.asarray(self.distances)
        if query.ndim != 1 or train.ndim != 1 or distances.ndim != 1:
            raise ValueError("match arrays must be one-dimensional")
        if not (len(query) == len(train) == len(distances)):
            raise ValueError("match arrays must have equal lengths")


@dataclass(frozen=True)
class ImagePair:
    """One bounded candidate pair with a deterministic category."""

    first_index: int
    second_index: int
    category: str


def _feature_count(feature: FeatureSet, label: str) -> int:
    if feature.descriptors.ndim != 2:
        raise ValueError(f"{label} descriptors must be two-dimensional")
    if feature.keypoints_xy.ndim != 2 or feature.keypoints_xy.shape[1] != 2:
        raise ValueError(f"{label} keypoints must have shape (N, 2)")
    if len(feature.keypoints_xy) != len(feature.descriptors):
        raise ValueError(f"{label} keypoints and descriptors have different lengths")
    return len(feature.descriptors)


def _one_to_one(
    matches: np.ndarray, quality: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Keep the best lower-quality match for every keypoint on each side."""

    if len(matches) <= 1:
        return matches, quality, 0
    order = np.argsort(np.asarray(quality).reshape(-1), kind="stable")
    used_query: set[int] = set()
    used_train: set[int] = set()
    keep: list[int] = []
    for index in order:
        query, train = (int(value) for value in matches[index])
        if query in used_query or train in used_train:
            continue
        used_query.add(query)
        used_train.add(train)
        keep.append(int(index))
    keep_array = np.asarray(keep, dtype=np.int64)
    if len(keep_array):
        return matches[keep_array], quality[keep_array], int(len(matches) - len(keep))
    return matches[:0], quality[:0], int(len(matches))


def match_adalam(
    left: FeatureSet,
    right: FeatureSet,
    *,
    device: Any,
    max_matches: int = 0,
) -> DescriptorMatches:
    """Match two ALIKED feature sets using Kornia's real AdaLAM implementation.

    Missing Kornia or malformed AdaLAM output is an error. This function never
    falls back to BFMatcher or another matcher.
    """

    left_count = _feature_count(left, "left")
    right_count = _feature_count(right, "right")
    if left_count == 0 or right_count == 0:
        return DescriptorMatches(
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            matcher="AdaLAM",
        )
    try:
        import kornia.feature as KF
        import torch
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "AdaLAM requires torch and kornia; install the project's [sfm] extra"
        ) from error
    desc1 = torch.from_numpy(np.asarray(left.descriptors, dtype=np.float32)).to(device)
    desc2 = torch.from_numpy(np.asarray(right.descriptors, dtype=np.float32)).to(device)
    kp1 = torch.from_numpy(np.asarray(left.keypoints_xy, dtype=np.float32)).to(device)
    kp2 = torch.from_numpy(np.asarray(right.keypoints_xy, dtype=np.float32)).to(device)
    if left.image_size_hw is None or right.image_size_hw is None:
        raise ValueError("AdaLAM requires image_size_hw for both feature sets")
    try:
        with torch.inference_mode():
            laf1 = KF.laf_from_center_scale_ori(kp1[None])
            laf2 = KF.laf_from_center_scale_ori(kp2[None])
            quality, raw_indices = KF.match_adalam(
                desc1,
                desc2,
                laf1,
                laf2,
                hw1=tuple(int(value) for value in left.image_size_hw),
                hw2=tuple(int(value) for value in right.image_size_hw),
            )
    except Exception as error:
        raise RuntimeError(f"AdaLAM failed for {left.image_path} and {right.image_path}") from error
    quality_array = quality.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    indices = raw_indices.detach().cpu().numpy().astype(np.int64, copy=False)
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise RuntimeError(f"AdaLAM returned unexpected match shape {indices.shape}")
    if len(quality_array) != len(indices):
        raise RuntimeError("AdaLAM returned a quality array with the wrong length")
    valid = (
        np.isfinite(quality_array)
        & (indices[:, 0] >= 0)
        & (indices[:, 0] < left_count)
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < right_count)
    )
    invalid_removed = int(np.count_nonzero(~valid))
    indices = indices[valid]
    quality_array = quality_array[valid]
    indices, quality_array, duplicate_removed = _one_to_one(indices, quality_array)
    if max_matches > 0 and len(indices) > max_matches:
        order = np.argsort(quality_array, kind="stable")[:max_matches]
        indices = indices[order]
        quality_array = quality_array[order]
    return DescriptorMatches(
        query_indices=np.ascontiguousarray(indices[:, 0], dtype=np.int32),
        train_indices=np.ascontiguousarray(indices[:, 1], dtype=np.int32),
        distances=np.ascontiguousarray(quality_array, dtype=np.float32),
        matcher="AdaLAM",
        invalid_matches_removed=invalid_removed,
        duplicate_matches_removed=duplicate_removed,
    )


def build_image_pairs(
    records: list[Any],
    *,
    temporal_window: int = 5,
    extra_stride: int = 0,
    stereo_window: int = 0,
) -> list[ImagePair]:
    """Build bounded temporal and synchronized stereo pairs.

    Temporal pairs use positions in each selected camera sequence, so uniform
    SVO sampling remains connected without creating an all-pairs O(N²) graph.
    ``stereo_window`` is measured in source-frame indices and includes the
    synchronized pair when set to zero.
    """

    if temporal_window < 0 or extra_stride < 0 or stereo_window < 0:
        raise ValueError("pair windows and strides must be non-negative")
    pairs: dict[tuple[int, int], str] = {}
    by_side: dict[str, list[int]] = {"left": [], "right": []}
    for index, record in enumerate(records):
        side = str(record.side).lower()
        if side not in by_side:
            raise ValueError(f"unsupported image side: {record.side}")
        by_side[side].append(index)
    for side, indices in by_side.items():
        indices.sort(key=lambda index: (int(records[index].frame), index))
        for position, first in enumerate(indices):
            for offset in range(1, temporal_window + 1):
                if position + offset < len(indices):
                    second = indices[position + offset]
                    pair = tuple(sorted((first, second)))
                    pairs[pair] = f"temporal_{side}"
            if extra_stride > temporal_window and position + extra_stride < len(indices):
                second = indices[position + extra_stride]
                pair = tuple(sorted((first, second)))
                pairs[pair] = f"stride_{side}"
    left_indices = by_side["left"]
    right_indices = by_side["right"]
    right_frames = [int(records[index].frame) for index in right_indices]
    for left_index in left_indices:
        left_frame = int(records[left_index].frame)
        first_right = bisect_left(right_frames, left_frame - stereo_window)
        after_right = bisect_right(right_frames, left_frame + stereo_window)
        for right_index in right_indices[first_right:after_right]:
            if abs(int(records[left_index].frame) - int(records[right_index].frame)) <= stereo_window:
                pair = tuple(sorted((left_index, right_index)))
                pairs[pair] = "synchronized_left_right"
    return [ImagePair(first, second, pairs[(first, second)]) for first, second in sorted(pairs)]


def match_descriptors(left: FeatureSet, right: FeatureSet, *, ratio: float = 0.8) -> DescriptorMatches:
    """Perform an explicit ORB-only ratio-test match for lightweight fixtures.

    Production ALIKED workflows call :func:`match_adalam`; this function is
    retained only for the small plumbing test backend and is never a fallback.
    """

    import cv2

    if left.descriptors.ndim != 2 or right.descriptors.ndim != 2:
        raise ValueError("feature descriptors must be two-dimensional")
    if len(left.descriptors) == 0 or len(right.descriptors) == 0:
        return DescriptorMatches(
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            matcher="ORB-BFMatcher",
        )
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(left.descriptors, right.descriptors, k=2)
    kept = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    return DescriptorMatches(
        np.asarray([item.queryIdx for item in kept], dtype=np.int32),
        np.asarray([item.trainIdx for item in kept], dtype=np.int32),
        np.asarray([item.distance for item in kept], dtype=np.float32),
        matcher="ORB-BFMatcher",
    )
