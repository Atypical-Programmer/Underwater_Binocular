"""Explicit ALIKED matchers and bounded image-pair selection."""

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


def load_lightglue_matcher(
    device: Any,
    *,
    filter_threshold: float = 0.1,
    depth_confidence: float = 0.95,
    width_confidence: float = 0.99,
) -> tuple[Any, dict[str, float]]:
    """Load one LightGlue matcher configured for ALIKED descriptors.

    The matcher is created once per reconstruction and reused for every image
    pair.  Keeping this separate from :func:`match_lightglue` avoids loading
    the LightGlue weights thousands of times during a large run.
    """

    values = {
        "filter_threshold": float(filter_threshold),
        "depth_confidence": float(depth_confidence),
        "width_confidence": float(width_confidence),
    }
    if not 0.0 <= values["filter_threshold"] <= 1.0:
        raise ValueError("LightGlue filter_threshold must be between 0 and 1")
    for name in ("depth_confidence", "width_confidence"):
        value = values[name]
        if value != -1.0 and not 0.0 <= value <= 1.0:
            raise ValueError(f"LightGlue {name} must be -1 or between 0 and 1")
    try:
        from lightglue import LightGlue
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "LightGlue matching requires the LightGlue package; install the project's [sfm] extra"
        ) from error
    try:
        matcher = LightGlue(features="aliked", **values).eval().to(device)
    except Exception as error:  # pragma: no cover - depends on model weights/runtime
        raise RuntimeError("LightGlue ALIKED matcher initialization failed") from error
    return matcher, values


def _lightglue_batch_item(value: Any, *, minimum_ndim: int) -> np.ndarray:
    """Convert a LightGlue batch/list output to one NumPy item."""

    if isinstance(value, (list, tuple)):
        if not value:
            return np.empty((0,), dtype=np.float32)
        value = value[0]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim >= minimum_ndim + 1 and array.shape[0] == 1:
        array = array[0]
    return array


def match_lightglue(
    left: FeatureSet,
    right: FeatureSet,
    *,
    matcher: Any,
    device: Any,
    max_matches: int = 0,
) -> DescriptorMatches:
    """Match two ALIKED feature sets with a real LightGlue model.

    LightGlue reports a confidence score where larger is better.  The common
    ``DescriptorMatches`` contract stores a lower-is-better distance, so this
    adapter stores ``1 - score`` and applies the same deterministic one-to-one
    filtering and optional match cap as the AdaLAM adapter.
    """

    left_count = _feature_count(left, "left")
    right_count = _feature_count(right, "right")
    if max_matches < 0:
        raise ValueError("max_matches must be non-negative")
    if left_count == 0 or right_count == 0:
        return DescriptorMatches(
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            matcher="LightGlue",
        )
    if left.image_size_hw is None or right.image_size_hw is None:
        raise ValueError("LightGlue requires image_size_hw for both feature sets")
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "LightGlue matching requires torch; install the project's [sfm] extra"
        ) from error

    def tensor(value: np.ndarray) -> Any:
        return torch.from_numpy(np.ascontiguousarray(value)).to(device)

    left_height, left_width = (int(value) for value in left.image_size_hw)
    right_height, right_width = (int(value) for value in right.image_size_hw)
    data = {
        "image0": {
            "keypoints": tensor(np.asarray(left.keypoints_xy, dtype=np.float32))[None],
            "descriptors": tensor(np.asarray(left.descriptors, dtype=np.float32))[None],
            # LightGlue expects (width, height), while FeatureSet stores (height, width).
            "image_size": torch.tensor(
                [[left_width, left_height]], dtype=torch.float32, device=device
            ),
        },
        "image1": {
            "keypoints": tensor(np.asarray(right.keypoints_xy, dtype=np.float32))[None],
            "descriptors": tensor(np.asarray(right.descriptors, dtype=np.float32))[None],
            "image_size": torch.tensor(
                [[right_width, right_height]], dtype=torch.float32, device=device
            ),
        },
    }
    try:
        with torch.inference_mode():
            output = matcher(data)
    except Exception as error:  # pragma: no cover - depends on model/runtime
        raise RuntimeError(f"LightGlue failed for {left.image_path} and {right.image_path}") from error
    if not isinstance(output, dict):
        raise RuntimeError("LightGlue returned a non-dictionary output")

    if "matches" in output:
        indices = _lightglue_batch_item(output["matches"], minimum_ndim=2)
        if indices.size == 0:
            indices = np.empty((0, 2), dtype=np.int64)
        if indices.ndim != 2 or indices.shape[1] != 2:
            raise RuntimeError(f"LightGlue returned unexpected match shape {indices.shape}")
        score_value = output.get("scores")
        if score_value is not None:
            scores = _lightglue_batch_item(score_value, minimum_ndim=1).reshape(-1)
        else:
            scores0_value = output.get("matching_scores0")
            if scores0_value is None:
                scores = np.ones(len(indices), dtype=np.float32)
            else:
                scores0 = _lightglue_batch_item(scores0_value, minimum_ndim=1).reshape(-1)
                scores = scores0[indices[:, 0].astype(np.int64, copy=False)]
    elif "matches0" in output:
        matches0 = _lightglue_batch_item(output["matches0"], minimum_ndim=1).reshape(-1)
        query = np.flatnonzero(matches0 >= 0).astype(np.int64, copy=False)
        indices = np.column_stack((query, matches0[query].astype(np.int64, copy=False)))
        score_value = output.get("matching_scores0")
        if score_value is None:
            scores = np.ones(len(indices), dtype=np.float32)
        else:
            scores0 = _lightglue_batch_item(score_value, minimum_ndim=1).reshape(-1)
            scores = scores0[query]
    else:
        raise RuntimeError("LightGlue output does not contain matches or matches0")

    indices = np.asarray(indices, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(scores) != len(indices):
        raise RuntimeError("LightGlue returned scores with a different length than matches")
    valid = (
        np.isfinite(scores)
        & (indices[:, 0] >= 0)
        & (indices[:, 0] < left_count)
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < right_count)
    )
    invalid_removed = int(np.count_nonzero(~valid))
    indices = indices[valid]
    scores = scores[valid]
    quality = 1.0 - scores
    indices, quality, duplicate_removed = _one_to_one(indices, quality)
    if max_matches > 0 and len(indices) > max_matches:
        order = np.argsort(quality, kind="stable")[:max_matches]
        indices = indices[order]
        quality = quality[order]
    return DescriptorMatches(
        query_indices=np.ascontiguousarray(indices[:, 0], dtype=np.int32),
        train_indices=np.ascontiguousarray(indices[:, 1], dtype=np.int32),
        distances=np.ascontiguousarray(quality, dtype=np.float32),
        matcher="LightGlue",
        invalid_matches_removed=invalid_removed,
        duplicate_matches_removed=duplicate_removed,
    )


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
