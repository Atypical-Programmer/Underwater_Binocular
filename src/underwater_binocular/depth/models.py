"""Common depth output model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DepthFrame:
    """One depth result.

    ``depth_m`` is the engine's Z/depth product in metres, not necessarily
    Euclidean ``range_m``. Invalid samples are represented by false mask
    entries and NaN in ``depth_m``.
    """

    depth_m: np.ndarray
    valid_mask: np.ndarray
    frame_index: int
    timestamp_ns: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth_m)
        mask = np.asarray(self.valid_mask, dtype=bool)
        if depth.ndim != 2 or mask.shape != depth.shape:
            raise ValueError("DepthFrame depth_m and valid_mask must be matching 2-D arrays")
        if not np.issubdtype(depth.dtype, np.floating):
            raise ValueError("DepthFrame depth_m must be floating point metres")
        if self.frame_index < 0:
            raise ValueError("DepthFrame frame_index cannot be negative")

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(np.asarray(self.depth_m).shape)  # type: ignore[return-value]

    def finite_depth(self) -> np.ndarray:
        """Return valid finite depth samples in metres."""

        depth = np.asarray(self.depth_m, dtype=np.float64)
        mask = np.asarray(self.valid_mask, dtype=bool) & np.isfinite(depth) & (depth > 0.0)
        return depth[mask]
