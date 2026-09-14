"""Shared depth statistics consumed by production outputs and diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .models import DepthFrame


def summarize_values(values: Iterable[float], *, prefix: str = "") -> dict[str, Any]:
    """Summarize positive finite metres with explicit invalid counts."""

    raw = np.asarray(list(values), dtype=np.float64).reshape(-1)
    valid = raw[np.isfinite(raw) & (raw > 0.0)]
    result: dict[str, Any] = {
        "count": int(valid.size),
        "invalid_count": int(raw.size - valid.size),
        "valid_ratio": float(valid.size / raw.size) if raw.size else 0.0,
    }
    if valid.size:
        for name, value in (
            ("min", np.min(valid)),
            ("p01", np.percentile(valid, 1)),
            ("p05", np.percentile(valid, 5)),
            ("median", np.median(valid)),
            ("mean", np.mean(valid)),
            ("std", np.std(valid)),
            ("p95", np.percentile(valid, 95)),
            ("p99", np.percentile(valid, 99)),
            ("max", np.max(valid)),
        ):
            result[f"{prefix}{name}_m"] = float(value)
    return result


def center_measurement(frame: DepthFrame, radius: int = 2) -> dict[str, Any]:
    """Return center depth and window median without calling image center optical."""

    depth = np.asarray(frame.depth_m, dtype=np.float64)
    height, width = depth.shape
    center_x, center_y = width // 2, height // 2
    y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
    x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
    window = depth[y0:y1, x0:x1]
    valid = np.asarray(frame.valid_mask, dtype=bool)[y0:y1, x0:x1]
    valid_values = window[valid & np.isfinite(window) & (window > 0.0)]
    center = float(depth[center_y, center_x])
    if not np.isfinite(center) or center <= 0.0 or not frame.valid_mask[center_y, center_x]:
        center = float("nan")
    return {
        "frame_index": frame.frame_index,
        "center_x": center_x,
        "center_y": center_y,
        "center_depth_m": center,
        "center_window_median_depth_m": float(np.median(valid_values)) if valid_values.size else float("nan"),
        "center_window_valid_count": int(valid_values.size),
    }


def region_statistics(frame: DepthFrame, regions: dict[str, tuple[slice, slice]]) -> dict[str, dict[str, Any]]:
    """Compute valid count/ratio and quantiles for named image regions."""

    result: dict[str, dict[str, Any]] = {}
    for name, (ys, xs) in regions.items():
        values = np.asarray(frame.depth_m, dtype=np.float64)[ys, xs]
        mask = np.asarray(frame.valid_mask, dtype=bool)[ys, xs]
        result[name] = summarize_values(values[mask])
        result[name]["pixel_count"] = int(values.size)
        result[name]["valid_pixel_count"] = int(np.count_nonzero(mask & np.isfinite(values) & (values > 0.0)))
        result[name]["valid_ratio"] = result[name]["valid_pixel_count"] / values.size if values.size else 0.0
    return result
