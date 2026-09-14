"""Depth diagnostics such as native/custom and center-edge summaries."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..depth.models import DepthFrame
from ..depth.statistics import region_statistics


def center_edge_diagnostic(frame: DepthFrame, *, valid_ratio_threshold: float = 0.1) -> dict[str, Any]:
    """Summarize a 3x3 grid and exclude low-validity regions from comparisons."""

    height, width = frame.shape
    y_edges = np.linspace(0, height, 4, dtype=int)
    x_edges = np.linspace(0, width, 4, dtype=int)
    regions: dict[str, tuple[slice, slice]] = {}
    for row in range(3):
        for column in range(3):
            regions[f"r{row}{column}"] = (slice(y_edges[row], y_edges[row + 1]), slice(x_edges[column], x_edges[column + 1]))
    stats = region_statistics(frame, regions)
    eligible = {name: value for name, value in stats.items() if value["valid_ratio"] >= valid_ratio_threshold}
    return {"valid_ratio_threshold": valid_ratio_threshold, "regions": stats, "eligible_regions": sorted(eligible)}
