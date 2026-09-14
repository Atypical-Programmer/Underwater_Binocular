"""Rectification diagnostics, separate from production depth export."""

from __future__ import annotations

import json
from pathlib import Path

from ..calibration.loaders import load_calibration_profile
from ..config.loaders import load_dataset_config
from ..geometry.rectification import rectify_calibration


def rectification_diagnostic(profile_path: Path, dataset_path: Path) -> str:
    """Build alpha=0 and alpha=1 summaries and report virtual projection values."""

    dataset = load_dataset_config(dataset_path)
    profile = load_calibration_profile(profile_path)
    models = [rectify_calibration(profile, alpha=alpha) for alpha in (0.0, 1.0)]
    value = {
        "status": "PASS",
        "dataset": dataset.dataset_id,
        "raw_semantic": "RAW_UNRECTIFIED",
        "rectified_semantic": "RECTIFIED",
        "alpha_policy": {
            "0": "crop/zoom to retain valid pixels; may discard source field of view",
            "1": "retain source pixels; may introduce invalid black borders",
        },
        "models": [
            {
                "alpha": model.alpha,
                "output_size": {"width": model.output_size[0], "height": model.output_size[1]},
                "focal_length_px": model.focal_length_px,
                "baseline_m": model.baseline_m,
                "principal_point_left_px": [float(model.p_left[0, 2]), float(model.p_left[1, 2])],
                "principal_point_right_px": [float(model.p_right[0, 2]), float(model.p_right[1, 2])],
                "roi_left": model.roi_left,
                "roi_right": model.roi_right,
            }
            for model in models
        ],
    }
    return json.dumps(value, indent=2)
