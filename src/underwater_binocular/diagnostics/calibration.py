"""Calibration diagnostics and source/runtime comparison entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..calibration.loaders import load_calibration_profile, sha256_file
from ..calibration.validation import validate_calibration
from ..config.loaders import load_dataset_config


def calibration_diagnostic(profile_path: Path, dataset_path: Path) -> str:
    """Return a JSON diagnostic for profile/config consistency without opening an SVO."""

    profile = load_calibration_profile(profile_path)
    dataset = load_dataset_config(dataset_path)
    value: dict[str, Any] = {
        "status": "PASS" if profile.resolution == dataset.resolution else "FAIL",
        "dataset": dataset.dataset_id,
        "profile": profile_path.as_posix(),
        "profile_hash": sha256_file(profile_path),
        "dataset_resolution": {"width": dataset.resolution[0], "height": dataset.resolution[1]},
        "profile_resolution": {"width": profile.resolution[0], "height": profile.resolution[1]},
        "validation": validate_calibration(profile),
        "physical_provenance": profile.provenance,
    }
    return json.dumps(value, indent=2, default=str)
