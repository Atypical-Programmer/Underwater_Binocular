"""Small, versioned behavioral anchors from the prior production runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from underwater_binocular.calibration.loaders import load_calibration_profile
from underwater_binocular.geometry.rectification import rectify_calibration

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "validation/reference/20260802_150233"


def _read(name: str) -> dict[str, object]:
    return json.loads((REFERENCE / name).read_text(encoding="utf-8"))


def test_calibration_reference_is_preserved() -> None:
    reference = _read("calibration_reference.json")
    calibration = load_calibration_profile(ROOT / "calibration/profiles/zed2i_37395692_custom.yaml")

    assert reference["stereo"]["convention"] == calibration.convention
    np.testing.assert_allclose(
        reference["stereo"]["translation_norm_m"],
        calibration.baseline_m,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        reference["diagnostics"]["rotation_determinant"],
        calibration.rotation_determinant,
        atol=1.0e-12,
    )


def test_rectification_reference_and_sgbm_reference_are_distinct_scales() -> None:
    calibration = load_calibration_profile(ROOT / "calibration/profiles/zed2i_37395692_custom.yaml")
    rectification = _read("rectification_reference.json")
    sgbm = _read("sgbm_reference.json")
    alpha0 = rectify_calibration(calibration, alpha=0.0)
    half = rectify_calibration(calibration, alpha=0.0, scale=0.5)

    np.testing.assert_allclose(
        rectification["custom_alpha0"]["focal_length_px"],
        alpha0.focal_length_px,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        sgbm["rectified_focal_length_px"],
        half.focal_length_px,
        atol=1.0e-9,
    )
    assert sgbm["disparity_units"] == "StereoSGBM fixed-point output divided by 16.0"


def test_depth_and_refractive_references_keep_their_scientific_status() -> None:
    depth = _read("depth_reference.json")
    refractive = _read("refractive_audit_reference.json")

    assert depth["meaning"] == "operational custom-calibrated SDK depth; not independent physical ground truth"
    assert refractive["r1_status"] == "NOT_IDENTIFIABLE"
    assert refractive["r1_definitive_depth"] is None
