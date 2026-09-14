"""Unit tests for the canonical calibration contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from underwater_binocular.calibration.loaders import (
    load_calibration_profile,
    load_opencv_calibration,
)
from underwater_binocular.calibration.validation import validate_calibration
from underwater_binocular.geometry.stereo import (
    baseline_m,
    camera_center_right_in_left,
    transform_points,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "calibration" / "profiles" / "zed2i_37395692_custom.yaml"
GENERATED = ROOT / "calibration" / "generated"


def test_canonical_profile_matches_recorded_calibration_anchor() -> None:
    calibration = load_calibration_profile(PROFILE)

    assert calibration.resolution == (1920, 1080)
    np.testing.assert_allclose(calibration.left.matrix[0, 0], 1443.326338, atol=1.0e-12)
    np.testing.assert_allclose(calibration.right.matrix[1, 2], 514.232843, atol=1.0e-12)
    np.testing.assert_allclose(calibration.baseline_m, 0.1237302227994842, atol=1.0e-15)
    assert calibration.convention == "X_right = R_left_to_right @ X_left + t_left_to_right"


def test_calibration_validation_reports_rotation_and_distinct_tx() -> None:
    calibration = load_calibration_profile(PROFILE)
    report = validate_calibration(calibration)

    assert report["rotation_determinant"] == calibration.rotation_determinant
    np.testing.assert_allclose(report["tx_abs_m"], 0.1224352, atol=1.0e-12)
    assert report["baseline_norm_m"] > report["tx_abs_m"]


def test_transform_convention_and_inverse_camera_center() -> None:
    calibration = load_calibration_profile(PROFILE)
    left_point = np.array([[0.25, -0.08, 2.4]], dtype=np.float64)
    right_point = transform_points(
        left_point,
        calibration.rotation_left_to_right,
        calibration.translation_left_to_right_m,
    )
    inverse_rotation, inverse_translation = calibration.inverse_transform()
    recovered = transform_points(right_point, inverse_rotation, inverse_translation)

    np.testing.assert_allclose(recovered, left_point, atol=1.0e-12)
    np.testing.assert_allclose(
        camera_center_right_in_left(
            calibration.rotation_left_to_right,
            calibration.translation_left_to_right_m,
        ),
        inverse_translation,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(baseline_m(calibration.translation_left_to_right_m), calibration.baseline_m)


def test_generated_zed_calibration_round_trips_to_profile() -> None:
    profile = load_calibration_profile(PROFILE)
    generated = load_opencv_calibration(GENERATED / "zed_custom_opencv.yml")

    np.testing.assert_allclose(generated.left.matrix, profile.left.matrix, atol=1.0e-12)
    np.testing.assert_allclose(generated.right.matrix, profile.right.matrix, atol=1.0e-12)
    np.testing.assert_allclose(
        generated.rotation_left_to_right,
        profile.rotation_left_to_right,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        generated.translation_left_to_right_m,
        profile.translation_left_to_right_m,
        atol=1.0e-12,
    )


def test_generated_colmap_config_is_explicit_and_frozen_by_default() -> None:
    value = json.loads((GENERATED / "colmap_camera.json").read_text(encoding="utf-8"))

    assert value["generated_file"] is True
    assert value["freeze_calibration_default"] is True
    assert value["camera_model"] == "OPENCV"
    assert value["left"]["params"][:4] == [1443.326338, 1441.541002, 967.033878, 540.469238]
