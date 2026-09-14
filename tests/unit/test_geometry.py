"""Pure geometry regression tests; these never import the ZED SDK."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from underwater_binocular.calibration.loaders import load_calibration_profile
from underwater_binocular.geometry.rectification import rectify_calibration
from underwater_binocular.geometry.refraction import (
    FlatPortModel,
    RefractiveModelNotIdentifiable,
    TotalInternalReflectionError,
    pair_frame_records,
    ray_plane_intersection,
    refract_vector,
    trace_flat_port_ray,
)
from underwater_binocular.geometry.rig_refraction import (
    RigCameraPose,
    RigFlatPortModel,
    project_water_point_to_raw_pixel,
    trace_rig_pixel,
    triangulate_rig_water_rays,
)
from underwater_binocular.geometry.triangulation import (
    StereoCorrespondence,
    triangulate_rectified,
)

ROOT = Path(__file__).resolve().parents[2]


def test_snell_law_and_total_internal_reflection() -> None:
    theta_air = math.radians(30.0)
    result = refract_vector(
        [math.sin(theta_air), 0.0, math.cos(theta_air)],
        [0.0, 0.0, 1.0],
        1.0,
        1.333,
    )
    assert abs(math.sin(theta_air) - 1.333 * result[0]) < 1.0e-10
    with pytest.raises(TotalInternalReflectionError):
        refract_vector([math.sin(math.radians(60.0)), 0.0, 0.5], [0.0, 0.0, 1.0], 1.5, 1.0)


def test_unknown_port_model_is_not_used_as_definitive_geometry() -> None:
    model = FlatPortModel(n_air=1.0, n_glass=1.5, n_water=1.333)

    assert model.status == "unknown"
    with pytest.raises(RefractiveModelNotIdentifiable):
        trace_flat_port_ray([960.0, 540.0], np.eye(3), np.zeros(5), model)


def test_rectification_preserves_recorded_custom_alpha_anchors() -> None:
    calibration = load_calibration_profile(ROOT / "calibration/profiles/zed2i_37395692_custom.yaml")
    alpha0 = rectify_calibration(calibration, alpha=0.0)
    alpha1 = rectify_calibration(calibration, alpha=1.0)
    half = rectify_calibration(calibration, alpha=0.0, scale=0.5)

    np.testing.assert_allclose(alpha0.focal_length_px, 3635.497171961014, rtol=0.0, atol=1.0e-9)
    np.testing.assert_allclose(alpha1.focal_length_px, 1423.952587176348, rtol=0.0, atol=1.0e-9)
    np.testing.assert_allclose(half.focal_length_px, 2140.9715416077433, rtol=0.0, atol=1.0e-9)
    np.testing.assert_allclose(alpha0.baseline_m, calibration.baseline_m, atol=1.0e-15)
    assert half.output_size == (960, 540)


def test_rectified_triangulation_and_ray_plane_intersection() -> None:
    t, point = ray_plane_intersection(
        [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 1.0]
    )
    assert t == 2.0
    np.testing.assert_allclose(point, [0.0, 0.0, 2.0], atol=1.0e-12)
    point_xyz = triangulate_rectified(
        StereoCorrespondence((650.0, 360.0), (610.0, 360.0)),
        focal_length_px=1000.0,
        baseline_m=0.12,
        principal_point_px=(640.0, 360.0),
    )
    np.testing.assert_allclose(point_xyz, [0.03, 0.0, 3.0], atol=1.0e-12)


def test_shared_plane_rig_round_trip() -> None:
    k = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])
    d = np.zeros(5, dtype=np.float64)
    left = RigCameraPose(np.eye(3), np.zeros(3))
    right = RigCameraPose(np.eye(3), np.array([0.12, 0.0, 0.0]))
    model = RigFlatPortModel.shared_parallel(1.0, 1.5, 1.333, [0.0, 0.0, 1.0], 0.02, 0.005)
    target = np.array([0.5, 0.2, 3.0])

    left_projection = project_water_point_to_raw_pixel("left", target, k, d, left, model)
    right_projection = project_water_point_to_raw_pixel("right", target, k, d, right, model)
    left_ray = trace_rig_pixel("left", left_projection["pixel_uv"], k, d, left, model)
    right_ray = trace_rig_pixel("right", right_projection["pixel_uv"], k, d, right, model)
    result = triangulate_rig_water_rays(left_ray, right_ray)

    np.testing.assert_allclose(result["midpoint_rig_m"], target, atol=1.0e-8)
    assert result["ray_gap_m"] < 1.0e-8
    assert result["positive_parameters"] is True


def test_frame_alignment_uses_explicit_frame_index() -> None:
    native = [{"frame_index": 0}, {"frame_index": 1}, {"frame_index": 2}]
    custom = [{"frame_index": 0}, {"frame_index": 2}]

    paired = pair_frame_records(native, custom)
    assert [row["frame_index"] for row in paired] == [0, 2]
