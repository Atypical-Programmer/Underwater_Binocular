"""Synthetic shared-plane rig tests for the final physical audit model."""

from __future__ import annotations

import unittest

import numpy as np

from rig_refractive_geometry import (
    RigCameraPose,
    RigFlatPortModel,
    RigPlane,
    trace_rig_pixel,
    triangulate_rig_water_rays,
    project_water_point_to_raw_pixel,
)


class RigRefractiveGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.k = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])
        self.d = np.zeros(5, dtype=np.float64)
        self.left = RigCameraPose(np.eye(3), np.zeros(3))
        self.right = RigCameraPose(np.eye(3), np.array([0.12, 0.0, 0.0]))

    def _model(self, n_air: float, n_glass: float, n_water: float, thickness: float = 0.005) -> RigFlatPortModel:
        return RigFlatPortModel.shared_parallel(n_air, n_glass, n_water, [0.0, 0.0, 1.0], 0.02, thickness)

    def _round_trip(self, target: np.ndarray, model: RigFlatPortModel) -> dict[str, float | np.ndarray]:
        left_projection = project_water_point_to_raw_pixel("left", target, self.k, self.d, self.left, model)
        right_projection = project_water_point_to_raw_pixel("right", target, self.k, self.d, self.right, model)
        left_ray = trace_rig_pixel("left", left_projection["pixel_uv"], self.k, self.d, self.left, model)
        right_ray = trace_rig_pixel("right", right_projection["pixel_uv"], self.k, self.d, self.right, model)
        triangulation = triangulate_rig_water_rays(left_ray, right_ray)
        return {
            "target": target,
            "left_pixel": left_projection["pixel_uv"],
            "right_pixel": right_projection["pixel_uv"],
            "midpoint": triangulation["midpoint_rig_m"],
            "ray_gap": triangulation["ray_gap_m"],
            "left_residual": left_projection["residual_m"],
            "right_residual": right_projection["residual_m"],
        }

    def test_case_a_no_refraction_recovers_pinhole(self) -> None:
        result = self._round_trip(np.array([0.5, 0.2, 3.0]), self._model(1.0, 1.0, 1.0))
        np.testing.assert_allclose(result["midpoint"], result["target"], atol=1.0e-9)
        self.assertLess(result["ray_gap"], 1.0e-9)

    def test_case_b_zero_glass_thickness_is_single_interface(self) -> None:
        # The model constructor rejects a zero thickness as a measured model;
        # the limiting case is represented by two coincident planes instead.
        inner = RigPlane(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.02]))
        model = RigFlatPortModel(1.0, 1.5, 1.333, inner_plane_shared=inner, outer_plane_shared=inner)
        ray = trace_rig_pixel("left", [700.0, 400.0], self.k, self.d, self.left, model)
        self.assertTrue(np.isfinite(ray.water_direction_rig).all())
        self.assertAlmostEqual(float(np.linalg.norm(ray.outer_point_rig - ray.inner_point_rig)), 0.0, places=12)

    def test_case_c_normal_incidence_does_not_bend(self) -> None:
        ray = trace_rig_pixel("left", [640.0, 360.0], self.k, self.d, self.left, self._model(1.0, 1.5, 1.333))
        np.testing.assert_allclose(ray.air_direction_rig, [0.0, 0.0, 1.0], atol=1.0e-12)
        np.testing.assert_allclose(ray.water_direction_rig, [0.0, 0.0, 1.0], atol=1.0e-12)

    def test_case_d_symmetric_target_round_trip(self) -> None:
        result = self._round_trip(np.array([0.0, 0.0, 3.0]), self._model(1.0, 1.5, 1.333))
        np.testing.assert_allclose(result["midpoint"], result["target"], atol=1.0e-8)
        self.assertLess(result["ray_gap"], 1.0e-8)

    def test_case_e_off_axis_target_round_trip(self) -> None:
        result = self._round_trip(np.array([0.5, 0.2, 3.0]), self._model(1.0, 1.5, 1.333))
        np.testing.assert_allclose(result["midpoint"], result["target"], atol=1.0e-8)
        self.assertLess(result["ray_gap"], 1.0e-8)

    def test_case_f_shared_plane_differs_from_local_separate_planes(self) -> None:
        shared = self._model(1.0, 1.5, 1.333)
        local_left = RigPlane(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.02]))
        local_right = RigPlane(np.array([0.0, 0.0, 1.0]), np.array([0.12, 0.0, 0.02]))
        separate = RigFlatPortModel(
            1.0,
            1.5,
            1.333,
            inner_plane_left=local_left,
            outer_plane_left=local_left,
            inner_plane_right=local_right,
            outer_plane_right=local_right,
            mode="separate_flat",
        )
        shared_ray = trace_rig_pixel("right", [740.0, 410.0], self.k, self.d, self.right, shared)
        separate_ray = trace_rig_pixel("right", [740.0, 410.0], self.k, self.d, self.right, separate)
        self.assertGreater(float(np.linalg.norm(shared_ray.outer_point_rig - separate_ray.outer_point_rig)), 1.0e-6)

    def test_case_g_dry_pinhole_round_trip(self) -> None:
        result = self._round_trip(np.array([0.25, -0.1, 2.0]), self._model(1.0, 1.0, 1.0))
        np.testing.assert_allclose(result["midpoint"], result["target"], atol=1.0e-9)

    def test_dome_is_not_silently_flat(self) -> None:
        dome = RigFlatPortModel(1.0, 1.5, 1.333, mode="dome")
        self.assertEqual(dome.status, "invalid_flat_port_model_for_dome")


if __name__ == "__main__":
    unittest.main()
