"""Unit tests for diagnostic refractive and stereo geometry helpers."""

from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from refractive_geometry import (
    FlatPortModel,
    RefractiveModelNotIdentifiable,
    StereoExtrinsics,
    TotalInternalReflectionError,
    pair_frame_records,
    ray_plane_intersection,
    refract_vector,
    trace_flat_port_ray,
    triangulate_two_rays,
)


class RefractiveGeometryTests(unittest.TestCase):
    def test_snell_normal_incidence(self) -> None:
        result = refract_vector([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 1.0, 1.333)
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=1.0e-12)

    def test_snell_known_angle(self) -> None:
        theta_air = math.radians(30.0)
        result = refract_vector(
            [math.sin(theta_air), 0.0, math.cos(theta_air)],
            [0.0, 0.0, 1.0],
            1.0,
            1.333,
        )
        theta_water = math.asin(float(result[0]))
        self.assertLess(
            abs(1.0 * math.sin(theta_air) - 1.333 * math.sin(theta_water)),
            1.0e-10,
        )

    def test_snell_refractive_index_identity(self) -> None:
        incident = np.array([0.2, 0.3, 0.932737905], dtype=np.float64)
        result = refract_vector(incident, [0.0, 0.0, 1.0], 1.5, 1.5)
        np.testing.assert_allclose(result, incident / np.linalg.norm(incident), atol=1.0e-12)

    def test_snell_rejects_total_internal_reflection(self) -> None:
        theta_glass = math.radians(60.0)
        with self.assertRaises(TotalInternalReflectionError):
            refract_vector(
                [math.sin(theta_glass), 0.0, math.cos(theta_glass)],
                [0.0, 0.0, 1.0],
                1.5,
                1.0,
            )

    def test_ray_plane_intersection(self) -> None:
        t, point = ray_plane_intersection(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0],
        )
        self.assertAlmostEqual(t, 2.0, places=12)
        np.testing.assert_allclose(point, [0.0, 0.0, 2.0], atol=1.0e-12)

    def test_two_ray_triangulation_exact_intersection(self) -> None:
        target = np.array([0.2, 0.0, 3.0])
        origin_left = np.zeros(3)
        origin_right = np.array([1.0, 0.0, 0.0])
        direction_left = target - origin_left
        direction_right = target - origin_right
        result = triangulate_two_rays(origin_left, direction_left, origin_right, direction_right)
        np.testing.assert_allclose(result.midpoint, target, atol=1.0e-12)
        self.assertLess(result.ray_gap_m, 1.0e-12)
        self.assertGreater(result.parameter_left_m, 0.0)
        self.assertGreater(result.parameter_right_m, 0.0)

    def test_two_ray_triangulation_skew(self) -> None:
        result = triangulate_two_rays(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        )
        self.assertAlmostEqual(result.ray_gap_m, 1.0, places=12)

    def test_unknown_flat_port_never_produces_definitive_ray(self) -> None:
        model = FlatPortModel(n_air=1.0, n_glass=1.5, n_water=1.333)
        self.assertEqual(model.status, "unknown")
        with self.assertRaises(RefractiveModelNotIdentifiable):
            trace_flat_port_ray([960.0, 540.0], np.eye(3), np.zeros(5), model)

    def test_custom_p_q_identity(self) -> None:
        k_left = np.array([[1443.326338, 0.0, 967.033878], [0.0, 1441.541002, 540.469238], [0.0, 0.0, 1.0]])
        k_right = np.array([[1449.830254, 0.0, 975.331674], [0.0, 1448.097563, 514.232843], [0.0, 0.0, 1.0]])
        d_left = np.array([0.3151847, -0.7495814, 0.002129304, 0.005759798, 6.688496])
        d_right = np.array([0.3023032, -0.1692313, 0.001685104, 0.01069086, 1.944058])
        r = np.array([[0.999968008, 0.00548864430, 0.00581871018], [-0.00548930265, 0.999984929, 0.0000971791651], [-0.00581808910, -0.000129116717, 0.999983066]])
        t = np.array([-122.4352, 0.1676, 17.8539], dtype=np.float64)
        _, _, p1, p2, q, _, _ = cv2.stereoRectify(
            k_left, d_left, k_right, d_right, (1920, 1080), r, t.reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=1.0, newImageSize=(1920, 1080)
        )
        f = float(p1[0, 0])
        baseline_mm = abs(float(p2[0, 3] / p2[0, 0]))
        self.assertLess(abs(float(p1[0, 2] - p2[0, 2])), 1.0e-9)
        self.assertLess(abs(float(p1[1, 2] - p2[1, 2])), 1.0e-9)
        self.assertLess(abs(float(p2[0, 3] + f * baseline_mm)), 1.0e-8)
        self.assertLess(abs(float(q[2, 3] - f)), 1.0e-8)
        self.assertLess(abs(float(q[3, 2] - 1.0 / baseline_mm)), 1.0e-12)

    def test_alpha_depth_invariance(self) -> None:
        k_left = np.array([[1443.326338, 0.0, 967.033878], [0.0, 1441.541002, 540.469238], [0.0, 0.0, 1.0]])
        k_right = np.array([[1449.830254, 0.0, 975.331674], [0.0, 1448.097563, 514.232843], [0.0, 0.0, 1.0]])
        d_left = np.array([0.3151847, -0.7495814, 0.002129304, 0.005759798, 6.688496])
        d_right = np.array([0.3023032, -0.1692313, 0.001685104, 0.01069086, 1.944058])
        r = np.array([[0.999968008, 0.00548864430, 0.00581871018], [-0.00548930265, 0.999984929, 0.0000971791651], [-0.00581808910, -0.000129116717, 0.999983066]])
        t = np.array([-122.4352, 0.1676, 17.8539], dtype=np.float64) / 1000.0
        point_left = np.array([[0.25, -0.08, 2.4]], dtype=np.float64)
        point_right = (r @ point_left[0]) + t
        pixel_left, _ = cv2.projectPoints(point_left, np.zeros(3), np.zeros(3), k_left, d_left)
        pixel_right, _ = cv2.projectPoints(point_right.reshape(1, 3), np.zeros(3), np.zeros(3), k_right, d_right)
        depths = []
        for alpha in (0.0, 1.0):
            r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
                k_left, d_left, k_right, d_right, (1920, 1080), r, t.reshape(3, 1),
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha, newImageSize=(1920, 1080)
            )
            rect_left = cv2.undistortPoints(pixel_left, k_left, d_left, R=r1, P=p1).reshape(2)
            rect_right = cv2.undistortPoints(pixel_right, k_right, d_right, R=r2, P=p2).reshape(2)
            self.assertLess(abs(float(rect_left[1] - rect_right[1])), 1.0e-6)
            disparity = float(rect_left[0] - rect_right[0])
            baseline_m = abs(float(p2[0, 3] / p2[0, 0])) / 1000.0
            depths.append(float(p1[0, 0] * baseline_m / disparity))
        self.assertLess(abs(depths[0] - depths[1]), 1.0e-9)

    def test_frame_alignment(self) -> None:
        native = [
            {"frame_index": 0, "median_depth_m": 2.0},
            {"frame_index": 1, "median_depth_m": 2.1},
            {"frame_index": 2, "median_depth_m": 2.2},
            {"frame_index": 3, "median_depth_m": 2.3},
        ]
        custom = [
            {"frame_index": 0, "median_depth_m": 2.4},
            {"frame_index": 2, "median_depth_m": 2.5},
            {"frame_index": 3, "median_depth_m": 2.6},
        ]
        paired = pair_frame_records(native, custom)
        self.assertEqual([row["frame_index"] for row in paired], [0, 2, 3])
        self.assertEqual(paired[1]["native"]["median_depth_m"], 2.2)
        self.assertEqual(paired[1]["custom"]["median_depth_m"], 2.5)


if __name__ == "__main__":
    unittest.main()
