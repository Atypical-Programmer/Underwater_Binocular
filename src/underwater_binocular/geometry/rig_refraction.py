"""Shared-plane rig-level refractive geometry, independent of ZED."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from .refraction import (
    RefractiveGeometryError,
    RefractiveModelNotIdentifiable,
    refract_vector,
    unit_vector,
)


@dataclass(frozen=True)
class RigCameraPose:
    """Camera pose ``X_rig = R_rig_from_camera @ X_camera + C_rig``."""

    rotation_rig_from_camera: np.ndarray
    center_rig_m: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation_rig_from_camera, dtype=np.float64)
        center = np.asarray(self.center_rig_m, dtype=np.float64).reshape(-1)
        if rotation.shape != (3, 3) or center.size != 3 or not np.isfinite(rotation).all() or not np.isfinite(center).all():
            raise RefractiveGeometryError("rig pose must contain finite R(3x3) and center(3)")
        if np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1.0e-6 or abs(float(np.linalg.det(rotation)) - 1.0) > 1.0e-6:
            raise RefractiveGeometryError("rig pose rotation is not a proper rotation")

    def direction_to_rig(self, direction_camera: Sequence[float] | np.ndarray) -> np.ndarray:
        return unit_vector(self.rotation_rig_from_camera @ unit_vector(direction_camera, name="direction_camera"), name="direction_rig")

    def point_to_rig(self, point_camera: Sequence[float] | np.ndarray) -> np.ndarray:
        return self.rotation_rig_from_camera @ np.asarray(point_camera, dtype=np.float64).reshape(3) + np.asarray(self.center_rig_m, dtype=np.float64).reshape(3)

    def direction_to_camera(self, direction_rig: Sequence[float] | np.ndarray) -> np.ndarray:
        return unit_vector(self.rotation_rig_from_camera.T @ unit_vector(direction_rig, name="direction_rig"), name="direction_camera")


@dataclass(frozen=True)
class RigPlane:
    """A physical plane in the common rig frame."""

    normal_rig: np.ndarray
    point_rig: np.ndarray

    def __post_init__(self) -> None:
        unit_vector(self.normal_rig, name="plane.normal_rig")
        point = np.asarray(self.point_rig, dtype=np.float64).reshape(-1)
        if point.size != 3 or not np.isfinite(point).all():
            raise RefractiveGeometryError("plane.point_rig must be finite 3-vector")

    @property
    def normal(self) -> np.ndarray:
        return unit_vector(self.normal_rig, name="plane.normal_rig")

    @property
    def point(self) -> np.ndarray:
        return np.asarray(self.point_rig, dtype=np.float64).reshape(3)


@dataclass(frozen=True)
class RigFlatPortModel:
    """Shared or separate planar-port model; dome geometry is never approximated."""

    n_air: float | None
    n_glass: float | None
    n_water: float | None
    inner_plane_shared: RigPlane | None = None
    outer_plane_shared: RigPlane | None = None
    inner_plane_left: RigPlane | None = None
    outer_plane_left: RigPlane | None = None
    inner_plane_right: RigPlane | None = None
    outer_plane_right: RigPlane | None = None
    mode: str = "shared_flat"

    @property
    def status(self) -> str:
        if self.mode == "dome":
            return "invalid_flat_port_model_for_dome"
        if any(value is None or not np.isfinite(value) or value <= 0.0 for value in (self.n_air, self.n_glass, self.n_water)):
            return "unknown"
        if self.mode == "shared_flat":
            return "known" if self.inner_plane_shared is not None and self.outer_plane_shared is not None else "unknown"
        if self.mode == "separate_flat":
            return "known" if all(value is not None for value in (self.inner_plane_left, self.outer_plane_left, self.inner_plane_right, self.outer_plane_right)) else "unknown"
        return "unsupported_port_geometry"

    @classmethod
    def shared_parallel(cls, n_air: float, n_glass: float, n_water: float, normal_rig: Sequence[float], inner_distance_m: float, glass_thickness_m: float) -> RigFlatPortModel:
        normal = unit_vector(normal_rig, name="normal_rig")
        if inner_distance_m <= 0.0 or glass_thickness_m <= 0.0:
            raise RefractiveGeometryError("inner distance and glass thickness must be positive")
        return cls(n_air, n_glass, n_water, RigPlane(normal, normal * inner_distance_m), RigPlane(normal, normal * (inner_distance_m + glass_thickness_m)))

    def planes_for(self, side: str) -> tuple[RigPlane, RigPlane]:
        if self.mode == "dome":
            raise RefractiveGeometryError("dome geometry requires a sphere model")
        if self.status != "known":
            raise RefractiveModelNotIdentifiable(f"rig port model status is {self.status}")
        if self.mode == "shared_flat":
            return self.inner_plane_shared, self.outer_plane_shared  # type: ignore[return-value]
        if side.lower() == "left":
            return self.inner_plane_left, self.outer_plane_left  # type: ignore[return-value]
        if side.lower() == "right":
            return self.inner_plane_right, self.outer_plane_right  # type: ignore[return-value]
        raise RefractiveGeometryError(f"unknown camera side: {side}")


@dataclass(frozen=True)
class RigWaterRay:
    """A traced ray in the common rig frame."""

    side: str
    air_direction_rig: np.ndarray
    inner_point_rig: np.ndarray
    glass_direction_rig: np.ndarray
    outer_point_rig: np.ndarray
    water_direction_rig: np.ndarray


def _intersect_plane(origin: np.ndarray, direction: np.ndarray, plane: RigPlane) -> np.ndarray:
    ray_direction = unit_vector(direction, name="ray_direction")
    denominator = float(np.dot(plane.normal, ray_direction))
    if abs(denominator) <= 1.0e-12:
        raise RefractiveGeometryError("ray is parallel to rig plane")
    t = float(np.dot(plane.normal, plane.point - origin) / denominator)
    if t < -1.0e-10:
        raise RefractiveGeometryError(f"ray intersects rig plane behind origin: t={t}")
    return origin + max(0.0, t) * ray_direction


def trace_rig_pixel(side: str, pixel_uv: Sequence[float] | np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray | Sequence[float] | None, camera_pose: RigCameraPose, port_model: RigFlatPortModel) -> RigWaterRay:
    """Trace one raw pixel through the physically shared rig planes."""

    if port_model.status != "known":
        raise RefractiveModelNotIdentifiable(f"rig port model is not known: {port_model.status}")
    inner_plane, outer_plane = port_model.planes_for(side)
    from .refraction import pixel_to_air_ray

    air_camera = pixel_to_air_ray(pixel_uv, camera_matrix, distortion)
    air_rig = camera_pose.direction_to_rig(air_camera)
    camera_origin = np.asarray(camera_pose.center_rig_m, dtype=np.float64).reshape(3)
    inner_point = _intersect_plane(camera_origin, air_rig, inner_plane)
    glass = refract_vector(air_rig, inner_plane.normal, float(port_model.n_air), float(port_model.n_glass))
    outer_point = _intersect_plane(inner_point, glass, outer_plane)
    water = refract_vector(glass, outer_plane.normal, float(port_model.n_glass), float(port_model.n_water))
    return RigWaterRay(side, air_rig, inner_point, glass, outer_point, water)


def triangulate_rig_water_rays(left_ray: RigWaterRay, right_ray: RigWaterRay) -> dict[str, float | np.ndarray]:
    """Return closest-point water-ray triangulation and conditioning metrics."""

    left_direction = unit_vector(left_ray.water_direction_rig, name="left_water_direction")
    right_direction = unit_vector(right_ray.water_direction_rig, name="right_water_direction")
    matrix = np.column_stack((left_direction, -right_direction))
    solution, _, rank, singular = np.linalg.lstsq(matrix, right_ray.outer_point_rig - left_ray.outer_point_rig, rcond=None)
    t_left, t_right = float(solution[0]), float(solution[1])
    point_left = left_ray.outer_point_rig + t_left * left_direction
    point_right = right_ray.outer_point_rig + t_right * right_direction
    return {
        "point_left_rig_m": point_left,
        "point_right_rig_m": point_right,
        "midpoint_rig_m": 0.5 * (point_left + point_right),
        "ray_gap_m": float(np.linalg.norm(point_left - point_right)),
        "t_left_m": t_left,
        "t_right_m": t_right,
        "rank": int(rank),
        "ray_angle_deg": float(np.degrees(np.arccos(np.clip(np.dot(left_direction, right_direction), -1.0, 1.0)))),
        "condition_number": float(np.inf if len(singular) < 2 or singular[-1] <= 1.0e-15 else singular[0] / singular[-1]),
        "positive_parameters": bool(t_left > 0.0 and t_right > 0.0),
    }


def _air_residual_for_outer_point(target_rig: np.ndarray, outer_point: np.ndarray, side: str, camera_pose: RigCameraPose, port_model: RigFlatPortModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inner_plane, outer_plane = port_model.planes_for(side)
    water_to_glass = unit_vector(outer_point - target_rig, name="water_to_glass")
    glass_to_air = refract_vector(water_to_glass, -outer_plane.normal, float(port_model.n_water), float(port_model.n_glass))
    inner_point = _intersect_plane(outer_point, glass_to_air, inner_plane)
    air_to_camera = refract_vector(glass_to_air, -inner_plane.normal, float(port_model.n_glass), float(port_model.n_air))
    camera_center = np.asarray(camera_pose.center_rig_m, dtype=np.float64).reshape(3)
    parameter = float(np.dot(camera_center - inner_point, air_to_camera))
    closest_camera = inner_point + parameter * air_to_camera
    if parameter <= 0.0:
        raise RefractiveGeometryError("backward air ray does not point toward camera")
    return camera_center - closest_camera, inner_point, air_to_camera


def project_water_point_to_raw_pixel(side: str, water_point_rig: Sequence[float] | np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray | Sequence[float] | None, camera_pose: RigCameraPose, port_model: RigFlatPortModel, initial_pixel_uv: Sequence[float] | None = None, max_iterations: int = 40, tolerance_m: float = 1.0e-10) -> dict[str, object]:
    """Numerically forward-project a water point through both port interfaces."""

    if port_model.status != "known":
        raise RefractiveModelNotIdentifiable(f"rig port model is not known: {port_model.status}")
    target = np.asarray(water_point_rig, dtype=np.float64).reshape(3)
    _inner_plane, outer_plane = port_model.planes_for(side)
    normal = outer_plane.normal
    camera_center = np.asarray(camera_pose.center_rig_m, dtype=np.float64).reshape(3)
    if initial_pixel_uv is None:
        target_camera = camera_pose.rotation_rig_from_camera.T @ (target - camera_center)
        if target_camera[2] <= 0.0:
            raise RefractiveGeometryError("water point is behind camera")
        k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        initial_pixel_uv = (float(k[0, 0] * target_camera[0] / target_camera[2] + k[0, 2]), float(k[1, 1] * target_camera[1] / target_camera[2] + k[1, 2]))
    reference = target - camera_center
    denominator = float(np.dot(normal, reference))
    if abs(denominator) <= 1.0e-12:
        raise RefractiveGeometryError("target ray is parallel to outer plane")
    outer_point = camera_center + reference * float(np.dot(normal, outer_plane.point - camera_center) / denominator)
    axis_seed = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    plane_axis_1 = unit_vector(np.cross(normal, axis_seed), name="plane_axis_1")
    plane_axis_2 = unit_vector(np.cross(normal, plane_axis_1), name="plane_axis_2")

    for iteration in range(max_iterations):
        residual, inner_point, air_to_camera = _air_residual_for_outer_point(target, outer_point, side, camera_pose, port_model)
        if np.linalg.norm(residual) <= tolerance_m:
            air_forward_rig = -air_to_camera
            air_forward_camera = camera_pose.direction_to_camera(air_forward_rig)
            if air_forward_camera[2] <= 0.0:
                raise RefractiveGeometryError("forward air ray points behind camera")
            distorted, _ = cv2.projectPoints(air_forward_camera.reshape(1, 1, 3), np.zeros(3), np.zeros(3), np.asarray(camera_matrix, dtype=np.float64), None if distortion is None else np.asarray(distortion, dtype=np.float64).reshape(-1, 1))
            return {"pixel_uv": distorted.reshape(2), "outer_point_rig_m": outer_point, "inner_point_rig_m": inner_point, "air_direction_forward_rig": air_forward_rig, "air_direction_forward_camera": air_forward_camera, "residual_m": float(np.linalg.norm(residual)), "iterations": iteration}
        step = max(1.0e-7, min(1.0e-4, np.linalg.norm(target - outer_point) * 1.0e-5))
        residual_1, _, _ = _air_residual_for_outer_point(target, outer_point + step * plane_axis_1, side, camera_pose, port_model)
        residual_2, _, _ = _air_residual_for_outer_point(target, outer_point + step * plane_axis_2, side, camera_pose, port_model)
        basis_seed = np.array([1.0, 0.0, 0.0]) if abs(air_to_camera[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        residual_axis_1 = unit_vector(np.cross(air_to_camera, basis_seed), name="residual_axis_1")
        residual_axis_2 = unit_vector(np.cross(air_to_camera, residual_axis_1), name="residual_axis_2")
        base_2d = np.asarray(
            [np.dot(residual, residual_axis_1), np.dot(residual, residual_axis_2)]
        )
        residual_1_2d = np.asarray(
            [np.dot(residual_1, residual_axis_1), np.dot(residual_1, residual_axis_2)]
        )
        residual_2_2d = np.asarray(
            [np.dot(residual_2, residual_axis_1), np.dot(residual_2, residual_axis_2)]
        )
        jacobian = np.column_stack(
            ((residual_1_2d - base_2d) / step, (residual_2_2d - base_2d) / step)
        )
        delta, _, rank, _ = np.linalg.lstsq(jacobian, -base_2d, rcond=None)
        if rank < 2 or not np.isfinite(delta).all():
            raise RefractiveGeometryError("forward refractive projection Jacobian is degenerate")
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > 0.05:
            delta *= 0.05 / delta_norm
        outer_point = outer_point + float(delta[0]) * plane_axis_1 + float(delta[1]) * plane_axis_2
    raise RefractiveGeometryError("forward refractive projection did not converge")
