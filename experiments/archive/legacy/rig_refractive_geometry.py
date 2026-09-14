"""Rig-level shared-plane refractive stereo geometry.

This is an audit-only extension of :mod:`refractive_geometry`.  Unlike the
camera-local helper, this module places both cameras and a shared flat pane in
one rig frame.  It therefore does not silently assume that the left and right
cameras have identical local pane distances or normals.

Rig convention:

* ``x=right, y=down, z=forward`` for each camera optical frame;
* ``X_rig = R_rig_from_camera @ X_camera + C_rig``;
* the left camera frame is used as the default rig frame in the diagnostics;
* a shared flat pane has one physical inner and outer plane for both cameras;
* a dome is explicitly unsupported by the flat-plane tracer and returns an
  invalid-model status rather than being approximated as a plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from refractive_geometry import (
    RefractiveGeometryError,
    RefractiveModelNotIdentifiable,
    TotalInternalReflectionError,
    pixel_to_air_ray,
    refract_vector,
    unit_vector,
)


@dataclass(frozen=True)
class RigCameraPose:
    """Camera pose in the common rig frame."""

    rotation_rig_from_camera: np.ndarray
    center_rig_m: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation_rig_from_camera, dtype=np.float64)
        center = np.asarray(self.center_rig_m, dtype=np.float64).reshape(-1)
        if rotation.shape != (3, 3) or center.size != 3:
            raise RefractiveGeometryError("rig pose must contain a 3x3 rotation and 3-vector center")
        if not np.isfinite(rotation).all() or not np.isfinite(center).all():
            raise RefractiveGeometryError("rig pose must be finite")
        if np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1.0e-6:
            raise RefractiveGeometryError("rig pose rotation is not orthonormal")
        if abs(float(np.linalg.det(rotation)) - 1.0) > 1.0e-6:
            raise RefractiveGeometryError("rig pose rotation determinant is not +1")

    def direction_to_rig(self, direction_camera: Sequence[float] | np.ndarray) -> np.ndarray:
        return unit_vector(
            self.rotation_rig_from_camera @ unit_vector(direction_camera, name="direction_camera"),
            name="direction_rig",
        )

    def point_to_rig(self, point_camera: Sequence[float] | np.ndarray) -> np.ndarray:
        return self.rotation_rig_from_camera @ np.asarray(point_camera, dtype=np.float64).reshape(3) + self.center_rig_m

    def direction_to_camera(self, direction_rig: Sequence[float] | np.ndarray) -> np.ndarray:
        return unit_vector(
            self.rotation_rig_from_camera.T @ unit_vector(direction_rig, name="direction_rig"),
            name="direction_camera",
        )


@dataclass(frozen=True)
class RigPlane:
    normal_rig: np.ndarray
    point_rig: np.ndarray

    def __post_init__(self) -> None:
        unit_vector(self.normal_rig, name="plane.normal_rig")
        point = np.asarray(self.point_rig, dtype=np.float64).reshape(-1)
        if point.size != 3 or not np.isfinite(point).all():
            raise RefractiveGeometryError("plane.point_rig must be a finite 3-vector")

    @property
    def normal(self) -> np.ndarray:
        return unit_vector(self.normal_rig, name="plane.normal_rig")

    @property
    def point(self) -> np.ndarray:
        return np.asarray(self.point_rig, dtype=np.float64).reshape(3)


@dataclass(frozen=True)
class RigFlatPortModel:
    """Shared/separate planar housing model; no dome approximation is implicit."""

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
        required_indices = (self.n_air, self.n_glass, self.n_water)
        if any(value is None or not np.isfinite(value) or value <= 0.0 for value in required_indices):
            return "unknown"
        if self.mode == "shared_flat":
            return "known" if self.inner_plane_shared is not None and self.outer_plane_shared is not None else "unknown"
        if self.mode == "separate_flat":
            complete = all(
                plane is not None
                for plane in (
                    self.inner_plane_left,
                    self.outer_plane_left,
                    self.inner_plane_right,
                    self.outer_plane_right,
                )
            )
            return "known" if complete else "unknown"
        return "unsupported_port_geometry"

    @classmethod
    def shared_parallel(
        cls,
        n_air: float,
        n_glass: float,
        n_water: float,
        normal_rig: Sequence[float],
        inner_distance_m: float,
        glass_thickness_m: float,
    ) -> "RigFlatPortModel":
        normal = unit_vector(normal_rig, name="normal_rig")
        if inner_distance_m <= 0.0 or glass_thickness_m <= 0.0:
            raise RefractiveGeometryError("inner distance and glass thickness must be positive")
        inner = RigPlane(normal, normal * inner_distance_m)
        outer = RigPlane(normal, normal * (inner_distance_m + glass_thickness_m))
        return cls(n_air, n_glass, n_water, inner_plane_shared=inner, outer_plane_shared=outer)

    def planes_for(self, side: str) -> tuple[RigPlane, RigPlane]:
        if self.mode == "dome":
            raise RefractiveGeometryError("FLAT_PORT_MODEL_INVALID_FOR_THIS_HOUSING: dome geometry requires a sphere model")
        if self.status not in ("known",):
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
    side: str
    air_direction_rig: np.ndarray
    inner_point_rig: np.ndarray
    glass_direction_rig: np.ndarray
    outer_point_rig: np.ndarray
    water_direction_rig: np.ndarray


def _intersect_plane(
    origin: np.ndarray,
    direction: np.ndarray,
    plane: RigPlane,
) -> np.ndarray:
    normal = plane.normal
    ray_direction = unit_vector(direction, name="ray_direction")
    denominator = float(np.dot(normal, ray_direction))
    if abs(denominator) <= 1.0e-12:
        raise RefractiveGeometryError("ray is parallel to rig plane")
    t = float(np.dot(normal, plane.point - origin) / denominator)
    if t < -1.0e-10:
        raise RefractiveGeometryError(f"ray intersects rig plane behind origin: t={t}")
    return origin + max(0.0, t) * ray_direction


def trace_rig_pixel(
    side: str,
    pixel_uv: Sequence[float] | np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray | Sequence[float] | None,
    camera_pose: RigCameraPose,
    port_model: RigFlatPortModel,
) -> RigWaterRay:
    """Trace one raw pixel through the physically shared rig-level planes."""

    if port_model.status != "known":
        raise RefractiveModelNotIdentifiable(f"rig port model is not known: {port_model.status}")
    inner_plane, outer_plane = port_model.planes_for(side)
    air_direction_camera = pixel_to_air_ray(pixel_uv, camera_matrix, distortion)
    camera_origin_rig = np.asarray(camera_pose.center_rig_m, dtype=np.float64).reshape(3)
    air_direction_rig = camera_pose.direction_to_rig(air_direction_camera)
    inner_point = _intersect_plane(camera_origin_rig, air_direction_rig, inner_plane)
    glass_direction = refract_vector(
        air_direction_rig,
        inner_plane.normal,
        float(port_model.n_air),
        float(port_model.n_glass),
    )
    outer_point = _intersect_plane(inner_point, glass_direction, outer_plane)
    water_direction = refract_vector(
        glass_direction,
        outer_plane.normal,
        float(port_model.n_glass),
        float(port_model.n_water),
    )
    return RigWaterRay(
        side=side,
        air_direction_rig=air_direction_rig,
        inner_point_rig=inner_point,
        glass_direction_rig=glass_direction,
        outer_point_rig=outer_point,
        water_direction_rig=water_direction,
    )


def triangulate_rig_water_rays(
    left_ray: RigWaterRay,
    right_ray: RigWaterRay,
) -> dict[str, float | np.ndarray]:
    """Closest-point triangulation and conditioning metrics for water rays."""

    d_left = unit_vector(left_ray.water_direction_rig, name="left_water_direction")
    d_right = unit_vector(right_ray.water_direction_rig, name="right_water_direction")
    o_left = left_ray.outer_point_rig
    o_right = right_ray.outer_point_rig
    matrix = np.column_stack((d_left, -d_right))
    solution, _, rank, singular_values = np.linalg.lstsq(matrix, o_right - o_left, rcond=None)
    t_left, t_right = float(solution[0]), float(solution[1])
    point_left = o_left + t_left * d_left
    point_right = o_right + t_right * d_right
    midpoint = 0.5 * (point_left + point_right)
    ray_angle = float(np.degrees(np.arccos(np.clip(np.dot(d_left, d_right), -1.0, 1.0))))
    condition = float(np.inf if len(singular_values) < 2 or singular_values[-1] <= 1.0e-15 else singular_values[0] / singular_values[-1])
    return {
        "point_left_rig_m": point_left,
        "point_right_rig_m": point_right,
        "midpoint_rig_m": midpoint,
        "ray_gap_m": float(np.linalg.norm(point_left - point_right)),
        "t_left_m": t_left,
        "t_right_m": t_right,
        "rank": int(rank),
        "ray_angle_deg": ray_angle,
        "condition_number": condition,
        "positive_parameters": bool(t_left > 0.0 and t_right > 0.0),
    }


def _air_residual_for_outer_point(
    target_rig: np.ndarray,
    outer_point: np.ndarray,
    side: str,
    camera_pose: RigCameraPose,
    port_model: RigFlatPortModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace a water point backwards to the camera side through both interfaces."""

    inner_plane, outer_plane = port_model.planes_for(side)
    water_to_glass = unit_vector(outer_point - target_rig, name="water_to_glass")
    glass_to_air = refract_vector(
        water_to_glass,
        -outer_plane.normal,
        float(port_model.n_water),
        float(port_model.n_glass),
    )
    inner_point = _intersect_plane(outer_point, glass_to_air, inner_plane)
    air_to_camera = refract_vector(
        glass_to_air,
        -inner_plane.normal,
        float(port_model.n_glass),
        float(port_model.n_air),
    )
    camera_center = np.asarray(camera_pose.center_rig_m, dtype=np.float64).reshape(3)
    parameter = float(np.dot(camera_center - inner_point, air_to_camera))
    closest_camera = inner_point + parameter * air_to_camera
    if parameter <= 0.0:
        raise RefractiveGeometryError("backward air ray does not point toward camera")
    error = camera_center - closest_camera
    return error, inner_point, air_to_camera


def project_water_point_to_raw_pixel(
    side: str,
    water_point_rig: Sequence[float] | np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray | Sequence[float] | None,
    camera_pose: RigCameraPose,
    port_model: RigFlatPortModel,
    initial_pixel_uv: Sequence[float] | None = None,
    max_iterations: int = 40,
    tolerance_m: float = 1.0e-10,
) -> dict[str, object]:
    """Forward-project a water point by solving the two-interface ray path.

    The unknown is the point on the outer physical plane.  At each iteration
    the water→glass→air path is traced backwards; the outer point is adjusted
    until the back-traced air ray passes through the camera center.  The final
    air ray is reversed and distorted with the camera K/D to produce a raw
    pixel.  This is a numerical forward projection, not a pinhole shortcut.
    """

    if port_model.status != "known":
        raise RefractiveModelNotIdentifiable(f"rig port model is not known: {port_model.status}")
    target = np.asarray(water_point_rig, dtype=np.float64).reshape(3)
    inner_plane, outer_plane = port_model.planes_for(side)
    normal = outer_plane.normal
    camera_center = np.asarray(camera_pose.center_rig_m, dtype=np.float64).reshape(3)
    if initial_pixel_uv is None:
        target_camera = camera_pose.rotation_rig_from_camera.T @ (target - camera_center)
        if target_camera[2] <= 0.0:
            raise RefractiveGeometryError("water point is behind camera")
        k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        initial_pixel_uv = (
            float(k[0, 0] * target_camera[0] / target_camera[2] + k[0, 2]),
            float(k[1, 1] * target_camera[1] / target_camera[2] + k[1, 2]),
        )
    reference = target - camera_center
    denominator = float(np.dot(normal, reference))
    if abs(denominator) <= 1.0e-12:
        raise RefractiveGeometryError("initial target ray is parallel to outer plane")
    outer_point = camera_center + reference * float(np.dot(normal, outer_plane.point - camera_center) / denominator)
    # Construct two orthonormal axes on the outer plane for the optimization.
    axis_seed = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    plane_axis_1 = unit_vector(np.cross(normal, axis_seed), name="plane_axis_1")
    plane_axis_2 = unit_vector(np.cross(normal, plane_axis_1), name="plane_axis_2")

    def residual_at(point: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _air_residual_for_outer_point(target, point, side, camera_pose, port_model)

    for iteration in range(max_iterations):
        residual, inner_point, air_to_camera = residual_at(outer_point)
        if np.linalg.norm(residual) <= tolerance_m:
            air_forward_rig = -air_to_camera
            air_forward_camera = camera_pose.direction_to_camera(air_forward_rig)
            if air_forward_camera[2] <= 0.0:
                raise RefractiveGeometryError("forward air ray points behind camera")
            distorted, _ = cv2.projectPoints(
                air_forward_camera.reshape(1, 1, 3),
                np.zeros(3),
                np.zeros(3),
                np.asarray(camera_matrix, dtype=np.float64),
                None if distortion is None else np.asarray(distortion, dtype=np.float64).reshape(-1, 1),
            )
            pixel = distorted.reshape(2)
            return {
                "pixel_uv": pixel,
                "outer_point_rig_m": outer_point,
                "inner_point_rig_m": inner_point,
                "air_direction_forward_rig": air_forward_rig,
                "air_direction_forward_camera": air_forward_camera,
                "residual_m": float(np.linalg.norm(residual)),
                "iterations": iteration,
            }
        step = max(1.0e-7, min(1.0e-4, np.linalg.norm(target - outer_point) * 1.0e-5))
        residual_1, _, _ = residual_at(outer_point + step * plane_axis_1)
        residual_2, _, _ = residual_at(outer_point + step * plane_axis_2)
        # Use the two plane-orthogonal residual components relative to the
        # current air ray; this avoids an arbitrary rig-axis projection.
        basis_seed = np.array([1.0, 0.0, 0.0]) if abs(air_to_camera[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        residual_axis_1 = unit_vector(np.cross(air_to_camera, basis_seed), name="residual_axis_1")
        residual_axis_2 = unit_vector(np.cross(air_to_camera, residual_axis_1), name="residual_axis_2")
        project_residual = lambda value: np.array([np.dot(value, residual_axis_1), np.dot(value, residual_axis_2)])
        base_2d = project_residual(residual)
        jacobian = np.column_stack(
            (
                (project_residual(residual_1) - base_2d) / step,
                (project_residual(residual_2) - base_2d) / step,
            )
        )
        delta, _, rank, _ = np.linalg.lstsq(jacobian, -base_2d, rcond=None)
        if rank < 2 or not np.isfinite(delta).all():
            raise RefractiveGeometryError("forward refractive projection Jacobian is degenerate")
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > 0.05:
            delta *= 0.05 / delta_norm
        outer_point = outer_point + float(delta[0]) * plane_axis_1 + float(delta[1]) * plane_axis_2
    raise RefractiveGeometryError("forward refractive projection did not converge")
