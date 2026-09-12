"""Explicit flat-port refractive stereo geometry.

This module is diagnostic-only.  It deliberately does not alter any of the
repository's production depth paths.  The coordinate convention is:

* each camera optical frame uses ``x=right, y=down, z=forward``;
* a ray direction points in its propagation direction;
* ``plane_normal_camera`` points from the preceding medium into the next
  medium (air -> glass -> water), so for a front-facing port it is ``+z``;
* the stereo common frame is the left camera frame;
* ``StereoExtrinsics`` stores a transform from right-camera coordinates to
  left-camera coordinates: ``X_left = R_right_to_left @ X_right + T``.

The public ray tracer accepts raw distorted pixels and first performs inverse
OpenCV distortion.  Rectified pixels are intentionally not accepted by the
physical tracer: their virtual-camera meaning must be supplied explicitly by
the caller before this module can be used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


EPS = 1.0e-12


class RefractiveGeometryError(ValueError):
    """Base error for invalid physical geometry or ray configuration."""


class RefractiveModelNotIdentifiable(RefractiveGeometryError):
    """Raised when a definitive flat-port ray cannot be computed."""


class TotalInternalReflectionError(RefractiveGeometryError):
    """Raised when Snell refraction has no transmitted ray."""


def unit_vector(value: Sequence[float] | np.ndarray, *, name: str = "vector") -> np.ndarray:
    """Return a finite unit vector, rejecting zero/non-finite inputs."""

    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != 3 or not np.isfinite(vector).all():
        raise RefractiveGeometryError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        raise RefractiveGeometryError(f"{name} must not be zero")
    return vector / norm


def refract_vector(
    incident_direction: Sequence[float] | np.ndarray,
    normal_from_medium1_to_medium2: Sequence[float] | np.ndarray,
    n1: float,
    n2: float,
) -> np.ndarray:
    """Apply vector Snell refraction for a transmitted ray.

    ``incident_direction`` points toward the interface.  The supplied normal
    must point from medium 1 into medium 2.  The function intentionally raises
    when the normal orientation is inconsistent instead of silently flipping
    it, and raises ``TotalInternalReflectionError`` when no transmitted ray
    exists.
    """

    incident = unit_vector(incident_direction, name="incident_direction")
    normal = unit_vector(
        normal_from_medium1_to_medium2,
        name="normal_from_medium1_to_medium2",
    )
    if not np.isfinite(n1) or not np.isfinite(n2) or n1 <= 0.0 or n2 <= 0.0:
        raise RefractiveGeometryError(f"refractive indices must be positive finite values: {n1}, {n2}")

    cos_incident = float(np.dot(incident, normal))
    if cos_incident <= EPS:
        raise RefractiveGeometryError(
            "surface normal must point from medium 1 into medium 2 and face the incident ray "
            f"(cos_incident={cos_incident:.16g})"
        )

    eta = float(n1 / n2)
    sin2_transmitted = eta * eta * max(0.0, 1.0 - cos_incident * cos_incident)
    if sin2_transmitted > 1.0 + 1.0e-12:
        raise TotalInternalReflectionError(
            f"total internal reflection: n1={n1}, n2={n2}, sin2_transmitted={sin2_transmitted}"
        )
    cos_transmitted = math_sqrt_clamped(1.0 - sin2_transmitted)
    tangential = incident - cos_incident * normal
    transmitted = eta * tangential + cos_transmitted * normal
    return unit_vector(transmitted, name="transmitted_direction")


def math_sqrt_clamped(value: float) -> float:
    if value < -1.0e-12:
        raise RefractiveGeometryError(f"negative square-root argument: {value}")
    return float(np.sqrt(max(0.0, value)))


@dataclass(frozen=True)
class FlatPortModel:
    """Physical parameters for a parallel air/glass/water flat port.

    ``None`` means the parameter is not measured/provided.  Such a model has
    status ``unknown`` and cannot produce a definitive refractive depth.
    The default normal is only a coordinate convention, not a claim about the
    installed housing.
    """

    n_air: float | None = None
    n_glass: float | None = None
    n_water: float | None = None
    camera_to_inner_interface_m: float | None = None
    glass_thickness_m: float | None = None
    plane_normal_camera: tuple[float, float, float] = (0.0, 0.0, 1.0)

    @property
    def missing_parameters(self) -> tuple[str, ...]:
        missing: list[str] = []
        for name in (
            "n_air",
            "n_glass",
            "n_water",
            "camera_to_inner_interface_m",
            "glass_thickness_m",
        ):
            value = getattr(self, name)
            if value is None or not np.isfinite(value) or value <= 0.0:
                missing.append(name)
        try:
            unit_vector(self.plane_normal_camera, name="plane_normal_camera")
        except RefractiveGeometryError:
            missing.append("plane_normal_camera")
        return tuple(missing)

    @property
    def status(self) -> str:
        return "known" if not self.missing_parameters else "unknown"

    def require_known(self) -> None:
        missing = self.missing_parameters
        if missing:
            raise RefractiveModelNotIdentifiable(
                "flat-port refractive model is not identifiable; missing/invalid parameters: "
                + ", ".join(missing)
            )


@dataclass(frozen=True)
class StereoExtrinsics:
    """Right-camera to left-camera rigid transform in metres."""

    rotation_right_to_left: np.ndarray
    translation_right_to_left_m: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation_right_to_left, dtype=np.float64)
        translation = np.asarray(self.translation_right_to_left_m, dtype=np.float64).reshape(-1)
        if rotation.shape != (3, 3) or translation.size != 3:
            raise RefractiveGeometryError("stereo extrinsics must be a 3x3 R and 3-vector T")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise RefractiveGeometryError("stereo extrinsics must be finite")
        if np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1.0e-6:
            raise RefractiveGeometryError("stereo R is not orthonormal")
        if abs(float(np.linalg.det(rotation)) - 1.0) > 1.0e-6:
            raise RefractiveGeometryError("stereo R determinant is not +1")

    @classmethod
    def from_opencv_left_to_right(
        cls,
        rotation_left_to_right: np.ndarray,
        translation_left_to_right_m: Sequence[float] | np.ndarray,
    ) -> "StereoExtrinsics":
        """Convert OpenCV ``X_right=R X_left+T`` to this class convention."""

        rotation = np.asarray(rotation_left_to_right, dtype=np.float64)
        translation = np.asarray(translation_left_to_right_m, dtype=np.float64).reshape(3)
        return cls(rotation.T, -rotation.T @ translation)

    def transform_ray_to_left(
        self,
        origin_right: Sequence[float] | np.ndarray,
        direction_right: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        origin = np.asarray(origin_right, dtype=np.float64).reshape(3)
        direction = unit_vector(direction_right, name="direction_right")
        return (
            self.rotation_right_to_left @ origin + self.translation_right_to_left_m,
            unit_vector(self.rotation_right_to_left @ direction, name="direction_left"),
        )


@dataclass(frozen=True)
class RefractedRay:
    """Ray and interface points in one camera's local optical frame."""

    air_direction_camera: np.ndarray
    inner_interface_point_camera: np.ndarray
    glass_direction_camera: np.ndarray
    outer_interface_point_camera: np.ndarray
    water_direction_camera: np.ndarray


@dataclass(frozen=True)
class TriangulationResult:
    """Closest-point solution for two rays in one common frame."""

    point_left: np.ndarray
    point_right: np.ndarray
    midpoint: np.ndarray
    ray_gap_m: float
    parameter_left_m: float
    parameter_right_m: float
    rank: int


def pixel_to_air_ray(
    pixel_uv: Sequence[float] | np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray | Sequence[float] | None,
) -> np.ndarray:
    """Recover an undistorted air-side ray from a raw pixel."""

    pixel = np.asarray(pixel_uv, dtype=np.float64).reshape(1, 1, 2)
    if pixel.shape != (1, 1, 2) or not np.isfinite(pixel).all():
        raise RefractiveGeometryError("pixel_uv must contain one finite (u, v) pair")
    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    d = None if distortion is None else np.asarray(distortion, dtype=np.float64).reshape(-1, 1)
    normalized = cv2.undistortPoints(pixel, k, d).reshape(2)
    return unit_vector([normalized[0], normalized[1], 1.0], name="air_ray")


def ray_plane_intersection(
    ray_origin: Sequence[float] | np.ndarray,
    ray_direction: Sequence[float] | np.ndarray,
    plane_point: Sequence[float] | np.ndarray,
    plane_normal: Sequence[float] | np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return ``(t, point)`` for ``ray_origin+t*ray_direction`` at a plane."""

    origin = np.asarray(ray_origin, dtype=np.float64).reshape(3)
    direction = unit_vector(ray_direction, name="ray_direction")
    point = np.asarray(plane_point, dtype=np.float64).reshape(3)
    normal = unit_vector(plane_normal, name="plane_normal")
    denominator = float(np.dot(normal, direction))
    if abs(denominator) <= EPS:
        raise RefractiveGeometryError("ray is parallel to plane")
    t = float(np.dot(normal, point - origin) / denominator)
    if t < -1.0e-10:
        raise RefractiveGeometryError(f"ray intersects the plane behind its origin: t={t}")
    t = max(0.0, t)
    return t, origin + t * direction


def trace_flat_port_ray(
    pixel_uv: Sequence[float] | np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray | Sequence[float] | None,
    model: FlatPortModel,
) -> RefractedRay:
    """Trace one raw pixel through parallel air/glass/water interfaces."""

    model.require_known()
    normal = unit_vector(model.plane_normal_camera, name="plane_normal_camera")
    air_direction = pixel_to_air_ray(pixel_uv, camera_matrix, distortion)
    camera_origin = np.zeros(3, dtype=np.float64)
    inner_plane_point = normal * float(model.camera_to_inner_interface_m)
    _, inner_point = ray_plane_intersection(
        camera_origin,
        air_direction,
        inner_plane_point,
        normal,
    )
    glass_direction = refract_vector(
        air_direction,
        normal,
        float(model.n_air),
        float(model.n_glass),
    )
    outer_plane_point = normal * (
        float(model.camera_to_inner_interface_m) + float(model.glass_thickness_m)
    )
    _, outer_point = ray_plane_intersection(
        inner_point,
        glass_direction,
        outer_plane_point,
        normal,
    )
    water_direction = refract_vector(
        glass_direction,
        normal,
        float(model.n_glass),
        float(model.n_water),
    )
    return RefractedRay(
        air_direction_camera=air_direction,
        inner_interface_point_camera=inner_point,
        glass_direction_camera=glass_direction,
        outer_interface_point_camera=outer_point,
        water_direction_camera=water_direction,
    )


def transform_local_ray_to_left(
    side: str,
    ray_origin_camera: Sequence[float] | np.ndarray,
    ray_direction_camera: Sequence[float] | np.ndarray,
    extrinsics: StereoExtrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a left/right local ray into the left-camera frame."""

    side_normalized = side.lower()
    if side_normalized == "left":
        return (
            np.asarray(ray_origin_camera, dtype=np.float64).reshape(3),
            unit_vector(ray_direction_camera, name="left_direction"),
        )
    if side_normalized == "right":
        return extrinsics.transform_ray_to_left(ray_origin_camera, ray_direction_camera)
    raise RefractiveGeometryError(f"side must be 'left' or 'right', got {side!r}")


def triangulate_two_rays(
    origin_left: Sequence[float] | np.ndarray,
    direction_left: Sequence[float] | np.ndarray,
    origin_right: Sequence[float] | np.ndarray,
    direction_right: Sequence[float] | np.ndarray,
) -> TriangulationResult:
    """Find the closest points on two 3-D rays using least squares."""

    o_left = np.asarray(origin_left, dtype=np.float64).reshape(3)
    o_right = np.asarray(origin_right, dtype=np.float64).reshape(3)
    d_left = unit_vector(direction_left, name="direction_left")
    d_right = unit_vector(direction_right, name="direction_right")
    matrix = np.column_stack((d_left, -d_right))
    rhs = o_right - o_left
    solution, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    t_left, t_right = float(solution[0]), float(solution[1])
    point_left = o_left + t_left * d_left
    point_right = o_right + t_right * d_right
    midpoint = 0.5 * (point_left + point_right)
    return TriangulationResult(
        point_left=point_left,
        point_right=point_right,
        midpoint=midpoint,
        ray_gap_m=float(np.linalg.norm(point_left - point_right)),
        parameter_left_m=t_left,
        parameter_right_m=t_right,
        rank=int(rank),
    )


def triangulation_depth_metrics(
    triangulation: TriangulationResult,
    left_outer_port_origin_left: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Return three explicitly different depth/distance definitions."""

    midpoint = triangulation.midpoint
    left_port = np.asarray(left_outer_port_origin_left, dtype=np.float64).reshape(3)
    return {
        "z_left_camera_m": float(midpoint[2]),
        "euclidean_distance_from_left_camera_m": float(np.linalg.norm(midpoint)),
        "distance_from_left_outer_port_m": float(np.linalg.norm(midpoint - left_port)),
        "ray_gap_m": float(triangulation.ray_gap_m),
    }


def refractive_stereo_correspondence(
    left_pixel_uv: Sequence[float] | np.ndarray,
    right_pixel_uv: Sequence[float] | np.ndarray,
    left_camera_matrix: np.ndarray,
    left_distortion: np.ndarray | Sequence[float] | None,
    right_camera_matrix: np.ndarray,
    right_distortion: np.ndarray | Sequence[float] | None,
    left_port_model: FlatPortModel,
    right_port_model: FlatPortModel,
    extrinsics: StereoExtrinsics,
) -> dict[str, Any]:
    """Trace two raw pixels and triangulate their refracted water rays."""

    left_ray = trace_flat_port_ray(left_pixel_uv, left_camera_matrix, left_distortion, left_port_model)
    right_ray = trace_flat_port_ray(right_pixel_uv, right_camera_matrix, right_distortion, right_port_model)
    left_origin, left_direction = transform_local_ray_to_left(
        "left",
        left_ray.outer_interface_point_camera,
        left_ray.water_direction_camera,
        extrinsics,
    )
    right_origin, right_direction = transform_local_ray_to_left(
        "right",
        right_ray.outer_interface_point_camera,
        right_ray.water_direction_camera,
        extrinsics,
    )
    triangulation = triangulate_two_rays(
        left_origin,
        left_direction,
        right_origin,
        right_direction,
    )
    metrics = triangulation_depth_metrics(triangulation, left_origin)
    view_angle = float(np.degrees(np.arccos(np.clip(left_direction[2], -1.0, 1.0))))
    return {
        "left_ray_origin_camera_m": left_ray.outer_interface_point_camera.tolist(),
        "left_ray_direction_water_camera": left_ray.water_direction_camera.tolist(),
        "right_ray_origin_camera_m": right_ray.outer_interface_point_camera.tolist(),
        "right_ray_direction_water_camera": right_ray.water_direction_camera.tolist(),
        "left_ray_origin_left_frame_m": left_origin.tolist(),
        "right_ray_origin_left_frame_m": right_origin.tolist(),
        "left_ray_direction_left_frame": left_direction.tolist(),
        "right_ray_direction_left_frame": right_direction.tolist(),
        "point_left_m": triangulation.point_left.tolist(),
        "point_right_m": triangulation.point_right.tolist(),
        "midpoint_left_frame_m": triangulation.midpoint.tolist(),
        "parameter_left_m": triangulation.parameter_left_m,
        "parameter_right_m": triangulation.parameter_right_m,
        "triangulation_rank": triangulation.rank,
        "view_angle_from_left_optical_axis_deg": view_angle,
        **metrics,
    }


def pair_frame_records(
    native_records: Iterable[dict[str, Any]],
    custom_records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Inner-join records by explicit frame index, never by list position."""

    native_by_frame = {int(record["frame_index"]): record for record in native_records}
    custom_by_frame = {int(record["frame_index"]): record for record in custom_records}
    common = sorted(set(native_by_frame) & set(custom_by_frame))
    return [
        {
            "frame_index": frame_index,
            "native": native_by_frame[frame_index],
            "custom": custom_by_frame[frame_index],
        }
        for frame_index in common
    ]
