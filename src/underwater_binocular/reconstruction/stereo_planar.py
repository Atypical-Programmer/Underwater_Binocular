"""Calibrated stereo reconstruction for scenes dominated by one plane.

COLMAP's ordinary incremental mapper estimates camera motion from a single
image pair.  That is under-constrained for a nearly planar scene.  A calibrated
binocular sequence has an additional measurement: every synchronized pair
provides metric 3-D points.  This module uses those points to estimate the
rigid motion between consecutive selected frames and writes a complete COLMAP
model without asking the monocular initializer to choose a homography pose.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..calibration.models import StereoCalibration
from .stereo_seed import _PAIR_ID_MAX, _camera_line, _read_keypoints

_IMAGE_RE = re.compile(r"^(left|right)/[^/]*_(\d+)\.png$", re.IGNORECASE)


@dataclass
class _StereoPoint:
    left_keypoint: int
    right_keypoint: int
    xyz_left: np.ndarray
    reprojection_error: float


@dataclass
class _StereoFrame:
    frame: int
    left_image_id: int
    right_image_id: int
    left_name: str
    right_name: str
    left_camera_id: int
    right_camera_id: int
    left_keypoints: np.ndarray
    right_keypoints: np.ndarray
    points: dict[int, _StereoPoint]


@dataclass
class _RigidMotion:
    rotation_next_from_current: np.ndarray
    translation_next_from_current: np.ndarray
    inlier_match_indices: np.ndarray
    median_error_m: float


def _pair_id(image_id1: int, image_id2: int) -> int:
    first, second = sorted((int(image_id1), int(image_id2)))
    return first * _PAIR_ID_MAX + second


def _image_records(connection: sqlite3.Connection) -> dict[tuple[str, int], tuple[int, str, int]]:
    records: dict[tuple[str, int], tuple[int, str, int]] = {}
    for image_id, name, camera_id in connection.execute(
        "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
    ):
        match = _IMAGE_RE.match(str(name).replace("\\", "/"))
        if match is None:
            continue
        key = (match.group(1).lower(), int(match.group(2)))
        if key in records:
            raise RuntimeError(f"duplicate image identity in COLMAP database: {key}")
        records[key] = (int(image_id), str(name), int(camera_id))
    return records


def _raw_matches(connection: sqlite3.Connection, image_id1: int, image_id2: int) -> np.ndarray:
    row = connection.execute(
        "SELECT rows, data FROM matches WHERE pair_id = ?",
        (_pair_id(image_id1, image_id2),),
    ).fetchone()
    if row is None:
        return np.empty((0, 2), dtype=np.int64)
    rows, blob = int(row[0]), row[1]
    values = np.frombuffer(blob, dtype=np.uint32)
    if values.size != rows * 2:
        raise RuntimeError(f"invalid raw match blob for pair {image_id1}-{image_id2}")
    matches = values.reshape(rows, 2).astype(np.int64, copy=False)
    if image_id1 > image_id2:
        matches = matches[:, ::-1]
    return matches


def _triangulate_frame(
    connection: sqlite3.Connection,
    *,
    left_image_id: int,
    right_image_id: int,
    calibration: StereoCalibration,
    max_reprojection_error: float,
) -> dict[int, _StereoPoint]:
    import cv2

    matches = _raw_matches(connection, left_image_id, right_image_id)
    if len(matches) == 0:
        raise RuntimeError(f"no synchronized stereo matches for {left_image_id}-{right_image_id}")
    left_keypoints = _read_keypoints(connection, left_image_id)
    right_keypoints = _read_keypoints(connection, right_image_id)
    if np.any(matches[:, 0] >= len(left_keypoints)) or np.any(matches[:, 1] >= len(right_keypoints)):
        raise RuntimeError("stereo match index exceeds the keypoint table")
    left_points = left_keypoints[matches[:, 0]]
    right_points = right_keypoints[matches[:, 1]]
    left_camera = calibration.left
    right_camera = calibration.right
    left_distortion = np.asarray(left_camera.distortion[:5], dtype=np.float64)
    right_distortion = np.asarray(right_camera.distortion[:5], dtype=np.float64)
    left_normalized = cv2.undistortPoints(
        left_points.reshape(-1, 1, 2), left_camera.matrix, left_distortion
    ).reshape(-1, 2)
    right_normalized = cv2.undistortPoints(
        right_points.reshape(-1, 1, 2), right_camera.matrix, right_distortion
    ).reshape(-1, 2)
    rotation = np.asarray(calibration.rotation_left_to_right, dtype=np.float64)
    translation = np.asarray(calibration.translation_left_to_right_m, dtype=np.float64).reshape(3)
    left_projection = np.hstack((np.eye(3), np.zeros((3, 1))))
    right_projection = np.hstack((rotation, translation[:, None]))
    homogeneous = cv2.triangulatePoints(
        left_projection,
        right_projection,
        left_normalized.T,
        right_normalized.T,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        points3d = (homogeneous[:3] / homogeneous[3:4]).T
    right_camera_points = (rotation @ points3d.T + translation[:, None]).T
    left_projected, _ = cv2.projectPoints(
        points3d, np.zeros(3), np.zeros(3), left_camera.matrix, left_distortion
    )
    right_projected, _ = cv2.projectPoints(
        points3d,
        cv2.Rodrigues(rotation)[0],
        translation,
        right_camera.matrix,
        right_distortion,
    )
    errors = np.maximum(
        np.linalg.norm(left_projected.reshape(-1, 2) - left_points, axis=1),
        np.linalg.norm(right_projected.reshape(-1, 2) - right_points, axis=1),
    )
    valid = (
        np.isfinite(points3d).all(axis=1)
        & np.isfinite(errors)
        & (points3d[:, 2] > 0.0)
        & (right_camera_points[:, 2] > 0.0)
        & (errors <= float(max_reprojection_error))
    )
    result: dict[int, _StereoPoint] = {}
    # Keep only one observation per keypoint even if a malformed match list is
    # supplied by an external matcher.
    used_right: set[int] = set()
    for index in np.flatnonzero(valid):
        left_index, right_index = (int(value) for value in matches[index])
        if left_index in result or right_index in used_right:
            continue
        used_right.add(right_index)
        result[left_index] = _StereoPoint(
            left_keypoint=left_index,
            right_keypoint=right_index,
            xyz_left=np.asarray(points3d[index], dtype=np.float64),
            reprojection_error=float(errors[index]),
        )
    if len(result) < 20:
        raise RuntimeError(
            f"stereo frame {left_image_id}-{right_image_id} has only {len(result)} valid points; "
            f"need at least 20 (max reprojection error={max_reprojection_error})"
        )
    return result


def _rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(source) < 3 or len(target) != len(source):
        return None
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    if np.linalg.matrix_rank(source_centered) < 2 or np.linalg.matrix_rank(target_centered) < 2:
        return None
    covariance = source_centered.T @ target_centered
    left, _, right_transposed = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(right_transposed.T @ left.T) < 0.0:
        correction[2, 2] = -1.0
    rotation = right_transposed.T @ correction @ left.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _estimate_rigid_motion(
    source: np.ndarray,
    target: np.ndarray,
    *,
    ransac_threshold_m: float,
    seed: int,
) -> _RigidMotion:
    if len(source) != len(target) or len(source) < 6:
        raise RuntimeError("too few 3-D correspondences for stereo motion estimation")
    rng = np.random.default_rng(seed)
    sample_count = min(500, max(80, len(source) * 2))
    best_indices = np.empty(0, dtype=np.int64)
    best_error = float("inf")
    for _ in range(sample_count):
        sample = rng.choice(len(source), size=3, replace=False)
        fitted = _rigid_fit(source[sample], target[sample])
        if fitted is None:
            continue
        rotation, translation = fitted
        errors = np.linalg.norm((rotation @ source.T).T + translation - target, axis=1)
        inliers = np.flatnonzero(errors <= ransac_threshold_m)
        score = float(np.median(errors[inliers])) if len(inliers) else float("inf")
        if len(inliers) > len(best_indices) or (
            len(inliers) == len(best_indices) and score < best_error
        ):
            best_indices = inliers
            best_error = score
    if len(best_indices) < 6:
        raise RuntimeError(
            f"stereo motion RANSAC found only {len(best_indices)} inliers out of {len(source)}"
        )
    fitted = _rigid_fit(source[best_indices], target[best_indices])
    if fitted is None:
        raise RuntimeError("stereo motion refinement failed")
    rotation, translation = fitted
    errors = np.linalg.norm((rotation @ source.T).T + translation - target, axis=1)
    # A second pass removes points that became inconsistent after refinement.
    refined_indices = np.flatnonzero(errors <= ransac_threshold_m)
    if len(refined_indices) >= 6:
        fitted = _rigid_fit(source[refined_indices], target[refined_indices])
        if fitted is not None:
            rotation, translation = fitted
            best_indices = refined_indices
            errors = np.linalg.norm((rotation @ source.T).T + translation - target, axis=1)
    return _RigidMotion(
        rotation_next_from_current=rotation,
        translation_next_from_current=translation,
        inlier_match_indices=np.asarray(best_indices, dtype=np.int64),
        median_error_m=float(np.median(errors[best_indices])),
    )


def _write_model(
    output_path: Path,
    *,
    connection: sqlite3.Connection,
    image_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    observations: dict[tuple[int, int], int],
    points: dict[int, tuple[np.ndarray, float, list[tuple[int, int]]]],
    image_keypoints: dict[int, np.ndarray],
    camera_ids: dict[int, int],
    image_names: dict[int, str],
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    camera_ids_used = sorted(set(camera_ids.values()))
    camera_lines = [_camera_line(connection, camera_id) for camera_id in camera_ids_used]
    (output_path / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"# Number of cameras: {len(camera_lines)}\n"
        + "\n".join(camera_lines)
        + "\n",
        encoding="utf-8",
    )

    def quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
        from .stereo_seed import _matrix_to_quaternion

        return _matrix_to_quaternion(rotation)

    image_lines: list[str] = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(image_names)}",
    ]
    observation_counts: list[int] = []
    for image_id in sorted(image_names):
        world_from_camera_rotation, world_from_camera_translation = image_poses[image_id]
        camera_from_world_rotation = world_from_camera_rotation.T
        camera_from_world_translation = -camera_from_world_rotation @ world_from_camera_translation
        qvec = quaternion(camera_from_world_rotation)
        image_lines.append(
            f"{image_id} {' '.join(f'{value:.17g}' for value in qvec)} "
            f"{' '.join(f'{value:.17g}' for value in camera_from_world_translation)} "
            f"{camera_ids[image_id]} {image_names[image_id]}"
        )
        values: list[str] = []
        count = 0
        for index, (x, y) in enumerate(image_keypoints[image_id]):
            point_id = observations.get((image_id, index), -1)
            if point_id >= 0:
                count += 1
            values.extend((f"{x:.9g}", f"{y:.9g}", str(point_id)))
        image_lines.append(" ".join(values))
        observation_counts.append(count)
    image_lines[3] += f", mean observations per image: {np.mean(observation_counts):.9g}"
    (output_path / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")

    point_lines = [
        "# 3D point list with one line of data per point:",
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, IMAGE_ID, POINT2D_IDX, ...",
        f"# Number of points: {len(points)}",
    ]
    for point_id in sorted(points):
        xyz, error, track = points[point_id]
        track_text = " ".join(f"{image_id} {point2d}" for image_id, point2d in track)
        point_lines.append(
            f"{point_id} {' '.join(f'{value:.17g}' for value in xyz)} 128 128 128 "
            f"{float(error):.9g} {track_text}"
        )
    (output_path / "points3D.txt").write_text("\n".join(point_lines) + "\n", encoding="utf-8")


def build_calibrated_stereo_planar_model(
    database_path: Path,
    output_path: Path,
    *,
    calibration: StereoCalibration,
    max_stereo_reprojection_error: float = 8.0,
    motion_ransac_threshold_m: float = 0.12,
    external_left_poses: Mapping[int, tuple[np.ndarray, np.ndarray]] | None = None,
    external_pose_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete COLMAP text model from synchronized calibrated pairs.

    When ``external_left_poses`` is supplied, the left-camera poses are fixed
    to the supplied world-from-camera trajectory.  Temporal matches are then
    accepted into tracks only when their stereo 3-D points agree in that
    external world frame.  The right-camera poses remain derived from the
    canonical rigid stereo calibration.
    """

    if max_stereo_reprojection_error <= 0.0 or motion_ransac_threshold_m <= 0.0:
        raise ValueError("stereo and motion thresholds must be positive")
    database_path = database_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    connection = sqlite3.connect(database_path)
    try:
        records = _image_records(connection)
        frames = sorted(
            frame
            for side, frame in records
            if side == "left" and ("right", frame) in records
        )
        if len(frames) < 2:
            raise RuntimeError("calibrated stereo planar reconstruction needs at least two synchronized frames")
        external_by_frame: dict[int, tuple[np.ndarray, np.ndarray]] | None = None
        if external_left_poses is not None:
            external_by_frame = {}
            missing = []
            for frame in frames:
                value = external_left_poses.get(frame)
                if value is None:
                    missing.append(frame)
                    continue
                rotation, translation = value
                rotation = np.asarray(rotation, dtype=np.float64)
                translation = np.asarray(translation, dtype=np.float64).reshape(-1)
                if (
                    rotation.shape != (3, 3)
                    or translation.shape != (3,)
                    or not np.isfinite(rotation).all()
                    or not np.isfinite(translation).all()
                    or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5)
                    or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-5)
                ):
                    raise ValueError(f"invalid external left-camera pose for frame {frame}")
                external_by_frame[frame] = (rotation, translation)
            if missing:
                raise ValueError(
                    "external pose source has no pose for selected frames: "
                    + ", ".join(str(frame) for frame in missing[:10])
                )
        stereo_frames: list[_StereoFrame] = []
        image_keypoints: dict[int, np.ndarray] = {}
        image_names: dict[int, str] = {}
        camera_ids: dict[int, int] = {}
        for frame in frames:
            left_id, left_name, left_camera_id = records[("left", frame)]
            right_id, right_name, right_camera_id = records[("right", frame)]
            image_keypoints[left_id] = _read_keypoints(connection, left_id)
            image_keypoints[right_id] = _read_keypoints(connection, right_id)
            image_names[left_id] = left_name
            image_names[right_id] = right_name
            camera_ids[left_id] = left_camera_id
            camera_ids[right_id] = right_camera_id
            stereo_frames.append(
                _StereoFrame(
                    frame=frame,
                    left_image_id=left_id,
                    right_image_id=right_id,
                    left_name=left_name,
                    right_name=right_name,
                    left_camera_id=left_camera_id,
                    right_camera_id=right_camera_id,
                    left_keypoints=image_keypoints[left_id],
                    right_keypoints=image_keypoints[right_id],
                    points=_triangulate_frame(
                        connection,
                        left_image_id=left_id,
                        right_image_id=right_id,
                        calibration=calibration,
                        max_reprojection_error=max_stereo_reprojection_error,
                    ),
                )
            )

        image_poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if external_by_frame is None:
            image_poses[stereo_frames[0].left_image_id] = (np.eye(3), np.zeros(3))
        else:
            image_poses.update(
                {
                    frame.left_image_id: external_by_frame[frame.frame]
                    for frame in stereo_frames
                }
            )
        motion_summaries: list[dict[str, Any]] = []
        left_track_nodes: dict[tuple[int, int], int] = {}
        parent: list[int] = []
        rank: list[int] = []

        def node(key: tuple[int, int]) -> int:
            if key not in left_track_nodes:
                left_track_nodes[key] = len(parent)
                parent.append(len(parent))
                rank.append(0)
            return left_track_nodes[key]

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(first: int, second: int) -> None:
            first_root, second_root = find(first), find(second)
            if first_root == second_root:
                return
            if rank[first_root] < rank[second_root]:
                first_root, second_root = second_root, first_root
            parent[second_root] = first_root
            if rank[first_root] == rank[second_root]:
                rank[first_root] += 1

        for current_index, current in enumerate(stereo_frames):
            for left_keypoint in current.points:
                node((current.left_image_id, left_keypoint))
            if current_index == 0:
                continue
            previous = stereo_frames[current_index - 1]
            matches = _raw_matches(
                connection, previous.left_image_id, current.left_image_id
            )
            source: list[np.ndarray] = []
            target: list[np.ndarray] = []
            match_keys: list[tuple[int, int]] = []
            for left_previous, left_current in matches:
                previous_point = previous.points.get(int(left_previous))
                current_point = current.points.get(int(left_current))
                if previous_point is None or current_point is None:
                    continue
                source.append(previous_point.xyz_left)
                target.append(current_point.xyz_left)
                match_keys.append((int(left_previous), int(left_current)))
            if external_by_frame is None:
                motion = _estimate_rigid_motion(
                    np.asarray(source, dtype=np.float64),
                    np.asarray(target, dtype=np.float64),
                    ransac_threshold_m=motion_ransac_threshold_m,
                    seed=current.frame,
                )
                previous_world_rotation, previous_world_translation = image_poses[
                    previous.left_image_id
                ]
                relative_rotation = motion.rotation_next_from_current
                relative_translation = motion.translation_next_from_current
                current_world_rotation = previous_world_rotation @ relative_rotation.T
                current_world_translation = (
                    previous_world_translation - current_world_rotation @ relative_translation
                )
                image_poses[current.left_image_id] = (
                    current_world_rotation,
                    current_world_translation,
                )
                inlier_indices = motion.inlier_match_indices
                median_error_m = motion.median_error_m
                translation_m = float(np.linalg.norm(relative_translation))
                rotation_degrees = float(
                    np.degrees(
                        np.arccos(
                            np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
                        )
                    )
                )
            else:
                previous_world_rotation, previous_world_translation = image_poses[
                    previous.left_image_id
                ]
                current_world_rotation, current_world_translation = image_poses[
                    current.left_image_id
                ]
                if source:
                    source_array = np.asarray(source, dtype=np.float64)
                    target_array = np.asarray(target, dtype=np.float64)
                    source_world = (
                        previous_world_rotation @ source_array.T
                    ).T + previous_world_translation
                    target_world = (
                        current_world_rotation @ target_array.T
                    ).T + current_world_translation
                    errors = np.linalg.norm(source_world - target_world, axis=1)
                    inlier_indices = np.flatnonzero(errors <= motion_ransac_threshold_m)
                    median_error_m = (
                        float(np.median(errors[inlier_indices]))
                        if len(inlier_indices)
                        else float("inf")
                    )
                else:
                    errors = np.empty(0, dtype=np.float64)
                    inlier_indices = np.empty(0, dtype=np.int64)
                    median_error_m = float("inf")
                if len(source) >= 6 and len(inlier_indices) < 6:
                    raise RuntimeError(
                        "external HDF5 poses reject temporal matches between frames "
                        f"{previous.frame} and {current.frame}: "
                        f"{len(inlier_indices)} inliers out of {len(source)} "
                        f"at threshold {motion_ransac_threshold_m} m"
                    )
                relative_rotation = current_world_rotation.T @ previous_world_rotation
                translation_m = float(
                    np.linalg.norm(current_world_translation - previous_world_translation)
                )
                rotation_degrees = float(
                    np.degrees(
                        np.arccos(
                            np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
                        )
                    )
                )
            for match_index in inlier_indices:
                left_previous, left_current = match_keys[int(match_index)]
                union(
                    node((previous.left_image_id, left_previous)),
                    node((current.left_image_id, left_current)),
                )
            motion_summaries.append(
                {
                    "from_frame": int(previous.frame),
                    "to_frame": int(current.frame),
                    "candidate_3d_matches": int(len(source)),
                    "inliers": int(len(inlier_indices)),
                    "median_error_m": float(median_error_m),
                    "translation_m": translation_m,
                    "rotation_degrees": rotation_degrees,
                    "source": "external_h5" if external_by_frame is not None else "visual_3d_rigid_fit",
                }
            )

        # Convert each stereo point into the world frame and group left-image
        # observations that survived the robust 3-D motion estimation.
        components: dict[int, list[tuple[_StereoFrame, _StereoPoint]]] = {}
        for frame in stereo_frames:
            world_rotation, world_translation = image_poses[frame.left_image_id]
            for point in frame.points.values():
                root = find(node((frame.left_image_id, point.left_keypoint)))
                world_xyz = world_rotation @ point.xyz_left + world_translation
                # Store a copy with its world coordinate for robust aggregation.
                copied = _StereoPoint(
                    point.left_keypoint,
                    point.right_keypoint,
                    world_xyz,
                    point.reprojection_error,
                )
                components.setdefault(root, []).append((frame, copied))

        observations: dict[tuple[int, int], int] = {}
        points: dict[int, tuple[np.ndarray, float, list[tuple[int, int]]]] = {}
        next_point_id = 1
        for values in components.values():
            # A malformed loop can merge two keypoints from one image. Keep one
            # observation per image, choosing the one closest to the component
            # median, because COLMAP tracks cannot contain duplicate image IDs.
            median_xyz = np.median(np.asarray([item[1].xyz_left for item in values]), axis=0)
            selected_by_image: dict[int, tuple[_StereoFrame, _StereoPoint]] = {}
            for frame, point in values:
                current = selected_by_image.get(frame.left_image_id)
                if current is None or np.linalg.norm(point.xyz_left - median_xyz) < np.linalg.norm(current[1].xyz_left - median_xyz):
                    selected_by_image[frame.left_image_id] = (frame, point)
            if not selected_by_image:
                continue
            point_id = next_point_id
            next_point_id += 1
            track: list[tuple[int, int]] = []
            errors: list[float] = []
            for frame, point in selected_by_image.values():
                left_key = (frame.left_image_id, point.left_keypoint)
                right_key = (frame.right_image_id, point.right_keypoint)
                if left_key not in observations and right_key not in observations:
                    observations[left_key] = point_id
                    observations[right_key] = point_id
                    track.extend(
                        [
                            (frame.left_image_id, point.left_keypoint),
                            (frame.right_image_id, point.right_keypoint),
                        ]
                    )
                    errors.append(point.reprojection_error)
            if len(track) >= 2:
                points[point_id] = (
                    median_xyz,
                    float(np.mean(errors)) if errors else 0.0,
                    track,
                )
        _write_model(
            output_path,
            connection=connection,
            image_poses={
                **image_poses,
                **{
                    frame.right_image_id: (
                        image_poses[frame.left_image_id][0]
                        @ np.asarray(calibration.rotation_left_to_right, dtype=np.float64).T,
                        image_poses[frame.left_image_id][1]
                        - (
                            image_poses[frame.left_image_id][0]
                            @ np.asarray(calibration.rotation_left_to_right, dtype=np.float64).T
                        )
                        @ np.asarray(calibration.translation_left_to_right_m, dtype=np.float64),
                    )
                    for frame in stereo_frames
                },
            },
            observations=observations,
            points=points,
            image_keypoints=image_keypoints,
            camera_ids=camera_ids,
            image_names=image_names,
        )
    finally:
        connection.close()
    all_stereo_counts = [len(frame.points) for frame in stereo_frames]
    result: dict[str, Any] = {
        "database": str(database_path),
        "output_path": str(output_path),
        "frames": int(len(stereo_frames)),
        "images": int(len(stereo_frames) * 2),
        "registered_images": int(len(stereo_frames) * 2),
        "points3D": int(len(points)),
        "scale": (
            "metric scale from external HDF5 ENU poses and the canonical calibrated stereo baseline; "
            "subject to HDF5 and calibration accuracy"
            if external_by_frame is not None
            else "metric scale from the canonical calibrated stereo baseline; subject to calibration accuracy"
        ),
        "stereo_points_per_frame": {
            "min": int(min(all_stereo_counts)),
            "median": float(np.median(all_stereo_counts)),
            "mean": float(np.mean(all_stereo_counts)),
            "max": int(max(all_stereo_counts)),
        },
        "motion_ransac_threshold_m": float(motion_ransac_threshold_m),
        "max_stereo_reprojection_error": float(max_stereo_reprojection_error),
        "pose_source": "external_h5" if external_by_frame is not None else "stereo_temporal_visual",
        "motion": motion_summaries,
    }
    if external_pose_metadata is not None:
        result["external_pose"] = dict(external_pose_metadata)
    return result
