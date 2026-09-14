"""Build a small calibrated-stereo seed model for COLMAP.

The ordinary incremental mapper estimates the first two-view pose from image
matches.  A scene dominated by one plane can make that estimate choose a
homography decomposition even when a calibrated stereo baseline is available.
This module creates a minimal COLMAP text model from one synchronized
left/right pair and the canonical stereo calibration.  The resulting points
are metric up to the calibration profile's scale and give the mapper a valid
3-D starting structure for planar scenes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from ..calibration.models import StereoCalibration

_COLMAP_CAMERA_MODELS = {1: "SIMPLE_PINHOLE", 2: "PINHOLE", 3: "SIMPLE_RADIAL", 4: "OPENCV", 5: "SIMPLE_RADIAL_FISHEYE", 6: "FULL_OPENCV"}
_PAIR_ID_MAX = 2_147_483_647


def _pair_id(image_id1: int, image_id2: int) -> int:
    first, second = sorted((int(image_id1), int(image_id2)))
    return first * _PAIR_ID_MAX + second


def _read_keypoints(connection: sqlite3.Connection, image_id: int) -> np.ndarray:
    row = connection.execute(
        "SELECT rows, data FROM keypoints WHERE image_id = ?", (int(image_id),)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"COLMAP database has no keypoints for image {image_id}")
    rows, data = int(row[0]), row[1]
    keypoints = np.frombuffer(data, dtype=np.float32)
    if keypoints.size != rows * 2:
        raise RuntimeError(f"invalid keypoint blob for image {image_id}")
    return np.asarray(keypoints.reshape(rows, 2), dtype=np.float64)


def _matrix_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper rotation matrix to COLMAP's w,x,y,z quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        values = (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0))
        values = (
            (matrix[2, 1] - matrix[1, 2]) / scale,
            0.25 * scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
        )
    elif matrix[1, 1] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0))
        values = (
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            0.25 * scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
        )
    else:
        scale = 2.0 * np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0))
        values = (
            (matrix[1, 0] - matrix[0, 1]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            0.25 * scale,
        )
    quaternion = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("stereo seed rotation is not a valid rotation")
    quaternion /= norm
    return tuple(float(value) for value in quaternion)


def _camera_line(connection: sqlite3.Connection, camera_id: int) -> str:
    row = connection.execute(
        "SELECT model, width, height, params FROM cameras WHERE camera_id = ?",
        (int(camera_id),),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"COLMAP database has no camera {camera_id}")
    model_id, width, height, params_blob = int(row[0]), int(row[1]), int(row[2]), row[3]
    model = _COLMAP_CAMERA_MODELS.get(model_id)
    if model is None:
        raise RuntimeError(f"unsupported COLMAP camera model id {model_id}")
    params = np.frombuffer(params_blob, dtype=np.float64)
    return f"{camera_id} {model} {width} {height} " + " ".join(f"{value:.17g}" for value in params)


def _image_info(connection: sqlite3.Connection, image_id: int) -> tuple[str, int]:
    row = connection.execute(
        "SELECT name, camera_id FROM images WHERE image_id = ?", (int(image_id),)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"COLMAP database has no image {image_id}")
    return str(row[0]), int(row[1])


def _triangulate_seed(
    connection: sqlite3.Connection,
    *,
    left_image_id: int,
    right_image_id: int,
    calibration: StereoCalibration,
    max_reprojection_error: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import cv2

    match_row = connection.execute(
        "SELECT rows, data FROM matches WHERE pair_id = ?",
        (_pair_id(left_image_id, right_image_id),),
    ).fetchone()
    if match_row is None:
        raise RuntimeError(
            f"COLMAP database has no raw matches for seed pair "
            f"{left_image_id}-{right_image_id}"
        )
    rows, data = int(match_row[0]), match_row[1]
    match_values = np.frombuffer(data, dtype=np.uint32)
    if match_values.size != rows * 2:
        raise RuntimeError("invalid raw match blob for stereo seed pair")
    matches = match_values.reshape(rows, 2)

    left_points = _read_keypoints(connection, left_image_id)[matches[:, 0]]
    right_points = _read_keypoints(connection, right_image_id)[matches[:, 1]]
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
    left_error = np.linalg.norm(left_projected.reshape(-1, 2) - left_points, axis=1)
    right_error = np.linalg.norm(right_projected.reshape(-1, 2) - right_points, axis=1)
    errors = np.maximum(left_error, right_error)
    valid = (
        np.isfinite(points3d).all(axis=1)
        & np.isfinite(errors)
        & (points3d[:, 2] > 0.0)
        & (right_camera_points[:, 2] > 0.0)
        & (errors <= float(max_reprojection_error))
    )

    # The custom matcher is expected to be one-to-one, but keep the seed
    # construction safe if a third-party database contains duplicate indices.
    used_left: set[int] = set()
    used_right: set[int] = set()
    selected: list[int] = []
    for index in np.flatnonzero(valid):
        left_index, right_index = (int(value) for value in matches[index])
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        selected.append(int(index))
    selected_indices = np.asarray(selected, dtype=np.int64)
    if len(selected_indices) < 20:
        raise RuntimeError(
            f"calibrated stereo seed has only {len(selected_indices)} valid points; "
            f"need at least 20 (max reprojection error={max_reprojection_error})"
        )
    return (
        points3d[selected_indices],
        matches[selected_indices],
        errors[selected_indices],
        np.asarray(selected_indices, dtype=np.int64),
    )


def build_calibrated_stereo_seed_model(
    database_path: Path,
    output_path: Path,
    *,
    calibration: StereoCalibration,
    left_image_id: int = 1,
    right_image_id: int = 2,
    max_reprojection_error: float = 8.0,
) -> dict[str, Any]:
    """Write a minimal two-image COLMAP text model from a known stereo pair."""

    if max_reprojection_error <= 0.0:
        raise ValueError("max_reprojection_error must be positive")
    database_path = database_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        left_name, left_camera_id = _image_info(connection, left_image_id)
        right_name, right_camera_id = _image_info(connection, right_image_id)
        points3d, matches, errors, _ = _triangulate_seed(
            connection,
            left_image_id=left_image_id,
            right_image_id=right_image_id,
            calibration=calibration,
            max_reprojection_error=max_reprojection_error,
        )
        left_keypoints = _read_keypoints(connection, left_image_id)
        right_keypoints = _read_keypoints(connection, right_image_id)
        camera_lines = [
            _camera_line(connection, left_camera_id),
            _camera_line(connection, right_camera_id),
        ]
    finally:
        connection.close()

    left_point_ids = {int(match[0]): index + 1 for index, match in enumerate(matches)}
    right_point_ids = {int(match[1]): index + 1 for index, match in enumerate(matches)}

    def points2d_line(keypoints: np.ndarray, point_ids: dict[int, int]) -> str:
        values: list[str] = []
        for point2d_index, (x, y) in enumerate(keypoints):
            values.extend((f"{x:.9g}", f"{y:.9g}", str(point_ids.get(point2d_index, -1))))
        return " ".join(values)

    left_quaternion = (1.0, 0.0, 0.0, 0.0)
    right_quaternion = _matrix_to_quaternion(calibration.rotation_left_to_right)
    translation = np.asarray(calibration.translation_left_to_right_m, dtype=np.float64).reshape(3)
    left_image_line = (
        f"{left_image_id} "
        f"{' '.join(f'{value:.17g}' for value in left_quaternion)} 0 0 0 "
        f"{left_camera_id} {left_name}"
    )
    right_image_line = (
        f"{right_image_id} "
        f"{' '.join(f'{value:.17g}' for value in right_quaternion)} "
        f"{' '.join(f'{value:.17g}' for value in translation)} "
        f"{right_camera_id} {right_name}"
    )

    (output_path / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"# Number of cameras: {len(camera_lines)}\n"
        + "\n".join(camera_lines)
        + "\n",
        encoding="utf-8",
    )
    (output_path / "images.txt").write_text(
        "# Image list with two lines of data per image:\n"
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "# POINTS2D[] as (X, Y, POINT3D_ID)\n"
        f"# Number of images: 2, mean observations per image: "
        f"{(len(left_point_ids) + len(right_point_ids)) / 2.0:.9g}\n"
        f"{left_image_line}\n{points2d_line(left_keypoints, left_point_ids)}\n"
        f"{right_image_line}\n{points2d_line(right_keypoints, right_point_ids)}\n",
        encoding="utf-8",
    )
    point_lines = [
        "# 3D point list with one line of data per point:",
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, IMAGE_ID, POINT2D_IDX, ...",
        f"# Number of points: {len(points3d)}",
    ]
    for point_id, (point, error, match) in enumerate(zip(points3d, errors, matches, strict=True), start=1):
        point_lines.append(
            f"{point_id} {' '.join(f'{value:.17g}' for value in point)} 128 128 128 "
            f"{float(error):.9g} {left_image_id} {int(match[0])} "
            f"{right_image_id} {int(match[1])}"
        )
    (output_path / "points3D.txt").write_text("\n".join(point_lines) + "\n", encoding="utf-8")
    return {
        "database": str(database_path),
        "output_path": str(output_path),
        "left_image_id": int(left_image_id),
        "right_image_id": int(right_image_id),
        "left_image": left_name,
        "right_image": right_name,
        "points3D": int(len(points3d)),
        "max_reprojection_error": float(max_reprojection_error),
        "median_reprojection_error": float(np.median(errors)),
        "max_observations_per_image": int(max(len(left_point_ids), len(right_point_ids))),
    }
