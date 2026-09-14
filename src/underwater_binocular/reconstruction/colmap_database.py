"""Small, testable COLMAP SQLite database writer."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from ..calibration.models import StereoCalibration
from .features import FeatureSet
from .matching import DescriptorMatches

PAIR_ID_MAX = 2_147_483_647
COLMAP_CAMERA_MODEL_IDS = {"PINHOLE": 1, "OPENCV": 4, "FULL_OPENCV": 6}


def _camera_params(calibration: StereoCalibration, side: str, camera_model: str) -> list[float]:
    camera = calibration.left if side == "left" else calibration.right
    base = [camera.fx_px, camera.fy_px, camera.cx_px, camera.cy_px]
    if camera_model == "PINHOLE":
        return base
    if camera_model == "OPENCV":
        return [*base, *camera.distortion[:4]]
    if camera_model == "FULL_OPENCV":
        return [*base, *camera.distortion[:8]]
    raise ValueError(f"unsupported COLMAP camera model: {camera_model}")


def create_colmap_database(
    path: Path,
    features: list[FeatureSet],
    matches: list[tuple[int, int, DescriptorMatches]],
    *,
    calibration: StereoCalibration | None = None,
    camera_model: str = "PINHOLE",
    insert_matches: bool = True,
) -> dict[str, int | str]:
    """Create a COLMAP database from explicit keypoints and AdaLAM matches.

    When ``insert_matches`` is false, the database is prepared for COLMAP's
    ``matches_importer``; this is the legacy-compatible path used before
    external geometric verification. If true, custom matches are inserted
    directly, which is useful for an offline database/plumbing run.
    """

    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    camera_model = camera_model.upper()
    if camera_model not in COLMAP_CAMERA_MODEL_IDS:
        raise ValueError(f"unsupported COLMAP camera model: {camera_model}")
    if not features:
        raise ValueError("COLMAP database requires at least one feature set")
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY, model INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL, params BLOB NOT NULL, prior_focal_length INTEGER NOT NULL);
            CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, camera_id INTEGER NOT NULL);
            CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB NOT NULL);
            CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB NOT NULL);
            CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB NOT NULL);
            CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB NOT NULL, config INTEGER NOT NULL, F BLOB, E BLOB, H BLOB, qvec BLOB, t BLOB);
            """
        )
        camera_id_by_side: dict[str, int] = {}
        if calibration is None:
            for image_id, feature in enumerate(features, start=1):
                height, width = _feature_image_size(feature)
                params = np.asarray([1.0, 1.0, width / 2.0, height / 2.0], dtype=np.float64)
                connection.execute(
                    "INSERT INTO cameras VALUES (?,?,?,?,?,?)",
                    (image_id, 1, width, height, params.tobytes(), 1),
                )
                camera_id_by_side[f"feature_{image_id}"] = image_id
        else:
            sides = sorted({feature.camera_side for feature in features}, key=("left", "right").index)
            for camera_id, side in enumerate(sides, start=1):
                side_features = [feature for feature in features if feature.camera_side == side]
                height, width = _feature_image_size(side_features[0])
                if (width, height) != calibration.resolution:
                    raise ValueError(
                        f"{side} image size {(width, height)} does not match calibration "
                        f"{calibration.resolution}"
                    )
                if any(_feature_image_size(feature) != (height, width) for feature in side_features):
                    raise ValueError(f"all {side} images must have the same dimensions")
                params = np.asarray(
                    _camera_params(calibration, side, camera_model), dtype=np.float64
                )
                connection.execute(
                    "INSERT INTO cameras VALUES (?,?,?,?,?,?)",
                    (
                        camera_id,
                        COLMAP_CAMERA_MODEL_IDS[camera_model],
                        width,
                        height,
                        params.tobytes(),
                        1,
                    ),
                )
                camera_id_by_side[side] = camera_id

        for image_id, feature in enumerate(features, start=1):
            if calibration is None:
                camera_id = camera_id_by_side[f"feature_{image_id}"]
            else:
                camera_id = camera_id_by_side[feature.camera_side]
            keypoint_name = feature.image_name or feature.image_path.name
            if "/" not in keypoint_name and "\\" not in keypoint_name:
                keypoint_name = keypoint_name.replace("\\", "/")
            connection.execute(
                "INSERT INTO images VALUES (?,?,?)", (image_id, keypoint_name, camera_id)
            )
            keypoints = np.ascontiguousarray(feature.keypoints_xy, dtype=np.float32)
            connection.execute(
                "INSERT INTO keypoints VALUES (?,?,?,?)",
                (image_id, len(keypoints), 2, keypoints.tobytes()),
            )
            # COLMAP's descriptors table is a SIFT-shaped uint8 storage slot.
            # The actual ALIKED float descriptors remain in features/*.npz;
            # imported raw AdaLAM matches are the source used by this workflow.
            descriptors = np.zeros((len(keypoints), 128), dtype=np.uint8)
            connection.execute(
                "INSERT INTO descriptors VALUES (?,?,?,?)",
                (image_id, len(descriptors), 128, descriptors.tobytes()),
            )
        if insert_matches:
            for image_id_left, image_id_right, value in matches:
                pair_id = _pair_id(image_id_left, image_id_right)
                data = np.column_stack((value.query_indices, value.train_indices)).astype(
                    np.uint32
                )
                connection.execute(
                    "INSERT INTO matches VALUES (?,?,?,?)",
                    (pair_id, len(data), 2, data.tobytes()),
                )
        connection.commit()
        return {
            "camera_model": camera_model,
            "cameras": len(camera_id_by_side),
            "images": len(features),
            "keypoints": sum(len(item.keypoints_xy) for item in features),
            "matches": sum(len(item.query_indices) for _, _, item in matches)
            if insert_matches
            else 0,
        }
    finally:
        connection.close()


def database_stats(path: Path) -> dict[str, int]:
    """Return row counts for the COLMAP tables, including verified geometries."""

    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(path)
    try:
        tables = [
            "cameras",
            "images",
            "keypoints",
            "descriptors",
            "matches",
            "two_view_geometries",
        ]
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }
        counts["verified_geometries_rows_ge_15"] = int(
            connection.execute(
                "SELECT COUNT(*) FROM two_view_geometries WHERE rows >= 15"
            ).fetchone()[0]
        )
        return counts
    finally:
        connection.close()


def _pair_id(image_id_left: int, image_id_right: int) -> int:
    first, second = sorted((int(image_id_left), int(image_id_right)))
    if first <= 0 or second <= 0 or first >= PAIR_ID_MAX or second >= PAIR_ID_MAX:
        raise ValueError("COLMAP image ids must be positive and below the pair-id limit")
    return first * PAIR_ID_MAX + second


def _feature_image_size(feature: FeatureSet) -> tuple[int, int]:
    if feature.image_size_hw is not None:
        return tuple(int(value) for value in feature.image_size_hw)
    height, width = _image_size(feature.image_path)
    return height, width


def _image_size(path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"could not read image for COLMAP database: {path}")
    return int(image.shape[0]), int(image.shape[1])
