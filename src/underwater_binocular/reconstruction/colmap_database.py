"""Small, testable COLMAP SQLite database writer."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from .features import FeatureSet
from .matching import DescriptorMatches


def create_colmap_database(path: Path, features: list[FeatureSet], matches: list[tuple[int, int, DescriptorMatches]]) -> dict[str, int]:
    """Create a minimal COLMAP database from explicit feature/match objects."""

    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
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
        for image_id, feature in enumerate(features, start=1):
            height, width = _image_size(feature.image_path)
            params = np.asarray([1.0, 1.0, width / 2.0, height / 2.0], dtype=np.float64)
            connection.execute("INSERT INTO cameras VALUES (?,?,?,?,?,?)", (image_id, 1, width, height, params.tobytes(), 1))
            connection.execute("INSERT INTO images VALUES (?,?,?)", (image_id, feature.image_path.name, image_id))
            keypoints = np.ascontiguousarray(feature.keypoints_xy, dtype=np.float32)
            connection.execute("INSERT INTO keypoints VALUES (?,?,?,?)", (image_id, len(keypoints), keypoints.shape[1] if keypoints.ndim == 2 else 0, keypoints.tobytes()))
            descriptors = np.ascontiguousarray(feature.descriptors, dtype=np.uint8)
            connection.execute("INSERT INTO descriptors VALUES (?,?,?,?)", (image_id, len(descriptors), descriptors.shape[1] if descriptors.ndim == 2 else 0, descriptors.tobytes()))
        for image_id_left, image_id_right, value in matches:
            pair_id = _pair_id(image_id_left, image_id_right)
            data = np.column_stack((value.query_indices, value.train_indices)).astype(np.uint32)
            connection.execute("INSERT INTO matches VALUES (?,?,?,?,?)", (pair_id, len(data), 2, data.tobytes()))
        connection.commit()
        return {"cameras": len(features), "images": len(features), "keypoints": sum(len(item.keypoints_xy) for item in features), "matches": sum(len(item.query_indices) for _, _, item in matches)}
    finally:
        connection.close()


def _pair_id(image_id_left: int, image_id_right: int) -> int:
    first, second = sorted((int(image_id_left), int(image_id_right)))
    return first * 2147483647 + second


def _image_size(path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"could not read image for COLMAP database: {path}")
    return int(image.shape[0]), int(image.shape[1])
