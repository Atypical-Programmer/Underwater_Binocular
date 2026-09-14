"""Unit tests for reconstruction data contracts without optional inference."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from underwater_binocular.calibration.loaders import load_calibration_profile
from underwater_binocular.reconstruction.colmap_database import (
    create_colmap_database,
    database_stats,
)
from underwater_binocular.reconstruction.features import (
    FeatureSet,
    ImageRecord,
    restore_keypoints_to_original,
)
from underwater_binocular.reconstruction.matching import (
    DescriptorMatches,
    build_image_pairs,
)

ROOT = Path(__file__).resolve().parents[2]


def test_resize_coordinate_transform_is_explicit() -> None:
    points = np.asarray([[10.0, 20.0], [100.0, 200.0]], dtype=np.float32)
    restored = restore_keypoints_to_original(
        points,
        original_size_hw=(1080, 1920),
        extracted_size_hw=(540, 960),
        coordinate_space="resized_image_pixels",
    )
    np.testing.assert_allclose(restored, points * 2.0)


def test_unknown_coordinate_space_is_rejected() -> None:
    points = np.zeros((1, 2), dtype=np.float32)
    try:
        restore_keypoints_to_original(
            points,
            original_size_hw=(10, 10),
            extracted_size_hw=(10, 10),
            coordinate_space="unknown",
        )
    except ValueError as error:
        assert "coordinate space" in str(error)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("unknown coordinate space was accepted")


def test_pair_generation_is_bounded_and_contains_synchronized_stereo() -> None:
    records = [
        ImageRecord(Path(f"left_{index}.png"), f"left/left_{index}.png", "left", index * 10)
        for index in range(4)
    ] + [
        ImageRecord(Path(f"right_{index}.png"), f"right/right_{index}.png", "right", index * 10)
        for index in range(4)
    ]
    pairs = build_image_pairs(records, temporal_window=1, stereo_window=0)

    assert len(pairs) == 10
    assert sum(pair.category == "synchronized_left_right" for pair in pairs) == 4
    assert len(pairs) < len(records) * (len(records) - 1) // 2


def test_colmap_database_contains_custom_keypoints_and_matches(tmp_path: Path) -> None:
    profile_path = ROOT / "calibration/profiles/zed2i_37395692_custom.yaml"
    calibration = load_calibration_profile(profile_path)
    features = [
        FeatureSet(
            image_path=tmp_path / "left.png",
            image_name="left/left.png",
            camera_side="left",
            keypoints_xy=np.asarray([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32),
            descriptors=np.zeros((2, 128), dtype=np.float32),
            model="ALIKED/aliked-n16",
            image_size_hw=(1080, 1920),
        ),
        FeatureSet(
            image_path=tmp_path / "right.png",
            image_name="right/right.png",
            camera_side="right",
            keypoints_xy=np.asarray([[11.0, 10.0], [21.0, 20.0]], dtype=np.float32),
            descriptors=np.zeros((2, 128), dtype=np.float32),
            model="ALIKED/aliked-n16",
            image_size_hw=(1080, 1920),
        ),
    ]
    matches = [
        (
            1,
            2,
            DescriptorMatches(
                np.asarray([0], dtype=np.int32),
                np.asarray([0], dtype=np.int32),
                np.asarray([0.1], dtype=np.float32),
                matcher="AdaLAM",
            ),
        )
    ]
    database = tmp_path / "database.db"
    result = create_colmap_database(
        database,
        features,
        matches,
        calibration=calibration,
        camera_model="FULL_OPENCV",
        insert_matches=True,
    )

    assert result["camera_model"] == "FULL_OPENCV"
    assert result["matches"] == 1
    stats = database_stats(database)
    assert stats["images"] == 2
    assert stats["keypoints"] == 2
    assert stats["matches"] == 1
    with sqlite3.connect(database) as connection:
        names = [row[0] for row in connection.execute("SELECT name FROM images ORDER BY image_id")]
    assert names == ["left/left.png", "right/right.png"]
