"""Tests for HDF5 inertial pose conversion and timestamp interpolation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from underwater_binocular.reconstruction.external_pose import (  # noqa: E402
    load_h5_inertial_pose_sequence,
)


def _write_pose_file(path: Path) -> None:
    dtype = np.dtype(
        [
            ("utc", "f8"),
            ("heading", "f8"),
            ("roll", "f8"),
            ("pitch", "f8"),
            ("lat", "f8"),
            ("lon", "f8"),
            ("altitude", "f8"),
            ("nav_status", "S4"),
            ("fault", "S1"),
            ("quality_ok", "?")
        ]
    )
    values = np.zeros(3, dtype=dtype)
    values["utc"] = [100.0, 100.2, 100.4]
    values["heading"] = [359.0, 1.0, 3.0]
    values["roll"] = [0.0, 0.0, 0.0]
    values["pitch"] = [0.0, 0.0, 0.0]
    values["lat"] = [16.0, 16.0, 16.00001]
    values["lon"] = [110.0, 110.00001, 110.00001]
    values["altitude"] = [-10.0, -10.1, -10.0]
    values["nav_status"] = b"05"
    values["fault"] = b"A"
    values["quality_ok"] = True
    with h5py.File(path, "w") as handle:
        handle.attrs["inertial_rate_hz"] = 10.0
        handle.attrs["enu_origin_lat"] = 16.0
        handle.attrs["enu_origin_lon"] = 110.0
        handle.create_dataset("inertial", data=values)


def test_h5_pose_interpolation_unwraps_heading_and_uses_enu(tmp_path: Path) -> None:
    path = tmp_path / "pose.h5"
    _write_pose_file(path)
    sequence = load_h5_inertial_pose_sequence(path)

    poses, metadata = sequence.interpolate_camera_poses(
        {0: 100_000_000_000, 1: 100_200_000_000, 2: 100_400_000_000}
    )

    assert metadata["actual_rate_hz_from_utc"] == pytest.approx(5.0)
    assert metadata["camera_frames"] == 3
    assert len(poses) == 3
    assert np.allclose(poses[0][1], np.zeros(3))
    assert np.isfinite(poses[1][0]).all()
    assert np.isfinite(poses[2][1]).all()
    # Heading 359 -> 1 degrees is interpolated through 360, not through 180.
    assert sequence.heading_unwrapped_deg[1] == pytest.approx(361.0)


def test_h5_pose_rejects_camera_time_outside_coverage(tmp_path: Path) -> None:
    path = tmp_path / "pose.h5"
    _write_pose_file(path)
    sequence = load_h5_inertial_pose_sequence(path)

    with pytest.raises(ValueError, match="outside HDF5 inertial coverage"):
        sequence.interpolate_camera_poses({0: 99_000_000_000})
