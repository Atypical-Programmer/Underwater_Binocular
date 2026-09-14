"""Depth-frame, statistics, and SGBM boundary tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from underwater_binocular.config.models import SgbmConfig, ZedSessionConfig
from underwater_binocular.depth.models import DepthFrame
from underwater_binocular.depth.sgbm import depth_from_disparity
from underwater_binocular.depth.statistics import center_measurement, region_statistics
from underwater_binocular.depth.zed_sdk import ZedSdkDepthEngine
from underwater_binocular.io.images import ImageSemantic


def test_disparity_to_depth_uses_corrected_pixel_units_and_metres() -> None:
    depth, valid = depth_from_disparity(
        np.array([[16.0, 32.0, 0.0, np.nan]], dtype=np.float32),
        focal_length_px=100.0,
        baseline_m=0.12,
    )

    np.testing.assert_allclose(depth[0, :2], [0.75, 0.375], atol=1.0e-7)
    assert valid.tolist() == [[True, True, False, False]]
    assert np.isnan(depth[0, 2:]).all()


def test_depth_statistics_keep_invalid_samples_out_of_quantiles() -> None:
    frame = DepthFrame(
        np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32),
        np.array([[True, True], [False, True]]),
        frame_index=7,
        timestamp_ns=123,
    )

    center = center_measurement(frame, radius=1)
    stats = region_statistics(frame, {"all": (slice(None), slice(None))})["all"]
    assert center["frame_index"] == 7
    assert center["center_window_valid_count"] == 3
    assert stats["valid_pixel_count"] == 3
    assert stats["valid_ratio"] == 0.75
    assert frame.finite_depth().tolist() == [1.0, 2.0, 4.0]


def test_sgbm_config_matches_recorded_diagnostic_shape() -> None:
    config = SgbmConfig(scale=0.5, num_disparities=192)
    config.validate()

    assert ImageSemantic.RAW_UNRECTIFIED.value == "RAW_UNRECTIFIED"
    assert config.num_disparities % 16 == 0
    assert config.scale == 0.5


def test_sdk_export_streams_frames_to_compact_outputs(tmp_path: Path) -> None:
    class FakeSession:
        config = ZedSessionConfig()
        runtime_metadata = {"resolution": {"width": 3, "height": 3}}

    class FakeEngine(ZedSdkDepthEngine):
        def iter_frames(self):
            for index, center in enumerate((2.0, 2.5)):
                depth = np.full((3, 3), center, dtype=np.float32)
                yield DepthFrame(depth, np.ones((3, 3), dtype=bool), index)

    summary = FakeEngine(FakeSession()).export(
        tmp_path,
        fps=30.0,
        include_video=False,
    )

    assert summary["frames"] == 2
    assert summary["video_frames"] == 0
    assert summary["center_pixel"]["median_m"] == 2.25
    assert len((tmp_path / "center_depth.csv").read_text(encoding="utf-8").splitlines()) == 3
