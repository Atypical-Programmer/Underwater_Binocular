"""Mocked tests for the single ZED I/O boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from underwater_binocular.calibration.loaders import load_calibration_profile
from underwater_binocular.calibration.zed import compare_runtime_calibration
from underwater_binocular.io.zed import ZedImagePair, ZedSession

ROOT = Path(__file__).resolve().parents[2]


class FakeSession(ZedSession):
    def __init__(self) -> None:
        self._frame_index = -1
        self._remaining = 1
        self.calls: list[str] = []

    def grab(self) -> bool:
        if self._remaining == 0:
            return False
        self._remaining -= 1
        self._frame_index += 1
        return True

    def retrieve_image(self, view_name: str = "LEFT") -> np.ndarray:
        self.calls.append(view_name)
        return np.zeros((2, 2, 4), dtype=np.uint8)

    def timestamp_ns(self) -> int:
        return 42


def test_raw_and_rectified_sdk_views_are_not_conflated() -> None:
    raw_session = FakeSession()
    raw = list(raw_session.iter_image_pairs(view_name="RAW_UNRECTIFIED"))
    rectified_session = FakeSession()
    rectified = list(rectified_session.iter_image_pairs(view_name="RECTIFIED"))

    assert raw_session.calls == ["LEFT_UNRECTIFIED", "RIGHT_UNRECTIFIED"]
    assert rectified_session.calls == ["LEFT", "RIGHT"]
    assert isinstance(raw[0], ZedImagePair)
    assert raw[0].semantic == "RAW_UNRECTIFIED"
    assert rectified[0].semantic == "RECTIFIED"


def test_runtime_calibration_comparison_checks_the_inverse_transform() -> None:
    calibration = load_calibration_profile(ROOT / "calibration/profiles/zed2i_37395692_custom.yaml")
    rotation, translation = calibration.inverse_transform()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    def camera(camera_model: object) -> dict[str, object]:
        value = camera_model
        return {
            "fx_px": value.fx_px,
            "fy_px": value.fy_px,
            "cx_px": value.cx_px,
            "cy_px": value.cy_px,
            "distortion": list(value.distortion),
        }

    runtime = {
        "resolution": {"width": 1920, "height": 1080},
        "raw": {
            "left": camera(calibration.left),
            "right": camera(calibration.right),
            "stereo_transform_m": transform.tolist(),
        },
        "rectified": {},
    }
    result = compare_runtime_calibration(runtime, calibration)

    assert result["status"] == "PASS"


def test_runtime_calibration_comparison_rejects_incomplete_distortion() -> None:
    calibration = load_calibration_profile(ROOT / "calibration/profiles/zed2i_37395692_custom.yaml")
    runtime = {
        "resolution": {"width": 1920, "height": 1080},
        "raw": {
            "left": {"fx_px": calibration.left.fx_px, "fy_px": calibration.left.fy_px, "cx_px": calibration.left.cx_px, "cy_px": calibration.left.cy_px, "distortion": []},
            "right": {"fx_px": calibration.right.fx_px, "fy_px": calibration.right.fy_px, "cx_px": calibration.right.cx_px, "cy_px": calibration.right.cy_px, "distortion": list(calibration.right.distortion)},
            "stereo_transform_m": np.eye(4).tolist(),
        },
        "rectified": {},
    }

    with pytest.raises(RuntimeError, match="calibration mismatch"):
        compare_runtime_calibration(runtime, calibration)
