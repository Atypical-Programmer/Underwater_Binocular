"""Unit tests for tracking configuration and pure replay bookkeeping."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from underwater_binocular.tracking.zed import (
    TrackingConfig,
    TrackingRecord,
    replay_tracking,
    start_tracking,
    summarize_tracking,
    write_tracking_outputs,
)


class _State:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name


class _FakePose:
    def __init__(self) -> None:
        self.valid = True
        self.pose_confidence = 90.0
        self.position = (0.0, 0.0, 0.0)

    def get_translation(self) -> _FakePose:
        return _FakeVector(self.position)

    def get_orientation(self) -> _FakePose:
        return _FakeVector((0.0, 0.0, 0.0, 1.0))

    def get(self) -> tuple[float, ...]:
        return self.position


class _FakeVector:
    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = values

    def get(self) -> tuple[float, ...]:
        return self.values


class _FakeCamera:
    def __init__(self) -> None:
        self.parameters = None
        self.position_calls = 0

    def enable_positional_tracking(self, parameters: object) -> int:
        self.parameters = parameters
        return 0

    def get_position(self, pose: _FakePose, _reference: object) -> _State:
        self.position_calls += 1
        pose.valid = True
        pose.pose_confidence = 90.0
        pose.position = (float(self.position_calls), 0.0, 0.0)
        return _State("OK")


class _FakeTrackingParameters:
    pass


class _FakeModes:
    GEN_1 = "gen1"
    GEN_3 = "gen3"


class _FakeErrorCodes:
    SUCCESS = 0


class _FakeSdk:
    POSITIONAL_TRACKING_MODE = _FakeModes
    ERROR_CODE = _FakeErrorCodes
    REFERENCE_FRAME = type("ReferenceFrame", (), {"WORLD": "world"})
    PositionalTrackingParameters = _FakeTrackingParameters
    Pose = _FakePose


class _FakeSession:
    def __init__(self, total_frames: int = 3) -> None:
        self.sdk = _FakeSdk
        self.camera = _FakeCamera()
        self.is_open = True
        self._position = -1
        self._total_frames = total_frames
        self.seeks: list[int] = []
        self.grabs = 0
        self._tracking_enabled = False

    def svo_number_of_frames(self) -> int:
        return self._total_frames

    def seek(self, position: int) -> None:
        self.seeks.append(position)
        self._position = position - 1

    def grab(self) -> bool:
        next_position = self._position + 1
        if next_position >= self._total_frames:
            return False
        self._position = next_position
        self.grabs += 1
        return True

    def svo_position(self) -> int:
        return self._position

    def timestamp_ns(self) -> int:
        return self._position * 100


def _record(
    frame: int,
    state: str,
    valid: bool,
    position: tuple[float, float, float],
) -> TrackingRecord:
    return TrackingRecord(
        frame_index=frame,
        svo_position=frame,
        timestamp_ns=frame * 100,
        tracking_state=state,
        pose_valid=valid,
        pose_confidence=90.0 if valid else float("nan"),
        tx_m=position[0],
        ty_m=position[1],
        tz_m=position[2],
        qx=0.0,
        qy=0.0,
        qz=0.0,
        qw=1.0,
    )


def test_gen3_is_selected_explicitly_without_fallback() -> None:
    session = _FakeSession()
    parameters = start_tracking(session, TrackingConfig(mode="GEN_3"))

    assert parameters.mode == "gen3"
    assert session.camera.parameters is parameters
    assert session._tracking_enabled is True


def test_unsupported_gen3_fails_loudly() -> None:
    class OnlyGen1:
        GEN_1 = "gen1"

    class SdkWithoutGen3(_FakeSdk):
        POSITIONAL_TRACKING_MODE = OnlyGen1

    session = _FakeSession()
    session.sdk = SdkWithoutGen3
    with pytest.raises(RuntimeError, match="does not support.*GEN_3"):
        start_tracking(session, TrackingConfig(mode="GEN_3"))


def test_replay_is_sequential_and_uses_one_initial_seek() -> None:
    session = _FakeSession(total_frames=5)
    records = replay_tracking(session, TrackingConfig(mode="GEN_1"), start_frame=1, max_frames=3)

    assert [record.svo_position for record in records] == [1, 2, 3]
    assert session.seeks == [1]
    assert session.grabs == 3
    assert session.camera.position_calls == 3


def test_summary_does_not_connect_poses_across_lost_states() -> None:
    records = [
        _record(0, "OK", True, (0.0, 0.0, 0.0)),
        _record(1, "OK", True, (1.0, 0.0, 0.0)),
        _record(2, "SEARCHING", False, (0.0, 0.0, 0.0)),
        _record(3, "OK", True, (10.0, 0.0, 0.0)),
        _record(4, "OFF", False, (0.0, 0.0, 0.0)),
        _record(5, "OK", True, (20.0, 0.0, 0.0)),
    ]
    summary = summarize_tracking(
        records,
        total_frames_reported=6,
        config=TrackingConfig(mode="GEN_1"),
        elapsed_seconds=0.1,
    )

    assert summary["valid_pose_frames"] == 4
    assert summary["searching_frames"] == 1
    assert summary["off_frames"] == 1
    assert summary["longest_valid_segment_frames"] == 2
    assert summary["trajectory_length_m"] == 1.0


def test_tracking_outputs_have_standard_columns(tmp_path: Path) -> None:
    records = [_record(0, "OK", True, (1.0, 2.0, 3.0))]
    output = tmp_path / "tracking"
    write_tracking_outputs(output, records, {"valid_pose_frames": 1}, {"status": "completed"})

    with (output / "trajectory.csv").open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header[:6] == [
        "frame_index",
        "svo_position",
        "timestamp_ns",
        "tracking_state",
        "pose_valid",
        "pose_confidence",
    ]
    assert "tx_m" in header and "qw" in header
    assert (output / "tracking_status.csv").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "run.json").is_file()
