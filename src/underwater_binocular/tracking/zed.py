"""End-to-end ZED positional tracking with explicit sequential replay."""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..calibration.loaders import load_calibration_profile, sha256_file
from ..config.loaders import load_dataset_config, load_zed_config
from ..config.models import CalibrationMode, DatasetConfig, ZedSessionConfig
from ..io.zed import ZedSession


@dataclass(frozen=True)
class TrackingConfig:
    """Tracking mode settings shared by every ZED tracking replay."""

    mode: str = "GEN_1"
    enable_area_memory: bool = False
    enable_imu_fusion: bool = True
    enable_pose_smoothing: bool = False
    set_gravity_as_origin: bool = True
    set_floor_as_origin: bool = False
    set_as_static: bool = False


@dataclass(frozen=True)
class TrackingRecord:
    """One attempted frame and its optional valid WORLD pose."""

    frame_index: int
    svo_position: int
    timestamp_ns: int
    tracking_state: str
    pose_valid: bool
    pose_confidence: float
    tx_m: float
    ty_m: float
    tz_m: float
    qx: float
    qy: float
    qz: float
    qw: float


TRACKING_STATUS_HEADER = [
    "frame_index",
    "svo_position",
    "timestamp_ns",
    "tracking_state",
    "pose_valid",
    "pose_confidence",
]
TRAJECTORY_HEADER = [
    *TRACKING_STATUS_HEADER,
    "tx_m",
    "ty_m",
    "tz_m",
    "qx",
    "qy",
    "qz",
    "qw",
]


def _status_name(status: Any) -> str:
    return str(status).rsplit(".", 1)[-1]


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repository_root(),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _enum_tracking_mode(sl: Any, mode: str) -> Any:
    normalized = str(mode).upper()
    if normalized not in {"GEN_1", "GEN_3"}:
        raise ValueError("tracking mode must be GEN_1 or GEN_3")
    namespace = getattr(sl, "POSITIONAL_TRACKING_MODE", None)
    if namespace is None or not hasattr(namespace, normalized):
        raise RuntimeError(
            f"ZED SDK does not support positional tracking mode {normalized}; "
            "no fallback mode was selected"
        )
    return getattr(namespace, normalized)


def start_tracking(session: ZedSession, config: TrackingConfig | None = None) -> Any:
    """Enable positional tracking on an open session using one shared policy."""

    if not session.is_open:
        raise RuntimeError("ZED session must be open before tracking starts")
    sl = session.sdk
    selected = config or TrackingConfig()
    mode_value = _enum_tracking_mode(sl, selected.mode)
    params = sl.PositionalTrackingParameters()
    params.mode = mode_value
    params.enable_area_memory = selected.enable_area_memory
    params.enable_imu_fusion = selected.enable_imu_fusion
    params.enable_pose_smoothing = selected.enable_pose_smoothing
    params.set_gravity_as_origin = selected.set_gravity_as_origin
    params.set_floor_as_origin = selected.set_floor_as_origin
    params.set_as_static = selected.set_as_static
    status = session.camera.enable_positional_tracking(params)
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(
            f"enabling positional tracking ({selected.mode.upper()}) failed: {_status_name(status)}"
        )
    session._tracking_enabled = True  # session owns cleanup of the active SDK resource
    return params


def _finite_vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size < length or not np.isfinite(vector[:length]).all():
        raise RuntimeError(f"ZED returned an invalid {label} vector")
    return tuple(float(item) for item in vector[:length])


def record_from_pose(
    pose: Any,
    tracking_state: Any,
    *,
    frame_index: int,
    svo_position: int,
    timestamp_ns: int,
) -> TrackingRecord:
    """Convert one SDK pose result into a serializable record.

    ZED's ``get_orientation().get()`` is written in the SDK's x,y,z,w order;
    the CSV header makes that order explicit.
    """

    state = _status_name(tracking_state)
    state_is_ok = state.upper() == "OK"
    pose_valid = bool(getattr(pose, "valid", False)) if state_is_ok else False
    try:
        confidence = float(pose.pose_confidence) if pose_valid else math.nan
    except (TypeError, ValueError):
        confidence = math.nan
    values = (math.nan,) * 7
    if pose_valid:
        translation = _finite_vector(pose.get_translation().get(), 3, "translation")
        quaternion = _finite_vector(pose.get_orientation().get(), 4, "orientation")
        values = (*translation, *quaternion)
        if not np.isfinite(confidence):
            confidence = math.nan
    return TrackingRecord(
        frame_index=int(frame_index),
        svo_position=int(svo_position),
        timestamp_ns=int(timestamp_ns),
        tracking_state=state,
        pose_valid=pose_valid,
        pose_confidence=confidence,
        tx_m=float(values[0]),
        ty_m=float(values[1]),
        tz_m=float(values[2]),
        qx=float(values[3]),
        qy=float(values[4]),
        qz=float(values[5]),
        qw=float(values[6]),
    )


def replay_tracking(
    session: ZedSession,
    config: TrackingConfig,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    max_frames: int = 0,
) -> list[TrackingRecord]:
    """Replay one SVO range sequentially after a pre-tracking seek.

    The initial ``seek`` is intentionally performed before tracking starts.
    Every subsequent frame is consumed with one ``grab`` and one
    ``get_position`` call; no random seeking occurs while the tracker is active.
    """

    if not session.is_open:
        raise RuntimeError("ZED session must be open before tracking replay")
    if start_frame < 0 or max_frames < 0:
        raise ValueError("start_frame and max_frames must be non-negative")
    total_frames = session.svo_number_of_frames()
    if start_frame >= total_frames:
        raise ValueError(
            f"start_frame {start_frame} is outside the SVO with {total_frames} frames"
        )
    requested_end = total_frames - 1 if end_frame is None else int(end_frame)
    if requested_end < start_frame:
        raise ValueError("end_frame must be greater than or equal to start_frame")
    requested_end = min(requested_end, total_frames - 1)
    if max_frames:
        requested_end = min(requested_end, start_frame + max_frames - 1)

    session.seek(start_frame)
    start_tracking(session, config)
    pose = session.sdk.Pose()
    records: list[TrackingRecord] = []
    while session.grab():
        svo_position = session.svo_position()
        if svo_position > requested_end:
            break
        state = session.camera.get_position(pose, session.sdk.REFERENCE_FRAME.WORLD)
        records.append(
            record_from_pose(
                pose,
                state,
                frame_index=len(records),
                svo_position=svo_position,
                timestamp_ns=session.timestamp_ns(),
            )
        )
        if svo_position >= requested_end:
            break
    if not records:
        raise RuntimeError("ZED tracking replay returned no frames")
    return records


def _longest_valid_segment(records: list[TrackingRecord]) -> int:
    longest = 0
    current = 0
    previous: TrackingRecord | None = None
    for record in records:
        consecutive = previous is not None and record.svo_position == previous.svo_position + 1
        if record.pose_valid and (previous is None or (previous.pose_valid and consecutive)):
            current += 1
        elif record.pose_valid:
            current = 1
        else:
            current = 0
        longest = max(longest, current)
        previous = record
    return longest


def summarize_tracking(
    records: list[TrackingRecord],
    *,
    total_frames_reported: int,
    config: TrackingConfig,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Calculate state counts and trajectory statistics without crossing lost gaps."""

    if not records:
        raise ValueError("cannot summarize an empty tracking replay")
    states = Counter(record.tracking_state for record in records)
    valid = [record for record in records if record.pose_valid]
    positions: list[np.ndarray] = []
    step_lengths: list[float] = []
    for previous, current in zip(records[:-1], records[1:], strict=True):
        if previous.pose_valid and current.pose_valid:
            if current.svo_position == previous.svo_position + 1:
                previous_position = np.asarray([previous.tx_m, previous.ty_m, previous.tz_m])
                current_position = np.asarray([current.tx_m, current.ty_m, current.tz_m])
                step_lengths.append(float(np.linalg.norm(current_position - previous_position)))
    for record in valid:
        positions.append(np.asarray([record.tx_m, record.ty_m, record.tz_m], dtype=np.float64))

    state_names = {str(key).upper(): int(value) for key, value in states.items()}
    searching = state_names.get("SEARCHING", 0)
    off = state_names.get("OFF", 0)
    lost = sum(
        count
        for name, count in state_names.items()
        if name not in {"OK", "SEARCHING", "OFF"}
    )
    lost += sum(1 for record in records if record.tracking_state.upper() == "OK" and not record.pose_valid)
    summary: dict[str, Any] = {
        "total_frames_attempted": len(records),
        "frames_grabbed": len(records),
        "total_svo_frames_reported": int(total_frames_reported),
        "valid_pose_frames": len(valid),
        "valid_pose_ratio": len(valid) / len(records),
        "searching_frames": searching,
        "lost_frames": lost,
        "off_frames": off,
        "tracking_state_counts": dict(sorted(states.items())),
        "longest_valid_segment_frames": _longest_valid_segment(records),
        "start_frame": records[0].svo_position,
        "end_frame": records[-1].svo_position,
        "mode": config.mode.upper(),
        "enable_area_memory": bool(config.enable_area_memory),
        "enable_imu_fusion": bool(config.enable_imu_fusion),
        "enable_pose_smoothing": bool(config.enable_pose_smoothing),
        "set_gravity_as_origin": bool(config.set_gravity_as_origin),
        "set_floor_as_origin": bool(config.set_floor_as_origin),
        "set_as_static": bool(config.set_as_static),
        "coordinate_system": "RIGHT_HANDED_Y_UP",
        "pose_reference_frame": "WORLD",
        "translation_unit": "metre",
        "quaternion_order": "x,y,z,w",
        "elapsed_seconds": float(elapsed_seconds),
    }
    if step_lengths:
        summary["trajectory_length_m"] = float(sum(step_lengths))
        summary["step_p50_m"] = float(np.percentile(step_lengths, 50))
        summary["step_p95_m"] = float(np.percentile(step_lengths, 95))
        summary["step_max_m"] = float(max(step_lengths))
    else:
        summary["trajectory_length_m"] = 0.0
        summary["step_p50_m"] = None
        summary["step_p95_m"] = None
        summary["step_max_m"] = None
    if len(positions) >= 1:
        summary["start_position_m"] = positions[0].tolist()
        summary["end_position_m"] = positions[-1].tolist()
    if len(positions) >= 2:
        summary["endpoint_displacement_m"] = float(np.linalg.norm(positions[-1] - positions[0]))
    else:
        summary["endpoint_displacement_m"] = None
    return summary


def _csv_number(value: float) -> str:
    return "NaN" if not np.isfinite(float(value)) else f"{float(value):.12g}"


def write_tracking_outputs(
    output_dir: Path,
    records: list[TrackingRecord],
    summary: dict[str, Any],
    run_metadata: dict[str, Any],
) -> None:
    """Write standard trajectory, status, summary, and provenance files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRAJECTORY_HEADER)
        for record in records:
            writer.writerow(
                [
                    record.frame_index,
                    record.svo_position,
                    record.timestamp_ns,
                    record.tracking_state,
                    int(record.pose_valid),
                    _csv_number(record.pose_confidence),
                    _csv_number(record.tx_m),
                    _csv_number(record.ty_m),
                    _csv_number(record.tz_m),
                    _csv_number(record.qx),
                    _csv_number(record.qy),
                    _csv_number(record.qz),
                    _csv_number(record.qw),
                ]
            )
    with (output_dir / "tracking_status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRACKING_STATUS_HEADER)
        for record in records:
            writer.writerow(
                [
                    record.frame_index,
                    record.svo_position,
                    record.timestamp_ns,
                    record.tracking_state,
                    int(record.pose_valid),
                    _csv_number(record.pose_confidence),
                ]
            )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "run.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _svo_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _software_versions(sdk: Any | None = None) -> dict[str, Any]:
    versions: dict[str, Any] = {
        "package": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    if sdk is not None:
        try:
            versions["zed_sdk"] = str(sdk.Camera.get_sdk_version())
        except (AttributeError, TypeError):
            versions["zed_sdk"] = "unknown"
    return versions


def _camera_metadata(info: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for attribute, key in (("camera_model", "camera_model"), ("serial_number", "serial_number")):
        value = getattr(info, attribute, None)
        if value is not None:
            result[key] = _status_name(value) if attribute == "camera_model" else int(value)
    configuration = getattr(info, "camera_configuration", None)
    resolution = getattr(configuration, "resolution", None)
    if resolution is not None:
        result["resolution"] = {
            "width": int(resolution.width),
            "height": int(resolution.height),
        }
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _compare_replays(
    first: list[TrackingRecord], second: list[TrackingRecord], output_path: Path
) -> dict[str, Any]:
    if len(first) != len(second):
        raise RuntimeError("independent tracking replays returned different row counts")
    differences: list[float] = []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "svo_position",
                "timestamp_ns",
                "gen1_tracking_state",
                "gen3_tracking_state",
                "both_pose_valid",
                "translation_difference_m",
            ]
        )
        for left, right in zip(first, second, strict=True):
            if (left.svo_position, left.timestamp_ns) != (right.svo_position, right.timestamp_ns):
                raise RuntimeError("GEN_1 and GEN_3 replay frame alignment differs")
            both_valid = left.pose_valid and right.pose_valid
            difference = math.nan
            if both_valid:
                difference = float(
                    np.linalg.norm(
                        np.asarray([left.tx_m, left.ty_m, left.tz_m])
                        - np.asarray([right.tx_m, right.ty_m, right.tz_m])
                    )
                )
                differences.append(difference)
            writer.writerow(
                [
                    left.frame_index,
                    left.svo_position,
                    left.timestamp_ns,
                    left.tracking_state,
                    right.tracking_state,
                    int(both_valid),
                    _csv_number(difference),
                ]
            )
    result: dict[str, Any] = {"aligned_rows": len(first), "both_pose_valid_rows": len(differences)}
    if differences:
        result["translation_difference_m"] = {
            "median": float(np.median(differences)),
            "p95": float(np.percentile(differences, 95)),
            "max": float(np.max(differences)),
        }
    return result


def run_tracking(
    dataset_path: Path,
    *,
    mode: str = "GEN_1",
    output_dir: Path,
    calibration_mode: CalibrationMode | str = CalibrationMode.NATIVE,
    svo_override: Path | None = None,
    profile_override: Path | None = None,
    zed_config_path: Path | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    max_frames: int = 0,
    enable_area_memory: bool = False,
) -> dict[str, Any]:
    """Run one or two independent ZED positional-tracking replays."""

    dataset_path = dataset_path.expanduser().resolve()
    dataset: DatasetConfig = load_dataset_config(dataset_path)
    selected_calibration_mode = CalibrationMode.parse(calibration_mode)
    if selected_calibration_mode is CalibrationMode.NATIVE and profile_override is not None:
        raise ValueError("--profile is only valid with --calibration-mode custom")
    profile_path = (
        (profile_override or dataset.calibration_profile).expanduser().resolve()
        if selected_calibration_mode is CalibrationMode.CUSTOM
        else None
    )
    calibration = load_calibration_profile(profile_path) if profile_path is not None else None
    svo_path = dataset.resolve_svo_path(svo_override)
    zed_config = load_zed_config(zed_config_path) if zed_config_path else ZedSessionConfig()
    normalized_mode = str(mode).upper()
    if normalized_mode == "BOTH":
        mode_names = ("GEN_1", "GEN_3")
    elif normalized_mode in {"GEN_1", "GEN_3"}:
        mode_names = (normalized_mode,)
    else:
        raise ValueError("mode must be GEN_1, GEN_3, or both")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    root_metadata: dict[str, Any] = {
        "status": "running",
        "result_status": "experimental",
        "purpose": "production",
        "retain_policy": "keep",
        "dataset": dataset.dataset_id,
        "dataset_config": str(dataset_path),
        "svo_path": str(svo_path),
        "svo_identity": _svo_identity(svo_path),
        "zed_sdk_config": str(zed_config_path.resolve()) if zed_config_path else None,
        "tracking_mode": normalized_mode,
        "tracking_configs": {
            name: TrackingConfig(mode=name, enable_area_memory=enable_area_memory).__dict__
            for name in mode_names
        },
        "frame_range": {
            "requested_start_frame": start_frame,
            "requested_end_frame": end_frame,
            "max_frames": max_frames,
        },
        "calibration_mode": selected_calibration_mode.value,
        "calibration_source": (
            "svo_embedded" if selected_calibration_mode is CalibrationMode.NATIVE else "custom_profile"
        ),
        "coordinate_system": zed_config.coordinate_system,
        "translation_unit": "metre",
        "pose_reference_frame": "WORLD",
        "sequential_replay": True,
        "independent_replays": normalized_mode == "BOTH",
        "git_sha": _git_sha(),
        "started_at_utc": started_at,
        "software_versions": _software_versions(),
    }
    if profile_path is not None:
        root_metadata["custom_calibration_path"] = str(profile_path)
        root_metadata["custom_calibration_sha256"] = sha256_file(profile_path)
    _write_json(output_dir / "run.json", root_metadata)

    branch_records: dict[str, list[TrackingRecord]] = {}
    branch_summaries: dict[str, dict[str, Any]] = {}
    try:
        for name in mode_names:
            branch_dir = output_dir if len(mode_names) == 1 else output_dir / name.lower()
            branch_dir.mkdir(parents=True, exist_ok=True)
            branch_config = TrackingConfig(mode=name, enable_area_memory=enable_area_memory)
            branch_started = time.perf_counter()
            branch_metadata = dict(root_metadata)
            branch_metadata.update(
                {
                    "status": "running",
                    "tracking_mode": name,
                    "output_dir": str(branch_dir),
                    "started_at_utc": _utc_now(),
                    "software_versions": _software_versions(),
                }
            )
            _write_json(branch_dir / "run.json", branch_metadata)
            runtime_metadata: dict[str, Any] | None = None
            with ZedSession(
                svo_path,
                profile_path,
                zed_config,
                calibration_mode=selected_calibration_mode,
                expected_calibration=calibration,
            ) as session:
                records = replay_tracking(
                    session,
                    branch_config,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    max_frames=max_frames,
                )
                total_frames = session.svo_number_of_frames()
                runtime_metadata = session.runtime_metadata
                branch_metadata["camera"] = _camera_metadata(session.camera_information())
                branch_metadata["runtime_calibration"] = runtime_metadata
                branch_metadata["software_versions"] = _software_versions(session.sdk)
            summary = summarize_tracking(
                records,
                total_frames_reported=total_frames,
                config=branch_config,
                elapsed_seconds=time.perf_counter() - branch_started,
            )
            branch_metadata.update(
                {
                    "status": "completed",
                    "result_status": "complete",
                    "finished_at_utc": _utc_now(),
                    "summary_file": str(branch_dir / "summary.json"),
                }
            )
            write_tracking_outputs(branch_dir, records, summary, branch_metadata)
            branch_records[name] = records
            branch_summaries[name] = summary

        if normalized_mode == "BOTH":
            comparison = _compare_replays(
                branch_records["GEN_1"], branch_records["GEN_3"], output_dir / "comparison.csv"
            )
            root_summary: dict[str, Any] = {
                "status": "completed",
                "result_status": "complete",
                "mode": "BOTH",
                "sequential_replay": True,
                "independent_replays": True,
                "branches": branch_summaries,
                "comparison": comparison,
            }
        else:
            root_summary = branch_summaries[normalized_mode]
        root_metadata.update(
            {
                "status": "completed",
                "result_status": "complete",
                "finished_at_utc": _utc_now(),
                "summary_file": str(output_dir / "summary.json"),
                "software_versions": (
                    branch_metadata.get("software_versions", root_metadata["software_versions"])
                    if mode_names
                    else root_metadata["software_versions"]
                ),
            }
        )
        _write_json(output_dir / "summary.json", root_summary)
        _write_json(output_dir / "run.json", root_metadata)
        return root_summary
    except Exception as error:
        root_metadata.update(
            {
                "status": "failed",
                "result_status": "failed",
                "finished_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        _write_json(output_dir / "run.json", root_metadata)
        raise
