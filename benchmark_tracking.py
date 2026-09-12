"""Benchmark ZED positional-tracking generations on an SVO2 recording.

The recording is replayed strictly sequentially.  No ``set_svo_position`` is
used, because positional tracking needs consecutive frames to maintain its
state.  The script writes one pose CSV and one summary JSON for a selected
tracking generation.  It is intentionally independent of the point-cloud
export so that the comparison measures tracking rather than disk I/O for
images and depth maps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any


def _prepare_windows_dll_search_path() -> None:
    """Make the ZED SDK DLLs discoverable before importing ``pyzed.sl``."""

    if os.name != "nt":
        return

    sdk_root = os.environ.get("ZED_SDK_ROOT_DIR")
    if not sdk_root:
        return

    sdk_root_path = Path(sdk_root)
    search_paths = [
        sdk_root_path / "bin",
        sdk_root_path / "dependencies" / "freeglut" / "bin",
        sdk_root_path / "dependencies" / "glew" / "bin",
        sdk_root_path / "dependencies" / "opencv" / "x64" / "vc16" / "bin",
    ]
    existing_paths = [str(path) for path in search_paths if path.is_dir()]
    if not existing_paths:
        return

    current_path = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join(
        [*existing_paths, *(path for path in current_path if path)]
    )
    if hasattr(os, "add_dll_directory"):
        handles = [os.add_dll_directory(path) for path in existing_paths]
        _prepare_windows_dll_search_path._dll_handles = handles  # type: ignore[attr-defined]


_prepare_windows_dll_search_path()

import numpy as np  # noqa: E402
import pyzed.sl as sl  # noqa: E402


DEFAULT_SVO = Path(__file__).resolve().with_name("20260802_150233.svo2")
DEFAULT_OUTPUT_DIR = (
    DEFAULT_SVO.parent / "output" / "20260802_150233_tracking_ab"
)

POSE_HEADER = [
    "frame_index",
    "svo_position",
    "timestamp_ns",
    "tracking_state",
    "pose_valid",
    "pose_confidence",
    "tx",
    "ty",
    "tz",
    "qx",
    "qy",
    "qz",
    "qw",
    *(f"m{row}{column}" for row in range(4) for column in range(4)),
]


def _status_name(status: Any) -> str:
    return str(status).rsplit(".", 1)[-1]


def _check_status(status: Any, operation: str) -> None:
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"{operation} failed: {_status_name(status)}")


def _csv_number(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "NaN"
    return f"{value:.12g}"


def _pose_values(
    pose: sl.Pose,
    tracking_state: Any,
) -> tuple[bool, float, list[float], np.ndarray | None]:
    is_ok = tracking_state == sl.POSITIONAL_TRACKING_STATE.OK
    pose_valid = bool(pose.valid) if is_ok else False
    pose_confidence = float(pose.pose_confidence) if is_ok else float("nan")
    if not is_ok or not pose_valid:
        return pose_valid, pose_confidence, [float("nan")] * 23, None

    translation = np.asarray(
        pose.get_translation().get(), dtype=np.float64
    ).reshape(-1)
    orientation = np.asarray(
        pose.get_orientation().get(), dtype=np.float64
    ).reshape(-1)
    matrix = np.asarray(pose.pose_data().m, dtype=np.float64).reshape(4, 4)
    values = [*translation[:3], *orientation[:4], *matrix.reshape(-1)]
    if len(values) != 23:
        raise RuntimeError(f"Unexpected pose value count: {len(values)}")
    return pose_valid, pose_confidence, values, matrix


def _make_init(svo_path: Path, depth_mode: Any) -> sl.InitParameters:
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = depth_mode
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    return init


def _make_tracking_parameters(mode: Any, enable_area_memory: bool) -> Any:
    params = sl.PositionalTrackingParameters()
    params.mode = mode
    params.enable_area_memory = enable_area_memory
    params.enable_imu_fusion = True
    params.enable_pose_smoothing = False
    params.set_gravity_as_origin = True
    params.set_as_static = False
    return params


def _run_tracking(
    svo_path: Path,
    output_dir: Path,
    mode_name: str,
    mode: Any,
    enable_area_memory: bool,
    depth_mode_name: str,
    depth_mode: Any,
    max_source_frames: int,
) -> tuple[Path, dict[str, Any]]:
    zed = sl.Camera()
    status = zed.open(_make_init(svo_path, depth_mode))
    _check_status(status, "open SVO2")

    rows: list[list[Any]] = []
    matrices: list[np.ndarray | None] = []
    positions: list[int] = []
    info = None
    started = time.monotonic()
    try:
        info = zed.get_camera_information()
        total_frames = int(zed.get_svo_number_of_frames())
        params = _make_tracking_parameters(mode, enable_area_memory)
        _check_status(
            zed.enable_positional_tracking(params),
            f"enable positional tracking ({mode_name})",
        )

        runtime = sl.RuntimeParameters()
        pose = sl.Pose()
        frame_index = 0
        last_report = -1
        while True:
            status = zed.grab(runtime)
            if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            _check_status(status, f"grab ({mode_name})")

            source_position = int(zed.get_svo_position())
            timestamp_ns = int(
                zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            )
            tracking_state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
            pose_valid, pose_confidence, values, matrix = _pose_values(
                pose, tracking_state
            )
            rows.append(
                [
                    frame_index,
                    source_position,
                    timestamp_ns,
                    _status_name(tracking_state),
                    int(pose_valid),
                    _csv_number(pose_confidence),
                    *(_csv_number(value) for value in values),
                ]
            )
            matrices.append(matrix)
            positions.append(source_position)
            frame_index += 1

            if max_source_frames > 0 and frame_index >= max_source_frames:
                break

            if source_position // 1000 != last_report:
                last_report = source_position // 1000
                elapsed = time.monotonic() - started
                print(
                    f"{mode_name}: source frame {source_position}/{total_frames - 1} "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )
    finally:
        zed.close()

    if info is None or not rows:
        raise RuntimeError(f"No frames were returned for {mode_name}")

    csv_path = output_dir / f"pose_{mode_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as pose_file:
        writer = csv.writer(pose_file)
        writer.writerow(POSE_HEADER)
        writer.writerows(rows)

    summary = _summarize(
        svo_path=svo_path,
        info=info,
        rows=rows,
        matrices=matrices,
        positions=positions,
        mode_name=mode_name,
        enable_area_memory=enable_area_memory,
        depth_mode_name=depth_mode_name,
        total_frames_reported=total_frames,
        elapsed_seconds=time.monotonic() - started,
    )
    summary_path = output_dir / f"summary_{mode_name}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return csv_path, summary


def _camera_model(info: Any) -> str:
    return _status_name(info.camera_model)


def _summarize(
    svo_path: Path,
    info: Any,
    rows: list[list[Any]],
    matrices: list[np.ndarray | None],
    positions: list[int],
    mode_name: str,
    enable_area_memory: bool,
    depth_mode_name: str,
    total_frames_reported: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    states = [str(row[3]) for row in rows]
    valid_indices = [index for index, matrix in enumerate(matrices) if matrix is not None]
    valid_matrices = [matrices[index] for index in valid_indices]
    valid_positions = np.asarray(
        [[float(rows[index][6]), float(rows[index][7]), float(rows[index][8])] for index in valid_indices],
        dtype=np.float64,
    )

    summary: dict[str, Any] = {
        "input_file": str(svo_path),
        "sdk_version": str(sl.Camera.get_sdk_version()),
        "camera_model": _camera_model(info),
        "mode": mode_name,
        "depth_mode": depth_mode_name,
        "enable_area_memory": bool(enable_area_memory),
        "enable_imu_fusion": True,
        "enable_pose_smoothing": False,
        "coordinate_system": "RIGHT_HANDED_Y_UP",
        "coordinate_units": "METER",
        "pose_reference_frame": "WORLD",
        "pose_camera": "LEFT_EYE",
        "total_svo_frames_reported": int(total_frames_reported),
        "frames_replayed": len(rows),
        "tracking_state_counts": dict(sorted(Counter(states).items())),
        "valid_pose_frames": len(valid_indices),
        "valid_pose_ratio": len(valid_indices) / len(rows),
        "elapsed_seconds": float(elapsed_seconds),
    }

    confidences = np.asarray(
        [float(rows[index][5]) for index in valid_indices], dtype=np.float64
    )
    finite_confidences = confidences[np.isfinite(confidences)]
    if finite_confidences.size:
        summary["pose_confidence"] = {
            "min": float(np.min(finite_confidences)),
            "p05": float(np.percentile(finite_confidences, 5)),
            "median": float(np.median(finite_confidences)),
            "p95": float(np.percentile(finite_confidences, 95)),
            "max": float(np.max(finite_confidences)),
        }

    if len(valid_positions) >= 2:
        steps = np.linalg.norm(np.diff(valid_positions, axis=0), axis=1)
        summary["translation_trajectory"] = {
            "path_length_m": float(np.sum(steps)),
            "endpoint_displacement_m": float(
                np.linalg.norm(valid_positions[-1] - valid_positions[0])
            ),
            "position_min_m": valid_positions.min(axis=0).tolist(),
            "position_max_m": valid_positions.max(axis=0).tolist(),
            "step_p50_m": float(np.percentile(steps, 50)),
            "step_p95_m": float(np.percentile(steps, 95)),
            "step_max_m": float(np.max(steps)),
        }

        gaps = np.diff(np.asarray([positions[index] for index in valid_indices]))
        summary["valid_frame_gap"] = {
            "max_source_frame_gap": int(np.max(gaps)),
            "non_consecutive_pairs": int(np.count_nonzero(gaps != 1)),
        }

    if valid_matrices and len(valid_matrices) >= 2:
        angles = []
        for first, second in zip(valid_matrices[:-1], valid_matrices[1:]):
            relative = first[:3, :3].T @ second[:3, :3]
            cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
            angles.append(math.degrees(math.acos(float(cosine))))
        summary["rotation_step_deg"] = {
            "p50": float(np.percentile(angles, 50)),
            "p95": float(np.percentile(angles, 95)),
            "max": float(np.max(angles)),
        }

    return summary


def _write_comparison(
    output_dir: Path,
    gen1_rows: list[list[str]],
    gen3_rows: list[list[str]],
) -> dict[str, Any]:
    if len(gen1_rows) != len(gen3_rows):
        raise RuntimeError("GEN_1 and GEN_3 CSV row counts differ")

    comparison_path = output_dir / "comparison_gen1_gen3.csv"
    header = [
        "frame_index",
        "svo_position",
        "timestamp_ns",
        "gen1_tracking_state",
        "gen3_tracking_state",
        "both_pose_valid",
        "translation_difference_m",
        "rotation_difference_deg",
    ]
    translation_differences: list[float] = []
    rotation_differences: list[float] = []
    with comparison_path.open("w", newline="", encoding="utf-8") as comparison_file:
        writer = csv.writer(comparison_file)
        writer.writerow(header)
        for gen1, gen3 in zip(gen1_rows, gen3_rows):
            if gen1[0] != gen3[0] or gen1[1] != gen3[1] or gen1[2] != gen3[2]:
                raise RuntimeError("GEN_1 and GEN_3 frame alignment differs")
            both_valid = gen1[4] == "1" and gen3[4] == "1"
            translation_difference = float("nan")
            rotation_difference = float("nan")
            if both_valid:
                t1 = np.asarray([float(gen1[6]), float(gen1[7]), float(gen1[8])])
                t3 = np.asarray([float(gen3[6]), float(gen3[7]), float(gen3[8])])
                translation_difference = float(np.linalg.norm(t1 - t3))
                m1 = np.asarray([float(value) for value in gen1[13:29]]).reshape(4, 4)
                m3 = np.asarray([float(value) for value in gen3[13:29]]).reshape(4, 4)
                relative = m1[:3, :3].T @ m3[:3, :3]
                cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
                rotation_difference = math.degrees(math.acos(float(cosine)))
                translation_differences.append(translation_difference)
                rotation_differences.append(rotation_difference)
            writer.writerow(
                [
                    gen1[0],
                    gen1[1],
                    gen1[2],
                    gen1[3],
                    gen3[3],
                    int(both_valid),
                    _csv_number(translation_difference),
                    _csv_number(rotation_difference),
                ]
            )

    result: dict[str, Any] = {
        "file": str(comparison_path),
        "aligned_rows": len(gen1_rows),
        "both_valid_rows": len(translation_differences),
    }
    if translation_differences:
        result["translation_difference_m"] = {
            "median": float(np.median(translation_differences)),
            "p95": float(np.percentile(translation_differences, 95)),
            "max": float(np.max(translation_differences)),
        }
        result["rotation_difference_deg"] = {
            "median": float(np.median(rotation_differences)),
            "p95": float(np.percentile(rotation_differences, 95)),
            "max": float(np.max(rotation_differences)),
        }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequentially benchmark ZED GEN_1 and GEN_3 tracking on an SVO2."
    )
    parser.add_argument("svo", nargs="?", type=Path, default=DEFAULT_SVO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mode",
        choices=("gen1", "gen3", "both"),
        default="both",
        help="Tracking generation to run (default: both).",
    )
    parser.add_argument(
        "--area-memory",
        action="store_true",
        help="Enable Area Memory for GEN_3; omitted for a pure-VIO comparison.",
    )
    parser.add_argument(
        "--depth-mode",
        choices=("none", "performance", "quality", "neural"),
        default="neural",
        help="Depth mode used while replaying (default: neural).",
    )
    parser.add_argument(
        "--max-source-frames",
        type=int,
        default=0,
        help="For a smoke test, stop after this many frames; 0 means full SVO.",
    )
    args = parser.parse_args()
    if args.max_source_frames < 0:
        parser.error("--max-source-frames must be >= 0")
    return args


def _load_csv_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        if header != POSE_HEADER:
            raise RuntimeError(f"Unexpected header in {path}")
        return list(reader)


def main() -> int:
    args = _parse_args()
    svo_path = args.svo.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not svo_path.is_file():
        raise FileNotFoundError(f"SVO/SVO2 file not found: {svo_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = []
    if args.mode in ("gen1", "both"):
        modes.append(("gen1", sl.POSITIONAL_TRACKING_MODE.GEN_1, False))
    if args.mode in ("gen3", "both"):
        modes.append(("gen3", sl.POSITIONAL_TRACKING_MODE.GEN_3, args.area_memory))

    depth_modes = {
        "none": sl.DEPTH_MODE.NONE,
        "performance": sl.DEPTH_MODE.PERFORMANCE,
        "quality": sl.DEPTH_MODE.QUALITY,
        "neural": sl.DEPTH_MODE.NEURAL,
    }
    depth_mode = depth_modes[args.depth_mode]

    results: dict[str, Any] = {
        "sdk_version": str(sl.Camera.get_sdk_version()),
        "input_file": str(svo_path),
        "coordinate_system": "RIGHT_HANDED_Y_UP",
        "coordinate_units": "METER",
        "pose_reference_frame": "WORLD",
        "sequential_replay": True,
        "depth_mode": args.depth_mode.upper(),
        "max_source_frames": args.max_source_frames,
        "requested_modes": [name for name, _, _ in modes],
    }

    for mode_name, mode, area_memory in modes:
        csv_path, summary = _run_tracking(
            svo_path=svo_path,
            output_dir=output_dir,
            mode_name=mode_name,
            mode=mode,
            enable_area_memory=area_memory,
            depth_mode_name=args.depth_mode.upper(),
            depth_mode=depth_mode,
            max_source_frames=args.max_source_frames,
        )
        results[mode_name] = {
            "pose_csv": str(csv_path),
            "summary": summary,
        }

    if args.mode == "both":
        gen1_rows = _load_csv_rows(output_dir / "pose_gen1.csv")
        gen3_rows = _load_csv_rows(output_dir / "pose_gen3.csv")
        results["comparison"] = _write_comparison(
            output_dir, gen1_rows=gen1_rows, gen3_rows=gen3_rows
        )

    results_path = output_dir / "experiment_summary.json"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Experiment summary: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
