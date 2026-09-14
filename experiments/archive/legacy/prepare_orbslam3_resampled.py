"""Build exact-frame ORB-SLAM3/Metashape exports from a stable SVO run.

The workflow has two deliberately separate stages:

* ``analyze`` finds the longest *continuous* run whose ORB-SLAM3 tracking
  state is valid.
* ``build`` consumes a bounded ORB-SLAM3 replay of that run, replays the SVO
  sequentially, saves the requested unrectified full-resolution stereo
  images, writes left/right Metashape YPR references, and colorizes the
  bounded replay's sparse MapPoints.

All camera calibration used for pose geometry, image projection, and the
ORB-SLAM3 settings is read from the independent ``Calibration/`` directory.
The SVO is used as an image/timestamp source only; its embedded calibration is
never used for the generated outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]


def _prepare_windows_dll_search_path() -> None:
    """Make ZED SDK DLLs discoverable before importing ``pyzed.sl``."""

    if os.name != "nt":
        return

    sdk_root = os.environ.get("ZED_SDK_ROOT_DIR") or r"C:\Program Files (x86)\ZED SDK.old"
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from prepare_orbslam3_stereo import read_calibration  # noqa: E402


VALID_TRACKING_STATES = {2, 5}  # ORB-SLAM3::Tracking::OK / OK_KLT
PLY_POINT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)
assert PLY_POINT_DTYPE.itemsize == 15


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _status_name(status: Any) -> str:
    return str(status).rsplit(".", 1)[-1]


def _check_status(status: Any, operation: str, sl: Any) -> None:
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"{operation} failed: {_status_name(status)}")


def _read_tracking_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    rows: list[dict[str, Any]] = []
    seen_processed: set[int] = set()
    seen_source: set[int] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"frame", "timestamp_sec", "tracking_state", "pose_valid"}
        missing = required.difference(fields)
        if missing:
            raise ValueError(f"{path} is missing tracking columns: {sorted(missing)}")

        for line_number, raw in enumerate(reader, start=2):
            try:
                processed_frame = int(raw["frame"])
                source_frame = int(raw.get("svo_position") or processed_frame)
                timestamp_sec = float(raw["timestamp_sec"])
                tracking_state = int(raw["tracking_state"])
                pose_valid = str(raw["pose_valid"]).strip() == "1"
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(f"{path}:{line_number}: invalid tracking row") from error

            if processed_frame in seen_processed:
                raise ValueError(f"{path}:{line_number}: duplicate processed frame {processed_frame}")
            if source_frame in seen_source:
                raise ValueError(f"{path}:{line_number}: duplicate SVO frame {source_frame}")
            if not math.isfinite(timestamp_sec):
                raise ValueError(f"{path}:{line_number}: non-finite timestamp")

            row: dict[str, Any] = {
                "processed_frame": processed_frame,
                "source_frame": source_frame,
                "svo_position": source_frame,
                "timestamp_sec": timestamp_sec,
                "tracking_state": tracking_state,
                "pose_valid_flag": pose_valid,
                "valid": pose_valid and tracking_state in VALID_TRACKING_STATES,
                "tracked_map_points": int(raw.get("tracked_map_points") or 0),
                "width": int(raw.get("width") or 0),
                "height": int(raw.get("height") or 0),
            }
            rows.append(row)
            seen_processed.add(processed_frame)
            seen_source.add(source_frame)

    rows.sort(key=lambda item: int(item["source_frame"]))
    return rows


def _find_valid_runs(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_source: int | None = None
    for row in rows:
        source_frame = int(row["source_frame"])
        if not row["valid"]:
            if current:
                runs.append(current)
                current = []
            previous_source = None
            continue

        if previous_source is None or source_frame != previous_source + 1:
            if current:
                runs.append(current)
            current = [row]
        else:
            current.append(row)
        previous_source = source_frame

    if current:
        runs.append(current)
    return runs


def _segment_summary(run: list[dict[str, Any]]) -> dict[str, Any]:
    if not run:
        raise ValueError("cannot summarize an empty segment")
    return {
        "start_frame": int(run[0]["source_frame"]),
        "end_frame": int(run[-1]["source_frame"]),
        "length": len(run),
        "start_processed_frame": int(run[0]["processed_frame"]),
        "end_processed_frame": int(run[-1]["processed_frame"]),
        "start_timestamp_sec": float(run[0]["timestamp_sec"]),
        "end_timestamp_sec": float(run[-1]["timestamp_sec"]),
    }


def analyze(tracking_log: Path, output_plan: Path, min_required: int) -> dict[str, Any]:
    if min_required <= 0:
        raise ValueError("min_required must be positive")
    rows = _read_tracking_log(tracking_log)
    runs = _find_valid_runs(rows)
    if not runs:
        raise RuntimeError("tracking log contains no continuous valid segment")

    # The earliest segment wins ties, making reruns deterministic.
    best = max(runs, key=lambda run: (len(run), -int(run[0]["source_frame"])))
    selected = _segment_summary(best)
    summaries = [_segment_summary(run) for run in runs]
    state_counts = Counter(str(row["tracking_state"]) for row in rows)

    plan: dict[str, Any] = {
        "schema_version": 1,
        "source_tracking_log": str(tracking_log.resolve()),
        "validity_rule": {
            "pose_valid": "1",
            "tracking_states": sorted(VALID_TRACKING_STATES),
            "description": "pose_valid == 1 and tracking_state in {2 (OK), 5 (OK_KLT)}",
        },
        "tracking_log": {
            "rows": len(rows),
            "valid_rows": int(sum(1 for row in rows if row["valid"])),
            "tracking_state_counts": dict(sorted(state_counts.items())),
        },
        "continuous_segments": summaries,
        "selected_segment": selected,
        "requested_sample_counts": [1500, 3000],
        "minimum_required_frames": min_required,
        "safe_for_requested_samples": bool(selected["length"] >= min_required),
    }
    if selected["length"] < min_required:
        _write_json(output_plan, plan)
        raise RuntimeError(
            "longest continuous valid segment has "
            f"{selected['length']} frames, fewer than the required {min_required}; "
            "3000-frame export is refused"
        )

    _write_json(output_plan, plan)
    print(
        "Selected continuous segment: "
        f"{selected['start_frame']}..{selected['end_frame']} "
        f"({selected['length']} frames)"
    )
    print(f"Wrote segment plan: {output_plan.resolve()}")
    return plan


def _round_timestamp(timestamp_sec: float) -> float:
    return round(float(timestamp_sec), 6)


def _quaternion_to_rotation(qx: float, qy: float, qz: float, qw: float) -> tuple[np.ndarray, float]:
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    if not np.all(np.isfinite(quaternion)):
        raise ValueError("trajectory contains a non-finite quaternion")
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"invalid quaternion norm: {norm}")
    x, y, z, w = quaternion / norm
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rotation, norm


def _metashape_ypr(rotation: np.ndarray) -> tuple[float, float, float]:
    """Agisoft's published yaw/pitch/roll extraction convention, degrees."""

    r21 = float(np.clip(rotation[2, 1], -1.0, 1.0))
    if r21 > 0.999:
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        pitch = math.pi / 2.0
        roll = 0.0
    elif r21 < -0.999:
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        pitch = -math.pi / 2.0
        roll = 0.0
    else:
        yaw = math.atan2(-float(rotation[0, 1]), float(rotation[1, 1]))
        roll = math.atan2(-float(rotation[2, 0]), float(rotation[2, 2]))
        pitch = math.asin(r21)
    if yaw > 0.0:
        yaw -= 2.0 * math.pi
    return -math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _read_trajectory(path: Path) -> dict[float, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[float, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 8:
                raise ValueError(f"{path}:{line_number}: expected 8 TUM fields")
            values = [float(value) for value in fields[:8]]
            timestamp, tx, ty, tz, qx, qy, qz, qw = values
            key = _round_timestamp(timestamp)
            if key in result:
                raise ValueError(f"{path}:{line_number}: duplicate timestamp {key}")
            rotation, quaternion_norm = _quaternion_to_rotation(qx, qy, qz, qw)
            translation = np.asarray([tx, ty, tz], dtype=np.float64)
            if not np.all(np.isfinite(translation)):
                raise ValueError(f"{path}:{line_number}: non-finite translation")
            result[key] = {
                "timestamp_sec": timestamp,
                "translation": translation,
                "rotation": rotation,
                "quaternion_norm": quaternion_norm,
            }
    if not result:
        raise ValueError(f"trajectory has no pose rows: {path}")
    return result


def _sample_rows(run: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("sample count must be positive")
    if len(run) < count:
        raise RuntimeError(
            f"requested {count} frames but stable continuous segment has only {len(run)}"
        )
    if count == 1:
        indices = np.asarray([0], dtype=np.int64)
    else:
        # Floor sampling includes both endpoints and cannot duplicate indices
        # when len(run) >= count.
        indices = np.floor(
            np.arange(count, dtype=np.float64) * (len(run) - 1) / (count - 1)
        ).astype(np.int64)
    if len(np.unique(indices)) != count:
        raise RuntimeError(f"sampling produced duplicate indices for count={count}")
    return [run[int(index)] for index in indices]


def _attach_poses(
    rows: list[dict[str, Any]], trajectory: dict[float, dict[str, Any]]
) -> list[dict[str, Any]]:
    attached: list[dict[str, Any]] = []
    unmatched: list[int] = []
    for row in rows:
        pose = trajectory.get(_round_timestamp(float(row["timestamp_sec"])))
        if pose is None:
            unmatched.append(int(row["source_frame"]))
            continue
        item = dict(row)
        item["translation"] = pose["translation"].copy()
        item["rotation"] = pose["rotation"].copy()
        item["quaternion_norm"] = float(pose["quaternion_norm"])
        item["timestamp_trajectory_sec"] = float(pose["timestamp_sec"])
        attached.append(item)
    if unmatched:
        raise RuntimeError(
            f"{len(unmatched)} valid stable frames are missing from CameraTrajectory.txt; "
            f"first missing source frames: {unmatched[:10]}"
        )
    return attached


def _ensure_sample_directory(sample_dir: Path) -> tuple[Path, Path]:
    if sample_dir.exists():
        existing = [path for path in sample_dir.rglob("*") if path.is_file()]
        if existing:
            raise FileExistsError(
                f"refusing to mix with existing sample output ({len(existing)} files): {sample_dir}"
            )
    left_dir = sample_dir / "left"
    right_dir = sample_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    return left_dir, right_dir


def _as_bgr(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _import_zed() -> Any:
    # Importing pyzed is delayed so ``analyze`` remains useful even on a host
    # that has the CSV tools but not the ZED SDK Python bindings.
    import pyzed.sl as sl  # type: ignore

    return sl


def _save_target_images(
    svo_path: Path,
    targets_by_source: dict[int, dict[str, Any]],
    sample_dirs: dict[int, tuple[Path, Path]],
) -> dict[int, dict[str, Path]]:
    sl = _import_zed()
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.camera_disable_self_calib = True
    init.depth_mode = sl.DEPTH_MODE.NONE

    zed = sl.Camera()
    _check_status(zed.open(init), "open SVO for unrectified image extraction", sl)
    left_mat = sl.Mat()
    right_mat = sl.Mat()
    left_view = getattr(sl.VIEW, "LEFT_UNRECTIFIED", sl.VIEW.LEFT)
    right_view = getattr(sl.VIEW, "RIGHT_UNRECTIFIED", sl.VIEW.RIGHT)
    saved: dict[int, dict[str, Path]] = {}
    total_targets = len(targets_by_source)
    try:
        info = zed.get_camera_information()
        source_resolution = info.camera_configuration.resolution
        expected_size = (int(source_resolution.width), int(source_resolution.height))
        if expected_size != (1920, 1080):
            raise RuntimeError(
                "full-resolution SVO is not 1920x1080: "
                f"{expected_size[0]}x{expected_size[1]}"
            )

        while len(saved) < total_targets:
            status = zed.grab()
            if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            _check_status(status, "grab during sequential image extraction", sl)
            source_frame = int(zed.get_svo_position())
            target = targets_by_source.get(source_frame)
            if target is None:
                continue

            _check_status(
                zed.retrieve_image(left_mat, left_view),
                f"retrieve unrectified left image at SVO frame {source_frame}",
                sl,
            )
            _check_status(
                zed.retrieve_image(right_mat, right_view),
                f"retrieve unrectified right image at SVO frame {source_frame}",
                sl,
            )
            left = _as_bgr(left_mat.get_data())
            right = _as_bgr(right_mat.get_data())
            if left.ndim != 3 or right.ndim != 3 or left.shape[2] != 3 or right.shape[2] != 3:
                raise RuntimeError(f"unexpected unrectified image shape at frame {source_frame}")
            if left.shape[:2] != (1080, 1920) or right.shape[:2] != (1080, 1920):
                raise RuntimeError(
                    f"unrectified image at frame {source_frame} is "
                    f"{left.shape[1]}x{left.shape[0]} / {right.shape[1]}x{right.shape[0]}"
                )

            left_dir, right_dir = sample_dirs[id(target["sample_owner"])]
            label_frame = int(target["source_frame"])
            left_path = left_dir / f"left_{label_frame:06d}.png"
            right_path = right_dir / f"right_{label_frame:06d}.png"
            compression = [cv2.IMWRITE_PNG_COMPRESSION, 3]
            if not cv2.imwrite(str(left_path), np.ascontiguousarray(left), compression):
                raise RuntimeError(f"could not write {left_path}")
            if not cv2.imwrite(str(right_path), np.ascontiguousarray(right), compression):
                raise RuntimeError(f"could not write {right_path}")
            saved[source_frame] = {"left": left_path, "right": right_path}
            if len(saved) % 100 == 0 or len(saved) == total_targets:
                print(f"Saved unrectified image pairs: {len(saved)}/{total_targets}", flush=True)
    finally:
        zed.close()

    missing = sorted(set(targets_by_source).difference(saved))
    if missing:
        raise RuntimeError(
            f"SVO ended before {len(missing)} requested image frames; first missing: {missing[:10]}"
        )
    return saved


def _homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def _validate_rotation(rotation: np.ndarray) -> tuple[float, float]:
    orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    determinant_error = abs(float(np.linalg.det(rotation)) - 1.0)
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation contains non-finite values")
    if orthogonality_error > 1e-4 or determinant_error > 1e-4:
        raise ValueError(
            "pose rotation is not a proper rotation: "
            f"orthogonality_error={orthogonality_error}, determinant_error={determinant_error}"
        )
    return orthogonality_error, determinant_error


def _calibration_metadata(values: dict[str, Any], rotation: np.ndarray, translation_mm: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, list):
            result[key] = [float(item) for item in value]
        else:
            result[key] = float(value)
    result["stereo_rotation_left_to_right"] = rotation.tolist()
    result["stereo_translation_left_to_right_mm"] = [float(item) for item in translation_mm]
    result["baseline_norm_mm"] = float(np.linalg.norm(translation_mm))
    return result


def _write_manifest(sample_dir: Path, entries: list[dict[str, Any]]) -> Path:
    path = sample_dir / "frame_manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "sample_index",
                "source_frame",
                "processed_frame",
                "svo_position",
                "timestamp_sec",
                "tracking_state",
                "pose_valid",
                "left_image",
                "right_image",
            ]
        )
        for index, entry in enumerate(entries):
            writer.writerow(
                [
                    index,
                    int(entry["source_frame"]),
                    int(entry["processed_frame"]),
                    int(entry["svo_position"]),
                    f"{float(entry['timestamp_sec']):.12f}",
                    int(entry["tracking_state"]),
                    1 if entry["valid"] else 0,
                    entry["left_image"],
                    entry["right_image"],
                ]
            )
    return path


def _write_metashape_reference(
    sample_dir: Path,
    entries: list[dict[str, Any]],
    right_to_left: np.ndarray,
) -> tuple[Path, dict[str, Any]]:
    path = sample_dir / "metashape_reference_ypr.csv"
    max_orthogonality_error = 0.0
    max_determinant_error = 0.0
    ypr_values: list[float] = []
    left_right_center_distances: list[float] = []

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["label", "x", "y", "z", "yaw", "pitch", "roll"])
        for entry in entries:
            left_rotation = np.asarray(entry["rotation"], dtype=np.float64)
            left_translation = np.asarray(entry["translation"], dtype=np.float64)
            left_orth, left_det = _validate_rotation(left_rotation)
            left_pose = _homogeneous(left_rotation, left_translation)
            right_pose = left_pose @ right_to_left
            right_rotation = right_pose[:3, :3]
            right_translation = right_pose[:3, 3]
            right_orth, right_det = _validate_rotation(right_rotation)
            max_orthogonality_error = max(max_orthogonality_error, left_orth, right_orth)
            max_determinant_error = max(max_determinant_error, left_det, right_det)

            left_ypr = _metashape_ypr(left_rotation)
            right_ypr = _metashape_ypr(right_rotation)
            ypr_values.extend([*left_ypr, *right_ypr])
            left_right_center_distances.append(float(np.linalg.norm(right_translation - left_translation)))
            frame = int(entry["source_frame"])
            for side, pose, ypr in (("left", left_pose, left_ypr), ("right", right_pose, right_ypr)):
                values = [*pose[:3, 3], *ypr]
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"non-finite Metashape pose at source frame {frame}, {side}")
                writer.writerow(
                    [
                        f"{side}_{frame:06d}.png",
                        f"{float(values[0]):.12f}",
                        f"{float(values[1]):.12f}",
                        f"{float(values[2]):.12f}",
                        f"{float(values[3]):.12f}",
                        f"{float(values[4]):.12f}",
                        f"{float(values[5]):.12f}",
                    ]
                )

    if not ypr_values or not np.all(np.isfinite(ypr_values)):
        raise ValueError("Metashape YPR output contains no finite values")
    expected_baseline_m = float(np.linalg.norm(right_to_left[:3, 3]))
    actual_min = min(left_right_center_distances)
    actual_max = max(left_right_center_distances)
    quality = {
        "csv_rows": len(entries) * 2,
        "label_rule": "left_{source_frame:06d}.png and right_{source_frame:06d}.png",
        "angle_order": "yaw,pitch,roll",
        "angle_units": "degrees",
        "coordinates": "ORB-SLAM3 local metric coordinates; no GPS/geographic reference",
        "max_rotation_orthogonality_error": max_orthogonality_error,
        "max_rotation_determinant_error": max_determinant_error,
        "calibrated_baseline_norm_m": expected_baseline_m,
        "pose_center_baseline_min_m": actual_min,
        "pose_center_baseline_max_m": actual_max,
        "pose_center_baseline_max_abs_error_m": max(abs(actual_min - expected_baseline_m), abs(actual_max - expected_baseline_m)),
    }
    return path, quality


def _read_map_points(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    points: list[list[float]] = []
    observations: list[int] = []
    invalid = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"map_point_id", "map_id", "observations", "x", "y", "z"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"MapPoint CSV is missing columns: {sorted(missing)}")
        for row in reader:
            try:
                point = [float(row["x"]), float(row["y"]), float(row["z"])]
                obs = int(row["observations"])
            except (TypeError, ValueError, KeyError):
                invalid += 1
                continue
            if not np.all(np.isfinite(point)):
                invalid += 1
                continue
            points.append(point)
            observations.append(obs)
    if not points:
        raise RuntimeError(f"MapPoint CSV contains no finite valid points: {path}")
    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(observations, dtype=np.int64),
        {"rows_invalid": invalid, "rows_valid": len(points)},
    )


def _colorize_map_points(
    points: np.ndarray,
    entries: list[dict[str, Any]],
    sample3000_dir: Path,
    values: dict[str, Any],
    output_ply: Path,
    max_color_observations: int = 16,
) -> dict[str, Any]:
    if max_color_observations <= 0:
        raise ValueError("max_color_observations must be positive")
    fx = float(values["left_fx"])
    fy = float(values["left_fy"])
    cx = float(values["left_cx"])
    cy = float(values["left_cy"])
    distortion = np.asarray(values["left_distortion"], dtype=np.float64).reshape(-1)
    if distortion.size < 5:
        raise ValueError("left independent calibration must contain five distortion coefficients")
    k1, k2, p1, p2, k3 = [float(item) for item in distortion[:5]]

    color_sum = np.zeros((len(points), 3), dtype=np.float64)
    color_count = np.zeros(len(points), dtype=np.int32)
    projection_count = 0
    used_images = 0
    for image_index, entry in enumerate(entries):
        eligible = np.flatnonzero(color_count < max_color_observations)
        if eligible.size == 0:
            break
        image_path = sample3000_dir / Path(str(entry["left_image"]))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not read saved left image for coloring: {image_path}")
        if image.shape[:2] != (1080, 1920):
            raise RuntimeError(f"saved left image has unexpected size: {image_path} -> {image.shape}")

        rotation = np.asarray(entry["rotation"], dtype=np.float64)
        translation = np.asarray(entry["translation"], dtype=np.float64)
        # T_wc maps camera coordinates to the ORB local world.  With row
        # vectors, camera coordinates are (world - t) @ R_wc.
        camera = (points[eligible] - translation) @ rotation
        z = camera[:, 2]
        valid = np.isfinite(camera).all(axis=1) & (z > 1e-8)
        if not np.any(valid):
            used_images += 1
            continue
        valid_indices = eligible[valid]
        camera_valid = camera[valid]
        x = camera_valid[:, 0] / camera_valid[:, 2]
        y = camera_valid[:, 1] / camera_valid[:, 2]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        u = np.rint(fx * xd + cx).astype(np.int64)
        v = np.rint(fy * yd + cy).astype(np.int64)
        in_image = (u >= 0) & (u < image.shape[1]) & (v >= 0) & (v < image.shape[0])
        if np.any(in_image):
            target_indices = valid_indices[in_image]
            pixels_rgb = image[v[in_image], u[in_image]][:, ::-1].astype(np.float64)
            np.add.at(color_sum, target_indices, pixels_rgb)
            np.add.at(color_count, target_indices, 1)
            projection_count += int(target_indices.size)
        used_images += 1
        if (image_index + 1) % 250 == 0 or image_index + 1 == len(entries):
            print(
                f"Colorized MapPoints with left images: {image_index + 1}/{len(entries)} "
                f"(covered {int(np.count_nonzero(color_count))}/{len(points)})",
                flush=True,
            )

    colors = np.full((len(points), 3), 160, dtype=np.uint8)
    colored = color_count > 0
    if np.any(colored):
        mean_colors = np.rint(color_sum[colored] / color_count[colored, None]).clip(0, 255)
        colors[colored] = mean_colors.astype(np.uint8)

    records = np.empty(len(points), dtype=PLY_POINT_DTYPE)
    records["x"] = points[:, 0].astype(np.float32)
    records["y"] = points[:, 1].astype(np.float32)
    records["z"] = points[:, 2].astype(np.float32)
    records["red"] = colors[:, 0]
    records["green"] = colors[:, 1]
    records["blue"] = colors[:, 2]
    if not np.isfinite(points).all():
        raise ValueError("non-finite MapPoint encountered while writing PLY")

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(records)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    with output_ply.open("wb") as handle:
        handle.write(header)
        records.tofile(handle)

    colored_count = int(np.count_nonzero(colored))
    return {
        "ply_vertices": len(records),
        "colored_points": colored_count,
        "uncolored_points": len(records) - colored_count,
        "color_coverage": colored_count / len(records),
        "color_samples": int(color_count.sum()),
        "projection_samples_in_image": projection_count,
        "images_used": used_images,
        "max_color_observations_per_point": max_color_observations,
        "uncolored_rgb": [160, 160, 160],
        "projection_model": "independent left pinhole intrinsics plus Brown-Conrady k1,k2,p1,p2,k3",
        "projection_image": "saved full-resolution left_<source_frame>.png, unrectified",
        "ply_format": "binary_little_endian; float32 xyz; uchar RGB",
    }


def _build_sample_metadata(
    sample_dir: Path,
    count: int,
    entries: list[dict[str, Any]],
    segment: dict[str, Any],
    svo_path: Path,
    calibration_root: Path,
    calibration_values: dict[str, Any],
    metashape_path: Path,
    metashape_quality: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sample_count": count,
        "source": {
            "svo2": str(svo_path.resolve()),
            "stable_run": str((sample_dir.parent / "stable_map").resolve()),
            "segment": segment,
            "sampling": "uniform floor indices including both continuous-segment endpoints",
            "source_frame_field": "tracking_log.csv:svo_position",
        },
        "images": {
            "left_count": count,
            "right_count": count,
            "resolution": {"width": 1920, "height": 1080},
            "format": "PNG",
            "color_order": "BGR source converted to normal 3-channel image by OpenCV; PNG is not rectified",
            "unrectified": True,
            "left_directory": str((sample_dir / "left").resolve()),
            "right_directory": str((sample_dir / "right").resolve()),
        },
        "calibration": {
            "root": str(calibration_root.resolve()),
            "source_files": [
                str((calibration_root / "标定结果" / "camera_intrinsics.yaml").resolve()),
                str((calibration_root / "标定结果" / "stereo_extrinsics.yaml").resolve()),
            ],
            "values": _calibration_metadata(
                calibration_values["values"],
                calibration_values["rotation"],
                calibration_values["translation_mm"],
            ),
        },
        "pose": {
            "left_transform": "ORB-SLAM3 CameraTrajectory.txt Twc",
            "right_transform": "Twc_left @ inverse(T_left_to_right)",
            "world_frame": "ORB-SLAM3 local metric world frame; no GPS/geographic conversion",
            "camera_axes": "+X right, +Y down, +Z forward (OpenCV camera convention)",
            "ypr_convention": "Agisoft Metashape mat2ypr convention",
            "ypr_order": "yaw,pitch,roll",
            "ypr_units": "degrees",
        },
        "quality": metashape_quality,
        "files": {
            "frame_manifest": str(manifest_path.resolve()),
            "metashape_reference_ypr": str(metashape_path.resolve()),
        },
        "selected_frame_range": {
            "first_source_frame": int(entries[0]["source_frame"]),
            "last_source_frame": int(entries[-1]["source_frame"]),
        },
    }


def _write_import_instructions(output_root: Path) -> Path:
    path = output_root / "metashape_import_instructions.txt"
    text = """Metashape import instructions for ORB-SLAM3 resampled stereo

1. Add all PNGs under sample1500/left, sample1500/right (or sample3000/left,
   sample3000/right) to the Metashape chunk. Keep the original file names.
2. Import the corresponding metashape_reference_ypr.csv as camera reference
   data. Map label to the image label/name and map x,y,z,yaw,pitch,roll to the
   corresponding columns. The labels include the .png extension and match the
   files exactly.
3. Select the [yaw,pitch,roll] angle order in the import dialog.
4. Use Local coordinates. These are ORB-SLAM3's local metric map coordinates,
   not GPS or a geographic CRS. Do not apply a ZED axis flip.
5. The CSV contains both left and right camera poses. The right pose is
   computed from the independent Calibration/ stereo left-to-right transform.

The sparse ORB-SLAM3 point cloud is stable_map/orbslam3_map_points_rgb.ply.
Its coordinates use the same local frame as stable_map/CameraTrajectory.txt.
"""
    path.write_text(text, encoding="utf-8")
    return path


def build(
    stable_dir: Path,
    plan_path: Path,
    svo_path: Path,
    calibration_root: Path,
    output_root: Path,
    min_observations: int = 2,
) -> dict[str, Any]:
    stable_dir = stable_dir.resolve()
    plan_path = plan_path.resolve()
    svo_path = svo_path.resolve()
    calibration_root = calibration_root.resolve()
    output_root = output_root.resolve()
    if not svo_path.is_file():
        raise FileNotFoundError(svo_path)
    if not calibration_root.is_dir():
        raise FileNotFoundError(calibration_root)
    with plan_path.open("r", encoding="utf-8") as handle:
        plan = json.load(handle)
    segment = dict(plan["selected_segment"])
    segment_start = int(segment["start_frame"])
    segment_end = int(segment["end_frame"])
    segment_length = segment_end - segment_start + 1
    if segment_length < 3000:
        raise RuntimeError(
            f"segment plan has only {segment_length} frames; refusing 3000-frame output"
        )

    tracking_path = stable_dir / "tracking_log.csv"
    trajectory_path = stable_dir / "CameraTrajectory.txt"
    map_points_path = stable_dir / "map_points_xyz.csv"
    stable_rows = _read_tracking_log(tracking_path)
    stable_by_source = {int(row["source_frame"]): row for row in stable_rows}
    expected_sources = list(range(segment_start, segment_end + 1))
    missing_sources = [source for source in expected_sources if source not in stable_by_source]
    invalid_sources = [
        source for source in expected_sources if source in stable_by_source and not stable_by_source[source]["valid"]
    ]
    if missing_sources or invalid_sources:
        raise RuntimeError(
            "bounded stable replay does not reproduce the planned continuous segment: "
            f"missing={len(missing_sources)}, invalid={len(invalid_sources)}; "
            f"first missing={missing_sources[:5]}, first invalid={invalid_sources[:5]}"
        )
    stable_run = [stable_by_source[source] for source in expected_sources]
    trajectory = _read_trajectory(trajectory_path)
    stable_run = _attach_poses(stable_run, trajectory)

    values, rotation_left_to_right, translation_left_to_right_mm = read_calibration(calibration_root.parent)
    # read_calibration expects the workspace root, so make the explicit
    # calibration-root relationship visible and verify it is the requested one.
    expected_calibration_root = calibration_root
    if expected_calibration_root.name != "Calibration":
        raise ValueError(f"calibration_root must be the independent Calibration directory: {calibration_root}")
    right_to_left = _homogeneous(
        rotation_left_to_right.T,
        -rotation_left_to_right.T @ (translation_left_to_right_mm / 1000.0),
    )
    _validate_rotation(right_to_left[:3, :3])
    calibration_payload = {
        "values": values,
        "rotation": rotation_left_to_right,
        "translation_mm": translation_left_to_right_mm,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    stable_output_dir = output_root / "stable_map"
    if stable_output_dir.resolve() != stable_dir.resolve():
        raise ValueError(
            "stable_dir must be the stable_map directory inside output_root so point-cloud paths remain auditable"
        )
    sample_specs = [(1500, output_root / "sample1500"), (3000, output_root / "sample3000")]
    sample_entries: dict[int, list[dict[str, Any]]] = {}
    sample_dirs: dict[int, tuple[Path, Path]] = {}
    all_targets: dict[int, dict[str, Any]] = {}
    for count, sample_dir in sample_specs:
        entries = _sample_rows(stable_run, count)
        left_dir, right_dir = _ensure_sample_directory(sample_dir)
        sample_dirs[count] = (left_dir, right_dir)
        for entry in entries:
            item = dict(entry)
            item["sample_owner"] = count
            item["left_image"] = f"left/left_{int(entry['source_frame']):06d}.png"
            item["right_image"] = f"right/right_{int(entry['source_frame']):06d}.png"
            sample_entries.setdefault(count, []).append(item)
            source_frame = int(entry["source_frame"])
            if source_frame in all_targets:
                # Both sample sets can share an endpoint.  The image path is
                # different, so retain one target descriptor per source for
                # extraction only; each set is written separately below.
                continue
            all_targets[source_frame] = item

    # Add sample-specific descriptors for the same source frame so extraction
    # writes both destination directories when the two sampling sets overlap.
    extraction_targets: dict[int, list[dict[str, Any]]] = {}
    for count, entries in sample_entries.items():
        for entry in entries:
            extraction_targets.setdefault(int(entry["source_frame"]), []).append(entry)

    # Save each shared-source image pair once into every sample directory that
    # requests it.  _save_target_images handles one path pair per target, so
    # use a small dedicated loop for the overlap-aware form.
    sl = _import_zed()
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.camera_disable_self_calib = True
    init.depth_mode = sl.DEPTH_MODE.NONE
    zed = sl.Camera()
    _check_status(zed.open(init), "open SVO for resampled image extraction", sl)
    left_mat = sl.Mat()
    right_mat = sl.Mat()
    left_view = getattr(sl.VIEW, "LEFT_UNRECTIFIED", sl.VIEW.LEFT)
    right_view = getattr(sl.VIEW, "RIGHT_UNRECTIFIED", sl.VIEW.RIGHT)
    extracted_sources: set[int] = set()
    try:
        info = zed.get_camera_information()
        resolution = info.camera_configuration.resolution
        if (int(resolution.width), int(resolution.height)) != (1920, 1080):
            raise RuntimeError(
                f"SVO full resolution is {resolution.width}x{resolution.height}, expected 1920x1080"
            )
        while len(extracted_sources) < len(extraction_targets):
            status = zed.grab()
            if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            _check_status(status, "grab during resampled image extraction", sl)
            source_frame = int(zed.get_svo_position())
            targets = extraction_targets.get(source_frame)
            if targets is None:
                continue
            _check_status(zed.retrieve_image(left_mat, left_view), "retrieve unrectified left", sl)
            _check_status(zed.retrieve_image(right_mat, right_view), "retrieve unrectified right", sl)
            left = _as_bgr(left_mat.get_data())
            right = _as_bgr(right_mat.get_data())
            if left.shape[:2] != (1080, 1920) or right.shape[:2] != (1080, 1920):
                raise RuntimeError(f"unexpected source image dimensions at frame {source_frame}")
            compression = [cv2.IMWRITE_PNG_COMPRESSION, 3]
            for target in targets:
                count = int(target["sample_owner"])
                left_dir, right_dir = sample_dirs[count]
                frame = int(target["source_frame"])
                left_path = left_dir / f"left_{frame:06d}.png"
                right_path = right_dir / f"right_{frame:06d}.png"
                if not cv2.imwrite(str(left_path), np.ascontiguousarray(left), compression):
                    raise RuntimeError(f"could not write {left_path}")
                if not cv2.imwrite(str(right_path), np.ascontiguousarray(right), compression):
                    raise RuntimeError(f"could not write {right_path}")
                target["left_path"] = left_path
                target["right_path"] = right_path
            extracted_sources.add(source_frame)
            if len(extracted_sources) % 100 == 0 or len(extracted_sources) == len(extraction_targets):
                print(
                    f"Saved unique unrectified source pairs: {len(extracted_sources)}/{len(extraction_targets)}",
                    flush=True,
                )
    finally:
        zed.close()
    missing_extracted = sorted(set(extraction_targets).difference(extracted_sources))
    if missing_extracted:
        raise RuntimeError(f"could not extract requested source frames: {missing_extracted[:10]}")

    sample_summaries: dict[str, Any] = {}
    for count, sample_dir in sample_specs:
        entries = sample_entries[count]
        for entry in entries:
            entry["left_image"] = str(Path(entry["left_image"]).as_posix())
            entry["right_image"] = str(Path(entry["right_image"]).as_posix())
        manifest_path = _write_manifest(sample_dir, entries)
        reference_path, reference_quality = _write_metashape_reference(sample_dir, entries, right_to_left)
        metadata = _build_sample_metadata(
            sample_dir=sample_dir,
            count=count,
            entries=entries,
            segment=segment,
            svo_path=svo_path,
            calibration_root=calibration_root,
            calibration_values=calibration_payload,
            metashape_path=reference_path,
            metashape_quality=reference_quality,
            manifest_path=manifest_path,
        )
        _write_json(sample_dir / "metashape_metadata.json", metadata)
        sample_summaries[str(count)] = {
            "left_images": len(list((sample_dir / "left").glob("*.png"))),
            "right_images": len(list((sample_dir / "right").glob("*.png"))),
            "manifest_rows": len(entries),
            "metashape_rows": reference_quality["csv_rows"],
            "metashape_reference": str(reference_path.resolve()),
        }

    points, observations, point_input_stats = _read_map_points(map_points_path)
    color_stats = _colorize_map_points(
        points=points,
        entries=sample_entries[3000],
        sample3000_dir=output_root / "sample3000",
        values=values,
        output_ply=stable_output_dir / "orbslam3_map_points_rgb.ply",
    )
    map_metadata = {
        "schema_version": 1,
        "source": {
            "map_points_xyz": str(map_points_path.resolve()),
            "camera_trajectory": str(trajectory_path.resolve()),
            "tracking_log": str(tracking_path.resolve()),
            "stable_segment": segment,
            "coordinate_frame": "same ORB-SLAM3 local metric frame as CameraTrajectory.txt",
        },
        "map_point_filter": {
            "C++_min_observations": min_observations,
            "input_rows_valid": point_input_stats["rows_valid"],
            "input_rows_invalid": point_input_stats["rows_invalid"],
            "observations_min": int(observations.min()),
            "observations_max": int(observations.max()),
        },
        "coloring": color_stats,
        "files": {
            "xyz_csv": str(map_points_path.resolve()),
            "rgb_ply": str((stable_output_dir / "orbslam3_map_points_rgb.ply").resolve()),
            "color_source": str((output_root / "sample3000" / "left").resolve()),
        },
    }
    _write_json(stable_output_dir / "map_metadata.json", map_metadata)
    instructions = _write_import_instructions(output_root)
    run_summary = {
        "schema_version": 1,
        "output_root": str(output_root.resolve()),
        "segment": segment,
        "samples": sample_summaries,
        "map": map_metadata,
        "metashape_import_instructions": str(instructions.resolve()),
        "calibration_source": str(calibration_root.resolve()),
        "svo_source": str(svo_path.resolve()),
    }
    _write_json(output_root / "run_summary.json", run_summary)
    print(f"Wrote resampled output: {output_root}")
    print(json.dumps({"segment": segment, "samples": sample_summaries, "map": color_stats}, ensure_ascii=False))
    return run_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ORB-SLAM3 exact-frame Metashape and sparse-map exports")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="find the longest continuous valid tracking segment")
    analyze_parser.add_argument("--tracking-log", type=Path, required=True)
    analyze_parser.add_argument("--output-plan", type=Path, required=True)
    analyze_parser.add_argument("--min-required", type=int, default=3000)

    build_parser = subparsers.add_parser("build", help="extract images and build Metashape/map exports")
    build_parser.add_argument("--stable-dir", type=Path, required=True)
    build_parser.add_argument("--plan", type=Path, required=True)
    build_parser.add_argument("--svo", type=Path, required=True)
    build_parser.add_argument("--calibration-root", type=Path, required=True)
    build_parser.add_argument("--output-root", type=Path, required=True)
    build_parser.add_argument("--min-observations", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "analyze":
        analyze(_resolve(args.tracking_log), _resolve(args.output_plan), args.min_required)
        return 0
    if args.command == "build":
        build(
            stable_dir=_resolve(args.stable_dir),
            plan_path=_resolve(args.plan),
            svo_path=_resolve(args.svo),
            calibration_root=_resolve(args.calibration_root),
            output_root=_resolve(args.output_root),
            min_observations=args.min_observations,
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
