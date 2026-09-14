"""Export point clouds for a uniformly sampled ORB-SLAM3 SVO run.

The ORB-SLAM3 run supplies the camera-to-world trajectory.  Depth and RGB are
retrieved again from the original SVO through the ZED SDK with the explicit
Custom OpenCV calibration file.  No old depth, pose, or native-SVO calibration
artifact is reused.

ORB-SLAM3 uses the OpenCV camera frame (+X right, +Y down, +Z forward), while
the SDK run below uses RIGHT_HANDED_Y_UP (+X right, +Y up, camera forward is
-Z).  The conversion ``diag(1, -1, -1)`` is therefore applied before the SDK
points are transformed by the ORB-SLAM3 TUM trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import rerun_custom_calibration as custom

cv2 = custom.cv2
np = custom.np


ROOT = Path(__file__).resolve().parent
DEFAULT_SVO = ROOT / "20260802_150233.svo2"
DEFAULT_RUN_DIR = ROOT / "output" / "20260802_150233_orbslam3_uniform10pct_custom"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "pointcloud"
DEFAULT_ORB_SETTINGS = ROOT / "SLAM" / "config" / "20260802_150233_stereo.yaml"
DEFAULT_CALIBRATION = ROOT / "Calibration" / "zed_custom_opencv.yml"
POINT_DTYPE = custom.POINT_DTYPE
POINT_RECORD_SIZE = int(POINT_DTYPE.itemsize)
HARD_OUTPUT_LIMIT_BYTES = 2_000_000_000
SAFE_OUTPUT_LIMIT_BYTES = 1_900_000_000
PLY_HEADER_RESERVE_BYTES = 512


def _round_timestamp(value: float) -> float:
    return round(float(value), 6)


def _quaternion_to_rotation(qx: float, qy: float, qz: float, qw: float) -> tuple[np.ndarray, float]:
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    if not np.isfinite(quaternion).all():
        raise RuntimeError("ORB trajectory contains a non-finite quaternion")
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError(f"Invalid ORB trajectory quaternion norm: {norm}")
    x, y, z, w = quaternion / norm
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rotation, norm


def _read_tracking_log(path: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    required = {
        "frame",
        "svo_position",
        "timestamp_sec",
        "tracking_state",
        "pose_valid",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"ORB tracking log is missing columns: {path}")
        rows: list[dict[str, Any]] = []
        by_source: dict[int, dict[str, Any]] = {}
        for expected_index, raw in enumerate(reader):
            frame = int(raw["frame"])
            source = int(raw["svo_position"])
            timestamp = float(raw["timestamp_sec"])
            state = int(raw["tracking_state"])
            pose_valid = raw["pose_valid"].strip() == "1"
            if frame != expected_index or source in by_source:
                raise RuntimeError(f"ORB tracking log has invalid frame indexing at row {expected_index}")
            if not math.isfinite(timestamp):
                raise RuntimeError(f"ORB tracking log has a non-finite timestamp at row {expected_index}")
            row = {
                "frame": frame,
                "svo_position": source,
                "timestamp_sec": timestamp,
                "tracking_state": state,
                "pose_valid": pose_valid,
                "tracked_map_points": int(raw.get("tracked_map_points") or 0),
                "width": int(raw.get("width") or 0),
                "height": int(raw.get("height") or 0),
            }
            rows.append(row)
            by_source[source] = row
    if not rows:
        raise RuntimeError(f"ORB tracking log is empty: {path}")
    sources = [int(row["svo_position"]) for row in rows]
    if sources != sorted(sources) or len(set(sources)) != len(sources):
        raise RuntimeError("ORB selected source positions are not strictly increasing")
    return rows, by_source


def _read_tum_trajectory(path: Path) -> dict[float, dict[str, Any]]:
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
                raise RuntimeError(f"{path}:{line_number}: expected 8 TUM fields")
            values = [float(value) for value in fields[:8]]
            timestamp, tx, ty, tz, qx, qy, qz, qw = values
            key = _round_timestamp(timestamp)
            rotation, quaternion_norm = _quaternion_to_rotation(qx, qy, qz, qw)
            candidate = {
                "timestamp_sec": timestamp,
                "translation": np.asarray([tx, ty, tz], dtype=np.float64),
                "rotation": rotation,
                "quaternion_norm": quaternion_norm,
            }
            if key in result:
                previous = result[key]
                same_pose = (
                    np.allclose(previous["translation"], candidate["translation"], atol=1e-9)
                    and np.allclose(previous["rotation"], candidate["rotation"], atol=1e-9)
                )
                if not same_pose:
                    raise RuntimeError(
                        f"{path}:{line_number}: duplicate timestamp {key} has different poses"
                    )
                # ORB-SLAM3 may repeat an identical trajectory row while
                # recovering/resetting a map.  One copy is sufficient because
                # the tracking log has the authoritative source-frame key.
                continue
            result[key] = candidate
    if not result:
        raise RuntimeError(f"ORB trajectory is empty: {path}")
    return result


def _attach_orb_poses(
    tracking_rows: list[dict[str, Any]], trajectory: dict[float, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attached: list[dict[str, Any]] = []
    unmatched_valid: list[int] = []
    unmatched_trajectory = 0
    for row in tracking_rows:
        pose = trajectory.get(_round_timestamp(float(row["timestamp_sec"])))
        if pose is None:
            if row["pose_valid"] and row["tracking_state"] in {2, 5}:
                unmatched_valid.append(int(row["svo_position"]))
            continue
        item = dict(row)
        item.update(pose)
        item["pose_valid"] = bool(row["pose_valid"] and row["tracking_state"] in {2, 5})
        if item["pose_valid"]:
            attached.append(item)

    tracking_keys = {_round_timestamp(float(row["timestamp_sec"])) for row in tracking_rows}
    unmatched_trajectory = sum(key not in tracking_keys for key in trajectory)
    if unmatched_valid:
        raise RuntimeError(
            "Valid ORB tracking rows are missing from CameraTrajectory.txt: "
            f"{unmatched_valid[:10]}"
        )
    return attached, {
        "trajectory_rows": len(trajectory),
        "trajectory_rows_without_tracking_row": unmatched_trajectory,
        "valid_tracking_rows_with_trajectory": len(attached),
    }


def _uniform_positions(total_frames: int, sample_count: int) -> list[int]:
    if total_frames <= 0 or sample_count <= 0 or sample_count > total_frames:
        raise RuntimeError(
            f"Invalid uniform sampling request: total={total_frames}, count={sample_count}"
        )
    if sample_count == 1:
        return [0]
    return [
        (index * (total_frames - 1)) // (sample_count - 1)
        for index in range(sample_count)
    ]


def _read_run_metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _write_sample_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "sample_index",
                "orb_frame",
                "svo_position",
                "timestamp_sec",
                "tracking_state",
                "pose_valid",
            ]
        )
        for index, row in enumerate(rows):
            writer.writerow(
                [
                    index,
                    row["frame"],
                    row["svo_position"],
                    f"{float(row['timestamp_sec']):.17g}",
                    row["tracking_state"],
                    int(row["pose_valid"]),
                ]
            )


def _read_orb_settings(path: Path) -> dict[str, Any]:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise RuntimeError(f"Could not open ORB-SLAM3 settings: {path}")
    try:
        def scalar(name: str) -> float:
            node = storage.getNode(name)
            if node.empty():
                raise RuntimeError(f"Missing ORB setting {name}")
            return float(node.real())

        transform = storage.getNode("Stereo.T_c1_c2").mat()
        if transform is None:
            raise RuntimeError("Missing Stereo.T_c1_c2 in ORB settings")
        return {
            "camera1": {key: scalar(f"Camera1.{key}") for key in ("fx", "fy", "cx", "cy")},
            "camera2": {key: scalar(f"Camera2.{key}") for key in ("fx", "fy", "cx", "cy")},
            "distortion1": [scalar(f"Camera1.{key}") for key in ("k1", "k2", "p1", "p2", "k3")],
            "distortion2": [scalar(f"Camera2.{key}") for key in ("k1", "k2", "p1", "p2", "k3")],
            "width": int(round(scalar("Camera.width"))),
            "height": int(round(scalar("Camera.height"))),
            "fps": scalar("Camera.fps"),
            "stereo_transform": np.asarray(transform, dtype=np.float64).reshape(4, 4),
        }
    finally:
        storage.release()


def _compare_orb_settings(
    settings: dict[str, Any], calibration: custom.Calibration
) -> dict[str, Any]:
    if settings["width"] <= 0 or settings["height"] <= 0:
        raise RuntimeError("ORB settings contain an invalid image size")
    scale_x = settings["width"] / calibration.width
    scale_y = settings["height"] / calibration.height
    if not math.isclose(scale_x, scale_y, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            "ORB image scaling is not uniform relative to Custom calibration: "
            f"ORB={settings['width']}x{settings['height']}, "
            f"Custom={calibration.width}x{calibration.height}"
        )
    image_scale = float(scale_x)
    if image_scale <= 0.0 or image_scale > 1.0:
        raise RuntimeError(f"Invalid ORB image scale relative to Custom calibration: {image_scale}")

    expected_t = np.eye(4, dtype=np.float64)
    r_lr = cv2.Rodrigues(calibration.r)[0]
    t_lr_m = calibration.t_mm / 1000.0
    expected_t[:3, :3] = r_lr.T
    expected_t[:3, 3] = -r_lr.T @ t_lr_m

    errors: dict[str, float] = {}
    for side, matrix, prefix in (
        ("left", calibration.k_left, "camera1"),
        ("right", calibration.k_right, "camera2"),
    ):
        for key, row, column in (("fx", 0, 0), ("fy", 1, 1), ("cx", 0, 2), ("cy", 1, 2)):
            errors[f"{side}_{key}_px"] = abs(
                float(settings[prefix][key]) - float(matrix[row, column]) * image_scale
            )
    for index, key in enumerate(("k1", "k2", "p1", "p2", "k3")):
        errors[f"left_{key}"] = abs(float(settings["distortion1"][index]) - float(calibration.d_left[index]))
        errors[f"right_{key}"] = abs(float(settings["distortion2"][index]) - float(calibration.d_right[index]))
    errors["stereo_transform"] = float(
        np.max(np.abs(settings["stereo_transform"] - expected_t))
    )
    if max(errors.values()) > 2e-4:
        raise RuntimeError(f"ORB settings do not match Custom calibration: {errors}")
    return {
        "passed": True,
        "image_scale": image_scale,
        "max_abs_errors": errors,
        "expected_orb_stereo_transform": expected_t.tolist(),
        "observed_orb_stereo_transform": settings["stereo_transform"].tolist(),
    }


def _append_frame(
    raw_file: Any,
    point_cloud_mat: Any,
    left_mat: Any,
    pose: dict[str, Any],
    pixel_stride: int,
    min_depth: float,
    max_depth: float,
    current_bytes: int,
) -> tuple[int, int]:
    point_cloud = np.asarray(point_cloud_mat.get_data())
    left_image = custom._as_bgr(left_mat.get_data())
    if point_cloud.ndim != 3 or point_cloud.shape[2] < 3:
        raise RuntimeError(f"Unexpected SDK XYZRGBA shape: {point_cloud.shape}")
    if left_image.shape[:2] != point_cloud.shape[:2]:
        raise RuntimeError(
            f"SDK XYZRGBA/left shape mismatch: {point_cloud.shape[:2]} != {left_image.shape[:2]}"
        )

    sampled_xyz_sdk = np.asarray(
        point_cloud[::pixel_stride, ::pixel_stride, :3], dtype=np.float32
    )
    sampled_bgr = np.asarray(
        left_image[::pixel_stride, ::pixel_stride], dtype=np.uint8
    )
    valid = np.isfinite(sampled_xyz_sdk).all(axis=2)
    optical_depth = -sampled_xyz_sdk[:, :, 2]
    valid &= np.isfinite(optical_depth)
    valid &= optical_depth > np.float32(min_depth)
    if np.isfinite(max_depth):
        valid &= optical_depth <= np.float32(max_depth)
    if not np.any(valid):
        return 0, current_bytes

    # SDK RIGHT_HANDED_Y_UP -> ORB/OpenCV camera coordinates.
    points_sdk = sampled_xyz_sdk[valid].astype(np.float64)
    points_cv = points_sdk * np.asarray([1.0, -1.0, -1.0], dtype=np.float64)
    rotation = np.asarray(pose["rotation"], dtype=np.float64)
    translation = np.asarray(pose["translation"], dtype=np.float64)
    points_world = points_cv @ rotation.T + translation
    finite = np.isfinite(points_world).all(axis=1)
    if not np.any(finite):
        return 0, current_bytes
    points_world = points_world[finite]
    colors_rgb = sampled_bgr[valid][finite][:, ::-1]

    records = np.empty(len(points_world), dtype=POINT_DTYPE)
    records["x"] = points_world[:, 0].astype(np.float32)
    records["y"] = points_world[:, 1].astype(np.float32)
    records["z"] = points_world[:, 2].astype(np.float32)
    records["red"] = colors_rgb[:, 0]
    records["green"] = colors_rgb[:, 1]
    records["blue"] = colors_rgb[:, 2]
    next_bytes = current_bytes + int(records.nbytes)
    if next_bytes + PLY_HEADER_RESERVE_BYTES >= HARD_OUTPUT_LIMIT_BYTES:
        raise RuntimeError(
            "Dense ORB-pose point cloud exceeded the 2 GB safety limit; "
            "increase --pixel-stride."
        )
    raw_file.write(records.tobytes(order="C"))
    return len(records), next_bytes


def _write_binary_ply_from_csv(csv_path: Path, ply_path: Path) -> dict[str, Any]:
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    records: list[tuple[float, float, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"x", "y", "z", "observations"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"ORB MapPoint CSV is missing columns: {csv_path}")
        for row in reader:
            if int(row["observations"]) < 2:
                continue
            point = tuple(float(row[key]) for key in ("x", "y", "z"))
            if np.isfinite(point).all():
                records.append(point)
    if not records:
        raise RuntimeError(f"ORB MapPoint CSV has no valid points: {csv_path}")
    values = np.asarray(records, dtype=np.float32)
    packed = np.empty(len(values), dtype=POINT_DTYPE)
    packed["x"] = values[:, 0]
    packed["y"] = values[:, 1]
    packed["z"] = values[:, 2]
    packed["red"] = 180
    packed["green"] = 180
    packed["blue"] = 180
    header = custom._ply_header(len(packed))
    tmp_path = ply_path.with_suffix(ply_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        handle.write(header)
        handle.write(packed.tobytes(order="C"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, ply_path)
    return {
        "file": str(ply_path),
        "format": "PLY binary_little_endian",
        "point_count": int(len(packed)),
        "file_size_bytes": int(ply_path.stat().st_size),
        "source": str(csv_path),
        "color": "constant gray; coordinates directly exported by ORB-SLAM3",
    }


def _export_dense(
    svo_path: Path,
    calibration_path: Path,
    output_dir: Path,
    poses_by_source: dict[int, dict[str, Any]],
    total_frames: int,
    pixel_stride: int,
    min_depth: float,
    max_depth: float,
) -> dict[str, Any]:
    dense_path = output_dir / "scene_world_rgb_orbslam3_pose.ply"
    raw_path = output_dir / "scene_world_rgb_orbslam3_pose.ply.records.tmp"
    ply_tmp = output_dir / "scene_world_rgb_orbslam3_pose.ply.tmp"
    point_count = 0
    raw_bytes = 0
    source_frames_read = 0
    depth_frames_used = 0
    started = time.monotonic()
    zed = custom.sl.Camera()
    custom._check_status(
        zed.open(custom._make_init(svo_path, calibration_path, custom.sl.DEPTH_MODE.NEURAL)),
        "open SVO for ORB-pose point cloud",
    )
    try:
        info = zed.get_camera_information()
        sdk_frame_count = int(zed.get_svo_number_of_frames())
        if sdk_frame_count != total_frames:
            raise RuntimeError(f"SVO frame count changed: expected {total_frames}, SDK={sdk_frame_count}")
        runtime = custom.sl.RuntimeParameters()
        runtime.confidence_threshold = 30
        runtime.texture_confidence_threshold = 100
        runtime.measure3D_reference_frame = custom.sl.REFERENCE_FRAME.CAMERA
        point_mat = custom.sl.Mat()
        left_mat = custom.sl.Mat()
        with raw_path.open("wb") as raw_file:
            while True:
                status = zed.grab(runtime)
                if status == custom.sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break
                custom._check_status(status, "grab during ORB-pose point-cloud export")
                source_frames_read += 1
                source_position = int(zed.get_svo_position())
                pose = poses_by_source.get(source_position)
                if pose is None:
                    continue
                custom._check_status(
                    zed.retrieve_image(left_mat, custom.sl.VIEW.LEFT),
                    f"retrieve SDK rectified left image at source frame {source_position}",
                )
                custom._check_status(
                    zed.retrieve_measure(point_mat, custom.sl.MEASURE.XYZRGBA),
                    f"retrieve SDK XYZRGBA at source frame {source_position}",
                )
                added, raw_bytes = _append_frame(
                    raw_file=raw_file,
                    point_cloud_mat=point_mat,
                    left_mat=left_mat,
                    pose=pose,
                    pixel_stride=pixel_stride,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    current_bytes=raw_bytes,
                )
                point_count += added
                depth_frames_used += 1
                if depth_frames_used % 100 == 0 or depth_frames_used == len(poses_by_source):
                    elapsed = time.monotonic() - started
                    print(
                        f"SDK depth frames: {depth_frames_used}/{len(poses_by_source)}, "
                        f"source={source_position}/{total_frames - 1}, "
                        f"points={point_count:,}, elapsed={elapsed:.1f}s",
                        flush=True,
                    )
            raw_file.flush()
            os.fsync(raw_file.fileno())
    finally:
        zed.close()

    if source_frames_read != total_frames:
        raise RuntimeError(
            f"SDK replay stopped early: {source_frames_read}/{total_frames} source frames"
        )
    if depth_frames_used != len(poses_by_source):
        raise RuntimeError(
            f"SDK depth frames do not cover ORB poses: {depth_frames_used}/{len(poses_by_source)}"
        )
    if point_count <= 0:
        raise RuntimeError("SDK/ORB dense point cloud contains no valid points")

    header = custom._ply_header(point_count)
    final_size = len(header) + raw_bytes
    if final_size >= HARD_OUTPUT_LIMIT_BYTES:
        raise RuntimeError(f"Dense PLY exceeds hard output limit: {final_size}")
    with ply_tmp.open("wb") as final_file:
        final_file.write(header)
        with raw_path.open("rb") as raw_file:
            shutil.copyfileobj(raw_file, final_file, length=8 * 1024 * 1024)
        final_file.flush()
        os.fsync(final_file.fileno())
    os.replace(ply_tmp, dense_path)
    raw_path.unlink(missing_ok=True)
    actual_size = dense_path.stat().st_size
    if actual_size != final_size:
        raise RuntimeError(f"Dense PLY size changed while writing: {actual_size} != {final_size}")
    return {
        "file": str(dense_path),
        "format": "PLY binary_little_endian",
        "fields": "float32 x,y,z; uint8 red,green,blue",
        "source": "ZED SDK MEASURE.XYZRGBA with Custom calibration; ORB-SLAM3 TUM pose",
        "coordinate_conversion": "SDK RIGHT_HANDED_Y_UP -> ORB OpenCV via diag(1,-1,-1)",
        "fusion": "p_world = T_orb_world_camera @ p_orb_camera",
        "point_count": point_count,
        "file_size_bytes": actual_size,
        "file_size_gb_decimal": actual_size / 1e9,
        "depth_frames_used": depth_frames_used,
        "source_frames_replayed": source_frames_read,
        "pixel_stride": pixel_stride,
        "min_optical_depth_m": min_depth,
        "max_optical_depth_m": max_depth,
        "elapsed_seconds": time.monotonic() - started,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Custom-calibration SDK depth fused with a sampled ORB-SLAM3 trajectory."
    )
    parser.add_argument("--svo", type=Path, default=DEFAULT_SVO)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--orb-settings", type=Path, default=DEFAULT_ORB_SETTINGS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--sample-count", type=int, default=3586)
    parser.add_argument("--pixel-stride", type=int, default=0)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=50.0)
    args = parser.parse_args()
    if args.sample_count <= 0:
        parser.error("--sample-count must be positive")
    if args.pixel_stride < 0:
        parser.error("--pixel-stride must be non-negative")
    if args.min_depth < 0 or args.max_depth <= args.min_depth:
        parser.error("require 0 <= --min-depth < --max-depth")
    return args


def main() -> int:
    args = _parse_args()
    svo_path = args.svo.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    orb_settings_path = args.orb_settings.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    tracking_path = run_dir / "tracking_log.csv"
    trajectory_path = run_dir / "CameraTrajectory.txt"
    map_points_path = run_dir / "map_points_xyz.csv"
    required = [svo_path, orb_settings_path, calibration_path, tracking_path, trajectory_path, map_points_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing ORB point-cloud input(s): " + ", ".join(missing))
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration = custom._read_opencv_calibration(calibration_path)
    orb_settings = _read_orb_settings(orb_settings_path)
    settings_audit = _compare_orb_settings(orb_settings, calibration)
    tracking_rows, tracking_by_source = _read_tracking_log(tracking_path)
    trajectory = _read_tum_trajectory(trajectory_path)
    orb_poses, trajectory_audit = _attach_orb_poses(tracking_rows, trajectory)

    zed_probe = custom.sl.Camera()
    custom._check_status(
        zed_probe.open(custom._make_init(svo_path, calibration_path, custom.sl.DEPTH_MODE.NEURAL)),
        "open SVO for Custom calibration runtime verification",
    )
    try:
        info = zed_probe.get_camera_information()
        total_frames = int(zed_probe.get_svo_number_of_frames())
        sdk_camera = custom._verify_sdk_override(info, calibration)
    finally:
        zed_probe.close()

    if len(tracking_rows) != args.sample_count:
        raise RuntimeError(
            f"ORB tracking row count is not the requested uniform sample count: "
            f"rows={len(tracking_rows)}, requested={args.sample_count}"
        )
    expected_positions = _uniform_positions(total_frames, args.sample_count)
    actual_positions = [int(row["svo_position"]) for row in tracking_rows]
    if actual_positions != expected_positions:
        raise RuntimeError(
            "ORB tracking source positions are not the requested uniform sample: "
            f"first_actual={actual_positions[:5]}, first_expected={expected_positions[:5]}, "
            f"last_actual={actual_positions[-5:]}, last_expected={expected_positions[-5:]}"
        )
    if len(orb_poses) == 0:
        raise RuntimeError("ORB-SLAM3 produced no valid trajectory poses")

    poses_by_source = {int(pose["svo_position"]): pose for pose in orb_poses}
    pixel_stride, estimated_points = custom._choose_pixel_stride(
        calibration.width,
        calibration.height,
        len(poses_by_source),
        args.pixel_stride,
    )
    if estimated_points * POINT_RECORD_SIZE + PLY_HEADER_RESERVE_BYTES >= SAFE_OUTPUT_LIMIT_BYTES:
        raise RuntimeError("Adaptive pixel stride failed the safe output estimate")

    print(f"ORB sampled rows: {len(tracking_rows)} / {total_frames} source frames")
    print(f"Valid ORB poses used for dense cloud: {len(poses_by_source)}")
    print(
        f"Pixel stride: {pixel_stride}; worst-case estimate: {estimated_points:,} points / "
        f"{estimated_points * POINT_RECORD_SIZE / 1e9:.3f} GB",
        flush=True,
    )
    sample_manifest_path = output_dir / "sample_positions.csv"
    _write_sample_manifest(sample_manifest_path, tracking_rows)
    dense = _export_dense(
        svo_path=svo_path,
        calibration_path=calibration_path,
        output_dir=output_dir,
        poses_by_source=poses_by_source,
        total_frames=total_frames,
        pixel_stride=pixel_stride,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    sparse = _write_binary_ply_from_csv(
        map_points_path, output_dir / "orbslam3_map_points.ply"
    )

    interval = np.diff(np.asarray(actual_positions, dtype=np.int64))
    state_counts = Counter(str(row["tracking_state"]) for row in tracking_rows)
    metadata = {
        "input_file": str(svo_path),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "sdk_version": str(custom.sl.Camera.get_sdk_version()),
        "verification": {
            "passed": True,
            "uniform_source_positions": True,
            "orb_settings_match_custom_calibration": settings_audit,
            "sdk_runtime_custom_override": sdk_camera[
                "custom_calibration_override_verification"
            ],
            "all_source_frames_replayed_for_depth": dense["source_frames_replayed"] == total_frames,
        },
        "sampling": {
            "source_frames": total_frames,
            "requested_ratio": 0.1,
            "sample_count": len(tracking_rows),
            "ratio_actual": len(tracking_rows) / total_frames,
            "rule": "uniform floor positions including source frame 0 and source frame N-1",
            "formula": "position_i = floor(i*(N-1)/(K-1)), i=0..K-1",
            "first_positions": actual_positions[:10],
            "last_positions": actual_positions[-10:],
            "interval_min": int(interval.min()) if interval.size else None,
            "interval_median": float(np.median(interval)) if interval.size else None,
            "interval_max": int(interval.max()) if interval.size else None,
        },
        "calibration": {
            "custom_opencv_file": str(calibration_path),
            "orb_settings_file": str(orb_settings_path),
            "image_size": {"width": calibration.width, "height": calibration.height},
            "left_K": calibration.k_left.tolist(),
            "right_K": calibration.k_right.tolist(),
            "left_D_14": calibration.d_left.tolist(),
            "right_D_14": calibration.d_right.tolist(),
            "R_rodrigues": calibration.r.tolist(),
            "T_mm": calibration.t_mm.tolist(),
            "translation_norm_mm": float(np.linalg.norm(calibration.t_mm)),
        },
        "orbslam3": {
            "tracking_log": str(tracking_path),
            "trajectory": str(trajectory_path),
            "tracking_rows": len(tracking_rows),
            "tracking_state_counts": dict(sorted(state_counts.items())),
            "valid_pose_rows": len(orb_poses),
            "trajectory_audit": trajectory_audit,
            "map_points_csv": str(map_points_path),
            "image_source": "ZED SDK VIEW::LEFT_UNRECTIFIED_BGR and RIGHT_UNRECTIFIED_BGR",
            "embedded_svo_calibration_for_slam": False,
            "geometry_source": "SLAM/config/20260802_150233_stereo.yaml generated from Calibration/",
        },
        "depth_and_fusion": {
            "depth_mode": "NEURAL",
            "depth_source": "ZED SDK MEASURE.XYZRGBA",
            "optional_opencv_calibration_file": str(calibration_path),
            "camera_disable_self_calib": True,
            "coordinate_units": "METER",
            "sdk_coordinate_system": "RIGHT_HANDED_Y_UP",
            "orb_camera_coordinate_system": "OpenCV optical (+X right,+Y down,+Z forward)",
            "sdk_to_orb_camera_conversion": "diag(1,-1,-1)",
            "orb_pose_transform": "T_orb_world_camera from ORB-SLAM3 SaveTrajectoryTUM",
            "min_optical_depth_m": args.min_depth,
            "max_optical_depth_m": args.max_depth,
        },
        "point_clouds": {
            "dense": dense,
            "sparse_orb_map": sparse,
        },
        "sample_manifest": str(sample_manifest_path),
        "source_run_metadata": _read_run_metadata(run_dir / "run_metadata.txt"),
        "outputs": {
            "dense_pointcloud": str(output_dir / "scene_world_rgb_orbslam3_pose.ply"),
            "sparse_pointcloud": str(output_dir / "orbslam3_map_points.ply"),
            "sample_manifest": str(sample_manifest_path),
            "metadata": str(output_dir / "metadata.json"),
        },
    }
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

    print(f"Completed ORB-SLAM3 Custom point-cloud export: {output_dir}")
    print(f"Dense: {dense['point_count']:,} points; {dense['file_size_gb_decimal']:.3f} GB")
    print(f"Sparse ORB map: {sparse['point_count']:,} points")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
