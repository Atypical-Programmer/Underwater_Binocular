"""Export a bounded, WORLD-frame RGB point cloud from a ZED SVO2 recording.

The pose pass is deliberately sequential.  The SVO is read from the first
frame to the last frame so the ZED positional tracker sees consecutive images;
only the 1000 previously exported depth/image frames are used for the point
cloud itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
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

import cv2  # noqa: E402  (DLL search path must be prepared first)
import numpy as np  # noqa: E402
import pyzed.sl as sl  # noqa: E402


DEFAULT_SVO = Path(__file__).resolve().with_name("20260802_150233.svo2")
DEFAULT_SOURCE_DIR = (
    DEFAULT_SVO.parent / "output" / "20260802_150233_sample1000"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SVO.parent / "output" / "20260802_150233_pointcloud1000"

HARD_OUTPUT_LIMIT_BYTES = 2_000_000_000
SAFE_OUTPUT_LIMIT_BYTES = 1_900_000_000
PLY_HEADER_RESERVE_BYTES = 512

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

# A packed binary PLY vertex: 3 float32 coordinates + 3 uint8 RGB values.
POINT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)
assert POINT_DTYPE.itemsize == 15


def _status_name(status: Any) -> str:
    return str(status).rsplit(".", 1)[-1]


def _check_status(status: Any, operation: str) -> None:
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"{operation} failed: {_status_name(status)}")


def _sample_positions(total_frames: int, max_frames: int) -> list[int]:
    """Return uniformly distributed source positions, including both ends."""

    if total_frames <= 0:
        return []
    if max_frames <= 0 or max_frames >= total_frames:
        return list(range(total_frames))

    sample_count = min(max_frames, total_frames)
    if sample_count == 1:
        return [0]
    return [
        (index * (total_frames - 1)) // (sample_count - 1)
        for index in range(sample_count)
    ]


def _csv_number(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "NaN"
    return f"{value:.12g}"


def _pose_values(
    pose: sl.Pose,
    tracking_state: Any,
) -> tuple[bool, float, list[float], np.ndarray | None]:
    """Extract a CSV-ready pose and its 4x4 camera-to-world matrix."""

    is_ok = tracking_state == sl.POSITIONAL_TRACKING_STATE.OK
    pose_valid = bool(pose.valid) if is_ok else False
    pose_confidence = float(pose.pose_confidence) if is_ok else float("nan")
    if not is_ok or not pose_valid:
        return pose_valid, pose_confidence, [float("nan")] * 23, None

    translation = np.asarray(pose.get_translation().get(), dtype=np.float64).reshape(-1)
    orientation = np.asarray(pose.get_orientation().get(), dtype=np.float64).reshape(-1)
    matrix = np.asarray(pose.pose_data().m, dtype=np.float64).reshape(4, 4)
    values = [*translation[:3], *orientation[:4], *matrix.reshape(-1)]
    if len(values) != 23:
        raise RuntimeError(f"Unexpected pose value count: {len(values)}")
    return pose_valid, pose_confidence, values, matrix


def _float_list(value: Any) -> list[float]:
    try:
        return [float(item) for item in np.asarray(value, dtype=np.float64).reshape(-1)]
    except (TypeError, ValueError):
        return []


def _camera_metadata(info: Any) -> dict[str, Any]:
    resolution = info.camera_configuration.resolution
    calibration = info.camera_configuration.calibration_parameters
    left = calibration.left_cam
    return {
        "camera_model": _status_name(info.camera_model),
        "serial_number": int(info.serial_number),
        "resolution": {"width": int(resolution.width), "height": int(resolution.height)},
        "left_rectified_intrinsics_px": {
            "fx": float(left.fx),
            "fy": float(left.fy),
            "cx": float(left.cx),
            "cy": float(left.cy),
        },
        "left_lens_distortion_model": str(left.lens_distortion_model),
        "left_distortion_coefficients": _float_list(left.disto),
    }


def _load_source_rows(source_dir: Path, sample_positions: list[int]) -> list[dict[str, str]]:
    pose_path = source_dir / "pose_world.csv"
    if not pose_path.is_file():
        raise FileNotFoundError(f"Source pose CSV not found: {pose_path}")

    with pose_path.open(newline="", encoding="utf-8") as pose_file:
        rows = list(csv.DictReader(pose_file))
    if len(rows) != len(sample_positions):
        raise RuntimeError(
            f"Source pose row count {len(rows)} does not match expected "
            f"sample count {len(sample_positions)}"
        )

    for index, row in enumerate(rows):
        if int(row["frame_index"]) != index:
            raise RuntimeError(f"Unexpected source frame index at row {index}")
        if int(row["svo_position"]) != sample_positions[index]:
            raise RuntimeError(
                f"Source SVO position mismatch at frame {index}: "
                f"{row['svo_position']} != {sample_positions[index]}"
            )
    return rows


def _run_sequential_pose_pass(
    svo_path: Path,
    target_positions: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    """Replay every SVO frame and retain pose only at target positions."""

    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    zed = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open SVO2: {_status_name(status)}")

    records: list[dict[str, Any] | None] = [None] * len(target_positions)
    target_to_index = {position: index for index, position in enumerate(target_positions)}
    info = None
    total_frames = 0
    try:
        info = zed.get_camera_information()
        total_frames = int(zed.get_svo_number_of_frames())
        tracking_status = zed.enable_positional_tracking(sl.PositionalTrackingParameters())
        _check_status(tracking_status, "enable_positional_tracking")

        runtime = sl.RuntimeParameters()
        pose = sl.Pose()
        started = time.monotonic()
        last_report = -1
        while True:
            status = zed.grab(runtime)
            if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            _check_status(status, "grab during sequential pose pass")

            source_position = int(zed.get_svo_position())
            tracking_state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
            target_index = target_to_index.get(source_position)
            if target_index is not None:
                timestamp_ns = int(
                    zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                )
                pose_valid, pose_confidence, values, matrix = _pose_values(
                    pose, tracking_state
                )
                records[target_index] = {
                    "row": [
                        target_index,
                        source_position,
                        timestamp_ns,
                        _status_name(tracking_state),
                        int(pose_valid),
                        _csv_number(pose_confidence),
                        *(_csv_number(value) for value in values),
                    ],
                    "matrix": matrix,
                    "pose_valid": pose_valid,
                    "tracking_state": _status_name(tracking_state),
                }

            if source_position // 1000 != last_report:
                last_report = source_position // 1000
                elapsed = time.monotonic() - started
                print(
                    f"Sequential pose pass: source frame {source_position}/{total_frames - 1} "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )

            if records and records[-1] is not None:
                break
    finally:
        zed.close()

    if info is None:
        raise RuntimeError("No camera information returned")
    if any(record is None for record in records):
        missing = [index for index, record in enumerate(records) if record is None]
        raise RuntimeError(f"Sequential pose pass missed sampled frames: {missing[:10]}")
    return [record for record in records if record is not None], _camera_metadata(info), total_frames


def _write_pose_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as pose_file:
        writer = csv.writer(pose_file)
        writer.writerow(POSE_HEADER)
        for record in records:
            writer.writerow(record["row"])


def _choose_pixel_stride(
    width: int,
    height: int,
    frame_count: int,
    requested_stride: int,
) -> tuple[int, int]:
    stride = max(1, requested_stride)
    while True:
        points_per_frame = ((width + stride - 1) // stride) * (
            (height + stride - 1) // stride
        )
        estimated_points = points_per_frame * frame_count
        estimated_bytes = estimated_points * POINT_DTYPE.itemsize + PLY_HEADER_RESERVE_BYTES
        if estimated_bytes < SAFE_OUTPUT_LIMIT_BYTES:
            return stride, estimated_points
        stride += 1


def _ply_header(point_count: int) -> bytes:
    return (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {point_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")


def _append_frame_points(
    raw_file: Any,
    depth_path: Path,
    image_path: Path,
    matrix: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    pixel_stride: int,
    min_depth: float,
    max_depth: float,
    current_bytes: int,
) -> tuple[int, int]:
    depth = np.load(depth_path, mmap_mode="r")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read left image: {image_path}")
    if depth.dtype != np.float32 or depth.ndim != 2:
        raise RuntimeError(f"Unexpected depth array at {depth_path}: {depth.dtype} {depth.shape}")
    if image.shape[:2] != depth.shape:
        raise RuntimeError(
            f"Image/depth shape mismatch at frame {depth_path.stem}: "
            f"{image.shape[:2]} != {depth.shape}"
        )

    sampled_depth = np.asarray(depth[::pixel_stride, ::pixel_stride], dtype=np.float32)
    sampled_bgr = image[::pixel_stride, ::pixel_stride]
    height, width = sampled_depth.shape
    u = np.arange(0, width * pixel_stride, pixel_stride, dtype=np.float32)
    v = np.arange(0, height * pixel_stride, pixel_stride, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    valid = np.isfinite(sampled_depth)
    valid &= sampled_depth > np.float32(min_depth)
    if np.isfinite(max_depth):
        valid &= sampled_depth <= np.float32(max_depth)
    if not np.any(valid):
        return 0, current_bytes

    distance = sampled_depth[valid].astype(np.float64)
    # RIGHT_HANDED_Y_UP is the ZED/OpenGL convention: +X right, +Y up,
    # and the camera looks along -Z.  DEPTH is a positive optical distance.
    points_camera = np.column_stack(
        (
            (uu[valid].astype(np.float64) - cx) * distance / fx,
            (cy - vv[valid].astype(np.float64)) * distance / fy,
            -distance,
        )
    )

    # pose_data().m is a column-vector camera-to-WORLD transform.  With row
    # vectors in NumPy this is points_camera @ R.T + t.
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    points_world = points_camera @ rotation.T + translation
    finite_points = np.isfinite(points_world).all(axis=1)
    if not np.any(finite_points):
        return 0, current_bytes
    points_world = points_world[finite_points]
    colors_rgb = sampled_bgr[valid][finite_points][:, ::-1]

    records = np.empty(len(points_world), dtype=POINT_DTYPE)
    records["x"] = points_world[:, 0].astype(np.float32)
    records["y"] = points_world[:, 1].astype(np.float32)
    records["z"] = points_world[:, 2].astype(np.float32)
    records["red"] = colors_rgb[:, 0]
    records["green"] = colors_rgb[:, 1]
    records["blue"] = colors_rgb[:, 2]

    next_bytes = current_bytes + records.nbytes
    if next_bytes + PLY_HEADER_RESERVE_BYTES >= HARD_OUTPUT_LIMIT_BYTES:
        raise RuntimeError(
            "Point cloud would exceed the hard 2,000,000,000-byte limit; "
            "increase pixel stride or lower the frame count."
        )
    raw_file.write(records.tobytes(order="C"))
    return len(records), next_bytes


def _build_point_cloud(
    output_dir: Path,
    source_dir: Path,
    records: list[dict[str, Any]],
    camera_meta: dict[str, Any],
    pixel_stride: int,
    min_depth: float,
    max_depth: float,
) -> tuple[Path, int, int, int]:
    intrinsics = camera_meta["left_rectified_intrinsics_px"]
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    raw_path = output_dir / "scene_world_rgb.ply.records.tmp"
    final_tmp_path = output_dir / "scene_world_rgb.ply.tmp"
    final_path = output_dir / "scene_world_rgb.ply"
    point_count = 0
    raw_bytes = 0
    valid_pose_frames = 0
    try:
        with raw_path.open("wb") as raw_file:
            for index, record in enumerate(records):
                if record["matrix"] is None or not record["pose_valid"]:
                    continue
                valid_pose_frames += 1
                depth_path = source_dir / "depth_raw" / f"depth_{index:06d}.npy"
                image_path = source_dir / "left" / f"left_{index:06d}.png"
                if not depth_path.is_file() or not image_path.is_file():
                    raise FileNotFoundError(
                        f"Missing source frame {index}: {depth_path} / {image_path}"
                    )
                added, raw_bytes = _append_frame_points(
                    raw_file=raw_file,
                    depth_path=depth_path,
                    image_path=image_path,
                    matrix=record["matrix"],
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    pixel_stride=pixel_stride,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    current_bytes=raw_bytes,
                )
                point_count += added
                if index % 25 == 0 or index == len(records) - 1:
                    print(
                        f"Point-cloud frames: {index + 1}/{len(records)}, "
                        f"points={point_count:,}, raw={raw_bytes / 1e6:.1f} MB",
                        flush=True,
                    )

        header = _ply_header(point_count)
        final_size = len(header) + raw_bytes
        if final_size >= HARD_OUTPUT_LIMIT_BYTES:
            raise RuntimeError(
                f"Final PLY size {final_size} exceeds the hard limit "
                f"{HARD_OUTPUT_LIMIT_BYTES}"
            )
        with final_tmp_path.open("wb") as final_file:
            final_file.write(header)
            with raw_path.open("rb") as raw_file:
                shutil.copyfileobj(raw_file, final_file, length=8 * 1024 * 1024)
            final_file.flush()
            os.fsync(final_file.fileno())
        os.replace(final_tmp_path, final_path)
    finally:
        if raw_path.exists():
            raw_path.unlink()
        if final_tmp_path.exists():
            final_tmp_path.unlink()

    actual_size = final_path.stat().st_size
    if actual_size >= HARD_OUTPUT_LIMIT_BYTES:
        raise RuntimeError(f"Final PLY size {actual_size} exceeds the hard limit")
    return final_path, point_count, actual_size, valid_pose_frames


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequentially track an SVO2 and export a bounded WORLD RGB point cloud."
    )
    parser.add_argument("svo", nargs="?", type=Path, default=DEFAULT_SVO)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing the 1000 sampled left/depth files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1000,
        help="Number of source samples to fuse across the complete SVO (default: 1000).",
    )
    parser.add_argument(
        "--pixel-stride",
        type=int,
        default=0,
        help="Pixel sampling stride; 0 selects one automatically under the size limit.",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.05,
        help="Ignore depth values at or below this value in meters.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=50.0,
        help="Ignore depth values above this value in meters (default: 50 m).",
    )
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")
    if args.pixel_stride < 0:
        parser.error("--pixel-stride must be >= 0")
    if args.min_depth < 0 or args.max_depth <= args.min_depth:
        parser.error("require 0 <= --min-depth < --max-depth")
    return args


def main() -> int:
    args = _parse_args()
    svo_path = args.svo.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not svo_path.is_file():
        raise FileNotFoundError(f"SVO/SVO2 file not found: {svo_path}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the total frame count once with the same recording and configuration
    # used for the sequential pose pass.  The pass itself validates it again.
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    zed = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open SVO2: {_status_name(status)}")
    try:
        total_frames = int(zed.get_svo_number_of_frames())
    finally:
        zed.close()

    target_positions = _sample_positions(total_frames, args.max_frames)
    if not target_positions:
        raise RuntimeError("No source frames available for point-cloud export")
    source_rows = _load_source_rows(source_dir, target_positions)
    del source_rows  # The old pose values are intentionally not reused.

    first_depth_path = source_dir / "depth_raw" / "depth_000000.npy"
    first_depth = np.load(first_depth_path, mmap_mode="r")
    if first_depth.dtype != np.float32 or first_depth.ndim != 2:
        raise RuntimeError(
            f"Unexpected source depth array: {first_depth.dtype} {first_depth.shape}"
        )
    resolution_height, resolution_width = first_depth.shape
    pixel_stride, estimated_points = _choose_pixel_stride(
        width=resolution_width,
        height=resolution_height,
        frame_count=len(target_positions),
        requested_stride=args.pixel_stride,
    )
    print(f"SVO total frames: {total_frames}")
    print(f"Target samples: {len(target_positions)}")
    print(
        f"Adaptive pixel stride: {pixel_stride} "
        f"(worst-case estimate {estimated_points:,} points, "
        f"{estimated_points * POINT_DTYPE.itemsize / 1e9:.3f} GB)",
        flush=True,
    )

    records, camera_meta, total_frames_from_pass = _run_sequential_pose_pass(
        svo_path=svo_path,
        target_positions=target_positions,
    )
    if total_frames_from_pass != total_frames:
        raise RuntimeError("SVO frame count changed between initialization and pose pass")

    pose_csv_path = output_dir / "pose_world_sequential.csv"
    _write_pose_csv(pose_csv_path, records)
    pointcloud_path, point_count, pointcloud_size, valid_pose_frames = _build_point_cloud(
        output_dir=output_dir,
        source_dir=source_dir,
        records=records,
        camera_meta=camera_meta,
        pixel_stride=pixel_stride,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )

    metadata = {
        "input_file": str(svo_path),
        "source_data_dir": str(source_dir),
        "sdk_version": str(sl.Camera.get_sdk_version()),
        "total_svo_frames": total_frames,
        "sampled_svo_frames": len(target_positions),
        "sampled_svo_position_start": target_positions[0],
        "sampled_svo_position_end": target_positions[-1],
        "depth": {
            "source": "pre-exported depth_raw/*.npy",
            "mode": "NEURAL",
            "dtype": "float32",
            "unit": "meter",
            "aligned_to": "LEFT",
            "min_filter_m": args.min_depth,
            "max_filter_m": args.max_depth,
        },
        "coordinate_system": "RIGHT_HANDED_Y_UP",
        "coordinate_units": "METER",
        "camera_frame": "LEFT_EYE",
        "pose": {
            "source": "ZED SDK positional tracking recomputed during sequential SVO replay",
            "reference_frame": "WORLD",
            "transform": "T_world_camera; p_world = T_world_camera @ p_camera",
            "matrix_layout": "4x4 row-major in CSV; translation in m03,m13,m23",
            "quaternion_order": "x,y,z,w",
            "world_origin": "first positional-tracking frame",
            "direct_svo_seek_used": False,
            "tracking_state_counts": {
                state: sum(record["tracking_state"] == state for record in records)
                for state in sorted({record["tracking_state"] for record in records})
            },
            "valid_pose_frames": valid_pose_frames,
        },
        "camera": camera_meta,
        "point_cloud": {
            "file": str(pointcloud_path),
            "format": "PLY binary_little_endian",
            "fields": "float32 x,y,z; uint8 red,green,blue",
            "fusion": "world-frame accumulation of sampled frames",
            "pixel_stride": pixel_stride,
            "input_depth_shape": [int(resolution_height), int(resolution_width)],
            "point_count": point_count,
            "file_size_bytes": pointcloud_size,
            "file_size_gb_decimal": pointcloud_size / 1e9,
            "hard_limit_bytes": HARD_OUTPUT_LIMIT_BYTES,
            "safe_target_limit_bytes": SAFE_OUTPUT_LIMIT_BYTES,
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        metadata_file.write("\n")

    print(f"Point cloud: {pointcloud_path}")
    print(f"Point count: {point_count:,}")
    print(f"File size: {pointcloud_size:,} bytes ({pointcloud_size / 1e9:.3f} GB)")
    print(f"Pose CSV: {pose_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
