"""Fuse the existing 1000 sampled ZED depth frames with a GEN_1 pose pass.

The pose CSV is produced by ``benchmark_tracking.py`` using a sequential SVO
replay.  The depth and RGB files are the already exported NEURAL, rectified
left-camera data.  This script deliberately does not seek through the SVO
again: it validates the frame correspondence, obtains the recorded camera
calibration, and writes a bounded WORLD-frame binary PLY.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import export_pointcloud as ep


DEFAULT_SVO = Path(__file__).resolve().with_name("20260802_150233.svo2")
DEFAULT_SOURCE_DIR = DEFAULT_SVO.parent / "output" / "20260802_150233_sample1000"
DEFAULT_TRACKING_DIR = (
    DEFAULT_SVO.parent / "output" / "20260802_150233_tracking_gen1_neural"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_SVO.parent / "output" / "20260802_150233_pointcloud1000_gen1"
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as json_file:
        value = json.load(json_file)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _open_camera_for_metadata(svo_path: Path) -> tuple[dict[str, Any], int]:
    init = ep.sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = ep.sl.DEPTH_MODE.NEURAL
    init.coordinate_units = ep.sl.UNIT.METER
    init.coordinate_system = ep.sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    zed = ep.sl.Camera()
    status = zed.open(init)
    ep._check_status(status, "open SVO2 for camera metadata")
    try:
        info = zed.get_camera_information()
        total_frames = int(zed.get_svo_number_of_frames())
        camera_meta = ep._camera_metadata(info)
    finally:
        zed.close()
    return camera_meta, total_frames


def _parse_matrix(row: dict[str, str]) -> np.ndarray | None:
    try:
        values = [
            float(row[f"m{matrix_row}{matrix_column}"])
            for matrix_row in range(4)
            for matrix_column in range(4)
        ]
    except (KeyError, TypeError, ValueError):
        return None
    matrix = np.asarray(values, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(matrix).all():
        return None
    return matrix


def _load_gen1_records(
    pose_csv: Path,
    target_positions: list[int],
    total_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not pose_csv.is_file():
        raise FileNotFoundError(f"GEN_1 pose CSV not found: {pose_csv}")

    with pose_csv.open(newline="", encoding="utf-8") as pose_file:
        reader = csv.DictReader(pose_file)
        if reader.fieldnames != ep.POSE_HEADER:
            raise RuntimeError(f"Unexpected GEN_1 pose CSV header: {pose_csv}")
        rows = list(reader)

    if len(rows) != total_frames:
        raise RuntimeError(
            f"GEN_1 pose row count {len(rows)} does not match SVO frame count "
            f"{total_frames}"
        )

    by_svo_position: dict[int, dict[str, str]] = {}
    for expected_index, row in enumerate(rows):
        try:
            frame_index = int(row["frame_index"])
            svo_position = int(row["svo_position"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Invalid frame index in {pose_csv}") from error
        if frame_index != expected_index or svo_position != expected_index:
            raise RuntimeError(
                "GEN_1 pose CSV is not a consecutive full replay at row "
                f"{expected_index}: frame_index={frame_index}, "
                f"svo_position={svo_position}"
            )
        if svo_position in by_svo_position:
            raise RuntimeError(f"Duplicate SVO position {svo_position}")
        by_svo_position[svo_position] = row

    selected: list[dict[str, Any]] = []
    state_counts: dict[str, int] = {}
    for selected_index, svo_position in enumerate(target_positions):
        source_row = by_svo_position.get(svo_position)
        if source_row is None:
            raise RuntimeError(f"GEN_1 pose is missing SVO position {svo_position}")

        tracking_state = source_row["tracking_state"]
        state_counts[tracking_state] = state_counts.get(tracking_state, 0) + 1
        pose_valid = source_row["pose_valid"] == "1"
        matrix = _parse_matrix(source_row) if pose_valid else None
        if pose_valid and matrix is None:
            raise RuntimeError(
                f"GEN_1 pose matrix is invalid at SVO position {svo_position}"
            )

        selected_row = [
            selected_index,
            svo_position,
            source_row["timestamp_ns"],
            tracking_state,
            source_row["pose_valid"],
            source_row["pose_confidence"],
            *(source_row[field] for field in ep.POSE_HEADER[6:]),
        ]
        selected.append(
            {
                "row": selected_row,
                "matrix": matrix,
                "pose_valid": pose_valid,
                "tracking_state": tracking_state,
            }
        )
    return selected, state_counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse 1000 sampled NEURAL depth frames using GEN_1 WORLD poses."
    )
    parser.add_argument("svo", nargs="?", type=Path, default=DEFAULT_SVO)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--tracking-dir", type=Path, default=DEFAULT_TRACKING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1000,
        help="Number of uniformly distributed source frames to fuse.",
    )
    parser.add_argument(
        "--pixel-stride",
        type=int,
        default=0,
        help="Pixel stride; 0 selects the largest density that stays below 1.9 GB.",
    )
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=50.0)
    args = parser.parse_args()
    if args.max_frames <= 0:
        parser.error("--max-frames must be > 0")
    if args.pixel_stride < 0:
        parser.error("--pixel-stride must be >= 0")
    if args.min_depth < 0 or args.max_depth <= args.min_depth:
        parser.error("require 0 <= --min-depth < --max-depth")
    return args


def main() -> int:
    args = _parse_args()
    svo_path = args.svo.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve()
    tracking_dir = args.tracking_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not svo_path.is_file():
        raise FileNotFoundError(f"SVO/SVO2 file not found: {svo_path}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source data directory not found: {source_dir}")
    if not tracking_dir.is_dir():
        raise FileNotFoundError(f"GEN_1 tracking directory not found: {tracking_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_meta, total_frames = _open_camera_for_metadata(svo_path)
    target_positions = ep._sample_positions(total_frames, args.max_frames)
    source_rows = ep._load_source_rows(source_dir, target_positions)
    del source_rows  # Only the frame correspondence is reused, never old poses.

    source_metadata_path = source_dir / "metadata.json"
    source_metadata = _load_json(source_metadata_path)
    if source_metadata.get("depth_mode") != "NEURAL":
        raise RuntimeError("Source depth data is not marked as NEURAL")
    if source_metadata.get("coordinate_system") != "RIGHT_HANDED_Y_UP":
        raise RuntimeError("Source coordinate system is not RIGHT_HANDED_Y_UP")
    if source_metadata.get("unit") != "METER":
        raise RuntimeError("Source depth data is not marked as meter units")

    first_depth_path = source_dir / "depth_raw" / "depth_000000.npy"
    first_depth = np.load(first_depth_path, mmap_mode="r")
    if first_depth.dtype != np.float32 or first_depth.ndim != 2:
        raise RuntimeError(
            f"Unexpected source depth array: {first_depth.dtype} {first_depth.shape}"
        )
    resolution_height, resolution_width = first_depth.shape
    expected_width = camera_meta["resolution"]["width"]
    expected_height = camera_meta["resolution"]["height"]
    if (resolution_width, resolution_height) != (expected_width, expected_height):
        raise RuntimeError(
            "Depth resolution does not match camera calibration: "
            f"depth={(resolution_width, resolution_height)}, "
            f"camera={(expected_width, expected_height)}"
        )

    pixel_stride, estimated_points = ep._choose_pixel_stride(
        width=resolution_width,
        height=resolution_height,
        frame_count=len(target_positions),
        requested_stride=args.pixel_stride,
    )
    pose_csv = tracking_dir / "pose_gen1.csv"
    records, state_counts = _load_gen1_records(
        pose_csv=pose_csv,
        target_positions=target_positions,
        total_frames=total_frames,
    )
    valid_pose_frames = sum(
        record["matrix"] is not None and record["pose_valid"]
        for record in records
    )
    if valid_pose_frames != len(records):
        raise RuntimeError(
            f"Cannot fuse all requested frames: {valid_pose_frames}/{len(records)} "
            "have valid GEN_1 poses"
        )

    print(f"SVO total frames: {total_frames}")
    print(f"Target samples: {len(target_positions)}")
    print(
        f"GEN_1 pose CSV: {pose_csv} ({total_frames} full-replay rows)"
    )
    print(
        f"Adaptive pixel stride: {pixel_stride} "
        f"(worst-case estimate {estimated_points:,} points, "
        f"{estimated_points * ep.POINT_DTYPE.itemsize / 1e9:.3f} GB)",
        flush=True,
    )

    pose_output = output_dir / "pose_world_gen1.csv"
    ep._write_pose_csv(pose_output, records)
    pointcloud_path, point_count, pointcloud_size, fused_frames = ep._build_point_cloud(
        output_dir=output_dir,
        source_dir=source_dir,
        records=records,
        camera_meta=camera_meta,
        pixel_stride=pixel_stride,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )

    tracking_summary = _load_json(tracking_dir / "summary_gen1.json")
    metadata = {
        "input_file": str(svo_path),
        "source_data_dir": str(source_dir),
        "tracking_data_dir": str(tracking_dir),
        "sdk_version": str(ep.sl.Camera.get_sdk_version()),
        "total_svo_frames": total_frames,
        "sampled_svo_frames": len(target_positions),
        "sampled_svo_position_start": target_positions[0],
        "sampled_svo_position_end": target_positions[-1],
        "sampling": source_metadata.get("sampling"),
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
            "source": "ZED SDK positional tracking recomputed by sequential full-SVO replay",
            "mode": "GEN_1",
            "depth_mode_during_replay": "NEURAL",
            "enable_area_memory": False,
            "enable_imu_fusion": True,
            "enable_pose_smoothing": False,
            "reference_frame": "WORLD",
            "transform": "T_world_camera; p_world = T_world_camera @ p_camera",
            "matrix_layout": "4x4 row-major in CSV; translation in m03,m13,m23",
            "quaternion_order": "x,y,z,w",
            "world_origin": "first positional-tracking frame",
            "pose_camera": "LEFT_EYE",
            "direct_svo_seek_used": False,
            "full_replay_frames": total_frames,
            "tracking_state_counts_sampled": state_counts,
            "valid_pose_frames_sampled": fused_frames,
            "full_replay_summary": tracking_summary,
        },
        "camera": camera_meta,
        "point_cloud": {
            "file": str(pointcloud_path),
            "format": "PLY binary_little_endian",
            "fields": "float32 x,y,z; uint8 red,green,blue",
            "fusion": "world-frame accumulation of sampled left-camera depth frames",
            "pixel_stride": pixel_stride,
            "input_depth_shape": [int(resolution_height), int(resolution_width)],
            "point_count": point_count,
            "file_size_bytes": pointcloud_size,
            "file_size_gb_decimal": pointcloud_size / 1e9,
            "hard_limit_bytes": ep.HARD_OUTPUT_LIMIT_BYTES,
            "safe_target_limit_bytes": ep.SAFE_OUTPUT_LIMIT_BYTES,
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        metadata_file.write("\n")

    print(f"Point cloud: {pointcloud_path}")
    print(f"Point count: {point_count:,}")
    print(f"File size: {pointcloud_size:,} bytes ({pointcloud_size / 1e9:.3f} GB)")
    print(f"Pose CSV: {pose_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
