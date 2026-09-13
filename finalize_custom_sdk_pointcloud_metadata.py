"""Validate and document an already-exported Custom-calibration point cloud.

This is intentionally a lightweight finalization pass.  It does not replay the
SVO or rewrite the large PLY.  It verifies the files produced by
``rerun_custom_calibration.py`` and opens the SVO once with the same Custom
calibration override to prove that the ZED SDK exposes the requested raw
intrinsics and stereo transform at runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import rerun_custom_calibration as pipeline


ROOT = Path(__file__).resolve().parent
DEFAULT_SVO = ROOT / "20260802_150233.svo2"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "20260802_150233_custom_sdk_pointcloud_full"
DEFAULT_SOURCE_CALIBRATION = ROOT / "Calibration" / "zed_custom_opencv.yml"
DEFAULT_INTRINSICS = ROOT / "Calibration" / "标定结果" / "camera_intrinsics.yaml"
DEFAULT_EXTRINSICS = ROOT / "Calibration" / "标定结果" / "stereo_extrinsics.yaml"
POINT_RECORD_SIZE = 15  # float32 xyz + uint8 rgb


def _number(value: str) -> float:
    parsed = float(value)
    if not pipeline.np.isfinite(parsed):
        raise RuntimeError(f"Non-finite numeric value in CSV: {value!r}")
    return parsed


def _calibration_errors(
    actual: pipeline.Calibration, expected: pipeline.Calibration
) -> dict[str, float]:
    pairs = {
        "width": (actual.width, expected.width),
        "height": (actual.height, expected.height),
        "K_LEFT": (actual.k_left, expected.k_left),
        "K_RIGHT": (actual.k_right, expected.k_right),
        "D_LEFT": (actual.d_left, expected.d_left),
        "D_RIGHT": (actual.d_right, expected.d_right),
        "R_rodrigues": (actual.r, expected.r),
        "T_mm": (actual.t_mm, expected.t_mm),
    }
    errors: dict[str, float] = {}
    for name, (actual_value, expected_value) in pairs.items():
        errors[name] = float(
            pipeline.np.max(
                pipeline.np.abs(
                    pipeline.np.asarray(actual_value, dtype=pipeline.np.float64)
                    - pipeline.np.asarray(expected_value, dtype=pipeline.np.float64)
                )
            )
        )
    return errors


def _read_ply(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Point cloud not found: {path}")

    header_lines: list[str] = []
    header_bytes = 0
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError("PLY header has no end_header marker")
            header_bytes += len(line)
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

    if header_lines[:2] != ["ply", "format binary_little_endian 1.0"]:
        raise RuntimeError(f"Unexpected PLY format in {path}: {header_lines[:2]}")
    vertex_line = next(
        (line for line in header_lines if line.startswith("element vertex ")), None
    )
    if vertex_line is None:
        raise RuntimeError(f"PLY has no vertex element: {path}")
    vertex_count = int(vertex_line.split()[2])
    expected_size = header_bytes + vertex_count * POINT_RECORD_SIZE
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"PLY size mismatch: header={header_bytes}, vertices={vertex_count}, "
            f"expected={expected_size}, actual={actual_size}"
        )

    # Read only one record as a cheap binary-layout sanity check.
    with path.open("rb") as handle:
        handle.seek(header_bytes)
        first = pipeline.np.fromfile(handle, dtype=pipeline.POINT_DTYPE, count=1)
    if first.size != 1 or not pipeline.np.isfinite(
        pipeline.np.asarray([first["x"][0], first["y"][0], first["z"][0]])
    ).all():
        raise RuntimeError("The first binary PLY vertex is invalid")

    return {
        "format": "binary_little_endian 1.0",
        "header_bytes": header_bytes,
        "vertex_count": vertex_count,
        "record_size_bytes": POINT_RECORD_SIZE,
        "expected_file_size_bytes": expected_size,
        "actual_file_size_bytes": actual_size,
        "size_check_passed": True,
    }


def _read_pose_csv(path: Path) -> dict[str, Any]:
    required = {
        "frame_index",
        "svo_position",
        "tracking_state",
        "pose_valid",
        "pose_confidence",
        "m03",
        "m13",
        "m23",
    }
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"Pose CSV is missing required columns: {path}")
        rows.extend(reader)
    if not rows:
        raise RuntimeError(f"Pose CSV is empty: {path}")

    frame_indices = [int(row["frame_index"]) for row in rows]
    svo_positions = [int(row["svo_position"]) for row in rows]
    expected = list(range(len(rows)))
    if frame_indices != expected or svo_positions != expected:
        raise RuntimeError("Pose CSV frame indices are not a complete sequential replay")

    states = Counter(row["tracking_state"] for row in rows)
    valid_rows = [row for row in rows if row["pose_valid"] == "1"]
    confidence = [
        _number(row["pose_confidence"])
        for row in valid_rows
        if row["pose_confidence"] not in {"NaN", "nan", "NAN"}
    ]
    positions = pipeline.np.asarray(
        [[_number(row[axis]) for axis in ("m03", "m13", "m23")] for row in valid_rows],
        dtype=pipeline.np.float64,
    )
    trajectory: dict[str, Any] = {}
    if len(positions) >= 2:
        steps = pipeline.np.linalg.norm(pipeline.np.diff(positions, axis=0), axis=1)
        trajectory = {
            "path_length_m": float(pipeline.np.sum(steps)),
            "endpoint_displacement_m": float(
                pipeline.np.linalg.norm(positions[-1] - positions[0])
            ),
            "position_min_m": positions.min(axis=0).tolist(),
            "position_max_m": positions.max(axis=0).tolist(),
            "step_p50_m": float(pipeline.np.percentile(steps, 50)),
            "step_p95_m": float(pipeline.np.percentile(steps, 95)),
            "step_max_m": float(pipeline.np.max(steps)),
        }

    confidence_summary: dict[str, Any] = {}
    if confidence:
        values = pipeline.np.asarray(confidence, dtype=pipeline.np.float64)
        confidence_summary = {
            "min": float(values.min()),
            "p05": float(pipeline.np.percentile(values, 5)),
            "median": float(pipeline.np.median(values)),
            "p95": float(pipeline.np.percentile(values, 95)),
            "max": float(values.max()),
        }

    return {
        "file": str(path),
        "rows": len(rows),
        "first_frame_index": frame_indices[0],
        "last_frame_index": frame_indices[-1],
        "tracking_state_counts": dict(sorted(states.items())),
        "valid_pose_frames": len(valid_rows),
        "valid_pose_ratio": len(valid_rows) / len(rows),
        "pose_confidence": confidence_summary,
        "trajectory": trajectory,
        "complete_sequential_indices": True,
    }


def _read_mapping_csv(path: Path, point_count: int) -> dict[str, Any]:
    required = {
        "map_index",
        "frame_index",
        "svo_position",
        "tracking_state",
        "pose_valid",
        "point_count",
    }
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"Mapping CSV is missing required columns: {path}")
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"Mapping CSV is empty: {path}")

    map_indices = [int(row["map_index"]) for row in rows]
    frame_indices = [int(row["frame_index"]) for row in rows]
    svo_positions = [int(row["svo_position"]) for row in rows]
    counts = [int(row["point_count"]) for row in rows]
    if map_indices != list(range(len(rows))):
        raise RuntimeError("Mapping CSV map_index is not sequential")
    if frame_indices != svo_positions:
        raise RuntimeError("Mapping CSV frame and SVO positions differ")
    if any(count <= 0 for count in counts):
        raise RuntimeError("Mapping CSV contains an empty mapping frame")
    sum_points = sum(counts)
    if sum_points != point_count:
        raise RuntimeError(
            f"Mapping/PLY point count mismatch: mapping={sum_points}, PLY={point_count}"
        )

    return {
        "file": str(path),
        "rows": len(rows),
        "first_svo_position": svo_positions[0],
        "last_svo_position": svo_positions[-1],
        "point_count_sum": sum_points,
        "point_count_min_per_frame": min(counts),
        "point_count_median_per_frame": float(pipeline.np.median(counts)),
        "point_count_max_per_frame": max(counts),
        "all_mapping_frames_valid": all(row["pose_valid"] == "1" for row in rows),
        "point_count_matches_ply": True,
    }


def _sdk_verification(
    svo_path: Path, calibration_path: Path, calibration: pipeline.Calibration
) -> tuple[str, int, dict[str, Any]]:
    zed = pipeline.sl.Camera()
    status = zed.open(
        pipeline._make_init(svo_path, calibration_path, pipeline.sl.DEPTH_MODE.NEURAL)
    )
    pipeline._check_status(status, "open SVO for point-cloud metadata verification")
    try:
        info = zed.get_camera_information()
        frame_count = int(zed.get_svo_number_of_frames())
        metadata = pipeline._verify_sdk_override(info, calibration)
        return str(pipeline.sl.Camera.get_sdk_version()), frame_count, metadata
    finally:
        zed.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an existing Custom-calibration ZED SDK point-cloud export."
    )
    parser.add_argument("--svo", type=Path, default=DEFAULT_SVO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-calibration", type=Path, default=DEFAULT_SOURCE_CALIBRATION)
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--extrinsics", type=Path, default=DEFAULT_EXTRINSICS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    svo_path = args.svo.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source_calibration_path = args.source_calibration.expanduser().resolve()
    intrinsics_path = args.intrinsics.expanduser().resolve()
    extrinsics_path = args.extrinsics.expanduser().resolve()
    runtime_calibration_path = output_dir / "zed_custom_opencv.yml"
    pointcloud_path = output_dir / "scene_world_rgb.ply"
    pose_path = output_dir / "pose_world.csv"
    mapping_path = output_dir / "mapping_frame_index.csv"
    area_map_path = output_dir / "area_map_custom.area"

    required_files = [
        svo_path,
        source_calibration_path,
        runtime_calibration_path,
        pointcloud_path,
        pose_path,
        mapping_path,
        area_map_path,
        intrinsics_path,
        extrinsics_path,
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required point-cloud artifact(s) missing: " + ", ".join(missing))

    runtime_calibration = pipeline._read_opencv_calibration(runtime_calibration_path)
    source_calibration = pipeline._read_opencv_calibration(source_calibration_path)
    cameras, extrinsics = pipeline._load_source_calibration(intrinsics_path, extrinsics_path)
    derived_calibration = pipeline._make_calibration(
        cameras,
        extrinsics,
        runtime_calibration.width,
        runtime_calibration.height,
        inverse=False,
    )
    pipeline._validate_calibration_against_source(runtime_calibration, cameras, extrinsics)

    source_file_errors = _calibration_errors(runtime_calibration, source_calibration)
    derived_errors = _calibration_errors(runtime_calibration, derived_calibration)
    calibration_file_match = max(source_file_errors.values()) <= 1e-6
    calibration_source_match = max(derived_errors.values()) <= 1e-6
    if not calibration_file_match or not calibration_source_match:
        raise RuntimeError(
            "Runtime output calibration does not exactly match the Custom source: "
            f"source_file_errors={source_file_errors}, derived_errors={derived_errors}"
        )

    ply = _read_ply(pointcloud_path)
    pose = _read_pose_csv(pose_path)
    mapping = _read_mapping_csv(mapping_path, ply["vertex_count"])
    sdk_version, sdk_frame_count, sdk_camera = _sdk_verification(
        svo_path, runtime_calibration_path, runtime_calibration
    )
    if sdk_frame_count != pose["rows"]:
        raise RuntimeError(
            f"SVO/pose frame count mismatch: SDK={sdk_frame_count}, pose={pose['rows']}"
        )

    area_map_size = area_map_path.stat().st_size
    if area_map_size <= 0:
        raise RuntimeError(f"Area Memory map is empty: {area_map_path}")

    metadata = {
        "input_file": str(svo_path),
        "output_dir": str(output_dir),
        "sdk_version": sdk_version,
        "verification": {
            "passed": True,
            "point_cloud_binary_layout": ply["size_check_passed"],
            "mapping_sum_equals_ply": mapping["point_count_matches_ply"],
            "pose_csv_is_complete_sequential_replay": pose["complete_sequential_indices"],
            "custom_file_matches_calibration_source": calibration_file_match,
            "custom_file_matches_intrinsics_extrinsics": calibration_source_match,
            "sdk_runtime_custom_override": sdk_camera[
                "custom_calibration_override_verification"
            ],
        },
        "calibration": {
            "source_opencv_file": str(source_calibration_path),
            "source_intrinsics": str(intrinsics_path),
            "source_extrinsics": str(extrinsics_path),
            "runtime_opencv_file": str(runtime_calibration_path),
            "image_size": {
                "width": runtime_calibration.width,
                "height": runtime_calibration.height,
            },
            "left_K": runtime_calibration.k_left.tolist(),
            "right_K": runtime_calibration.k_right.tolist(),
            "left_D_14": runtime_calibration.d_left.tolist(),
            "right_D_14": runtime_calibration.d_right.tolist(),
            "R_rodrigues": runtime_calibration.r.tolist(),
            "T_mm": runtime_calibration.t_mm.tolist(),
            "translation_norm_mm": float(pipeline.np.linalg.norm(runtime_calibration.t_mm)),
            "source_file_max_abs_error": max(source_file_errors.values()),
            "derived_source_max_abs_error": max(derived_errors.values()),
            "source_file_errors": source_file_errors,
            "derived_source_errors": derived_errors,
        },
        "camera_runtime_readback": sdk_camera,
        "processing": {
            "depth_mode": "NEURAL",
            "depth_source": "ZED SDK MEASURE.XYZRGBA",
            "coordinate_units": "METER",
            "coordinate_system": "RIGHT_HANDED_Y_UP",
            "tracking_mode": "GEN_1",
            "enable_imu_fusion": True,
            "enable_area_memory": True,
            "enable_pose_smoothing": False,
            "camera_disable_self_calib": True,
            "optional_opencv_calibration_file": str(runtime_calibration_path),
            "pose_reference_frame": "WORLD",
            "pose_camera": "LEFT_EYE",
            "pose_transform": "T_world_left; p_world = T_world_left @ p_left",
            "sequential_replay": True,
            "replayed_frames": pose["rows"],
            "mapping_frames": mapping["rows"],
            "mapping_selection": (
                "uniform source positions excluding initial unavailable frame; "
                "all source frames replayed sequentially"
            ),
            "pixel_stride": 5,
            "pixel_stride_selection": "automatically selected from the safe output-size limit",
            "min_optical_depth_m": 0.05,
            "max_optical_depth_m": 50.0,
        },
        "replay": {
            "sdk_reported_total_frames": sdk_frame_count,
            "pose": pose,
            "mapping": mapping,
        },
        "point_cloud": {
            "file": str(pointcloud_path),
            "format": ply["format"],
            "fields": "float32 x,y,z; uint8 red,green,blue",
            "source": "ZED SDK MEASURE.XYZRGBA with Custom calibration override",
            "fusion": "same-frame T_world_left transform and streaming accumulation",
            "point_count": ply["vertex_count"],
            "file_size_bytes": ply["actual_file_size_bytes"],
            "file_size_gb_decimal": ply["actual_file_size_bytes"] / 1e9,
            "pixel_stride": 5,
            "min_optical_depth_m": 0.05,
            "max_optical_depth_m": 50.0,
        },
        "outputs": {
            "pointcloud": str(pointcloud_path),
            "pose_csv": str(pose_path),
            "mapping_frame_index_csv": str(mapping_path),
            "area_map": {
                "file": str(area_map_path),
                "size_bytes": area_map_size,
            },
            "runtime_calibration": str(runtime_calibration_path),
        },
    }
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

    print(f"Validation passed: {output_dir}")
    print(f"SDK: {sdk_version}; replayed frames: {sdk_frame_count}")
    print(f"Custom calibration override: PASS")
    print(f"Point cloud: {ply['vertex_count']:,} points; {ply['actual_file_size_bytes']:,} bytes")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
