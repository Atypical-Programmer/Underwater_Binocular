"""Convert an ORB-SLAM3 TUM trajectory to a Metashape YPR reference file.

ORB-SLAM3's CameraTrajectory.txt is written by SaveTrajectoryTUM() as a
camera-to-world (Twc) pose:

    timestamp tx ty tz qx qy qz qw

The quaternion and translation use ORB-SLAM3's OpenCV camera frame
(+X right, +Y down, +Z forward).  This is already the camera-frame
convention expected by Metashape's YPR conversion, so no ZED-specific
axis flip is applied here.  Applying the ZED ``diag(1, -1, -1)`` conversion
to this file would rotate the trajectory incorrectly.

By default only frames marked pose_valid=1 and tracking_state=2 (OK) in the
ORB-SLAM3 tracking log are exported.  This keeps camera labels aligned with
the frames for which the SLAM system reports a valid pose.  Use
``--include-nonvalid`` only when an import of every trajectory line is
specifically required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = WORKSPACE / "output" / "20260802_150233_orbslam3_stereo"
DEFAULT_TRAJECTORY = DEFAULT_RESULT_DIR / "CameraTrajectory.txt"
DEFAULT_TRACKING_LOG = DEFAULT_RESULT_DIR / "tracking_log.csv"
DEFAULT_OUTPUT = DEFAULT_RESULT_DIR / "metashape_reference_orbslam3_ypr.csv"
DEFAULT_METADATA = DEFAULT_RESULT_DIR / "metashape_reference_orbslam3_ypr_metadata.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ORB-SLAM3 CameraTrajectory.txt into "
            "label,x,y,z,yaw,pitch,roll for Metashape."
        )
    )
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--tracking-log", type=Path, default=DEFAULT_TRACKING_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--include-nonvalid",
        action="store_true",
        help="also export trajectory rows whose tracking log pose is invalid",
    )
    return parser.parse_args()


def _round_timestamp(timestamp_sec: float) -> float:
    # CameraTrajectory.txt is written with six fractional digits.  Rounding
    # to that precision joins it robustly to tracking_log.csv, which stores
    # the original Python float representation.
    return round(float(timestamp_sec), 6)


def _read_tracking_log(path: Path) -> tuple[dict[float, dict[str, Any]], dict[str, Any]]:
    rows_by_timestamp: dict[float, dict[str, Any]] = {}
    state_counts: Counter[str] = Counter()
    valid_counts: Counter[str] = Counter()
    total_rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"frame", "timestamp_sec", "tracking_state", "pose_valid"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"tracking log missing columns: {sorted(missing)}")

        for row in reader:
            total_rows += 1
            timestamp = float(row["timestamp_sec"])
            key = _round_timestamp(timestamp)
            if key in rows_by_timestamp:
                raise ValueError(
                    f"duplicate tracking-log timestamp after rounding to 6 decimals: {key}"
                )

            state = str(row["tracking_state"])
            pose_valid = str(row["pose_valid"])
            state_counts[state] += 1
            valid_counts[pose_valid] += 1
            rows_by_timestamp[key] = {
                "frame": int(row["frame"]),
                "timestamp_sec": timestamp,
                "tracking_state": state,
                "pose_valid": pose_valid,
            }

    return rows_by_timestamp, {
        "rows": total_rows,
        "tracking_state_counts": dict(sorted(state_counts.items())),
        "pose_valid_counts": dict(sorted(valid_counts.items())),
    }


def _quaternion_to_rotation(qx: float, qy: float, qz: float, qw: float) -> tuple[np.ndarray, float]:
    """Return a proper rotation matrix and the unnormalised quaternion norm."""

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
    """Extract Metashape's yaw, pitch, roll angles in degrees.

    This follows Agisoft's published mat2ypr extraction convention.  The
    yaw sign and wrap are intentional: Metashape's yaw is reported clockwise
    in the aerial-camera convention and is equivalent modulo 360 degrees.
    """

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

    return (-math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


def _trajectory_rows(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 8:
                raise ValueError(
                    f"{path}:{line_number}: expected 8 TUM columns, got {len(fields)}"
                )
            yield line_number, [float(value) for value in fields[:8]]


def convert(
    trajectory_path: Path,
    tracking_log_path: Path,
    output_path: Path,
    metadata_path: Path,
    include_nonvalid: bool = False,
) -> dict[str, Any]:
    if not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)
    if not tracking_log_path.is_file():
        raise FileNotFoundError(tracking_log_path)

    tracking_by_timestamp, tracking_meta = _read_tracking_log(tracking_log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    source_rows = 0
    output_rows = 0
    skipped_nonvalid = 0
    unmatched_timestamps = 0
    duplicate_frames: list[int] = []
    used_frames: set[int] = set()
    state_counts_exported: Counter[str] = Counter()
    quaternion_norms: list[float] = []
    max_rotation_orthogonality_error = 0.0
    max_rotation_determinant_error = 0.0
    max_abs_position = 0.0
    first_frame: int | None = None
    last_frame: int | None = None
    first_timestamp: float | None = None
    last_timestamp: float | None = None

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["label", "x", "y", "z", "yaw", "pitch", "roll"])

        for line_number, values in _trajectory_rows(trajectory_path):
            source_rows += 1
            timestamp, tx, ty, tz, qx, qy, qz, qw = values
            tracking = tracking_by_timestamp.get(_round_timestamp(timestamp))
            if tracking is None:
                unmatched_timestamps += 1
                continue

            is_valid = tracking["pose_valid"] == "1" and tracking["tracking_state"] == "2"
            if not include_nonvalid and not is_valid:
                skipped_nonvalid += 1
                continue

            frame = int(tracking["frame"])
            if frame in used_frames:
                duplicate_frames.append(frame)
                continue
            used_frames.add(frame)

            rotation, quaternion_norm = _quaternion_to_rotation(qx, qy, qz, qw)
            quaternion_norms.append(quaternion_norm)
            orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
            determinant_error = abs(float(np.linalg.det(rotation)) - 1.0)
            max_rotation_orthogonality_error = max(
                max_rotation_orthogonality_error, orthogonality_error
            )
            max_rotation_determinant_error = max(max_rotation_determinant_error, determinant_error)

            yaw, pitch, roll = _metashape_ypr(rotation)
            label = f"left_{frame:06d}.png"
            writer.writerow(
                [
                    label,
                    f"{tx:.12f}",
                    f"{ty:.12f}",
                    f"{tz:.12f}",
                    f"{yaw:.12f}",
                    f"{pitch:.12f}",
                    f"{roll:.12f}",
                ]
            )

            output_rows += 1
            state_counts_exported[tracking["tracking_state"]] += 1
            first_frame = frame if first_frame is None else min(first_frame, frame)
            last_frame = frame if last_frame is None else max(last_frame, frame)
            first_timestamp = timestamp if first_timestamp is None else min(first_timestamp, timestamp)
            last_timestamp = timestamp if last_timestamp is None else max(last_timestamp, timestamp)
            max_abs_position = max(max_abs_position, abs(tx), abs(ty), abs(tz))

    if duplicate_frames:
        raise ValueError(
            "trajectory produced duplicate valid frame labels: "
            f"{duplicate_frames[:10]}"
        )
    if unmatched_timestamps:
        raise ValueError(
            f"{unmatched_timestamps} trajectory rows could not be matched to tracking_log.csv"
        )
    if output_rows == 0:
        raise ValueError("no trajectory rows were exported")

    metadata: dict[str, Any] = {
        "source": {
            "trajectory": str(trajectory_path.resolve()),
            "tracking_log": str(tracking_log_path.resolve()),
            "trajectory_format": "ORB-SLAM3 TUM CameraTrajectory.txt",
        },
        "input_rows": {
            "trajectory_rows": source_rows,
            "tracking_log_rows": tracking_meta["rows"],
            "trajectory_rows_unmatched": unmatched_timestamps,
            "trajectory_rows_skipped_nonvalid": skipped_nonvalid,
            "output_rows": output_rows,
        },
        "frame_alignment": {
            "label_pattern": "left_{frame:06d}.png",
            "frame_index_source": "tracking_log.csv:frame",
            "first_frame": first_frame,
            "last_frame": last_frame,
            "first_timestamp_sec": first_timestamp,
            "last_timestamp_sec": last_timestamp,
            "timestamp_join": "round both timestamps to 6 decimal places",
            "filter": (
                "none (--include-nonvalid)"
                if include_nonvalid
                else "tracking_log.pose_valid == 1 and tracking_state == 2 (OK)"
            ),
        },
        "pose_definition": {
            "transform": "Twc: ORB-SLAM3 camera frame to ORB-SLAM3 local world/map frame",
            "camera_axes": "+X right, +Y down, +Z forward (OpenCV camera convention)",
            "translation_units": "meters (stereo baseline in ORB-SLAM3 settings is metric)",
            "world_frame": "ORB-SLAM3 local map frame; no GPS/geographic georeference",
            "quaternion_input_order": "qx,qy,qz,qw",
            "axis_conversion": (
                "none; CameraTrajectory.txt already contains OpenCV-camera Twc. "
                "Do not apply ZED RHT_Y_UP diag(1,-1,-1) to this trajectory."
            ),
            "ypr_input_order": "yaw,pitch,roll",
            "ypr_units": "degrees",
            "ypr_convention": "Agisoft Metashape mat2ypr convention",
        },
        "quality": {
            "tracking_state_counts_all": tracking_meta["tracking_state_counts"],
            "pose_valid_counts_all": tracking_meta["pose_valid_counts"],
            "tracking_state_counts_exported": dict(sorted(state_counts_exported.items())),
            "quaternion_norm_min": min(quaternion_norms),
            "quaternion_norm_max": max(quaternion_norms),
            "max_rotation_orthogonality_error": max_rotation_orthogonality_error,
            "max_rotation_determinant_error": max_rotation_determinant_error,
            "max_abs_translation_component_m": max_abs_position,
        },
        "output": {
            "csv": str(output_path.resolve()),
            "header": "label,x,y,z,yaw,pitch,roll",
        },
    }

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return metadata


def main() -> int:
    args = _parse_args()
    metadata = convert(
        trajectory_path=args.trajectory,
        tracking_log_path=args.tracking_log,
        output_path=args.output,
        metadata_path=args.metadata,
        include_nonvalid=args.include_nonvalid,
    )
    # Keep console output ASCII-only so the script also works with the
    # default Windows cp1252 console encoding.
    print(f"Output CSV: {args.output.resolve()}")
    print(f"Metadata JSON: {args.metadata.resolve()}")
    print(f"Trajectory rows: {metadata['input_rows']['trajectory_rows']}")
    print(f"Exported rows: {metadata['input_rows']['output_rows']}")
    print(f"Skipped non-valid poses: {metadata['input_rows']['trajectory_rows_skipped_nonvalid']}")
    print(
        "Coordinates: ORB-SLAM3 local metric world; "
        "camera +X right/+Y down/+Z forward; no ZED axis flip"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
