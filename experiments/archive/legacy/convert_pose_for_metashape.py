"""Convert the GEN_1 ZED camera poses to a Metashape reference CSV.

Input poses are ZED camera-to-WORLD matrices in RIGHT_HANDED_Y_UP, where the
camera looks along -Z and +Y is up.  Metashape's local camera frame uses +X
right, +Y down and +Z forward.  The direct camera transform therefore uses
diag(1, -1, -1, 1).  For Metashape OPK reference angles, the same conversion
is represented by extracting OPK from the ZED 3x3 rotation after the camera
axis convention is accounted for; equivalently, this is the rotation matrix
used by Metashape's camera-to-world OPK convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT = (
    Path(__file__).resolve().parent
    / "output"
    / "20260802_150233_pointcloud1000_gen1"
    / "pose_world_gen1.csv"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "output"
    / "20260802_150233_pointcloud1000_gen1"
    / "metashape_reference_gen1.csv"
)
DEFAULT_METADATA = (
    Path(__file__).resolve().parent
    / "output"
    / "20260802_150233_pointcloud1000_gen1"
    / "metashape_reference_gen1_metadata.json"
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

# ZED RIGHT_HANDED_Y_UP uses +X right, +Y up, and the optical viewing
# direction along -Z. Metashape's camera frame for exterior orientation uses
# +X right, +Y down, and +Z forward. This is a camera-local conversion only;
# the WORLD coordinates of the camera center do not change.
ZED_TO_METASHAPE_CAMERA = np.diag([1.0, -1.0, -1.0])


def _parse_matrix(row: dict[str, str]) -> np.ndarray:
    values = [
        float(row[f"m{matrix_row}{matrix_column}"])
        for matrix_row in range(4)
        for matrix_column in range(4)
    ]
    matrix = np.asarray(values, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError("pose matrix contains non-finite values")
    return matrix


def _orthonormalize(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the closest proper rotation and its maximum correction."""

    u, _, vh = np.linalg.svd(rotation)
    corrected = u @ vh
    if np.linalg.det(corrected) < 0:
        u[:, -1] *= -1.0
        corrected = u @ vh
    correction = float(np.max(np.abs(corrected - rotation)))
    return corrected, correction


def _rotation_diagnostics(rotation: np.ndarray) -> tuple[float, float]:
    orth_error = float(
        np.max(np.abs(rotation.T @ rotation - np.eye(3, dtype=np.float64)))
    )
    determinant = float(np.linalg.det(rotation))
    return orth_error, determinant


def _rx(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotation_to_opk(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert a Metashape camera-to-world rotation matrix to OPK degrees.

    Metashape's OPK rotation matrix is represented as
    Rx(omega) @ Ry(phi) @ Rz(kappa) for the camera-to-world orientation
    after the camera-axis convention is removed.  The formulas below are the
    inverse of that product and include the two gimbal-lock branches.
    """

    rotation, correction = _orthonormalize(rotation)
    sine_phi = float(np.clip(rotation[0, 2], -1.0, 1.0))
    phi = math.asin(sine_phi)
    cosine_phi = math.cos(phi)

    if abs(cosine_phi) > 1e-10:
        omega = math.atan2(-rotation[1, 2], rotation[2, 2])
        kappa = math.atan2(-rotation[0, 1], rotation[0, 0])
    elif sine_phi > 0.0:
        # phi = +90 degrees; choose kappa = 0.
        omega = math.atan2(rotation[1, 0], rotation[1, 1])
        kappa = 0.0
    else:
        # phi = -90 degrees; choose kappa = 0.
        omega = math.atan2(-rotation[1, 0], rotation[1, 1])
        kappa = 0.0

    recovered = _rx(omega) @ _ry(phi) @ _rz(kappa)
    roundtrip_error = float(np.max(np.abs(recovered - rotation)))
    return (
        np.degrees([omega, phi, kappa]).astype(np.float64),
        max(correction, roundtrip_error),
    )


def _number(value: float) -> str:
    return f"{float(value):.12f}"


def convert(
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    rigid_only: bool = False,
) -> dict[str, Any]:
    with input_path.open(newline="", encoding="utf-8") as pose_file:
        reader = csv.DictReader(pose_file)
        if reader.fieldnames != POSE_HEADER:
            raise RuntimeError(f"Unexpected pose CSV header: {input_path}")
        rows = list(reader)

    if not rows:
        raise RuntimeError("No pose rows found")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_rotation_error = 0.0
    max_orthonormalization_correction = 0.0
    max_position_consistency_error = 0.0
    min_confidence = math.inf
    max_confidence = -math.inf
    input_tracking_valid_rows = 0
    skipped_tracking_invalid_rows = 0
    non_rigid_input_rows = 0
    orthonormalized_rows = 0
    skipped_non_rigid_rows = 0
    output_rows = 0
    previous_frame_index = None
    previous_svo_position = None

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["label", "x", "y", "z", "omega", "phi", "kappa"])

        for row in rows:
            frame_index = int(row["frame_index"])
            svo_position = int(row["svo_position"])
            if previous_frame_index is not None and frame_index <= previous_frame_index:
                raise RuntimeError(
                    f"Frame index is not strictly increasing: "
                    f"{previous_frame_index} -> {frame_index}"
                )
            if previous_svo_position is not None and svo_position <= previous_svo_position:
                raise RuntimeError(
                    f"SVO position is not strictly increasing: "
                    f"{previous_svo_position} -> {svo_position}"
                )
            previous_frame_index = frame_index
            previous_svo_position = svo_position

            # Metashape cannot use an invalid reference row. The first SVO
            # initialization frame is UNAVAILABLE in this recording, so it is
            # deliberately omitted rather than emitting NaN values.
            if row["tracking_state"] != "OK" or row["pose_valid"] != "1":
                skipped_tracking_invalid_rows += 1
                continue
            input_tracking_valid_rows += 1

            matrix = _parse_matrix(row)
            rotation_zed = matrix[:3, :3]
            orth_error, determinant = _rotation_diagnostics(rotation_zed)
            is_rigid = orth_error <= 1e-3 and abs(determinant - 1.0) <= 1e-3
            if not is_rigid:
                non_rigid_input_rows += 1
                if rigid_only:
                    skipped_non_rigid_rows += 1
                    continue
                rotation_zed, correction = _orthonormalize(rotation_zed)
                orthonormalized_rows += 1
                max_orthonormalization_correction = max(
                    max_orthonormalization_correction, correction
                )

            # ZED and Metashape use different local camera axes. Apply the
            # axis conversion on the right because the pose maps camera-local
            # coordinates to the common WORLD frame. The camera center remains
            # in the original metric WORLD coordinates.
            rotation_metashape = rotation_zed @ ZED_TO_METASHAPE_CAMERA
            opk, rotation_error = _rotation_to_opk(rotation_metashape)
            max_rotation_error = max(max_rotation_error, rotation_error)

            confidence = float(row["pose_confidence"])
            if np.isfinite(confidence):
                min_confidence = min(min_confidence, confidence)
                max_confidence = max(max_confidence, confidence)

            csv_position = np.asarray(
                [float(row["tx"]), float(row["ty"]), float(row["tz"])],
                dtype=np.float64,
            )
            position_error = float(np.max(np.abs(csv_position - matrix[:3, 3])))
            max_position_consistency_error = max(
                max_position_consistency_error, position_error
            )

            writer.writerow(
                [
                    f"left_{svo_position:06d}.png",
                    *(_number(value) for value in matrix[:3, 3]),
                    _number(opk[0]),
                    _number(opk[1]),
                    _number(opk[2]),
                ]
            )
            output_rows += 1

    if output_rows == 0:
        raise RuntimeError("No usable pose rows remained for Metashape export")

    if not np.isfinite(min_confidence):
        min_confidence = float("nan")
    if not np.isfinite(max_confidence):
        max_confidence = float("nan")
    metadata = {
        "source_pose_csv": str(input_path.resolve()),
        "output_reference_csv": str(output_path.resolve()),
        "source_row_count": len(rows),
        "row_count": output_rows,
        "image_label_pattern": "left_%06d.png, using SVO source position",
        "input_tracking_valid_rows": input_tracking_valid_rows,
        "skipped_tracking_invalid_rows": skipped_tracking_invalid_rows,
        "non_rigid_input_rows": non_rigid_input_rows,
        "orthonormalized_rows": orthonormalized_rows,
        "skipped_non_rigid_rows": skipped_non_rigid_rows,
        "rigid_only": rigid_only,
        "coordinate_units": "METER",
        "world_coordinate_system": "ZED RIGHT_HANDED_Y_UP / WORLD",
        "camera_coordinate_conversion": {
            "zed_camera": "+X right, +Y up, viewing direction -Z",
            "metashape_camera": "+X right, +Y down, viewing direction +Z",
            "zed_to_metashape_camera_axis_matrix": ZED_TO_METASHAPE_CAMERA.tolist(),
            "rotation_transform": "R_metashape = R_zed @ diag(1,-1,-1)",
            "position_conversion": "none; camera centers remain meters in the existing WORLD frame",
            "matrix_inverse": "not used",
        },
        "rotation_format": "Metashape Omega, Phi, Kappa in degrees",
        "rotation_source": "ZED pose 3x3 camera-to-WORLD rotation after local-axis conversion",
        "max_rotation_roundtrip_error": max_rotation_error,
        "max_orthonormalization_correction": max_orthonormalization_correction,
        "max_position_consistency_error_m": max_position_consistency_error,
        "pose_confidence_min": min_confidence,
        "pose_confidence_max": max_confidence,
        "import_reference_columns": "nxyzabc",
        "recommended_rotation_convention": "Omega-Phi-Kappa / OPK",
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        metadata_file.write("\n")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert GEN_1 ZED poses to a Metashape OPK reference CSV."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--rigid-only",
        action="store_true",
        help="Skip input pose matrices that are not proper rigid rotations.",
    )
    args = parser.parse_args()

    metadata = convert(
        input_path=args.input.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        metadata_path=args.metadata.expanduser().resolve(),
        rigid_only=args.rigid_only,
    )
    print(f"Converted rows: {metadata['row_count']}")
    print(f"Output CSV: {args.output.expanduser().resolve()}")
    print(f"Max OPK round-trip error: {metadata['max_rotation_roundtrip_error']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
