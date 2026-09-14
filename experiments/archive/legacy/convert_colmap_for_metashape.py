"""Export COLMAP camera poses as Metashape reference CSV files.

COLMAP stores each image pose in ``images.txt`` as a world-to-camera
transform:

    x_camera = R_wc @ x_world + t_wc

The camera center and camera-to-world rotation are therefore::

    C_world = -R_wc.T @ t_wc
    R_cw = R_wc.T

COLMAP and Metashape use the same OpenCV optical camera axes here (+X right,
+Y down, +Z forward).  The exported YPR/OPK angles are consequently extracted
from the camera-to-world rotation without an additional axis flip.

The direct-import files have the seven columns expected by Metashape's
Reference CSV importer:

    label,x,y,z,yaw,pitch,roll

and an OPK variant with:

    label,x,y,z,omega,phi,kappa

The COLMAP models in this project are independent reconstruction components.
Each model has its own arbitrary local coordinate frame, so a separate CSV is
written for every model directory rather than merging unrelated coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = (
    WORKSPACE
    / "output"
    / "20260802_150233_sample1000"
    / "sfm_aliked_adalam_2000_full_opencv_refined"
    / "sparse_all_models_strict_text"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE
    / "output"
    / "20260802_150233_sample1000"
    / "sfm_aliked_adalam_2000_full_opencv_refined"
    / "metashape_export"
)

FRAME_RE = re.compile(r"_(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)


@dataclass(frozen=True)
class ColmapPose:
    model_id: int
    image_id: int
    camera_id: int
    source_name: str
    label: str
    side: str
    frame_index: int | None
    qvec_wxyz: tuple[float, float, float, float]
    t_wc: np.ndarray
    r_wc: np.ndarray
    center_world: np.ndarray
    r_cw: np.ndarray


def _quaternion_to_rotation(qvec_wxyz: Iterable[float]) -> tuple[np.ndarray, float]:
    q = np.asarray(tuple(float(value) for value in qvec_wxyz), dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("COLMAP pose contains a non-finite quaternion")
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"invalid COLMAP quaternion norm: {norm}")
    qw, qx, qy, qz = q / norm
    rotation = np.asarray(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )
    return rotation, norm


def _parse_model(images_path: Path, model_id: int) -> list[ColmapPose]:
    if not images_path.is_file():
        raise FileNotFoundError(images_path)

    poses: list[ColmapPose] = []
    seen_labels: set[str] = set()
    with images_path.open("r", encoding="utf-8-sig") as handle:
        line_iterator = iter(handle)
        for line_number, line in enumerate(line_iterator, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            fields = stripped.split(maxsplit=9)
            if len(fields) != 10:
                raise ValueError(
                    f"{images_path}:{line_number}: expected COLMAP image pose line "
                    f"with 10 fields, got {len(fields)}"
                )

            # The second line of every COLMAP image record contains the 2D
            # observations. It is intentionally skipped; poses are on line 1.
            try:
                next(line_iterator)
            except StopIteration as error:
                raise ValueError(
                    f"{images_path}:{line_number}: missing POINTS2D line"
                ) from error

            image_id = int(fields[0])
            qvec = tuple(float(value) for value in fields[1:5])
            t_wc = np.asarray([float(value) for value in fields[5:8]], dtype=np.float64)
            camera_id = int(fields[8])
            source_name = fields[9].replace("\\", "/")
            label = Path(source_name).name
            if not label:
                raise ValueError(f"{images_path}:{line_number}: empty image label")
            label_key = label.casefold()
            if label_key in seen_labels:
                raise ValueError(f"duplicate image label in model {model_id}: {label}")
            seen_labels.add(label_key)

            if not np.all(np.isfinite(t_wc)):
                raise ValueError(f"{images_path}:{line_number}: non-finite translation")
            r_wc, _ = _quaternion_to_rotation(qvec)
            r_cw = r_wc.T
            center_world = -r_cw @ t_wc
            if not np.all(np.isfinite(center_world)):
                raise ValueError(f"{images_path}:{line_number}: non-finite camera center")

            frame_match = FRAME_RE.search(label)
            frame_index = int(frame_match.group(1)) if frame_match else None
            if source_name.lower().startswith("left/"):
                side = "left"
            elif source_name.lower().startswith("right/"):
                side = "right"
            else:
                side = "unknown"

            poses.append(
                ColmapPose(
                    model_id=model_id,
                    image_id=image_id,
                    camera_id=camera_id,
                    source_name=source_name,
                    label=label,
                    side=side,
                    frame_index=frame_index,
                    qvec_wxyz=tuple(float(value) for value in qvec),
                    t_wc=t_wc,
                    r_wc=r_wc,
                    center_world=center_world,
                    r_cw=r_cw,
                )
            )

    if not poses:
        raise ValueError(f"no COLMAP image poses found in {images_path}")
    return poses


def _sort_key(pose: ColmapPose) -> tuple[int, int, int]:
    # Keep left/right rows together when frame numbers are available.
    frame = pose.frame_index if pose.frame_index is not None else 2**31 - 1
    side_order = {"left": 0, "right": 1}.get(pose.side, 2)
    return frame, side_order, pose.image_id


def _metashape_ypr(rotation_cw: np.ndarray) -> tuple[float, float, float]:
    """Extract Metashape-compatible yaw, pitch, roll in degrees.

    This is the inverse of Metashape's camera-to-world ``ypr2mat`` convention,
    including its clockwise yaw sign and wrap to the usual [-360, 0] range.
    """

    r21 = float(np.clip(rotation_cw[2, 1], -1.0, 1.0))
    if r21 > 0.999:
        yaw = math.atan2(float(rotation_cw[1, 0]), float(rotation_cw[0, 0]))
        pitch = math.pi / 2.0
        roll = 0.0
    elif r21 < -0.999:
        yaw = math.atan2(float(rotation_cw[1, 0]), float(rotation_cw[0, 0]))
        pitch = -math.pi / 2.0
        roll = 0.0
    else:
        yaw = math.atan2(-float(rotation_cw[0, 1]), float(rotation_cw[1, 1]))
        roll = math.atan2(-float(rotation_cw[2, 0]), float(rotation_cw[2, 2]))
        pitch = math.asin(r21)

    if yaw > 0.0:
        yaw -= 2.0 * math.pi
    return -math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _rx(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotation_to_opk(rotation_cw: np.ndarray) -> tuple[tuple[float, float, float], float]:
    """Extract Metashape Omega, Phi, Kappa and return round-trip error."""

    u, _, vh = np.linalg.svd(rotation_cw)
    corrected = u @ vh
    if np.linalg.det(corrected) < 0.0:
        u[:, -1] *= -1.0
        corrected = u @ vh

    sine_phi = float(np.clip(corrected[0, 2], -1.0, 1.0))
    phi = math.asin(sine_phi)
    cosine_phi = math.cos(phi)
    if abs(cosine_phi) > 1e-10:
        omega = math.atan2(-corrected[1, 2], corrected[2, 2])
        kappa = math.atan2(-corrected[0, 1], corrected[0, 0])
    elif sine_phi > 0.0:
        omega = math.atan2(corrected[1, 0], corrected[1, 1])
        kappa = 0.0
    else:
        omega = math.atan2(-corrected[1, 0], corrected[1, 1])
        kappa = 0.0

    recovered = _rx(omega) @ _ry(phi) @ _rz(kappa)
    roundtrip_error = float(np.max(np.abs(recovered - corrected)))
    angles = np.degrees([omega, phi, kappa]).astype(np.float64)
    return (float(angles[0]), float(angles[1]), float(angles[2])), roundtrip_error


def _number(value: float) -> str:
    return f"{float(value):.12f}"


def _write_reference_csv(
    path: Path,
    poses: list[ColmapPose],
    angle_kind: str,
) -> tuple[int, float]:
    if angle_kind not in {"ypr", "opk"}:
        raise ValueError(f"unsupported angle kind: {angle_kind}")

    if angle_kind == "ypr":
        header = ["label", "x", "y", "z", "yaw", "pitch", "roll"]
    else:
        header = ["label", "x", "y", "z", "omega", "phi", "kappa"]

    max_roundtrip_error = 0.0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for pose in sorted(poses, key=_sort_key):
            if angle_kind == "ypr":
                angles = _metashape_ypr(pose.r_cw)
            else:
                angles, roundtrip_error = _rotation_to_opk(pose.r_cw)
                max_roundtrip_error = max(max_roundtrip_error, roundtrip_error)
            values = [*pose.center_world.tolist(), *angles]
            if not np.all(np.isfinite(values)):
                raise ValueError(f"non-finite Metashape {angle_kind} pose: {pose.label}")
            writer.writerow([pose.label, *(_number(value) for value in values)])

    return len(poses), max_roundtrip_error


def _write_pose_details(path: Path, poses: list[ColmapPose]) -> None:
    header = [
        "model_id",
        "image_id",
        "camera_id",
        "side",
        "frame_index",
        "label",
        "source_name",
        "x",
        "y",
        "z",
        "qw",
        "qx",
        "qy",
        "qz",
        "yaw",
        "pitch",
        "roll",
        "omega",
        "phi",
        "kappa",
        *(f"r_cw_{row}{column}" for row in range(3) for column in range(3)),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for pose in sorted(poses, key=_sort_key):
            ypr = _metashape_ypr(pose.r_cw)
            opk, _ = _rotation_to_opk(pose.r_cw)
            writer.writerow(
                [
                    pose.model_id,
                    pose.image_id,
                    pose.camera_id,
                    pose.side,
                    "" if pose.frame_index is None else pose.frame_index,
                    pose.label,
                    pose.source_name,
                    *(_number(value) for value in pose.center_world),
                    *(_number(value) for value in pose.qvec_wxyz),
                    *(_number(value) for value in ypr),
                    *(_number(value) for value in opk),
                    *(_number(value) for value in pose.r_cw.reshape(-1)),
                ]
            )


def _camera_center_stats(poses: list[ColmapPose]) -> dict[str, Any]:
    frames = [pose.frame_index for pose in poses if pose.frame_index is not None]
    side_counts: dict[str, int] = {}
    for pose in poses:
        side_counts[pose.side] = side_counts.get(pose.side, 0) + 1
    return {
        "images": len(poses),
        "side_counts": dict(sorted(side_counts.items())),
        "camera_ids": sorted({pose.camera_id for pose in poses}),
        "frame_min": min(frames) if frames else None,
        "frame_max": max(frames) if frames else None,
        "unique_frames": len(set(frames)),
        "coordinate_units": "COLMAP reconstruction units; absolute metric scale is not guaranteed",
    }


def _write_instructions(path: Path, output_root: Path, model_root: Path) -> None:
    source_database = model_root.parent / "database.db"
    text = f"""Metashape import instructions for COLMAP camera poses

Source reconstruction:
  {source_database.resolve()}

Important:
  COLMAP model directories 0, 1, ... are independent local coordinate frames.
  Import one model CSV into one Metashape chunk. Do not combine model 0 and
  models 1-4 as if their XYZ coordinates were in one common frame.
  COLMAP reconstruction scale is not guaranteed to be metric because the
  COLMAP mapper used independent left/right camera poses rather than a fixed
  stereo rig constraint. Use a known scale bar/baseline in Metashape if needed.

For a model:
  1. Add the corresponding registered images from the sample1000/left and
     sample1000/right folders. Labels in the CSV are the image file names,
     including .png.
  2. Open the Reference pane and choose Import.
  3. Select the *_ypr.csv file, comma delimiter, and map columns as
     label, x, y, z, yaw, pitch, roll. Select the [yaw,pitch,roll] angle order.
  4. Leave the project in a local/non-geographic coordinate system. These
     poses have no GPS datum.

Alternative orientation:
  The *_opk.csv file uses [omega,phi,kappa] instead. Select that angle order
  in the import dialog.

Files in this directory:
  metashape_reference_modelN_ypr.csv  direct Reference CSV import
  metashape_reference_modelN_opk.csv  direct Reference CSV import (OPK)
  colmap_pose_modelN_details.csv      full audit/details CSV
  metashape_export_manifest.json      model and file summary
"""
    path.write_text(text, encoding="utf-8")


def _parse_model_selection(value: str, available: list[int]) -> list[int]:
    if value.strip().lower() == "all":
        return available
    selected = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    unknown = [model_id for model_id in selected if model_id not in available]
    if unknown:
        raise ValueError(f"requested model(s) not found: {unknown}; available={available}")
    if not selected:
        raise ValueError("--models must be 'all' or a comma-separated model list")
    return selected


def convert(model_root: Path, output_root: Path, model_selection: str) -> dict[str, Any]:
    model_root = model_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not model_root.is_dir():
        raise FileNotFoundError(model_root)

    available = sorted(
        int(path.name)
        for path in model_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if not available:
        raise ValueError(f"no numeric COLMAP model directories under {model_root}")
    selected = _parse_model_selection(model_selection, available)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_models: list[dict[str, Any]] = []
    for model_id in selected:
        model_dir = model_root / str(model_id)
        poses = _parse_model(model_dir / "images.txt", model_id)
        ypr_path = output_root / f"metashape_reference_model{model_id}_ypr.csv"
        opk_path = output_root / f"metashape_reference_model{model_id}_opk.csv"
        details_path = output_root / f"colmap_pose_model{model_id}_details.csv"
        metadata_path = output_root / f"metashape_reference_model{model_id}_metadata.json"

        ypr_rows, _ = _write_reference_csv(ypr_path, poses, "ypr")
        opk_rows, opk_error = _write_reference_csv(opk_path, poses, "opk")
        _write_pose_details(details_path, poses)
        stats = _camera_center_stats(poses)
        metadata = {
            "source_model_id": model_id,
            "source_model_directory": str(model_dir),
            "source_images_txt": str((model_dir / "images.txt").resolve()),
            "pose_conversion": {
                "colmap_transform": "x_camera = R_wc @ x_world + t_wc",
                "camera_center": "C_world = -R_wc.T @ t_wc",
                "camera_to_world_rotation": "R_cw = R_wc.T",
                "camera_axes": "+X right, +Y down, +Z forward",
                "axis_flip": "none",
            },
            "metashape": {
                "ypr_columns": "label,x,y,z,yaw,pitch,roll",
                "opk_columns": "label,x,y,z,omega,phi,kappa",
                "angle_units": "degrees",
                "coordinate_system": "local Euclidean; no GPS/geographic datum",
                "scale_note": "COLMAP reconstruction units; not guaranteed metric",
            },
            "quality": {
                **stats,
                "opk_roundtrip_max_abs_matrix_error": opk_error,
            },
            "outputs": {
                "ypr_csv": str(ypr_path.resolve()),
                "opk_csv": str(opk_path.resolve()),
                "details_csv": str(details_path.resolve()),
            },
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_models.append(
            {
                "model_id": model_id,
                "images": ypr_rows,
                "ypr_csv": str(ypr_path.resolve()),
                "opk_csv": str(opk_path.resolve()),
                "details_csv": str(details_path.resolve()),
                "metadata_json": str(metadata_path.resolve()),
                **stats,
            }
        )

    manifest = {
        "source_model_root": str(model_root),
        "selected_models": selected,
        "coordinate_frame_policy": "one independent Metashape CSV per COLMAP model",
        "models": manifest_models,
        "instructions": str((output_root / "metashape_import_instructions.txt").resolve()),
    }
    (output_root / "metashape_export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_instructions(
        output_root / "metashape_import_instructions.txt", output_root, model_root
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert COLMAP images.txt poses to Metashape Reference CSV files."
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--models",
        default="all",
        help="all or comma-separated numeric model IDs (default: all)",
    )
    args = parser.parse_args()
    manifest = convert(args.model_root, args.output_dir, args.models)
    print(f"Output directory: {Path(manifest['instructions']).parent}")
    print(f"Models converted: {len(manifest['models'])}")
    for item in manifest["models"]:
        print(
            f"Model {item['model_id']}: {item['images']} images, "
            f"frames {item['frame_min']}..{item['frame_max']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
