"""Export a COLMAP sparse model for import into Agisoft Metashape.

The primary output is a COLMAP text package.  Metashape 2.3 and newer can
import this format directly and preserve the COLMAP FULL_OPENCV camera model.
For the older Metashape 1.7 installation on this workstation, this script
also invokes the embedded Metashape Python runner to create an Agisoft XML
camera file.  The XML is a compatibility export: Metashape 1.7 stores its
own calibration model, so COLMAP's k5/k6 are retained in the manifest but
cannot be represented by that legacy XML calibration schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_MODEL = (
    WORKSPACE
    / "output"
    / "20260802_150233_sample1000"
    / "sfm_aliked_adalam_100_full_opencv_refined"
    / "sparse"
    / "0_text"
)
DEFAULT_IMAGES = WORKSPACE / "output" / "20260802_150233_sample1000" / "left"
DEFAULT_OUTPUT = (
    WORKSPACE
    / "output"
    / "20260802_150233_sample1000"
    / "sfm_aliked_adalam_100_full_opencv_refined"
    / "metashape_export"
)
DEFAULT_METASHAPE = Path(r"C:\Program Files\Agisoft\Metashape Pro\metashape.exe")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export COLMAP FULL_OPENCV poses for Agisoft Metashape."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metashape", type=Path, default=DEFAULT_METASHAPE)
    parser.add_argument(
        "--skip-metashape-xml",
        action="store_true",
        help="Only write the exact COLMAP package and CSV diagnostics.",
    )
    return parser.parse_args()


def _quaternion_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"invalid COLMAP quaternion norm: {norm}")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _read_camera(path: Path) -> dict[str, Any]:
    cameras: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 5:
                raise ValueError(f"{path}:{line_number}: invalid camera row")
            camera_id = int(fields[0])
            model = fields[1]
            width = int(fields[2])
            height = int(fields[3])
            params = [float(value) for value in fields[4:]]
            cameras[camera_id] = {
                "id": camera_id,
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    if not cameras:
        raise ValueError(f"no camera rows found in {path}")
    if any(camera["model"] != "FULL_OPENCV" for camera in cameras.values()):
        models = sorted({camera["model"] for camera in cameras.values()})
        raise ValueError(f"expected FULL_OPENCV camera model, found {models}")
    return cameras


def _read_poses(path: Path) -> list[dict[str, Any]]:
    poses: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            # COLMAP images.txt uses a second, variable-length POINTS2D row.
            # A pose row has exactly 10 whitespace-separated fields.
            if len(fields) != 10:
                continue
            try:
                image_id = int(fields[0])
                qw, qx, qy, qz = (float(value) for value in fields[1:5])
                tx, ty, tz = (float(value) for value in fields[5:8])
                camera_id = int(fields[8])
            except ValueError:
                continue
            name = fields[9]
            rotation_wc = _quaternion_to_rotation(qw, qx, qy, qz)
            rotation_cw = rotation_wc.T
            center = -rotation_cw @ np.asarray([tx, ty, tz], dtype=np.float64)
            if not np.isfinite(center).all():
                raise ValueError(f"{path}:{line_number}: non-finite camera center")
            poses.append(
                {
                    "image_id": image_id,
                    "camera_id": camera_id,
                    "name": name,
                    "qw": qw,
                    "qx": qx,
                    "qy": qy,
                    "qz": qz,
                    "tx": tx,
                    "ty": ty,
                    "tz": tz,
                    "rotation_wc": rotation_wc,
                    "rotation_cw": rotation_cw,
                    "center": center,
                }
            )
    if not poses:
        raise ValueError(f"no COLMAP pose rows found in {path}")
    poses.sort(key=lambda row: row["image_id"])
    names = [row["name"] for row in poses]
    if len(names) != len(set(names)):
        raise ValueError("duplicate image names in COLMAP pose file")
    return poses


def _rotation_to_opk(rotation: np.ndarray) -> tuple[float, float, float]:
    """Extract Metashape OPK angles from a camera-to-world matrix."""

    sine_phi = float(np.clip(rotation[0, 2], -1.0, 1.0))
    phi = math.asin(sine_phi)
    cosine_phi = math.cos(phi)
    if abs(cosine_phi) > 1e-10:
        omega = math.atan2(-float(rotation[1, 2]), float(rotation[2, 2]))
        kappa = math.atan2(-float(rotation[0, 1]), float(rotation[0, 0]))
    elif sine_phi > 0.0:
        omega = math.atan2(float(rotation[1, 0]), float(rotation[1, 1]))
        kappa = 0.0
    else:
        omega = math.atan2(-float(rotation[1, 0]), float(rotation[1, 1]))
        kappa = 0.0
    return tuple(float(value) for value in np.degrees([omega, phi, kappa]))


def _number(value: float) -> str:
    return f"{float(value):.12f}"


def _copy_colmap_package(model_dir: Path, output_dir: Path) -> dict[str, str]:
    package_dir = output_dir / "colmap_full_opencv"
    package_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for filename in ("cameras.txt", "images.txt", "points3D.txt"):
        source = model_dir / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = package_dir / filename
        shutil.copy2(source, destination)
        copied[filename] = str(destination.resolve())
    return copied


def _write_pose_exports(
    poses: list[dict[str, Any]],
    cameras: dict[int, dict[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    reference_path = output_dir / "metashape_camera_reference_opk.csv"
    matrix_path = output_dir / "camera_poses_camera_to_world.csv"

    with reference_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["label", "x", "y", "z", "omega", "phi", "kappa"])
        for pose in poses:
            omega, phi, kappa = _rotation_to_opk(pose["rotation_cw"])
            writer.writerow(
                [
                    pose["name"],
                    *(_number(value) for value in pose["center"]),
                    _number(omega),
                    _number(phi),
                    _number(kappa),
                ]
            )

    matrix_header = [
        "image_id",
        "camera_id",
        "label",
        "cx",
        "cy",
        "cz",
        *(f"r{row}{column}" for row in range(3) for column in range(3)),
    ]
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(matrix_header)
        for pose in poses:
            writer.writerow(
                [
                    pose["image_id"],
                    pose["camera_id"],
                    pose["name"],
                    *(_number(value) for value in pose["center"]),
                    *(_number(value) for value in pose["rotation_cw"].reshape(-1)),
                ]
            )

    return {
        "reference_opk_csv": str(reference_path.resolve()),
        "camera_to_world_matrix_csv": str(matrix_path.resolve()),
    }


def _run_metashape_xml_export(
    model_dir: Path,
    image_dir: Path,
    xml_path: Path,
    metashape_exe: Path,
) -> dict[str, Any]:
    if not metashape_exe.is_file():
        return {"status": "skipped", "reason": f"Metashape executable not found: {metashape_exe}"}
    runner = WORKSPACE / "export_colmap_poses_to_metashape_xml.py"
    if not runner.is_file():
        raise FileNotFoundError(runner)
    command = [
        str(metashape_exe),
        "-r",
        str(runner),
        "--model",
        str(model_dir.resolve()),
        "--images",
        str(image_dir.resolve()),
        "--output",
        str(xml_path.resolve()),
    ]
    completed = subprocess.run(
        command,
        cwd=str(WORKSPACE),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path = xml_path.with_suffix(".metashape.log")
    log_path.write_text(
        "COMMAND:\n"
        + subprocess.list2cmdline(command)
        + "\n\nSTDOUT:\n"
        + completed.stdout
        + "\nSTDERR:\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0 or not xml_path.is_file():
        raise RuntimeError(
            "Metashape XML export failed; see "
            f"{log_path} (exit code {completed.returncode})"
        )
    return {
        "status": "created",
        "xml": str(xml_path.resolve()),
        "log": str(log_path.resolve()),
        "metashape_executable": str(metashape_exe.resolve()),
    }


def export(
    model_dir: Path,
    image_dir: Path,
    output_dir: Path,
    metashape_exe: Path,
    skip_metashape_xml: bool = False,
) -> dict[str, Any]:
    model_dir = model_dir.expanduser().resolve()
    image_dir = image_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    cameras = _read_camera(model_dir / "cameras.txt")
    poses = _read_poses(model_dir / "images.txt")
    for pose in poses:
        if pose["camera_id"] not in cameras:
            raise ValueError(
                f"pose image {pose['name']} references missing camera "
                f"{pose['camera_id']}"
            )
        if not (image_dir / pose["name"]).is_file():
            raise FileNotFoundError(image_dir / pose["name"])

    output_dir.mkdir(parents=True, exist_ok=True)
    colmap_paths = _copy_colmap_package(model_dir, output_dir)
    pose_paths = _write_pose_exports(poses, cameras, output_dir)

    xml_path = output_dir / "metashape_cameras.xml"
    if skip_metashape_xml:
        xml_result: dict[str, Any] = {"status": "skipped", "reason": "--skip-metashape-xml"}
    else:
        xml_result = _run_metashape_xml_export(
            model_dir=model_dir,
            image_dir=image_dir,
            xml_path=xml_path,
            metashape_exe=metashape_exe.expanduser().resolve(),
        )

    camera_models = {
        str(camera_id): {
            "model": camera["model"],
            "width": camera["width"],
            "height": camera["height"],
            "parameter_order": "fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6",
            "parameters": camera["params"],
        }
        for camera_id, camera in sorted(cameras.items())
    }
    manifest: dict[str, Any] = {
        "source": {
            "model_dir": str(model_dir),
            "images_dir": str(image_dir),
            "algorithm": "ALIKED + AdaLAM + COLMAP",
            "model_selection": "200-photo FULL_OPENCV stereo model from 100 synchronized frames",
        },
        "colmap": {
            "camera_model": "FULL_OPENCV",
            "camera_count": len(cameras),
            "camera_ids": sorted(cameras),
            "parameter_order": "fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6",
            "models_by_camera_id": camera_models,
            "registered_pose_count": len(poses),
            "pose_definition": "COLMAP world-to-camera quaternion/translation inverted to camera-to-world",
            "camera_axes": "+X right, +Y down, +Z forward",
            "world_units": "COLMAP reconstruction units; no metric/geographic scale was supplied",
        },
        "outputs": {
            **colmap_paths,
            **pose_paths,
            "metashape_xml": xml_result,
        },
        "metashape_import": {
            "exact_full_opencv": (
                "Use the colmap_full_opencv/cameras.txt package with Metashape 2.3+; "
                "the FULL_OPENCV 12-parameter camera is preserved."
            ),
            "legacy_xml": (
                "metashape_cameras.xml is generated by the installed Metashape 1.7.4. "
                "Its transforms are camera-to-world; its legacy calibration stores "
                "f,cx,cy,b1,b2,k1-k4,p1,p2, while source k5/k6 remain in this manifest."
            ),
            "reference_csv": (
                "metashape_camera_reference_opk.csv is a source-reference CSV; "
                "it contains pose centers and OPK degrees but not calibration."
            ),
        },
    }
    manifest_path = output_dir / "metashape_export_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest["outputs"]["manifest"] = str(manifest_path.resolve())
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    args = _parse_args()
    manifest = export(
        model_dir=args.model,
        image_dir=args.images,
        output_dir=args.output_dir,
        metashape_exe=args.metashape,
        skip_metashape_xml=args.skip_metashape_xml,
    )
    print(f"Registered poses exported: {manifest['colmap']['registered_pose_count']}")
    print(f"Camera model: {manifest['colmap']['camera_model']}")
    print(f"Output directory: {Path(manifest['outputs']['manifest']).parent}")
    print(f"Metashape XML status: {manifest['outputs']['metashape_xml']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
