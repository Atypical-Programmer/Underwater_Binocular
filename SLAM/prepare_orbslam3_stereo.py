"""Generate an ORB-SLAM3 stereo settings file from Calibration/ only.

The SVO2 file is deliberately not opened here.  The independent stereo file
stores the source transform as OpenCV left-camera -> right-camera.  ORB-SLAM3
expects T_c1_c2 in its camera convention, so this script writes the inverse;
ORB-SLAM3 inverts it once more when constructing OpenCV's rectification maps.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np


FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def read_block(text: str, name: str, next_name: str | None = None) -> str:
    start = re.search(rf"(?ms)^\s*{re.escape(name)}:\s*", text)
    if not start:
        raise ValueError(f"missing section: {name}")
    block = text[start.end() :]
    if next_name:
        end = re.search(rf"(?ms)^\s*{re.escape(next_name)}:\s*", block)
        if end:
            block = block[: end.start()]
    return block


def read_scalar(block: str, name: str) -> float:
    match = re.search(rf"(?m)^\s*{re.escape(name)}:\s*({FLOAT})", block)
    if not match:
        raise ValueError(f"missing value: {name}")
    return float(match.group(1))


def read_vector(block: str, name: str) -> list[float]:
    match = re.search(rf"(?ms)^\s*{re.escape(name)}:\s*\[([^\]]+)\]", block)
    if not match:
        raise ValueError(f"missing vector: {name}")
    return [float(value) for value in re.findall(FLOAT, match.group(1))]


def read_calibration(root: Path) -> tuple[dict[str, float | list[float]], np.ndarray, np.ndarray]:
    intrinsics_path = root / "Calibration" / "标定结果" / "camera_intrinsics.yaml"
    extrinsics_path = root / "Calibration" / "标定结果" / "stereo_extrinsics.yaml"
    intrinsics_text = intrinsics_path.read_text(encoding="utf-8")
    extrinsics_text = extrinsics_path.read_text(encoding="utf-8")

    left = read_block(intrinsics_text, "left", "right")
    right = read_block(intrinsics_text, "right")
    values: dict[str, float | list[float]] = {}
    for prefix, block in (("left", left), ("right", right)):
        for key in ("fx", "fy", "cx", "cy"):
            values[f"{prefix}_{key}"] = read_scalar(block, key)
        distortion = read_vector(block, "distortion_coefficients")
        if len(distortion) < 5:
            raise ValueError(f"{prefix} distortion must contain k1,k2,p1,p2,k3")
        values[f"{prefix}_distortion"] = distortion[:5]

    r_section = read_block(extrinsics_text, "R", "T")
    rows = re.findall(r"\[([^\]]+)\]", r_section)
    if len(rows) < 3:
        raise ValueError("stereo R must contain three rows")
    rotation = np.array(
        [[float(value) for value in re.findall(FLOAT, row)] for row in rows[:3]],
        dtype=np.float64,
    )
    translation = np.array(read_vector(extrinsics_text, "T"), dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("invalid stereo extrinsics shape")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=2e-5
    ):
        raise ValueError("stereo R is not a valid rotation")
    return values, rotation, translation


def format_number(value: float) -> str:
    return f"{float(value):.12g}"


def write_settings(
    output_path: Path,
    values: dict[str, float | list[float]],
    rotation_source: np.ndarray,
    translation_source_mm: np.ndarray,
    image_scale: float,
) -> None:
    if not math.isfinite(image_scale) or image_scale <= 0.0 or image_scale > 1.0:
        raise ValueError("image scale must be in the interval (0, 1]")

    width = int(round(1920 * image_scale))
    height = int(round(1080 * image_scale))

    # The Calibration extrinsics are used as OpenCV left -> right.  ORB's
    # Settings loader expects T_c1_c2 whose inverse is that OpenCV transform.
    rotation_orb = rotation_source.T
    translation_orb_m = -rotation_source.T @ (translation_source_mm / 1000.0)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_orb
    transform[:3, 3] = translation_orb_m

    ld = values["left_distortion"]
    rd = values["right_distortion"]
    assert isinstance(ld, list) and isinstance(rd, list)
    lines = [
        "%YAML:1.0",
        "",
        "# Generated only from Calibration/标定结果/camera_intrinsics.yaml and",
        "# Calibration/标定结果/stereo_extrinsics.yaml. No SVO calibration is read.",
        "# Input images are resized uniformly before ORB-SLAM3.",
        "File.version: \"1.0\"",
        "Camera.type: \"PinHole\"",
        "",
        f"Camera1.fx: {format_number(float(values['left_fx']) * image_scale)}",
        f"Camera1.fy: {format_number(float(values['left_fy']) * image_scale)}",
        f"Camera1.cx: {format_number(float(values['left_cx']) * image_scale)}",
        f"Camera1.cy: {format_number(float(values['left_cy']) * image_scale)}",
        f"Camera1.k1: {format_number(ld[0])}",
        f"Camera1.k2: {format_number(ld[1])}",
        f"Camera1.p1: {format_number(ld[2])}",
        f"Camera1.p2: {format_number(ld[3])}",
        f"Camera1.k3: {format_number(ld[4])}",
        "",
        f"Camera2.fx: {format_number(float(values['right_fx']) * image_scale)}",
        f"Camera2.fy: {format_number(float(values['right_fy']) * image_scale)}",
        f"Camera2.cx: {format_number(float(values['right_cx']) * image_scale)}",
        f"Camera2.cy: {format_number(float(values['right_cy']) * image_scale)}",
        f"Camera2.k1: {format_number(rd[0])}",
        f"Camera2.k2: {format_number(rd[1])}",
        f"Camera2.p1: {format_number(rd[2])}",
        f"Camera2.p2: {format_number(rd[3])}",
        f"Camera2.k3: {format_number(rd[4])}",
        "",
        f"Camera.width: {width}",
        f"Camera.height: {height}",
        "Camera.fps: 30",
        "Camera.RGB: 0",
        "",
        "Stereo.ThDepth: 60.0",
        "Stereo.T_c1_c2: !!opencv-matrix",
        "  rows: 4",
        "  cols: 4",
        # ORB-SLAM3's Converter::toSophus reads this matrix as CV_32F.
        "  dt: f",
        "  data: ["
        + ",".join(format_number(value) for value in transform.reshape(-1))
        + "]",
    ]
    lines.extend(
        [
            "",
            "ORBextractor.nFeatures: 1500",
            "ORBextractor.scaleFactor: 1.2",
            "ORBextractor.nLevels: 8",
            "ORBextractor.iniThFAST: 20",
            "ORBextractor.minThFAST: 7",
            "",
            "Viewer.KeyFrameSize: 0.05",
            "Viewer.KeyFrameLineWidth: 1.0",
            "Viewer.GraphLineWidth: 0.9",
            "Viewer.PointSize: 2.0",
            "Viewer.CameraSize: 0.08",
            "Viewer.CameraLineWidth: 3.0",
            "Viewer.ViewpointX: 0.0",
            "Viewer.ViewpointY: -0.7",
            "Viewer.ViewpointZ: -1.8",
            "Viewer.ViewpointF: 500.0",
            "Viewer.imageViewScale: 1.0",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "20260802_150233_stereo.yaml",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="uniform image scale applied before ORB-SLAM3 (default: 1.0)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else (root / args.output)
    values, rotation, translation = read_calibration(root)
    write_settings(output, values, rotation, translation, args.scale)
    print(f"Wrote {output.resolve()}")
    print(f"Source baseline norm: {np.linalg.norm(translation):.6f} mm")
    print(f"ORB T_c1_c2 translation norm: {np.linalg.norm(translation) / 1000.0:.6f} m")
    print(f"Processed image size: {round(1920 * args.scale)}x{round(1080 * args.scale)}")


if __name__ == "__main__":
    main()
