"""Independently compute stereo depth with OpenCV StereoSGBM.

This script deliberately does not use the ZED depth engine.  It reads raw
left/right images from the SVO2, loads the independent calibration files in
``Calibration/``, rectifies both images with OpenCV, computes disparity with
``StereoSGBM``, and converts disparity to depth with the rectified focal length
and baseline.

The complete recording produces one centre-depth CSV and two histograms.  Raw
float32 depth maps are kept in memory one frame at a time; use
``--save-depth-maps-every`` when selected raw maps are needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
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
        sdk_root_path / "dependencies" / "freeglut_2.8" / "x64",
        sdk_root_path / "dependencies" / "glew" / "bin",
        sdk_root_path / "dependencies" / "glew-1.12.0" / "x64",
        sdk_root_path / "dependencies" / "opencv" / "x64" / "vc16" / "bin",
        sdk_root_path / "dependencies" / "opencv_3.1.0" / "x64",
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

if hasattr(sys.stdout, "reconfigure"):
    # Windows PowerShell may expose a legacy cp1252 stdout.  Calibration
    # paths contain Chinese characters, so keep diagnostics printable even
    # when the console has not been switched to UTF-8.
    sys.stdout.reconfigure(errors="backslashreplace")

import cv2  # noqa: E402  (the DLL search path must be prepared first)
import numpy as np  # noqa: E402
import pyzed.sl as sl  # noqa: E402


DEFAULT_SVO = Path(__file__).resolve().with_name("20260802_150233.svo2")
DEFAULT_CALIBRATION_DIR = (
    Path(__file__).resolve().parent / "Calibration" / "标定结果"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "output"
    / "20260802_150233_sgbm_depth_histogram"
)

CSV_HEADER = [
    "frame_index",
    "svo_position",
    "timestamp_ns",
    "center_x",
    "center_y",
    "center_disparity_px",
    "center_depth_m",
    "center_window_median_depth_m",
    "center_window_valid_count",
]


@dataclass(frozen=True)
class StereoCalibration:
    left_k: np.ndarray
    right_k: np.ndarray
    left_d: np.ndarray
    right_d: np.ndarray
    rotation: np.ndarray
    translation_mm: np.ndarray


@dataclass(frozen=True)
class Rectification:
    r_left: np.ndarray
    r_right: np.ndarray
    p_left: np.ndarray
    p_right: np.ndarray
    q: np.ndarray
    left_map_x: np.ndarray
    left_map_y: np.ndarray
    right_map_x: np.ndarray
    right_map_y: np.ndarray
    focal_length_px: float
    baseline_m: float
    roi_left: tuple[int, int, int, int]
    roi_right: tuple[int, int, int, int]


def _status_name(status: Any) -> str:
    return str(status).rsplit(".", 1)[-1]


def _check_status(status: Any, operation: str) -> None:
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"{operation} failed: {_status_name(status)}")


def _parse_numbers(text: str) -> list[float]:
    pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    return [float(value) for value in re.findall(pattern, text)]


def _source_section(text: str, name: str, next_name: str | None = None) -> str:
    end_pattern = rf"^\s*(?:{re.escape(next_name)}\s*:|\Z)" if next_name else r"\Z"
    match = re.search(
        rf"(?ms)^\s*{re.escape(name)}\s*:\s*(.*?)(?={end_pattern})",
        text,
    )
    if match is None:
        raise RuntimeError(f"Missing {name!r} section in calibration")
    return match.group(1)


def _source_scalar(section: str, name: str) -> float:
    match = re.search(
        rf"(?m)^\s*{re.escape(name)}\s*:\s*([-+0-9.eE]+)", section
    )
    if match is None:
        raise RuntimeError(f"Missing {name!r} in calibration")
    return float(match.group(1))


def _source_list(section: str, name: str) -> list[float]:
    match = re.search(
        rf"(?m)^\s*{re.escape(name)}\s*:\s*\[([^\]]+)\]", section
    )
    if match is None:
        raise RuntimeError(f"Missing {name!r} list in calibration")
    return _parse_numbers(match.group(1))


def _load_calibration(intrinsics_path: Path, extrinsics_path: Path) -> StereoCalibration:
    intrinsics_text = intrinsics_path.read_text(encoding="utf-8")
    extrinsics_text = extrinsics_path.read_text(encoding="utf-8")

    camera_values: dict[str, dict[str, Any]] = {}
    for name in ("left", "right"):
        section = _source_section(intrinsics_text, name, "right" if name == "left" else None)
        distortion = _source_list(section, "distortion_coefficients")
        if len(distortion) < 5:
            raise RuntimeError(f"Expected five distortion coefficients for {name}")
        camera_values[name] = {
            "fx": _source_scalar(section, "fx"),
            "fy": _source_scalar(section, "fy"),
            "cx": _source_scalar(section, "cx"),
            "cy": _source_scalar(section, "cy"),
            "distortion": distortion[:5],
        }

    r_section = _source_section(extrinsics_text, "R", "T")
    r_rows = [
        _parse_numbers(row)
        for row in re.findall(r"\[([^\]]+)\]", r_section)
    ]
    if len(r_rows) != 3 or any(len(row) != 3 for row in r_rows):
        raise RuntimeError(f"Expected a 3x3 stereo rotation matrix, got {r_rows}")
    translation = np.asarray(_source_list(extrinsics_text, "T"), dtype=np.float64)
    if translation.size != 3:
        raise RuntimeError("Stereo translation must contain three values")

    def camera_matrix(values: dict[str, Any]) -> np.ndarray:
        return np.asarray(
            [
                [values["fx"], 0.0, values["cx"]],
                [0.0, values["fy"], values["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    return StereoCalibration(
        left_k=camera_matrix(camera_values["left"]),
        right_k=camera_matrix(camera_values["right"]),
        left_d=np.asarray(camera_values["left"]["distortion"], dtype=np.float64),
        right_d=np.asarray(camera_values["right"]["distortion"], dtype=np.float64),
        rotation=np.asarray(r_rows, dtype=np.float64),
        translation_mm=translation,
    )


def _make_rectification(
    calibration: StereoCalibration,
    input_size: tuple[int, int],
    output_size: tuple[int, int],
    invert_extrinsics: bool,
    rectify_alpha: float,
) -> Rectification:
    rotation = calibration.rotation
    translation = calibration.translation_mm.reshape(3, 1)
    if invert_extrinsics:
        rotation = rotation.T
        translation = -rotation @ translation

    r_left, r_right, p_left, p_right, q, roi_left, roi_right = cv2.stereoRectify(
        calibration.left_k,
        calibration.left_d,
        calibration.right_k,
        calibration.right_d,
        input_size,
        rotation,
        translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=rectify_alpha,
        newImageSize=output_size,
    )
    left_map_x, left_map_y = cv2.initUndistortRectifyMap(
        calibration.left_k,
        calibration.left_d,
        r_left,
        p_left,
        output_size,
        cv2.CV_32FC1,
    )
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(
        calibration.right_k,
        calibration.right_d,
        r_right,
        p_right,
        output_size,
        cv2.CV_32FC1,
    )

    focal_length_px = float(p_left[0, 0])
    baseline_m = abs(float(p_right[0, 3] / p_right[0, 0])) / 1000.0
    if not np.isfinite(focal_length_px) or focal_length_px <= 0:
        raise RuntimeError(f"Invalid rectified focal length: {focal_length_px}")
    if not np.isfinite(baseline_m) or baseline_m <= 0:
        raise RuntimeError(f"Invalid rectified baseline: {baseline_m}")

    return Rectification(
        r_left=r_left,
        r_right=r_right,
        p_left=p_left,
        p_right=p_right,
        q=q,
        left_map_x=left_map_x,
        left_map_y=left_map_y,
        right_map_x=right_map_x,
        right_map_y=right_map_y,
        focal_length_px=focal_length_px,
        baseline_m=baseline_m,
        roi_left=tuple(int(value) for value in roi_left),
        roi_right=tuple(int(value) for value in roi_right),
    )


def _as_gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise RuntimeError(f"Expected a grayscale-compatible image, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _make_matcher(
    num_disparities: int,
    block_size: int,
    uniqueness_ratio: int,
    speckle_window_size: int,
    speckle_range: int,
) -> cv2.StereoSGBM:
    channels = 1
    p1 = 8 * channels * block_size * block_size
    p2 = 32 * channels * block_size * block_size
    mode = getattr(cv2, "STEREO_SGBM_MODE_SGBM_3WAY", cv2.STEREO_SGBM_MODE_SGBM)
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=1,
        preFilterCap=63,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        mode=mode,
    )


def _depth_from_disparity(
    disparity: np.ndarray,
    focal_length_px: float,
    baseline_m: float,
    max_depth_m: float,
) -> np.ndarray:
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 1.0)
    depth[valid] = (focal_length_px * baseline_m / disparity[valid]).astype(np.float32)
    depth_valid = np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth_m)
    depth[~depth_valid] = np.nan
    return depth


def _center_values(
    disparity: np.ndarray,
    depth: np.ndarray,
    radius: int,
) -> tuple[float, float, float, int]:
    height, width = depth.shape
    center_x = width // 2
    center_y = height // 2
    center_disparity = float(disparity[center_y, center_x])
    center_depth = float(depth[center_y, center_x])
    y0 = max(0, center_y - radius)
    y1 = min(height, center_y + radius + 1)
    x0 = max(0, center_x - radius)
    x1 = min(width, center_x + radius + 1)
    window = depth[y0:y1, x0:x1]
    valid_window = window[np.isfinite(window) & (window > 0.0)]
    window_median = (
        float(np.median(valid_window)) if valid_window.size else float("nan")
    )
    return center_disparity, center_depth, window_median, int(valid_window.size)


def _valid_values(values: list[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values_array) & (values_array > 0.0)
    return values_array[valid]


def _stats(values: list[float]) -> dict[str, Any]:
    finite = _valid_values(values)
    result: dict[str, Any] = {
        "count": int(finite.size),
        "invalid_count": int(len(values) - finite.size),
        "valid_ratio": float(finite.size / len(values)) if values else 0.0,
    }
    if finite.size == 0:
        return result
    result.update(
        {
            "min_m": float(np.min(finite)),
            "p01_m": float(np.percentile(finite, 1)),
            "p05_m": float(np.percentile(finite, 5)),
            "median_m": float(np.median(finite)),
            "mean_m": float(np.mean(finite)),
            "std_m": float(np.std(finite)),
            "p95_m": float(np.percentile(finite, 95)),
            "p99_m": float(np.percentile(finite, 99)),
            "max_m": float(np.max(finite)),
        }
    )
    return result


def _write_depth_preview(path: Path, depth: np.ndarray, title: str) -> None:
    valid = np.isfinite(depth) & (depth > 0.0)
    canvas = np.zeros(depth.shape, dtype=np.uint8)
    values = depth[valid].astype(np.float64)
    if values.size:
        low = float(np.percentile(values, 2))
        high = float(np.percentile(values, 98))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.min(values))
            high = low + 1.0
        canvas[valid] = np.clip(
            (depth[valid] - low) * 255.0 / (high - low), 0, 255
        ).astype(np.uint8)
    color = cv2.applyColorMap(canvas, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    cv2.putText(
        color,
        title,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), np.ascontiguousarray(color)):
        raise RuntimeError(f"Could not write depth preview: {path}")


def _write_disparity_preview(path: Path, disparity: np.ndarray, title: str) -> None:
    valid = np.isfinite(disparity) & (disparity > 1.0)
    canvas = np.zeros(disparity.shape, dtype=np.uint8)
    values = disparity[valid].astype(np.float64)
    if values.size:
        low = float(np.percentile(values, 2))
        high = float(np.percentile(values, 98))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.min(values))
            high = low + 1.0
        canvas[valid] = np.clip(
            (disparity[valid] - low) * 255.0 / (high - low), 0, 255
        ).astype(np.uint8)
    color = cv2.applyColorMap(canvas, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    cv2.putText(
        color,
        title,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), np.ascontiguousarray(color)):
        raise RuntimeError(f"Could not write disparity preview: {path}")


def _write_histogram(path: Path, values: list[float], title: str) -> dict[str, Any]:
    finite = _valid_values(values)
    if finite.size == 0:
        raise RuntimeError(f"No valid values available for histogram: {title}")
    low = float(np.percentile(finite, 1))
    high = float(np.percentile(finite, 99))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = low + 1.0
    clipped = np.clip(finite, low, high)
    counts, _ = np.histogram(clipped, bins=100, range=(low, high))

    canvas = np.full((820, 1400, 3), 255, dtype=np.uint8)
    left, top, right, bottom = 105, 80, 1340, 700
    plot_width = right - left
    plot_height = bottom - top
    maximum = max(1, int(np.max(counts)))
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = int(round(bottom - fraction * plot_height))
        cv2.line(canvas, (left, y), (right, y), (225, 225, 225), 1)
        cv2.putText(
            canvas,
            str(int(round(fraction * maximum))),
            (left - 75, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )

    bar_width = plot_width / len(counts)
    for index, count in enumerate(counts):
        x0 = int(round(left + index * bar_width))
        x1 = max(x0 + 1, int(round(left + (index + 1) * bar_width)) - 1)
        y = int(round(bottom - (int(count) / maximum) * plot_height))
        cv2.rectangle(canvas, (x0, y), (x1, bottom), (190, 80, 35), -1)

    cv2.line(canvas, (left, top), (left, bottom), (30, 30, 30), 2)
    cv2.line(canvas, (left, bottom), (right, bottom), (30, 30, 30), 2)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = int(round(left + fraction * plot_width))
        value = low + fraction * (high - low)
        cv2.line(canvas, (x, bottom), (x, bottom + 8), (30, 30, 30), 2)
        cv2.putText(
            canvas,
            f"{value:.2f}",
            (x - 35, bottom + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        title,
        (left, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Depth (m)",
        (right - 125, bottom + 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"valid n={finite.size:,}; plotted range=p01..p99; median={np.median(finite):.3f} m",
        (left, 775),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Could not write histogram: {path}")
    return {
        "file": str(path),
        "bins": int(len(counts)),
        "plot_range_p01_p99_m": [low, high],
        "clipped_values": int(np.count_nonzero((finite < low) | (finite > high))),
    }


def _write_rectification_metadata(
    path: Path,
    calibration: StereoCalibration,
    rectification: Rectification,
    input_size: tuple[int, int],
    output_size: tuple[int, int],
    invert_extrinsics: bool,
    rectify_alpha: float,
) -> None:
    payload = {
        "input_image_size": {"width": input_size[0], "height": input_size[1]},
        "rectified_image_size": {"width": output_size[0], "height": output_size[1]},
        "rectify_alpha": rectify_alpha,
        "extrinsics_direction": "inverse_of_source_R_T"
        if invert_extrinsics
        else "source_R_T_used_directly_as_left_to_right",
        "left_K": calibration.left_k.tolist(),
        "right_K": calibration.right_k.tolist(),
        "left_D": calibration.left_d.tolist(),
        "right_D": calibration.right_d.tolist(),
        "R_left_to_right": calibration.rotation.tolist(),
        "T_mm": calibration.translation_mm.tolist(),
        "input_baseline_norm_m": float(np.linalg.norm(calibration.translation_mm) / 1000.0),
        "R1": rectification.r_left.tolist(),
        "R2": rectification.r_right.tolist(),
        "P1": rectification.p_left.tolist(),
        "P2": rectification.p_right.tolist(),
        "Q": rectification.q.tolist(),
        "rectified_focal_length_px": rectification.focal_length_px,
        "rectified_baseline_m": rectification.baseline_m,
        "valid_roi_left": list(rectification.roi_left),
        "valid_roi_right": list(rectification.roi_right),
        "depth_formula": "depth_m = rectified_focal_length_px * rectified_baseline_m / disparity_px",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independently compute calibrated stereo depth with "
            "OpenCV StereoSGBM."
        )
    )
    parser.add_argument("svo", nargs="?", type=Path, default=DEFAULT_SVO)
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR / "camera_intrinsics.yaml",
    )
    parser.add_argument(
        "--extrinsics",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR / "stereo_extrinsics.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Rectified output scale; camera intrinsics are rectified for this size.",
    )
    parser.add_argument(
        "--rectify-alpha",
        type=float,
        default=0.0,
        help=(
            "OpenCV stereoRectify alpha in [0, 1]. 0 maximizes valid pixels by "
            "zooming/cropping; 1 preserves the widest source view."
        ),
    )
    parser.add_argument(
        "--num-disparities",
        type=int,
        default=256,
        help="SGBM disparity search range; must be a positive multiple of 16.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=5,
        help="Odd SGBM matching block size (default: 5).",
    )
    parser.add_argument(
        "--uniqueness-ratio",
        type=int,
        default=8,
        help="SGBM uniqueness ratio (default: 8).",
    )
    parser.add_argument(
        "--speckle-window-size",
        type=int,
        default=100,
        help="SGBM speckle window size (default: 100).",
    )
    parser.add_argument(
        "--speckle-range",
        type=int,
        default=2,
        help="SGBM speckle range (default: 2).",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=100.0,
        help="Discard SGBM depths above this value in metres (default: 100).",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--window-radius", type=int, default=2)
    parser.add_argument(
        "--save-depth-maps-every",
        type=int,
        default=0,
        help="Save raw float32 maps at this interval; 0 saves none.",
    )
    parser.add_argument(
        "--invert-extrinsics",
        action="store_true",
        help="Debug option: invert the source R/T before stereoRectify.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.scale <= 1.0:
        parser.error("--scale must be in the interval (0, 1]")
    if not 0.0 <= args.rectify_alpha <= 1.0:
        parser.error("--rectify-alpha must be in the interval [0, 1]")
    if args.num_disparities <= 0 or args.num_disparities % 16:
        parser.error("--num-disparities must be a positive multiple of 16")
    if args.block_size < 3 or args.block_size % 2 == 0:
        parser.error("--block-size must be an odd integer >= 3")
    if args.max_depth <= 0:
        parser.error("--max-depth must be > 0")
    if args.max_frames < 0 or args.window_radius < 0 or args.save_depth_maps_every < 0:
        parser.error("frame/window/map intervals must be >= 0")
    return args


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    known = [
        "center_depth.csv",
        "center_depth_histogram.png",
        "center_window_depth_histogram.png",
        "rectification.json",
        "summary.json",
    ]
    existing = [path / name for name in known if (path / name).exists()]
    if existing and not overwrite:
        raise RuntimeError(
            "Output files already exist; use --overwrite or choose another directory: "
            + ", ".join(str(item) for item in existing)
        )
    if overwrite:
        for item in existing:
            item.unlink()


def main() -> int:
    args = _parse_args()
    svo_path = args.svo.expanduser().resolve()
    intrinsics_path = args.intrinsics.expanduser().resolve()
    extrinsics_path = args.extrinsics.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path, label in (
        (svo_path, "SVO2"),
        (intrinsics_path, "intrinsics calibration"),
        (extrinsics_path, "extrinsics calibration"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
    _prepare_output_dir(output_dir, args.overwrite)

    calibration = _load_calibration(intrinsics_path, extrinsics_path)
    zed = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    _check_status(zed.open(init), "open SVO2 for raw stereo images")

    center_depths: list[float] = []
    window_depths: list[float] = []
    center_disparities: list[float] = []
    window_valid_counts: list[int] = []
    preview_frames: list[int] = []
    started = time.monotonic()
    csv_path = output_dir / "center_depth.csv"
    depth_maps_dir = output_dir / "depth_maps"
    if args.save_depth_maps_every > 0:
        depth_maps_dir.mkdir(parents=True, exist_ok=True)

    info = None
    replayed_frames = 0
    total_frames = 0
    width = height = 0
    output_width = output_height = 0
    rectification: Rectification | None = None
    try:
        info = zed.get_camera_information()
        total_frames = int(zed.get_svo_number_of_frames())
        width = int(info.camera_configuration.resolution.width)
        height = int(info.camera_configuration.resolution.height)
        output_width = max(2, int(round(width * args.scale)))
        output_height = max(2, int(round(height * args.scale)))
        input_size = (width, height)
        output_size = (output_width, output_height)
        rectification = _make_rectification(
            calibration,
            input_size=input_size,
            output_size=output_size,
            invert_extrinsics=args.invert_extrinsics,
            rectify_alpha=args.rectify_alpha,
        )
        _write_rectification_metadata(
            output_dir / "rectification.json",
            calibration,
            rectification,
            input_size,
            output_size,
            args.invert_extrinsics,
            args.rectify_alpha,
        )

        replay_limit = (
            min(args.max_frames, total_frames)
            if args.max_frames > 0
            else total_frames
        )
        preview_targets = {
            target
            for target in (0, (replay_limit - 1) // 2, replay_limit - 1)
            if target >= 0
        }
        center_x = output_width // 2
        center_y = output_height // 2
        matcher = _make_matcher(
            args.num_disparities,
            args.block_size,
            args.uniqueness_ratio,
            args.speckle_window_size,
            args.speckle_range,
        )
        left_mat = sl.Mat()
        right_mat = sl.Mat()
        runtime = sl.RuntimeParameters()
        print(f"SDK version: {sl.Camera.get_sdk_version()}")
        print(f"Camera: {_status_name(info.camera_model)}")
        print(f"Input: {svo_path}")
        print(f"Intrinsics: {intrinsics_path}")
        print(f"Extrinsics: {extrinsics_path}")
        print(f"Raw image size: {width}x{height}")
        print(f"Rectified size: {output_width}x{output_height}")
        print(f"SGBM focal length: {rectification.focal_length_px:.9f} px")
        print(f"SGBM baseline: {rectification.baseline_m:.9f} m")
        print(
            "SGBM: "
            f"num_disparities={args.num_disparities}, block_size={args.block_size}, "
            f"uniqueness={args.uniqueness_ratio}"
        )
        print(f"SVO frames: {total_frames}; replay limit: {replay_limit}")
        print(f"Output: {output_dir}")

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(CSV_HEADER)
            while replayed_frames < replay_limit:
                status = zed.grab(runtime)
                if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break
                _check_status(status, "grab")
                _check_status(
                    zed.retrieve_image(left_mat, sl.VIEW.LEFT_UNRECTIFIED),
                    f"retrieve raw left image at frame {replayed_frames}",
                )
                _check_status(
                    zed.retrieve_image(right_mat, sl.VIEW.RIGHT_UNRECTIFIED),
                    f"retrieve raw right image at frame {replayed_frames}",
                )
                left_raw = _as_gray(left_mat.get_data())
                right_raw = _as_gray(right_mat.get_data())
                left_rectified = cv2.remap(
                    left_raw,
                    rectification.left_map_x,
                    rectification.left_map_y,
                    cv2.INTER_LINEAR,
                )
                right_rectified = cv2.remap(
                    right_raw,
                    rectification.right_map_x,
                    rectification.right_map_y,
                    cv2.INTER_LINEAR,
                )
                disparity = matcher.compute(left_rectified, right_rectified).astype(
                    np.float32
                ) / 16.0
                depth = _depth_from_disparity(
                    disparity,
                    rectification.focal_length_px,
                    rectification.baseline_m,
                    args.max_depth,
                )
                if depth.shape != (output_height, output_width):
                    raise RuntimeError(
                        f"Unexpected SGBM depth shape {depth.shape}; "
                        f"expected {(output_height, output_width)}"
                    )

                center_disparity, center_depth, window_median, window_count = _center_values(
                    disparity, depth, args.window_radius
                )
                center_disparities.append(center_disparity)
                center_depths.append(center_depth)
                window_depths.append(window_median)
                window_valid_counts.append(window_count)

                writer.writerow(
                    [
                        replayed_frames,
                        int(zed.get_svo_position()),
                        int(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()),
                        center_x,
                        center_y,
                        f"{center_disparity:.12g}"
                        if np.isfinite(center_disparity)
                        else "NaN",
                        f"{center_depth:.12g}"
                        if np.isfinite(center_depth)
                        else "NaN",
                        f"{window_median:.12g}"
                        if np.isfinite(window_median)
                        else "NaN",
                        window_count,
                    ]
                )

                if (
                    args.save_depth_maps_every > 0
                    and replayed_frames % args.save_depth_maps_every == 0
                ):
                    np.save(
                        depth_maps_dir / f"depth_{replayed_frames:06d}.npy",
                        depth,
                    )

                if replayed_frames in preview_targets:
                    depth_preview_path = output_dir / (
                        f"depth_preview_frame_{replayed_frames:06d}.png"
                    )
                    disparity_preview_path = output_dir / (
                        f"disparity_preview_frame_{replayed_frames:06d}.png"
                    )
                    _write_depth_preview(
                        depth_preview_path,
                        depth,
                        f"SGBM depth; frame {replayed_frames}; centre {center_depth:.3f} m",
                    )
                    _write_disparity_preview(
                        disparity_preview_path,
                        disparity,
                        f"SGBM disparity; frame {replayed_frames}; centre {center_disparity:.2f} px",
                    )
                    preview_frames.append(replayed_frames)

                replayed_frames += 1
                if replayed_frames % 100 == 0 or replayed_frames == 1:
                    elapsed = time.monotonic() - started
                    rate = replayed_frames / elapsed if elapsed > 0 else 0.0
                    print(
                        f"Replay: {replayed_frames}/{replay_limit} frames, "
                        f"{rate:.2f} frames/s, centre={center_depth:.3f} m, "
                        f"disparity={center_disparity:.2f} px",
                        flush=True,
                    )
            csv_file.flush()
            os.fsync(csv_file.fileno())
    finally:
        zed.close()

    if rectification is None or info is None:
        raise RuntimeError("Rectification was not initialized")
    if replayed_frames == 0:
        raise RuntimeError("No stereo frames were processed")
    if replayed_frames != replay_limit:
        raise RuntimeError(
            f"SVO ended before the requested replay limit: "
            f"{replayed_frames}/{replay_limit} frames"
        )

    histogram_path = output_dir / "center_depth_histogram.png"
    window_histogram_path = output_dir / "center_window_depth_histogram.png"
    histogram_metadata = _write_histogram(
        histogram_path,
        center_depths,
        "SGBM centre depth histogram (exact centre pixel)",
    )
    window_histogram_metadata = _write_histogram(
        window_histogram_path,
        window_depths,
        "SGBM centre depth histogram (5x5 median)",
    )
    elapsed_seconds = time.monotonic() - started
    summary = {
        "method": "OpenCV StereoSGBM; independent of ZED MEASURE.DEPTH",
        "input_file": str(svo_path),
        "intrinsics_file": str(intrinsics_path),
        "extrinsics_file": str(extrinsics_path),
        "sdk_version": str(sl.Camera.get_sdk_version()),
        "camera_model": _status_name(info.camera_model),
        "input_resolution": {"width": width, "height": height},
        "rectified_resolution": {
            "width": output_width,
            "height": output_height,
        },
        "rectified_focal_length_px": rectification.focal_length_px,
        "rectified_baseline_m": rectification.baseline_m,
        "rectify_alpha": args.rectify_alpha,
        "extrinsics_inverted": args.invert_extrinsics,
        "depth_formula": "depth_m = f_rectified_px * baseline_m / disparity_px",
        "sgbm": {
            "num_disparities": args.num_disparities,
            "block_size": args.block_size,
            "uniqueness_ratio": args.uniqueness_ratio,
            "speckle_window_size": args.speckle_window_size,
            "speckle_range": args.speckle_range,
            "max_depth_m": args.max_depth,
        },
        "total_svo_frames": total_frames,
        "frames_replayed": replayed_frames,
        "center_pixel": {
            "x": output_width // 2,
            "y": output_height // 2,
        },
        "center_window_radius": args.window_radius,
        "center_disparity_stats_px": {
            "count": int(_valid_values(center_disparities).size),
            "median_px": float(np.median(_valid_values(center_disparities)))
            if _valid_values(center_disparities).size
            else None,
        },
        "center_pixel_stats_m": _stats(center_depths),
        "center_window_median_stats_m": _stats(window_depths),
        "center_window_valid_count_stats": {
            "min": int(np.min(window_valid_counts)),
            "median": float(np.median(window_valid_counts)),
            "max": int(np.max(window_valid_counts)),
        },
        "files": {
            "center_depth_csv": str(csv_path),
            "center_depth_histogram": histogram_metadata,
            "center_window_depth_histogram": window_histogram_metadata,
            "rectification": str(output_dir / "rectification.json"),
            "depth_preview_frames": preview_frames,
            "raw_depth_maps_directory": (
                str(depth_maps_dir) if args.save_depth_maps_every > 0 else None
            ),
            "raw_depth_map_interval": args.save_depth_maps_every,
        },
        "elapsed_seconds": elapsed_seconds,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Finished: {replayed_frames} frames in {elapsed_seconds:.1f}s")
    print(f"Center depth CSV: {csv_path}")
    print(f"Histogram: {histogram_path}")
    print(f"Window histogram: {window_histogram_path}")
    print(f"Rectification: {output_dir / 'rectification.json'}")
    print(f"Summary: {summary_path}")
    print(
        "Exact centre depth: "
        f"valid={summary['center_pixel_stats_m']['count']}/{replayed_frames}, "
        f"median={summary['center_pixel_stats_m'].get('median_m')} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
