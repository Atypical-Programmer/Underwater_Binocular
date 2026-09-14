"""Sample a few stereo depths using the calibration embedded in an SVO2.

The SVO is opened without ``optional_opencv_calibration_file``.  Native
rectified LEFT/RIGHT images are matched with OpenCV StereoSGBM, and depth is
computed from the native rectified focal length and stereo baseline:

    depth_m = focal_length_px * baseline_m / disparity_px

This is a small diagnostic tool rather than a full-recording depth export.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


def _prepare_windows_dll_search_path() -> None:
    """Make the ZED SDK DLLs discoverable before importing pyzed.sl."""

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
    os.environ["PATH"] = os.pathsep.join(
        [*existing_paths, *(p for p in os.environ.get("PATH", "").split(os.pathsep) if p)]
    )
    if hasattr(os, "add_dll_directory"):
        _prepare_windows_dll_search_path._dll_handles = [  # type: ignore[attr-defined]
            os.add_dll_directory(path) for path in existing_paths
        ]


_prepare_windows_dll_search_path()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyzed.sl as sl  # noqa: E402


DEFAULT_SVO = Path(__file__).resolve().with_name("20260802_150233.svo2")
DEFAULT_OUTPUT = (
    Path(__file__).resolve()
    / "output"
    / "20260802_150233_native_svo_depth_samples"
)


def _status_name(status: Any) -> str:
    return str(status).rsplit(".", 1)[-1]


def _check_status(status: Any, operation: str) -> None:
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"{operation} failed: {_status_name(status)}")


def _as_gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise RuntimeError(f"Unexpected image shape: {image.shape}")
    return image


def _matrix4(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Native stereo transform contains non-finite values")
    return matrix


def _camera_dict(camera: Any) -> dict[str, Any]:
    return {
        "fx": float(camera.fx),
        "fy": float(camera.fy),
        "cx": float(camera.cx),
        "cy": float(camera.cy),
        "distortion": [
            float(value)
            for value in np.asarray(camera.disto, dtype=np.float64).reshape(-1)
        ],
        "lens_distortion_model": str(camera.lens_distortion_model),
    }


def _depth_from_disparity(disparity_px: float, focal_px: float, baseline_m: float) -> float:
    if not np.isfinite(disparity_px) or disparity_px <= 1.0:
        return float("nan")
    return float(focal_px * baseline_m / disparity_px)


def _sample_frames(text: str) -> list[int]:
    values = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not values or values[0] < 0:
        raise ValueError("--sample-frames must contain non-negative frame indices")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate several depths with native SVO2 calibration."
    )
    parser.add_argument("--svo", type=Path, default=DEFAULT_SVO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sample-frames",
        default="0,20,40,60,80,99",
        help="Comma-separated sequential SVO frame indices to report.",
    )
    parser.add_argument("--num-disparities", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--uniqueness-ratio", type=int, default=8)
    parser.add_argument("--window-radius", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    svo_path = args.svo.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    sample_frames = _sample_frames(args.sample_frames)
    if not svo_path.is_file():
        raise FileNotFoundError(f"SVO2 file not found: {svo_path}")
    if args.num_disparities <= 0 or args.num_disparities % 16:
        raise ValueError("--num-disparities must be a positive multiple of 16")
    if args.block_size < 3 or args.block_size % 2 == 0:
        raise ValueError("--block-size must be an odd integer >= 3")
    if args.window_radius < 0:
        raise ValueError("--window-radius must be non-negative")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "native_svo_depth_samples.csv"
    output_json = output_dir / "native_calibration.json"
    if not args.overwrite and (output_csv.exists() or output_json.exists()):
        raise FileExistsError(
            f"Output exists: {output_dir}; use --overwrite to replace it"
        )

    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    # Calibration transforms follow the configured coordinate unit.  Make the
    # baseline and the resulting depth unambiguously metric.
    init.coordinate_units = sl.UNIT.METER
    init.camera_disable_self_calib = True

    zed = sl.Camera()
    _check_status(zed.open(init), "open SVO2 with embedded native calibration")
    left_mat = sl.Mat()
    right_mat = sl.Mat()
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=args.num_disparities,
        blockSize=args.block_size,
        P1=8 * args.block_size * args.block_size,
        P2=32 * args.block_size * args.block_size,
        disp12MaxDiff=1,
        uniquenessRatio=args.uniqueness_ratio,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    rows: list[dict[str, Any]] = []
    try:
        info = zed.get_camera_information()
        configuration = info.camera_configuration
        resolution = configuration.resolution
        raw = configuration.calibration_parameters_raw
        rectified = configuration.calibration_parameters
        raw_transform = _matrix4(raw.stereo_transform.m)
        rectified_transform = _matrix4(rectified.stereo_transform.m)
        focal_px = float(rectified.left_cam.fx)
        baseline_m = abs(float(rectified_transform[0, 3]))
        if not np.isfinite(focal_px) or focal_px <= 0:
            raise RuntimeError(f"Invalid native rectified focal length: {focal_px}")
        if not np.isfinite(baseline_m) or baseline_m <= 0:
            raise RuntimeError(f"Invalid native rectified baseline: {baseline_m}")

        total_frames = int(zed.get_svo_number_of_frames())
        if sample_frames[-1] >= total_frames:
            raise ValueError(
                f"Requested frame {sample_frames[-1]}, but SVO has {total_frames} frames"
            )
        print(f"SDK version: {sl.Camera.get_sdk_version()}")
        print(f"Camera: {_status_name(info.camera_model)}")
        print(f"SVO: {svo_path}")
        print(f"Native image size: {int(resolution.width)}x{int(resolution.height)}")
        print(f"Native raw left fx/fy: {raw.left_cam.fx:.9f}, {raw.left_cam.fy:.9f} px")
        print(f"Native raw right fx/fy: {raw.right_cam.fx:.9f}, {raw.right_cam.fy:.9f} px")
        print(f"Native rectified left fx/fy: {rectified.left_cam.fx:.9f}, {rectified.left_cam.fy:.9f} px")
        print(f"Native rectified right fx/fy: {rectified.right_cam.fx:.9f}, {rectified.right_cam.fy:.9f} px")
        print(f"Native rectified baseline: {baseline_m:.9f} m")
        print(f"Samples: {sample_frames}")

        runtime = sl.RuntimeParameters()
        target_set = set(sample_frames)
        max_frame = sample_frames[-1]
        for frame_index in range(max_frame + 1):
            status = zed.grab(runtime)
            if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            _check_status(status, f"grab frame {frame_index}")
            if frame_index not in target_set:
                continue
            _check_status(
                zed.retrieve_image(left_mat, sl.VIEW.LEFT),
                f"retrieve native left frame {frame_index}",
            )
            _check_status(
                zed.retrieve_image(right_mat, sl.VIEW.RIGHT),
                f"retrieve native right frame {frame_index}",
            )
            left = _as_gray(left_mat.get_data())
            right = _as_gray(right_mat.get_data())
            disparity = matcher.compute(left, right).astype(np.float32) / 16.0
            height, width = disparity.shape
            cx = width // 2
            cy = height // 2
            center_disparity = float(disparity[cy, cx])
            center_depth = _depth_from_disparity(
                center_disparity, focal_px, baseline_m
            )
            radius = args.window_radius
            window = disparity[
                max(0, cy - radius) : min(height, cy + radius + 1),
                max(0, cx - radius) : min(width, cx + radius + 1),
            ]
            valid_window = window[np.isfinite(window) & (window > 1.0)]
            window_disparity = (
                float(np.median(valid_window))
                if valid_window.size
                else float("nan")
            )
            window_depth = _depth_from_disparity(
                window_disparity, focal_px, baseline_m
            )
            row = {
                "frame_index": frame_index,
                "svo_position": int(zed.get_svo_position()),
                "center_x": cx,
                "center_y": cy,
                "center_disparity_px": center_disparity,
                "center_depth_m": center_depth,
                "window_median_disparity_px": window_disparity,
                "window_median_depth_m": window_depth,
                "window_valid_count": int(valid_window.size),
            }
            rows.append(row)
            print(
                f"frame {frame_index:>4}: disparity={center_disparity:>8.3f} px, "
                f"center depth={center_depth:>8.4f} m, "
                f"5x5 depth={window_depth:>8.4f} m"
            )

        native_metadata = {
            "method": "OpenCV StereoSGBM on SVO native rectified images",
            "depth_formula": "depth_m = native_rectified_focal_px * native_rectified_baseline_m / disparity_px",
            "svo": str(svo_path),
            "sdk_version": sl.Camera.get_sdk_version(),
            "camera_model": _status_name(info.camera_model),
            "resolution": {
                "width": int(resolution.width),
                "height": int(resolution.height),
            },
            "total_svo_frames": total_frames,
            "sample_frames": sample_frames,
            "raw_calibration": {
                "left": _camera_dict(raw.left_cam),
                "right": _camera_dict(raw.right_cam),
                "stereo_transform": raw_transform.tolist(),
            },
            "rectified_calibration": {
                "left": _camera_dict(rectified.left_cam),
                "right": _camera_dict(rectified.right_cam),
                "stereo_transform": rectified_transform.tolist(),
            },
            "depth_parameters": {
                "focal_length_px": focal_px,
                "baseline_m": baseline_m,
                "num_disparities": args.num_disparities,
                "block_size": args.block_size,
                "uniqueness_ratio": args.uniqueness_ratio,
                "window_radius": args.window_radius,
            },
            "samples": rows,
        }
        output_json.write_text(
            json.dumps(native_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
                "frame_index", "svo_position", "center_x", "center_y",
                "center_disparity_px", "center_depth_m",
                "window_median_disparity_px", "window_median_depth_m",
                "window_valid_count",
            ])
            writer.writeheader()
            writer.writerows(rows)
    finally:
        zed.close()

    print(f"Saved calibration and samples to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
