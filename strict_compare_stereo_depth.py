"""Run native-SVO and custom-calibration stereo depth in isolated pipelines.

The two branches deliberately do not share images, rectification maps,
disparities, focal lengths, or baselines:

* native branch: open the SVO with no external calibration, retrieve native
  rectified LEFT/RIGHT images, and use the SVO embedded rectified calibration;
* custom branch: open a fresh SVO handle with the custom calibration file,
  retrieve raw LEFT_UNRECTIFIED/RIGHT_UNRECTIFIED images, parse the custom
  OpenCV calibration files, rectify with OpenCV, and use the custom P1/baseline.

Both branches then run the same OpenCV StereoSGBM settings and save one
float32 depth map per frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


def _prepare_windows_dll_search_path() -> None:
    if os.name != "nt":
        return
    sdk_root = os.environ.get("ZED_SDK_ROOT_DIR")
    if not sdk_root:
        return
    root = Path(sdk_root)
    search_paths = [
        root / "bin",
        root / "dependencies" / "freeglut" / "bin",
        root / "dependencies" / "freeglut_2.8" / "x64",
        root / "dependencies" / "glew" / "bin",
        root / "dependencies" / "glew-1.12.0" / "x64",
        root / "dependencies" / "opencv" / "x64" / "vc16" / "bin",
        root / "dependencies" / "opencv_3.1.0" / "x64",
    ]
    existing = [str(path) for path in search_paths if path.is_dir()]
    if not existing:
        return
    os.environ["PATH"] = os.pathsep.join(
        [*existing, *(item for item in os.environ.get("PATH", "").split(os.pathsep) if item)]
    )
    if hasattr(os, "add_dll_directory"):
        _prepare_windows_dll_search_path._dll_handles = [  # type: ignore[attr-defined]
            os.add_dll_directory(path) for path in existing
        ]


_prepare_windows_dll_search_path()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyzed.sl as sl  # noqa: E402


DEFAULT_SVO = Path(__file__).resolve().with_name("20260802_150233.svo2")
DEFAULT_CUSTOM_DIR = Path(__file__).resolve().parent / "Calibration" / "标定结果"
DEFAULT_CUSTOM_YML = Path(__file__).resolve().parent / "Calibration" / "zed_custom_opencv.yml"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "output"
    / "20260802_150233_strict_calibration_compare_20"
)


def _status_name(status: Any) -> str:
    return str(status).rsplit(".", 1)[-1]


def _check_status(status: Any, operation: str) -> None:
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"{operation} failed: {_status_name(status)}")


def _as_gray(data: np.ndarray) -> np.ndarray:
    image = np.asarray(data)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 2:
        return image
    raise RuntimeError(f"Unexpected image shape: {image.shape}")


def _matrix4(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Non-finite stereo transform")
    return matrix


def _camera_metadata(camera: Any) -> dict[str, Any]:
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


def _extract_numbers(text: str) -> list[float]:
    pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    return [float(value) for value in re.findall(pattern, text)]


def _yaml_section(text: str, name: str, next_name: str | None = None) -> str:
    end = rf"^\s*{re.escape(next_name)}\s*:" if next_name else r"\Z"
    match = re.search(
        rf"(?ms)^\s*{re.escape(name)}\s*:\s*(.*?)(?={end})",
        text,
    )
    if match is None:
        raise RuntimeError(f"Missing section {name!r}")
    return match.group(1)


def _load_custom_calibration(intrinsics_path: Path, extrinsics_path: Path) -> dict[str, Any]:
    intrinsics_text = intrinsics_path.read_text(encoding="utf-8-sig")
    extrinsics_text = extrinsics_path.read_text(encoding="utf-8-sig")

    def camera(name: str, next_name: str | None) -> dict[str, Any]:
        section = _yaml_section(intrinsics_text, name, next_name)
        values: dict[str, float] = {}
        for key in ("fx", "fy", "cx", "cy"):
            match = re.search(rf"(?m)^\s*{key}\s*:\s*([^#\r\n]+)", section)
            if match is None:
                raise RuntimeError(f"Missing {name}.{key}")
            values[key] = float(match.group(1).strip())
        distortion_match = re.search(
            r"(?m)^\s*distortion_coefficients\s*:\s*\[([^]]+)\]", section
        )
        if distortion_match is None:
            raise RuntimeError(f"Missing {name}.distortion_coefficients")
        distortion = _extract_numbers(distortion_match.group(1))
        if len(distortion) < 5:
            raise RuntimeError(f"{name} requires five distortion coefficients")
        return {**values, "distortion": np.asarray(distortion[:5], dtype=np.float64)}

    left = camera("left", "right")
    right = camera("right", None)
    t_match = re.search(r"(?m)^\s*T\s*:\s*\[([^]]+)\]", extrinsics_text)
    r_match = re.search(r"(?ms)^\s*R\s*:\s*(.*?)(?=^\s*T\s*:)", extrinsics_text)
    if t_match is None or r_match is None:
        raise RuntimeError("Missing R/T in custom extrinsics")
    translation_mm = np.asarray(_extract_numbers(t_match.group(1)), dtype=np.float64)
    rotation = np.asarray(_extract_numbers(r_match.group(1)), dtype=np.float64)
    if translation_mm.size != 3 or rotation.size != 9:
        raise RuntimeError("Custom R/T has unexpected dimensions")
    return {
        "left": left,
        "right": right,
        "rotation": rotation.reshape(3, 3),
        "translation_mm": translation_mm,
        "source_intrinsics": str(intrinsics_path),
        "source_extrinsics": str(extrinsics_path),
    }


def _custom_rectification(
    calibration: dict[str, Any],
    image_size: tuple[int, int],
    output_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    if output_size is None:
        output_size = image_size
    left = calibration["left"]
    right = calibration["right"]
    left_k = np.array(
        [[left["fx"], 0.0, left["cx"]], [0.0, left["fy"], left["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    right_k = np.array(
        [[right["fx"], 0.0, right["cx"]], [0.0, right["fy"], right["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
        left_k,
        left["distortion"],
        right_k,
        right["distortion"],
        image_size,
        calibration["rotation"],
        calibration["translation_mm"].reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
        newImageSize=output_size,
    )
    map_l = cv2.initUndistortRectifyMap(
        left_k, left["distortion"], r1, p1, output_size, cv2.CV_32FC1
    )
    map_r = cv2.initUndistortRectifyMap(
        right_k, right["distortion"], r2, p2, output_size, cv2.CV_32FC1
    )
    focal = float(p1[0, 0])
    baseline_m = abs(float(p2[0, 3] / p2[0, 0])) / 1000.0
    if focal <= 0 or baseline_m <= 0:
        raise RuntimeError(f"Invalid custom rectification focal/baseline: {focal}, {baseline_m}")
    return {
        "left_map_x": map_l[0],
        "left_map_y": map_l[1],
        "right_map_x": map_r[0],
        "right_map_y": map_r[1],
        "p1": p1,
        "p2": p2,
        "q": q,
        "roi_left": list(roi1),
        "roi_right": list(roi2),
        "focal_length_px": focal,
        "baseline_m": baseline_m,
    }


def _make_matcher() -> cv2.StereoSGBM:
    block_size = 5
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=256,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def _depth_from_disparity(disparity: np.ndarray, focal: float, baseline: float) -> np.ndarray:
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 1.0)
    depth[valid] = (focal * baseline / disparity[valid]).astype(np.float32)
    depth[depth > 100.0] = np.nan
    return depth


def _center_values(
    disparity: np.ndarray, depth: np.ndarray, radius: int = 2
) -> tuple[float, float, float, int]:
    height, width = depth.shape
    cx, cy = width // 2, height // 2
    center_disparity = float(disparity[cy, cx])
    center_depth = float(depth[cy, cx])
    window = depth[
        max(0, cy - radius) : min(height, cy + radius + 1),
        max(0, cx - radius) : min(width, cx + radius + 1),
    ]
    valid = window[np.isfinite(window) & (window > 0.0)]
    median = float(np.median(valid)) if valid.size else float("nan")
    return center_disparity, center_depth, median, int(valid.size)


def _stats(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    result: dict[str, Any] = {
        "count": int(finite.size),
        "invalid_count": int(len(values) - finite.size),
        "valid_ratio": float(finite.size / len(values)) if values else 0.0,
    }
    if finite.size:
        result.update(
            {
                "min_m": float(np.min(finite)),
                "p05_m": float(np.percentile(finite, 5)),
                "median_m": float(np.median(finite)),
                "mean_m": float(np.mean(finite)),
                "std_m": float(np.std(finite)),
                "p95_m": float(np.percentile(finite, 95)),
                "max_m": float(np.max(finite)),
            }
        )
    return result


def _init_parameters(svo_path: Path, custom_yml: Path | None) -> sl.InitParameters:
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.METER
    init.camera_disable_self_calib = True
    if custom_yml is not None:
        init.optional_opencv_calibration_file = str(custom_yml)
    return init


def _save_branch(
    branch_dir: Path,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    center_depths: list[float],
    window_depths: list[float],
    center_disparities: list[float],
) -> dict[str, Any]:
    branch_dir.mkdir(parents=True, exist_ok=True)
    csv_path = branch_dir / "center_depth.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "frame_index", "svo_position", "center_x", "center_y",
            "center_disparity_px", "center_depth_m",
            "center_window_median_depth_m", "center_window_valid_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        **metadata,
        "center_pixel_stats_m": _stats(center_depths),
        "center_window_median_stats_m": _stats(window_depths),
        "center_disparity_stats_px": _stats(
            [value for value in center_disparities if np.isfinite(value) and value > 1.0]
        ),
        "files": {
            "center_depth_csv": str(csv_path),
            "depth_maps_directory": str(branch_dir / "depth_maps"),
            "depth_map_count": len(rows),
        },
    }
    (branch_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _run_native_branch(
    svo_path: Path, branch_dir: Path, frame_count: int
) -> dict[str, Any]:
    print("\n=== NATIVE branch: fresh SVO open, embedded calibration, native rectified images ===")
    zed = sl.Camera()
    _check_status(zed.open(_init_parameters(svo_path, None)), "open native SVO")
    left_mat, right_mat = sl.Mat(), sl.Mat()
    try:
        info = zed.get_camera_information()
        configuration = info.camera_configuration
        resolution = configuration.resolution
        raw = configuration.calibration_parameters_raw
        rectified = configuration.calibration_parameters
        native_transform = _matrix4(rectified.stereo_transform.m)
        focal = float(rectified.left_cam.fx)
        baseline = abs(float(native_transform[0, 3]))
        width, height = int(resolution.width), int(resolution.height)
        total_frames = int(zed.get_svo_number_of_frames())
        if frame_count > total_frames:
            raise ValueError(f"Requested {frame_count} native frames, SVO has {total_frames}")
        matcher = _make_matcher()
        runtime = sl.RuntimeParameters()
        depth_dir = branch_dir / "depth_maps"
        depth_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        center_depths: list[float] = []
        window_depths: list[float] = []
        center_disparities: list[float] = []
        started = time.monotonic()
        for frame_index in range(frame_count):
            _check_status(zed.grab(runtime), f"native grab frame {frame_index}")
            _check_status(zed.retrieve_image(left_mat, sl.VIEW.LEFT), "native retrieve left")
            _check_status(zed.retrieve_image(right_mat, sl.VIEW.RIGHT), "native retrieve right")
            left = _as_gray(left_mat.get_data())
            right = _as_gray(right_mat.get_data())
            disparity = matcher.compute(left, right).astype(np.float32) / 16.0
            depth = _depth_from_disparity(disparity, focal, baseline)
            np.save(depth_dir / f"depth_{frame_index:06d}.npy", depth)
            center_d, center_z, window_z, window_count = _center_values(disparity, depth)
            center_disparities.append(center_d)
            center_depths.append(center_z)
            window_depths.append(window_z)
            rows.append(
                {
                    "frame_index": frame_index,
                    "svo_position": int(zed.get_svo_position()),
                    "center_x": width // 2,
                    "center_y": height // 2,
                    "center_disparity_px": center_d,
                    "center_depth_m": center_z,
                    "center_window_median_depth_m": window_z,
                    "center_window_valid_count": window_count,
                }
            )
        metadata = {
            "branch": "native",
            "parameter_source": "SVO embedded calibration only",
            "image_source": "VIEW.LEFT and VIEW.RIGHT (native rectified)",
            "depth_method": "OpenCV StereoSGBM; no ZED MEASURE.DEPTH",
            "svo": str(svo_path),
            "sdk_version": sl.Camera.get_sdk_version(),
            "camera_model": _status_name(info.camera_model),
            "frames_processed": frame_count,
            "total_svo_frames": total_frames,
            "image_size": {"width": width, "height": height},
            "raw_calibration_from_svo": {
                "left": _camera_metadata(raw.left_cam),
                "right": _camera_metadata(raw.right_cam),
            },
            "rectified_calibration_from_svo": {
                "left": _camera_metadata(rectified.left_cam),
                "right": _camera_metadata(rectified.right_cam),
                "stereo_transform_m": native_transform.tolist(),
            },
            "depth_parameters": {
                "focal_length_px": focal,
                "baseline_m": baseline,
                "formula": "depth_m = native_rectified_focal_px * native_rectified_baseline_m / disparity_px",
                "sgbm_num_disparities": 256,
                "sgbm_block_size": 5,
                "sgbm_uniqueness_ratio": 8,
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        summary = _save_branch(
            branch_dir, metadata, rows, center_depths, window_depths, center_disparities
        )
        print(
            f"NATIVE: raw fx={raw.left_cam.fx:.6f}; rectified fx={focal:.6f}; "
            f"baseline={baseline:.9f} m; center={summary['center_pixel_stats_m']}"
        )
        return summary
    finally:
        zed.close()


def _run_custom_branch(
    svo_path: Path,
    custom_yml: Path,
    calibration: dict[str, Any],
    branch_dir: Path,
    frame_count: int,
) -> dict[str, Any]:
    print("\n=== CUSTOM branch: fresh SVO open, custom calibration, raw images + OpenCV rectification ===")
    zed = sl.Camera()
    _check_status(zed.open(_init_parameters(svo_path, custom_yml)), "open custom-calibration SVO")
    left_mat, right_mat = sl.Mat(), sl.Mat()
    try:
        info = zed.get_camera_information()
        configuration = info.camera_configuration
        resolution = configuration.resolution
        width, height = int(resolution.width), int(resolution.height)
        total_frames = int(zed.get_svo_number_of_frames())
        if frame_count > total_frames:
            raise ValueError(f"Requested {frame_count} custom frames, SVO has {total_frames}")
        # Verify that the calibration file used to open this branch is the
        # same calibration that was parsed below.  The depth computation still
        # uses only the explicitly parsed OpenCV parameters and raw images.
        sdk_raw = configuration.calibration_parameters_raw
        observed_internal = _matrix4(sdk_raw.stereo_transform.m)
        sdk_errors = {
            "left_fx_px": abs(float(sdk_raw.left_cam.fx) - calibration["left"]["fx"]),
            "left_fy_px": abs(float(sdk_raw.left_cam.fy) - calibration["left"]["fy"]),
            "left_cx_px": abs(float(sdk_raw.left_cam.cx) - calibration["left"]["cx"]),
            "left_cy_px": abs(float(sdk_raw.left_cam.cy) - calibration["left"]["cy"]),
            "right_fx_px": abs(float(sdk_raw.right_cam.fx) - calibration["right"]["fx"]),
            "right_fy_px": abs(float(sdk_raw.right_cam.fy) - calibration["right"]["fy"]),
            "right_cx_px": abs(float(sdk_raw.right_cam.cx) - calibration["right"]["cx"]),
            "right_cy_px": abs(float(sdk_raw.right_cam.cy) - calibration["right"]["cy"]),
            # The SDK transform representation changes with its coordinate
            # system convention.  Its translation norm is convention
            # independent and must match the custom T magnitude.
            "stereo_translation_norm_m": abs(
                float(np.linalg.norm(observed_internal[:3, 3]))
                - float(np.linalg.norm(calibration["translation_mm"]) / 1000.0)
            ),
        }
        if max(sdk_errors.values()) > 2e-2:
            raise RuntimeError(
                "The custom calibration applied while opening the SVO does not "
                f"match the parsed calibration: {sdk_errors}"
            )
        rectification = _custom_rectification(calibration, (width, height))
        matcher = _make_matcher()
        runtime = sl.RuntimeParameters()
        depth_dir = branch_dir / "depth_maps"
        depth_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        center_depths: list[float] = []
        window_depths: list[float] = []
        center_disparities: list[float] = []
        started = time.monotonic()
        for frame_index in range(frame_count):
            _check_status(zed.grab(runtime), f"custom grab frame {frame_index}")
            _check_status(
                zed.retrieve_image(left_mat, sl.VIEW.LEFT_UNRECTIFIED),
                "custom retrieve raw left",
            )
            _check_status(
                zed.retrieve_image(right_mat, sl.VIEW.RIGHT_UNRECTIFIED),
                "custom retrieve raw right",
            )
            left_raw = _as_gray(left_mat.get_data())
            right_raw = _as_gray(right_mat.get_data())
            left = cv2.remap(
                left_raw,
                rectification["left_map_x"],
                rectification["left_map_y"],
                cv2.INTER_LINEAR,
            )
            right = cv2.remap(
                right_raw,
                rectification["right_map_x"],
                rectification["right_map_y"],
                cv2.INTER_LINEAR,
            )
            disparity = matcher.compute(left, right).astype(np.float32) / 16.0
            depth = _depth_from_disparity(
                disparity,
                rectification["focal_length_px"],
                rectification["baseline_m"],
            )
            np.save(depth_dir / f"depth_{frame_index:06d}.npy", depth)
            center_d, center_z, window_z, window_count = _center_values(disparity, depth)
            center_disparities.append(center_d)
            center_depths.append(center_z)
            window_depths.append(window_z)
            rows.append(
                {
                    "frame_index": frame_index,
                    "svo_position": int(zed.get_svo_position()),
                    "center_x": width // 2,
                    "center_y": height // 2,
                    "center_disparity_px": center_d,
                    "center_depth_m": center_z,
                    "center_window_median_depth_m": window_z,
                    "center_window_valid_count": window_count,
                }
            )
        metadata = {
            "branch": "custom",
            "parameter_source": "Calibration/ camera_intrinsics.yaml + stereo_extrinsics.yaml",
            "image_source": "VIEW.LEFT_UNRECTIFIED and VIEW.RIGHT_UNRECTIFIED",
            "depth_method": "OpenCV stereoRectify + StereoSGBM; no ZED MEASURE.DEPTH",
            "svo": str(svo_path),
            "custom_opencv_calibration_file_used_to_open_svo": str(custom_yml),
            "custom_sdk_calibration_verification": {
                "passed": True,
                "max_abs_errors": sdk_errors,
                "observed_internal_stereo_transform_m": observed_internal.tolist(),
                "source_R_left_to_right": calibration["rotation"].tolist(),
                "source_T_norm_m": float(
                    np.linalg.norm(calibration["translation_mm"]) / 1000.0
                ),
            },
            "sdk_version": sl.Camera.get_sdk_version(),
            "camera_model": _status_name(info.camera_model),
            "frames_processed": frame_count,
            "total_svo_frames": total_frames,
            "image_size": {"width": width, "height": height},
            "custom_raw_calibration": {
                "left": {
                    "fx": calibration["left"]["fx"],
                    "fy": calibration["left"]["fy"],
                    "cx": calibration["left"]["cx"],
                    "cy": calibration["left"]["cy"],
                    "distortion": calibration["left"]["distortion"].tolist(),
                },
                "right": {
                    "fx": calibration["right"]["fx"],
                    "fy": calibration["right"]["fy"],
                    "cx": calibration["right"]["cx"],
                    "cy": calibration["right"]["cy"],
                    "distortion": calibration["right"]["distortion"].tolist(),
                },
                "R_left_to_right": calibration["rotation"].tolist(),
                "T_mm": calibration["translation_mm"].tolist(),
            },
            "opencv_rectification": {
                "P1": rectification["p1"].tolist(),
                "P2": rectification["p2"].tolist(),
                "Q": rectification["q"].tolist(),
                "roi_left": rectification["roi_left"],
                "roi_right": rectification["roi_right"],
            },
            "depth_parameters": {
                "focal_length_px": rectification["focal_length_px"],
                "baseline_m": rectification["baseline_m"],
                "formula": "depth_m = custom_rectified_P1_fx * custom_rectified_baseline_m / disparity_px",
                "sgbm_num_disparities": 256,
                "sgbm_block_size": 5,
                "sgbm_uniqueness_ratio": 8,
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        summary = _save_branch(
            branch_dir, metadata, rows, center_depths, window_depths, center_disparities
        )
        print(
            f"CUSTOM: raw fx={calibration['left']['fx']:.6f}; "
            f"rectified fx={rectification['focal_length_px']:.6f}; "
            f"baseline={rectification['baseline_m']:.9f} m; "
            f"center={summary['center_pixel_stats_m']}"
        )
        return summary
    finally:
        zed.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("svo", nargs="?", type=Path, default=DEFAULT_SVO)
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_CUSTOM_DIR / "camera_intrinsics.yaml")
    parser.add_argument("--extrinsics", type=Path, default=DEFAULT_CUSTOM_DIR / "stereo_extrinsics.yaml")
    parser.add_argument("--custom-opencv", type=Path, default=DEFAULT_CUSTOM_YML)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    return args


def main() -> int:
    args = _parse_args()
    svo_path = args.svo.expanduser().resolve()
    intrinsics_path = args.intrinsics.expanduser().resolve()
    extrinsics_path = args.extrinsics.expanduser().resolve()
    custom_yml = args.custom_opencv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path, label in (
        (svo_path, "SVO2"),
        (intrinsics_path, "custom intrinsics"),
        (extrinsics_path, "custom extrinsics"),
        (custom_yml, "custom OpenCV calibration"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty; use --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "native").mkdir(exist_ok=True)
    (output_dir / "custom").mkdir(exist_ok=True)

    # Parse the custom files before running either branch.  The custom branch
    # receives this parsed object only; the native branch never sees it.
    custom_calibration = _load_custom_calibration(intrinsics_path, extrinsics_path)

    # Each branch opens and closes its own SVO handle and starts at position 0.
    native_summary = _run_native_branch(svo_path, output_dir / "native", args.frames)
    custom_summary = _run_custom_branch(
        svo_path,
        custom_yml,
        custom_calibration,
        output_dir / "custom",
        args.frames,
    )
    comparison = {
        "protocol": "isolated fresh acquisition and calibration in each branch",
        "frames_per_branch": args.frames,
        "native_summary": native_summary,
        "custom_summary": custom_summary,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nCompleted isolated comparison: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
