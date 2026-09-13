"""Compare rectification/depth combinations by their radial depth profile.

This is an audit tool, not a production depth pipeline.  Each combination
opens a fresh SVO handle, reads raw ``VIEW.LEFT_UNRECTIFIED`` and
``VIEW.RIGHT_UNRECTIFIED`` frames, applies exactly one OpenCV rectification,
and computes a new SGBM disparity/depth map.  The report keeps the image
center and the rectified optical center separate because custom alpha=0
rectification can move the virtual principal point away from the image center.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


def _prepare_windows_dll_search_path() -> None:
    if os.name != "nt":
        return
    root_text = os.environ.get("ZED_SDK_ROOT_DIR")
    if not root_text:
        return
    root = Path(root_text)
    paths = [
        root / "bin",
        root / "dependencies" / "freeglut" / "bin",
        root / "dependencies" / "freeglut_2.8" / "x64",
        root / "dependencies" / "glew" / "bin",
        root / "dependencies" / "glew-1.12.0" / "x64",
        root / "dependencies" / "opencv" / "x64" / "vc16" / "bin",
        root / "dependencies" / "opencv_3.1.0" / "x64",
    ]
    existing = [str(path) for path in paths if path.is_dir()]
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


ROOT = Path(__file__).resolve().parent
DEFAULT_SVO = ROOT / "20260802_150233.svo2"
DEFAULT_CALIBRATION = ROOT / "Calibration" / "zed_custom_opencv.yml"
DEFAULT_OUTPUT = ROOT / "RECTIFICATION_PROFILE_EXPERIMENT.json"
WIDTH = 1920
HEIGHT = 1080


def _status_name(value: Any) -> str:
    return str(value).rsplit(".", 1)[-1]


def _check(status: Any, operation: str) -> None:
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"{operation} failed: {_status_name(status)}")


def _as_gray(data: np.ndarray) -> np.ndarray:
    image = np.asarray(data)
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise RuntimeError(f"Unexpected image shape: {image.shape}")
    return np.ascontiguousarray(image, dtype=np.uint8)


def _read_matrix(fs: cv2.FileStorage, name: str) -> np.ndarray:
    value = fs.getNode(name).mat()
    if value is None:
        raise RuntimeError(f"Missing {name} in calibration file")
    return np.asarray(value, dtype=np.float64)


def _load_calibration(path: Path) -> dict[str, np.ndarray]:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Could not open calibration: {path}")
    try:
        rotation_source = _read_matrix(fs, "R")
        if rotation_source.size == 3:
            rotation = cv2.Rodrigues(rotation_source.reshape(3, 1))[0]
            rotation_encoding = "Rodrigues rotation vector"
        elif rotation_source.size == 9:
            rotation = rotation_source.reshape(3, 3)
            rotation_encoding = "3x3 rotation matrix"
        else:
            raise RuntimeError(f"Unexpected R size in calibration file: {rotation_source.shape}")
        data = {
            "K_left": _read_matrix(fs, "K_LEFT").reshape(3, 3),
            "K_right": _read_matrix(fs, "K_RIGHT").reshape(3, 3),
            "D_left": _read_matrix(fs, "D_LEFT").reshape(-1),
            "D_right": _read_matrix(fs, "D_RIGHT").reshape(-1),
            "R": rotation,
            "T_mm": _read_matrix(fs, "T").reshape(3),
            "R_encoding": rotation_encoding,
        }
    finally:
        fs.release()
    return data


def _rectification(
    calibration: dict[str, np.ndarray],
    alpha: float,
    centered_projection: bool,
) -> dict[str, Any]:
    k_left = calibration["K_left"]
    k_right = calibration["K_right"]
    d_left = calibration["D_left"]
    d_right = calibration["D_right"]
    r_left, r_right, p_left, p_right, q, roi_left, roi_right = cv2.stereoRectify(
        k_left,
        d_left,
        k_right,
        d_right,
        (WIDTH, HEIGHT),
        calibration["R"],
        calibration["T_mm"].reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,
        newImageSize=(WIDTH, HEIGHT),
    )
    if centered_projection:
        # This is a valid virtual-camera translation, not a second
        # rectification.  Keep the stereo focal/baseline relation unchanged.
        p_left = p_left.copy()
        p_right = p_right.copy()
        p_left[0, 2] = WIDTH / 2.0
        p_right[0, 2] = WIDTH / 2.0
        p_left[1, 2] = HEIGHT / 2.0
        p_right[1, 2] = HEIGHT / 2.0
    map_left = cv2.initUndistortRectifyMap(
        k_left, d_left, r_left, p_left, (WIDTH, HEIGHT), cv2.CV_32FC1
    )
    map_right = cv2.initUndistortRectifyMap(
        k_right, d_right, r_right, p_right, (WIDTH, HEIGHT), cv2.CV_32FC1
    )
    focal = float(p_left[0, 0])
    baseline_m = abs(float(p_right[0, 3] / p_right[0, 0])) / 1000.0
    if not math.isfinite(focal) or focal <= 0.0 or not math.isfinite(baseline_m) or baseline_m <= 0.0:
        raise RuntimeError(f"Invalid rectification f/B: {focal}, {baseline_m}")
    return {
        "alpha": float(alpha),
        "centered_projection": bool(centered_projection),
        "R_left": r_left,
        "R_right": r_right,
        "P_left": p_left,
        "P_right": p_right,
        "Q": q,
        "left_map_x": map_left[0],
        "left_map_y": map_left[1],
        "right_map_x": map_right[0],
        "right_map_y": map_right[1],
        "focal_px": focal,
        "baseline_m": baseline_m,
        "roi_left": tuple(int(v) for v in roi_left),
        "roi_right": tuple(int(v) for v in roi_right),
    }


def _matcher(config: dict[str, int]) -> cv2.StereoSGBM:
    block = int(config["block_size"])
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=int(config["num_disparities"]),
        blockSize=block,
        P1=8 * block * block,
        P2=32 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=int(config["uniqueness_ratio"]),
        speckleWindowSize=int(config["speckle_window_size"]),
        speckleRange=int(config["speckle_range"]),
        preFilterCap=63,
        mode=getattr(cv2, "STEREO_SGBM_MODE_SGBM_3WAY", cv2.STEREO_SGBM_MODE_SGBM),
    )


def _depth(disparity: np.ndarray, focal_px: float, baseline_m: float) -> np.ndarray:
    result = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 1.0)
    result[valid] = (focal_px * baseline_m / disparity[valid]).astype(np.float32)
    result[(result <= 0.0) | (result > 100.0)] = np.nan
    return result


def _band_masks(cx: float, cy: float) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    radius = np.sqrt(((xx - cx) / (WIDTH / 2.0)) ** 2 + ((yy - cy) / (HEIGHT / 2.0)) ** 2)
    return {
        "center_r025": radius <= 0.25,
        "inner_mid_r025_050": (radius > 0.25) & (radius <= 0.50),
        "outer_mid_r050_075": (radius > 0.50) & (radius < 0.75),
        "edge_r075": radius >= 0.75,
    }


def _band_values(depth: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, dict[str, float | int | None]]:
    output: dict[str, dict[str, float | int | None]] = {}
    for name, mask in masks.items():
        values = depth[mask]
        values = values[np.isfinite(values) & (values > 0.0)]
        output[name] = {
            "median_m": float(np.median(values)) if values.size else None,
            "valid_count": int(values.size),
            "valid_ratio": float(values.size / np.count_nonzero(mask)),
        }
    return output


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]["median_m"]) for row in rows if row[key]["median_m"] is not None]
    if not values:
        return {"valid_frames": 0}
    return {
        "valid_frames": len(values),
        "median_of_frame_medians_m": float(np.median(values)),
        "min_m": float(np.min(values)),
        "max_m": float(np.max(values)),
        "median_valid_ratio": float(np.median([row[key]["valid_ratio"] for row in rows])),
        "median_valid_count": int(np.median([row[key]["valid_count"] for row in rows])),
    }


def _profile_summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    names = [
        f"{prefix}center_r025",
        f"{prefix}inner_mid_r025_050",
        f"{prefix}outer_mid_r050_075",
        f"{prefix}edge_r075",
    ]
    aggregate = {name.removeprefix(prefix): _aggregate(rows, name) for name in names}
    complete = [
        row for row in rows
        if all(row[name]["median_m"] is not None for name in names)
    ]
    monotonic = [
        float(row[names[3]]["median_m"]) - float(row[names[0]]["median_m"])
        for row in complete
    ]
    ordered = [
        all(float(row[names[i]]["median_m"]) < float(row[names[i + 1]]["median_m"]) for i in range(3))
        for row in complete
    ]
    return {
        "bands": aggregate,
        "complete_frames": len(complete),
        "strict_center_to_edge_monotonic_frames": int(sum(ordered)),
        "edge_minus_center_frame_delta_stats_m": {
            "median": float(np.median(monotonic)) if monotonic else None,
            "min": float(np.min(monotonic)) if monotonic else None,
            "max": float(np.max(monotonic)) if monotonic else None,
        },
    }


def _open_svo(svo: Path, calibration_path: Path) -> sl.Camera:
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.METER
    init.camera_disable_self_calib = True
    init.optional_opencv_calibration_file = str(calibration_path)
    zed = sl.Camera()
    _check(zed.open(init), "open SVO with custom calibration")
    return zed


def _run_combination(
    svo: Path,
    calibration_path: Path,
    rectification: dict[str, Any],
    matcher_config: dict[str, int],
    positions: list[int],
) -> dict[str, Any]:
    zed = _open_svo(svo, calibration_path)
    try:
        left_mat, right_mat = sl.Mat(), sl.Mat()
        matcher = _matcher(matcher_config)
        image_masks = _band_masks(WIDTH / 2.0, HEIGHT / 2.0)
        optical_masks = _band_masks(
            float(rectification["P_left"][0, 2]),
            float(rectification["P_left"][1, 2]),
        )
        rows: list[dict[str, Any]] = []
        for position in positions:
            _check(zed.set_svo_position(position), f"set SVO position {position}")
            _check(zed.grab(), f"grab SVO position {position}")
            _check(zed.retrieve_image(left_mat, sl.VIEW.LEFT_UNRECTIFIED), "retrieve raw left")
            _check(zed.retrieve_image(right_mat, sl.VIEW.RIGHT_UNRECTIFIED), "retrieve raw right")
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
            depth = _depth(disparity, rectification["focal_px"], rectification["baseline_m"])
            image_bands = _band_values(depth, image_masks)
            optical_bands = _band_values(depth, optical_masks)
            rows.append({
                "frame": int(position),
                "image_bands": image_bands,
                "optical_bands": optical_bands,
                "overall_valid_ratio": float(np.count_nonzero(np.isfinite(depth)) / depth.size),
            })
            print(
                f"  frame={position:5d} image center/edge="
                f"{image_bands['center_r025']['median_m']} / {image_bands['edge_r075']['median_m']} m; "
                f"optical center/edge="
                f"{optical_bands['center_r025']['median_m']} / {optical_bands['edge_r075']['median_m']} m",
                flush=True,
            )
        return {
            "rectification": {
                "alpha": rectification["alpha"],
                "centered_projection": rectification["centered_projection"],
                "focal_px": rectification["focal_px"],
                "baseline_m": rectification["baseline_m"],
                "P_left": rectification["P_left"].tolist(),
                "P_right": rectification["P_right"].tolist(),
                "roi_left": list(rectification["roi_left"]),
                "roi_right": list(rectification["roi_right"]),
                "source": "Calibration/zed_custom_opencv.yml; raw unrectified views; exactly one OpenCV remap",
            },
            "matcher": matcher_config,
            "positions": positions,
            "rows": rows,
            "image_center_profile": _profile_summary(rows, "image_bands.") if False else None,
        }
    finally:
        zed.close()


def _fix_result_profile(result: dict[str, Any]) -> None:
    rows = result["rows"]
    # Remove the placeholder written by _run_combination before generating
    # the real summaries.  The image-center summary must be retained; it is
    # intentionally different from the optical-center summary when P1.cx/cy
    # is not at the output image center.
    result.pop("image_center_profile", None)
    for coordinate in ("image_bands", "optical_bands"):
        prefix = coordinate + "."
        profile = _profile_summary(
            [{prefix + key: row[coordinate][key] for key in row[coordinate]} for row in rows],
            prefix,
        )
        result[coordinate.replace("_bands", "") + "_center_profile"] = profile


def _parse_positions(text: str) -> list[int]:
    values = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not values or values[0] < 0:
        raise ValueError("positions must contain non-negative integers")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svo", type=Path, default=DEFAULT_SVO)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--positions",
        default="0,5000,10000,15000,20000,25000,30000,35000,35854",
    )
    args = parser.parse_args()
    svo = args.svo.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not svo.is_file() or not calibration_path.is_file():
        raise FileNotFoundError(f"missing SVO or calibration: {svo}, {calibration_path}")
    positions = _parse_positions(args.positions)
    calibration = _load_calibration(calibration_path)
    rectifications = [
        _rectification(calibration, 0.0, False),
        _rectification(calibration, 0.0, True),
        _rectification(calibration, 0.5, False),
        _rectification(calibration, 0.5, True),
        _rectification(calibration, 1.0, False),
        _rectification(calibration, 1.0, True),
    ]
    matcher_configs = [
        {"name": "default_b5_u8_n256", "num_disparities": 256, "block_size": 5, "uniqueness_ratio": 8, "speckle_window_size": 100, "speckle_range": 2},
        {"name": "b3_u5_n256", "num_disparities": 256, "block_size": 3, "uniqueness_ratio": 5, "speckle_window_size": 100, "speckle_range": 2},
        {"name": "b7_u8_n256", "num_disparities": 256, "block_size": 7, "uniqueness_ratio": 8, "speckle_window_size": 100, "speckle_range": 2},
        {"name": "default_b5_u8_n512", "num_disparities": 512, "block_size": 5, "uniqueness_ratio": 8, "speckle_window_size": 100, "speckle_range": 2},
    ]
    started = time.monotonic()
    combinations: list[dict[str, Any]] = []
    for rectification in rectifications:
        for matcher_config in matcher_configs:
            print(
                f"\\n=== alpha={rectification['alpha']} centered={rectification['centered_projection']} "
                f"matcher={matcher_config['name']} f={rectification['focal_px']:.3f} ===",
                flush=True,
            )
            result = _run_combination(
                svo, calibration_path, rectification, matcher_config, positions
            )
            _fix_result_profile(result)
            combinations.append(result)
    report = {
        "script": Path(__file__).name,
        "svo": str(svo),
        "calibration": str(calibration_path),
        "sdk_version": str(sl.Camera.get_sdk_version()),
        "protocol": "fresh SVO handle per combination; raw unrectified images; one OpenCV remap; independent SGBM matcher",
        "positions": positions,
        "calibration_values": {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in calibration.items()
        },
        "interpretation": {
            "image_center": "output coordinate (960,540)",
            "optical_center": "rectified P1 principal point; may differ from image center",
            "desired_profile": "center_r025 < inner_mid_r025_050 < outer_mid_r050_075 < edge_r075",
            "depth_formula": "Z_m = rectified_P1_fx_px * rectified_baseline_m / disparity_px",
            "not_ground_truth": "SGBM profile agreement with a prior is not proof of physical metric accuracy",
        },
        "combinations": combinations,
        "elapsed_seconds": time.monotonic() - started,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\\nSaved report: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
