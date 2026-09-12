"""Standalone underwater stereo scale and geometry audit.

This diagnostic does not replace any production pipeline.  It opens two fresh
SVO handles and processes the same consecutive frames in two independent
image geometries:

* native: SVO ``VIEW.LEFT/RIGHT`` and the SVO embedded rectified calibration;
* custom: SVO ``VIEW.LEFT_UNRECTIFIED/RIGHT_UNRECTIFIED`` followed by one
  OpenCV ``stereoRectify``/``remap`` operation from ``Calibration/``.

For every frame it computes the four requested paraxial cases:

    Z_native                 = f_native * B_native / d_native
    Z_native_refractive      = n * Z_native
    Z_custom                 = f_custom * B_custom / d_custom
    Z_custom_refractive      = n * Z_custom

It also compares formula depth with ``cv2.reprojectImageTo3D`` and reports a
paraxial-versus-Snell correction without assuming that the calibration is a
physical refractive-camera model.  The script intentionally keeps all native
and custom disparities separate: they are measured in different rectified
image coordinate systems.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _prepare_windows_dll_search_path() -> None:
    if os.name != "nt":
        return
    sdk_root = os.environ.get("ZED_SDK_ROOT_DIR")
    if not sdk_root:
        return
    root = Path(sdk_root)
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
DEFAULT_INTRINSICS = ROOT / "Calibration" / "标定结果" / "camera_intrinsics.yaml"
DEFAULT_EXTRINSICS = ROOT / "Calibration" / "标定结果" / "stereo_extrinsics.yaml"
DEFAULT_OUTPUT = ROOT / "output" / "20260802_150233_refractive_depth_audit"


@dataclass(frozen=True)
class CustomCalibration:
    left_k: np.ndarray
    right_k: np.ndarray
    left_d: np.ndarray
    right_d: np.ndarray
    rotation: np.ndarray
    translation_mm: np.ndarray
    declared_baseline_mm: float
    reprojection_error_px: float
    scale_error_percent: float


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
    focal_px: float
    fy_px: float
    baseline_m: float
    q_length_scale_to_m: float
    alpha: float
    roi_left: tuple[int, int, int, int]
    roi_right: tuple[int, int, int, int]


CASES = (
    "Z_native",
    "Z_native_refractive",
    "Z_custom",
    "Z_custom_refractive",
)


def _status_name(status: Any) -> str:
    return str(status).rsplit(".", 1)[-1]


def _check_status(status: Any, operation: str) -> None:
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"{operation} failed: {_status_name(status)}")


def _matrix4(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise RuntimeError("non-finite 4x4 matrix")
    return matrix


def _camera_metadata(camera: Any) -> dict[str, Any]:
    return {
        "fx_px": float(camera.fx),
        "fy_px": float(camera.fy),
        "cx_px": float(camera.cx),
        "cy_px": float(camera.cy),
        "distortion": [
            float(value)
            for value in np.asarray(camera.disto, dtype=np.float64).reshape(-1)
        ],
        "lens_distortion_model": str(camera.lens_distortion_model),
    }


def _numbers(text: str) -> list[float]:
    pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    return [float(value) for value in re.findall(pattern, text)]


def _section(text: str, name: str, next_name: str | None = None) -> str:
    end = rf"^\s*{re.escape(next_name)}\s*:" if next_name else r"\Z"
    match = re.search(
        rf"(?ms)^\s*{re.escape(name)}\s*:\s*(.*?)(?={end})",
        text,
    )
    if match is None:
        raise RuntimeError(f"missing calibration section {name!r}")
    return match.group(1)


def _scalar(text: str, name: str) -> float:
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*:\s*([-+0-9.eE]+)", text)
    if match is None:
        raise RuntimeError(f"missing scalar {name!r}")
    return float(match.group(1))


def _list(text: str, name: str) -> list[float]:
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*:\s*\[([^\]]+)\]", text)
    if match is None:
        raise RuntimeError(f"missing list {name!r}")
    return _numbers(match.group(1))


def _load_custom_calibration(
    intrinsics_path: Path, extrinsics_path: Path
) -> CustomCalibration:
    intrinsics_text = intrinsics_path.read_text(encoding="utf-8-sig")
    extrinsics_text = extrinsics_path.read_text(encoding="utf-8-sig")

    def camera(name: str, next_name: str | None) -> tuple[np.ndarray, np.ndarray]:
        section = _section(intrinsics_text, name, next_name)
        fx, fy, cx, cy = (_scalar(section, key) for key in ("fx", "fy", "cx", "cy"))
        distortion = _list(section, "distortion_coefficients")
        if len(distortion) < 5:
            raise RuntimeError(f"{name} must contain at least five distortion coefficients")
        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        return k, np.asarray(distortion[:5], dtype=np.float64)

    left_k, left_d = camera("left", "right")
    right_k, right_d = camera("right", None)
    r_text = _section(extrinsics_text, "R", "T")
    rows = [_numbers(row) for row in re.findall(r"\[([^\]]+)\]", r_text)]
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise RuntimeError(f"expected a 3x3 R matrix, got {rows}")
    rotation = np.asarray(rows, dtype=np.float64)
    translation = np.asarray(_list(extrinsics_text, "T"), dtype=np.float64)
    if translation.size != 3:
        raise RuntimeError("expected a 3-vector T")
    return CustomCalibration(
        left_k=left_k,
        right_k=right_k,
        left_d=left_d,
        right_d=right_d,
        rotation=rotation,
        translation_mm=translation,
        declared_baseline_mm=_scalar(extrinsics_text, "baseline"),
        reprojection_error_px=_scalar(extrinsics_text, "reprojection_error"),
        scale_error_percent=_scalar(extrinsics_text, "scale_error_percent"),
    )


def _make_custom_rectification(
    calibration: CustomCalibration,
    input_size: tuple[int, int],
    output_size: tuple[int, int],
    alpha: float,
) -> Rectification:
    r_left, r_right, p_left, p_right, q, roi_left, roi_right = cv2.stereoRectify(
        calibration.left_k,
        calibration.left_d,
        calibration.right_k,
        calibration.right_d,
        input_size,
        calibration.rotation,
        calibration.translation_mm.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,
        newImageSize=output_size,
    )
    map_left = cv2.initUndistortRectifyMap(
        calibration.left_k,
        calibration.left_d,
        r_left,
        p_left,
        output_size,
        cv2.CV_32FC1,
    )
    map_right = cv2.initUndistortRectifyMap(
        calibration.right_k,
        calibration.right_d,
        r_right,
        p_right,
        output_size,
        cv2.CV_32FC1,
    )
    focal = float(p_left[0, 0])
    fy = float(p_left[1, 1])
    baseline_mm = abs(float(p_right[0, 3] / p_right[0, 0]))
    if focal <= 0.0 or fy <= 0.0 or baseline_mm <= 0.0:
        raise RuntimeError(f"invalid custom rectification: f={focal}, fy={fy}, B={baseline_mm}")
    return Rectification(
        r_left=r_left,
        r_right=r_right,
        p_left=p_left,
        p_right=p_right,
        q=q,
        left_map_x=map_left[0],
        left_map_y=map_left[1],
        right_map_x=map_right[0],
        right_map_y=map_right[1],
        focal_px=focal,
        fy_px=fy,
        baseline_m=baseline_mm / 1000.0,
        q_length_scale_to_m=0.001,
        alpha=alpha,
        roi_left=tuple(int(value) for value in roi_left),
        roi_right=tuple(int(value) for value in roi_right),
    )


def _make_native_projection(rectified: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    f = float(rectified.left_cam.fx)
    fy = float(rectified.left_cam.fy)
    cx = float(rectified.left_cam.cx)
    cy = float(rectified.left_cam.cy)
    transform = _matrix4(rectified.stereo_transform.m)
    baseline = abs(float(transform[0, 3]))
    if f <= 0.0 or fy <= 0.0 or baseline <= 0.0:
        raise RuntimeError(f"invalid native rectified calibration: f={f}, fy={fy}, B={baseline}")
    p1 = np.array([[f, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    p2 = p1.copy()
    p2[0, 3] = -f * baseline
    q = np.array(
        [[1.0, 0.0, 0.0, -cx], [0.0, 1.0, 0.0, -cy], [0.0, 0.0, 0.0, f], [0.0, 0.0, 1.0 / baseline, 0.0]],
        dtype=np.float64,
    )
    return p1, p2, q, f, baseline


def _as_gray(data: np.ndarray) -> np.ndarray:
    image = np.asarray(data)
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise RuntimeError(f"unexpected image shape: {image.shape}")
    return np.ascontiguousarray(image.astype(np.uint8, copy=False))


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
        mode=getattr(cv2, "STEREO_SGBM_MODE_SGBM_3WAY", cv2.STEREO_SGBM_MODE_SGBM),
    )


def _depth(disparity: np.ndarray, f: float, baseline_m: float) -> np.ndarray:
    result = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 1.0)
    result[valid] = (f * baseline_m / disparity[valid]).astype(np.float32)
    return result


def _stats(values: list[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array) & (array > 0.0)]
    result: dict[str, Any] = {
        "count": int(finite.size),
        "invalid_count": int(array.size - finite.size),
    }
    if finite.size:
        result.update(
            {
                "min": float(np.min(finite)),
                "p05": float(np.percentile(finite, 5)),
                "median": float(np.median(finite)),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "p95": float(np.percentile(finite, 95)),
                "max": float(np.max(finite)),
            }
        )
    return result


def _rotation_diagnostics(rotation: np.ndarray) -> dict[str, float]:
    return {
        "orthonormal_max_abs": float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
        "det_minus_one": float(np.linalg.det(rotation) - 1.0),
    }


def _region_definitions(
    width: int,
    height: int,
    native_p1: np.ndarray,
    custom_p1: np.ndarray,
) -> dict[str, tuple[int, int, int, int]]:
    def box(cx: float, cy: float, radius: int) -> tuple[int, int, int, int]:
        x = int(round(cx))
        y = int(round(cy))
        return (
            max(0, x - radius),
            min(width, x + radius + 1),
            max(0, y - radius),
            min(height, y + radius + 1),
        )

    return {
        "image_center_11x11": box(width / 2.0, height / 2.0, 5),
        "native_optical_center_11x11": box(native_p1[0, 2], native_p1[1, 2], 5),
        "custom_optical_center_11x11": box(custom_p1[0, 2], custom_p1[1, 2], 5),
        "central_40_percent": (
            int(round(width * 0.30)),
            int(round(width * 0.70)),
            int(round(height * 0.30)),
            int(round(height * 0.70)),
        ),
    }


def _new_region_accumulator(regions: dict[str, tuple[int, int, int, int]]) -> dict[str, Any]:
    return {
        name: {
            case: {"frame_medians": [], "frame_valid_counts": [], "total_valid_pixels": 0}
            for case in CASES
        }
        for name in regions
    }


def _record_regions(
    accumulator: dict[str, Any],
    regions: dict[str, tuple[int, int, int, int]],
    case_depths: dict[str, np.ndarray],
) -> None:
    for name, (x0, x1, y0, y1) in regions.items():
        for case, depth_map in case_depths.items():
            values = depth_map[y0:y1, x0:x1]
            valid = values[np.isfinite(values) & (values > 0.0)]
            entry = accumulator[name][case]
            entry["frame_valid_counts"].append(int(valid.size))
            entry["total_valid_pixels"] += int(valid.size)
            if valid.size:
                entry["frame_medians"].append(float(np.median(valid)))


def _finalize_regions(accumulator: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, cases in accumulator.items():
        result[name] = {}
        for case, entry in cases.items():
            frame_medians = entry["frame_medians"]
            frame_counts = np.asarray(entry["frame_valid_counts"], dtype=np.int64)
            result[name][case] = {
                "frame_median_stats_m": _stats(frame_medians),
                "frame_medians_m": [float(value) for value in frame_medians],
                "frame_count_with_valid_median": int(len(frame_medians)),
                "total_valid_pixels": int(entry["total_valid_pixels"]),
                "median_valid_pixels_per_frame": float(np.median(frame_counts)) if frame_counts.size else 0.0,
                "max_valid_pixels_per_frame": int(np.max(frame_counts)) if frame_counts.size else 0,
            }
    return result


def _ratio_stats(numerator: list[float], denominator: list[float]) -> dict[str, Any]:
    count = min(len(numerator), len(denominator))
    if count == 0:
        return {"count": 0}
    values = [
        float(a / b)
        for a, b in zip(numerator[:count], denominator[:count])
        if np.isfinite(a) and np.isfinite(b) and a > 0.0 and b > 0.0
    ]
    return _stats(values)


def _q_identity(
    p1: np.ndarray,
    p2: np.ndarray,
    q: np.ndarray,
    q_length_unit_to_m: float,
) -> dict[str, float]:
    f = float(p1[0, 0])
    b_unit = abs(float(p2[0, 3] / p2[0, 0]))
    return {
        "p1_fx_minus_fy_abs": float(abs(p1[0, 0] - p1[1, 1])),
        "p2_fx_minus_p1_fx_abs": float(abs(p2[0, 0] - p1[0, 0])),
        "p2_tx_plus_fx_baseline_abs_in_calibration_units": float(abs(p2[0, 3] + f * b_unit)),
        "baseline_from_p2_m": float(b_unit * q_length_unit_to_m),
        "q_23_minus_f_abs": float(abs(q[2, 3] - f)),
        "q_32_minus_inverse_baseline_abs": float(abs(q[3, 2] - 1.0 / b_unit)),
        "q_length_unit_to_m": float(q_length_unit_to_m),
    }


def _q_validation(
    disparity: np.ndarray,
    formula_depth_m: np.ndarray,
    q: np.ndarray,
    q_length_unit_to_m: float,
    rng: np.random.Generator,
    samples_per_frame: int,
    differences: list[float],
    relative_differences: list[float],
) -> None:
    xyz = cv2.reprojectImageTo3D(np.asarray(disparity, dtype=np.float32), q)
    q_depth_m = xyz[:, :, 2].astype(np.float64) * q_length_unit_to_m
    formula = formula_depth_m.astype(np.float64)
    valid = (
        np.isfinite(disparity)
        & (disparity > 1.0)
        & np.isfinite(formula)
        & (formula > 0.0)
        & np.isfinite(q_depth_m)
        & (q_depth_m > 0.0)
    )
    positions = np.flatnonzero(valid)
    if positions.size > samples_per_frame:
        positions = rng.choice(positions, size=samples_per_frame, replace=False)
    if positions.size == 0:
        return
    flat_formula = formula.reshape(-1)[positions]
    flat_q = q_depth_m.reshape(-1)[positions]
    diff = np.abs(flat_formula - flat_q)
    differences.extend(float(value) for value in diff)
    relative_differences.extend(
        float(value)
        for value in diff / np.maximum(np.abs(flat_formula), 1e-12)
    )


def _feature_correspondence_check(
    left: np.ndarray,
    right: np.ndarray,
    disparity: np.ndarray,
    focal_px: float,
    baseline_m: float,
    max_matches: int = 100,
) -> dict[str, Any]:
    orb = cv2.ORB_create(nfeatures=2000)
    key_left, desc_left = orb.detectAndCompute(left, None)
    key_right, desc_right = orb.detectAndCompute(right, None)
    if desc_left is None or desc_right is None:
        return {"status": "insufficient_features", "keypoints_left": len(key_left), "keypoints_right": len(key_right)}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc_left, desc_right, k=2)
    manual: list[float] = []
    sampled_sgbm: list[float] = []
    vertical: list[float] = []
    for pair in pairs:
        if len(pair) != 2 or pair[0].distance >= 0.75 * pair[1].distance:
            continue
        match = pair[0]
        ul, vl = key_left[match.queryIdx].pt
        ur, vr = key_right[match.trainIdx].pt
        d = ul - ur
        if d <= 1.0 or abs(vl - vr) > 2.0:
            continue
        x = int(round(ul))
        y = int(round(vl))
        if not (0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]):
            continue
        sgbm_d = float(disparity[y, x])
        if not np.isfinite(sgbm_d) or sgbm_d <= 1.0:
            continue
        manual.append(float(d))
        sampled_sgbm.append(sgbm_d)
        vertical.append(float(abs(vl - vr)))
    if not manual:
        return {
            "status": "no_valid_rectified_matches",
            "keypoints_left": len(key_left),
            "keypoints_right": len(key_right),
            "ratio_test_matches": len(pairs),
        }
    manual_array = np.asarray(manual[:max_matches], dtype=np.float64)
    sgbm_array = np.asarray(sampled_sgbm[:max_matches], dtype=np.float64)
    manual_depth = focal_px * baseline_m / manual_array
    sgbm_depth = focal_px * baseline_m / sgbm_array
    return {
        "status": "ok",
        "keypoints_left": len(key_left),
        "keypoints_right": len(key_right),
        "ratio_test_matches": len(pairs),
        "valid_matches_used": int(manual_array.size),
        "manual_disparity_definition": "d_px = u_left - u_right",
        "manual_disparity_stats_px": _stats(manual_array),
        "sgbm_disparity_at_left_keypoint_stats_px": _stats(sgbm_array),
        "vertical_error_stats_px": _stats(np.asarray(vertical[:max_matches], dtype=np.float64)),
        "manual_depth_stats_m": _stats(manual_depth),
        "sgbm_depth_at_same_keypoint_stats_m": _stats(sgbm_depth),
        "median_manual_minus_sgbm_disparity_px": float(np.median(manual_array - sgbm_array)),
        "median_manual_minus_sgbm_depth_m": float(np.median(manual_depth - sgbm_depth)),
    }


def _snell_summary(
    f_px: float,
    baseline_m: float,
    disparity_values: list[float],
    n: float,
) -> dict[str, Any]:
    d = np.asarray(disparity_values, dtype=np.float64)
    d = d[np.isfinite(d) & (d > 1.0)]
    if d.size == 0:
        return {"status": "no_valid_disparities", "n": n}
    factor = np.sqrt(n * n + (n * n - 1.0) * (d / (2.0 * f_px)) ** 2)
    return {
        "status": "ok",
        "n": float(n),
        "paraxial_factor": float(n),
        "disparity_stats_px": _stats(d),
        "exact_snell_factor_stats": _stats(factor),
        "exact_over_paraxial_percent_stats": _stats(100.0 * factor / n),
        "exact_minus_paraxial_percent_median": float(100.0 * (np.median(factor) / n - 1.0)),
        "formula_without_h": (
            "Z_snell = (f*B/d) * sqrt(n^2 + (n^2-1)*(d/(2f))^2); "
            "camera-center-to-port distance h is not supplied"
        ),
        "unknown_parameters": ["camera_to_flat_port_distance_h", "glass_thickness", "glass_refractive_index"],
    }


def _open_svo(svo_path: Path) -> sl.Camera:
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.METER
    init.camera_disable_self_calib = True
    zed = sl.Camera()
    _check_status(zed.open(init), f"open SVO {svo_path}")
    return zed


def _run_native(
    svo_path: Path,
    frame_count: int,
    n: float,
    regions: dict[str, tuple[int, int, int, int]],
    rng: np.random.Generator,
    q_differences: list[float],
    q_relative_differences: list[float],
) -> dict[str, Any]:
    zed = _open_svo(svo_path)
    try:
        info = zed.get_camera_information()
        configuration = info.camera_configuration
        resolution = configuration.resolution
        width, height = int(resolution.width), int(resolution.height)
        total_frames = int(zed.get_svo_number_of_frames())
        if frame_count > total_frames:
            raise ValueError(f"requested {frame_count} frames but SVO has {total_frames}")
        raw = configuration.calibration_parameters_raw
        rectified = configuration.calibration_parameters
        p1, p2, q, focal, baseline = _make_native_projection(rectified)
        accumulator = _new_region_accumulator(regions)
        matcher = _make_matcher()
        runtime = sl.RuntimeParameters()
        left_mat, right_mat = sl.Mat(), sl.Mat()
        disparity_values: list[float] = []
        center_disparities: list[float] = []
        first_feature_check: dict[str, Any] | None = None
        for frame_index in range(frame_count):
            _check_status(zed.grab(runtime), f"native grab frame {frame_index}")
            _check_status(zed.retrieve_image(left_mat, sl.VIEW.LEFT), "native retrieve left")
            _check_status(zed.retrieve_image(right_mat, sl.VIEW.RIGHT), "native retrieve right")
            left = _as_gray(left_mat.get_data())
            right = _as_gray(right_mat.get_data())
            if left.shape != (height, width) or right.shape != left.shape:
                raise RuntimeError(f"native image size mismatch: {left.shape}, {right.shape}")
            disparity = matcher.compute(left, right).astype(np.float32) / 16.0
            depth_native = _depth(disparity, focal, baseline)
            depth_native_refractive = depth_native * np.float32(n)
            case_depths = {
                "Z_native": depth_native,
                "Z_native_refractive": depth_native_refractive,
                "Z_custom": np.full_like(depth_native, np.nan),
                "Z_custom_refractive": np.full_like(depth_native, np.nan),
            }
            _record_regions(accumulator, regions, case_depths)
            valid_disparity = disparity[np.isfinite(disparity) & (disparity > 1.0)]
            if valid_disparity.size:
                chosen = valid_disparity
                if chosen.size > 5000:
                    chosen = rng.choice(chosen, size=5000, replace=False)
                disparity_values.extend(float(value) for value in chosen)
            center_disparities.append(float(disparity[height // 2, width // 2]))
            _q_validation(
                disparity,
                depth_native,
                q,
                1.0,
                rng,
                5000,
                q_differences,
                q_relative_differences,
            )
            if frame_index == 0:
                first_feature_check = _feature_correspondence_check(
                    left, right, disparity, focal, baseline
                )
        region_result = _finalize_regions(accumulator)
        for values in region_result.values():
            values["Z_custom"] = {"status": "not applicable in native image geometry"}
            values["Z_custom_refractive"] = {"status": "not applicable in native image geometry"}
        return {
            "image_source": "VIEW.LEFT + VIEW.RIGHT; SVO-native rectified coordinates",
            "parameter_source": "SVO embedded calibration, read with no external calibration override",
            "frames_processed": frame_count,
            "total_svo_frames": total_frames,
            "image_size": {"width": width, "height": height},
            "raw_calibration": {"left": _camera_metadata(raw.left_cam), "right": _camera_metadata(raw.right_cam), "stereo_transform_m": _matrix4(raw.stereo_transform.m).tolist()},
            "rectified_calibration": {"left": _camera_metadata(rectified.left_cam), "right": _camera_metadata(rectified.right_cam), "stereo_transform_m": _matrix4(rectified.stereo_transform.m).tolist()},
            "P1": p1.tolist(),
            "P2": p2.tolist(),
            "Q": q.tolist(),
            "focal_length_px": focal,
            "fy_rectified_px": float(rectified.left_cam.fy),
            "baseline_m": baseline,
            "baseline_definition": "abs(native rectified stereo_transform[0,3]); horizontal rectified baseline",
            "depth_formula": "Z_native_m = f_native_rect_px * B_native_m / d_native_px",
            "disparity_units": "raw StereoSGBM int16 fixed-point / 16.0; positive d = u_left - u_right",
            "center_disparity_stats_px": _stats(center_disparities),
            "all_valid_disparity_sample_stats_px": _stats(disparity_values),
            "region_results": region_result,
            "snell_summary": _snell_summary(focal, baseline, disparity_values, n),
            "feature_correspondence_check_frame_0": first_feature_check,
            "q_identity": _q_identity(p1, p2, q, 1.0),
            "rectification": {
                "flags": "CALIB_ZERO_DISPARITY",
                "alpha": "native SDK value; no second rectification applied",
                "input_image_size": [width, height],
                "output_image_size": [width, height],
                "T_units": "metres in SVO SDK transform",
            },
        }
    finally:
        zed.close()


def _run_custom(
    svo_path: Path,
    rectification: Rectification,
    frame_count: int,
    n: float,
    width: int,
    height: int,
    regions: dict[str, tuple[int, int, int, int]],
    rng: np.random.Generator,
    q_differences: list[float],
    q_relative_differences: list[float],
) -> dict[str, Any]:
    zed = _open_svo(svo_path)
    try:
        total_frames = int(zed.get_svo_number_of_frames())
        if frame_count > total_frames:
            raise ValueError(f"requested {frame_count} frames but SVO has {total_frames}")
        accumulator = _new_region_accumulator(regions)
        matcher = _make_matcher()
        runtime = sl.RuntimeParameters()
        left_mat, right_mat = sl.Mat(), sl.Mat()
        disparity_values: list[float] = []
        center_disparities: list[float] = []
        first_feature_check: dict[str, Any] | None = None
        for frame_index in range(frame_count):
            _check_status(zed.grab(runtime), f"custom grab frame {frame_index}")
            _check_status(
                zed.retrieve_image(left_mat, sl.VIEW.LEFT_UNRECTIFIED),
                "custom retrieve unrectified left",
            )
            _check_status(
                zed.retrieve_image(right_mat, sl.VIEW.RIGHT_UNRECTIFIED),
                "custom retrieve unrectified right",
            )
            left_raw = _as_gray(left_mat.get_data())
            right_raw = _as_gray(right_mat.get_data())
            expected_shape = (height, width)
            if left_raw.shape != expected_shape or right_raw.shape != expected_shape:
                raise RuntimeError(f"custom raw image size mismatch: {left_raw.shape}, {right_raw.shape}")
            left = cv2.remap(left_raw, rectification.left_map_x, rectification.left_map_y, cv2.INTER_LINEAR)
            right = cv2.remap(right_raw, rectification.right_map_x, rectification.right_map_y, cv2.INTER_LINEAR)
            disparity = matcher.compute(left, right).astype(np.float32) / 16.0
            depth_custom = _depth(disparity, rectification.focal_px, rectification.baseline_m)
            depth_custom_refractive = depth_custom * np.float32(n)
            case_depths = {
                "Z_native": np.full_like(depth_custom, np.nan),
                "Z_native_refractive": np.full_like(depth_custom, np.nan),
                "Z_custom": depth_custom,
                "Z_custom_refractive": depth_custom_refractive,
            }
            _record_regions(accumulator, regions, case_depths)
            valid_disparity = disparity[np.isfinite(disparity) & (disparity > 1.0)]
            if valid_disparity.size:
                chosen = valid_disparity
                if chosen.size > 5000:
                    chosen = rng.choice(chosen, size=5000, replace=False)
                disparity_values.extend(float(value) for value in chosen)
            center_disparities.append(float(disparity[height // 2, width // 2]))
            _q_validation(
                disparity,
                depth_custom,
                rectification.q,
                rectification.q_length_scale_to_m,
                rng,
                5000,
                q_differences,
                q_relative_differences,
            )
            if frame_index == 0:
                first_feature_check = _feature_correspondence_check(
                    left, right, disparity, rectification.focal_px, rectification.baseline_m
                )
        region_result = _finalize_regions(accumulator)
        for values in region_result.values():
            values["Z_native"] = {"status": "not applicable in custom image geometry"}
            values["Z_native_refractive"] = {"status": "not applicable in custom image geometry"}
        return {
            "image_source": "VIEW.LEFT_UNRECTIFIED + VIEW.RIGHT_UNRECTIFIED; one OpenCV remap into custom rectified coordinates",
            "parameter_source": "camera_intrinsics.yaml + stereo_extrinsics.yaml parsed independently",
            "frames_processed": frame_count,
            "total_svo_frames": total_frames,
            "image_size": {"width": width, "height": height},
            "rectification_alpha": rectification.alpha,
            "R1": rectification.r_left.tolist(),
            "R2": rectification.r_right.tolist(),
            "P1": rectification.p_left.tolist(),
            "P2": rectification.p_right.tolist(),
            "Q": rectification.q.tolist(),
            "valid_roi_left": list(rectification.roi_left),
            "valid_roi_right": list(rectification.roi_right),
            "focal_length_px": rectification.focal_px,
            "fy_rectified_px": rectification.fy_px,
            "baseline_m": rectification.baseline_m,
            "baseline_definition": "abs(P2[0,3] / P2[0,0]); OpenCV rectified baseline; T and P units are mm before conversion",
            "depth_formula": "Z_custom_m = P1[0,0]_px * (abs(P2[0,3]/P2[0,0])/1000)_m / d_custom_px",
            "disparity_units": "raw StereoSGBM int16 fixed-point / 16.0; positive d = u_left - u_right",
            "center_disparity_stats_px": _stats(center_disparities),
            "all_valid_disparity_sample_stats_px": _stats(disparity_values),
            "region_results": region_result,
            "snell_summary": _snell_summary(rectification.focal_px, rectification.baseline_m, disparity_values, n),
            "feature_correspondence_check_frame_0": first_feature_check,
            "q_identity": _q_identity(rectification.p_left, rectification.p_right, rectification.q, rectification.q_length_scale_to_m),
            "rectification": {
                "flags": "CALIB_ZERO_DISPARITY",
                "alpha": rectification.alpha,
                "input_image_size": [1920, 1080],
                "output_image_size": [width, height],
                "R_direction_assumed_by_this_script": "source R/T supplied directly as left-to-right to OpenCV stereoRectify",
                "T_units": "millimetres in calibration files and OpenCV P/Q; converted to metres only in depth formula",
                "raw_source_views": ["VIEW.LEFT_UNRECTIFIED", "VIEW.RIGHT_UNRECTIFIED"],
                "rectification_count": 1,
            },
        }
    finally:
        zed.close()


def _add_cross_branch_ratios(result: dict[str, Any]) -> None:
    native_regions = result["native"]["region_results"]
    custom_regions = result["custom"]["region_results"]
    ratios: dict[str, Any] = {}
    region_pairs = {
        "optical_center_corresponding": (
            "native_optical_center_11x11",
            "custom_optical_center_11x11",
        ),
        "image_center_same_output_coordinates": (
            "image_center_11x11",
            "image_center_11x11",
        ),
        "central_area_same_output_coordinates": (
            "central_40_percent",
            "central_40_percent",
        ),
    }
    for name, (native_name, custom_name) in region_pairs.items():
        native_data = native_regions[native_name]
        custom_data = custom_regions[custom_name]

        def pair_ratio(native_case: str, custom_case: str) -> dict[str, Any]:
            native_values = native_data[native_case].get("frame_medians_m", [])
            custom_values = custom_data[custom_case].get("frame_medians_m", [])
            paired = [
                float(custom_value / native_value)
                for custom_value, native_value in zip(custom_values, native_values)
                if np.isfinite(custom_value)
                and np.isfinite(native_value)
                and custom_value > 0.0
                and native_value > 0.0
            ]
            return _stats(paired)

        ratios[name] = {
            "native_region": native_name,
            "custom_region": custom_name,
            "note": "Z3/Z2 is computed from per-frame medians in each pipeline's own rectified coordinate system; native and custom disparities are never mixed",
            "Z3_over_Z2": pair_ratio("Z_native_refractive", "Z_custom"),
            "Z3_over_Z1": pair_ratio("Z_native", "Z_custom"),
            "Z4_over_Z2": pair_ratio("Z_native_refractive", "Z_custom_refractive"),
        }
    result["cross_branch_comparison"] = ratios


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("svo", nargs="?", type=Path, default=DEFAULT_SVO)
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--extrinsics", type=Path, default=DEFAULT_EXTRINSICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--rectify-alpha", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--n-water", type=float, default=1.333)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if not 0.0 <= args.rectify_alpha <= 1.0:
        parser.error("--rectify-alpha must be in [0, 1]")
    if not 0.0 < args.scale <= 1.0:
        parser.error("--scale must be in (0, 1]")
    if args.n_water <= 1.0 or not math.isfinite(args.n_water):
        parser.error("--n-water must be finite and > 1")
    return args


def main() -> int:
    args = _parse_args()
    svo_path = args.svo.expanduser().resolve()
    intrinsics_path = args.intrinsics.expanduser().resolve()
    extrinsics_path = args.extrinsics.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path, label in (
        (svo_path, "SVO2"),
        (intrinsics_path, "custom intrinsics"),
        (extrinsics_path, "custom extrinsics"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "refractive_depth_audit.json"
    if output_path.exists() and not args.overwrite:
        raise RuntimeError(f"output exists; use --overwrite or another directory: {output_path}")

    started = time.monotonic()
    calibration = _load_custom_calibration(intrinsics_path, extrinsics_path)
    rotation_diag = _rotation_diagnostics(calibration.rotation)
    if rotation_diag["orthonormal_max_abs"] > 1e-5 or abs(rotation_diag["det_minus_one"]) > 1e-5:
        raise RuntimeError(f"custom R failed rotation diagnostics: {rotation_diag}")
    if np.any(np.diag(calibration.left_k)[:2] <= 0) or np.any(np.diag(calibration.right_k)[:2] <= 0):
        raise RuntimeError("custom focal lengths are not positive")

    # Probe native resolution once.  This is intentionally a fresh handle and
    # uses the embedded calibration only for geometry metadata.
    probe = _open_svo(svo_path)
    try:
        info = probe.get_camera_information()
        resolution = info.camera_configuration.resolution
        width = int(resolution.width)
        height = int(resolution.height)
        total_frames = int(probe.get_svo_number_of_frames())
        native_rectified = info.camera_configuration.calibration_parameters
        native_p1 = np.array(
            [[float(native_rectified.left_cam.fx), 0.0, float(native_rectified.left_cam.cx)],
             [0.0, float(native_rectified.left_cam.fy), float(native_rectified.left_cam.cy)],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    finally:
        probe.close()
    if calibration.left_k[0, 2] <= 0 or calibration.left_k[1, 2] <= 0:
        raise RuntimeError("custom principal point is invalid")
    for side, matrix in (("left", calibration.left_k), ("right", calibration.right_k)):
        cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
        if not (0.0 <= cx < 1920.0 and 0.0 <= cy < 1080.0):
            raise RuntimeError(f"custom {side} principal point is outside 1920x1080: ({cx}, {cy})")
    if (width, height) != (1920, 1080):
        raise RuntimeError(f"unexpected SVO resolution for this calibration: {(width, height)}")
    if args.scale != 1.0:
        output_size = (max(2, int(round(width * args.scale))), max(2, int(round(height * args.scale))))
        native_p1 = native_p1.copy()
        native_p1[0, :] *= args.scale
        native_p1[1, :] *= args.scale
        native_p1[2, 2] = 1.0
        width, height = output_size
    else:
        output_size = (width, height)
    # The current audit is run at full resolution by default.  If a scaled
    # run is requested, native images are resized explicitly and custom maps
    # target the same output size.
    custom_rectification = _make_custom_rectification(
        calibration,
        input_size=(1920, 1080),
        output_size=output_size,
        alpha=args.rectify_alpha,
    )
    regions = _region_definitions(width, height, native_p1, custom_rectification.p_left)
    rng = np.random.default_rng(args.seed)
    q_differences: list[float] = []
    q_relative_differences: list[float] = []

    # The full-resolution default is the requested same-frames experiment.
    # For a scaled run, the native branch is resized after retrieval by the
    # helper below; the custom branch maps directly to output_size.
    # Keep the default path strict and fail rather than silently resizing.
    if args.scale != 1.0:
        raise RuntimeError("scaled runs are intentionally unsupported in this audit; use --scale 1")

    native_result = _run_native(
        svo_path,
        min(args.frames, total_frames),
        args.n_water,
        regions,
        rng,
        q_differences,
        q_relative_differences,
    )
    custom_result = _run_custom(
        svo_path,
        custom_rectification,
        min(args.frames, total_frames),
        args.n_water,
        width,
        height,
        regions,
        rng,
        q_differences,
        q_relative_differences,
    )

    result: dict[str, Any] = {
        "script": Path(__file__).name,
        "protocol": "same first N SVO frames; native and custom branches use independent fresh SVO handles and never share disparity",
        "input": {
            "svo": str(svo_path),
            "intrinsics": str(intrinsics_path),
            "extrinsics": str(extrinsics_path),
            "sdk_version": str(sl.Camera.get_sdk_version()),
            "camera_model": _status_name(info.camera_model),
            "serial_number": int(info.serial_number),
            "image_size": {"width": width, "height": height},
            "total_svo_frames": total_frames,
            "frames_requested": args.frames,
            "frames_processed_per_branch": min(args.frames, total_frames),
        },
        "refractive_model": {
            "n_water_over_n_air": args.n_water,
            "Z_native": "f_native_rect_px * B_native_m / d_native_px",
            "Z_native_refractive": "n * Z_native",
            "Z_custom": "f_custom_rect_px * B_custom_rect_m / d_custom_px",
            "Z_custom_refractive": "n * Z_custom",
            "snell_model": "(fB/d) * sqrt(n^2 + (n^2-1)*(d/(2f))^2)",
            "snell_h_model": "not evaluated: h, glass thickness, and glass refractive index are not supplied",
        },
        "custom_calibration": {
            "left_K": calibration.left_k.tolist(),
            "right_K": calibration.right_k.tolist(),
            "left_D": calibration.left_d.tolist(),
            "right_D": calibration.right_d.tolist(),
            "R": calibration.rotation.tolist(),
            "T_mm": calibration.translation_mm.tolist(),
            "declared_baseline_mm": calibration.declared_baseline_mm,
            "T_x_abs_mm": float(abs(calibration.translation_mm[0])),
            "T_norm_mm": float(np.linalg.norm(calibration.translation_mm)),
            "reprojection_error_px": calibration.reprojection_error_px,
            "scale_error_percent_relative_to_120mm_comment": calibration.scale_error_percent,
            "rotation_diagnostics": rotation_diag,
        },
        "native": native_result,
        "custom": custom_result,
        "q_validation": {
            "sample_count": len(q_differences),
            "sample_count_required": 1000,
            "max_absolute_difference_m": float(max(q_differences)) if q_differences else None,
            "mean_absolute_difference_m": float(np.mean(q_differences)) if q_differences else None,
            "max_relative_difference": float(max(q_relative_differences)) if q_relative_differences else None,
            "mean_relative_difference": float(np.mean(q_relative_differences)) if q_relative_differences else None,
            "status": "PASS" if len(q_differences) >= 1000 and q_differences and max(q_differences) < 1e-5 else "INSUFFICIENT_OR_FAILED",
            "note": "Native Q uses metre baseline; custom OpenCV Q uses millimetre T and is converted by 0.001 before comparison.",
        },
        "baseline_comparison": {
            "B_nominal_mm": 120.0,
            "B_declared_mm": calibration.declared_baseline_mm,
            "B_Tx_mm": float(abs(calibration.translation_mm[0])),
            "B_norm_mm": float(np.linalg.norm(calibration.translation_mm)),
            "B_rectified_mm": float(abs(custom_rectification.p_right[0, 3] / custom_rectification.p_right[0, 0])),
            "declared_minus_nominal_percent": float(100.0 * (calibration.declared_baseline_mm / 120.0 - 1.0)),
            "norm_minus_nominal_percent": float(100.0 * (np.linalg.norm(calibration.translation_mm) / 120.0 - 1.0)),
            "norm_minus_declared_percent": float(100.0 * (np.linalg.norm(calibration.translation_mm) / calibration.declared_baseline_mm - 1.0)),
            "rectified_minus_norm_mm": float(abs(custom_rectification.p_right[0, 3] / custom_rectification.p_right[0, 0]) - np.linalg.norm(calibration.translation_mm)),
            "Tz_mm": float(calibration.translation_mm[2]),
            "baseline_angle_from_x_deg": float(math.degrees(math.atan2(math.hypot(calibration.translation_mm[1], calibration.translation_mm[2]), abs(calibration.translation_mm[0])))),
        },
        "validation_checks": {
            "positive_focal_lengths": True,
            "principal_points_inside_1920x1080": True,
            "rotation_orthonormal_and_det_one": rotation_diag["orthonormal_max_abs"] <= 1e-5 and abs(rotation_diag["det_minus_one"]) <= 1e-5,
            "translation_norm_reasonable_range_10_to_500_mm": 10.0 < float(np.linalg.norm(calibration.translation_mm)) < 500.0,
            "declared_tx_norm_rectified_baseline_logged_separately": True,
            "image_size_consistent": (width, height) == (1920, 1080),
            "resize_scale": "1.0; no implicit resize",
            "disparity_fixed_point_division": "/16.0 explicitly applied in both branches",
            "positive_depth_filter": "d > 1 px and finite; non-positive depth is invalid",
            "q_formula_consistency": "see q_validation and per-branch q_identity",
            "duplicate_rectification": "none: native uses SDK rectified views; custom uses raw unrectified views plus one remap",
            "calibration_source_logged": True,
            "raw_vs_rectified_image_type_logged": True,
            "silent_fallback": "none; missing or invalid calibration raises",
        },
        "runtime_seconds": time.monotonic() - started,
    }
    _add_cross_branch_ratios(result)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Native rectified f/B: {native_result['focal_length_px']:.9f} px / {native_result['baseline_m']:.9f} m")
    print(f"Custom rectified f/B: {custom_result['focal_length_px']:.9f} px / {custom_result['baseline_m']:.9f} m")
    print(f"n = {args.n_water:.6f}; frames per branch = {min(args.frames, total_frames)}")
    print(f"Q samples: {len(q_differences)}; max abs difference = {max(q_differences) if q_differences else None} m")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
