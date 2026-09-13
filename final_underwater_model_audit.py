"""Reproducible final audit of native, empirical, and refractive candidates.

This file is diagnostic-only.  It reads raw SVO stereo images and the two
calibration sources, then writes small machine-readable artifacts for the
final report.  It never changes a production depth or SLAM path.

The three candidates are kept separate:

* A: SVO native raw K/D/T, a best-available physical/dry candidate;
* B: external custom raw K/D/T, an empirical pinhole candidate;
* R1: native raw K/D/T plus a measured rig-level port model.  R1 is left
  explicitly NOT_IDENTIFIABLE when the installed housing is undocumented.

The legacy custom-K/T-plus-Snell calculation is generated only by the older
diagnostic and is labelled as a double-counting control, never as physical
underwater depth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pyzed.sl as sl

from debug_refractive_depth_check import (
    CustomCalibration,
    Rectification,
    _as_gray,
    _check_status,
    _load_custom_calibration,
    _make_custom_rectification,
)
from native_refractive_depth_check import collect_native_metadata
from refractive_geometry import pixel_to_air_ray, refract_vector
from rig_refractive_geometry import (
    RigCameraPose,
    RigFlatPortModel,
    trace_rig_pixel,
    triangulate_rig_water_rays,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_SVO = ROOT / "20260802_150233.svo2"
DEFAULT_OUTPUT = ROOT / "output" / "20260802_150233_final_underwater_audit"
N_WATER_VALUES = (1.330, 1.333, 1.336, 1.340)
N_GLASS_VALUES = (1.47, 1.50, 1.52)
H_VALUES_M = (0.005, 0.010, 0.020, 0.030)
THICKNESS_VALUES_M = (0.003, 0.005, 0.008, 0.010)


def _find_calibration_file(name: str) -> Path:
    matches = sorted((ROOT / "Calibration").rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one Calibration/{name}, found {matches}")
    return matches[0]


def _camera_matrix_from_metadata(camera: dict[str, Any]) -> np.ndarray:
    return np.array(
        [[float(camera["fx_px"]), 0.0, float(camera["cx_px"])],
         [0.0, float(camera["fy_px"]), float(camera["cy_px"])],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _camera_matrix_from_object(camera: Any) -> np.ndarray:
    return np.array(
        [[float(camera.fx), 0.0, float(camera.cx)],
         [0.0, float(camera.fy), float(camera.cy)],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _camera_distortion_from_metadata(camera: dict[str, Any]) -> np.ndarray:
    return np.asarray(camera["distortion"], dtype=np.float64).reshape(-1)


def _finite_or_none(value: float | int | np.floating | None) -> float | int | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _rotation_angle_and_vector(rotation: np.ndarray) -> tuple[float, list[float]]:
    vector, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))
    vector = vector.reshape(3)
    return float(np.degrees(np.linalg.norm(vector))), vector.tolist()


def _distortion_radial_mapping_audit(
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    width: int,
    height: int,
    *,
    samples: int = 101,
    angles: int = 16,
) -> dict[str, Any]:
    """Check sampled radial distortion monotonicity without claiming a physical model."""

    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    d = np.asarray(distortion, dtype=np.float64).reshape(-1, 1)
    cx, cy = float(k[0, 2]), float(k[1, 2])
    fx, fy = float(k[0, 0]), float(k[1, 1])
    max_radius = max(
        math.hypot((u - cx) / fx, (v - cy) / fy)
        for u, v in ((0.0, 0.0), (width - 1.0, 0.0), (0.0, height - 1.0), (width - 1.0, height - 1.0))
    )
    radius_values = np.linspace(0.0, max_radius, samples)
    negative_derivatives = 0
    nonfinite = 0
    maximum_displacement = 0.0
    derivative_values: list[float] = []
    for angle in np.linspace(0.0, 2.0 * math.pi, angles, endpoint=False):
        normalized = np.column_stack((radius_values * math.cos(angle), radius_values * math.sin(angle), np.ones_like(radius_values)))
        try:
            projected, _ = cv2.projectPoints(normalized, np.zeros(3), np.zeros(3), k, d)
        except cv2.error:
            return {"status": "ERROR", "reason": "OpenCV projectPoints rejected distortion vector"}
        pixels = projected.reshape(-1, 2)
        radial_pixels = np.linalg.norm(pixels - np.array([cx, cy]), axis=1)
        if not np.isfinite(radial_pixels).all():
            nonfinite += int(np.count_nonzero(~np.isfinite(radial_pixels)))
            continue
        displacement = np.abs(radial_pixels - radius_values * 0.5 * (fx + fy))
        maximum_displacement = max(maximum_displacement, float(np.max(displacement)))
        derivative = np.diff(radial_pixels)
        derivative_values.extend(derivative.tolist())
        negative_derivatives += int(np.count_nonzero(derivative < -1.0e-6))
    return {
        "status": "PASS" if negative_derivatives == 0 and nonfinite == 0 else "WARNING",
        "normalized_corner_radius_max": float(max_radius),
        "sample_count_per_angle": samples,
        "angle_count": angles,
        "negative_radial_steps": negative_derivatives,
        "nonfinite_projected_samples": nonfinite,
        "maximum_radial_displacement_px": maximum_displacement,
        "foldover_definition": "sampled negative radial derivative or non-finite projection; this is a diagnostic, not proof of global invertibility",
        "minimum_sampled_radial_derivative_px_per_step": float(min(derivative_values)) if derivative_values else None,
    }


def _rectified_projection_from_metadata(native: dict[str, Any]) -> dict[str, Any]:
    left = native["rectified_reference"]["left"]
    right = native["rectified_reference"]["right"]
    transform = np.asarray(native["rectified_reference"]["stereo_transform_m"], dtype=np.float64)
    focal = float(left["fx_px"])
    fy = float(left["fy_px"])
    cx = float(left["cx_px"])
    cy = float(left["cy_px"])
    baseline = abs(float(transform[0, 3]))
    p1 = np.array([[focal, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    p2 = np.array(
        [[float(right["fx_px"]), 0.0, float(right["cx_px"]), -float(right["fx_px"]) * baseline],
         [0.0, float(right["fy_px"]), float(right["cy_px"]), 0.0],
         [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    q = np.array(
        [[1.0, 0.0, 0.0, -cx], [0.0, 1.0, 0.0, -cy], [0.0, 0.0, 0.0, focal], [0.0, 0.0, 1.0 / baseline, 0.0]],
        dtype=np.float64,
    )
    return {
        "P1": p1,
        "P2": p2,
        "Q": q,
        "focal_px": focal,
        "fy_px": fy,
        "baseline_m": baseline,
    }


def _rotation_diagnostics(rotation: np.ndarray) -> dict[str, float]:
    return {
        "orthonormal_max_abs": float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
        "det_minus_one": float(np.linalg.det(rotation) - 1.0),
    }


def _baseline_entry(
    name: str,
    rotation_left_to_right: np.ndarray,
    translation_left_to_right_m: np.ndarray,
    *,
    declared_mm: float | None = None,
    rectified_m: float | None = None,
) -> dict[str, Any]:
    rotation = np.asarray(rotation_left_to_right, dtype=np.float64)
    translation = np.asarray(translation_left_to_right_m, dtype=np.float64).reshape(3)
    tilt = math.degrees(math.atan2(float(np.linalg.norm(translation[1:])), abs(float(translation[0]))))
    return {
        "name": name,
        "convention": "X_right = R_left_to_right X_left + T_left_to_right",
        "R_left_to_right": rotation.tolist(),
        "T_left_to_right_m": translation.tolist(),
        "T_left_to_right_mm": (1000.0 * translation).tolist(),
        "T_x_abs_m": float(abs(translation[0])),
        "T_y_m": float(translation[1]),
        "T_z_m": float(translation[2]),
        "T_norm_m": float(np.linalg.norm(translation)),
        "tilt_from_baseline_axis_deg": float(tilt),
        "declared_baseline_mm": declared_mm,
        "rectified_baseline_m": rectified_m,
    }


def _build_calibration_report(
    native: dict[str, Any],
    custom: CustomCalibration,
    custom_rect0: Rectification,
    custom_rect1: Rectification,
    input_size: tuple[int, int],
    svo_path: Path,
    intrinsics_path: Path,
    extrinsics_path: Path,
) -> dict[str, Any]:
    native_raw_left = _camera_matrix_from_metadata(native["raw"]["left"])
    native_raw_right = _camera_matrix_from_metadata(native["raw"]["right"])
    native_raw_transform = np.asarray(native["raw"]["stereo_transform_right_to_left_m"], dtype=np.float64)
    native_r_rl = native_raw_transform[:3, :3]
    native_t_rl = native_raw_transform[:3, 3]
    native_r_lr = native_r_rl.T
    native_t_lr = -native_r_lr @ native_t_rl
    custom_r_lr = np.asarray(custom.rotation, dtype=np.float64)
    custom_t_lr = np.asarray(custom.translation_mm, dtype=np.float64).reshape(3) / 1000.0
    native_rect = _rectified_projection_from_metadata(native)
    rotation_delta = custom_r_lr @ native_r_lr.T
    rotation_delta_angle, rotation_delta_vector = _rotation_angle_and_vector(rotation_delta)
    delta_t = custom_t_lr - native_t_lr

    focal_ratios: list[dict[str, Any]] = []
    for coordinate_system, native_cameras, custom_cameras in (
        ("raw", native["raw"], {"left": {"fx_px": custom.left_k[0, 0], "fy_px": custom.left_k[1, 1]}, "right": {"fx_px": custom.right_k[0, 0], "fy_px": custom.right_k[1, 1]}}),
        ("rectified_alpha1", native["rectified_reference"], {"left": {"fx_px": custom_rect1.p_left[0, 0], "fy_px": custom_rect1.p_left[1, 1]}, "right": {"fx_px": custom_rect1.p_right[0, 0], "fy_px": custom_rect1.p_right[1, 1]}}),
    ):
        for side in ("left", "right"):
            for axis in ("fx_px", "fy_px"):
                native_f = float(native_cameras[side][axis])
                custom_f = float(custom_cameras[side][axis])
                for n_water in N_WATER_VALUES:
                    predicted = n_water * native_f
                    focal_ratios.append(
                        {
                            "coordinate_system": coordinate_system,
                            "side": side,
                            "axis": axis,
                            "n_water_over_n_air": n_water,
                            "native_focal_px": native_f,
                            "custom_focal_px": custom_f,
                            "custom_over_native": custom_f / native_f,
                            "n_times_native_px": predicted,
                            "custom_minus_n_times_native_px": custom_f - predicted,
                            "percent_custom_vs_n_times_native": 100.0 * (custom_f / predicted - 1.0),
                        }
                    )

    custom_camera_dict = {
        "left": {"K": custom.left_k.tolist(), "D_first5": custom.left_d[:5].tolist()},
        "right": {"K": custom.right_k.tolist(), "D_first5": custom.right_d[:5].tolist()},
    }
    native_camera_dict = {
        "left": {"K": native_raw_left.tolist(), "D": native["raw"]["left"]["distortion"]},
        "right": {"K": native_raw_right.tolist(), "D": native["raw"]["right"]["distortion"]},
    }
    p1p2q = {
        "native_sdk_rectified_reference": {key: value.tolist() for key, value in native_rect.items() if isinstance(value, np.ndarray)},
        "custom_opencv_alpha0": {"P1": custom_rect0.p_left.tolist(), "P2": custom_rect0.p_right.tolist(), "Q": custom_rect0.q.tolist(), "focal_px": custom_rect0.focal_px, "baseline_m": custom_rect0.baseline_m, "principal_point_checks": custom_rect0.principal_point_checks},
        "custom_opencv_alpha1": {"P1": custom_rect1.p_left.tolist(), "P2": custom_rect1.p_right.tolist(), "Q": custom_rect1.q.tolist(), "focal_px": custom_rect1.focal_px, "baseline_m": custom_rect1.baseline_m, "principal_point_checks": custom_rect1.principal_point_checks},
    }
    checks = {
        "input_size_matches_svo_and_custom_calibration": list(input_size) == [1920, 1080],
        "native_raw_focal_positive": bool(np.all(np.diag(native_raw_left)[:2] > 0.0) and np.all(np.diag(native_raw_right)[:2] > 0.0)),
        "custom_focal_positive": bool(np.all(np.diag(custom.left_k)[:2] > 0.0) and np.all(np.diag(custom.right_k)[:2] > 0.0)),
        "native_rotation": _rotation_diagnostics(native_r_rl),
        "custom_rotation": _rotation_diagnostics(custom_r_lr),
        "native_raw_transform_units": "meter from SDK read with coordinate_units=METER",
        "custom_transform_units": "millimetre in YAML, converted exactly once to metre",
        "custom_P2_identity_alpha0": float(custom_rect0.p_right[0, 3] / custom_rect0.p_right[0, 0] + custom_rect0.baseline_m),
        "custom_P2_identity_alpha1": float(custom_rect1.p_right[0, 3] / custom_rect1.p_right[0, 0] + custom_rect1.baseline_m),
        "native_P2_identity": float(native_rect["P2"][0, 3] / native_rect["P2"][0, 0] + native_rect["baseline_m"]),
        "custom_alpha0_alpha1_baseline_difference_m": float(custom_rect0.baseline_m - custom_rect1.baseline_m),
        "no_duplicate_rectification": "native branch uses SVO SDK rectified views; custom branch uses raw unrectified views plus one OpenCV remap",
    }
    return {
        "script": Path(__file__).name,
        "status": "CALIBRATION_COMPARISON_COMPLETE_PHYSICAL_PROVENANCE_UNKNOWN",
        "input": {"svo": str(svo_path), "intrinsics": str(intrinsics_path), "extrinsics": str(extrinsics_path), "image_size": list(input_size)},
        "model_candidates": {
            "A_native_dry_candidate": {"classification": "BEST_AVAILABLE_PHYSICAL_DRY_CANDIDATE", "physical_medium": "UNKNOWN", "uses": "native raw K/D and embedded right-to-left R/T for raw-pixel pinhole geometry"},
            "B_custom_empirical_pinhole": {"classification": "EMPIRICAL_UNDERWATER_PINHOLE_CANDIDATE", "physicality": "UNKNOWN_UNTIL_PROVENANCE_AND_METRIC_VALIDATION", "uses": "custom raw K/D/R/T and its own OpenCV rectification"},
            "R1_native_plus_refractive": {"classification": "PHYSICAL_MODEL_NOT_IDENTIFIABLE", "requires": "measured rig-level housing geometry and refractive indices"},
            "R2_custom_without_extra_snell": {"classification": "VALID_EMPIRICAL_CONTROL", "uses": "custom f/B/d only; no second refractive multiplier"},
            "R3_custom_plus_snell": {"classification": "LEGACY_DOUBLE_COUNTING_CONTROL_ONLY", "uses": "not a physical final estimate"},
        },
        "native_raw": {
            "camera_model": native["camera_model"],
            "serial_number": native["serial_number"],
            "sdk_version": native["sdk_version"],
            "self_calibration_disabled_in_audit_reader": True,
            "calibration_semantics": "SVO embedded raw SDK calibration; factory/self-calibration provenance is UNKNOWN, and this audit does not relabel it as air calibration",
            **native_camera_dict,
            "R_right_to_left": native_r_rl.tolist(),
            "T_right_to_left_m": native_t_rl.tolist(),
        },
        "custom_raw": {
            **custom_camera_dict,
            "R_left_to_right": custom_r_lr.tolist(),
            "T_left_to_right_mm": (1000.0 * custom_t_lr).tolist(),
            "declared_baseline_mm": float(custom.declared_baseline_mm),
            "reprojection_error_px": float(custom.reprojection_error_px),
            "scale_error_percent": float(custom.scale_error_percent),
        },
        "baseline_audit": {
            "nominal_hardware_baseline_mm": 120.0,
            "native_raw": _baseline_entry("native_raw", native_r_lr, native_t_lr, rectified_m=native_rect["baseline_m"]),
            "custom_yaml": _baseline_entry("custom_yaml", custom_r_lr, custom_t_lr, declared_mm=float(custom.declared_baseline_mm), rectified_m=custom_rect1.baseline_m),
            "delta_custom_minus_native_T_m": delta_t.tolist(),
            "delta_custom_minus_native_T_norm_m": float(np.linalg.norm(delta_t)),
            "interpretation": "The custom Tz and tilt may be an effective/systematic refractive calibration term or a convention/calibration artifact; repository evidence cannot prove it is a physical camera-center displacement.",
        },
        "rotation_audit": {
            "comparison_convention": "both rotations converted to left-to-right before subtraction",
            "R_delta_custom_times_native_inverse": rotation_delta.tolist(),
            "angle_deg": rotation_delta_angle,
            "rotation_vector_rad": rotation_delta_vector,
        },
        "focal_ratio_audit": focal_ratios,
        "radial_mapping_audit": {
            "native_left": _distortion_radial_mapping_audit(native_raw_left, _camera_distortion_from_metadata(native["raw"]["left"]), input_size[0], input_size[1]),
            "native_right": _distortion_radial_mapping_audit(native_raw_right, _camera_distortion_from_metadata(native["raw"]["right"]), input_size[0], input_size[1]),
            "custom_left": _distortion_radial_mapping_audit(custom.left_k, custom.left_d, input_size[0], input_size[1]),
            "custom_right": _distortion_radial_mapping_audit(custom.right_k, custom.right_d, input_size[0], input_size[1]),
        },
        "P1_P2_Q": p1p2q,
        "provenance": {
            "calibration_medium": "UNKNOWN",
            "housing_used": "UNKNOWN",
            "same_housing_as_recording": "UNKNOWN",
            "port_type_and_shared_geometry": "UNKNOWN",
            "target_type": "UNKNOWN",
            "target_square_size_or_metric_scale": "UNKNOWN; ABSOLUTE_SCALE_NOT_IDENTIFIABLE_FROM_REPOSITORY_EVIDENCE",
            "calibration_image_count": 35,
            "software": "partially known: ZED SDK 5.4.1 and OpenCV/Python are recorded; calibration capture software/version is UNKNOWN",
            "intrinsic_stereo_model": "native RAD_TAN 12-coefficient metadata versus custom OpenCV 5-coefficient RAD_TAN-like model; physical provenance UNKNOWN",
            "evidence_checked": [
                "README.md",
                "20260802_150233_underwater_stereo_detailed_report.md",
                "Calibration/zed_custom_opencv.yml",
                "Calibration/标定结果/camera_intrinsics.yaml",
                "Calibration/标定结果/stereo_extrinsics.yaml",
                "Calibration/标定结果/calib_full_params.xlsx",
                "Calibration/标定结果/*.pdf",
                "all tracked source scripts and visible git history",
            ],
        },
        "checks": checks,
        "conventions": {
            "raw_pixel_ray": "inverse distortion with the model-specific raw K/D",
            "rectified_disparity": "only paired with the P1/P2/Q generated from the same calibration and alpha",
            "Z": "left-camera optical-axis coordinate; not camera-to-target range and not interface-to-target distance",
            "range": "Euclidean distance from the left camera origin for dry pinhole rows; refractive range would require the outer interface origin",
        },
    }


def _patch_ncc(left: np.ndarray, right: np.ndarray, p_left: np.ndarray, p_right: np.ndarray, radius: int = 5) -> float:
    u1, v1 = np.rint(p_left).astype(int)
    u2, v2 = np.rint(p_right).astype(int)
    if min(u1 - radius, v1 - radius, u2 - radius, v2 - radius) < 0:
        return float("nan")
    if v1 + radius >= left.shape[0] or u1 + radius >= left.shape[1] or v2 + radius >= right.shape[0] or u2 + radius >= right.shape[1]:
        return float("nan")
    a = left[v1 - radius:v1 + radius + 1, u1 - radius:u1 + radius + 1].astype(np.float64)
    b = right[v2 - radius:v2 + radius + 1, u2 - radius:u2 + radius + 1].astype(np.float64)
    a -= float(np.mean(a))
    b -= float(np.mean(b))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.sum(a * b) / denominator) if denominator > 1.0e-9 else float("nan")


def _match_quality(
    frame_index: int,
    left: np.ndarray,
    right: np.ndarray,
    calibration: CustomCalibration,
    rectification: Rectification,
    max_selected: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Produce model-agnostic Set A and custom-consistent Set B flags."""

    orb = cv2.ORB_create(nfeatures=6000, fastThreshold=7)
    key_left, desc_left = orb.detectAndCompute(left, None)
    key_right, desc_right = orb.detectAndCompute(right, None)
    summary: dict[str, Any] = {
        "frame_index": frame_index,
        "keypoints_left": len(key_left),
        "keypoints_right": len(key_right),
        "ratio_candidates": 0,
        "mutual_candidates": 0,
        "set_a_count": 0,
        "set_b_count": 0,
    }
    if desc_left is None or desc_right is None:
        return [], summary
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward = matcher.knnMatch(desc_left, desc_right, k=2)
    backward = matcher.knnMatch(desc_right, desc_left, k=2)
    reverse_best: dict[int, tuple[int, float]] = {}
    for pair in backward:
        if len(pair) != 2:
            continue
        best, second = pair
        ratio = float(best.distance / max(second.distance, 1.0))
        if ratio <= 0.85:
            reverse_best[best.queryIdx] = (best.trainIdx, ratio)

    candidates: list[dict[str, Any]] = []
    for pair in forward:
        if len(pair) != 2:
            continue
        best, second = pair
        ratio = float(best.distance / max(second.distance, 1.0))
        if ratio > 0.85:
            continue
        reverse = reverse_best.get(best.trainIdx)
        mutual = bool(reverse is not None and reverse[0] == best.queryIdx)
        p_left = np.asarray(key_left[best.queryIdx].pt, dtype=np.float64)
        p_right = np.asarray(key_right[best.trainIdx].pt, dtype=np.float64)
        candidates.append({
            "frame_index": frame_index,
            "query_index": int(best.queryIdx),
            "train_index": int(best.trainIdx),
            "u_left_px": float(p_left[0]),
            "v_left_px": float(p_left[1]),
            "u_right_px": float(p_right[0]),
            "v_right_px": float(p_right[1]),
            "ratio": ratio,
            "descriptor_distance": float(best.distance),
            "mutual": mutual,
            "ncc": float("nan"),
            "lk_forward_backward_px": float("nan"),
            "custom_rectified_vertical_px": float("nan"),
            "custom_rectified_disparity_px": float("nan"),
            "custom_ransac_inlier": False,
            "set_a": False,
            "set_b": False,
            "selected_set_a": False,
            "selected_set_b": False,
        })
    summary["ratio_candidates"] = len(candidates)
    summary["mutual_candidates"] = sum(bool(item["mutual"]) for item in candidates)
    if not candidates:
        return [], summary

    points_left = np.asarray([[item["u_left_px"], item["v_left_px"]] for item in candidates], dtype=np.float32).reshape(-1, 1, 2)
    points_right = np.asarray([[item["u_right_px"], item["v_right_px"]] for item in candidates], dtype=np.float32).reshape(-1, 1, 2)
    try:
        forward_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(left, right, points_left, None, winSize=(31, 31), maxLevel=3)
        backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(right, left, points_right, None, winSize=(31, 31), maxLevel=3)
    except cv2.error:
        forward_points = backward_points = None
        forward_status = backward_status = None
    for index, item in enumerate(candidates):
        p_left = points_left[index, 0].astype(np.float64)
        p_right = points_right[index, 0].astype(np.float64)
        item["ncc"] = _patch_ncc(left, right, p_left, p_right)
        if forward_points is not None and backward_points is not None and forward_status is not None and backward_status is not None and forward_status[index, 0] and backward_status[index, 0]:
            item["lk_forward_backward_px"] = float(np.linalg.norm(backward_points[index, 0] - p_left))

    raw_left = np.asarray([[item["u_left_px"], item["v_left_px"]] for item in candidates], dtype=np.float64)
    raw_right = np.asarray([[item["u_right_px"], item["v_right_px"]] for item in candidates], dtype=np.float64)
    rect_left = cv2.undistortPoints(raw_left.reshape(-1, 1, 2), calibration.left_k, calibration.left_d, R=rectification.r_left, P=rectification.p_left).reshape(-1, 2)
    rect_right = cv2.undistortPoints(raw_right.reshape(-1, 1, 2), calibration.right_k, calibration.right_d, R=rectification.r_right, P=rectification.p_right).reshape(-1, 2)
    fundamental_mask = np.zeros(len(candidates), dtype=bool)
    finite = np.isfinite(rect_left).all(axis=1) & np.isfinite(rect_right).all(axis=1)
    finite_indices = np.flatnonzero(finite)
    if finite_indices.size >= 8:
        _, mask = cv2.findFundamentalMat(rect_left[finite], rect_right[finite], cv2.FM_RANSAC, 1.5, 0.999, 5000)
        if mask is not None:
            fundamental_mask[finite_indices] = mask.reshape(-1).astype(bool)

    for index, item in enumerate(candidates):
        vertical = float(abs(rect_left[index, 1] - rect_right[index, 1])) if finite[index] else float("nan")
        disparity = float(rect_left[index, 0] - rect_right[index, 0]) if finite[index] else float("nan")
        item["custom_rectified_vertical_px"] = vertical
        item["custom_rectified_disparity_px"] = disparity
        item["custom_ransac_inlier"] = bool(fundamental_mask[index])
        ncc = float(item["ncc"])
        fb = float(item["lk_forward_backward_px"])
        item["set_a"] = bool(item["mutual"] and item["ratio"] <= 0.80 and math.isfinite(ncc) and ncc >= 0.20 and math.isfinite(fb) and fb <= 3.0)
        item["set_b"] = bool(item["set_a"] and item["custom_ransac_inlier"] and math.isfinite(vertical) and vertical <= 2.0 and math.isfinite(disparity) and disparity > 1.0)

    set_a = sorted((item for item in candidates if item["set_a"]), key=lambda item: (-float(item["ncc"]), float(item["lk_forward_backward_px"]), float(item["ratio"])))
    set_b = sorted((item for item in candidates if item["set_b"]), key=lambda item: (-float(item["ncc"]), float(item["lk_forward_backward_px"]), float(item["ratio"])))
    for item in set_a[:max_selected]:
        item["selected_set_a"] = True
    for item in set_b[:max_selected]:
        item["selected_set_b"] = True
    summary["set_a_count"] = int(sum(bool(item["set_a"]) for item in candidates))
    summary["set_b_count"] = int(sum(bool(item["set_b"]) for item in candidates))
    summary["selected_set_a_count"] = int(sum(bool(item["selected_set_a"]) for item in candidates))
    summary["selected_set_b_count"] = int(sum(bool(item["selected_set_b"]) for item in candidates))
    return candidates, summary


def _triangulate_pinhole(
    left_pixel: tuple[float, float],
    right_pixel: tuple[float, float],
    left_k: np.ndarray,
    left_d: np.ndarray,
    right_k: np.ndarray,
    right_d: np.ndarray,
    rotation_right_to_left: np.ndarray,
    translation_right_to_left_m: np.ndarray,
) -> dict[str, Any]:
    left_direction = pixel_to_air_ray(left_pixel, left_k, left_d)
    right_direction_local = pixel_to_air_ray(right_pixel, right_k, right_d)
    right_direction = rotation_right_to_left @ right_direction_local
    right_direction /= np.linalg.norm(right_direction)
    left_origin = np.zeros(3, dtype=np.float64)
    right_origin = np.asarray(translation_right_to_left_m, dtype=np.float64).reshape(3)
    matrix = np.column_stack((left_direction, -right_direction))
    singular = np.linalg.svd(matrix, compute_uv=False)
    solution, _, rank, _ = np.linalg.lstsq(matrix, right_origin - left_origin, rcond=None)
    t_left, t_right = float(solution[0]), float(solution[1])
    point_left = left_origin + t_left * left_direction
    point_right = right_origin + t_right * right_direction
    midpoint = 0.5 * (point_left + point_right)
    angle = float(np.degrees(np.arccos(np.clip(np.dot(left_direction, right_direction), -1.0, 1.0))))
    condition = float(np.inf if singular[-1] <= 1.0e-15 else singular[0] / singular[-1])
    return {
        "t_left_m": t_left,
        "t_right_m": t_right,
        "rank": int(rank),
        "ray_angle_deg": angle,
        "condition_number": condition,
        "ray_gap_m": float(np.linalg.norm(point_left - point_right)),
        "z_left_camera_m": float(midpoint[2]),
        "range_from_left_camera_m": float(np.linalg.norm(midpoint)),
        "midpoint_x_left_m": float(midpoint[0]),
        "midpoint_y_left_m": float(midpoint[1]),
        "positive_parameters": bool(t_left > 0.0 and t_right > 0.0),
        "angle_gt_0_25_deg": bool(angle > 0.25),
        "angle_gt_0_5_deg": bool(angle > 0.5),
        "angle_gt_1_deg": bool(angle > 1.0),
        "rank2_positive_status": "PASS" if int(rank) == 2 and t_left > 0.0 and t_right > 0.0 else "FAIL",
    }


def _correspondence_rows_for_csv(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "frame_index", "match_id", "query_index", "train_index", "u_left_px", "v_left_px", "u_right_px", "v_right_px",
        "ratio", "descriptor_distance", "mutual", "ncc", "lk_forward_backward_px", "custom_rectified_vertical_px",
        "custom_rectified_disparity_px", "custom_ransac_inlier", "set_a", "set_b", "selected_set_a", "selected_set_b",
    ]
    result = []
    for index, row in enumerate(all_rows):
        result.append({"match_id": index, **{field: row.get(field) for field in fields if field != "match_id"}})
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _triangulation_rows(
    selected_records: list[dict[str, Any]],
    native_left_k: np.ndarray,
    native_left_d: np.ndarray,
    native_right_k: np.ndarray,
    native_right_d: np.ndarray,
    native_r_rl: np.ndarray,
    native_t_rl: np.ndarray,
    custom: CustomCalibration,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    custom_r_rl = custom.rotation.T
    custom_t_rl = -custom.rotation.T @ (custom.translation_mm / 1000.0)
    for record in selected_records:
        sets = []
        if record["selected_set_a"]:
            sets.append("Set_A_model_agnostic")
        if record["selected_set_b"]:
            sets.append("Set_B_custom_consistent")
        for set_name in sets:
            for model_name, values in (
                ("A_native_dry_pinhole", (native_left_k, native_left_d, native_right_k, native_right_d, native_r_rl, native_t_rl)),
                ("B_custom_empirical_pinhole", (custom.left_k, custom.left_d, custom.right_k, custom.right_d, custom_r_rl, custom_t_rl)),
            ):
                try:
                    metrics = _triangulate_pinhole(
                        (record["u_left_px"], record["v_left_px"]),
                        (record["u_right_px"], record["v_right_px"]),
                        *values,
                    )
                    status = metrics["rank2_positive_status"]
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                    metrics = {"error": str(error)}
                    status = "ERROR"
                rows.append({"frame_index": record["frame_index"], "match_id": record["match_id"], "set": set_name, "model": model_name, "physical_status": "pinhole_candidate_only", "recommended_angle_threshold_deg": 0.5, "ray_gap_threshold_m": 0.01, "status": status, **metrics})
            rows.append({
                "frame_index": record["frame_index"], "match_id": record["match_id"], "set": set_name,
                "model": "R1_native_dry_plus_explicit_refractive_rig", "physical_status": "NOT_IDENTIFIABLE_WITHOUT_MEASURED_PORT_PARAMETERS",
                "recommended_angle_threshold_deg": 0.5, "ray_gap_threshold_m": 0.01, "status": "NOT_IDENTIFIABLE",
                "reason": "native K/D/T are available but housing plane/index/interface measurements are not",
            })
    return rows


def _ray_field_rows(
    native: dict[str, Any],
    custom: CustomCalibration,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    normalized_positions = ((0.05, 0.05), (0.50, 0.05), (0.95, 0.05), (0.05, 0.50), (0.50, 0.50), (0.95, 0.50), (0.05, 0.95), (0.50, 0.95), (0.95, 0.95))
    for side in ("left", "right"):
        native_k = _camera_matrix_from_metadata(native["raw"][side])
        native_d = _camera_distortion_from_metadata(native["raw"][side])
        custom_k = custom.left_k if side == "left" else custom.right_k
        custom_d = custom.left_d if side == "left" else custom.right_d
        for sample_id, (x, y) in enumerate(normalized_positions):
            pixel = (x * (width - 1), y * (height - 1))
            native_ray = pixel_to_air_ray(pixel, native_k, native_d)
            custom_ray = pixel_to_air_ray(pixel, custom_k, custom_d)
            custom_native_angle = float(np.degrees(np.arccos(np.clip(np.dot(native_ray, custom_ray), -1.0, 1.0))))
            for n_water in N_WATER_VALUES:
                try:
                    single_interface = refract_vector(native_ray, [0.0, 0.0, 1.0], 1.0, n_water)
                    native_water_angle = float(np.degrees(np.arctan2(np.linalg.norm(single_interface[:2]), single_interface[2])))
                    native_air_angle = float(np.degrees(np.arctan2(np.linalg.norm(native_ray[:2]), native_ray[2])))
                except ValueError:
                    native_water_angle = float("nan")
                    native_air_angle = float("nan")
                rows.append({
                    "side": side, "sample_id": sample_id, "u_px": pixel[0], "v_px": pixel[1], "n_water": n_water,
                    "native_air_ray_x": native_ray[0], "native_air_ray_y": native_ray[1], "native_air_ray_z": native_ray[2],
                    "custom_air_ray_x": custom_ray[0], "custom_air_ray_y": custom_ray[1], "custom_air_ray_z": custom_ray[2],
                    "native_air_angle_deg": native_air_angle, "custom_air_angle_deg": float(np.degrees(np.arctan2(np.linalg.norm(custom_ray[:2]), custom_ray[2]))),
                    "custom_minus_native_air_ray_angle_deg": custom_native_angle,
                    "native_single_interface_water_angle_deg": native_water_angle,
                    "single_interface_vs_custom_angle_deg": native_water_angle - float(np.degrees(np.arctan2(np.linalg.norm(custom_ray[:2]), custom_ray[2]))) if math.isfinite(native_water_angle) else float("nan"),
                    "status": "single_interface_hypothetical_not_rig_model",
                })
    return rows


def _sensitivity_rows(
    selected_by_set: dict[str, list[dict[str, Any]]],
    native_left_k: np.ndarray,
    native_left_d: np.ndarray,
    native_right_k: np.ndarray,
    native_right_d: np.ndarray,
    native_r_rl: np.ndarray,
    native_t_rl: np.ndarray,
    native_dry_z_by_match: dict[tuple[int, int], float],
) -> list[dict[str, Any]]:
    left_pose = RigCameraPose(np.eye(3), np.zeros(3))
    right_pose = RigCameraPose(native_r_rl, native_t_rl)
    rows: list[dict[str, Any]] = []
    for set_name, records in selected_by_set.items():
        records = records[:50]
        for n_water in N_WATER_VALUES:
            for n_glass in N_GLASS_VALUES:
                for h_m in H_VALUES_M:
                    for thickness_m in THICKNESS_VALUES_M:
                        z_values: list[float] = []
                        range_values: list[float] = []
                        gap_values: list[float] = []
                        angle_values: list[float] = []
                        dry_values: list[float] = []
                        failures = 0
                        model = RigFlatPortModel.shared_parallel(1.0, n_glass, n_water, [0.0, 0.0, 1.0], h_m, thickness_m)
                        for record in records:
                            try:
                                left_ray = trace_rig_pixel("left", [record["u_left_px"], record["v_left_px"]], native_left_k, native_left_d, left_pose, model)
                                right_ray = trace_rig_pixel("right", [record["u_right_px"], record["v_right_px"]], native_right_k, native_right_d, right_pose, model)
                                metrics = triangulate_rig_water_rays(left_ray, right_ray)
                                if int(metrics["rank"]) != 2 or not bool(metrics["positive_parameters"]):
                                    failures += 1
                                    continue
                                z_values.append(float(np.asarray(metrics["midpoint_rig_m"])[2]))
                                range_values.append(float(np.linalg.norm(np.asarray(metrics["midpoint_rig_m"]))))
                                gap_values.append(float(metrics["ray_gap_m"]))
                                angle_values.append(float(metrics["ray_angle_deg"]))
                                dry = native_dry_z_by_match.get((int(record["frame_index"]), int(record["match_id"])))
                                if dry is not None and math.isfinite(dry):
                                    dry_values.append(dry)
                            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                                failures += 1
                        z_stats = _stats(z_values)
                        range_stats = _stats(range_values)
                        gap_stats = _stats(gap_values)
                        angle_stats = _stats(angle_values)
                        dry_stats = _stats(dry_values)
                        rows.append({
                            "status": "HYPOTHETICAL_R1_SENSITIVITY_ONLY",
                            "physical_status": "NOT_IDENTIFIABLE",
                            "set": set_name,
                            "port_geometry": "shared_parallel_flat_port_assumption_in_rig_frame",
                            "interface_parallelism": "assumed exactly parallel; not measured",
                            "n_air": 1.0, "n_glass": n_glass, "n_water": n_water,
                            "camera_to_inner_interface_m": h_m, "glass_thickness_m": thickness_m,
                            "input_correspondence_count": len(records), "valid_count": z_stats.get("count", 0), "failure_count": failures,
                            "z_left_rig_median": z_stats.get("median"), "z_left_rig_p05": z_stats.get("p05"), "z_left_rig_p95": z_stats.get("p95"),
                            "range_from_left_camera_median": range_stats.get("median"), "ray_gap_median": gap_stats.get("median"), "ray_angle_deg_median": angle_stats.get("median"),
                            "native_dry_z_median_for_same_rows": dry_stats.get("median"),
                            "median_ratio_to_native_dry_z": (z_stats.get("median") / dry_stats.get("median")) if z_stats.get("median") and dry_stats.get("median") else None,
                        })
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("svo", nargs="?", type=Path, default=DEFAULT_SVO)
    parser.add_argument("--intrinsics", type=Path, default=None)
    parser.add_argument("--extrinsics", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=20, help="evenly scattered frame positions, including frame 0")
    parser.add_argument("--max-matches-per-frame", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    svo_path = args.svo.expanduser().resolve()
    intrinsics_path = (args.intrinsics or _find_calibration_file("camera_intrinsics.yaml")).expanduser().resolve()
    extrinsics_path = (args.extrinsics or _find_calibration_file("stereo_extrinsics.yaml")).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.frames < 10:
        raise ValueError("--frames must be at least 10 for a scattered final audit")
    for path, label in ((svo_path, "SVO2"), (intrinsics_path, "intrinsics"), (extrinsics_path, "extrinsics")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    output_names = ("native_vs_custom_calibration.json", "ray_field_comparison.csv", "rig_refractive_sensitivity.csv", "correspondence_quality.csv", "triangulation_quality.csv", "native_refractive_depth_check.json")
    if not args.overwrite:
        existing = [output_dir / name for name in output_names if (output_dir / name).exists()]
        if existing:
            raise RuntimeError(f"outputs exist; use --overwrite: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    native = collect_native_metadata(svo_path)
    width = int(native["resolution"]["width"])
    height = int(native["resolution"]["height"])
    custom = _load_custom_calibration(intrinsics_path, extrinsics_path)
    custom_rect0 = _make_custom_rectification(custom, (width, height), (width, height), 0.0)
    custom_rect1 = _make_custom_rectification(custom, (width, height), (width, height), 1.0)
    calibration_report = _build_calibration_report(native, custom, custom_rect0, custom_rect1, (width, height), svo_path, intrinsics_path, extrinsics_path)
    (output_dir / "native_vs_custom_calibration.json").write_text(json.dumps(calibration_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    native_left_k = _camera_matrix_from_metadata(native["raw"]["left"])
    native_right_k = _camera_matrix_from_metadata(native["raw"]["right"])
    native_left_d = _camera_distortion_from_metadata(native["raw"]["left"])
    native_right_d = _camera_distortion_from_metadata(native["raw"]["right"])
    native_transform = np.asarray(native["raw"]["stereo_transform_right_to_left_m"], dtype=np.float64)
    native_r_rl = native_transform[:3, :3]
    native_t_rl = native_transform[:3, 3]
    frame_indices = sorted({int(round(value)) for value in np.linspace(0, int(native["total_svo_frames"]) - 1, args.frames)})
    zed = __import__("debug_refractive_depth_check", fromlist=["_open_svo"])._open_svo(svo_path)
    left_mat, right_mat = sl.Mat(), sl.Mat()
    all_matches: list[dict[str, Any]] = []
    frame_summaries: list[dict[str, Any]] = []
    try:
        for frame_index in frame_indices:
            _check_status(zed.set_svo_position(frame_index), f"set SVO position {frame_index}")
            _check_status(zed.grab(sl.RuntimeParameters()), f"grab frame {frame_index}")
            _check_status(zed.retrieve_image(left_mat, sl.VIEW.LEFT_UNRECTIFIED), f"retrieve left raw {frame_index}")
            _check_status(zed.retrieve_image(right_mat, sl.VIEW.RIGHT_UNRECTIFIED), f"retrieve right raw {frame_index}")
            left = _as_gray(left_mat.get_data())
            right = _as_gray(right_mat.get_data())
            matches, summary = _match_quality(frame_index, left, right, custom, custom_rect1, args.max_matches_per_frame)
            for match in matches:
                all_matches.append(match)
            frame_summaries.append(summary)
    finally:
        zed.close()

    correspondence_fields = [
        "frame_index", "match_id", "query_index", "train_index", "u_left_px", "v_left_px", "u_right_px", "v_right_px",
        "ratio", "descriptor_distance", "mutual", "ncc", "lk_forward_backward_px", "custom_rectified_vertical_px",
        "custom_rectified_disparity_px", "custom_ransac_inlier", "set_a", "set_b", "selected_set_a", "selected_set_b",
    ]
    csv_rows = []
    per_frame_match_id: dict[int, int] = {}
    for row in all_matches:
        frame = int(row["frame_index"])
        match_id = per_frame_match_id.get(frame, 0)
        per_frame_match_id[frame] = match_id + 1
        row["match_id"] = match_id
        csv_rows.append({field: row.get(field) for field in correspondence_fields})
    _write_csv(output_dir / "correspondence_quality.csv", csv_rows, correspondence_fields)

    selected_records = [row for row in all_matches if row["selected_set_a"] or row["selected_set_b"]]
    # Match IDs are frame-local and are used only for audit joins.
    for row in selected_records:
        row["match_id"] = next(item["match_id"] for item in csv_rows if item["frame_index"] == row["frame_index"] and item["u_left_px"] == row["u_left_px"] and item["u_right_px"] == row["u_right_px"])
    triangulation_rows = _triangulation_rows(selected_records, native_left_k, native_left_d, native_right_k, native_right_d, native_r_rl, native_t_rl, custom)
    triangulation_fields = [
        "frame_index", "match_id", "set", "model", "physical_status", "recommended_angle_threshold_deg", "ray_gap_threshold_m", "status", "t_left_m", "t_right_m", "rank", "ray_angle_deg", "condition_number", "ray_gap_m", "z_left_camera_m", "range_from_left_camera_m", "midpoint_x_left_m", "midpoint_y_left_m", "positive_parameters", "angle_gt_0_25_deg", "angle_gt_0_5_deg", "angle_gt_1_deg", "rank2_positive_status", "reason", "error",
    ]
    _write_csv(output_dir / "triangulation_quality.csv", triangulation_rows, triangulation_fields)

    dry_by_match: dict[tuple[int, int], float] = {}
    for row in triangulation_rows:
        if row.get("model") == "A_native_dry_pinhole" and row.get("set") == "Set_A_model_agnostic" and row.get("z_left_camera_m") is not None and row.get("status") == "PASS":
            dry_by_match[(int(row["frame_index"]), int(row["match_id"]))] = float(row["z_left_camera_m"])
    selected_by_set = {
        "Set_A_model_agnostic": [row for row in selected_records if row["selected_set_a"]],
        "Set_B_custom_consistent": [row for row in selected_records if row["selected_set_b"]],
    }
    sensitivity_rows = _sensitivity_rows(selected_by_set, native_left_k, native_left_d, native_right_k, native_right_d, native_r_rl, native_t_rl, dry_by_match)
    sensitivity_fields = [
        "status", "physical_status", "set", "port_geometry", "interface_parallelism", "n_air", "n_glass", "n_water", "camera_to_inner_interface_m", "glass_thickness_m", "input_correspondence_count", "valid_count", "failure_count", "z_left_rig_median", "z_left_rig_p05", "z_left_rig_p95", "range_from_left_camera_median", "ray_gap_median", "ray_angle_deg_median", "native_dry_z_median_for_same_rows", "median_ratio_to_native_dry_z",
    ]
    _write_csv(output_dir / "rig_refractive_sensitivity.csv", sensitivity_rows, sensitivity_fields)
    ray_rows = _ray_field_rows(native, custom, width, height)
    _write_csv(output_dir / "ray_field_comparison.csv", ray_rows, list(ray_rows[0].keys()) if ray_rows else ["status"])

    native_report_path = output_dir / "native_refractive_depth_check.json"
    native_report = {
        "script": Path(__file__).name,
        "status": "NOT_IDENTIFIABLE",
        "model": "R1_native_raw_plus_measured_rig_refraction",
        "definitive_depth": None,
        "native_metadata_source": "native_refractive_depth_check.py",
        "native_candidate": native,
        "missing_physical_parameters": ["housing/port type", "shared/separate port planes in rig frame", "n_air", "n_glass", "n_water", "camera_to_inner_interface_m", "glass_thickness_m", "plane normals", "independent metric GT"],
        "same_raw_correspondences_used_for_controls": len(selected_records),
        "guardrail": "R1 remains null; rig_refractive_sensitivity.csv is hypothetical parameter sensitivity only",
    }
    native_report_path.write_text(json.dumps(native_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Frames: {frame_indices}")
    print(f"Correspondence candidates: {len(all_matches)}; Set A: {sum(int(row['selected_set_a']) for row in all_matches)}; Set B: {sum(int(row['selected_set_b']) for row in all_matches)}")
    print(f"Triangulation rows: {len(triangulation_rows)}; rig sensitivity rows: {len(sensitivity_rows)}")
    print("R1 native + explicit refraction: NOT_IDENTIFIABLE; no definitive underwater depth emitted")
    print("R2 custom pinhole: empirical/internal-consistency candidate only")
    print("R3 custom + Snell: legacy double-counting control only")
    print(f"Elapsed seconds: {time.monotonic() - started:.2f}")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
