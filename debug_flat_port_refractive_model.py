"""Correspondence-level flat-port refractive geometry diagnostic.

This diagnostic is deliberately separate from the production SGBM/ZED/SLAM
pipelines.  It uses raw unrectified stereo images, feature correspondences,
per-pixel inverse distortion, and the explicit air -> glass -> water ray model
in ``refractive_geometry.py``.  Without measured port parameters it writes
``refractive_depth=None`` and reports ``not_identifiable_without_port_parameters``;
the hypothetical sensitivity CSV is not a reconstruction result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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

from debug_refractive_depth_check import (  # noqa: E402
    CustomCalibration,
    _as_gray,
    _camera_metadata,
    _check_status,
    _load_custom_calibration,
    _matrix4,
    _open_svo,
    _status_name,
)
from refractive_geometry import (  # noqa: E402
    FlatPortModel,
    RefractiveModelNotIdentifiable,
    StereoExtrinsics,
    pixel_to_air_ray,
    refractive_stereo_correspondence,
    triangulate_two_rays,
    triangulation_depth_metrics,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_SVO = ROOT / "20260802_150233.svo2"
DEFAULT_OUTPUT = ROOT / "output" / "20260802_150233_flat_port_refractive_audit"
SENSITIVITY_N_WATER = (1.330, 1.333, 1.340)
SENSITIVITY_N_GLASS = (1.47, 1.50, 1.52)
SENSITIVITY_H_M = (0.005, 0.010, 0.020, 0.030)
SENSITIVITY_THICKNESS_M = (0.003, 0.005, 0.008, 0.010)


@dataclass(frozen=True)
class AlphaGeometry:
    alpha: float
    r_left: np.ndarray
    r_right: np.ndarray
    p_left: np.ndarray
    p_right: np.ndarray
    q: np.ndarray
    roi_left: tuple[int, int, int, int]
    roi_right: tuple[int, int, int, int]

    @property
    def focal_px(self) -> float:
        return float(self.p_left[0, 0])

    @property
    def baseline_m(self) -> float:
        return abs(float(self.p_right[0, 3] / self.p_right[0, 0])) / 1000.0

    @property
    def principal_point_offset_px(self) -> tuple[float, float]:
        return (
            float(self.p_left[0, 2] - self.p_right[0, 2]),
            float(self.p_left[1, 2] - self.p_right[1, 2]),
        )


def _find_calibration_file(name: str) -> Path:
    matches = sorted((ROOT / "Calibration").rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one Calibration/{name}, found {matches}")
    return matches[0]


def _camera_matrix(camera: Any) -> np.ndarray:
    return np.array(
        [[float(camera.fx), 0.0, float(camera.cx)], [0.0, float(camera.fy), float(camera.cy)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _camera_distortion(camera: Any) -> np.ndarray:
    return np.asarray(camera.disto, dtype=np.float64).reshape(-1)


def _make_custom_alpha_geometry(
    calibration: CustomCalibration,
    input_size: tuple[int, int],
    alpha: float,
) -> AlphaGeometry:
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
        newImageSize=input_size,
    )
    geometry = AlphaGeometry(
        alpha=alpha,
        r_left=r_left,
        r_right=r_right,
        p_left=p_left,
        p_right=p_right,
        q=q,
        roi_left=tuple(int(value) for value in roi_left),
        roi_right=tuple(int(value) for value in roi_right),
    )
    cx_delta, cy_delta = geometry.principal_point_offset_px
    if abs(cx_delta) >= 1.0e-6 or abs(cy_delta) >= 1.0e-6:
        raise RuntimeError(
            f"CALIB_ZERO_DISPARITY principal point equality failed for alpha={alpha}: "
            f"cx_delta={cx_delta}, cy_delta={cy_delta}"
        )
    if geometry.focal_px <= 0.0 or geometry.baseline_m <= 0.0:
        raise RuntimeError(f"invalid alpha geometry: {geometry}")
    return geometry


def _rectify_points(points: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray, geometry: AlphaGeometry, side: str) -> np.ndarray:
    if side == "left":
        rotation, projection = geometry.r_left, geometry.p_left
    elif side == "right":
        rotation, projection = geometry.r_right, geometry.p_right
    else:
        raise ValueError(side)
    return cv2.undistortPoints(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 2),
        camera_matrix,
        distortion,
        R=rotation,
        P=projection,
    ).reshape(-1, 2)


def _match_raw_frame(
    left: np.ndarray,
    right: np.ndarray,
    calibration: CustomCalibration,
    alpha_geometry: AlphaGeometry,
    max_matches: int = 150,
) -> dict[str, Any]:
    """ORB ratio test + rectified RANSAC + vertical epipolar consistency."""

    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=7)
    key_left, desc_left = orb.detectAndCompute(left, None)
    key_right, desc_right = orb.detectAndCompute(right, None)
    empty = {
        "status": "insufficient_features",
        "keypoints_left": len(key_left),
        "keypoints_right": len(key_right),
        "ratio_test_candidates": 0,
        "matches": [],
    }
    if desc_left is None or desc_right is None:
        return empty
    raw_matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_left, desc_right, k=2)
    candidates: list[tuple[Any, float]] = []
    for pair in raw_matches:
        if len(pair) != 2 or pair[0].distance >= 0.75 * pair[1].distance:
            continue
        candidates.append((pair[0], float(pair[0].distance / max(pair[1].distance, 1.0))))
    if len(candidates) < 8:
        empty["status"] = "insufficient_ratio_matches"
        empty["ratio_test_candidates"] = len(candidates)
        return empty
    raw_left = np.asarray([key_left[m.queryIdx].pt for m, _ in candidates], dtype=np.float64)
    raw_right = np.asarray([key_right[m.trainIdx].pt for m, _ in candidates], dtype=np.float64)
    rect_left = _rectify_points(raw_left, calibration.left_k, calibration.left_d, alpha_geometry, "left")
    rect_right = _rectify_points(raw_right, calibration.right_k, calibration.right_d, alpha_geometry, "right")
    fundamental, mask = cv2.findFundamentalMat(
        rect_left,
        rect_right,
        cv2.FM_RANSAC,
        1.5,
        0.999,
        5000,
    )
    if mask is None:
        mask_array = np.zeros(len(candidates), dtype=bool)
    else:
        mask_array = mask.reshape(-1).astype(bool)
    accepted: list[dict[str, Any]] = []
    for index, (match, ratio) in enumerate(candidates):
        if index >= mask_array.size or not mask_array[index]:
            continue
        vertical_error = float(abs(rect_left[index, 1] - rect_right[index, 1]))
        disparity = float(rect_left[index, 0] - rect_right[index, 0])
        if not np.isfinite(vertical_error) or not np.isfinite(disparity):
            continue
        if vertical_error > 2.0 or disparity <= 1.0:
            continue
        accepted.append(
            {
                "uL": float(raw_left[index, 0]),
                "vL": float(raw_left[index, 1]),
                "uR": float(raw_right[index, 0]),
                "vR": float(raw_right[index, 1]),
                "rectified_uL_alpha1": float(rect_left[index, 0]),
                "rectified_vL_alpha1": float(rect_left[index, 1]),
                "rectified_uR_alpha1": float(rect_right[index, 0]),
                "rectified_vR_alpha1": float(rect_right[index, 1]),
                "custom_rectified_disparity_alpha1_px": disparity,
                "rectified_vertical_error_alpha1_px": vertical_error,
                "descriptor_distance": float(match.distance),
                "ratio": ratio,
            }
        )
    accepted.sort(key=lambda item: (item["ratio"], item["descriptor_distance"]))
    return {
        "status": "ok" if accepted else "no_high_quality_correspondences",
        "keypoints_left": len(key_left),
        "keypoints_right": len(key_right),
        "ratio_test_candidates": len(candidates),
        "ransac_inliers": int(np.count_nonzero(mask_array)),
        "fundamental_matrix_found": fundamental is not None,
        "high_quality_matches": len(accepted[:max_matches]),
        "matches": accepted[:max_matches],
    }


def _pinhole_depth_from_raw_correspondence(
    left_pixel: tuple[float, float],
    right_pixel: tuple[float, float],
    left_k: np.ndarray,
    left_d: np.ndarray,
    right_k: np.ndarray,
    right_d: np.ndarray,
    extrinsics: StereoExtrinsics,
) -> dict[str, float] | None:
    try:
        left_direction = pixel_to_air_ray(left_pixel, left_k, left_d)
        right_direction = pixel_to_air_ray(right_pixel, right_k, right_d)
        right_origin_left, right_direction_left = extrinsics.transform_ray_to_left(
            np.zeros(3, dtype=np.float64), right_direction
        )
        triangulation = triangulate_two_rays(
            np.zeros(3, dtype=np.float64),
            left_direction,
            right_origin_left,
            right_direction_left,
        )
        metrics = triangulation_depth_metrics(triangulation, np.zeros(3, dtype=np.float64))
        if triangulation.parameter_left_m <= 0.0 or triangulation.parameter_right_m <= 0.0:
            return None
        if metrics["z_left_camera_m"] <= 0.0:
            return None
        return metrics
    except (ValueError, RuntimeError):
        return None


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


def _alpha_depth_for_record(
    record: dict[str, Any],
    calibration: CustomCalibration,
    geometry: AlphaGeometry,
) -> dict[str, float] | None:
    left = np.array([[record["uL"], record["vL"]]], dtype=np.float64)
    right = np.array([[record["uR"], record["vR"]]], dtype=np.float64)
    rect_left = _rectify_points(left, calibration.left_k, calibration.left_d, geometry, "left")[0]
    rect_right = _rectify_points(right, calibration.right_k, calibration.right_d, geometry, "right")[0]
    disparity = float(rect_left[0] - rect_right[0])
    vertical_error = float(abs(rect_left[1] - rect_right[1]))
    if disparity <= 1.0 or vertical_error > 2.0:
        return None
    return {
        "d_px": disparity,
        "vertical_error_px": vertical_error,
        "z_m": float(geometry.focal_px * geometry.baseline_m / disparity),
    }


def _alpha_invariance(
    records: list[dict[str, Any]],
    calibration: CustomCalibration,
    alpha0: AlphaGeometry,
    alpha1: AlphaGeometry,
    minimum_count: int = 100,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for record in records:
        value0 = _alpha_depth_for_record(record, calibration, alpha0)
        value1 = _alpha_depth_for_record(record, calibration, alpha1)
        if value0 is None or value1 is None:
            continue
        relative = abs(value0["z_m"] - value1["z_m"]) / max(abs(value1["z_m"]), 1.0e-12)
        comparisons.append(
            {
                "frame": int(record["frame"]),
                "correspondence_id": int(record["correspondence_id"]),
                "d_alpha0_px": value0["d_px"],
                "d_alpha1_px": value1["d_px"],
                "z_alpha0_m": value0["z_m"],
                "z_alpha1_m": value1["z_m"],
                "relative_difference": float(relative),
                "vertical_error_alpha0_px": value0["vertical_error_px"],
                "vertical_error_alpha1_px": value1["vertical_error_px"],
            }
        )
    relative_values = [item["relative_difference"] for item in comparisons]
    return {
        "status": "PASS" if len(comparisons) >= minimum_count and _stats(relative_values).get("p95", math.inf) < 1.0e-5 else "INSUFFICIENT_OR_FAILED",
        "minimum_high_quality_correspondences": minimum_count,
        "high_quality_correspondence_count": len(comparisons),
        "relative_difference_stats": _stats(relative_values),
        "alpha0": {
            "focal_px": alpha0.focal_px,
            "baseline_m": alpha0.baseline_m,
            "principal_point_offset_px": alpha0.principal_point_offset_px,
        },
        "alpha1": {
            "focal_px": alpha1.focal_px,
            "baseline_m": alpha1.baseline_m,
            "principal_point_offset_px": alpha1.principal_point_offset_px,
        },
        "comparison_definition": "same raw pixel correspondence is independently mapped into alpha=0 and alpha=1 virtual rectified coordinates; Z=fB/d is compared within each coordinate system",
        "records": comparisons,
    }


def _field_band(u: float, v: float, width: int, height: int) -> str:
    x = (u - width / 2.0) / (width / 2.0)
    y = (v - height / 2.0) / (height / 2.0)
    radius = math.hypot(x, y)
    if radius <= 0.25:
        return "central"
    if radius <= 0.60:
        return "mid_fov"
    return "near_edge"


def _add_disparity_bands(records: list[dict[str, Any]]) -> tuple[float, float]:
    disparities = np.asarray(
        [record["custom_rectified_disparity_alpha1_px"] for record in records],
        dtype=np.float64,
    )
    if disparities.size == 0:
        return math.nan, math.nan
    p33, p66 = np.percentile(disparities, [33.333333, 66.666667])
    for record in records:
        disparity = float(record["custom_rectified_disparity_alpha1_px"])
        if disparity <= p33:
            band = "far_disparity"
        elif disparity <= p66:
            band = "mid_disparity"
        else:
            band = "near_disparity"
        record["field_band"] = _field_band(float(record["uL"]), float(record["vL"]), 1920, 1080)
        record["disparity_band"] = band
    return float(p33), float(p66)


def _sensitivity_rows(
    records: list[dict[str, Any]],
    calibration: CustomCalibration,
    extrinsics: StereoExtrinsics,
) -> list[dict[str, Any]]:
    fields = ("central", "mid_fov", "near_edge")
    disparity_bands = ("far_disparity", "mid_disparity", "near_disparity")
    rows: list[dict[str, Any]] = []
    for n_water in SENSITIVITY_N_WATER:
        for n_glass in SENSITIVITY_N_GLASS:
            for h_m in SENSITIVITY_H_M:
                for thickness_m in SENSITIVITY_THICKNESS_M:
                    model = FlatPortModel(
                        n_air=1.0,
                        n_glass=n_glass,
                        n_water=n_water,
                        camera_to_inner_interface_m=h_m,
                        glass_thickness_m=thickness_m,
                    )
                    for field in fields:
                        for disparity_band in disparity_bands:
                            selected = [
                                record
                                for record in records
                                if record.get("field_band") == field
                                and record.get("disparity_band") == disparity_band
                                and record.get("custom_depth") is not None
                            ]
                            refractive_z: list[float] = []
                            ray_gaps: list[float] = []
                            relative: list[float] = []
                            for record in selected:
                                try:
                                    result = refractive_stereo_correspondence(
                                        (record["uL"], record["vL"]),
                                        (record["uR"], record["vR"]),
                                        calibration.left_k,
                                        calibration.left_d,
                                        calibration.right_k,
                                        calibration.right_d,
                                        model,
                                        model,
                                        extrinsics,
                                    )
                                except (ValueError, RuntimeError):
                                    continue
                                z = float(result["z_left_camera_m"])
                                refractive_z.append(z)
                                ray_gaps.append(float(result["ray_gap_m"]))
                                relative.append(z / float(record["custom_depth"]) - 1.0)
                            z_stats = _stats(refractive_z)
                            gap_stats = _stats(ray_gaps)
                            relative_stats = _stats(relative)
                            rows.append(
                                {
                                    "status": "hypothetical_sensitivity_only",
                                    "parameter_provenance": "hypothetical_grid_not_measured",
                                    "n_air": 1.0,
                                    "n_glass": n_glass,
                                    "n_water": n_water,
                                    "camera_to_inner_interface_m": h_m,
                                    "glass_thickness_m": thickness_m,
                                    "field_band": field,
                                    "disparity_band": disparity_band,
                                    "correspondence_count": len(refractive_z),
                                    "z_left_camera_median": z_stats.get("median"),
                                    "z_left_camera_mean": z_stats.get("mean"),
                                    "ray_gap_median": gap_stats.get("median"),
                                    "relative_to_custom_pinhole_median": relative_stats.get("median"),
                                    "relative_to_custom_pinhole_p95": relative_stats.get("p95"),
                                }
                            )
    return rows


def _write_sensitivity_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "status",
        "parameter_provenance",
        "n_air",
        "n_glass",
        "n_water",
        "camera_to_inner_interface_m",
        "glass_thickness_m",
        "field_band",
        "disparity_band",
        "correspondence_count",
        "z_left_camera_median",
        "z_left_camera_mean",
        "ray_gap_median",
        "relative_to_custom_pinhole_median",
        "relative_to_custom_pinhole_p95",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _model_from_args(args: argparse.Namespace) -> FlatPortModel:
    normal = tuple(float(value) for value in (args.plane_normal or (0.0, 0.0, 1.0)))
    return FlatPortModel(
        n_air=args.n_air,
        n_glass=args.n_glass,
        n_water=args.n_water,
        camera_to_inner_interface_m=args.port_distance,
        glass_thickness_m=args.glass_thickness,
        plane_normal_camera=normal,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("svo", nargs="?", type=Path, default=DEFAULT_SVO)
    parser.add_argument("--intrinsics", type=Path, default=None)
    parser.add_argument("--extrinsics", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=10, help="number of evenly scattered SVO positions; must be at least 10")
    parser.add_argument("--max-matches-per-frame", type=int, default=150)
    parser.add_argument("--n-air", type=float, default=None)
    parser.add_argument("--n-glass", type=float, default=None)
    parser.add_argument("--n-water", type=float, default=None)
    parser.add_argument("--port-distance", type=float, default=None, help="camera to inner port interface, metres")
    parser.add_argument("--glass-thickness", type=float, default=None, help="glass thickness, metres")
    parser.add_argument("--plane-normal", type=float, nargs=3, default=None, help="measured port normal in the camera frame; default is only a hypothetical +z convention")
    parser.add_argument("--port-parameters-provenance", choices=("not_provided", "assumed", "measured"), default="not_provided")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.frames < 10:
        parser.error("--frames must be at least 10 to include frame 0 plus nine scattered frames")
    if args.max_matches_per_frame <= 0:
        parser.error("--max-matches-per-frame must be positive")
    if args.port_parameters_provenance == "measured":
        values = (args.n_air, args.n_glass, args.n_water, args.port_distance, args.glass_thickness)
        if any(value is None for value in values):
            parser.error("measured port provenance requires all five physical parameters")
        if args.plane_normal is None:
            parser.error("measured port provenance also requires --plane-normal")
    return args


def main() -> int:
    args = _parse_args()
    svo_path = args.svo.expanduser().resolve()
    intrinsics_path = (args.intrinsics or _find_calibration_file("camera_intrinsics.yaml")).expanduser().resolve()
    extrinsics_path = (args.extrinsics or _find_calibration_file("stereo_extrinsics.yaml")).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not svo_path.is_file() or not intrinsics_path.is_file() or not extrinsics_path.is_file():
        raise FileNotFoundError(f"missing input: svo={svo_path}, intrinsics={intrinsics_path}, extrinsics={extrinsics_path}")
    output_names = (
        "refractive_correspondence_check.json",
        "refractive_sensitivity.csv",
        "alpha_invariance_check.json",
    )
    if not args.overwrite:
        existing = [output_dir / name for name in output_names if (output_dir / name).exists()]
        if existing:
            raise RuntimeError(f"outputs exist; use --overwrite: {existing}")
    started = time.monotonic()
    calibration = _load_custom_calibration(intrinsics_path, extrinsics_path)
    model = _model_from_args(args)
    alpha0 = _make_custom_alpha_geometry(calibration, (1920, 1080), 0.0)
    alpha1 = _make_custom_alpha_geometry(calibration, (1920, 1080), 1.0)

    probe = _open_svo(svo_path)
    try:
        info = probe.get_camera_information()
        total_frames = int(probe.get_svo_number_of_frames())
        resolution = info.camera_configuration.resolution
        width, height = int(resolution.width), int(resolution.height)
        raw_native = info.camera_configuration.calibration_parameters_raw
        native_left_k = _camera_matrix(raw_native.left_cam)
        native_right_k = _camera_matrix(raw_native.right_cam)
        native_left_d = _camera_distortion(raw_native.left_cam)
        native_right_d = _camera_distortion(raw_native.right_cam)
        native_transform = _matrix4(raw_native.stereo_transform.m)
        native_extrinsics = StereoExtrinsics(native_transform[:3, :3], native_transform[:3, 3])
    finally:
        probe.close()
    if (width, height) != (1920, 1080):
        raise RuntimeError(f"expected 1920x1080 raw frames, got {(width, height)}")
    frame_indices = sorted(
        {int(round(value)) for value in np.linspace(0, total_frames - 1, args.frames)}
    )
    if len(frame_indices) < 10 or frame_indices[0] != 0:
        raise RuntimeError(f"could not construct frame 0 plus nine scattered frame indices: {frame_indices}")

    custom_extrinsics = StereoExtrinsics.from_opencv_left_to_right(
        calibration.rotation,
        calibration.translation_mm / 1000.0,
    )
    zed = _open_svo(svo_path)
    left_mat, right_mat = sl.Mat(), sl.Mat()
    correspondence_records: list[dict[str, Any]] = []
    frame_summaries: list[dict[str, Any]] = []
    try:
        for frame_index in frame_indices:
            _check_status(zed.set_svo_position(frame_index), f"set SVO position {frame_index}")
            _check_status(zed.grab(sl.RuntimeParameters()), f"grab frame {frame_index}")
            _check_status(zed.retrieve_image(left_mat, sl.VIEW.LEFT_UNRECTIFIED), f"retrieve left {frame_index}")
            _check_status(zed.retrieve_image(right_mat, sl.VIEW.RIGHT_UNRECTIFIED), f"retrieve right {frame_index}")
            left = _as_gray(left_mat.get_data())
            right = _as_gray(right_mat.get_data())
            match_result = _match_raw_frame(left, right, calibration, alpha1, args.max_matches_per_frame)
            frame_summaries.append(
                {
                    "frame": frame_index,
                    "status": match_result["status"],
                    "high_quality_matches": match_result.get("high_quality_matches", 0),
                    "ratio_test_candidates": match_result.get("ratio_test_candidates", 0),
                    "ransac_inliers": match_result.get("ransac_inliers", 0),
                }
            )
            for match_index, match in enumerate(match_result.get("matches", [])):
                native_depth = _pinhole_depth_from_raw_correspondence(
                    (match["uL"], match["vL"]),
                    (match["uR"], match["vR"]),
                    native_left_k,
                    native_left_d,
                    native_right_k,
                    native_right_d,
                    native_extrinsics,
                )
                custom_depth = _pinhole_depth_from_raw_correspondence(
                    (match["uL"], match["vL"]),
                    (match["uR"], match["vR"]),
                    calibration.left_k,
                    calibration.left_d,
                    calibration.right_k,
                    calibration.right_d,
                    custom_extrinsics,
                )
                record: dict[str, Any] = {
                    "frame": frame_index,
                    "correspondence_id": match_index,
                    "uL": match["uL"],
                    "vL": match["vL"],
                    "uR": match["uR"],
                    "vR": match["vR"],
                    "native_depth": None if native_depth is None else native_depth["z_left_camera_m"],
                    "custom_depth": None if custom_depth is None else custom_depth["z_left_camera_m"],
                    "native_pinhole_distance_from_left_camera_m": None if native_depth is None else native_depth["euclidean_distance_from_left_camera_m"],
                    "custom_pinhole_distance_from_left_camera_m": None if custom_depth is None else custom_depth["euclidean_distance_from_left_camera_m"],
                    "custom_rectified_disparity_alpha1_px": match["custom_rectified_disparity_alpha1_px"],
                    "rectified_vertical_error_alpha1_px": match["rectified_vertical_error_alpha1_px"],
                    "ratio": match["ratio"],
                    "descriptor_distance": match["descriptor_distance"],
                    "refractive_depth": None,
                    "ray_gap_m": None,
                    "view_angle_deg": None,
                }
                correspondence_records.append(record)
    finally:
        zed.close()

    p33, p66 = _add_disparity_bands(correspondence_records)
    plane_normal_provided = args.plane_normal is not None
    model_parameters_known = model.status == "known" and plane_normal_provided
    definitive_refractive = model_parameters_known and args.port_parameters_provenance == "measured"
    if definitive_refractive:
        for record in correspondence_records:
            try:
                result = refractive_stereo_correspondence(
                    (record["uL"], record["vL"]),
                    (record["uR"], record["vR"]),
                    calibration.left_k,
                    calibration.left_d,
                    calibration.right_k,
                    calibration.right_d,
                    model,
                    model,
                    custom_extrinsics,
                )
            except (ValueError, RuntimeError, RefractiveModelNotIdentifiable):
                continue
            record["refractive_depth"] = result["z_left_camera_m"]
            record["ray_gap_m"] = result["ray_gap_m"]
            record["view_angle_deg"] = result["view_angle_from_left_optical_axis_deg"]

    if not model_parameters_known:
        refractive_status = "not_identifiable_without_port_parameters"
    elif args.port_parameters_provenance != "measured":
        refractive_status = "not_identifiable_without_measured_port_parameters"
    else:
        refractive_status = "measured_parameters_supplied"
    correspondence_output = {
        "script": Path(__file__).name,
        "status": refractive_status,
        "definitive_refractive_depth_generated": definitive_refractive,
        "scope": "raw unrectified pixel correspondences; per-pixel inverse distortion; feature ratio test + rectified RANSAC + vertical consistency",
        "coordinate_convention": "camera x=right, y=down, z=forward; common stereo frame is left camera; water ray origin is outer glass/water interface",
        "input": {
            "svo": str(svo_path),
            "intrinsics": str(intrinsics_path),
            "extrinsics": str(extrinsics_path),
            "sdk_version": str(sl.Camera.get_sdk_version()),
            "camera_model": _status_name(info.camera_model),
            "serial_number": int(info.serial_number),
            "image_size": [width, height],
            "total_svo_frames": total_frames,
            "frame_indices": frame_indices,
        },
        "calibration_provenance": {
            "medium": "unknown",
            "housing": "unknown",
            "flat_port_or_dome": "unknown",
            "target_square_size_or_metric_scale": "unknown",
            "evidence_checked": [
                "README.md",
                "Calibration/zed_custom_opencv.yml",
                "Calibration/标定结果/*.yaml",
                "Calibration/标定结果/calib_full_params.xlsx",
                "Calibration/标定结果/*.pdf",
                "git history (two commits; no calibration provenance commit)",
            ],
            "status": "UNKNOWN",
        },
        "port_model": {
            "status": "known" if model_parameters_known else "unknown",
            "parameter_provenance": args.port_parameters_provenance,
            "n_air": model.n_air,
            "n_glass": model.n_glass,
            "n_water": model.n_water,
            "camera_to_inner_interface_m": model.camera_to_inner_interface_m,
            "glass_thickness_m": model.glass_thickness_m,
            "plane_normal_camera": list(model.plane_normal_camera),
            "plane_normal_provided": plane_normal_provided,
            "missing_parameters": [
                *model.missing_parameters,
                *([] if plane_normal_provided else ["plane_normal_camera"]),
            ],
            "no_default_physical_parameters_used": True,
        },
        "native_pinhole_model": {
            "depth_definition": "Z_left_camera from raw SDK calibration and closest-point triangulation",
            "transform_convention": "SDK raw stereo_transform interpreted as right-camera-to-left-camera because X_left=R X_right+T",
            "left": {"K": native_left_k.tolist(), "D": native_left_d.tolist()},
            "right": {"K": native_right_k.tolist(), "D": native_right_d.tolist()},
            "stereo_transform_right_to_left_m": native_transform.tolist(),
        },
        "custom_pinhole_model": {
            "depth_definition": "Z_left_camera from raw custom K/D and closest-point triangulation",
            "transform_convention": "custom YAML R/T interpreted as OpenCV left-to-right; refractive common-frame transform is its inverse",
            "left": {"K": calibration.left_k.tolist(), "D": calibration.left_d.tolist()},
            "right": {"K": calibration.right_k.tolist(), "D": calibration.right_d.tolist()},
            "stereo_transform_left_to_right_mm": {"R": calibration.rotation.tolist(), "T": calibration.translation_mm.tolist()},
        },
        "frame_summaries": frame_summaries,
        "correspondence_count": len(correspondence_records),
        "disparity_percentile_cutoffs_px": {"p33": p33, "p66": p66},
        "correspondences": correspondence_records,
        "notes": [
            "native_depth and custom_depth are raw-pixel pinhole triangulation comparisons, not independent GT",
            "refractive_depth remains null unless all port parameters are supplied with provenance=measured",
            "ordinary rectified SGBM disparity is not used as a general refractive triangulation input",
        ],
    }
    correspondence_path = output_dir / "refractive_correspondence_check.json"
    alpha_path = output_dir / "alpha_invariance_check.json"
    sensitivity_path = output_dir / "refractive_sensitivity.csv"
    correspondence_path.write_text(json.dumps(correspondence_output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    alpha_output = _alpha_invariance(correspondence_records, calibration, alpha0, alpha1)
    alpha_output.update(
        {
            "frame_indices": frame_indices,
            "correspondence_source": str(correspondence_path),
            "same_raw_correspondences": True,
            "depth_formula": "Z_alpha = f_alpha_rect_px * B_alpha_rect_m / d_alpha_px",
        }
    )
    alpha_path.write_text(json.dumps(alpha_output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sensitivity_rows = _sensitivity_rows(correspondence_records, calibration, custom_extrinsics)
    _write_sensitivity_csv(sensitivity_path, sensitivity_rows)

    implementation_status = "PASS" if len(correspondence_records) > 0 and len(frame_indices) >= 10 else "FAIL"
    frame_alignment_status = "PASS" if all(frame_indices[i] < frame_indices[i + 1] for i in range(len(frame_indices) - 1)) else "FAIL"
    p_q_status = "PASS" if abs(alpha1.principal_point_offset_px[0]) < 1.0e-6 and abs(alpha1.principal_point_offset_px[1]) < 1.0e-6 else "FAIL"
    alpha_status = str(alpha_output["status"])
    physical_status = "PARTIAL" if model_parameters_known else "NOT IDENTIFIABLE"
    print(f"Processed frames: {frame_indices}")
    print(f"Correspondences: {len(correspondence_records)}; sensitivity rows: {len(sensitivity_rows)}")
    print(f"Outputs: {correspondence_path}, {sensitivity_path}, {alpha_path}")
    print(f"Elapsed seconds: {time.monotonic() - started:.2f}")
    print("Implementation geometry: " + implementation_status)
    print("Frame alignment: " + frame_alignment_status)
    print("P/Q consistency: " + p_q_status)
    print("Alpha invariance: " + alpha_status)
    print("Physical refractive model: " + physical_status)
    print("Calibration medium provenance: UNKNOWN")
    print("Independent metric GT: NOT AVAILABLE")
    print("Extra x1.333 correction: NOT JUSTIFIED")
    print("Absolute depth accuracy: NOT VERIFIED")
    print("Final verdict: B2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
