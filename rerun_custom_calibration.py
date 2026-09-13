"""Re-run ZED SVO2 tracking and point-cloud generation with custom calibration.

The input calibration is an OpenCV stereo calibration.  It is loaded by the
ZED SDK through ``InitParameters.optional_opencv_calibration_file`` so that
the SDK recomputes rectification, depth, and positional tracking from the
custom camera model instead of using the calibration embedded in the SVO.

The SVO is always replayed sequentially in one pass.  That same pass computes
a pose for every frame and fuses SDK XYZRGBA data from uniformly distributed
valid mapping frames into a bounded WORLD-frame binary PLY.  A single pass is
important when Area Memory or IMU fusion is enabled: an independent second
replay can legitimately produce a different tracking state or trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
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

    current_path = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join(
        [*existing_paths, *(path for path in current_path if path)]
    )

    if hasattr(os, "add_dll_directory"):
        handles = [os.add_dll_directory(path) for path in existing_paths]
        _prepare_windows_dll_search_path._dll_handles = handles  # type: ignore[attr-defined]


_prepare_windows_dll_search_path()

import cv2  # noqa: E402  (DLL search path must be prepared first)
import numpy as np  # noqa: E402
import pyzed.sl as sl  # noqa: E402


DEFAULT_SVO = Path(__file__).resolve().with_name("20260802_150233.svo2")
DEFAULT_CALIBRATION_DIR = (
    Path(__file__).resolve().parent / "Calibration" / "标定结果"
)
DEFAULT_OPENCV_CALIBRATION = (
    Path(__file__).resolve().parent / "Calibration" / "zed_custom_opencv.yml"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "output"
    / "20260802_150233_custom_gen1_full"
)

POSE_HEADER = [
    "frame_index",
    "svo_position",
    "timestamp_ns",
    "tracking_state",
    "pose_valid",
    "pose_confidence",
    "tx",
    "ty",
    "tz",
    "qx",
    "qy",
    "qz",
    "qw",
    *(f"m{row}{column}" for row in range(4) for column in range(4)),
]
MAPPING_HEADER = [
    "map_index",
    "frame_index",
    "svo_position",
    "timestamp_ns",
    "tracking_state",
    "pose_valid",
    "point_count",
]

HARD_OUTPUT_LIMIT_BYTES = 2_000_000_000
SAFE_OUTPUT_LIMIT_BYTES = 1_900_000_000
PLY_HEADER_RESERVE_BYTES = 512
AREA_EXPORT_TIMEOUT_SECONDS = 1800.0
# A packed binary PLY vertex: 3 float32 coordinates + 3 uint8 RGB values.
POINT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)
assert POINT_DTYPE.itemsize == 15


@dataclass(frozen=True)
class Calibration:
    """OpenCV stereo calibration in the convention expected by ZED."""

    width: int
    height: int
    k_left: np.ndarray
    k_right: np.ndarray
    d_left: np.ndarray
    d_right: np.ndarray
    r: np.ndarray
    t_mm: np.ndarray
    direction: str


@dataclass
class PoseRecord:
    frame_index: int
    svo_position: int
    timestamp_ns: int
    tracking_state: str
    pose_valid: bool
    pose_confidence: float
    values: list[float]
    matrix: np.ndarray | None

    def csv_row(self) -> list[Any]:
        return [
            self.frame_index,
            self.svo_position,
            self.timestamp_ns,
            self.tracking_state,
            int(self.pose_valid),
            _csv_number(self.pose_confidence),
            *(_csv_number(value) for value in self.values),
        ]


def _status_name(status: Any) -> str:
    return str(status).rsplit(".", 1)[-1]


def _check_status(status: Any, operation: str) -> None:
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"{operation} failed: {_status_name(status)}")


def _csv_number(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "NaN"
    return f"{value:.12g}"


def _float_list(value: Any) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=np.float64).reshape(-1)]


def _matrix4(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Non-finite 4x4 matrix")
    return matrix


def _parse_numbers(text: str) -> list[float]:
    pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    return [float(value) for value in re.findall(pattern, text)]


def _source_section(text: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^\s*{re.escape(name)}\s*:\s*(.*?)(?=^\s*(?:left|right)\s*:|\Z)",
        text,
    )
    if match is None:
        raise RuntimeError(f"Missing {name!r} section in camera calibration")
    return match.group(1)


def _source_scalar(section: str, name: str) -> float:
    match = re.search(
        rf"(?m)^\s*{re.escape(name)}\s*:\s*([-+0-9.eE]+)", section
    )
    if match is None:
        raise RuntimeError(f"Missing {name!r} in camera calibration")
    return float(match.group(1))


def _source_list(section: str, name: str) -> list[float]:
    match = re.search(
        rf"(?m)^\s*{re.escape(name)}\s*:\s*\[([^\]]+)\]", section
    )
    if match is None:
        raise RuntimeError(f"Missing {name!r} list in camera calibration")
    return _parse_numbers(match.group(1))


def _load_source_calibration(
    intrinsics_path: Path,
    extrinsics_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    intrinsics_text = intrinsics_path.read_text(encoding="utf-8")
    extrinsics_text = extrinsics_path.read_text(encoding="utf-8")

    cameras: dict[str, Any] = {}
    for name in ("left", "right"):
        section = _source_section(intrinsics_text, name)
        distortion = _source_list(section, "distortion_coefficients")
        if len(distortion) != 5:
            raise RuntimeError(
                f"Expected five OpenCV radial-tangential coefficients for {name}, "
                f"got {len(distortion)}"
            )
        cameras[name] = {
            "fx": _source_scalar(section, "fx"),
            "fy": _source_scalar(section, "fy"),
            "cx": _source_scalar(section, "cx"),
            "cy": _source_scalar(section, "cy"),
            "distortion_coefficients": distortion,
        }

    r_match = re.search(
        r"(?ms)^\s*R\s*:\s*(.*?)(?=^\s*T\s*:)", extrinsics_text
    )
    if r_match is None:
        raise RuntimeError("Missing R matrix in stereo calibration")
    r_rows = [
        _parse_numbers(row)
        for row in re.findall(r"\[([^\]]+)\]", r_match.group(1))
    ]
    if len(r_rows) != 3 or any(len(row) != 3 for row in r_rows):
        raise RuntimeError(f"Expected a 3x3 R matrix, got {r_rows}")

    t_values = _source_list(extrinsics_text, "T")
    if len(t_values) != 3:
        raise RuntimeError(f"Expected a 3-vector T, got {t_values}")

    extrinsics = {
        "baseline_mm": _source_scalar(extrinsics_text, "baseline"),
        "reprojection_error_px": _source_scalar(
            extrinsics_text, "reprojection_error"
        ),
        "rotation": np.asarray(r_rows, dtype=np.float64),
        "translation_mm": np.asarray(t_values, dtype=np.float64),
        "annotation": "right_cam_to_left",
        "scale_error_percent": _source_scalar(
            extrinsics_text, "scale_error_percent"
        ),
    }
    return cameras, extrinsics


def _make_calibration(
    cameras: dict[str, Any],
    extrinsics: dict[str, Any],
    width: int,
    height: int,
    inverse: bool = False,
) -> Calibration:
    r_source = np.asarray(extrinsics["rotation"], dtype=np.float64)
    t_source = np.asarray(extrinsics["translation_mm"], dtype=np.float64)
    if inverse:
        r_matrix = r_source.T
        t_mm = -r_source.T @ t_source
        direction = "inverse_of_declared_right_to_left"
    else:
        r_matrix = r_source
        t_mm = t_source
        direction = "source_R_T_as_OpenCV_left_to_right"

    rotation_vector = cv2.Rodrigues(r_matrix)[0].reshape(-1)
    return Calibration(
        width=int(width),
        height=int(height),
        k_left=np.asarray(
            [
                [cameras["left"]["fx"], 0.0, cameras["left"]["cx"]],
                [0.0, cameras["left"]["fy"], cameras["left"]["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        k_right=np.asarray(
            [
                [cameras["right"]["fx"], 0.0, cameras["right"]["cx"]],
                [0.0, cameras["right"]["fy"], cameras["right"]["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        d_left=np.asarray(cameras["left"]["distortion_coefficients"] + [0.0] * 9),
        d_right=np.asarray(cameras["right"]["distortion_coefficients"] + [0.0] * 9),
        r=rotation_vector.astype(np.float64),
        t_mm=np.asarray(t_mm, dtype=np.float64),
        direction=direction,
    )


def _write_opencv_calibration(path: Path, calibration: Calibration) -> None:
    def matrix_data(matrix: np.ndarray) -> str:
        return ", ".join(f"{float(value):.16g}" for value in matrix.reshape(-1))

    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""%YAML:1.0
---
Size: [ {calibration.width}, {calibration.height} ]
K_LEFT: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ {matrix_data(calibration.k_left)} ]
K_RIGHT: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ {matrix_data(calibration.k_right)} ]
D_LEFT: !!opencv-matrix
   rows: 1
   cols: 14
   dt: d
   data: [ {matrix_data(calibration.d_left)} ]
D_RIGHT: !!opencv-matrix
   rows: 1
   cols: 14
   dt: d
   data: [ {matrix_data(calibration.d_right)} ]
R: !!opencv-matrix
   rows: 3
   cols: 1
   dt: d
   data: [ {matrix_data(calibration.r.reshape(3, 1))} ]
T: !!opencv-matrix
   rows: 3
   cols: 1
   dt: d
   data: [ {matrix_data(calibration.t_mm.reshape(3, 1))} ]
"""
    path.write_text(text, encoding="utf-8")


def _read_opencv_calibration(path: Path) -> Calibration:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise RuntimeError(f"Could not open OpenCV calibration: {path}")
    try:
        size_node = storage.getNode("Size")
        if not size_node.isSeq() or size_node.size() != 2:
            raise RuntimeError("OpenCV calibration Size must be [width, height]")
        width = int(round(size_node.at(0).real()))
        height = int(round(size_node.at(1).real()))

        def read_matrix(name: str) -> np.ndarray:
            value = storage.getNode(name).mat()
            if value is None:
                raise RuntimeError(f"Missing matrix {name} in {path}")
            value = np.asarray(value, dtype=np.float64)
            if not np.isfinite(value).all():
                raise RuntimeError(f"Matrix {name} contains non-finite values")
            return value

        k_left = read_matrix("K_LEFT")
        k_right = read_matrix("K_RIGHT")
        d_left = read_matrix("D_LEFT").reshape(-1)
        d_right = read_matrix("D_RIGHT").reshape(-1)
        r = read_matrix("R").reshape(-1)
        t_mm = read_matrix("T").reshape(-1)
    finally:
        storage.release()

    if k_left.shape != (3, 3) or k_right.shape != (3, 3):
        raise RuntimeError("K_LEFT and K_RIGHT must be 3x3 matrices")
    if d_left.size < 5 or d_right.size < 5:
        raise RuntimeError("D_LEFT and D_RIGHT must contain at least five values")
    if r.size != 3 or t_mm.size != 3:
        raise RuntimeError("R and T must be 3-vectors")

    return Calibration(
        width=width,
        height=height,
        k_left=k_left,
        k_right=k_right,
        d_left=d_left,
        d_right=d_right,
        r=r,
        t_mm=t_mm,
        direction="loaded_opencv_file",
    )


def _validate_calibration_against_source(
    calibration: Calibration,
    cameras: dict[str, Any],
    extrinsics: dict[str, Any],
) -> None:
    if calibration.width <= 0 or calibration.height <= 0:
        raise RuntimeError("Calibration image size is invalid")
    for name, matrix in (("K_LEFT", calibration.k_left), ("K_RIGHT", calibration.k_right)):
        if not np.isfinite(matrix).all() or np.any(np.diag(matrix)[:2] <= 0):
            raise RuntimeError(f"{name} is invalid")
    rotation = cv2.Rodrigues(calibration.r)[0]
    rotation_error = np.max(np.abs(rotation.T @ rotation - np.eye(3)))
    if rotation_error > 1e-6 or abs(np.linalg.det(rotation) - 1.0) > 1e-6:
        raise RuntimeError(
            f"Calibration rotation is not a proper rotation: error={rotation_error}"
        )
    if calibration.t_mm[0] == 0 or np.linalg.norm(calibration.t_mm) <= 0:
        raise RuntimeError("Calibration translation is invalid")

    expected_k_left = np.asarray(
        [
            [cameras["left"]["fx"], 0.0, cameras["left"]["cx"]],
            [0.0, cameras["left"]["fy"], cameras["left"]["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    expected_k_right = np.asarray(
        [
            [cameras["right"]["fx"], 0.0, cameras["right"]["cx"]],
            [0.0, cameras["right"]["fy"], cameras["right"]["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    expected_r = np.asarray(extrinsics["rotation"], dtype=np.float64)
    file_r = cv2.Rodrigues(calibration.r)[0]
    expected_t = np.asarray(extrinsics["translation_mm"], dtype=np.float64)

    checks = {
        "K_LEFT": (calibration.k_left, expected_k_left),
        "K_RIGHT": (calibration.k_right, expected_k_right),
        "D_LEFT": (calibration.d_left[:5], np.asarray(cameras["left"]["distortion_coefficients"])),
        "D_RIGHT": (calibration.d_right[:5], np.asarray(cameras["right"]["distortion_coefficients"])),
        "R": (file_r, expected_r),
        "T": (calibration.t_mm, expected_t),
    }
    for name, (actual, expected) in checks.items():
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        if error > 1e-3:
            raise RuntimeError(f"{name} in OpenCV calibration differs from source by {error}")


def _make_init(
    svo_path: Path,
    calibration_path: Path,
    depth_mode: Any,
    sdk_verbose: int = 0,
) -> sl.InitParameters:
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = depth_mode
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.camera_disable_self_calib = True
    init.optional_opencv_calibration_file = str(calibration_path)
    init.depth_stabilization = 0
    init.enable_image_enhancement = False
    init.sdk_verbose = sdk_verbose
    return init


def _make_tracking_parameters() -> sl.PositionalTrackingParameters:
    params = sl.PositionalTrackingParameters()
    params.mode = sl.POSITIONAL_TRACKING_MODE.GEN_1
    params.enable_area_memory = True
    params.enable_imu_fusion = True
    params.enable_pose_smoothing = False
    params.set_gravity_as_origin = True
    params.set_floor_as_origin = False
    params.set_as_static = False
    return params


def _camera_info_metadata(info: Any) -> dict[str, Any]:
    configuration = info.camera_configuration
    resolution = configuration.resolution
    raw = configuration.calibration_parameters_raw
    rectified = configuration.calibration_parameters

    def camera_metadata(camera: Any) -> dict[str, Any]:
        return {
            "fx": float(camera.fx),
            "fy": float(camera.fy),
            "cx": float(camera.cx),
            "cy": float(camera.cy),
            "distortion_coefficients": _float_list(camera.disto),
            "lens_distortion_model": str(camera.lens_distortion_model),
        }

    return {
        "camera_model": _status_name(info.camera_model),
        "serial_number": int(info.serial_number),
        "resolution": {"width": int(resolution.width), "height": int(resolution.height)},
        "fps": float(configuration.fps),
        "raw_calibration": {
            "left": camera_metadata(raw.left_cam),
            "right": camera_metadata(raw.right_cam),
            "stereo_transform": _matrix4(raw.stereo_transform.m).tolist(),
        },
        "rectified_calibration": {
            "left": camera_metadata(rectified.left_cam),
            "right": camera_metadata(rectified.right_cam),
            "stereo_transform": _matrix4(rectified.stereo_transform.m).tolist(),
        },
    }


def _verify_sdk_override(info: Any, calibration: Calibration) -> dict[str, Any]:
    metadata = _camera_info_metadata(info)
    raw = info.camera_configuration.calibration_parameters_raw
    raw_left = raw.left_cam
    raw_right = raw.right_cam
    expected_r = cv2.Rodrigues(calibration.r)[0]
    expected_t_m = calibration.t_mm / 1000.0
    expected_internal = np.eye(4, dtype=np.float64)
    expected_internal[:3, :3] = expected_r.T
    expected_internal[:3, 3] = -expected_r.T @ expected_t_m
    observed_internal = _matrix4(raw.stereo_transform.m)

    errors = {
        "left_fx_px": abs(float(raw_left.fx) - float(calibration.k_left[0, 0])),
        "left_fy_px": abs(float(raw_left.fy) - float(calibration.k_left[1, 1])),
        "left_cx_px": abs(float(raw_left.cx) - float(calibration.k_left[0, 2])),
        "left_cy_px": abs(float(raw_left.cy) - float(calibration.k_left[1, 2])),
        "right_fx_px": abs(float(raw_right.fx) - float(calibration.k_right[0, 0])),
        "right_fy_px": abs(float(raw_right.fy) - float(calibration.k_right[1, 1])),
        "right_cx_px": abs(float(raw_right.cx) - float(calibration.k_right[0, 2])),
        "right_cy_px": abs(float(raw_right.cy) - float(calibration.k_right[1, 2])),
        "stereo_transform_max_abs": float(
            np.max(np.abs(observed_internal - expected_internal))
        ),
    }
    if max(errors.values()) > 2e-3:
        raise RuntimeError(
            "ZED SDK did not expose the requested custom calibration exactly: "
            f"{errors}"
        )
    metadata["custom_calibration_override_verification"] = {
        "passed": True,
        "expected_sdk_internal_stereo_transform": expected_internal.tolist(),
        "observed_sdk_internal_stereo_transform": observed_internal.tolist(),
        "max_abs_errors": errors,
    }
    return metadata


def _probe_svo_resolution(
    svo_path: Path,
    calibration_path: Path,
) -> tuple[int, int, int]:
    zed = sl.Camera()
    status = zed.open(
        _make_init(svo_path, calibration_path, sl.DEPTH_MODE.NONE)
    )
    _check_status(status, "open SVO for resolution probe")
    try:
        info = zed.get_camera_information()
        resolution = info.camera_configuration.resolution
        return int(resolution.width), int(resolution.height), int(zed.get_svo_number_of_frames())
    finally:
        zed.close()


def _as_gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _as_bgr(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _probe_rectification(
    svo_path: Path,
    calibration: Calibration,
    frames: int,
) -> dict[str, Any]:
    """Measure feature alignment after applying an external calibration candidate."""

    r_matrix = cv2.Rodrigues(calibration.r)[0]
    t = calibration.t_mm.reshape(3, 1)
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        calibration.k_left,
        calibration.d_left,
        calibration.k_right,
        calibration.d_right,
        (calibration.width, calibration.height),
        r_matrix,
        t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
        newImageSize=(calibration.width, calibration.height),
    )
    map_l = cv2.initUndistortRectifyMap(
        calibration.k_left,
        calibration.d_left,
        r1,
        p1,
        (calibration.width, calibration.height),
        cv2.CV_32FC1,
    )
    map_r = cv2.initUndistortRectifyMap(
        calibration.k_right,
        calibration.d_right,
        r2,
        p2,
        (calibration.width, calibration.height),
        cv2.CV_32FC1,
    )

    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.camera_disable_self_calib = True
    init.sdk_verbose = 0
    zed = sl.Camera()
    _check_status(zed.open(init), "open SVO for rectification probe")
    left_mat = sl.Mat()
    right_mat = sl.Mat()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    orb = cv2.ORB_create(nfeatures=1200)
    vertical_errors: list[float] = []
    positive_disparity: list[float] = []
    match_counts: list[int] = []
    try:
        for _ in range(max(1, frames)):
            status = zed.grab()
            if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            _check_status(status, "grab during rectification probe")
            _check_status(
                zed.retrieve_image(left_mat, sl.VIEW.LEFT_UNRECTIFIED),
                "retrieve unrectified left image",
            )
            _check_status(
                zed.retrieve_image(right_mat, sl.VIEW.RIGHT_UNRECTIFIED),
                "retrieve unrectified right image",
            )
            left = cv2.remap(
                _as_gray(left_mat.get_data()), map_l[0], map_l[1], cv2.INTER_LINEAR
            )
            right = cv2.remap(
                _as_gray(right_mat.get_data()), map_r[0], map_r[1], cv2.INTER_LINEAR
            )
            key_l, desc_l = orb.detectAndCompute(left, None)
            key_r, desc_r = orb.detectAndCompute(right, None)
            if desc_l is None or desc_r is None:
                match_counts.append(0)
                continue
            knn = matcher.knnMatch(desc_l, desc_r, k=2)
            good = [
                pair[0]
                for pair in knn
                if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
            ]
            if not good:
                match_counts.append(0)
                continue
            ys = np.asarray(
                [abs(key_l[m.queryIdx].pt[1] - key_r[m.trainIdx].pt[1]) for m in good],
                dtype=np.float64,
            )
            disparities = np.asarray(
                [key_l[m.queryIdx].pt[0] - key_r[m.trainIdx].pt[0] for m in good],
                dtype=np.float64,
            )
            # Robustly score only the best half of descriptor matches.
            keep = np.argsort(ys)[: max(1, len(ys) // 2)]
            vertical_errors.append(float(np.median(ys[keep])))
            positive_disparity.append(float(np.mean(disparities[keep] > 0)))
            match_counts.append(len(good))
    finally:
        zed.close()

    result: dict[str, Any] = {
        "direction": calibration.direction,
        "frames_tested": len(match_counts),
        "match_counts": match_counts,
        "median_vertical_error_px": (
            float(np.median(vertical_errors)) if vertical_errors else None
        ),
        "positive_disparity_fraction": (
            float(np.median(positive_disparity)) if positive_disparity else None
        ),
    }
    return result


def _probe_depth_validity(
    svo_path: Path,
    calibration_path: Path,
    frames: int,
) -> dict[str, Any]:
    zed = sl.Camera()
    status = zed.open(
        _make_init(svo_path, calibration_path, sl.DEPTH_MODE.NEURAL)
    )
    if status > sl.ERROR_CODE.SUCCESS:
        return {"open_status": _status_name(status), "valid_depth_ratios": []}
    depth_mat = sl.Mat()
    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 30
    # The underwater sequence has very low texture-confidence values in large
    # regions.  A threshold of 30 removes every depth pixel on this recording,
    # so retain the full texture range and use the geometric confidence filter
    # below for conservative filtering.
    runtime.texture_confidence_threshold = 100
    ratios: list[float] = []
    try:
        for _ in range(max(1, frames)):
            status = zed.grab(runtime)
            if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if status != sl.ERROR_CODE.SUCCESS:
                break
            _check_status(
                zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH),
                "retrieve depth during direction probe",
            )
            depth = np.asarray(depth_mat.get_data())
            valid = np.isfinite(depth) & (depth > 0)
            ratios.append(float(np.mean(valid)))
    finally:
        zed.close()
    return {
        "open_status": "SUCCESS",
        "valid_depth_ratios": ratios,
        "median_valid_depth_ratio": float(np.median(ratios)) if ratios else 0.0,
    }


def _select_direction_and_calibration(
    svo_path: Path,
    cameras: dict[str, Any],
    extrinsics: dict[str, Any],
    width: int,
    height: int,
    output_path: Path,
    probe_frames: int,
    skip_probe: bool,
) -> tuple[Calibration, dict[str, Any]]:
    candidates = [
        _make_calibration(cameras, extrinsics, width, height, inverse=False),
        _make_calibration(cameras, extrinsics, width, height, inverse=True),
    ]
    diagnostics: dict[str, Any] = {
        "source_extrinsic_annotation": extrinsics["annotation"],
        "candidates": [],
    }

    if skip_probe:
        selected = candidates[0]
        diagnostics["selection"] = {
            "selected": selected.direction,
            "reason": "direction probe explicitly skipped; source R/T used",
        }
        _write_opencv_calibration(output_path, selected)
        return selected, diagnostics

    with tempfile.TemporaryDirectory(prefix="zed_calibration_probe_") as temp_dir:
        scores: list[tuple[float, int, Calibration]] = []
        for index, candidate in enumerate(candidates):
            candidate_path = Path(temp_dir) / f"candidate_{index}.yml"
            _write_opencv_calibration(candidate_path, candidate)
            loaded = _read_opencv_calibration(candidate_path)
            geometry = _probe_rectification(svo_path, loaded, probe_frames)
            depth = _probe_depth_validity(svo_path, candidate_path, probe_frames)
            diagnostics["candidates"].append(
                {
                    "direction": candidate.direction,
                    "rotation_vector": candidate.r.tolist(),
                    "translation_mm": candidate.t_mm.tolist(),
                    "geometry": geometry,
                    "depth": depth,
                }
            )
            y_error = geometry["median_vertical_error_px"]
            positive = geometry["positive_disparity_fraction"]
            depth_ratio = depth.get("median_valid_depth_ratio", 0.0)
            if y_error is None:
                score = -1e9
            else:
                score = (
                    -float(y_error)
                    + 0.5 * float(positive or 0.0)
                    + 0.25 * math.log1p(max(0.0, float(depth_ratio)) * 100.0)
                )
            scores.append((score, index, candidate))

        scores.sort(key=lambda item: item[0], reverse=True)
        best_score, best_index, selected = scores[0]
        second_score = scores[1][0]
        best_diag = diagnostics["candidates"][best_index]
        diagnostics["selection"] = {
            "selected": selected.direction,
            "selected_candidate_index": best_index,
            "score": float(best_score),
            "score_margin": float(best_score - second_score),
        }

        best_geometry = best_diag["geometry"]
        best_depth = best_diag["depth"]
        if (
            best_geometry["median_vertical_error_px"] is None
            or best_geometry["match_counts"]
            and max(best_geometry["match_counts"]) < 10
            or best_depth.get("open_status") != "SUCCESS"
            or best_depth.get("median_valid_depth_ratio", 0.0) <= 0.0
        ):
            raise RuntimeError(
                "Calibration direction could not be validated from the SVO probe: "
                f"{json.dumps(diagnostics, ensure_ascii=False)}"
            )
        if best_score - second_score < 0.05:
            raise RuntimeError(
                "Calibration direction is ambiguous; refusing to run with an "
                f"unverified R/T convention: {json.dumps(diagnostics, ensure_ascii=False)}"
            )

    _write_opencv_calibration(output_path, selected)
    return selected, diagnostics


def _pose_values(
    pose: sl.Pose,
    tracking_state: Any,
) -> tuple[bool, float, list[float], np.ndarray | None]:
    is_ok = tracking_state == sl.POSITIONAL_TRACKING_STATE.OK
    pose_valid = bool(pose.valid) if is_ok else False
    pose_confidence = float(pose.pose_confidence) if is_ok else float("nan")
    if not is_ok or not pose_valid:
        return pose_valid, pose_confidence, [float("nan")] * 23, None

    translation = np.asarray(pose.get_translation().get(), dtype=np.float64).reshape(-1)
    orientation = np.asarray(pose.get_orientation().get(), dtype=np.float64).reshape(-1)
    matrix = _matrix4(pose.pose_data().m)
    values = [*translation[:3], *orientation[:4], *matrix.reshape(-1)]
    if len(values) != 23:
        raise RuntimeError(f"Unexpected pose value count: {len(values)}")
    return pose_valid, pose_confidence, values, matrix


def _uniform_sample(values: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    if count > len(values):
        raise RuntimeError(f"Cannot select {count} frames from only {len(values)} valid frames")
    if count == 1:
        return [values[0]]
    return [values[(index * (len(values) - 1)) // (count - 1)] for index in range(count)]


def _choose_pixel_stride(width: int, height: int, frame_count: int, requested: int) -> tuple[int, int]:
    stride = max(1, requested)
    while True:
        points_per_frame = ((width + stride - 1) // stride) * (
            (height + stride - 1) // stride
        )
        estimated_points = points_per_frame * frame_count
        estimated_bytes = estimated_points * POINT_DTYPE.itemsize + PLY_HEADER_RESERVE_BYTES
        if estimated_bytes < SAFE_OUTPUT_LIMIT_BYTES:
            return stride, estimated_points
        stride += 1


def _ply_header(point_count: int) -> bytes:
    return (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {point_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")


def _append_xyzrgba_frame(
    raw_file: Any,
    point_cloud_mat: sl.Mat,
    left_mat: sl.Mat,
    pose_matrix: np.ndarray,
    pixel_stride: int,
    min_depth: float,
    max_depth: float,
    current_bytes: int,
) -> tuple[int, int]:
    point_cloud = np.asarray(point_cloud_mat.get_data())
    left_image = _as_bgr(left_mat.get_data())
    if point_cloud.ndim != 3 or point_cloud.shape[2] < 3:
        raise RuntimeError(f"Unexpected XYZRGBA shape: {point_cloud.shape}")
    if left_image.shape[:2] != point_cloud.shape[:2]:
        raise RuntimeError(
            f"XYZRGBA/left image size mismatch: {point_cloud.shape[:2]} != {left_image.shape[:2]}"
        )

    sampled_xyz = np.asarray(point_cloud[::pixel_stride, ::pixel_stride, :3], dtype=np.float32)
    sampled_bgr = np.asarray(left_image[::pixel_stride, ::pixel_stride], dtype=np.uint8)
    valid = np.isfinite(sampled_xyz).all(axis=2)
    optical_depth = -sampled_xyz[:, :, 2]
    valid &= np.isfinite(optical_depth)
    valid &= optical_depth > np.float32(min_depth)
    if np.isfinite(max_depth):
        valid &= optical_depth <= np.float32(max_depth)
    if not np.any(valid):
        return 0, current_bytes

    points_camera = sampled_xyz[valid].astype(np.float64)
    rotation = pose_matrix[:3, :3]
    translation = pose_matrix[:3, 3]
    # pose_data().m maps left-camera column vectors to WORLD.  With NumPy row
    # vectors this is p_camera @ R.T + t.
    points_world = points_camera @ rotation.T + translation
    finite = np.isfinite(points_world).all(axis=1)
    if not np.any(finite):
        return 0, current_bytes
    points_world = points_world[finite]
    colors_rgb = sampled_bgr[valid][finite][:, ::-1]

    records = np.empty(len(points_world), dtype=POINT_DTYPE)
    records["x"] = points_world[:, 0].astype(np.float32)
    records["y"] = points_world[:, 1].astype(np.float32)
    records["z"] = points_world[:, 2].astype(np.float32)
    records["red"] = colors_rgb[:, 0]
    records["green"] = colors_rgb[:, 1]
    records["blue"] = colors_rgb[:, 2]

    next_bytes = current_bytes + records.nbytes
    if next_bytes + PLY_HEADER_RESERVE_BYTES >= HARD_OUTPUT_LIMIT_BYTES:
        raise RuntimeError(
            "Point cloud would exceed the hard 2,000,000,000-byte limit; "
            "increase pixel stride or reduce mapping frames."
        )
    raw_file.write(records.tobytes(order="C"))
    return len(records), next_bytes


def _save_area_map(
    zed: sl.Camera,
    path: Path,
    allow_empty: bool = False,
) -> dict[str, Any]:
    status = zed.save_area_map(str(path))
    _check_status(status, "save custom Area Memory map")
    # A full 35,855-frame replay can build a substantially larger Area Memory
    # graph than a short smoke test.  Export is asynchronous in the SDK and
    # can take several minutes; do not mistake that normal work for failure.
    deadline = time.monotonic() + AREA_EXPORT_TIMEOUT_SECONDS
    next_report = time.monotonic() + 30.0
    export_state = zed.get_area_export_state()
    while export_state == sl.AREA_EXPORTING_STATE.RUNNING:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Timed out while exporting the custom Area Memory map after "
                f"{AREA_EXPORT_TIMEOUT_SECONDS:.0f}s; "
                f"state={_status_name(export_state)}"
            )
        if time.monotonic() >= next_report:
            remaining = max(0.0, deadline - time.monotonic())
            print(
                "Area Memory export still running; "
                f"remaining timeout {remaining:.0f}s",
                flush=True,
            )
            next_report = time.monotonic() + 30.0
        time.sleep(0.5)
        export_state = zed.get_area_export_state()
    if export_state == sl.AREA_EXPORTING_STATE.FILE_EMPTY and allow_empty:
        return {
            "path": str(path),
            "saved": False,
            "state": _status_name(export_state),
            "reason": "short smoke-test replay did not build an Area Memory map",
        }
    if export_state != sl.AREA_EXPORTING_STATE.SUCCESS:
        raise RuntimeError(
            f"Area Memory export failed: {_status_name(export_state)}"
        )
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"ZED SDK reported success but area map is missing: {path}")
    return {
        "path": str(path),
        "saved": True,
        "state": _status_name(export_state),
        "size_bytes": int(path.stat().st_size),
    }


def _run_mapping_pass(
    svo_path: Path,
    calibration_path: Path,
    output_dir: Path,
    mapping_positions: list[int],
    pose_export_positions: set[int] | None,
    pixel_stride: int,
    min_depth: float,
    max_depth: float,
    max_source_frames: int,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    """Replay the SVO once and write pose plus the selected WORLD point cloud.

    ``mapping_positions`` are nominal, uniformly distributed source positions.
    They are generated before this pass because the SVO frame count is known.
    If tracking is temporarily invalid exactly at a nominal position, the
    first later valid pose is used for that mapping slot.  This preserves the
    requested number of valid mapping frames without ever jumping over SVO
    frames during tracking.
    """

    zed = sl.Camera()
    _check_status(
        zed.open(_make_init(svo_path, calibration_path, sl.DEPTH_MODE.NEURAL)),
        "open SVO for single sequential custom pass",
    )
    pose_csv_tmp = output_dir / "pose_world.csv.tmp"
    mapping_csv_tmp = output_dir / "mapping_frame_index.csv.tmp"
    raw_path = output_dir / "scene_world_rgb.ply.records.tmp"
    ply_tmp = output_dir / "scene_world_rgb.ply.tmp"
    pose_csv_path = output_dir / "pose_world.csv"
    mapping_csv_path = output_dir / "mapping_frame_index.csv"
    area_map_path = output_dir / "area_map_custom.area"
    point_count = 0
    raw_bytes = 0
    mapping_rows: list[list[Any]] = []
    records: list[PoseRecord] = []
    next_mapping_index = 0
    pose_rows_written = 0
    started = time.monotonic()
    total_frames = 0
    area_metadata: dict[str, Any] | None = None

    try:
        info = zed.get_camera_information()
        total_frames = int(zed.get_svo_number_of_frames())
        replay_count = (
            min(max_source_frames, total_frames)
            if max_source_frames > 0
            else total_frames
        )
        if not mapping_positions:
            raise RuntimeError("At least one mapping position is required")
        if any(position < 0 or position >= replay_count for position in mapping_positions):
            raise RuntimeError(
                "Mapping positions are outside the replay range: "
                f"positions={mapping_positions[:3]}...{mapping_positions[-3:]}, "
                f"replay_count={replay_count}"
            )
        if (int(info.camera_configuration.resolution.width),
                int(info.camera_configuration.resolution.height)) != (1920, 1080):
            raise RuntimeError(
                "Unexpected SDK output resolution during mapping pass: "
                f"{info.camera_configuration.resolution.width}x"
                f"{info.camera_configuration.resolution.height}"
            )
        _check_status(
            zed.enable_positional_tracking(_make_tracking_parameters()),
            "enable custom GEN_1 positional tracking for single pass",
        )
        runtime = sl.RuntimeParameters()
        runtime.confidence_threshold = 30
        runtime.texture_confidence_threshold = 100
        runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA
        pose = sl.Pose()
        point_cloud_mat = sl.Mat()
        left_mat = sl.Mat()
        with (
            pose_csv_tmp.open("w", newline="", encoding="utf-8") as pose_file,
            raw_path.open("wb") as raw_file,
        ):
            pose_writer = csv.writer(pose_file)
            pose_writer.writerow(POSE_HEADER)
            while True:
                status = zed.grab(runtime)
                if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break
                _check_status(status, "grab during single sequential custom pass")
                frame_index = len(records)
                source_position = int(zed.get_svo_position())
                if source_position != frame_index:
                    raise RuntimeError(
                        f"Single-pass SVO position mismatch: frame={frame_index}, "
                        f"svo_position={source_position}"
                    )
                timestamp_ns = int(
                    zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                )
                tracking_state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
                pose_valid, pose_confidence, values, matrix = _pose_values(
                    pose, tracking_state
                )
                record = PoseRecord(
                    frame_index=frame_index,
                    svo_position=source_position,
                    timestamp_ns=timestamp_ns,
                    tracking_state=_status_name(tracking_state),
                    pose_valid=pose_valid,
                    pose_confidence=pose_confidence,
                    values=values,
                    matrix=matrix,
                )
                records.append(record)

                # The target list is sorted.  Normally source_position equals
                # the target.  If that target has an invalid pose, the next
                # valid frame is accepted as the same mapping slot.
                mapping_target = None
                if (
                    next_mapping_index < len(mapping_positions)
                    and source_position >= mapping_positions[next_mapping_index]
                    and record.pose_valid
                ):
                    mapping_target = mapping_positions[next_mapping_index]

                # The SVO is always replayed frame-by-frame, but the caller can
                # restrict the persisted pose CSV to a uniform subset.  If a
                # nominal mapping target had to advance to a later valid frame,
                # keep that actual mapping pose as well so the two exports stay
                # cross-referenceable.
                if (
                    pose_export_positions is None
                    or source_position in pose_export_positions
                    or mapping_target is not None
                ):
                    pose_writer.writerow(record.csv_row())
                    pose_rows_written += 1

                if mapping_target is not None:
                    if matrix is None or not record.pose_valid:
                        raise RuntimeError(
                            f"Selected mapping frame has invalid pose: {source_position}"
                        )
                    _check_status(
                        zed.retrieve_image(left_mat, sl.VIEW.LEFT),
                        f"retrieve custom rectified left image at {source_position}",
                    )
                    _check_status(
                        zed.retrieve_measure(point_cloud_mat, sl.MEASURE.XYZRGBA),
                        f"retrieve custom XYZRGBA at {source_position}",
                    )
                    added, raw_bytes = _append_xyzrgba_frame(
                        raw_file=raw_file,
                        point_cloud_mat=point_cloud_mat,
                        left_mat=left_mat,
                        pose_matrix=matrix,
                        pixel_stride=pixel_stride,
                        min_depth=min_depth,
                        max_depth=max_depth,
                        current_bytes=raw_bytes,
                    )
                    point_count += added
                    mapping_rows.append(
                        [
                            len(mapping_rows),
                            frame_index,
                            source_position,
                            timestamp_ns,
                            record.tracking_state,
                            int(record.pose_valid),
                            added,
                        ]
                    )
                    next_mapping_index += 1

                if frame_index % 1000 == 0:
                    elapsed = time.monotonic() - started
                    print(
                        f"Replay: {frame_index}/{total_frames} frames, "
                        f"mapped={len(mapping_rows)}/{len(mapping_positions)}, "
                        f"points={point_count:,} ({elapsed:.1f}s)",
                        flush=True,
                    )
                if max_source_frames > 0 and len(records) >= max_source_frames:
                    break
            pose_file.flush()
            os.fsync(pose_file.fileno())

        if len(records) != replay_count:
            raise RuntimeError(
                f"Single pass returned {len(records)} frames, expected {replay_count}"
            )
        if next_mapping_index != len(mapping_positions):
            missing_targets = mapping_positions[next_mapping_index:]
            raise RuntimeError(
                f"Mapped {len(mapping_rows)} valid frames, expected "
                f"{len(mapping_positions)}; unresolved nominal targets: "
                f"{missing_targets[:10]}"
            )
        if point_count <= 0:
            raise RuntimeError("No valid XYZRGBA points were produced")

        area_metadata = _save_area_map(
            zed,
            area_map_path,
            allow_empty=max_source_frames > 0,
        )

        header = _ply_header(point_count)
        final_size = len(header) + raw_bytes
        if final_size >= HARD_OUTPUT_LIMIT_BYTES:
            raise RuntimeError(
                f"Final PLY size {final_size} exceeds the hard limit "
                f"{HARD_OUTPUT_LIMIT_BYTES}"
            )
        with ply_tmp.open("wb") as ply_file:
            ply_file.write(header)
            with raw_path.open("rb") as raw_file:
                shutil.copyfileobj(raw_file, ply_file, length=8 * 1024 * 1024)
            ply_file.flush()
            os.fsync(ply_file.fileno())
        os.replace(ply_tmp, output_dir / "scene_world_rgb.ply")
        os.replace(pose_csv_tmp, pose_csv_path)
        with mapping_csv_tmp.open("w", newline="", encoding="utf-8") as mapping_file:
            writer = csv.writer(mapping_file)
            writer.writerow(MAPPING_HEADER)
            writer.writerows(mapping_rows)
        os.replace(mapping_csv_tmp, mapping_csv_path)
    finally:
        zed.close()
        for path in (raw_path, ply_tmp, pose_csv_tmp, mapping_csv_tmp):
            if path.exists():
                path.unlink()

    pointcloud_path = output_dir / "scene_world_rgb.ply"
    actual_size = pointcloud_path.stat().st_size
    if actual_size >= HARD_OUTPUT_LIMIT_BYTES:
        raise RuntimeError(
            f"Final PLY size {actual_size} exceeds the hard limit {HARD_OUTPUT_LIMIT_BYTES}"
        )
    states = Counter(record.tracking_state for record in records)
    valid_records = [record for record in records if record.pose_valid]
    final_summary = {
        "frames_replayed": len(records),
        "total_svo_frames_reported": total_frames,
        "tracking_state_counts": dict(sorted(states.items())),
        "valid_pose_frames": len(valid_records),
        "pose_csv_rows": pose_rows_written,
        "valid_pose_ratio": len(valid_records) / len(records),
        "mapped_frames": len(mapping_rows),
        "point_count": point_count,
        "pointcloud_file_size_bytes": actual_size,
        "elapsed_seconds": time.monotonic() - started,
        "area_map": area_metadata,
    }
    if valid_records:
        confidences = np.asarray(
            [record.pose_confidence for record in valid_records], dtype=np.float64
        )
        finite = confidences[np.isfinite(confidences)]
        if finite.size:
            final_summary["pose_confidence"] = {
                "min": float(np.min(finite)),
                "p05": float(np.percentile(finite, 5)),
                "median": float(np.median(finite)),
                "p95": float(np.percentile(finite, 95)),
                "max": float(np.max(finite)),
            }
        positions = np.asarray(
            [record.matrix[:3, 3] for record in valid_records], dtype=np.float64
        )
        if len(positions) >= 2:
            steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            final_summary["trajectory"] = {
                "path_length_m": float(np.sum(steps)),
                "endpoint_displacement_m": float(
                    np.linalg.norm(positions[-1] - positions[0])
                ),
                "position_min_m": positions.min(axis=0).tolist(),
                "position_max_m": positions.max(axis=0).tolist(),
                "step_p50_m": float(np.percentile(steps, 50)),
                "step_p95_m": float(np.percentile(steps, 95)),
                "step_max_m": float(np.max(steps)),
            }
    return (
        final_summary,
        {
            "mapping_rows": mapping_rows,
            "nominal_mapping_positions": mapping_positions,
            "pose_rows_written": pose_rows_written,
        },
        pointcloud_path,
        pose_csv_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run an SVO2 with custom OpenCV calibration using ZED GEN_1 "
            "tracking and export a bounded WORLD point cloud."
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
    parser.add_argument(
        "--opencv-calibration",
        type=Path,
        default=DEFAULT_OPENCV_CALIBRATION,
        help="Generated ZED-compatible OpenCV calibration path.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--map-frames",
        type=int,
        default=1000,
        help="Uniformly distributed valid frames used for dense point-cloud fusion.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=None,
        help=(
            "Uniformly retain this fraction of the complete replay for the pose "
            "CSV and point-cloud mapping; all source frames are still replayed "
            "sequentially for tracking."
        ),
    )
    parser.add_argument(
        "--pixel-stride",
        type=int,
        default=0,
        help="Pixel stride; 0 chooses the largest density below the safe size limit.",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.05,
        help="Minimum optical depth in meters for point-cloud filtering.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=50.0,
        help="Maximum optical depth in meters for point-cloud filtering.",
    )
    parser.add_argument(
        "--max-source-frames",
        type=int,
        default=0,
        help="Smoke-test limit; 0 means the complete SVO.",
    )
    parser.add_argument(
        "--direction-probe-frames",
        type=int,
        default=5,
        help="Frames used to validate the two possible stereo extrinsic directions.",
    )
    parser.add_argument(
        "--skip-direction-probe",
        action="store_true",
        help="Use source R/T directly; intended only for debugging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite only known output artifacts in an existing output directory.",
    )
    args = parser.parse_args()
    if args.map_frames <= 0:
        parser.error("--map-frames must be > 0")
    if args.sample_ratio is not None and not 0.0 < args.sample_ratio <= 1.0:
        parser.error("--sample-ratio must be in the interval (0, 1]")
    if args.pixel_stride < 0:
        parser.error("--pixel-stride must be >= 0")
    if args.min_depth < 0 or args.max_depth <= args.min_depth:
        parser.error("require 0 <= --min-depth < --max-depth")
    if args.max_source_frames < 0:
        parser.error("--max-source-frames must be >= 0")
    if args.direction_probe_frames <= 0:
        parser.error("--direction-probe-frames must be > 0")
    return args


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    known = [
        "pose_world.csv",
        "mapping_frame_index.csv",
        "scene_world_rgb.ply",
        "area_map_custom.area",
        "metadata.json",
        "zed_custom_opencv.yml",
    ]
    existing = [path / name for name in known if (path / name).exists()]
    if existing and not overwrite:
        raise RuntimeError(
            "Output artifacts already exist; use --overwrite or choose another "
            f"directory: {', '.join(str(item) for item in existing)}"
        )
    if overwrite:
        for item in existing:
            item.unlink()


def main() -> int:
    args = _parse_args()
    svo_path = args.svo.expanduser().resolve()
    intrinsics_path = args.intrinsics.expanduser().resolve()
    extrinsics_path = args.extrinsics.expanduser().resolve()
    opencv_calibration_path = args.opencv_calibration.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not svo_path.is_file():
        raise FileNotFoundError(f"SVO/SVO2 file not found: {svo_path}")
    if not intrinsics_path.is_file() or not extrinsics_path.is_file():
        raise FileNotFoundError(
            f"Calibration source files not found: {intrinsics_path}, {extrinsics_path}"
        )
    _prepare_output_dir(output_dir, args.overwrite)

    cameras, extrinsics = _load_source_calibration(intrinsics_path, extrinsics_path)
    # The calibration file is first generated from the source files.  A short
    # SDK-open probe then verifies the actual SVO resolution and the override.
    width = 1920
    height = 1080
    source_calibration = _make_calibration(
        cameras, extrinsics, width, height, inverse=False
    )
    _write_opencv_calibration(opencv_calibration_path, source_calibration)
    loaded_calibration = _read_opencv_calibration(opencv_calibration_path)
    _validate_calibration_against_source(loaded_calibration, cameras, extrinsics)

    # Probe with the generated file; this also verifies that the SVO can be
    # opened without falling back to factory calibration.
    svo_width, svo_height, reported_total_frames = _probe_svo_resolution(
        svo_path, opencv_calibration_path
    )
    if (svo_width, svo_height) != (loaded_calibration.width, loaded_calibration.height):
        raise RuntimeError(
            "Calibration resolution does not match SVO resolution: "
            f"calibration={loaded_calibration.width}x{loaded_calibration.height}, "
            f"svo={svo_width}x{svo_height}"
        )

    selected_calibration, direction_diagnostics = _select_direction_and_calibration(
        svo_path=svo_path,
        cameras=cameras,
        extrinsics=extrinsics,
        width=svo_width,
        height=svo_height,
        output_path=opencv_calibration_path,
        probe_frames=args.direction_probe_frames,
        skip_probe=args.skip_direction_probe,
    )
    loaded_calibration = _read_opencv_calibration(opencv_calibration_path)
    if selected_calibration.direction != "source_R_T_as_OpenCV_left_to_right":
        # Re-validate an inverse candidate against the source by comparing the
        # generated file to the selected candidate, not to the source R/T.
        if np.max(np.abs(loaded_calibration.t_mm - selected_calibration.t_mm)) > 1e-3:
            raise RuntimeError("Selected calibration file does not match the candidate")

    # Open once with NEURAL to verify all custom raw parameters before the
    # expensive sequential replay.
    zed = sl.Camera()
    _check_status(
        zed.open(_make_init(svo_path, opencv_calibration_path, sl.DEPTH_MODE.NEURAL)),
        "open SVO for custom calibration verification",
    )
    try:
        info = zed.get_camera_information()
        if int(zed.get_svo_number_of_frames()) != reported_total_frames:
            raise RuntimeError("SVO frame count changed during verification")
        sdk_calibration_metadata = _verify_sdk_override(info, selected_calibration)
    finally:
        zed.close()

    full_run = args.max_source_frames == 0
    if full_run:
        print(f"SVO total frames: {reported_total_frames}")
    else:
        print(
            f"Smoke-test replay limit: {args.max_source_frames} / "
            f"{reported_total_frames} frames"
        )
    print(f"Custom calibration: {opencv_calibration_path}")
    print(f"Selected extrinsic direction: {selected_calibration.direction}")

    replay_count = (
        min(args.max_source_frames, reported_total_frames)
        if args.max_source_frames > 0
        else reported_total_frames
    )
    # On this SVO the initial grab at source position 0 does not yet have a
    # valid WORLD pose.  Exclude that initialization frame from the nominal
    # uniform grid; the single replay still validates every selected frame and
    # can move a target forward to the next valid pose if needed.
    candidate_positions = list(range(1, replay_count))
    if not candidate_positions:
        candidate_positions = [0]
    if args.sample_ratio is None and args.map_frames > len(candidate_positions):
        raise RuntimeError(
            f"Only {len(candidate_positions)} candidate mapping positions are "
            f"available; cannot produce the requested {args.map_frames} frames"
        )
    if args.sample_ratio is None:
        mapping_positions = _uniform_sample(candidate_positions, args.map_frames)
        pose_export_positions: set[int] | None = None
        pose_sampling_metadata = {
            "enabled": False,
            "ratio": None,
            "selected_frame_count": None,
            "selection": "all_replayed_frames",
        }
    else:
        # Round against the complete replay count so 35,855 * 0.10 produces
        # the advertised 3,586 frames.  The initial source frame is excluded
        # because it has no valid WORLD pose on this recording.
        sample_count = max(
            1,
            min(
                len(candidate_positions),
                int(math.floor(replay_count * args.sample_ratio + 0.5)),
            ),
        )
        mapping_positions = _uniform_sample(candidate_positions, sample_count)
        pose_export_positions = set(mapping_positions)
        pose_sampling_metadata = {
            "enabled": True,
            "ratio": float(args.sample_ratio),
            "selected_frame_count": len(mapping_positions),
            "selection": (
                "uniform_source_positions_excluding_initial_unavailable_frame; "
                "all source frames still replayed sequentially"
            ),
        }
    pixel_stride, estimated_points = _choose_pixel_stride(
        width=svo_width,
        height=svo_height,
        frame_count=len(mapping_positions),
        requested=args.pixel_stride,
    )
    print(
        f"Export/mapping frames: {len(mapping_positions)}; "
        f"pose CSV rows requested: "
        f"{len(pose_export_positions) if pose_export_positions is not None else 'all'}; "
        f"pixel stride: {pixel_stride}; "
        f"worst-case estimate: {estimated_points:,} points / "
        f"{estimated_points * POINT_DTYPE.itemsize / 1e9:.3f} GB",
        flush=True,
    )

    final_summary, mapping_details, pointcloud_path, pose_csv_path = _run_mapping_pass(
        svo_path=svo_path,
        calibration_path=opencv_calibration_path,
        output_dir=output_dir,
        mapping_positions=mapping_positions,
        pose_export_positions=pose_export_positions,
        pixel_stride=pixel_stride,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        max_source_frames=args.max_source_frames,
    )

    output_calibration_path = output_dir / "zed_custom_opencv.yml"
    # ``--opencv-calibration`` may intentionally point into the output
    # directory.  In that case the calibration has already been written and
    # copying it onto itself raises SameFileError on Windows.
    if opencv_calibration_path.resolve() != output_calibration_path.resolve():
        shutil.copyfile(opencv_calibration_path, output_calibration_path)
    mapping_rows = mapping_details["mapping_rows"]
    metadata = {
        "input_file": str(svo_path),
        "output_dir": str(output_dir),
        "sdk_version": str(sl.Camera.get_sdk_version()),
        "camera": sdk_calibration_metadata,
        "calibration": {
            "source_intrinsics": str(intrinsics_path),
            "source_extrinsics": str(extrinsics_path),
            "opencv_file": str(output_dir / "zed_custom_opencv.yml"),
            "image_size": {"width": svo_width, "height": svo_height},
            "left_K": selected_calibration.k_left.tolist(),
            "right_K": selected_calibration.k_right.tolist(),
            "left_D_14": selected_calibration.d_left.tolist(),
            "right_D_14": selected_calibration.d_right.tolist(),
            "opencv_R_rodrigues": selected_calibration.r.tolist(),
            "opencv_T_mm": selected_calibration.t_mm.tolist(),
            "baseline_x_mm": float(selected_calibration.t_mm[0]),
            "translation_norm_mm": float(np.linalg.norm(selected_calibration.t_mm)),
            "source_reprojection_error_px": float(extrinsics["reprojection_error_px"]),
            "source_scale_error_percent": float(extrinsics["scale_error_percent"]),
            "direction_selection": direction_diagnostics,
        },
        "processing": {
            "depth_mode": "NEURAL",
            "depth_stabilization": 0,
            "coordinate_units": "METER",
            "coordinate_system": "RIGHT_HANDED_Y_UP",
            "pose_reference_frame": "WORLD",
            "pose_camera": "LEFT_EYE",
            "pose_transform": "T_world_left; p_world = T_world_left @ p_left",
            "tracking_mode": "GEN_1",
            "enable_imu_fusion": True,
            "enable_area_memory": True,
            "enable_pose_smoothing": False,
            "set_gravity_as_origin": True,
            "runtime_confidence_threshold": 30,
            "runtime_texture_confidence_threshold": 100,
            "sequential_replay": True,
            "max_source_frames": args.max_source_frames,
            "single_sequential_pass": True,
            "pose_output_sampling": pose_sampling_metadata,
            "mapping_selection": (
                "uniform_source_positions_excluding_initial_unavailable_frame; "
                "invalid targets advance to the next valid pose"
            ),
        },
        "replay": {
            "reported_total_frames": reported_total_frames,
            "final_pass": final_summary,
            "nominal_mapping_svo_positions": mapping_positions,
            "actual_mapping_svo_positions": [
                int(row[2]) for row in mapping_details["mapping_rows"]
            ],
            "mapping_rows": len(mapping_rows),
            "pose_csv_rows": int(mapping_details["pose_rows_written"]),
        },
        "point_cloud": {
            "file": str(pointcloud_path),
            "format": "PLY binary_little_endian",
            "fields": "float32 x,y,z; uint8 red,green,blue",
            "source": "ZED SDK MEASURE.XYZRGBA using custom calibration",
            "fusion": "T_world_left transform and streaming accumulation",
            "pixel_stride": pixel_stride,
            "min_optical_depth_m": args.min_depth,
            "max_optical_depth_m": args.max_depth,
            "point_count": final_summary["point_count"],
            "file_size_bytes": final_summary["pointcloud_file_size_bytes"],
            "file_size_gb_decimal": final_summary["pointcloud_file_size_bytes"] / 1e9,
            "hard_limit_bytes": HARD_OUTPUT_LIMIT_BYTES,
        },
        "outputs": {
            "pose_csv": str(pose_csv_path),
            "mapping_frame_index_csv": str(output_dir / "mapping_frame_index.csv"),
            "pointcloud": str(pointcloud_path),
            "area_map": final_summary["area_map"],
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        metadata_file.write("\n")

    print(f"Completed: {output_dir}")
    print(f"Pose CSV: {pose_csv_path}")
    print(f"Point cloud: {pointcloud_path}")
    print(
        f"Point count: {final_summary['point_count']:,}; "
        f"size: {final_summary['pointcloud_file_size_bytes'] / 1e9:.3f} GB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
