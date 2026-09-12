"""Dump a detailed comparison between SVO-embedded and external calibration.

This report reads the SVO calibration directly by opening the SVO without an
external calibration file.  It separately parses the files under
``Calibration/`` and derives the OpenCV custom rectification matrices.  No
depth maps or previous reports are used as calibration input.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
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

from strict_compare_stereo_depth import (  # noqa: E402
    _camera_metadata,
    _check_status,
    _custom_rectification,
    _load_custom_calibration,
    _matrix4,
    _status_name,
)


DEFAULT_SVO = Path(__file__).resolve().with_name("20260802_150233.svo2")
DEFAULT_INTRINSICS = Path(__file__).resolve().parent / "Calibration" / "标定结果" / "camera_intrinsics.yaml"
DEFAULT_EXTRINSICS = Path(__file__).resolve().parent / "Calibration" / "标定结果" / "stereo_extrinsics.yaml"
DEFAULT_OPENCV = Path(__file__).resolve().parent / "Calibration" / "zed_custom_opencv.yml"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "20260802_150233_calibration_comparison.json"


def _camera_dict_from_custom(camera: dict[str, Any]) -> dict[str, Any]:
    return {
        "fx": float(camera["fx"]),
        "fy": float(camera["fy"]),
        "cx": float(camera["cx"]),
        "cy": float(camera["cy"]),
        "distortion": [float(value) for value in camera["distortion"]],
        "lens_distortion_model": "OpenCV RAD_TAN (5 coefficients)",
    }


def _read_opencv_file(path: Path) -> dict[str, Any]:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise RuntimeError(f"Could not open OpenCV calibration file: {path}")
    try:
        size_node = storage.getNode("Size")
        size = [int(round(size_node.at(index).real())) for index in range(size_node.size())]
        matrices: dict[str, Any] = {}
        for name in ("K_LEFT", "K_RIGHT", "D_LEFT", "D_RIGHT", "R", "T"):
            matrix = storage.getNode(name).mat()
            if matrix is None:
                raise RuntimeError(f"Missing {name} in {path}")
            matrices[name] = np.asarray(matrix, dtype=np.float64)
        return {"size": size, **matrices}
    finally:
        storage.release()


def _open_native_svo(svo_path: Path) -> dict[str, Any]:
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.METER
    init.camera_disable_self_calib = True
    zed = sl.Camera()
    _check_status(zed.open(init), "open SVO for native calibration comparison")
    try:
        info = zed.get_camera_information()
        config = info.camera_configuration
        resolution = config.resolution
        raw = config.calibration_parameters_raw
        rectified = config.calibration_parameters
        return {
            "sdk_version": sl.Camera.get_sdk_version(),
            "camera_model": _status_name(info.camera_model),
            "serial_number": int(info.serial_number),
            "resolution": {"width": int(resolution.width), "height": int(resolution.height)},
            "fps": float(config.fps),
            "total_svo_frames": int(zed.get_svo_number_of_frames()),
            "coordinate_units_used_for_transform": "METER",
            "raw": {
                "left": _camera_metadata(raw.left_cam),
                "right": _camera_metadata(raw.right_cam),
                "stereo_transform_m": _matrix4(raw.stereo_transform.m).tolist(),
            },
            "rectified": {
                "left": _camera_metadata(rectified.left_cam),
                "right": _camera_metadata(rectified.right_cam),
                "stereo_transform_m": _matrix4(rectified.stereo_transform.m).tolist(),
            },
        }
    finally:
        zed.close()


def _matrix_diff(a: Any, b: Any) -> dict[str, float]:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    delta = left - right
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "frobenius": float(np.linalg.norm(delta)),
    }


def _camera_intrinsic_diff(native: dict[str, Any], custom: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for side in ("left", "right"):
        result[side] = {}
        for key in ("fx", "fy", "cx", "cy"):
            n = float(native[side][key])
            c = float(custom[side][key])
            result[side][key] = {
                "native": n,
                "custom": c,
                "custom_minus_native": c - n,
                "custom_over_native": c / n if n else None,
            }
        native_d = np.asarray(native[side]["distortion"], dtype=np.float64)
        custom_d = np.asarray(custom[side]["distortion"], dtype=np.float64)
        result[side]["distortion"] = {
            "native_length": int(native_d.size),
            "custom_length": int(custom_d.size),
            "native_first5": native_d[:5].tolist(),
            "custom_first5": custom_d[:5].tolist(),
            "custom_minus_native_first5": (custom_d[:5] - native_d[:5]).tolist(),
        }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("svo", nargs="?", type=Path, default=DEFAULT_SVO)
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--extrinsics", type=Path, default=DEFAULT_EXTRINSICS)
    parser.add_argument("--opencv-calibration", type=Path, default=DEFAULT_OPENCV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    svo_path = args.svo.expanduser().resolve()
    intrinsics_path = args.intrinsics.expanduser().resolve()
    extrinsics_path = args.extrinsics.expanduser().resolve()
    opencv_path = args.opencv_calibration.expanduser().resolve()
    for path, label in (
        (svo_path, "SVO2"),
        (intrinsics_path, "custom intrinsics"),
        (extrinsics_path, "custom extrinsics"),
        (opencv_path, "custom OpenCV calibration"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    native = _open_native_svo(svo_path)
    custom_source = _load_custom_calibration(intrinsics_path, extrinsics_path)
    extrinsics_text = extrinsics_path.read_text(encoding="utf-8-sig")

    def source_scalar(name: str) -> float:
        match = re.search(rf"(?m)^\s*{re.escape(name)}\s*:\s*([-+0-9.eE]+)", extrinsics_text)
        if match is None:
            raise RuntimeError(f"Missing {name} in {extrinsics_path}")
        return float(match.group(1))

    custom_source_scalars = {
        "baseline_declared_mm": source_scalar("baseline"),
        "reprojection_error_px": source_scalar("reprojection_error"),
        "scale_error_percent": source_scalar("scale_error_percent"),
    }
    custom = {
        "source_files": {
            "intrinsics": str(intrinsics_path),
            "extrinsics": str(extrinsics_path),
            "opencv_calibration": str(opencv_path),
        },
        "resolution_from_opencv_file": _read_opencv_file(opencv_path)["size"],
        "raw": {
            "left": _camera_dict_from_custom(custom_source["left"]),
            "right": _camera_dict_from_custom(custom_source["right"]),
        },
        "R_left_to_right": custom_source["rotation"].tolist(),
        "T_left_to_right_mm": custom_source["translation_mm"].tolist(),
        "baseline": {
            "yaml_declared_baseline_mm": custom_source_scalars["baseline_declared_mm"],
            "T_x_abs_mm": float(abs(custom_source["translation_mm"][0])),
            "T_norm_mm": float(np.linalg.norm(custom_source["translation_mm"])),
        },
        "quality": {
            "reprojection_error_px": custom_source_scalars["reprojection_error_px"],
            "scale_error_percent": custom_source_scalars["scale_error_percent"],
        },
    }
    opencv = _read_opencv_file(opencv_path)
    custom_rect_full = _custom_rectification(
        custom_source,
        (int(native["resolution"]["width"]), int(native["resolution"]["height"])),
    )
    custom_rect_half = _custom_rectification(
        custom_source,
        (int(native["resolution"]["width"]), int(native["resolution"]["height"])),
        (960, 540),
    )

    native_raw_left = native["raw"]["left"]
    native_raw_right = native["raw"]["right"]
    custom_raw_left = custom["raw"]["left"]
    custom_raw_right = custom["raw"]["right"]
    native_raw_transform = np.asarray(native["raw"]["stereo_transform_m"], dtype=np.float64)
    native_rect_transform = np.asarray(native["rectified"]["stereo_transform_m"], dtype=np.float64)
    report = {
        "input": {
            "svo": str(svo_path),
            "sdk_native_read_mode": "SVO opened without optional_opencv_calibration_file; DEPTH_MODE.NONE",
            "custom_read_mode": "Calibration files parsed separately; custom OpenCV file listed for provenance",
        },
        "native_svo_embedded": native,
        "custom_calibration_files": custom,
        "custom_opencv_file_matrices": {
            "size": opencv["size"],
            "K_LEFT": opencv["K_LEFT"].tolist(),
            "K_RIGHT": opencv["K_RIGHT"].tolist(),
            "D_LEFT": opencv["D_LEFT"].tolist(),
            "D_RIGHT": opencv["D_RIGHT"].tolist(),
            "R_rodrigues": opencv["R"].reshape(-1).tolist(),
            "T_mm": opencv["T"].reshape(-1).tolist(),
            "custom_source_vs_opencv": {
                "left_K": _matrix_diff(
                    opencv["K_LEFT"],
                    np.array(
                        [[custom_raw_left["fx"], 0.0, custom_raw_left["cx"]],
                         [0.0, custom_raw_left["fy"], custom_raw_left["cy"]],
                         [0.0, 0.0, 1.0]]
                    ),
                ),
                "right_K": _matrix_diff(
                    opencv["K_RIGHT"],
                    np.array(
                        [[custom_raw_right["fx"], 0.0, custom_raw_right["cx"]],
                         [0.0, custom_raw_right["fy"], custom_raw_right["cy"]],
                         [0.0, 0.0, 1.0]]
                    ),
                ),
                "T_mm": _matrix_diff(opencv["T"].reshape(-1), custom_source["translation_mm"]),
            },
        },
        "raw_intrinsics_difference_custom_minus_native": _camera_intrinsic_diff(
            native["raw"], custom["raw"]
        ),
        "baseline_comparison": {
            "native_raw_transform_norm_m": float(np.linalg.norm(native_raw_transform[:3, 3])),
            "native_rectified_transform_x_m": float(native_rect_transform[0, 3]),
            "custom_T_norm_m": float(custom["baseline"]["T_norm_mm"] / 1000.0),
            "custom_yaml_declared_baseline_m": float(
                custom["baseline"]["yaml_declared_baseline_mm"] / 1000.0
            ),
            "custom_T_x_abs_m": float(custom["baseline"]["T_x_abs_mm"] / 1000.0),
            "custom_T_norm_minus_native_raw_m": float(
                custom["baseline"]["T_norm_mm"] / 1000.0
                - np.linalg.norm(native_raw_transform[:3, 3])
            ),
        },
        "rectification_comparison": {
            "native_svo_rectified": {
                "left": native["rectified"]["left"],
                "right": native["rectified"]["right"],
                "stereo_transform_m": native["rectified"]["stereo_transform_m"],
            },
            "custom_opencv_alpha0_full_1920x1080": {
                "image_size": [1920, 1080],
                "P1": custom_rect_full["p1"].tolist(),
                "P2": custom_rect_full["p2"].tolist(),
                "focal_length_px": custom_rect_full["focal_length_px"],
                "baseline_m": custom_rect_full["baseline_m"],
                "roi_left": custom_rect_full["roi_left"],
                "roi_right": custom_rect_full["roi_right"],
            },
            "custom_opencv_alpha0_half_960x540": {
                "image_size": [960, 540],
                "P1": custom_rect_half["p1"].tolist(),
                "P2": custom_rect_half["p2"].tolist(),
                "focal_length_px": custom_rect_half["focal_length_px"],
                "baseline_m": custom_rect_half["baseline_m"],
                "roi_left": custom_rect_half["roi_left"],
                "roi_right": custom_rect_half["roi_right"],
            },
        },
        "interpretation": {
            "native_raw_focal_is_not_custom_raw_focal": True,
            "custom_raw_focal_must_not_be_used_directly_with_rectified_disparity": True,
            "custom_rectified_focal_is_P1_00_after_stereoRectify": True,
            "native_and_custom_stereo_transform_conventions_should_not_be_subtracted_without_coordinate_convention_alignment": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"SVO native raw left fx/fy: {native_raw_left['fx']:.9f}, {native_raw_left['fy']:.9f} px")
    print(f"SVO native raw right fx/fy: {native_raw_right['fx']:.9f}, {native_raw_right['fy']:.9f} px")
    print(f"SVO native rectified left fx/fy: {native['rectified']['left']['fx']:.9f}, {native['rectified']['left']['fy']:.9f} px")
    print(f"SVO native baseline: {np.linalg.norm(native_raw_transform[:3, 3]):.9f} m")
    print(f"Custom raw left fx/fy: {custom_raw_left['fx']:.9f}, {custom_raw_left['fy']:.9f} px")
    print(f"Custom raw right fx/fy: {custom_raw_right['fx']:.9f}, {custom_raw_right['fy']:.9f} px")
    print(f"Custom rectified fx full: {custom_rect_full['focal_length_px']:.9f} px")
    print(f"Custom rectified fx half: {custom_rect_half['focal_length_px']:.9f} px")
    print(f"Custom baseline norm: {custom['baseline']['T_norm_mm']:.9f} mm")
    print(f"Report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
