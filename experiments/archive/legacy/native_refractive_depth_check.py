"""Audit-only native-raw refractive model entry point.

The native raw K/D/R/T are the best available dry/physical stereo candidate
in this repository, but they are not proof that the recorded camera was in
air or that the housing was absent.  This script deliberately stops before
producing a definitive underwater metric depth because the repository does
not contain the installed port geometry and refractive indices.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from debug_refractive_depth_check import (
    _camera_metadata,
    _matrix4,
    _open_svo,
    _status_name,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_SVO = ROOT / "20260802_150233.svo2"
DEFAULT_OUTPUT = ROOT / "output" / "20260802_150233_final_underwater_audit" / "native_refractive_depth_check.json"


def collect_native_metadata(svo_path: Path) -> dict[str, Any]:
    """Read only the SVO-embedded calibration with self-calibration disabled."""

    zed = _open_svo(svo_path)
    try:
        info = zed.get_camera_information()
        configuration = info.camera_configuration
        resolution = configuration.resolution
        raw = configuration.calibration_parameters_raw
        rectified = configuration.calibration_parameters
        raw_transform = _matrix4(raw.stereo_transform.m)
        raw_r = raw_transform[:3, :3]
        raw_t = raw_transform[:3, 3]
        if np.max(np.abs(raw_r.T @ raw_r - np.eye(3))) > 1.0e-5 or abs(float(np.linalg.det(raw_r)) - 1.0) > 1.0e-5:
            raise RuntimeError("SVO raw stereo rotation failed orthonormality checks")
        return {
            "sdk_version": str(__import__("pyzed.sl", fromlist=["sl"]).Camera.get_sdk_version()),
            "camera_model": _status_name(info.camera_model),
            "serial_number": int(info.serial_number),
            "resolution": {"width": int(resolution.width), "height": int(resolution.height)},
            "fps": float(configuration.fps),
            "total_svo_frames": int(zed.get_svo_number_of_frames()),
            "read_mode": {
                "calibration_source": "SVO embedded calibration_parameters_raw",
                "image_geometry_for_candidate": "VIEW.LEFT_UNRECTIFIED / VIEW.RIGHT_UNRECTIFIED raw distorted pixels",
                "camera_disable_self_calib": True,
                "depth_mode": "NONE",
                "coordinate_units": "METER",
            },
            "raw": {
                "left": _camera_metadata(raw.left_cam),
                "right": _camera_metadata(raw.right_cam),
                "stereo_transform_right_to_left_m": raw_transform.tolist(),
                "translation_right_to_left_m": raw_t.tolist(),
                "camera_center_right_in_left_rig_m": raw_t.tolist(),
                "baseline_norm_m": float(np.linalg.norm(raw_t)),
                "baseline_x_abs_m": float(abs(raw_t[0])),
            },
            "rectified_reference": {
                "left": _camera_metadata(rectified.left_cam),
                "right": _camera_metadata(rectified.right_cam),
                "stereo_transform_m": _matrix4(rectified.stereo_transform.m).tolist(),
                "status": "reference only; not used to claim a physical refractive model",
            },
        }
    finally:
        zed.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("svo", nargs="?", type=Path, default=DEFAULT_SVO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    svo_path = args.svo.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not svo_path.is_file():
        raise FileNotFoundError(f"SVO2 not found: {svo_path}")
    native = collect_native_metadata(svo_path)
    report = {
        "script": Path(__file__).name,
        "status": "NOT_IDENTIFIABLE",
        "model_candidate": "R1_NATIVE_DRY_CANDIDATE_PLUS_EXPLICIT_FLAT_PORT_REFRACTION",
        "depth_estimate": None,
        "absolute_depth_claim": "PROHIBITED",
        "reason": "The repository does not identify the installed housing geometry, port type, interface distances, glass index, water index, or plane orientation.",
        "required_for_definitive_r1": [
            "calibration medium and housing provenance",
            "flat-port versus dome confirmation",
            "shared/separate port geometry in a common rig frame",
            "camera-to-inner-interface distance for each camera",
            "glass thickness and refractive index",
            "water refractive index or measured temperature/salinity",
            "port plane normals and camera-to-rig poses",
        ],
        "native_candidate": native,
        "coordinate_convention": {
            "camera_axes": "x=right, y=down, z=forward",
            "native_sdk_transform": "X_left = R_right_to_left X_right + T_right_to_left",
            "raw_pixels": "undistorted by native raw K/D only before a hypothetical physical ray trace",
            "Z_definition_if_computed": "left-camera-rig optical-axis coordinate, distinct from Euclidean range and interface-to-target distance",
        },
        "guardrails": [
            "No n-water multiplication is applied here.",
            "No custom effective K/T is combined with Snell as a final physical result.",
            "No definitive R1 depth is emitted while physical parameters are unknown.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Native raw candidate: {native['raw']['left']['fx_px']:.6f}/{native['raw']['right']['fx_px']:.6f} px")
    print(f"Native baseline norm: {native['raw']['baseline_norm_m']:.9f} m")
    print("Explicit refractive depth: NOT_IDENTIFIABLE; no definitive depth written")
    print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
