"""Generate a Left | Right | ZED SDK depth GIF from an SVO2 recording.

The exporter opens the SVO with the project's Custom OpenCV calibration and
disables the embedded/native self-calibration.  A first pass estimates one
global depth display range for the requested sequence; a second pass renders
the same consecutive frames into an animated GIF.
"""

from __future__ import annotations

import argparse
import json
import os
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
    current = [item for item in os.environ.get("PATH", "").split(os.pathsep) if item]
    os.environ["PATH"] = os.pathsep.join([*existing, *current])
    if hasattr(os, "add_dll_directory"):
        handles = [os.add_dll_directory(path) for path in existing]
        _prepare_windows_dll_search_path._dll_handles = handles  # type: ignore[attr-defined]


_prepare_windows_dll_search_path()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import pyzed.sl as sl  # noqa: E402


ROOT = Path(__file__).resolve().parent
DEFAULT_SVO = ROOT / "20260802_150233.svo2"
DEFAULT_CALIBRATION = ROOT / "Calibration" / "zed_custom_opencv.yml"
DEFAULT_OUTPUT = ROOT / "notebook_assets" / "zed_sdk_custom_depth_60frames.gif"
DEFAULT_METADATA = ROOT / "notebook_assets" / "zed_sdk_custom_depth_60frames_metadata.json"


def _status_name(status: Any) -> str:
    return str(status).rsplit(".", 1)[-1]


def _check_status(status: Any, operation: str) -> None:
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"{operation} failed: {_status_name(status)}")


def _depth_array(depth_mat: sl.Mat) -> np.ndarray:
    data = np.asarray(depth_mat.get_data())
    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]
    if data.ndim != 2:
        raise RuntimeError(f"Expected 2-D depth, got {data.shape}")
    return np.array(data, dtype=np.float32, copy=True)


def _as_bgr(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return np.array(image, copy=True)
    raise RuntimeError(f"Unexpected camera image shape: {image.shape}")


def _make_init(svo_path: Path, calibration_path: Path) -> sl.InitParameters:
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.camera_disable_self_calib = True
    init.optional_opencv_calibration_file = str(calibration_path)
    init.depth_stabilization = 0
    init.enable_image_enhancement = False
    return init


def _open_camera(svo_path: Path, calibration_path: Path) -> tuple[sl.Camera, Any, int]:
    zed = sl.Camera()
    _check_status(
        zed.open(_make_init(svo_path, calibration_path)),
        "open SVO with Custom calibration",
    )
    info = zed.get_camera_information()
    return zed, info, int(zed.get_svo_number_of_frames())


def _verify_runtime_calibration(info: Any, calibration_path: Path) -> dict[str, Any]:
    storage = cv2.FileStorage(str(calibration_path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise RuntimeError(f"Could not open Custom calibration: {calibration_path}")
    try:
        k_left = np.asarray(storage.getNode("K_LEFT").mat(), dtype=np.float64)
        k_right = np.asarray(storage.getNode("K_RIGHT").mat(), dtype=np.float64)
        r_vector = np.asarray(storage.getNode("R").mat(), dtype=np.float64).reshape(3, 1)
        t_mm = np.asarray(storage.getNode("T").mat(), dtype=np.float64).reshape(3)
    finally:
        storage.release()

    raw = info.camera_configuration.calibration_parameters_raw
    expected_rotation = cv2.Rodrigues(r_vector)[0]
    expected_t_m = t_mm / 1000.0
    expected_transform = np.eye(4, dtype=np.float64)
    expected_transform[:3, :3] = expected_rotation.T
    expected_transform[:3, 3] = -expected_rotation.T @ expected_t_m
    observed_transform = np.asarray(raw.stereo_transform.m, dtype=np.float64).reshape(4, 4)
    errors = {
        "left_fx_px": abs(float(raw.left_cam.fx) - float(k_left[0, 0])),
        "left_fy_px": abs(float(raw.left_cam.fy) - float(k_left[1, 1])),
        "left_cx_px": abs(float(raw.left_cam.cx) - float(k_left[0, 2])),
        "left_cy_px": abs(float(raw.left_cam.cy) - float(k_left[1, 2])),
        "right_fx_px": abs(float(raw.right_cam.fx) - float(k_right[0, 0])),
        "right_fy_px": abs(float(raw.right_cam.fy) - float(k_right[1, 1])),
        "right_cx_px": abs(float(raw.right_cam.cx) - float(k_right[0, 2])),
        "right_cy_px": abs(float(raw.right_cam.cy) - float(k_right[1, 2])),
        "stereo_transform_max_abs": float(
            np.max(np.abs(observed_transform - expected_transform))
        ),
    }
    passed = max(errors.values()) <= 2e-3
    if not passed:
        raise RuntimeError(f"Custom calibration runtime verification failed: {errors}")
    return {
        "passed": True,
        "sdk_raw_left": {
            "fx_px": float(raw.left_cam.fx),
            "fy_px": float(raw.left_cam.fy),
            "cx_px": float(raw.left_cam.cx),
            "cy_px": float(raw.left_cam.cy),
        },
        "sdk_raw_right": {
            "fx_px": float(raw.right_cam.fx),
            "fy_px": float(raw.right_cam.fy),
            "cx_px": float(raw.right_cam.cx),
            "cy_px": float(raw.right_cam.cy),
        },
        "baseline_norm_m": float(np.linalg.norm(observed_transform[:3, 3])),
        "max_abs_errors": errors,
    }


def _make_runtime() -> sl.RuntimeParameters:
    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 30
    runtime.texture_confidence_threshold = 100
    runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA
    return runtime


def _seek_and_grab(zed: sl.Camera, runtime: sl.RuntimeParameters, frame_index: int) -> int:
    _check_status(zed.set_svo_position(frame_index), f"seek to frame {frame_index}")
    _check_status(zed.grab(runtime), f"grab frame {frame_index}")
    actual = int(zed.get_svo_position())
    if actual != frame_index:
        raise RuntimeError(f"Frame seek mismatch: requested {frame_index}, got {actual}")
    return actual


def _read_depth_frame(zed: sl.Camera, depth_mat: sl.Mat, frame_index: int) -> np.ndarray:
    _check_status(
        zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH),
        f"retrieve SDK depth at frame {frame_index}",
    )
    return _depth_array(depth_mat)


def _estimate_display_range(
    svo_path: Path,
    calibration_path: Path,
    frame_indices: list[int],
    max_depth: float,
    sample_step: int,
) -> tuple[float, float, dict[str, Any]]:
    zed, info, total_frames = _open_camera(svo_path, calibration_path)
    try:
        runtime = _make_runtime()
        depth_mat = sl.Mat()
        samples: list[np.ndarray] = []
        for frame_index in frame_indices:
            _seek_and_grab(zed, runtime, frame_index)
            depth = _read_depth_frame(zed, depth_mat, frame_index)
            sampled = depth[::sample_step, ::sample_step]
            valid = np.isfinite(sampled) & (sampled > 0.0) & (sampled <= max_depth)
            if np.any(valid):
                samples.append(sampled[valid].astype(np.float32, copy=False))
    finally:
        zed.close()
    if not samples:
        raise RuntimeError("No valid SDK depth samples found")
    values = np.concatenate(samples).astype(np.float64, copy=False)
    low, high = np.percentile(values, (2.0, 98.0))
    if high <= low:
        high = low + 1.0
    return float(low), float(high), {
        "sample_step": sample_step,
        "sample_count": int(values.size),
        "valid_min_m": float(np.min(values)),
        "valid_median_m": float(np.median(values)),
        "valid_max_m": float(np.max(values)),
        "display_low_m": float(low),
        "display_high_m": float(high),
        "sdk_total_frames": total_frames,
        "resolution": {
            "width": int(info.camera_configuration.resolution.width),
            "height": int(info.camera_configuration.resolution.height),
        },
    }


def _center_median(depth: np.ndarray, radius: int) -> float:
    height, width = depth.shape
    cx, cy = width // 2, height // 2
    window = depth[
        max(0, cy - radius):min(height, cy + radius + 1),
        max(0, cx - radius):min(width, cx + radius + 1),
    ]
    valid = np.isfinite(window) & (window > 0.0)
    values = window[valid]
    return float(np.median(values)) if values.size else float("nan")


def _label_panel(panel: np.ndarray, title: str) -> np.ndarray:
    panel = np.array(panel, copy=True)
    banner_height = max(34, panel.shape[0] // 11)
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, banner_height), (0, 0, 0), -1)
    cv2.putText(
        panel,
        title,
        (12, max(24, banner_height - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def _render_composite(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    depth: np.ndarray,
    frame_index: int,
    display_low: float,
    display_high: float,
    panel_width: int,
    max_depth: float,
    center_radius: int,
) -> tuple[Image.Image, dict[str, Any]]:
    source_height, source_width = depth.shape
    panel_height = max(2, int(round(panel_width * source_height / source_width)))
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth)
    gray = np.zeros(depth.shape, dtype=np.uint8)
    gray[valid] = np.clip(
        (depth[valid] - display_low) * 255.0 / (display_high - display_low),
        0.0,
        255.0,
    ).astype(np.uint8)
    depth_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    depth_bgr[~valid] = 0

    left_panel = cv2.resize(left_bgr, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    right_panel = cv2.resize(right_bgr, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    depth_panel = cv2.resize(depth_bgr, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    center_depth = _center_median(depth, center_radius)
    cv2.drawMarker(
        depth_panel,
        (panel_width // 2, panel_height // 2),
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=max(12, panel_width // 28),
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    left_panel = _label_panel(left_panel, "LEFT | SDK VIEW.LEFT")
    right_panel = _label_panel(right_panel, "RIGHT | SDK VIEW.RIGHT")
    depth_title = (
        f"DEPTH | SDK MEASURE.DEPTH | center {center_depth:.2f} m"
        if np.isfinite(center_depth)
        else "DEPTH | SDK MEASURE.DEPTH | center invalid"
    )
    depth_panel = _label_panel(depth_panel, depth_title)

    composite = np.hstack([left_panel, right_panel, depth_panel])
    footer_height = max(30, panel_height // 9)
    footer = np.zeros((footer_height, composite.shape[1], 3), dtype=np.uint8)
    footer_text = (
        f"Custom calibration | frame {frame_index:06d} | "
        f"display {display_low:.2f}-{display_high:.2f} m | blue=near, red=far"
    )
    cv2.putText(
        footer,
        footer_text,
        (12, max(22, footer_height - 9)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    composite = np.vstack([composite, footer])
    rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
    return image, {
        "frame_index": frame_index,
        "center_window_median_m": center_depth,
        "valid_ratio": float(np.count_nonzero(valid) / valid.size),
        "width": int(composite.shape[1]),
        "height": int(composite.shape[0]),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Left | Right | ZED SDK depth animated GIF."
    )
    parser.add_argument("--svo", type=Path, default=DEFAULT_SVO)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--start-frame", type=int, default=1000)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--panel-width", type=int, default=512)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--center-radius", type=int, default=15)
    parser.add_argument("--sample-step", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    svo_path = args.svo.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve()
    if not svo_path.is_file():
        raise FileNotFoundError(f"SVO file not found: {svo_path}")
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
    if args.frames <= 0 or args.fps <= 0 or args.panel_width <= 0:
        raise ValueError("frames, fps, and panel-width must be positive")
    if args.start_frame < 0 or args.sample_step <= 0 or args.center_radius < 0:
        raise ValueError("start-frame, sample-step, and center-radius are invalid")

    # Open once before the two passes to validate the actual SDK override.
    zed, info, total_frames = _open_camera(svo_path, calibration_path)
    try:
        runtime_verification = _verify_runtime_calibration(info, calibration_path)
    finally:
        zed.close()
    if args.start_frame + args.frames > total_frames:
        raise ValueError(
            f"Requested frames [{args.start_frame}, {args.start_frame + args.frames - 1}] "
            f"outside SVO range [0, {total_frames - 1}]"
        )
    frame_indices = list(range(args.start_frame, args.start_frame + args.frames))
    display_low, display_high, range_metadata = _estimate_display_range(
        svo_path,
        calibration_path,
        frame_indices,
        args.max_depth,
        args.sample_step,
    )

    zed, info, second_total_frames = _open_camera(svo_path, calibration_path)
    gif_frames: list[Image.Image] = []
    frame_metadata: list[dict[str, Any]] = []
    try:
        runtime = _make_runtime()
        left_mat, right_mat, depth_mat = sl.Mat(), sl.Mat(), sl.Mat()
        for frame_number, frame_index in enumerate(frame_indices, start=1):
            actual = _seek_and_grab(zed, runtime, frame_index)
            _check_status(
                zed.retrieve_image(left_mat, sl.VIEW.LEFT),
                f"retrieve left image at frame {actual}",
            )
            _check_status(
                zed.retrieve_image(right_mat, sl.VIEW.RIGHT),
                f"retrieve right image at frame {actual}",
            )
            depth = _read_depth_frame(zed, depth_mat, actual)
            image, per_frame = _render_composite(
                _as_bgr(np.array(left_mat.get_data(), copy=True)),
                _as_bgr(np.array(right_mat.get_data(), copy=True)),
                depth,
                actual,
                display_low,
                display_high,
                args.panel_width,
                args.max_depth,
                args.center_radius,
            )
            gif_frames.append(image)
            frame_metadata.append(per_frame)
            print(
                f"Rendered {frame_number}/{len(frame_indices)}: frame {actual}; "
                f"center={per_frame['center_window_median_m']:.3f} m; "
                f"valid={per_frame['valid_ratio']:.1%}",
                flush=True,
            )
    finally:
        zed.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # GIF stores frame delays in centiseconds.  Round to the actual representable
    # delay so metadata and playback behavior agree.
    duration_ms = max(20, int(round((1000.0 / args.fps) / 10.0) * 10))
    actual_playback_fps = 1000.0 / duration_ms
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    metadata = {
        "input_file": str(svo_path),
        "sdk_version": str(sl.Camera.get_sdk_version()),
        "calibration_file": str(calibration_path),
        "calibration_runtime_verification": runtime_verification,
        "native_svo_calibration_used": False,
        "processing": {
            "depth_source": "ZED SDK MEASURE.DEPTH",
            "depth_mode": "NEURAL",
            "camera_disable_self_calib": True,
            "optional_opencv_calibration_file": str(calibration_path),
            "view_left": "VIEW.LEFT",
            "view_right": "VIEW.RIGHT",
            "coordinate_units": "METER",
            "frames": len(frame_indices),
            "start_frame": args.start_frame,
            "end_frame": frame_indices[-1],
            "requested_playback_fps": args.fps,
            "playback_fps": actual_playback_fps,
            "duration_seconds": len(frame_indices) * duration_ms / 1000.0,
            "panel_width": args.panel_width,
            "global_display_range_m": [display_low, display_high],
            "global_display_range_method": "2nd-98th percentile over sampled valid depth values",
            "max_depth_filter_m": args.max_depth,
            "center_window_radius_px": args.center_radius,
        },
        "svo": {
            "reported_total_frames": total_frames,
            "second_pass_total_frames": second_total_frames,
            "resolution": range_metadata["resolution"],
            "frame_indices": frame_indices,
            "sequential_requested_frames": True,
        },
        "display_range_statistics": range_metadata,
        "frames": frame_metadata,
        "output": {
            "gif": str(output_path),
            "format": "GIF89a",
            "size_bytes": int(output_path.stat().st_size),
            "width": frame_metadata[0]["width"],
            "height": frame_metadata[0]["height"],
            "frame_count": len(gif_frames),
            "duration_ms_per_frame": duration_ms,
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"GIF: {output_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Frames: {len(gif_frames)}; size: {output_path.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
