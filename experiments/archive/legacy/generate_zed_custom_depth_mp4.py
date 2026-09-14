"""Generate a compressed Left | Right | ZED SDK depth MP4.

The video uses the same Custom-calibration and ZED SDK depth path as the GIF
exporter, but writes full-color BGR frames through OpenCV's MP4 writer instead
of quantizing each frame to a 256-color GIF palette.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import generate_zed_custom_depth_gif as helpers


sl = helpers.sl
ROOT = Path(__file__).resolve().parent
DEFAULT_SVO = ROOT / "20260802_150233.svo2"
DEFAULT_CALIBRATION = ROOT / "Calibration" / "zed_custom_opencv.yml"
DEFAULT_OUTPUT = ROOT / "notebook_assets" / "zed_sdk_custom_depth_120frames.mp4"
DEFAULT_METADATA = ROOT / "notebook_assets" / "zed_sdk_custom_depth_120frames_metadata.json"


def _render_composite_bgr(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    depth: np.ndarray,
    frame_index: int,
    display_low: float,
    display_high: float,
    panel_width: int,
    max_depth: float,
    center_radius: int,
) -> tuple[np.ndarray, dict[str, Any]]:
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
    center_depth = helpers._center_median(depth, center_radius)
    cv2.drawMarker(
        depth_panel,
        (panel_width // 2, panel_height // 2),
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=max(12, panel_width // 28),
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    left_panel = helpers._label_panel(left_panel, "LEFT | SDK VIEW.LEFT")
    right_panel = helpers._label_panel(right_panel, "RIGHT | SDK VIEW.RIGHT")
    depth_title = (
        f"DEPTH | SDK MEASURE.DEPTH | center {center_depth:.2f} m"
        if np.isfinite(center_depth)
        else "DEPTH | SDK MEASURE.DEPTH | center invalid"
    )
    depth_panel = helpers._label_panel(depth_panel, depth_title)

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
    return composite, {
        "frame_index": frame_index,
        "center_window_median_m": center_depth,
        "valid_ratio": float(np.count_nonzero(valid) / valid.size),
        "width": int(composite.shape[1]),
        "height": int(composite.shape[0]),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a compressed Left | Right | ZED SDK depth MP4."
    )
    parser.add_argument("--svo", type=Path, default=DEFAULT_SVO)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--start-frame", type=int, default=1000)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--panel-width", type=int, default=512)
    parser.add_argument("--codec", type=str, default="mp4v")
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--center-radius", type=int, default=15)
    parser.add_argument("--sample-step", type=int, default=8)
    return parser.parse_args()


def _verify_video(path: Path, expected_frames: int, expected_size: tuple[int, int]) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen generated MP4: {path}")
    reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    decoded_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[1::-1] != expected_size:
                raise RuntimeError(
                    f"Generated MP4 frame size changed: {frame.shape[1::-1]} vs {expected_size}"
                )
            decoded_frames += 1
    finally:
        capture.release()
    if decoded_frames != expected_frames:
        raise RuntimeError(
            f"Generated MP4 frame count mismatch: decoded={decoded_frames}, expected={expected_frames}"
        )
    return {
        "reported_frame_count": reported_frames,
        "decoded_frame_count": decoded_frames,
        "reported_fps": reported_fps,
        "width": width,
        "height": height,
        "passed": True,
    }


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
    if len(args.codec) != 4:
        raise ValueError("codec must be a four-character OpenCV codec code")

    zed, info, total_frames = helpers._open_camera(svo_path, calibration_path)
    try:
        runtime_verification = helpers._verify_runtime_calibration(info, calibration_path)
    finally:
        zed.close()
    if args.start_frame + args.frames > total_frames:
        raise ValueError(
            f"Requested frames [{args.start_frame}, {args.start_frame + args.frames - 1}] "
            f"outside SVO range [0, {total_frames - 1}]"
        )
    frame_indices = list(range(args.start_frame, args.start_frame + args.frames))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display_low, display_high, range_metadata = helpers._estimate_display_range(
        svo_path,
        calibration_path,
        frame_indices,
        args.max_depth,
        args.sample_step,
    )

    zed, info, second_total_frames = helpers._open_camera(svo_path, calibration_path)
    writer: cv2.VideoWriter | None = None
    frame_metadata: list[dict[str, Any]] = []
    try:
        runtime = helpers._make_runtime()
        left_mat, right_mat, depth_mat = sl.Mat(), sl.Mat(), sl.Mat()
        for frame_number, frame_index in enumerate(frame_indices, start=1):
            actual = helpers._seek_and_grab(zed, runtime, frame_index)
            helpers._check_status(
                zed.retrieve_image(left_mat, sl.VIEW.LEFT),
                f"retrieve left image at frame {actual}",
            )
            helpers._check_status(
                zed.retrieve_image(right_mat, sl.VIEW.RIGHT),
                f"retrieve right image at frame {actual}",
            )
            depth = helpers._read_depth_frame(zed, depth_mat, actual)
            composite, per_frame = _render_composite_bgr(
                helpers._as_bgr(np.array(left_mat.get_data(), copy=True)),
                helpers._as_bgr(np.array(right_mat.get_data(), copy=True)),
                depth,
                actual,
                display_low,
                display_high,
                args.panel_width,
                args.max_depth,
                args.center_radius,
            )
            if writer is None:
                height, width = composite.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*args.codec),
                    args.fps,
                    (width, height),
                    True,
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        f"Could not open MP4 writer with codec {args.codec!r}: {output_path}"
                    )
            writer.write(composite)
            frame_metadata.append(per_frame)
            print(
                f"Rendered {frame_number}/{len(frame_indices)}: frame {actual}; "
                f"center={per_frame['center_window_median_m']:.3f} m; "
                f"valid={per_frame['valid_ratio']:.1%}",
                flush=True,
            )
    finally:
        if writer is not None:
            writer.release()
        zed.close()

    if not frame_metadata:
        raise RuntimeError("No MP4 frames were written")
    video_verification = _verify_video(
        output_path,
        len(frame_indices),
        (frame_metadata[0]["width"], frame_metadata[0]["height"]),
    )
    metadata = {
        "input_file": str(svo_path),
        "sdk_version": str(sl.Camera.get_sdk_version()),
        "calibration_file": str(calibration_path),
        "calibration_runtime_verification": runtime_verification,
        "native_svo_calibration_used": False,
        "processing": {
            "left_source": "ZED SDK VIEW.LEFT",
            "right_source": "ZED SDK VIEW.RIGHT",
            "depth_source": "ZED SDK MEASURE.DEPTH",
            "depth_mode": "NEURAL",
            "camera_disable_self_calib": True,
            "optional_opencv_calibration_file": str(calibration_path),
            "coordinate_units": "METER",
            "frames": len(frame_indices),
            "start_frame": args.start_frame,
            "end_frame": frame_indices[-1],
            "requested_playback_fps": args.fps,
            "actual_container_fps": video_verification["reported_fps"],
            "duration_seconds": len(frame_indices) / video_verification["reported_fps"],
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
        "verification": video_verification,
        "output": {
            "video": str(output_path),
            "format": "MP4",
            "codec": args.codec,
            "size_bytes": int(output_path.stat().st_size),
            "width": frame_metadata[0]["width"],
            "height": frame_metadata[0]["height"],
            "frame_count": len(frame_indices),
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"MP4: {output_path}")
    print(f"Metadata: {metadata_path}")
    print(
        f"Frames: {len(frame_indices)}; size: {output_path.stat().st_size:,} bytes; "
        f"decoded verification: PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
