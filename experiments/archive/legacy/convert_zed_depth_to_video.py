"""Export the native ZED SDK depth measure from an SVO/SVO2 to video.

The video is generated from ``sl.MEASURE.DEPTH`` using the SDK's embedded
calibration and ``DEPTH_MODE.NEURAL``.  No OpenCV stereo matching or custom
calibration file is used.  The SVO is replayed twice: the first pass estimates
a stable robust display range, and the second pass writes every depth frame.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable


def _prepare_windows_dll_search_path() -> None:
    """Make ZED SDK DLLs discoverable before importing ``pyzed.sl``."""

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

    current = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join([*existing, *(p for p in current if p)])
    if hasattr(os, "add_dll_directory"):
        handles = [os.add_dll_directory(path) for path in existing]
        _prepare_windows_dll_search_path._dll_handles = handles  # type: ignore[attr-defined]


_prepare_windows_dll_search_path()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyzed.sl as sl  # noqa: E402


DEFAULT_SVO = Path(__file__).resolve().with_name("20260802_150233.svo2")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "output" / "20260802_150233_zed_depth_all.mp4"
)


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
        raise RuntimeError(f"Expected a single-channel 2-D depth measure, got {data.shape}")
    return np.asarray(data, dtype=np.float32)


def _valid_mask(depth: np.ndarray, max_depth: float) -> np.ndarray:
    return np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth)


def _open_svo(svo_path: Path) -> sl.Camera:
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    zed = sl.Camera()
    _check_status(zed.open(init), "open SVO2 with native ZED SDK depth")
    _check_status(zed.set_svo_position(0), "reset SVO position to frame 0")
    return zed


def _replay_depth(
    svo_path: Path,
    max_frames: int,
    callback: Callable[[int, int, np.ndarray], None],
) -> tuple[Any, int, tuple[int, int]]:
    """Replay from frame zero and call ``callback(frame, svo_position, depth)``."""

    zed = _open_svo(svo_path)
    try:
        info = zed.get_camera_information()
        total_frames = int(zed.get_svo_number_of_frames())
        limit = total_frames if max_frames == 0 else min(max_frames, total_frames)
        depth_mat = sl.Mat()
        runtime = sl.RuntimeParameters()
        count = 0
        shape: tuple[int, int] | None = None
        while count < limit:
            status = zed.grab(runtime)
            if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            _check_status(status, "grab SVO frame")
            _check_status(
                zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH),
                "retrieve native ZED MEASURE.DEPTH",
            )
            depth = _depth_array(depth_mat)
            if shape is None:
                shape = depth.shape
            elif depth.shape != shape:
                raise RuntimeError(
                    f"Depth shape changed at frame {count}: {depth.shape}, expected {shape}"
                )
            callback(count, int(zed.get_svo_position()), depth)
            count += 1
        if shape is None:
            raise RuntimeError("No native ZED depth frames were replayed")
        return info, count, shape
    finally:
        zed.close()


def _estimate_range(
    svo_path: Path,
    max_frames: int,
    low_percentile: float,
    high_percentile: float,
    max_depth: float,
    frame_stride: int,
    pixel_stride: int,
) -> tuple[Any, int, tuple[int, int], float, float, int]:
    samples: list[np.ndarray] = []
    sampled_frame_count = 0

    def collect(frame_index: int, _svo_position: int, depth: np.ndarray) -> None:
        nonlocal sampled_frame_count
        if frame_index > 0 and frame_index % 500 == 0:
            print(f"Range pass replayed {frame_index} frames", flush=True)
        if frame_index % frame_stride != 0:
            return
        sampled = depth[::pixel_stride, ::pixel_stride]
        valid = _valid_mask(sampled, max_depth)
        if np.any(valid):
            samples.append(np.asarray(sampled[valid], dtype=np.float32))
            sampled_frame_count += 1

    info, count, shape = _replay_depth(svo_path, max_frames, collect)
    if not samples:
        raise RuntimeError("No valid native ZED depth values found during range estimation")
    values = np.concatenate(samples)
    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, high_percentile))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise RuntimeError(f"Invalid native depth display range: {low} .. {high}")
    return info, count, shape, low, high, sampled_frame_count


def _even(value: int) -> int:
    value = max(2, value)
    return value if value % 2 == 0 else value - 1


def _render_depth(
    depth: np.ndarray,
    low: float,
    high: float,
    max_depth: float,
    frame_index: int,
    total_frames: int,
    fps: float,
    scale: float,
) -> tuple[np.ndarray, float]:
    valid = _valid_mask(depth, max_depth)
    gray = np.zeros(depth.shape, dtype=np.uint8)
    gray[valid] = np.clip(
        (depth[valid] - low) * 255.0 / (high - low), 0.0, 255.0
    ).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~valid] = 0

    valid_ratio = float(np.count_nonzero(valid) / valid.size)
    label = (
        f"ZED SDK MEASURE.DEPTH | frame {frame_index:06d}/{total_frames - 1:06d} | "
        f"range={low:.2f}-{high:.2f} m | valid={valid_ratio:.1%} | blue=near red=far"
    )
    bar_height = 36
    cv2.rectangle(color, (0, 0), (min(color.shape[1] - 1, 1180), bar_height), (0, 0, 0), -1)
    cv2.putText(
        color,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if scale != 1.0:
        color = cv2.resize(
            color,
            (_even(round(color.shape[1] * scale)), _even(round(color.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(color), valid_ratio


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export every native ZED SDK MEASURE.DEPTH frame from an SVO2 to video."
    )
    parser.add_argument("svo", nargs="?", type=Path, default=DEFAULT_SVO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--codec", default="mp4v", help="FourCC codec (default: mp4v)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all SVO frames")
    parser.add_argument("--scale", type=float, default=1.0, help="Video scale (default: 1.0)")
    parser.add_argument("--percentile-low", type=float, default=5.0)
    parser.add_argument("--percentile-high", type=float, default=95.0)
    parser.add_argument("--max-depth", type=float, default=100.0)
    parser.add_argument(
        "--range-frame-stride",
        type=int,
        default=30,
        help="Use every Nth frame for display-range estimation (default: 30)",
    )
    parser.add_argument(
        "--range-pixel-stride",
        type=int,
        default=16,
        help="Use every Nth pixel for display-range estimation (default: 16)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if len(args.codec) != 4:
        parser.error("--codec must contain exactly four characters")
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be greater than 0")
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")
    if not np.isfinite(args.scale) or args.scale <= 0:
        parser.error("--scale must be greater than 0")
    if not 0 <= args.percentile_low < args.percentile_high <= 100:
        parser.error("percentiles must satisfy 0 <= low < high <= 100")
    if not np.isfinite(args.max_depth) or args.max_depth <= 0:
        parser.error("--max-depth must be greater than 0")
    if args.range_frame_stride <= 0 or args.range_pixel_stride <= 0:
        parser.error("range strides must be greater than 0")
    return args


def main() -> int:
    args = _parse_args()
    svo_path = args.svo.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not svo_path.is_file():
        raise FileNotFoundError(svo_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}; use --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    print("Pass 1/2: replaying native ZED depth to estimate a stable display range")
    info, estimated_count, shape, low, high, sampled_frame_count = _estimate_range(
        svo_path,
        args.max_frames,
        args.percentile_low,
        args.percentile_high,
        args.max_depth,
        args.range_frame_stride,
        args.range_pixel_stride,
    )
    total_svo_frames = int(estimated_count if args.max_frames else 0)
    if total_svo_frames == 0:
        # The second replay obtains the authoritative complete SVO count below.
        probe = _open_svo(svo_path)
        try:
            total_svo_frames = int(probe.get_svo_number_of_frames())
        finally:
            probe.close()
    requested_frames = total_svo_frames if args.max_frames == 0 else min(args.max_frames, total_svo_frames)
    resolution = info.camera_configuration.resolution
    recorded_fps = float(info.camera_configuration.fps)
    fps = args.fps if np.isfinite(args.fps) and args.fps > 0 else recorded_fps
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    print(f"SDK version: {sl.Camera.get_sdk_version()}")
    print(f"Camera: {_status_name(info.camera_model)}")
    print(f"Native depth mode: NEURAL; measure: MEASURE.DEPTH; unit: METER")
    print(f"SVO frames: {total_svo_frames}; requested export: {requested_frames}")
    print(f"Depth resolution: {shape[1]}x{shape[0]}; display range: {low:.3f} .. {high:.3f} m")
    print(f"Range-estimation frames with valid samples: {sampled_frame_count}")

    output_width = _even(round(shape[1] * args.scale))
    output_height = _even(round(shape[0] * args.scale))
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    exported_count = 0
    valid_ratios: list[float] = []

    def write_frame(frame_index: int, _svo_position: int, depth: np.ndarray) -> None:
        nonlocal exported_count
        frame, valid_ratio = _render_depth(
            depth,
            low,
            high,
            args.max_depth,
            frame_index,
            requested_frames,
            fps,
            args.scale,
        )
        writer.write(frame)
        exported_count += 1
        valid_ratios.append(valid_ratio)
        if exported_count % 500 == 0 or exported_count == requested_frames:
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"Rendered {exported_count}/{requested_frames} frames "
                f"({exported_count / elapsed:.1f} fps wall-clock)"
            )

    print("Pass 2/2: replaying from SVO frame 0 and writing every native depth frame")
    try:
        _info2, replayed_count, shape2 = _replay_depth(svo_path, args.max_frames, write_frame)
    finally:
        writer.release()
    if replayed_count != requested_frames or exported_count != requested_frames:
        raise RuntimeError(
            f"Incomplete export: replayed={replayed_count}, written={exported_count}, "
            f"expected={requested_frames}"
        )
    if shape2 != shape:
        raise RuntimeError(f"Depth shape changed between passes: {shape} vs {shape2}")

    elapsed = time.monotonic() - started
    metadata_path = output_path.with_name(output_path.stem + "_metadata.json")
    metadata = {
        "input_file": str(svo_path),
        "output_video": str(output_path),
        "sdk_version": str(sl.Camera.get_sdk_version()),
        "camera_model": _status_name(info.camera_model),
        "depth_source": "pyzed.sl.MEASURE.DEPTH",
        "depth_mode": "NEURAL",
        "calibration_source": "embedded SVO/ZED SDK calibration",
        "custom_calibration_used": False,
        "coordinate_units": "METER",
        "input_resolution": {"width": int(resolution.width), "height": int(resolution.height)},
        "depth_resolution": {"width": shape[1], "height": shape[0]},
        "output_resolution": {"width": output_width, "height": output_height},
        "recorded_fps": recorded_fps,
        "output_fps": fps,
        "codec": args.codec,
        "scale": args.scale,
        "total_svo_frames": total_svo_frames,
        "exported_frames": exported_count,
        "display_range_m": {"low": low, "high": high},
        "display_percentiles": {"low": args.percentile_low, "high": args.percentile_high},
        "max_valid_depth_m": args.max_depth,
        "invalid_pixels": "black",
        "colormap": "TURBO; blue=near, red=far",
        "mean_valid_ratio": float(np.mean(valid_ratios)),
        "range_estimation": {
            "frame_stride": args.range_frame_stride,
            "pixel_stride": args.range_pixel_stride,
            "sampled_frames_with_valid_values": sampled_frame_count,
        },
        "elapsed_seconds": elapsed,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Output video: {output_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Exported all {exported_count} native ZED depth frames in {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
