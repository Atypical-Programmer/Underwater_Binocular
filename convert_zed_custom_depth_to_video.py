"""Export the ZED SDK depth map using the Custom calibration.

This is the production video exporter for the current project decision:

* the depth engine is ZED SDK ``MEASURE.DEPTH`` / ``DEPTH_MODE.NEURAL``;
* the SVO's embedded/native calibration is not used;
* ``Calibration/zed_custom_opencv.yml`` is supplied to ``InitParameters``;
* the complete SVO is replayed sequentially and rendered frame by frame.

The raw float32 depth maps are not written to disk.  A complete 1920x1080
recording would require roughly 300 GB of float32 ``.npy`` data.  The depth
array is converted directly to a high-contrast video frame, while the centre
pixel/window values are retained in a CSV and summarized in JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _prepare_windows_dll_search_path() -> None:
    """Make the ZED SDK DLLs discoverable before importing ``pyzed.sl``."""

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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyzed.sl as sl  # noqa: E402


DEFAULT_SVO = Path(__file__).resolve().with_name("20260802_150233.svo2")
DEFAULT_CALIBRATION = (
    Path(__file__).resolve().parent / "Calibration" / "zed_custom_opencv.yml"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "20260802_150233_custom_sdk_depth_all.mp4"
)

CSV_HEADER = [
    "frame_index",
    "svo_position",
    "timestamp_ns",
    "center_x",
    "center_y",
    "center_depth_m",
    "center_window_median_depth_m",
    "center_window_valid_count",
]


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
    return np.array(data, dtype=np.float32, copy=True)


def _valid_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values) & (values > 0.0)
    return values[valid]


def _stats(values: list[float]) -> dict[str, Any]:
    finite = _valid_values(np.asarray(values, dtype=np.float64))
    result: dict[str, Any] = {
        "count": int(finite.size),
        "invalid_count": int(len(values) - finite.size),
        "valid_ratio": float(finite.size / len(values)) if values else 0.0,
    }
    if finite.size == 0:
        return result
    result.update(
        {
            "min_m": float(np.min(finite)),
            "p01_m": float(np.percentile(finite, 1)),
            "p05_m": float(np.percentile(finite, 5)),
            "median_m": float(np.median(finite)),
            "mean_m": float(np.mean(finite)),
            "std_m": float(np.std(finite)),
            "p95_m": float(np.percentile(finite, 95)),
            "p99_m": float(np.percentile(finite, 99)),
            "max_m": float(np.max(finite)),
        }
    )
    return result


def _csv_number(value: float) -> str:
    value = float(value)
    return "NaN" if not np.isfinite(value) else f"{value:.12g}"


def _center_values(
    depth: np.ndarray,
    radius: int,
) -> tuple[float, float, int]:
    height, width = depth.shape
    center_x = width // 2
    center_y = height // 2
    center_depth = float(depth[center_y, center_x])
    y0 = max(0, center_y - radius)
    y1 = min(height, center_y + radius + 1)
    x0 = max(0, center_x - radius)
    x1 = min(width, center_x + radius + 1)
    window = _valid_values(depth[y0:y1, x0:x1])
    window_median = float(np.median(window)) if window.size else float("nan")
    return center_depth, window_median, int(window.size)


def _calibration_metadata(path: Path) -> dict[str, Any]:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise RuntimeError(f"Could not open Custom calibration file: {path}")
    try:
        size_node = storage.getNode("Size")
        if not size_node.isSeq() or size_node.size() != 2:
            raise RuntimeError(f"Calibration Size must be [width, height]: {path}")

        def matrix(name: str) -> np.ndarray:
            value = storage.getNode(name).mat()
            if value is None:
                raise RuntimeError(f"Missing {name} in calibration file: {path}")
            return np.asarray(value, dtype=np.float64)

        width = int(round(size_node.at(0).real()))
        height = int(round(size_node.at(1).real()))
        left_k = matrix("K_LEFT")
        right_k = matrix("K_RIGHT")
        left_d = matrix("D_LEFT").reshape(-1)
        right_d = matrix("D_RIGHT").reshape(-1)
        r = matrix("R").reshape(-1)
        t = matrix("T").reshape(-1)
    finally:
        storage.release()

    if left_k.shape != (3, 3) or right_k.shape != (3, 3):
        raise RuntimeError(f"Custom calibration K matrices must be 3x3: {path}")
    if left_d.size < 5 or right_d.size < 5 or r.size != 3 or t.size != 3:
        raise RuntimeError(f"Invalid Custom calibration vectors: {path}")
    return {
        "file": str(path),
        "size": {"width": width, "height": height},
        "left_intrinsics": {
            "fx_px": float(left_k[0, 0]),
            "fy_px": float(left_k[1, 1]),
            "cx_px": float(left_k[0, 2]),
            "cy_px": float(left_k[1, 2]),
        },
        "right_intrinsics": {
            "fx_px": float(right_k[0, 0]),
            "fy_px": float(right_k[1, 1]),
            "cx_px": float(right_k[0, 2]),
            "cy_px": float(right_k[1, 2]),
        },
        "left_distortion_coefficients": left_d[:5].tolist(),
        "right_distortion_coefficients": right_d[:5].tolist(),
        "R_vector": r.tolist(),
        "T_mm": t.tolist(),
        "baseline_norm_mm": float(np.linalg.norm(t)),
    }


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


def _render_depth(
    depth: np.ndarray,
    display_low: float,
    display_high: float,
    max_depth: float,
    frame_index: int,
    total_frames: int,
    fps: float,
    scale: float,
) -> tuple[np.ndarray, float]:
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth)
    gray = np.zeros(depth.shape, dtype=np.uint8)
    gray[valid] = np.clip(
        (depth[valid] - display_low) * 255.0 / (display_high - display_low),
        0.0,
        255.0,
    ).astype(np.uint8)
    # TURBO: low depth is blue; high depth is red.
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~valid] = 0

    valid_ratio = float(np.count_nonzero(valid) / valid.size)
    label = (
        "ZED SDK MEASURE.DEPTH | Custom calibration | "
        f"frame {frame_index:06d}/{total_frames - 1:06d} | "
        f"display={display_low:.2f}-{display_high:.2f} m | "
        f"valid={valid_ratio:.1%} | blue=near red=far"
    )
    cv2.rectangle(
        color,
        (0, 0),
        (min(color.shape[1] - 1, 1490), 38),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        color,
        label,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.61,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if scale != 1.0:
        width = max(2, int(round(color.shape[1] * scale)))
        height = max(2, int(round(color.shape[0] * scale)))
        width -= width % 2
        height -= height % 2
        color = cv2.resize(color, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(color), valid_ratio


def _write_histogram(path: Path, values: list[float], title: str) -> dict[str, Any]:
    finite = _valid_values(np.asarray(values, dtype=np.float64))
    if finite.size == 0:
        raise RuntimeError(f"No valid values available for histogram: {title}")
    low = float(np.percentile(finite, 1))
    high = float(np.percentile(finite, 99))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = low + 1.0
    counts, _ = np.histogram(np.clip(finite, low, high), bins=100, range=(low, high))
    canvas = np.full((820, 1400, 3), 255, dtype=np.uint8)
    left, top, right, bottom = 105, 80, 1340, 700
    plot_width = right - left
    plot_height = bottom - top
    maximum = max(1, int(np.max(counts)))
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = int(round(bottom - fraction * plot_height))
        cv2.line(canvas, (left, y), (right, y), (225, 225, 225), 1)
        cv2.putText(
            canvas,
            str(int(round(fraction * maximum))),
            (left - 75, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
    bar_width = plot_width / len(counts)
    for index, count in enumerate(counts):
        x0 = int(round(left + index * bar_width))
        x1 = max(x0 + 1, int(round(left + (index + 1) * bar_width)) - 1)
        y = int(round(bottom - int(count) / maximum * plot_height))
        cv2.rectangle(canvas, (x0, y), (x1, bottom), (190, 80, 35), -1)
    cv2.line(canvas, (left, top), (left, bottom), (30, 30, 30), 2)
    cv2.line(canvas, (left, bottom), (right, bottom), (30, 30, 30), 2)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = int(round(left + fraction * plot_width))
        value = low + fraction * (high - low)
        cv2.line(canvas, (x, bottom), (x, bottom + 8), (30, 30, 30), 2)
        cv2.putText(
            canvas,
            f"{value:.2f}",
            (x - 35, bottom + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        title,
        (left, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Depth (m)",
        (right - 125, bottom + 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"valid n={finite.size:,}; plotted p01..p99; median={np.median(finite):.3f} m",
        (left, 775),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Could not write histogram: {path}")
    return {
        "file": str(path),
        "bins": 100,
        "plot_range_p01_p99_m": [low, high],
        "clipped_values": int(np.count_nonzero((finite < low) | (finite > high))),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export all ZED SDK MEASURE.DEPTH frames using Custom calibration."
    )
    parser.add_argument("svo", nargs="?", type=Path, default=DEFAULT_SVO)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all SVO frames")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--display-min-depth", type=float, default=1.5)
    parser.add_argument("--display-max-depth", type=float, default=3.5)
    parser.add_argument("--max-depth", type=float, default=100.0)
    parser.add_argument("--window-radius", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if len(args.codec) != 4:
        parser.error("--codec must contain exactly four characters")
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be greater than 0")
    if args.max_frames < 0 or args.window_radius < 0:
        parser.error("--max-frames and --window-radius must be >= 0")
    if not np.isfinite(args.scale) or args.scale <= 0:
        parser.error("--scale must be greater than 0")
    if (
        not np.isfinite(args.display_min_depth)
        or not np.isfinite(args.display_max_depth)
        or args.display_min_depth >= args.display_max_depth
    ):
        parser.error("display depth range must be finite and min < max")
    if not np.isfinite(args.max_depth) or args.max_depth <= 0:
        parser.error("--max-depth must be greater than 0")
    return args


def main() -> int:
    args = _parse_args()
    svo_path = args.svo.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not svo_path.is_file():
        raise FileNotFoundError(svo_path)
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}; use --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    calibration = _calibration_metadata(calibration_path)
    zed = sl.Camera()
    _check_status(
        zed.open(_make_init(svo_path, calibration_path)),
        "open SVO2 with Custom calibration and ZED SDK depth",
    )

    writer: cv2.VideoWriter | None = None
    csv_path = output_path.with_name(output_path.stem + "_center_depth.csv")
    histogram_path = output_path.with_name(output_path.stem + "_center_depth_histogram.png")
    window_histogram_path = output_path.with_name(
        output_path.stem + "_center_window_depth_histogram.png"
    )
    metadata_path = output_path.with_name(output_path.stem + "_metadata.json")
    started = time.monotonic()
    center_depths: list[float] = []
    window_depths: list[float] = []
    window_counts: list[int] = []
    valid_ratios: list[float] = []
    replayed = 0
    total_frames = 0
    info = None

    try:
        info = zed.get_camera_information()
        total_frames = int(zed.get_svo_number_of_frames())
        replay_limit = (
            min(args.max_frames, total_frames)
            if args.max_frames > 0
            else total_frames
        )
        resolution = info.camera_configuration.resolution
        recorded_fps = float(info.camera_configuration.fps)
        fps = args.fps if np.isfinite(args.fps) and args.fps > 0 else recorded_fps
        runtime = sl.RuntimeParameters()
        runtime.confidence_threshold = 30
        runtime.texture_confidence_threshold = 100
        runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA
        depth_mat = sl.Mat()

        print(f"SDK version: {sl.Camera.get_sdk_version()}")
        print(f"Camera: {_status_name(info.camera_model)}")
        print("Depth source: ZED SDK MEASURE.DEPTH; mode: NEURAL")
        print(f"Custom calibration: {calibration_path}")
        print("Native SVO calibration: excluded")
        print(f"SVO frames: {total_frames}; export limit: {replay_limit}")
        print(
            f"Display range: {args.display_min_depth:.3f} .. "
            f"{args.display_max_depth:.3f} m; blue=near, red=far"
        )

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(CSV_HEADER)
            while replayed < replay_limit:
                status = zed.grab(runtime)
                if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break
                _check_status(status, f"grab frame {replayed}")
                _check_status(
                    zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH),
                    f"retrieve Custom-calibrated SDK depth at frame {replayed}",
                )
                depth = _depth_array(depth_mat)
                if writer is None:
                    out_width = max(2, int(round(depth.shape[1] * args.scale)))
                    out_height = max(2, int(round(depth.shape[0] * args.scale)))
                    out_width -= out_width % 2
                    out_height -= out_height % 2
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*args.codec),
                        fps,
                        (out_width, out_height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not open video writer: {output_path}")
                    expected_shape = depth.shape
                elif depth.shape != expected_shape:
                    raise RuntimeError(
                        f"Depth shape changed at frame {replayed}: "
                        f"{depth.shape} vs {expected_shape}"
                    )

                center_depth, window_median, window_count = _center_values(
                    depth, args.window_radius
                )
                center_depths.append(center_depth)
                window_depths.append(window_median)
                window_counts.append(window_count)

                frame, valid_ratio = _render_depth(
                    depth,
                    args.display_min_depth,
                    args.display_max_depth,
                    args.max_depth,
                    replayed,
                    replay_limit,
                    fps,
                    args.scale,
                )
                writer.write(frame)
                valid_ratios.append(valid_ratio)

                csv_writer.writerow(
                    [
                        replayed,
                        int(zed.get_svo_position()),
                        int(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()),
                        depth.shape[1] // 2,
                        depth.shape[0] // 2,
                        _csv_number(center_depth),
                        _csv_number(window_median),
                        window_count,
                    ]
                )
                replayed += 1
                if replayed % 500 == 0 or replayed == 1:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    print(
                        f"Rendered {replayed}/{replay_limit} frames; "
                        f"{replayed / elapsed:.2f} fps wall-clock; "
                        f"centre={center_depth:.3f} m; valid={valid_ratio:.1%}",
                        flush=True,
                    )
            csv_file.flush()
            os.fsync(csv_file.fileno())
    finally:
        if writer is not None:
            writer.release()
        zed.close()

    if replayed == 0:
        raise RuntimeError("No SDK depth frames were exported")
    if replayed != replay_limit:
        raise RuntimeError(f"Incomplete export: {replayed}/{replay_limit} frames")
    if info is None:
        raise RuntimeError("Camera information was not available")

    center_histogram = _write_histogram(
        histogram_path,
        center_depths,
        "ZED SDK Custom-calibrated centre depth (exact centre pixel)",
    )
    window_histogram = _write_histogram(
        window_histogram_path,
        window_depths,
        f"ZED SDK Custom-calibrated centre depth ({2 * args.window_radius + 1}x{2 * args.window_radius + 1} median)",
    )
    elapsed = time.monotonic() - started
    metadata = {
        "input_file": str(svo_path),
        "output_video": str(output_path),
        "sdk_version": str(sl.Camera.get_sdk_version()),
        "camera_model": _status_name(info.camera_model),
        "depth_source": "pyzed.sl.MEASURE.DEPTH",
        "depth_mode": "NEURAL",
        "calibration_source": "Calibration/zed_custom_opencv.yml",
        "custom_calibration_used": True,
        "native_svo_calibration_used": False,
        "calibration": calibration,
        "camera_disable_self_calib": True,
        "coordinate_units": "METER",
        "measure3D_reference_frame": "CAMERA",
        "runtime_confidence_threshold": 30,
        "runtime_texture_confidence_threshold": 100,
        "input_resolution": {
            "width": int(info.camera_configuration.resolution.width),
            "height": int(info.camera_configuration.resolution.height),
        },
        "depth_resolution": {"width": int(expected_shape[1]), "height": int(expected_shape[0])},
        "output_resolution": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
        "recorded_fps": float(info.camera_configuration.fps),
        "output_fps": fps,
        "codec": args.codec,
        "scale": args.scale,
        "total_svo_frames": total_frames,
        "exported_frames": replayed,
        "display_range_m": {
            "low": args.display_min_depth,
            "high": args.display_max_depth,
            "purpose": "fixed global high-contrast visualization; values outside are clipped",
        },
        "max_valid_depth_m": args.max_depth,
        "invalid_pixels": "black",
        "colormap": "TURBO; blue=near, red=far",
        "center_pixel": {
            "x": int(expected_shape[1] // 2),
            "y": int(expected_shape[0] // 2),
        },
        "center_window_radius": args.window_radius,
        "center_pixel_stats_m": _stats(center_depths),
        "center_window_median_stats_m": _stats(window_depths),
        "center_window_valid_count_stats": _stats(
            [float(value) for value in window_counts]
        ),
        "mean_valid_ratio": float(np.mean(valid_ratios)),
        "files": {
            "center_depth_csv": str(csv_path),
            "center_depth_histogram": center_histogram,
            "center_window_depth_histogram": window_histogram,
        },
        "elapsed_seconds": elapsed,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Finished: {replayed} frames in {elapsed:.1f} s")
    print(f"Output video: {output_path}")
    print(f"Metadata: {metadata_path}")
    print(
        "Centre depth median: "
        f"{metadata['center_pixel_stats_m'].get('median_m')} m; "
        f"valid={metadata['center_pixel_stats_m']['count']}/{replayed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
