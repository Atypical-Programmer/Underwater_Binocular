"""Regenerate calibrated ZED depth maps and analyse the image centre.

The SVO is replayed sequentially with the calibration generated from
``Calibration/``.  A depth map is retrieved for every frame, but raw maps are
kept in memory unless ``--save-depth-maps-every`` is explicitly requested.
This avoids producing roughly 300 GB of float32 data for the complete SVO.

The main statistic is the exact centre pixel of the left-camera depth map.
For robustness, the script also records the median of a small centre window.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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

import cv2  # noqa: E402  (the DLL search path must be prepared first)
import numpy as np  # noqa: E402
import pyzed.sl as sl  # noqa: E402


DEFAULT_SVO = Path(__file__).resolve().with_name("20260802_150233.svo2")
DEFAULT_CALIBRATION = (
    Path(__file__).resolve().parent / "Calibration" / "zed_custom_opencv.yml"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "output"
    / "20260802_150233_custom_depth_histogram"
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


def _csv_number(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "NaN"
    return f"{value:.12g}"


def _depth_array(depth_mat: sl.Mat) -> np.ndarray:
    """Copy a ZED depth measure into a two-dimensional float32 array."""

    depth = np.asarray(depth_mat.get_data(), dtype=np.float32)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise RuntimeError(f"Expected a 2-D depth map, got {depth.shape}")
    return np.array(depth, dtype=np.float32, copy=True)


def _valid_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values) & (values > 0.0)
    return values[valid]


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
    window_values = _valid_values(depth[y0:y1, x0:x1])
    window_median = (
        float(np.median(window_values)) if window_values.size else float("nan")
    )
    return center_depth, window_median, int(window_values.size)


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


def _read_calibration_metadata(path: Path) -> dict[str, Any]:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise RuntimeError(f"Could not open calibration file: {path}")
    try:
        size_node = storage.getNode("Size")
        if not size_node.isSeq() or size_node.size() != 2:
            raise RuntimeError(f"Calibration Size must be [width, height]: {path}")
        width = int(round(size_node.at(0).real()))
        height = int(round(size_node.at(1).real()))

        def matrix(name: str) -> np.ndarray:
            value = storage.getNode(name).mat()
            if value is None:
                raise RuntimeError(f"Missing {name} in calibration file: {path}")
            return np.asarray(value, dtype=np.float64)

        k_left = matrix("K_LEFT")
        k_right = matrix("K_RIGHT")
        t_mm = matrix("T").reshape(-1)
    finally:
        storage.release()

    if k_left.shape != (3, 3) or k_right.shape != (3, 3) or t_mm.size != 3:
        raise RuntimeError(f"Invalid matrices in calibration file: {path}")
    return {
        "file": str(path),
        "size": {"width": width, "height": height},
        "left_intrinsics": {
            "fx_px": float(k_left[0, 0]),
            "fy_px": float(k_left[1, 1]),
            "cx_px": float(k_left[0, 2]),
            "cy_px": float(k_left[1, 2]),
        },
        "right_intrinsics": {
            "fx_px": float(k_right[0, 0]),
            "fy_px": float(k_right[1, 1]),
            "cx_px": float(k_right[0, 2]),
            "cy_px": float(k_right[1, 2]),
        },
        "translation_mm": t_mm.tolist(),
        "baseline_norm_mm": float(np.linalg.norm(t_mm)),
    }


def _make_init(
    svo_path: Path,
    calibration_path: Path,
    depth_mode: Any,
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
    return init


def _write_depth_preview(path: Path, depth: np.ndarray, title: str) -> None:
    valid = np.isfinite(depth) & (depth > 0.0)
    canvas = np.zeros(depth.shape, dtype=np.uint8)
    values = depth[valid].astype(np.float64)
    if values.size:
        low = float(np.percentile(values, 2))
        high = float(np.percentile(values, 98))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.min(values))
            high = low + 1.0
        canvas[valid] = np.clip(
            (depth[valid] - low) * 255.0 / (high - low), 0, 255
        ).astype(np.uint8)

    color = cv2.applyColorMap(canvas, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    cv2.putText(
        color,
        title,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), np.ascontiguousarray(color)):
        raise RuntimeError(f"Could not write depth preview: {path}")


def _write_histogram(path: Path, values: list[float], title: str) -> dict[str, Any]:
    finite = _valid_values(np.asarray(values, dtype=np.float64))
    if finite.size == 0:
        raise RuntimeError(f"No valid values available for histogram: {title}")

    low = float(np.percentile(finite, 1))
    high = float(np.percentile(finite, 99))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = low + 1.0

    clipped = np.clip(finite, low, high)
    counts, edges = np.histogram(clipped, bins=100, range=(low, high))
    canvas = np.full((820, 1400, 3), 255, dtype=np.uint8)
    left, top, right, bottom = 105, 80, 1340, 700
    plot_width = right - left
    plot_height = bottom - top
    maximum = max(1, int(np.max(counts)))

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = int(round(bottom - fraction * plot_height))
        cv2.line(canvas, (left, y), (right, y), (225, 225, 225), 1)
        label = str(int(round(fraction * maximum)))
        cv2.putText(
            canvas,
            label,
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
        y = int(round(bottom - (int(count) / maximum) * plot_height))
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
        f"valid n={finite.size:,}; plotted range=p01..p99; median={np.median(finite):.3f} m",
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
        "bins": int(len(counts)),
        "plot_range_p01_p99_m": [low, high],
        "clipped_values": int(np.count_nonzero((finite < low) | (finite > high))),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay an SVO2 with Calibration/zed_custom_opencv.yml, "
            "regenerate depth maps, and plot centre-depth histograms."
        )
    )
    parser.add_argument("svo", nargs="?", type=Path, default=DEFAULT_SVO)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="ZED-compatible calibration generated from Calibration/ parameters.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--depth-mode",
        choices=("performance", "quality", "ultra", "neural"),
        default="neural",
        help="ZED depth mode (default: neural).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to replay; 0 means the complete SVO.",
    )
    parser.add_argument(
        "--window-radius",
        type=int,
        default=2,
        help="Radius of the centre window; 2 means a 5x5 window.",
    )
    parser.add_argument(
        "--save-depth-maps-every",
        type=int,
        default=0,
        help=(
            "Save raw float32 depth maps at this frame interval; 0 saves none. "
            "Use 1 for every frame."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow known output files to be overwritten.",
    )
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")
    if args.window_radius < 0:
        parser.error("--window-radius must be >= 0")
    if args.save_depth_maps_every < 0:
        parser.error("--save-depth-maps-every must be >= 0")
    return args


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    known = [
        "center_depth.csv",
        "center_depth_histogram.png",
        "center_window_depth_histogram.png",
        "summary.json",
    ]
    existing = [path / name for name in known if (path / name).exists()]
    if existing and not overwrite:
        raise RuntimeError(
            "Output files already exist; use --overwrite or choose another directory: "
            + ", ".join(str(item) for item in existing)
        )
    if overwrite:
        for item in existing:
            item.unlink()


def main() -> int:
    args = _parse_args()
    svo_path = args.svo.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not svo_path.is_file():
        raise FileNotFoundError(f"SVO/SVO2 file not found: {svo_path}")
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
    _prepare_output_dir(output_dir, args.overwrite)

    calibration_metadata = _read_calibration_metadata(calibration_path)
    depth_modes = {
        "performance": sl.DEPTH_MODE.PERFORMANCE,
        "quality": sl.DEPTH_MODE.QUALITY,
        "ultra": sl.DEPTH_MODE.ULTRA,
        "neural": sl.DEPTH_MODE.NEURAL,
    }
    depth_mode = depth_modes[args.depth_mode]

    zed = sl.Camera()
    _check_status(
        zed.open(_make_init(svo_path, calibration_path, depth_mode)),
        "open SVO2 with custom calibration",
    )

    center_depths: list[float] = []
    window_depths: list[float] = []
    window_valid_counts: list[int] = []
    preview_paths: list[str] = []
    started = time.monotonic()
    csv_path = output_dir / "center_depth.csv"
    depth_maps_dir = output_dir / "depth_maps"
    if args.save_depth_maps_every > 0:
        depth_maps_dir.mkdir(parents=True, exist_ok=True)

    replayed_frames = 0
    total_frames = 0
    info = None
    center_x = 0
    center_y = 0
    try:
        info = zed.get_camera_information()
        total_frames = int(zed.get_svo_number_of_frames())
        resolution = info.camera_configuration.resolution
        width = int(resolution.width)
        height = int(resolution.height)
        if (width, height) != (
            calibration_metadata["size"]["width"],
            calibration_metadata["size"]["height"],
        ):
            raise RuntimeError(
                "Calibration and SVO resolutions differ: "
                f"calibration={calibration_metadata['size']}, "
                f"svo={{'width': {width}, 'height': {height}}}"
            )
        center_x = width // 2
        center_y = height // 2
        replay_limit = (
            min(args.max_frames, total_frames)
            if args.max_frames > 0
            else total_frames
        )
        preview_targets = {
            target
            for target in (0, (replay_limit - 1) // 2, replay_limit - 1)
            if target >= 0
        }

        print(f"SDK version: {sl.Camera.get_sdk_version()}")
        print(f"Camera: {_status_name(info.camera_model)}")
        print(f"Calibration: {calibration_path}")
        print(f"Resolution: {width}x{height}; centre=({center_x}, {center_y})")
        print(f"Depth mode: {args.depth_mode.upper()}")
        print(f"SVO frames: {total_frames}; replay limit: {replay_limit}")
        print(f"Output: {output_dir}")

        runtime = sl.RuntimeParameters()
        runtime.confidence_threshold = 30
        runtime.texture_confidence_threshold = 100
        runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA
        depth_mat = sl.Mat()

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(CSV_HEADER)
            while replayed_frames < replay_limit:
                status = zed.grab(runtime)
                if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break
                _check_status(status, "grab")

                _check_status(
                    zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH),
                    f"retrieve depth at frame {replayed_frames}",
                )
                depth = _depth_array(depth_mat)
                if depth.shape != (height, width):
                    raise RuntimeError(
                        f"Unexpected depth shape at frame {replayed_frames}: "
                        f"{depth.shape}; expected {(height, width)}"
                    )

                center_depth, window_median, window_count = _center_values(
                    depth, args.window_radius
                )
                center_depths.append(center_depth)
                window_depths.append(window_median)
                window_valid_counts.append(window_count)

                svo_position = int(zed.get_svo_position())
                timestamp_ns = int(
                    zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                )
                writer.writerow(
                    [
                        replayed_frames,
                        svo_position,
                        timestamp_ns,
                        center_x,
                        center_y,
                        _csv_number(center_depth),
                        _csv_number(window_median),
                        window_count,
                    ]
                )

                if (
                    args.save_depth_maps_every > 0
                    and replayed_frames % args.save_depth_maps_every == 0
                ):
                    np.save(
                        depth_maps_dir / f"depth_{replayed_frames:06d}.npy",
                        depth,
                    )

                if replayed_frames in preview_targets:
                    preview_path = output_dir / (
                        f"depth_preview_frame_{replayed_frames:06d}.png"
                    )
                    _write_depth_preview(
                        preview_path,
                        depth,
                        f"Depth preview; frame {replayed_frames}; centre {center_depth:.3f} m",
                    )
                    preview_paths.append(str(preview_path))

                replayed_frames += 1
                if replayed_frames % 1000 == 0 or replayed_frames == 1:
                    elapsed = time.monotonic() - started
                    rate = replayed_frames / elapsed if elapsed > 0 else 0.0
                    print(
                        f"Replay: {replayed_frames}/{replay_limit} frames, "
                        f"{rate:.1f} frames/s, "
                        f"centre={center_depth:.3f} m",
                        flush=True,
                    )
            csv_file.flush()
            os.fsync(csv_file.fileno())
    finally:
        zed.close()

    if replayed_frames == 0:
        raise RuntimeError("No depth frames were generated")
    if replayed_frames != replay_limit:
        raise RuntimeError(
            f"SVO ended before the requested replay limit: "
            f"{replayed_frames}/{replay_limit} frames"
        )

    histogram_path = output_dir / "center_depth_histogram.png"
    window_histogram_path = output_dir / "center_window_depth_histogram.png"
    histogram_metadata = _write_histogram(
        histogram_path,
        center_depths,
        "Centre depth histogram (exact centre pixel)",
    )
    window_histogram_metadata = _write_histogram(
        window_histogram_path,
        window_depths,
        f"Centre depth histogram ({2 * args.window_radius + 1}x{2 * args.window_radius + 1} median)",
    )

    elapsed_seconds = time.monotonic() - started
    metadata = {
        "input_file": str(svo_path),
        "calibration": calibration_metadata,
        "sdk_version": str(sl.Camera.get_sdk_version()),
        "camera_model": _status_name(info.camera_model),
        "resolution": {"width": width, "height": height},
        "recorded_fps": float(info.camera_configuration.fps),
        "depth_mode": args.depth_mode.upper(),
        "coordinate_units": "METER",
        "runtime_confidence_threshold": 30,
        "runtime_texture_confidence_threshold": 100,
        "measure3D_reference_frame": "CAMERA",
        "sequential_replay": True,
        "total_svo_frames": total_frames,
        "frames_replayed": replayed_frames,
        "center_pixel": {"x": center_x, "y": center_y},
        "center_window": {
            "radius": args.window_radius,
            "width": 2 * args.window_radius + 1,
            "height": 2 * args.window_radius + 1,
        },
        "center_pixel_stats_m": _stats(center_depths),
        "center_window_median_stats_m": _stats(window_depths),
        "center_window_valid_count_stats": _stats(
            [float(value) for value in window_valid_counts]
        ),
        "files": {
            "center_depth_csv": str(csv_path),
            "center_depth_histogram": histogram_metadata,
            "center_window_depth_histogram": window_histogram_metadata,
            "depth_preview_images": preview_paths,
            "raw_depth_maps_directory": (
                str(depth_maps_dir) if args.save_depth_maps_every > 0 else None
            ),
            "raw_depth_map_interval": args.save_depth_maps_every,
        },
        "elapsed_seconds": elapsed_seconds,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Finished: {replayed_frames} frames in {elapsed_seconds:.1f}s")
    print(f"Center depth CSV: {csv_path}")
    print(f"Histogram: {histogram_path}")
    print(f"Window histogram: {window_histogram_path}")
    print(f"Summary: {summary_path}")
    print(
        "Exact centre depth: "
        f"valid={metadata['center_pixel_stats_m']['count']}/"
        f"{replayed_frames}, "
        f"median={metadata['center_pixel_stats_m'].get('median_m')} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
