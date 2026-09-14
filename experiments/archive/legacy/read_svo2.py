"""Export stereo images, depth, and WORLD camera poses from a ZED SVO/SVO2 file."""

from __future__ import annotations

import argparse
import csv
import json
import os
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
        sdk_root_path / "dependencies" / "glew" / "bin",
        sdk_root_path / "dependencies" / "opencv" / "x64" / "vc16" / "bin",
    ]
    existing_paths = [str(path) for path in search_paths if path.is_dir()]
    if not existing_paths:
        return

    current_path = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join(
        [*existing_paths, *(path for path in current_path if path)]
    )

    if hasattr(os, "add_dll_directory"):
        handles = []
        for path in existing_paths:
            handles.append(os.add_dll_directory(path))
        # Keep the handles alive for the lifetime of the process.
        _prepare_windows_dll_search_path._dll_handles = handles  # type: ignore[attr-defined]


_prepare_windows_dll_search_path()

import cv2  # noqa: E402  (DLL search path must be prepared first)
import numpy as np  # noqa: E402
import pyzed.sl as sl  # noqa: E402


DEFAULT_SVO = Path(__file__).resolve().with_name("20260802_150233.svo2")
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


def _status_name(status: Any) -> str:
    """Return a compact enum name that is convenient in CSV/JSON files."""

    return str(status).rsplit(".", 1)[-1]


def _check_status(status: Any, operation: str) -> None:
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"{operation} failed: {_status_name(status)}")


def _default_output_dir(svo_path: Path, max_frames: int) -> Path:
    suffix = f"sample{max_frames}" if max_frames > 0 else "all"
    return svo_path.parent / "output" / f"{svo_path.stem}_{suffix}"


def _sample_positions(total_frames: int, max_frames: int) -> list[int]:
    """Choose evenly spaced source positions, including the first and last frame."""

    if total_frames <= 0:
        return []
    if max_frames <= 0 or max_frames >= total_frames:
        return list(range(total_frames))

    sample_count = min(max_frames, total_frames)
    if sample_count == 1:
        return [0]

    # Integer floor distributes the 35/36-frame gaps while guaranteeing that
    # the first and last source frames are represented.
    return [
        (index * (total_frames - 1)) // (sample_count - 1)
        for index in range(sample_count)
    ]


def _as_bgr(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), np.ascontiguousarray(image)):
        raise RuntimeError(f"Could not write PNG: {path}")


def _csv_number(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "NaN"
    return f"{value:.12g}"


def _pose_values(
    pose: sl.Pose,
    tracking_state: Any,
) -> tuple[bool, float, list[float]]:
    """Return valid flag, confidence, and translation/quaternion/matrix values."""

    is_ok = tracking_state == sl.POSITIONAL_TRACKING_STATE.OK
    pose_valid = bool(pose.valid) if is_ok else False
    pose_confidence = float(pose.pose_confidence) if is_ok else float("nan")

    if not is_ok or not pose_valid:
        return pose_valid, pose_confidence, [float("nan")] * 23

    translation = np.asarray(pose.get_translation().get(), dtype=np.float64).reshape(-1)
    orientation = np.asarray(pose.get_orientation().get(), dtype=np.float64).reshape(-1)
    matrix = np.asarray(pose.pose_data().m, dtype=np.float64).reshape(4, 4)
    values = [*translation[:3], *orientation[:4], *matrix.reshape(-1)]
    if len(values) != 23:
        raise RuntimeError(f"Unexpected pose value count: {len(values)}")
    return pose_valid, pose_confidence, values


def _metadata(
    svo_path: Path,
    output_dir: Path,
    info: Any,
    total_frames: int,
    exported_frames: int,
    first_svo_position: int | None,
    last_svo_position: int | None,
    first_timestamp_ns: int | None,
    last_timestamp_ns: int | None,
    requested_max_frames: int,
    sample_positions: list[int],
) -> dict[str, Any]:
    resolution = info.camera_configuration.resolution
    fps = info.camera_configuration.fps
    if len(sample_positions) > 1:
        gaps = np.diff(np.asarray(sample_positions, dtype=np.int64))
        sampling_interval = {
            "nominal_frames": (total_frames - 1) / (len(sample_positions) - 1),
            "minimum_frames": int(gaps.min()),
            "maximum_frames": int(gaps.max()),
        }
    else:
        sampling_interval = {
            "nominal_frames": None,
            "minimum_frames": None,
            "maximum_frames": None,
        }
    return {
        "input_file": str(svo_path),
        "output_dir": str(output_dir),
        "sdk_version": str(sl.Camera.get_sdk_version()),
        "camera_model": _status_name(info.camera_model),
        "serial_number": int(info.serial_number),
        "resolution": {"width": int(resolution.width), "height": int(resolution.height)},
        "recorded_fps": float(fps),
        "total_frames": int(total_frames),
        "depth_mode": "NEURAL",
        "unit": "METER",
        "coordinate_system": "RIGHT_HANDED_Y_UP",
        "svo_real_time_mode": False,
        "positional_tracking_enabled": True,
        "pose_reference_frame": "WORLD",
        "pose_camera": "LEFT",
        "depth_aligned_to": "LEFT",
        "raw_depth_dtype": "float32",
        "raw_depth_unit": "meter",
        "sampling": {
            "mode": "uniform_svo_position",
            "requested_max_frames": int(requested_max_frames),
            "source_frame_count": int(total_frames),
            "sampled_frame_count": int(len(sample_positions)),
            "interval": sampling_interval,
            "includes_first_source_frame": bool(sample_positions and sample_positions[0] == 0),
            "includes_last_source_frame": bool(
                sample_positions and sample_positions[-1] == total_frames - 1
            ),
        },
        "export_range": {
            "frame_index_start": 0 if exported_frames else None,
            "frame_index_end_inclusive": exported_frames - 1 if exported_frames else None,
            "svo_position_start": first_svo_position,
            "svo_position_end": last_svo_position,
            "timestamp_start_ns": first_timestamp_ns,
            "timestamp_end_ns": last_timestamp_ns,
            "count": int(exported_frames),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export rectified stereo images, raw depth, preview depth, and WORLD poses."
    )
    parser.add_argument(
        "svo",
        nargs="?",
        type=Path,
        default=DEFAULT_SVO,
        help=f"SVO/SVO2 input file (default: {DEFAULT_SVO.name})",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help=(
            "Maximum number of frames sampled uniformly across the complete recording; "
            "use 0 for every frame (default: 100)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        "--save-dir",
        dest="output_dir",
        type=Path,
        default=None,
        help="Output bundle directory (the old --save-dir name is also accepted).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the preview window (recommended for headless/batch export).",
    )
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")
    return args


def main() -> int:
    args = _parse_args()
    svo_path = args.svo.expanduser().resolve()
    if not svo_path.is_file():
        raise FileNotFoundError(f"SVO/SVO2 file not found: {svo_path}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir(svo_path, args.max_frames)
    )
    left_dir = output_dir / "left"
    right_dir = output_dir / "right"
    depth_raw_dir = output_dir / "depth_raw"
    depth_preview_dir = output_dir / "depth_preview"
    for directory in (left_dir, right_dir, depth_raw_dir, depth_preview_dir):
        directory.mkdir(parents=True, exist_ok=True)

    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    zed = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open SVO2: {_status_name(status)}")

    display_enabled = not args.no_display
    exported_frames = 0
    first_svo_position: int | None = None
    last_svo_position: int | None = None
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None
    info = None

    try:
        info = zed.get_camera_information()
        tracking_parameters = sl.PositionalTrackingParameters()
        tracking_status = zed.enable_positional_tracking(tracking_parameters)
        _check_status(tracking_status, "enable_positional_tracking")

        total_frames = int(zed.get_svo_number_of_frames())
        resolution = info.camera_configuration.resolution
        print(f"SDK version: {sl.Camera.get_sdk_version()}")
        print(f"Camera: {_status_name(info.camera_model)}")
        print(f"Resolution: {resolution.width}x{resolution.height}")
        print(f"Total frames: {total_frames}")
        print(f"Output: {output_dir}")

        runtime = sl.RuntimeParameters()
        sample_positions = _sample_positions(total_frames, args.max_frames)
        print(
            f"Sampling: {len(sample_positions)} frames across the recording "
            f"(requested max: {args.max_frames})"
        )
        if len(sample_positions) > 1:
            gaps = np.diff(np.asarray(sample_positions, dtype=np.int64))
            print(
                "Adaptive interval: "
                f"{(total_frames - 1) / (len(sample_positions) - 1):.3f} frames "
                f"(actual gaps {int(gaps.min())}-{int(gaps.max())})"
            )
        left_mat = sl.Mat()
        right_mat = sl.Mat()
        depth_mat = sl.Mat()
        depth_preview_mat = sl.Mat()
        pose = sl.Pose()

        pose_csv_path = output_dir / "pose_world.csv"
        with pose_csv_path.open("w", newline="", encoding="utf-8") as pose_file:
            writer = csv.writer(pose_file)
            writer.writerow(POSE_HEADER)

            for source_position in sample_positions:
                _check_status(
                    zed.set_svo_position(source_position),
                    "set SVO position",
                )
                status = zed.grab(runtime)
                if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break
                _check_status(status, "grab")

                frame_index = exported_frames
                svo_position = int(zed.get_svo_position())
                timestamp_ns = int(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())

                _check_status(
                    zed.retrieve_image(left_mat, sl.VIEW.LEFT),
                    "retrieve left image",
                )
                _check_status(
                    zed.retrieve_image(right_mat, sl.VIEW.RIGHT),
                    "retrieve right image",
                )
                _check_status(
                    zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH),
                    "retrieve depth measure",
                )
                _check_status(
                    zed.retrieve_image(depth_preview_mat, sl.VIEW.DEPTH),
                    "retrieve depth preview",
                )

                left_image = _as_bgr(left_mat.get_data())
                right_image = _as_bgr(right_mat.get_data())
                depth = np.array(depth_mat.get_data(), dtype=np.float32, copy=True)
                depth_preview = _as_bgr(depth_preview_mat.get_data())

                _write_png(left_dir / f"left_{frame_index:06d}.png", left_image)
                _write_png(right_dir / f"right_{frame_index:06d}.png", right_image)
                np.save(depth_raw_dir / f"depth_{frame_index:06d}.npy", depth)
                _write_png(
                    depth_preview_dir / f"depth_{frame_index:06d}.png",
                    depth_preview,
                )

                tracking_state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
                tracking_state_name = _status_name(tracking_state)
                pose_valid, pose_confidence, pose_data = _pose_values(pose, tracking_state)
                writer.writerow(
                    [
                        frame_index,
                        svo_position,
                        timestamp_ns,
                        tracking_state_name,
                        int(pose_valid),
                        _csv_number(pose_confidence),
                        *(_csv_number(value) for value in pose_data),
                    ]
                )
                pose_file.flush()

                if first_svo_position is None:
                    first_svo_position = svo_position
                    first_timestamp_ns = timestamp_ns
                last_svo_position = svo_position
                last_timestamp_ns = timestamp_ns
                exported_frames += 1

                if display_enabled:
                    cv2.imshow("ZED LEFT", left_image)
                    if cv2.waitKey(1) & 0xFF == 27:
                        print("Stopped by user (Esc).")
                        break

        metadata = _metadata(
            svo_path=svo_path,
            output_dir=output_dir,
            info=info,
            total_frames=total_frames,
            exported_frames=exported_frames,
            first_svo_position=first_svo_position,
            last_svo_position=last_svo_position,
            first_timestamp_ns=first_timestamp_ns,
            last_timestamp_ns=last_timestamp_ns,
            requested_max_frames=args.max_frames,
            sample_positions=sample_positions,
        )
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
            metadata_file.write("\n")

        print(f"Exported frames: {exported_frames}")
        print(f"Pose CSV: {output_dir / 'pose_world.csv'}")
        return 0
    finally:
        zed.close()
        if display_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
