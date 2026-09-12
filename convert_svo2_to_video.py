"""Convert a ZED SVO/SVO2 recording to a side-by-side stereo video.

Each output frame contains the left image followed by the right image:

    [ LEFT | RIGHT ]

The default image source is the rectified ``VIEW.LEFT``/``VIEW.RIGHT`` pair.
Use ``--unrectified`` when the original camera images are required instead.
"""

from __future__ import annotations

import argparse
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
        # Keep the handles alive for the lifetime of the process.
        _prepare_windows_dll_search_path._dll_handles = handles  # type: ignore[attr-defined]


_prepare_windows_dll_search_path()

import cv2  # noqa: E402  (the DLL search path must be prepared first)
import numpy as np  # noqa: E402
import pyzed.sl as sl  # noqa: E402


DEFAULT_SVO = Path(__file__).resolve().with_name("20260802_150233.svo2")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "output"
    / "20260802_150233_left_right.mp4"
)


def _status_name(status: Any) -> str:
    return str(status).rsplit(".", 1)[-1]


def _check_status(status: Any, operation: str) -> None:
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"{operation} failed: {_status_name(status)}")


def _as_bgr(image: np.ndarray) -> np.ndarray:
    """Convert a ZED image to a contiguous 8-bit, three-channel BGR array."""

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise RuntimeError(f"Expected a 3- or 4-channel image, got {image.shape}")
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _even(value: int) -> int:
    """Return a positive even dimension accepted by common video codecs."""

    value = max(2, value)
    return value if value % 2 == 0 else value - 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a ZED SVO/SVO2 into a horizontal [LEFT | RIGHT] video."
    )
    parser.add_argument(
        "svo",
        nargs="?",
        type=Path,
        default=DEFAULT_SVO,
        help=f"SVO/SVO2 input file (default: {DEFAULT_SVO.name})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output video path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC codec used by OpenCV (default: mp4v).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale each eye before concatenating (default: 1.0).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum number of frames; 0 means the complete recording (default: 0).",
    )
    parser.add_argument(
        "--unrectified",
        action="store_true",
        help="Use LEFT_UNRECTIFIED/RIGHT_UNRECTIFIED instead of rectified views.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output video.",
    )
    args = parser.parse_args()
    if len(args.codec) != 4:
        parser.error("--codec must contain exactly four characters")
    if not np.isfinite(args.scale) or args.scale <= 0:
        parser.error("--scale must be greater than 0")
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")
    return args


def _write_metadata(
    path: Path,
    svo_path: Path,
    output_path: Path,
    info: Any,
    total_frames: int,
    exported_frames: int,
    output_width: int,
    output_height: int,
    fps: float,
    codec: str,
    scale: float,
    unrectified: bool,
    elapsed_seconds: float,
) -> None:
    resolution = info.camera_configuration.resolution
    metadata = {
        "input_file": str(svo_path),
        "output_file": str(output_path),
        "sdk_version": str(sl.Camera.get_sdk_version()),
        "camera_model": _status_name(info.camera_model),
        "input_resolution": {
            "width": int(resolution.width),
            "height": int(resolution.height),
        },
        "output_resolution": {"width": output_width, "height": output_height},
        "recorded_fps": fps,
        "codec": codec,
        "scale": scale,
        "image_source": "LEFT_UNRECTIFIED + RIGHT_UNRECTIFIED"
        if unrectified
        else "LEFT + RIGHT",
        "layout": "horizontal; left camera on the left, right camera on the right",
        "total_svo_frames": total_frames,
        "exported_frames": exported_frames,
        "elapsed_seconds": elapsed_seconds,
    }
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    svo_path = args.svo.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not svo_path.is_file():
        raise FileNotFoundError(f"SVO/SVO2 file not found: {svo_path}")
    if output_path == svo_path:
        raise ValueError("The output video path must differ from the SVO/SVO2 input path")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}; use --overwrite to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    # Depth is not needed for image export and disabling it avoids unnecessary GPU work.
    init.depth_mode = sl.DEPTH_MODE.NONE

    zed = sl.Camera()
    _check_status(zed.open(init), "open SVO2")

    writer: cv2.VideoWriter | None = None
    exported_frames = 0
    started = time.monotonic()
    try:
        info = zed.get_camera_information()
        total_frames = int(zed.get_svo_number_of_frames())
        camera_fps = float(info.camera_configuration.fps)
        fps = camera_fps if np.isfinite(camera_fps) and camera_fps > 0 else 30.0

        if args.unrectified:
            left_view = sl.VIEW.LEFT_UNRECTIFIED
            right_view = sl.VIEW.RIGHT_UNRECTIFIED
            source_name = "LEFT_UNRECTIFIED + RIGHT_UNRECTIFIED"
        else:
            left_view = sl.VIEW.LEFT
            right_view = sl.VIEW.RIGHT
            source_name = "LEFT + RIGHT"

        left_mat = sl.Mat()
        right_mat = sl.Mat()
        runtime = sl.RuntimeParameters()
        fourcc = cv2.VideoWriter_fourcc(*args.codec)

        print(f"SDK version: {sl.Camera.get_sdk_version()}")
        print(f"Camera: {_status_name(info.camera_model)}")
        print(f"Input: {svo_path}")
        print(f"Image source: {source_name}")
        print(f"SVO frames: {total_frames}")

        while args.max_frames == 0 or exported_frames < args.max_frames:
            status = zed.grab(runtime)
            if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            _check_status(status, "grab")

            _check_status(
                zed.retrieve_image(left_mat, left_view),
                "retrieve left image",
            )
            _check_status(
                zed.retrieve_image(right_mat, right_view),
                "retrieve right image",
            )

            left = _as_bgr(left_mat.get_data())
            right = _as_bgr(right_mat.get_data())
            if left.shape != right.shape:
                raise RuntimeError(
                    "Left/right image shapes differ: "
                    f"{left.shape} vs {right.shape}"
                )

            frame = np.hstack((left, right))
            if args.scale != 1.0:
                output_width = _even(int(round(frame.shape[1] * args.scale)))
                output_height = _even(int(round(frame.shape[0] * args.scale)))
                frame = cv2.resize(
                    frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_AREA
                    if args.scale < 1.0
                    else cv2.INTER_LINEAR,
                )
            else:
                output_height = _even(frame.shape[0])
                output_width = _even(frame.shape[1])
                if (output_width, output_height) != (frame.shape[1], frame.shape[0]):
                    frame = frame[:output_height, :output_width]

            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_path), fourcc, fps, (output_width, output_height)
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        f"Could not open video writer for {output_path} "
                        f"with codec {args.codec!r}"
                    )
                print(
                    f"Output: {output_path} ({output_width}x{output_height} @ {fps:g} fps)"
                )

            writer.write(frame)
            exported_frames += 1

            if exported_frames % 1000 == 0 or exported_frames == 1:
                elapsed = time.monotonic() - started
                rate = exported_frames / elapsed if elapsed > 0 else 0.0
                progress = (
                    f"/{total_frames}" if total_frames > 0 else ""
                )
                print(
                    f"frame {exported_frames}{progress}, "
                    f"{rate:.1f} frames/s",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.release()
        zed.close()

    if exported_frames == 0:
        raise RuntimeError("No frames were exported")

    elapsed_seconds = time.monotonic() - started
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    _write_metadata(
        path=metadata_path,
        svo_path=svo_path,
        output_path=output_path,
        info=info,
        total_frames=total_frames,
        exported_frames=exported_frames,
        output_width=output_width,
        output_height=output_height,
        fps=fps,
        codec=args.codec,
        scale=args.scale,
        unrectified=args.unrectified,
        elapsed_seconds=elapsed_seconds,
    )
    print(
        f"Finished: {exported_frames} frames in {elapsed_seconds:.1f}s\n"
        f"Video: {output_path}\n"
        f"Metadata: {metadata_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
