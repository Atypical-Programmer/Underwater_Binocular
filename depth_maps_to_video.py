"""Render saved floating-point depth maps as a high-contrast video.

The input ``.npy`` files are expected to contain 2-D depth arrays in metres.
Invalid values (NaN, infinity, non-positive values, and values above the
configured maximum) are rendered black.  A robust range is estimated from
all frames and then reused for every frame, so the colour scale is stable
throughout the video.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def _depth_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("depth_*.npy"))
    if not files:
        raise FileNotFoundError(f"No depth_*.npy files found in {input_dir}")
    return files


def _load_depth(path: Path) -> np.ndarray:
    depth = np.load(path, mmap_mode="r", allow_pickle=False)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2-D depth map in {path}, got {depth.shape}")
    if not np.issubdtype(depth.dtype, np.number):
        raise TypeError(f"Expected a numeric depth map in {path}, got {depth.dtype}")
    return np.asarray(depth, dtype=np.float32)


def _valid_mask(depth: np.ndarray, max_depth: float) -> np.ndarray:
    return np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth)


def _estimate_range(
    paths: list[Path],
    low_percentile: float,
    high_percentile: float,
    max_depth: float,
    sample_stride: int,
) -> tuple[float, float]:
    samples: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    for path in paths:
        depth = _load_depth(path)
        if shape is None:
            shape = depth.shape
        elif depth.shape != shape:
            raise ValueError(
                f"Depth map shape mismatch: {path} has {depth.shape}, expected {shape}"
            )
        sampled = depth[::sample_stride, ::sample_stride]
        valid = _valid_mask(sampled, max_depth)
        if np.any(valid):
            samples.append(np.asarray(sampled[valid], dtype=np.float32))

    if not samples:
        raise ValueError("No valid depth values found in the input maps")
    values = np.concatenate(samples)
    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, high_percentile))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"Invalid display range estimated from depth maps: {low}, {high}")
    return low, high


def _render_depth(
    depth: np.ndarray,
    low: float,
    high: float,
    max_depth: float,
    frame_index: int,
    fps: float,
) -> tuple[np.ndarray, float]:
    valid = _valid_mask(depth, max_depth)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    scaled = (depth - low) * (255.0 / (high - low))
    normalized[valid] = np.clip(scaled[valid], 0.0, 255.0).astype(np.uint8)

    # Turbo gives a high-contrast continuous map: blue is near, red is far.
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color[~valid] = 0

    valid_ratio = float(np.count_nonzero(valid) / valid.size)
    label = (
        f"frame {frame_index:06d} | t={frame_index / fps:.2f}s | "
        f"range={low:.2f}-{high:.2f} m | valid={valid_ratio:.1%}"
    )
    cv2.rectangle(color, (0, 0), (min(color.shape[1] - 1, 760), 34), (0, 0, 0), -1)
    cv2.putText(
        color,
        label,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.ascontiguousarray(color), valid_ratio


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render saved .npy depth maps as a stable, high-contrast video."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing depth_*.npy maps")
    parser.add_argument("--output", type=Path, required=True, help="Output video path")
    parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate (default: 30)")
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC codec used by OpenCV (default: mp4v)",
    )
    parser.add_argument(
        "--percentile-low",
        type=float,
        default=1.0,
        help="Lower valid-depth percentile for contrast stretching (default: 1)",
    )
    parser.add_argument(
        "--percentile-high",
        type=float,
        default=99.0,
        help="Upper valid-depth percentile for contrast stretching (default: 99)",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=100.0,
        help="Depth values above this are invalid (default: 100 m)",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=8,
        help="Pixel stride used while estimating the global range (default: 8)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing video")
    args = parser.parse_args()
    if len(args.codec) != 4:
        parser.error("--codec must contain exactly four characters")
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be greater than 0")
    if not np.isfinite(args.max_depth) or args.max_depth <= 0:
        parser.error("--max-depth must be greater than 0")
    if not 0 <= args.percentile_low < args.percentile_high <= 100:
        parser.error("percentiles must satisfy 0 <= low < high <= 100")
    if args.sample_stride <= 0:
        parser.error("--sample-stride must be greater than 0")
    return args


def main() -> int:
    args = _parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; use --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    paths = _depth_files(input_dir)
    started = time.monotonic()
    low, high = _estimate_range(
        paths,
        args.percentile_low,
        args.percentile_high,
        args.max_depth,
        args.sample_stride,
    )

    first_depth = _load_depth(paths[0])
    height, width = first_depth.shape
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    valid_ratios: list[float] = []
    try:
        for frame_index, path in enumerate(paths):
            depth = _load_depth(path)
            if depth.shape != (height, width):
                raise ValueError(
                    f"Depth map shape mismatch: {path} has {depth.shape}, expected {(height, width)}"
                )
            frame, valid_ratio = _render_depth(
                depth, low, high, args.max_depth, frame_index, args.fps
            )
            writer.write(frame)
            valid_ratios.append(valid_ratio)
            if (frame_index + 1) % 10 == 0 or frame_index + 1 == len(paths):
                print(f"Rendered {frame_index + 1}/{len(paths)} depth frames")
    finally:
        writer.release()

    elapsed = time.monotonic() - started
    metadata_path = output_path.with_name(output_path.stem + "_metadata.json")
    metadata = {
        "input_directory": str(input_dir),
        "output_video": str(output_path),
        "frame_count": len(paths),
        "resolution": {"width": width, "height": height},
        "fps": args.fps,
        "codec": args.codec,
        "depth_unit": "m",
        "display_range_m": {"low": low, "high": high},
        "display_percentiles": {
            "low": args.percentile_low,
            "high": args.percentile_high,
        },
        "max_valid_depth_m": args.max_depth,
        "invalid_pixels": "black",
        "colormap": "TURBO; blue=near, red=far",
        "mean_valid_ratio": float(np.mean(valid_ratios)),
        "elapsed_seconds": elapsed,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Output video: {output_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Global display range: {low:.3f} .. {high:.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
