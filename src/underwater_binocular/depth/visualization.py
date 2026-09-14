"""Visualization/export consumers for :class:`DepthFrame` only."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .models import DepthFrame


def render_depth_bgr(
    frame: DepthFrame,
    *,
    min_depth_m: float,
    max_depth_m: float,
    colormap: int | None = None,
) -> np.ndarray:
    """Render depth without recomputing it; invalid values are black."""

    import cv2

    if not min_depth_m < max_depth_m:
        raise ValueError("min_depth_m must be lower than max_depth_m")
    valid = np.asarray(frame.valid_mask, dtype=bool) & np.isfinite(frame.depth_m) & (frame.depth_m > 0.0)
    normalized = np.zeros(frame.depth_m.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        (np.asarray(frame.depth_m)[valid] - min_depth_m) * 255.0 / (max_depth_m - min_depth_m),
        0.0,
        255.0,
    ).astype(np.uint8)
    result = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO if colormap is None else colormap)
    result[~valid] = 0
    return result


class DepthVideoWriter:
    """Incremental visualization writer that never buffers a depth sequence."""

    def __init__(
        self,
        output_path: Path,
        *,
        fps: float,
        min_depth_m: float,
        max_depth_m: float,
    ) -> None:
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        if not min_depth_m < max_depth_m:
            raise ValueError("min_depth_m must be lower than max_depth_m")
        self.output_path = output_path
        self.fps = float(fps)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self._writer = None
        self.count = 0

    def write(self, frame: DepthFrame) -> None:
        """Render and append one frame, creating the writer from its shape."""

        import cv2

        image = render_depth_bgr(
            frame,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
        )
        if self._writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = image.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self.output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (width, height),
            )
            if not self._writer.isOpened():
                raise OSError(f"could not open video writer: {self.output_path}")
        self._writer.write(image)
        self.count += 1

    def close(self) -> None:
        """Release the writer; safe when no frames were written."""

        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> DepthVideoWriter:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def export_depth_video(
    frames: Iterable[DepthFrame],
    output_path: Path,
    *,
    fps: float,
    min_depth_m: float = 1.5,
    max_depth_m: float = 3.5,
) -> int:
    """Write a visualization video by consuming already-computed frames."""

    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("cannot export an empty depth stream") from None
    with DepthVideoWriter(
        output_path,
        fps=fps,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    ) as writer:
        writer.write(first)
        for frame in iterator:
            writer.write(frame)
        return writer.count
