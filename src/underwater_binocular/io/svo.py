"""SVO-oriented wrappers that deliberately delegate to :class:`ZedSession`."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .zed import ZedImagePair, ZedSession


def iter_svo_image_pairs(
    svo_path: Path,
    calibration_path: Path,
    *,
    expected_calibration: Any | None = None,
) -> Iterator[ZedImagePair]:
    """Replay raw image pairs through the single shared ZED session."""

    with ZedSession(svo_path, calibration_path, expected_calibration=expected_calibration) as session:
        yield from session.iter_image_pairs()
