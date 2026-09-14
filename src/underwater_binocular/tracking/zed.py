"""Shared ZED tracking configuration for GEN_1/GEN_3 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..io.zed import ZedSession


@dataclass(frozen=True)
class TrackingConfig:
    """Tracking mode settings; GEN_1/GEN_3 are configuration, not duplicate code."""

    mode: str = "GEN_1"
    enable_area_memory: bool = True
    enable_imu_fusion: bool = True
    enable_pose_smoothing: bool = False
    set_gravity_as_origin: bool = True
    set_floor_as_origin: bool = False
    set_as_static: bool = False


def start_tracking(session: ZedSession, config: TrackingConfig | None = None) -> Any:
    """Enable positional tracking on an open session using one shared policy."""

    if not session.is_open:
        raise RuntimeError("ZED session must be open before tracking starts")
    sl = session._sl  # centralized session owns the optional SDK import
    if sl is None:
        raise RuntimeError("ZED SDK is not loaded")
    selected = config or TrackingConfig()
    params = sl.PositionalTrackingParameters()
    params.mode = getattr(sl.POSITIONAL_TRACKING_MODE, selected.mode.upper())
    params.enable_area_memory = selected.enable_area_memory
    params.enable_imu_fusion = selected.enable_imu_fusion
    params.enable_pose_smoothing = selected.enable_pose_smoothing
    params.set_gravity_as_origin = selected.set_gravity_as_origin
    params.set_floor_as_origin = selected.set_floor_as_origin
    params.set_as_static = selected.set_as_static
    status = session.camera.enable_positional_tracking(params)
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"enabling positional tracking failed: {status}")
    return params
