"""Typed configuration models used by production and integration entry points."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class CalibrationMode(str, Enum):
    """How a ZED session obtains stereo calibration."""

    NATIVE = "native"
    CUSTOM = "custom"

    @classmethod
    def parse(cls, value: CalibrationMode | str) -> CalibrationMode:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as error:
            raise ValueError("calibration mode must be native or custom") from error


def _resolution(value: Any) -> tuple[int, int]:
    if isinstance(value, Mapping):
        width, height = value.get("width"), value.get("height")
    else:
        width, height = value
    result = (int(width), int(height))
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError(f"resolution must be positive, got {result}")
    return result


def _as_bool(value: Any, *, default: bool) -> bool:
    """Parse YAML booleans without treating the string ``"false"`` as true."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    raise ValueError(f"expected a boolean value, got {value!r}")


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset identity and external paths.

    ``svo_path`` may be ``None`` in a versioned config. The caller must then
    provide ``svo_path_env`` or an explicit CLI override; no developer path is
    silently substituted.
    """

    dataset_id: str
    svo_path: Path | None
    calibration_profile: Path
    camera_serial: int | None
    resolution: tuple[int, int]
    fps: float
    expected_frames: int | None
    svo_path_env: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], *, base_dir: Path) -> DatasetConfig:
        raw_svo = values.get("svo_path")
        svo_path = None if raw_svo in (None, "") else Path(str(raw_svo))
        raw_profile = values.get("calibration_profile")
        if raw_profile in (None, ""):
            raise ValueError("dataset config requires calibration_profile")
        profile = Path(str(raw_profile))
        return cls(
            dataset_id=str(values.get("id", "")),
            svo_path=None if svo_path is None else (base_dir / svo_path).resolve() if not svo_path.is_absolute() else svo_path,
            calibration_profile=(base_dir / profile).resolve() if not profile.is_absolute() else profile,
            camera_serial=None if values.get("camera_serial") is None else int(values["camera_serial"]),
            resolution=_resolution(values.get("resolution")),
            fps=float(values.get("fps", 0.0)),
            expected_frames=None if values.get("expected_frames") is None else int(values["expected_frames"]),
            svo_path_env=None if values.get("svo_path_env") in (None, "") else str(values["svo_path_env"]),
            metadata={
                key: value
                for key, value in values.items()
                if key
                not in {
                    "id",
                    "svo_path",
                    "calibration_profile",
                    "camera_serial",
                    "resolution",
                    "fps",
                    "expected_frames",
                    "svo_path_env",
                }
            },
        )

    def resolve_svo_path(self, override: Path | None = None) -> Path:
        """Resolve the SVO from an explicit override, config, or named env var."""

        if override is not None:
            return override.expanduser().resolve()
        if self.svo_path is not None:
            return self.svo_path.expanduser().resolve()
        if self.svo_path_env:
            import os

            value = os.environ.get(self.svo_path_env)
            if value:
                return Path(value).expanduser().resolve()
        raise FileNotFoundError(
            f"dataset {self.dataset_id!r} has no SVO path; set {self.svo_path_env or 'an override'}"
        )


@dataclass(frozen=True)
class ZedSessionConfig:
    """ZED initialization settings with explicit calibration override policy."""

    depth_mode: str = "NEURAL"
    coordinate_units: str = "METER"
    coordinate_system: str = "RIGHT_HANDED_Y_UP"
    reference_frame: str = "CAMERA"
    camera_disable_self_calib: bool = True
    depth_stabilization: int = 0
    enable_image_enhancement: bool = False
    confidence_threshold: int = 30
    texture_confidence_threshold: int = 100
    svo_real_time_mode: bool = False
    display_min_depth_m: float = 1.5
    display_max_depth_m: float = 3.5
    max_valid_depth_m: float = 100.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ZedSessionConfig:
        result = cls(
            depth_mode=str(values.get("depth_mode", "NEURAL")),
            coordinate_units=str(values.get("coordinate_units", "METER")),
            coordinate_system=str(values.get("coordinate_system", "RIGHT_HANDED_Y_UP")),
            reference_frame=str(values.get("reference_frame", "CAMERA")),
            camera_disable_self_calib=_as_bool(values.get("camera_disable_self_calib"), default=True),
            depth_stabilization=int(values.get("depth_stabilization", 0)),
            enable_image_enhancement=_as_bool(values.get("enable_image_enhancement"), default=False),
            confidence_threshold=int(values.get("confidence_threshold", 30)),
            texture_confidence_threshold=int(values.get("texture_confidence_threshold", 100)),
            svo_real_time_mode=_as_bool(values.get("svo_real_time_mode"), default=False),
            display_min_depth_m=float(values.get("display_min_depth_m", 1.5)),
            display_max_depth_m=float(values.get("display_max_depth_m", 3.5)),
            max_valid_depth_m=float(values.get("max_valid_depth_m", 100.0)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.coordinate_units.upper() != "METER":
            raise ValueError("package ZED sessions must use METER coordinates")
        if not self.camera_disable_self_calib:
            raise ValueError("production ZED sessions must disable SDK self-calibration")
        if self.depth_stabilization < 0:
            raise ValueError("depth_stabilization cannot be negative")
        for name, value in (
            ("confidence_threshold", self.confidence_threshold),
            ("texture_confidence_threshold", self.texture_confidence_threshold),
        ):
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be in [0, 100]")
        if not 0.0 < self.display_min_depth_m < self.display_max_depth_m:
            raise ValueError("display depth range must be positive and ordered")
        if self.max_valid_depth_m <= 0.0:
            raise ValueError("max_valid_depth_m must be positive")


@dataclass(frozen=True)
class SgbmConfig:
    """OpenCV StereoSGBM parameters; disparity is converted from fixed-point /16."""

    rectify_alpha: float = 0.0
    scale: float = 1.0
    num_disparities: int = 256
    block_size: int = 5
    uniqueness_ratio: int = 8
    speckle_window_size: int = 100
    speckle_range: int = 2
    min_disparity_px: float = 1.0
    max_depth_m: float = 100.0
    center_window_radius: int = 2

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> SgbmConfig:
        result = cls(
            **{
                field_name: values[field_name]
                for field_name in cls.__dataclass_fields__
                if field_name in values
            }
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not 0.0 <= float(self.rectify_alpha) <= 1.0:
            raise ValueError("rectify_alpha must be within [0, 1]")
        if not 0.0 < float(self.scale) <= 1.0:
            raise ValueError("scale must be within (0, 1]")
        if self.num_disparities <= 0 or self.num_disparities % 16:
            raise ValueError("num_disparities must be a positive multiple of 16")
        if self.block_size < 3 or self.block_size % 2 == 0:
            raise ValueError("block_size must be odd and at least 3")
        if self.max_depth_m <= 0.0 or self.min_disparity_px <= 0.0:
            raise ValueError("depth and disparity limits must be positive")
