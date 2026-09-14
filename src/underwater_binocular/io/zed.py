"""Single ZED SDK session implementation.

The proprietary ``pyzed.sl`` module is imported only when a session is opened,
so calibration, geometry, and unit tests remain runnable without the SDK.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..calibration.models import StereoCalibration
from ..calibration.zed import compare_runtime_calibration, runtime_calibration_metadata
from ..config.models import CalibrationMode, ZedSessionConfig


class ZedDependencyError(RuntimeError):
    """Raised when the vendor ZED Python binding is unavailable."""


def _prepare_windows_dll_search_path() -> None:
    """Add paths from ``ZED_SDK_ROOT_DIR`` without assuming a developer path."""

    if os.name != "nt":
        return
    sdk_root = os.environ.get("ZED_SDK_ROOT_DIR")
    if not sdk_root:
        return
    root = Path(sdk_root)
    candidates = [
        root / "bin",
        root / "dependencies" / "freeglut" / "bin",
        root / "dependencies" / "freeglut_2.8" / "x64",
        root / "dependencies" / "glew" / "bin",
        root / "dependencies" / "glew-1.12.0" / "x64",
        root / "dependencies" / "opencv" / "x64" / "vc16" / "bin",
        root / "dependencies" / "opencv_3.1.0" / "x64",
    ]
    existing = [str(path) for path in candidates if path.is_dir()]
    if existing:
        os.environ["PATH"] = os.pathsep.join([*existing, os.environ.get("PATH", "")])
        if hasattr(os, "add_dll_directory"):
            _prepare_windows_dll_search_path._dll_handles = [  # type: ignore[attr-defined]
                os.add_dll_directory(path) for path in existing
            ]


def import_zed() -> Any:
    """Import ``pyzed.sl`` with an actionable installation error."""

    _prepare_windows_dll_search_path()
    try:
        import pyzed.sl as sl
    except ImportError as error:  # pragma: no cover - depends on vendor install
        raise ZedDependencyError(
            "ZED SDK Python bindings are unavailable; install the vendor-supported ZED SDK "
            "and expose it in the active environment"
        ) from error
    return sl


def _status_name(status: Any) -> str:
    return str(status).rsplit(".", 1)[-1]


def _check_status(status: Any, operation: str, sl: Any) -> None:
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"{operation} failed: {_status_name(status)}")


def _enum_member(namespace: Any, name: str) -> Any:
    try:
        return getattr(namespace, name)
    except AttributeError as error:
        raise RuntimeError(f"ZED SDK does not expose {namespace.__name__}.{name}") from error


@dataclass(frozen=True)
class ZedImagePair:
    """One synchronized pair of raw or rectified ZED images."""

    frame_index: int
    timestamp_ns: int
    left: np.ndarray
    right: np.ndarray
    semantic: str


class ZedSession:
    """Own one SVO handle and all SDK initialization/runtime verification."""

    def __init__(
        self,
        svo_path: Path,
        calibration_path: Path | None = None,
        config: ZedSessionConfig | None = None,
        *,
        calibration_mode: CalibrationMode | str = CalibrationMode.CUSTOM,
        expected_calibration: StereoCalibration | None = None,
    ) -> None:
        self.svo_path = Path(svo_path).expanduser().resolve()
        self.calibration_mode = CalibrationMode.parse(calibration_mode)
        self.calibration_path = (
            None if calibration_path is None else Path(calibration_path).expanduser().resolve()
        )
        self.config = config or ZedSessionConfig()
        self.config.validate()
        if self.calibration_mode is CalibrationMode.NATIVE and expected_calibration is not None:
            raise ValueError("native ZED sessions cannot compare against a custom calibration profile")
        if self.calibration_mode is CalibrationMode.CUSTOM and self.calibration_path is None:
            raise ValueError("custom ZED sessions require a calibration path")
        self.expected_calibration = expected_calibration
        self._sl: Any | None = None
        self._camera: Any | None = None
        self._runtime: Any | None = None
        self._frame_index = -1
        self._tracking_enabled = False
        self._runtime_metadata: dict[str, Any] | None = None

    @property
    def is_open(self) -> bool:
        return self._camera is not None

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        if self._runtime_metadata is None:
            raise RuntimeError("ZED session is not open")
        return self._runtime_metadata

    @property
    def camera(self) -> Any:
        if self._camera is None:
            raise RuntimeError("ZED session is not open")
        return self._camera

    @property
    def sdk(self) -> Any:
        """Return the loaded SDK namespace for an already-open session."""

        if self._sl is None:
            raise RuntimeError("ZED SDK is not loaded")
        return self._sl

    @property
    def frame_index(self) -> int:
        """Return the last successfully grabbed SVO frame position."""

        return self._frame_index

    def camera_information(self) -> Any:
        """Return camera information without opening another SDK handle."""

        return self.camera.get_camera_information()

    def svo_number_of_frames(self) -> int:
        """Return the number of frames reported by the currently open SVO."""

        value = int(self.camera.get_svo_number_of_frames())
        if value <= 0:
            raise RuntimeError(f"ZED reported an invalid SVO frame count: {value}")
        return value

    def svo_position(self) -> int:
        """Return the SDK's current SVO position."""

        if hasattr(self.camera, "get_svo_position"):
            return int(self.camera.get_svo_position())
        return self._frame_index

    def _make_init_parameters(self) -> Any:
        sl = self._sl or import_zed()
        init = sl.InitParameters()
        init.set_from_svo_file(str(self.svo_path))
        init.svo_real_time_mode = bool(self.config.svo_real_time_mode)
        init.depth_mode = _enum_member(sl.DEPTH_MODE, self.config.depth_mode.upper())
        init.coordinate_units = _enum_member(sl.UNIT, self.config.coordinate_units.upper())
        init.coordinate_system = _enum_member(sl.COORDINATE_SYSTEM, self.config.coordinate_system.upper())
        init.camera_disable_self_calib = bool(self.config.camera_disable_self_calib)
        if self.calibration_mode is CalibrationMode.CUSTOM:
            init.optional_opencv_calibration_file = str(self.calibration_path)
        init.depth_stabilization = int(self.config.depth_stabilization)
        if hasattr(init, "enable_image_enhancement"):
            init.enable_image_enhancement = bool(self.config.enable_image_enhancement)
        return init

    def open(self) -> ZedSession:
        """Open the SVO and verify the selected native or custom calibration policy."""

        if self.is_open:
            return self
        if not self.svo_path.is_file():
            raise FileNotFoundError(f"SVO not found: {self.svo_path}")
        if self.calibration_mode is CalibrationMode.CUSTOM and not self.calibration_path.is_file():
            raise FileNotFoundError(f"custom calibration not found: {self.calibration_path}")
        self._sl = import_zed()
        camera = self._sl.Camera()
        status = camera.open(self._make_init_parameters())
        _check_status(status, "opening SVO", self._sl)
        self._camera = camera
        try:
            self._runtime_metadata = runtime_calibration_metadata(camera.get_camera_information())
            self._runtime_metadata["calibration_mode"] = self.calibration_mode.value
            if self.calibration_mode is CalibrationMode.NATIVE:
                self._runtime_metadata["calibration_source"] = "svo_embedded"
                self._runtime_metadata["verification"] = {
                    "status": "NOT_APPLICABLE",
                    "reason": "native mode uses the calibration embedded in the SVO",
                }
            else:
                self._runtime_metadata["calibration_source"] = str(self.calibration_path)
                if self.expected_calibration is not None:
                    verification = compare_runtime_calibration(
                        self._runtime_metadata,
                        self.expected_calibration,
                    )
                    self._runtime_metadata["verification"] = verification
                else:
                    self._runtime_metadata["verification"] = {
                        "status": "NOT_REQUESTED",
                        "reason": "custom profile was supplied without an expected calibration object",
                    }
            runtime = self._make_runtime_parameters()
            self._runtime = runtime
        except Exception:
            self.close()
            raise
        return self

    def _make_runtime_parameters(self) -> Any:
        sl = self._sl
        if sl is None:
            raise RuntimeError("ZED SDK is not loaded")
        runtime = sl.RuntimeParameters()
        if hasattr(runtime, "confidence_threshold"):
            runtime.confidence_threshold = self.config.confidence_threshold
        if hasattr(runtime, "texture_confidence_threshold"):
            runtime.texture_confidence_threshold = self.config.texture_confidence_threshold
        if hasattr(runtime, "measure3D_reference_frame"):
            runtime.measure3D_reference_frame = _enum_member(
                sl.REFERENCE_FRAME, self.config.reference_frame.upper()
            )
        return runtime

    def close(self) -> None:
        """Close the SDK handle; safe to call after a failed open."""

        if self._camera is not None:
            try:
                if self._tracking_enabled and hasattr(self._camera, "disable_positional_tracking"):
                    self._camera.disable_positional_tracking()
            finally:
                self._camera.close()
        self._camera = None
        self._runtime = None
        self._frame_index = -1
        self._tracking_enabled = False

    def __enter__(self) -> ZedSession:
        return self.open()

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()

    def grab(self) -> bool:
        """Grab the next sequential frame and return whether it was successful."""

        status = self.camera.grab(self._runtime)
        if status == self._sl.ERROR_CODE.SUCCESS:
            self._frame_index += 1
            return True
        if status == self._sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            return False
        _check_status(status, "grabbing SVO frame", self._sl)
        return False

    def seek(self, frame_index: int) -> None:
        """Seek to an SVO position; the next successful grab is that frame."""

        if frame_index < 0:
            raise ValueError("frame_index cannot be negative")
        if self._tracking_enabled:
            raise RuntimeError("SVO seeking is only allowed before positional tracking starts")
        status = self.camera.set_svo_position(int(frame_index))
        _check_status(status, "seeking SVO", self._sl)
        self._frame_index = frame_index - 1

    def timestamp_ns(self) -> int:
        """Return the current frame timestamp in nanoseconds."""

        return int(self.camera.get_timestamp(self._sl.TIME_REFERENCE.IMAGE).get_nanoseconds())

    def retrieve_image(self, view_name: str = "LEFT") -> np.ndarray:
        """Retrieve a BGR image from the requested SDK view."""

        mat = self._sl.Mat()
        view = _enum_member(self._sl.VIEW, view_name.upper())
        status = self.camera.retrieve_image(mat, view)
        _check_status(status, f"retrieving image {view_name}", self._sl)
        value = mat.get_data()
        if value is None:
            raise RuntimeError(f"ZED returned no image data for {view_name}")
        return np.asarray(value).copy()

    def retrieve_depth(self) -> np.ndarray:
        """Retrieve SDK ``MEASURE.DEPTH`` in metres as a float32 array."""

        mat = self._sl.Mat()
        status = self.camera.retrieve_measure(mat, self._sl.MEASURE.DEPTH)
        _check_status(status, "retrieving MEASURE.DEPTH", self._sl)
        value = mat.get_data()
        if value is None:
            raise RuntimeError("ZED returned no depth data")
        depth = np.asarray(value, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise RuntimeError(f"unexpected ZED depth shape: {depth.shape}")
        return depth.copy()

    def iter_image_pairs(self, *, view_name: str = "LEFT_AND_RIGHT") -> Iterator[ZedImagePair]:
        """Yield synchronized image pairs with explicit raw/rectified semantics."""

        normalized_view = view_name.upper()
        if normalized_view not in {"LEFT_AND_RIGHT", "RAW_UNRECTIFIED", "RECTIFIED"}:
            raise ValueError("view_name must be LEFT_AND_RIGHT, RAW_UNRECTIFIED, or RECTIFIED")
        if normalized_view == "RECTIFIED":
            left_view, right_view = "LEFT", "RIGHT"
            semantic = "RECTIFIED"
        else:
            left_view, right_view = "LEFT_UNRECTIFIED", "RIGHT_UNRECTIFIED"
            semantic = "RAW_UNRECTIFIED"
        while self.grab():
            left = self.retrieve_image(left_view)
            right = self.retrieve_image(right_view)
            yield ZedImagePair(self._frame_index, self.timestamp_ns(), left, right, semantic)

    def iter_depth_frames(self) -> Iterator[Any]:
        """Yield :class:`DepthFrame` objects from the SDK depth measure."""

        from ..depth.models import DepthFrame

        while self.grab():
            depth = self.retrieve_depth()
            valid = np.isfinite(depth) & (depth > 0.0)
            yield DepthFrame(
                depth_m=depth,
                valid_mask=valid,
                frame_index=self._frame_index,
                timestamp_ns=self.timestamp_ns(),
                metadata={"engine": "zed-neural", "semantic": "SDK_MEASURE_DEPTH"},
            )
