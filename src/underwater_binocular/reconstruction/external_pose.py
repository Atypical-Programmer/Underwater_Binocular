"""External inertial pose loading for constrained stereo reconstruction.

The calibration run HDF5 file stores one navigation/INS trajectory rather
than separate left and right camera poses.  This module converts its standard
NED/FRD attitude and geodetic position fields to a local ENU world frame and
interpolates them at the actual SVO timestamps.

The convention used here is explicit:

* navigation world: ENU (east, north, up);
* navigation body: FRD (forward, right, down);
* camera: OpenCV optical axes (right, down, forward);
* ``heading``: degrees clockwise from north;
* ``roll``/``pitch``: aerospace 3-2-1 angles in the NED frame.

The HDF5 source does not contain a sensor-to-camera lever arm, so callers may
provide one in the body frame.  With the default zero lever arm, the INS
reference point is treated as the left-camera center; that assumption is
recorded in the returned metadata.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_REQUIRED_FIELDS = (
    "utc",
    "heading",
    "roll",
    "pitch",
    "lat",
    "lon",
    "altitude",
)

# [forward, right, down] = R_body_from_camera [right, down, forward].
_BODY_FROM_CAMERA = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# [east, north, up] = R_enu_from_ned [north, east, down].
_ENU_FROM_NED = np.asarray(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


def _utc_text(seconds: float) -> str:
    return datetime.fromtimestamp(float(seconds), timezone.utc).isoformat()


def _rotation_enu_from_body(
    heading_deg: float,
    roll_deg: float,
    pitch_deg: float,
) -> np.ndarray:
    """Return the body-to-world rotation for the documented NED convention."""

    yaw, pitch, roll = np.deg2rad([heading_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    # Aerospace yaw-pitch-roll body-to-NED rotation.  Columns are the body
    # forward/right/down axes expressed in north/east/down coordinates.
    ned_from_body = np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    return _ENU_FROM_NED @ ned_from_body


def _local_enu(
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    altitude_m: np.ndarray,
    *,
    origin_latitude_deg: float,
    origin_longitude_deg: float,
) -> np.ndarray:
    """Convert a small geodetic trajectory to local ENU metres."""

    earth_radius_m = 6_378_137.0
    east = np.deg2rad(longitude_deg - origin_longitude_deg) * earth_radius_m * np.cos(
        np.deg2rad(origin_latitude_deg)
    )
    north = np.deg2rad(latitude_deg - origin_latitude_deg) * earth_radius_m
    up = altitude_m - altitude_m[0]
    return np.column_stack((east, north, up)).astype(np.float64, copy=False)


@dataclass(frozen=True)
class H5InertialPoseSequence:
    """A validated and locally expressed HDF5 inertial trajectory."""

    timestamps_s: np.ndarray
    heading_unwrapped_deg: np.ndarray
    roll_deg: np.ndarray
    pitch_deg: np.ndarray
    positions_enu_m: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        count = len(self.timestamps_s)
        arrays = (
            self.heading_unwrapped_deg,
            self.roll_deg,
            self.pitch_deg,
            self.positions_enu_m,
        )
        if count < 2 or any(len(value) != count for value in arrays):
            raise ValueError("HDF5 inertial pose sequence must contain at least two aligned samples")
        if self.positions_enu_m.shape != (count, 3):
            raise ValueError("HDF5 inertial positions must have shape (N, 3)")
        if not np.isfinite(self.timestamps_s).all() or not np.isfinite(self.positions_enu_m).all():
            raise ValueError("HDF5 inertial timestamps and positions must be finite")
        if np.any(np.diff(self.timestamps_s) <= 0.0):
            raise ValueError("HDF5 inertial timestamps must be strictly increasing")

    @property
    def start_time_s(self) -> float:
        return float(self.timestamps_s[0])

    @property
    def end_time_s(self) -> float:
        return float(self.timestamps_s[-1])

    def interpolate_camera_poses(
        self,
        frame_timestamps_ns: Mapping[int, int],
        *,
        time_offset_s: float = 0.0,
        lever_arm_body_m: Sequence[float] | np.ndarray | None = None,
    ) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
        """Interpolate world-from-left-camera poses at SVO timestamps.

        ``time_offset_s`` is added to each camera timestamp before looking it
        up in the HDF5 stream.  This makes a positive value mean that the H5
        query is shifted later than the SVO timestamp.  The returned world
        origin is moved to the first selected left-camera center while keeping
        the ENU orientation and metre scale.
        """

        if not frame_timestamps_ns:
            raise ValueError("at least one camera frame timestamp is required for HDF5 pose alignment")
        offset = float(time_offset_s)
        if not np.isfinite(offset):
            raise ValueError("HDF5 pose time offset must be finite")
        if lever_arm_body_m is None:
            lever_arm = np.zeros(3, dtype=np.float64)
        else:
            lever_arm = np.asarray(lever_arm_body_m, dtype=np.float64).reshape(-1)
            if lever_arm.shape != (3,) or not np.isfinite(lever_arm).all():
                raise ValueError("HDF5 pose lever arm must be a finite 3-vector in metres")

        frames = sorted(int(frame) for frame in frame_timestamps_ns)
        camera_times = np.asarray(
            [int(frame_timestamps_ns[frame]) / 1.0e9 + offset for frame in frames],
            dtype=np.float64,
        )
        tolerance_s = 1.0e-6
        if camera_times[0] < self.start_time_s - tolerance_s or camera_times[-1] > self.end_time_s + tolerance_s:
            raise ValueError(
                "camera timestamps fall outside HDF5 inertial coverage: "
                f"camera={_utc_text(camera_times[0])}..{_utc_text(camera_times[-1])}, "
                f"HDF5={_utc_text(self.start_time_s)}..{_utc_text(self.end_time_s)}"
            )
        query_times = np.clip(camera_times, self.start_time_s, self.end_time_s)
        heading = np.interp(query_times, self.timestamps_s, self.heading_unwrapped_deg)
        roll = np.interp(query_times, self.timestamps_s, self.roll_deg)
        pitch = np.interp(query_times, self.timestamps_s, self.pitch_deg)
        positions = np.column_stack(
            [np.interp(query_times, self.timestamps_s, self.positions_enu_m[:, axis]) for axis in range(3)]
        )
        poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        centers: list[np.ndarray] = []
        rotations: list[np.ndarray] = []
        for index in range(len(frames)):
            enu_from_body = _rotation_enu_from_body(heading[index], roll[index], pitch[index])
            enu_from_camera = enu_from_body @ _BODY_FROM_CAMERA
            center = positions[index] + enu_from_body @ lever_arm
            rotations.append(enu_from_camera)
            centers.append(center)
        origin = centers[0]
        for index, frame in enumerate(frames):
            rotation = rotations[index]
            center = centers[index]
            poses[frame] = (rotation, center - origin)
        selected_positions = np.asarray([pose[1] for pose in poses.values()], dtype=np.float64)
        left_indices = np.searchsorted(self.timestamps_s, query_times, side="right") - 1
        right_indices = left_indices + 1
        left_indices = np.clip(left_indices, 0, len(self.timestamps_s) - 1)
        right_indices = np.clip(right_indices, 0, len(self.timestamps_s) - 1)
        interpolation_gaps = np.minimum(
            np.abs(query_times - self.timestamps_s[left_indices]),
            np.abs(query_times - self.timestamps_s[right_indices]),
        )
        metadata = {
            **self.metadata,
            "camera_frames": len(frames),
            "camera_time_start_utc": _utc_text(float(camera_times[0])),
            "camera_time_end_utc": _utc_text(float(camera_times[-1])),
            "time_offset_s": offset,
            "lever_arm_body_m": lever_arm.tolist(),
            "lever_arm_assumption": (
                "zero; HDF5 INS reference point treated as left-camera center"
                if np.allclose(lever_arm, 0.0)
                else "provided in body forward/right/down axes"
            ),
            "world_frame": "local ENU; origin at first selected left-camera center",
            "camera_frame": "OpenCV optical: x=right, y=down, z=forward",
            "attitude_convention": "heading clockwise from north; aerospace NED/FRD yaw-pitch-roll",
            "interpolation_max_gap_s": float(np.max(interpolation_gaps))
            if len(query_times)
            else 0.0,
            "selected_position_extent_m": [
                float(np.ptp(selected_positions[:, axis])) for axis in range(3)
            ],
        }
        return poses, metadata


def load_h5_inertial_pose_sequence(path: Path) -> H5InertialPoseSequence:
    """Read and validate the ``inertial`` compound dataset from an HDF5 run."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HDF5 pose file not found: {path}")
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - exercised in missing-extra environments
        raise RuntimeError(
            "HDF5 pose support requires h5py; install the SfM extra with "
            "python -m pip install -e '.[sfm]'"
        ) from error
    with h5py.File(path, "r") as handle:
        if "inertial" not in handle:
            raise ValueError(f"HDF5 pose file has no inertial dataset: {path}")
        dataset = handle["inertial"]
        fields = set(dataset.dtype.names or ())
        missing = [field for field in _REQUIRED_FIELDS if field not in fields]
        if missing:
            raise ValueError(f"HDF5 inertial dataset is missing fields: {', '.join(missing)}")
        values = dataset[()]
        root_attrs = {str(key): value for key, value in handle.attrs.items()}

    if len(values) < 2:
        raise ValueError("HDF5 inertial dataset must contain at least two rows")
    numeric = {
        field: np.asarray(values[field], dtype=np.float64)
        for field in _REQUIRED_FIELDS
    }
    if any(not np.isfinite(value).all() for value in numeric.values()):
        raise ValueError("HDF5 inertial pose fields must be finite")
    order = np.argsort(numeric["utc"], kind="stable")
    for field in numeric:
        numeric[field] = numeric[field][order]
    if np.any(np.diff(numeric["utc"]) <= 0.0):
        raise ValueError("HDF5 inertial UTC timestamps must be strictly increasing")

    origin_lat = float(root_attrs.get("enu_origin_lat", numeric["lat"][0]))
    origin_lon = float(root_attrs.get("enu_origin_lon", numeric["lon"][0]))
    positions = _local_enu(
        numeric["lat"],
        numeric["lon"],
        numeric["altitude"],
        origin_latitude_deg=origin_lat,
        origin_longitude_deg=origin_lon,
    )
    heading_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(numeric["heading"])))
    dt = np.diff(numeric["utc"])
    actual_hz = 1.0 / float(np.median(dt))
    quality_counts: dict[str, int] = {}
    with h5py.File(path, "r") as handle:
        dataset = handle["inertial"]
        for field in ("nav_status", "fault", "quality_ok"):
            if field not in (dataset.dtype.names or ()):
                continue
            raw = dataset[field][()][order]
            for value in np.unique(raw):
                if isinstance(value, bytes):
                    label = value.decode("utf-8", errors="replace")
                elif isinstance(value, np.bytes_):
                    label = bytes(value).decode("utf-8", errors="replace")
                else:
                    label = str(value)
                quality_counts[f"{field}={label}"] = int(np.sum(raw == value))
    metadata = {
        "source_path": str(path),
        "dataset": "inertial",
        "rows": int(len(values)),
        "hdf5_nominal_rate_hz": (
            float(root_attrs["inertial_rate_hz"])
            if "inertial_rate_hz" in root_attrs
            else None
        ),
        "actual_rate_hz_from_utc": actual_hz,
        "timestamp_start_utc": _utc_text(numeric["utc"][0]),
        "timestamp_end_utc": _utc_text(numeric["utc"][-1]),
        "timestamp_interval_s": float(np.median(dt)),
        "enu_origin_lat": origin_lat,
        "enu_origin_lon": origin_lon,
        "quality_counts": quality_counts,
    }
    return H5InertialPoseSequence(
        timestamps_s=numeric["utc"],
        heading_unwrapped_deg=heading_unwrapped,
        roll_deg=numeric["roll"],
        pitch_deg=numeric["pitch"],
        positions_enu_m=positions,
        metadata=metadata,
    )
