"""Production ZED SDK depth engine and metadata-aware exporter."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .. import __version__
from ..calibration.loaders import load_calibration_profile, sha256_file
from ..config.loaders import load_dataset_config, load_zed_config
from ..config.models import DatasetConfig
from ..io.zed import ZedSession
from .models import DepthFrame
from .statistics import center_measurement, summarize_values
from .visualization import DepthVideoWriter


def _git_commit() -> str | None:
    try:
        value = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return value.stdout.strip() or None


class ZedSdkDepthEngine:
    """Yield ZED ``MEASURE.DEPTH`` frames in metres from one verified session."""

    def __init__(self, session: ZedSession) -> None:
        self.session = session

    def iter_frames(self) -> Iterator[DepthFrame]:
        """Iterate the SVO sequentially; no second hidden SVO handle is opened."""

        yield from self.session.iter_depth_frames()

    def export(
        self,
        output_dir: Path,
        *,
        fps: float,
        display_min_depth_m: float = 1.5,
        display_max_depth_m: float = 3.5,
        max_valid_depth_m: float = 100.0,
        include_video: bool = True,
    ) -> dict[str, Any]:
        """Export a video and compact center statistics from computed frames."""

        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / "depth.mp4"
        csv_path = output_dir / "center_depth.csv"
        video_writer = (
            DepthVideoWriter(
                video_path,
                fps=fps,
                min_depth_m=display_min_depth_m,
                max_depth_m=display_max_depth_m,
            )
            if include_video
            else None
        )
        csv_handle = None
        csv_writer = None
        frame_count = 0
        video_count = 0
        center_values: list[float] = []
        window_values: list[float] = []
        try:
            for frame in self.iter_frames():
                if max_valid_depth_m > 0.0:
                    depth = frame.depth_m.copy()
                    invalid = depth > max_valid_depth_m
                    depth[invalid] = float("nan")
                    frame = DepthFrame(
                        depth,
                        frame.valid_mask & ~invalid,
                        frame.frame_index,
                        frame.timestamp_ns,
                        frame.metadata,
                    )
                if video_writer is not None:
                    video_writer.write(frame)
                row = center_measurement(frame)
                if csv_writer is None:
                    csv_handle = csv_path.open("w", newline="", encoding="utf-8")
                    csv_writer = csv.DictWriter(csv_handle, fieldnames=list(row))
                    csv_writer.writeheader()
                csv_writer.writerow(row)
                frame_count += 1
                center_values.append(row["center_depth_m"])
                window_values.append(row["center_window_median_depth_m"])
        finally:
            if csv_handle is not None:
                csv_handle.close()
            if video_writer is not None:
                video_writer.close()
                video_count = video_writer.count
        summary = {
            "schema_version": 1,
            "engine": "zed-neural",
            "depth_source": "pyzed.sl.MEASURE.DEPTH",
            "depth_mode": self.session.config.depth_mode,
            "frames": frame_count,
            "video_frames": video_count,
            "center_pixel": summarize_values(center_values),
            "center_window_median": summarize_values(window_values),
            "runtime_calibration": self.session.runtime_metadata,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        return summary


def _default_zed_config(dataset_path: Path) -> Path:
    return dataset_path.parents[1] / "depth" / "zed_neural.yaml"


def _run_metadata(dataset: DatasetConfig, profile_path: Path, command: list[str], output_dir: Path, session: ZedSession) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp": __import__("datetime").datetime.now().astimezone().isoformat(),
        "git_commit": _git_commit(),
        "dataset": dataset.dataset_id,
        "calibration_profile": profile_path.as_posix(),
        "calibration_hash": sha256_file(profile_path),
        "command": command,
        "resolved_config": session.config.__dict__,
        "software_versions": {"underwater_binocular": __version__, "python": platform.python_version()},
        "platform": platform.platform(),
        "outputs": [],
        "status": "running",
        "sdk_runtime": session.runtime_metadata,
    }


def run_depth_export(args: Any) -> int:
    """CLI adapter for production ZED depth export."""

    dataset_config = load_dataset_config(args.dataset)
    root = args.dataset.resolve().parents[2]
    zed_config_path = args.config or (root / "configs" / "depth" / "zed_neural.yaml")
    zed_config = load_zed_config(zed_config_path)
    profile_path = dataset_config.calibration_profile
    profile = load_calibration_profile(profile_path)
    svo_path = dataset_config.resolve_svo_path(args.svo)
    output_dir = (args.output or (root / "outputs" / f"{dataset_config.dataset_id}_zed_neural")).resolve()
    with ZedSession(svo_path, profile_path.parent.parent / "generated" / "zed_custom_opencv.yml", zed_config, expected_calibration=profile) as session:
        metadata = _run_metadata(dataset_config, profile_path, ["underwater", "depth", "export"], output_dir, session)
        (output_dir / "run.json").parent.mkdir(parents=True, exist_ok=True)
        (output_dir / "run.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
        summary = ZedSdkDepthEngine(session).export(
            output_dir,
            fps=dataset_config.fps,
            display_min_depth_m=zed_config.display_min_depth_m,
            display_max_depth_m=zed_config.display_max_depth_m,
            max_valid_depth_m=zed_config.max_valid_depth_m,
            include_video=args.format == "mp4",
        )
        metadata["outputs"] = [path.name for path in output_dir.iterdir() if path.is_file()]
        metadata["summary"] = summary
        metadata["status"] = "complete"
        (output_dir / "run.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Depth export complete: {output_dir}")
    return 0
