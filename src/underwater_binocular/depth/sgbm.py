"""OpenCV SGBM depth engine for diagnostics and cross-checks."""

from __future__ import annotations

import json
import platform
import subprocess
from typing import Any

import cv2
import numpy as np

from .. import __version__
from ..calibration.loaders import load_calibration_profile, sha256_file
from ..config.loaders import load_dataset_config, load_sgbm_config
from ..config.models import SgbmConfig, ZedSessionConfig
from ..geometry.rectification import RectifiedStereoModel, rectify_calibration
from ..io.images import as_gray
from ..io.zed import ZedSession
from .models import DepthFrame
from .statistics import center_measurement, summarize_values


def depth_from_disparity(
    disparity_px: np.ndarray,
    *,
    focal_length_px: float,
    baseline_m: float,
    min_disparity_px: float = 1.0,
    max_depth_m: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert fixed-point-corrected disparity to optical-axis depth in metres."""

    disparity = np.asarray(disparity_px, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > min_disparity_px)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[valid] = focal_length_px * baseline_m / disparity[valid]
    valid &= np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth_m)
    depth[~valid] = np.nan
    return depth, valid


class SgbmDepthEngine:
    """Rectify a raw pair, match with OpenCV StereoSGBM, and return ``DepthFrame``."""

    def __init__(self, calibration: Any, config: SgbmConfig) -> None:
        config.validate()
        self.calibration = calibration
        self.config = config
        self.rectified: RectifiedStereoModel = rectify_calibration(
            calibration,
            alpha=config.rectify_alpha,
            scale=config.scale,
        )
        channels = 1
        p1 = 8 * channels * config.block_size * config.block_size
        p2 = 32 * channels * config.block_size * config.block_size
        mode = getattr(cv2, "STEREO_SGBM_MODE_SGBM_3WAY", cv2.STEREO_SGBM_MODE_SGBM)
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=config.num_disparities,
            blockSize=config.block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=1,
            preFilterCap=63,
            uniquenessRatio=config.uniqueness_ratio,
            speckleWindowSize=config.speckle_window_size,
            speckleRange=config.speckle_range,
            mode=mode,
        )

    def compute(self, left_raw: np.ndarray, right_raw: np.ndarray, *, frame_index: int = 0, timestamp_ns: int | None = None) -> DepthFrame:
        """Compute one depth frame from ``RAW_UNRECTIFIED`` images."""

        left_rectified, right_rectified = self.rectified.rectify_pair(left_raw, right_raw)
        disparity_fixed = self.matcher.compute(as_gray(left_rectified), as_gray(right_rectified))
        disparity_px = disparity_fixed.astype(np.float32) / 16.0
        depth, valid = depth_from_disparity(
            disparity_px,
            focal_length_px=self.rectified.focal_length_px,
            baseline_m=self.rectified.baseline_m,
            min_disparity_px=self.config.min_disparity_px,
            max_depth_m=self.config.max_depth_m,
        )
        return DepthFrame(
            depth,
            valid,
            frame_index,
            timestamp_ns,
            {
                "engine": "sgbm",
                "raw_semantic": "RAW_UNRECTIFIED",
                "rectified_semantic": "RECTIFIED",
                "disparity_fixed_point_divisor": 16.0,
                "focal_length_px": self.rectified.focal_length_px,
                "baseline_m": self.rectified.baseline_m,
            },
        )


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def run_sgbm_export(args: Any) -> int:
    """CLI adapter for a diagnostic SGBM replay; outputs are disposable."""

    dataset_config = load_dataset_config(args.dataset)
    root = args.dataset.resolve().parents[2]
    config_path = args.config or (root / "configs" / "depth" / "sgbm.yaml")
    config = load_sgbm_config(config_path)
    profile_path = dataset_config.calibration_profile
    calibration = load_calibration_profile(profile_path)
    svo_path = dataset_config.resolve_svo_path(args.svo)
    output_dir = (args.output or (root / "outputs" / f"{dataset_config.dataset_id}_sgbm")).resolve()
    zed_config = ZedSessionConfig(depth_mode="NONE")
    rows: list[dict[str, Any]] = []
    with ZedSession(svo_path, profile_path.parent.parent / "generated" / "zed_custom_opencv.yml", zed_config, expected_calibration=calibration) as session:
        engine = SgbmDepthEngine(calibration, config)
        for pair in session.iter_image_pairs(view_name="RAW_UNRECTIFIED"):
            frame = engine.compute(
                pair.left,
                pair.right,
                frame_index=pair.frame_index,
                timestamp_ns=pair.timestamp_ns,
            )
            rows.append(center_measurement(frame, config.center_window_radius))
            if len(rows) >= (dataset_config.expected_frames or 0) and dataset_config.expected_frames:
                break
    output_dir.mkdir(parents=True, exist_ok=True)
    center = [row["center_depth_m"] for row in rows]
    windows = [row["center_window_median_depth_m"] for row in rows]
    summary = {
        "schema_version": 1,
        "method": "OpenCV StereoSGBM; independent of ZED MEASURE.DEPTH",
        "dataset": dataset_config.dataset_id,
        "engine": "SgbmDepthEngine",
        "calibration_profile": profile_path.as_posix(),
        "calibration_hash": sha256_file(profile_path),
        "frames_replayed": len(rows),
        "center_pixel_stats_m": summarize_values(center),
        "center_window_median_stats_m": summarize_values(windows),
        "rectified_focal_length_px": engine.rectified.focal_length_px,
        "rectified_baseline_m": engine.rectified.baseline_m,
        "sgbm": config.__dict__,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    run = {
        "schema_version": 1,
        "timestamp": __import__("datetime").datetime.now().astimezone().isoformat(),
        "git_commit": _git_commit(),
        "dataset": dataset_config.dataset_id,
        "calibration_profile": profile_path.as_posix(),
        "calibration_hash": sha256_file(profile_path),
        "software_versions": {"underwater_binocular": __version__, "python": platform.python_version()},
        "outputs": ["summary.json"],
        "status": "complete",
    }
    (output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    print(f"SGBM diagnostic export complete: {output_dir}")
    return 0
