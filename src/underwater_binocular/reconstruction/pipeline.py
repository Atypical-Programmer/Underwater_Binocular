"""End-to-end ALIKED matcher + COLMAP reconstruction workflow."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..calibration.conversion import write_colmap_camera_config, write_zed_opencv_calibration
from ..calibration.loaders import load_calibration_profile, sha256_file
from ..calibration.models import StereoCalibration
from ..config.loaders import load_dataset_config
from ..config.models import DatasetConfig, ZedSessionConfig
from ..io.images import as_bgr, write_image
from ..io.zed import ZedSession
from .aliked import AlikedConfig, extract_aliked_features, load_torch_device
from .colmap_database import create_colmap_database, database_stats
from .colmap_runner import find_colmap_executable, run_colmap_pipeline
from .features import ImageRecord
from .matching import (
    DescriptorMatches,
    ImagePair,
    build_image_pairs,
    load_lightglue_matcher,
    match_adalam,
    match_lightglue,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git_sha() -> str | None:
    root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _svo_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def select_svo_positions(
    total_frames: int,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    num_frames: int = 50,
    frame_step: int = 1,
) -> list[int]:
    """Select deterministic positions, retaining both ends when sampling."""

    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if start_frame < 0 or frame_step <= 0 or num_frames < 0:
        raise ValueError("start_frame/frame_step must be positive and num_frames non-negative")
    if start_frame >= total_frames:
        raise ValueError(f"start_frame {start_frame} is outside {total_frames} frames")
    last = total_frames - 1 if end_frame is None else min(int(end_frame), total_frames - 1)
    if last < start_frame:
        raise ValueError("end_frame must be greater than or equal to start_frame")
    candidates = list(range(start_frame, last + 1, frame_step))
    if not candidates:
        raise ValueError("frame selection produced no positions")
    if num_frames == 0 or num_frames >= len(candidates):
        return candidates
    if num_frames == 1:
        return [candidates[0]]
    indices = np.linspace(0, len(candidates) - 1, num_frames, dtype=np.int64)
    return [candidates[int(index)] for index in indices]


def _record_from_manifest(output_dir: Path, value: dict[str, Any]) -> ImageRecord:
    try:
        relative_path = Path(str(value["path"]))
        image_name = str(value.get("name", ""))
        if not image_name:
            parts = relative_path.parts
            image_name = (
                Path(*parts[1:]).as_posix()
                if parts and parts[0].lower() == "images"
                else relative_path.as_posix()
            )
        record = ImageRecord(
            path=(output_dir / relative_path).resolve(),
            name=image_name.replace("\\", "/"),
            side=str(value["side"]),
            frame=int(value["frame"]),
            timestamp_ns=int(value.get("timestamp_ns", 0)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("invalid image_manifest.json record") from error
    if record.side not in {"left", "right"} or not record.path.is_file():
        raise FileNotFoundError(f"manifest image is unavailable: {record.path}")
    return record


def _load_existing_manifest(
    output_dir: Path,
    *,
    dataset_id: str,
    svo_path: Path,
    include_right: bool,
    start_frame: int,
    end_frame: int | None,
    num_frames: int,
    frame_step: int,
    resume: bool,
) -> tuple[list[ImageRecord], dict[str, Any]] | None:
    if not resume:
        return None
    path = output_dir / "image_manifest.json"
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read existing image manifest: {path}") from error
    if not isinstance(manifest, dict):
        raise RuntimeError(f"image manifest root must be an object: {path}")
    if manifest.get("dataset") != dataset_id:
        return None
    if manifest.get("svo_identity") != _svo_identity(svo_path):
        return None
    if bool(manifest.get("include_right")) != include_right:
        return None
    if manifest.get("selection") != {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "num_frames": num_frames,
        "frame_step": frame_step,
    }:
        return None
    values = manifest.get("records")
    if not isinstance(values, list) or not values:
        return None
    records = [_record_from_manifest(output_dir, value) for value in values]
    return records, manifest


def _manifest_records(
    output_dir: Path,
    records: list[ImageRecord],
    *,
    dataset: DatasetConfig,
    svo_path: Path,
    include_right: bool,
    positions: list[int],
    selection: dict[str, Any],
    runtime_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "dataset": dataset.dataset_id,
        "svo_path": str(svo_path),
        "svo_identity": _svo_identity(svo_path),
        "output_dir": str(output_dir),
        "include_right": bool(include_right),
        "image_view": "RAW_UNRECTIFIED",
        "image_coordinate_space": "original_image_pixels",
        "selection": selection,
        "selected_svo_positions": positions,
        "selected_frame_count": len(positions),
        "records": [
            {
                "path": record.path.relative_to(output_dir).as_posix(),
                "name": record.name,
                "side": record.side,
                "frame": record.frame,
                "timestamp_ns": record.timestamp_ns,
                "sha256": sha256_file(record.path),
            }
            for record in records
        ],
        "runtime_calibration": runtime_metadata,
    }


def extract_svo_images(
    *,
    dataset: DatasetConfig,
    svo_path: Path,
    calibration_path: Path,
    output_dir: Path,
    include_right: bool,
    start_frame: int,
    end_frame: int | None,
    num_frames: int,
    frame_step: int,
    zed_config: ZedSessionConfig,
    expected_calibration: Any,
) -> tuple[list[ImageRecord], dict[str, Any]]:
    """Extract raw synchronized images through one sequential ZED session."""

    image_root = output_dir / "images"
    records: list[ImageRecord] = []
    with ZedSession(
        svo_path,
        calibration_path,
        zed_config,
        expected_calibration=expected_calibration,
    ) as session:
        total_frames = session.svo_number_of_frames()
        positions = select_svo_positions(
            total_frames,
            start_frame=start_frame,
            end_frame=end_frame,
            num_frames=num_frames,
            frame_step=frame_step,
        )
        targets = set(positions)
        session.seek(positions[0])
        while session.grab():
            source_position = session.svo_position()
            if source_position > positions[-1]:
                break
            if source_position not in targets:
                continue
            timestamp_ns = session.timestamp_ns()
            left_name = Path("left") / f"left_{source_position:06d}.png"
            left_path = image_root / left_name
            write_image(left_path, as_bgr(session.retrieve_image("LEFT_UNRECTIFIED")))
            records.append(
                ImageRecord(
                    path=left_path,
                    name=left_name.as_posix(),
                    side="left",
                    frame=source_position,
                    timestamp_ns=timestamp_ns,
                )
            )
            if include_right:
                right_name = Path("right") / f"right_{source_position:06d}.png"
                right_path = image_root / right_name
                write_image(right_path, as_bgr(session.retrieve_image("RIGHT_UNRECTIFIED")))
                records.append(
                    ImageRecord(
                        path=right_path,
                        name=right_name.as_posix(),
                        side="right",
                        frame=source_position,
                        timestamp_ns=timestamp_ns,
                    )
                )
            if len({record.frame for record in records if record.side == "left"}) == len(positions):
                break
        runtime_metadata = dict(session.runtime_metadata)
    selected_left = [record.frame for record in records if record.side == "left"]
    if selected_left != positions:
        raise RuntimeError(
            f"SVO extraction missed requested positions: expected {positions[:8]}... "
            f"got {selected_left[:8]}..."
        )
    if include_right:
        selected_right = [record.frame for record in records if record.side == "right"]
        if selected_right != positions:
            raise RuntimeError("SVO extraction produced a left/right frame identity mismatch")
    manifest = _manifest_records(
        output_dir,
        records,
        dataset=dataset,
        svo_path=svo_path,
        include_right=include_right,
        positions=positions,
        selection={
            "start_frame": start_frame,
            "end_frame": end_frame,
            "num_frames": num_frames,
            "frame_step": frame_step,
        },
        runtime_metadata=runtime_metadata,
    )
    _write_json(output_dir / "image_manifest.json", manifest)
    return records, manifest


def _write_pairs(path: Path, records: list[ImageRecord], pairs: list[ImagePair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            handle.write(f"{records[pair.first_index].name} {records[pair.second_index].name}\n")


def _write_raw_matches(
    path: Path,
    records: list[ImageRecord],
    pairs: list[ImagePair],
    values: list[DescriptorMatches],
    *,
    min_raw_matches: int,
) -> tuple[int, int, list[dict[str, Any]]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    accepted_pairs = 0
    total_matches = 0
    details: list[dict[str, Any]] = []
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for pair, matches in zip(pairs, values, strict=True):
            accepted = len(matches.query_indices) >= min_raw_matches
            detail = {
                "image1": records[pair.first_index].name,
                "image2": records[pair.second_index].name,
                "side1": records[pair.first_index].side,
                "side2": records[pair.second_index].side,
                "frame1": records[pair.first_index].frame,
                "frame2": records[pair.second_index].frame,
                "category": pair.category,
                "raw_matches": int(len(matches.query_indices)),
                "accepted": accepted,
                "matcher": matches.matcher,
                "invalid_matches_removed": matches.invalid_matches_removed,
                "duplicate_matches_removed": matches.duplicate_matches_removed,
            }
            details.append(detail)
            if not accepted:
                continue
            handle.write(f"{detail['image1']} {detail['image2']}\n")
            for query, train in zip(matches.query_indices, matches.train_indices, strict=True):
                handle.write(f"{int(query)} {int(train)}\n")
            handle.write("\n")
            accepted_pairs += 1
            total_matches += int(len(matches.query_indices))
    return accepted_pairs, total_matches, details


def _run_matching(
    records: list[ImageRecord],
    features: list[Any],
    *,
    matching_dir: Path,
    device: Any,
    temporal_window: int,
    extra_stride: int,
    stereo_window: int,
    min_raw_matches: int,
    max_matches_per_pair: int,
    match_one: Callable[[Any, Any], DescriptorMatches],
    algorithm: str,
    implementation: str,
    matcher_config: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any], list[tuple[int, int, DescriptorMatches]]]:
    """Run one explicit matcher for all bounded candidate pairs."""

    if len(records) != len(features):
        raise ValueError("records and features must have equal lengths")
    if min_raw_matches < 0 or max_matches_per_pair < 0:
        raise ValueError("match thresholds must be non-negative")
    pairs = build_image_pairs(
        records,
        temporal_window=temporal_window,
        extra_stride=extra_stride,
        stereo_window=stereo_window,
    )
    _write_pairs(matching_dir / "pairs.txt", records, pairs)
    started = time.perf_counter()
    matched_values: list[DescriptorMatches] = []
    accepted_database_matches: list[tuple[int, int, DescriptorMatches]] = []
    for pair_index, pair in enumerate(pairs, start=1):
        result = match_one(features[pair.first_index], features[pair.second_index])
        matched_values.append(result)
        if len(result.query_indices) >= min_raw_matches:
            accepted_database_matches.append(
                (pair.first_index + 1, pair.second_index + 1, result)
            )
        if pair_index == 1 or pair_index % 250 == 0 or pair_index == len(pairs):
            print(
                f"[{algorithm}] {pair_index}/{len(pairs)} pairs; "
                f"{len(result.query_indices)} matches; {len(accepted_database_matches)} accepted",
                flush=True,
            )
    match_path = matching_dir / "matches_raw.txt"
    accepted_pairs, total_matches, details = _write_raw_matches(
        match_path,
        records,
        pairs,
        matched_values,
        min_raw_matches=min_raw_matches,
    )
    category_counts = Counter(pair.category for pair in pairs)
    accepted_categories = Counter(
        detail["category"] for detail in details if detail["accepted"]
    )
    match_counts = [int(len(value.query_indices)) for value in matched_values]
    summary: dict[str, Any] = {
        "algorithm": algorithm,
        "implementation": implementation,
        "candidate_pairs": len(pairs),
        "pairs_with_matches": accepted_pairs,
        "accepted_pairs": accepted_pairs,
        "total_accepted_raw_matches": total_matches,
        "min_raw_matches": min_raw_matches,
        "max_matches_per_pair": max_matches_per_pair,
        "temporal_window": temporal_window,
        "extra_stride": extra_stride,
        "stereo_window": stereo_window,
        "candidate_pair_categories": dict(sorted(category_counts.items())),
        "accepted_pair_categories": dict(sorted(accepted_categories.items())),
        "match_count_statistics": {
            "min": int(min(match_counts)) if match_counts else 0,
            "median": float(np.median(match_counts)) if match_counts else 0.0,
            "mean": float(np.mean(match_counts)) if match_counts else 0.0,
            "max": int(max(match_counts)) if match_counts else 0,
        },
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "pairs": details,
    }
    if matcher_config is not None:
        summary["config"] = dict(matcher_config)
    _write_json(matching_dir / "summary.json", summary)
    return match_path, summary, accepted_database_matches


def run_adalam_matching(
    records: list[ImageRecord],
    features: list[Any],
    *,
    matching_dir: Path,
    device: Any,
    temporal_window: int,
    extra_stride: int,
    stereo_window: int,
    min_raw_matches: int,
    max_matches_per_pair: int,
) -> tuple[Path, dict[str, Any], list[tuple[int, int, DescriptorMatches]]]:
    """Run real AdaLAM for all bounded candidate pairs."""

    return _run_matching(
        records,
        features,
        matching_dir=matching_dir,
        device=device,
        temporal_window=temporal_window,
        extra_stride=extra_stride,
        stereo_window=stereo_window,
        min_raw_matches=min_raw_matches,
        max_matches_per_pair=max_matches_per_pair,
        match_one=lambda left, right: match_adalam(
            left,
            right,
            device=device,
            max_matches=max_matches_per_pair,
        ),
        algorithm="AdaLAM",
        implementation="kornia.feature.match_adalam",
    )


def run_lightglue_matching(
    records: list[ImageRecord],
    features: list[Any],
    *,
    matching_dir: Path,
    device: Any,
    temporal_window: int,
    extra_stride: int,
    stereo_window: int,
    min_raw_matches: int,
    max_matches_per_pair: int,
    filter_threshold: float,
    depth_confidence: float,
    width_confidence: float,
) -> tuple[Path, dict[str, Any], list[tuple[int, int, DescriptorMatches]]]:
    """Run LightGlue configured for the cached ALIKED descriptors."""

    matcher, matcher_config = load_lightglue_matcher(
        device,
        filter_threshold=filter_threshold,
        depth_confidence=depth_confidence,
        width_confidence=width_confidence,
    )
    return _run_matching(
        records,
        features,
        matching_dir=matching_dir,
        device=device,
        temporal_window=temporal_window,
        extra_stride=extra_stride,
        stereo_window=stereo_window,
        min_raw_matches=min_raw_matches,
        max_matches_per_pair=max_matches_per_pair,
        match_one=lambda left, right: match_lightglue(
            left,
            right,
            matcher=matcher,
            device=device,
            max_matches=max_matches_per_pair,
        ),
        algorithm="LightGlue",
        implementation="lightglue.LightGlue",
        matcher_config=matcher_config,
    )


def _feature_statistics(features: list[Any]) -> dict[str, float | int]:
    counts = [len(feature.keypoints_xy) for feature in features]
    return {
        "min": int(min(counts)) if counts else 0,
        "median": float(np.median(counts)) if counts else 0.0,
        "mean": float(np.mean(counts)) if counts else 0.0,
        "max": int(max(counts)) if counts else 0,
    }


def _runtime_software(torch: Any | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "package": __version__,
        "python": platform.python_version(),
        "torch": getattr(torch, "__version__", _version("torch")),
        "kornia": _version("kornia"),
        "lightglue": _version("lightglue"),
    }
    if torch is not None:
        result["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            result["cuda_device"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return result


def _default_output_dir(dataset_id: str) -> Path:
    root = Path(__file__).resolve().parents[3]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / "outputs" / dataset_id / "sfm" / run_id


def _prepare_zed_calibration(profile_path: Path, calibration: StereoCalibration) -> Path:
    """Regenerate and return the OpenCV FileStorage file required by ZED."""

    generated_path = profile_path.parent.parent / "generated" / "zed_custom_opencv.yml"
    return write_zed_opencv_calibration(calibration, profile_path, generated_path)


def run_aliked_colmap(
    dataset_path: Path,
    *,
    output_dir: Path | None = None,
    svo_override: Path | None = None,
    profile_override: Path | None = None,
    num_frames: int = 50,
    include_right: bool = False,
    matcher: str = "adalam",
    start_frame: int = 0,
    end_frame: int | None = None,
    frame_step: int = 1,
    image_view: str = "RAW_UNRECTIFIED",
    device: str = "cuda",
    aliked_model: str = "aliked-n16",
    resize: int | None = 1024,
    max_keypoints: int = 800,
    detection_threshold: float = 0.2,
    nms_radius: int = 2,
    temporal_window: int = 5,
    extra_stride: int = 0,
    stereo_window: int = 0,
    min_raw_matches: int = 20,
    max_matches_per_pair: int = 0,
    camera_model: str = "FULL_OPENCV",
    freeze_calibration: bool = True,
    colmap_executable: Path | None = None,
    colmap_threads: int = 8,
    min_geometric_inliers: int = 15,
    max_geometric_error: float = 4.0,
    min_model_size: int = 10,
    mapper_min_num_matches: int = 15,
    init_min_num_inliers: int = 50,
    init_min_tri_angle: float = 2.0,
    calibrated_stereo_planar: bool = False,
    stereo_max_reprojection_error: float = 8.0,
    stereo_motion_ransac_threshold_m: float = 0.12,
    lightglue_filter_threshold: float = 0.1,
    lightglue_depth_confidence: float = 0.95,
    lightglue_width_confidence: float = 0.99,
    pose_h5: Path | None = None,
    pose_time_offset_s: float = 0.0,
    pose_lever_arm_body_m: list[float] | tuple[float, float, float] | None = None,
    skip_colmap: bool = False,
    resume: bool = True,
    purpose: str | None = None,
    retain_policy: str | None = None,
) -> dict[str, Any]:
    """Execute image extraction through COLMAP, or stop explicitly before COLMAP."""

    if image_view.upper() != "RAW_UNRECTIFIED":
        raise ValueError(
            "the package ALIKED workflow currently writes RAW_UNRECTIFIED images; "
            "rectified-camera support must be implemented with a matching derived calibration"
        )
    dataset_path = dataset_path.expanduser().resolve()
    dataset = load_dataset_config(dataset_path)
    profile_path = (profile_override or dataset.calibration_profile).expanduser().resolve()
    calibration = load_calibration_profile(profile_path)
    pose_h5 = pose_h5.expanduser().resolve() if pose_h5 is not None else None
    if pose_h5 is not None and not pose_h5.is_file():
        raise FileNotFoundError(f"HDF5 pose file not found: {pose_h5}")
    zed_calibration_path = _prepare_zed_calibration(profile_path, calibration)
    svo_path = dataset.resolve_svo_path(svo_override)
    if not svo_path.is_file():
        raise FileNotFoundError(f"SVO/SVO2 file not found: {svo_path}")
    output = (output_dir or _default_output_dir(dataset.dataset_id)).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = AlikedConfig(
        model_name=aliked_model,
        resize=resize,
        max_keypoints=max_keypoints,
        detection_threshold=detection_threshold,
        nms_radius=nms_radius,
        device=device,
    )
    matcher_key = str(matcher).strip().lower()
    matcher_names = {"adalam": "AdaLAM", "lightglue": "LightGlue"}
    if matcher_key not in matcher_names:
        raise ValueError("matcher must be adalam or lightglue")
    matcher_name = matcher_names[matcher_key]
    zed_config = ZedSessionConfig()
    started_at = _utc_now()
    effective_purpose = purpose or ("smoke" if skip_colmap else "production")
    effective_retain_policy = retain_policy or ("keep_summary" if skip_colmap else "keep")
    if effective_purpose not in {"production", "baseline", "diagnostic", "smoke", "ablation"}:
        raise ValueError("purpose must be production, baseline, diagnostic, smoke, or ablation")
    if effective_retain_policy not in {"keep", "keep_summary", "archive", "disposable"}:
        raise ValueError("retain_policy must be keep, keep_summary, archive, or disposable")
    if calibrated_stereo_planar and pose_h5 is not None:
        mapping_method = "calibrated_stereo_h5_planar"
    else:
        mapping_method = "calibrated_stereo_planar" if calibrated_stereo_planar else "incremental_mapper"
    scale_description = (
        (
            "metric scale from HDF5 ENU pose and the canonical calibrated stereo baseline; "
            "subject to HDF5 and calibration accuracy"
        )
        if pose_h5 is not None and calibrated_stereo_planar
        else "metric scale from the canonical calibrated stereo baseline; subject to calibration accuracy"
        if calibrated_stereo_planar
        else "arbitrary local SfM scale unless externally constrained"
    )
    run_metadata: dict[str, Any] = {
        "status": "running",
        "result_status": "experimental",
        "purpose": effective_purpose,
        "retain_policy": effective_retain_policy,
        "dataset": dataset.dataset_id,
        "dataset_config": str(dataset_path),
        "svo_path": str(svo_path),
        "svo_identity": _svo_identity(svo_path),
        "image_view": image_view.upper(),
        "include_right": bool(include_right),
        "frame_selection": {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "num_frames": num_frames,
            "frame_step": frame_step,
        },
        "pair_policy": {
            "temporal_window": temporal_window,
            "extra_stride": extra_stride,
            "stereo_window": stereo_window,
            "bounded_graph": True,
        },
        "feature_algorithm": "ALIKED",
        "matcher_algorithm": matcher_name,
        "aliked": config.to_mapping(),
        "calibration_profile": str(profile_path),
        "calibration_sha256": sha256_file(profile_path),
        "zed_opencv_calibration": str(zed_calibration_path),
        "zed_opencv_calibration_sha256": sha256_file(zed_calibration_path),
        "calibration_mode": "frozen" if freeze_calibration else "refined",
        "colmap_camera_model": camera_model.upper(),
        "mapping_method": mapping_method,
        "calibrated_stereo_planar": {
            "enabled": bool(calibrated_stereo_planar),
            "max_reprojection_error_px": float(stereo_max_reprojection_error),
            "motion_ransac_threshold_m": float(stereo_motion_ransac_threshold_m),
        },
        "matcher_config": (
            {
                "filter_threshold": float(lightglue_filter_threshold),
                "depth_confidence": float(lightglue_depth_confidence),
                "width_confidence": float(lightglue_width_confidence),
            }
            if matcher_key == "lightglue"
            else None
        ),
        "external_pose": {
            "enabled": pose_h5 is not None,
            "source": str(pose_h5) if pose_h5 is not None else None,
            "time_offset_s": float(pose_time_offset_s),
            "lever_arm_body_m": (
                [float(value) for value in pose_lever_arm_body_m]
                if pose_lever_arm_body_m is not None
                else [0.0, 0.0, 0.0]
            ),
            "mode": "fixed_left_camera_poses" if pose_h5 is not None else None,
        },
        "colmap_executable": str(colmap_executable) if colmap_executable else None,
        "scale": scale_description,
        "git_sha": _git_sha(),
        "started_at_utc": started_at,
        "software": _runtime_software(),
    }
    _write_json(output / "run.json", run_metadata)
    try:
        existing = _load_existing_manifest(
            output,
            dataset_id=dataset.dataset_id,
            svo_path=svo_path,
            include_right=include_right,
            start_frame=start_frame,
            end_frame=end_frame,
            num_frames=num_frames,
            frame_step=frame_step,
            resume=resume,
        )
        if existing is None:
            records, manifest = extract_svo_images(
                dataset=dataset,
                svo_path=svo_path,
                calibration_path=zed_calibration_path,
                output_dir=output,
                include_right=include_right,
                start_frame=start_frame,
                end_frame=end_frame,
                num_frames=num_frames,
                frame_step=frame_step,
                zed_config=zed_config,
                expected_calibration=calibration,
            )
        else:
            records, manifest = existing
        run_metadata["image_manifest"] = str(output / "image_manifest.json")
        run_metadata["selected_images"] = len(records)
        run_metadata["selected_frames"] = len({record.frame for record in records if record.side == "left"})

        features, feature_summary = extract_aliked_features(
            records,
            output / "features",
            config,
            resume=resume,
        )
        torch, torch_device = load_torch_device(device)
        matching_kwargs = {
            "records": records,
            "features": features,
            "matching_dir": output / "matching",
            "device": torch_device,
            "temporal_window": temporal_window,
            "extra_stride": extra_stride,
            "stereo_window": stereo_window,
            "min_raw_matches": min_raw_matches,
            "max_matches_per_pair": max_matches_per_pair,
        }
        if matcher_key == "lightglue":
            match_path, match_summary, database_matches = run_lightglue_matching(
                **matching_kwargs,
                filter_threshold=lightglue_filter_threshold,
                depth_confidence=lightglue_depth_confidence,
                width_confidence=lightglue_width_confidence,
            )
        else:
            match_path, match_summary, database_matches = run_adalam_matching(**matching_kwargs)
        colmap_dir = output / "colmap"
        colmap_dir.mkdir(parents=True, exist_ok=True)
        camera_config_path = colmap_dir / "camera.json"
        write_colmap_camera_config(
            calibration,
            profile_path,
            camera_config_path,
            camera_model=camera_model,
        )
        database_path = colmap_dir / "database.db"
        database_summary = create_colmap_database(
            database_path,
            features,
            database_matches,
            calibration=calibration,
            camera_model=camera_model,
            insert_matches=skip_colmap,
        )
        if skip_colmap:
            colmap_result: dict[str, Any] = {
                "status": "NOT_EXECUTED",
                "reason": "explicit --skip-colmap",
                "database_stats": database_stats(database_path),
            }
        else:
            executable = find_colmap_executable(colmap_executable)
            run_metadata["colmap_executable"] = str(executable)
            colmap_result = run_colmap_pipeline(
                colmap_executable=executable,
                image_dir=output / "images",
                database_path=database_path,
                match_path=match_path,
                colmap_dir=colmap_dir,
                freeze_calibration=freeze_calibration,
                min_geometric_inliers=min_geometric_inliers,
                max_geometric_error=max_geometric_error,
                min_model_size=min_model_size,
                min_num_matches=mapper_min_num_matches,
                init_min_num_inliers=init_min_num_inliers,
                init_min_tri_angle=init_min_tri_angle,
                threads=colmap_threads,
                calibration=calibration,
                calibrated_stereo_planar=calibrated_stereo_planar,
                stereo_max_reprojection_error=stereo_max_reprojection_error,
                stereo_motion_ransac_threshold_m=stereo_motion_ransac_threshold_m,
                external_pose_h5=pose_h5,
                image_timestamps_ns={
                    int(record.frame): int(record.timestamp_ns)
                    for record in records
                    if record.side == "left"
                },
                external_pose_time_offset_s=pose_time_offset_s,
                external_pose_lever_arm_body_m=pose_lever_arm_body_m,
            )
        model = colmap_result.get("model", {})
        summary: dict[str, Any] = {
            "status": "completed",
            "images_processed": len(records),
            "features_per_image": _feature_statistics(features),
            "candidate_pairs": match_summary["candidate_pairs"],
            "pairs_with_matches": match_summary["pairs_with_matches"],
            "matches_per_pair": match_summary["match_count_statistics"],
            "registered_images": model.get("registered_images") if isinstance(model, dict) else None,
            "points3D": model.get("points3D") if isinstance(model, dict) else None,
            "observations": None,
            "mean_track_length": None,
            "mean_reprojection_error": None,
            "calibration_mode": "frozen" if freeze_calibration else "refined",
            "mapping_method": mapping_method,
            "scale": scale_description,
            "feature_summary": feature_summary,
            "matching_summary": {
                key: value for key, value in match_summary.items() if key != "pairs"
            },
            "database_summary": database_summary,
            "colmap": colmap_result,
        }
        _write_json(output / "summary.json", summary)
        run_metadata.update(
            {
                "status": "completed",
                "result_status": "complete",
                "finished_at_utc": _utc_now(),
                "output_structure": {
                    "images": str(output / "images"),
                    "features": str(output / "features"),
                    "matching": str(output / "matching"),
                    "colmap": str(output / "colmap"),
                    "export": str(output / "export"),
                },
                "feature_summary": feature_summary,
                "matching_summary": {
                    key: value for key, value in match_summary.items() if key != "pairs"
                },
                "database_summary": database_summary,
                "colmap": colmap_result,
                "software": _runtime_software(torch),
            }
        )
        (output / "export").mkdir(parents=True, exist_ok=True)
        _write_json(output / "run.json", run_metadata)
        return summary
    except Exception as error:
        run_metadata.update(
            {
                "status": "failed",
                "result_status": "failed",
                "finished_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        _write_json(output / "run.json", run_metadata)
        raise
