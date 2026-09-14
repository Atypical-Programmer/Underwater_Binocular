"""COLMAP process boundary with frozen/refined calibration modes."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..calibration.conversion import write_colmap_camera_config
from ..calibration.loaders import load_calibration_profile
from ..calibration.models import StereoCalibration
from .colmap_database import database_stats
from .stereo_planar import build_calibrated_stereo_planar_model


def find_colmap_executable(executable: Path | None = None) -> Path:
    """Resolve an explicit executable or a PATH installation."""

    if executable is not None:
        candidate = executable.expanduser().resolve()
    else:
        candidate = Path(shutil.which("colmap") or "")
    if not candidate.is_file():
        raise FileNotFoundError(
            "COLMAP executable not found; pass --colmap-executable or add colmap to PATH"
        )
    return candidate


def build_matches_importer_command(
    colmap_executable: Path,
    database_path: Path,
    match_path: Path,
    *,
    min_geometric_inliers: int = 15,
    max_geometric_error: float = 4.0,
    threads: int = 8,
) -> list[str]:
    """Build the command that imports externally generated raw matches."""

    return [
        str(colmap_executable),
        "matches_importer",
        "--database_path",
        str(database_path),
        "--match_list_path",
        str(match_path),
        "--match_type",
        "raw",
        "--SiftMatching.use_gpu",
        "0",
        "--SiftMatching.num_threads",
        str(threads),
        "--TwoViewGeometry.min_num_inliers",
        str(min_geometric_inliers),
        "--TwoViewGeometry.max_error",
        str(max_geometric_error),
        "--TwoViewGeometry.confidence",
        "0.999",
    ]


def build_colmap_mapper_command(
    colmap_executable: Path,
    database_path: Path,
    image_path: Path,
    output_path: Path,
    *,
    freeze_calibration: bool = True,
    min_model_size: int = 10,
    min_num_matches: int = 15,
    init_min_num_inliers: int = 50,
    max_geometric_error: float = 4.0,
    init_min_tri_angle: float = 2.0,
    threads: int = 8,
) -> list[str]:
    """Build a deterministic mapper command; refinement is never implicit."""

    command = [
        str(colmap_executable),
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--output_path",
        str(output_path),
        "--Mapper.multiple_models",
        "0",
        "--Mapper.max_num_models",
        "1",
        "--Mapper.min_model_size",
        str(min_model_size),
        "--Mapper.min_num_matches",
        str(min_num_matches),
        "--Mapper.num_threads",
        str(threads),
        "--Mapper.ba_use_gpu",
        "0",
        "--Mapper.ba_refine_focal_length",
        "1" if not freeze_calibration else "0",
        "--Mapper.ba_refine_principal_point",
        "1" if not freeze_calibration else "0",
        "--Mapper.ba_refine_extra_params",
        "1" if not freeze_calibration else "0",
        "--Mapper.init_min_num_inliers",
        str(init_min_num_inliers),
        "--Mapper.init_max_error",
        str(max_geometric_error),
        "--Mapper.init_min_tri_angle",
        str(init_min_tri_angle),
    ]
    return command


def build_fixed_pose_bundle_adjuster_command(
    colmap_executable: Path,
    input_path: Path,
    output_path: Path,
    *,
    max_num_iterations: int = 100,
) -> list[str]:
    """Build a point-only bundle-adjustment command for externally fixed poses."""

    return [
        str(colmap_executable),
        "bundle_adjuster",
        "--input_path",
        str(input_path),
        "--output_path",
        str(output_path),
        "--BundleAdjustment.max_num_iterations",
        str(max_num_iterations),
        "--BundleAdjustment.refine_focal_length",
        "0",
        "--BundleAdjustment.refine_principal_point",
        "0",
        "--BundleAdjustment.refine_extra_params",
        "0",
        "--BundleAdjustment.refine_extrinsics",
        "0",
        "--BundleAdjustment.use_gpu",
        "0",
    ]


def _run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace", newline="\n") as handle:
        result = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(
            f"COLMAP command failed with exit code {result.returncode}; tail of {log_path}:\n"
            + "\n".join(tail)
        )


def _model_counts(text_model_path: Path) -> dict[str, int]:
    images_path = text_model_path / "images.txt"
    points_path = text_model_path / "points3D.txt"
    registered_images = 0
    points3d = 0
    if images_path.is_file():
        registered_images = sum(
            1
            for line in images_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip() and not line.startswith("#") and ".png" in line
        )
    if points_path.is_file():
        points3d = sum(
            1
            for line in points_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip() and not line.startswith("#")
        )
    return {"registered_images": registered_images, "points3D": points3d}


def run_colmap_pipeline(
    *,
    colmap_executable: Path,
    image_dir: Path,
    database_path: Path,
    match_path: Path,
    colmap_dir: Path,
    freeze_calibration: bool,
    min_geometric_inliers: int = 15,
    max_geometric_error: float = 4.0,
    min_model_size: int = 10,
    min_num_matches: int = 15,
    init_min_num_inliers: int = 50,
    init_min_tri_angle: float = 2.0,
    threads: int = 8,
    calibration: StereoCalibration | None = None,
    calibrated_stereo_planar: bool = False,
    stereo_max_reprojection_error: float = 8.0,
    stereo_motion_ransac_threshold_m: float = 0.12,
    external_pose_h5: Path | None = None,
    image_timestamps_ns: Mapping[int, int] | None = None,
    external_pose_time_offset_s: float = 0.0,
    external_pose_lever_arm_body_m: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Import explicit matcher outputs and run the selected COLMAP path."""

    executable = find_colmap_executable(colmap_executable)
    for required in (image_dir, database_path, match_path):
        if not required.exists():
            raise FileNotFoundError(f"COLMAP input does not exist: {required}")
    logs_dir = colmap_dir / "logs"
    sparse_dir = colmap_dir / "sparse"
    logs_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    importer = build_matches_importer_command(
        executable,
        database_path,
        match_path,
        min_geometric_inliers=min_geometric_inliers,
        max_geometric_error=max_geometric_error,
        threads=threads,
    )
    _run_logged(importer, logs_dir / "matches_importer.log")
    imported_stats = database_stats(database_path)
    if calibrated_stereo_planar:
        if calibration is None:
            raise ValueError("calibrated_stereo_planar requires a stereo calibration")
        external_left_poses = None
        external_pose_metadata: Mapping[str, Any] | None = None
        if external_pose_h5 is not None:
            if image_timestamps_ns is None:
                raise ValueError("external HDF5 poses require SVO image timestamps")
            from .external_pose import load_h5_inertial_pose_sequence

            sequence = load_h5_inertial_pose_sequence(external_pose_h5)
            external_left_poses, external_pose_metadata = sequence.interpolate_camera_poses(
                image_timestamps_ns,
                time_offset_s=external_pose_time_offset_s,
                lever_arm_body_m=external_pose_lever_arm_body_m,
            )
        external = external_left_poses is not None
        mapping_method = (
            "calibrated_stereo_h5_planar" if external else "calibrated_stereo_planar"
        )
        if external:
            seed_text_path = sparse_dir / "calibrated_stereo_h5_planar_seed_text"
            seed_model_path = sparse_dir / "calibrated_stereo_h5_planar_seed"
            planar_model_path = sparse_dir / "calibrated_stereo_h5_planar"
            planar_text_path = sparse_dir / "calibrated_stereo_h5_planar_text"
        else:
            seed_text_path = sparse_dir / "calibrated_stereo_planar_text"
            seed_model_path = None
            planar_model_path = sparse_dir / "calibrated_stereo_planar"
            planar_text_path = seed_text_path
        for path in (seed_text_path, seed_model_path, planar_model_path, planar_text_path):
            if path is not None and path.exists():
                shutil.rmtree(path)
        if seed_model_path is not None:
            seed_model_path.mkdir(parents=True, exist_ok=True)
        planar_model_path.mkdir(parents=True, exist_ok=True)
        stereo_summary = build_calibrated_stereo_planar_model(
            database_path,
            seed_text_path,
            calibration=calibration,
            max_stereo_reprojection_error=stereo_max_reprojection_error,
            motion_ransac_threshold_m=stereo_motion_ransac_threshold_m,
            external_left_poses=external_left_poses,
            external_pose_metadata=external_pose_metadata,
        )
        stereo_summary_path = colmap_dir / "calibrated_stereo_planar_summary.json"
        stereo_summary_path.write_text(
            json.dumps(stereo_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        seed_converter = [
            str(executable),
            "model_converter",
            "--input_path",
            str(seed_text_path),
            "--output_path",
            str(seed_model_path or planar_model_path),
            "--output_type",
            "BIN",
        ]
        _run_logged(seed_converter, logs_dir / "calibrated_stereo_model_converter.log")
        commands: dict[str, list[str]] = {
            "matches_importer": importer,
            "model_converter_seed": seed_converter,
        }
        if external:
            if seed_model_path is None:
                raise RuntimeError("internal error: HDF5 pose seed model path is unavailable")
            bundle_adjuster = build_fixed_pose_bundle_adjuster_command(
                executable,
                seed_model_path,
                planar_model_path,
            )
            _run_logged(bundle_adjuster, logs_dir / "calibrated_stereo_h5_bundle_adjuster.log")
            commands["bundle_adjuster_fixed_poses"] = bundle_adjuster
            if planar_text_path.exists():
                shutil.rmtree(planar_text_path)
            planar_text_path.mkdir(parents=True, exist_ok=True)
            converter = [
                str(executable),
                "model_converter",
                "--input_path",
                str(planar_model_path),
                "--output_path",
                str(planar_text_path),
                "--output_type",
                "TXT",
            ]
            _run_logged(converter, logs_dir / "calibrated_stereo_h5_model_converter.log")
            commands["model_converter"] = converter
        else:
            commands["model_converter"] = seed_converter
        analyzer = [
            str(executable),
            "model_analyzer",
            "--path",
            str(planar_model_path),
        ]
        analyzer_log = (
            "calibrated_stereo_h5_model_analyzer.log"
            if external
            else "calibrated_stereo_model_analyzer.log"
        )
        _run_logged(analyzer, logs_dir / analyzer_log)
        text_model_path = planar_text_path
        model_summary = {
            "model_path": str(planar_model_path),
            "text_model_path": str(text_model_path),
            "model_dirs": [str(planar_model_path)],
            "mapping_method": mapping_method,
            **_model_counts(text_model_path),
        }
        result = {
            "status": "completed",
            "executable": str(executable),
            "calibration_mode": "frozen",
            "mapping_method": mapping_method,
            "scale": (
                "metric scale from HDF5 ENU pose and the canonical calibrated stereo baseline; "
                "subject to HDF5 and calibration accuracy"
                if external
                else "metric scale from the canonical calibrated stereo baseline; subject to calibration accuracy"
            ),
            "database_stats_after_import": imported_stats,
            "model": model_summary,
            "calibrated_stereo": stereo_summary,
            "commands": {**commands, "model_analyzer": analyzer},
        }
        if external:
            result["external_pose"] = dict(external_pose_metadata or {})
            result["initial_model"] = {
                "model_path": str(seed_model_path),
                "text_model_path": str(seed_text_path),
            }
        (colmap_dir / "model_summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return result
    mapper = build_colmap_mapper_command(
        executable,
        database_path,
        image_dir,
        sparse_dir,
        freeze_calibration=freeze_calibration,
        min_model_size=min_model_size,
        min_num_matches=min_num_matches,
        init_min_num_inliers=init_min_num_inliers,
        max_geometric_error=max_geometric_error,
        init_min_tri_angle=init_min_tri_angle,
        threads=threads,
    )
    _run_logged(mapper, logs_dir / "mapper.log")
    model_dirs = sorted(path for path in sparse_dir.iterdir() if path.is_dir())
    if not model_dirs:
        raise RuntimeError("COLMAP mapper completed without producing a sparse model")
    model_path = model_dirs[0]
    text_model_path = sparse_dir / f"{model_path.name}_text"
    text_model_path.mkdir(parents=True, exist_ok=True)
    converter = [
        str(executable),
        "model_converter",
        "--input_path",
        str(model_path),
        "--output_path",
        str(text_model_path),
        "--output_type",
        "TXT",
    ]
    _run_logged(converter, logs_dir / "model_converter.log")
    model_summary = {
        "model_path": str(model_path),
        "text_model_path": str(text_model_path),
        "model_dirs": [str(path) for path in model_dirs],
        **_model_counts(text_model_path),
    }
    result = {
        "status": "completed",
        "executable": str(executable),
        "calibration_mode": "frozen" if freeze_calibration else "refined",
        "database_stats_after_import": imported_stats,
        "model": model_summary,
        "commands": {"matches_importer": importer, "mapper": mapper, "model_converter": converter},
    }
    (colmap_dir / "model_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def run_colmap_from_cli(args: Any) -> int:
    """Run an explicitly configured external COLMAP command or fail loudly."""

    executable = find_colmap_executable(args.colmap_executable)
    profile_path = args.calibration.resolve()
    profile = load_calibration_profile(profile_path)
    output_dir = Path("outputs") / "colmap"
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_path = output_dir / "colmap_camera.json"
    write_colmap_camera_config(profile, profile_path, camera_path)
    metadata = {"calibration_profile": profile_path.as_posix(), "camera_config": camera_path.as_posix(), "calibration_mode": "frozen" if args.freeze_calibration else "refined", "scale": "arbitrary local SfM scale unless externally constrained"}
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if not args.remainder:
        print(f"COLMAP configuration ready: {output_dir}")
        return 0
    result = subprocess.run([str(executable), *args.remainder], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"COLMAP failed with exit code {result.returncode}")
    return result.returncode
