"""Stable, lightweight command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__


def repository_root() -> Path:
    """Return the source checkout root when running from an installed package."""

    return Path(__file__).resolve().parents[2]


def _default_profile() -> Path:
    return repository_root() / "calibration" / "profiles" / "zed2i_37395692_custom.yaml"


def build_parser() -> argparse.ArgumentParser:
    """Build the public parser without importing optional SDK/SfM modules."""

    parser = argparse.ArgumentParser(prog="underwater", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    calibration = commands.add_parser("calibration", help="validate or generate calibration artifacts")
    calibration_commands = calibration.add_subparsers(dest="calibration_command", required=True)
    validate = calibration_commands.add_parser("validate")
    validate.add_argument("--profile", type=Path, default=_default_profile())
    validate.set_defaults(handler=_handle_calibration_validate)
    generate = calibration_commands.add_parser("generate")
    generate.add_argument("--profile", type=Path, default=_default_profile())
    generate.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root() / "calibration" / "generated",
    )
    generate.add_argument("--orb-scale", type=float, default=0.5)
    generate.set_defaults(handler=_handle_calibration_generate)

    depth = commands.add_parser("depth", help="run a production or diagnostic depth engine")
    depth_commands = depth.add_subparsers(dest="depth_command", required=True)
    export = depth_commands.add_parser("export")
    export.add_argument("--dataset", type=Path, required=True)
    export.add_argument("--engine", choices=("zed-neural", "sgbm"), default="zed-neural")
    export.add_argument("--config", type=Path)
    export.add_argument("--svo", type=Path)
    export.add_argument("--output", type=Path)
    export.add_argument(
        "--format",
        choices=("mp4", "summary"),
        default="summary",
        help="write a streaming MP4 visualization or summary-only outputs",
    )
    export.set_defaults(handler=_handle_depth_export)

    diagnostic = commands.add_parser("diagnostic", help="run read-only calibration/rectification diagnostics")
    diagnostic_commands = diagnostic.add_subparsers(dest="diagnostic_command", required=True)
    for name, handler in (
        ("calibration", _handle_diagnostic_calibration),
        ("rectification", _handle_diagnostic_rectification),
    ):
        item = diagnostic_commands.add_parser(name)
        item.add_argument("--dataset", type=Path, required=True)
        item.add_argument("--profile", type=Path, default=_default_profile())
        item.set_defaults(handler=handler)

    tracking = commands.add_parser("tracking", help="run sequential ZED positional tracking")
    tracking_commands = tracking.add_subparsers(dest="tracking_command", required=True)
    zed_tracking = tracking_commands.add_parser(
        "zed", help="replay a ZED SVO with GEN_1 or GEN_3 positional tracking"
    )
    zed_tracking.add_argument("--dataset", type=Path, required=True)
    zed_tracking.add_argument(
        "--mode",
        type=str.upper,
        choices=("GEN_1", "GEN_3", "BOTH"),
        default="GEN_1",
    )
    zed_tracking.add_argument("--profile", type=Path)
    zed_tracking.add_argument(
        "--calibration-mode",
        choices=("native", "custom"),
        default="native",
        help="native uses calibration embedded in the SVO; custom uses the canonical profile",
    )
    zed_tracking.add_argument("--zed-config", type=Path)
    zed_tracking.add_argument("--svo", type=Path)
    zed_tracking.add_argument("--start-frame", type=int, default=0)
    zed_tracking.add_argument("--end-frame", type=int)
    zed_tracking.add_argument("--max-frames", type=int, default=0)
    zed_tracking.add_argument("--output", type=Path)
    zed_tracking.add_argument(
        "--area-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable Area Memory for the selected replay (default: disabled)",
    )
    zed_tracking.set_defaults(handler=_handle_tracking_zed)

    sfm = commands.add_parser("sfm", help="external SfM integrations")
    sfm_commands = sfm.add_subparsers(dest="sfm_command", required=True)
    aliked_colmap = sfm_commands.add_parser(
        "aliked-colmap", help="run real ALIKED + AdaLAM + COLMAP reconstruction"
    )
    aliked_colmap.add_argument("--dataset", type=Path, required=True)
    aliked_colmap.add_argument("--profile", type=Path)
    aliked_colmap.add_argument("--svo", type=Path)
    aliked_colmap.add_argument("--output", type=Path)
    aliked_colmap.add_argument("--num-frames", type=int, default=50)
    aliked_colmap.add_argument("--start-frame", type=int, default=0)
    aliked_colmap.add_argument("--end-frame", type=int)
    aliked_colmap.add_argument("--frame-step", type=int, default=1)
    aliked_colmap.add_argument("--include-right", action="store_true")
    aliked_colmap.add_argument("--device", default="cuda")
    aliked_colmap.add_argument("--aliked-model", default="aliked-n16")
    aliked_colmap.add_argument(
        "--resize", type=int, default=1024, help="ALIKED long-edge resize; 0 disables it"
    )
    aliked_colmap.add_argument("--max-keypoints", type=int, default=800)
    aliked_colmap.add_argument("--detection-threshold", type=float, default=0.2)
    aliked_colmap.add_argument("--nms-radius", type=int, default=2)
    aliked_colmap.add_argument("--temporal-window", type=int, default=5)
    aliked_colmap.add_argument("--extra-stride", type=int, default=0)
    aliked_colmap.add_argument("--stereo-window", type=int, default=0)
    aliked_colmap.add_argument("--min-raw-matches", type=int, default=20)
    aliked_colmap.add_argument("--max-matches-per-pair", type=int, default=0)
    aliked_colmap.add_argument(
        "--camera-model",
        choices=("PINHOLE", "OPENCV", "FULL_OPENCV"),
        default="FULL_OPENCV",
    )
    aliked_colmap.add_argument(
        "--freeze-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="freeze COLMAP calibration by default; use --no-freeze-calibration to refine",
    )
    aliked_colmap.add_argument("--colmap-executable", type=Path)
    aliked_colmap.add_argument("--colmap-threads", type=int, default=8)
    aliked_colmap.add_argument("--min-geometric-inliers", type=int, default=15)
    aliked_colmap.add_argument("--max-geometric-error", type=float, default=4.0)
    aliked_colmap.add_argument("--min-model-size", type=int, default=10)
    aliked_colmap.add_argument("--mapper-min-num-matches", type=int, default=15)
    aliked_colmap.add_argument("--init-min-num-inliers", type=int, default=50)
    aliked_colmap.add_argument("--init-min-tri-angle", type=float, default=2.0)
    aliked_colmap.add_argument(
        "--calibrated-stereo-planar",
        action="store_true",
        help="use known synchronized stereo geometry for near-planar scenes and write a complete COLMAP model",
    )
    aliked_colmap.add_argument(
        "--stereo-max-reprojection-error",
        type=float,
        default=8.0,
        help="maximum per-camera reprojection error for calibrated stereo points (pixels)",
    )
    aliked_colmap.add_argument(
        "--stereo-motion-ransac-threshold-m",
        type=float,
        default=0.12,
        help="3-D rigid-motion RANSAC threshold between adjacent selected frames (meters)",
    )
    aliked_colmap.add_argument(
        "--pose-h5",
        type=Path,
        help="HDF5 inertial trajectory used to fix left-camera poses in calibrated stereo mode",
    )
    aliked_colmap.add_argument(
        "--pose-time-offset-s",
        type=float,
        default=0.0,
        help="seconds added to each SVO timestamp before HDF5 pose interpolation",
    )
    aliked_colmap.add_argument(
        "--pose-lever-arm-body-m",
        type=float,
        nargs=3,
        metavar=("FORWARD", "RIGHT", "DOWN"),
        default=None,
        help="INS reference point to left-camera-center lever arm in body metres",
    )
    aliked_colmap.add_argument(
        "--skip-colmap",
        action="store_true",
        help="stop after writing the custom-feature/match database; mark COLMAP NOT EXECUTED",
    )
    aliked_colmap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse only provenance-matching images/features",
    )
    aliked_colmap.set_defaults(handler=_handle_sfm_aliked_colmap)
    colmap = sfm_commands.add_parser("colmap", help="run the decomposed COLMAP integration")
    colmap.add_argument("--calibration", type=Path, default=_default_profile())
    colmap.add_argument(
        "--freeze-calibration", action=argparse.BooleanOptionalAction, default=True
    )
    colmap.add_argument("--colmap-executable", type=Path)
    colmap.add_argument("remainder", nargs=argparse.REMAINDER)
    colmap.set_defaults(handler=_handle_sfm_colmap)

    outputs = commands.add_parser("outputs", help="inventory and conservatively prune local outputs")
    outputs_commands = outputs.add_subparsers(dest="outputs_command", required=True)
    inventory = outputs_commands.add_parser(
        "inventory", help="scan output/, outputs/, and cache/ and emit JSON/CSV/Markdown reports"
    )
    inventory.add_argument("--repo", type=Path, default=repository_root())
    inventory.add_argument("--output-dir", type=Path, default=repository_root())
    inventory.add_argument(
        "--no-hash-duplicates",
        action="store_true",
        help="skip hashes even for same-size duplicate candidates",
    )
    inventory.set_defaults(handler=_handle_outputs_inventory)
    prune = outputs_commands.add_parser(
        "prune", help="dry-run by default; apply only an explicit category or manifest"
    )
    prune.add_argument("--repo", type=Path, default=repository_root())
    prune.add_argument("--apply", action="store_true", help="apply the explicit cleanup selection")
    prune.add_argument(
        "--dry-run",
        action="store_true",
        help="show the selected paths without changing files (the default)",
    )
    prune.add_argument("--category", choices=("empty",), help="narrow built-in selection")
    prune.add_argument("--manifest", type=Path, help="JSON manifest containing exact relative paths")
    prune.set_defaults(handler=_handle_outputs_prune)
    return parser


def _handle_calibration_validate(args: argparse.Namespace) -> int:
    from .calibration.loaders import load_calibration_profile
    from .calibration.validation import validate_calibration

    calibration = load_calibration_profile(args.profile)
    report = validate_calibration(calibration)
    print(f"Calibration: PASS ({args.profile})")
    print(f"resolution={calibration.resolution[0]}x{calibration.resolution[1]}")
    print(f"baseline_norm_m={calibration.baseline_m:.12g}")
    print(f"rotation_determinant={report['rotation_determinant']:.12g}")
    return 0


def _handle_calibration_generate(args: argparse.Namespace) -> int:
    from .calibration.conversion import generate_derived_calibrations

    paths = generate_derived_calibrations(args.profile, args.output_dir, orb_scale=args.orb_scale)
    for path in paths:
        print(path)
    return 0


def _handle_depth_export(args: argparse.Namespace) -> int:
    if args.engine == "zed-neural":
        from .depth.zed_sdk import run_depth_export

        return run_depth_export(args)
    from .depth.sgbm import run_sgbm_export

    return run_sgbm_export(args)


def _handle_diagnostic_calibration(args: argparse.Namespace) -> int:
    from .diagnostics.calibration import calibration_diagnostic

    print(calibration_diagnostic(args.profile, args.dataset))
    return 0


def _handle_diagnostic_rectification(args: argparse.Namespace) -> int:
    from .diagnostics.rectification import rectification_diagnostic

    print(rectification_diagnostic(args.profile, args.dataset))
    return 0


def _default_run_output(dataset_id: str, category: str) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return repository_root() / "outputs" / dataset_id / category / run_id


def _handle_tracking_zed(args: argparse.Namespace) -> int:
    from .config.loaders import load_dataset_config
    from .tracking.zed import run_tracking

    dataset = load_dataset_config(args.dataset)
    output = args.output or _default_run_output(dataset.dataset_id, "tracking")
    run_tracking(
        args.dataset,
        mode=args.mode,
        output_dir=output,
        calibration_mode=args.calibration_mode,
        svo_override=args.svo,
        profile_override=args.profile,
        zed_config_path=args.zed_config,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        max_frames=args.max_frames,
        enable_area_memory=args.area_memory,
    )
    print(f"Tracking completed: {output}")
    return 0


def _handle_outputs_inventory(args: argparse.Namespace) -> int:
    from .outputs.inventory import build_inventory, write_inventory_reports

    inventory = build_inventory(args.repo, hash_duplicates=not args.no_hash_duplicates)
    paths = write_inventory_reports(inventory, args.output_dir)
    print(json.dumps({"entries": inventory["entry_count"], "reports": [str(path) for path in paths]}, indent=2))
    return 0


def _handle_outputs_prune(args: argparse.Namespace) -> int:
    from .outputs.inventory import prune_outputs

    if args.apply and args.dry_run:
        raise ValueError("--apply and --dry-run cannot be used together")
    actions = prune_outputs(
        args.repo,
        apply=args.apply,
        category=args.category,
        manifest=args.manifest,
    )
    for action in actions:
        print(action)
    if not actions:
        print("No paths selected.")
    return 0


def _handle_sfm_aliked_colmap(args: argparse.Namespace) -> int:
    from .reconstruction.pipeline import run_aliked_colmap

    summary = run_aliked_colmap(
        args.dataset,
        output_dir=args.output,
        svo_override=args.svo,
        profile_override=args.profile,
        num_frames=args.num_frames,
        include_right=args.include_right,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        frame_step=args.frame_step,
        device=args.device,
        aliked_model=args.aliked_model,
        resize=args.resize,
        max_keypoints=args.max_keypoints,
        detection_threshold=args.detection_threshold,
        nms_radius=args.nms_radius,
        temporal_window=args.temporal_window,
        extra_stride=args.extra_stride,
        stereo_window=args.stereo_window,
        min_raw_matches=args.min_raw_matches,
        max_matches_per_pair=args.max_matches_per_pair,
        camera_model=args.camera_model,
        freeze_calibration=args.freeze_calibration,
        colmap_executable=args.colmap_executable,
        colmap_threads=args.colmap_threads,
        min_geometric_inliers=args.min_geometric_inliers,
        max_geometric_error=args.max_geometric_error,
        min_model_size=args.min_model_size,
        mapper_min_num_matches=args.mapper_min_num_matches,
        init_min_num_inliers=args.init_min_num_inliers,
        init_min_tri_angle=args.init_min_tri_angle,
        calibrated_stereo_planar=args.calibrated_stereo_planar,
        stereo_max_reprojection_error=args.stereo_max_reprojection_error,
        stereo_motion_ransac_threshold_m=args.stereo_motion_ransac_threshold_m,
        pose_h5=args.pose_h5,
        pose_time_offset_s=args.pose_time_offset_s,
        pose_lever_arm_body_m=args.pose_lever_arm_body_m,
        skip_colmap=args.skip_colmap,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _handle_sfm_colmap(args: argparse.Namespace) -> int:
    from .reconstruction.colmap_runner import run_colmap_from_cli

    return run_colmap_from_cli(args)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
