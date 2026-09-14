"""Stable, lightweight command-line entry point."""

from __future__ import annotations

import argparse
import sys
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

    sfm = commands.add_parser("sfm", help="external SfM integrations")
    sfm_commands = sfm.add_subparsers(dest="sfm_command", required=True)
    colmap = sfm_commands.add_parser("colmap", help="run the decomposed COLMAP integration")
    colmap.add_argument("--calibration", type=Path, default=_default_profile())
    colmap.add_argument(
        "--freeze-calibration", action=argparse.BooleanOptionalAction, default=True
    )
    colmap.add_argument("--colmap-executable", type=Path)
    colmap.add_argument("remainder", nargs=argparse.REMAINDER)
    colmap.set_defaults(handler=_handle_sfm_colmap)
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
