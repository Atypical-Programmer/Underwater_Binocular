"""COLMAP process boundary with frozen/refined calibration modes."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..calibration.conversion import write_colmap_camera_config
from ..calibration.loaders import load_calibration_profile


def build_colmap_mapper_command(colmap_executable: Path, database_path: Path, image_path: Path, output_path: Path, *, freeze_calibration: bool = True) -> list[str]:
    """Build a deterministic mapper command; refinement is never implicit."""

    command = [str(colmap_executable), "mapper", "--database_path", str(database_path), "--image_path", str(image_path), "--output_path", str(output_path)]
    if freeze_calibration:
        command.extend(["--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0", "--Mapper.ba_refine_extra_params", "0"])
    else:
        command.extend(["--Mapper.ba_refine_focal_length", "1", "--Mapper.ba_refine_principal_point", "1", "--Mapper.ba_refine_extra_params", "1"])
    return command


def run_colmap_from_cli(args: Any) -> int:
    """Run an explicitly configured external COLMAP command or fail loudly."""

    executable = args.colmap_executable
    if executable is None:
        executable = Path(shutil.which("colmap") or "")
    if not executable or not executable.is_file():
        raise FileNotFoundError("COLMAP executable not found; pass --colmap-executable or add colmap to PATH")
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
