"""Compatibility wrapper for the package-owned ORB-SLAM3 generator."""

from __future__ import annotations

import argparse
from pathlib import Path

from underwater_binocular.calibration.conversion import generate_derived_calibrations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    root = args.root.resolve()
    profile = root / "calibration" / "profiles" / "zed2i_37395692_custom.yaml"
    output = args.output
    if output is None:
        output = root / "calibration" / "generated" / "orbslam3_stereo.yaml"
    if not output.is_absolute():
        output = root / output
    generated = generate_derived_calibrations(profile, output.parent, orb_scale=args.scale)
    generated_orb = next(path for path in generated if path.name == "orbslam3_stereo.yaml")
    if generated_orb.resolve() != output.resolve():
        output.write_text(generated_orb.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
