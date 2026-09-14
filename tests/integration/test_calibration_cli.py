"""Integration checks for generated calibration artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from underwater_binocular.calibration.conversion import generate_derived_calibrations

ROOT = Path(__file__).resolve().parents[2]


def test_generation_is_reproducible_in_a_temporary_directory(tmp_path: Path) -> None:
    profile = ROOT / "calibration/profiles/zed2i_37395692_custom.yaml"
    paths = generate_derived_calibrations(profile, tmp_path, orb_scale=0.5)

    assert [path.name for path in paths] == [
        "zed_custom_opencv.yml",
        "orbslam3_stereo.yaml",
        "colmap_camera.json",
    ]
    zed_text = paths[0].read_text(encoding="utf-8")
    assert zed_text.startswith("%YAML:1.0\n---\n# GENERATED FILE - DO NOT EDIT")
    assert "# source hash: " in zed_text
    colmap = json.loads(paths[2].read_text(encoding="utf-8"))
    assert colmap["left"]["width"] == 1920
    assert colmap["right"]["height"] == 1080
