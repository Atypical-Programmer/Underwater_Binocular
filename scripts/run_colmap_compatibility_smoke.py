"""Run a small real ALIKED + AdaLAM + COLMAP compatibility audit.

The script reuses already extracted legacy images but invokes the package
workflow and the external COLMAP importer/mapper/model-converter boundary.
It never opens the SVO when the generated image manifest matches the SVO
identity, which makes it usable on a machine without ``pyzed.sl``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _svo_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _frame_number(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", type=Path, default=Path("configs/datasets/20260802_150233.yaml"))
    parser.add_argument("--svo", type=Path, required=True)
    parser.add_argument("--source-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--colmap", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    dataset = (repo / args.dataset).resolve() if not args.dataset.is_absolute() else args.dataset.resolve()
    svo = args.svo.expanduser().resolve()
    source = args.source_images.expanduser().resolve()
    output = args.output.expanduser().resolve()
    profile = None if args.profile is None else args.profile.expanduser().resolve()
    colmap = args.colmap.expanduser().resolve()
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")
    if not svo.is_file() or not source.is_dir() or not colmap.is_file():
        raise FileNotFoundError("SVO, source image directory, and COLMAP executable must exist")

    left = sorted((source / "left").glob("left_*.png"), key=_frame_number)
    right = sorted((source / "right").glob("right_*.png"), key=_frame_number)
    left_by_frame = {_frame_number(path): path for path in left}
    right_by_frame = {_frame_number(path): path for path in right}
    positions = sorted(set(left_by_frame) & set(right_by_frame))[: args.frames]
    if len(positions) != args.frames:
        raise RuntimeError(f"only {len(positions)} synchronized image pairs are available")

    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for side, paths in (("left", left_by_frame), ("right", right_by_frame)):
        for position in positions:
            source_path = paths[position]
            relative = Path("images") / side / source_path.name
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
            records.append(
                {
                    "path": relative.as_posix(),
                    "name": f"{side}/{source_path.name}",
                    "side": side,
                    "frame": position,
                    "timestamp_ns": position * 1_000_000_000 // 30,
                    "sha256": None,
                }
            )
    manifest = {
        "dataset": "20260802_150233",
        "svo_path": str(svo),
        "svo_identity": _svo_identity(svo),
        "output_dir": str(output),
        "include_right": True,
        "image_view": "RAW_UNRECTIFIED",
        "image_coordinate_space": "original_image_pixels",
        "selection": {"start_frame": 0, "end_frame": None, "num_frames": args.frames, "frame_step": 1},
        "selected_svo_positions": positions,
        "selected_frame_count": len(positions),
        "records": records,
        "runtime_calibration": None,
    }
    (output / "image_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    from underwater_binocular.reconstruction.pipeline import run_aliked_colmap

    summary = run_aliked_colmap(
        dataset,
        output_dir=output,
        svo_override=svo,
        profile_override=profile,
        num_frames=args.frames,
        include_right=True,
        device=args.device,
        resize=1024,
        max_keypoints=800,
        temporal_window=3,
        stereo_window=0,
        min_raw_matches=10,
        camera_model="FULL_OPENCV",
        freeze_calibration=True,
        colmap_executable=colmap,
        colmap_threads=4,
        min_geometric_inliers=10,
        max_geometric_error=4.0,
        min_model_size=3,
        mapper_min_num_matches=10,
        init_min_num_inliers=10,
        init_min_tri_angle=0.5,
        skip_colmap=False,
        resume=True,
        purpose="diagnostic",
        retain_policy="keep_summary",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
