"""Run ALIKED + AdaLAM matching and COLMAP sparse SfM on a frame subset.

The script intentionally keeps the neural feature stage GPU-bounded:
one image is resident on CUDA at a time, descriptors/keypoints are copied to
CPU immediately, and AdaLAM is evaluated one image pair at a time.  COLMAP is
run with GPU matching/BA disabled because the custom ALIKED/AdaLAM matches are
already imported and geometric verification is CPU-side in this workflow.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import kornia.feature as KF
import numpy as np
import torch
from lightglue import ALIKED


PAIR_ID_MAX = 2_147_483_647
COLMAP_CAMERA_MODEL_IDS = {
    "PINHOLE": 1,
    "OPENCV": 4,
    "FULL_OPENCV": 6,
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    name: str
    side: str
    frame: int


def log(message: str) -> None:
    print(message, flush=True)


def json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def frame_number(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot parse frame number from {path.name}") from exc


def select_side_images(directory: Path, side: str, num_frames: int) -> list[Path]:
    paths = sorted(directory.glob(f"{side}_*.png"), key=frame_number)
    if len(paths) < num_frames:
        raise FileNotFoundError(
            f"{directory} contains {len(paths)} {side} PNGs, but {num_frames} are required"
        )
    selected = paths[:num_frames]
    actual = [frame_number(path) for path in selected]
    expected = list(range(actual[0], actual[0] + num_frames))
    if actual != expected:
        raise RuntimeError(
            f"The selected {side} frames are not contiguous: {actual[:8]} ..."
        )
    return selected


def select_images(input_dir: Path, num_images: int) -> list[Path]:
    """Backward-compatible left-only selector."""
    return select_side_images(input_dir, "left", num_images)


def select_image_records(
    input_dir: Path, num_frames: int, include_right: bool
) -> tuple[Path, list[ImageRecord]]:
    """Select synchronized left/right frames and names relative to COLMAP's image root."""
    input_dir = input_dir.resolve()
    if (input_dir / "left").is_dir():
        image_root = input_dir
        left_dir = input_dir / "left"
    elif input_dir.name.lower() == "left":
        image_root = input_dir.parent
        left_dir = input_dir
    else:
        image_root = input_dir
        left_dir = input_dir

    right_dir = image_root / "right"
    left_paths = select_side_images(left_dir, "left", num_frames)
    right_paths: list[Path] = []
    if include_right:
        if not right_dir.is_dir():
            raise FileNotFoundError(f"Right-image directory does not exist: {right_dir}")
        right_paths = select_side_images(right_dir, "right", num_frames)
        left_frames = [frame_number(path) for path in left_paths]
        right_frames = [frame_number(path) for path in right_paths]
        if left_frames != right_frames:
            raise RuntimeError(
                "Left/right frame numbers do not match: "
                f"left={left_frames[:8]}..., right={right_frames[:8]}..."
            )

    def record(path: Path, side: str) -> ImageRecord:
        return ImageRecord(
            path=path,
            name=path.relative_to(image_root).as_posix(),
            side=side,
            frame=frame_number(path),
        )

    records: list[ImageRecord] = []
    if include_right:
        right_by_frame = {frame_number(path): path for path in right_paths}
        for left_path in left_paths:
            frame = frame_number(left_path)
            records.append(record(left_path, "left"))
            records.append(record(right_by_frame[frame], "right"))
    else:
        records = [record(path, "left") for path in left_paths]
    return image_root, records


def read_rgb_tensor(path: Path, device: torch.device) -> tuple[torch.Tensor, tuple[int, int]]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"OpenCV could not read {path}")
    height, width = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().div_(255.0)
    return tensor.to(device), (height, width)


def load_feature_file(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "keypoints": np.asarray(data["keypoints"], dtype=np.float32),
            "descriptors": np.asarray(data["descriptors"], dtype=np.float32),
            "scores": np.asarray(data["scores"], dtype=np.float32),
            "image_size": np.asarray(data["image_size"], dtype=np.int32),
        }


def extract_features(
    image_records: list[ImageRecord],
    feature_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    feature_dir.mkdir(parents=True, exist_ok=True)
    log(
        f"[ALIKED] model={args.aliked_model}, resize={args.resize}, "
        f"max_keypoints={args.max_keypoints}, threshold={args.detection_threshold}"
    )
    model = ALIKED(
        model_name=args.aliked_model,
        max_num_keypoints=args.max_keypoints,
        detection_threshold=args.detection_threshold,
        nms_radius=args.nms_radius,
    ).eval().to(device)

    features: list[dict[str, np.ndarray]] = []
    peak_mb = 0.0
    total_start = time.perf_counter()
    for index, image_record in enumerate(image_records):
        feature_path = feature_dir / f"{index:06d}.npz"
        if args.resume and feature_path.exists():
            feature = load_feature_file(feature_path)
            features.append(feature)
            log(f"[ALIKED] {index + 1:04d}/{len(image_records)} resume {image_record.name}: "
                f"{len(feature['keypoints'])} keypoints")
            continue

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        image_tensor, image_size = read_rgb_tensor(image_record.path, device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.extract(image_tensor, resize=args.resize)
        keypoints = output["keypoints"][0].detach().cpu().numpy().astype(np.float32, copy=False)
        descriptors = output["descriptors"][0].detach().cpu().numpy().astype(np.float32, copy=False)
        scores = output["keypoint_scores"][0].detach().cpu().numpy().astype(np.float32, copy=False)
        feature = {
            "keypoints": np.ascontiguousarray(keypoints),
            "descriptors": np.ascontiguousarray(descriptors),
            "scores": np.ascontiguousarray(scores),
            "image_size": np.asarray([image_size[0], image_size[1]], dtype=np.int32),
        }
        np.savez_compressed(feature_path, **feature)
        features.append(feature)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            current_peak = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            peak_mb = max(peak_mb, current_peak)
        else:
            current_peak = 0.0
        elapsed = time.perf_counter() - start
        log(
            f"[ALIKED] {index + 1:04d}/{len(image_records)} {image_record.name}: "
            f"{len(keypoints)} keypoints, {elapsed:.2f}s, peak={current_peak:.0f} MB"
        )

        del output, image_tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    summary = {
        "num_images": len(image_records),
        "num_left_images": sum(item.side == "left" for item in image_records),
        "num_right_images": sum(item.side == "right" for item in image_records),
        "feature_dir": str(feature_dir),
        "model": args.aliked_model,
        "resize": args.resize,
        "max_keypoints": args.max_keypoints,
        "detection_threshold": args.detection_threshold,
        "nms_radius": args.nms_radius,
        "total_keypoints": int(sum(len(item["keypoints"]) for item in features)),
        "mean_keypoints": float(np.mean([len(item["keypoints"]) for item in features])),
        "max_single_image_cuda_mb": peak_mb,
        "elapsed_seconds": time.perf_counter() - total_start,
    }
    json_dump(feature_dir.parent / "feature_summary.json", summary)
    return features, summary


def build_pairs(
    image_records: list[ImageRecord],
    window: int,
    extra_stride: int,
    stereo_window: int,
) -> list[tuple[int, int]]:
    """Build temporal same-camera pairs plus synchronized cross-camera pairs."""
    pairs: set[tuple[int, int]] = set()

    indices_by_side: dict[str, list[int]] = {}
    for index, record in enumerate(image_records):
        indices_by_side.setdefault(record.side, []).append(index)

    for indices in indices_by_side.values():
        indices.sort(key=lambda index: image_records[index].frame)
        for position, first in enumerate(indices):
            for offset in range(1, window + 1):
                if position + offset >= len(indices):
                    break
                second = indices[position + offset]
                if image_records[second].frame - image_records[first].frame == offset:
                    pairs.add(tuple(sorted((first, second))))
            if extra_stride > window and position + extra_stride < len(indices):
                second = indices[position + extra_stride]
                if image_records[second].frame - image_records[first].frame == extra_stride:
                    pairs.add(tuple(sorted((first, second))))

    left_indices = indices_by_side.get("left", [])
    right_indices = indices_by_side.get("right", [])
    if left_indices and right_indices and stereo_window >= 0:
        for left_index in left_indices:
            left_frame = image_records[left_index].frame
            for right_index in right_indices:
                if abs(left_frame - image_records[right_index].frame) <= stereo_window:
                    pairs.add(tuple(sorted((left_index, right_index))))
    return sorted(pairs)


def enforce_one_to_one_matches(
    matches: np.ndarray, quality: np.ndarray
) -> tuple[np.ndarray, int]:
    """Remove repeated keypoint indices while keeping the best AdaLAM pairs."""
    if len(matches) <= 1:
        return matches, 0
    quality_values = np.asarray(quality).reshape(-1)
    order = np.argsort(quality_values, kind="stable")
    used_first: set[int] = set()
    used_second: set[int] = set()
    keep: list[int] = []
    for match_index in order:
        first, second = (int(value) for value in matches[match_index])
        if first in used_first or second in used_second:
            continue
        used_first.add(first)
        used_second.add(second)
        keep.append(int(match_index))
    keep_array = np.asarray(keep, dtype=np.int64)
    filtered = matches[keep_array] if len(keep_array) else matches[:0]
    return filtered, int(len(matches) - len(filtered))


def match_features(
    image_records: list[ImageRecord],
    features: list[dict[str, np.ndarray]],
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    pairs = build_pairs(
        image_records,
        args.temporal_window,
        args.extra_stride,
        args.stereo_window,
    )
    match_path = output_dir / "matches_raw.txt"
    stats: list[dict[str, Any]] = []
    accepted = 0
    total_matches = 0
    start_all = time.perf_counter()

    log(
        f"[AdaLAM] {len(pairs)} candidate pairs, temporal_window={args.temporal_window}, "
        f"stereo_window={args.stereo_window}, min_raw_matches={args.min_raw_matches}, "
        f"device={device}"
    )
    with match_path.open("w", encoding="utf-8", newline="\n") as handle:
        for pair_index, (first, second) in enumerate(pairs, start=1):
            record1 = image_records[first]
            record2 = image_records[second]
            item1 = features[first]
            item2 = features[second]
            start = time.perf_counter()
            desc1 = torch.from_numpy(item1["descriptors"]).to(device)
            desc2 = torch.from_numpy(item2["descriptors"]).to(device)
            kp1 = torch.from_numpy(item1["keypoints"]).to(device)
            kp2 = torch.from_numpy(item2["keypoints"]).to(device)
            with torch.inference_mode():
                laf1 = KF.laf_from_center_scale_ori(kp1[None])
                laf2 = KF.laf_from_center_scale_ori(kp2[None])
                quality, match_indices = KF.match_adalam(
                    desc1,
                    desc2,
                    laf1,
                    laf2,
                    hw1=tuple(int(value) for value in item1["image_size"]),
                    hw2=tuple(int(value) for value in item2["image_size"]),
                )
            quality_cpu = quality.detach().cpu().numpy()
            matches = match_indices.detach().cpu().numpy().astype(np.int32, copy=False)
            if matches.ndim != 2 or matches.shape[1] != 2:
                raise RuntimeError(f"Unexpected AdaLAM output shape {matches.shape}")
            matches, duplicate_matches_removed = enforce_one_to_one_matches(matches, quality_cpu)
            if args.max_matches_per_pair > 0 and len(matches) > args.max_matches_per_pair:
                matches = matches[: args.max_matches_per_pair]

            num_matches = int(len(matches))
            accepted_pair = num_matches >= args.min_raw_matches
            stats.append(
                {
                    "image1": record1.name,
                    "image2": record2.name,
                    "side1": record1.side,
                    "side2": record2.side,
                    "frame1": record1.frame,
                    "frame2": record2.frame,
                    "index1": first,
                    "index2": second,
                    "raw_matches": num_matches,
                    "duplicate_matches_removed": duplicate_matches_removed,
                    "accepted": accepted_pair,
                    "elapsed_seconds": time.perf_counter() - start,
                }
            )
            if accepted_pair:
                handle.write(f"{record1.name} {record2.name}\n")
                for left_index, right_index in matches:
                    handle.write(f"{int(left_index)} {int(right_index)}\n")
                handle.write("\n")
                accepted += 1
                total_matches += num_matches

            del desc1, desc2, kp1, kp2, laf1, laf2, quality, quality_cpu, match_indices
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if pair_index == 1 or pair_index % 25 == 0 or pair_index == len(pairs):
                log(
                    f"[AdaLAM] {pair_index:03d}/{len(pairs)} "
                    f"{record1.name}-{record2.name}: "
                    f"{num_matches} raw ({accepted} accepted)"
                )

    summary = {
        "candidate_pairs": len(pairs),
        "accepted_pairs": accepted,
        "total_accepted_raw_matches": total_matches,
        "temporal_window": args.temporal_window,
        "stereo_window": args.stereo_window,
        "extra_stride": args.extra_stride,
        "min_raw_matches": args.min_raw_matches,
        "max_matches_per_pair": args.max_matches_per_pair,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - start_all,
        "pairs": stats,
    }
    json_dump(output_dir / "adalam_match_summary.json", summary)
    return match_path, summary


def camera_params_for_side(
    args: argparse.Namespace, side: str, camera_model: str
) -> list[float]:
    if side == "left":
        values = (args.fx, args.fy, args.cx, args.cy, args.k1, args.k2,
                  args.p1, args.p2, args.k3, args.k4, args.k5, args.k6)
    elif side == "right":
        values = (args.right_fx, args.right_fy, args.right_cx, args.right_cy,
                  args.right_k1, args.right_k2, args.right_p1, args.right_p2,
                  args.right_k3, args.right_k4, args.right_k5, args.right_k6)
    else:
        raise ValueError(f"Unsupported camera side: {side}")

    if camera_model == "PINHOLE":
        return list(values[:4])
    if camera_model == "OPENCV":
        return list(values[:8])
    if camera_model == "FULL_OPENCV":
        return list(values)
    raise ValueError(f"Unsupported COLMAP camera model: {camera_model}")


def create_colmap_database(
    image_records: list[ImageRecord],
    features: list[dict[str, np.ndarray]],
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    database_path = output_dir / "database.db"
    if database_path.exists():
        database_path.unlink()
    connection = sqlite3.connect(str(database_path))
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE cameras (
            camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            model INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            params BLOB,
            prior_focal_length INTEGER NOT NULL
        );
        CREATE TABLE images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name TEXT NOT NULL UNIQUE,
            camera_id INTEGER NOT NULL,
            CONSTRAINT image_id_check CHECK(image_id >= 0 AND image_id < 2147483647),
            FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
        );
        CREATE UNIQUE INDEX index_name ON images(name);
        CREATE TABLE keypoints (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        );
        CREATE TABLE descriptors (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        );
        CREATE TABLE matches (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB
        );
        CREATE TABLE two_view_geometries (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            config INTEGER NOT NULL,
            F BLOB,
            E BLOB,
            H BLOB,
            qvec BLOB,
            tvec BLOB
        );
        """
    )

    camera_model = args.colmap_camera_model.upper()
    camera_sides = ["left"]
    if any(record.side == "right" for record in image_records):
        camera_sides.append("right")

    camera_id_by_side: dict[str, int] = {}
    for camera_id, side in enumerate(camera_sides, start=1):
        side_features = [
            feature for record, feature in zip(image_records, features) if record.side == side
        ]
        if not side_features:
            raise RuntimeError(f"No features found for camera side: {side}")
        side_size = tuple(int(value) for value in side_features[0]["image_size"])
        if any(tuple(int(value) for value in item["image_size"]) != side_size
               for item in side_features):
            raise RuntimeError(f"All {side} images must have the same dimensions")
        height, width = side_size
        params = np.asarray(
            camera_params_for_side(args, side, camera_model), dtype=np.float64
        )
        connection.execute(
            "INSERT INTO cameras(camera_id,model,width,height,params,prior_focal_length) "
            "VALUES(?,?,?,?,?,?)",
            (
                camera_id,
                COLMAP_CAMERA_MODEL_IDS[camera_model],
                width,
                height,
                params.tobytes(),
                1,
            ),
        )
        camera_id_by_side[side] = camera_id

    for image_id, (image_record, feature) in enumerate(
        zip(image_records, features), start=1
    ):
        keypoints = np.ascontiguousarray(feature["keypoints"], dtype=np.float32)
        # COLMAP requires a descriptor row for every image. Matching was done
        # by AdaLAM above, so these are deliberately inert uint8 placeholders.
        dummy_descriptors = np.zeros((len(keypoints), 128), dtype=np.uint8)
        connection.execute(
            "INSERT INTO images(image_id,name,camera_id) VALUES(?,?,?)",
            (image_id, image_record.name, camera_id_by_side[image_record.side]),
        )
        connection.execute(
            "INSERT INTO keypoints(image_id,rows,cols,data) VALUES(?,?,?,?)",
            (image_id, len(keypoints), 2, keypoints.tobytes()),
        )
        connection.execute(
            "INSERT INTO descriptors(image_id,rows,cols,data) VALUES(?,?,?,?)",
            (image_id, len(dummy_descriptors), 128, dummy_descriptors.tobytes()),
        )
    connection.commit()
    connection.close()
    return database_path


def database_stats(database_path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(str(database_path))
    tables = ["cameras", "images", "keypoints", "descriptors", "matches", "two_view_geometries"]
    counts = {table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in tables}
    verified = int(
        connection.execute(
            "SELECT COUNT(*) FROM two_view_geometries WHERE rows >= 15"
        ).fetchone()[0]
    )
    counts["verified_geometries_rows_ge_15"] = verified
    connection.close()
    return counts


def run_logged(command: list[str], log_path: Path) -> None:
    log(f"[COLMAP] {' '.join(command)}")
    with log_path.open("w", encoding="utf-8", errors="replace", newline="\n") as handle:
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(
            f"COLMAP failed with exit code {completed.returncode}; tail of {log_path}:\n"
            + "\n".join(tail)
        )


def run_colmap(
    image_dir: Path,
    database_path: Path,
    match_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    colmap = Path(args.colmap_exe)
    if not colmap.exists():
        raise FileNotFoundError(f"COLMAP executable not found: {colmap}")
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    run_logged(
        [
            str(colmap),
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
            str(args.colmap_threads),
            "--TwoViewGeometry.min_num_inliers",
            str(args.min_geometric_inliers),
            "--TwoViewGeometry.max_error",
            str(args.max_geometric_error),
            "--TwoViewGeometry.confidence",
            "0.999",
        ],
        output_dir / "colmap_matches_importer.log",
    )
    imported_stats = database_stats(database_path)
    json_dump(output_dir / "colmap_database_stats.json", imported_stats)
    log(f"[COLMAP] imported database stats: {imported_stats}")

    run_logged(
        [
            str(colmap),
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse_dir),
            "--Mapper.multiple_models",
            "0",
            "--Mapper.max_num_models",
            "1",
            "--Mapper.min_model_size",
            str(args.min_model_size),
            "--Mapper.min_num_matches",
            str(args.mapper_min_num_matches),
            "--Mapper.num_threads",
            str(args.colmap_threads),
            "--Mapper.ba_use_gpu",
            "0",
            "--Mapper.ba_refine_focal_length",
            str(args.mapper_refine_focal_length),
            "--Mapper.ba_refine_principal_point",
            str(args.mapper_refine_principal_point),
            "--Mapper.ba_refine_extra_params",
            str(args.mapper_refine_extra_params),
            "--Mapper.max_extra_param",
            str(args.mapper_max_extra_param),
            "--Mapper.init_min_num_inliers",
            str(args.init_min_num_inliers),
            "--Mapper.init_max_error",
            str(args.max_geometric_error),
            "--Mapper.init_min_tri_angle",
            str(args.mapper_init_min_tri_angle),
        ],
        output_dir / "colmap_mapper.log",
    )

    model_dirs = sorted(path for path in sparse_dir.iterdir() if path.is_dir())
    model_summary: dict[str, Any] = {"model_dirs": [str(path) for path in model_dirs]}
    if model_dirs:
        model_path = model_dirs[0]
        analyzer_log = output_dir / "colmap_model_analyzer.log"
        run_logged([str(colmap), "model_analyzer", "--path", str(model_path)], analyzer_log)
        text_path = sparse_dir / f"{model_path.name}_text"
        text_path.mkdir(parents=True, exist_ok=True)
        run_logged(
            [
                str(colmap),
                "model_converter",
                "--input_path",
                str(model_path),
                "--output_path",
                str(text_path),
                "--output_type",
                "TXT",
            ],
            output_dir / "colmap_model_converter.log",
        )
        images_txt = text_path / "images.txt"
        points_txt = text_path / "points3D.txt"
        registered_images = 0
        points3d = 0
        if images_txt.exists():
            registered_images = sum(
                1
                for line in images_txt.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip() and not line.startswith("#") and ".png" in line
            )
        if points_txt.exists():
            points3d = sum(
                1
                for line in points_txt.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip() and not line.startswith("#")
            )
        model_summary.update(
            {
                "model_path": str(model_path),
                "registered_images": registered_images,
                "points3D": points3d,
                "text_model_path": str(text_path),
            }
        )
        log(
            f"[COLMAP] model={model_path.name}, registered_images={registered_images}, "
            f"points3D={points3d}"
        )
    else:
        log("[COLMAP] mapper produced no sparse model")
    return {"database": imported_stats, "model": model_summary}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent
    default_input = repo / "output" / "20260802_150233_sample1000"
    default_output = (
        repo / "output" / "20260802_150233_sample1000"
        / "sfm_aliked_adalam_2000_full_opencv_refined"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--colmap-exe",
        default=r"D:\Underwater\Software\colmap-x64-windows-cuda\bin\colmap.exe",
    )
    parser.add_argument("--num-images", type=int, default=1000,
                        help="Number of synchronized frames to select from each camera")
    parser.add_argument("--include-right", action="store_true",
                        help="Include the corresponding right-camera images")
    parser.add_argument("--temporal-window", type=int, default=5)
    parser.add_argument(
        "--stereo-window",
        type=int,
        default=0,
        help="Match left frame t with right frames in [t-window,t+window]; 0 means same frame",
    )
    parser.add_argument("--extra-stride", type=int, default=0)
    parser.add_argument("--aliked-model", default="aliked-n16")
    parser.add_argument("--resize", type=int, default=1024)
    parser.add_argument("--max-keypoints", type=int, default=800)
    parser.add_argument("--detection-threshold", type=float, default=0.2)
    parser.add_argument("--nms-radius", type=int, default=2)
    parser.add_argument("--min-raw-matches", type=int, default=20)
    parser.add_argument("--max-matches-per-pair", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.75)
    parser.add_argument("--colmap-threads", type=int, default=8)
    parser.add_argument("--min-geometric-inliers", type=int, default=15)
    parser.add_argument("--max-geometric-error", type=float, default=4.0)
    parser.add_argument("--mapper-min-num-matches", type=int, default=15)
    parser.add_argument("--init-min-num-inliers", type=int, default=50)
    parser.add_argument("--mapper-init-min-tri-angle", type=float, default=2.0)
    parser.add_argument("--mapper-refine-focal-length", type=int, choices=[0, 1], default=1)
    parser.add_argument("--mapper-refine-principal-point", type=int, choices=[0, 1], default=0)
    parser.add_argument("--mapper-refine-extra-params", type=int, choices=[0, 1], default=1)
    parser.add_argument("--mapper-max-extra-param", type=float, default=100.0)
    parser.add_argument("--min-model-size", type=int, default=10)
    parser.add_argument(
        "--colmap-camera-model",
        choices=["PINHOLE", "OPENCV", "FULL_OPENCV"],
        default="FULL_OPENCV",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-colmap", action="store_true")

    # Left-camera calibration from Calibration/zed_custom_opencv.yml.
    parser.add_argument("--fx", type=float, default=1443.326338)
    parser.add_argument("--fy", type=float, default=1441.541002)
    parser.add_argument("--cx", type=float, default=967.033878)
    parser.add_argument("--cy", type=float, default=540.469238)
    parser.add_argument("--k1", type=float, default=0.3151847)
    parser.add_argument("--k2", type=float, default=-0.7495814)
    parser.add_argument("--p1", type=float, default=0.002129304)
    parser.add_argument("--p2", type=float, default=0.005759798)
    parser.add_argument("--k3", type=float, default=6.688496)
    parser.add_argument("--k4", type=float, default=0.0)
    parser.add_argument("--k5", type=float, default=0.0)
    parser.add_argument("--k6", type=float, default=0.0)
    # Right-camera calibration from Calibration/zed_custom_opencv.yml.
    parser.add_argument("--right-fx", type=float, default=1449.830254)
    parser.add_argument("--right-fy", type=float, default=1448.097563)
    parser.add_argument("--right-cx", type=float, default=975.331674)
    parser.add_argument("--right-cy", type=float, default=514.232843)
    parser.add_argument("--right-k1", type=float, default=0.3023032)
    parser.add_argument("--right-k2", type=float, default=-0.1692313)
    parser.add_argument("--right-p1", type=float, default=0.001685104)
    parser.add_argument("--right-p2", type=float, default=0.01069086)
    parser.add_argument("--right-k3", type=float, default=1.944058)
    parser.add_argument("--right-k4", type=float, default=0.0)
    parser.add_argument("--right-k5", type=float, default=0.0)
    parser.add_argument("--right-k6", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = output_dir / "features"

    image_root, image_records = select_image_records(
        input_dir, args.num_images, args.include_right
    )
    manifest = {
        "input_dir": str(input_dir),
        "image_root": str(image_root),
        "num_frames": args.num_images,
        "num_images": len(image_records),
        "include_right": args.include_right,
        "images": [record.name for record in image_records],
        "records": [
            {"name": record.name, "side": record.side, "frame": record.frame}
            for record in image_records
        ],
        "first_image": image_records[0].name,
        "last_image": image_records[-1].name,
        "selection": "first contiguous synchronized frames sorted by camera and numeric frame index",
    }
    json_dump(output_dir / "image_manifest.json", manifest)

    if args.device.lower() == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "cuda":
        if not 0.1 <= args.gpu_memory_fraction <= 1.0:
            raise ValueError("--gpu-memory-fraction must be in [0.1, 1.0]")
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
        total_mb = torch.cuda.get_device_properties(device).total_memory / (1024.0 * 1024.0)
        free_mb, total_driver_mb = torch.cuda.mem_get_info(device)
        log(
            f"[CUDA] {torch.cuda.get_device_name(device)}, allocator_fraction="
            f"{args.gpu_memory_fraction:.2f}, driver_free={free_mb / 1024**2:.2f} GB/"
            f"{total_driver_mb / 1024**2:.2f} GB, total={total_mb / 1024:.2f} GB"
        )
    else:
        log("[CUDA] disabled; feature extraction and AdaLAM will run on CPU")

    features, feature_summary = extract_features(image_records, feature_dir, device, args)
    match_path, match_summary = match_features(image_records, features, output_dir, device, args)
    database_path = create_colmap_database(image_records, features, output_dir, args)
    json_dump(
        output_dir / "pipeline_config.json",
        {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "kornia": __import__("kornia").__version__,
            "input_dir": str(input_dir),
            "image_root": str(image_root),
            "output_dir": str(output_dir),
            "include_right": args.include_right,
            "num_frames": args.num_images,
            "stereo_window": args.stereo_window,
            "feature_summary": feature_summary,
            "match_summary_without_pair_details": {
                key: value for key, value in match_summary.items() if key != "pairs"
            },
            "colmap_camera_model": args.colmap_camera_model.upper(),
            "colmap_camera_params_by_side": {
                side: camera_params_for_side(args, side, args.colmap_camera_model.upper())
                for side in ("left", "right")
                if side == "left" or args.include_right
            },
            "mapper_ba_refine_focal_length": args.mapper_refine_focal_length,
            "mapper_ba_refine_principal_point": args.mapper_refine_principal_point,
            "mapper_ba_refine_extra_params": args.mapper_refine_extra_params,
            "mapper_max_extra_param": args.mapper_max_extra_param,
            "colmap_custom_descriptors": "zero uint8 placeholders; custom ALIKED descriptors are in features/*.npz",
        },
    )
    log(f"[DB] created {database_path} ({database_path.stat().st_size / 1024**2:.1f} MB)")

    result: dict[str, Any] = {
        "feature_summary": feature_summary,
        "adalam_summary": {
            key: value for key, value in match_summary.items() if key != "pairs"
        },
        "database_path": str(database_path),
        "match_path": str(match_path),
    }
    if not args.skip_colmap:
        result["colmap"] = run_colmap(image_root, database_path, match_path, output_dir, args)
    json_dump(output_dir / "run_summary.json", result)
    log(f"[DONE] results: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # make failures visible in the terminal and logs
        log(f"[ERROR] {type(exc).__name__}: {exc}")
        raise
