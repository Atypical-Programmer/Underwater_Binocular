"""Audit custom rectification with an independent left-right SGBM check.

This diagnostic deliberately starts each combination from a fresh SVO handle,
retrieves raw unrectified views, performs one OpenCV remap, computes both
left-to-right and right-to-left SGBM disparities, and keeps only pixels whose
two disparities agree.  It is intended to distinguish a calibration/profile
effect from dense matcher outliers at weakly textured image borders.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

import rectification_profile_experiment as base


def _matcher(min_disparity: int, num_disparities: int = 256) -> Any:
    block = 5
    return base.cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block,
        P1=8 * block * block,
        P2=32 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=getattr(
            base.cv2,
            "STEREO_SGBM_MODE_SGBM_3WAY",
            base.cv2.STEREO_SGBM_MODE_SGBM,
        ),
    )


def _band_masks(cx: float, cy: float) -> dict[str, np.ndarray]:
    return base._band_masks(cx, cy)


def _values(depth: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, mask in masks.items():
        values = depth[mask]
        values = values[np.isfinite(values) & (values > 0.0)]
        result[name] = {
            "median_m": float(np.median(values)) if values.size else None,
            "valid_count": int(values.size),
            "valid_ratio": float(values.size / max(1, np.count_nonzero(mask))),
        }
    return result


def _profile(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    names = [
        "center_r025",
        "inner_mid_r025_050",
        "outer_mid_r050_075",
        "edge_r075",
    ]
    values: dict[str, list[float]] = {name: [] for name in names}
    deltas: list[float] = []
    strict = 0
    for row in rows:
        bands = row[key]
        medians = [bands[name]["median_m"] for name in names]
        if all(value is not None for value in medians):
            numeric = [float(value) for value in medians]
            for name, value in zip(names, numeric):
                values[name].append(value)
            deltas.append(numeric[-1] - numeric[0])
            strict += int(all(numeric[i] < numeric[i + 1] for i in range(3)))
    aggregate: dict[str, Any] = {}
    for name in names:
        array = np.asarray(values[name], dtype=np.float64)
        aggregate[name] = {
            "valid_frames": int(array.size),
            "median_of_frame_medians_m": float(np.median(array)) if array.size else None,
            "min_m": float(np.min(array)) if array.size else None,
            "max_m": float(np.max(array)) if array.size else None,
        }
    delta_array = np.asarray(deltas, dtype=np.float64)
    return {
        "bands": aggregate,
        "complete_frames": int(delta_array.size),
        "strict_center_to_edge_monotonic_frames": int(strict),
        "edge_minus_center_median_m": float(np.median(delta_array)) if delta_array.size else None,
        "edge_minus_center_min_m": float(np.min(delta_array)) if delta_array.size else None,
        "edge_minus_center_max_m": float(np.max(delta_array)) if delta_array.size else None,
    }


def _run(
    svo: Path,
    calibration_path: Path,
    rectification: dict[str, Any],
    positions: list[int],
    num_disparities: int,
) -> dict[str, Any]:
    zed = base._open_svo(svo, calibration_path)
    try:
        left_mat, right_mat = base.sl.Mat(), base.sl.Mat()
        left_matcher = _matcher(0, num_disparities)
        right_matcher = _matcher(-num_disparities, num_disparities)
        image_masks = _band_masks(base.WIDTH / 2.0, base.HEIGHT / 2.0)
        optical_masks = _band_masks(
            float(rectification["P_left"][0, 2]),
            float(rectification["P_left"][1, 2]),
        )
        remap_left_valid = base.cv2.remap(
            np.ones((base.HEIGHT, base.WIDTH), dtype=np.uint8),
            rectification["left_map_x"],
            rectification["left_map_y"],
            base.cv2.INTER_NEAREST,
            borderMode=base.cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        remap_right_valid = base.cv2.remap(
            np.ones((base.HEIGHT, base.WIDTH), dtype=np.uint8),
            rectification["right_map_x"],
            rectification["right_map_y"],
            base.cv2.INTER_NEAREST,
            borderMode=base.cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        rows: list[dict[str, Any]] = []
        yy, xx = np.indices((base.HEIGHT, base.WIDTH), dtype=np.int32)
        for position in positions:
            base._check(zed.set_svo_position(position), f"set SVO position {position}")
            base._check(zed.grab(), f"grab SVO position {position}")
            base._check(
                zed.retrieve_image(left_mat, base.sl.VIEW.LEFT_UNRECTIFIED),
                "retrieve raw left",
            )
            base._check(
                zed.retrieve_image(right_mat, base.sl.VIEW.RIGHT_UNRECTIFIED),
                "retrieve raw right",
            )
            left_raw = base._as_gray(left_mat.get_data())
            right_raw = base._as_gray(right_mat.get_data())
            left = base.cv2.remap(
                left_raw,
                rectification["left_map_x"],
                rectification["left_map_y"],
                base.cv2.INTER_LINEAR,
            )
            right = base.cv2.remap(
                right_raw,
                rectification["right_map_x"],
                rectification["right_map_y"],
                base.cv2.INTER_LINEAR,
            )
            disparity_left = left_matcher.compute(left, right).astype(np.float32) / 16.0
            disparity_right = right_matcher.compute(right, left).astype(np.float32) / 16.0

            xr = np.rint(xx.astype(np.float32) - disparity_left).astype(np.int32)
            inside = (xr >= 0) & (xr < base.WIDTH)
            sampled_right = np.full(disparity_left.shape, np.nan, dtype=np.float32)
            sampled_right[inside] = disparity_right[yy[inside], xr[inside]]
            left_right_error = np.abs(disparity_left + sampled_right)
            valid = (
                np.isfinite(disparity_left)
                & (disparity_left > 1.0)
                & np.isfinite(sampled_right)
                & (sampled_right < -1.0)
                & (left_right_error <= 1.5)
                & (remap_left_valid > 0)
                & (remap_right_valid[yy, np.clip(xr, 0, base.WIDTH - 1)] > 0)
            )
            depth = np.full(disparity_left.shape, np.nan, dtype=np.float32)
            depth[valid] = (
                rectification["focal_px"]
                * rectification["baseline_m"]
                / disparity_left[valid]
            ).astype(np.float32)
            depth[(depth <= 0.0) | (depth > 100.0)] = np.nan
            row = {
                "frame": int(position),
                "image_bands": _values(depth, image_masks),
                "optical_bands": _values(depth, optical_masks),
                "raw_sgbm_valid_ratio": float(
                    np.count_nonzero(
                        np.isfinite(disparity_left)
                        & (disparity_left > 1.0)
                    )
                    / disparity_left.size
                ),
                "lr_consistent_valid_ratio": float(np.count_nonzero(valid) / valid.size),
                "lr_consistency_error_median_px": float(
                    np.median(left_right_error[valid])
                )
                if np.any(valid)
                else None,
            }
            rows.append(row)
            center = row["image_bands"]["center_r025"]["median_m"]
            edge = row["image_bands"]["edge_r075"]["median_m"]
            print(
                f"  frame={position:5d} LR-consistent image center/edge="
                f"{center} / {edge} m; valid={row['lr_consistent_valid_ratio']:.4f}",
                flush=True,
            )
        return {
            "rectification": {
                "alpha": rectification["alpha"],
                "centered_projection": rectification["centered_projection"],
                "focal_px": rectification["focal_px"],
                "baseline_m": rectification["baseline_m"],
                "P_left": rectification["P_left"].tolist(),
                "P_right": rectification["P_right"].tolist(),
                "roi_left": list(rectification["roi_left"]),
                "roi_right": list(rectification["roi_right"]),
            },
                "matcher": {
                    "left_to_right_min_disparity": 0,
                    "right_to_left_min_disparity": -num_disparities,
                    "num_disparities": num_disparities,
                "lr_threshold_px": 1.5,
            },
            "rows": rows,
            "image_center_profile": _profile(rows, "image_bands"),
            "optical_center_profile": _profile(rows, "optical_bands"),
        }
    finally:
        zed.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svo", type=Path, default=base.DEFAULT_SVO)
    parser.add_argument("--calibration", type=Path, default=base.DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=base.ROOT / "RECTIFICATION_LR_CONSISTENCY_EXPERIMENT.json")
    parser.add_argument("--positions", default="0,5000,10000,15000,20000,25000,30000,35000,35854")
    parser.add_argument("--num-disparities", type=int, default=256)
    parser.add_argument(
        "--variant",
        choices=("all", "alpha0", "alpha0_centered", "alpha05", "alpha05_centered", "alpha1", "alpha1_centered"),
        default="all",
        help="Run one rectification variant instead of all six.",
    )
    args = parser.parse_args()
    svo = args.svo.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    output = args.output.expanduser().resolve()
    positions = base._parse_positions(args.positions)
    if args.num_disparities <= 0 or args.num_disparities % 16 != 0:
        parser.error("--num-disparities must be a positive multiple of 16")
    calibration = base._load_calibration(calibration_path)
    rectifications = [
        base._rectification(calibration, 0.0, False),
        base._rectification(calibration, 0.0, True),
        base._rectification(calibration, 0.5, False),
        base._rectification(calibration, 0.5, True),
        base._rectification(calibration, 1.0, False),
        base._rectification(calibration, 1.0, True),
    ]
    if args.variant != "all":
        variant_index = {
            "alpha0": 0,
            "alpha0_centered": 1,
            "alpha05": 2,
            "alpha05_centered": 3,
            "alpha1": 4,
            "alpha1_centered": 5,
        }[args.variant]
        rectifications = [rectifications[variant_index]]
    started = time.monotonic()
    combinations: list[dict[str, Any]] = []
    for rectification in rectifications:
        print(
            f"\n=== alpha={rectification['alpha']} centered={rectification['centered_projection']} "
            f"f={rectification['focal_px']:.3f} ===",
            flush=True,
        )
        combinations.append(
            _run(svo, calibration_path, rectification, positions, args.num_disparities)
        )
    report = {
        "script": Path(__file__).name,
        "svo": str(svo),
        "calibration": str(calibration_path),
        "protocol": "fresh SVO handle per combination; raw unrectified views; one OpenCV remap; independent bidirectional SGBM; left-right threshold 1.5 px",
        "variant": args.variant,
        "num_disparities": args.num_disparities,
        "positions": positions,
        "calibration_values": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in calibration.items()
        },
        "interpretation": {
            "image_center": "output coordinate (960,540)",
            "optical_center": "rectified P1 principal point",
            "desired_profile": "center_r025 < inner_mid_r025_050 < outer_mid_r050_075 < edge_r075",
            "depth_formula": "Z_m = rectified_P1_fx_px * rectified_baseline_m / disparity_px",
            "lr_filter": "at xL, xR=round(xL-dLR); require |dLR+dRL(xR)| <= 1.5 px",
            "not_ground_truth": "profile agreement is not independent metric validation",
        },
        "combinations": combinations,
        "elapsed_seconds": time.monotonic() - started,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved report: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
