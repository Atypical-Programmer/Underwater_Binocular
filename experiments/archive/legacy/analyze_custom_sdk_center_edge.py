"""Compare centre and peripheral regions in Custom-calibrated SDK depth.

The final depth video stores only colourized frames and centre statistics.  This
diagnostic reopens the SVO with the same Custom calibration and samples evenly
spaced frames directly from ``MEASURE.DEPTH``.  It compares the central 1/3
image cell with the eight surrounding 1/3 cells, so "most edge regions" has an
explicit definition instead of relying on a few hand-picked pixels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyzed.sl as sl

import regenerate_depth_histogram as sdk_helpers


DEFAULT_SVO = Path(__file__).resolve().with_name("20260802_150233.svo2")
DEFAULT_CALIBRATION = (
    Path(__file__).resolve().parent / "Calibration" / "zed_custom_opencv.yml"
)
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("CUSTOM_SDK_CENTER_EDGE_100.json")


def _median(depth: np.ndarray, bounds: tuple[int, int, int, int]) -> float:
    y0, y1, x0, x1 = bounds
    values = np.asarray(depth[y0:y1, x0:x1], dtype=np.float32)
    valid = np.isfinite(values) & (values > 0.0) & (values <= 100.0)
    if not np.any(valid):
        return float("nan")
    return float(np.median(values[valid]))


def _stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    valid = array[np.isfinite(array) & (array > 0.0)]
    result: dict[str, Any] = {
        "count": int(valid.size),
        "invalid_count": int(array.size - valid.size),
        "valid_ratio": float(valid.size / array.size) if array.size else 0.0,
    }
    if valid.size:
        result.update(
            {
                "min_m": float(np.min(valid)),
                "p05_m": float(np.percentile(valid, 5)),
                "median_m": float(np.median(valid)),
                "mean_m": float(np.mean(valid)),
                "p95_m": float(np.percentile(valid, 95)),
                "max_m": float(np.max(valid)),
            }
        )
    return result


def _compare(left: list[float], right: list[float]) -> dict[str, Any]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0.0) & (b > 0.0)
    delta = b[valid] - a[valid]
    result: dict[str, Any] = {
        "comparable_frames": int(delta.size),
        "center_lower_count": int(np.count_nonzero(delta > 0.0)),
        "center_lower_ratio": (
            float(np.count_nonzero(delta > 0.0) / delta.size) if delta.size else 0.0
        ),
        "center_equal_count": int(np.count_nonzero(delta == 0.0)),
        "center_higher_count": int(np.count_nonzero(delta < 0.0)),
    }
    if delta.size:
        result.update(
            {
                "edge_minus_center_median_m": float(np.median(delta)),
                "edge_minus_center_p05_m": float(np.percentile(delta, 5)),
                "edge_minus_center_p95_m": float(np.percentile(delta, 95)),
                "edge_minus_center_min_m": float(np.min(delta)),
                "edge_minus_center_max_m": float(np.max(delta)),
            }
        )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare centre and edge regions in Custom-calibrated SDK depth."
    )
    parser.add_argument("svo", nargs="?", type=Path, default=DEFAULT_SVO)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.sample_count <= 0:
        parser.error("--sample-count must be greater than zero")
    return args


def main() -> int:
    args = _parse_args()
    svo_path = args.svo.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not svo_path.is_file():
        raise FileNotFoundError(svo_path)
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)

    zed = sl.Camera()
    sdk_helpers._check_status(
        zed.open(
            sdk_helpers._make_init(
                svo_path, calibration_path, sl.DEPTH_MODE.NEURAL
            )
        ),
        "open SVO2 with Custom calibration",
    )

    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    try:
        total_frames = int(zed.get_svo_number_of_frames())
        sample_positions = np.unique(
            np.rint(
                np.linspace(0, total_frames - 1, min(args.sample_count, total_frames))
            ).astype(np.int64)
        )
        info = zed.get_camera_information()
        runtime = sl.RuntimeParameters()
        runtime.confidence_threshold = 30
        runtime.texture_confidence_threshold = 100
        runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA
        depth_mat = sl.Mat()
        edge_names = (
            "top_left",
            "top",
            "top_right",
            "left",
            "right",
            "bottom_left",
            "bottom",
            "bottom_right",
        )

        for index, target in enumerate(sample_positions):
            sdk_helpers._check_status(
                zed.set_svo_position(int(target)),
                f"seek to SVO position {int(target)}",
            )
            status = zed.grab(runtime)
            if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            sdk_helpers._check_status(status, f"grab sampled frame {int(target)}")
            sdk_helpers._check_status(
                zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH),
                f"retrieve sampled SDK depth {int(target)}",
            )
            depth = sdk_helpers._depth_array(depth_mat)
            height, width = depth.shape
            y = [0, height // 3, 2 * height // 3, height]
            x = [0, width // 3, 2 * width // 3, width]
            regions = {
                "top_left": (y[0], y[1], x[0], x[1]),
                "top": (y[0], y[1], x[1], x[2]),
                "top_right": (y[0], y[1], x[2], x[3]),
                "left": (y[1], y[2], x[0], x[1]),
                "center": (y[1], y[2], x[1], x[2]),
                "right": (y[1], y[2], x[2], x[3]),
                "bottom_left": (y[2], y[3], x[0], x[1]),
                "bottom": (y[2], y[3], x[1], x[2]),
                "bottom_right": (y[2], y[3], x[2], x[3]),
            }
            medians = {name: _median(depth, bounds) for name, bounds in regions.items()}
            center_pixel = float(depth[height // 2, width // 2])
            center_window = _median(
                depth,
                (
                    max(0, height // 2 - 2),
                    min(height, height // 2 + 3),
                    max(0, width // 2 - 2),
                    min(width, width // 2 + 3),
                ),
            )
            edge_values = [medians[name] for name in edge_names]
            valid_edges = [value for value in edge_values if np.isfinite(value)]
            center = medians["center"]
            majority_lower = None
            if np.isfinite(center) and valid_edges:
                majority_lower = sum(center < value for value in valid_edges) > (
                    len(valid_edges) / 2.0
                )
            rows.append(
                {
                    "sample_index": index,
                    "requested_position": int(target),
                    "actual_position_after_grab": int(zed.get_svo_position()),
                    "regions_m": medians,
                    "center_pixel_m": center_pixel,
                    "center_window_5x5_m": center_window,
                    "valid_edge_region_count": len(valid_edges),
                    "center_lower_than_edge_majority": majority_lower,
                }
            )
            if (index + 1) % 10 == 0 or index + 1 == len(sample_positions):
                print(f"Sampled {index + 1}/{len(sample_positions)} frames", flush=True)

        if not rows:
            raise RuntimeError("No sampled SDK depth frames were collected")
    finally:
        zed.close()

    center = [row["regions_m"]["center"] for row in rows]
    center_pixel = [row["center_pixel_m"] for row in rows]
    center_window = [row["center_window_5x5_m"] for row in rows]
    edge_pooled = [
        row["regions_m"][name]
        for row in rows
        for name in edge_names
    ]
    comparisons = {
        name: _compare(
            [row["regions_m"]["center"] for row in rows],
            [row["regions_m"][name] for row in rows],
        )
        for name in edge_names
    }
    center_pixel_comparisons = {
        name: _compare(
            center_pixel,
            [row["regions_m"][name] for row in rows],
        )
        for name in edge_names
    }
    majority_rows = [
        row
        for row in rows
        if row["center_lower_than_edge_majority"] is not None
    ]
    majority_true = sum(
        row["center_lower_than_edge_majority"] is True for row in majority_rows
    )
    majority_summary = {
        "comparable_frames": len(majority_rows),
        "center_lower_than_more_than_half_of_valid_edge_regions": majority_true,
        "ratio": float(majority_true / len(majority_rows)) if majority_rows else 0.0,
    }
    exact_majority_values: list[bool] = []
    for row in rows:
        value = row["center_pixel_m"]
        edge_values = [row["regions_m"][name] for name in edge_names]
        edge_values = [item for item in edge_values if np.isfinite(item)]
        if np.isfinite(value) and edge_values:
            exact_majority_values.append(
                sum(value < item for item in edge_values) > len(edge_values) / 2.0
            )
    exact_majority_true = sum(exact_majority_values)

    metadata = {
        "input_file": str(svo_path),
        "calibration_file": str(calibration_path),
        "calibration_source": "Calibration/zed_custom_opencv.yml",
        "native_svo_calibration_used": False,
        "depth_source": "pyzed.sl.MEASURE.DEPTH",
        "depth_mode": "NEURAL",
        "camera_model": sdk_helpers._status_name(info.camera_model),
        "total_svo_frames": total_frames,
        "sampled_frames": len(rows),
        "region_definition": {
            "grid": "3x3 equal image cells",
            "center": "middle cell",
            "edge_regions": list(edge_names),
            "interpretation": "center lower than edge means centre is nearer",
        },
        "center_region_stats_m": _stats(center),
        "center_pixel_stats_m": _stats(center_pixel),
        "center_window_5x5_stats_m": _stats(center_window),
        "all_edge_region_medians_pooled_stats_m": _stats(edge_pooled),
        "center_vs_each_edge_region": comparisons,
        "center_vs_edge_majority": majority_summary,
        "center_pixel_vs_each_edge_region": center_pixel_comparisons,
        "center_pixel_vs_edge_majority": {
            "comparable_frames": len(exact_majority_values),
            "center_pixel_lower_than_more_than_half_of_valid_edge_regions": int(
                exact_majority_true
            ),
            "ratio": (
                float(exact_majority_true / len(exact_majority_values))
                if exact_majority_values
                else 0.0
            ),
        },
        "rows": rows,
        "elapsed_seconds": time.monotonic() - started,
    }
    output_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Output: {output_path}")
    print(
        "Center pixel median: "
        f"{metadata['center_pixel_stats_m'].get('median_m')} m; "
        "center-cell median: "
        f"{metadata['center_region_stats_m'].get('median_m')} m; "
        "edge-pooled median: "
        f"{metadata['all_edge_region_medians_pooled_stats_m'].get('median_m')} m"
    )
    print(
        "Center cell nearer than edge majority: "
        f"{majority_true}/{len(majority_rows)} "
        f"({majority_summary['ratio']:.1%})"
    )
    print(
        "Center pixel nearer than edge majority: "
        f"{exact_majority_true}/{len(exact_majority_values)} "
        f"({exact_majority_true / len(exact_majority_values):.1%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
