"""Conservative classification rules for legacy and current output runs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

ACTIONS = ("KEEP", "MIGRATE", "ARCHIVE", "DELETE", "MANUAL_REVIEW")
_DATASET_RE = re.compile(r"(\d{8}_\d{6})")


def _text(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def dataset_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        match = _DATASET_RE.search(part)
        if match:
            return match.group(1)
    return None


def _metadata_value(metadata: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def classify_path(
    path: Path,
    *,
    root_name: str,
    file_count: int,
    contains_summary: bool,
    contains_run_json: bool,
    contains_large_binary: bool,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a conservative retention classification.

    Only well-known historical names receive automatic actions. Unknown paths
    deliberately become ``MANUAL_REVIEW`` so inventory cannot silently bless a
    destructive cleanup.
    """

    metadata = metadata or {}
    name = path.name.lower()
    full = path.as_posix().lower()
    dataset = _metadata_value(metadata, "dataset", "dataset_id") or dataset_from_path(path)
    category = _metadata_value(metadata, "category")
    purpose = _metadata_value(metadata, "purpose")
    status = _metadata_value(metadata, "result_status", "status")
    provenance = "unknown"
    if contains_run_json and metadata:
        provenance = "run_metadata"
    elif contains_summary or metadata:
        provenance = "summary_or_metadata"
    elif file_count == 0:
        provenance = "empty"

    action = "MANUAL_REVIEW"
    suggested_name: str | None = None
    reason = "Name or provenance is not covered by a conservative rule."

    if file_count == 0 and path.is_dir():
        action = "DELETE"
        reason = "Known empty output directory; delete only after manifest and recheck."
    elif root_name == "cache":
        action = "DELETE"
        category = category or "cache"
        purpose = purpose or "disposable"
        reason = "Cache is reproducible and disposable; retain only if a current run references it."
    elif name == "20260802_150233_custom_depth_histogram":
        action = "MIGRATE"
        category = category or "depth"
        purpose = purpose or "reference"
        suggested_name = "depth__custom__zed_sdk__full35855__reference"
        reason = "Authoritative custom SDK depth histogram; extract compact reference before archiving."
    elif name == "20260802_150233_sgbm_depth_histogram_1000":
        action = "MIGRATE"
        category = category or "depth"
        purpose = purpose or "reference"
        suggested_name = "depth__custom__sgbm_halfres__1000f__reference"
        reason = "Authoritative 1000-frame SGBM cross-check; retain compact statistics and provenance."
    elif "calibration_comparison" in name:
        action = "MIGRATE"
        category = category or "calibration"
        purpose = purpose or "reference"
        suggested_name = "calibration__comparison__20260802_150233__reference"
        reason = "Calibration comparison is a compact validation reference, not a disposable run."
    elif name == "20260802_150233_tracking_gen1_neural":
        action = "MIGRATE"
        category = category or "tracking"
        purpose = purpose or "baseline"
        suggested_name = "tracking__native__gen1_neural__full35855__baseline"
        status = status or "complete"
        reason = "Historical native GEN_1 full replay is the legacy native baseline."
    elif name == "20260802_150233_tracking_ab_performance":
        action = "ARCHIVE"
        category = category or "tracking"
        purpose = purpose or "ablation"
        suggested_name = "tracking__native__gen1_performance__full35855__ablation"
        reason = "Performance-mode replay is useful as an ablation, not the primary baseline."
    elif name == "20260802_150233_tracking_custom_gen1_neural":
        action = "ARCHIVE"
        category = category or "tracking"
        purpose = purpose or "diagnostic"
        suggested_name = "tracking__custom__gen1_neural__full35855__ANOMALOUS"
        status = status or "anomalous"
        reason = "Custom-calibration trajectory is a large anomalous historical comparison."
    elif name == "20260802_150233_tracking_custom_gen1_smoke":
        action = "ARCHIVE"
        category = category or "tracking"
        purpose = purpose or "smoke"
        suggested_name = "tracking__custom__gen1_neural__smoke__diagnostic"
        reason = "Small custom smoke replay is retained only as diagnostic evidence."
    elif name == "20260802_150233_custom_sdk_pointcloud_full":
        action = "MIGRATE"
        category = category or "pointcloud"
        purpose = purpose or "reference"
        suggested_name = "pointcloud__custom__zed_sdk__1000mapped__reference"
        reason = "Authoritative custom SDK point cloud with pose and mapping metadata."
    elif name in {"20260802_150233_custom_gen1_full", "20260802_150233_custom_gen1_sample10pct"}:
        action = "ARCHIVE"
        category = category or "pointcloud"
        purpose = purpose or "diagnostic"
        suggested_name = (
            "pointcloud__custom__gen1__full__ANOMALOUS"
            if name.endswith("_full")
            else "pointcloud__custom__gen1__sample10pct__ANOMALOUS"
        )
        status = status or "anomalous"
        reason = "Custom GEN_1 point cloud is retained as an anomalous comparison summary."
    elif name in {"20260802_150233_pointcloud1000", "20260802_150233_pointcloud1000_gen1"}:
        action = "DELETE"
        category = category or "pointcloud"
        purpose = purpose or "diagnostic"
        reason = "Low-provenance duplicate candidate; delete only after candidate hashes and summary are recorded."
    elif name == "20260802_150233_orbslam3_resampled":
        action = "MIGRATE"
        category = category or "slam"
        purpose = purpose or "baseline"
        suggested_name = "slam__orbslam3__diagnostic_full__35855f__baseline"
        reason = "ORB-SLAM3 diagnostic full replay contains the most complete compact trajectory evidence."
    elif name == "20260802_150233_orbslam3_stereo":
        action = "MIGRATE"
        category = category or "slam"
        purpose = purpose or "baseline"
        suggested_name = "slam__orbslam3__stereo_halfres__35855f__baseline"
        reason = "ORB-SLAM3 stereo replay is retained as the stereo baseline."
    elif name == "20260802_150233_orbslam3_uniform10pct_custom":
        action = "ARCHIVE"
        category = category or "slam"
        purpose = purpose or "diagnostic"
        suggested_name = "slam__orbslam3__sampled_custom__diagnostic"
        reason = "Sampled custom ORB-SLAM3 result is a large derivative diagnostic."
    elif name == "20260802_150233_sample1000":
        action = "MANUAL_REVIEW"
        category = category or "sfm"
        purpose = purpose or "ablation"
        suggested_name = "sfm__aliked_adalam__sample1000__ablation"
        reason = "Mixed legacy sample bundle must be split by sub-run before any deletion."
    elif name == "20260802_150233_refractive_depth_audit_v2_smoke":
        action = "ARCHIVE"
        category = category or "diagnostics"
        purpose = purpose or "smoke"
        reason = "Small refractive audit smoke evidence can be kept in the diagnostics archive."
    elif name.startswith("20260802_150233_strict_calibration_compare"):
        action = "MIGRATE"
        category = category or "calibration"
        purpose = purpose or "reference"
        reason = "Strict calibration comparison contains compact summaries plus bulky frame samples."
    elif name.endswith(".mp4") or name.endswith(".mp4.json"):
        action = "MANUAL_REVIEW"
        category = category or "diagnostics"
        purpose = purpose or "diagnostic"
        reason = "Video retention depends on metadata, SVO availability, and unique scientific role."
    elif "sgbm" in full or "depth" in full:
        category = category or "depth"
    elif "tracking" in full:
        category = category or "tracking"
    elif "orbslam" in full or "slam" in full:
        category = category or "slam"
    elif "sfm" in full or "colmap" in full:
        category = category or "sfm"

    return {
        "dataset": dataset,
        "category": category or "unknown",
        "status": status or ("empty" if file_count == 0 else "unknown"),
        "purpose": purpose or "unknown",
        "provenance_quality": provenance,
        "suggested_action": action if action in ACTIONS else "MANUAL_REVIEW",
        "suggested_new_name": suggested_name,
        "reason": reason,
    }
