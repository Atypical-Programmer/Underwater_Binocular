"""Tests for conservative output inventory and prune policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from underwater_binocular.outputs.inventory import (
    build_inventory,
    prune_outputs,
    write_inventory_reports,
)


def test_inventory_emits_required_fields_and_unknown_manual_review(tmp_path: Path) -> None:
    output = tmp_path / "output"
    known = output / "20260802_150233_tracking_gen1_neural"
    known.mkdir(parents=True)
    (known / "summary.json").write_text(json.dumps({"dataset": "20260802_150233"}), encoding="utf-8")
    unknown = output / "new_unclassified_run"
    unknown.mkdir()
    (unknown / "payload.txt").write_text("payload", encoding="utf-8")

    inventory = build_inventory(tmp_path, hash_duplicates=False)

    required = {
        "path", "size_bytes", "file_count", "mtime", "dataset", "category", "status", "purpose",
        "provenance_quality", "contains_summary", "contains_run_json", "contains_large_binary",
        "suggested_action", "suggested_new_name", "reason",
    }
    assert required <= set(inventory["entries"][0])
    unknown_entry = next(item for item in inventory["entries"] if item["path"].endswith("new_unclassified_run"))
    assert unknown_entry["suggested_action"] == "MANUAL_REVIEW"


def test_inventory_hashes_only_same_size_duplicate_candidates(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    first = output / "first.ply"
    second = output / "second.ply"
    first.write_bytes(b"same-payload")
    second.write_bytes(b"same-payload")

    inventory = build_inventory(tmp_path)

    assert len(inventory["duplicate_groups"]) == 1
    group = inventory["duplicate_groups"][0]
    assert group["same_hash"] is True
    hashes = {item["sha256"] for item in group["files"]}
    assert len(hashes) == 1
    assert next(iter(hashes))


def test_inventory_reports_and_prune_are_safe_by_default(tmp_path: Path) -> None:
    (tmp_path / "output" / "empty_run").mkdir(parents=True)
    inventory = build_inventory(tmp_path, hash_duplicates=False)
    paths = write_inventory_reports(inventory, tmp_path)
    assert all(path.is_file() for path in paths)

    dry_run = prune_outputs(tmp_path, category="empty")
    assert dry_run == ["DRY-RUN output/empty_run"]
    assert (tmp_path / "output" / "empty_run").is_dir()

    applied = prune_outputs(tmp_path, apply=True, category="empty")
    assert applied == ["DELETE output/empty_run"]
    assert not (tmp_path / "output" / "empty_run").exists()


def test_manifest_prune_is_limited_to_output_roots(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"paths": ["README.md"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="under output, outputs, or cache"):
        prune_outputs(tmp_path, apply=True, manifest=manifest)
