"""Repeatable, conservative inventory of local output trees."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .classify import classify_path

ROOT_NAMES = ("output", "outputs", "cache")
LARGE_BINARY_SUFFIXES = {".area", ".db", ".mp4", ".npy", ".ply", ".svo", ".svo2"}
DUPLICATE_SUFFIXES = {".db", ".mp4", ".ply"}
LARGE_FILE_BYTES = 100 * 1024 * 1024
_METADATA_NAMES = {"run.json", "metadata.json", "summary.json", "run_summary.json"}


def _utc_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _measure(path: Path) -> tuple[int, int, list[tuple[Path, int]]]:
    if path.is_file():
        size = int(path.stat().st_size)
        return size, 1, [(path, size)]
    total = 0
    count = 0
    files: list[tuple[Path, int]] = []
    for child in path.rglob("*"):
        if child.is_symlink() or not child.is_file():
            continue
        try:
            size = int(child.stat().st_size)
        except OSError:
            continue
        total += size
        count += 1
        files.append((child, size))
    return total, count, files


def _read_metadata(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {}
    candidates: list[Path] = []
    for child in path.rglob("*"):
        if child.is_file() and child.name.lower() in _METADATA_NAMES:
            candidates.append(child)
            if child.name.lower() == "run.json":
                break
        if len(candidates) >= 24:
            break
    candidates.sort(key=lambda item: (item.name.lower() != "run.json", len(item.parts)))
    for candidate in candidates:
        try:
            if candidate.stat().st_size > 2 * 1024 * 1024:
                continue
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def build_inventory(repo_root: Path, *, hash_duplicates: bool = True) -> dict[str, Any]:
    """Scan immediate top-level entries under ``output``, ``outputs``, and ``cache``.

    Hashing is restricted to duplicate candidates: same-size ``.ply``, ``.mp4``
    or ``.db`` files, and same-size files at least 100 MiB. Dense arrays are
    never hashed unless they are explicitly a candidate under that rule.
    """

    repo_root = repo_root.expanduser().resolve()
    entries: list[dict[str, Any]] = []
    candidates: list[tuple[Path, str, int]] = []
    roots: list[dict[str, Any]] = []
    for root_name in ROOT_NAMES:
        root = repo_root / root_name
        root_info = {"name": root_name, "path": root_name, "exists": root.exists()}
        if not root.exists():
            roots.append(root_info)
            continue
        children = sorted(root.iterdir(), key=lambda item: item.name.lower())
        root_info["top_level_entries"] = len(children)
        roots.append(root_info)
        for child in children:
            size, file_count, files = _measure(child)
            metadata = _read_metadata(child)
            contains_large_binary = any(
                suffix in LARGE_BINARY_SUFFIXES and file_size >= LARGE_FILE_BYTES
                for file_path, file_size in files
                for suffix in (file_path.suffix.lower(),)
            )
            contains_summary = any(
                file_path.name.lower() in _METADATA_NAMES - {"run.json"} for file_path, _ in files
            )
            contains_run_json = any(file_path.name.lower() == "run.json" for file_path, _ in files)
            classification = classify_path(
                child,
                root_name=root_name,
                file_count=file_count,
                contains_summary=contains_summary,
                contains_run_json=contains_run_json,
                contains_large_binary=contains_large_binary,
                metadata=metadata,
            )
            entry = {
                "path": _relative(child, repo_root),
                "size_bytes": size,
                "file_count": file_count,
                "mtime": _utc_mtime(child),
                "dataset": classification["dataset"],
                "category": classification["category"],
                "status": classification["status"],
                "purpose": classification["purpose"],
                "provenance_quality": classification["provenance_quality"],
                "contains_summary": contains_summary,
                "contains_run_json": contains_run_json,
                "contains_large_binary": contains_large_binary,
                "suggested_action": classification["suggested_action"],
                "suggested_new_name": classification["suggested_new_name"],
                "reason": classification["reason"],
                "duplicate_group": None,
                "sha256": None,
                "metadata_files": sorted(
                    _relative(file_path, repo_root)
                    for file_path, _ in files
                    if file_path.name.lower() in _METADATA_NAMES
                )[:24],
            }
            entries.append(entry)
            for file_path, file_size in files:
                suffix = file_path.suffix.lower()
                if suffix in DUPLICATE_SUFFIXES or file_size >= LARGE_FILE_BYTES:
                    candidates.append((file_path, suffix, file_size))

    grouped: dict[tuple[str, int], list[tuple[Path, str, int]]] = {}
    for candidate in candidates:
        grouped.setdefault((candidate[1], candidate[2]), []).append(candidate)
    duplicate_groups: list[dict[str, Any]] = []
    for group_number, ((suffix, size), group) in enumerate(sorted(grouped.items()), start=1):
        if len(group) < 2:
            continue
        group_id = f"duplicate-{group_number:04d}"
        files: list[dict[str, Any]] = []
        for file_path, _, file_size in group:
            relative = _relative(file_path, repo_root)
            digest = _sha256(file_path) if hash_duplicates else None
            files.append({"path": relative, "size_bytes": file_size, "sha256": digest})
            for entry in entries:
                if relative == entry["path"] or relative.startswith(entry["path"].rstrip("/") + "/"):
                    current = entry["duplicate_group"] or []
                    if group_id not in current:
                        current.append(group_id)
                    entry["duplicate_group"] = current
                    if entry["sha256"] is None and digest:
                        entry["sha256"] = digest if relative == entry["path"] else None
        duplicate_groups.append(
            {
                "duplicate_group": group_id,
                "suffix": suffix,
                "size_bytes": size,
                "candidate_count": len(group),
                "same_hash": bool(len({item["sha256"] for item in files if item["sha256"]}) == 1),
                "files": files,
            }
        )

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repo_root),
        "roots": roots,
        "entry_count": len(entries),
        "entries": entries,
        "duplicate_groups": duplicate_groups,
        "hash_policy": {
            "enabled": hash_duplicates,
            "large_file_threshold_bytes": LARGE_FILE_BYTES,
            "candidate_suffixes": sorted(DUPLICATE_SUFFIXES),
            "never_hash_all_files": True,
        },
    }


def _csv_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def _markdown(inventory: dict[str, Any]) -> str:
    entries = inventory["entries"]
    total = sum(int(entry["size_bytes"]) for entry in entries)
    lines = [
        "# Outputs inventory",
        "",
        f"Generated: `{inventory['generated_at_utc']}`",
        f"Top-level entries: **{len(entries)}**; measured bytes: **{total:,}**",
        "",
        "Hashing is limited to same-size duplicate candidates; it does not hash the entire output tree.",
        "",
        "| Path | Size bytes | Files | Dataset | Category | Status | Action | New name |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for entry in sorted(entries, key=lambda item: (-int(item["size_bytes"]), item["path"])):
        lines.append(
            "| `{path}` | {size:,} | {files:,} | {dataset} | {category} | {status} | **{action}** | {new} |".format(
                path=entry["path"],
                size=int(entry["size_bytes"]),
                files=int(entry["file_count"]),
                dataset=entry["dataset"] or "",
                category=entry["category"],
                status=entry["status"],
                action=entry["suggested_action"],
                new=entry["suggested_new_name"] or "",
            )
        )
    lines.extend(["", "## Duplicate candidate groups", ""])
    if not inventory["duplicate_groups"]:
        lines.append("No same-size duplicate candidates were found.")
    else:
        for group in inventory["duplicate_groups"]:
            lines.append(
                f"- `{group['duplicate_group']}` ({group['suffix']}, {group['size_bytes']:,} bytes, "
                f"same_hash={group['same_hash']}):"
            )
            for item in group["files"]:
                lines.append(f"  - `{item['path']}` sha256=`{item['sha256'] or 'not computed'}`")
    return "\n".join(lines) + "\n"


def write_inventory_reports(inventory: dict[str, Any], output_dir: Path) -> tuple[Path, Path, Path]:
    """Write JSON, CSV, and Markdown reports and return their paths."""

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "outputs_inventory.json"
    csv_path = output_dir / "outputs_inventory.csv"
    markdown_path = output_dir / "outputs_inventory.md"
    json_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = [
        "path", "size_bytes", "file_count", "mtime", "dataset", "category", "status", "purpose",
        "provenance_quality", "contains_summary", "contains_run_json", "contains_large_binary",
        "suggested_action", "suggested_new_name", "reason", "duplicate_group", "sha256",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in inventory["entries"]:
            writer.writerow({field: _csv_value(entry.get(field)) for field in fields})
    markdown_path.write_text(_markdown(inventory), encoding="utf-8")
    return json_path, csv_path, markdown_path


def _safe_repo_path(repo_root: Path, value: str) -> Path:
    root = repo_root.resolve()
    path = (root / value).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path is outside repository: {value}") from error
    if not relative.parts or relative.parts[0].lower() not in ROOT_NAMES:
        raise ValueError(f"prune path must be under output, outputs, or cache: {value}")
    return path


def prune_outputs(
    repo_root: Path,
    *,
    apply: bool = False,
    category: str | None = None,
    manifest: Path | None = None,
) -> list[str]:
    """Plan or apply only an explicit, narrow cleanup selection.

    Without ``apply`` this function is always a dry-run. Applying requires
    either ``category=empty`` or a JSON manifest containing exact relative
    paths. Non-empty directories are rejected even when listed in a manifest.
    """

    repo_root = repo_root.expanduser().resolve()
    if apply and category is None and manifest is None:
        raise ValueError("--apply requires --category empty or --manifest <json>")
    if category is not None and category != "empty":
        raise ValueError("supported prune category is only: empty")
    inventory = build_inventory(repo_root, hash_duplicates=False)
    selected: list[Path] = []
    if category == "empty":
        selected = [
            _safe_repo_path(repo_root, entry["path"])
            for entry in inventory["entries"]
            if entry["file_count"] == 0 and (repo_root / entry["path"]).is_dir()
        ]
    if manifest is not None:
        value = json.loads(manifest.expanduser().resolve().read_text(encoding="utf-8"))
        values = value.get("paths", value) if isinstance(value, dict) else value
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError("prune manifest must be a JSON list or {\"paths\": [...]}")
        selected.extend(_safe_repo_path(repo_root, item) for item in values)
    unique = sorted({path for path in selected}, key=lambda item: str(item))
    actions = [f"DELETE {path.relative_to(repo_root).as_posix()}" for path in unique]
    if not apply:
        return ["DRY-RUN " + action.removeprefix("DELETE ") for action in actions]
    for path in unique:
        if path.is_dir():
            try:
                next(path.iterdir())
            except StopIteration:
                path.rmdir()
            else:
                raise ValueError(f"refusing to recursively delete non-empty directory: {path}")
        elif path.is_file():
            path.unlink()
        else:
            raise FileNotFoundError(path)
    return actions
