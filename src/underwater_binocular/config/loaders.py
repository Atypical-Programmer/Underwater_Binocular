"""YAML configuration loading with explicit path resolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .models import DatasetConfig, SgbmConfig, ZedSessionConfig


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one mapping YAML file and fail on non-mapping roots."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def load_dataset_config(path: Path) -> DatasetConfig:
    """Load a dataset config and resolve its profile relative to the config."""

    path = path.expanduser().resolve()
    return DatasetConfig.from_mapping(load_yaml(path), base_dir=path.parent)


def load_zed_config(path: Path) -> ZedSessionConfig:
    """Load a ZED session config."""

    return ZedSessionConfig.from_mapping(load_yaml(path))


def load_sgbm_config(path: Path) -> SgbmConfig:
    """Load an SGBM config."""

    return SgbmConfig.from_mapping(load_yaml(path))


def resolve_local_path(value: str | Path | None, *, base_dir: Path) -> Path | None:
    """Resolve a config/CLI path without inventing a machine-specific default."""

    if value in (None, ""):
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()
