"""Read-only refractive diagnostics and identifiability guardrails."""

from __future__ import annotations

from typing import Any

from ..geometry.refraction import FlatPortModel


def refractive_status(model: FlatPortModel) -> dict[str, Any]:
    """Return status without inventing missing housing/port parameters."""

    return {
        "status": model.status,
        "missing_parameters": list(model.missing_parameters),
        "definitive_depth_allowed": model.status == "known",
        "guardrail": "NOT_IDENTIFIABLE" if model.status != "known" else "measured parameters supplied",
    }
