from __future__ import annotations

from typing import Any


def trace_step(
    layer: str,
    *,
    allowed: bool,
    reasons: list[str] | tuple[str, ...] | None = None,
    warnings: list[str] | tuple[str, ...] | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "layer": layer,
        "allowed": bool(allowed),
        "reasons": list(reasons or []),
        "warnings": list(warnings or []),
        "data": data or {},
    }
