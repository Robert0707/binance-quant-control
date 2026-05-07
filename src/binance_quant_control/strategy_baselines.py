from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import CONFIG_DIR

DEFAULT_BASELINES_PATH = CONFIG_DIR / "official-strategy-baselines.yaml"


@dataclass(frozen=True, slots=True)
class StrategyBaseline:
    baseline_id: str
    title: str
    applies_to_routes: tuple[str, ...]
    strategy_family: str
    thesis: str
    local_template: Path
    source_notes: tuple[str, ...]
    official_sources: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_id": self.baseline_id,
            "title": self.title,
            "applies_to_routes": list(self.applies_to_routes),
            "strategy_family": self.strategy_family,
            "thesis": self.thesis,
            "local_template": str(self.local_template),
            "source_notes": list(self.source_notes),
            "official_sources": [dict(item) for item in self.official_sources],
        }


def resolve_baselines_path(path: str | Path | None = None) -> Path:
    candidate = Path(path or DEFAULT_BASELINES_PATH).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()
    return (CONFIG_DIR / candidate).resolve()


def _resolve_project_path(raw: str) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()
    return (CONFIG_DIR / candidate).resolve()


def load_strategy_baselines(path: str | Path | None = None) -> dict[str, StrategyBaseline]:
    config_path = resolve_baselines_path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw_baselines = payload.get("baselines") or {}
    baselines: dict[str, StrategyBaseline] = {}
    for baseline_id, raw in raw_baselines.items():
        if not isinstance(raw, dict):
            continue
        baselines[baseline_id] = StrategyBaseline(
            baseline_id=str(baseline_id),
            title=str(raw.get("title") or baseline_id),
            applies_to_routes=tuple(str(item) for item in (raw.get("applies_to_routes") or [])),
            strategy_family=str(raw.get("strategy_family") or "unknown"),
            thesis=str(raw.get("thesis") or ""),
            local_template=_resolve_project_path(str(raw.get("local_template") or "")),
            source_notes=tuple(str(item) for item in (raw.get("source_notes") or [])),
            official_sources=tuple(dict(item) for item in (raw.get("official_sources") or []) if isinstance(item, dict)),
        )
    return baselines


def baseline_for_route(route_id: str, path: str | Path | None = None) -> StrategyBaseline | None:
    for baseline in load_strategy_baselines(path).values():
        if route_id in baseline.applies_to_routes:
            return baseline
    return None
