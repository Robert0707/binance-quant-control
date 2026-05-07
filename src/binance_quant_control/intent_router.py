from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import CONFIG_DIR

DEFAULT_INTENT_CONFIG_PATH = CONFIG_DIR / "operator-intent.default.yaml"


@dataclass(frozen=True, slots=True)
class OperatorIntent:
    intent_id: str
    score: int
    summary: str
    operator_prompt: str
    actions: tuple[str, ...]
    resources: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "intent_id": self.intent_id,
            "score": self.score,
            "summary": self.summary,
            "operator_prompt": self.operator_prompt,
            "actions": list(self.actions),
            "resources": list(self.resources),
        }


def resolve_intent_config_path(path: str | Path | None = None) -> Path:
    candidate = Path(path or DEFAULT_INTENT_CONFIG_PATH).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (CONFIG_DIR / candidate).resolve()


def resolve_operator_intent(message: str, path: str | Path | None = None) -> OperatorIntent:
    config_path = resolve_intent_config_path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    intents = payload.get("intents") or []
    if not isinstance(intents, list):
        raise ValueError(f"Intent config must contain a list of intents: {config_path}")

    normalized = message.strip().lower()
    best: OperatorIntent | None = None
    for raw in intents:
        if not isinstance(raw, dict):
            continue
        phrases = [str(item).lower() for item in (raw.get("phrases") or [])]
        hits = sum(1 for phrase in phrases if phrase and phrase in normalized)
        if hits <= 0:
            continue
        intent = OperatorIntent(
            intent_id=str(raw.get("id") or "unknown"),
            score=hits,
            summary=str(raw.get("summary") or ""),
            operator_prompt=str(raw.get("operator_prompt") or ""),
            actions=tuple(str(item) for item in (raw.get("actions") or [])),
            resources=tuple(str(item) for item in (raw.get("resources") or [])),
        )
        if best is None or intent.score > best.score:
            best = intent

    if best is not None:
        return best

    fallback = payload.get("fallback") or {}
    return OperatorIntent(
        intent_id=str(fallback.get("id") or "fallback"),
        score=0,
        summary=str(fallback.get("summary") or "No strong operator intent matched."),
        operator_prompt=str(fallback.get("operator_prompt") or ""),
        actions=tuple(str(item) for item in (fallback.get("actions") or [])),
        resources=tuple(str(item) for item in (fallback.get("resources") or [])),
    )
