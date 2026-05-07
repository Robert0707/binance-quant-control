from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import STATE_DIR, ensure_runtime_dirs

SKIPPED_SIGNAL_JOURNAL = STATE_DIR / "journals" / "skipped-signals.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class SkippedSignalRecord:
    timestamp: str
    symbol: str
    side: str
    route_id: str
    strategy_family: str
    gate: str
    blockers: list[str]
    signal_score: float | None = None
    expectancy_r: float | None = None
    payoff_ratio: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def append_skipped_signal(
    *,
    symbol: str,
    side: str,
    route_id: str,
    strategy_family: str,
    gate: str,
    blockers: list[str],
    signal_score: float | None = None,
    expectancy_r: float | None = None,
    payoff_ratio: float | None = None,
    metadata: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> Path:
    ensure_runtime_dirs()
    journal_path = Path(path) if path else SKIPPED_SIGNAL_JOURNAL
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    record = SkippedSignalRecord(
        timestamp=_utc_now_iso(),
        symbol=symbol.upper(),
        side=side.upper(),
        route_id=route_id,
        strategy_family=strategy_family,
        gate=gate,
        blockers=blockers,
        signal_score=signal_score,
        expectancy_r=expectancy_r,
        payoff_ratio=payoff_ratio,
        metadata=metadata or {},
    )
    with journal_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return journal_path


def read_skipped_signals(path: str | Path | None = None, *, limit: int = 0) -> list[dict[str, Any]]:
    journal_path = Path(path) if path else SKIPPED_SIGNAL_JOURNAL
    if not journal_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:] if limit > 0 else rows
