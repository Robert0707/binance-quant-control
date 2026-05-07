from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class TradingSignal:
    signal_id: str
    symbol: str
    side: str
    interval: str
    strategy_family: str
    route_id: str
    source: str = "hermes-ai-trader"
    status: str = "draft"
    signal_score: float = 0.0
    expectancy_r: float = 0.0
    payoff_ratio: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    stop_loss_ratio: float = 100.0
    trade_count: int = 0
    blockers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)

    @property
    def approved(self) -> bool:
        return self.status == "approved" and not self.blockers

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_signal_id(symbol: str, interval: str, strategy_family: str, side: str) -> str:
    return ":".join(
        [
            symbol.upper(),
            interval,
            strategy_family,
            side.upper(),
        ]
    )
