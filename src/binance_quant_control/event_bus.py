from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class TradingEvent:
    event_type: str
    payload: dict[str, Any]
    source: str = "hermes-ai-trader"
    timestamp: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EventHandler = Callable[[TradingEvent], None]


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._events: list[TradingEvent] = []

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers[event_type].append(handler)

    def publish(self, event: TradingEvent) -> None:
        self._events.append(event)
        for handler in self._handlers.get(event.event_type, []):
            handler(event)
        for handler in self._handlers.get("*", []):
            handler(event)

    def history(self, *, limit: int = 0) -> list[TradingEvent]:
        return self._events[-limit:] if limit > 0 else list(self._events)


@dataclass(frozen=True, slots=True)
class PluginSpec:
    name: str
    stage: str
    enabled: bool
    module: str
    description: str = ""
    input_events: tuple[str, ...] = ()
    output_events: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_plugin_lifecycle() -> list[PluginSpec]:
    return [
        PluginSpec(
            "universe",
            "pre_signal",
            True,
            "candidate_universe",
            "coin discovery",
            (),
            ("universe.ready",),
        ),
        PluginSpec(
            "features",
            "pre_signal",
            True,
            "feature_registry",
            "feature manifest",
            ("universe.ready",),
            ("features.ready",),
        ),
        PluginSpec(
            "alpha",
            "signal",
            True,
            "alpha_families",
            "strategy family scoring",
            ("features.ready",),
            ("signal.created",),
        ),
        PluginSpec(
            "committee",
            "review",
            True,
            "trading_committee",
            "structured debate",
            ("signal.created",),
            ("committee.reviewed",),
        ),
        PluginSpec(
            "portfolio",
            "risk",
            True,
            "portfolio_construction",
            "risk targets",
            ("signal.created",),
            ("portfolio.targeted",),
        ),
        PluginSpec(
            "risk",
            "risk",
            True,
            "professional_entry_gate",
            "pre-trade denial",
            ("portfolio.targeted", "committee.reviewed"),
            ("gate.evaluated",),
        ),
        PluginSpec(
            "execution",
            "execution",
            False,
            "live_execution",
            "disabled until gates pass",
            ("gate.evaluated",),
            ("order.intent.created", "order.submitted"),
        ),
        PluginSpec(
            "monitoring",
            "post_trade",
            True,
            "skipped_signal_journal",
            "denial ledger",
            ("gate.evaluated", "order.submitted", "fill.received"),
            ("monitoring.recorded",),
        ),
    ]
