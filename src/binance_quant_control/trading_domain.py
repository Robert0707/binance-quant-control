from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

TradeSide = str
OrderKind = str
PositionSide = str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class OrderIntent:
    symbol: str
    side: TradeSide
    order_kind: OrderKind
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    reduce_only: bool = False
    time_in_force: str = "GTC"
    client_order_id: str | None = None
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def notional_usdt(self) -> float | None:
        if self.price is None:
            return None
        return round(max(self.quantity, 0.0) * max(self.price, 0.0), 8)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OrderSnapshot:
    order_id: str | int | None
    client_order_id: str | None
    symbol: str
    side: TradeSide
    order_kind: OrderKind
    quantity: float
    status: str
    price: float | None = None
    stop_price: float | None = None
    filled_quantity: float = 0.0
    avg_fill_price: float | None = None
    reduce_only: bool = False
    time_in_force: str = "GTC"
    source: str = "unknown"
    timestamp: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_quantity(self) -> float:
        return round(max(self.quantity - self.filled_quantity, 0.0), 8)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class Broker(Protocol):
    name: str

    def submit_order(self, order: OrderIntent) -> dict[str, Any]:
        """Submit an order intent through a concrete broker/exchange adapter."""

    def orders(self) -> list[OrderSnapshot]:
        """Return recently known order snapshots."""

    def positions(self) -> list[PositionSnapshot]:
        """Return current position snapshots."""


@runtime_checkable
class DataSource(Protocol):
    name: str

    def latest_features(self, symbol: str, interval: str) -> dict[str, Any]:
        """Return live-safe feature values for a symbol/interval pair."""


@dataclass(frozen=True, slots=True)
class FillEvent:
    symbol: str
    side: TradeSide
    quantity: float
    price: float
    fee_usdt: float = 0.0
    order_id: str | None = None
    liquidity: str = "unknown"
    timestamp: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def gross_notional_usdt(self) -> float:
        return round(max(self.quantity, 0.0) * max(self.price, 0.0), 8)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    mark_price: float
    unrealized_pnl_usdt: float = 0.0
    open_risk_pct: float = 0.0
    route_id: str = ""
    correlation_group: str = ""
    timestamp: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.side != "FLAT" and abs(self.quantity) > 0.0

    @property
    def gross_notional_usdt(self) -> float:
        return round(abs(self.quantity) * max(self.mark_price, 0.0), 8)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
