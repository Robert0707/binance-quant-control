from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .convergence import calculate_loss_streak, calculate_profit_factor
from .order_journal import read_closed_trade_reviews


@dataclass(frozen=True, slots=True)
class SideRiskEvaluation:
    allowed: bool
    route_id: str
    side: str
    sample_count: int
    profit_factor: float
    net_pnl_usdt: float
    stop_loss_ratio: float
    avg_r_multiple: float
    loss_streak: int
    threshold_profit_factor: float
    min_samples: int
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "route_id": self.route_id,
            "side": self.side,
            "sample_count": self.sample_count,
            "profit_factor": round(self.profit_factor, 4),
            "net_pnl_usdt": round(self.net_pnl_usdt, 8),
            "stop_loss_ratio": round(self.stop_loss_ratio, 2),
            "avg_r_multiple": round(self.avg_r_multiple, 4),
            "loss_streak": self.loss_streak,
            "threshold_profit_factor": round(self.threshold_profit_factor, 4),
            "min_samples": self.min_samples,
            "reasons": self.reasons,
        }


def _pnl(row: dict[str, Any]) -> float:
    try:
        return float(row.get("realized_pnl_usdt") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _side(row: dict[str, Any]) -> str:
    return str(row.get("side") or "").upper()


def _r_multiple(row: dict[str, Any]) -> float | None:
    value = row.get("realized_r_multiple")
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_route_side_risk(
    *,
    route_id: str,
    side: str,
    min_samples: int = 30,
    min_profit_factor: float = 0.8,
    max_stop_loss_ratio: float = 70.0,
    reviews: list[dict[str, Any]] | None = None,
) -> SideRiskEvaluation:
    normalized_route = str(route_id or "")
    normalized_side = str(side or "").upper()
    rows = [
        row
        for row in (reviews if reviews is not None else read_closed_trade_reviews())
        if str(row.get("route_id") or "") == normalized_route and _side(row) == normalized_side
    ]
    pnls = [_pnl(row) for row in rows]
    profit_factor = calculate_profit_factor(pnls)
    net_pnl = sum(pnls)
    loss_streak = calculate_loss_streak(pnls)
    stop_loss_count = sum(1 for row in rows if str(row.get("exit_reason") or "") == "stop_loss")
    stop_loss_ratio = (stop_loss_count / len(rows) * 100.0) if rows else 0.0
    r_values = [value for row in rows if (value := _r_multiple(row)) is not None]
    avg_r = sum(r_values) / len(r_values) if r_values else 0.0
    reasons: list[str] = []
    enough_samples = len(rows) >= max(int(min_samples), 1)
    if enough_samples and profit_factor < float(min_profit_factor):
        reasons.append(
            f"Route-side historical PF {profit_factor:.4f} is below "
            f"{float(min_profit_factor):.4f} for {normalized_route}/{normalized_side} "
            f"over {len(rows)} reviews."
        )
    if enough_samples and net_pnl < 0.0 and profit_factor < 1.0:
        reasons.append(
            f"Route-side net PnL is still negative ({net_pnl:.4f} USDT) for "
            f"{normalized_route}/{normalized_side}."
        )
    if enough_samples and stop_loss_ratio >= float(max_stop_loss_ratio) and profit_factor < 1.0:
        reasons.append(
            f"Route-side stop-loss ratio is {stop_loss_ratio:.2f}% for "
            f"{normalized_route}/{normalized_side}; entries are being stopped too often."
        )
    return SideRiskEvaluation(
        allowed=not reasons,
        route_id=normalized_route,
        side=normalized_side,
        sample_count=len(rows),
        profit_factor=profit_factor,
        net_pnl_usdt=net_pnl,
        stop_loss_ratio=stop_loss_ratio,
        avg_r_multiple=avg_r,
        loss_streak=loss_streak,
        threshold_profit_factor=float(min_profit_factor),
        min_samples=max(int(min_samples), 1),
        reasons=reasons,
    )
