from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PayoffObjectiveTargets:
    min_trades: int = 100
    min_profit_factor: float = 1.5
    min_expectancy_r: float = 0.10
    min_payoff_ratio: float = 1.15
    min_win_rate: float = 65.0
    max_stop_loss_ratio: float = 35.0
    min_slippage_resilience: float = 0.0
    min_walk_forward_stability: float = 0.0


PROMOTION_DECISION_RANK = {
    "reject": 0,
    "watchlist": 1,
    "promote": 2,
    "elite_candidate": 3,
}


def metric_float(value: Any, default: float = 0.0) -> float:
    if value in {"inf", "+inf"}:
        return 9999.0
    if value == "-inf":
        return -9999.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    if math.isinf(number):
        return 9999.0 if number > 0.0 else -9999.0
    return number


def promotion_decision_rank(value: Any) -> int:
    return PROMOTION_DECISION_RANK.get(str(value or "").strip().lower(), 0)


def _bounded_ratio(value: float, target: float, *, cap: float = 3.0) -> float:
    floor = max(float(target), 1e-9)
    return min(max(float(value) / floor, -cap), cap)


def _sample_ratio(trade_count: float, min_trades: int) -> float:
    if min_trades <= 0:
        return min(max(trade_count / 100.0, 0.0), 1.0)
    return min(max(trade_count / float(min_trades), 0.0), 1.0)


def payoff_objective_score(
    metrics: Mapping[str, Any],
    *,
    targets: PayoffObjectiveTargets | None = None,
    min_trades: int | None = None,
) -> float:
    """Score candidates by fixed-risk edge before headline win rate.

    The score is intentionally bounded so a tiny-sample ``PF=inf`` row cannot
    dominate mature but merely good cohorts.
    """

    target = targets or PayoffObjectiveTargets()
    trade_floor = int(min_trades if min_trades is not None else target.min_trades)
    trade_count = metric_float(metrics.get("trade_count"))
    expectancy_r = metric_float(metrics.get("expectancy_r"))
    payoff_ratio = min(metric_float(metrics.get("payoff_ratio")), 5.0)
    profit_factor = min(metric_float(metrics.get("profit_factor")), 5.0)
    win_rate = metric_float(metrics.get("win_rate"))
    stop_loss_ratio = metric_float(metrics.get("stop_loss_ratio"))
    edge_points = metric_float(metrics.get("expectancy_edge_points"))
    slippage_resilience = metric_float(metrics.get("slippage_resilience"))
    walk_forward_stability = metric_float(metrics.get("walk_forward_stability"))
    total_return = metric_float(metrics.get("total_return_pct"))
    drawdown = max(metric_float(metrics.get("max_drawdown_pct")), 1.0)
    return_over_drawdown = metric_float(metrics.get("return_over_drawdown"), total_return / drawdown)
    sample = _sample_ratio(trade_count, trade_floor)
    stop_quality = max((target.max_stop_loss_ratio - stop_loss_ratio) / max(target.max_stop_loss_ratio, 1.0), -2.0)
    score = (
        sample * 18.0
        + _bounded_ratio(expectancy_r, target.min_expectancy_r) * 28.0
        + _bounded_ratio(payoff_ratio, target.min_payoff_ratio) * 18.0
        + _bounded_ratio(profit_factor, target.min_profit_factor) * 14.0
        + _bounded_ratio(win_rate, target.min_win_rate) * 6.0
        + stop_quality * 6.0
        + min(max(edge_points / 10.0, -2.0), 3.0) * 4.0
        + _bounded_ratio(slippage_resilience, target.min_slippage_resilience) * 6.0
        + _bounded_ratio(walk_forward_stability, target.min_walk_forward_stability) * 6.0
        + min(max(return_over_drawdown, -2.0), 5.0) * 3.0
    )
    if trade_floor > 0 and trade_count < trade_floor:
        score *= max(0.05, sample)
    if expectancy_r <= 0.0:
        score *= 0.35
    if payoff_ratio < target.min_payoff_ratio:
        score *= 0.60
    if target.min_slippage_resilience > 0.0 and slippage_resilience < target.min_slippage_resilience:
        score *= 0.70
    if target.min_walk_forward_stability > 0.0 and walk_forward_stability < target.min_walk_forward_stability:
        score *= 0.70
    return round(score, 6)


def payoff_objective_sort_key(
    metrics: Mapping[str, Any],
    *,
    targets: PayoffObjectiveTargets | None = None,
    min_trades: int | None = None,
) -> tuple[float, ...]:
    target = targets or PayoffObjectiveTargets()
    trade_floor = int(min_trades if min_trades is not None else target.min_trades)
    trade_count = metric_float(metrics.get("trade_count"))
    return (
        payoff_objective_score(metrics, targets=target, min_trades=trade_floor),
        _sample_ratio(trade_count, trade_floor),
        metric_float(metrics.get("expectancy_r")),
        min(metric_float(metrics.get("payoff_ratio")), 5.0),
        min(metric_float(metrics.get("profit_factor")), 5.0),
        metric_float(metrics.get("slippage_resilience")),
        metric_float(metrics.get("walk_forward_stability")),
        metric_float(metrics.get("expectancy_edge_points")),
        metric_float(metrics.get("total_return_pct")),
        -metric_float(metrics.get("stop_loss_ratio")),
        trade_count,
    )
