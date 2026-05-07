from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .asset_routing import RouteValidationSpec


@dataclass(frozen=True, slots=True)
class ConvergenceMetrics:
    trade_count: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    loss_streak: int
    expectancy_r: float = 0.0
    payoff_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_count": self.trade_count,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "loss_streak": self.loss_streak,
            "expectancy_r": round(self.expectancy_r, 4),
            "payoff_ratio": round(self.payoff_ratio, 4),
        }


def build_cohort_id(*, asset_class: str, strategy_profile: str, market: str, interval: str) -> str:
    return ":".join(
        [
            asset_class or "unknown",
            strategy_profile or "unprofiled",
            market or "futures",
            interval or "unknown",
        ]
    )


def calculate_profit_factor(realized_pnls: list[float]) -> float:
    gross_profit = sum(item for item in realized_pnls if item > 0.0)
    gross_loss = abs(sum(item for item in realized_pnls if item < 0.0))
    if gross_loss <= 0.0:
        return float("inf") if gross_profit > 0.0 else 0.0
    return gross_profit / gross_loss


def calculate_expectancy_stats(realized_r_multiples: list[float]) -> dict[str, float]:
    """Return fixed-risk expectancy metrics from realized R multiples."""

    values = [float(item) for item in realized_r_multiples]
    trade_count = len(values)
    if trade_count <= 0:
        return {
            "expectancy_r": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "payoff_ratio": 0.0,
            "break_even_win_rate": 0.0,
            "expectancy_edge_points": 0.0,
        }
    wins = [item for item in values if item > 0.0]
    losses = [abs(item) for item in values if item < 0.0]
    win_rate = len(wins) / trade_count
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    expectancy = sum(values) / trade_count
    if avg_loss <= 0.0:
        payoff_ratio = float("inf") if avg_win > 0.0 else 0.0
        break_even_win_rate = 0.0 if avg_win > 0.0 else 1.0
    else:
        payoff_ratio = avg_win / avg_loss
        break_even_win_rate = 1.0 / (1.0 + payoff_ratio) if payoff_ratio > 0.0 else 1.0
    edge_points = (win_rate - break_even_win_rate) * 100.0
    return {
        "expectancy_r": round(expectancy, 4),
        "avg_win_r": round(avg_win, 4),
        "avg_loss_r": round(avg_loss, 4),
        "payoff_ratio": round(payoff_ratio, 4) if payoff_ratio != float("inf") else float("inf"),
        "break_even_win_rate": round(break_even_win_rate * 100.0, 4),
        "expectancy_edge_points": round(edge_points, 4),
    }


def calculate_loss_streak(realized_pnls: list[float]) -> int:
    streak = 0
    for pnl in reversed(realized_pnls):
        if pnl < 0.0:
            streak += 1
        else:
            break
    return streak


def calculate_max_drawdown_pct(realized_pnls: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for pnl in realized_pnls:
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, ((peak - equity) / peak) * 100.0)
    return max_drawdown


def evaluate_convergence(metrics: ConvergenceMetrics, spec: RouteValidationSpec) -> dict[str, Any]:
    screening_passed = (
        metrics.trade_count >= spec.screening_min_trades
        and metrics.win_rate >= spec.screening_min_win_rate
        and metrics.profit_factor >= spec.screening_min_profit_factor
        and metrics.expectancy_r >= spec.screening_min_expectancy_r
        and metrics.payoff_ratio >= spec.screening_min_payoff_ratio
    )
    validation_ready = metrics.trade_count >= spec.validation_min_simulated_trades
    validation_passed = (
        validation_ready
        and metrics.win_rate >= spec.validation_min_win_rate
        and metrics.profit_factor >= spec.validation_min_profit_factor
        and metrics.expectancy_r >= spec.validation_min_expectancy_r
        and metrics.payoff_ratio >= spec.validation_min_payoff_ratio
        and metrics.max_drawdown_pct <= spec.max_drawdown_pct
        and metrics.loss_streak <= spec.max_loss_streak
    )
    elite_passed = (
        spec.elite_enabled
        and metrics.trade_count >= spec.elite_min_trades
        and metrics.win_rate >= spec.elite_min_win_rate
        and metrics.profit_factor >= spec.elite_min_profit_factor
        and metrics.max_drawdown_pct <= spec.max_drawdown_pct
        and metrics.loss_streak <= spec.max_loss_streak
    )
    if elite_passed:
        promotion_decision = "elite_candidate"
    elif validation_passed:
        promotion_decision = "promote"
    elif screening_passed:
        promotion_decision = "watchlist"
    else:
        promotion_decision = "reject"
    return {
        "screening_status": "passed" if screening_passed else "failed",
        "validation_status": (
            "passed" if validation_passed else "insufficient_sample" if screening_passed and not validation_ready else "failed"
        ),
        "elite_status": "elite_candidate" if elite_passed else "not_elite",
        "promotion_decision": promotion_decision,
        "thresholds": spec.to_dict(),
    }
