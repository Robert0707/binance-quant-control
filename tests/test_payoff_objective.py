from __future__ import annotations

from binance_quant_control.payoff_objective import (
    PayoffObjectiveTargets,
    payoff_objective_score,
    payoff_objective_sort_key,
    promotion_decision_rank,
)


def test_payoff_objective_prefers_larger_payoff_over_headline_win_rate() -> None:
    targets = PayoffObjectiveTargets(
        min_trades=100,
        min_profit_factor=1.5,
        min_expectancy_r=0.10,
        min_payoff_ratio=1.15,
        min_win_rate=65.0,
        max_stop_loss_ratio=35.0,
    )
    high_win_low_payoff = {
        "trade_count": 120,
        "win_rate": 80.0,
        "stop_loss_ratio": 20.0,
        "profit_factor": 1.0252,
        "expectancy_r": 0.002,
        "payoff_ratio": 0.2563,
        "expectancy_edge_points": 0.4,
        "total_return_pct": 0.1,
    }
    lower_win_good_payoff = {
        "trade_count": 120,
        "win_rate": 66.0,
        "stop_loss_ratio": 34.0,
        "profit_factor": 1.7,
        "expectancy_r": 0.18,
        "payoff_ratio": 1.65,
        "expectancy_edge_points": 11.0,
        "total_return_pct": 5.0,
    }

    assert payoff_objective_score(lower_win_good_payoff, targets=targets) > payoff_objective_score(
        high_win_low_payoff,
        targets=targets,
    )
    assert payoff_objective_sort_key(lower_win_good_payoff, targets=targets) > payoff_objective_sort_key(
        high_win_low_payoff,
        targets=targets,
    )


def test_promotion_decision_rank_is_explicit_not_lexicographic() -> None:
    assert promotion_decision_rank("reject") < promotion_decision_rank("watchlist")
    assert promotion_decision_rank("watchlist") < promotion_decision_rank("promote")
    assert promotion_decision_rank("promote") < promotion_decision_rank("elite_candidate")


def test_payoff_objective_penalizes_non_robust_machine_candidates() -> None:
    targets = PayoffObjectiveTargets(
        min_trades=100,
        min_profit_factor=1.25,
        min_expectancy_r=0.05,
        min_payoff_ratio=1.2,
        min_win_rate=45.0,
        max_stop_loss_ratio=55.0,
        min_slippage_resilience=0.7,
        min_walk_forward_stability=0.6,
    )
    fragile = {
        "trade_count": 120,
        "win_rate": 70.0,
        "stop_loss_ratio": 30.0,
        "profit_factor": 1.4,
        "expectancy_r": 0.08,
        "payoff_ratio": 1.3,
        "slippage_resilience": 0.0,
        "walk_forward_stability": 0.0,
    }
    robust = fragile | {
        "slippage_resilience": 0.85,
        "walk_forward_stability": 0.75,
    }

    assert payoff_objective_score(robust, targets=targets) > payoff_objective_score(fragile, targets=targets)
