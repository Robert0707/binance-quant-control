from binance_quant_control.asset_routing import resolve_symbol_route
from binance_quant_control.convergence import (
    ConvergenceMetrics,
    build_cohort_id,
    evaluate_convergence,
)


def test_build_cohort_id_is_stable() -> None:
    cohort_id = build_cohort_id(
        asset_class="meme_high_beta",
        strategy_profile="meme-momentum",
        market="futures",
        interval="1h",
    )
    assert cohort_id == "meme_high_beta:meme-momentum:futures:1h"


def test_evaluate_convergence_distinguishes_screening_validation_and_elite() -> None:
    route = resolve_symbol_route("BTCUSDT")
    short_sample = evaluate_convergence(
        ConvergenceMetrics(
            trade_count=99,
            win_rate=99.0,
            profit_factor=5.0,
            max_drawdown_pct=8.0,
            loss_streak=1,
        ),
        route.validation,
    )
    screening_only = evaluate_convergence(
        ConvergenceMetrics(
            trade_count=100,
            win_rate=82.0,
            profit_factor=1.25,
            max_drawdown_pct=8.0,
            loss_streak=1,
            expectancy_r=0.06,
            payoff_ratio=1.05,
        ),
        route.validation,
    )
    validated = evaluate_convergence(
        ConvergenceMetrics(
            trade_count=100,
            win_rate=89.0,
            profit_factor=1.6,
            max_drawdown_pct=7.0,
            loss_streak=1,
            expectancy_r=0.12,
            payoff_ratio=1.2,
        ),
        route.validation,
    )
    elite = evaluate_convergence(
        ConvergenceMetrics(
            trade_count=100,
            win_rate=91.0,
            profit_factor=1.7,
            max_drawdown_pct=6.0,
            loss_streak=1,
            expectancy_r=0.18,
            payoff_ratio=1.4,
        ),
        route.validation,
    )

    assert short_sample["screening_status"] == "failed"
    assert short_sample["thresholds"]["screening_min_trades"] == 100
    assert short_sample["promotion_decision"] == "reject"
    assert screening_only["screening_status"] == "passed"
    assert screening_only["validation_status"] == "failed"
    assert screening_only["promotion_decision"] == "watchlist"
    assert validated["validation_status"] == "passed"
    assert validated["promotion_decision"] == "promote"
    assert elite["elite_status"] == "elite_candidate"
    assert elite["promotion_decision"] == "elite_candidate"


def test_evaluate_convergence_rejects_win_rate_without_expectancy() -> None:
    route = resolve_symbol_route("BTCUSDT")

    result = evaluate_convergence(
        ConvergenceMetrics(
            trade_count=100,
            win_rate=72.0,
            profit_factor=1.6,
            max_drawdown_pct=6.0,
            loss_streak=1,
            expectancy_r=0.0,
            payoff_ratio=0.8,
        ),
        route.validation,
    )

    assert result["screening_status"] == "failed"
    assert result["promotion_decision"] == "reject"
