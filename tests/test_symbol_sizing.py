from __future__ import annotations

from pathlib import Path

from binance_quant_control.asset_routing import resolve_symbol_route
from binance_quant_control.strategy import load_strategy_config
from binance_quant_control.symbol_sizing import build_symbol_sizing_plan


def test_symbol_sizing_boosts_core_high_confidence() -> None:
    route = resolve_symbol_route("BTCUSDT")
    strategy = load_strategy_config(Path("config/strategy-btc-volatility.yaml"))

    plan = build_symbol_sizing_plan(
        symbol="BTCUSDT",
        route=route,
        strategy=strategy,
        latest={"adx": 28.0, "realized_vol_20": 0.6},
        analysis={"score": 88, "convergence": 0.86},
        equity_usdt=1000.0,
        available_balance_usdt=800.0,
        min_notional_usdt=5.0,
        volume_rank=3,
    )

    assert plan.recommended_leverage >= 5
    assert plan.recommended_margin_usdt > 5.0
    assert any(reason in plan.reasons for reason in {"strong-confidence", "alpha-testnet-conviction"})


def test_symbol_sizing_reduces_high_beta_extreme_volatility() -> None:
    route = resolve_symbol_route("DOGEUSDT")
    strategy = load_strategy_config(Path("config/strategy-meme-momentum.yaml"))

    plan = build_symbol_sizing_plan(
        symbol="DOGEUSDT",
        route=route,
        strategy=strategy,
        latest={"adx": 9.0, "realized_vol_20": 2.8},
        analysis={"score": 54, "convergence": 0.55},
        equity_usdt=1000.0,
        available_balance_usdt=800.0,
        min_notional_usdt=5.0,
        volume_rank=45,
        news_risk={"risk_level": "high"},
    )

    assert plan.recommended_leverage == 1
    assert plan.recommended_margin_pct < 0.1
    assert plan.warnings


def test_symbol_sizing_uses_flow_news_and_route_feedback() -> None:
    route = resolve_symbol_route("DOGEUSDT")
    strategy = load_strategy_config(Path("config/strategy-meme-momentum.yaml"))

    plan = build_symbol_sizing_plan(
        symbol="DOGEUSDT",
        route=route,
        strategy=strategy,
        latest={"adx": 24.0, "realized_vol_20": 0.8},
        analysis={"score": 86, "convergence": 0.9},
        equity_usdt=1000.0,
        available_balance_usdt=800.0,
        min_notional_usdt=5.0,
        volume_rank=18,
        news_risk={"risk_level": "high"},
        signal_scores={
            "flow_score": 38.0,
            "event_risk_score": 25.0,
            "composite_convergence_score": 55.0,
        },
        route_side_risk={
            "sample_count": 169,
            "profit_factor": 0.491,
            "net_pnl_usdt": -5.155,
        },
    )

    assert plan.recommended_leverage <= 2
    assert plan.recommended_margin_pct < 0.06
    assert any("Weak flow" in item for item in plan.warnings)
    assert any("Route/side" in item for item in plan.warnings)


def test_symbol_sizing_reduces_high_stop_loss_route_side() -> None:
    route = resolve_symbol_route("SOLUSDT")
    strategy = load_strategy_config(Path("config/strategy-major-alt-trend.yaml"))

    plan = build_symbol_sizing_plan(
        symbol="SOLUSDT",
        route=route,
        strategy=strategy,
        latest={"adx": 26.0, "realized_vol_20": 0.7},
        analysis={"score": 92, "convergence": 0.92},
        equity_usdt=1000.0,
        available_balance_usdt=800.0,
        min_notional_usdt=5.0,
        volume_rank=8,
        signal_scores={
            "flow_score": 75.0,
            "event_risk_score": 70.0,
            "composite_convergence_score": 82.0,
        },
        route_side_risk={
            "sample_count": 151,
            "profit_factor": 0.92,
            "net_pnl_usdt": -1.2,
            "stop_loss_ratio": 72.0,
            "avg_r_multiple": -0.25,
        },
    )

    assert plan.recommended_margin_pct < 0.3
    assert plan.max_account_risk_pct < 0.014
    assert any("stop-loss ratio" in item for item in plan.warnings)
    assert any("average R" in item for item in plan.warnings)


def test_symbol_sizing_forces_unknown_high_event_risk_to_one_x() -> None:
    route = resolve_symbol_route("UBUSDT")
    strategy = load_strategy_config(Path("config/strategy-defensive-default.yaml"))

    plan = build_symbol_sizing_plan(
        symbol="UBUSDT",
        route=route,
        strategy=strategy,
        latest={"adx": 30.0, "realized_vol_20": 0.5},
        analysis={"score": 100, "convergence": 1.0},
        equity_usdt=5000.0,
        available_balance_usdt=5000.0,
        min_notional_usdt=5.0,
        volume_rank=55,
        news_risk={"risk_level": "high"},
        signal_scores={
            "flow_score": 99.0,
            "event_risk_score": 20.0,
            "composite_convergence_score": 80.0,
        },
    )

    assert plan.recommended_leverage == 1
    assert plan.max_account_risk_pct <= 0.003
    assert any("forced to 1x" in item for item in plan.warnings)


def test_symbol_sizing_raises_leverage_to_satisfy_exchange_min_notional_when_safe() -> None:
    route = resolve_symbol_route("ETHUSDT")
    strategy = load_strategy_config(Path("config/strategy-live-pilot.yaml"))

    plan = build_symbol_sizing_plan(
        symbol="ETHUSDT",
        route=route,
        strategy=strategy,
        latest={"adx": 24.0, "realized_vol_20": 0.5},
        analysis={"score": 78, "convergence": 0.9},
        equity_usdt=5.0,
        available_balance_usdt=0.84,
        min_notional_usdt=20.0,
        volume_rank=45,
    )

    assert plan.recommended_margin_usdt * plan.recommended_leverage >= 20.0 - 0.00001
    assert plan.recommended_leverage <= plan.max_leverage
    assert "leverage-raised-to-satisfy-exchange-min-notional" in plan.reasons


def test_symbol_sizing_treats_old_route_loss_as_advisory_for_market_bot_promotion() -> None:
    route = resolve_symbol_route("ETHUSDT")
    strategy = load_strategy_config(Path("config/strategy-live-pilot.yaml"))

    base_kwargs = {
        "symbol": "ETHUSDT",
        "route": route,
        "strategy": strategy,
        "latest": {"adx": 24.0, "realized_vol_20": 0.5},
        "analysis": {"score": 78, "convergence": 0.9},
        "equity_usdt": 5.0,
        "available_balance_usdt": 0.84,
        "min_notional_usdt": 20.0,
        "volume_rank": 45,
        "route_side_risk": {
            "sample_count": 40,
            "profit_factor": 0.7,
            "net_pnl_usdt": -2.0,
            "stop_loss_ratio": 40.0,
            "avg_r_multiple": -0.05,
        },
    }

    blocked_by_legacy_history = build_symbol_sizing_plan(
        **base_kwargs,
        market_bot_promoted=False,
    )
    promoted = build_symbol_sizing_plan(
        **base_kwargs,
        market_bot_promoted=True,
    )

    assert blocked_by_legacy_history.recommended_leverage <= 2
    assert blocked_by_legacy_history.recommended_margin_usdt * blocked_by_legacy_history.recommended_leverage < 20.0
    assert promoted.recommended_leverage > blocked_by_legacy_history.recommended_leverage
    assert promoted.recommended_margin_usdt * promoted.recommended_leverage >= 20.0 - 0.00001
    assert any("advisory" in item for item in promoted.warnings)
