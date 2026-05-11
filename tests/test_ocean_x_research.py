from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from scripts.research_ocean_x_btc_evidence import (
    TradingViewConvergenceParams,
    _dataset_quality,
    _default_tradingview_param_grid,
    _evaluate_gate,
    _limit_tradingview_param_grid,
    _limit_tradingview_shortlist,
    _normalize_regime_filters,
    _normalize_side_filter,
    _simulate_signal_trades,
    apply_regime_filter,
    build_tradingview_signal_features,
)


def _feature_frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "ocean_proxy_signal": ["L", "L", "S"],
            "close": [110.0, 90.0, 88.0],
            "open": [100.0, 95.0, 92.0],
            "high": [112.0, 96.0, 94.0],
            "low": [99.0, 88.0, 87.0],
            "ema_fast": [105.0, 95.0, 92.0],
            "ema_slow": [100.0, 100.0, 96.0],
            "sma_200": [99.0, 99.0, 97.0],
            "macd_hist": [1.0, -1.0, -1.0],
            "plus_di": [30.0, 10.0, 10.0],
            "minus_di": [10.0, 30.0, 35.0],
            "supertrend_direction": [1.0, -1.0, -1.0],
            "trend_magic_direction": [1.0, -1.0, -1.0],
            "follow_line_direction": [1.0, -1.0, -1.0],
            "adx": [24.0, 24.0, 24.0],
            "bb_percent_b": [0.5, 0.5, 0.5],
            "rsi_14": [55.0, 55.0, 45.0],
            "stoch_rsi_k": [35.0, 50.0, 70.0],
            "mfi_14": [58.0, 42.0, 40.0],
            "vwap": [101.0, 94.0, 91.0],
            "volume_zscore_20": [0.8, 0.2, 0.9],
            "volume_ratio_20": [1.2, 1.0, 1.25],
            "jumbo_power": [20.0, -20.0, -30.0],
            "jumbo_power_ma": [10.0, -10.0, -20.0],
            "jumbo_long_signal": [True, False, False],
            "jumbo_short_signal": [False, False, True],
            "taker_flow_imbalance": [0.2, -0.2, -0.25],
            "fib_pullback_long_zone": [True, False, False],
            "fib_ote_long_zone": [False, False, False],
            "fib_pullback_short_zone": [False, False, True],
            "fib_ote_short_zone": [False, False, False],
            "liquidity_reclaim_long_20": [False, False, False],
            "liquidity_reclaim_short_20": [False, False, True],
        },
        index=index,
    )


def test_regime_filter_keeps_only_trend_aligned_signal() -> None:
    filtered = apply_regime_filter(_feature_frame(), signal="L", regime_filter="trend")

    assert filtered["ocean_proxy_signal"].tolist() == ["L", "", "S"]


def test_regime_filter_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported regime filter"):
        _normalize_regime_filters(["mystery"])


def test_dataset_quality_requires_coverage_and_bars() -> None:
    frame = pd.DataFrame(
        {"close": [1.0, 2.0]},
        index=pd.to_datetime(["2026-03-26T14:00:00Z", "2026-05-02T23:00:00Z"]),
    )

    quality = _dataset_quality(
        frame,
        requested_start=date(2024, 5, 3),
        requested_end=date(2026, 5, 3),
        min_coverage_ratio=0.65,
        min_dataset_bars=1000,
    )

    assert quality["mature"] is False
    assert quality["bars"] == 2
    assert quality["coverage_ratio"] < 0.1


def test_tradingview_supertrend_macd_builds_transparent_long_signal() -> None:
    params = TradingViewConvergenceParams(
        family="tv_supertrend_macd",
        side="long",
        stop_loss_pct=1.1,
        take_profit_pct=0.7,
        max_hold_bars=12,
        fee_bps=4.0,
        slippage_bps=2.0,
        min_adx=20.0,
        min_trend_votes=4,
        min_volume_z=0.5,
        min_abs_taker_flow=0.1,
    )

    features = build_tradingview_signal_features(_feature_frame(), params=params)

    assert features["ocean_proxy_signal"].tolist() == ["L", "", ""]
    assert features["tv_family"].tolist() == ["tv_supertrend_macd"] * 3


def test_side_filter_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unsupported side filter"):
        _normalize_side_filter("sideways")


def test_tradingview_shortlist_keeps_payoff_shaped_configs() -> None:
    low_payoff = TradingViewConvergenceParams(
        family="tv_supertrend_macd",
        side="short",
        stop_loss_pct=1.3,
        take_profit_pct=0.9,
        max_hold_bars=18,
        fee_bps=4.0,
        slippage_bps=2.0,
    )
    high_payoff = TradingViewConvergenceParams(
        family="tv_supertrend_macd",
        side="short",
        stop_loss_pct=1.0,
        take_profit_pct=2.5,
        max_hold_bars=48,
        fee_bps=4.0,
        slippage_bps=2.0,
    )
    shortlist = [
        {
            "signal": "S",
            "params": low_payoff,
            "pre_screen": {"train_signals": 200, "test_signals": 80, "full_signals": 280},
        },
        {
            "signal": "S",
            "params": high_payoff,
            "pre_screen": {"train_signals": 90, "test_signals": 34, "full_signals": 124},
        },
    ]

    limited = _limit_tradingview_shortlist(
        shortlist,
        max_items=2,
        min_train_trades=70,
        min_test_trades=30,
    )

    assert high_payoff in {item["params"] for item in limited}


def test_tradingview_param_limit_keeps_entry_stoch_variants() -> None:
    limited = _limit_tradingview_param_grid(
        _default_tradingview_param_grid(max_per_trade_risk_pct=2.5),
        max_configs=420,
    )

    assert any(
        item.family == "tv_vwap_trend"
        and item.side == "long"
        and item.stop_loss_pct == 2.0
        and item.take_profit_pct == 5.0
        and item.min_entry_stoch == 60.0
        for item in limited
    )


def test_expectancy_gate_reports_walk_forward_pf_sample_stability_without_blocking() -> None:
    strong = {
        "trade_count": 80,
        "profit_factor": 1.4,
        "expectancy_pct": 0.08,
        "payoff_ratio": 1.4,
        "max_drawdown_pct": 5.0,
        "max_loss_streak": 3,
        "stop_loss_ratio": 30.0,
    }
    passed, blockers, diagnostics = _evaluate_gate(
        train=strong,
        test=strong,
        full=strong,
        window_results=[
            {"trade_count": 35, "profit_factor": 1.3, "expectancy_pct": 0.05, "total_return_pct": 2.0},
            {"trade_count": 35, "profit_factor": 1.3, "expectancy_pct": 0.05, "total_return_pct": 2.0},
            {"trade_count": 5, "profit_factor": 9.0, "expectancy_pct": 0.9, "total_return_pct": 4.0},
            {"trade_count": 35, "profit_factor": 0.8, "expectancy_pct": -0.03, "total_return_pct": -1.0},
        ],
        walk_forward_windows=4,
        gate_mode="expectancy",
        target_win_rate=45,
        min_train_trades=70,
        min_test_trades=30,
        min_profit_factor=1.2,
        min_stop_loss_ratio=55,
        min_expectancy_pct=0.03,
        min_payoff_ratio=1.2,
        max_drawdown_pct=20,
        max_loss_streak=8,
    )

    assert passed is False
    assert diagnostics["positive_train_test_windows"] == 2
    assert "walk-forward-train-test-stability-too-low" not in blockers
    assert "walk-forward-min-expectancy-negative" in blockers


def test_expectancy_gate_does_not_require_every_window_to_meet_pf_sample_floor() -> None:
    strong = {
        "trade_count": 80,
        "profit_factor": 1.4,
        "expectancy_pct": 0.08,
        "payoff_ratio": 1.4,
        "max_drawdown_pct": 5.0,
        "max_loss_streak": 3,
        "stop_loss_ratio": 30.0,
    }
    passed, blockers, diagnostics = _evaluate_gate(
        train=strong,
        test=strong,
        full=strong,
        window_results=[
            {"trade_count": 35, "profit_factor": 1.3, "expectancy_pct": 0.05, "total_return_pct": 2.0},
            {"trade_count": 20, "profit_factor": 1.1, "expectancy_pct": 0.04, "total_return_pct": 1.0},
            {"trade_count": 35, "profit_factor": 1.3, "expectancy_pct": 0.05, "total_return_pct": 2.0},
            {"trade_count": 20, "profit_factor": 1.1, "expectancy_pct": 0.04, "total_return_pct": 1.0},
        ],
        walk_forward_windows=4,
        gate_mode="expectancy",
        target_win_rate=45,
        min_train_trades=70,
        min_test_trades=30,
        min_profit_factor=1.2,
        min_stop_loss_ratio=55,
        min_expectancy_pct=0.03,
        min_payoff_ratio=1.2,
        max_drawdown_pct=20,
        max_loss_streak=8,
    )

    assert passed is True
    assert diagnostics["positive_train_test_windows"] == 2
    assert "walk-forward-positive-window-count-too-low" not in blockers


def test_loss_cooldown_declusters_repeated_failed_signals() -> None:
    index = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    features = pd.DataFrame(
        {
            "ocean_proxy_signal": ["L", "L", "L", "L", "", "", "", ""],
            "open": [100.0] * 8,
            "high": [100.1] * 8,
            "low": [98.5] * 8,
            "close": [99.0] * 8,
        },
        index=index,
    )
    no_cooldown = TradingViewConvergenceParams(
        family="tv_supertrend_macd",
        side="long",
        stop_loss_pct=1.0,
        take_profit_pct=2.0,
        max_hold_bars=1,
        fee_bps=4.0,
        slippage_bps=2.0,
        cooldown_bars_after_loss=0,
    )
    cooldown = TradingViewConvergenceParams(
        family="tv_supertrend_macd",
        side="long",
        stop_loss_pct=1.0,
        take_profit_pct=2.0,
        max_hold_bars=1,
        fee_bps=4.0,
        slippage_bps=2.0,
        cooldown_bars_after_loss=2,
    )

    assert len(_simulate_signal_trades(features, signal="L", params=cooldown)) < len(
        _simulate_signal_trades(features, signal="L", params=no_cooldown)
    )


def test_quality_flow_regime_requires_volume_and_flow_confirmation() -> None:
    frame = _feature_frame()
    frame.loc[frame.index[0], "volume_zscore_20"] = -0.5
    frame.loc[frame.index[0], "volume_ratio_20"] = 1.0

    filtered = apply_regime_filter(frame, signal="L", regime_filter="quality_flow")

    assert filtered["ocean_proxy_signal"].tolist() == ["", "", "S"]


def test_breakeven_trigger_reduces_full_stop_loss_after_favorable_move() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    features = pd.DataFrame(
        {
            "ocean_proxy_signal": ["L", "", "", ""],
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 101.2, 100.5, 100.0],
            "low": [100.0, 99.8, 98.8, 100.0],
            "close": [100.0, 100.2, 99.0, 100.0],
        },
        index=index,
    )
    params = TradingViewConvergenceParams(
        family="tv_supertrend_macd",
        side="long",
        stop_loss_pct=1.0,
        take_profit_pct=3.0,
        max_hold_bars=2,
        fee_bps=0.0,
        slippage_bps=0.0,
        breakeven_trigger_r=1.0,
    )

    trades = _simulate_signal_trades(features, signal="L", params=params)

    assert trades[0]["exit_reason"] == "breakeven_stop"
    assert trades[0]["pnl_pct"] == 0.0


def test_entry_stoch_gate_filters_low_momentum_long_entry_bar() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    features = pd.DataFrame(
        {
            "ocean_proxy_signal": ["L", "", "", ""],
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 103.0, 100.0, 100.0],
            "low": [100.0, 99.0, 100.0, 100.0],
            "close": [100.0, 102.0, 100.0, 100.0],
            "stoch_rsi_k": [70.0, 45.0, 70.0, 70.0],
        },
        index=index,
    )
    ungated = TradingViewConvergenceParams(
        family="tv_supertrend_macd",
        side="long",
        stop_loss_pct=1.0,
        take_profit_pct=2.0,
        max_hold_bars=2,
        fee_bps=0.0,
        slippage_bps=0.0,
    )
    gated = TradingViewConvergenceParams(
        family="tv_supertrend_macd",
        side="long",
        stop_loss_pct=1.0,
        take_profit_pct=2.0,
        max_hold_bars=2,
        fee_bps=0.0,
        slippage_bps=0.0,
        min_entry_stoch=60.0,
    )

    assert len(_simulate_signal_trades(features, signal="L", params=ungated)) == 1
    assert _simulate_signal_trades(features, signal="L", params=gated) == []
