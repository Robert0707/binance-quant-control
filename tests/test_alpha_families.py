from __future__ import annotations

import pandas as pd

from binance_quant_control.alpha_families import (
    ai_family_router_signal,
    build_diagnostic_family_signals,
    build_strategy_family_signals,
    liquidity_reclaim_signal,
    range_mean_reversion_signal,
    reversal_squeeze_signal,
    select_best_family,
    strategy_family_trade_decision,
    trend_pullback_signal,
    vwap_reclaim_signal,
)
from binance_quant_control.analysis import score_bias


def test_strategy_family_pool_selects_independent_breakout_signal() -> None:
    row = pd.Series(
        {
            "close": 110.0,
            "ema_fast": 98.0,
            "ema_slow": 100.0,
            "macd_hist": 0.0,
            "adx": 14.0,
            "plus_di": 28.0,
            "minus_di": 9.0,
            "squeeze_released": True,
            "squeeze_off": True,
            "squeeze_momentum": 2.0,
            "donchian_breakout_up": True,
            "donchian_breakout_down": False,
            "bb_bandwidth": 0.09,
            "keltner_width_pct": 0.04,
            "keltner_upper": 105.0,
            "keltner_lower": 95.0,
            "volume_ratio_20": 2.2,
            "rsi_14": 64.0,
            "qqe_direction": 1,
            "jumbo_power": 55.0,
            "jumbo_power_ma": 40.0,
            "stoch_rsi_k": 55.0,
            "bb_percent_b": 0.72,
        }
    )

    signals = build_strategy_family_signals(row)
    by_family = {item.family: item for item in signals}
    best = select_best_family(signals)

    assert tuple(by_family) == (
        "trend_continuation",
        "breakout",
        "trend_pullback",
        "liquidity_reclaim",
        "vwap_reclaim",
        "mean_reversion",
    )
    assert by_family["trend_continuation"].bias == "neutral"
    assert by_family["breakout"].bias == "long"
    assert by_family["trend_pullback"].bias == "neutral"
    assert by_family["liquidity_reclaim"].bias == "neutral"
    assert by_family["vwap_reclaim"].bias == "neutral"
    assert by_family["mean_reversion"].bias == "neutral"
    assert best.bias == "long"
    assert best.family == "breakout"

    diagnostic = {item.family: item for item in build_diagnostic_family_signals(row)}
    assert {"volatility_breakout", "high_beta_momentum"}.issubset(diagnostic)


def test_reversal_squeeze_uses_funding_oi_and_taker_flow() -> None:
    row = pd.Series({"stoch_rsi_k": 12.0, "bb_percent_b": 0.08})

    signal = reversal_squeeze_signal(
        row,
        {
            "funding_rate": -0.0012,
            "open_interest_change_pct": 5.0,
            "taker_buy_sell_ratio": 0.88,
        },
    )

    assert signal.bias == "long"
    assert signal.confidence == 1.0
    assert "crowded-short-squeeze-risk" in signal.reasons


def test_range_mean_reversion_is_separate_from_trend_breakout() -> None:
    row = pd.Series(
        {
            "adx": 12.0,
            "rsi_14": 31.0,
            "stoch_rsi_k": 10.0,
            "bb_percent_b": 0.10,
        }
    )

    signal = range_mean_reversion_signal(row)

    assert signal.family == "range_mean_reversion"
    assert signal.bias == "long"
    assert signal.reasons == ("range-oversold",)


def test_strategy_family_trade_decision_uses_only_requested_family() -> None:
    row = pd.Series(
        {
            "close": 90.0,
            "ema_fast": 92.0,
            "ema_slow": 100.0,
            "macd_hist": -1.2,
            "adx": 31.0,
            "plus_di": 8.0,
            "minus_di": 25.0,
            "squeeze_released": True,
            "squeeze_momentum": 2.0,
            "donchian_breakout_up": True,
            "bb_bandwidth": 0.10,
            "keltner_width_pct": 0.03,
            "keltner_upper": 85.0,
            "keltner_lower": 75.0,
            "volume_ratio_20": 2.1,
            "rsi_14": 62.0,
            "qqe_direction": 1,
            "stoch_rsi_k": 55.0,
            "bb_percent_b": 0.65,
        }
    )

    breakout = strategy_family_trade_decision(row, market="futures", family="breakout")
    trend = strategy_family_trade_decision(row, market="futures", family="trend_continuation")

    assert breakout["recommended_action"] == "BUY"
    assert breakout["entry_ready"] is True
    assert breakout["strategy_family"] == "breakout"
    assert breakout["selected_strategy_family"]["family"] == "breakout"
    assert trend["recommended_action"] == "SELL"
    assert trend["selected_strategy_family"]["family"] == "trend_continuation"


def test_ai_family_router_selects_best_active_family() -> None:
    row = pd.Series(
        {
            "close": 116.0,
            "ema_fast": 114.0,
            "ema_slow": 110.0,
            "sma_200": 96.0,
            "adx": 22.0,
            "plus_di": 24.0,
            "minus_di": 15.0,
            "rsi_14": 51.0,
            "stoch_rsi_k": 42.0,
            "bb_percent_b": 0.52,
            "volume_ratio_20": 1.25,
            "supertrend_direction": 1,
            "trend_magic_direction": 0,
            "follow_line_direction": 1,
            "jumbo_power": 18.0,
            "jumbo_power_ma": 16.0,
            "fib_pullback_long_zone": True,
            "fib_ote_long_zone": True,
            "donchian_breakout_up": False,
            "squeeze_released": False,
        }
    )

    signal = ai_family_router_signal(row)
    decision = strategy_family_trade_decision(row, market="futures", family="ai_family_router")

    assert signal.family == "ai_family_router"
    assert signal.bias == "long"
    assert "router-selected:trend_pullback" in signal.reasons
    assert decision["recommended_action"] == "BUY"
    assert decision["entry_ready"] is True
    assert decision["strategy_family"] == "ai_family_router"
    assert decision["routed_strategy_family"] == "trend_pullback"


def test_trend_pullback_signal_uses_fibonacci_zone_without_breakout_chase() -> None:
    row = pd.Series(
        {
            "close": 116.0,
            "ema_fast": 114.0,
            "ema_slow": 110.0,
            "sma_200": 96.0,
            "adx": 22.0,
            "plus_di": 24.0,
            "minus_di": 15.0,
            "rsi_14": 51.0,
            "stoch_rsi_k": 42.0,
            "bb_percent_b": 0.52,
            "volume_ratio_20": 1.25,
            "supertrend_direction": 1,
            "trend_magic_direction": 0,
            "follow_line_direction": 1,
            "jumbo_power": 18.0,
            "jumbo_power_ma": 16.0,
            "fib_pullback_long_zone": True,
            "fib_ote_long_zone": True,
            "fib_pullback_short_zone": False,
            "fib_ote_short_zone": False,
            "donchian_breakout_up": False,
            "squeeze_released": False,
        }
    )

    signal = trend_pullback_signal(row)
    decision = strategy_family_trade_decision(row, market="futures", family="trend_pullback")

    assert signal.family == "trend_pullback"
    assert signal.bias == "long"
    assert signal.confidence >= 0.6
    assert "fib-ote-long-zone" in signal.reasons
    assert decision["recommended_action"] == "BUY"
    assert decision["entry_ready"] is True
    assert decision["selected_strategy_family"]["family"] == "trend_pullback"


def test_liquidity_reclaim_signal_uses_sweep_and_flow_confirmation() -> None:
    row = pd.Series(
        {
            "close": 101.0,
            "adx": 18.0,
            "plus_di": 20.0,
            "minus_di": 18.0,
            "rsi_14": 44.0,
            "stoch_rsi_k": 32.0,
            "bb_percent_b": 0.38,
            "volume_ratio_20": 1.35,
            "liquidity_sweep_low_20": True,
            "liquidity_reclaim_long_20": True,
            "liquidity_sweep_high_20": False,
            "liquidity_reclaim_short_20": False,
            "liquidity_close_position": 0.72,
        }
    )

    signal = liquidity_reclaim_signal(
        row,
        {
            "taker_flow_imbalance": 0.08,
            "order_book_imbalance": 0.02,
            "funding_rate": -0.0006,
            "open_interest_change_pct": 2.5,
        },
    )
    decision = strategy_family_trade_decision(
        row,
        market="futures",
        family="liquidity_reclaim",
        market_context={"taker_flow_imbalance": 0.08},
    )

    assert signal.family == "liquidity_reclaim"
    assert signal.bias == "long"
    assert "sell-side-liquidity-sweep-reclaimed" in signal.reasons
    assert decision["recommended_action"] == "BUY"
    assert decision["entry_ready"] is True
    assert decision["selected_strategy_family"]["family"] == "liquidity_reclaim"


def test_vwap_reclaim_signal_uses_rolling_vwap_bands() -> None:
    row = pd.Series(
        {
            "close": 100.8,
            "vwap_rolling_48": 100.2,
            "vwap_distance_pct_48": 0.35,
            "vwap_reclaim_long_48": True,
            "vwap_reclaim_short_48": False,
            "vwap_mid_reclaim_long_48": False,
            "vwap_mid_reclaim_short_48": False,
            "bb_percent_b": 0.46,
            "rsi_14": 48.0,
            "stoch_rsi_k": 38.0,
            "volume_ratio_20": 1.18,
            "obv_zscore_20": 0.8,
            "adx": 18.0,
            "plus_di": 21.0,
            "minus_di": 17.0,
        }
    )

    signal = vwap_reclaim_signal(row)
    decision = strategy_family_trade_decision(row, market="futures", family="vwap_reclaim")

    assert signal.family == "vwap_reclaim"
    assert signal.bias == "long"
    assert "vwap-lower-band-reclaimed" in signal.reasons
    assert decision["recommended_action"] == "BUY"
    assert decision["entry_ready"] is True
    assert decision["selected_strategy_family"]["family"] == "vwap_reclaim"


def test_score_bias_groups_correlated_trend_indicators_once() -> None:
    row = pd.Series(
        {
            "close": 120.0,
            "ema_fast": 115.0,
            "ema_slow": 100.0,
            "sma_200": 90.0,
            "rsi_14": 50.0,
            "macd_hist": 1.5,
            "breakout_high_20": 130.0,
            "breakout_low_20": 80.0,
            "realized_vol_20": 0.4,
            "adx": 31.0,
            "plus_di": 28.0,
            "minus_di": 9.0,
            "bb_percent_b": 0.50,
            "bb_bandwidth": 0.08,
            "volume_zscore_20": 0.0,
            "obv_zscore_20": 0.0,
            "vwap": 120.0,
            "supertrend_direction": 1,
            "trend_magic_direction": 1,
            "follow_line_direction": 1,
            "jumbo_power": 0.0,
            "jumbo_power_ma": 0.0,
            "mfi_14": 50.0,
            "cci_20": 0.0,
            "stoch_k": 50.0,
            "williams_r_14": -50.0,
            "ultimate_oscillator_proxy": 50.0,
            "volume_ratio_20": 1.0,
            "vwma_20": 0.0,
            "fib_pullback_long_zone": False,
            "fib_ote_long_zone": False,
            "fib_pullback_short_zone": False,
            "fib_ote_short_zone": False,
            "fib_retrace_from_high_89": 0.0,
            "fib_retrace_from_low_89": 0.0,
            "squeeze_released": False,
            "squeeze_off": False,
            "squeeze_momentum": 0.0,
            "donchian_breakout_up": False,
            "donchian_breakout_down": False,
            "keltner_upper": 130.0,
            "keltner_lower": 100.0,
            "keltner_width_pct": 0.04,
            "qqe_direction": 0,
            "stoch_rsi_k": 50.0,
        }
    )

    analysis = score_bias(row, "futures", None, market_context={}, strategy=None)
    groups = {item["name"]: item for item in analysis["signal_groups"]}

    assert groups["trend_alignment"]["bias"] == "long"
    assert groups["trend_alignment"]["counted"] is False
    assert groups["trend_alignment"]["duplicate_of_strategy_family"] == "trend_continuation"
    assert groups["strategy_family"]["bias"] == "long"
    assert analysis["signal_counts"]["bullish"] == 1
    assert analysis["convergence"] == 1.0
    assert analysis["recommended_action"] == "HOLD"
    assert "Signal convergence" not in " ".join(analysis["entry_blockers"])


def test_score_bias_requires_independent_group_support_for_strong_entry() -> None:
    row = pd.Series(
        {
            "close": 120.0,
            "ema_fast": 115.0,
            "ema_slow": 100.0,
            "sma_200": 90.0,
            "rsi_14": 62.0,
            "macd_hist": 1.5,
            "breakout_high_20": 118.0,
            "breakout_low_20": 80.0,
            "realized_vol_20": 0.4,
            "adx": 31.0,
            "plus_di": 28.0,
            "minus_di": 9.0,
            "bb_percent_b": 0.86,
            "bb_bandwidth": 0.12,
            "volume_zscore_20": 1.5,
            "obv_zscore_20": 1.4,
            "vwap": 118.0,
            "supertrend_direction": 1,
            "trend_magic_direction": 1,
            "follow_line_direction": 1,
            "jumbo_power": 58.0,
            "jumbo_power_ma": 40.0,
            "mfi_14": 62.0,
            "cci_20": 80.0,
            "stoch_k": 65.0,
            "williams_r_14": -40.0,
            "ultimate_oscillator_proxy": 61.0,
            "volume_ratio_20": 1.8,
            "vwma_20": 117.0,
            "fib_pullback_long_zone": False,
            "fib_ote_long_zone": False,
            "fib_pullback_short_zone": False,
            "fib_ote_short_zone": False,
            "fib_retrace_from_high_89": 0.0,
            "fib_retrace_from_low_89": 0.0,
            "squeeze_released": True,
            "squeeze_off": True,
            "squeeze_momentum": 2.0,
            "donchian_breakout_up": True,
            "donchian_breakout_down": False,
            "keltner_upper": 116.0,
            "keltner_lower": 100.0,
            "keltner_width_pct": 0.04,
            "qqe_direction": 1,
            "stoch_rsi_k": 60.0,
        }
    )

    analysis = score_bias(
        row,
        "futures",
        None,
        market_context={"taker_buy_sell_ratio": 1.12, "order_book_imbalance": 0.15},
        strategy=None,
    )

    assert analysis["score"] >= 65
    assert analysis["entry_ready"] is True
    assert analysis["recommended_action"] == "BUY"
    assert analysis["signal_counts"]["bullish"] >= 4
    assert {item["name"] for item in analysis["signal_groups"]} >= {
        "momentum_power",
        "price_structure",
        "volume_flow",
        "strategy_family",
    }
