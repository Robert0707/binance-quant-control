from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from scripts.research_ocean_x_btc_evidence import (
    TradingViewConvergenceParams,
    _dataset_quality,
    _normalize_regime_filters,
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
