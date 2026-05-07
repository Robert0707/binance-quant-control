from __future__ import annotations

import pandas as pd

from binance_quant_control.analysis import enrich_indicators
from binance_quant_control.indicators import qqe_mod, squeeze_momentum


def _frame(length: int = 120) -> pd.DataFrame:
    rows = []
    price = 100.0
    for idx in range(length):
        drift = 0.05 if idx < length // 2 else 0.18
        open_price = price
        close_price = price + drift
        high_price = max(open_price, close_price) + 0.4
        low_price = min(open_price, close_price) - 0.4
        rows.append(
            {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": 1000.0 + idx * 10.0,
                "quote_asset_volume": (1000.0 + idx * 10.0) * close_price,
                "taker_buy_quote_volume": (1000.0 + idx * 10.0) * close_price * 0.58,
            }
        )
        price = close_price
    return pd.DataFrame(rows, index=pd.date_range("2026-01-01", periods=length, freq="1h", tz="UTC"))


def test_squeeze_momentum_matches_pasted_lazybear_bb_multiplier_behavior() -> None:
    df = _frame()

    pasted = squeeze_momentum(df, length=20, bb_mult=2.0, kc_mult=1.5, match_pasted_source=True)
    canonical = squeeze_momentum(df, length=20, bb_mult=2.0, kc_mult=1.5, match_pasted_source=False)

    assert {"squeeze_no", "squeeze_bb_upper", "squeeze_kc_upper"}.issubset(pasted.columns)
    assert pasted["squeeze_bb_upper"].notna().sum() > 0
    assert not pasted["squeeze_bb_upper"].equals(canonical["squeeze_bb_upper"])


def test_qqe_mod_exposes_primary_secondary_and_bollinger_signals() -> None:
    result = qqe_mod(_frame()["close"])

    expected = {
        "qqe_primary_rsi",
        "qqe_secondary_rsi",
        "qqe_bollinger_upper",
        "qqe_bollinger_lower",
        "qqe_up_signal",
        "qqe_down_signal",
    }
    assert expected.issubset(result.columns)
    assert result["qqe_direction"].dropna().isin([-1, 0, 1]).all()


def test_enrich_indicators_includes_tradingview_gate_columns() -> None:
    enriched = enrich_indicators(_frame(), "1h")

    expected = {
        "jumbo_trend_confirmed",
        "jumbo_volume_momentum",
        "jumbo_momentum_confirmed",
        "qqe_primary_rsi",
        "squeeze_no",
        "chandelier_direction",
        "ichimoku_direction",
        "psar_direction",
        "fib_swing_high_89",
        "fib_swing_low_89",
        "fib_retrace_from_high_89",
        "fib_ote_long_zone",
        "liquidity_swing_high_20",
        "liquidity_swing_low_20",
        "liquidity_sweep_high_20",
        "liquidity_sweep_low_20",
        "liquidity_reclaim_long_20",
        "liquidity_reclaim_short_20",
        "vwap_rolling_48",
        "vwap_upper_1_48",
        "vwap_lower_1_48",
        "vwap_reclaim_long_48",
        "vwap_reclaim_short_48",
    }
    assert expected.issubset(enriched.columns)


def test_enrich_indicators_marks_fibonacci_pullback_zone() -> None:
    rows = []
    for idx in range(99):
        price = 50.0 + idx * 0.35
        rows.append(
            {
                "open": price - 0.2,
                "high": 200.0 if idx == 98 else price + 0.5,
                "low": price - 0.5,
                "close": 88.0 if idx == 98 else price,
                "volume": 1000.0 + idx,
                "quote_asset_volume": (1000.0 + idx) * price,
                "taker_buy_quote_volume": (1000.0 + idx) * price * 0.58,
            }
        )
    rows.append(
        {
            "open": 86.0,
            "high": 87.0,
            "low": 83.0,
            "close": 85.0,
            "volume": 2500.0,
            "quote_asset_volume": 212500.0,
            "taker_buy_quote_volume": 123250.0,
        }
    )
    df = pd.DataFrame(rows, index=pd.date_range("2026-01-01", periods=len(rows), freq="1h", tz="UTC"))

    enriched = enrich_indicators(df, "1h")
    latest = enriched.iloc[-1]

    assert bool(latest["fib_pullback_long_zone"]) is True
    assert bool(latest["fib_ote_long_zone"]) is True
    assert 0.618 <= latest["fib_retrace_from_high_89"] <= 0.786


def test_enrich_indicators_marks_liquidity_sweep_reclaim() -> None:
    rows = []
    for idx in range(25):
        price = 100.0 + (idx % 3) * 0.1
        rows.append(
            {
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price + 0.1,
                "volume": 1000.0,
                "quote_asset_volume": 100000.0,
                "taker_buy_quote_volume": 52000.0,
            }
        )
    rows.append(
        {
            "open": 100.1,
            "high": 101.0,
            "low": 98.2,
            "close": 100.6,
            "volume": 1800.0,
            "quote_asset_volume": 181080.0,
            "taker_buy_quote_volume": 110000.0,
        }
    )
    df = pd.DataFrame(rows, index=pd.date_range("2026-01-01", periods=len(rows), freq="1h", tz="UTC"))

    enriched = enrich_indicators(df, "1h")
    latest = enriched.iloc[-1]

    assert bool(latest["liquidity_sweep_low_20"]) is True
    assert bool(latest["liquidity_reclaim_long_20"]) is True
    assert latest["liquidity_close_position"] >= 0.58


def test_enrich_indicators_marks_vwap_band_reclaim() -> None:
    rows = []
    price = 100.0
    for idx in range(60):
        rows.append(
            {
                "open": price,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price + (idx % 2) * 0.1,
                "volume": 1200.0,
                "quote_asset_volume": 120000.0,
                "taker_buy_quote_volume": 61000.0,
            }
        )
    rows.append(
        {
            "open": 99.6,
            "high": 100.6,
            "low": 95.0,
            "close": 100.2,
            "volume": 2600.0,
            "quote_asset_volume": 260520.0,
            "taker_buy_quote_volume": 150000.0,
        }
    )
    df = pd.DataFrame(rows, index=pd.date_range("2026-01-01", periods=len(rows), freq="1h", tz="UTC"))

    enriched = enrich_indicators(df, "1h")
    latest = enriched.iloc[-1]

    assert latest["vwap_rolling_48"] > 0
    assert bool(latest["vwap_reclaim_long_48"]) is True
