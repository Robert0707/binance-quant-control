from __future__ import annotations

import pandas as pd

from binance_quant_control.volume_structure import (
    summarize_htf_volume_imbalance,
    summarize_volume_bubbles,
    summarize_volume_profile,
)


def _frame() -> pd.DataFrame:
    rows = []
    price = 100.0
    for idx in range(120):
        open_price = price
        close_price = price + (0.3 if idx % 3 else -0.1)
        if idx == 119:
            close_price = open_price + 2.0
        volume = 1000.0 + idx
        if idx == 119:
            volume = 9000.0
        rows.append(
            {
                "open": open_price,
                "high": max(open_price, close_price) + 0.5,
                "low": min(open_price, close_price) - 0.5,
                "close": close_price,
                "volume": volume,
            }
        )
        price = close_price
    return pd.DataFrame(rows, index=pd.date_range("2026-01-01", periods=120, freq="15min", tz="UTC"))


def test_volume_profile_reports_poc_and_value_area() -> None:
    payload = summarize_volume_profile(_frame(), rows=20, lookback=120)

    assert payload["available"] is True
    assert payload["poc"] > 0
    assert payload["vah"] > payload["val"]
    assert payload["close_position"] in {"above-value", "below-value", "near-poc", "inside-value"}


def test_volume_bubbles_detects_large_buy_cluster() -> None:
    payload = summarize_volume_bubbles(_frame())

    assert payload["available"] is True
    assert payload["cluster"] in {"medium", "big"}
    assert payload["side"] == "buy"
    assert payload["volume_ratio"] > 1.0


def test_htf_volume_imbalance_tracks_latest_spike_zone() -> None:
    payload = summarize_htf_volume_imbalance(_frame(), lookback=120, percentile=95.0)

    assert payload["available"] is True
    assert payload["direction"] == "bullish"
    assert payload["zone_high"] >= payload["zone_low"]
    assert payload["volume_ratio"] >= 1.0
