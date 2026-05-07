from __future__ import annotations

import pandas as pd

from binance_quant_control.market_context import (
    estimate_slippage_from_order_book,
    summarize_market_context,
    summarize_taker_flow,
)


def test_summarize_taker_flow_builds_ratio_and_imbalance() -> None:
    df = pd.DataFrame(
        [
            {"quote_asset_volume": 100.0, "taker_buy_quote_volume": 60.0},
            {"quote_asset_volume": 90.0, "taker_buy_quote_volume": 45.0},
        ]
    )
    payload = summarize_taker_flow(df, lookback=2)
    assert payload["taker_buy_sell_ratio"] > 1.0
    assert payload["taker_flow_imbalance"] > 0.0


def test_estimate_slippage_from_order_book_uses_depth_levels() -> None:
    order_book = {
        "bids": [["100.0", "1.0"], ["99.5", "3.0"]],
        "asks": [["100.5", "1.0"], ["101.0", "3.0"]],
    }
    payload = estimate_slippage_from_order_book(
        order_book,
        side="BUY",
        target_notional_usdt=250.0,
        fallback_slippage_bps=3.0,
    )
    assert payload["spread_bps"] > 0.0
    assert payload["estimated_slippage_bps"] >= 0.0
    assert payload["levels_consumed"] >= 2.0


def test_summarize_market_context_collects_futures_inputs() -> None:
    class FakeClient:
        def order_book(self, symbol: str, market: str, limit: int = 20):
            return {"bids": [["99.9", "5"]], "asks": [["100.1", "4"]]}

        def funding_rate_history(self, symbol: str, limit: int = 2):
            return [{"fundingRate": "0.0001"}, {"fundingRate": "0.0009"}]

        def open_interest_hist(self, symbol: str, period: str = "5m", limit: int = 2):
            return [
                {"sumOpenInterestValue": "1000"},
                {"sumOpenInterestValue": "1050"},
            ]

    df = pd.DataFrame(
        [
            {
                "open": 99.0,
                "high": 101.0,
                "low": 98.0,
                "close": 100.0,
                "volume": 1000.0,
                "quote_asset_volume": 100.0,
                "taker_buy_quote_volume": 55.0,
            },
            {
                "open": 100.0,
                "high": 103.0,
                "low": 99.0,
                "close": 102.0,
                "volume": 1200.0,
                "quote_asset_volume": 120.0,
                "taker_buy_quote_volume": 70.0,
            },
        ]
    )
    payload = summarize_market_context(client=FakeClient(), symbol="BTCUSDT", market="futures", df=df)
    assert payload["funding_rate"] == 0.0009
    assert payload["open_interest_change_pct"] == 5.0
    assert payload["spread_bps"] > 0.0
    assert "volume_profile" in payload
    assert "volume_bubbles" in payload
    assert "htf_volume_imbalance" in payload
