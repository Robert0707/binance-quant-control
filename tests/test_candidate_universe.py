from __future__ import annotations

from types import SimpleNamespace

import binance_quant_control.candidate_universe as universe


class FakeClient:
    def __init__(self, settings):
        self.settings = settings

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def exchange_info(self, symbol: str, market: str = "futures"):
        return {
            "symbols": [
                {"symbol": "BTCUSDT", "status": "TRADING", "contractType": "PERPETUAL"},
                {"symbol": "ETHUSDT", "status": "TRADING", "contractType": "PERPETUAL"},
                {"symbol": "BADUSDT", "status": "BREAK", "contractType": "PERPETUAL"},
                {"symbol": "BTCUSDC", "status": "TRADING", "contractType": "PERPETUAL"},
            ]
        }

    def ticker_24hr(self, market: str = "futures"):
        return [
            {"symbol": "ETHUSDT", "quoteVolume": "500"},
            {"symbol": "BTCUSDT", "quoteVolume": "1000"},
            {"symbol": "BADUSDT", "quoteVolume": "2000"},
            {"symbol": "BTCUSDC", "quoteVolume": "3000"},
        ]


def test_fetch_top_futures_symbols_filters_and_ranks(monkeypatch) -> None:
    monkeypatch.setattr(universe, "BinanceClient", FakeClient)

    result = universe.fetch_top_futures_symbols(SimpleNamespace(), limit=2)

    assert [item.symbol for item in result] == ["BTCUSDT", "ETHUSDT"]
    assert [item.rank for item in result] == [1, 2]
