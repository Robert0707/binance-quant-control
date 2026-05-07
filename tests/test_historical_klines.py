from __future__ import annotations

from binance_quant_control.historical_klines import fetch_recent_klines


def _row(open_time: int) -> list[object]:
    return [
        open_time,
        "1",
        "1",
        "1",
        "1",
        "1",
        open_time + 59_999,
        "1",
        1,
        "1",
        "1",
        "0",
    ]


def test_fetch_recent_klines_paginates_backwards() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = []

        def klines(self, symbol, interval, limit, market, *, end_time=None):
            self.calls.append({"limit": limit, "end_time": end_time})
            if end_time is None:
                return [_row(open_time) for open_time in range(1_500, 3_000)]
            return [_row(open_time) for open_time in range(0, 1_500)]

    client = FakeClient()

    rows = fetch_recent_klines(client, "BTCUSDT", "1h", 2_000, "futures")

    assert len(rows) == 2_000
    assert rows[0][0] == 1_000
    assert rows[-1][0] == 2_999
    assert client.calls == [
        {"limit": 1500, "end_time": None},
        {"limit": 500, "end_time": 1499},
    ]
