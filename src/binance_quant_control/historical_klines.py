from __future__ import annotations

from typing import Any

MAX_BINANCE_KLINE_LIMIT = 1500


def _client_klines(
    client: Any,
    symbol: str,
    interval: str,
    limit: int,
    market: str,
    *,
    end_time: int | None,
) -> Any:
    try:
        return client.klines(symbol, interval, limit, market, end_time=end_time)
    except TypeError:
        return client.klines(symbol, interval, limit, market)


def fetch_recent_klines(
    client: Any,
    symbol: str,
    interval: str,
    limit: int,
    market: str,
) -> list[list[Any]]:
    """Fetch recent klines with backwards pagination when more than one Binance page is needed."""
    target = max(int(limit), 1)
    rows: list[list[Any]] = []
    end_time: int | None = None
    seen_open_times: set[int] = set()
    while len(rows) < target:
        page_limit = min(MAX_BINANCE_KLINE_LIMIT, target - len(rows))
        page = _client_klines(
            client,
            symbol,
            interval,
            page_limit,
            market,
            end_time=end_time,
        )
        if not page:
            break
        normalized_page = [list(item) for item in page]
        new_rows: list[list[Any]] = []
        for row in normalized_page:
            try:
                open_time = int(float(row[0]))
            except (TypeError, ValueError, IndexError):
                continue
            if open_time in seen_open_times:
                continue
            seen_open_times.add(open_time)
            new_rows.append(row)
        if not new_rows:
            break
        rows = new_rows + rows
        earliest = min(int(float(row[0])) for row in new_rows)
        next_end_time = earliest - 1
        if end_time is not None and next_end_time >= end_time:
            break
        end_time = next_end_time
        if len(normalized_page) < page_limit:
            break
    return rows[-target:]
