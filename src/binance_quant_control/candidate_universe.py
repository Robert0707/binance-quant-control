from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .asset_routing import normalize_symbol
from .binance_api import BinanceClient
from .config import Settings


@dataclass(frozen=True, slots=True)
class UniverseSymbol:
    symbol: str
    quote_volume_usdt: float
    rank: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tradeable_usdt_perpetuals(exchange_info: dict[str, Any]) -> set[str]:
    symbols: set[str] = set()
    for item in exchange_info.get("symbols") or []:
        symbol = str(item.get("symbol") or "").upper()
        if not symbol.endswith("USDT"):
            continue
        if str(item.get("status") or "").upper() != "TRADING":
            continue
        if str(item.get("contractType") or "PERPETUAL").upper() != "PERPETUAL":
            continue
        symbols.add(symbol)
    return symbols


def fetch_top_futures_symbols(
    settings: Settings,
    *,
    limit: int = 60,
    include_symbols: list[str] | None = None,
) -> list[UniverseSymbol]:
    with BinanceClient(settings) as client:
        exchange_info = client.exchange_info("", "futures")
        tickers = client.ticker_24hr("futures")

    tradeable = _tradeable_usdt_perpetuals(exchange_info)
    rows: list[tuple[str, float]] = []
    for item in tickers if isinstance(tickers, list) else []:
        symbol = str(item.get("symbol") or "").upper()
        if symbol not in tradeable:
            continue
        try:
            quote_volume = float(item.get("quoteVolume") or 0.0)
        except (TypeError, ValueError):
            quote_volume = 0.0
        if quote_volume > 0.0:
            rows.append((symbol, quote_volume))

    rows.sort(key=lambda row: row[1], reverse=True)
    selected: list[UniverseSymbol] = []
    seen: set[str] = set()
    for rank, (symbol, quote_volume) in enumerate(rows, start=1):
        if rank > limit:
            break
        seen.add(symbol)
        selected.append(
            UniverseSymbol(
                symbol=symbol,
                quote_volume_usdt=round(quote_volume, 3),
                rank=rank,
                source="binance-futures-24hr-volume",
            )
        )

    for raw_symbol in include_symbols or []:
        symbol = normalize_symbol(raw_symbol)
        if symbol in seen or symbol not in tradeable:
            continue
        seen.add(symbol)
        selected.append(
            UniverseSymbol(
                symbol=symbol,
                quote_volume_usdt=0.0,
                rank=0,
                source="configured-include",
            )
        )
    return selected


def volume_rank_map(symbols: list[UniverseSymbol]) -> dict[str, int]:
    return {item.symbol: item.rank for item in symbols if item.rank > 0}
