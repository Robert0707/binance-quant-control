from __future__ import annotations

from typing import Any

import pandas as pd

from .volume_structure import (
    summarize_htf_volume_imbalance,
    summarize_volume_bubbles,
    summarize_volume_profile,
)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_taker_flow(df: pd.DataFrame, lookback: int = 20) -> dict[str, float]:
    if df.empty:
        return {
            "taker_buy_sell_ratio": 1.0,
            "taker_flow_imbalance": 0.0,
        }

    window = df.tail(max(1, lookback))
    buy_quote = float(pd.to_numeric(window["taker_buy_quote_volume"], errors="coerce").fillna(0.0).sum())
    total_quote = float(pd.to_numeric(window["quote_asset_volume"], errors="coerce").fillna(0.0).sum())
    sell_quote = max(total_quote - buy_quote, 0.0)
    total_taker = buy_quote + sell_quote
    ratio = buy_quote / sell_quote if sell_quote > 0 else (2.0 if buy_quote > 0 else 1.0)
    imbalance = ((buy_quote - sell_quote) / total_taker) if total_taker > 0 else 0.0
    return {
        "taker_buy_sell_ratio": round(ratio, 6),
        "taker_flow_imbalance": round(imbalance, 6),
    }


def summarize_order_book(order_book: dict[str, Any] | None) -> dict[str, float]:
    bids = (order_book or {}).get("bids") or []
    asks = (order_book or {}).get("asks") or []
    best_bid = _float(bids[0][0]) if bids else 0.0
    best_ask = _float(asks[0][0]) if asks else 0.0
    best_bid_qty = _float(bids[0][1]) if bids else 0.0
    best_ask_qty = _float(asks[0][1]) if asks else 0.0
    spread = max(best_ask - best_bid, 0.0) if best_bid > 0 and best_ask > 0 else 0.0
    mid = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else max(best_bid, best_ask, 0.0)
    spread_bps = (spread / mid) * 10_000.0 if mid > 0 else 0.0
    total_top_qty = best_bid_qty + best_ask_qty
    imbalance = ((best_bid_qty - best_ask_qty) / total_top_qty) if total_top_qty > 0 else 0.0
    return {
        "best_bid": round(best_bid, 8),
        "best_ask": round(best_ask, 8),
        "spread_bps": round(spread_bps, 4),
        "order_book_imbalance": round(imbalance, 6),
    }


def estimate_slippage_from_order_book(
    order_book: dict[str, Any] | None,
    *,
    side: str,
    target_notional_usdt: float,
    fallback_slippage_bps: float,
) -> dict[str, float]:
    summary = summarize_order_book(order_book)
    book_side = "asks" if side.upper() == "BUY" else "bids"
    levels = (order_book or {}).get(book_side) or []
    reference_price = summary["best_ask"] if side.upper() == "BUY" else summary["best_bid"]
    remaining_quote = max(float(target_notional_usdt), 0.0)
    filled_quote = 0.0
    filled_base = 0.0
    levels_consumed = 0

    for raw_price, raw_qty, *_ in levels:
        price = _float(raw_price)
        quantity = _float(raw_qty)
        if price <= 0.0 or quantity <= 0.0 or remaining_quote <= 0.0:
            continue
        level_quote = price * quantity
        take_quote = min(level_quote, remaining_quote)
        take_base = take_quote / price
        filled_quote += take_quote
        filled_base += take_base
        remaining_quote -= take_quote
        levels_consumed += 1
        if remaining_quote <= 1e-9:
            break

    if filled_base <= 0.0 or reference_price <= 0.0:
        return {
            **summary,
            "estimated_slippage_bps": round(float(fallback_slippage_bps), 4),
            "fill_ratio": 0.0,
            "levels_consumed": 0.0,
        }

    average_fill_price = filled_quote / filled_base
    slippage_bps = abs(average_fill_price - reference_price) / reference_price * 10_000.0
    fill_ratio = filled_quote / target_notional_usdt if target_notional_usdt > 0 else 1.0
    if fill_ratio < 0.999:
        slippage_bps = max(slippage_bps, float(fallback_slippage_bps))
    return {
        **summary,
        "estimated_slippage_bps": round(max(slippage_bps, 0.0), 4),
        "fill_ratio": round(min(fill_ratio, 1.0), 4),
        "levels_consumed": float(levels_consumed),
    }


def summarize_market_context(
    *,
    client: Any,
    symbol: str,
    market: str,
    df: pd.DataFrame,
) -> dict[str, Any]:
    context: dict[str, Any] = {"data_quality_notes": []}
    context.update(summarize_taker_flow(df))
    context["volume_profile"] = summarize_volume_profile(df, rows=20, lookback=240)
    context["volume_bubbles"] = summarize_volume_bubbles(df)
    context["htf_volume_imbalance"] = summarize_htf_volume_imbalance(df)

    try:
        order_book = client.order_book(symbol, market, limit=20)
    except Exception:
        order_book = None
        context["data_quality_notes"].append("order-book-unavailable")
    context.update(summarize_order_book(order_book))

    if market != "futures":
        return context

    try:
        funding_history = client.funding_rate_history(symbol, limit=2)
    except Exception:
        funding_history = []
    latest_funding = funding_history[-1] if funding_history else {}
    if latest_funding:
        context["funding_rate"] = round(_float(latest_funding.get("fundingRate")), 8)
    else:
        context["funding_rate"] = None
        context["data_quality_notes"].append("funding-rate-unavailable")

    try:
        open_interest_hist = client.open_interest_hist(symbol, period="5m", limit=2)
    except Exception:
        open_interest_hist = []
    latest_oi = open_interest_hist[-1] if open_interest_hist else {}
    previous_oi = open_interest_hist[-2] if len(open_interest_hist) >= 2 else {}
    if latest_oi:
        latest_oi_value = _float(latest_oi.get("sumOpenInterestValue"))
        previous_oi_value = _float(previous_oi.get("sumOpenInterestValue"))
        oi_change_pct = 0.0
        if previous_oi_value > 0:
            oi_change_pct = ((latest_oi_value - previous_oi_value) / previous_oi_value) * 100.0
        context["open_interest_value"] = round(latest_oi_value, 6)
        context["open_interest_change_pct"] = round(oi_change_pct, 4)
    else:
        context["open_interest_value"] = None
        context["open_interest_change_pct"] = None
        context["data_quality_notes"].append("open-interest-unavailable")
    return context
