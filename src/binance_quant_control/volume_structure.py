from __future__ import annotations

from typing import Any

import pandas as pd


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(parsed):
        return default
    return parsed


def _safe_quantile(series: pd.Series, percentile: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    return float(clean.quantile(max(0.0, min(100.0, percentile)) / 100.0))


def summarize_volume_profile(
    df: pd.DataFrame,
    *,
    rows: int = 20,
    lookback: int = 240,
) -> dict[str, Any]:
    """Approximate TradingIQ-style volume profile levels from local candles.

    TradingView's lower-timeframe allocation is not exactly reproducible from
    OHLCV alone, so this deterministic version allocates each candle's volume
    across touched price rows and derives POC/VAH/VAL for gating.
    """

    if df.empty:
        return {"available": False, "reason": "empty-frame"}
    window = df.tail(max(int(lookback), int(rows), 1))
    high = _float(window["high"].max())
    low = _float(window["low"].min())
    close = _float(window["close"].iloc[-1])
    if high <= low or close <= 0.0:
        return {"available": False, "reason": "invalid-price-range"}

    row_count = max(5, min(200, int(rows)))
    step = (high - low) / row_count
    total_vol = [0.0 for _ in range(row_count)]
    buy_vol = [0.0 for _ in range(row_count)]
    sell_vol = [0.0 for _ in range(row_count)]

    for _, candle in window.iterrows():
        candle_high = _float(candle.get("high"))
        candle_low = _float(candle.get("low"))
        volume = _float(candle.get("volume"))
        if candle_high <= candle_low or volume <= 0.0:
            continue
        direction = 1.0 if _float(candle.get("close")) >= _float(candle.get("open")) else -1.0
        dn_idx = max(0, min(row_count - 1, int((candle_low - low) / step)))
        up_idx = max(0, min(row_count - 1, int((candle_high - low) / step)))
        span = max(1, up_idx - dn_idx + 1)
        allocated = volume / span
        for idx in range(dn_idx, up_idx + 1):
            total_vol[idx] += allocated
            if direction >= 0:
                buy_vol[idx] += allocated
            else:
                sell_vol[idx] += allocated

    if not any(total_vol):
        return {"available": False, "reason": "zero-volume"}
    poc_idx = max(range(row_count), key=lambda idx: total_vol[idx])
    total_volume = sum(total_vol)
    target_value_area = total_volume * 0.7
    selected = {poc_idx}
    selected_volume = total_vol[poc_idx]
    up_idx = poc_idx + 1
    down_idx = poc_idx - 1
    while selected_volume < target_value_area and (down_idx >= 0 or up_idx < row_count):
        up_volume = total_vol[up_idx] if up_idx < row_count else -1.0
        down_volume = total_vol[down_idx] if down_idx >= 0 else -1.0
        if up_volume >= down_volume:
            selected.add(up_idx)
            selected_volume += max(up_volume, 0.0)
            up_idx += 1
        else:
            selected.add(down_idx)
            selected_volume += max(down_volume, 0.0)
            down_idx -= 1

    vah_idx = max(selected)
    val_idx = min(selected)
    poc = low + (poc_idx + 0.5) * step
    vah = low + (vah_idx + 1.0) * step
    val = low + val_idx * step
    buy_total = sum(buy_vol)
    sell_total = sum(sell_vol)
    delta = buy_total - sell_total
    if close > vah:
        position = "above-value"
    elif close < val:
        position = "below-value"
    elif abs(close - poc) <= step:
        position = "near-poc"
    else:
        position = "inside-value"

    return {
        "available": True,
        "rows": row_count,
        "lookback": len(window),
        "poc": round(poc, 8),
        "vah": round(vah, 8),
        "val": round(val, 8),
        "close_position": position,
        "close_to_poc_pct": round(((close - poc) / close) * 100.0, 4),
        "value_area_width_pct": round(((vah - val) / close) * 100.0, 4),
        "buy_volume": round(buy_total, 4),
        "sell_volume": round(sell_total, 4),
        "delta": round(delta, 4),
        "delta_ratio": round(delta / total_volume, 6) if total_volume else 0.0,
        "total_volume": round(total_volume, 4),
    }


def summarize_volume_bubbles(
    df: pd.DataFrame,
    *,
    short_len: int = 20,
    mid_len: int = 50,
    long_len: int = 100,
    small_pct: float = 75.0,
    medium_pct: float = 90.0,
    big_pct: float = 97.0,
) -> dict[str, Any]:
    """Detect QuantAlgo-style elevated volume/delta clusters."""

    if df.empty or "volume" not in df:
        return {"available": False, "reason": "volume-unavailable"}
    window = df.copy()
    volume = pd.to_numeric(window["volume"], errors="coerce").fillna(0.0)
    if float(volume.sum()) <= 0.0:
        return {"available": False, "reason": "zero-volume"}

    candle_delta = volume.where(window["close"] >= window["open"], -volume)
    abs_delta = candle_delta.abs()
    latest_volume = _float(volume.iloc[-1])
    latest_delta = _float(candle_delta.iloc[-1])
    checks = {
        "short": volume.tail(short_len),
        "mid": volume.tail(mid_len),
        "long": volume.tail(long_len),
    }
    delta_checks = {
        "short": abs_delta.tail(short_len),
        "mid": abs_delta.tail(mid_len),
        "long": abs_delta.tail(long_len),
    }

    def consensus(series_map: dict[str, pd.Series], percentile: float, value: float) -> int:
        return sum(1 for series in series_map.values() if value >= _safe_quantile(series, percentile))

    vol_big_hits = consensus(checks, big_pct, latest_volume)
    vol_med_hits = consensus(checks, medium_pct, latest_volume)
    vol_small_hits = consensus(checks, small_pct, latest_volume)
    delta_big_hits = consensus(delta_checks, big_pct, abs(latest_delta))
    delta_med_hits = consensus(delta_checks, medium_pct, abs(latest_delta))
    delta_small_hits = consensus(delta_checks, small_pct, abs(latest_delta))

    if vol_big_hits >= 2 or delta_big_hits >= 2:
        cluster = "big"
    elif vol_med_hits >= 2 or delta_med_hits >= 2:
        cluster = "medium"
    elif vol_small_hits >= 2 or delta_small_hits >= 2:
        cluster = "small"
    else:
        cluster = "none"

    if latest_delta > 0 and _float(window["close"].iloc[-1]) >= _float(window["open"].iloc[-1]):
        side = "buy"
    elif latest_delta < 0 and _float(window["close"].iloc[-1]) < _float(window["open"].iloc[-1]):
        side = "sell"
    elif cluster != "none":
        side = "mixed"
    else:
        side = "neutral"

    avg_mid_volume = _float(volume.tail(mid_len).mean())
    return {
        "available": True,
        "cluster": cluster,
        "side": side,
        "volume": round(latest_volume, 4),
        "delta": round(latest_delta, 4),
        "volume_ratio": round(latest_volume / avg_mid_volume, 4) if avg_mid_volume > 0 else 0.0,
        "vol_consensus_hits": {
            "small": vol_small_hits,
            "medium": vol_med_hits,
            "big": vol_big_hits,
        },
        "delta_consensus_hits": {
            "small": delta_small_hits,
            "medium": delta_med_hits,
            "big": delta_big_hits,
        },
    }


def summarize_htf_volume_imbalance(
    df: pd.DataFrame,
    *,
    lookback: int = 96,
    percentile: float = 95.0,
) -> dict[str, Any]:
    """Approximate LuxAlgo-style HTF volume spike + imbalance projection.

    It looks for the latest high-percentile volume event and reports whether
    that event projects a bullish or bearish imbalance zone that price has not
    fully invalidated yet.
    """

    required = {"open", "high", "low", "close", "volume"}
    if df.empty or not required.issubset(df.columns):
        return {"available": False, "reason": "missing-ohlcv"}
    window = df.tail(max(10, int(lookback))).copy()
    volume = pd.to_numeric(window["volume"], errors="coerce").fillna(0.0)
    if float(volume.sum()) <= 0.0:
        return {"available": False, "reason": "zero-volume"}

    threshold = _safe_quantile(volume, percentile)
    spike_mask = volume >= threshold
    if not bool(spike_mask.any()):
        return {"available": True, "active": False, "reason": "no-spike"}

    spike_idx = spike_mask[spike_mask].index[-1]
    spike = window.loc[spike_idx]
    close = _float(window["close"].iloc[-1])
    open_price = _float(spike["open"])
    high = _float(spike["high"])
    low = _float(spike["low"])
    spike_close = _float(spike["close"])
    spike_volume = _float(spike["volume"])
    body_mid = (open_price + spike_close) / 2.0
    direction = "bullish" if spike_close >= open_price else "bearish"
    zone_low = min(open_price, spike_close, body_mid)
    zone_high = max(open_price, spike_close, body_mid)
    if direction == "bullish":
        invalidated = close < low
        distance_pct = ((close - zone_high) / close) * 100.0 if close > 0 else 0.0
    else:
        invalidated = close > high
        distance_pct = ((zone_low - close) / close) * 100.0 if close > 0 else 0.0

    return {
        "available": True,
        "active": not invalidated,
        "direction": direction,
        "spike_time": str(spike_idx),
        "spike_volume": round(spike_volume, 4),
        "threshold_volume": round(threshold, 4),
        "volume_ratio": round(spike_volume / threshold, 4) if threshold > 0 else 0.0,
        "zone_low": round(zone_low, 8),
        "zone_high": round(zone_high, 8),
        "invalidated": invalidated,
        "distance_pct": round(distance_pct, 4),
    }
