from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .indicators import interval_bars_per_year, true_range, zscore


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def _regime_bucket(value: float, *, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "mid"


def _signed_bucket(value: float, *, threshold: float = 0.0) -> str:
    if value > threshold:
        return "positive"
    if value < -threshold:
        return "negative"
    return "neutral"


def build_ai_ml_features(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Point-in-time ML features for AI trader gates.

    These columns are not human chart signals. They describe state, payoff
    potential, and execution quality so later models can decide whether to
    exploit, explore, or skip a candidate.
    """

    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    log_ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    bars_per_year = interval_bars_per_year(interval)
    tr = true_range(df).astype(float)
    atr_14 = df["atr_14"] if "atr_14" in df.columns else tr.ewm(alpha=1 / 14, adjust=False).mean()
    atr_pct = atr_14 / close.replace(0, np.nan)
    realized_vol_20 = log_ret.rolling(20).std() * math.sqrt(bars_per_year)
    realized_vol_80 = log_ret.rolling(80).std() * math.sqrt(bars_per_year)
    vol_ratio = realized_vol_20 / realized_vol_80.replace(0, np.nan)
    range_position_80 = (close - low.rolling(80).min()) / (high.rolling(80).max() - low.rolling(80).min()).replace(0, np.nan)
    trend_slope_20 = close.pct_change(20)
    trend_slope_80 = close.pct_change(80)
    trend_efficiency_20 = (close - close.shift(20)).abs() / close.diff().abs().rolling(20).sum().replace(0, np.nan)
    volume_z_20 = zscore(volume, 20)
    quote_proxy = close * volume
    quote_volume_z_20 = zscore(quote_proxy, 20)
    candle_body_pct = (close - df["open"].astype(float)).abs() / close.replace(0, np.nan)
    candle_range_pct = (high - low) / close.replace(0, np.nan)
    wick_imbalance = ((high - close).abs() - (close - low).abs()) / (high - low).replace(0, np.nan)
    turbulence_20 = (log_ret.abs() / log_ret.rolling(20).std().replace(0, np.nan)).rolling(5).mean()

    out["ml_return_1"] = log_ret
    out["ml_return_5"] = close.pct_change(5)
    out["ml_return_20"] = close.pct_change(20)
    out["ml_atr_pct_14"] = atr_pct
    out["ml_realized_vol_20"] = realized_vol_20
    out["ml_realized_vol_80"] = realized_vol_80
    out["ml_vol_ratio_20_80"] = vol_ratio
    out["ml_trend_slope_20"] = trend_slope_20
    out["ml_trend_slope_80"] = trend_slope_80
    out["ml_trend_efficiency_20"] = trend_efficiency_20
    out["ml_range_position_80"] = range_position_80
    out["ml_volume_z_20"] = volume_z_20
    out["ml_quote_volume_z_20"] = quote_volume_z_20
    out["ml_candle_body_pct"] = candle_body_pct
    out["ml_candle_range_pct"] = candle_range_pct
    out["ml_wick_imbalance"] = wick_imbalance
    out["ml_turbulence_20"] = turbulence_20
    out["ml_liquidity_pressure"] = (quote_volume_z_20.fillna(0.0) + volume_z_20.fillna(0.0)) / 2.0
    out["ml_payoff_potential_long"] = (high.rolling(20).max().shift(1) - close) / atr_14.replace(0, np.nan)
    out["ml_payoff_potential_short"] = (close - low.rolling(20).min().shift(1)) / atr_14.replace(0, np.nan)

    regimes = pd.DataFrame(index=df.index)
    regimes["ml_volatility_regime"] = out["ml_vol_ratio_20_80"].map(
        lambda value: _regime_bucket(_finite(value, 1.0), low=0.85, high=1.25)
    )
    regimes["ml_trend_regime"] = out["ml_trend_slope_80"].map(
        lambda value: _signed_bucket(_finite(value), threshold=0.03)
    )
    regimes["ml_liquidity_regime"] = out["ml_liquidity_pressure"].map(
        lambda value: _regime_bucket(_finite(value), low=-0.5, high=1.0)
    )
    regimes["ml_turbulence_regime"] = out["ml_turbulence_20"].map(
        lambda value: _regime_bucket(_finite(value), low=0.8, high=1.5)
    )
    regimes["ml_session_bucket"] = [
        "asia" if ts.hour < 8 else "europe" if ts.hour < 16 else "us"
        for ts in pd.to_datetime(df.index, utc=True)
    ]
    return pd.concat([out, regimes], axis=1)
