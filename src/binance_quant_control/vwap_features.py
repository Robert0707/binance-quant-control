from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_vwap_bands(df: pd.DataFrame, *, window: int = 48) -> pd.DataFrame:
    """Rolling VWAP bands for crypto sessions where exchange hours do not reset.

    TradingView's VWAP concept is volume-weighted fair value with optional
    standard-deviation bands. Crypto futures do not have a clean cash-session
    open, so this feature uses a rolling anchor to keep the signal adaptive.
    """

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = df["volume"].replace(0, np.nan)
    rolling_volume = volume.rolling(window).sum().replace(0, np.nan)
    rolling_vwap = (typical_price * volume).rolling(window).sum() / rolling_volume
    weighted_second_moment = ((typical_price**2) * volume).rolling(window).sum() / rolling_volume
    weighted_variance = weighted_second_moment - rolling_vwap**2
    deviation = np.sqrt(weighted_variance.clip(lower=0.0))
    close_position = (
        (df["close"] - df["low"])
        / (df["high"] - df["low"]).where((df["high"] - df["low"]) > 0)
    ).clip(lower=0.0, upper=1.0)

    lower_1 = rolling_vwap - deviation
    upper_1 = rolling_vwap + deviation
    lower_2 = rolling_vwap - deviation * 2.0
    upper_2 = rolling_vwap + deviation * 2.0
    previous_close = df["close"].shift(1)

    long_reclaim = (
        ((df["low"] <= lower_2) | (previous_close < lower_1.shift(1)))
        & (df["close"] > lower_1)
        & (close_position >= 0.56)
    )
    short_reclaim = (
        ((df["high"] >= upper_2) | (previous_close > upper_1.shift(1)))
        & (df["close"] < upper_1)
        & (close_position <= 0.44)
    )
    mid_reclaim_long = (previous_close < rolling_vwap.shift(1)) & (df["close"] > rolling_vwap) & (
        close_position >= 0.58
    )
    mid_reclaim_short = (previous_close > rolling_vwap.shift(1)) & (df["close"] < rolling_vwap) & (
        close_position <= 0.42
    )

    return pd.DataFrame(
        {
            f"vwap_rolling_{window}": rolling_vwap,
            f"vwap_deviation_{window}": deviation,
            f"vwap_upper_1_{window}": upper_1,
            f"vwap_lower_1_{window}": lower_1,
            f"vwap_upper_2_{window}": upper_2,
            f"vwap_lower_2_{window}": lower_2,
            f"vwap_distance_pct_{window}": (
                (df["close"] - rolling_vwap) / rolling_vwap.replace(0, np.nan)
            )
            * 100.0,
            f"vwap_reclaim_long_{window}": long_reclaim.fillna(False),
            f"vwap_reclaim_short_{window}": short_reclaim.fillna(False),
            f"vwap_mid_reclaim_long_{window}": mid_reclaim_long.fillna(False),
            f"vwap_mid_reclaim_short_{window}": mid_reclaim_short.fillna(False),
        },
        index=df.index,
    )
