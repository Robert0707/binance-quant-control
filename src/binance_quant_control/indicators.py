from __future__ import annotations

import math

import numpy as np
import pandas as pd


def interval_bars_per_year(interval: str) -> float:
    mapping = {
        "1m": 60 * 24 * 365,
        "3m": 20 * 24 * 365,
        "5m": 12 * 24 * 365,
        "15m": 4 * 24 * 365,
        "30m": 2 * 24 * 365,
        "1h": 24 * 365,
        "2h": 12 * 365,
        "4h": 6 * 365,
        "6h": 4 * 365,
        "8h": 3 * 365,
        "12h": 2 * 365,
        "1d": 365,
        "3d": 365 / 3,
        "1w": 52,
    }
    return mapping.get(interval, 24 * 365)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def macd(series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    fast = ema(series, fast_period)
    slow = ema(series, slow_period)
    line = fast - slow
    signal = ema(line, signal_period)
    hist = line - signal
    return pd.DataFrame({"macd": line, "macd_signal": signal, "macd_hist": hist})


def bollinger_bands(series: pd.Series, window: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    basis = sma(series, window)
    deviation = series.rolling(window).std()
    upper = basis + std_dev * deviation
    lower = basis - std_dev * deviation
    bandwidth = (upper - lower) / basis.replace(0, np.nan)
    percent_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame(
        {
            "bb_basis": basis,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
            "bb_percent_b": percent_b,
        }
    )


def stochastic_oscillator(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    lowest_low = df["low"].rolling(k_window).min()
    highest_high = df["high"].rolling(k_window).max()
    percent_k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    percent_d = percent_k.rolling(d_window).mean()
    return pd.DataFrame({"stoch_k": percent_k, "stoch_d": percent_d})


def stoch_rsi(
    series: pd.Series,
    *,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> pd.DataFrame:
    rsi_value = rsi(series, rsi_period)
    lowest = rsi_value.rolling(stoch_period).min()
    highest = rsi_value.rolling(stoch_period).max()
    raw = 100.0 * (rsi_value - lowest) / (highest - lowest).replace(0, np.nan)
    k = raw.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return pd.DataFrame({"stoch_rsi_k": k, "stoch_rsi_d": d})


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume.fillna(0.0)).cumsum()


def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    mean_tp = typical_price.rolling(window).mean()
    mad = typical_price.rolling(window).apply(lambda values: np.mean(np.abs(values - np.mean(values))), raw=True)
    return (typical_price - mean_tp) / (0.015 * mad.replace(0, np.nan))


def williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    highest_high = df["high"].rolling(window).max()
    lowest_low = df["low"].rolling(window).min()
    return -100 * (highest_high - df["close"]) / (highest_high - lowest_low).replace(0, np.nan)


def money_flow_index(df: pd.DataFrame, window: int = 14) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    raw_flow = typical_price * df["volume"]
    delta = typical_price.diff()
    positive = raw_flow.where(delta > 0, 0.0)
    negative = raw_flow.where(delta < 0, 0.0)
    positive_sum = positive.rolling(window).sum()
    negative_sum = negative.abs().rolling(window).sum()
    ratio = positive_sum / negative_sum.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))


def adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_value = true_range.ewm(alpha=1 / window, adjust=False).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr_value.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr_value.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    adx_line = dx.ewm(alpha=1 / window, adjust=False).mean()
    return pd.DataFrame({"adx": adx_line, "plus_di": plus_di, "minus_di": minus_di})


def vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_volume = df["volume"].cumsum().replace(0, np.nan)
    return (typical_price * df["volume"]).cumsum() / cumulative_volume


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def realized_volatility(close: pd.Series, interval: str, window: int = 20) -> pd.Series:
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window).std() * math.sqrt(interval_bars_per_year(interval))


def vwma(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    weighted = close * volume
    return weighted.rolling(window).sum() / volume.rolling(window).sum().replace(0, np.nan)


def donchian_channels(
    df: pd.DataFrame,
    *,
    upper_window: int = 20,
    lower_window: int = 20,
) -> pd.DataFrame:
    upper = df["high"].rolling(upper_window).max().shift(1)
    lower = df["low"].rolling(lower_window).min().shift(1)
    mid = (upper + lower) / 2.0
    width_pct = (upper - lower) / df["close"].replace(0, np.nan)
    return pd.DataFrame(
        {
            "donchian_upper": upper,
            "donchian_lower": lower,
            "donchian_mid": mid,
            "donchian_width_pct": width_pct,
            "donchian_breakout_up": df["high"] >= upper,
            "donchian_breakout_down": df["low"] <= lower,
        }
    )


def keltner_channels(
    df: pd.DataFrame,
    *,
    window: int = 20,
    multiplier: float = 1.5,
) -> pd.DataFrame:
    basis = ema(df["close"], window)
    atr_value = atr(df, window)
    upper = basis + multiplier * atr_value
    lower = basis - multiplier * atr_value
    width_pct = (upper - lower) / basis.replace(0, np.nan)
    return pd.DataFrame(
        {
            "keltner_basis": basis,
            "keltner_upper": upper,
            "keltner_lower": lower,
            "keltner_width_pct": width_pct,
        }
    )


def squeeze_momentum(
    df: pd.DataFrame,
    *,
    length: int = 20,
    bb_mult: float = 2.0,
    kc_mult: float = 1.5,
    use_true_range: bool = True,
    match_pasted_source: bool = True,
) -> pd.DataFrame:
    """LazyBear-style Squeeze Momentum translated from the provided Pine.

    The user's pasted Pine snippet uses `multKC` for the Bollinger deviation
    line, even though it also defines a separate BB multiplier input. The
    default keeps that strict pasted-source behavior; set
    `match_pasted_source=False` to use the canonical BB multiplier instead.
    """

    source = df["close"]
    basis = sma(source, length)
    dev_multiplier = kc_mult if match_pasted_source else bb_mult
    deviation = dev_multiplier * source.rolling(length).std(ddof=0)
    upper_bb = basis + deviation
    lower_bb = basis - deviation

    ma = sma(source, length)
    range_series = true_range(df) if use_true_range else df["high"] - df["low"]
    range_ma = sma(range_series, length)
    upper_kc = ma + range_ma * kc_mult
    lower_kc = ma - range_ma * kc_mult

    squeeze_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    squeeze_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
    squeeze_no = ~(squeeze_on | squeeze_off)
    highest = df["high"].rolling(length).max()
    lowest = df["low"].rolling(length).min()
    baseline = ((highest + lowest) / 2.0 + sma(df["close"], length)) / 2.0
    detrended = df["close"] - baseline
    x = np.arange(length, dtype=float)

    def linreg_last(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        slope, intercept = np.polyfit(x, values, 1)
        return float(intercept + slope * (length - 1))

    momentum = detrended.rolling(length).apply(linreg_last, raw=True)
    return pd.DataFrame(
        {
            "squeeze_on": squeeze_on,
            "squeeze_off": squeeze_off,
            "squeeze_no": squeeze_no,
            "squeeze_momentum": momentum,
            "squeeze_released": squeeze_on.shift(1, fill_value=False) & squeeze_off,
            "squeeze_bb_upper": upper_bb,
            "squeeze_bb_lower": lower_bb,
            "squeeze_kc_upper": upper_kc,
            "squeeze_kc_lower": lower_kc,
        }
    )


def chandelier_exit(
    df: pd.DataFrame,
    *,
    length: int = 22,
    multiplier: float = 3.0,
    use_close: bool = True,
) -> pd.DataFrame:
    atr_stop = atr(df, length) * multiplier
    high_source = df["close"] if use_close else df["high"]
    low_source = df["close"] if use_close else df["low"]
    raw_long_stop = high_source.rolling(length).max() - atr_stop
    raw_short_stop = low_source.rolling(length).min() + atr_stop
    long_stop: list[float] = []
    short_stop: list[float] = []
    direction: list[int] = []
    for idx in range(len(df)):
        close = float(df["close"].iloc[idx])
        prev_close = float(df["close"].iloc[idx - 1]) if idx > 0 else close
        raw_long = float(raw_long_stop.iloc[idx]) if not pd.isna(raw_long_stop.iloc[idx]) else close
        raw_short = float(raw_short_stop.iloc[idx]) if not pd.isna(raw_short_stop.iloc[idx]) else close
        if idx == 0:
            long_now = raw_long
            short_now = raw_short
            dir_now = 1
        else:
            prev_long = long_stop[-1]
            prev_short = short_stop[-1]
            long_now = max(raw_long, prev_long) if prev_close > prev_long else raw_long
            short_now = min(raw_short, prev_short) if prev_close < prev_short else raw_short
            prev_dir = direction[-1]
            if close > prev_short:
                dir_now = 1
            elif close < prev_long:
                dir_now = -1
            else:
                dir_now = prev_dir
        long_stop.append(long_now)
        short_stop.append(short_now)
        direction.append(dir_now)
    direction_series = pd.Series(direction, index=df.index)
    return pd.DataFrame(
        {
            "chandelier_long_stop": pd.Series(long_stop, index=df.index),
            "chandelier_short_stop": pd.Series(short_stop, index=df.index),
            "chandelier_direction": direction_series,
            "chandelier_buy": (direction_series == 1) & (direction_series.shift(1) == -1),
            "chandelier_sell": (direction_series == -1) & (direction_series.shift(1) == 1),
        }
    )


def _crosses(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    prev_a = series_a.shift(1)
    prev_b = series_b.shift(1)
    return ((prev_a <= prev_b) & (series_a > series_b)) | ((prev_a >= prev_b) & (series_a < series_b))


def _calculate_qqe(
    close: pd.Series,
    *,
    rsi_length: int = 6,
    smoothing: int = 5,
    qqe_factor: float = 3.0,
) -> pd.DataFrame:
    wilders_length = rsi_length * 2 - 1
    smoothed_rsi = ema(rsi(close, rsi_length).fillna(50.0), smoothing)
    atr_rsi = smoothed_rsi.diff().abs()
    dynamic_atr = ema(atr_rsi, wilders_length) * qqe_factor
    long_band: list[float] = []
    short_band: list[float] = []
    directions: list[int] = []
    for idx in range(len(close)):
        rsi_now = float(smoothed_rsi.iloc[idx])
        delta = float(dynamic_atr.iloc[idx]) if not pd.isna(dynamic_atr.iloc[idx]) else 0.0
        new_long = rsi_now - delta
        new_short = rsi_now + delta
        if idx == 0:
            long_now = new_long
            short_now = new_short
            direction = 0
        else:
            prev_rsi = float(smoothed_rsi.iloc[idx - 1])
            prev_long = long_band[-1]
            prev_short = short_band[-1]
            long_now = max(prev_long, new_long) if prev_rsi > prev_long and rsi_now > prev_long else new_long
            short_now = min(prev_short, new_short) if prev_rsi < prev_short and rsi_now < prev_short else new_short
            if rsi_now > prev_short and prev_rsi <= prev_short:
                direction = 1
            elif rsi_now < prev_long and prev_rsi >= prev_long:
                direction = -1
            else:
                direction = directions[-1]
        long_band.append(long_now)
        short_band.append(short_now)
        directions.append(direction)
    direction_series = pd.Series(directions, index=close.index)
    return pd.DataFrame(
        {
            "trend_line": pd.Series(
                [
                    long_band[idx] if directions[idx] == 1 else short_band[idx]
                    for idx in range(len(close))
                ],
                index=close.index,
            ),
            "rsi": smoothed_rsi,
            "long_band": pd.Series(long_band, index=close.index),
            "short_band": pd.Series(short_band, index=close.index),
            "direction": direction_series,
        }
    )


def qqe_mod(
    close: pd.Series,
    *,
    rsi_length: int = 6,
    smoothing: int = 5,
    qqe_factor: float = 3.0,
    threshold: float = 3.0,
    secondary_rsi_length: int = 6,
    secondary_smoothing: int = 5,
    secondary_qqe_factor: float = 1.61,
    secondary_threshold: float = 3.0,
    bollinger_length: int = 50,
    bollinger_multiplier: float = 0.35,
) -> pd.DataFrame:
    primary = _calculate_qqe(
        close,
        rsi_length=rsi_length,
        smoothing=smoothing,
        qqe_factor=qqe_factor,
    )
    secondary = _calculate_qqe(
        close,
        rsi_length=secondary_rsi_length,
        smoothing=secondary_smoothing,
        qqe_factor=secondary_qqe_factor,
    )
    primary_centered = primary["trend_line"] - 50.0
    basis = sma(primary_centered, bollinger_length)
    deviation = bollinger_multiplier * primary_centered.rolling(bollinger_length).std(ddof=0)
    upper = basis + deviation
    lower = basis - deviation
    primary_rsi_centered = primary["rsi"] - 50.0
    secondary_rsi_centered = secondary["rsi"] - 50.0
    up_signal = (secondary_rsi_centered > secondary_threshold) & (primary_rsi_centered > upper)
    down_signal = (secondary_rsi_centered < -secondary_threshold) & (primary_rsi_centered < lower)
    signal_direction = pd.Series(0, index=close.index)
    signal_direction = signal_direction.mask(secondary["direction"] > 0, 1)
    signal_direction = signal_direction.mask(secondary["direction"] < 0, -1)
    signal_direction = signal_direction.mask(up_signal, 1).mask(down_signal, -1)
    return pd.DataFrame(
        {
            "qqe_rsi": secondary["rsi"],
            "qqe_long_band": secondary["long_band"],
            "qqe_short_band": secondary["short_band"],
            "qqe_direction": signal_direction,
            "qqe_up_signal": up_signal,
            "qqe_down_signal": down_signal,
            "qqe_primary_trend_line": primary["trend_line"],
            "qqe_primary_rsi": primary["rsi"],
            "qqe_primary_direction": primary["direction"],
            "qqe_secondary_trend_line": secondary["trend_line"],
            "qqe_secondary_rsi": secondary["rsi"],
            "qqe_secondary_direction": secondary["direction"],
            "qqe_bollinger_basis": basis,
            "qqe_bollinger_upper": upper,
            "qqe_bollinger_lower": lower,
            "qqe_primary_cross_up": (primary["rsi"].shift(1) <= 50.0) & (primary["rsi"] > 50.0),
            "qqe_primary_cross_down": (primary["rsi"].shift(1) >= 50.0) & (primary["rsi"] < 50.0),
        }
    )


def ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    tenkan = (df["high"].rolling(9).max() + df["low"].rolling(9).min()) / 2.0
    kijun = (df["high"].rolling(26).max() + df["low"].rolling(26).min()) / 2.0
    span_a = ((tenkan + kijun) / 2.0).shift(26)
    span_b = ((df["high"].rolling(52).max() + df["low"].rolling(52).min()) / 2.0).shift(26)
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)
    bullish = (df["close"] > cloud_top) & (tenkan > kijun)
    bearish = (df["close"] < cloud_bottom) & (tenkan < kijun)
    direction = pd.Series(0, index=df.index)
    direction = direction.mask(bullish, 1).mask(bearish, -1)
    return pd.DataFrame(
        {
            "ichimoku_tenkan": tenkan,
            "ichimoku_kijun": kijun,
            "ichimoku_span_a": span_a,
            "ichimoku_span_b": span_b,
            "ichimoku_direction": direction,
        }
    )


def parabolic_sar(
    df: pd.DataFrame,
    *,
    step: float = 0.02,
    max_step: float = 0.2,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"psar": [], "psar_direction": []}, index=df.index)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    bullish = True
    af = step
    ep = float(high.iloc[0])
    sar = float(low.iloc[0])
    values: list[float] = []
    directions: list[int] = []
    for idx in range(len(df)):
        if idx == 0:
            values.append(sar)
            directions.append(1)
            continue
        sar = sar + af * (ep - sar)
        if bullish:
            sar = min(sar, float(low.iloc[idx - 1]))
            if idx > 1:
                sar = min(sar, float(low.iloc[idx - 2]))
            if low.iloc[idx] < sar:
                bullish = False
                sar = ep
                ep = float(low.iloc[idx])
                af = step
            else:
                if high.iloc[idx] > ep:
                    ep = float(high.iloc[idx])
                    af = min(af + step, max_step)
        else:
            sar = max(sar, float(high.iloc[idx - 1]))
            if idx > 1:
                sar = max(sar, float(high.iloc[idx - 2]))
            if high.iloc[idx] > sar:
                bullish = True
                sar = ep
                ep = float(high.iloc[idx])
                af = step
            else:
                if low.iloc[idx] < ep:
                    ep = float(low.iloc[idx])
                    af = min(af + step, max_step)
        values.append(sar)
        directions.append(1 if close.iloc[idx] >= sar else -1)
    return pd.DataFrame(
        {
            "psar": pd.Series(values, index=df.index),
            "psar_direction": pd.Series(directions, index=df.index),
        }
    )


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Regular SuperTrend approximation used as a deterministic trend filter."""

    hl2 = (df["high"] + df["low"]) / 2.0
    atr_value = atr(df, period)
    basic_upper = hl2 + multiplier * atr_value
    basic_lower = hl2 - multiplier * atr_value
    final_upper: list[float] = []
    final_lower: list[float] = []
    trends: list[int] = []
    lines: list[float] = []

    for idx in range(len(df)):
        close = float(df["close"].iloc[idx])
        prev_close = float(df["close"].iloc[idx - 1]) if idx > 0 else close
        upper = float(basic_upper.iloc[idx])
        lower = float(basic_lower.iloc[idx])
        if idx == 0 or np.isnan(upper) or np.isnan(lower):
            final_upper.append(upper)
            final_lower.append(lower)
            trends.append(1)
            lines.append(lower)
            continue

        prev_upper = final_upper[-1]
        prev_lower = final_lower[-1]
        upper = upper if upper < prev_upper or prev_close > prev_upper else prev_upper
        lower = lower if lower > prev_lower or prev_close < prev_lower else prev_lower
        trend = trends[-1]
        if trends[-1] == -1 and close > prev_upper:
            trend = 1
        elif trends[-1] == 1 and close < prev_lower:
            trend = -1
        final_upper.append(upper)
        final_lower.append(lower)
        trends.append(trend)
        lines.append(lower if trend == 1 else upper)

    return pd.DataFrame(
        {
            "supertrend": pd.Series(lines, index=df.index),
            "supertrend_direction": pd.Series(trends, index=df.index),
        }
    )


def trend_magic(
    df: pd.DataFrame,
    *,
    cci_period: int = 20,
    atr_period: int = 5,
    atr_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Trend Magic style CCI + ATR trailing line."""

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_sma = true_range.rolling(atr_period).mean()
    cci_value = cci(df, cci_period)
    upper_trail = df["low"] - atr_sma * atr_multiplier
    lower_trail = df["high"] + atr_sma * atr_multiplier
    magic: list[float] = []
    directions: list[int] = []

    for idx in range(len(df)):
        cci_now = float(cci_value.iloc[idx]) if not pd.isna(cci_value.iloc[idx]) else 0.0
        up_now = float(upper_trail.iloc[idx]) if not pd.isna(upper_trail.iloc[idx]) else float(df["low"].iloc[idx])
        down_now = float(lower_trail.iloc[idx]) if not pd.isna(lower_trail.iloc[idx]) else float(df["high"].iloc[idx])
        previous = magic[-1] if magic else float(df["close"].iloc[idx])
        if cci_now >= 0.0:
            current = previous if up_now < previous else up_now
            direction = 1
        else:
            current = previous if down_now > previous else down_now
            direction = -1
        magic.append(current)
        directions.append(direction)

    return pd.DataFrame(
        {
            "trend_magic": pd.Series(magic, index=df.index),
            "trend_magic_direction": pd.Series(directions, index=df.index),
        }
    )


def follow_line(
    df: pd.DataFrame,
    *,
    atr_period: int = 5,
    bb_period: int = 21,
    bb_deviation: float = 1.0,
    use_atr_filter: bool = True,
) -> pd.DataFrame:
    """Follow Line style Bollinger breakout trail with optional ATR buffer."""

    bb = bollinger_bands(df["close"], bb_period, bb_deviation)
    atr_value = atr(df, atr_period)
    lines: list[float] = []
    trends: list[int] = []
    bb_signal = 0

    for idx in range(len(df)):
        close = float(df["close"].iloc[idx])
        high = float(df["high"].iloc[idx])
        low = float(df["low"].iloc[idx])
        upper = float(bb["bb_upper"].iloc[idx]) if not pd.isna(bb["bb_upper"].iloc[idx]) else np.nan
        lower = float(bb["bb_lower"].iloc[idx]) if not pd.isna(bb["bb_lower"].iloc[idx]) else np.nan
        atr_now = float(atr_value.iloc[idx]) if not pd.isna(atr_value.iloc[idx]) else 0.0
        previous_line = lines[-1] if lines else close

        if not np.isnan(upper) and close > upper:
            bb_signal = 1
        elif not np.isnan(lower) and close < lower:
            bb_signal = -1

        if bb_signal == 1:
            line = low - atr_now if use_atr_filter else low
            line = max(line, previous_line)
        elif bb_signal == -1:
            line = high + atr_now if use_atr_filter else high
            line = min(line, previous_line)
        else:
            line = previous_line

        previous_trend = trends[-1] if trends else 0
        if line > previous_line:
            trend = 1
        elif line < previous_line:
            trend = -1
        else:
            trend = previous_trend
        lines.append(line)
        trends.append(trend)

    trend_series = pd.Series(trends, index=df.index)
    return pd.DataFrame(
        {
            "follow_line": pd.Series(lines, index=df.index),
            "follow_line_direction": trend_series,
            "follow_line_buy": (trend_series.shift(1) == -1) & (trend_series == 1),
            "follow_line_sell": (trend_series.shift(1) == 1) & (trend_series == -1),
        }
    )


def _bounded_norm(series: pd.Series, scale: float) -> pd.Series:
    scaled = series * scale
    return scaled / (1.0 + scaled.abs())


def jumbo_power(df: pd.DataFrame) -> pd.DataFrame:
    """JUMBO Pro inspired composite power score.

    The output intentionally keeps the TradingView idea as a transparent local
    feature rather than a black-box signal: trend, volume, momentum, VWAP,
    Bollinger position, and ATR volatility are exposed separately.
    """

    ema_fast = ema(df["close"], 13)
    ema_slow = ema(df["close"], 55)
    macd_frame = macd(df["close"], 12, 26, 9)
    adx_frame = adx(df, 14)
    bb = bollinger_bands(df["close"], 20, 2.0)
    st = supertrend(df, 10, 3.0)
    atr_value = atr(df, 14)
    rsi_value = rsi(df["close"], 14).fillna(50.0)
    mfi_value = money_flow_index(df, 14).fillna(50.0)
    obv_value = on_balance_volume(df["close"], df["volume"])
    obv_ema = ema(obv_value, 20)
    obv_delta = obv_ema.diff().fillna(0.0)
    obv_trend = ema(obv_delta, 5)
    volume_sma = sma(df["volume"], 20)
    volume_ratio = df["volume"] / volume_sma.replace(0, np.nan)
    volume_confirmed = (volume_ratio > 1.2) & (df["volume"] > df["volume"].shift(1))
    vwap_value = vwap(df)

    ema_slope = (ema_fast - ema_slow) / atr_value.replace(0, np.nan)
    macd_baseline = ema(macd_frame["macd_hist"].abs(), 8).replace(0, np.nan)
    macd_score = _bounded_norm(macd_frame["macd_hist"] / macd_baseline, 1.0)
    adx_norm = (adx_frame["adx"] / 50.0).clip(upper=1.0).fillna(0.0)
    trend_score = (_bounded_norm(ema_slope, 0.5) * 0.4 + macd_score * 0.3 + adx_norm * 0.3) * adx_norm

    volume_score_raw = _bounded_norm(volume_ratio.fillna(1.0) - 1.0, 1.5)
    obv_score_raw = _bounded_norm(obv_delta / (df["volume"].replace(0, np.nan) * 10.0), 1.0)
    volume_score = (volume_score_raw * 0.5 + obv_score_raw.fillna(0.0) * 0.5).clip(-1.0, 1.0)

    rsi_score = (rsi_value - 50.0) / 50.0
    mfi_score = (mfi_value - 50.0) / 50.0
    rsi_momentum = rsi_value.diff().fillna(0.0)
    momentum_score = (rsi_score * 0.4 + mfi_score * 0.4 + _bounded_norm(rsi_momentum, 0.1) * 0.2).clip(
        -1.0,
        1.0,
    )

    vwap_distance = ((df["close"] - vwap_value) / vwap_value.replace(0, np.nan)) * 100.0
    vwap_score = _bounded_norm(vwap_distance.fillna(0.0), 0.02)
    bb_score = _bounded_norm(bb["bb_percent_b"].fillna(0.5) - 0.5, 2.0)
    atr_pct = (atr_value / df["close"].replace(0, np.nan)) * 100.0
    volatility_score = _bounded_norm(atr_pct.fillna(0.0) - 1.5, 0.5)

    composite = (
        trend_score.fillna(0.0) * 0.30
        + volume_score.fillna(0.0) * 0.25
        + momentum_score.fillna(0.0) * 0.20
        + vwap_score.fillna(0.0) * 0.10
        + bb_score.fillna(0.0) * 0.10
        + volatility_score.fillna(0.0) * 0.05
    )
    power = composite * 100.0
    power_smooth = ema(power, 3)
    power_ma = ema(power_smooth, 5)
    golden_cross = (power_smooth.shift(1) <= power_ma.shift(1)) & (power_smooth > power_ma)
    death_cross = (power_smooth.shift(1) >= power_ma.shift(1)) & (power_smooth < power_ma)
    bullish_st = st["supertrend_direction"] > 0
    bearish_st = st["supertrend_direction"] < 0
    trend_up = ema_fast > ema_slow
    trend_down = ema_fast < ema_slow
    trend_confirmed = trend_up & bullish_st & (df["close"] > bb["bb_basis"])
    trend_reversed = trend_down & bearish_st & (df["close"] < bb["bb_basis"])
    volume_momentum = volume_confirmed & (obv_trend > 0.0)
    volume_weakness = volume_confirmed & (obv_trend < 0.0)
    momentum_confirmed = (rsi_value > 50.0) & (mfi_value > 50.0) & (adx_frame["plus_di"] > adx_frame["minus_di"])
    momentum_weak = (rsi_value < 50.0) & (mfi_value < 50.0) & (adx_frame["minus_di"] > adx_frame["plus_di"])
    bull = power_smooth > 35.0
    bear = power_smooth < -35.0
    strong_bull = (power_smooth > 52.5) & (power_smooth > power_ma)
    strong_bear = (power_smooth < -52.5) & (power_smooth < power_ma)
    long_signal = trend_confirmed & bull & volume_momentum & momentum_confirmed & (rsi_value < 70.0)
    short_signal = trend_reversed & bear & volume_weakness & momentum_weak & (rsi_value > 30.0)
    return pd.DataFrame(
        {
            "jumbo_power": power_smooth,
            "jumbo_power_ma": power_ma,
            "jumbo_trend_score": trend_score.fillna(0.0),
            "jumbo_volume_score": volume_score.fillna(0.0),
            "jumbo_momentum_score": momentum_score.fillna(0.0),
            "jumbo_bull": bull,
            "jumbo_bear": bear,
            "jumbo_golden_cross": golden_cross,
            "jumbo_death_cross": death_cross,
            "jumbo_trend_confirmed": trend_confirmed,
            "jumbo_trend_reversed": trend_reversed,
            "jumbo_volume_momentum": volume_momentum,
            "jumbo_volume_weakness": volume_weakness,
            "jumbo_momentum_confirmed": momentum_confirmed,
            "jumbo_momentum_weak": momentum_weak,
            "jumbo_long_signal": long_signal,
            "jumbo_short_signal": short_signal,
            "jumbo_strong_long": long_signal & strong_bull & (adx_frame["adx"] > 20.0),
            "jumbo_strong_short": short_signal & strong_bear & (adx_frame["adx"] > 20.0),
            "volume_ratio_20": volume_ratio.fillna(1.0),
            "vwma_20": vwma(df["close"], df["volume"], 20),
        }
    )
