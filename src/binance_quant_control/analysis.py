from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .alpha_families import (
    ACTIVE_STRATEGY_FAMILIES,
    build_diagnostic_family_signals,
    build_strategy_family_signals,
    select_best_family,
)
from .binance_api import BinanceClient
from .blave_api import BlaveClient
from .config import REPORTS_DIR, Settings, ensure_runtime_dirs
from .decision_trace import trace_step
from .indicators import (
    adx,
    atr,
    bollinger_bands,
    cci,
    chandelier_exit,
    donchian_channels,
    ema,
    follow_line,
    ichimoku,
    jumbo_power,
    keltner_channels,
    macd,
    money_flow_index,
    on_balance_volume,
    parabolic_sar,
    qqe_mod,
    realized_volatility,
    rsi,
    sma,
    squeeze_momentum,
    stoch_rsi,
    stochastic_oscillator,
    supertrend,
    trend_magic,
    vwap,
    williams_r,
    zscore,
)
from .market_context import summarize_market_context
from .signals import SignalBias, decide_trade_action
from .strategy import StrategyConfig
from .vwap_features import rolling_vwap_bands


@dataclass(slots=True)
class AnalysisArtifacts:
    run_id: str
    output_dir: Path
    report_json: Path
    report_md: Path
    chart_path: Path | None


@dataclass(frozen=True, slots=True)
class MultiTimeframeTrendResult:
    name: str
    bias: SignalBias
    score: float
    confidence: float
    htf_score: int
    ltf_score: int


@dataclass(frozen=True, slots=True)
class TimeframeStructure:
    interval: str
    bias: SignalBias
    score: float
    confidence: float
    ema7: float
    ema25: float
    ema89: float
    close: float
    momentum: float
    adx: float


@dataclass(frozen=True, slots=True)
class MultiTimeframeStructureResult:
    name: str
    bias: SignalBias
    score: float
    confidence: float
    alignment: str
    structures: tuple[TimeframeStructure, ...]


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _direction(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _ema_context_score(close: pd.Series, span: int) -> int:
    if close.empty:
        return 0
    ema_series = ema(close, span)
    slope = 0.0
    if len(ema_series) >= 2:
        slope = float(ema_series.iloc[-1] - ema_series.iloc[-2])
    context = float(close.iloc[-1] - ema_series.iloc[-1])
    momentum = 0.0
    if len(close) >= 3:
        momentum = float(close.iloc[-1] - close.iloc[-3])
    return _direction(slope) + _direction(context) + _direction(momentum)


def evaluate_multi_timeframe_trend(ltf: pd.DataFrame, htf: pd.DataFrame) -> MultiTimeframeTrendResult:
    """Evaluate higher-timeframe trend alignment for lower-timeframe entries.

    The helper is deterministic and intentionally lightweight: it only looks at
    close prices, checks EMA slope and price-vs-EMA context on the higher
    timeframe, and confirms the lower timeframe is aligned with the same
    direction.
    """

    if "close" not in ltf or "close" not in htf:
        raise KeyError("Both frames must include a close column.")

    ltf_close = pd.to_numeric(ltf["close"], errors="coerce").dropna()
    htf_close = pd.to_numeric(htf["close"], errors="coerce").dropna()

    htf_score = _ema_context_score(htf_close, 200)
    ltf_score = _ema_context_score(ltf_close, 50)
    score = (htf_score + ltf_score) / 4.0

    if htf_score >= 3 and ltf_score >= 3:
        bias: SignalBias = "long"
    elif htf_score <= -3 and ltf_score <= -3:
        bias = "short"
    else:
        bias = "neutral"

    confidence = min(abs(htf_score), abs(ltf_score)) / 5.0

    return MultiTimeframeTrendResult(
        name="multi_timeframe_trend",
        bias=bias,
        score=score,
        confidence=confidence,
        htf_score=htf_score,
        ltf_score=ltf_score,
    )


def _timeframe_structure(interval: str, df: pd.DataFrame) -> TimeframeStructure:
    close_series = pd.to_numeric(df["close"], errors="coerce").dropna()
    if close_series.empty:
        return TimeframeStructure(
            interval=interval,
            bias="neutral",
            score=0.0,
            confidence=0.0,
            ema7=0.0,
            ema25=0.0,
            ema89=0.0,
            close=0.0,
            momentum=0.0,
            adx=0.0,
        )

    ema7 = ema(close_series, 7)
    ema25 = ema(close_series, 25)
    ema89 = ema(close_series, 89)
    close = float(close_series.iloc[-1])
    ema7_value = float(ema7.iloc[-1])
    ema25_value = float(ema25.iloc[-1])
    ema89_value = float(ema89.iloc[-1])
    momentum = float(close_series.iloc[-1] - close_series.iloc[-4]) if len(close_series) >= 4 else 0.0

    bullish = 0
    bearish = 0
    if ema7_value > ema25_value > ema89_value:
        bullish += 2
    elif ema7_value < ema25_value < ema89_value:
        bearish += 2
    if close > ema25_value:
        bullish += 1
    elif close < ema25_value:
        bearish += 1
    if momentum > 0.0:
        bullish += 1
    elif momentum < 0.0:
        bearish += 1

    bias: SignalBias = "neutral"
    if bullish >= 3 and bullish > bearish:
        bias = "long"
    elif bearish >= 3 and bearish > bullish:
        bias = "short"

    total = bullish + bearish
    confidence = (max(bullish, bearish) / total) if total else 0.0
    score = (bullish - bearish) / max(total, 1)
    adx_value = 0.0
    if {"high", "low", "close"}.issubset(df.columns) and len(df) >= 20:
        adx_frame = adx(df, 14)
        adx_value = _float(adx_frame["adx"].iloc[-1])

    return TimeframeStructure(
        interval=interval,
        bias=bias,
        score=round(score, 4),
        confidence=round(confidence, 4),
        ema7=round(ema7_value, 8),
        ema25=round(ema25_value, 8),
        ema89=round(ema89_value, 8),
        close=round(close, 8),
        momentum=round(momentum, 8),
        adx=round(adx_value, 4),
    )


def evaluate_multi_timeframe_structure(frames: dict[str, pd.DataFrame]) -> MultiTimeframeStructureResult:
    """Evaluate 1m/15m/4h/1d style EMA 7/25/89 alignment for strategy context."""

    ordered_intervals = [item for item in ("1m", "15m", "1h", "4h", "1d") if item in frames]
    ordered_intervals.extend(interval for interval in frames if interval not in ordered_intervals)
    structures = tuple(_timeframe_structure(interval, frames[interval]) for interval in ordered_intervals)
    if not structures:
        return MultiTimeframeStructureResult(
            name="multi_timeframe_structure",
            bias="neutral",
            score=0.0,
            confidence=0.0,
            alignment="unavailable",
            structures=(),
        )

    weights = {"1m": 0.7, "15m": 1.0, "1h": 1.1, "4h": 1.4, "1d": 1.6}
    long_weight = sum(weights.get(item.interval, 1.0) * item.confidence for item in structures if item.bias == "long")
    short_weight = sum(weights.get(item.interval, 1.0) * item.confidence for item in structures if item.bias == "short")
    total_weight = sum(weights.get(item.interval, 1.0) * item.confidence for item in structures)
    weighted_score = sum(weights.get(item.interval, 1.0) * item.score for item in structures) / max(
        sum(weights.get(item.interval, 1.0) for item in structures),
        1.0,
    )

    if long_weight > short_weight:
        bias: SignalBias = "long"
    elif short_weight > long_weight:
        bias = "short"
    else:
        bias = "neutral"

    actionable = [item for item in structures if item.bias in {"long", "short"}]
    aligned_count = sum(1 for item in actionable if item.bias == bias) if bias != "neutral" else 0
    conflict_count = sum(1 for item in actionable if item.bias != bias) if bias != "neutral" else len(actionable)
    if not actionable:
        alignment = "neutral"
    elif conflict_count == 0 and aligned_count >= 2:
        alignment = "strong"
    elif aligned_count >= max(1, conflict_count):
        alignment = "mixed"
    else:
        alignment = "conflicted"

    confidence = (max(long_weight, short_weight) / total_weight) if total_weight else 0.0
    return MultiTimeframeStructureResult(
        name="multi_timeframe_structure",
        bias=bias,
        score=round(weighted_score, 4),
        confidence=round(confidence, 4),
        alignment=alignment,
        structures=structures,
    )


def prepare_klines_frame(klines: list[list[Any]]) -> pd.DataFrame:
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    df = pd.DataFrame(klines, columns=columns)
    if df.empty:
        raise RuntimeError("No kline data returned from Binance.")
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    if df["open_time"].dropna().gt(10**14).any():
        df["open_time"] = df["open_time"] / 1000
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("open_time").sort_index()
    return df


def enrich_indicators(df: pd.DataFrame, interval: str, strategy: StrategyConfig | None = None) -> pd.DataFrame:
    ema_fast_span = strategy.signal.ema_fast if strategy else 21
    ema_slow_span = strategy.signal.ema_slow if strategy else 55
    rsi_length = strategy.signal.rsi_length if strategy else 14
    macd_fast = strategy.signal.macd_fast if strategy else 12
    macd_slow = strategy.signal.macd_slow if strategy else 26
    macd_signal = strategy.signal.macd_signal if strategy else 9
    breakout_length = strategy.signal.breakout_length if strategy else 20
    base = df.copy()
    core = pd.DataFrame(
        {
            "sma_200": sma(base["close"], 200),
            "ema_fast": ema(base["close"], ema_fast_span),
            "ema_slow": ema(base["close"], ema_slow_span),
            "rsi_14": rsi(base["close"], rsi_length),
            "atr_14": atr(base, 14),
            "breakout_high_20": base["high"].rolling(breakout_length).max().shift(1),
            "breakout_low_20": base["low"].rolling(breakout_length).min().shift(1),
            "realized_vol_20": realized_volatility(base["close"], interval, 20),
        },
        index=base.index,
    )
    fib_window = 89
    fib_swing_high = base["high"].rolling(fib_window).max().shift(1)
    fib_swing_low = base["low"].rolling(fib_window).min().shift(1)
    fib_range = (fib_swing_high - fib_swing_low).where((fib_swing_high - fib_swing_low) > 0)
    fib_position = ((base["close"] - fib_swing_low) / fib_range).clip(lower=-1.0, upper=2.0)
    fib_retrace_from_high = ((fib_swing_high - base["close"]) / fib_range).clip(lower=-1.0, upper=2.0)
    fib_retrace_from_low = ((base["close"] - fib_swing_low) / fib_range).clip(lower=-1.0, upper=2.0)
    fib_trend_up = (core["ema_fast"] > core["ema_slow"]) & (base["close"] > core["ema_slow"])
    fib_trend_down = (core["ema_fast"] < core["ema_slow"]) & (base["close"] < core["ema_slow"])
    fib = pd.DataFrame(
        {
            "fib_swing_high_89": fib_swing_high,
            "fib_swing_low_89": fib_swing_low,
            "fib_range_89": fib_range,
            "fib_position_89": fib_position,
            "fib_retrace_from_high_89": fib_retrace_from_high,
            "fib_retrace_from_low_89": fib_retrace_from_low,
            "fib_382_long_89": fib_swing_high - fib_range * 0.382,
            "fib_500_long_89": fib_swing_high - fib_range * 0.500,
            "fib_618_long_89": fib_swing_high - fib_range * 0.618,
            "fib_786_long_89": fib_swing_high - fib_range * 0.786,
            "fib_382_short_89": fib_swing_low + fib_range * 0.382,
            "fib_500_short_89": fib_swing_low + fib_range * 0.500,
            "fib_618_short_89": fib_swing_low + fib_range * 0.618,
            "fib_786_short_89": fib_swing_low + fib_range * 0.786,
            "fib_pullback_long_zone": fib_trend_up & fib_retrace_from_high.between(0.382, 0.786),
            "fib_ote_long_zone": fib_trend_up & fib_retrace_from_high.between(0.618, 0.786),
            "fib_pullback_short_zone": fib_trend_down & fib_retrace_from_low.between(0.382, 0.786),
            "fib_ote_short_zone": fib_trend_down & fib_retrace_from_low.between(0.618, 0.786),
        },
        index=base.index,
    )
    liquidity_window = 20
    liquidity_high = base["high"].rolling(liquidity_window).max().shift(1)
    liquidity_low = base["low"].rolling(liquidity_window).min().shift(1)
    liquidity_range = (liquidity_high - liquidity_low).where((liquidity_high - liquidity_low) > 0)
    sweep_buffer = pd.concat(
        [
            core["atr_14"] * 0.05,
            base["close"] * 0.0005,
            liquidity_range * 0.01,
        ],
        axis=1,
    ).max(axis=1)
    sweep_high = (base["high"] > liquidity_high + sweep_buffer) & (base["close"] < liquidity_high)
    sweep_low = (base["low"] < liquidity_low - sweep_buffer) & (base["close"] > liquidity_low)
    close_position = ((base["close"] - base["low"]) / (base["high"] - base["low"]).where((base["high"] - base["low"]) > 0)).clip(
        lower=0.0,
        upper=1.0,
    )
    liquidity = pd.DataFrame(
        {
            "liquidity_swing_high_20": liquidity_high,
            "liquidity_swing_low_20": liquidity_low,
            "liquidity_range_20": liquidity_range,
            "liquidity_sweep_high_20": sweep_high,
            "liquidity_sweep_low_20": sweep_low,
            "liquidity_reclaim_long_20": sweep_low & (close_position >= 0.58),
            "liquidity_reclaim_short_20": sweep_high & (close_position <= 0.42),
            "liquidity_close_position": close_position,
        },
        index=base.index,
    )
    frames = [
        core,
        fib,
        liquidity,
        macd(base["close"], macd_fast, macd_slow, macd_signal),
        bollinger_bands(base["close"], 20, 2.0),
        adx(base, 14),
        stochastic_oscillator(base, 14, 3),
        stoch_rsi(base["close"], rsi_period=14, stoch_period=14, smooth_k=3, smooth_d=3),
        donchian_channels(base, upper_window=20, lower_window=20),
        rolling_vwap_bands(base, window=48),
        keltner_channels(base, window=20, multiplier=1.5),
        squeeze_momentum(base, length=20, bb_mult=2.0, kc_mult=1.5),
        chandelier_exit(base, length=22, multiplier=3.0),
        qqe_mod(base["close"], rsi_length=6, smoothing=5, qqe_factor=3.0, threshold=3.0),
        ichimoku(base),
        parabolic_sar(base),
        supertrend(base, 10, 3.0),
        trend_magic(base, cci_period=20, atr_period=5, atr_multiplier=1.0),
        follow_line(base, atr_period=5, bb_period=21, bb_deviation=1.0),
        jumbo_power(base),
    ]
    features = pd.concat(frames, axis=1)
    obv_series = on_balance_volume(base["close"], base["volume"])
    williams = williams_r(base, 14)
    features = pd.concat(
        [
            features,
            pd.DataFrame(
                {
                    "obv": obv_series,
                    "obv_zscore_20": zscore(obv_series, 20),
                    "volume_zscore_20": zscore(base["volume"], 20),
                    "vwap": vwap(base),
                    "mfi_14": money_flow_index(base, 14),
                    "cci_20": cci(base, 20),
                    "williams_r_14": williams,
                    "ultimate_oscillator_proxy": (
                        core["rsi_14"].fillna(50.0) * 0.4
                        + features["stoch_k"].fillna(50.0) * 0.3
                        + (100.0 + williams.fillna(-50.0)) * 0.3
                    ),
                },
                index=base.index,
            ),
        ],
        axis=1,
    )
    return pd.concat([base, features], axis=1).copy()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(parsed):
        return default
    return parsed


def _optional_rounded(value: Any, digits: int) -> float | None:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _positive_price(value: Any) -> float | None:
    parsed = _float(value, 0.0)
    return parsed if parsed > 0.0 else None


def _snapshot_metric(snapshot: dict[str, Any], flat_key: str, nested_key: str, nested_field: str = "-") -> float:
    if flat_key in snapshot:
        return _float(snapshot.get(flat_key))
    nested = snapshot.get(nested_key)
    if isinstance(nested, dict):
        return _float(nested.get(nested_field))
    return _float(nested)


def _selected_strategy_family_name(analysis: dict[str, Any] | None) -> str:
    selected = (analysis or {}).get("selected_strategy_family") or {}
    family = str((analysis or {}).get("strategy_family") or selected.get("family") or "")
    return family if family in ACTIVE_STRATEGY_FAMILIES else ""


def _indicator_invalidation_candidates(
    latest: pd.Series,
    *,
    side: str,
    family: str,
) -> list[tuple[str, float]]:
    close = _float(latest.get("close"), 0.0)
    if close <= 0.0:
        return []

    below: list[tuple[str, float]] = []
    above: list[tuple[str, float]] = []
    for name in (
        "supertrend",
        "trend_magic",
        "follow_line",
        "chandelier_long_stop",
        "psar",
        "donchian_lower",
        "keltner_lower",
        "bb_lower",
        "bb_basis",
        "fib_786_long_89",
        "fib_swing_low_89",
        "vwap_lower_2_48",
        "vwap_lower_1_48",
        "vwap_rolling_48",
    ):
        value = _positive_price(latest.get(name))
        if value is not None and value < close:
            below.append((name, value))
    for name in (
        "supertrend",
        "trend_magic",
        "follow_line",
        "chandelier_short_stop",
        "psar",
        "donchian_upper",
        "keltner_upper",
        "bb_upper",
        "bb_basis",
        "fib_786_short_89",
        "fib_swing_high_89",
        "vwap_upper_2_48",
        "vwap_upper_1_48",
        "vwap_rolling_48",
    ):
        value = _positive_price(latest.get(name))
        if value is not None and value > close:
            above.append((name, value))

    if side == "BUY":
        preferred = {
            "trend_continuation": ("supertrend", "trend_magic", "follow_line", "chandelier_long_stop", "psar"),
            "breakout": ("donchian_lower", "keltner_lower", "bb_basis", "supertrend", "follow_line"),
            "trend_pullback": ("fib_786_long_89", "fib_swing_low_89", "trend_magic", "follow_line", "bb_basis"),
            "vwap_reclaim": ("vwap_lower_2_48", "vwap_lower_1_48", "vwap_rolling_48", "bb_lower", "bb_basis"),
            "mean_reversion": ("bb_lower", "keltner_lower", "donchian_lower", "bb_basis"),
        }.get(family, ("supertrend", "trend_magic", "follow_line", "chandelier_long_stop"))
        return [item for item in below if item[0] in preferred]

    preferred = {
        "trend_continuation": ("supertrend", "trend_magic", "follow_line", "chandelier_short_stop", "psar"),
        "breakout": ("donchian_upper", "keltner_upper", "bb_basis", "supertrend", "follow_line"),
        "trend_pullback": ("fib_786_short_89", "fib_swing_high_89", "trend_magic", "follow_line", "bb_basis"),
        "vwap_reclaim": ("vwap_upper_2_48", "vwap_upper_1_48", "vwap_rolling_48", "bb_upper", "bb_basis"),
        "mean_reversion": ("bb_upper", "keltner_upper", "donchian_upper", "bb_basis"),
    }.get(family, ("supertrend", "trend_magic", "follow_line", "chandelier_short_stop"))
    return [item for item in above if item[0] in preferred]


def _strategy_stop_distance(
    latest: pd.Series,
    *,
    side: str,
    strategy: StrategyConfig | None,
    family: str = "",
) -> dict[str, Any]:
    close = _float(latest.get("close"), 0.0)
    atr_value = max(_float(latest.get("atr_14"), 0.0), close * 0.003)
    atr_multiple = strategy.risk.atr_stop_multiple if strategy else 1.5
    atr_distance = max(atr_value * atr_multiple, close * 0.002)
    max_distance = max(atr_distance * 1.6, close * 0.006)
    min_distance = max(atr_distance * 0.85, close * 0.002)
    wick_buffer = max(atr_value * 0.15, close * 0.001)
    candidates = _indicator_invalidation_candidates(latest, side=side, family=family)

    usable: list[tuple[str, float, float]] = []
    for source, price in candidates:
        stop_price = price - wick_buffer if side == "BUY" else price + wick_buffer
        distance = abs(close - stop_price)
        if min_distance <= distance <= max_distance:
            usable.append((source, stop_price, distance))

    if usable:
        # Select the structure line closest to the configured ATR risk budget.
        # This keeps stops indicator-aware without letting a tight line inside
        # normal candle noise become the whole risk model.
        source, stop_price, distance = min(usable, key=lambda item: abs(item[2] - atr_distance))
        return {
            "stop_price": stop_price,
            "risk_distance": distance,
            "source": f"{source}_buffered",
            "atr_distance": atr_distance,
            "candidate_count": len(usable),
        }

    fallback_price = close - atr_distance if side == "BUY" else close + atr_distance
    return {
        "stop_price": fallback_price,
        "risk_distance": atr_distance,
        "source": "atr",
        "atr_distance": atr_distance,
        "candidate_count": 0,
    }


def indicator_trade_plan_side(
    latest: pd.Series,
    *,
    side: str,
    strategy: StrategyConfig | None = None,
    family: str = "",
) -> dict[str, Any]:
    close = _float(latest.get("close"), 0.0)
    tp_levels = strategy.risk.take_profit_r_multiples if strategy else (1.0, 2.0)
    tp1_multiple = tp_levels[0] if tp_levels else 1.0
    tp2_multiple = tp_levels[1] if len(tp_levels) > 1 else max(tp1_multiple * 2, 2.0)
    tp3_multiple = tp_levels[2] if len(tp_levels) > 2 else max(tp2_multiple * 1.4, tp2_multiple + 1.0)
    stop = _strategy_stop_distance(latest, side=side, strategy=strategy, family=family)
    risk_distance = float(stop["risk_distance"])
    if side == "BUY":
        take_profit_levels = [round(close + multiple * risk_distance, 6) for multiple in (tp1_multiple, tp2_multiple, tp3_multiple)]
    else:
        take_profit_levels = [round(close - multiple * risk_distance, 6) for multiple in (tp1_multiple, tp2_multiple, tp3_multiple)]
    return {
        "entry_reference": round(close, 6),
        "invalidation": round(float(stop["stop_price"]), 6),
        "invalidation_source": str(stop["source"]),
        "risk_distance": round(risk_distance, 6),
        "risk_distance_source": "indicator" if stop["source"] != "atr" else "atr",
        "atr_risk_distance": round(float(stop["atr_distance"]), 6),
        "indicator_candidate_count": int(stop["candidate_count"]),
        "strategy_family": family or "score_model",
        "take_profit_1": take_profit_levels[0],
        "take_profit_2": take_profit_levels[1],
        "take_profit_3": take_profit_levels[2],
        "take_profit_levels": take_profit_levels,
        "r_multiple_to_tp1": round(tp1_multiple, 4),
        "r_multiple_to_tp2": round(tp2_multiple, 4),
        "r_multiple_to_tp3": round(tp3_multiple, 4),
    }


def score_bias(
    latest: pd.Series,
    market: str,
    blave_snapshot: dict[str, Any] | None,
    market_context: dict[str, Any] | None = None,
    strategy: StrategyConfig | None = None,
) -> dict[str, Any]:
    score = 50
    notes: list[str] = []
    bullish = 0
    bearish = 0
    close = _float(latest["close"])
    ema_fast_value = _float(latest["ema_fast"])
    ema_slow_value = _float(latest["ema_slow"])
    sma_200 = _float(latest["sma_200"])
    rsi_value = _float(latest["rsi_14"], 50.0)
    macd_hist_value = _float(latest["macd_hist"])
    breakout_high = _float(latest["breakout_high_20"])
    breakout_low = _float(latest["breakout_low_20"])
    realized_vol = _float(latest["realized_vol_20"])
    adx_value = _float(latest["adx"])
    plus_di = _float(latest["plus_di"])
    minus_di = _float(latest["minus_di"])
    bb_percent_b = _float(latest["bb_percent_b"], 0.5)
    bb_bandwidth = _float(latest["bb_bandwidth"], 0.0)
    fib_retrace_from_high = _float(latest.get("fib_retrace_from_high_89"), 0.0)
    fib_retrace_from_low = _float(latest.get("fib_retrace_from_low_89"), 0.0)
    fib_ote_long = bool(latest.get("fib_ote_long_zone", False))
    fib_ote_short = bool(latest.get("fib_ote_short_zone", False))
    fib_pullback_long = bool(latest.get("fib_pullback_long_zone", False))
    fib_pullback_short = bool(latest.get("fib_pullback_short_zone", False))
    liquidity_reclaim_long = bool(latest.get("liquidity_reclaim_long_20", False))
    liquidity_reclaim_short = bool(latest.get("liquidity_reclaim_short_20", False))
    liquidity_sweep_high = bool(latest.get("liquidity_sweep_high_20", False))
    liquidity_sweep_low = bool(latest.get("liquidity_sweep_low_20", False))
    liquidity_close_position = _float(latest.get("liquidity_close_position"), 0.5)
    volume_z = _float(latest["volume_zscore_20"])
    obv_z = _float(latest["obv_zscore_20"])
    vwap_value = _float(latest["vwap"])
    taker_ratio = _float((market_context or {}).get("taker_buy_sell_ratio"), 1.0)
    taker_imbalance = _float((market_context or {}).get("taker_flow_imbalance"), 0.0)
    funding_rate = _float((market_context or {}).get("funding_rate"), 0.0)
    open_interest_change_pct = _float((market_context or {}).get("open_interest_change_pct"), 0.0)
    order_book_imbalance = _float((market_context or {}).get("order_book_imbalance"), 0.0)
    spread_bps = _float((market_context or {}).get("spread_bps"), 0.0)
    multi_timeframe = (market_context or {}).get("multi_timeframe_structure") or {}
    mtf_bias = str(multi_timeframe.get("bias") or "neutral")
    mtf_alignment = str(multi_timeframe.get("alignment") or "neutral")
    mtf_confidence = _float(multi_timeframe.get("confidence"), 0.0)
    volume_profile = (market_context or {}).get("volume_profile") or {}
    profile_position = str(volume_profile.get("close_position") or "")
    profile_delta_ratio = _float(volume_profile.get("delta_ratio"), 0.0)
    volume_bubbles = (market_context or {}).get("volume_bubbles") or {}
    bubble_cluster = str(volume_bubbles.get("cluster") or "none")
    bubble_side = str(volume_bubbles.get("side") or "neutral")
    htf_imbalance = (market_context or {}).get("htf_volume_imbalance") or {}
    imbalance_active = bool(htf_imbalance.get("active"))
    imbalance_direction = str(htf_imbalance.get("direction") or "neutral")
    supertrend_direction = int(_float(latest.get("supertrend_direction"), 0.0))
    trend_magic_direction = int(_float(latest.get("trend_magic_direction"), 0.0))
    follow_line_direction = int(_float(latest.get("follow_line_direction"), 0.0))
    jumbo_power_value = _float(latest.get("jumbo_power"), 0.0)
    jumbo_power_ma = _float(latest.get("jumbo_power_ma"), 0.0)
    mfi_value = _float(latest.get("mfi_14"), 50.0)
    cci_value = _float(latest.get("cci_20"), 0.0)
    stoch_k = _float(latest.get("stoch_k"), 50.0)
    williams_value = _float(latest.get("williams_r_14"), -50.0)
    ultimate_proxy = _float(latest.get("ultimate_oscillator_proxy"), 50.0)
    volume_ratio_20 = _float(latest.get("volume_ratio_20"), 1.0)
    vwma_20 = _float(latest.get("vwma_20"), 0.0)
    family_signals = build_strategy_family_signals(latest, market_context)
    diagnostic_family_signals = build_diagnostic_family_signals(latest, market_context)
    best_family = select_best_family(family_signals)

    def _group_from_strengths(
        name: str,
        *,
        long_strength: float,
        short_strength: float,
        max_strength: float,
        max_delta: float,
        reasons: list[str],
        neutral_delta: float = 0.0,
    ) -> dict[str, Any]:
        net = long_strength - short_strength
        if long_strength > short_strength and long_strength >= 1.0:
            bias: SignalBias = "long"
            confidence = min(long_strength / max(max_strength, 1.0), 1.0)
        elif short_strength > long_strength and short_strength >= 1.0:
            bias = "short"
            confidence = min(short_strength / max(max_strength, 1.0), 1.0)
        else:
            bias = "neutral"
            confidence = min(max(long_strength, short_strength) / max(max_strength, 1.0), 1.0)
        return {
            "name": name,
            "bias": bias,
            "score_delta": round((net / max(max_strength, 1.0)) * max_delta + neutral_delta, 4),
            "confidence": round(confidence, 4),
            "reasons": reasons,
            "counted": True,
        }

    signal_groups: list[dict[str, Any]] = []

    trend_long = 0.0
    trend_short = 0.0
    trend_reasons: list[str] = []
    if close > ema_slow_value and ema_fast_value > ema_slow_value:
        trend_long += 2.0
        trend_reasons.append("Price and fast EMA are above the slow EMA.")
    elif close < ema_slow_value and ema_fast_value < ema_slow_value:
        trend_short += 2.0
        trend_reasons.append("Price and fast EMA are below the slow EMA.")
    if close > sma_200:
        trend_long += 1.0
        trend_reasons.append("Price is above SMA200.")
    elif close < sma_200:
        trend_short += 1.0
        trend_reasons.append("Price is below SMA200.")
    if macd_hist_value > 0:
        trend_long += 1.0
        trend_reasons.append("MACD histogram is positive.")
    elif macd_hist_value < 0:
        trend_short += 1.0
        trend_reasons.append("MACD histogram is negative.")
    if adx_value >= 20:
        if plus_di > minus_di:
            trend_long += 1.0
            trend_reasons.append("ADX/DI confirms directional strength on the long side.")
        elif minus_di > plus_di:
            trend_short += 1.0
            trend_reasons.append("ADX/DI confirms directional strength on the short side.")
    else:
        trend_reasons.append("ADX says trend strength is only moderate; mean-reversion risk remains.")
    trend_filter_votes = supertrend_direction + trend_magic_direction + follow_line_direction
    if trend_filter_votes >= 2:
        trend_long += 2.0
        trend_reasons.append("SuperTrend, Trend Magic, and Follow Line mostly align long.")
    elif trend_filter_votes <= -2:
        trend_short += 2.0
        trend_reasons.append("SuperTrend, Trend Magic, and Follow Line mostly align short.")
    else:
        trend_reasons.append("Trend filters are mixed; avoid oversized entries.")
    trend_neutral_delta = 0.0
    if mtf_bias == "long" and mtf_alignment in {"strong", "mixed"}:
        trend_long += 2.0 if mtf_alignment == "strong" else 1.0
        trend_reasons.append(
            f"Multi-timeframe EMA 7/25/89 structure supports longs ({mtf_alignment}, confidence {mtf_confidence:.2f})."
        )
    elif mtf_bias == "short" and mtf_alignment in {"strong", "mixed"}:
        trend_short += 2.0 if mtf_alignment == "strong" else 1.0
        trend_reasons.append(
            f"Multi-timeframe EMA 7/25/89 structure supports shorts ({mtf_alignment}, confidence {mtf_confidence:.2f})."
        )
    elif mtf_alignment == "conflicted":
        trend_neutral_delta -= 3.0
        trend_reasons.append("Multi-timeframe EMA 7/25/89 structure is conflicted; entry quality is lower.")
    signal_groups.append(
        _group_from_strengths(
            "trend_alignment",
            long_strength=trend_long,
            short_strength=trend_short,
            max_strength=9.0,
            max_delta=18.0,
            reasons=trend_reasons,
            neutral_delta=trend_neutral_delta,
        )
    )

    momentum_long = 0.0
    momentum_short = 0.0
    momentum_delta = 0.0
    momentum_reasons: list[str] = []
    if 52 <= rsi_value <= 68:
        momentum_long += 1.0
        momentum_reasons.append("RSI sits in a healthy momentum zone.")
    elif rsi_value >= 72:
        momentum_delta -= 2.0
        momentum_reasons.append("RSI is extended; chase risk is elevated.")
    elif rsi_value <= 40:
        momentum_short += 1.0
        momentum_reasons.append("RSI is weak.")
    if jumbo_power_value >= 52.5 and jumbo_power_value > jumbo_power_ma:
        momentum_long += 2.0
        momentum_reasons.append(f"JUMBO composite power is strongly bullish ({jumbo_power_value:.1f}).")
    elif jumbo_power_value >= 35.0 and jumbo_power_value > jumbo_power_ma:
        momentum_long += 1.0
        momentum_reasons.append(f"JUMBO composite power is bullish ({jumbo_power_value:.1f}).")
    elif jumbo_power_value <= -52.5 and jumbo_power_value < jumbo_power_ma:
        momentum_short += 2.0
        momentum_reasons.append(f"JUMBO composite power is strongly bearish ({jumbo_power_value:.1f}).")
    elif jumbo_power_value <= -35.0 and jumbo_power_value < jumbo_power_ma:
        momentum_short += 1.0
        momentum_reasons.append(f"JUMBO composite power is bearish ({jumbo_power_value:.1f}).")
    if 50.0 < mfi_value < 80.0 and cci_value > 0.0:
        momentum_long += 1.0
        momentum_reasons.append("MFI and CCI confirm constructive buy pressure.")
    elif 20.0 < mfi_value < 50.0 and cci_value < 0.0:
        momentum_short += 1.0
        momentum_reasons.append("MFI and CCI confirm distribution pressure.")
    if stoch_k >= 80.0 and williams_value >= -20.0 and ultimate_proxy >= 65.0:
        momentum_delta -= 1.5
        momentum_reasons.append("Oscillator stack is extended; prefer staged entries or wait for pullback.")
    elif stoch_k <= 20.0 and williams_value <= -80.0 and ultimate_proxy <= 35.0:
        momentum_delta -= 1.0
        momentum_reasons.append("Oscillator stack is deeply oversold; short entries need extra confirmation.")
    signal_groups.append(
        _group_from_strengths(
            "momentum_power",
            long_strength=momentum_long,
            short_strength=momentum_short,
            max_strength=4.0,
            max_delta=10.0,
            reasons=momentum_reasons,
            neutral_delta=momentum_delta,
        )
    )

    structure_long = 0.0
    structure_short = 0.0
    structure_delta = 0.0
    structure_reasons: list[str] = []
    if breakout_high and close > breakout_high:
        structure_long += 1.0
        structure_reasons.append("Fresh 20-bar breakout confirmed.")
    elif breakout_low and close < breakout_low:
        structure_short += 1.0
        structure_reasons.append("20-bar support has failed.")
    if close > vwap_value > 0:
        structure_long += 1.0
        structure_reasons.append("Price is holding above VWAP.")
    elif vwap_value > 0 and close < vwap_value:
        structure_short += 1.0
        structure_reasons.append("Price is below VWAP.")
    if bb_percent_b >= 0.8:
        structure_long += 1.0
        structure_reasons.append("Price is pressing the upper Bollinger band.")
    elif bb_percent_b <= 0.2:
        structure_short += 1.0
        structure_reasons.append("Price is pressing the lower Bollinger band.")
    if bb_bandwidth <= 0.05:
        structure_reasons.append("Bollinger bandwidth is compressed; breakout risk is elevated.")
    if fib_ote_long:
        structure_long += 2.0
        structure_reasons.append(
            f"Price is in the 0.618-0.786 Fibonacci long pullback zone ({fib_retrace_from_high:.3f})."
        )
    elif fib_pullback_long:
        structure_long += 1.0
        structure_reasons.append(
            f"Price is in the 0.382-0.786 Fibonacci long pullback zone ({fib_retrace_from_high:.3f})."
        )
    if fib_ote_short:
        structure_short += 2.0
        structure_reasons.append(
            f"Price is in the 0.618-0.786 Fibonacci short pullback zone ({fib_retrace_from_low:.3f})."
        )
    elif fib_pullback_short:
        structure_short += 1.0
        structure_reasons.append(
            f"Price is in the 0.382-0.786 Fibonacci short pullback zone ({fib_retrace_from_low:.3f})."
        )
    if profile_position == "near-poc":
        structure_delta -= 2.0
        structure_reasons.append("Price is near volume-profile POC; chop risk is higher unless breakout confirms.")
    if liquidity_reclaim_long:
        structure_long += 2.0
        structure_reasons.append("Price swept prior sell-side liquidity and reclaimed back inside the range.")
    elif liquidity_sweep_low:
        structure_delta -= 1.0
        structure_reasons.append("Price swept below prior liquidity but has not reclaimed cleanly yet.")
    if liquidity_reclaim_short:
        structure_short += 2.0
        structure_reasons.append("Price swept prior buy-side liquidity and rejected back inside the range.")
    elif liquidity_sweep_high:
        structure_delta -= 1.0
        structure_reasons.append("Price swept above prior liquidity but has not rejected cleanly yet.")
    signal_groups.append(
        _group_from_strengths(
            "price_structure",
            long_strength=structure_long,
            short_strength=structure_short,
            max_strength=7.0,
            max_delta=9.0,
            reasons=structure_reasons,
            neutral_delta=structure_delta,
        )
    )

    volume_long = 0.0
    volume_short = 0.0
    volume_delta = 0.0
    volume_reasons: list[str] = []
    if volume_z >= 1.0:
        volume_reasons.append("Volume is expanding relative to the last 20 bars.")
    if volume_ratio_20 >= 1.5:
        volume_reasons.append(f"Volume ratio is elevated at {volume_ratio_20:.2f}x.")
    if close > vwma_20 > 0:
        volume_long += 1.0
        volume_reasons.append("Price is above VWMA20, so volume-weighted trend support is positive.")
    elif vwma_20 > 0 and close < vwma_20:
        volume_short += 1.0
        volume_reasons.append("Price is below VWMA20, so volume-weighted trend support is weak.")
    if obv_z >= 1.0:
        volume_long += 1.0
        volume_reasons.append("OBV confirms accumulation.")
    elif obv_z <= -1.0:
        volume_short += 1.0
        volume_reasons.append("OBV confirms distribution.")
    if taker_ratio >= 1.08:
        volume_long += 1.0
        volume_reasons.append(f"Taker flow favors buyers ({taker_ratio:.2f}x buy/sell).")
    elif taker_ratio <= 0.92:
        volume_short += 1.0
        volume_reasons.append(f"Taker flow favors sellers ({taker_ratio:.2f}x buy/sell).")
    if order_book_imbalance >= 0.12:
        volume_long += 1.0
        volume_reasons.append("Top-of-book depth leans to the bid side.")
    elif order_book_imbalance <= -0.12:
        volume_short += 1.0
        volume_reasons.append("Top-of-book depth leans to the ask side.")
    if profile_position == "above-value" and profile_delta_ratio > 0.05:
        volume_long += 1.0
        volume_reasons.append("Volume profile shows price above value with positive delta.")
    elif profile_position == "below-value" and profile_delta_ratio < -0.05:
        volume_short += 1.0
        volume_reasons.append("Volume profile shows price below value with negative delta.")
    if bubble_cluster in {"medium", "big"}:
        if bubble_side == "buy":
            volume_long += 1.0
            volume_reasons.append(f"{bubble_cluster.title()} buy-side volume bubble detected.")
        elif bubble_side == "sell":
            volume_short += 1.0
            volume_reasons.append(f"{bubble_cluster.title()} sell-side volume bubble detected.")
        else:
            volume_delta -= 2.0
            volume_reasons.append(f"{bubble_cluster.title()} mixed volume bubble detected; directional conviction is unclear.")
    if imbalance_active and imbalance_direction == "bullish":
        volume_long += 1.0
        volume_reasons.append("Active bullish HTF volume-spike imbalance is still respected.")
    elif imbalance_active and imbalance_direction == "bearish":
        volume_short += 1.0
        volume_reasons.append("Active bearish HTF volume-spike imbalance is still respected.")
    signal_groups.append(
        _group_from_strengths(
            "volume_flow",
            long_strength=volume_long,
            short_strength=volume_short,
            max_strength=7.0,
            max_delta=12.0,
            reasons=volume_reasons,
            neutral_delta=volume_delta,
        )
    )

    context_long = 0.0
    context_short = 0.0
    context_delta = 0.0
    context_reasons: list[str] = []
    context_regime = ""
    if realized_vol > 1.2:
        context_delta -= 5.0
        context_reasons.append("Realized volatility is very high; position sizing should shrink.")
    if spread_bps >= 12.0:
        context_delta -= 4.0
        context_reasons.append(f"Spread is wide at {spread_bps:.2f} bps; execution quality is weaker.")
    if open_interest_change_pct >= 3.0 and abs(taker_imbalance) < 0.05:
        context_delta -= 2.0
        context_reasons.append("Open interest is rising without decisive taker flow; crowding risk is building.")
    if (liquidity_sweep_high or liquidity_sweep_low) and volume_ratio_20 < 1.05:
        context_delta -= 2.0
        context_reasons.append("Liquidity sweep happened without enough relative volume; false-signal risk is higher.")
    if liquidity_reclaim_long and liquidity_close_position >= 0.58:
        context_long += 1.0
        context_reasons.append("Liquidity reclaim closed in the upper candle range after sell-side sweep.")
    elif liquidity_reclaim_short and liquidity_close_position <= 0.42:
        context_short += 1.0
        context_reasons.append("Liquidity reclaim closed in the lower candle range after buy-side sweep.")
    if funding_rate >= 0.0008 and open_interest_change_pct >= 3.0 and taker_ratio >= 1.08:
        context_short += 1.0
        context_delta -= 4.0
        context_reasons.append("Funding and open interest show a crowded long structure.")
        context_regime = "crowded-long"
    elif funding_rate <= -0.0008 and open_interest_change_pct >= 3.0 and taker_ratio <= 0.92:
        context_long += 1.0
        context_reasons.append("Funding and open interest show a crowded short structure with squeeze risk.")
        context_regime = "crowded-short"
    signal_groups.append(
        _group_from_strengths(
            "context_risk",
            long_strength=context_long,
            short_strength=context_short,
            max_strength=1.0,
            max_delta=6.0,
            reasons=context_reasons,
            neutral_delta=context_delta,
        )
    )

    if best_family.bias == "long":
        signal_groups.append(
            {
                "name": "strategy_family",
                "bias": "long",
                "score_delta": round(max(6.0, best_family.confidence * 14.0), 4),
                "confidence": best_family.confidence,
                "reasons": [f"Best alpha family is long: {best_family.family}.", *best_family.reasons],
                "counted": True,
                "family": best_family.family,
            }
        )
    elif best_family.bias == "short":
        signal_groups.append(
            {
                "name": "strategy_family",
                "bias": "short",
                "score_delta": round(-max(6.0, best_family.confidence * 14.0), 4),
                "confidence": best_family.confidence,
                "reasons": [f"Best alpha family is short: {best_family.family}.", *best_family.reasons],
                "counted": True,
                "family": best_family.family,
            }
        )
    else:
        signal_groups.append(
            {
                "name": "strategy_family",
                "bias": "neutral",
                "score_delta": 0.0,
                "confidence": 0.0,
                "reasons": ["No alpha family has enough independent confirmation."],
                "counted": True,
                "family": best_family.family,
            }
        )

    blave_notes: list[str] = []
    blave_long = 0.0
    blave_short = 0.0
    if blave_snapshot:
        hc = _snapshot_metric(blave_snapshot, "holder_concentration", "holder_concentration")
        whale = _snapshot_metric(blave_snapshot, "whale_hunter_24h_oi", "whale_hunter", "24h-score_oi")
        up_prob = _snapshot_metric(blave_snapshot, "up_prob", "statistics", "up_prob")
        if hc > 0.5:
            blave_notes.append(f"Blave holder concentration is supportive ({hc:.2f}).")
            blave_long += 1.0
        elif hc < -0.5:
            blave_notes.append(f"Blave holder concentration is bearish ({hc:.2f}).")
            blave_short += 1.0
        if whale > 0.5:
            blave_notes.append(f"Whale Hunter shows positive pressure ({whale:.2f}).")
            blave_long += 1.0
        elif whale < -0.5:
            blave_notes.append(f"Whale Hunter shows negative pressure ({whale:.2f}).")
            blave_short += 1.0
        if up_prob > 55:
            blave_notes.append(f"Blave historical up probability is elevated ({up_prob:.1f}%).")
            blave_long += 1.0
    if blave_snapshot:
        signal_groups.append(
            _group_from_strengths(
                "external_alpha",
                long_strength=blave_long,
                short_strength=blave_short,
                max_strength=3.0,
                max_delta=6.0,
                reasons=blave_notes,
            )
        )

    overlap_by_family = {
        "trend_continuation": {"trend_alignment"},
        "breakout": {"price_structure"},
        "trend_pullback": {"trend_alignment", "price_structure"},
        "mean_reversion": {"momentum_power"},
        "liquidity_reclaim": {"price_structure", "context_risk"},
        "vwap_reclaim": {"price_structure", "volume_flow"},
    }
    duplicate_groups = overlap_by_family.get(best_family.family, set())
    for group in signal_groups:
        if group["name"] in duplicate_groups and (group.get("bias") != "neutral" or group.get("score_delta")):
            group["counted"] = False
            group["duplicate_of_strategy_family"] = best_family.family

    for group in signal_groups:
        notes.extend(str(reason) for reason in group.get("reasons", []) if reason)

    score_delta = sum(
        _float(group.get("score_delta"))
        for group in signal_groups
        if bool(group.get("counted", True)) or str(group.get("bias")) == "neutral"
    )
    score = int(round(max(0.0, min(100.0, 50.0 + score_delta))))

    regime = "trend-up" if ema_fast_value > ema_slow_value and close > ema_slow_value else "trend-down"
    if 45 <= rsi_value <= 55 and adx_value < 20:
        regime = "range"
    elif bb_bandwidth <= 0.05 and adx_value < 20:
        regime = "squeeze"
    if context_regime:
        regime = context_regime

    counted_directional_groups = [
        group
        for group in signal_groups
        if bool(group.get("counted", True)) and group.get("bias") in {"long", "short"}
    ]
    bullish = sum(1 for group in counted_directional_groups if group.get("bias") == "long")
    bearish = sum(1 for group in counted_directional_groups if group.get("bias") == "short")
    total_signals = bullish + bearish
    long_confidence = sum(_float(group.get("confidence")) for group in counted_directional_groups if group.get("bias") == "long")
    short_confidence = sum(
        _float(group.get("confidence")) for group in counted_directional_groups if group.get("bias") == "short"
    )
    total_confidence = long_confidence + short_confidence
    convergence = round(max(long_confidence, short_confidence) / total_confidence, 3) if total_confidence else 0.0
    decision = decide_trade_action(
        market=market,
        score=score,
        convergence=convergence,
        adx_value=adx_value,
        regime=regime,
        min_score_long=strategy.risk.min_score_long if strategy else 65,
        max_score_short=strategy.risk.max_score_short if strategy else 35,
        min_convergence=strategy.risk.min_convergence if strategy else 0.6,
        min_adx=strategy.risk.min_adx if strategy else 20.0,
    )
    family_biases = {item.family: item.bias for item in family_signals}
    family_supports_action = (
        (decision.action == "BUY" and best_family.bias == "long")
        or (decision.action == "SELL" and best_family.bias == "short")
        or decision.action == "HOLD"
    )
    if decision.action in {"BUY", "SELL"} and not family_supports_action:
        decision = type(decision)(
            action="HOLD",
            directional_bias=decision.directional_bias,
            blockers=(
                *decision.blockers,
                "Active strategy family does not confirm the score-model direction.",
            ),
            rationale=decision.rationale,
        )
    decision_trace = [
        trace_step(
            "score_model",
            allowed=decision.allowed,
            reasons=list(decision.blockers),
            warnings=list(decision.rationale),
            data={
                "score": score,
                "convergence": convergence,
                "regime": regime,
                "action": decision.action,
                "bias": decision.directional_bias,
                "signal_groups": signal_groups,
            },
        ),
        trace_step(
            "strategy_family_pool",
            allowed=best_family.bias != "neutral" and family_supports_action,
            reasons=[]
            if best_family.bias != "neutral" and family_supports_action
            else ["no-active-family-confirmation" if best_family.bias == "neutral" else "family-score-direction-mismatch"],
            data={
                "active_families": family_biases,
                "selected": best_family.to_dict(),
            },
        ),
    ]

    return {
        "score": score,
        "bias": decision.directional_bias,
        "regime": regime,
        "signal_counts": {"bullish": bullish, "bearish": bearish, "total": total_signals},
        "signal_groups": signal_groups,
        "convergence": convergence,
        "recommended_action": decision.action,
        "entry_ready": decision.allowed,
        "entry_blockers": list(decision.blockers),
        "decision_notes": list(decision.rationale),
        "notes": notes + blave_notes,
        "strategy_families": [item.to_dict() for item in family_signals],
        "diagnostic_strategy_families": [item.to_dict() for item in diagnostic_family_signals],
        "selected_strategy_family": best_family.to_dict(),
        "decision_trace": decision_trace,
    }


def build_trade_plan(
    latest: pd.Series,
    bias: str,
    market: str,
    strategy: StrategyConfig | None = None,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    family = _selected_strategy_family_name(analysis)
    long_plan = indicator_trade_plan_side(latest, side="BUY", strategy=strategy, family=family)
    short_plan = indicator_trade_plan_side(latest, side="SELL", strategy=strategy, family=family)
    if market == "spot":
        short_plan["note"] = "Spot lane is analysis-only here; no short execution workflow is enabled."
    return {"preferred_bias": bias, "long": long_plan, "short": short_plan}


def render_chart(df: pd.DataFrame, symbol: str, interval: str, output_path: Path) -> None:
    latest = df.tail(180)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
    axes[0].plot(latest.index, latest["close"], label="Close", color="#0b7285", linewidth=1.6)
    axes[0].plot(latest.index, latest["ema_fast"], label="EMA 21", color="#f08c00", linewidth=1.2)
    axes[0].plot(latest.index, latest["ema_slow"], label="EMA 55", color="#c92a2a", linewidth=1.2)
    axes[0].set_title(f"{symbol} {interval} market structure")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.2)

    axes[1].plot(latest.index, latest["rsi_14"], color="#5f3dc4", linewidth=1.2)
    axes[1].axhline(70, linestyle="--", color="#adb5bd")
    axes[1].axhline(50, linestyle=":", color="#ced4da")
    axes[1].axhline(30, linestyle="--", color="#adb5bd")
    axes[1].set_ylabel("RSI")
    axes[1].grid(alpha=0.2)

    axes[2].bar(latest.index, latest["macd_hist"], color="#74c0fc", alpha=0.7)
    axes[2].plot(latest.index, latest["macd"], color="#1c7ed6", linewidth=1.1, label="MACD")
    axes[2].plot(latest.index, latest["macd_signal"], color="#e03131", linewidth=1.1, label="Signal")
    axes[2].set_ylabel("MACD")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_markdown_report(payload: dict[str, Any], path: Path) -> None:
    analysis = payload["analysis"]
    plan = payload["trade_plan"]
    lines = [
        f"# {payload['symbol']} {payload['market']} analysis",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Interval: `{payload['interval']}`",
        f"- Bias: `{analysis['bias']}`",
        f"- Setup score: `{analysis['score']}` / 100",
        f"- Regime: `{analysis['regime']}`",
        f"- Signal convergence: `{analysis['convergence']}`",
        "",
        "## Market Snapshot",
        "",
        f"- Close: `{payload['latest']['close']}`",
        f"- SMA200: `{payload['latest']['sma_200']}`",
        f"- EMA21 / EMA55: `{payload['latest']['ema_fast']}` / `{payload['latest']['ema_slow']}`",
        f"- RSI14: `{payload['latest']['rsi_14']}`",
        f"- MACD hist: `{payload['latest']['macd_hist']}`",
        f"- ATR14: `{payload['latest']['atr_14']}`",
        f"- ADX / +DI / -DI: `{payload['latest']['adx']}` / `{payload['latest']['plus_di']}` / `{payload['latest']['minus_di']}`",
        f"- VWAP / BB %B / BB Width: `{payload['latest']['vwap']}` / `{payload['latest']['bb_percent_b']}` / `{payload['latest']['bb_bandwidth']}`",
        f"- Rolling VWAP48 distance/reclaim L/S: `{payload['latest'].get('vwap_distance_pct_48')}` / `{payload['latest'].get('vwap_reclaim_long_48')}` `{payload['latest'].get('vwap_reclaim_short_48')}`",
        f"- Fib 89 retrace H/L / OTE long/short: `{payload['latest'].get('fib_retrace_from_high_89')}` / `{payload['latest'].get('fib_retrace_from_low_89')}` / `{payload['latest'].get('fib_ote_long_zone')}` `{payload['latest'].get('fib_ote_short_zone')}`",
        f"- Liquidity sweep/reclaim high/low: `{payload['latest'].get('liquidity_sweep_high_20')}` `{payload['latest'].get('liquidity_sweep_low_20')}` / `{payload['latest'].get('liquidity_reclaim_short_20')}` `{payload['latest'].get('liquidity_reclaim_long_20')}`",
        f"- SuperTrend / TrendMagic / FollowLine: `{payload['latest'].get('supertrend_direction')}` / `{payload['latest'].get('trend_magic_direction')}` / `{payload['latest'].get('follow_line_direction')}`",
        f"- JUMBO Power / MFI / CCI: `{payload['latest'].get('jumbo_power')}` / `{payload['latest'].get('mfi_14')}` / `{payload['latest'].get('cci_20')}`",
        f"- StochK / WilliamsR / UO proxy: `{payload['latest'].get('stoch_k')}` / `{payload['latest'].get('williams_r_14')}` / `{payload['latest'].get('ultimate_oscillator_proxy')}`",
        f"- Squeeze / Donchian / Chandelier / QQE: `{payload['latest'].get('squeeze_released')}` / `{payload['latest'].get('donchian_breakout_up')}` `{payload['latest'].get('donchian_breakout_down')}` / `{payload['latest'].get('chandelier_direction')}` / `{payload['latest'].get('qqe_direction')}`",
        f"- Selected alpha family: `{analysis.get('selected_strategy_family', {}).get('family')}` `{analysis.get('selected_strategy_family', {}).get('bias')}`",
        "- Signal groups: `"
        + ", ".join(
            f"{item.get('name')}={item.get('bias')}/{item.get('confidence')} counted={item.get('counted')}"
            for item in analysis.get("signal_groups", [])
        )
        + "`",
        f"- Diagnostic-only families: `{', '.join(item.get('family', '') for item in analysis.get('diagnostic_strategy_families', []))}`",
        f"- Realized vol(20): `{payload['latest']['realized_vol_20']}`",
        f"- Taker buy/sell ratio: `{payload['latest'].get('taker_buy_sell_ratio')}`",
        f"- Funding / OI change / spread: `{payload['latest'].get('funding_rate')}` / `{payload['latest'].get('open_interest_change_pct')}` / `{payload['latest'].get('spread_bps')}` bps",
        "",
        "## Trader Notes",
        "",
    ]
    for note in analysis["notes"]:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Playbook",
            "",
            f"- Long invalidation: `{plan['long']['invalidation']}`",
            f"- Long invalidation source: `{plan['long'].get('invalidation_source')}`",
            f"- Long TP1 / TP2: `{plan['long']['take_profit_1']}` / `{plan['long']['take_profit_2']}`",
            f"- Short invalidation: `{plan['short']['invalidation']}`",
            f"- Short invalidation source: `{plan['short'].get('invalidation_source')}`",
            f"- Short TP1 / TP2: `{plan['short']['take_profit_1']}` / `{plan['short']['take_profit_2']}`",
            "",
            "## Risk",
            "",
            "- This is a decision-support report, not live execution advice.",
            "- Use testnet or paper sizing first, then validate fees, slippage, and latency before any live deployment.",
            "- Keep withdrawal permission disabled on API keys.",
        ]
    )
    if payload.get("blave"):
        lines.extend(["", "## Blave Snapshot", ""])
        for key, value in payload["blave"].items():
            lines.append(f"- {key}: `{value}`")
    if payload.get("market_context"):
        lines.extend(["", "## Market Context", ""])
        for key, value in payload["market_context"].items():
            lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    settings: Settings,
    *,
    symbol: str,
    market: str,
    interval: str,
    limit: int,
    use_blave: bool,
    render_chart_flag: bool,
    strategy: StrategyConfig | None = None,
    output_dir: Path | None = None,
) -> tuple[dict[str, Any], AnalysisArtifacts]:
    ensure_runtime_dirs()
    run_id = f"{now_stamp()}-{symbol.lower()}-{market}-{interval}"
    root = output_dir or (REPORTS_DIR / run_id)
    root.mkdir(parents=True, exist_ok=True)
    report_json = root / "analysis.json"
    report_md = root / "analysis.md"
    chart_path = root / "chart.png" if render_chart_flag else None

    with BinanceClient(settings) as client:
        raw_klines = client.klines(symbol, interval, limit, market)
        df = enrich_indicators(prepare_klines_frame(raw_klines), interval, strategy=strategy)
        market_context = summarize_market_context(client=client, symbol=symbol, market=market, df=df)
        timeframe_frames: dict[str, pd.DataFrame] = {interval: df}
        for tf in ("1m", "15m", "4h", "1d"):
            if tf == interval:
                continue
            tf_limit = 500 if tf in {"1m", "15m"} else 360
            try:
                tf_klines = client.klines(symbol, tf, tf_limit, market)
                timeframe_frames[tf] = enrich_indicators(
                    prepare_klines_frame(tf_klines),
                    tf,
                    strategy=strategy,
                )
            except Exception as exc:
                market_context["data_quality_notes"].append(f"multi-timeframe-{tf}-unavailable:{exc}")
        mtf = evaluate_multi_timeframe_structure(timeframe_frames)
        market_context["multi_timeframe_structure"] = {
            "name": mtf.name,
            "bias": mtf.bias,
            "score": mtf.score,
            "confidence": mtf.confidence,
            "alignment": mtf.alignment,
            "structures": [asdict(item) for item in mtf.structures],
            "ema_stack": "7/25/89",
        }
    latest = df.iloc[-1]

    blave_snapshot = None
    if use_blave and settings.has_blave_credentials:
        with BlaveClient(settings) as blave:
            row = blave.latest_snapshot(symbol)
        if row:
            blave_snapshot = {
                "holder_concentration": _float((row.get("holder_concentration") or {}).get("-")),
                "whale_hunter_24h_oi": _float((row.get("whale_hunter") or {}).get("24h-score_oi")),
                "price_change_24h": _float((row.get("price_change") or {}).get("24h")),
                "up_prob": _float((row.get("statistics") or {}).get("up_prob")),
                "exp_value": _float((row.get("statistics") or {}).get("exp_value")),
            }

    analysis = score_bias(
        latest,
        market,
        blave_snapshot,
        market_context=market_context,
        strategy=strategy,
    )
    trade_plan = build_trade_plan(latest, analysis["bias"], market, strategy=strategy, analysis=analysis)

    latest_payload = {
        "close": round(_float(latest["close"]), 6),
        "sma_200": round(_float(latest["sma_200"]), 6),
        "ema_fast": round(_float(latest["ema_fast"]), 6),
        "ema_slow": round(_float(latest["ema_slow"]), 6),
        "rsi_14": round(_float(latest["rsi_14"]), 4),
        "atr_14": round(_float(latest["atr_14"]), 6),
        "macd": round(_float(latest["macd"]), 6),
        "macd_signal": round(_float(latest["macd_signal"]), 6),
        "macd_hist": round(_float(latest["macd_hist"]), 6),
        "adx": round(_float(latest["adx"]), 4),
        "plus_di": round(_float(latest["plus_di"]), 4),
        "minus_di": round(_float(latest["minus_di"]), 4),
        "bb_percent_b": round(_float(latest["bb_percent_b"]), 4),
        "bb_bandwidth": round(_float(latest["bb_bandwidth"]), 6),
        "fib_swing_high_89": round(_float(latest.get("fib_swing_high_89")), 6),
        "fib_swing_low_89": round(_float(latest.get("fib_swing_low_89")), 6),
        "fib_retrace_from_high_89": round(_float(latest.get("fib_retrace_from_high_89")), 4),
        "fib_retrace_from_low_89": round(_float(latest.get("fib_retrace_from_low_89")), 4),
        "fib_pullback_long_zone": bool(latest.get("fib_pullback_long_zone", False)),
        "fib_pullback_short_zone": bool(latest.get("fib_pullback_short_zone", False)),
        "fib_ote_long_zone": bool(latest.get("fib_ote_long_zone", False)),
        "fib_ote_short_zone": bool(latest.get("fib_ote_short_zone", False)),
        "liquidity_swing_high_20": round(_float(latest.get("liquidity_swing_high_20")), 6),
        "liquidity_swing_low_20": round(_float(latest.get("liquidity_swing_low_20")), 6),
        "liquidity_sweep_high_20": bool(latest.get("liquidity_sweep_high_20", False)),
        "liquidity_sweep_low_20": bool(latest.get("liquidity_sweep_low_20", False)),
        "liquidity_reclaim_long_20": bool(latest.get("liquidity_reclaim_long_20", False)),
        "liquidity_reclaim_short_20": bool(latest.get("liquidity_reclaim_short_20", False)),
        "liquidity_close_position": round(_float(latest.get("liquidity_close_position"), 0.5), 4),
        "vwap": round(_float(latest["vwap"]), 6),
        "vwap_rolling_48": round(_float(latest.get("vwap_rolling_48")), 6),
        "vwap_distance_pct_48": round(_float(latest.get("vwap_distance_pct_48")), 4),
        "vwap_upper_1_48": round(_float(latest.get("vwap_upper_1_48")), 6),
        "vwap_lower_1_48": round(_float(latest.get("vwap_lower_1_48")), 6),
        "vwap_upper_2_48": round(_float(latest.get("vwap_upper_2_48")), 6),
        "vwap_lower_2_48": round(_float(latest.get("vwap_lower_2_48")), 6),
        "vwap_reclaim_long_48": bool(latest.get("vwap_reclaim_long_48", False)),
        "vwap_reclaim_short_48": bool(latest.get("vwap_reclaim_short_48", False)),
        "vwap_mid_reclaim_long_48": bool(latest.get("vwap_mid_reclaim_long_48", False)),
        "vwap_mid_reclaim_short_48": bool(latest.get("vwap_mid_reclaim_short_48", False)),
        "vwma_20": round(_float(latest["vwma_20"]), 6),
        "supertrend": round(_float(latest["supertrend"]), 6),
        "supertrend_direction": int(_float(latest["supertrend_direction"])),
        "trend_magic": round(_float(latest["trend_magic"]), 6),
        "trend_magic_direction": int(_float(latest["trend_magic_direction"])),
        "follow_line": round(_float(latest["follow_line"]), 6),
        "follow_line_direction": int(_float(latest["follow_line_direction"])),
        "jumbo_power": round(_float(latest["jumbo_power"]), 4),
        "jumbo_power_ma": round(_float(latest["jumbo_power_ma"]), 4),
        "mfi_14": round(_float(latest["mfi_14"], 50.0), 4),
        "cci_20": round(_float(latest["cci_20"]), 4),
        "stoch_k": round(_float(latest["stoch_k"], 50.0), 4),
        "stoch_d": round(_float(latest["stoch_d"], 50.0), 4),
        "williams_r_14": round(_float(latest["williams_r_14"], -50.0), 4),
        "ultimate_oscillator_proxy": round(_float(latest["ultimate_oscillator_proxy"], 50.0), 4),
        "volume_ratio_20": round(_float(latest["volume_ratio_20"], 1.0), 4),
        "squeeze_on": bool(latest["squeeze_on"]),
        "squeeze_off": bool(latest["squeeze_off"]),
        "squeeze_released": bool(latest["squeeze_released"]),
        "squeeze_momentum": round(_float(latest["squeeze_momentum"]), 6),
        "donchian_upper": round(_float(latest["donchian_upper"]), 6),
        "donchian_lower": round(_float(latest["donchian_lower"]), 6),
        "donchian_breakout_up": bool(latest["donchian_breakout_up"]),
        "donchian_breakout_down": bool(latest["donchian_breakout_down"]),
        "keltner_upper": round(_float(latest["keltner_upper"]), 6),
        "keltner_lower": round(_float(latest["keltner_lower"]), 6),
        "chandelier_direction": int(_float(latest["chandelier_direction"])),
        "qqe_direction": int(_float(latest["qqe_direction"])),
        "qqe_rsi": round(_float(latest["qqe_rsi"], 50.0), 4),
        "stoch_rsi_k": round(_float(latest["stoch_rsi_k"], 50.0), 4),
        "stoch_rsi_d": round(_float(latest["stoch_rsi_d"], 50.0), 4),
        "ichimoku_direction": int(_float(latest["ichimoku_direction"])),
        "psar_direction": int(_float(latest["psar_direction"])),
        "obv_zscore_20": round(_float(latest["obv_zscore_20"]), 4),
        "volume_zscore_20": round(_float(latest["volume_zscore_20"]), 4),
        "breakout_high_20": round(_float(latest["breakout_high_20"]), 6),
        "breakout_low_20": round(_float(latest["breakout_low_20"]), 6),
        "realized_vol_20": round(_float(latest["realized_vol_20"]), 6),
        "volume": round(_float(latest["volume"]), 4),
        "taker_buy_sell_ratio": round(_float(market_context.get("taker_buy_sell_ratio"), 1.0), 6),
        "taker_flow_imbalance": round(_float(market_context.get("taker_flow_imbalance"), 0.0), 6),
        "funding_rate": _optional_rounded(market_context.get("funding_rate"), 8),
        "open_interest_change_pct": _optional_rounded(market_context.get("open_interest_change_pct"), 4),
        "order_book_imbalance": round(_float(market_context.get("order_book_imbalance"), 0.0), 6),
        "spread_bps": round(_float(market_context.get("spread_bps"), 0.0), 4),
        "multi_timeframe_bias": (market_context.get("multi_timeframe_structure") or {}).get("bias"),
        "multi_timeframe_alignment": (market_context.get("multi_timeframe_structure") or {}).get("alignment"),
        "volume_profile_position": (market_context.get("volume_profile") or {}).get("close_position"),
        "volume_bubble_cluster": (market_context.get("volume_bubbles") or {}).get("cluster"),
        "volume_bubble_side": (market_context.get("volume_bubbles") or {}).get("side"),
        "htf_volume_imbalance_direction": (market_context.get("htf_volume_imbalance") or {}).get("direction"),
        "htf_volume_imbalance_active": (market_context.get("htf_volume_imbalance") or {}).get("active"),
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "symbol": symbol.upper(),
        "market": market,
        "interval": interval,
        "limit": limit,
        "strategy_profile": strategy.profile if strategy else "built-in-defaults",
        "strategy_path": str(strategy.path) if strategy else None,
        "analysis": analysis,
        "latest": latest_payload,
        "trade_plan": trade_plan,
        "blave": blave_snapshot,
        "market_context": market_context,
        "artifacts": {
            "output_dir": str(root),
            "report_json": str(report_json),
            "report_md": str(report_md),
            "chart_path": str(chart_path) if chart_path else None,
        },
    }
    report_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(payload, report_md)
    if chart_path:
        render_chart(df, symbol.upper(), interval, chart_path)

    return payload, AnalysisArtifacts(
        run_id=run_id,
        output_dir=root,
        report_json=report_json,
        report_md=report_md,
        chart_path=chart_path,
    )
