from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .signals import SignalBias
from .strategy import StrategyConfig

ACTIVE_STRATEGY_FAMILIES: tuple[str, ...] = (
    "ai_family_router",
    "trend_continuation",
    "breakout",
    "trend_pullback",
    "liquidity_reclaim",
    "vwap_reclaim",
    "mean_reversion",
)


@dataclass(frozen=True, slots=True)
class StrategyFamilySignal:
    family: str
    bias: SignalBias
    score: float
    confidence: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _float(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        parsed = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    if pd.isna(parsed):
        return default
    return parsed


def _bool(row: pd.Series, key: str) -> bool:
    return bool(row.get(key, False))


def _signal_from_votes(
    *,
    family: str,
    long_votes: int,
    short_votes: int,
    max_votes: int,
    reasons: list[str],
) -> StrategyFamilySignal:
    if long_votes > short_votes and long_votes >= 2:
        bias: SignalBias = "long"
        raw = long_votes
    elif short_votes > long_votes and short_votes >= 2:
        bias = "short"
        raw = short_votes
    else:
        bias = "neutral"
        raw = max(long_votes, short_votes)
    confidence = min(raw / max(max_votes, 1), 1.0)
    signed_score = (long_votes - short_votes) / max(max_votes, 1)
    return StrategyFamilySignal(
        family=family,
        bias=bias,
        score=round(signed_score, 4),
        confidence=round(confidence, 4),
        reasons=tuple(reasons),
    )


def trend_continuation_signal(row: pd.Series) -> StrategyFamilySignal:
    long_votes = 0
    short_votes = 0
    reasons: list[str] = []
    close = _float(row, "close")
    ema_fast = _float(row, "ema_fast")
    ema_slow = _float(row, "ema_slow")
    macd_hist = _float(row, "macd_hist")
    adx = _float(row, "adx")
    plus_di = _float(row, "plus_di")
    minus_di = _float(row, "minus_di")
    if close > ema_slow and ema_fast > ema_slow:
        long_votes += 1
        reasons.append("ema-stack-long")
    elif close < ema_slow and ema_fast < ema_slow:
        short_votes += 1
        reasons.append("ema-stack-short")
    if macd_hist > 0:
        long_votes += 1
        reasons.append("macd-positive")
    elif macd_hist < 0:
        short_votes += 1
        reasons.append("macd-negative")
    if adx >= 20 and plus_di > minus_di:
        long_votes += 1
        reasons.append("adx-di-long")
    elif adx >= 20 and minus_di > plus_di:
        short_votes += 1
        reasons.append("adx-di-short")
    return _signal_from_votes(
        family="trend_continuation",
        long_votes=long_votes,
        short_votes=short_votes,
        max_votes=3,
        reasons=reasons,
    )


def volatility_breakout_signal(row: pd.Series) -> StrategyFamilySignal:
    long_votes = 0
    short_votes = 0
    reasons: list[str] = []
    squeeze_momentum = _float(row, "squeeze_momentum")
    if _bool(row, "squeeze_released") or _bool(row, "squeeze_off"):
        if squeeze_momentum > 0:
            long_votes += 1
            reasons.append("squeeze-release-up")
        elif squeeze_momentum < 0:
            short_votes += 1
            reasons.append("squeeze-release-down")
    if _bool(row, "donchian_breakout_up"):
        long_votes += 1
        reasons.append("donchian-breakout-up")
    elif _bool(row, "donchian_breakout_down"):
        short_votes += 1
        reasons.append("donchian-breakout-down")
    if _float(row, "bb_bandwidth") > _float(row, "keltner_width_pct"):
        if _float(row, "close") > _float(row, "keltner_upper"):
            long_votes += 1
            reasons.append("keltner-expansion-up")
        elif _float(row, "close") < _float(row, "keltner_lower"):
            short_votes += 1
            reasons.append("keltner-expansion-down")
    return _signal_from_votes(
        family="volatility_breakout",
        long_votes=long_votes,
        short_votes=short_votes,
        max_votes=3,
        reasons=reasons,
    )


def breakout_signal(row: pd.Series) -> StrategyFamilySignal:
    """High-yield breakout family with volatility expansion and volume filters.

    This intentionally absorbs the old high-beta momentum inputs instead of
    allowing a separate correlated family to vote again.
    """

    long_votes = 0
    short_votes = 0
    reasons: list[str] = []
    squeeze_momentum = _float(row, "squeeze_momentum")
    volume_ratio = _float(row, "volume_ratio_20", 1.0)
    rsi = _float(row, "rsi_14", 50.0)

    if _bool(row, "squeeze_released") or _bool(row, "squeeze_off"):
        if squeeze_momentum > 0:
            long_votes += 1
            reasons.append("squeeze-release-up")
        elif squeeze_momentum < 0:
            short_votes += 1
            reasons.append("squeeze-release-down")

    if _bool(row, "donchian_breakout_up"):
        long_votes += 1
        reasons.append("donchian-breakout-up")
    elif _bool(row, "donchian_breakout_down"):
        short_votes += 1
        reasons.append("donchian-breakout-down")

    if _float(row, "bb_bandwidth") > _float(row, "keltner_width_pct"):
        if _float(row, "close") > _float(row, "keltner_upper"):
            long_votes += 1
            reasons.append("keltner-expansion-up")
        elif _float(row, "close") < _float(row, "keltner_lower"):
            short_votes += 1
            reasons.append("keltner-expansion-down")

    if volume_ratio >= 1.5:
        if rsi >= 58 and _float(row, "qqe_direction") >= 0:
            long_votes += 1
            reasons.append("volume-rsi-breakout-up")
        elif rsi <= 42 and _float(row, "qqe_direction") <= 0:
            short_votes += 1
            reasons.append("volume-rsi-breakout-down")

    return _signal_from_votes(
        family="breakout",
        long_votes=long_votes,
        short_votes=short_votes,
        max_votes=4,
        reasons=reasons,
    )


def trend_pullback_signal(row: pd.Series) -> StrategyFamilySignal:
    """Fibonacci/OTE pullback family for trend entries without late breakout chase."""

    long_votes = 0
    short_votes = 0
    reasons: list[str] = []
    close = _float(row, "close")
    ema_fast = _float(row, "ema_fast")
    ema_slow = _float(row, "ema_slow")
    sma_200 = _float(row, "sma_200")
    adx = _float(row, "adx")
    plus_di = _float(row, "plus_di")
    minus_di = _float(row, "minus_di")
    rsi = _float(row, "rsi_14", 50.0)
    stoch_k = _float(row, "stoch_rsi_k", 50.0)
    bb_percent_b = _float(row, "bb_percent_b", 0.5)
    volume_ratio = _float(row, "volume_ratio_20", 1.0)
    trend_votes = int(_float(row, "supertrend_direction")) + int(_float(row, "trend_magic_direction")) + int(
        _float(row, "follow_line_direction")
    )
    jumbo_power = _float(row, "jumbo_power")
    jumbo_ma = _float(row, "jumbo_power_ma")

    uptrend = ema_fast > ema_slow and close > ema_slow and (sma_200 <= 0 or close > sma_200)
    downtrend = ema_fast < ema_slow and close < ema_slow and (sma_200 <= 0 or close < sma_200)
    if uptrend and adx >= 16 and plus_di >= minus_di:
        long_votes += 1
        reasons.append("pullback-uptrend-context")
    elif downtrend and adx >= 16 and minus_di >= plus_di:
        short_votes += 1
        reasons.append("pullback-downtrend-context")

    if _bool(row, "fib_ote_long_zone"):
        long_votes += 2
        reasons.append("fib-ote-long-zone")
    elif _bool(row, "fib_pullback_long_zone"):
        long_votes += 1
        reasons.append("fib-pullback-long-zone")
    if _bool(row, "fib_ote_short_zone"):
        short_votes += 2
        reasons.append("fib-ote-short-zone")
    elif _bool(row, "fib_pullback_short_zone"):
        short_votes += 1
        reasons.append("fib-pullback-short-zone")

    if 42.0 <= rsi <= 62.0 and stoch_k <= 65.0 and bb_percent_b <= 0.78:
        long_votes += 1
        reasons.append("oscillator-reset-long")
    elif 38.0 <= rsi <= 58.0 and stoch_k >= 35.0 and bb_percent_b >= 0.22:
        short_votes += 1
        reasons.append("oscillator-reset-short")

    if trend_votes >= 1:
        long_votes += 1
        reasons.append("trend-trails-not-bearish")
    elif trend_votes <= -1:
        short_votes += 1
        reasons.append("trend-trails-not-bullish")

    if volume_ratio >= 1.05:
        if jumbo_power >= jumbo_ma - 8.0 and jumbo_power > -25.0:
            long_votes += 1
            reasons.append("volume-jumbo-supports-pullback-long")
        elif jumbo_power <= jumbo_ma + 8.0 and jumbo_power < 25.0:
            short_votes += 1
            reasons.append("volume-jumbo-supports-pullback-short")

    return _signal_from_votes(
        family="trend_pullback",
        long_votes=long_votes,
        short_votes=short_votes,
        max_votes=6,
        reasons=reasons,
    )


def liquidity_reclaim_signal(
    row: pd.Series,
    market_context: dict[str, Any] | None = None,
) -> StrategyFamilySignal:
    """Liquidity sweep/reclaim family for failed breakouts back into range.

    The family is a structure-first research lane: it needs a prior swing sweep,
    a close back inside the range, and confirmation from volume or taker flow.
    """

    context = market_context or {}
    taker_imbalance = float(context.get("taker_flow_imbalance") or 0.0)
    order_book_imbalance = float(context.get("order_book_imbalance") or 0.0)
    funding = float(context.get("funding_rate") or 0.0)
    oi_change = float(context.get("open_interest_change_pct") or 0.0)
    long_votes = 0
    short_votes = 0
    reasons: list[str] = []
    rsi = _float(row, "rsi_14", 50.0)
    stoch_k = _float(row, "stoch_rsi_k", 50.0)
    close_position = _float(row, "liquidity_close_position", 0.5)
    volume_ratio = _float(row, "volume_ratio_20", 1.0)
    bb_percent_b = _float(row, "bb_percent_b", 0.5)
    adx = _float(row, "adx")

    if _bool(row, "liquidity_reclaim_long_20"):
        long_votes += 2
        reasons.append("sell-side-liquidity-sweep-reclaimed")
    elif _bool(row, "liquidity_sweep_low_20"):
        reasons.append("sell-side-liquidity-sweep-no-reclaim")
    if _bool(row, "liquidity_reclaim_short_20"):
        short_votes += 2
        reasons.append("buy-side-liquidity-sweep-rejected")
    elif _bool(row, "liquidity_sweep_high_20"):
        reasons.append("buy-side-liquidity-sweep-no-rejection")

    if volume_ratio >= 1.15:
        if close_position >= 0.58:
            long_votes += 1
            reasons.append("reclaim-volume-confirmed-long")
        elif close_position <= 0.42:
            short_votes += 1
            reasons.append("reclaim-volume-confirmed-short")

    if taker_imbalance >= 0.06 or order_book_imbalance >= 0.10:
        long_votes += 1
        reasons.append("flow-confirms-reclaim-long")
    elif taker_imbalance <= -0.06 or order_book_imbalance <= -0.10:
        short_votes += 1
        reasons.append("flow-confirms-reclaim-short")

    if funding <= -0.0005 and oi_change >= 2.0 and _bool(row, "liquidity_reclaim_long_20"):
        long_votes += 1
        reasons.append("crowded-shorts-after-sell-side-sweep")
    elif funding >= 0.0005 and oi_change >= 2.0 and _bool(row, "liquidity_reclaim_short_20"):
        short_votes += 1
        reasons.append("crowded-longs-after-buy-side-sweep")

    if _bool(row, "liquidity_reclaim_long_20") and rsi <= 62.0 and stoch_k <= 70.0 and bb_percent_b <= 0.72:
        long_votes += 1
        reasons.append("long-reclaim-not-overextended")
    elif _bool(row, "liquidity_reclaim_short_20") and rsi >= 38.0 and stoch_k >= 30.0 and bb_percent_b >= 0.28:
        short_votes += 1
        reasons.append("short-reclaim-not-overextended")

    if adx >= 32.0:
        if long_votes > short_votes and _float(row, "minus_di") > _float(row, "plus_di"):
            long_votes = max(0, long_votes - 1)
            reasons.append("strong-downtrend-penalizes-long-reclaim")
        elif short_votes > long_votes and _float(row, "plus_di") > _float(row, "minus_di"):
            short_votes = max(0, short_votes - 1)
            reasons.append("strong-uptrend-penalizes-short-reclaim")

    return _signal_from_votes(
        family="liquidity_reclaim",
        long_votes=long_votes,
        short_votes=short_votes,
        max_votes=6,
        reasons=reasons,
    )


def vwap_reclaim_signal(row: pd.Series) -> StrategyFamilySignal:
    """VWAP fair-value reclaim family using rolling VWAP bands as structure."""

    long_votes = 0
    short_votes = 0
    reasons: list[str] = []
    close = _float(row, "close")
    vwap_value = _float(row, "vwap_rolling_48")
    distance_pct = _float(row, "vwap_distance_pct_48")
    bb_percent_b = _float(row, "bb_percent_b", 0.5)
    rsi = _float(row, "rsi_14", 50.0)
    stoch_k = _float(row, "stoch_rsi_k", 50.0)
    volume_ratio = _float(row, "volume_ratio_20", 1.0)
    obv_zscore = _float(row, "obv_zscore_20")
    adx = _float(row, "adx")
    plus_di = _float(row, "plus_di")
    minus_di = _float(row, "minus_di")

    if _bool(row, "vwap_reclaim_long_48"):
        long_votes += 2
        reasons.append("vwap-lower-band-reclaimed")
    elif _bool(row, "vwap_mid_reclaim_long_48") and distance_pct <= 0.35:
        long_votes += 1
        reasons.append("vwap-midline-reclaimed-long")

    if _bool(row, "vwap_reclaim_short_48"):
        short_votes += 2
        reasons.append("vwap-upper-band-rejected")
    elif _bool(row, "vwap_mid_reclaim_short_48") and distance_pct >= -0.35:
        short_votes += 1
        reasons.append("vwap-midline-reclaimed-short")

    if close > 0.0 and vwap_value > 0.0:
        if close >= vwap_value and distance_pct <= 0.65:
            long_votes += 1
            reasons.append("price-above-vwap-without-chase")
        elif close <= vwap_value and distance_pct >= -0.65:
            short_votes += 1
            reasons.append("price-below-vwap-without-chase")

    if volume_ratio >= 1.05 or abs(obv_zscore) >= 0.7:
        if long_votes > short_votes and obv_zscore >= -0.4:
            long_votes += 1
            reasons.append("volume-confirms-vwap-long")
        elif short_votes > long_votes and obv_zscore <= 0.4:
            short_votes += 1
            reasons.append("volume-confirms-vwap-short")

    if long_votes > short_votes:
        if rsi <= 64.0 and stoch_k <= 72.0 and bb_percent_b <= 0.78:
            long_votes += 1
            reasons.append("long-reclaim-not-overextended")
        if adx >= 30.0 and minus_di > plus_di:
            long_votes = max(0, long_votes - 1)
            reasons.append("strong-downtrend-penalizes-vwap-long")
    elif short_votes > long_votes:
        if rsi >= 36.0 and stoch_k >= 28.0 and bb_percent_b >= 0.22:
            short_votes += 1
            reasons.append("short-reclaim-not-overextended")
        if adx >= 30.0 and plus_di > minus_di:
            short_votes = max(0, short_votes - 1)
            reasons.append("strong-uptrend-penalizes-vwap-short")

    return _signal_from_votes(
        family="vwap_reclaim",
        long_votes=long_votes,
        short_votes=short_votes,
        max_votes=6,
        reasons=reasons,
    )


def high_beta_momentum_signal(row: pd.Series) -> StrategyFamilySignal:
    long_votes = 0
    short_votes = 0
    reasons: list[str] = []
    volume_ratio = _float(row, "volume_ratio_20", 1.0)
    rsi = _float(row, "rsi_14", 50.0)
    if volume_ratio >= 1.5 and _bool(row, "donchian_breakout_up"):
        long_votes += 1
        reasons.append("volume-breakout-up")
    elif volume_ratio >= 1.5 and _bool(row, "donchian_breakout_down"):
        short_votes += 1
        reasons.append("volume-breakout-down")
    if rsi >= 58 and _float(row, "qqe_direction") >= 0:
        long_votes += 1
        reasons.append("rsi-qqe-strength")
    elif rsi <= 42 and _float(row, "qqe_direction") <= 0:
        short_votes += 1
        reasons.append("rsi-qqe-weakness")
    if _float(row, "jumbo_power") > _float(row, "jumbo_power_ma") and _float(row, "jumbo_power") >= 35:
        long_votes += 1
        reasons.append("jumbo-power-long")
    elif _float(row, "jumbo_power") < _float(row, "jumbo_power_ma") and _float(row, "jumbo_power") <= -35:
        short_votes += 1
        reasons.append("jumbo-power-short")
    return _signal_from_votes(
        family="high_beta_momentum",
        long_votes=long_votes,
        short_votes=short_votes,
        max_votes=3,
        reasons=reasons,
    )


def reversal_squeeze_signal(row: pd.Series, market_context: dict[str, Any] | None = None) -> StrategyFamilySignal:
    context = market_context or {}
    funding = float(context.get("funding_rate") or 0.0)
    oi_change = float(context.get("open_interest_change_pct") or 0.0)
    taker_ratio = float(context.get("taker_buy_sell_ratio") or 1.0)
    stoch_k = _float(row, "stoch_rsi_k", 50.0)
    bb_percent_b = _float(row, "bb_percent_b", 0.5)
    long_votes = 0
    short_votes = 0
    reasons: list[str] = []
    if funding <= -0.0008 and oi_change >= 3.0 and taker_ratio <= 0.92:
        long_votes += 2
        reasons.append("crowded-short-squeeze-risk")
    elif funding >= 0.0008 and oi_change >= 3.0 and taker_ratio >= 1.08:
        short_votes += 2
        reasons.append("crowded-long-squeeze-risk")
    if stoch_k <= 20 and bb_percent_b <= 0.15:
        long_votes += 1
        reasons.append("oversold-reversal-zone")
    elif stoch_k >= 80 and bb_percent_b >= 0.85:
        short_votes += 1
        reasons.append("overbought-reversal-zone")
    return _signal_from_votes(
        family="reversal_squeeze",
        long_votes=long_votes,
        short_votes=short_votes,
        max_votes=3,
        reasons=reasons,
    )


def range_mean_reversion_signal(row: pd.Series) -> StrategyFamilySignal:
    long_votes = 0
    short_votes = 0
    reasons: list[str] = []
    adx = _float(row, "adx")
    rsi = _float(row, "rsi_14", 50.0)
    stoch_k = _float(row, "stoch_rsi_k", 50.0)
    bb_percent_b = _float(row, "bb_percent_b", 0.5)
    if adx <= 18 and rsi <= 35 and stoch_k <= 20 and bb_percent_b <= 0.15:
        long_votes += 3
        reasons.append("range-oversold")
    elif adx <= 18 and rsi >= 65 and stoch_k >= 80 and bb_percent_b >= 0.85:
        short_votes += 3
        reasons.append("range-overbought")
    return _signal_from_votes(
        family="range_mean_reversion",
        long_votes=long_votes,
        short_votes=short_votes,
        max_votes=3,
        reasons=reasons,
    )


def mean_reversion_signal(row: pd.Series, market_context: dict[str, Any] | None = None) -> StrategyFamilySignal:
    """Range/reversal family using oscillator location and squeeze/crowding context."""

    range_signal = range_mean_reversion_signal(row)
    squeeze_signal = reversal_squeeze_signal(row, market_context)
    long_votes = 0
    short_votes = 0
    reasons: list[str] = []
    for signal in (range_signal, squeeze_signal):
        if signal.bias == "long":
            long_votes += max(1, int(round(signal.confidence * 3)))
            reasons.extend(signal.reasons)
        elif signal.bias == "short":
            short_votes += max(1, int(round(signal.confidence * 3)))
            reasons.extend(signal.reasons)
    if range_signal.bias != "neutral" and squeeze_signal.bias == range_signal.bias:
        max_votes = 6
    else:
        max_votes = 3
    return _signal_from_votes(
        family="mean_reversion",
        long_votes=long_votes,
        short_votes=short_votes,
        max_votes=max_votes,
        reasons=reasons,
    )


def build_strategy_family_signals(
    row: pd.Series,
    market_context: dict[str, Any] | None = None,
) -> tuple[StrategyFamilySignal, ...]:
    return (
        trend_continuation_signal(row),
        breakout_signal(row),
        trend_pullback_signal(row),
        liquidity_reclaim_signal(row, market_context),
        vwap_reclaim_signal(row),
        mean_reversion_signal(row, market_context),
    )


def build_diagnostic_family_signals(
    row: pd.Series,
    market_context: dict[str, Any] | None = None,
) -> tuple[StrategyFamilySignal, ...]:
    return (
        volatility_breakout_signal(row),
        high_beta_momentum_signal(row),
        reversal_squeeze_signal(row, market_context),
        range_mean_reversion_signal(row),
    )


def select_best_family(signals: tuple[StrategyFamilySignal, ...]) -> StrategyFamilySignal:
    actionable = [item for item in signals if item.bias != "neutral"]
    if not actionable:
        return StrategyFamilySignal("none", "neutral", 0.0, 0.0, ())
    return max(actionable, key=lambda item: (item.confidence, abs(item.score)))


def ai_family_router_signal(
    row: pd.Series,
    market_context: dict[str, Any] | None = None,
) -> StrategyFamilySignal:
    candidates = [
        item
        for item in build_strategy_family_signals(row, market_context)
        if item.family in {"trend_continuation", "breakout", "trend_pullback", "liquidity_reclaim", "vwap_reclaim"}
        and item.bias != "neutral"
    ]
    if not candidates:
        return StrategyFamilySignal("ai_family_router", "neutral", 0.0, 0.0, ())
    best = max(
        candidates,
        key=lambda item: (
            item.confidence,
            abs(item.score),
            1 if item.family in {"trend_pullback", "liquidity_reclaim", "vwap_reclaim"} else 0,
        ),
    )
    return StrategyFamilySignal(
        family="ai_family_router",
        bias=best.bias,
        score=best.score,
        confidence=best.confidence,
        reasons=(f"router-selected:{best.family}", *best.reasons),
    )


def strategy_family_trade_decision(
    row: pd.Series,
    *,
    market: str,
    family: str,
    market_context: dict[str, Any] | None = None,
    strategy: StrategyConfig | None = None,
) -> dict[str, Any]:
    """Build an analysis-like decision from one independent strategy family."""

    base_signals = build_strategy_family_signals(row, market_context)
    family_map = {signal.family: signal for signal in base_signals}
    family_map["ai_family_router"] = ai_family_router_signal(row, market_context)
    signal = family_map.get(family)
    if signal is None:
        valid = ", ".join(ACTIVE_STRATEGY_FAMILIES)
        raise ValueError(f"Unknown strategy family {family!r}; expected one of: {valid}")
    routed_family = ""
    if family == "ai_family_router":
        routed_family = next(
            (
                reason.split(":", 1)[1]
                for reason in signal.reasons
                if reason.startswith("router-selected:")
            ),
            "",
        )

    min_confidence = strategy.risk.min_convergence if strategy else 0.6
    adx_value = _float(row, "adx")
    blockers: list[str] = []
    notes: list[str] = []
    action = "HOLD"
    bias = "neutral"

    if signal.bias == "long":
        action = "BUY"
        bias = "long-bias"
    elif signal.bias == "short":
        action = "SELL" if market == "futures" else "HOLD"
        bias = "short-bias" if market == "futures" else "defensive"
    else:
        blockers.append(f"{family} has no actionable directional edge.")

    if signal.confidence < min_confidence:
        blockers.append(f"{family} confidence {signal.confidence:.3f} is below required {min_confidence:.3f}.")

    effective_family = routed_family or family

    if effective_family == "trend_continuation":
        min_adx = strategy.risk.min_adx if strategy else 20.0
        if adx_value < min_adx:
            blockers.append(f"{family} ADX {adx_value:.2f} is below required {min_adx:.2f}.")
    elif effective_family == "breakout":
        if not any("breakout" in reason or "release" in reason for reason in signal.reasons):
            blockers.append("breakout family lacks a fresh breakout or squeeze-release trigger.")
    elif effective_family == "trend_pullback":
        if adx_value < 16.0:
            blockers.append(f"{family} ADX {adx_value:.2f} is below required 16.00.")
        if not any(reason.startswith("fib-") for reason in signal.reasons):
            blockers.append("trend_pullback family lacks a Fibonacci pullback zone.")
        if signal.bias == "long" and _float(row, "bb_percent_b", 0.5) >= 0.86:
            blockers.append("trend_pullback long is too extended above the Bollinger range.")
        elif signal.bias == "short" and _float(row, "bb_percent_b", 0.5) <= 0.14:
            blockers.append("trend_pullback short is too extended below the Bollinger range.")
    elif effective_family == "liquidity_reclaim":
        if not any("sweep" in reason and ("reclaimed" in reason or "rejected" in reason) for reason in signal.reasons):
            blockers.append("liquidity_reclaim family lacks a completed sweep and reclaim/rejection.")
        taker_flow = abs(float((market_context or {}).get("taker_flow_imbalance") or 0.0))
        if _float(row, "volume_ratio_20", 1.0) < 1.05 and taker_flow < 0.05:
            blockers.append("liquidity_reclaim lacks volume or taker-flow confirmation.")
        if signal.bias == "long" and _float(row, "liquidity_close_position", 0.5) < 0.58:
            blockers.append("liquidity_reclaim long did not close high enough in the candle.")
        elif signal.bias == "short" and _float(row, "liquidity_close_position", 0.5) > 0.42:
            blockers.append("liquidity_reclaim short did not close low enough in the candle.")
    elif effective_family == "vwap_reclaim":
        has_reclaim = any(
            "vwap-" in reason and ("reclaimed" in reason or "rejected" in reason)
            for reason in signal.reasons
        )
        if not has_reclaim:
            blockers.append("vwap_reclaim lacks a rolling VWAP band or midline reclaim trigger.")
        if _float(row, "vwap_rolling_48", 0.0) <= 0.0:
            blockers.append("vwap_reclaim rolling VWAP is unavailable.")
        if _float(row, "volume_ratio_20", 1.0) < 1.02 and abs(_float(row, "obv_zscore_20")) < 0.5:
            blockers.append("vwap_reclaim lacks volume or OBV confirmation.")
        if signal.bias == "long" and _float(row, "bb_percent_b", 0.5) >= 0.84:
            blockers.append("vwap_reclaim long is too extended above the Bollinger range.")
        elif signal.bias == "short" and _float(row, "bb_percent_b", 0.5) <= 0.16:
            blockers.append("vwap_reclaim short is too extended below the Bollinger range.")
    elif effective_family == "mean_reversion":
        if adx_value > 24.0 and not any("squeeze-risk" in reason for reason in signal.reasons):
            blockers.append(f"mean_reversion ADX {adx_value:.2f} is too high for range reversion.")

    if action == "SELL" and market != "futures":
        blockers.append("Short setups are disabled outside the futures lane.")
    if blockers:
        action = "HOLD"

    signed = signal.confidence if signal.bias == "long" else -signal.confidence if signal.bias == "short" else 0.0
    score = int(round(50.0 + signed * 50.0))
    notes.append(
        f"{family} decision uses one independent family only; other indicators are filters, not extra votes."
    )
    notes.extend(signal.reasons)

    return {
        "score": max(0, min(100, score)),
        "bias": bias,
        "regime": family,
        "signal_counts": {
            "bullish": 1 if signal.bias == "long" else 0,
            "bearish": 1 if signal.bias == "short" else 0,
            "total": 1 if signal.bias != "neutral" else 0,
        },
        "convergence": signal.confidence,
        "recommended_action": action,
        "entry_ready": action in {"BUY", "SELL"} and not blockers,
        "entry_blockers": blockers,
        "decision_notes": notes,
        "notes": notes,
        "strategy_families": [item.to_dict() for item in family_map.values()],
        "selected_strategy_family": signal.to_dict(),
        "strategy_family": family,
        "requested_strategy_family": family,
        "routed_strategy_family": routed_family or family,
        "decision_trace": [
            {
                "layer": "strategy_family",
                "family": family,
                "allowed": action in {"BUY", "SELL"} and not blockers,
                "action": action,
                "reasons": blockers or list(signal.reasons),
            }
        ],
    }
