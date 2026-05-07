from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .alpha_families import (
    build_strategy_family_signals,
    select_best_family,
    strategy_family_trade_decision,
)
from .analysis import enrich_indicators, indicator_trade_plan_side, prepare_klines_frame, score_bias
from .asset_routing import RouteValidationSpec, resolve_symbol_route
from .binance_api import BinanceClient
from .config import REPORTS_DIR, Settings, ensure_runtime_dirs
from .convergence import (
    ConvergenceMetrics,
    calculate_expectancy_stats,
    calculate_loss_streak,
    calculate_profit_factor,
    evaluate_convergence,
)
from .exit_profiles import runner_stop_after_target, staged_take_profit_weights
from .feature_registry import build_feature_manifest
from .historical_klines import fetch_recent_klines
from .strategy import StrategyConfig
from .volume_structure import (
    summarize_htf_volume_imbalance,
    summarize_volume_bubbles,
    summarize_volume_profile,
)

PURE_STOP_REASONS = {"stop_loss", "stop_priority_same_bar"}
PARTIAL_TP_THEN_STOP_REASONS = {"partial_tp_then_stop"}


@dataclass(frozen=True, slots=True)
class BacktestTrade:
    side: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_r: float
    exit_reason: str
    analysis_score: int
    analysis_convergence: float
    strategy_family: str = ""
    routed_strategy_family: str = ""


EntryFilter = Callable[
    [pd.Series, pd.Series, dict[str, Any], int],
    bool | tuple[bool, str],
]

BacktestFrameCache = dict[tuple[Any, ...], pd.DataFrame]
BacktestMarketContextCache = dict[tuple[Any, ...], dict[int, dict[str, Any]]]


def _backtest_frame_cache_key(
    *,
    strategy: StrategyConfig,
    symbol: str,
    market: str,
    interval: str,
    limit: int,
) -> tuple[Any, ...]:
    return (
        symbol.upper(),
        market,
        interval,
        int(limit),
        int(strategy.signal.ema_fast),
        int(strategy.signal.ema_slow),
        int(strategy.signal.rsi_length),
        int(strategy.signal.macd_fast),
        int(strategy.signal.macd_slow),
        int(strategy.signal.macd_signal),
        int(strategy.signal.breakout_length),
    )


def _slippage_factor(side: str, slippage_bps: float, entering: bool) -> float:
    value = slippage_bps / 10_000.0
    if side == "BUY":
        return 1.0 + value if entering else 1.0 - value
    return 1.0 - value if entering else 1.0 + value


def _max_drawdown(equity_curve: list[float]) -> float:
    peak = 1.0
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak:
            max_dd = max(max_dd, (peak - value) / peak)
    return max_dd


def _metric_float(value: Any, *, infinity_value: float = 9999.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isinf(parsed):
        return infinity_value
    if math.isnan(parsed):
        return 0.0
    return parsed


def _json_metric(value: float) -> float | str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return 0.0
    return round(value, 4)


def _gross_exposure_fraction(entry_price: float, risk_distance: float, strategy: StrategyConfig) -> float:
    if entry_price <= 0 or risk_distance <= 0:
        return 0.0
    risk_pct = risk_distance / entry_price
    if risk_pct <= 0:
        return 0.0
    risk_capped_fraction = strategy.risk.max_account_risk_pct / risk_pct
    notional_capped_fraction = strategy.risk.max_notional_pct * strategy.risk.default_leverage
    return max(0.0, min(risk_capped_fraction, notional_capped_fraction))


def _backtest_market_context(df: pd.DataFrame, end_idx: int) -> dict[str, Any]:
    window = df.iloc[: end_idx + 1]
    if window.empty:
        return {}
    taker_context: dict[str, float] = {
        "taker_buy_sell_ratio": 1.0,
        "taker_flow_imbalance": 0.0,
    }
    if {"quote_asset_volume", "taker_buy_quote_volume"}.issubset(window.columns):
        buy_quote = float(pd.to_numeric(window["taker_buy_quote_volume"], errors="coerce").fillna(0.0).tail(20).sum())
        total_quote = float(pd.to_numeric(window["quote_asset_volume"], errors="coerce").fillna(0.0).tail(20).sum())
        sell_quote = max(total_quote - buy_quote, 0.0)
        total_taker = buy_quote + sell_quote
        taker_context = {
            "taker_buy_sell_ratio": round(buy_quote / sell_quote, 6) if sell_quote > 0 else 1.0,
            "taker_flow_imbalance": round((buy_quote - sell_quote) / total_taker, 6) if total_taker > 0 else 0.0,
        }
    return {
        **taker_context,
        "order_book_imbalance": 0.0,
        "spread_bps": 0.0,
        "funding_rate": 0.0,
        "open_interest_change_pct": 0.0,
        "multi_timeframe_structure": _backtest_multi_timeframe_structure(window),
        "volume_profile": summarize_volume_profile(window, rows=20, lookback=240),
        "volume_bubbles": summarize_volume_bubbles(window),
        "htf_volume_imbalance": summarize_htf_volume_imbalance(window),
    }


def _backtest_timeframe_structure(interval: str, window: pd.DataFrame) -> dict[str, Any]:
    if len(window) < 60:
        return {
            "interval": interval,
            "bias": "neutral",
            "score": 0.0,
            "confidence": 0.0,
            "alignment": "insufficient",
        }
    close = pd.to_numeric(window["close"], errors="coerce").dropna()
    if close.empty:
        return {
            "interval": interval,
            "bias": "neutral",
            "score": 0.0,
            "confidence": 0.0,
            "alignment": "unavailable",
        }
    ema7 = close.ewm(span=7, adjust=False).mean().iloc[-1]
    ema25 = close.ewm(span=25, adjust=False).mean().iloc[-1]
    ema89 = close.ewm(span=89, adjust=False).mean().iloc[-1]
    latest = float(close.iloc[-1])
    momentum = float(close.iloc[-1] - close.iloc[-5]) if len(close) >= 5 else 0.0
    bullish = 0
    bearish = 0
    if ema7 > ema25 > ema89:
        bullish += 2
    elif ema7 < ema25 < ema89:
        bearish += 2
    if latest > ema25:
        bullish += 1
    elif latest < ema25:
        bearish += 1
    if momentum > 0.0:
        bullish += 1
    elif momentum < 0.0:
        bearish += 1
    if bullish > bearish and bullish >= 3:
        bias = "long"
    elif bearish > bullish and bearish >= 3:
        bias = "short"
    else:
        bias = "neutral"
    total = bullish + bearish
    return {
        "interval": interval,
        "bias": bias,
        "score": round((bullish - bearish) / max(total, 1), 4),
        "confidence": round(max(bullish, bearish) / max(total, 1), 4) if total else 0.0,
        "alignment": "directional" if bias != "neutral" else "neutral",
    }


def _backtest_multi_timeframe_structure(window: pd.DataFrame) -> dict[str, Any]:
    structures = [
        _backtest_timeframe_structure("signal", window.tail(120)),
        _backtest_timeframe_structure("context", window.tail(240)),
    ]
    actionable = [item for item in structures if item["bias"] in {"long", "short"}]
    if not actionable:
        bias = "neutral"
        alignment = "neutral"
    else:
        long_weight = sum(float(item["confidence"]) for item in actionable if item["bias"] == "long")
        short_weight = sum(float(item["confidence"]) for item in actionable if item["bias"] == "short")
        if long_weight > short_weight:
            bias = "long"
        elif short_weight > long_weight:
            bias = "short"
        else:
            bias = "neutral"
        conflict = any(item["bias"] != bias for item in actionable) if bias != "neutral" else bool(actionable)
        alignment = "conflicted" if conflict else ("strong" if len(actionable) >= 2 else "mixed")
    total_conf = sum(float(item["confidence"]) for item in actionable)
    selected_conf = sum(float(item["confidence"]) for item in actionable if item["bias"] == bias)
    return {
        "bias": bias,
        "alignment": alignment,
        "confidence": round(selected_conf / total_conf, 4) if total_conf > 0 and bias != "neutral" else 0.0,
        "structures": structures,
    }


def _group_biases(analysis: dict[str, Any]) -> dict[str, str]:
    groups = analysis.get("signal_groups")
    if not isinstance(groups, list):
        return {}
    return {str(item.get("name")): str(item.get("bias")) for item in groups if isinstance(item, dict)}


def _entry_quality_filter(
    previous: pd.Series,
    analysis: dict[str, Any],
    score_model_analysis: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Cheap deterministic veto to reduce known fast-stop entry patterns."""

    action = str(analysis.get("recommended_action") or "")
    strategy_family = str(analysis.get("strategy_family") or "")
    routed_family = str(analysis.get("routed_strategy_family") or strategy_family)
    is_mean_reversion = routed_family == "mean_reversion"
    is_trend_pullback = routed_family == "trend_pullback"
    is_liquidity_reclaim = routed_family == "liquidity_reclaim"
    is_vwap_reclaim = routed_family == "vwap_reclaim"
    score_model = score_model_analysis or analysis
    score_model_action = str(score_model.get("recommended_action") or "")
    score_model_score = float(score_model.get("score") or 50.0)
    score_model_groups = _group_biases(score_model)
    trend_votes = int(float(previous.get("supertrend_direction", 0) or 0)) + int(
        float(previous.get("trend_magic_direction", 0) or 0)
    ) + int(float(previous.get("follow_line_direction", 0) or 0))
    jumbo_power = float(previous.get("jumbo_power", 0.0) or 0.0)
    jumbo_ma = float(previous.get("jumbo_power_ma", 0.0) or 0.0)
    mfi = float(previous.get("mfi_14", 50.0) or 50.0)
    bb_percent_b = float(previous.get("bb_percent_b", 0.5) or 0.5)
    adx_value = float(previous.get("adx", 0.0) or 0.0)
    family = select_best_family(build_strategy_family_signals(previous))
    tv_long_votes = sum(
        1
        for key in (
            "chandelier_direction",
            "ichimoku_direction",
            "psar_direction",
            "qqe_direction",
        )
        if float(previous.get(key, 0.0) or 0.0) > 0.0
    )
    tv_short_votes = sum(
        1
        for key in (
            "chandelier_direction",
            "ichimoku_direction",
            "psar_direction",
            "qqe_direction",
        )
        if float(previous.get(key, 0.0) or 0.0) < 0.0
    )
    squeeze_on = bool(previous.get("squeeze_on", False))
    squeeze_released = bool(previous.get("squeeze_released", False))
    squeeze_momentum = float(previous.get("squeeze_momentum", 0.0) or 0.0)
    fib_pullback_long = bool(previous.get("fib_pullback_long_zone", False))
    fib_pullback_short = bool(previous.get("fib_pullback_short_zone", False))
    fib_ote_long = bool(previous.get("fib_ote_long_zone", False))
    fib_ote_short = bool(previous.get("fib_ote_short_zone", False))
    liquidity_reclaim_long = bool(previous.get("liquidity_reclaim_long_20", False))
    liquidity_reclaim_short = bool(previous.get("liquidity_reclaim_short_20", False))
    liquidity_close_position = float(previous.get("liquidity_close_position", 0.5) or 0.5)
    vwap_reclaim_long = bool(previous.get("vwap_reclaim_long_48", False)) or bool(
        previous.get("vwap_mid_reclaim_long_48", False)
    )
    vwap_reclaim_short = bool(previous.get("vwap_reclaim_short_48", False)) or bool(
        previous.get("vwap_mid_reclaim_short_48", False)
    )
    vwap_distance_pct = float(previous.get("vwap_distance_pct_48", 0.0) or 0.0)
    volume_ratio = float(previous.get("volume_ratio_20", 1.0) or 1.0)

    if action == "BUY":
        if routed_family in {"trend_continuation", "breakout", "trend_pullback"} and score_model_action != "BUY":
            return False, "score-model-not-long"
        if routed_family == "breakout" and score_model_groups.get("volume_flow") != "long":
            return False, "breakout-volume-flow-not-long"
        if (
            routed_family == "trend_continuation"
            and "momentum_power" in score_model_groups
            and score_model_groups.get("momentum_power") not in {"long", "neutral"}
        ):
            return False, "trend-momentum-opposes-long"
        if family.bias == "short" and family.confidence >= 0.67:
            return False, f"alpha-family-opposes-long:{family.family}"
        if routed_family == "trend_continuation" and squeeze_on and not squeeze_released:
            return False, "squeeze-not-released-for-long"
        if routed_family == "breakout" and squeeze_momentum <= 0.0:
            return False, "squeeze-momentum-not-long"
        if routed_family == "trend_pullback" and not (fib_pullback_long or fib_ote_long):
            return False, "fib-pullback-zone-not-long"
        if routed_family == "liquidity_reclaim" and not liquidity_reclaim_long:
            return False, "liquidity-reclaim-not-long"
        if routed_family == "liquidity_reclaim" and liquidity_close_position < 0.58:
            return False, "liquidity-reclaim-close-not-high"
        if routed_family == "liquidity_reclaim" and volume_ratio < 1.05:
            return False, "liquidity-reclaim-volume-too-low"
        if routed_family == "vwap_reclaim" and not vwap_reclaim_long:
            return False, "vwap-reclaim-not-long"
        if routed_family == "vwap_reclaim" and vwap_distance_pct > 0.85:
            return False, "vwap-reclaim-long-too-far-above-value"
        obv_strength = abs(float(previous.get("obv_zscore_20", 0.0) or 0.0))
        if routed_family == "vwap_reclaim" and volume_ratio < 1.02 and obv_strength < 0.5:
            return False, "vwap-reclaim-volume-too-low"
        if routed_family == "trend_pullback" and bb_percent_b >= 0.86:
            return False, "pullback-long-too-extended"
        if is_mean_reversion and adx_value >= 24.0 and trend_votes <= -2:
            return False, "mean-reversion-trend-stack-opposes-long"
        if is_trend_pullback and trend_votes <= -1:
            return False, "pullback-trend-filter-opposes-long"
        if is_liquidity_reclaim and adx_value >= 32.0 and trend_votes <= -2:
            return False, "liquidity-reclaim-strong-trend-opposes-long"
        if is_vwap_reclaim and adx_value >= 30.0 and trend_votes <= -2:
            return False, "vwap-reclaim-strong-trend-opposes-long"
        if (
            not is_mean_reversion
            and not is_trend_pullback
            and not is_liquidity_reclaim
            and not is_vwap_reclaim
            and trend_votes <= 0
        ):
            return False, "trend-filter-not-long"
        if tv_short_votes >= 3 and tv_short_votes > tv_long_votes and (not is_mean_reversion or adx_value >= 24.0):
            return False, "tradingview-stack-opposes-long"
        if is_trend_pullback and jumbo_power < jumbo_ma - 18.0 and jumbo_power < -20.0:
            return False, "pullback-jumbo-power-opposes-long"
        if (
            not is_mean_reversion
            and not is_trend_pullback
            and not is_liquidity_reclaim
            and not is_vwap_reclaim
            and jumbo_power < jumbo_ma
            and jumbo_power < 20.0
        ):
            return False, "jumbo-power-not-long"
        if bb_percent_b >= 1.08 and mfi >= 80.0:
            return False, "long-chase-extension"
    elif action == "SELL":
        if routed_family in {"trend_continuation", "breakout", "trend_pullback"} and score_model_action != "SELL":
            return False, "score-model-not-short"
        if routed_family == "breakout" and score_model_groups.get("volume_flow") != "short":
            return False, "breakout-volume-flow-not-short"
        if (
            routed_family == "trend_continuation"
            and "momentum_power" in score_model_groups
            and score_model_groups.get("momentum_power") not in {"short", "neutral"}
        ):
            return False, "trend-momentum-opposes-short"
        if family.bias == "long" and family.confidence >= 0.67:
            return False, f"alpha-family-opposes-short:{family.family}"
        if routed_family == "trend_continuation" and squeeze_on and not squeeze_released:
            return False, "squeeze-not-released-for-short"
        if routed_family == "breakout" and squeeze_momentum >= 0.0:
            return False, "squeeze-momentum-not-short"
        if routed_family == "trend_pullback" and not (fib_pullback_short or fib_ote_short):
            return False, "fib-pullback-zone-not-short"
        if routed_family == "liquidity_reclaim" and not liquidity_reclaim_short:
            return False, "liquidity-reclaim-not-short"
        if routed_family == "liquidity_reclaim" and liquidity_close_position > 0.42:
            return False, "liquidity-reclaim-close-not-low"
        if routed_family == "liquidity_reclaim" and volume_ratio < 1.05:
            return False, "liquidity-reclaim-volume-too-low"
        if routed_family == "vwap_reclaim" and not vwap_reclaim_short:
            return False, "vwap-reclaim-not-short"
        if routed_family == "vwap_reclaim" and vwap_distance_pct < -0.85:
            return False, "vwap-reclaim-short-too-far-below-value"
        obv_strength = abs(float(previous.get("obv_zscore_20", 0.0) or 0.0))
        if routed_family == "vwap_reclaim" and volume_ratio < 1.02 and obv_strength < 0.5:
            return False, "vwap-reclaim-volume-too-low"
        if routed_family == "trend_pullback" and bb_percent_b <= 0.14:
            return False, "pullback-short-too-extended"
        if is_mean_reversion and adx_value >= 24.0 and trend_votes >= 2:
            return False, "mean-reversion-trend-stack-opposes-short"
        if is_trend_pullback and trend_votes >= 1:
            return False, "pullback-trend-filter-opposes-short"
        if is_liquidity_reclaim and adx_value >= 32.0 and trend_votes >= 2:
            return False, "liquidity-reclaim-strong-trend-opposes-short"
        if is_vwap_reclaim and adx_value >= 30.0 and trend_votes >= 2:
            return False, "vwap-reclaim-strong-trend-opposes-short"
        if (
            not is_mean_reversion
            and not is_trend_pullback
            and not is_liquidity_reclaim
            and not is_vwap_reclaim
            and trend_votes >= 0
        ):
            return False, "trend-filter-not-short"
        if tv_long_votes >= 3 and tv_long_votes > tv_short_votes and (not is_mean_reversion or adx_value >= 24.0):
            return False, "tradingview-stack-opposes-short"
        if is_trend_pullback and jumbo_power > jumbo_ma + 18.0 and jumbo_power > 20.0:
            return False, "pullback-jumbo-power-opposes-short"
        if (
            not is_mean_reversion
            and not is_trend_pullback
            and not is_liquidity_reclaim
            and not is_vwap_reclaim
            and jumbo_power > jumbo_ma
            and jumbo_power > -20.0
        ):
            return False, "jumbo-power-not-short"
        if bb_percent_b <= -0.08 and mfi <= 20.0:
            return False, "short-chase-extension"
    if (
        not is_mean_reversion
        and not is_liquidity_reclaim
        and not is_vwap_reclaim
        and adx_value < 15.0
    ):
        return False, "trend-strength-too-low"
    if is_mean_reversion and score_model_score >= 75.0 and action == "SELL":
        return False, "mean-reversion-score-model-strong-long"
    if is_mean_reversion and score_model_score <= 25.0 and action == "BUY":
        return False, "mean-reversion-score-model-strong-short"
    return True, ""


def _symbol_family_entry_filter(
    *,
    symbol: str,
    interval: str,
    strategy_family: str,
    previous: pd.Series,
    analysis: dict[str, Any],
) -> tuple[bool, str]:
    """Symbol-role vetoes that keep core symbols from sharing one loose gate."""

    side = str(analysis.get("recommended_action") or "")
    symbol = symbol.upper()
    interval = str(interval)
    if not symbol or not interval:
        return True, ""
    adx_value = float(previous.get("adx", 0.0) or 0.0)
    bb_percent_b = float(previous.get("bb_percent_b", 0.5) or 0.5)
    bb_bandwidth = float(previous.get("bb_bandwidth", 0.0) or 0.0)
    rsi_value = float(previous.get("rsi_14", 50.0) or 50.0)
    stoch_k = float(previous.get("stoch_rsi_k", 50.0) or 50.0)
    volume_ratio = float(previous.get("volume_ratio_20", 1.0) or 1.0)
    vwap_distance = float(previous.get("vwap_distance_pct_48", 0.0) or 0.0)

    if strategy_family == "vwap_reclaim":
        if symbol in {"XAUTUSDT", "PAXGUSDT"} and interval not in {"4h", "1d"}:
            return False, "gold-vwap-lane-requires-higher-timeframe"
        if symbol in {"BTCUSDT", "ETHUSDT"} and interval == "1h" and abs(vwap_distance) > 0.55:
            return False, "core-vwap-reclaim-too-far-from-value"
        alt_symbols = {"SOLUSDT", "BNBUSDT", "XRPUSDT", "LINKUSDT", "AAVEUSDT", "TRXUSDT"}
        if symbol in alt_symbols and volume_ratio < 1.03:
            return False, "alt-vwap-reclaim-needs-volume"
        if symbol in {"LINKUSDT", "AAVEUSDT"} and adx_value >= 28.0:
            return False, "defi-vwap-reclaim-avoids-strong-trend"
        if side == "BUY" and bb_percent_b > 0.82:
            return False, "symbol-vwap-long-too-extended"
        if side == "SELL" and bb_percent_b < 0.18:
            return False, "symbol-vwap-short-too-extended"

    if strategy_family == "mean_reversion":
        if symbol in {"BTCUSDT", "ETHUSDT"}:
            return False, "core-mean-reversion-disabled"
        if symbol == "XRPUSDT" and interval == "4h" and side == "BUY" and bb_bandwidth < 0.06:
            return False, "xrp-mean-reversion-long-bandwidth-too-low"
        if symbol == "TRXUSDT" and interval == "4h" and side == "SELL" and stoch_k > 96.0:
            return False, "trx-mean-reversion-short-stoch-too-extended"
        if adx_value >= 22.0 and symbol not in {"XAUTUSDT", "PAXGUSDT"}:
            return False, "symbol-mean-reversion-adx-too-high"
        if side == "BUY" and (rsi_value > 42.0 or bb_percent_b > 0.32):
            return False, "symbol-mean-reversion-long-not-discounted"
        if side == "SELL" and (rsi_value < 58.0 or bb_percent_b < 0.68):
            return False, "symbol-mean-reversion-short-not-premium"

    if strategy_family == "breakout" and symbol != "SOLUSDT" and volume_ratio < 1.35:
        return False, "non-sol-breakout-volume-too-low"

    return True, ""


def _staged_take_profit_weights(
    parts: int,
    strategy: StrategyConfig,
    confidence: float,
    *,
    strategy_family: str = "",
) -> list[float]:
    return staged_take_profit_weights(
        parts,
        exit_profile=strategy.risk.exit_profile,
        trailing_stop_enabled=bool(strategy.risk.trailing_stop_enabled),
        confidence=confidence,
        strategy_family=strategy_family,
    )


def _runner_stop_after_target(
    *,
    side: str,
    current_stop: float,
    entry_price: float,
    close_price: float,
    strategy: StrategyConfig,
    hit_count: int,
    initial_risk_distance: float | None = None,
) -> float:
    return runner_stop_after_target(
        side=side,
        current_stop=current_stop,
        entry_price=entry_price,
        close_price=close_price,
        trailing_callback_pct=strategy.risk.trailing_callback_pct,
        hit_count=hit_count,
        exit_profile=strategy.risk.exit_profile,
        initial_risk_distance=initial_risk_distance,
    )


def _position_remaining_fraction(position: dict[str, Any]) -> float:
    closed = sum(float(item.get("weight", 0.0)) for item in position.get("closed_targets", []))
    return max(0.0, 1.0 - closed)


def _position_exit_weights(position: dict[str, Any], exit_price: float) -> list[dict[str, float]]:
    remaining_weight = _position_remaining_fraction(position)
    exits = [
        {"price": float(target["price"]), "weight": float(target["weight"])}
        for target in position.get("closed_targets", [])
    ]
    if remaining_weight > 1e-9:
        exits.append({"price": float(exit_price), "weight": remaining_weight})
    return exits


def _normalized_exit_reason(exit_reason: str, closed_targets: list[dict[str, Any]]) -> str:
    if not closed_targets:
        return exit_reason
    if exit_reason == "take_profit":
        return "staged_take_profit"
    if exit_reason in {"stop_loss", "stop_priority_same_bar"}:
        return "partial_tp_then_stop"
    if exit_reason == "end_of_data":
        return "partial_tp_then_end"
    return exit_reason


def _compound_total_return_pct(trades: list[dict[str, Any]]) -> float:
    equity = 1.0
    for trade in trades:
        equity *= 1.0 + (_metric_float(trade.get("pnl_pct")) / 100.0)
    return round((equity - 1.0) * 100.0, 4)


def _exit_reason_ratio(trades: list[dict[str, Any]], reasons: set[str]) -> float:
    if not trades:
        return 0.0
    return round(
        (sum(1 for trade in trades if str(trade.get("exit_reason") or "") in reasons) / len(trades))
        * 100.0,
        2,
    )


def _trade_fold_metrics(trades: list[dict[str, Any]], *, fold: int, start_index: int, end_index: int) -> dict[str, Any]:
    pnls = [_metric_float(trade.get("pnl_pct")) for trade in trades]
    r_values = [_metric_float(trade.get("pnl_r")) for trade in trades]
    wins = sum(1 for pnl in pnls if pnl > 0.0)
    losses = sum(1 for pnl in pnls if pnl <= 0.0)
    equity = [1.0]
    running = 1.0
    for pnl in pnls:
        running *= 1.0 + (pnl / 100.0)
        equity.append(running)
    profit_factor = calculate_profit_factor(pnls)
    return {
        "fold": fold,
        "start_index": start_index,
        "end_index": end_index,
        "start_time": str(trades[0].get("entry_time") or "") if trades else "",
        "end_time": str(trades[-1].get("exit_time") or "") if trades else "",
        "trade_count": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round((wins / len(trades)) * 100.0, 2) if trades else 0.0,
        "avg_pnl_pct": round(sum(pnls) / len(pnls), 4) if pnls else 0.0,
        "avg_r": round(sum(r_values) / len(r_values), 4) if r_values else 0.0,
        "total_return_pct": _compound_total_return_pct(trades),
        "max_drawdown_pct": round(_max_drawdown(equity) * 100.0, 4),
        "profit_factor": _json_metric(profit_factor),
        "loss_streak": calculate_loss_streak(pnls),
        "stop_loss_ratio": _exit_reason_ratio(trades, PURE_STOP_REASONS),
        "partial_tp_then_stop_ratio": _exit_reason_ratio(trades, PARTIAL_TP_THEN_STOP_REASONS),
    }


def _chronological_trade_folds(
    trades: list[dict[str, Any]],
    *,
    target_folds: int,
    min_trades_per_fold: int,
) -> list[dict[str, Any]]:
    if not trades:
        return []
    min_trades = max(1, int(min_trades_per_fold))
    max_possible_folds = len(trades) // min_trades
    fold_count = min(max(1, int(target_folds)), max_possible_folds)
    if fold_count <= 1:
        return []
    fold_size = math.ceil(len(trades) / fold_count)
    folds: list[dict[str, Any]] = []
    for fold_index in range(fold_count):
        start = fold_index * fold_size
        end = min(len(trades), (fold_index + 1) * fold_size)
        fold_trades = trades[start:end]
        if len(fold_trades) < min_trades:
            continue
        folds.append(
            _trade_fold_metrics(
                fold_trades,
                fold=fold_index + 1,
                start_index=start,
                end_index=end,
            )
        )
    return folds


def audit_backtest_robustness(
    summary: dict[str, Any],
    validation: RouteValidationSpec,
    *,
    folds: int = 4,
    min_trades_per_fold: int = 3,
) -> dict[str, Any]:
    """Validate that a backtest edge is distributed across time, not one lucky slice."""

    trades = list(summary.get("trades") or [])
    fold_rows = _chronological_trade_folds(
        trades,
        target_folds=folds,
        min_trades_per_fold=min_trades_per_fold,
    )
    target_profit_factor = max(1.0, float(validation.screening_min_profit_factor))
    min_fold_profit_factor = min(1.0, target_profit_factor)
    required_positive_folds = math.ceil(max(len(fold_rows), 1) * 0.67)
    positive_folds = sum(1 for item in fold_rows if _metric_float(item.get("total_return_pct")) > 0.0)
    folds_below_one_pf = sum(1 for item in fold_rows if _metric_float(item.get("profit_factor")) < min_fold_profit_factor)
    fold_profit_factors = [_metric_float(item.get("profit_factor")) for item in fold_rows]
    fold_drawdowns = [_metric_float(item.get("max_drawdown_pct")) for item in fold_rows]

    reasons: list[str] = []
    if len(trades) < int(validation.screening_min_trades):
        reasons.append(
            f"overall-trade-count-below-screening-floor:{len(trades)}/{validation.screening_min_trades}"
        )
    if len(fold_rows) < 2:
        reasons.append("insufficient-chronological-folds-for-robustness-audit")
    if _metric_float(summary.get("profit_factor")) < target_profit_factor:
        reasons.append(
            "overall-profit-factor-below-robustness-floor:"
            f"{_metric_float(summary.get('profit_factor')):.4f}/{target_profit_factor:.4f}"
        )
    if _metric_float(summary.get("expectancy_r")) < float(validation.screening_min_expectancy_r):
        reasons.append(
            "overall-expectancy-r-below-screening-floor:"
            f"{_metric_float(summary.get('expectancy_r')):.4f}/{validation.screening_min_expectancy_r:.4f}"
        )
    if _metric_float(summary.get("payoff_ratio")) < float(validation.screening_min_payoff_ratio):
        reasons.append(
            "overall-payoff-ratio-below-screening-floor:"
            f"{_metric_float(summary.get('payoff_ratio')):.4f}/{validation.screening_min_payoff_ratio:.4f}"
        )
    if _metric_float(summary.get("max_drawdown_pct")) > float(validation.max_drawdown_pct):
        reasons.append(
            "overall-drawdown-above-route-limit:"
            f"{_metric_float(summary.get('max_drawdown_pct')):.4f}/{validation.max_drawdown_pct:.4f}"
        )
    if int(summary.get("loss_streak") or 0) > int(validation.max_loss_streak):
        reasons.append(
            f"overall-loss-streak-above-route-limit:{summary.get('loss_streak')}/{validation.max_loss_streak}"
        )
    if fold_rows and positive_folds < required_positive_folds:
        reasons.append(f"positive-fold-count-too-low:{positive_folds}/{required_positive_folds}")
    allowed_bad_folds = 1 if len(fold_rows) >= 4 else 0
    if fold_rows and folds_below_one_pf > allowed_bad_folds:
        reasons.append(f"too-many-folds-below-1pf:{folds_below_one_pf}/{len(fold_rows)}")
    if fold_drawdowns and max(fold_drawdowns) > float(validation.max_drawdown_pct):
        reasons.append(
            "fold-drawdown-above-route-limit:"
            f"{max(fold_drawdowns):.4f}/{validation.max_drawdown_pct:.4f}"
        )

    status = "passed" if not reasons else "failed"
    if any(reason.startswith("insufficient") or reason.startswith("overall-trade-count") for reason in reasons):
        status = "insufficient_sample"

    return {
        "status": status,
        "passed": not reasons,
        "fold_count": len(fold_rows),
        "target_folds": int(folds),
        "min_trades_per_fold": int(min_trades_per_fold),
        "target_profit_factor": round(target_profit_factor, 4),
        "target_expectancy_r": round(float(validation.screening_min_expectancy_r), 4),
        "target_payoff_ratio": round(float(validation.screening_min_payoff_ratio), 4),
        "min_fold_profit_factor": round(min_fold_profit_factor, 4),
        "positive_fold_count": positive_folds,
        "required_positive_fold_count": required_positive_folds if fold_rows else 0,
        "folds_below_one_pf": folds_below_one_pf,
        "mean_fold_profit_factor": round(sum(fold_profit_factors) / len(fold_profit_factors), 4) if fold_profit_factors else 0.0,
        "min_fold_profit_factor_observed": round(min(fold_profit_factors), 4) if fold_profit_factors else 0.0,
        "max_fold_drawdown_pct": round(max(fold_drawdowns), 4) if fold_drawdowns else 0.0,
        "reasons": reasons,
        "folds": fold_rows,
        "applied_principles": [
            "chronological-fold-validation",
            "profit-factor-must-persist-across-regimes",
            "drawdown-and-loss-streak-gate-before-promotion",
            "reject-single-window-backtest-overfitting",
        ],
    }


def simulate_backtest(
    df: pd.DataFrame,
    market: str,
    strategy: StrategyConfig,
    *,
    symbol: str = "",
    interval: str = "",
    entry_filter: EntryFilter | None = None,
    strategy_family: str | None = None,
    market_context_cache: dict[int, dict[str, Any]] | None = None,
    require_score_model_confirmation: bool = True,
    lightweight_market_context: bool = False,
) -> dict[str, Any]:
    warmup = max(220, strategy.signal.breakout_length + 5)
    fee_rate = strategy.execution.fee_bps / 10_000.0
    slippage_bps = strategy.execution.slippage_bps
    trades: list[BacktestTrade] = []
    equity_curve = [1.0]
    equity = 1.0
    position: dict[str, Any] | None = None
    entry_veto_count = 0
    entry_veto_reasons: dict[str, int] = {}

    for idx in range(warmup + 1, len(df)):
        current = df.iloc[idx]
        previous = df.iloc[idx - 1]

        if position is not None:
            high = float(current["high"])
            low = float(current["low"])
            close = float(current["close"])
            exit_price: float | None = None
            exit_reason = ""
            stop_price = float(position["stop_price"])
            closed_targets = position["closed_targets"]
            target_hits: list[dict[str, Any]] = []
            bars_held = idx - int(position["entry_idx"])
            if position["side"] == "BUY":
                stop_hit = low <= stop_price
                for target in position["targets"]:
                    if target["index"] not in position["hit_target_indices"] and high >= target["price"]:
                        target_hits.append(target)
                if stop_hit and target_hits:
                    exit_price = stop_price
                    exit_reason = "stop_priority_same_bar"
                elif stop_hit:
                    exit_price = stop_price
                    exit_reason = "stop_loss"
                elif target_hits:
                    for target in target_hits:
                        closed_targets.append(target)
                        position["hit_target_indices"].add(target["index"])
                    last_target = target_hits[-1]
                    if strategy.risk.trailing_stop_enabled:
                        position["stop_price"] = _runner_stop_after_target(
                            side="BUY",
                            current_stop=float(position["stop_price"]),
                            entry_price=float(position["entry_price"]),
                            close_price=close,
                            strategy=strategy,
                            hit_count=len(position["hit_target_indices"]),
                            initial_risk_distance=float(position["risk_distance"]),
                        )
                    if _position_remaining_fraction(position) <= 1e-9:
                        exit_price = last_target["price"]
                        exit_reason = "take_profit"
                    elif not strategy.risk.trailing_stop_enabled:
                        exit_price = last_target["price"]
                        exit_reason = "partial_take_profit_exit"
            else:
                stop_hit = high >= stop_price
                for target in position["targets"]:
                    if target["index"] not in position["hit_target_indices"] and low <= target["price"]:
                        target_hits.append(target)
                if stop_hit and target_hits:
                    exit_price = stop_price
                    exit_reason = "stop_priority_same_bar"
                elif stop_hit:
                    exit_price = stop_price
                    exit_reason = "stop_loss"
                elif target_hits:
                    for target in target_hits:
                        closed_targets.append(target)
                        position["hit_target_indices"].add(target["index"])
                    last_target = target_hits[-1]
                    if strategy.risk.trailing_stop_enabled:
                        position["stop_price"] = _runner_stop_after_target(
                            side="SELL",
                            current_stop=float(position["stop_price"]),
                            entry_price=float(position["entry_price"]),
                            close_price=close,
                            strategy=strategy,
                            hit_count=len(position["hit_target_indices"]),
                            initial_risk_distance=float(position["risk_distance"]),
                        )
                    if _position_remaining_fraction(position) <= 1e-9:
                        exit_price = last_target["price"]
                        exit_reason = "take_profit"
                    elif not strategy.risk.trailing_stop_enabled:
                        exit_price = last_target["price"]
                        exit_reason = "partial_take_profit_exit"

            if (
                exit_price is None
                and strategy.risk.time_limit_bars > 0
                and bars_held >= strategy.risk.time_limit_bars
            ):
                exit_price = close * _slippage_factor(position["side"], slippage_bps, entering=False)
                exit_reason = "time_limit"

            if exit_price is None and idx == len(df) - 1:
                exit_price = float(current["close"]) * _slippage_factor(position["side"], slippage_bps, entering=False)
                exit_reason = "end_of_data"

            if exit_price is not None:
                all_exits = _position_exit_weights(position, exit_price)
                scaled_gross_return = 0.0
                for realized_exit in all_exits:
                    gross_return = (
                        (float(realized_exit["price"]) - position["entry_price"]) / position["entry_price"]
                    )
                    if position["side"] == "SELL":
                        gross_return *= -1.0
                    scaled_gross_return += (
                        gross_return * position["gross_exposure_fraction"] * float(realized_exit["weight"])
                    )
                fee_drag = (2 * fee_rate) * position["gross_exposure_fraction"]
                net_return = scaled_gross_return - fee_drag
                equity *= 1.0 + net_return
                equity_curve.append(equity)
                risk_pct = strategy.risk.max_account_risk_pct
                pnl_r = net_return / risk_pct if risk_pct else 0.0
                trades.append(
                    BacktestTrade(
                        side=position["side"],
                        entry_time=str(position["entry_time"]),
                        exit_time=str(df.index[idx]),
                        entry_price=round(position["entry_price"], 8),
                        exit_price=round(exit_price, 8),
                        pnl_pct=round(net_return * 100.0, 4),
                        pnl_r=round(pnl_r, 4),
                        exit_reason=_normalized_exit_reason(exit_reason, closed_targets),
                        analysis_score=position["analysis_score"],
                        analysis_convergence=position["analysis_convergence"],
                        strategy_family=str(position.get("strategy_family") or ""),
                        routed_strategy_family=str(position.get("routed_strategy_family") or ""),
                    )
                )
                position = None
                continue
            continue

        context_idx = idx - 1
        if lightweight_market_context:
            market_context = {}
        elif market_context_cache is not None and context_idx in market_context_cache:
            market_context = market_context_cache[context_idx]
        else:
            market_context = _backtest_market_context(df, context_idx)
            if market_context_cache is not None:
                market_context_cache[context_idx] = market_context
        if strategy_family:
            analysis = strategy_family_trade_decision(
                previous,
                market=market,
                family=strategy_family,
                market_context=market_context,
                strategy=strategy,
            )
            score_model_analysis = (
                score_bias(previous, market, None, market_context=market_context, strategy=strategy)
                if require_score_model_confirmation
                else analysis
            )
        else:
            analysis = score_bias(previous, market, None, market_context=market_context, strategy=strategy)
            score_model_analysis = analysis
        if not bool(analysis.get("entry_ready")):
            continue
        quality_allowed, quality_reason = _entry_quality_filter(previous, analysis, score_model_analysis)
        if quality_allowed and strategy_family:
            quality_allowed, quality_reason = _symbol_family_entry_filter(
                symbol=symbol,
                interval=interval,
                strategy_family=str(analysis.get("routed_strategy_family") or strategy_family),
                previous=previous,
                analysis=analysis,
            )
        if not quality_allowed:
            entry_veto_count += 1
            entry_veto_reasons[quality_reason] = entry_veto_reasons.get(quality_reason, 0) + 1
            continue
        if entry_filter is not None:
            filter_result = entry_filter(previous, current, analysis, idx)
            if isinstance(filter_result, tuple):
                entry_allowed, veto_reason = filter_result
            else:
                entry_allowed = bool(filter_result)
                veto_reason = "entry-filter-veto"
            if not entry_allowed:
                reason = str(veto_reason or "entry-filter-veto")
                entry_veto_count += 1
                entry_veto_reasons[reason] = entry_veto_reasons.get(reason, 0) + 1
                continue
        side = str(analysis["recommended_action"])

        entry_price = float(current["open"]) * _slippage_factor(side, slippage_bps, entering=True)
        selected_family = analysis.get("selected_strategy_family") or {}
        side_plan = indicator_trade_plan_side(
            previous,
            side=side,
            strategy=strategy,
            family=str(analysis.get("strategy_family") or selected_family.get("family") or ""),
        )
        stop_price = float(side_plan["invalidation"])
        if side == "BUY":
            risk_distance = entry_price - stop_price
        else:
            risk_distance = stop_price - entry_price
        risk_pct = risk_distance / entry_price if entry_price else 0.0
        gross_exposure_fraction = _gross_exposure_fraction(entry_price, risk_distance, strategy)
        if gross_exposure_fraction <= 0:
            continue
        confidence = float(analysis["convergence"])
        tp_multiples = list(strategy.risk.take_profit_r_multiples or (strategy.primary_tp_multiple,))
        family_name = str(
            analysis.get("routed_strategy_family")
            or analysis.get("strategy_family")
            or selected_family.get("family")
            or ""
        )
        tp_weights = _staged_take_profit_weights(
            len(tp_multiples),
            strategy,
            confidence,
            strategy_family=family_name,
        )
        if side == "BUY":
            targets = [
                {
                    "index": target_idx,
                    "price": entry_price + (risk_distance * multiple),
                    "weight": tp_weights[target_idx],
                }
                for target_idx, multiple in enumerate(tp_multiples)
            ]
        else:
            targets = [
                {
                    "index": target_idx,
                    "price": entry_price - (risk_distance * multiple),
                    "weight": tp_weights[target_idx],
                }
                for target_idx, multiple in enumerate(tp_multiples)
            ]
        position = {
            "side": side,
            "entry_time": df.index[idx],
            "entry_idx": idx,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "targets": targets,
            "closed_targets": [],
            "hit_target_indices": set(),
            "risk_pct": risk_pct,
            "risk_distance": risk_distance,
            "gross_exposure_fraction": gross_exposure_fraction,
            "analysis_score": int(analysis["score"]),
            "analysis_convergence": float(analysis["convergence"]),
            "strategy_family": str(analysis.get("strategy_family") or ""),
            "routed_strategy_family": family_name,
        }

    wins = sum(1 for trade in trades if trade.pnl_pct > 0)
    losses = sum(1 for trade in trades if trade.pnl_pct <= 0)
    avg_pnl_pct = sum(trade.pnl_pct for trade in trades) / len(trades) if trades else 0.0
    avg_r = sum(trade.pnl_r for trade in trades) / len(trades) if trades else 0.0
    profit_factor = calculate_profit_factor([trade.pnl_pct for trade in trades])
    expectancy_stats = calculate_expectancy_stats([trade.pnl_r for trade in trades])
    trade_dicts = [asdict(trade) for trade in trades]
    return {
        "trade_count": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round((wins / len(trades)) * 100.0, 2) if trades else 0.0,
        "avg_pnl_pct": round(avg_pnl_pct, 4),
        "avg_r": round(avg_r, 4),
        **expectancy_stats,
        "ending_equity": round(equity, 6),
        "total_return_pct": round((equity - 1.0) * 100.0, 4),
        "max_drawdown_pct": round(_max_drawdown(equity_curve) * 100.0, 4),
        "profit_factor": round(profit_factor, 4),
        "loss_streak": calculate_loss_streak([trade.pnl_pct for trade in trades]),
        "stop_loss_ratio": _exit_reason_ratio(trade_dicts, PURE_STOP_REASONS),
        "partial_tp_then_stop_ratio": _exit_reason_ratio(trade_dicts, PARTIAL_TP_THEN_STOP_REASONS),
        "entry_veto_count": entry_veto_count,
        "entry_veto_reasons": entry_veto_reasons,
        "fee_bps": strategy.execution.fee_bps,
        "slippage_bps": strategy.execution.slippage_bps,
        "trades": trade_dicts,
    }


def write_backtest_markdown(payload: dict[str, Any], path: Path) -> None:
    summary = payload["summary"]
    robustness = payload.get("robustness") or {}
    lines = [
        f"# {payload['symbol']} {payload['interval']} backtest",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Strategy profile: `{payload['strategy_profile']}`",
        f"- Trades: `{summary['trade_count']}`",
        f"- Win rate: `{summary['win_rate']}`%",
        f"- Avg PnL per trade: `{summary['avg_pnl_pct']}`%",
        f"- Avg R: `{summary['avg_r']}`",
        f"- Total return: `{summary['total_return_pct']}`%",
        f"- Max drawdown: `{summary['max_drawdown_pct']}`%",
        f"- Profit factor: `{summary['profit_factor']}`",
        f"- Fee / slippage assumptions: `{summary['fee_bps']}` / `{summary['slippage_bps']}` bps",
        f"- Robustness gate: `{robustness.get('status', 'unknown')}`",
        f"- Robustness folds: `{robustness.get('positive_fold_count', 0)}` / `{robustness.get('fold_count', 0)}` positive",
        f"- Robustness reasons: `{', '.join(robustness.get('reasons') or []) or 'none'}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_backtest(
    settings: Settings,
    *,
    strategy: StrategyConfig,
    symbol: str,
    market: str,
    interval: str,
    limit: int,
    output_dir: Path | None = None,
    strategy_family: str | None = None,
    frame_cache: BacktestFrameCache | None = None,
    market_context_cache: BacktestMarketContextCache | None = None,
    entry_filter: EntryFilter | None = None,
    research_entry_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = output_dir or (REPORTS_DIR / f"{run_id}-{symbol.lower()}-{market}-{interval}-backtest")
    root.mkdir(parents=True, exist_ok=True)
    cache_key = _backtest_frame_cache_key(
        strategy=strategy,
        symbol=symbol,
        market=market,
        interval=interval,
        limit=limit,
    )
    if frame_cache is not None and cache_key in frame_cache:
        df = frame_cache[cache_key].copy(deep=False)
    else:
        with BinanceClient(settings) as client:
            raw_klines = fetch_recent_klines(client, symbol, interval, limit, market)
        df = enrich_indicators(prepare_klines_frame(raw_klines), interval, strategy=strategy)
        if frame_cache is not None:
            frame_cache[cache_key] = df
    context_cache_for_frame = None
    if market_context_cache is not None:
        context_cache_for_frame = market_context_cache.setdefault(cache_key, {})
    summary = simulate_backtest(
        df,
        market,
        strategy,
        symbol=symbol,
        interval=interval,
        entry_filter=entry_filter,
        strategy_family=strategy_family,
        market_context_cache=context_cache_for_frame,
    )
    route = resolve_symbol_route(symbol.upper())
    robustness = audit_backtest_robustness(summary, route.validation)
    convergence = evaluate_convergence(
        ConvergenceMetrics(
            trade_count=int(summary["trade_count"]),
            win_rate=float(summary["win_rate"]),
            profit_factor=float(summary["profit_factor"]),
            max_drawdown_pct=float(summary["max_drawdown_pct"]),
            loss_streak=int(summary["loss_streak"]),
            expectancy_r=float(summary.get("expectancy_r") or 0.0),
            payoff_ratio=float(summary.get("payoff_ratio") or 0.0),
        ),
        route.validation,
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "symbol": symbol.upper(),
        "market": market,
        "interval": interval,
        "limit": limit,
        "strategy_profile": strategy.profile,
        "strategy_family": strategy_family,
        "require_score_model_confirmation": True,
        "strategy_path": str(strategy.path),
        "research_entry_gate": research_entry_gate or {"enabled": False},
        "feature_manifest": build_feature_manifest(),
        "summary": summary,
        "convergence": convergence,
        "robustness": robustness,
        "artifacts": {
            "output_dir": str(root),
            "report_json": str(root / "backtest.json"),
            "report_md": str(root / "backtest.md"),
        },
    }
    report_json = root / "backtest.json"
    report_md = root / "backtest.md"
    report_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_backtest_markdown(payload, report_md)
    return payload
