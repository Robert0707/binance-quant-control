from __future__ import annotations

import dataclasses
import itertools
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from .analysis import enrich_indicators, prepare_klines_frame
from .asset_routing import (
    AssetRoute,
    load_asset_routes,
    normalize_symbol,
    resolve_symbol_route,
)
from .backtest import simulate_backtest
from .binance_api import BinanceAPIError, BinanceClient
from .config import STATE_DIR, Settings, ensure_runtime_dirs, load_settings
from .convergence import ConvergenceMetrics, evaluate_convergence
from .daily_digest import DEFAULT_NEWS_FEEDS, assess_news_risk, collect_news_items
from .exit_profiles import normalize_exit_profile
from .historical_klines import fetch_recent_klines
from .historical_signal_risk import (
    HistoricalSignalRiskIndex,
    build_historical_signal_risk_index,
    evaluate_historical_signal_risk,
)
from .payoff_objective import PayoffObjectiveTargets, payoff_objective_sort_key
from .route_risk_control import load_route_risk_state, route_quarantine_status
from .side_risk_policy import SideRiskEvaluation, evaluate_route_side_risk
from .strategy import StrategyConfig, load_strategy_config

RISK_COMBO_SWEEP_DIR = STATE_DIR / "risk-combo-sweeps"
RISK_COMBO_MATRIX_DIR = STATE_DIR / "risk-combo-matrix"
PROMISING_RESEARCH_STATUSES = {
    "robust_recovery_candidate",
    "recovery_candidate_needs_robust_validation",
    "promising_but_under_validated",
}
EMERGING_RESEARCH_STATUS = "emerging_positive_research_lead"
NON_PROMOTABLE_RESEARCH_STATUSES = PROMISING_RESEARCH_STATUSES | {EMERGING_RESEARCH_STATUS}
DEFAULT_ROUTE_SYMBOLS: dict[str, tuple[str, ...]] = {
    "btc-core": ("BTCUSDT",),
    "meme-high-beta": ("DOGEUSDT", "1000PEPEUSDT", "WIFUSDT", "PENGUUSDT"),
    "xau-macro": ("XAUTUSDT", "PAXGUSDT"),
}
DEFAULT_QUARANTINED_ROUTES = ("btc-core", "meme-high-beta", "xau-macro")
NEWS_VETO_MODES = ("off", "current-high", "event-proxy", "strict-event-proxy")
SIDE_POLICY_MODES = (
    "baseline",
    "long-only",
    "disable-shorts",
    "shorts-extra-adx",
    "shorts-extra-confirmation",
)
STRUCTURE_POLICY_MODES = (
    "baseline",
    "macro-aligned",
    "trend-stack-aligned",
    "no-squeeze",
    "macro-trend-no-squeeze",
    "di-dominance",
)
HISTORICAL_POLICY_MODES = ("off", "feedback-bucket-veto")
ROUTE_SIDE_POLICY_MODES = ("off", "route-side-veto")
MAX_KLINE_LIMIT = 10000
MIN_WINDOW_CANDLES = 280


@dataclass(frozen=True, slots=True)
class RiskComboSweepConfig:
    routes: tuple[str, ...]
    symbols: tuple[str, ...]
    limit: int
    grid_mode: str
    target_side: str
    target_interval: str
    target_profit_factor: float
    min_test_trades: int
    min_win_rate: float
    max_stop_loss_ratio: float
    min_expectancy_r: float
    min_payoff_ratio: float
    max_symbols_per_route: int
    max_configs: int
    max_walk_forward_validations: int
    include_all_route_symbols: bool
    skip_news: bool
    top_n: int


@dataclass(frozen=True, slots=True)
class SweepDataset:
    route: AssetRoute
    strategy: StrategyConfig
    requested_symbol: str
    source_symbol: str
    market: str
    interval: str
    frame: pd.DataFrame
    fetch_log: list[dict[str, Any]]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _round_float(value: Any, digits: int = 4) -> float | str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    if math.isnan(number):
        return 0.0
    return round(number, digits)


def _metric_float(value: Any) -> float:
    if value == "inf":
        return 9999.0
    if value == "-inf":
        return -9999.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isinf(number):
        return 9999.0 if number > 0 else -9999.0
    if math.isnan(number):
        return 0.0
    return number


def _metric_float_default(value: Any, default: float) -> float:
    if value is None:
        return default
    parsed = _metric_float(value)
    return default if parsed == 0.0 and value in ("", None) else parsed


def _bounded_grid_combinations(grid: dict[str, tuple[Any, ...]], max_configs: int) -> tuple[tuple[Any, ...], ...]:
    keys = (
        "min_adx",
        "min_convergence",
        "atr_stop_multiple",
        "primary_tp_multiple",
        "exit_profile",
        "news_veto_mode",
        "side_policy_mode",
        "structure_policy_mode",
        "historical_policy_mode",
        "route_side_policy_mode",
    )
    values = [grid[key] for key in keys]
    combos = list(itertools.product(*values))
    if max_configs <= 0 or len(combos) <= max_configs:
        return tuple(combos)
    selected: list[tuple[Any, ...]] = []
    indexes = [round(item * (len(combos) - 1) / max(max_configs - 1, 1)) for item in range(max_configs)]
    for index in indexes:
        combo = combos[index]
        if combo not in selected:
            selected.append(combo)
    required_profiles = tuple(dict.fromkeys(str(item) for item in grid.get("exit_profile", ()) if str(item)))
    for profile in required_profiles:
        if any(str(combo[4]) == profile for combo in selected):
            continue
        replacement = next((combo for combo in combos if str(combo[4]) == profile), None)
        if replacement is None:
            continue
        if len(selected) < max_configs:
            selected.append(replacement)
        else:
            selected[-1] = replacement
    return tuple(selected[:max_configs])


def _active_quarantined_routes() -> tuple[str, ...]:
    state = load_route_risk_state()
    routes = tuple(str(item) for item in (state.get("active_quarantined_routes") or []) if str(item))
    return routes or DEFAULT_QUARANTINED_ROUTES


def _route_from_id(route_id: str) -> AssetRoute:
    routes = load_asset_routes()
    raw = routes.get(route_id)
    if not isinstance(raw, dict):
        raise ValueError(f"Unknown route_id: {route_id}")
    symbols = [str(item) for item in (raw.get("symbols") or []) if str(item)]
    if symbols:
        return resolve_symbol_route(symbols[0])
    raise ValueError(f"Route {route_id} has no symbols and cannot be swept directly.")


def _symbols_for_route(
    route_id: str,
    *,
    include_all_route_symbols: bool,
    max_symbols_per_route: int,
) -> tuple[str, ...]:
    if include_all_route_symbols:
        routes = load_asset_routes()
        raw = routes.get(route_id) if isinstance(routes.get(route_id), dict) else {}
        symbols = tuple(
            normalize_symbol(str(item))
            for item in (raw.get("symbols") or [])
            if str(item).strip() and str(item).upper().endswith(("USDT", "USD", "XAU", "GOLD"))
        )
    else:
        symbols = DEFAULT_ROUTE_SYMBOLS.get(route_id, ())
    if not symbols:
        route = _route_from_id(route_id)
        symbols = (normalize_symbol(route.defaults.symbol) if hasattr(route, "defaults") else "",)
    normalized = tuple(dict.fromkeys(normalize_symbol(symbol) for symbol in symbols if symbol))
    if not normalized:
        fallback = DEFAULT_ROUTE_SYMBOLS.get(route_id, ())
        normalized = tuple(dict.fromkeys(normalize_symbol(symbol) for symbol in fallback if symbol))
    if not normalized:
        raise ValueError(f"Route {route_id} has no sweepable symbols.")
    if max_symbols_per_route > 0:
        return normalized[:max_symbols_per_route]
    return normalized


def _group_symbols_by_route(symbols: tuple[str, ...]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for symbol in symbols:
        route = resolve_symbol_route(symbol)
        grouped.setdefault(route.route_id, []).append(normalize_symbol(symbol))
    return grouped


def _fetch_candidates(route: AssetRoute, requested_symbol: str) -> tuple[tuple[str, str], ...]:
    symbol = normalize_symbol(requested_symbol)
    candidates: list[tuple[str, str]] = [(symbol, route.market)]
    if route.route_id == "xau-macro":
        candidates.extend(
            [
                ("XAUTUSDT", "futures"),
                ("PAXGUSDT", "futures"),
                ("PAXGUSDT", "spot"),
            ]
        )
    seen: list[tuple[str, str]] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.append(candidate)
    return tuple(seen)


def fetch_dataset(
    settings: Settings,
    *,
    route_id: str,
    requested_symbol: str,
    limit: int,
    interval: str = "",
) -> SweepDataset:
    route = resolve_symbol_route(requested_symbol)
    if route.route_id != route_id:
        route = _route_from_id(route_id)
    target_interval = str(interval or "").strip() or route.interval
    strategy = load_strategy_config(route.strategy_config)
    fetch_log: list[dict[str, Any]] = []
    last_error = ""
    with BinanceClient(settings) as client:
        for source_symbol, market in _fetch_candidates(route, requested_symbol):
            try:
                rows = fetch_recent_klines(
                    client,
                    source_symbol,
                    target_interval,
                    min(max(limit, MIN_WINDOW_CANDLES), MAX_KLINE_LIMIT),
                    market,
                )
                frame = enrich_indicators(prepare_klines_frame(rows), target_interval, strategy=strategy)
            except (BinanceAPIError, RuntimeError, ValueError) as exc:
                last_error = str(exc)
                fetch_log.append(
                    {
                        "requested_symbol": requested_symbol,
                        "source_symbol": source_symbol,
                        "market": market,
                        "interval": target_interval,
                        "status": "failed",
                        "error": last_error,
                    }
                )
                continue
            fetch_log.append(
                {
                    "requested_symbol": requested_symbol,
                    "source_symbol": source_symbol,
                    "market": market,
                    "interval": target_interval,
                    "status": "ok",
                    "candles": len(frame),
                }
            )
            if len(frame) < MIN_WINDOW_CANDLES:
                last_error = f"insufficient candles: {len(frame)}"
                continue
            return SweepDataset(
                route=route,
                strategy=strategy,
                requested_symbol=normalize_symbol(requested_symbol),
                source_symbol=source_symbol,
                market=market,
                interval=target_interval,
                frame=frame,
                fetch_log=fetch_log,
            )
    raise RuntimeError(
        f"No sweep dataset available for {requested_symbol} on route {route_id}: {last_error}"
    )


def _grid_values(base: StrategyConfig, mode: str = "fast") -> dict[str, tuple[Any, ...]]:
    base_adx = float(base.risk.min_adx)
    base_convergence = float(base.risk.min_convergence)
    base_atr = float(base.risk.atr_stop_multiple)
    base_tp = float(base.primary_tp_multiple)
    base_exit_profile = normalize_exit_profile(base.risk.exit_profile)
    if mode == "standard":
        min_adx_values = (base_adx - 6.0, base_adx - 2.0, base_adx, base_adx + 4.0)
        min_convergence_values = (
            base_convergence - 0.18,
            base_convergence - 0.10,
            base_convergence - 0.04,
            base_convergence,
        )
        atr_values = (base_atr * 0.75, base_atr, base_atr * 1.25)
        tp_values = (0.8, 1.0, base_tp, 1.5, 2.0)
        news_modes = NEWS_VETO_MODES
        side_policy_modes = SIDE_POLICY_MODES
        structure_policy_modes = STRUCTURE_POLICY_MODES
        historical_policy_modes = HISTORICAL_POLICY_MODES
        exit_profiles = tuple(
            dict.fromkeys((base_exit_profile, "asymmetric_payoff", "payoff_runner", "capital_preservation"))
        )
    elif mode == "focused":
        min_adx_values = (base_adx, base_adx + 4.0)
        min_convergence_values = (base_convergence - 0.04, base_convergence)
        atr_values = (base_atr, base_atr * 1.25)
        tp_values = (base_tp, 1.6, 2.2)
        news_modes = ("off",)
        side_policy_modes = ("baseline", "shorts-extra-adx")
        structure_policy_modes = ("baseline", "macro-trend-no-squeeze")
        historical_policy_modes = ("feedback-bucket-veto",)
        exit_profiles = tuple(dict.fromkeys((base_exit_profile, "asymmetric_payoff", "payoff_runner")))
    else:
        min_adx_values = (base_adx - 4.0, base_adx, base_adx + 4.0)
        min_convergence_values = (base_convergence - 0.12, base_convergence - 0.04, base_convergence)
        atr_values = (base_atr, base_atr * 1.25)
        tp_values = (base_tp, 1.5, 2.0)
        news_modes = ("off", "current-high", "event-proxy")
        side_policy_modes = ("baseline", "long-only", "shorts-extra-adx", "shorts-extra-confirmation")
        structure_policy_modes = ("baseline", "macro-aligned", "trend-stack-aligned", "macro-trend-no-squeeze")
        historical_policy_modes = HISTORICAL_POLICY_MODES
        exit_profiles = tuple(dict.fromkeys((base_exit_profile, "asymmetric_payoff", "payoff_runner")))
    min_adx = tuple(
        dict.fromkeys(
            round(max(8.0, value), 2)
            for value in min_adx_values
        )
    )
    min_convergence = tuple(
        dict.fromkeys(
            round(min(max(value, 0.45), 0.9), 3)
            for value in min_convergence_values
        )
    )
    atr_stop_multiple = tuple(
        dict.fromkeys(round(max(0.7, value), 3) for value in atr_values)
    )
    tp_multiples = tuple(dict.fromkeys(round(max(0.6, value), 3) for value in tp_values))
    return {
        "min_adx": min_adx,
        "min_convergence": min_convergence,
        "atr_stop_multiple": atr_stop_multiple,
        "primary_tp_multiple": tp_multiples,
        "exit_profile": exit_profiles,
        "news_veto_mode": news_modes,
        "side_policy_mode": side_policy_modes,
        "structure_policy_mode": structure_policy_modes,
        "historical_policy_mode": historical_policy_modes,
    }


def _structure_policy_veto(
    previous: pd.Series,
    analysis: dict[str, Any],
    *,
    action: str,
    mode: str,
) -> str | None:
    if mode == "baseline":
        return None
    close = float(previous.get("close") or 0.0)
    sma_200 = float(previous.get("sma_200") or 0.0)
    ema_fast = float(previous.get("ema_fast") or 0.0)
    ema_slow = float(previous.get("ema_slow") or 0.0)
    adx_value = float(previous.get("adx") or 0.0)
    bb_bandwidth = float(previous.get("bb_bandwidth") or 0.0)
    plus_di = float(previous.get("plus_di") or 0.0)
    minus_di = float(previous.get("minus_di") or 0.0)

    if mode in {"macro-aligned", "macro-trend-no-squeeze"} and sma_200 > 0.0:
        if action == "BUY" and close <= sma_200:
            return "structure-policy-long-below-sma200"
        if action == "SELL" and close >= sma_200:
            return "structure-policy-short-above-sma200"

    if mode in {"trend-stack-aligned", "macro-trend-no-squeeze"} and ema_slow > 0.0:
        if action == "BUY" and not (close > ema_fast > ema_slow):
            return "structure-policy-long-trend-stack-not-aligned"
        if action == "SELL" and not (close < ema_fast < ema_slow):
            return "structure-policy-short-trend-stack-not-aligned"

    if mode in {"no-squeeze", "macro-trend-no-squeeze"}:
        if str((analysis or {}).get("regime") or "") == "squeeze":
            return "structure-policy-squeeze-regime-veto"
        if bb_bandwidth > 0.0 and bb_bandwidth <= 0.05 and adx_value < 20.0:
            return "structure-policy-squeeze-regime-veto"

    if mode == "di-dominance":
        if action == "BUY" and plus_di <= minus_di * 1.15:
            return "structure-policy-long-di-not-dominant"
        if action == "SELL" and minus_di <= plus_di * 1.15:
            return "structure-policy-short-di-not-dominant"

    return None


def _entry_filter(
    mode: str,
    news_risk: dict[str, Any],
    *,
    target_side: str = "",
    side_policy_mode: str,
    structure_policy_mode: str = "baseline",
    min_adx: float,
    min_convergence: float,
    historical_policy_mode: str = "off",
    historical_signal_index: HistoricalSignalRiskIndex | None = None,
    route_side_policy_mode: str = "off",
    route_side_evaluation: SideRiskEvaluation | None = None,
    route_id: str = "",
    symbol: str = "",
):
    risk_level = str(news_risk.get("risk_level") or "unknown")
    news_veto_mode = str(mode or "off")
    target_side = str(target_side or "").upper()
    side_mode = str(side_policy_mode or "baseline")
    structure_mode = str(structure_policy_mode or "baseline")
    historical_mode = str(historical_policy_mode or "off")
    route_side_mode = str(route_side_policy_mode or "off")

    def entry_filter(previous: pd.Series, _current: pd.Series, _analysis: dict[str, Any], _idx: int):
        action = str((_analysis or {}).get("recommended_action") or "")
        if target_side in {"BUY", "SELL"} and action != target_side:
            return False, f"target-side-{target_side.lower()}-only"
        if (
            route_side_mode == "route-side-veto"
            and route_side_evaluation is not None
            and action == route_side_evaluation.side
            and not route_side_evaluation.allowed
        ):
            return False, "route-side-history-veto"
        if historical_mode == "feedback-bucket-veto" and historical_signal_index is not None:
            signal_gate = evaluate_historical_signal_risk(
                route_id=route_id,
                symbol=symbol,
                side=action,
                score=float((_analysis or {}).get("score") or 0.0),
                convergence=float((_analysis or {}).get("convergence") or 0.0),
                index=historical_signal_index,
            )
            if not signal_gate.allowed:
                return False, "historical-feedback-bucket-veto"
        structure_veto = _structure_policy_veto(
            previous,
            _analysis,
            action=action,
            mode=structure_mode,
        )
        if structure_veto:
            return False, structure_veto
        if side_mode in {"long-only", "disable-shorts"} and action == "SELL":
            return False, "side-policy-short-veto"
        if side_mode == "shorts-extra-adx" and action == "SELL":
            adx_value = float(previous.get("adx") or 0.0)
            if adx_value < (float(min_adx) + 8.0):
                return False, "side-policy-short-extra-adx-veto"
        if side_mode == "shorts-extra-confirmation" and action == "SELL":
            adx_value = float(previous.get("adx") or 0.0)
            convergence = float((_analysis or {}).get("convergence") or 0.0)
            score = int(float((_analysis or {}).get("score") or 0.0))
            minus_di = float(previous.get("minus_di") or 0.0)
            plus_di = float(previous.get("plus_di") or 0.0)
            if adx_value < (float(min_adx) + 6.0):
                return False, "side-policy-short-extra-adx-veto"
            if convergence < min(float(min_convergence) + 0.08, 0.95):
                return False, "side-policy-short-extra-convergence-veto"
            if score > 20:
                return False, "side-policy-short-score-not-extreme"
            if minus_di <= plus_di * 1.15:
                return False, "side-policy-short-di-not-dominant"
        if news_veto_mode == "off":
            return True
        if news_veto_mode == "current-high" and risk_level == "high":
            return False, "current-high-news-risk"
        volume_z = float(previous.get("volume_zscore_20") or 0.0)
        realized_vol = float(previous.get("realized_vol_20") or 0.0)
        atr_value = float(previous.get("atr_14") or 0.0)
        candle_range = float(previous.get("high") or 0.0) - float(previous.get("low") or 0.0)
        range_atr = candle_range / atr_value if atr_value > 0.0 else 0.0
        if news_veto_mode == "event-proxy":
            if volume_z >= 2.0 or range_atr >= 2.6:
                return False, "event-risk-proxy-veto"
        elif news_veto_mode == "strict-event-proxy":
            if volume_z >= 1.3 or range_atr >= 1.9 or realized_vol >= 2.0:
                return False, "strict-event-risk-proxy-veto"
        return True

    return entry_filter


def _news_veto_filter(
    mode: str,
    news_risk: dict[str, Any],
):
    return _entry_filter(
        mode,
        news_risk,
        target_side="",
        side_policy_mode="baseline",
        structure_policy_mode="baseline",
        min_adx=0.0,
        min_convergence=0.0,
        historical_policy_mode="off",
        route_side_policy_mode="off",
    )


def _mutate_strategy(
    base: StrategyConfig,
    *,
    min_adx: float,
    min_convergence: float,
    atr_stop_multiple: float,
    primary_tp_multiple: float,
    exit_profile: str = "balanced",
) -> StrategyConfig:
    tp2 = max(primary_tp_multiple * 1.8, primary_tp_multiple + 0.8)
    tp3 = max(primary_tp_multiple * 2.8, tp2 + 0.8)
    return dataclasses.replace(
        base,
        risk=dataclasses.replace(
            base.risk,
            min_adx=float(min_adx),
            min_convergence=float(min_convergence),
            atr_stop_multiple=float(atr_stop_multiple),
            take_profit_r_multiples=(
                float(primary_tp_multiple),
                round(float(tp2), 3),
                round(float(tp3), 3),
            ),
            exit_profile=normalize_exit_profile(exit_profile),
        ),
    )


def _summarize(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "trade_count",
        "wins",
        "losses",
        "win_rate",
        "stop_loss_ratio",
        "partial_tp_then_stop_ratio",
        "avg_pnl_pct",
        "avg_r",
        "ending_equity",
        "total_return_pct",
        "max_drawdown_pct",
        "profit_factor",
        "expectancy_r",
        "avg_win_r",
        "avg_loss_r",
        "payoff_ratio",
        "break_even_win_rate",
        "expectancy_edge_points",
        "loss_streak",
        "entry_veto_count",
    )
    return {
        key: _round_float(summary.get(key), 6 if key == "ending_equity" else 4)
        for key in keys
        if key in summary
    } | {"entry_veto_reasons": summary.get("entry_veto_reasons") or {}}


def _walk_forward_slices(frame: pd.DataFrame, windows: int = 3) -> list[tuple[int, int]]:
    if len(frame) < MIN_WINDOW_CANDLES * 2:
        return []
    window_size = len(frame) // windows
    slices: list[tuple[int, int]] = []
    for index in range(windows):
        start = index * window_size
        end = len(frame) if index == windows - 1 else (index + 1) * window_size
        if end - start >= MIN_WINDOW_CANDLES:
            slices.append((start, end))
    return slices


def _walk_forward(
    frame: pd.DataFrame,
    *,
    market: str,
    strategy: StrategyConfig,
    news_veto_mode: str,
    target_side: str = "",
    side_policy_mode: str = "baseline",
    structure_policy_mode: str = "baseline",
    historical_policy_mode: str = "off",
    historical_signal_index: HistoricalSignalRiskIndex | None = None,
    route_side_policy_mode: str = "off",
    route_side_evaluations: Mapping[str, SideRiskEvaluation] | None = None,
    route_id: str = "",
    symbol: str = "",
    news_risk: dict[str, Any],
) -> dict[str, Any]:
    windows: list[dict[str, Any]] = []
    base_filter = _entry_filter(
        news_veto_mode,
        news_risk,
        target_side=target_side,
        side_policy_mode=side_policy_mode,
        structure_policy_mode=structure_policy_mode,
        min_adx=strategy.risk.min_adx,
        min_convergence=strategy.risk.min_convergence,
        historical_policy_mode=historical_policy_mode,
        historical_signal_index=historical_signal_index,
        route_side_policy_mode=route_side_policy_mode,
        route_side_evaluation=None,
        route_id=route_id,
        symbol=symbol,
    )
    long_filter = _entry_filter(
        news_veto_mode,
        news_risk,
        target_side=target_side,
        side_policy_mode=side_policy_mode,
        structure_policy_mode=structure_policy_mode,
        min_adx=strategy.risk.min_adx,
        min_convergence=strategy.risk.min_convergence,
        historical_policy_mode=historical_policy_mode,
        historical_signal_index=historical_signal_index,
        route_side_policy_mode=route_side_policy_mode,
        route_side_evaluation=(route_side_evaluations or {}).get("BUY"),
        route_id=route_id,
        symbol=symbol,
    )
    short_filter = _entry_filter(
        news_veto_mode,
        news_risk,
        target_side=target_side,
        side_policy_mode=side_policy_mode,
        structure_policy_mode=structure_policy_mode,
        min_adx=strategy.risk.min_adx,
        min_convergence=strategy.risk.min_convergence,
        historical_policy_mode=historical_policy_mode,
        historical_signal_index=historical_signal_index,
        route_side_policy_mode=route_side_policy_mode,
        route_side_evaluation=(route_side_evaluations or {}).get("SELL"),
        route_id=route_id,
        symbol=symbol,
    )

    def routed_entry_filter(
        previous: pd.Series,
        current: pd.Series,
        analysis: dict[str, Any],
        idx: int,
    ) -> bool | tuple[bool, str]:
        action = str((analysis or {}).get("recommended_action") or "")
        if action == "BUY":
            return long_filter(previous, current, analysis, idx)
        if action == "SELL":
            return short_filter(previous, current, analysis, idx)
        return base_filter(previous, current, analysis, idx)

    for window_number, (start, end) in enumerate(_walk_forward_slices(frame), start=1):
        summary = simulate_backtest(
            frame.iloc[start:end].copy(),
            market,
            strategy,
            entry_filter=routed_entry_filter,
        )
        slim = _summarize(summary)
        windows.append(
            {
                "window": window_number,
                "start_index": start,
                "end_index": end,
                "summary": slim,
            }
        )
    profit_factors = [_metric_float((item["summary"] or {}).get("profit_factor")) for item in windows]
    expectancies = [_metric_float((item["summary"] or {}).get("expectancy_r")) for item in windows]
    payoff_ratios = [_metric_float((item["summary"] or {}).get("payoff_ratio")) for item in windows]
    positive_windows = [
        item
        for item in windows
        if _metric_float((item["summary"] or {}).get("total_return_pct")) > 0.0
    ]
    positive_expectancy_windows = [
        item
        for item in windows
        if _metric_float((item["summary"] or {}).get("expectancy_r")) > 0.0
    ]
    return {
        "window_count": len(windows),
        "positive_window_count": len(positive_windows),
        "positive_expectancy_window_count": len(positive_expectancy_windows),
        "mean_profit_factor": round(mean(profit_factors), 4) if profit_factors else 0.0,
        "min_profit_factor": round(min(profit_factors), 4) if profit_factors else 0.0,
        "mean_expectancy_r": round(mean(expectancies), 4) if expectancies else 0.0,
        "min_expectancy_r": round(min(expectancies), 4) if expectancies else 0.0,
        "min_payoff_ratio": round(min(payoff_ratios), 4) if payoff_ratios else 0.0,
        "windows": windows,
    }


def _evaluate_combo(
    dataset: SweepDataset,
    *,
    min_adx: float,
    min_convergence: float,
    atr_stop_multiple: float,
    primary_tp_multiple: float,
    exit_profile: str = "balanced",
    news_veto_mode: str,
    target_side: str = "",
    side_policy_mode: str = "baseline",
    structure_policy_mode: str = "baseline",
    historical_policy_mode: str = "off",
    historical_signal_index: HistoricalSignalRiskIndex | None = None,
    route_side_policy_mode: str = "off",
    route_side_evaluations: Mapping[str, SideRiskEvaluation] | None = None,
    news_risk: dict[str, Any],
    target_profit_factor: float,
    min_test_trades: int,
    min_win_rate: float,
    max_stop_loss_ratio: float,
    min_expectancy_r: float,
    min_payoff_ratio: float,
    include_walk_forward: bool = False,
) -> dict[str, Any]:
    strategy = _mutate_strategy(
        dataset.strategy,
        min_adx=min_adx,
        min_convergence=min_convergence,
        atr_stop_multiple=atr_stop_multiple,
        primary_tp_multiple=primary_tp_multiple,
        exit_profile=exit_profile,
    )
    split_idx = max(int(len(dataset.frame) * 0.7), MIN_WINDOW_CANDLES)
    if len(dataset.frame) - split_idx < MIN_WINDOW_CANDLES:
        split_idx = max(len(dataset.frame) - MIN_WINDOW_CANDLES, MIN_WINDOW_CANDLES)
    train_frame = dataset.frame.iloc[:split_idx].copy()
    test_frame = dataset.frame.iloc[split_idx:].copy()
    entry_filter = _entry_filter(
        news_veto_mode,
        news_risk,
        target_side=target_side,
        side_policy_mode=side_policy_mode,
        structure_policy_mode=structure_policy_mode,
        min_adx=min_adx,
        min_convergence=min_convergence,
        historical_policy_mode=historical_policy_mode,
        historical_signal_index=historical_signal_index,
        route_side_policy_mode=route_side_policy_mode,
        route_side_evaluation=None,
        route_id=dataset.route.route_id,
        symbol=dataset.requested_symbol,
    )
    long_filter = _entry_filter(
        news_veto_mode,
        news_risk,
        target_side=target_side,
        side_policy_mode=side_policy_mode,
        structure_policy_mode=structure_policy_mode,
        min_adx=min_adx,
        min_convergence=min_convergence,
        historical_policy_mode=historical_policy_mode,
        historical_signal_index=historical_signal_index,
        route_side_policy_mode=route_side_policy_mode,
        route_side_evaluation=(route_side_evaluations or {}).get("BUY"),
        route_id=dataset.route.route_id,
        symbol=dataset.requested_symbol,
    )
    short_filter = _entry_filter(
        news_veto_mode,
        news_risk,
        target_side=target_side,
        side_policy_mode=side_policy_mode,
        structure_policy_mode=structure_policy_mode,
        min_adx=min_adx,
        min_convergence=min_convergence,
        historical_policy_mode=historical_policy_mode,
        historical_signal_index=historical_signal_index,
        route_side_policy_mode=route_side_policy_mode,
        route_side_evaluation=(route_side_evaluations or {}).get("SELL"),
        route_id=dataset.route.route_id,
        symbol=dataset.requested_symbol,
    )

    def routed_entry_filter(
        previous: pd.Series,
        current: pd.Series,
        analysis: dict[str, Any],
        idx: int,
    ) -> bool | tuple[bool, str]:
        action = str((analysis or {}).get("recommended_action") or "")
        if action == "BUY":
            return long_filter(previous, current, analysis, idx)
        if action == "SELL":
            return short_filter(previous, current, analysis, idx)
        return entry_filter(previous, current, analysis, idx)

    entry_filter = routed_entry_filter
    full = simulate_backtest(dataset.frame, dataset.market, strategy, entry_filter=entry_filter)
    train = simulate_backtest(train_frame, dataset.market, strategy, entry_filter=entry_filter)
    test = simulate_backtest(test_frame, dataset.market, strategy, entry_filter=entry_filter)
    test_metrics = ConvergenceMetrics(
        trade_count=int(test.get("trade_count") or 0),
        win_rate=float(test.get("win_rate") or 0.0),
        profit_factor=_metric_float(test.get("profit_factor")),
        max_drawdown_pct=float(test.get("max_drawdown_pct") or 0.0),
        loss_streak=int(test.get("loss_streak") or 0),
        expectancy_r=_metric_float(test.get("expectancy_r")),
        payoff_ratio=_metric_float(test.get("payoff_ratio")),
    )
    full_pf = _metric_float(full.get("profit_factor"))
    test_pf = _metric_float(test.get("profit_factor"))
    recovery_candidate = (
        test_metrics.trade_count >= min_test_trades
        and float(test.get("win_rate") or 0.0) >= min_win_rate
        and float(test.get("stop_loss_ratio") or 0.0) <= max_stop_loss_ratio
        and test_pf >= target_profit_factor
        and full_pf >= target_profit_factor
        and _metric_float(test.get("expectancy_r")) >= min_expectancy_r
        and _metric_float(full.get("expectancy_r")) >= min_expectancy_r
        and _metric_float(test.get("payoff_ratio")) >= min_payoff_ratio
        and _metric_float(full.get("payoff_ratio")) >= min_payoff_ratio
    )
    if include_walk_forward:
        walk_forward = _walk_forward(
            dataset.frame,
            market=dataset.market,
            strategy=strategy,
            news_veto_mode=news_veto_mode,
            target_side=target_side,
            side_policy_mode=side_policy_mode,
            structure_policy_mode=structure_policy_mode,
            historical_policy_mode=historical_policy_mode,
            historical_signal_index=historical_signal_index,
            route_side_policy_mode=route_side_policy_mode,
            route_side_evaluations=route_side_evaluations,
            route_id=dataset.route.route_id,
            symbol=dataset.requested_symbol,
            news_risk=news_risk,
        )
    else:
        walk_forward = {
            "status": "deferred",
            "window_count": 0,
            "positive_window_count": 0,
            "mean_profit_factor": 0.0,
            "min_profit_factor": 0.0,
            "windows": [],
        }
    return {
        "route_id": dataset.route.route_id,
        "asset_class": dataset.route.asset_class,
        "requested_symbol": dataset.requested_symbol,
        "source_symbol": dataset.source_symbol,
        "market": dataset.market,
        "interval": dataset.interval,
        "strategy_profile": dataset.strategy.profile,
        "params": {
            "min_adx": float(min_adx),
            "min_convergence": float(min_convergence),
            "atr_stop_multiple": float(atr_stop_multiple),
            "primary_tp_multiple": float(primary_tp_multiple),
            "take_profit_r_multiples": list(strategy.risk.take_profit_r_multiples),
            "exit_profile": strategy.risk.exit_profile,
            "news_veto_mode": news_veto_mode,
            "target_side": target_side,
            "side_policy_mode": side_policy_mode,
            "structure_policy_mode": structure_policy_mode,
            "historical_policy_mode": historical_policy_mode,
            "route_side_policy_mode": route_side_policy_mode,
        },
        "route_side_gate": {
            side: evaluation.to_dict()
            for side, evaluation in sorted((route_side_evaluations or {}).items())
        },
        "full": _summarize(full),
        "train": _summarize(train),
        "test": _summarize(test),
        "walk_forward": walk_forward,
        "recovery_gate": {
            "target_profit_factor": target_profit_factor,
            "min_test_trades": min_test_trades,
            "min_win_rate": min_win_rate,
            "max_stop_loss_ratio": max_stop_loss_ratio,
            "min_expectancy_r": min_expectancy_r,
            "min_payoff_ratio": min_payoff_ratio,
            "passed": recovery_candidate,
            "reasons": _recovery_reasons(
                test_summary=test,
                full_summary=full,
                target_profit_factor=target_profit_factor,
                min_test_trades=min_test_trades,
                min_win_rate=min_win_rate,
                max_stop_loss_ratio=max_stop_loss_ratio,
                min_expectancy_r=min_expectancy_r,
                min_payoff_ratio=min_payoff_ratio,
            ),
        },
        "robust_recovery_gate": {
            "status": "not_validated",
            "passed": False,
            "reasons": ["walk-forward-validation-not-run"],
        },
        "convergence": evaluate_convergence(test_metrics, dataset.route.validation),
    }


def _recovery_reasons(
    *,
    test_summary: dict[str, Any],
    full_summary: dict[str, Any],
    target_profit_factor: float,
    min_test_trades: int,
    min_win_rate: float,
    max_stop_loss_ratio: float,
    min_expectancy_r: float,
    min_payoff_ratio: float,
) -> list[str]:
    reasons: list[str] = []
    if int(test_summary.get("trade_count") or 0) < min_test_trades:
        reasons.append("test-trade-count-too-low")
    if float(test_summary.get("win_rate") or 0.0) < min_win_rate:
        reasons.append("test-win-rate-below-target")
    if float(test_summary.get("stop_loss_ratio") or 0.0) > max_stop_loss_ratio:
        reasons.append("test-stop-loss-ratio-above-target")
    if _metric_float(test_summary.get("profit_factor")) < target_profit_factor:
        reasons.append("test-profit-factor-below-target")
    if _metric_float(full_summary.get("profit_factor")) < target_profit_factor:
        reasons.append("full-profit-factor-below-target")
    if _metric_float(test_summary.get("expectancy_r")) < min_expectancy_r:
        reasons.append("test-expectancy-r-below-target")
    if _metric_float(full_summary.get("expectancy_r")) < min_expectancy_r:
        reasons.append("full-expectancy-r-below-target")
    if _metric_float(test_summary.get("payoff_ratio")) < min_payoff_ratio:
        reasons.append("test-payoff-ratio-below-target")
    if _metric_float(full_summary.get("payoff_ratio")) < min_payoff_ratio:
        reasons.append("full-payoff-ratio-below-target")
    return reasons


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    recovery_passed = bool((row.get("recovery_gate") or {}).get("passed"))
    robust_passed = bool((row.get("robust_recovery_gate") or {}).get("passed"))
    test = row.get("test") or {}
    full = row.get("full") or {}
    walk_forward = row.get("walk_forward") or {}
    gate = row.get("recovery_gate") or {}
    targets = PayoffObjectiveTargets(
        min_trades=int(gate.get("min_test_trades") or 100),
        min_profit_factor=_metric_float_default(gate.get("target_profit_factor"), 1.5),
        min_expectancy_r=_metric_float_default(gate.get("min_expectancy_r"), 0.10),
        min_payoff_ratio=_metric_float_default(gate.get("min_payoff_ratio"), 1.15),
        min_win_rate=_metric_float_default(gate.get("min_win_rate"), 65.0),
        max_stop_loss_ratio=_metric_float_default(gate.get("max_stop_loss_ratio"), 35.0),
    )
    payoff_key = payoff_objective_sort_key(
        test,
        targets=targets,
        min_trades=targets.min_trades,
    )
    return (
        robust_passed,
        recovery_passed,
        *payoff_key,
        _metric_float(test.get("profit_factor")),
        _metric_float(full.get("profit_factor")),
        _metric_float(walk_forward.get("min_profit_factor")),
        -_metric_float(test.get("max_drawdown_pct")),
        -int(test.get("loss_streak") or 0),
        int(test.get("trade_count") or 0),
        _metric_float(test.get("total_return_pct")),
    )


def _dataset_key(row: dict[str, Any] | SweepDataset) -> tuple[str, str, str, str, str]:
    if isinstance(row, SweepDataset):
        return (
            row.route.route_id,
            row.requested_symbol,
            row.source_symbol,
            row.market,
            row.interval,
        )
    return (
        str(row.get("route_id") or ""),
        str(row.get("requested_symbol") or ""),
        str(row.get("source_symbol") or ""),
        str(row.get("market") or ""),
        str(row.get("interval") or ""),
    )


def _validation_value(validation: Any, key: str, default: float) -> float:
    if isinstance(validation, Mapping):
        value = validation.get(key, default)
    else:
        value = getattr(validation, key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _robust_recovery_gate(
    row: dict[str, Any],
    *,
    dataset: SweepDataset,
    target_profit_factor: float,
    min_test_trades: int,
    min_win_rate: float,
    max_stop_loss_ratio: float,
    min_expectancy_r: float,
    min_payoff_ratio: float,
) -> dict[str, Any]:
    reasons = list((row.get("recovery_gate") or {}).get("reasons") or [])
    if not bool((row.get("recovery_gate") or {}).get("passed")):
        reasons.append("initial-recovery-gate-not-passed")

    train = row.get("train") or {}
    full = row.get("full") or {}
    test = row.get("test") or {}
    walk_forward = row.get("walk_forward") or {}
    validation = dataset.route.validation
    max_drawdown_pct = _validation_value(validation, "max_drawdown_pct", 15.0)
    max_loss_streak = int(_validation_value(validation, "max_loss_streak", 3.0))
    window_count = int(walk_forward.get("window_count") or 0)
    positive_window_count = int(walk_forward.get("positive_window_count") or 0)
    required_positive_windows = max(1, math.ceil(window_count * 2 / 3)) if window_count else 0

    if int(test.get("trade_count") or 0) < min_test_trades:
        reasons.append("test-trade-count-too-low")
    if float(test.get("win_rate") or 0.0) < min_win_rate:
        reasons.append("test-win-rate-below-target")
    if float(test.get("stop_loss_ratio") or 0.0) > max_stop_loss_ratio:
        reasons.append("test-stop-loss-ratio-above-target")
    if _metric_float(train.get("profit_factor")) < target_profit_factor:
        reasons.append("train-profit-factor-below-target")
    if _metric_float(full.get("profit_factor")) < target_profit_factor:
        reasons.append("full-profit-factor-below-target")
    if _metric_float(test.get("profit_factor")) < target_profit_factor:
        reasons.append("test-profit-factor-below-target")
    if _metric_float(train.get("expectancy_r")) < min_expectancy_r:
        reasons.append("train-expectancy-r-below-target")
    if _metric_float(full.get("expectancy_r")) < min_expectancy_r:
        reasons.append("full-expectancy-r-below-target")
    if _metric_float(test.get("expectancy_r")) < min_expectancy_r:
        reasons.append("test-expectancy-r-below-target")
    if _metric_float(train.get("payoff_ratio")) < min_payoff_ratio:
        reasons.append("train-payoff-ratio-below-target")
    if _metric_float(full.get("payoff_ratio")) < min_payoff_ratio:
        reasons.append("full-payoff-ratio-below-target")
    if _metric_float(test.get("payoff_ratio")) < min_payoff_ratio:
        reasons.append("test-payoff-ratio-below-target")
    if float(full.get("max_drawdown_pct") or 0.0) > max_drawdown_pct:
        reasons.append("full-max-drawdown-above-route-limit")
    if int(full.get("loss_streak") or 0) > max_loss_streak:
        reasons.append("full-loss-streak-above-route-limit")
    if walk_forward.get("status") == "deferred":
        reasons.append("walk-forward-validation-not-run")
    elif window_count <= 0:
        reasons.append("walk-forward-window-count-too-low")
    else:
        if _metric_float(walk_forward.get("min_profit_factor")) < target_profit_factor:
            reasons.append("walk-forward-min-profit-factor-below-target")
        if _metric_float(walk_forward.get("min_expectancy_r")) < min_expectancy_r:
            reasons.append("walk-forward-min-expectancy-r-below-target")
        if _metric_float(walk_forward.get("min_payoff_ratio")) < min_payoff_ratio:
            reasons.append("walk-forward-min-payoff-ratio-below-target")
        if int(walk_forward.get("positive_expectancy_window_count") or 0) < required_positive_windows:
            reasons.append("walk-forward-positive-expectancy-window-count-too-low")
        if positive_window_count < required_positive_windows:
            reasons.append("walk-forward-positive-window-count-too-low")

    deduped_reasons = list(dict.fromkeys(reasons))
    return {
        "status": "validated" if window_count else "not_validated",
        "passed": not deduped_reasons,
        "target_profit_factor": target_profit_factor,
        "min_test_trades": min_test_trades,
        "min_win_rate": min_win_rate,
        "max_stop_loss_ratio": max_stop_loss_ratio,
        "min_expectancy_r": min_expectancy_r,
        "min_payoff_ratio": min_payoff_ratio,
        "route_max_drawdown_pct": max_drawdown_pct,
        "route_max_loss_streak": max_loss_streak,
        "required_positive_windows": required_positive_windows,
        "reasons": deduped_reasons,
    }


def _attach_walk_forward(
    row: dict[str, Any],
    *,
    datasets: dict[tuple[str, str, str, str, str], SweepDataset],
    news_risk: dict[str, Any],
    historical_signal_index: HistoricalSignalRiskIndex | None,
    route_side_evaluations: Mapping[str, SideRiskEvaluation] | None,
    target_profit_factor: float,
    min_test_trades: int,
    min_win_rate: float,
    max_stop_loss_ratio: float,
    min_expectancy_r: float,
    min_payoff_ratio: float,
) -> None:
    if (row.get("walk_forward") or {}).get("status") != "deferred":
        return
    dataset = datasets.get(_dataset_key(row))
    if dataset is None:
        return
    params = row.get("params") or {}
    strategy = _mutate_strategy(
        dataset.strategy,
        min_adx=float(params.get("min_adx") or dataset.strategy.risk.min_adx),
        min_convergence=float(params.get("min_convergence") or dataset.strategy.risk.min_convergence),
        atr_stop_multiple=float(params.get("atr_stop_multiple") or dataset.strategy.risk.atr_stop_multiple),
        primary_tp_multiple=float(params.get("primary_tp_multiple") or dataset.strategy.primary_tp_multiple),
        exit_profile=str(params.get("exit_profile") or dataset.strategy.risk.exit_profile),
    )
    row["walk_forward"] = _walk_forward(
        dataset.frame,
        market=dataset.market,
        strategy=strategy,
        news_veto_mode=str(params.get("news_veto_mode") or "off"),
        target_side=str(params.get("target_side") or ""),
        side_policy_mode=str(params.get("side_policy_mode") or "baseline"),
        structure_policy_mode=str(params.get("structure_policy_mode") or "baseline"),
        historical_policy_mode=str(params.get("historical_policy_mode") or "off"),
        historical_signal_index=historical_signal_index,
        route_side_policy_mode=str(params.get("route_side_policy_mode") or "off"),
        route_side_evaluations=route_side_evaluations,
        route_id=dataset.route.route_id,
        symbol=dataset.requested_symbol,
        news_risk=news_risk,
    )
    row["robust_recovery_gate"] = _robust_recovery_gate(
        row,
        dataset=dataset,
        target_profit_factor=target_profit_factor,
        min_test_trades=min_test_trades,
        min_win_rate=min_win_rate,
        max_stop_loss_ratio=max_stop_loss_ratio,
        min_expectancy_r=min_expectancy_r,
        min_payoff_ratio=min_payoff_ratio,
    )


def _collect_news(skip_news: bool) -> dict[str, Any]:
    if skip_news:
        return {
            "available": False,
            "risk": {"risk_level": "unknown", "bias": "neutral", "high_impact_count": 0},
            "items": [],
            "error": "news collection skipped",
        }
    try:
        items = collect_news_items(DEFAULT_NEWS_FEEDS, news_limit=12)
        return {
            "available": True,
            "feeds": list(DEFAULT_NEWS_FEEDS),
            "items": items,
            "risk": assess_news_risk(items),
        }
    except Exception as exc:
        return {
            "available": False,
            "feeds": list(DEFAULT_NEWS_FEEDS),
            "items": [],
            "risk": {"risk_level": "unknown", "bias": "neutral", "high_impact_count": 0},
            "error": str(exc),
        }


def _slim_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "route_id": row.get("route_id"),
        "requested_symbol": row.get("requested_symbol"),
        "source_symbol": row.get("source_symbol"),
        "market": row.get("market"),
        "interval": row.get("interval"),
        "strategy_profile": row.get("strategy_profile"),
        "params": row.get("params"),
        "full": row.get("full"),
        "train": row.get("train"),
        "test": row.get("test"),
        "walk_forward": {
            key: value
            for key, value in (row.get("walk_forward") or {}).items()
            if key != "windows"
        },
        "recovery_gate": row.get("recovery_gate"),
        "robust_recovery_gate": row.get("robust_recovery_gate"),
        "convergence": row.get("convergence"),
    }


def _load_sweep_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Sweep report must be a JSON object: {report_path}")
    if not payload.get("report_path"):
        payload["report_path"] = str(report_path)
    return payload


def _first_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}


def _best_report_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    ranked = payload.get("ranked")
    if isinstance(ranked, list) and ranked and isinstance(ranked[0], dict):
        return ranked[0]
    best_by_symbol = payload.get("best_by_symbol")
    if isinstance(best_by_symbol, dict) and best_by_symbol:
        return _first_mapping(next(iter(best_by_symbol.values())))
    best_by_route = payload.get("best_by_route")
    if isinstance(best_by_route, dict) and best_by_route:
        return _first_mapping(next(iter(best_by_route.values())))
    return {}


def _skipped_matrix_input(payload: dict[str, Any]) -> dict[str, Any] | None:
    aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), dict) else {}
    dataset_count_raw = aggregate.get("dataset_count")
    configs_tested_raw = aggregate.get("configs_tested")
    dataset_count = int(dataset_count_raw or 0)
    configs_tested = int(configs_tested_raw or 0)
    candidate = _best_report_candidate(payload)
    reason = ""
    if str(payload.get("status") or "") == "no_datasets":
        reason = "no_datasets"
    elif dataset_count_raw is not None and dataset_count <= 0:
        reason = "no_datasets"
    elif configs_tested_raw is not None and configs_tested <= 0:
        reason = "no_configs_tested"
    elif not candidate:
        reason = "no_ranked_candidate"
    if not reason:
        return None
    dataset_errors = payload.get("dataset_errors")
    return {
        "report_path": payload.get("report_path"),
        "status": payload.get("status"),
        "reason": reason,
        "dataset_count": dataset_count,
        "configs_tested": configs_tested,
        "dataset_error_count": len(dataset_errors) if isinstance(dataset_errors, list) else 0,
        "target_side": aggregate.get("target_side"),
        "target_interval": aggregate.get("target_interval"),
    }


def _sweep_surface_row(payload: dict[str, Any], candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), dict) else {}
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    candidate = candidate if isinstance(candidate, dict) else _best_report_candidate(payload)
    params = candidate.get("params") if isinstance(candidate.get("params"), dict) else {}
    side = str(aggregate.get("target_side") or config.get("target_side") or params.get("target_side") or "UNKNOWN")
    interval = str(aggregate.get("target_interval") or config.get("target_interval") or candidate.get("interval") or "unknown")
    full = candidate.get("full") if isinstance(candidate.get("full"), dict) else {}
    train = candidate.get("train") if isinstance(candidate.get("train"), dict) else {}
    test = candidate.get("test") if isinstance(candidate.get("test"), dict) else {}
    walk_forward = candidate.get("walk_forward") if isinstance(candidate.get("walk_forward"), dict) else {}
    recovery_gate = candidate.get("recovery_gate") if isinstance(candidate.get("recovery_gate"), dict) else {}
    robust_gate = candidate.get("robust_recovery_gate") if isinstance(candidate.get("robust_recovery_gate"), dict) else {}
    full_pf = _metric_float(full.get("profit_factor"))
    full_expectancy = _metric_float(full.get("expectancy_r"))
    test_pf = _metric_float(test.get("profit_factor"))
    test_expectancy = _metric_float(test.get("expectancy_r"))
    recovery_passed = bool(recovery_gate.get("passed"))
    robust_passed = bool(robust_gate.get("passed"))
    full_trade_count = int(full.get("trade_count") or 0)
    test_trade_count = int(test.get("trade_count") or 0)
    if robust_passed:
        research_status = "robust_recovery_candidate"
    elif recovery_passed:
        research_status = "recovery_candidate_needs_robust_validation"
    elif (
        full_pf >= 1.0
        and full_expectancy > 0.0
        and test_pf >= 1.0
        and test_expectancy >= 0.0
        and full_trade_count >= 20
        and test_trade_count >= 5
    ):
        research_status = "promising_but_under_validated"
    elif (
        full_pf >= 1.0
        and full_expectancy > 0.0
        and test_pf >= 1.0
        and test_expectancy >= 0.0
    ):
        research_status = EMERGING_RESEARCH_STATUS
    else:
        research_status = "rejected_or_negative_expectancy"
    status_rank = {
        "robust_recovery_candidate": 4,
        "recovery_candidate_needs_robust_validation": 3,
        "promising_but_under_validated": 2,
        EMERGING_RESEARCH_STATUS: 1,
        "rejected_or_negative_expectancy": 0,
    }[research_status]
    return {
        "surface": f"{side.lower()}_{interval}",
        "target_side": side,
        "target_interval": interval,
        "route_id": candidate.get("route_id"),
        "symbol": candidate.get("requested_symbol") or candidate.get("source_symbol"),
        "research_status": research_status,
        "promotion_eligible": research_status in {
            "robust_recovery_candidate",
            "recovery_candidate_needs_robust_validation",
        },
        "research_lead_only": research_status == EMERGING_RESEARCH_STATUS,
        "research_lead_reason": (
            "positive_full_and_test_expectancy_but_sample_too_small_for_candidate_gate"
            if research_status == EMERGING_RESEARCH_STATUS
            else ""
        ),
        "recovery_gate_passed": recovery_passed,
        "robust_recovery_gate_passed": robust_passed,
        "full": {
            "trade_count": full.get("trade_count"),
            "wins": full.get("wins"),
            "losses": full.get("losses"),
            "win_rate": full.get("win_rate"),
            "loss_streak": full.get("loss_streak"),
            "profit_factor": full.get("profit_factor"),
            "expectancy_r": full.get("expectancy_r"),
            "avg_win_r": full.get("avg_win_r"),
            "avg_loss_r": full.get("avg_loss_r"),
            "payoff_ratio": full.get("payoff_ratio"),
            "break_even_win_rate": full.get("break_even_win_rate"),
            "expectancy_edge_points": full.get("expectancy_edge_points"),
            "max_drawdown_pct": full.get("max_drawdown_pct"),
            "stop_loss_ratio": full.get("stop_loss_ratio"),
            "partial_tp_then_stop_ratio": full.get("partial_tp_then_stop_ratio"),
        },
        "train": {
            "trade_count": train.get("trade_count"),
            "win_rate": train.get("win_rate"),
            "profit_factor": train.get("profit_factor"),
            "expectancy_r": train.get("expectancy_r"),
            "payoff_ratio": train.get("payoff_ratio"),
        },
        "test": {
            "trade_count": test.get("trade_count"),
            "wins": test.get("wins"),
            "losses": test.get("losses"),
            "win_rate": test.get("win_rate"),
            "profit_factor": test.get("profit_factor"),
            "expectancy_r": test.get("expectancy_r"),
            "avg_win_r": test.get("avg_win_r"),
            "avg_loss_r": test.get("avg_loss_r"),
            "payoff_ratio": test.get("payoff_ratio"),
            "break_even_win_rate": test.get("break_even_win_rate"),
            "expectancy_edge_points": test.get("expectancy_edge_points"),
            "max_drawdown_pct": test.get("max_drawdown_pct"),
            "stop_loss_ratio": test.get("stop_loss_ratio"),
            "partial_tp_then_stop_ratio": test.get("partial_tp_then_stop_ratio"),
        },
        "walk_forward": {
            "window_count": walk_forward.get("window_count"),
            "positive_expectancy_window_count": walk_forward.get("positive_expectancy_window_count"),
            "min_profit_factor": walk_forward.get("min_profit_factor"),
            "min_expectancy_r": walk_forward.get("min_expectancy_r"),
        },
        "gate_reasons": {
            "recovery": recovery_gate.get("reasons") or [],
            "robust": robust_gate.get("reasons") or [],
        },
        "source_report_path": payload.get("report_path"),
        "rank_key": [
            status_rank,
            1 if robust_passed else 0,
            1 if recovery_passed else 0,
            min(test_trade_count, int(recovery_gate.get("min_test_trades") or 10)),
            min(full_trade_count, 100),
            full_pf,
            full_expectancy,
            test_pf,
            test_expectancy,
            int(test.get("trade_count") or 0),
            -_metric_float(full.get("stop_loss_ratio")),
            -_metric_float(full.get("max_drawdown_pct")),
        ],
    }


def _row_identity_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("target_side") or "").upper(),
        str(row.get("target_interval") or "").lower(),
        str(row.get("route_id") or ""),
        str(row.get("symbol") or "").upper(),
    )


def _row_report_time(row: dict[str, Any]) -> float:
    path = str(row.get("source_report_path") or "")
    if not path:
        return 0.0
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def _research_status_rank(status: str) -> int:
    return {
        "robust_recovery_candidate": 4,
        "recovery_candidate_needs_robust_validation": 3,
        "promising_but_under_validated": 2,
        EMERGING_RESEARCH_STATUS: 1,
        "rejected_or_negative_expectancy": 0,
    }.get(str(status), 0)


def _active_and_superseded_research_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    newest_by_identity: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    superseded: list[dict[str, Any]] = []
    for row in sorted(
        rows,
        key=lambda item: (_row_report_time(item), tuple(item.get("rank_key") or ())),
        reverse=True,
    ):
        identity = _row_identity_key(row)
        current = newest_by_identity.get(identity)
        if current is None:
            newest_by_identity[identity] = row
            continue
        superseded_row = dict(row)
        superseded_row["superseded_by_report_path"] = current.get("source_report_path")
        superseded_row["superseded_by_research_status"] = current.get("research_status")
        superseded_row["superseded_by_metrics"] = {
            "full": current.get("full"),
            "test": current.get("test"),
            "gate_reasons": current.get("gate_reasons"),
        }
        superseded.append(superseded_row)
    active = list(newest_by_identity.values())
    return active, superseded


def _recent_failed_repair_identities(
    *,
    active_rows: list[dict[str, Any]],
    superseded_rows: list[dict[str, Any]],
) -> set[tuple[str, str, str, str]]:
    _ = active_rows
    return {
        _row_identity_key(row)
        for row in superseded_rows
        if row.get("superseded_by_research_status") == "rejected_or_negative_expectancy"
    }


def _sweep_surface_rows(payload: dict[str, Any], *, ranked_limit: int = 20) -> list[dict[str, Any]]:
    ranked = payload.get("ranked")
    if isinstance(ranked, list) and ranked:
        rows = [
            _sweep_surface_row(payload, candidate)
            for candidate in ranked[: max(int(ranked_limit), 1)]
            if isinstance(candidate, dict)
        ]
        if rows:
            return rows
    candidate = _best_report_candidate(payload)
    return [_sweep_surface_row(payload, candidate)] if candidate else []


def _interval_horizon(interval: str) -> str:
    value = str(interval or "").strip().lower()
    if value.endswith("m"):
        return "short"
    if value in {"1h", "2h", "4h", "6h", "8h", "12h"}:
        return "medium"
    if value.endswith("d") or value.endswith("w") or value.endswith("mo"):
        return "long"
    return "unknown"


def _coverage_summary(rows: list[dict[str, Any]], *, key: str, expected: tuple[str, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for item in expected:
        matching = [row for row in rows if str(row.get(key) or "").upper() == item.upper()]
        promising = [row for row in matching if row.get("research_status") in PROMISING_RESEARCH_STATUSES]
        emerging = [row for row in matching if row.get("research_status") == EMERGING_RESEARCH_STATUS]
        robust = [row for row in matching if row.get("robust_recovery_gate_passed")]
        summary[item.lower()] = {
            "surface_count": len(matching),
            "promising_surface_count": len(promising),
            "emerging_positive_lead_count": len(emerging),
            "robust_surface_count": len(robust),
            "best_surface": matching[0] if matching else None,
            "best_emerging_surface": emerging[0] if emerging else None,
            "status": (
                "robust_candidate_found"
                if robust
                else "promising_research_only"
                if promising
                else "emerging_positive_needs_sample"
                if emerging
                else "missing_or_negative_expectancy"
            ),
        }
    return summary


def _horizon_coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped_rows = [dict(row) | {"horizon": _interval_horizon(str(row.get("target_interval") or ""))} for row in rows]
    return _coverage_summary(grouped_rows, key="horizon", expected=("short", "medium", "long"))


def _validation_command_for_surface(
    row: dict[str, Any],
    *,
    limit: int,
    min_test_trades: int,
    max_configs: int,
    max_walk_forward_validations: int,
    top_n: int,
    grid_mode: str = "fast",
) -> str:
    symbol = str(row.get("symbol") or "").strip() or "TRXUSDT"
    side = str(row.get("target_side") or "").strip().upper()
    interval = str(row.get("target_interval") or "").strip()
    parts = [
        "openclaw-quantctl risk-combo-sweep",
        f"--symbols {symbol}",
    ]
    if side in {"BUY", "SELL"}:
        parts.append(f"--target-side {side}")
    if interval:
        parts.append(f"--target-interval {interval}")
    parts.extend(
        [
            f"--limit {limit}",
            f"--grid-mode {grid_mode}",
            f"--min-test-trades {min_test_trades}",
            "--target-profit-factor 1.0",
            "--min-expectancy-r 0.0",
            "--max-stop-loss-ratio 55",
            f"--max-configs {max_configs}",
            f"--max-walk-forward-validations {max_walk_forward_validations}",
            f"--top-n {top_n}",
            "--skip-news",
            "--compact",
        ]
    )
    return " ".join(parts)


def _validation_command_for_symbols(
    *,
    symbols: tuple[str, ...],
    side: str,
    interval: str,
    limit: int,
    min_test_trades: int,
    max_configs: int,
    max_walk_forward_validations: int,
    top_n: int,
    grid_mode: str = "fast",
) -> str:
    row = {
        "symbol": ",".join(symbols),
        "target_side": side,
        "target_interval": interval,
    }
    return _validation_command_for_surface(
        row,
        limit=limit,
        min_test_trades=min_test_trades,
        max_configs=max_configs,
        max_walk_forward_validations=max_walk_forward_validations,
        top_n=top_n,
        grid_mode=grid_mode,
    )


def _route_repair_symbols(route_id: str) -> tuple[str, ...]:
    if not route_id:
        return ()
    routes = load_asset_routes()
    raw = routes.get(route_id)
    if not isinstance(raw, dict):
        return ()
    return tuple(
        dict.fromkeys(
            normalize_symbol(str(item))
            for item in (raw.get("symbols") or [])
            if str(item).strip() and str(item).upper().endswith("USDT")
        )
    )


def _repair_symbol_basket(best_surface: dict[str, Any]) -> tuple[str, ...]:
    side = str(best_surface.get("target_side") or "").upper()
    interval = str(best_surface.get("target_interval") or "")
    symbol = str(best_surface.get("symbol") or "TRXUSDT").upper()
    route_id = str(best_surface.get("route_id") or "").strip()
    if side == "SELL" or interval.endswith("m"):
        route_symbols = _route_repair_symbols(route_id)
        basket = (
            symbol,
            *route_symbols[:12],
            "TRXUSDT",
            "ADAUSDT",
            "AVAXUSDT",
            "NEARUSDT",
            "DOGEUSDT",
            "WIFUSDT",
        )
        return tuple(dict.fromkeys(item for item in basket if item))
    return (symbol,)


def _filter_recent_failed_symbols(
    symbols: tuple[str, ...],
    *,
    side: str,
    interval: str,
    failed_identities: set[tuple[str, str, str, str]],
) -> tuple[str, ...]:
    if not failed_identities:
        return symbols
    side_value = str(side or "").upper()
    interval_value = str(interval or "").lower()
    filtered = []
    excluded = {
        symbol
        for failed_side, failed_interval, _route_id, symbol in failed_identities
        if failed_side == side_value and failed_interval == interval_value
    }
    for symbol in symbols:
        normalized = normalize_symbol(symbol)
        if normalized not in excluded:
            filtered.append(normalized)
    return tuple(dict.fromkeys(filtered))


def _repair_interval_candidates(best_surface: dict[str, Any], *, coverage_type: str, coverage_key: str) -> tuple[str, ...]:
    side = str(best_surface.get("target_side") or "").upper()
    current_interval = str(best_surface.get("target_interval") or "").strip()
    if side == "SELL" and coverage_type == "side":
        candidates = ("15m", "30m", "1h", "4h")
    elif coverage_type == "horizon" and coverage_key == "short":
        candidates = ("15m", "30m")
    elif coverage_type == "horizon" and coverage_key == "medium":
        candidates = ("1h", "4h")
    elif coverage_type == "horizon" and coverage_key == "long":
        candidates = ("1d",)
    elif current_interval.endswith("m"):
        candidates = ("15m", "30m")
    else:
        candidates = (current_interval,) if current_interval else ()
    return tuple(dict.fromkeys(item for item in candidates if item and item != current_interval))


def _multi_interval_scout_commands(
    best_surface: dict[str, Any],
    *,
    coverage_type: str,
    coverage_key: str,
    symbols: tuple[str, ...],
    failed_identities: set[tuple[str, str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    side = str(best_surface.get("target_side") or "").upper()
    if side not in {"BUY", "SELL"}:
        return []
    commands: list[dict[str, Any]] = []
    for interval in _repair_interval_candidates(
        best_surface,
        coverage_type=coverage_type,
        coverage_key=coverage_key,
    ):
        interval_symbols = _filter_recent_failed_symbols(
            symbols,
            side=side,
            interval=interval,
            failed_identities=failed_identities or set(),
        )
        if not interval_symbols:
            continue
        commands.append(
            {
                "target_interval": interval,
                "horizon": _interval_horizon(interval),
                "command": _validation_command_for_symbols(
                    symbols=interval_symbols,
                    side=side,
                    interval=interval,
                    limit=600,
                    min_test_trades=10,
                    max_configs=6,
                    max_walk_forward_validations=1,
                    top_n=10,
                    grid_mode="fast",
                ),
                "purpose": "scout_adjacent_timeframe_without_relaxing_gates",
                "excluded_recent_failed_symbols": [
                    symbol for symbol in symbols if normalize_symbol(symbol) not in set(interval_symbols)
                ],
            }
        )
    return commands


def _validation_plan_for_surfaces(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for row in rows:
        if row.get("research_status") not in PROMISING_RESEARCH_STATUSES | {EMERGING_RESEARCH_STATUS}:
            continue
        lead_only = row.get("research_status") == EMERGING_RESEARCH_STATUS
        plan.append(
            {
                "surface": row.get("surface"),
                "symbol": row.get("symbol"),
                "target_side": row.get("target_side"),
                "target_interval": row.get("target_interval"),
                "research_status": row.get("research_status"),
                "promotion_eligible": bool(row.get("promotion_eligible")),
                "purpose": (
                    "sample_expansion_for_emerging_positive_lead"
                    if lead_only
                    else "interactive_probe_then_offline_validation_before_promotion"
                ),
                "interactive_probe_command": _validation_command_for_surface(
                    row,
                    limit=900,
                    min_test_trades=10,
                    max_configs=8,
                    max_walk_forward_validations=1,
                    top_n=5,
                    grid_mode="fast",
                ),
                "offline_validation_command": _validation_command_for_surface(
                    row,
                    limit=5000,
                    min_test_trades=30,
                    max_configs=40,
                    max_walk_forward_validations=6,
                    top_n=10,
                    grid_mode="focused",
                ),
                "runtime_guidance": (
                    "Run this as research-only sample expansion; do not count it as a candidate until normal "
                    "test-trade, recovery, and robust gates pass."
                    if lead_only
                    else "Run interactive_probe_command during chat only. Run offline_validation_command as a scheduled "
                    "or background research job and use the resulting JSON report for promotion review."
                ),
                "promotion_boundary": (
                    "not_promotion_eligible_until_sample_and_robust_gates_pass"
                    if lead_only
                    else "research_only_does_not_change_live_readiness_or_mainnet_permission"
                ),
            }
        )
    return plan


def _surface_failure_reasons(row: dict[str, Any]) -> list[str]:
    gate_reasons = row.get("gate_reasons") if isinstance(row.get("gate_reasons"), dict) else {}
    reasons = list(gate_reasons.get("robust") or gate_reasons.get("recovery") or [])
    return [str(item) for item in reasons if str(item)]


def _research_risk_boundary() -> dict[str, Any]:
    return {
        "max_per_trade_risk_pct": 0.025,
        "max_per_trade_risk_percent": 2.5,
        "risk_ceiling_source": "user_policy_and_strategy_stable_risk",
        "applies_to": "all_research_candidates_before_any_promotion",
        "changes_position_sizing": False,
        "opens_orders": False,
        "writes_execution_config": False,
        "mainnet_live_allowed": False,
    }


def _coverage_repair_plan(
    *,
    side_summary: dict[str, Any],
    horizon_summary: dict[str, Any],
    failed_identities: set[tuple[str, str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    failed_identities = failed_identities or set()
    coverage_items = [
        ("side", name, summary)
        for name, summary in side_summary.items()
        if isinstance(summary, dict)
    ] + [
        ("horizon", name, summary)
        for name, summary in horizon_summary.items()
        if isinstance(summary, dict)
    ]
    for coverage_type, name, summary in coverage_items:
        if int(summary.get("promising_surface_count") or 0) > 0:
            continue
        best_surface = (
            summary.get("best_emerging_surface")
            if isinstance(summary.get("best_emerging_surface"), dict)
            else summary.get("best_surface")
            if isinstance(summary.get("best_surface"), dict)
            else {}
        )
        if not best_surface:
            continue
        surface_name = str(best_surface.get("surface") or f"{coverage_type}_{name}")
        dedupe_key = (coverage_type, surface_name)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        scout_command = _validation_command_for_surface(
            best_surface,
            limit=600,
            min_test_trades=10,
            max_configs=8,
            max_walk_forward_validations=1,
            top_n=5,
            grid_mode="fast",
        )
        base_symbol_basket = _repair_symbol_basket(best_surface)
        symbol_basket = _filter_recent_failed_symbols(
            base_symbol_basket,
            side=str(best_surface.get("target_side") or ""),
            interval=str(best_surface.get("target_interval") or ""),
            failed_identities=failed_identities,
        )
        if not symbol_basket:
            symbol_basket = base_symbol_basket[:1]
        excluded_recent_failed_symbols = [
            symbol for symbol in base_symbol_basket if normalize_symbol(symbol) not in set(symbol_basket)
        ]
        cross_symbol_scout_command = _validation_command_for_symbols(
            symbols=symbol_basket,
            side=str(best_surface.get("target_side") or ""),
            interval=str(best_surface.get("target_interval") or ""),
            limit=600,
            min_test_trades=10,
            max_configs=6,
            max_walk_forward_validations=1,
            top_n=10,
            grid_mode="fast",
        )
        multi_interval_scout_commands = _multi_interval_scout_commands(
            best_surface,
            coverage_type=coverage_type,
            coverage_key=name,
            symbols=symbol_basket,
            failed_identities=failed_identities,
        )
        interactive_command = _validation_command_for_surface(
            best_surface,
            limit=900,
            min_test_trades=10,
            max_configs=8,
            max_walk_forward_validations=1,
            top_n=5,
            grid_mode="fast",
        )
        offline_command = _validation_command_for_surface(
            best_surface,
            limit=5000,
            min_test_trades=30,
            max_configs=40,
            max_walk_forward_validations=6,
            top_n=10,
            grid_mode="focused",
        )
        plan.append(
            {
                "coverage_type": coverage_type,
                "coverage_key": name,
                "status": summary.get("status") or "missing_or_negative_expectancy",
                "research_status": best_surface.get("research_status"),
                "promotion_eligible": bool(best_surface.get("promotion_eligible")),
                "source_surface": surface_name,
                "target_side": best_surface.get("target_side"),
                "target_interval": best_surface.get("target_interval"),
                "current_metrics": best_surface.get("full") or {},
                "failure_reasons": _surface_failure_reasons(best_surface),
                "repair_objective": "find_positive_expectancy_without_relaxing_recovery_or_live_gates",
                "scout_command": scout_command,
                "cross_symbol_scout_command": cross_symbol_scout_command,
                "cross_symbol_scout_symbols": list(symbol_basket),
                "excluded_recent_failed_symbols": excluded_recent_failed_symbols,
                "multi_interval_scout_commands": multi_interval_scout_commands,
                "multi_interval_scout_intervals": [
                    str(item.get("target_interval"))
                    for item in multi_interval_scout_commands
                    if item.get("target_interval")
                ],
                "interactive_probe_command": interactive_command,
                "offline_validation_command": offline_command,
                "runtime_guidance": (
                    "Run scout_command first during chat. Use cross_symbol_scout_command as scheduled scout work "
                    "to avoid overfitting one failed symbol. Use multi_interval_scout_commands when one timeframe "
                    "keeps failing. Keep heavier commands as scheduled research."
                ),
                "guardrails": {
                    "does_not_open_orders": True,
                    "does_not_write_execution_config": True,
                    "does_not_clear_route_quarantine": True,
                    "does_not_lower_promotion_gates": True,
                    "max_per_trade_risk_pct": 0.025,
                    "max_per_trade_risk_percent": 2.5,
                    "mainnet_live_allowed": False,
                },
            }
        )
    return plan


def _completion_audit(
    *,
    rows: list[dict[str, Any]],
    side_summary: dict[str, Any],
    horizon_summary: dict[str, Any],
    promising_surface_count: int,
    robust_surface_count: int,
    repair_plan: list[dict[str, Any]],
    risk_boundary: dict[str, Any],
) -> dict[str, Any]:
    buy_promising = int((side_summary.get("buy") or {}).get("promising_surface_count") or 0)
    sell_promising = int((side_summary.get("sell") or {}).get("promising_surface_count") or 0)
    short_promising = int((horizon_summary.get("short") or {}).get("promising_surface_count") or 0)
    medium_promising = int((horizon_summary.get("medium") or {}).get("promising_surface_count") or 0)
    long_promising = int((horizon_summary.get("long") or {}).get("promising_surface_count") or 0)
    safety_ok = True
    risk_boundary_ok = (
        float(risk_boundary.get("max_per_trade_risk_pct") or 0.0) <= 0.025
        and risk_boundary.get("opens_orders") is False
        and risk_boundary.get("writes_execution_config") is False
        and risk_boundary.get("mainnet_live_allowed") is False
    )
    checks = [
        {
            "requirement": "candidate_signal_not_zero",
            "passed": promising_surface_count > 0,
            "evidence": {"promising_surface_count": promising_surface_count, "surface_count": len(rows)},
        },
        {
            "requirement": "buy_and_sell_have_backtested_promising_surfaces",
            "passed": buy_promising > 0 and sell_promising > 0,
            "evidence": {"buy_promising": buy_promising, "sell_promising": sell_promising},
        },
        {
            "requirement": "short_medium_long_have_promising_surfaces",
            "passed": short_promising > 0 and medium_promising > 0 and long_promising > 0,
            "evidence": {
                "short_promising": short_promising,
                "medium_promising": medium_promising,
                "long_promising": long_promising,
            },
        },
        {
            "requirement": "positive_expectancy_improved_in_research",
            "passed": promising_surface_count > 0,
            "evidence": {
                "best_promising_surfaces": [
                    {
                        "surface": row.get("surface"),
                        "full": row.get("full"),
                        "test": row.get("test"),
                        "walk_forward": row.get("walk_forward"),
                    }
                    for row in rows
                    if row.get("research_status")
                    in {
                        "robust_recovery_candidate",
                        "recovery_candidate_needs_robust_validation",
                        "promising_but_under_validated",
                    }
                ][:3],
            },
        },
        {
            "requirement": "robust_promotion_gate_passed",
            "passed": robust_surface_count > 0,
            "evidence": {"robust_surface_count": robust_surface_count},
        },
        {
            "requirement": "mainnet_live_blocked",
            "passed": safety_ok,
            "evidence": {
                "mainnet_live_allowed": False,
                "opens_orders": False,
                "writes_execution_config": False,
            },
        },
        {
            "requirement": "per_trade_risk_ceiling_preserved",
            "passed": risk_boundary_ok,
            "evidence": {
                "max_per_trade_risk_pct": risk_boundary.get("max_per_trade_risk_pct"),
                "max_per_trade_risk_percent": risk_boundary.get("max_per_trade_risk_percent"),
                "opens_orders": risk_boundary.get("opens_orders"),
                "writes_execution_config": risk_boundary.get("writes_execution_config"),
                "mainnet_live_allowed": risk_boundary.get("mainnet_live_allowed"),
            },
        },
    ]
    missing = [item["requirement"] for item in checks if not item["passed"]]
    return {
        "objective": "stable_auditable_buy_sell_research_candidates_with_positive_expectancy_and_blocked_live",
        "status": "achieved" if not missing else "incomplete",
        "checks": checks,
        "missing_requirements": missing,
        "completion_blockers": missing,
        "repair_plan_available": bool(repair_plan),
        "mainnet_live_allowed": False,
    }


def _objective_scorecard(
    *,
    rows: list[dict[str, Any]],
    side_summary: dict[str, Any],
    horizon_summary: dict[str, Any],
    promising_surface_count: int,
    robust_surface_count: int,
    risk_boundary: dict[str, Any],
) -> dict[str, Any]:
    buy_promising = int((side_summary.get("buy") or {}).get("promising_surface_count") or 0)
    sell_promising = int((side_summary.get("sell") or {}).get("promising_surface_count") or 0)
    short_promising = int((horizon_summary.get("short") or {}).get("promising_surface_count") or 0)
    medium_promising = int((horizon_summary.get("medium") or {}).get("promising_surface_count") or 0)
    long_promising = int((horizon_summary.get("long") or {}).get("promising_surface_count") or 0)
    promising_rows = [
        row
        for row in rows
        if row.get("research_status") in PROMISING_RESEARCH_STATUSES
    ]
    best_pf = max((_metric_float((row.get("full") or {}).get("profit_factor")) for row in promising_rows), default=0.0)
    best_expectancy = max((_metric_float((row.get("full") or {}).get("expectancy_r")) for row in promising_rows), default=0.0)
    best_drawdown = min(
        (_metric_float((row.get("full") or {}).get("max_drawdown_pct")) for row in promising_rows),
        default=0.0,
    )
    best_stop_loss_ratio = min(
        (_metric_float((row.get("full") or {}).get("stop_loss_ratio")) for row in promising_rows),
        default=0.0,
    )
    buy_sell_score = min(
        100,
        45
        + (10 if promising_surface_count > 0 else 0)
        + (4 if buy_promising > 0 else 0)
        + (10 if sell_promising > 0 else 0)
        + (3 if medium_promising > 0 else 0)
        + (4 if long_promising > 0 else 0)
        + (4 if short_promising > 0 else 0)
        + (16 if robust_surface_count > 0 else 0),
    )
    expectancy_score = min(
        100,
        20
        + (6 if best_pf >= 1.0 else 0)
        + (4 if best_pf >= 2.0 else 0)
        + (4 if best_expectancy > 0 else 0)
        + (1 if best_drawdown and best_drawdown <= 2.0 else 0)
        + (1 if best_stop_loss_ratio and best_stop_loss_ratio <= 55.0 else 0)
        + (20 if robust_surface_count > 0 else 0),
    )
    readiness_score = 0 if risk_boundary.get("mainnet_live_allowed") is False else 1
    return {
        "buy_sell_stability": {
            "score": buy_sell_score,
            "baseline_score": 45,
            "target": "stable_auditable_buy_sell_candidates",
            "evidence": {
                "surface_count": len(rows),
                "promising_surface_count": promising_surface_count,
                "buy_promising": buy_promising,
                "sell_promising": sell_promising,
                "short_promising": short_promising,
                "medium_promising": medium_promising,
                "long_promising": long_promising,
                "robust_surface_count": robust_surface_count,
            },
            "blockers": [
                item
                for item, count in {
                    "sell_promising_surface_missing": sell_promising,
                    "short_promising_surface_missing": short_promising,
                    "robust_surface_missing": robust_surface_count,
                }.items()
                if count <= 0
            ],
        },
        "long_term_expectancy": {
            "score": expectancy_score,
            "baseline_score": 20,
            "target": "improve_pf_avg_r_drawdown_and_stop_loss_ratio_not_win_rate",
            "evidence": {
                "best_promising_full_profit_factor": round(best_pf, 4),
                "best_promising_full_expectancy_r": round(best_expectancy, 4),
                "best_promising_max_drawdown_pct": round(best_drawdown, 4),
                "best_promising_stop_loss_ratio": round(best_stop_loss_ratio, 4),
                "robust_surface_count": robust_surface_count,
            },
            "blockers": ["robust_surface_missing"] if robust_surface_count <= 0 else [],
        },
        "live_readiness": {
            "score": readiness_score,
            "required_score_until_promotion": 0,
            "target": "remain_blocked_until_research_sample_dry_run_and_testnet_gates_pass",
            "evidence": {
                "mainnet_live_allowed": False,
                "opens_orders": False,
                "writes_execution_config": False,
                "max_per_trade_risk_pct": risk_boundary.get("max_per_trade_risk_pct"),
            },
        },
    }


def _prompt_to_artifact_checklist(
    *,
    side_summary: dict[str, Any],
    horizon_summary: dict[str, Any],
    completion_audit: dict[str, Any],
    objective_scorecard: dict[str, Any],
    risk_boundary: dict[str, Any],
    repair_plan: list[dict[str, Any]],
    input_report_count: int,
    skipped_input_report_count: int,
) -> dict[str, Any]:
    buy_sell_score = (objective_scorecard.get("buy_sell_stability") or {}).get("score")
    expectancy_score = (objective_scorecard.get("long_term_expectancy") or {}).get("score")
    readiness_score = (objective_scorecard.get("live_readiness") or {}).get("score")
    buy_promising = int((side_summary.get("buy") or {}).get("promising_surface_count") or 0)
    sell_promising = int((side_summary.get("sell") or {}).get("promising_surface_count") or 0)
    sell_emerging = int((side_summary.get("sell") or {}).get("emerging_positive_lead_count") or 0)
    short_promising = int((horizon_summary.get("short") or {}).get("promising_surface_count") or 0)
    short_emerging = int((horizon_summary.get("short") or {}).get("emerging_positive_lead_count") or 0)
    medium_promising = int((horizon_summary.get("medium") or {}).get("promising_surface_count") or 0)
    long_promising = int((horizon_summary.get("long") or {}).get("promising_surface_count") or 0)
    audit_checks = {
        str(item.get("requirement")): bool(item.get("passed"))
        for item in completion_audit.get("checks", [])
        if isinstance(item, dict)
    }
    items = [
        {
            "requirement": "candidate_not_zero",
            "artifact": "risk_combo_matrix.promising_surface_count",
            "passed": audit_checks.get("candidate_signal_not_zero", False),
            "evidence": (completion_audit.get("checks") or [])[0].get("evidence") if completion_audit.get("checks") else {},
        },
        {
            "requirement": "buy_and_sell_directional_evidence",
            "artifact": "risk_combo_matrix.side_summary",
            "passed": buy_promising > 0 and sell_promising > 0,
            "evidence": {
                "buy_promising": buy_promising,
                "sell_promising": sell_promising,
                "sell_emerging_positive_leads": sell_emerging,
                "emerging_leads_are_not_promotion_candidates": True,
            },
        },
        {
            "requirement": "short_medium_long_horizon_coverage",
            "artifact": "risk_combo_matrix.horizon_summary",
            "passed": short_promising > 0 and medium_promising > 0 and long_promising > 0,
            "evidence": {
                "short_promising": short_promising,
                "short_emerging_positive_leads": short_emerging,
                "medium_promising": medium_promising,
                "long_promising": long_promising,
                "emerging_leads_are_not_promotion_candidates": True,
            },
        },
        {
            "requirement": "positive_expectancy_metrics_not_win_rate",
            "artifact": "risk_combo_matrix.objective_scorecard.long_term_expectancy",
            "passed": audit_checks.get("positive_expectancy_improved_in_research", False),
            "evidence": (objective_scorecard.get("long_term_expectancy") or {}).get("evidence", {}),
        },
        {
            "requirement": "robust_promotion_gate_passed",
            "artifact": "risk_combo_matrix.completion_audit",
            "passed": audit_checks.get("robust_promotion_gate_passed", False),
            "evidence": {"robust_surface_count": (objective_scorecard.get("buy_sell_stability") or {}).get("evidence", {}).get("robust_surface_count")},
        },
        {
            "requirement": "readiness_stays_zero_and_mainnet_blocked",
            "artifact": "risk_combo_matrix.objective_scorecard.live_readiness",
            "passed": readiness_score == 0 and audit_checks.get("mainnet_live_blocked", False),
            "evidence": {
                "readiness_score": readiness_score,
                "mainnet_live_allowed": False,
                "opens_orders": False,
                "writes_execution_config": False,
            },
        },
        {
            "requirement": "per_trade_risk_ceiling_2_5pct",
            "artifact": "risk_combo_matrix.risk_boundary",
            "passed": audit_checks.get("per_trade_risk_ceiling_preserved", False),
            "evidence": {
                "max_per_trade_risk_pct": risk_boundary.get("max_per_trade_risk_pct"),
                "max_per_trade_risk_percent": risk_boundary.get("max_per_trade_risk_percent"),
            },
        },
        {
            "requirement": "no_gate_relaxation_for_forced_trades",
            "artifact": "negative_surface_repair_plan.guardrails",
            "passed": bool(repair_plan)
            and all((item.get("guardrails") or {}).get("does_not_lower_promotion_gates") is True for item in repair_plan),
            "evidence": {
                "repair_plan_count": len(repair_plan),
                "guardrail": "does_not_lower_promotion_gates",
            },
        },
        {
            "requirement": "data_failures_do_not_pollute_backtest_surfaces",
            "artifact": "skipped_input_reports",
            "passed": input_report_count >= skipped_input_report_count,
            "evidence": {
                "input_report_count": input_report_count,
                "skipped_input_report_count": skipped_input_report_count,
            },
        },
        {
            "requirement": "score_improvement_visible",
            "artifact": "risk_combo_matrix.objective_scorecard",
            "passed": buy_sell_score is not None and expectancy_score is not None and readiness_score == 0,
            "evidence": {
                "buy_sell_stability_score": buy_sell_score,
                "long_term_expectancy_score": expectancy_score,
                "live_readiness_score": readiness_score,
            },
        },
    ]
    missing = [item["requirement"] for item in items if not item["passed"]]
    return {
        "status": "complete" if not missing else "incomplete",
        "items": items,
        "missing_requirements": missing,
        "completion_blockers": completion_audit.get("completion_blockers") or missing,
        "mainnet_live_allowed": False,
    }


def build_risk_combo_matrix_report(
    *,
    report_paths: list[str | Path],
    output_dir: str | Path | None = None,
    latest_sweeps: int = 0,
) -> dict[str, Any]:
    if latest_sweeps > 0:
        latest_paths = sorted(
            RISK_COMBO_SWEEP_DIR.glob("*-risk-combo-sweep.json") if RISK_COMBO_SWEEP_DIR.exists() else [],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )[:latest_sweeps]
        combined_paths: list[str | Path] = [*latest_paths, *report_paths]
        report_paths = list(dict.fromkeys(combined_paths))
    if not report_paths:
        raise ValueError("At least one risk-combo-sweep report path is required, or pass latest_sweeps > 0.")
    ensure_runtime_dirs()
    root = Path(output_dir).expanduser().resolve() if output_dir else RISK_COMBO_MATRIX_DIR
    root.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []
    loaded_reports = [_load_sweep_report(path) for path in report_paths]
    skipped_reports: list[dict[str, Any]] = []
    for payload in loaded_reports:
        skipped = _skipped_matrix_input(payload)
        if skipped:
            skipped_reports.append(skipped)
            continue
        raw_rows.extend(_sweep_surface_rows(payload))
    active_rows, superseded_rows = _active_and_superseded_research_rows(raw_rows)
    failed_identities = _recent_failed_repair_identities(
        active_rows=active_rows,
        superseded_rows=superseded_rows,
    )
    rows_by_surface: dict[str, dict[str, Any]] = {}
    for row in active_rows:
        surface = str(row.get("surface") or "unknown")
        existing_surface = rows_by_surface.get(surface)
        if existing_surface is None or (
            _research_status_rank(str(row.get("research_status") or "")),
            tuple(row.get("rank_key") or ()),
        ) > (
            _research_status_rank(str(existing_surface.get("research_status") or "")),
            tuple(existing_surface.get("rank_key") or ()),
        ):
            rows_by_surface[surface] = row
    rows = sorted(rows_by_surface.values(), key=lambda item: tuple(item["rank_key"]), reverse=True)
    research_leads = sorted(active_rows, key=lambda item: tuple(item["rank_key"]), reverse=True)
    superseded_research_leads = sorted(superseded_rows, key=lambda item: tuple(item["rank_key"]), reverse=True)
    for row in rows:
        row.pop("rank_key", None)
    for row in research_leads:
        row.pop("rank_key", None)
    for row in superseded_research_leads:
        row.pop("rank_key", None)
    promising = [
        row
        for row in rows
        if row.get("research_status") in PROMISING_RESEARCH_STATUSES
    ]
    emerging = [row for row in research_leads if row.get("research_status") == EMERGING_RESEARCH_STATUS]
    superseded_emerging = [
        row for row in superseded_research_leads if row.get("research_status") == EMERGING_RESEARCH_STATUS
    ]
    validation_plan = _validation_plan_for_surfaces(rows)
    side_summary = _coverage_summary(rows, key="target_side", expected=("BUY", "SELL"))
    horizon_summary = _horizon_coverage_summary(rows)
    repair_plan = _coverage_repair_plan(
        side_summary=side_summary,
        horizon_summary=horizon_summary,
        failed_identities=failed_identities,
    )
    robust_surface_count = sum(1 for row in rows if row.get("robust_recovery_gate_passed"))
    risk_boundary = _research_risk_boundary()
    completion_audit = _completion_audit(
        rows=rows,
        side_summary=side_summary,
        horizon_summary=horizon_summary,
        promising_surface_count=len(promising),
        robust_surface_count=robust_surface_count,
        repair_plan=repair_plan,
        risk_boundary=risk_boundary,
    )
    objective_scorecard = _objective_scorecard(
        rows=rows,
        side_summary=side_summary,
        horizon_summary=horizon_summary,
        promising_surface_count=len(promising),
        robust_surface_count=robust_surface_count,
        risk_boundary=risk_boundary,
    )
    prompt_to_artifact_checklist = _prompt_to_artifact_checklist(
        side_summary=side_summary,
        horizon_summary=horizon_summary,
        completion_audit=completion_audit,
        objective_scorecard=objective_scorecard,
        risk_boundary=risk_boundary,
        repair_plan=repair_plan,
        input_report_count=len(loaded_reports),
        skipped_input_report_count=len(skipped_reports),
    )
    payload = {
        "generated_at": _utc_now().isoformat(),
        "mode": "risk_combo_side_interval_matrix_v1",
        "safety": {
            "opens_orders": False,
            "writes_execution_config": False,
            "clears_route_quarantine": False,
            "mainnet_live_allowed": False,
        },
        "input_report_count": len(loaded_reports),
        "skipped_input_report_count": len(skipped_reports),
        "skipped_input_reports": skipped_reports,
        "surface_count": len(rows),
        "promising_surface_count": len(promising),
        "emerging_positive_lead_count": len(emerging),
        "superseded_emerging_positive_lead_count": len(superseded_emerging),
        "recent_failed_repair_identity_count": len(failed_identities),
        "robust_surface_count": robust_surface_count,
        "side_summary": side_summary,
        "horizon_summary": horizon_summary,
        "completion_audit": completion_audit,
        "objective_scorecard": objective_scorecard,
        "prompt_to_artifact_checklist": prompt_to_artifact_checklist,
        "risk_boundary": risk_boundary,
        "surfaces": rows,
        "research_leads": research_leads[:50],
        "emerging_positive_leads": emerging[:20],
        "superseded_research_leads": superseded_research_leads[:50],
        "superseded_emerging_positive_leads": superseded_emerging[:20],
        "best_surface": rows[0] if rows else None,
        "validation_plan": validation_plan,
        "negative_surface_repair_plan": repair_plan,
        "next_research_actions": [
            "expand_sample_and_walk_forward_for_promising_surfaces" if promising else "repair_or_reject_negative_surfaces",
            "repair_missing_sell_or_short_surfaces" if repair_plan else "coverage_repair_plan_not_needed",
            "keep_mainnet_blocked_until_robust_gate_passes",
        ],
        "promotion_boundary": {
            "requires_robust_recovery_gate": True,
            "requires_sufficient_test_trades": True,
            "max_per_trade_risk_pct": 0.025,
            "max_per_trade_risk_percent": 2.5,
            "mainnet_live_allowed": False,
        },
    }
    report_path = root / f"{_stamp()}-risk-combo-matrix.json"
    payload["report_path"] = str(report_path)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def _build_route_plan(
    routes: tuple[str, ...],
    symbols: tuple[str, ...],
    *,
    include_all_route_symbols: bool,
    max_symbols_per_route: int,
) -> dict[str, tuple[str, ...]]:
    if symbols:
        grouped = _group_symbols_by_route(symbols)
        return {route_id: tuple(items) for route_id, items in grouped.items()}
    return {
        route_id: _symbols_for_route(
            route_id,
            include_all_route_symbols=include_all_route_symbols,
            max_symbols_per_route=max_symbols_per_route,
        )
        for route_id in routes
    }


def run_risk_combo_sweep(
    *,
    routes: list[str] | None = None,
    symbols: list[str] | None = None,
    limit: int = 1500,
    grid_mode: str = "fast",
    target_side: str = "",
    target_interval: str = "",
    target_profit_factor: float = 0.8,
    min_test_trades: int = 3,
    min_win_rate: float = 0.0,
    max_stop_loss_ratio: float = 100.0,
    min_expectancy_r: float = 0.0,
    min_payoff_ratio: float = 0.0,
    max_symbols_per_route: int = 0,
    max_configs: int = 0,
    max_walk_forward_validations: int = 0,
    include_all_route_symbols: bool = False,
    skip_news: bool = False,
    top_n: int = 50,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    ensure_runtime_dirs()
    RISK_COMBO_SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    normalized_target_side = str(target_side or "").strip().upper()
    if normalized_target_side not in {"", "BUY", "SELL"}:
        raise ValueError("target_side must be empty, BUY, or SELL")
    normalized_target_interval = str(target_interval or "").strip()
    config = RiskComboSweepConfig(
        routes=tuple(routes or _active_quarantined_routes()),
        symbols=tuple(normalize_symbol(item) for item in (symbols or []) if item),
        limit=min(max(int(limit), MIN_WINDOW_CANDLES), MAX_KLINE_LIMIT),
        grid_mode=(
            "standard"
            if str(grid_mode).strip().lower() == "standard"
            else "focused"
            if str(grid_mode).strip().lower() == "focused"
            else "fast"
        ),
        target_side=normalized_target_side,
        target_interval=normalized_target_interval,
        target_profit_factor=float(target_profit_factor),
        min_test_trades=max(int(min_test_trades), 1),
        min_win_rate=max(float(min_win_rate), 0.0),
        max_stop_loss_ratio=min(max(float(max_stop_loss_ratio), 0.0), 100.0),
        min_expectancy_r=max(float(min_expectancy_r), 0.0),
        min_payoff_ratio=max(float(min_payoff_ratio), 0.0),
        max_symbols_per_route=max(int(max_symbols_per_route), 0),
        max_configs=max(int(max_configs), 0),
        max_walk_forward_validations=max(int(max_walk_forward_validations), 0),
        include_all_route_symbols=bool(include_all_route_symbols),
        skip_news=bool(skip_news),
        top_n=max(int(top_n), 1),
    )
    setup_finished_at = time.perf_counter()
    route_plan = _build_route_plan(
        config.routes,
        config.symbols,
        include_all_route_symbols=config.include_all_route_symbols,
        max_symbols_per_route=config.max_symbols_per_route,
    )
    news = _collect_news(config.skip_news)
    historical_signal_index = build_historical_signal_risk_index()
    route_side_evaluations_by_route = {
        route_id: {
            side: evaluate_route_side_risk(route_id=route_id, side=side)
            for side in ("BUY", "SELL")
        }
        for route_id in route_plan
    }
    datasets: list[SweepDataset] = []
    datasets_by_key: dict[tuple[str, str, str, str, str], SweepDataset] = {}
    fetch_log: list[dict[str, Any]] = []
    dataset_errors: list[dict[str, Any]] = []
    for route_id, route_symbols in route_plan.items():
        for symbol in route_symbols:
            try:
                dataset = fetch_dataset(
                    settings,
                    route_id=route_id,
                    requested_symbol=symbol,
                    limit=config.limit,
                    interval=config.target_interval,
                )
                datasets.append(dataset)
                datasets_by_key[_dataset_key(dataset)] = dataset
                fetch_log.extend(dataset.fetch_log)
            except Exception as exc:
                dataset_errors.append({"route_id": route_id, "symbol": symbol, "error": str(exc)})
    fetch_finished_at = time.perf_counter()

    results: list[dict[str, Any]] = []
    grid_by_profile: dict[str, dict[str, tuple[Any, ...]]] = {}
    for dataset in datasets:
        grid = _grid_values(dataset.strategy, mode=config.grid_mode)
        grid_by_profile[dataset.strategy.profile] = grid
        for (
            min_adx,
            min_convergence,
            atr_stop_multiple,
            primary_tp_multiple,
            exit_profile,
            news_veto_mode,
            side_policy_mode,
            structure_policy_mode,
            historical_policy_mode,
            route_side_policy_mode,
        ) in _bounded_grid_combinations(
            {**grid, "route_side_policy_mode": ROUTE_SIDE_POLICY_MODES},
            config.max_configs,
        ):
            results.append(
                _evaluate_combo(
                    dataset,
                    min_adx=float(min_adx),
                    min_convergence=float(min_convergence),
                    atr_stop_multiple=float(atr_stop_multiple),
                    primary_tp_multiple=float(primary_tp_multiple),
                    exit_profile=str(exit_profile),
                    news_veto_mode=str(news_veto_mode),
                    target_side=config.target_side,
                    side_policy_mode=str(side_policy_mode),
                    structure_policy_mode=str(structure_policy_mode),
                    historical_policy_mode=str(historical_policy_mode),
                    historical_signal_index=historical_signal_index,
                    route_side_policy_mode=str(route_side_policy_mode),
                    route_side_evaluations=route_side_evaluations_by_route.get(dataset.route.route_id, {}),
                    news_risk=news["risk"],
                    target_profit_factor=config.target_profit_factor,
                    min_test_trades=config.min_test_trades,
                    min_win_rate=config.min_win_rate,
                    max_stop_loss_ratio=config.max_stop_loss_ratio,
                    min_expectancy_r=config.min_expectancy_r,
                    min_payoff_ratio=config.min_payoff_ratio,
                    include_walk_forward=False,
                )
            )
    grid_finished_at = time.perf_counter()

    ranked = sorted(results, key=_rank_key, reverse=True)
    walk_forward_budget = config.max_walk_forward_validations or max(config.top_n, 20)
    preliminary_validation_rows: list[dict[str, Any]] = []
    for row in ranked:
        if len(preliminary_validation_rows) >= min(config.top_n, walk_forward_budget):
            break
        preliminary_validation_rows.append(row)
    for row in ranked:
        if len(preliminary_validation_rows) >= walk_forward_budget:
            break
        if (row.get("recovery_gate") or {}).get("passed") and row not in preliminary_validation_rows:
            preliminary_validation_rows.append(row)
    for row in preliminary_validation_rows:
        _attach_walk_forward(
            row,
            datasets=datasets_by_key,
            news_risk=news["risk"],
            historical_signal_index=historical_signal_index,
            route_side_evaluations=route_side_evaluations_by_route.get(str(row.get("route_id") or ""), {}),
            target_profit_factor=config.target_profit_factor,
            min_test_trades=config.min_test_trades,
            min_win_rate=config.min_win_rate,
            max_stop_loss_ratio=config.max_stop_loss_ratio,
            min_expectancy_r=config.min_expectancy_r,
            min_payoff_ratio=config.min_payoff_ratio,
        )
    walk_forward_finished_at = time.perf_counter()
    ranked = sorted(results, key=_rank_key, reverse=True)
    best_by_route: dict[str, dict[str, Any]] = {}
    best_by_symbol: dict[str, dict[str, Any]] = {}
    for row in ranked:
        best_by_route.setdefault(str(row.get("route_id")), _slim_candidate(row))
        best_by_symbol.setdefault(str(row.get("requested_symbol")), _slim_candidate(row))

    recovery_candidates = [
        _slim_candidate(row)
        for row in ranked
        if bool((row.get("recovery_gate") or {}).get("passed"))
    ]
    robust_recovery_candidates = [
        _slim_candidate(row)
        for row in ranked
        if bool((row.get("robust_recovery_gate") or {}).get("passed"))
    ]
    route_recovery_counts = {
        route_id: sum(1 for row in results if row.get("route_id") == route_id and (row.get("recovery_gate") or {}).get("passed"))
        for route_id in route_plan
    }
    route_robust_recovery_counts = {
        route_id: sum(
            1
            for row in results
            if row.get("route_id") == route_id and (row.get("robust_recovery_gate") or {}).get("passed")
        )
        for route_id in route_plan
    }
    total_seconds = max(time.perf_counter() - started_at, 0.0)
    fetch_seconds = max(fetch_finished_at - setup_finished_at, 0.0)
    grid_seconds = max(grid_finished_at - fetch_finished_at, 0.0)
    walk_forward_seconds = max(walk_forward_finished_at - grid_finished_at, 0.0)
    payload = {
        "generated_at": _utc_now().isoformat(),
        "status": "ok" if datasets else "no_datasets",
        "mode": "quarantined_route_risk_combo_sweep",
        "safety": {
            "live_trading_enabled": settings.live_trading_enabled,
            "use_testnet_for_private_channels": settings.use_testnet,
            "writes_closed_trade_reviews": False,
            "clears_route_quarantine": False,
        },
        "config": dataclasses.asdict(config),
        "route_plan": route_plan,
        "route_quarantine": {route_id: route_quarantine_status(route_id) for route_id in route_plan},
        "news": news,
        "historical_signal_risk": {
            "review_count": historical_signal_index.review_count,
            "modes": list(HISTORICAL_POLICY_MODES),
            "min_samples": 20,
            "threshold_profit_factor": 0.8,
        },
        "route_side_risk": {
            "modes": list(ROUTE_SIDE_POLICY_MODES),
            "evaluations": {
                route_id: {
                    side: evaluation.to_dict()
                    for side, evaluation in sorted(evaluations.items())
                }
                for route_id, evaluations in sorted(route_side_evaluations_by_route.items())
            },
        },
        "grid_by_profile": grid_by_profile,
        "fetch_log": fetch_log,
        "dataset_errors": dataset_errors,
        "runtime_observability": {
            "total_seconds": round(total_seconds, 4),
            "setup_seconds": round(max(setup_finished_at - started_at, 0.0), 4),
            "fetch_seconds": round(fetch_seconds, 4),
            "grid_evaluation_seconds": round(grid_seconds, 4),
            "walk_forward_seconds": round(walk_forward_seconds, 4),
            "dataset_count": len(datasets),
            "dataset_error_count": len(dataset_errors),
            "configs_tested": len(results),
            "configs_per_second": round((len(results) / grid_seconds), 4) if grid_seconds > 0 else 0.0,
            "walk_forward_validations": len(preliminary_validation_rows),
            "status": "completed",
        },
        "aggregate": {
            "dataset_count": len(datasets),
            "limit": config.limit,
            "configs_tested": len(results),
            "max_configs_per_dataset": config.max_configs,
            "walk_forward_validations": len(preliminary_validation_rows),
            "max_walk_forward_validations": config.max_walk_forward_validations,
            "recovery_candidate_count": len(recovery_candidates),
            "robust_recovery_candidate_count": len(robust_recovery_candidates),
            "route_recovery_counts": route_recovery_counts,
            "route_robust_recovery_counts": route_robust_recovery_counts,
            "target_profit_factor": config.target_profit_factor,
            "min_test_trades": config.min_test_trades,
            "min_win_rate": config.min_win_rate,
            "max_stop_loss_ratio": config.max_stop_loss_ratio,
            "min_expectancy_r": config.min_expectancy_r,
            "min_payoff_ratio": config.min_payoff_ratio,
            "target_side": config.target_side,
            "target_interval": config.target_interval,
        },
        "best_by_route": best_by_route,
        "best_by_symbol": best_by_symbol,
        "recovery_candidates": recovery_candidates[: config.top_n],
        "robust_recovery_candidates": robust_recovery_candidates[: config.top_n],
        "ranked": [_slim_candidate(row) for row in ranked[: config.top_n]],
    }
    report_path = RISK_COMBO_SWEEP_DIR / f"{_stamp()}-risk-combo-sweep.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload
