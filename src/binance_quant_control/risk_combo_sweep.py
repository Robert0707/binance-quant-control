from __future__ import annotations

import dataclasses
import itertools
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
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
) -> SweepDataset:
    route = resolve_symbol_route(requested_symbol)
    if route.route_id != route_id:
        route = _route_from_id(route_id)
    strategy = load_strategy_config(route.strategy_config)
    fetch_log: list[dict[str, Any]] = []
    last_error = ""
    with BinanceClient(settings) as client:
        for source_symbol, market in _fetch_candidates(route, requested_symbol):
            try:
                rows = fetch_recent_klines(
                    client,
                    source_symbol,
                    route.interval,
                    min(max(limit, MIN_WINDOW_CANDLES), MAX_KLINE_LIMIT),
                    market,
                )
                frame = enrich_indicators(prepare_klines_frame(rows), route.interval, strategy=strategy)
            except (BinanceAPIError, RuntimeError, ValueError) as exc:
                last_error = str(exc)
                fetch_log.append(
                    {
                        "requested_symbol": requested_symbol,
                        "source_symbol": source_symbol,
                        "market": market,
                        "interval": route.interval,
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
                    "interval": route.interval,
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
                interval=route.interval,
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
    side_mode = str(side_policy_mode or "baseline")
    structure_mode = str(structure_policy_mode or "baseline")
    historical_mode = str(historical_policy_mode or "off")
    route_side_mode = str(route_side_policy_mode or "off")

    def entry_filter(previous: pd.Series, _current: pd.Series, _analysis: dict[str, Any], _idx: int):
        action = str((_analysis or {}).get("recommended_action") or "")
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
    ensure_runtime_dirs()
    RISK_COMBO_SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
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
                dataset = fetch_dataset(settings, route_id=route_id, requested_symbol=symbol, limit=config.limit)
                datasets.append(dataset)
                datasets_by_key[_dataset_key(dataset)] = dataset
                fetch_log.extend(dataset.fetch_log)
            except Exception as exc:
                dataset_errors.append({"route_id": route_id, "symbol": symbol, "error": str(exc)})

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
