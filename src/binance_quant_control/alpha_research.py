from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .alpha_families import ACTIVE_STRATEGY_FAMILIES
from .asset_routing import resolve_symbol_route
from .backtest import (
    BacktestFrameCache,
    BacktestMarketContextCache,
    audit_backtest_robustness,
    run_backtest,
)
from .candidate_universe import UniverseSymbol, fetch_top_futures_symbols
from .config import CONFIG_DIR, REPORTS_DIR, Settings, ensure_runtime_dirs
from .feature_label_gate import (
    FeatureLabelGateConfig,
    build_feature_label_entry_gate,
    load_feature_label_gate_index,
)
from .feature_registry import build_feature_manifest
from .historical_signal_risk import build_historical_signal_risk_index
from .order_journal import read_closed_trade_reviews
from .payoff_objective import PayoffObjectiveTargets, payoff_objective_score
from .research_entry_gate import ResearchEntryGateConfig, build_research_entry_gate
from .strategy import StrategyConfig, load_strategy_config
from .symbol_strategy_map import (
    SymbolStrategySpec,
    filter_symbol_interval_families,
    load_symbol_strategy_map,
    resolve_symbol_interval_family_sides,
)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _json_metric(value: Any) -> float | str:
    parsed = _float(value)
    if math.isinf(parsed):
        return "inf" if parsed > 0.0 else "-inf"
    if math.isnan(parsed):
        return 0.0
    return round(parsed, 4)


def _metric(value: Any) -> float:
    if value == "inf":
        return 9999.0
    return _float(value)


def resolve_alpha_research_config_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0] == CONFIG_DIR.name:
        return (CONFIG_DIR.parent / candidate).resolve()
    return (CONFIG_DIR / candidate).resolve()


def _load_alpha_research_config(path: str | Path) -> dict[str, Any]:
    candidate = resolve_alpha_research_config_path(path)
    return yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}


def _slippage_variant(strategy: StrategyConfig, slippage_bps: float) -> StrategyConfig:
    return replace(
        strategy,
        execution=replace(strategy.execution, slippage_bps=float(slippage_bps)),
    )


def _return_over_drawdown(summary: dict[str, Any]) -> float:
    drawdown = max(_float(summary.get("max_drawdown_pct")), 1.0)
    return _float(summary.get("total_return_pct")) / drawdown


def _fold_stability(robustness: dict[str, Any]) -> float:
    folds = list(robustness.get("folds") or [])
    if not folds:
        return 0.0
    positive = sum(1 for item in folds if _float(item.get("total_return_pct")) > 0.0)
    pf_ok = sum(1 for item in folds if _metric(item.get("profit_factor")) >= 1.0)
    return round(((positive / len(folds)) * 0.6 + (pf_ok / len(folds)) * 0.4), 4)


def _slippage_resilience(base_return: float, stressed_returns: list[float]) -> float:
    if not stressed_returns:
        return 0.0
    if base_return <= 0:
        return 0.0
    worst = min(stressed_returns)
    return round(max(0.0, min(1.0, worst / base_return)), 4)


def _compound_return_pct(pnls: list[float]) -> float:
    equity = 1.0
    for pnl in pnls:
        equity *= 1.0 + (pnl / 100.0)
    return round((equity - 1.0) * 100.0, 4)


def _finite_metric(value: Any) -> float | None:
    parsed = _metric(value)
    if parsed >= 9999.0:
        return None
    return parsed


def _weighted_average(rows: list[dict[str, Any]], key: str) -> float:
    total_trades = sum(int(item.get("trade_count") or 0) for item in rows)
    if total_trades <= 0:
        return 0.0
    return round(
        sum(_float(item.get(key)) * int(item.get("trade_count") or 0) for item in rows) / total_trades,
        4,
    )


def _mean_finite(rows: list[dict[str, Any]], key: str) -> float:
    values = [
        _finite_metric(item.get(key))
        for item in rows
        if int(item.get("trade_count") or 0) > 0
    ]
    finite = [item for item in values if item is not None]
    if not finite:
        return 0.0
    return round(sum(finite) / len(finite), 4)


def _aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    thresholds: dict[str, float],
    enforce_win_rate_gate: bool = True,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(group_key) or "unknown"), []).append(row)
    aggregates: list[dict[str, Any]] = []
    for name, group_rows in grouped.items():
        trades = sum(int(item.get("trade_count") or 0) for item in group_rows)
        stop_ratio = _weighted_average(group_rows, "stop_loss_ratio")
        win_rate = _weighted_average(group_rows, "win_rate")
        expectancy_r = _weighted_average(group_rows, "expectancy_r")
        payoff_ratio = _weighted_average(group_rows, "payoff_ratio")
        finite_profit_factor = _mean_finite(group_rows, "profit_factor")
        promoted = sum(1 for item in group_rows if item.get("promotion_eligible"))
        positive_rows = sum(1 for item in group_rows if _float(item.get("total_return_pct")) > 0.0)
        blockers: list[str] = []
        if trades < int(thresholds.get("min_trades", 0.0)):
            blockers.append("aggregate-trade-count-below-floor")
        if enforce_win_rate_gate and win_rate < float(thresholds.get("min_win_rate", 0.0)):
            blockers.append("aggregate-win-rate-below-floor")
        if stop_ratio > float(thresholds.get("max_stop_loss_ratio", 100.0)):
            blockers.append("aggregate-stop-loss-ratio-above-ceiling")
        if finite_profit_factor and finite_profit_factor < float(thresholds.get("min_profit_factor", 1.0)):
            blockers.append("aggregate-profit-factor-below-floor")
        if expectancy_r < float(thresholds.get("min_expectancy_r", 0.0)):
            blockers.append("aggregate-expectancy-r-below-floor")
        if payoff_ratio < float(thresholds.get("min_payoff_ratio", 0.0)):
            blockers.append("aggregate-payoff-ratio-below-floor")
        aggregates.append(
            {
                group_key: name,
                "row_count": len(group_rows),
                "trade_count": trades,
                "weighted_win_rate": win_rate,
                "weighted_stop_loss_ratio": stop_ratio,
                "weighted_partial_tp_then_stop_ratio": _weighted_average(group_rows, "partial_tp_then_stop_ratio"),
                "finite_avg_profit_factor": finite_profit_factor,
                "weighted_expectancy_r": expectancy_r,
                "weighted_payoff_ratio": payoff_ratio,
                "positive_row_count": positive_rows,
                "promotion_eligible_count": promoted,
                "health": "ok" if not blockers else "blocked",
                "blockers": blockers,
            }
        )
    aggregates.sort(
        key=lambda item: (
            item["health"] == "ok",
            _float(item.get("finite_avg_profit_factor")),
            _float(item.get("weighted_expectancy_r")),
            _float(item.get("weighted_payoff_ratio")),
            _float(item.get("weighted_win_rate")),
            -_float(item.get("weighted_stop_loss_ratio")),
            int(item.get("trade_count") or 0),
        ),
        reverse=True,
    )
    return aggregates


def _aggregate_symbol_interval_rows(
    rows: list[dict[str, Any]],
    *,
    thresholds: dict[str, float],
    enforce_win_rate_gate: bool = True,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"{row.get('symbol') or 'unknown'}:{row.get('interval') or 'unknown'}"
        grouped.setdefault(key, []).append(row)
    aggregates: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        symbol, interval = key.split(":", 1)
        trades = sum(int(item.get("trade_count") or 0) for item in group_rows)
        stop_ratio = _weighted_average(group_rows, "stop_loss_ratio")
        win_rate = _weighted_average(group_rows, "win_rate")
        expectancy_r = _weighted_average(group_rows, "expectancy_r")
        payoff_ratio = _weighted_average(group_rows, "payoff_ratio")
        finite_profit_factor = _mean_finite(group_rows, "profit_factor")
        blockers: list[str] = []
        if trades < int(thresholds.get("min_trades", 0.0)):
            blockers.append("symbol-interval-trade-count-below-floor")
        if enforce_win_rate_gate and win_rate < float(thresholds.get("min_win_rate", 0.0)):
            blockers.append("symbol-interval-win-rate-below-floor")
        if stop_ratio > float(thresholds.get("max_stop_loss_ratio", 100.0)):
            blockers.append("symbol-interval-stop-loss-ratio-above-ceiling")
        if finite_profit_factor and finite_profit_factor < float(thresholds.get("min_profit_factor", 1.0)):
            blockers.append("symbol-interval-profit-factor-below-floor")
        if expectancy_r < float(thresholds.get("min_expectancy_r", 0.0)):
            blockers.append("symbol-interval-expectancy-r-below-floor")
        if payoff_ratio < float(thresholds.get("min_payoff_ratio", 0.0)):
            blockers.append("symbol-interval-payoff-ratio-below-floor")
        aggregates.append(
            {
                "symbol": symbol,
                "interval": interval,
                "row_count": len(group_rows),
                "families": sorted({str(item.get("strategy_family") or "unknown") for item in group_rows}),
                "trade_count": trades,
                "weighted_win_rate": win_rate,
                "weighted_stop_loss_ratio": stop_ratio,
                "weighted_partial_tp_then_stop_ratio": _weighted_average(group_rows, "partial_tp_then_stop_ratio"),
                "finite_avg_profit_factor": finite_profit_factor,
                "weighted_expectancy_r": expectancy_r,
                "weighted_payoff_ratio": payoff_ratio,
                "promotion_eligible_count": sum(1 for item in group_rows if item.get("promotion_eligible")),
                "health": "ok" if not blockers else "blocked",
                "blockers": blockers,
            }
        )
    aggregates.sort(
        key=lambda item: (
            item["health"] == "ok",
            _float(item.get("finite_avg_profit_factor")),
            _float(item.get("weighted_expectancy_r")),
            _float(item.get("weighted_payoff_ratio")),
            _float(item.get("weighted_win_rate")),
            -_float(item.get("weighted_stop_loss_ratio")),
            int(item.get("trade_count") or 0),
        ),
        reverse=True,
    )
    return aggregates


def _research_performance_summary(
    rows: list[dict[str, Any]],
    *,
    thresholds: dict[str, float],
    target_metrics: dict[str, Any],
    enforce_win_rate_gate: bool = True,
) -> dict[str, Any]:
    trades = sum(int(item.get("trade_count") or 0) for item in rows)
    summary = {
        "row_count": len(rows),
        "trade_count": trades,
        "promotion_eligible_count": sum(1 for item in rows if item.get("promotion_eligible")),
        "positive_row_count": sum(1 for item in rows if _float(item.get("total_return_pct")) > 0.0),
        "weighted_win_rate": _weighted_average(rows, "win_rate"),
        "weighted_stop_loss_ratio": _weighted_average(rows, "stop_loss_ratio"),
        "weighted_partial_tp_then_stop_ratio": _weighted_average(rows, "partial_tp_then_stop_ratio"),
        "finite_avg_profit_factor": _mean_finite(rows, "profit_factor"),
        "weighted_expectancy_r": _weighted_average(rows, "expectancy_r"),
        "weighted_payoff_ratio": _weighted_average(rows, "payoff_ratio"),
    }
    target_win = float(target_metrics.get("win_rate") or 0.0)
    target_stop = float(target_metrics.get("max_stop_loss_ratio") or 100.0)
    target_expectancy = float(target_metrics.get("expectancy_r") or thresholds.get("min_expectancy_r") or 0.0)
    target_payoff = float(target_metrics.get("payoff_ratio") or thresholds.get("min_payoff_ratio") or 0.0)
    summary["target_metrics"] = {
        "win_rate": target_win,
        "max_stop_loss_ratio": target_stop,
        "expectancy_r": target_expectancy,
        "payoff_ratio": target_payoff,
        "win_rate_mode": "hard" if enforce_win_rate_gate else "advisory",
    }
    summary["target_gap"] = {
        "win_rate_points": round(target_win - float(summary["weighted_win_rate"]), 4),
        "stop_loss_ratio_points": round(float(summary["weighted_stop_loss_ratio"]) - target_stop, 4),
        "expectancy_r": round(target_expectancy - float(summary["weighted_expectancy_r"]), 4),
        "payoff_ratio": round(target_payoff - float(summary["weighted_payoff_ratio"]), 4),
    }
    summary["meets_target"] = (
        trades >= int(thresholds.get("min_trades", 0.0))
        and (not enforce_win_rate_gate or float(summary["weighted_win_rate"]) >= target_win)
        and float(summary["weighted_stop_loss_ratio"]) <= target_stop
        and float(summary["weighted_expectancy_r"]) >= target_expectancy
        and float(summary["weighted_payoff_ratio"]) >= target_payoff
        and summary["promotion_eligible_count"] > 0
    )
    summary["by_family"] = _aggregate_rows(
        rows,
        group_key="strategy_family",
        thresholds=thresholds,
        enforce_win_rate_gate=enforce_win_rate_gate,
    )
    summary["by_symbol"] = _aggregate_rows(
        rows,
        group_key="symbol",
        thresholds=thresholds,
        enforce_win_rate_gate=enforce_win_rate_gate,
    )
    summary["by_interval"] = _aggregate_rows(
        rows,
        group_key="interval",
        thresholds=thresholds,
        enforce_win_rate_gate=enforce_win_rate_gate,
    )
    summary["by_symbol_interval"] = _aggregate_symbol_interval_rows(
        rows,
        thresholds=thresholds,
        enforce_win_rate_gate=enforce_win_rate_gate,
    )
    summary["execution_recommendation"] = (
        "paper_or_testnet_candidate_available"
        if summary["promotion_eligible_count"]
        else "block_new_entries_and_continue_research"
    )
    return summary


def _load_optional_symbol_strategy_map(config: dict[str, Any]) -> dict[str, SymbolStrategySpec]:
    raw = config.get("symbol_strategy_map") or {}
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return {}
    return load_symbol_strategy_map(raw.get("path") or None)


def _resolved_symbol_family_plan(
    *,
    symbols: list[str],
    intervals: list[str],
    families: list[str],
    symbol_specs: dict[str, SymbolStrategySpec],
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for symbol in symbols:
        spec = symbol_specs.get(symbol)
        for interval in intervals:
            filtered = filter_symbol_interval_families(symbol, interval, families, symbol_specs)
            plan.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "families": filtered,
                    "active": bool(filtered),
                    "inactive_reason": None if filtered else "symbol-interval-quarantined-by-strategy-map",
                    "symbol_strategy_map_applied": spec is not None,
                    "primary_family": spec.primary_family if spec else None,
                    "execution_lane": spec.execution_lane if spec else None,
                    "route_id": spec.route_id if spec else None,
                }
            )
    return plan


def _symbol_promotion_thresholds(
    *,
    symbol: str,
    default_min_trades: int,
    default_min_profit_factor: float,
    default_min_win_rate: float,
    default_max_stop_loss_ratio: float,
    symbol_specs: dict[str, SymbolStrategySpec],
) -> tuple[int, float, float, float]:
    spec = symbol_specs.get(symbol)
    if spec is None:
        return (
            default_min_trades,
            default_min_profit_factor,
            default_min_win_rate,
            default_max_stop_loss_ratio,
        )
    return (
        max(default_min_trades, spec.promotion.min_trades),
        max(default_min_profit_factor, spec.promotion.min_profit_factor),
        max(default_min_win_rate, spec.promotion.min_win_rate),
        min(default_max_stop_loss_ratio, spec.promotion.max_stop_loss_ratio),
    )


def _promotion_threshold(
    symbol_strategy_spec: SymbolStrategySpec | None,
    key: str,
    default: float,
) -> float:
    if symbol_strategy_spec is None:
        return float(default)
    value = getattr(symbol_strategy_spec.promotion, key, None)
    if value is None:
        return float(default)
    if key.startswith("max_"):
        return min(float(default), float(value))
    return max(float(default), float(value))


def _symbol_strategy_variant(
    strategy: StrategyConfig,
    symbol_strategy_spec: SymbolStrategySpec | None,
) -> StrategyConfig:
    if symbol_strategy_spec is None:
        return strategy
    raw_risk = (symbol_strategy_spec.strategy_overrides or {}).get("risk") or {}
    if not isinstance(raw_risk, dict) or not raw_risk:
        return strategy
    risk_updates: dict[str, Any] = {}
    for key in (
        "atr_stop_multiple",
        "take_profit_r_multiples",
        "trailing_callback_pct",
        "time_limit_bars",
        "trailing_stop_enabled",
        "exit_profile",
    ):
        if key in raw_risk:
            value = raw_risk[key]
            if key == "take_profit_r_multiples":
                value = tuple(float(item) for item in value)
            risk_updates[key] = value
    if not risk_updates:
        return strategy
    return replace(strategy, risk=replace(strategy.risk, **risk_updates))


def _compose_entry_filters(*filters: Any) -> Any:
    active_filters = [item for item in filters if item is not None]
    if not active_filters:
        return None

    def composed(previous: Any, current: Any, analysis: dict[str, Any], idx: int) -> bool | tuple[bool, str]:
        for entry_filter in active_filters:
            result = entry_filter(previous, current, analysis, idx)
            if isinstance(result, tuple):
                allowed, reason = result
            else:
                allowed = bool(result)
                reason = "entry-filter-veto"
            if not allowed:
                return False, str(reason or "entry-filter-veto")
        return True, ""

    return composed


def _symbol_side_entry_filter(
    *,
    allowed_sides: tuple[str, ...],
    reason_prefix: str = "symbol-strategy-side-policy",
) -> Any:
    normalized = tuple(str(item).upper() for item in allowed_sides if str(item).strip())
    if set(normalized) >= {"BUY", "SELL"}:
        return None

    def side_filter(_previous: Any, _current: Any, analysis: dict[str, Any], _idx: int) -> tuple[bool, str]:
        side = str(analysis.get("recommended_action") or "").upper()
        if side in normalized:
            return True, ""
        return False, f"{reason_prefix}:{side or 'UNKNOWN'}"

    return side_filter


def _symbol_entry_profile_filter(
    *,
    symbol_strategy_spec: SymbolStrategySpec | None,
) -> Any:
    if symbol_strategy_spec is None or not symbol_strategy_spec.entry_filters:
        return None
    profile = symbol_strategy_spec.entry_filters
    min_signed_di = profile.get("min_signed_di")
    max_abs_obv_zscore = profile.get("max_abs_obv_zscore")
    min_obv_zscore = profile.get("min_obv_zscore")
    max_obv_zscore = profile.get("max_obv_zscore")
    min_volume_zscore = profile.get("min_volume_zscore")
    max_volume_zscore = profile.get("max_volume_zscore")
    min_adx = profile.get("min_adx")
    max_adx = profile.get("max_adx")
    blocked_routed_families = tuple(str(item) for item in profile.get("blocked_routed_families") or ())
    required_routed_families = tuple(str(item) for item in profile.get("required_routed_families") or ())

    def profile_filter(previous: Any, _current: Any, analysis: dict[str, Any], _idx: int) -> tuple[bool, str]:
        action = str(analysis.get("recommended_action") or "").upper()
        if action not in {"BUY", "SELL"}:
            return True, ""
        routed_family = str(analysis.get("routed_strategy_family") or analysis.get("strategy_family") or "")
        if blocked_routed_families and routed_family in blocked_routed_families:
            return False, "symbol-entry-profile-blocked-routed-family"
        if required_routed_families and routed_family not in required_routed_families:
            return False, "symbol-entry-profile-required-routed-family"
        plus_di = _float(previous.get("plus_di"))
        minus_di = _float(previous.get("minus_di"))
        signed_di = plus_di - minus_di if action == "BUY" else minus_di - plus_di
        obv_zscore = _float(previous.get("obv_zscore_20"))
        volume_zscore = _float(previous.get("volume_zscore_20"))
        adx = _float(previous.get("adx"))
        if min_signed_di is not None and signed_di < float(min_signed_di):
            return False, "symbol-entry-profile-min-signed-di"
        if max_abs_obv_zscore is not None and abs(obv_zscore) > float(max_abs_obv_zscore):
            return False, "symbol-entry-profile-max-abs-obv-zscore"
        if min_obv_zscore is not None and obv_zscore < float(min_obv_zscore):
            return False, "symbol-entry-profile-min-obv-zscore"
        if max_obv_zscore is not None and obv_zscore > float(max_obv_zscore):
            return False, "symbol-entry-profile-max-obv-zscore"
        if min_volume_zscore is not None and volume_zscore < float(min_volume_zscore):
            return False, "symbol-entry-profile-min-volume-zscore"
        if max_volume_zscore is not None and volume_zscore > float(max_volume_zscore):
            return False, "symbol-entry-profile-max-volume-zscore"
        if min_adx is not None and adx < float(min_adx):
            return False, "symbol-entry-profile-min-adx"
        if max_adx is not None and adx > float(max_adx):
            return False, "symbol-entry-profile-max-adx"
        return True, ""

    return profile_filter


def _out_of_sample_return_pct(summary: dict[str, Any], fraction: float = 0.30) -> float:
    trades = list(summary.get("trades") or [])
    if not trades:
        return _float(summary.get("total_return_pct"))
    start = max(0, int(len(trades) * (1.0 - max(0.05, min(fraction, 0.80)))))
    pnls = [_float(item.get("pnl_pct")) for item in trades[start:]]
    return _compound_return_pct(pnls)


def _rank_score(
    *,
    summary: dict[str, Any],
    robustness: dict[str, Any],
    slippage_resilience: float,
    weights: dict[str, float],
    min_trades: int = 0,
) -> float:
    out_of_sample_return = _out_of_sample_return_pct(summary)
    return_drawdown = _return_over_drawdown(summary)
    profit_factor = min(_metric(summary.get("profit_factor")), 5.0)
    raw_trade_count = _float(summary.get("trade_count"))
    trade_count = min(raw_trade_count / 30.0, 1.0)
    walk_forward = _fold_stability(robustness)
    payoff_score = payoff_objective_score(
        {
            **summary,
            "return_over_drawdown": return_drawdown,
            "trade_count": raw_trade_count,
        },
        targets=PayoffObjectiveTargets(min_trades=max(int(min_trades), 1) if min_trades else 30),
        min_trades=max(int(min_trades), 1) if min_trades else 30,
    )
    score = (
        out_of_sample_return * float(weights.get("out_of_sample_total_return", 0.30))
        + return_drawdown * 10.0 * float(weights.get("return_over_drawdown", 0.25))
        + profit_factor * 10.0 * float(weights.get("profit_factor", 0.18))
        + trade_count * 10.0 * float(weights.get("trade_count", 0.08))
        + slippage_resilience * 10.0 * float(weights.get("slippage_resilience", 0.10))
        + walk_forward * 10.0 * float(weights.get("walk_forward_stability", 0.09))
        + payoff_score * float(weights.get("payoff_objective", 0.35))
    )
    if min_trades > 0 and raw_trade_count < min_trades:
        sample_factor = max(0.05, raw_trade_count / max(float(min_trades), 1.0))
        score *= sample_factor
    if str(robustness.get("status") or "") == "insufficient_sample":
        score *= 0.25
    elif robustness and not bool(robustness.get("passed", False)):
        score *= 0.5
    return round(score, 4)


def _rank_score_from_row(
    row: dict[str, Any],
    *,
    slippage_resilience: float,
    weights: dict[str, float],
    min_trades: int = 0,
) -> float:
    profit_factor = min(_metric(row.get("profit_factor")), 5.0)
    raw_trade_count = _float(row.get("trade_count"))
    trade_count = min(raw_trade_count / 30.0, 1.0)
    payoff_score = payoff_objective_score(
        row,
        targets=PayoffObjectiveTargets(min_trades=max(int(min_trades), 1) if min_trades else 30),
        min_trades=max(int(min_trades), 1) if min_trades else 30,
    )
    score = (
        _float(row.get("out_of_sample_total_return_pct")) * float(weights.get("out_of_sample_total_return", 0.30))
        + _float(row.get("return_over_drawdown")) * 10.0 * float(weights.get("return_over_drawdown", 0.25))
        + profit_factor * 10.0 * float(weights.get("profit_factor", 0.18))
        + trade_count * 10.0 * float(weights.get("trade_count", 0.08))
        + slippage_resilience * 10.0 * float(weights.get("slippage_resilience", 0.10))
        + _float(row.get("walk_forward_stability")) * 10.0 * float(weights.get("walk_forward_stability", 0.09))
        + payoff_score * float(weights.get("payoff_objective", 0.35))
    )
    if min_trades > 0 and raw_trade_count < min_trades:
        score *= max(0.05, raw_trade_count / max(float(min_trades), 1.0))
    if str(row.get("robustness_status") or "") == "insufficient_sample":
        score *= 0.25
    elif not bool(row.get("base_promotion_candidate", False)):
        score *= 0.5
    return round(score, 4)


def _summarize_result(
    *,
    symbol: str,
    interval: str,
    family: str,
    payload: dict[str, Any],
    stressed_returns: list[float],
    weights: dict[str, float],
    min_trades: int,
    min_profit_factor: float,
    min_win_rate: float,
    max_stop_loss_ratio: float,
    min_expectancy_r: float,
    min_payoff_ratio: float,
    universe_symbol: UniverseSymbol | None = None,
    symbol_strategy_spec: SymbolStrategySpec | None = None,
    enforce_win_rate_gate: bool = True,
) -> dict[str, Any]:
    summary = payload["summary"]
    robustness = payload.get("robustness") or {}
    base_return = _float(summary.get("total_return_pct"))
    resilience = _slippage_resilience(base_return, stressed_returns)
    ranking_score = _rank_score(
        summary=summary,
        robustness=robustness,
        slippage_resilience=resilience,
        weights=weights,
        min_trades=min_trades,
    )
    trades = list(summary.get("trades") or [])
    pnls = [_float(item.get("pnl_pct")) for item in trades]
    trade_count = int(summary.get("trade_count") or 0)
    oos_return = _out_of_sample_return_pct(summary)
    robustness_passed = bool(robustness.get("passed", False))
    stop_loss_ratio = (
        round(
            (
                sum(
                    1
                    for item in trades
                    if str(item.get("exit_reason") or "") in {"stop_loss", "stop_priority_same_bar"}
                )
                / len(trades)
            )
            * 100.0,
            2,
        )
        if trades
        else 0.0
    )
    profit_factor = _metric(summary.get("profit_factor"))
    win_rate = _float(summary.get("win_rate"))
    expectancy_r = _float(summary.get("expectancy_r"))
    payoff_ratio = _float(summary.get("payoff_ratio"))
    base_promotion_candidate = (
        trade_count >= int(min_trades)
        and base_return > 0.0
        and oos_return > 0.0
        and robustness_passed
        and profit_factor >= min_profit_factor
        and (not enforce_win_rate_gate or win_rate >= min_win_rate)
        and stop_loss_ratio <= max_stop_loss_ratio
        and expectancy_r >= min_expectancy_r
        and payoff_ratio >= min_payoff_ratio
    )
    row = {
        "symbol": symbol,
        "interval": interval,
        "strategy_family": family,
        "cohort_id": f"{symbol}:{interval}:{family}",
        "universe_rank": universe_symbol.rank if universe_symbol else None,
        "quote_volume_usdt": universe_symbol.quote_volume_usdt if universe_symbol else None,
        "universe_source": universe_symbol.source if universe_symbol else "configured",
        "ranking_score": ranking_score,
        "out_of_sample_total_return_pct": oos_return,
        "total_return_pct": summary.get("total_return_pct"),
        "max_drawdown_pct": summary.get("max_drawdown_pct"),
        "return_over_drawdown": round(_return_over_drawdown(summary), 4),
        "profit_factor": summary.get("profit_factor"),
        "expectancy_r": summary.get("expectancy_r"),
        "avg_win_r": summary.get("avg_win_r"),
        "avg_loss_r": summary.get("avg_loss_r"),
        "payoff_ratio": summary.get("payoff_ratio"),
        "break_even_win_rate": summary.get("break_even_win_rate"),
        "expectancy_edge_points": summary.get("expectancy_edge_points"),
        "trade_count": trade_count,
        "win_rate": summary.get("win_rate"),
        "stop_loss_ratio": stop_loss_ratio,
        "partial_tp_then_stop_ratio": round(
            (
                sum(1 for item in trades if str(item.get("exit_reason") or "") == "partial_tp_then_stop")
                / len(trades)
            )
            * 100.0,
            2,
        )
        if trades
        else 0.0,
        "avg_pnl_pct": round(sum(pnls) / len(pnls), 4) if pnls else 0.0,
        "walk_forward_stability": _fold_stability(robustness),
        "slippage_resilience": resilience,
        "stressed_returns": stressed_returns,
        "stress_tested": bool(stressed_returns),
        "robustness_status": robustness.get("status"),
        "base_promotion_candidate": base_promotion_candidate,
        "promotion_eligible": base_promotion_candidate and (not stressed_returns or resilience > 0.0),
        "promotion_min_trades": int(min_trades),
        "win_rate_mode": "hard" if enforce_win_rate_gate else "advisory",
        "report_json": payload["artifacts"]["report_json"],
    }
    if symbol_strategy_spec is not None:
        row["symbol_strategy"] = {
            "primary_family": symbol_strategy_spec.primary_family,
            "allowed_families": list(symbol_strategy_spec.allowed_families),
            "interval_families": {
                interval: list(families)
                for interval, families in sorted(symbol_strategy_spec.interval_families.items())
            },
            "interval_family_sides": {
                interval: {family: list(sides) for family, sides in sorted(family_sides.items())}
                for interval, family_sides in sorted(symbol_strategy_spec.interval_family_sides.items())
            },
            "execution_lane": symbol_strategy_spec.execution_lane,
            "route_id": symbol_strategy_spec.route_id,
            "asset_class": symbol_strategy_spec.asset_class,
            "promotion": symbol_strategy_spec.promotion.to_dict(),
            "risk_filters": list(symbol_strategy_spec.risk_filters),
            "thesis": symbol_strategy_spec.thesis,
        }
        if family != symbol_strategy_spec.primary_family:
            row["research_note"] = "secondary-family-research-only"
    gate = payload.get("research_entry_gate") or {}
    if gate:
        row["research_entry_gate"] = {
            key: value
            for key, value in gate.items()
            if key not in {"route_side"}
        }
        row["route_side_gate"] = gate.get("route_side") or {}
        row["entry_veto_reasons"] = summary.get("entry_veto_reasons") or {}
    return row


def _with_stress_metrics(
    row: dict[str, Any],
    *,
    stressed_returns: list[float],
    weights: dict[str, float],
) -> dict[str, Any]:
    updated = dict(row)
    base_return = _float(updated.get("total_return_pct"))
    resilience = _slippage_resilience(base_return, stressed_returns)
    updated["slippage_resilience"] = resilience
    updated["stressed_returns"] = stressed_returns
    updated["stress_tested"] = bool(stressed_returns)
    updated["ranking_score"] = _rank_score_from_row(
        updated,
        slippage_resilience=resilience,
        weights=weights,
        min_trades=int(updated.get("promotion_min_trades") or 0),
    )
    base_candidate = bool(updated.get("base_promotion_candidate", updated.get("promotion_eligible")))
    updated["promotion_eligible"] = base_candidate and bool(stressed_returns) and resilience > 0.0
    if updated["promotion_eligible"]:
        updated.pop("promotion_blocker", None)
    elif base_candidate:
        updated["promotion_blocker"] = "failed-slippage-stress"
    return updated


def _ranked_universe_snapshot(symbols: list[UniverseSymbol]) -> dict[str, Any]:
    ranked = [item for item in symbols if item.rank > 0]
    if not ranked:
        return {
            "alpha_sources": [
                "configured-symbols",
                "24h-volume-rank-unavailable",
                "relative-strength/funding/OI/news hooks remain evaluated downstream when data is available",
            ],
            "high_beta_proxy": [],
        }
    ranked.sort(key=lambda item: item.rank)
    high_beta_proxy = [
        {
            "symbol": item.symbol,
            "rank": item.rank,
            "quote_volume_usdt": item.quote_volume_usdt,
            "source": item.source,
            "alpha_interpretation": "24h quote-volume leadership / high-attention rotation proxy",
        }
        for item in ranked[:10]
    ]
    return {
        "alpha_sources": [
            "24h Binance futures quote-volume expansion",
            "high beta / high-attention rank proxy from liquid USDT perpetuals",
            "per-candidate OI/funding/taker-flow context inside reversal_squeeze family",
            "event-risk context through structured digest when configured",
        ],
        "high_beta_proxy": high_beta_proxy,
    }


def run_aggressive_alpha_research(
    settings: Settings,
    *,
    config_path: str | Path = "aggressive-alpha-research.default.yaml",
    output_dir: Path | None = None,
    symbol_overrides: list[str] | None = None,
    interval_overrides: list[str] | None = None,
    limit_override: int | None = None,
    family_overrides: list[str] | None = None,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    config = _load_alpha_research_config(config_path)
    safety = config.get("safety") or {}
    if bool(safety.get("mainnet_live_allowed")):
        raise ValueError("Aggressive alpha research lane must remain mainnet_live_allowed=false.")

    strategy_path = (config.get("strategy") or {}).get("strategy_config") or "strategy-aggressive-alpha-research.yaml"
    strategy = load_strategy_config(strategy_path)
    universe = config.get("universe") or {}
    configured_symbols = [str(item).upper() for item in universe.get("symbols") or [strategy.defaults.symbol]]
    universe_errors: list[str] = []
    universe_symbols: list[UniverseSymbol] = []
    if symbol_overrides:
        symbols = [str(item).upper() for item in symbol_overrides]
    elif bool(universe.get("include_top_futures_volume", False)):
        try:
            universe_symbols = fetch_top_futures_symbols(
                settings,
                limit=int(universe.get("top_volume_limit") or len(configured_symbols) or 20),
                include_symbols=configured_symbols,
            )
            symbols = [item.symbol for item in universe_symbols]
        except Exception as exc:
            universe_errors.append(str(exc))
            symbols = configured_symbols
    else:
        symbols = configured_symbols
    universe_by_symbol = {item.symbol: item for item in universe_symbols}
    intervals = [str(item) for item in (interval_overrides or universe.get("intervals") or [strategy.defaults.interval])]
    limit = int(limit_override or universe.get("limit") or strategy.defaults.limit)
    ranking = config.get("ranking") or {}
    risk_control = ResearchEntryGateConfig.from_mapping(config.get("research_entry_gate"))
    feature_label_gate_config = FeatureLabelGateConfig.from_mapping(config.get("feature_label_gate"))
    feature_label_gate_index = load_feature_label_gate_index(feature_label_gate_config)
    symbol_strategy_specs = _load_optional_symbol_strategy_map(config)
    weights = ranking.get("weights") or {}
    min_trades = int(ranking.get("min_trades") or 0)
    min_profit_factor = float(ranking.get("min_profit_factor") or 1.0)
    min_win_rate = float(ranking.get("min_win_rate") or 0.0)
    max_stop_loss_ratio = float(ranking.get("max_stop_loss_ratio") or 100.0)
    min_expectancy_r = float(ranking.get("min_expectancy_r") or 0.0)
    min_payoff_ratio = float(ranking.get("min_payoff_ratio") or 0.0)
    enforce_win_rate_gate = bool(ranking.get("enforce_win_rate_gate", True))
    promotion_thresholds = {
        "min_trades": float(min_trades),
        "min_profit_factor": float(min_profit_factor),
        "min_win_rate": float(min_win_rate),
        "max_stop_loss_ratio": float(max_stop_loss_ratio),
        "min_expectancy_r": float(min_expectancy_r),
        "min_payoff_ratio": float(min_payoff_ratio),
        "enforce_win_rate_gate": enforce_win_rate_gate,
    }
    target_metrics = ranking.get("target_metrics") or {
        "win_rate": min_win_rate,
        "max_stop_loss_ratio": max_stop_loss_ratio,
        "expectancy_r": min_expectancy_r,
        "payoff_ratio": min_payoff_ratio,
    }
    slippage_cases = [float(item) for item in ranking.get("slippage_bps_cases") or [strategy.execution.slippage_bps]]
    stress_top_n = max(0, int(ranking.get("stress_top_n") or 0))
    configured_families = family_overrides or ranking.get("strategy_families") or list(ACTIVE_STRATEGY_FAMILIES)
    families = [str(item) for item in configured_families if str(item) in ACTIVE_STRATEGY_FAMILIES]
    families = list(dict.fromkeys(families))
    if not families:
        families = list(ACTIVE_STRATEGY_FAMILIES)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = output_dir or REPORTS_DIR / f"{stamp}-aggressive-alpha-research"
    root.mkdir(parents=True, exist_ok=True)
    base_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    skipped_symbol_intervals: list[dict[str, str]] = []
    progress_path = root / "alpha-research-progress.json"
    total_base_cohorts = sum(
        len(filter_symbol_interval_families(symbol, interval, families, symbol_strategy_specs))
        for symbol in symbols
        for interval in intervals
    )

    def write_progress(
        *,
        phase: str,
        completed_base_cohorts: int,
        current: dict[str, Any] | None = None,
    ) -> None:
        progress_path.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    "phase": phase,
                    "completed_base_cohorts": completed_base_cohorts,
                    "total_base_cohorts": total_base_cohorts,
                    "row_count": len(base_rows),
                    "error_count": len(errors),
                    "current": current or {},
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    write_progress(phase="base", completed_base_cohorts=0)
    completed_base_cohorts = 0
    frame_cache: BacktestFrameCache = {}
    market_context_cache: BacktestMarketContextCache = {}
    review_rows = read_closed_trade_reviews() if risk_control.enabled else []
    historical_signal_index = (
        build_historical_signal_risk_index(review_rows)
        if risk_control.enabled and risk_control.historical_signal_veto
        else None
    )

    for symbol in symbols:
        for interval in intervals:
            symbol_families = filter_symbol_interval_families(symbol, interval, families, symbol_strategy_specs)
            if not symbol_families:
                spec = symbol_strategy_specs.get(symbol)
                configured_empty = spec is not None and interval in spec.interval_families
                target = skipped_symbol_intervals if configured_empty else errors
                target.append(
                    {
                        "symbol": symbol,
                        "interval": interval,
                        "family": "none",
                        "reason" if configured_empty else "error": (
                            "symbol-interval-quarantined-by-strategy-map"
                            if configured_empty
                            else "symbol-strategy-map-filtered-all-families"
                        ),
                    }
                )
                continue
            for family in symbol_families:
                pair_dir = root / f"{symbol.lower()}-{interval}" / family
                try:
                    write_progress(
                        phase="base",
                        completed_base_cohorts=completed_base_cohorts,
                        current={"symbol": symbol, "interval": interval, "strategy_family": family},
                    )
                    (
                        symbol_min_trades,
                        symbol_min_profit_factor,
                        symbol_min_win_rate,
                        symbol_max_stop_loss_ratio,
                    ) = _symbol_promotion_thresholds(
                        symbol=symbol,
                        default_min_trades=min_trades,
                        default_min_profit_factor=min_profit_factor,
                        default_min_win_rate=min_win_rate,
                        default_max_stop_loss_ratio=max_stop_loss_ratio,
                        symbol_specs=symbol_strategy_specs,
                    )
                    symbol_spec = symbol_strategy_specs.get(symbol)
                    strategy_variant = _symbol_strategy_variant(strategy, symbol_spec)
                    symbol_min_expectancy_r = _promotion_threshold(
                        symbol_spec,
                        "min_expectancy_r",
                        min_expectancy_r,
                    )
                    symbol_min_payoff_ratio = _promotion_threshold(
                        symbol_spec,
                        "min_payoff_ratio",
                        min_payoff_ratio,
                    )
                    route_obj = resolve_symbol_route(symbol)
                    entry_gate, entry_gate_metadata = build_research_entry_gate(
                        route_id=(
                            symbol_strategy_specs[symbol].route_id
                            if symbol in symbol_strategy_specs and symbol_strategy_specs[symbol].route_id
                            else route_obj.route_id
                        ),
                        symbol=symbol,
                        config=risk_control,
                        reviews=review_rows,
                        historical_signal_index=historical_signal_index,
                    )
                    label_gate, label_gate_metadata = build_feature_label_entry_gate(
                        symbol=symbol,
                        interval=interval,
                        config=feature_label_gate_config,
                        index=feature_label_gate_index,
                    )
                    allowed_sides = resolve_symbol_interval_family_sides(
                        symbol,
                        interval,
                        family,
                        symbol_strategy_specs,
                    )
                    side_gate = _symbol_side_entry_filter(allowed_sides=allowed_sides)
                    profile_gate = _symbol_entry_profile_filter(symbol_strategy_spec=symbol_spec)
                    combined_entry_gate = _compose_entry_filters(entry_gate, label_gate, side_gate, profile_gate)
                    base_payload = run_backtest(
                        settings,
                        strategy=strategy_variant,
                        symbol=symbol,
                        market=strategy.defaults.market,
                        interval=interval,
                        limit=limit,
                        output_dir=pair_dir / "base",
                        strategy_family=family,
                        frame_cache=frame_cache,
                        market_context_cache=market_context_cache,
                        entry_filter=combined_entry_gate,
                        research_entry_gate={
                            **entry_gate_metadata,
                            "feature_label_gate": label_gate_metadata,
                        },
                    )
                    route = base_payload.get("robustness") or {}
                    if not route:
                        route = audit_backtest_robustness(base_payload["summary"], route_obj.validation)
                    base_rows.append(
                        _summarize_result(
                            symbol=symbol,
                            interval=interval,
                            family=family,
                            payload={**base_payload, "robustness": route},
                            stressed_returns=[],
                            weights=weights,
                            min_trades=symbol_min_trades,
                            min_profit_factor=symbol_min_profit_factor,
                            min_win_rate=symbol_min_win_rate,
                            max_stop_loss_ratio=symbol_max_stop_loss_ratio,
                            min_expectancy_r=symbol_min_expectancy_r,
                            min_payoff_ratio=symbol_min_payoff_ratio,
                            universe_symbol=universe_by_symbol.get(symbol),
                            symbol_strategy_spec=symbol_spec,
                            enforce_win_rate_gate=enforce_win_rate_gate,
                        )
                    )
                except Exception as exc:
                    errors.append({"symbol": symbol, "interval": interval, "family": family, "error": str(exc)})
                finally:
                    completed_base_cohorts += 1
                    write_progress(
                        phase="base",
                        completed_base_cohorts=completed_base_cohorts,
                        current={"symbol": symbol, "interval": interval, "strategy_family": family},
                    )

    stress_required = len(slippage_cases) > 1 and stress_top_n > 0
    if stress_required:
        for row in base_rows:
            if row.get("promotion_eligible"):
                row["promotion_eligible"] = False
                row["promotion_blocker"] = "awaiting-slippage-stress"
    base_rows.sort(key=lambda item: item["ranking_score"], reverse=True)
    rows_by_cohort = {str(item["cohort_id"]): item for item in base_rows}
    candidates_for_stress = [
        item
        for item in base_rows
        if int(item.get("trade_count") or 0) >= int(item.get("promotion_min_trades") or min_trades)
        and _float(item.get("total_return_pct")) > 0.0
        and _float(item.get("out_of_sample_total_return_pct")) > 0.0
    ][:stress_top_n]
    for row in candidates_for_stress:
        write_progress(
            phase="stress",
            completed_base_cohorts=completed_base_cohorts,
            current={
                "symbol": row.get("symbol"),
                "interval": row.get("interval"),
                "strategy_family": row.get("strategy_family"),
            },
        )
        stressed_returns: list[float] = []
        symbol = str(row["symbol"])
        interval = str(row["interval"])
        family = str(row["strategy_family"])
        pair_dir = root / f"{symbol.lower()}-{interval}" / family
        strategy_variant = _symbol_strategy_variant(strategy, symbol_strategy_specs.get(symbol))
        for slippage_bps in slippage_cases:
            if slippage_bps == strategy.execution.slippage_bps:
                continue
            try:
                variant = _slippage_variant(strategy_variant, slippage_bps)
                route_obj = resolve_symbol_route(symbol)
                entry_gate, entry_gate_metadata = build_research_entry_gate(
                    route_id=(
                        symbol_strategy_specs[symbol].route_id
                        if symbol in symbol_strategy_specs and symbol_strategy_specs[symbol].route_id
                        else route_obj.route_id
                    ),
                    symbol=symbol,
                    config=risk_control,
                    reviews=review_rows,
                    historical_signal_index=historical_signal_index,
                )
                label_gate, label_gate_metadata = build_feature_label_entry_gate(
                    symbol=symbol,
                    interval=interval,
                    config=feature_label_gate_config,
                    index=feature_label_gate_index,
                )
                allowed_sides = resolve_symbol_interval_family_sides(
                    symbol,
                    interval,
                    family,
                    symbol_strategy_specs,
                )
                side_gate = _symbol_side_entry_filter(allowed_sides=allowed_sides)
                profile_gate = _symbol_entry_profile_filter(symbol_strategy_spec=symbol_strategy_specs.get(symbol))
                combined_entry_gate = _compose_entry_filters(entry_gate, label_gate, side_gate, profile_gate)
                stressed = run_backtest(
                    settings,
                    strategy=variant,
                    symbol=symbol,
                    market=strategy.defaults.market,
                    interval=interval,
                    limit=limit,
                    output_dir=pair_dir / f"slippage-{slippage_bps:g}bps",
                    strategy_family=family,
                    frame_cache=frame_cache,
                    market_context_cache=market_context_cache,
                    entry_filter=combined_entry_gate,
                    research_entry_gate={
                        **entry_gate_metadata,
                        "feature_label_gate": label_gate_metadata,
                    },
                )
                stressed_returns.append(_float(stressed["summary"].get("total_return_pct")))
            except Exception as exc:
                errors.append({"symbol": symbol, "interval": interval, "family": family, "error": str(exc)})
        rows_by_cohort[str(row["cohort_id"])] = _with_stress_metrics(
            row,
            stressed_returns=stressed_returns,
            weights=weights,
        )

    rows = list(rows_by_cohort.values())
    rows.sort(key=lambda item: item["ranking_score"], reverse=True)
    performance_summary = _research_performance_summary(
        rows,
        thresholds=promotion_thresholds,
        target_metrics=target_metrics,
        enforce_win_rate_gate=enforce_win_rate_gate,
    )
    feature_manifest = build_feature_manifest()
    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "mode": "aggressive_alpha_research",
        "mainnet_live_allowed": False,
        "strategy_profile": strategy.profile,
        "symbols": symbols,
        "intervals": intervals,
        "strategy_families": families,
        "limit": limit,
        "safety": {
            "allowed_modes": list(safety.get("allowed_modes") or ["backtest", "paper", "testnet"]),
            "mainnet_live_allowed": False,
        },
        "universe_selection": {
            "configured_symbols": configured_symbols,
            "include_top_futures_volume": bool(universe.get("include_top_futures_volume", False)),
            "top_volume_limit": int(universe.get("top_volume_limit") or 0),
            "selected": [item.to_dict() for item in universe_symbols],
            "errors": universe_errors,
            **_ranked_universe_snapshot(universe_symbols),
        },
        "symbol_strategy_map": {
            "enabled": bool(symbol_strategy_specs),
            "symbols": {
                symbol: spec.to_dict()
                for symbol, spec in sorted(symbol_strategy_specs.items())
                if symbol in symbols
            },
        },
        "feature_label_gate": {
            "enabled": feature_label_gate_config.enabled,
            "config": feature_label_gate_config.to_dict(),
            "dataset_loaded": feature_label_gate_index is not None,
            "example_count": feature_label_gate_index.example_count if feature_label_gate_index is not None else 0,
            "symbol_intervals": feature_label_gate_index.symbol_intervals if feature_label_gate_index is not None else [],
        },
        "resolved_symbol_family_plan": _resolved_symbol_family_plan(
            symbols=symbols,
            intervals=intervals,
            families=families,
            symbol_specs=symbol_strategy_specs,
        ),
        "ranking_method": [
            "independent strategy family cohort",
            "base-first 60-symbol sweep",
            "historical route-side and signal-bucket entry gate when enabled",
            "top-cohort slippage stress test",
            "out-of-sample total return",
            "max drawdown",
            "return/drawdown",
            "profit factor",
            "trade count",
            "slippage sensitivity",
            "walk-forward stability",
        ],
        "feature_manifest": feature_manifest,
        "top": rows[:10],
        "rows": rows,
        "performance_summary": performance_summary,
        "execution_recommendation": performance_summary["execution_recommendation"],
        "errors": errors,
        "skipped_symbol_intervals": skipped_symbol_intervals,
        "stress_test": {
            "stress_top_n": stress_top_n,
            "slippage_bps_cases": slippage_cases,
            "stressed_cohort_count": len(candidates_for_stress),
        },
        "progress_path": str(progress_path),
        "promotion_thresholds": promotion_thresholds,
        "research_entry_gate": risk_control.to_dict()
        | {
            "review_count": len(review_rows),
        },
    }
    report_path = root / "alpha-research-ranking.json"
    payload["output_dir"] = str(root)
    payload["report_path"] = str(report_path)
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_progress(phase="complete", completed_base_cohorts=completed_base_cohorts)
    return payload
