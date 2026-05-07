from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import CONFIG_DIR, STATE_DIR, ensure_runtime_dirs
from .payoff_objective import (
    PayoffObjectiveTargets,
    payoff_objective_score,
    payoff_objective_sort_key,
)

HIGH_WIN_ITERATION_DIR = STATE_DIR / "high-win-iteration"


@dataclass(frozen=True, slots=True)
class HighWinTargets:
    min_trades: int = 100
    min_win_rate: float = 65.0
    max_stop_loss_ratio: float = 35.0
    min_profit_factor: float = 1.50
    min_expectancy_r: float = 0.10
    min_payoff_ratio: float = 1.15
    max_per_trade_risk_pct: float = 2.50
    min_promoted_symbols: int = 0
    required_symbols: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PidGains:
    win_rate_kp: float = 0.015
    stop_loss_kp: float = 0.020
    profit_factor_kp: float = 0.080
    sample_kp: float = 0.010
    integral_ki: float = 0.002
    derivative_kd: float = 0.004
    max_convergence_delta: float = 0.10
    max_adx_delta: float = 8.0
    max_limit: int = 10000


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _float(value: Any, default: float = 0.0) -> float:
    if value in ("inf", "+inf"):
        return 9999.0
    if value == "-inf":
        return -9999.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    if math.isinf(number):
        return 9999.0 if number > 0.0 else -9999.0
    return number


def _finite_float(value: Any) -> float | None:
    parsed = _float(value)
    if abs(parsed) >= 9999.0:
        return None
    return parsed


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _round(value: Any, digits: int = 4) -> float:
    return round(_float(value), digits)


def _resolve_project_path(path: str | Path, *, config_relative: bool = False) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0] == CONFIG_DIR.name:
        return (CONFIG_DIR.parent / candidate).resolve()
    if candidate.parts and candidate.parts[0] == STATE_DIR.name:
        return (STATE_DIR.parent / candidate).resolve()
    if config_relative:
        return (CONFIG_DIR / candidate).resolve()
    return (CONFIG_DIR.parent / candidate).resolve()


def resolve_high_win_iteration_config_path(path: str | Path) -> Path:
    return _resolve_project_path(path, config_relative=True)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    candidate = resolve_high_win_iteration_config_path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"High-win iteration config not found: {candidate}")
    return yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}


def _targets(raw: dict[str, Any]) -> HighWinTargets:
    data = raw.get("targets") or {}
    portfolio = raw.get("portfolio_gate") or {}
    return HighWinTargets(
        min_trades=max(_int(data.get("min_trades"), 100), 1),
        min_win_rate=max(_float(data.get("min_win_rate"), 65.0), 0.0),
        max_stop_loss_ratio=min(max(_float(data.get("max_stop_loss_ratio"), 35.0), 0.0), 100.0),
        min_profit_factor=max(_float(data.get("min_profit_factor"), 1.50), 0.0),
        min_expectancy_r=max(_float(data.get("min_expectancy_r"), 0.10), 0.0),
        min_payoff_ratio=max(_float(data.get("min_payoff_ratio"), 1.15), 0.0),
        max_per_trade_risk_pct=max(_float(data.get("max_per_trade_risk_pct"), 2.50), 0.0),
        min_promoted_symbols=max(_int(portfolio.get("min_promoted_symbols"), 0), 0),
        required_symbols=tuple(
            str(item).upper()
            for item in (portfolio.get("required_symbols") or [])
            if str(item).strip()
        ),
    )


def _pid_gains(raw: dict[str, Any]) -> PidGains:
    data = raw.get("pid") or {}
    return PidGains(
        win_rate_kp=max(_float(data.get("win_rate_kp"), 0.015), 0.0),
        stop_loss_kp=max(_float(data.get("stop_loss_kp"), 0.020), 0.0),
        profit_factor_kp=max(_float(data.get("profit_factor_kp"), 0.080), 0.0),
        sample_kp=max(_float(data.get("sample_kp"), 0.010), 0.0),
        integral_ki=max(_float(data.get("integral_ki"), 0.002), 0.0),
        derivative_kd=max(_float(data.get("derivative_kd"), 0.004), 0.0),
        max_convergence_delta=min(max(_float(data.get("max_convergence_delta"), 0.10), 0.0), 0.25),
        max_adx_delta=max(_float(data.get("max_adx_delta"), 8.0), 0.0),
        max_limit=max(_int(data.get("max_limit"), 10000), 500),
    )


def _report_paths(raw: dict[str, Any], key: str) -> list[Path]:
    reports = raw.get("reports") or {}
    paths = reports.get(key) or []
    return [_resolve_project_path(item) for item in paths if str(item).strip()]


def _load_json_report(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not path.exists():
        return None, {"path": str(path), "error": "report-not-found"}
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        return None, {"path": str(path), "error": f"invalid-json: {exc}"}


def _weighted_average(rows: list[dict[str, Any]], key: str) -> float:
    trade_count = sum(_int(row.get("trade_count")) for row in rows)
    if trade_count <= 0:
        return 0.0
    value = sum(_float(row.get(key)) * _int(row.get("trade_count")) for row in rows) / trade_count
    return round(value, 4)


def _finite_average(rows: list[dict[str, Any]], key: str) -> float:
    values = [
        _finite_float(row.get(key))
        for row in rows
        if _int(row.get("trade_count")) > 0
    ]
    finite = [value for value in values if value is not None]
    if not finite:
        return 0.0
    return round(sum(finite) / len(finite), 4)


def _alpha_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("performance_summary") or {}
    rows = [row for row in (report.get("rows") or []) if isinstance(row, dict)]
    if summary:
        return {
            "row_count": _int(summary.get("row_count"), len(rows)),
            "trade_count": _int(summary.get("trade_count")),
            "promotion_eligible_count": _int(summary.get("promotion_eligible_count")),
            "weighted_win_rate": _round(summary.get("weighted_win_rate")),
            "weighted_stop_loss_ratio": _round(summary.get("weighted_stop_loss_ratio")),
            "finite_avg_profit_factor": _round(summary.get("finite_avg_profit_factor")),
            "weighted_expectancy_r": _round(summary.get("weighted_expectancy_r")),
            "weighted_payoff_ratio": _round(summary.get("weighted_payoff_ratio")),
            "positive_row_count": _int(summary.get("positive_row_count")),
        }
    return {
        "row_count": len(rows),
        "trade_count": sum(_int(row.get("trade_count")) for row in rows),
        "promotion_eligible_count": sum(1 for row in rows if bool(row.get("promotion_eligible"))),
        "weighted_win_rate": _weighted_average(rows, "win_rate"),
        "weighted_stop_loss_ratio": _weighted_average(rows, "stop_loss_ratio"),
        "finite_avg_profit_factor": _finite_average(rows, "profit_factor"),
        "weighted_expectancy_r": _weighted_average(rows, "expectancy_r"),
        "weighted_payoff_ratio": _weighted_average(rows, "payoff_ratio"),
        "positive_row_count": sum(1 for row in rows if _float(row.get("total_return_pct")) > 0.0),
    }


def _target_gaps(summary: dict[str, Any], targets: HighWinTargets) -> dict[str, float]:
    return {
        "sample_trades": float(max(targets.min_trades - _int(summary.get("trade_count")), 0)),
        "win_rate_points": round(targets.min_win_rate - _float(summary.get("weighted_win_rate")), 4),
        "stop_loss_ratio_points": round(_float(summary.get("weighted_stop_loss_ratio")) - targets.max_stop_loss_ratio, 4),
        "profit_factor": round(targets.min_profit_factor - _float(summary.get("finite_avg_profit_factor")), 4),
        "expectancy_r": round(targets.min_expectancy_r - _float(summary.get("weighted_expectancy_r")), 4),
        "payoff_ratio": round(targets.min_payoff_ratio - _float(summary.get("weighted_payoff_ratio")), 4),
    }


def _gate_blockers(summary: dict[str, Any], targets: HighWinTargets) -> list[str]:
    blockers: list[str] = []
    if _int(summary.get("trade_count")) < targets.min_trades:
        blockers.append("aggregate-trade-count-below-floor")
    if _float(summary.get("weighted_win_rate")) < targets.min_win_rate:
        blockers.append("aggregate-win-rate-below-floor")
    if _float(summary.get("weighted_stop_loss_ratio")) > targets.max_stop_loss_ratio:
        blockers.append("aggregate-stop-loss-ratio-above-ceiling")
    if _float(summary.get("finite_avg_profit_factor")) < targets.min_profit_factor:
        blockers.append("aggregate-profit-factor-below-floor")
    if _float(summary.get("weighted_expectancy_r")) < targets.min_expectancy_r:
        blockers.append("aggregate-expectancy-r-below-floor")
    if _float(summary.get("weighted_payoff_ratio")) < targets.min_payoff_ratio:
        blockers.append("aggregate-payoff-ratio-below-floor")
    if _int(summary.get("promotion_eligible_count")) <= 0:
        blockers.append("no-promotion-eligible-cohort")
    return blockers


def _portfolio_gate(
    *,
    alpha_reports: list[tuple[Path, dict[str, Any]]],
    targets: HighWinTargets,
) -> dict[str, Any]:
    promoted_rows: list[dict[str, Any]] = []
    for path, report in alpha_reports:
        for row in (report.get("rows") or []):
            if not isinstance(row, dict) or not bool(row.get("promotion_eligible")):
                continue
            if not _row_target_shape(row, targets):
                continue
            promoted_rows.append({**_slim_row_metrics(row), "source_report": str(path)})
    promoted_symbols = sorted(
        {
            str(row.get("symbol") or "").upper()
            for row in promoted_rows
            if str(row.get("symbol") or "").strip()
        }
    )
    required_symbols = list(targets.required_symbols)
    missing_required = sorted(symbol for symbol in required_symbols if symbol not in promoted_symbols)
    blockers: list[str] = []
    if targets.min_promoted_symbols and len(promoted_symbols) < targets.min_promoted_symbols:
        blockers.append("promoted-symbol-count-below-floor")
    if missing_required:
        blockers.append("required-promoted-symbols-missing")
    enabled = bool(targets.min_promoted_symbols or required_symbols)
    return {
        "enabled": enabled,
        "passed": not blockers,
        "blockers": blockers,
        "min_promoted_symbols": targets.min_promoted_symbols,
        "required_symbols": required_symbols,
        "promoted_symbol_count": len(promoted_symbols),
        "promoted_symbols": promoted_symbols,
        "missing_required_symbols": missing_required,
        "promoted_rows": promoted_rows,
    }


def _group_family_stats(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in (report.get("rows") or []) if isinstance(row, dict)]
    families = list(dict.fromkeys(str(item) for item in (report.get("strategy_families") or []) if str(item)))
    for row in rows:
        family = str(row.get("strategy_family") or "unknown")
        if family not in families:
            families.append(family)
    stats: list[dict[str, Any]] = []
    for family in families:
        family_rows = [row for row in rows if str(row.get("strategy_family") or "unknown") == family]
        stats.append(
            {
                "strategy_family": family,
                "row_count": len(family_rows),
                "trade_count": sum(_int(row.get("trade_count")) for row in family_rows),
                "weighted_win_rate": _weighted_average(family_rows, "win_rate"),
                "weighted_stop_loss_ratio": _weighted_average(family_rows, "stop_loss_ratio"),
                "finite_avg_profit_factor": _finite_average(family_rows, "profit_factor"),
                "weighted_expectancy_r": _weighted_average(family_rows, "expectancy_r"),
                "weighted_payoff_ratio": _weighted_average(family_rows, "payoff_ratio"),
                "promotion_eligible_count": sum(1 for row in family_rows if bool(row.get("promotion_eligible"))),
            }
        )
    stats.sort(
        key=lambda item: (
            _int(item.get("trade_count")),
            _float(item.get("weighted_expectancy_r")),
            _float(item.get("weighted_payoff_ratio")),
            _float(item.get("finite_avg_profit_factor")),
        ),
        reverse=True,
    )
    return stats


def _zero_sample_families(family_stats: list[dict[str, Any]]) -> list[str]:
    return [
        str(item.get("strategy_family"))
        for item in family_stats
        if _int(item.get("trade_count")) <= 0 and str(item.get("strategy_family") or "")
    ]


def _cohort_key(row: dict[str, Any]) -> str:
    cohort_id = str(row.get("cohort_id") or "").strip()
    if cohort_id:
        return cohort_id
    return ":".join(
        [
            str(row.get("symbol") or "unknown"),
            str(row.get("interval") or "unknown"),
            str(row.get("strategy_family") or "unknown"),
        ]
    )


def _row_target_shape(row: dict[str, Any], targets: HighWinTargets) -> bool:
    return (
        _int(row.get("trade_count")) > 0
        and _float(row.get("win_rate")) >= targets.min_win_rate
        and _float(row.get("stop_loss_ratio")) <= targets.max_stop_loss_ratio
        and _float(row.get("profit_factor")) >= targets.min_profit_factor
        and _float(row.get("expectancy_r")) >= targets.min_expectancy_r
        and _float(row.get("payoff_ratio")) >= targets.min_payoff_ratio
    )


def _row_regressed(row: dict[str, Any], targets: HighWinTargets) -> bool:
    return (
        _int(row.get("trade_count")) > 0
        and (
            _float(row.get("win_rate")) < targets.min_win_rate
            or _float(row.get("stop_loss_ratio")) > targets.max_stop_loss_ratio
            or _float(row.get("profit_factor")) < targets.min_profit_factor
            or _float(row.get("expectancy_r")) < targets.min_expectancy_r
            or _float(row.get("payoff_ratio")) < targets.min_payoff_ratio
            or _float(row.get("total_return_pct")) <= 0.0
        )
    )


def _slim_row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "cohort_id": _cohort_key(row),
        "symbol": str(row.get("symbol") or ""),
        "interval": str(row.get("interval") or ""),
        "strategy_family": str(row.get("strategy_family") or ""),
        "trade_count": _int(row.get("trade_count")),
        "win_rate": _round(row.get("win_rate")),
        "stop_loss_ratio": _round(row.get("stop_loss_ratio")),
        "profit_factor": _round(row.get("profit_factor")),
        "expectancy_r": _round(row.get("expectancy_r")),
        "payoff_ratio": _round(row.get("payoff_ratio")),
        "total_return_pct": _round(row.get("total_return_pct")),
    }


def _sample_expansion_regressions(
    *,
    path: Path,
    report: dict[str, Any],
    peer_reports: list[tuple[Path, dict[str, Any]]],
    targets: HighWinTargets,
) -> list[dict[str, Any]]:
    current_limit = _int(report.get("limit"))
    if current_limit <= 0:
        return []
    current_rows = [row for row in (report.get("rows") or []) if isinstance(row, dict)]
    peer_rows_by_cohort: dict[str, list[tuple[Path, int, dict[str, Any]]]] = {}
    for peer_path, peer_report in peer_reports:
        if peer_path == path:
            continue
        peer_limit = _int(peer_report.get("limit"))
        if peer_limit <= current_limit:
            continue
        for row in (peer_report.get("rows") or []):
            if isinstance(row, dict):
                peer_rows_by_cohort.setdefault(_cohort_key(row), []).append((peer_path, peer_limit, row))

    regressions: list[dict[str, Any]] = []
    for row in current_rows:
        if not _row_target_shape(row, targets):
            continue
        cohort = _cohort_key(row)
        current_trades = _int(row.get("trade_count"))
        for peer_path, peer_limit, peer_row in peer_rows_by_cohort.get(cohort, []):
            if _int(peer_row.get("trade_count")) < current_trades:
                continue
            if not _row_regressed(peer_row, targets):
                continue
            regressions.append(
                {
                    "cohort_id": cohort,
                    "short_sample_report": str(path),
                    "expanded_sample_report": str(peer_path),
                    "short_sample_limit": current_limit,
                    "expanded_sample_limit": peer_limit,
                    "short_sample": _slim_row_metrics(row),
                    "expanded_sample": _slim_row_metrics(peer_row),
                    "reason": "short-sample-target-shape-failed-after-sample-expansion",
                }
            )
            break
    return regressions


def _estimated_limit_for_target_trades(row: dict[str, Any], current_limit: int, targets: HighWinTargets) -> int:
    trade_count = max(_int(row.get("trade_count")), 1)
    return _ceil_to_step(current_limit * (targets.min_trades / trade_count) * 1.20)


def _cohort_expansion_candidates(
    *,
    alpha_reports: list[tuple[Path, dict[str, Any]]],
    alpha_evaluations: list[dict[str, Any]],
    targets: HighWinTargets,
    gains: PidGains,
    top_n: int = 8,
    min_trade_count: int = 10,
    max_limit: int | None = None,
) -> list[dict[str, Any]]:
    candidate_min_trades = max(int(min_trade_count), 1)
    expansion_max_limit = max(int(max_limit or gains.max_limit), gains.max_limit)
    regressed_cohorts = {
        str(item.get("cohort_id"))
        for evaluation in alpha_evaluations
        for item in (evaluation.get("sample_expansion_regressions") or [])
        if isinstance(item, dict) and item.get("cohort_id")
    }
    latest_by_cohort: dict[str, tuple[Path, int, dict[str, Any]]] = {}
    for path, report in alpha_reports:
        limit = _int(report.get("limit"))
        if limit <= 0:
            continue
        for row in (report.get("rows") or []):
            if not isinstance(row, dict):
                continue
            cohort = _cohort_key(row)
            previous = latest_by_cohort.get(cohort)
            if previous is None or limit > previous[1]:
                latest_by_cohort[cohort] = (path, limit, row)

    candidates: list[dict[str, Any]] = []
    for cohort, (path, limit, row) in latest_by_cohort.items():
        trade_count = _int(row.get("trade_count"))
        if (
            cohort in regressed_cohorts
            or trade_count < candidate_min_trades
            or trade_count >= targets.min_trades
        ):
            continue
        if not _row_target_shape(row, targets):
            continue
        if _float(row.get("total_return_pct")) <= 0.0:
            continue
        estimated_limit = _estimated_limit_for_target_trades(row, limit, targets)
        suggested_limit = min(max(estimated_limit, limit), expansion_max_limit)
        sample_ratio = trade_count / max(float(targets.min_trades), 1.0)
        capacity_penalty = 1.0 if estimated_limit > expansion_max_limit else 0.0
        objective_targets = PayoffObjectiveTargets(
            min_trades=targets.min_trades,
            min_profit_factor=targets.min_profit_factor,
            min_expectancy_r=targets.min_expectancy_r,
            min_payoff_ratio=targets.min_payoff_ratio,
            min_win_rate=targets.min_win_rate,
            max_stop_loss_ratio=targets.max_stop_loss_ratio,
        )
        score = payoff_objective_score(row, targets=objective_targets, min_trades=targets.min_trades)
        score += sample_ratio * 8.0
        score -= capacity_penalty
        candidates.append(
            {
                **_slim_row_metrics(row),
                "source_report": str(path),
                "current_limit": limit,
                "sample_gap": targets.min_trades - trade_count,
                "estimated_limit_for_target_trades": estimated_limit,
                "suggested_next_limit": suggested_limit,
                "sample_capacity_limited": estimated_limit > expansion_max_limit,
                "min_trade_count": candidate_min_trades,
                "expansion_score": round(score, 6),
                "reason": "target-shaped-but-under-min-trades",
            }
        )
    candidates.sort(
        key=lambda item: (
            not bool(item.get("sample_capacity_limited")),
            _int(item.get("trade_count")),
            _float(item.get("expectancy_r")),
            _float(item.get("payoff_ratio")),
            _float(item.get("expansion_score")),
            _float(item.get("profit_factor")),
        ),
        reverse=True,
    )
    return candidates[: max(int(top_n), 1)]


def _ceil_to_step(value: float, step: int = 500) -> int:
    return int(math.ceil(value / float(step)) * step)


def _next_limit(report: dict[str, Any], summary: dict[str, Any], targets: HighWinTargets, gains: PidGains) -> int:
    current_limit = max(_int(report.get("limit"), 1500), 1)
    trade_count = max(_int(summary.get("trade_count")), 1)
    if trade_count >= targets.min_trades:
        return min(max(current_limit, 5000), gains.max_limit)
    factor = min(max((targets.min_trades / trade_count) * 1.25, 1.5), 4.0)
    return min(max(_ceil_to_step(current_limit * factor), current_limit), gains.max_limit)


def _load_pid_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {"runs": 0, "integral": {}, "previous_error": {}}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"runs": 0, "integral": {}, "previous_error": {}}
    if not isinstance(payload, dict):
        return {"runs": 0, "integral": {}, "previous_error": {}}
    return payload


def _write_pid_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _pid_control(
    *,
    report: dict[str, Any],
    summary: dict[str, Any],
    targets: HighWinTargets,
    gains: PidGains,
    pid_state: dict[str, Any],
) -> dict[str, Any]:
    gaps = _target_gaps(summary, targets)
    normalized_error = {
        "win_rate": max(gaps["win_rate_points"], 0.0) / max(targets.min_win_rate, 1.0),
        "stop_loss_ratio": max(gaps["stop_loss_ratio_points"], 0.0) / max(targets.max_stop_loss_ratio, 1.0),
        "profit_factor": max(gaps["profit_factor"], 0.0) / max(targets.min_profit_factor, 0.1),
        "expectancy_r": max(gaps["expectancy_r"], 0.0) / max(targets.min_expectancy_r, 0.01),
        "payoff_ratio": max(gaps["payoff_ratio"], 0.0) / max(targets.min_payoff_ratio, 0.1),
        "sample": gaps["sample_trades"] / max(float(targets.min_trades), 1.0),
    }
    integral_raw = pid_state.get("integral") if isinstance(pid_state.get("integral"), dict) else {}
    previous_raw = pid_state.get("previous_error") if isinstance(pid_state.get("previous_error"), dict) else {}
    integral = {
        key: min(max(_float(integral_raw.get(key)) + normalized_error[key], -10.0), 10.0)
        for key in normalized_error
    }
    derivative = {
        key: normalized_error[key] - _float(previous_raw.get(key))
        for key in normalized_error
    }
    convergence_signal = (
        normalized_error["win_rate"] * gains.win_rate_kp
        + normalized_error["stop_loss_ratio"] * gains.stop_loss_kp
        + normalized_error["profit_factor"] * gains.profit_factor_kp
        + sum(integral.values()) * gains.integral_ki
        + sum(derivative.values()) * gains.derivative_kd
    )
    adx_signal = (
        max(gaps["stop_loss_ratio_points"], 0.0) * 0.35
        + max(gaps["win_rate_points"], 0.0) * 0.10
    )
    pid_state["runs"] = _int(pid_state.get("runs")) + 1
    pid_state["integral"] = integral
    pid_state["previous_error"] = normalized_error
    return {
        "scope": "research-gate-only",
        "writes_execution_config": False,
        "normalized_error": {key: round(value, 6) for key, value in normalized_error.items()},
        "integral": {key: round(value, 6) for key, value in integral.items()},
        "derivative": {key: round(value, 6) for key, value in derivative.items()},
        "suggested_bias": {
            "next_limit": _next_limit(report, summary, targets, gains),
            "min_convergence_delta": round(min(max(convergence_signal, 0.0), gains.max_convergence_delta), 4),
            "min_adx_delta": round(min(max(adx_signal, 0.0), gains.max_adx_delta), 4),
            "risk_combo_grid_mode": "focused" if gaps["stop_loss_ratio_points"] > 0.0 or gaps["profit_factor"] > 0.0 else "fast",
            "do_not_widen_stops_first": True,
        },
    }


def _alpha_evaluation(
    *,
    path: Path,
    report: dict[str, Any],
    peer_reports: list[tuple[Path, dict[str, Any]]],
    targets: HighWinTargets,
    gains: PidGains,
    pid_state: dict[str, Any],
) -> dict[str, Any]:
    summary = _alpha_summary(report)
    blockers = _gate_blockers(summary, targets)
    family_stats = _group_family_stats(report)
    sample_expansion_regressions = _sample_expansion_regressions(
        path=path,
        report=report,
        peer_reports=peer_reports,
        targets=targets,
    )
    if sample_expansion_regressions:
        blockers.append("sample-expansion-regression")
    return {
        "path": str(path),
        "generated_at": report.get("generated_at"),
        "strategy_profile": report.get("strategy_profile"),
        "limit": _int(report.get("limit")),
        "mainnet_live_allowed": bool(report.get("mainnet_live_allowed", False)),
        "execution_recommendation": report.get("execution_recommendation"),
        "summary": summary,
        "target_gaps": _target_gaps(summary, targets),
        "gate": {
            "passed": not blockers,
            "blockers": blockers,
            "targets": {
                "min_trades": targets.min_trades,
                "min_win_rate": targets.min_win_rate,
                "max_stop_loss_ratio": targets.max_stop_loss_ratio,
                "min_profit_factor": targets.min_profit_factor,
                "min_expectancy_r": targets.min_expectancy_r,
                "min_payoff_ratio": targets.min_payoff_ratio,
            },
        },
        "family_stats": family_stats,
        "zero_sample_families": _zero_sample_families(family_stats),
        "sample_expansion_regressions": sample_expansion_regressions,
        "pid_controller": _pid_control(
            report=report,
            summary=summary,
            targets=targets,
            gains=gains,
            pid_state=pid_state,
        ),
    }


def _sweep_evaluation(path: Path, report: dict[str, Any], targets: HighWinTargets) -> dict[str, Any]:
    aggregate = report.get("aggregate") or {}
    robust_count = _int(aggregate.get("robust_recovery_candidate_count"))
    recovery_count = _int(aggregate.get("recovery_candidate_count"))
    target_matches = (
        _float(aggregate.get("target_profit_factor")) >= targets.min_profit_factor
        and _int(aggregate.get("min_test_trades")) >= targets.min_trades
        and _float(aggregate.get("min_win_rate")) >= targets.min_win_rate
        and _float(aggregate.get("max_stop_loss_ratio"), 100.0) <= targets.max_stop_loss_ratio
        and _float(aggregate.get("min_expectancy_r")) >= targets.min_expectancy_r
        and _float(aggregate.get("min_payoff_ratio")) >= targets.min_payoff_ratio
    )
    blockers: list[str] = []
    if robust_count <= 0:
        blockers.append("no-robust-recovery-candidate")
    if not target_matches:
        blockers.append("sweep-thresholds-not-strict-enough")
    return {
        "path": str(path),
        "generated_at": report.get("generated_at"),
        "status": report.get("status"),
        "aggregate": aggregate,
        "strict_recovery_gate": {
            "passed": robust_count > 0 and target_matches,
            "blockers": blockers,
        },
        "recovery_candidate_count": recovery_count,
        "robust_recovery_candidate_count": robust_count,
        "best_by_symbol": report.get("best_by_symbol") or {},
    }


def _sweep_under_sampled_candidates(
    sweep_evaluations: list[dict[str, Any]],
    targets: HighWinTargets,
    *,
    min_trade_count: int = 10,
    top_n: int = 6,
) -> list[dict[str, Any]]:
    latest_rows: dict[tuple[str, str, str, str, str], tuple[int, dict[str, Any], dict[str, Any]]] = {}
    for evaluation in sweep_evaluations:
        current_limit = _int((evaluation.get("aggregate") or {}).get("limit"), 0)
        best_by_symbol = evaluation.get("best_by_symbol") or {}
        if not isinstance(best_by_symbol, dict):
            continue
        for symbol, row in best_by_symbol.items():
            if not isinstance(row, dict):
                continue
            params = row.get("params") or {}
            key = (
                str(symbol).upper(),
                str(row.get("route_id") or ""),
                str(row.get("interval") or ""),
                str(row.get("strategy_profile") or ""),
                str(params.get("exit_profile") or ""),
            )
            previous = latest_rows.get(key)
            if previous is None or current_limit >= previous[0]:
                latest_rows[key] = (current_limit, evaluation, row)

    candidates: list[dict[str, Any]] = []
    for (symbol, _route_id, _interval, _profile, _exit_profile), (current_limit, evaluation, row) in latest_rows.items():
        full = row.get("full") or {}
        test = row.get("test") or {}
        robust_gate = row.get("robust_recovery_gate") or {}
        if not isinstance(full, dict) or not isinstance(test, dict):
            continue
        full_trades = _int(full.get("trade_count"))
        if full_trades < max(min_trade_count, 1) or full_trades >= targets.min_trades:
            continue
        if _float(full.get("profit_factor")) < targets.min_profit_factor:
            continue
        if _float(full.get("expectancy_r")) < targets.min_expectancy_r:
            continue
        if _float(full.get("payoff_ratio")) < targets.min_payoff_ratio:
            continue
        if _float(full.get("stop_loss_ratio")) > targets.max_stop_loss_ratio + 10.0:
            continue
        reasons = list(robust_gate.get("reasons") or []) if isinstance(robust_gate, dict) else []
        non_sample_reasons = [
            str(reason)
            for reason in reasons
            if str(reason)
            and str(reason)
            not in {
                "test-trade-count-too-low",
                "initial-recovery-gate-not-passed",
                "walk-forward-min-profit-factor-below-target",
                "walk-forward-min-payoff-ratio-below-target",
            }
        ]
        if non_sample_reasons:
            continue
        baseline_limit = current_limit if current_limit > 0 else 5000
        estimated_limit = _ceil_to_step(
            max(500.0, baseline_limit * (targets.min_trades / max(full_trades, 1)) * 1.2)
        )
        candidates.append(
            {
                "symbol": symbol,
                "source_report": evaluation.get("path"),
                "route_id": row.get("route_id"),
                "interval": row.get("interval"),
                "strategy_profile": row.get("strategy_profile"),
                "params": row.get("params") or {},
                "full": full,
                "test": test,
                "current_limit": current_limit,
                "estimated_limit_for_target_trades": estimated_limit,
                "sample_gap": targets.min_trades - full_trades,
                "reason": "sweep-target-shaped-but-under-min-trades",
            }
        )
    candidates.sort(
        key=lambda item: (
            _int((item.get("full") or {}).get("trade_count")),
            _float((item.get("full") or {}).get("expectancy_r")),
            _float((item.get("full") or {}).get("profit_factor")),
            _float((item.get("full") or {}).get("payoff_ratio")),
        ),
        reverse=True,
    )
    return candidates[: max(int(top_n), 1)]


def _best_alpha(evaluations: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not evaluations:
        return None
    return max(
        evaluations,
        key=lambda item: (
            bool((item.get("gate") or {}).get("passed")),
            not bool(item.get("sample_expansion_regressions")),
            _int((item.get("summary") or {}).get("promotion_eligible_count")),
            *payoff_objective_sort_key(
                {
                    "trade_count": _int((item.get("summary") or {}).get("trade_count")),
                    "win_rate": _float((item.get("summary") or {}).get("weighted_win_rate")),
                    "stop_loss_ratio": _float((item.get("summary") or {}).get("weighted_stop_loss_ratio")),
                    "profit_factor": _float((item.get("summary") or {}).get("finite_avg_profit_factor")),
                    "expectancy_r": _float((item.get("summary") or {}).get("weighted_expectancy_r")),
                    "payoff_ratio": _float((item.get("summary") or {}).get("weighted_payoff_ratio")),
                },
                targets=PayoffObjectiveTargets(
                    min_trades=_int(((item.get("gate") or {}).get("targets") or {}).get("min_trades"), 100),
                    min_profit_factor=_float(((item.get("gate") or {}).get("targets") or {}).get("min_profit_factor"), 1.5),
                    min_expectancy_r=_float(((item.get("gate") or {}).get("targets") or {}).get("min_expectancy_r"), 0.10),
                    min_payoff_ratio=_float(((item.get("gate") or {}).get("targets") or {}).get("min_payoff_ratio"), 1.15),
                    min_win_rate=_float(((item.get("gate") or {}).get("targets") or {}).get("min_win_rate"), 65.0),
                    max_stop_loss_ratio=_float(((item.get("gate") or {}).get("targets") or {}).get("max_stop_loss_ratio"), 35.0),
                ),
            ),
            _float((item.get("summary") or {}).get("finite_avg_profit_factor")),
            _float((item.get("summary") or {}).get("weighted_expectancy_r")),
            _float((item.get("summary") or {}).get("weighted_payoff_ratio")),
            _float((item.get("summary") or {}).get("weighted_win_rate")),
            -_float((item.get("summary") or {}).get("weighted_stop_loss_ratio")),
            _int((item.get("summary") or {}).get("trade_count")),
        ),
    )


def best_alpha_from_iteration(payload: dict[str, Any]) -> dict[str, Any] | None:
    evaluations = [item for item in (payload.get("alpha_evaluations") or []) if isinstance(item, dict)]
    return _best_alpha(evaluations)


def high_win_gap_score(payload: dict[str, Any]) -> float:
    best = best_alpha_from_iteration(payload)
    if best is None:
        return 9999.0
    gaps = best.get("target_gaps") or {}
    targets = ((best.get("gate") or {}).get("targets") or {})
    return round(
        max(_float(gaps.get("sample_trades")), 0.0) / max(_float(targets.get("min_trades")), 1.0)
        + max(_float(gaps.get("win_rate_points")), 0.0) / max(_float(targets.get("min_win_rate")), 1.0)
        + max(_float(gaps.get("stop_loss_ratio_points")), 0.0) / max(_float(targets.get("max_stop_loss_ratio")), 1.0)
        + max(_float(gaps.get("profit_factor")), 0.0) / max(_float(targets.get("min_profit_factor")), 0.1)
        + max(_float(gaps.get("expectancy_r")), 0.0) / max(_float(targets.get("min_expectancy_r")), 0.01)
        + max(_float(gaps.get("payoff_ratio")), 0.0) / max(_float(targets.get("min_payoff_ratio")), 0.1),
        6,
    )


def _top_symbols_from_alpha(report: dict[str, Any] | None, *, fallback: list[str]) -> list[str]:
    if not report:
        return fallback
    rows = [row for row in (report.get("rows") or []) if isinstance(row, dict)]
    rows.sort(
        key=lambda item: (
            _float(item.get("ranking_score")),
            _float(item.get("profit_factor")),
            _float(item.get("win_rate")),
            _int(item.get("trade_count")),
        ),
        reverse=True,
    )
    symbols: list[str] = []
    for row in rows:
        symbol = str(row.get("symbol") or "").upper()
        if symbol and symbol not in symbols:
            symbols.append(symbol)
        if len(symbols) >= 6:
            break
    return symbols or fallback


def _unique_nonempty(values: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in values if item))


def _cohort_expansion_command(config: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return ""
    commands = config.get("commands") or {}
    symbols = ",".join(_unique_nonempty([str(item.get("symbol") or "").upper() for item in candidates]))
    intervals = ",".join(_unique_nonempty([str(item.get("interval") or "") for item in candidates]))
    next_limit = max(_int(item.get("suggested_next_limit")) for item in candidates)
    template = str(
        commands.get("cohort_expansion")
        or (
            "openclaw-quantctl alpha-research --config config/core-high-win-research.default.yaml "
            "--symbols {symbols} --intervals {intervals} --limit {limit} "
            "--output-dir state/cohort-expansion-high-win-next --compact"
        )
    ).strip()
    try:
        return template.format(symbols=symbols, intervals=intervals, limit=next_limit)
    except (KeyError, ValueError):
        return (
            "openclaw-quantctl alpha-research --config config/core-high-win-research.default.yaml "
            f"--symbols {symbols} --intervals {intervals} --limit {next_limit} "
            "--output-dir state/cohort-expansion-high-win-next --compact"
        )


def _sweep_expansion_command(config: dict[str, Any], candidates: list[dict[str, Any]], targets: HighWinTargets) -> str:
    if not candidates:
        return ""
    symbols = ",".join(_unique_nonempty([str(item.get("symbol") or "").upper() for item in candidates]))
    next_limit = max(_int(item.get("estimated_limit_for_target_trades")) for item in candidates)
    commands = config.get("commands") or {}
    template = str(
        commands.get("strict_risk_combo_sweep")
        or (
            "openclaw-quantctl risk-combo-sweep --symbols {symbols} --limit {limit} "
            "--grid-mode focused --min-test-trades {min_trades} --min-win-rate {min_win_rate} "
            "--max-stop-loss-ratio {max_stop_loss_ratio} --target-profit-factor {min_profit_factor} "
            "--min-expectancy-r {min_expectancy_r} --min-payoff-ratio {min_payoff_ratio} "
            "--max-configs 80 --max-walk-forward-validations 12 --skip-news --compact"
        )
    ).strip()
    replacements = {
        "{symbols}": symbols,
        "{limit}": str(next_limit),
        "{min_trades}": str(targets.min_trades),
        "{min_win_rate}": f"{targets.min_win_rate:g}",
        "{max_stop_loss_ratio}": f"{targets.max_stop_loss_ratio:g}",
        "{min_profit_factor}": f"{targets.min_profit_factor:g}",
        "{min_expectancy_r}": f"{targets.min_expectancy_r:g}",
        "{min_payoff_ratio}": f"{targets.min_payoff_ratio:g}",
    }
    command = template
    for key, value in replacements.items():
        command = command.replace(key, value)
    if "--symbols " in command:
        command = _replace_option_value(command, "--symbols", symbols)
    else:
        command = f"{command} --symbols {symbols}"
    if "--limit " in command:
        command = _replace_option_value(command, "--limit", str(next_limit))
    else:
        command = f"{command} --limit {next_limit}"
    return command


def _replace_option_value(command: str, option: str, value: str) -> str:
    parts = command.split()
    for index, part in enumerate(parts[:-1]):
        if part == option:
            parts[index + 1] = value
            return " ".join(parts)
    return command


def _action(
    *,
    code: str,
    priority: str,
    reason: str,
    command: str = "",
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {"code": code, "priority": priority, "reason": reason}
    if command:
        payload["command"] = command
    if params:
        payload["params"] = params
    return payload


def _next_actions(
    *,
    config: dict[str, Any],
    alpha_reports: list[dict[str, Any]],
    alpha_evaluations: list[dict[str, Any]],
    best_alpha_report: dict[str, Any] | None,
    best_alpha_eval: dict[str, Any] | None,
    sweep_evaluations: list[dict[str, Any]],
    cohort_expansion_candidates: list[dict[str, Any]],
    sweep_under_sampled_candidates: list[dict[str, Any]],
    targets: HighWinTargets,
) -> list[dict[str, Any]]:
    commands = config.get("commands") or {}
    fallback_symbols = list(commands.get("risk_sweep_symbols") or ["PAXGUSDT", "ETHUSDT", "XRPUSDT", "TRXUSDT"])
    sweep_symbols = ",".join(_top_symbols_from_alpha(best_alpha_report, fallback=fallback_symbols))
    core_command = str(
        commands.get("core_l5000")
        or "openclaw-quantctl alpha-research --config config/core-high-win-research.default.yaml "
        "--output-dir state/core-10-high-win-l5000 --compact"
    ).strip()
    replacement_command = str(
        commands.get("replacement_scout_l5000")
        or "openclaw-quantctl alpha-research --config config/core-replacement-scout.default.yaml "
        "--output-dir state/replacement-scout-expectancy-l5000 --compact"
    ).strip()
    risk_sweep_command = str(
        commands.get("strict_risk_combo_sweep")
        or (
            "openclaw-quantctl risk-combo-sweep "
            f"--symbols {sweep_symbols} --limit 5000 --grid-mode focused "
            f"--min-test-trades {targets.min_trades} --min-win-rate {targets.min_win_rate:g} "
            f"--max-stop-loss-ratio {targets.max_stop_loss_ratio:g} "
            f"--target-profit-factor {targets.min_profit_factor:g} --skip-news --compact"
        )
    ).strip()

    actions: list[dict[str, Any]] = []
    if not alpha_reports:
        actions.append(
            _action(
                code="run-core-high-win-sample",
                priority="critical",
                reason="No alpha research report was supplied, so the gate has no evidence.",
                command=core_command,
            )
        )
        return actions

    if best_alpha_eval is None:
        return actions
    summary = best_alpha_eval.get("summary") or {}
    gaps = best_alpha_eval.get("target_gaps") or {}
    pid_bias = ((best_alpha_eval.get("pid_controller") or {}).get("suggested_bias") or {})
    regressions = [
        regression
        for evaluation in alpha_evaluations
        for regression in (evaluation.get("sample_expansion_regressions") or [])
        if isinstance(regression, dict)
    ]
    if regressions:
        actions.append(
            _action(
                code="reject-short-sample-regression",
                priority="high",
                reason=(
                    "At least one cohort looked target-shaped in a short sample but failed after "
                    "sample expansion; do not promote it from the short report."
                ),
                command=core_command,
                params={
                    "regressed_cohorts": list(
                        dict.fromkeys(str(item.get("cohort_id")) for item in regressions if item.get("cohort_id"))
                    )
                },
            )
        )
    if cohort_expansion_candidates:
        actions.append(
            _action(
                code="expand-target-shaped-under-sampled-cohorts",
                priority="critical",
                reason=(
                    "Some cohorts already meet the expectancy/PF/payoff shape but have fewer than 100 trades; "
                    "expand those cohorts before broadening the universe again."
                ),
                command=_cohort_expansion_command(config, cohort_expansion_candidates),
                params={
                    "candidates": cohort_expansion_candidates,
                    "symbols": _unique_nonempty(
                        [str(item.get("symbol") or "").upper() for item in cohort_expansion_candidates]
                    ),
                    "intervals": _unique_nonempty(
                        [str(item.get("interval") or "") for item in cohort_expansion_candidates]
                    ),
                },
            )
        )
    if sweep_under_sampled_candidates:
        actions.append(
            _action(
                code="expand-payoff-shaped-risk-sweep-candidates",
                priority="critical",
                reason=(
                    "A risk-combo row has positive PF/expectancy/payoff shape but too few trades; "
                    "expand that exact symbol/exit setup before claiming it is tradable."
                ),
                command=_sweep_expansion_command(config, sweep_under_sampled_candidates, targets),
                params={
                    "candidates": sweep_under_sampled_candidates,
                    "symbols": _unique_nonempty(
                        [str(item.get("symbol") or "").upper() for item in sweep_under_sampled_candidates]
                    ),
                },
            )
        )
    if _int(summary.get("trade_count")) < targets.min_trades:
        actions.append(
            _action(
                code="increase-sample-before-relaxing-rules",
                priority="critical",
                reason="The best report has too few trades for promotion; collect more candles before judging edge quality.",
                command=core_command,
                params={"suggested_next_limit": pid_bias.get("next_limit")},
            )
        )
    if _float(gaps.get("stop_loss_ratio_points")) > 0.0:
        actions.append(
            _action(
                code="tighten-structure-and-stoploss-guard",
                priority="high",
                reason="Pure stop-loss ratio is above the target; mature bots reduce entries first instead of widening stops first.",
                command=risk_sweep_command,
                params={
                    "min_adx_delta": pid_bias.get("min_adx_delta"),
                    "min_convergence_delta": pid_bias.get("min_convergence_delta"),
                },
            )
        )
    if _float(gaps.get("profit_factor")) > 0.0:
        actions.append(
            _action(
                code="reject-low-pf-cohorts-and-scout-replacements",
                priority="high",
                reason="Profit factor is below the recovery floor; the next iteration should search better symbol-family fit.",
                command=replacement_command,
            )
        )
    if _float(gaps.get("expectancy_r")) > 0.0 or _float(gaps.get("payoff_ratio")) > 0.0:
        actions.append(
            _action(
                code="improve-risk-reward-expectancy",
                priority="high",
                reason=(
                    "Fixed-risk expectancy or payoff ratio is below target; reduce low-R partial exits, "
                    "require cleaner structure, and favor setups with larger average winner than loser."
                ),
                command=risk_sweep_command,
                params={
                    "min_expectancy_r": targets.min_expectancy_r,
                    "min_payoff_ratio": targets.min_payoff_ratio,
                },
            )
        )
    if _float(gaps.get("win_rate_points")) > 0.0:
        actions.append(
            _action(
                code="raise-confirmation-pressure",
                priority="medium",
                reason="Win rate is below target; bias the next sweep toward higher convergence, ADX, structure, and feedback-bucket vetoes.",
                command=risk_sweep_command,
                params={
                    "min_adx_delta": pid_bias.get("min_adx_delta"),
                    "min_convergence_delta": pid_bias.get("min_convergence_delta"),
                },
            )
        )
    for family in best_alpha_eval.get("zero_sample_families") or []:
        actions.append(
            _action(
                code="family-insufficient-sample",
                priority="medium",
                reason=f"{family} is connected but produced no sample in the supplied report; expand history before loosening it.",
                command=core_command,
                params={"strategy_family": family},
            )
        )
    if not any((item.get("strict_recovery_gate") or {}).get("passed") for item in sweep_evaluations):
        actions.append(
            _action(
                code="run-strict-risk-combo-sweep",
                priority="medium",
                reason="No strict robust risk-combo candidate is available yet.",
                command=risk_sweep_command,
            )
        )
    return actions


def compact_high_win_iteration(payload: dict[str, Any]) -> dict[str, Any]:
    best = best_alpha_from_iteration(payload)
    sample_regressions = [
        {
            "cohort_id": item.get("cohort_id"),
            "short_sample_limit": item.get("short_sample_limit"),
            "expanded_sample_limit": item.get("expanded_sample_limit"),
            "short_sample": item.get("short_sample"),
            "expanded_sample": item.get("expanded_sample"),
        }
        for evaluation in (payload.get("alpha_evaluations") or [])
        if isinstance(evaluation, dict)
        for item in (evaluation.get("sample_expansion_regressions") or [])
        if isinstance(item, dict)
    ]
    return {
        "mode": payload.get("mode"),
        "safety": payload.get("safety"),
        "targets": payload.get("targets"),
        "report_path": payload.get("report_path"),
        "best_alpha_path": payload.get("best_alpha_path"),
        "promotion_allowed": payload.get("promotion_allowed"),
        "safe_to_open_new_entries": payload.get("safe_to_open_new_entries"),
        "execution_recommendation": payload.get("execution_recommendation"),
        "best_alpha_gate": payload.get("best_alpha_gate"),
        "portfolio_gate": payload.get("portfolio_gate"),
        "gap_score": high_win_gap_score(payload),
        "best_alpha_summary": (best or {}).get("summary"),
        "cohort_expansion_candidate_count": len(payload.get("cohort_expansion_candidates") or []),
        "cohort_expansion_candidates": payload.get("cohort_expansion_candidates") or [],
        "sweep_under_sampled_candidate_count": len(payload.get("sweep_under_sampled_candidates") or []),
        "sweep_under_sampled_candidates": payload.get("sweep_under_sampled_candidates") or [],
        "sample_expansion_regression_count": len(sample_regressions),
        "sample_expansion_regressions": sample_regressions,
        "next_action_codes": [item.get("code") for item in (payload.get("next_actions") or [])],
        "next_actions": payload.get("next_actions") or [],
    }


def suggested_next_limit(payload: dict[str, Any]) -> int:
    for action in payload.get("next_actions") or []:
        params = action.get("params") if isinstance(action, dict) else {}
        if isinstance(params, dict) and _int(params.get("suggested_next_limit")) > 0:
            return _int(params.get("suggested_next_limit"))
    return 0


def run_high_win_iteration(
    *,
    config_path: str | Path = "high-win-iteration.default.yaml",
    alpha_report_paths: list[str | Path] | None = None,
    sweep_report_paths: list[str | Path] | None = None,
    output_dir: str | Path | None = None,
    write_pid_state: bool = True,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    config = _load_yaml(config_path)
    targets = _targets(config)
    gains = _pid_gains(config)
    resolved_alpha_paths = [
        _resolve_project_path(path)
        for path in (alpha_report_paths or [])
        if str(path).strip()
    ] or _report_paths(config, "alpha_reports")
    resolved_sweep_paths = [
        _resolve_project_path(path)
        for path in (sweep_report_paths or [])
        if str(path).strip()
    ] or _report_paths(config, "sweep_reports")

    report_errors: list[dict[str, Any]] = []
    alpha_reports: list[tuple[Path, dict[str, Any]]] = []
    for path in resolved_alpha_paths:
        report, error = _load_json_report(path)
        if error:
            report_errors.append(error)
        elif report is not None:
            alpha_reports.append((path, report))
    sweep_reports: list[tuple[Path, dict[str, Any]]] = []
    for path in resolved_sweep_paths:
        report, error = _load_json_report(path)
        if error:
            report_errors.append(error)
        elif report is not None:
            sweep_reports.append((path, report))

    pid_state_path = _resolve_project_path(
        (config.get("pid") or {}).get("state_path") or "state/high-win-iteration/pid-state.json"
    )
    pid_state = _load_pid_state(pid_state_path)
    alpha_evaluations = [
        _alpha_evaluation(
            path=path,
            report=report,
            peer_reports=alpha_reports,
            targets=targets,
            gains=gains,
            pid_state=pid_state,
        )
        for path, report in alpha_reports
    ]
    sweep_evaluations = [
        _sweep_evaluation(path, report, targets)
        for path, report in sweep_reports
    ]
    if write_pid_state:
        _write_pid_state(pid_state_path, pid_state)

    portfolio_gate = _portfolio_gate(alpha_reports=alpha_reports, targets=targets)
    candidate_config = config.get("candidate_expansion") or {}
    cohort_expansion_candidates = _cohort_expansion_candidates(
        alpha_reports=alpha_reports,
        alpha_evaluations=alpha_evaluations,
        targets=targets,
        gains=gains,
        top_n=_int(candidate_config.get("top_n"), 8),
        min_trade_count=_int(candidate_config.get("min_trade_count"), 10),
        max_limit=_int(candidate_config.get("max_limit"), gains.max_limit),
    ) if bool(candidate_config.get("enabled", True)) else []
    sweep_under_sampled_candidates = _sweep_under_sampled_candidates(
        sweep_evaluations,
        targets,
        min_trade_count=_int(candidate_config.get("min_trade_count"), 10),
        top_n=_int(candidate_config.get("top_n"), 8),
    )

    best_alpha_eval = _best_alpha(alpha_evaluations)
    best_alpha_report = None
    if best_alpha_eval is not None:
        best_path = best_alpha_eval.get("path")
        best_alpha_report = next((report for path, report in alpha_reports if str(path) == best_path), None)
    promotion_allowed = bool(best_alpha_eval and (best_alpha_eval.get("gate") or {}).get("passed"))
    if portfolio_gate.get("enabled") and not portfolio_gate.get("passed"):
        promotion_allowed = False
    safe_to_open_new_entries = promotion_allowed
    execution_recommendation = (
        "paper_or_testnet_candidate_available_require_live_readiness"
        if promotion_allowed
        else "block_new_entries_and_continue_research"
    )
    root = _resolve_project_path(output_dir) if output_dir else HIGH_WIN_ITERATION_DIR
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": _utc_now().isoformat(),
        "mode": "expectancy_research_iteration",
        "safety": {
            "mainnet_live_allowed": False,
            "writes_execution_config": False,
            "opens_orders": False,
            "max_per_trade_risk_pct": targets.max_per_trade_risk_pct,
            "research_gate_only": True,
        },
        "targets": {
            "min_trades": targets.min_trades,
            "min_win_rate": targets.min_win_rate,
            "max_stop_loss_ratio": targets.max_stop_loss_ratio,
            "min_profit_factor": targets.min_profit_factor,
            "min_expectancy_r": targets.min_expectancy_r,
            "min_payoff_ratio": targets.min_payoff_ratio,
            "max_per_trade_risk_pct": targets.max_per_trade_risk_pct,
            "min_promoted_symbols": targets.min_promoted_symbols,
            "required_symbols": list(targets.required_symbols),
        },
        "input_reports": {
            "alpha_reports": [str(path) for path, _report in alpha_reports],
            "sweep_reports": [str(path) for path, _report in sweep_reports],
            "errors": report_errors,
        },
        "alpha_evaluations": alpha_evaluations,
        "sweep_evaluations": sweep_evaluations,
        "cohort_expansion_candidates": cohort_expansion_candidates,
        "sweep_under_sampled_candidates": sweep_under_sampled_candidates,
        "portfolio_gate": portfolio_gate,
        "best_alpha_path": (best_alpha_eval or {}).get("path"),
        "best_alpha_gate": (best_alpha_eval or {}).get("gate"),
        "promotion_allowed": promotion_allowed,
        "safe_to_open_new_entries": safe_to_open_new_entries,
        "execution_recommendation": execution_recommendation,
        "next_actions": _next_actions(
            config=config,
            alpha_reports=[report for _path, report in alpha_reports],
            alpha_evaluations=alpha_evaluations,
            best_alpha_report=best_alpha_report,
            best_alpha_eval=best_alpha_eval,
            sweep_evaluations=sweep_evaluations,
            cohort_expansion_candidates=cohort_expansion_candidates,
            sweep_under_sampled_candidates=sweep_under_sampled_candidates,
            targets=targets,
        ),
        "operator_note": (
            "This controller only adjusts research pressure and batch commands. "
            "It never enables mainnet or sends testnet/live orders."
        ),
    }
    report_path = root / f"{_stamp()}-high-win-iteration.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload
