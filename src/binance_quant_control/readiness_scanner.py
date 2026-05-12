from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analysis import run_analysis
from .binance_api import BinanceAPIError
from .config import CONFIG_DIR, STATE_DIR, ensure_runtime_dirs, load_settings
from .hermes_ai_trader import DEFAULT_BLUEPRINT_PATH, run_hermes_ai_trader
from .live_execution import build_live_execution_plan
from .skipped_signal_journal import append_skipped_signal
from .strategy import load_strategy_config

READINESS_SCAN_DIR = STATE_DIR / "hermes-readiness-scan"
RISK_COMBO_MATRIX_DIR = STATE_DIR / "risk-combo-matrix"
DEFAULT_SCANNER_STRATEGY_CONFIG = CONFIG_DIR / "strategy-live-pilot.yaml"
ACTION_PRIORITY = {
    "execute_ready_dry_run_only": 0,
    "wait_for_kill_switch_clear": 10,
    "repair_exchange_sizing_or_margin": 20,
    "wait_for_portfolio_capacity_or_flat_position": 30,
    "wait_for_market_state": 40,
    "wait_for_signal_quality": 50,
    "repair_strategy_performance_or_route_history": 60,
    "repair_route_history_or_wait_for_quarantine_clear": 70,
    "repair_optimizer_or_market_bot_bridge": 80,
    "repair_private_api_or_symbol_lane": 90,
    "repair_hermes_gate_before_live_readiness": 100,
    "hold_candidate_until_blockers_clear": 110,
    "wait_for_candidate_scan": 120,
}
INTERVAL_HORIZON = {
    "1m": "short",
    "3m": "short",
    "5m": "short",
    "15m": "short",
    "30m": "short",
    "1h": "short",
    "2h": "medium",
    "4h": "medium",
    "6h": "medium",
    "8h": "medium",
    "12h": "medium",
    "1d": "long",
    "3d": "long",
    "1w": "long",
    "1M": "long",
}
RESEARCH_HORIZON_INTERVALS = {
    "short": "15m",
    "medium": "4h",
    "long": "1d",
}
RESEARCH_SMOKE_SWEEP = {
    "limit": 600,
    "grid_mode": "fast",
    "min_test_trades": 10,
    "target_profit_factor": 1.0,
    "min_expectancy_r": 0.0,
    "max_stop_loss_ratio": 55,
    "max_configs": 8,
    "max_walk_forward_validations": 1,
    "top_n": 5,
}
RISK_COMBO_READINESS_STATUSES = {
    "emerging_positive_research_lead",
    "promising_but_under_validated",
    "robust_research_candidate_found",
}


@dataclass(frozen=True, slots=True)
class CandidateScanResult:
    rank: int
    symbol: str
    side: str
    interval: str
    route_id: str
    strategy_family: str
    machine_state: str
    pre_gate_allowed: bool
    scanned: bool
    allowed: bool
    next_action: str
    blocker_taxonomy: dict[str, list[str]]
    warning_taxonomy: dict[str, list[str]]
    live_plan: dict[str, Any] | None
    error: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _signal_from_queue_item(item: dict[str, Any]) -> dict[str, Any]:
    signal = item.get("signal") if isinstance(item.get("signal"), dict) else {}
    return signal


def _latest_json_report(root: Path, pattern: str) -> tuple[dict[str, Any], str]:
    candidates = sorted(
        root.glob(pattern) if root.exists() else [],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {}, ""
    latest = candidates[0]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, str(latest)
    return payload if isinstance(payload, dict) else {}, str(latest)


def _risk_combo_surface_key(surface: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(surface.get("symbol") or "").upper(),
        str(surface.get("target_side") or "").upper(),
        str(surface.get("target_interval") or ""),
    )


def _risk_combo_surface_to_queue_item(
    surface: dict[str, Any],
    *,
    rank: int,
    matrix_path: str,
) -> dict[str, Any] | None:
    symbol, side, interval = _risk_combo_surface_key(surface)
    if not symbol or side not in {"BUY", "SELL"} or not interval:
        return None
    status = str(surface.get("research_status") or "")
    if status not in RISK_COMBO_READINESS_STATUSES and not surface.get("robust_recovery_gate_passed"):
        return None
    route_id = str(surface.get("route_id") or "risk-combo-research")
    return {
        "rank": rank,
        "signal": {
            "symbol": symbol,
            "side": side,
            "interval": interval,
            "strategy_family": "risk_combo_matrix",
            "route_id": route_id,
            "source": "risk_combo_matrix",
            "risk_combo_research_status": status,
            "risk_combo_research_lead_only": bool(surface.get("research_lead_only")),
            "risk_combo_promotion_eligible": bool(surface.get("promotion_eligible")),
            "risk_combo_source_report_path": surface.get("source_report_path"),
            "risk_combo_matrix_path": matrix_path,
        },
        "machine_state": "candidate_ready",
        "open_order_gate": {"allowed": True, "blockers": []},
        "source": "risk_combo_matrix",
    }


def _risk_combo_matrix_queue_items(
    *,
    existing_keys: set[tuple[str, str, str]],
    matrix_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
    matrix_payload, matrix_path = _latest_json_report(
        matrix_dir or RISK_COMBO_MATRIX_DIR,
        "*-risk-combo-matrix.json",
    )
    if not matrix_payload:
        return [], matrix_path

    raw_surfaces: list[dict[str, Any]] = []
    best_surface = matrix_payload.get("best_surface")
    if isinstance(best_surface, dict):
        raw_surfaces.append(best_surface)
    for section_name in ("side_summary", "horizon_summary"):
        section = matrix_payload.get(section_name)
        if not isinstance(section, dict):
            continue
        for row in section.values():
            if not isinstance(row, dict):
                continue
            for key in ("best_surface", "best_emerging_surface"):
                surface = row.get(key)
                if isinstance(surface, dict):
                    raw_surfaces.append(surface)

    seen = set(existing_keys)
    queue_items: list[dict[str, Any]] = []
    for surface in raw_surfaces:
        surface_key = _risk_combo_surface_key(surface)
        if surface_key in seen:
            continue
        item = _risk_combo_surface_to_queue_item(
            surface,
            rank=50 + len(queue_items),
            matrix_path=matrix_path,
        )
        if item is None:
            continue
        seen.add(surface_key)
        queue_items.append(item)
    return queue_items, matrix_path


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return None
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _text_taxonomy(items: list[str]) -> dict[str, list[str]]:
    taxonomy: dict[str, list[str]] = {
        "kill_switch": [],
        "market_state": [],
        "signal_quality": [],
        "strategy_performance": [],
        "exchange_constraints": [],
        "route_history": [],
        "optimizer_legacy": [],
        "portfolio": [],
        "execution_api": [],
        "unknown": [],
    }
    for item in items:
        lowered = item.lower()
        if "kill-switch" in lowered or "trading is paused" in lowered:
            taxonomy["kill_switch"].append(item)
        elif (
            "exchange minimum" in lowered
            or "min_notional" in lowered
            or "notional" in lowered
            or "quantity" in lowered
            or "margin" in lowered
            or "balance" in lowered
            or "leverage" in lowered
            or "liquidation" in lowered
        ):
            taxonomy["exchange_constraints"].append(item)
        elif "optimizer" in lowered or "promotion" in lowered:
            taxonomy["optimizer_legacy"].append(item)
        elif (
            "quarantined" in lowered
            or "route/side" in lowered
            or "route-side" in lowered
            or "route side" in lowered
            or "historical" in lowered
        ):
            taxonomy["route_history"].append(item)
        elif (
            "open regular order" in lowered
            or "open order" in lowered
            or "non-flat" in lowered
            or "position" in lowered
            or "daily trade" in lowered
            or "portfolio" in lowered
        ):
            taxonomy["portfolio"].append(item)
        elif (
            "adx" in lowered
            or "trend" in lowered
            or "volume" in lowered
            or "obv" in lowered
            or "flow" in lowered
            or "spread" in lowered
            or "volatility" in lowered
            or "book" in lowered
            or "slippage" in lowered
        ):
            taxonomy["market_state"].append(item)
        elif (
            "profit factor" in lowered
            or "expectancy" in lowered
            or "payoff" in lowered
            or "win rate" in lowered
            or "stop-loss" in lowered
            or "sample" in lowered
            or "recent" in lowered
            or "net pnl" in lowered
        ):
            taxonomy["strategy_performance"].append(item)
        elif (
            "analysis score" in lowered
            or "convergence" in lowered
            or "strategy policy" in lowered
            or "entry" in lowered
            or "reward/risk" in lowered
            or "composite" in lowered
            or "structure" in lowered
            or "quality" in lowered
            or "hold" in lowered
        ):
            taxonomy["signal_quality"].append(item)
        elif "api" in lowered or "auth" in lowered or "invalid symbol" in lowered or "rejected" in lowered:
            taxonomy["execution_api"].append(item)
        else:
            taxonomy["unknown"].append(item)
    return {key: value for key, value in taxonomy.items() if value}


def _next_action(*, allowed: bool, scanned: bool, taxonomy: dict[str, list[str]], pre_gate_allowed: bool) -> str:
    if allowed:
        return "execute_ready_dry_run_only"
    if not scanned:
        if taxonomy.get("route_history"):
            return "repair_route_history_or_wait_for_quarantine_clear"
        if taxonomy.get("strategy_performance"):
            return "repair_strategy_performance_or_route_history"
        if taxonomy.get("portfolio"):
            return "wait_for_portfolio_capacity_or_flat_position"
        if not pre_gate_allowed:
            return "repair_hermes_gate_before_live_readiness"
        return "wait_for_candidate_scan"
    if taxonomy.get("kill_switch"):
        return "wait_for_kill_switch_clear"
    if taxonomy.get("exchange_constraints"):
        return "repair_exchange_sizing_or_margin"
    if taxonomy.get("portfolio"):
        return "wait_for_portfolio_capacity_or_flat_position"
    if taxonomy.get("market_state"):
        return "wait_for_market_state"
    if taxonomy.get("signal_quality"):
        return "wait_for_signal_quality"
    if taxonomy.get("strategy_performance"):
        return "repair_strategy_performance_or_route_history"
    if taxonomy.get("route_history"):
        return "repair_route_history_or_wait_for_quarantine_clear"
    if taxonomy.get("optimizer_legacy"):
        return "repair_optimizer_or_market_bot_bridge"
    if taxonomy.get("execution_api"):
        return "repair_private_api_or_symbol_lane"
    return "hold_candidate_until_blockers_clear"


def _non_kill_taxonomy(taxonomy: dict[str, list[str]]) -> dict[str, list[str]]:
    return {key: value for key, value in taxonomy.items() if key != "kill_switch" and value}


def _is_ready_after_global_unlock(result: CandidateScanResult) -> bool:
    return (
        result.scanned
        and result.pre_gate_allowed
        and bool(result.blocker_taxonomy.get("kill_switch"))
        and not _non_kill_taxonomy(result.blocker_taxonomy)
    )


def _candidate_symbol(result: CandidateScanResult) -> str:
    return f"{result.symbol}:{result.side}"


def _unlock_candidate_summary(result: CandidateScanResult) -> dict[str, Any]:
    live_plan = result.live_plan if isinstance(result.live_plan, dict) else {}
    market_bot_gate = live_plan.get("market_bot_gate") if isinstance(live_plan.get("market_bot_gate"), dict) else {}
    return {
        "rank": result.rank,
        "symbol": result.symbol,
        "side": result.side,
        "route_id": result.route_id,
        "strategy_family": result.strategy_family,
        "next_action_after_global_unlock": "execute_ready_dry_run_only",
        "analysis_score": live_plan.get("analysis_score"),
        "analysis_convergence": live_plan.get("analysis_convergence"),
        "planned_account_risk_pct": live_plan.get("planned_account_risk_pct"),
        "gross_notional_usdt": live_plan.get("gross_notional_usdt"),
        "min_notional_usdt": live_plan.get("min_notional_usdt"),
        "market_bot_allowed": market_bot_gate.get("allowed"),
    }


def _command_parts_to_string(parts: list[str]) -> str:
    return " ".join(parts)


def _build_execution_ticket(
    *,
    selected: CandidateScanResult | None,
    strategy_config: str | Path,
    market: str,
    limit: int,
    margin_notional_usdt: float | None,
    execution_mode: str,
) -> dict[str, Any] | None:
    if selected is None or not selected.allowed or not isinstance(selected.live_plan, dict):
        return None
    plan = selected.live_plan
    interval = selected.interval or "4h"
    readiness_command = [
        ".venv/bin/binance-quant-control",
        "live-readiness",
        "--strategy-config",
        str(strategy_config),
        "--symbol",
        selected.symbol,
        "--market",
        market,
        "--interval",
        interval,
        "--side",
        selected.side,
        "--execution-mode",
        execution_mode,
        "--compact",
    ]
    execution_command = [
        ".venv/bin/binance-quant-control",
        "live-pilot",
        "--strategy-config",
        str(strategy_config),
        "--symbol",
        selected.symbol,
        "--market",
        market,
        "--interval",
        interval,
        "--side",
        selected.side,
        "--execution-mode",
        execution_mode,
        "--execute",
        "--compact",
    ]
    if limit > 0:
        readiness_command.extend(["--limit", str(limit)])
        execution_command.extend(["--limit", str(limit)])
    if margin_notional_usdt is not None:
        formatted_margin = f"{float(margin_notional_usdt):.8g}"
        readiness_command.extend(["--margin-notional-usdt", formatted_margin])
        execution_command.extend(["--margin-notional-usdt", formatted_margin])

    professional_gate = plan.get("professional_entry_gate") if isinstance(plan.get("professional_entry_gate"), dict) else {}
    layers = professional_gate.get("layers") if isinstance(professional_gate.get("layers"), dict) else {}
    strategy_performance = layers.get("strategy_performance") if isinstance(layers.get("strategy_performance"), dict) else {}
    market_bot_gate = plan.get("market_bot_gate") if isinstance(plan.get("market_bot_gate"), dict) else {}
    matched_row = market_bot_gate.get("matched_row") if isinstance(market_bot_gate.get("matched_row"), dict) else {}

    return {
        "state": "ready_for_operator_testnet_execution",
        "opens_orders": False,
        "requires_explicit_operator_execute": True,
        "selected_rank": selected.rank,
        "symbol": selected.symbol,
        "side": selected.side,
        "market": market,
        "interval": interval,
        "route_id": selected.route_id,
        "strategy_family": selected.strategy_family,
        "strategy_config": str(strategy_config),
        "risk_snapshot": {
            "quantity": plan.get("quantity"),
            "price": plan.get("price"),
            "leverage": plan.get("leverage"),
            "margin_notional_usdt": plan.get("margin_notional_usdt"),
            "gross_notional_usdt": plan.get("gross_notional_usdt"),
            "min_notional_usdt": plan.get("min_notional_usdt"),
            "planned_account_risk_pct": plan.get("planned_account_risk_pct"),
            "analysis_score": plan.get("analysis_score"),
            "analysis_convergence": plan.get("analysis_convergence"),
            "adx_value": plan.get("adx_value"),
        },
        "expectancy_evidence": {
            "scope": strategy_performance.get("scope"),
            "trade_count": strategy_performance.get("count"),
            "profit_factor": strategy_performance.get("profit_factor"),
            "expectancy_r": strategy_performance.get("expectancy_r"),
            "payoff_ratio": strategy_performance.get("payoff_ratio"),
            "win_rate": strategy_performance.get("win_rate"),
            "break_even_win_rate": strategy_performance.get("break_even_win_rate"),
            "source_report_path": strategy_performance.get("source_report_path")
            or market_bot_gate.get("report_path"),
            "feature_manifest_hash": market_bot_gate.get("feature_manifest_hash"),
            "cohort_id": strategy_performance.get("source_cohort_id") or matched_row.get("cohort_id"),
        },
        "preflight_command": _command_parts_to_string(readiness_command),
        "operator_testnet_execute_command": _command_parts_to_string(execution_command),
        "post_execute_required_checks": [
            "rescan positions and open orders",
            "verify stop-loss and staged take-profit protection coverage",
            "append execution outcome to live order journal",
        ],
    }


def _build_machine_action_queue(
    *,
    results: list[CandidateScanResult],
    selected: CandidateScanResult | None,
) -> list[dict[str, Any]]:
    if selected:
        return [
            {
                "priority": ACTION_PRIORITY["execute_ready_dry_run_only"],
                "action": "execute_ready_dry_run_only",
                "scope": "candidate",
                "candidate_count": 1,
                "candidates": [_candidate_symbol(selected)],
                "blocker_classes": [],
                "note": "Dry-run readiness passed. This still does not execute orders.",
            }
        ]

    action_rows: dict[str, dict[str, Any]] = {}
    kill_candidates = [
        _candidate_symbol(result)
        for result in results
        if result.blocker_taxonomy.get("kill_switch")
    ]
    unlock_ready_candidates = [
        _candidate_symbol(result)
        for result in results
        if _is_ready_after_global_unlock(result)
    ]
    if kill_candidates:
        action_rows["wait_for_kill_switch_clear"] = {
            "priority": ACTION_PRIORITY["wait_for_kill_switch_clear"],
            "action": "wait_for_kill_switch_clear",
            "scope": "global",
            "candidate_count": len(kill_candidates),
            "candidates": kill_candidates,
            "unlock_ready_candidate_count": len(unlock_ready_candidates),
            "unlock_ready_candidates": unlock_ready_candidates,
            "blocker_classes": ["kill_switch"],
            "note": (
                "Global kill-switch blocks all execution paths. "
                f"{len(unlock_ready_candidates)} candidate(s) have no other hard blocker; "
                "rescan after it clears."
            ),
        }

    for result in results:
        taxonomy = _non_kill_taxonomy(result.blocker_taxonomy)
        if not taxonomy and not result.blocker_taxonomy:
            taxonomy = {"unknown": ["candidate did not pass but no blocker taxonomy was produced"]}
        if not taxonomy:
            continue
        action = _next_action(
            allowed=False,
            scanned=result.scanned,
            taxonomy=taxonomy,
            pre_gate_allowed=result.pre_gate_allowed,
        )
        row = action_rows.setdefault(
            action,
            {
                "priority": ACTION_PRIORITY.get(action, 999),
                "action": action,
                "scope": "candidate",
                "candidate_count": 0,
                "candidates": [],
                "blocker_classes": set(),
                "note": "",
            },
        )
        row["candidate_count"] += 1
        row["candidates"].append(_candidate_symbol(result))
        row["blocker_classes"].update(taxonomy.keys())

    normalized: list[dict[str, Any]] = []
    for row in action_rows.values():
        blocker_classes = row.get("blocker_classes")
        if isinstance(blocker_classes, set):
            row["blocker_classes"] = sorted(blocker_classes)
        normalized.append(row)
    return sorted(normalized, key=lambda row: (int(row.get("priority") or 999), str(row.get("action") or "")))


def _compact_plan(plan: dict[str, Any]) -> dict[str, Any]:
    challenge = plan.get("challenge") if isinstance(plan.get("challenge"), dict) else {}
    return {
        "allowed": plan.get("allowed"),
        "symbol": plan.get("symbol"),
        "market": plan.get("market"),
        "side": plan.get("side"),
        "quantity": plan.get("quantity"),
        "price": plan.get("price"),
        "leverage": plan.get("leverage"),
        "margin_notional_usdt": plan.get("margin_notional_usdt"),
        "gross_notional_usdt": plan.get("gross_notional_usdt"),
        "min_notional_usdt": plan.get("min_notional_usdt"),
        "planned_account_risk_pct": plan.get("planned_account_risk_pct"),
        "analysis_score": plan.get("analysis_score"),
        "analysis_convergence": plan.get("analysis_convergence"),
        "adx_value": plan.get("adx_value"),
        "execution_mode": plan.get("execution_mode"),
        "violations": plan.get("violations") or [],
        "warnings": plan.get("warnings") or [],
        "professional_entry_gate": plan.get("professional_entry_gate"),
        "market_bot_gate": challenge.get("market_bot_gate"),
        "route_quarantine": challenge.get("route_quarantine"),
        "route_side_risk": challenge.get("route_side_risk"),
        "historical_signal_risk": challenge.get("historical_signal_risk"),
    }


def _scan_candidate_live_readiness(
    *,
    item: dict[str, Any],
    strategy_config: str | Path,
    market: str,
    limit: int,
    margin_notional_usdt: float | None,
    execution_mode: str,
) -> CandidateScanResult:
    signal = _signal_from_queue_item(item)
    symbol = str(signal.get("symbol") or "UNRESOLVED").upper()
    side = str(signal.get("side") or "BUY").upper()
    interval = str(signal.get("interval") or "")
    route_id = str(signal.get("route_id") or "unrouted")
    strategy_family = str(signal.get("strategy_family") or "unknown")
    open_gate = item.get("open_order_gate") if isinstance(item.get("open_order_gate"), dict) else {}
    pre_gate_allowed = bool(open_gate.get("allowed"))
    machine_state = str(item.get("machine_state") or "")
    if not pre_gate_allowed or machine_state != "candidate_ready":
        blockers = [str(entry) for entry in open_gate.get("blockers") or []]
        taxonomy = _text_taxonomy(blockers)
        return CandidateScanResult(
            rank=int(item.get("rank") or 0),
            symbol=symbol,
            side=side,
            interval=interval,
            route_id=route_id,
            strategy_family=strategy_family,
            machine_state=machine_state,
            pre_gate_allowed=pre_gate_allowed,
            scanned=False,
            allowed=False,
            next_action=_next_action(
                allowed=False,
                scanned=False,
                taxonomy=taxonomy,
                pre_gate_allowed=pre_gate_allowed,
            ),
            blocker_taxonomy=taxonomy,
            warning_taxonomy={},
            live_plan=None,
            error="",
        )

    settings = load_settings()
    strategy = load_strategy_config(strategy_config)
    resolved_market = market or strategy.defaults.market
    resolved_interval = interval or strategy.defaults.interval
    resolved_limit = limit or strategy.defaults.limit
    analysis, _artifacts = run_analysis(
        settings,
        symbol=symbol,
        market=resolved_market,
        interval=resolved_interval,
        limit=max(resolved_limit, 240),
        use_blave=strategy.defaults.use_blave,
        render_chart_flag=False,
        strategy=strategy,
    )
    plan = build_live_execution_plan(
        settings,
        strategy,
        analysis,
        side_override=side,
        margin_notional_usdt=margin_notional_usdt,
        execution_mode=execution_mode,
    )
    plan_payload = plan.to_dict()
    blockers = [str(entry) for entry in plan_payload.get("violations") or []]
    warnings = [str(entry) for entry in plan_payload.get("warnings") or []]
    taxonomy = _text_taxonomy(blockers)
    return CandidateScanResult(
        rank=int(item.get("rank") or 0),
        symbol=symbol,
        side=side,
        interval=resolved_interval,
        route_id=route_id,
        strategy_family=strategy_family,
        machine_state=machine_state,
        pre_gate_allowed=pre_gate_allowed,
        scanned=True,
        allowed=bool(plan_payload.get("allowed")),
        next_action=_next_action(
            allowed=bool(plan_payload.get("allowed")),
            scanned=True,
            taxonomy=taxonomy,
            pre_gate_allowed=pre_gate_allowed,
        ),
        blocker_taxonomy=taxonomy,
        warning_taxonomy=_text_taxonomy(warnings),
        live_plan=_compact_plan(plan_payload),
        error="",
    )


def _scan_error_result(item: dict[str, Any], error: str) -> CandidateScanResult:
    signal = _signal_from_queue_item(item)
    taxonomy = _text_taxonomy([error])
    return CandidateScanResult(
        rank=int(item.get("rank") or 0),
        symbol=str(signal.get("symbol") or "UNRESOLVED").upper(),
        side=str(signal.get("side") or "BUY").upper(),
        interval=str(signal.get("interval") or ""),
        route_id=str(signal.get("route_id") or "unrouted"),
        strategy_family=str(signal.get("strategy_family") or "unknown"),
        machine_state=str(item.get("machine_state") or ""),
        pre_gate_allowed=bool((item.get("open_order_gate") or {}).get("allowed"))
        if isinstance(item.get("open_order_gate"), dict)
        else False,
        scanned=True,
        allowed=False,
        next_action=_next_action(
            allowed=False,
            scanned=True,
            taxonomy=taxonomy,
            pre_gate_allowed=True,
        ),
        blocker_taxonomy=taxonomy,
        warning_taxonomy={},
        live_plan=None,
        error=error,
    )


def _flatten_taxonomy(taxonomy: dict[str, list[str]]) -> list[str]:
    blockers: list[str] = []
    for values in taxonomy.values():
        blockers.extend(str(item) for item in values)
    return list(dict.fromkeys(blockers))


def _readiness_denial_gate(result: CandidateScanResult) -> str:
    if result.error:
        return "ai_readiness_scan_error"
    if not result.pre_gate_allowed or not result.scanned:
        return "ai_readiness_pre_gate_skip"
    return "ai_readiness_live_plan_denial"


def _readiness_metric(result: CandidateScanResult, key: str) -> float | None:
    plan = result.live_plan if isinstance(result.live_plan, dict) else {}
    professional_gate = plan.get("professional_entry_gate") if isinstance(plan.get("professional_entry_gate"), dict) else {}
    layers = professional_gate.get("layers") if isinstance(professional_gate.get("layers"), dict) else {}
    strategy_performance = layers.get("strategy_performance") if isinstance(layers.get("strategy_performance"), dict) else {}
    market_bot_gate = plan.get("market_bot_gate") if isinstance(plan.get("market_bot_gate"), dict) else {}
    matched_row = market_bot_gate.get("matched_row") if isinstance(market_bot_gate.get("matched_row"), dict) else {}
    value = strategy_performance.get(key)
    if value is None:
        value = matched_row.get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _research_candidate_row(result: CandidateScanResult) -> dict[str, Any]:
    plan = result.live_plan if isinstance(result.live_plan, dict) else {}
    professional_gate = plan.get("professional_entry_gate") if isinstance(plan.get("professional_entry_gate"), dict) else {}
    layers = professional_gate.get("layers") if isinstance(professional_gate.get("layers"), dict) else {}
    execution_quality = layers.get("execution_quality") if isinstance(layers.get("execution_quality"), dict) else {}
    market_bot_gate = plan.get("market_bot_gate") if isinstance(plan.get("market_bot_gate"), dict) else {}
    matched_row = market_bot_gate.get("matched_row") if isinstance(market_bot_gate.get("matched_row"), dict) else {}
    blocker_classes = sorted(result.blocker_taxonomy)
    performance_pf = _readiness_metric(result, "profit_factor")
    performance_expectancy = _readiness_metric(result, "expectancy_r")
    stop_loss_ratio = _readiness_metric(result, "stop_loss_ratio")
    sample_count = _readiness_metric(result, "count")
    horizon = INTERVAL_HORIZON.get(result.interval, "unknown")
    promotion_gaps = []
    if performance_pf is None or performance_pf < 1.0:
        promotion_gaps.append("profit_factor_below_1.0_or_missing")
    if performance_expectancy is None or performance_expectancy <= 0.0:
        promotion_gaps.append("expectancy_r_not_positive")
    if sample_count is None or sample_count < 30.0:
        promotion_gaps.append("sample_count_below_30")
    if stop_loss_ratio is not None and stop_loss_ratio > 0.55:
        promotion_gaps.append("stop_loss_ratio_above_55pct")
    quality_score = 0.0
    if result.scanned:
        quality_score += 20.0
    if result.pre_gate_allowed:
        quality_score += 15.0
    if result.allowed:
        quality_score += 25.0
    if performance_pf is not None:
        quality_score += min(20.0, max(0.0, performance_pf) * 10.0)
    if performance_expectancy is not None and performance_expectancy > 0:
        quality_score += min(10.0, performance_expectancy * 20.0)
    if stop_loss_ratio is not None:
        quality_score += max(0.0, 10.0 - max(0.0, stop_loss_ratio - 45.0) / 5.0)
    near_ready_market_only = (
        result.scanned
        and not result.allowed
        and not promotion_gaps
        and set(blocker_classes) == {"market_state"}
    )
    return {
        "rank": result.rank,
        "symbol": result.symbol,
        "side": result.side,
        "interval": result.interval,
        "route_id": result.route_id,
        "strategy_family": result.strategy_family,
        "horizon": horizon,
        "research_status": "reviewable_signal" if result.scanned else "pre_gate_rejected_signal",
        "trade_readiness_allowed": result.allowed,
        "next_research_action": result.next_action,
        "blocker_classes": blocker_classes,
        "direction_evidence": {
            "analysis_score": plan.get("analysis_score"),
            "analysis_convergence": plan.get("analysis_convergence"),
            "adx_value": plan.get("adx_value"),
            "machine_state": result.machine_state,
            "pre_gate_allowed": result.pre_gate_allowed,
        },
        "expectancy_metrics": {
            "profit_factor": performance_pf,
            "expectancy_r": performance_expectancy,
            "payoff_ratio": _readiness_metric(result, "payoff_ratio"),
            "stop_loss_ratio": stop_loss_ratio,
            "sample_count": sample_count,
            "market_bot_profit_factor": matched_row.get("profit_factor"),
            "market_bot_expectancy_r": matched_row.get("expectancy_r"),
            "market_bot_sample_count": matched_row.get("count"),
        },
        "risk_metrics": {
            "planned_account_risk_pct": plan.get("planned_account_risk_pct"),
            "gross_notional_usdt": plan.get("gross_notional_usdt"),
            "min_notional_usdt": plan.get("min_notional_usdt"),
            "reward_risk": execution_quality.get("reward_risk"),
            "net_profit_to_risk": execution_quality.get("net_profit_to_risk"),
        },
        "research_quality_score": round(quality_score, 4),
        "positive_expectancy_gap": promotion_gaps,
        "near_ready_market_only": near_ready_market_only,
        "promotion_boundary": "research_only_not_trade_permission",
    }


def _build_research_candidate_report(results: list[CandidateScanResult]) -> dict[str, Any]:
    rows = [_research_candidate_row(result) for result in results]
    rows.sort(key=lambda row: float(row.get("research_quality_score") or 0.0), reverse=True)
    side_counts: dict[str, int] = {}
    reviewable_side_counts: dict[str, int] = {}
    horizon_counts: dict[str, int] = {}
    reviewable_horizon_counts: dict[str, int] = {}
    side_horizon_counts: dict[str, int] = {}
    reviewable_side_horizon_counts: dict[str, int] = {}
    for row in rows:
        side = str(row.get("side") or "UNKNOWN")
        horizon = str(row.get("horizon") or "unknown")
        side_horizon = f"{side.lower()}_{horizon}"
        side_counts[side] = side_counts.get(side, 0) + 1
        horizon_counts[horizon] = horizon_counts.get(horizon, 0) + 1
        side_horizon_counts[side_horizon] = side_horizon_counts.get(side_horizon, 0) + 1
        if row.get("research_status") == "reviewable_signal":
            reviewable_side_counts[side] = reviewable_side_counts.get(side, 0) + 1
            reviewable_horizon_counts[horizon] = reviewable_horizon_counts.get(horizon, 0) + 1
            reviewable_side_horizon_counts[side_horizon] = reviewable_side_horizon_counts.get(side_horizon, 0) + 1
    coverage_gaps = []
    for side in ("BUY", "SELL"):
        if reviewable_side_counts.get(side, 0) == 0:
            coverage_gaps.append(f"missing_reviewable_{side.lower()}_research_candidate")
    for horizon in ("short", "medium", "long"):
        if reviewable_horizon_counts.get(horizon, 0) == 0:
            coverage_gaps.append(f"missing_reviewable_{horizon}_horizon_candidate")
    for side in ("BUY", "SELL"):
        for horizon in ("short", "medium", "long"):
            side_horizon = f"{side.lower()}_{horizon}"
            if reviewable_side_horizon_counts.get(side_horizon, 0) == 0:
                coverage_gaps.append(f"missing_reviewable_{side_horizon}_research_candidate")
    if not any(row.get("trade_readiness_allowed") for row in rows):
        coverage_gaps.append("no_trade_readiness_allowed_candidate")
    expansion_plan: list[dict[str, Any]] = []
    seed_symbols = [str(row.get("symbol")) for row in rows if str(row.get("symbol") or "UNRESOLVED") != "UNRESOLVED"]
    seed_symbol = seed_symbols[0] if seed_symbols else "BTCUSDT"
    for side in ("BUY", "SELL"):
        for horizon, interval in RESEARCH_HORIZON_INTERVALS.items():
            side_horizon = f"{side.lower()}_{horizon}"
            if reviewable_side_horizon_counts.get(side_horizon, 0) > 0:
                continue
            expansion_plan.append(
                {
                    "surface": f"{side.lower()}_{horizon}_research",
                    "target_side": side,
                    "horizon": horizon,
                    "target_interval": interval,
                    "seed_symbol": seed_symbol,
                    "purpose": "research_only_candidate_generation",
                    "command": (
                        "openclaw-quantctl risk-combo-sweep "
                        f"--symbols {seed_symbol} "
                        f"--target-side {side} --target-interval {interval} "
                        f"--limit {RESEARCH_SMOKE_SWEEP['limit']} "
                        f"--grid-mode {RESEARCH_SMOKE_SWEEP['grid_mode']} "
                        f"--min-test-trades {RESEARCH_SMOKE_SWEEP['min_test_trades']} "
                        "--target-profit-factor 1.0 --min-expectancy-r 0.0 "
                        f"--max-stop-loss-ratio {RESEARCH_SMOKE_SWEEP['max_stop_loss_ratio']} "
                        f"--max-configs {RESEARCH_SMOKE_SWEEP['max_configs']} "
                        f"--max-walk-forward-validations {RESEARCH_SMOKE_SWEEP['max_walk_forward_validations']} "
                        f"--top-n {RESEARCH_SMOKE_SWEEP['top_n']} --skip-news --compact"
                    ),
                    "command_note": (
                        "Quick bounded discovery sweep only; use it to surface auditable research candidates, then "
                        "rerun wider validation before promotion. target_side and target_interval are research-only "
                        "sweep controls and do not change live readiness or execution config."
                    ),
                    "smoke_sweep_budget": RESEARCH_SMOKE_SWEEP,
                    "promotion_boundary": "does_not_change_live_readiness_or_mainnet_permission",
                }
            )
    near_ready_candidates = [row for row in rows if row.get("near_ready_market_only")]
    return {
        "mode": "research_candidate_report_v1",
        "objective": "surface_auditable_buy_sell_research_candidates_without_live_permission",
        "candidate_count": len(rows),
        "reviewable_candidate_count": sum(1 for row in rows if row.get("research_status") == "reviewable_signal"),
        "trade_allowed_count": sum(1 for row in rows if row.get("trade_readiness_allowed")),
        "near_ready_count": len(near_ready_candidates),
        "side_counts": side_counts,
        "reviewable_side_counts": reviewable_side_counts,
        "horizon_counts": horizon_counts,
        "reviewable_horizon_counts": reviewable_horizon_counts,
        "side_horizon_counts": side_horizon_counts,
        "reviewable_side_horizon_counts": reviewable_side_horizon_counts,
        "coverage_gaps": coverage_gaps,
        "research_expansion_plan": expansion_plan[:12],
        "research_next_actions": [
            "expand_short_and_long_interval_research_lanes"
            if any("horizon" in gap for gap in coverage_gaps)
            else "keep_horizon_mix_under_observation",
            "repair_or_expand_short_side_candidate_generation"
            if "missing_reviewable_sell_research_candidate" in coverage_gaps
            else "keep_short_side_under_backtest_review",
            "improve_expectancy_before_promotion",
        ],
        "expectancy_improvement_targets": {
            "profit_factor_min": 1.0,
            "expectancy_r_min": 0.0,
            "sample_count_min": 30,
            "stop_loss_ratio_max": 0.55,
            "risk_ceiling_pct": 0.025,
        },
        "top_candidates": rows[:10],
        "near_ready_candidates": near_ready_candidates[:5],
        "promotion_boundary": {
            "mainnet_live_allowed": False,
            "opens_orders": False,
            "writes_execution_config": False,
            "requires_positive_expectancy_and_readiness_before_trade": True,
        },
    }


def _record_readiness_denials(
    *,
    results: list[CandidateScanResult],
    path: Path,
    execution_mode: str,
) -> int:
    count = 0
    for result in results:
        if result.allowed:
            continue
        blockers = _flatten_taxonomy(result.blocker_taxonomy)
        if result.error and result.error not in blockers:
            blockers.append(result.error)
        if not blockers:
            blockers = ["candidate did not pass readiness"]
        plan = result.live_plan if isinstance(result.live_plan, dict) else {}
        signal_score = plan.get("analysis_score")
        try:
            signal_score_value = float(signal_score) if signal_score is not None else None
        except (TypeError, ValueError):
            signal_score_value = None
        append_skipped_signal(
            symbol=result.symbol,
            side=result.side,
            route_id=result.route_id,
            strategy_family=result.strategy_family,
            gate=_readiness_denial_gate(result),
            blockers=blockers,
            signal_score=signal_score_value,
            expectancy_r=_readiness_metric(result, "expectancy_r"),
            payoff_ratio=_readiness_metric(result, "payoff_ratio"),
            metadata={
                "rank": result.rank,
                "machine_state": result.machine_state,
                "pre_gate_allowed": result.pre_gate_allowed,
                "scanned": result.scanned,
                "next_action": result.next_action,
                "execution_mode": execution_mode,
                "warning_taxonomy": result.warning_taxonomy,
            },
            path=path,
        )
        count += 1
    return count


def run_ai_readiness_scan(
    *,
    blueprint_config: str | Path = DEFAULT_BLUEPRINT_PATH,
    strategy_config: str | Path = DEFAULT_SCANNER_STRATEGY_CONFIG,
    output_dir: str | Path | None = None,
    market: str = "futures",
    limit: int = 0,
    margin_notional_usdt: float | None = None,
    execution_mode: str = "testnet_exploration",
    max_candidates: int = 0,
    exclude_symbols: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    root_dir = Path(output_dir).expanduser().resolve() if output_dir else READINESS_SCAN_DIR
    root_dir.mkdir(parents=True, exist_ok=True)
    hermes_payload = run_hermes_ai_trader(
        blueprint_config=blueprint_config,
        output_dir=root_dir / "hermes-ai-trader",
    )
    queue = [item for item in hermes_payload.get("candidate_queue") or [] if isinstance(item, dict)]
    queue.sort(key=lambda item: int(item.get("rank") or 999999))
    existing_keys = {
        (
            str(_signal_from_queue_item(item).get("symbol") or "").upper(),
            str(_signal_from_queue_item(item).get("side") or "").upper(),
            str(_signal_from_queue_item(item).get("interval") or ""),
        )
        for item in queue
    }
    risk_combo_queue, risk_combo_matrix_path = _risk_combo_matrix_queue_items(existing_keys=existing_keys)
    queue = [*risk_combo_queue, *queue]
    excluded_symbols = {
        str(symbol or "").upper()
        for symbol in (exclude_symbols or ())
        if str(symbol or "").strip()
    }
    if excluded_symbols:
        queue = [
            item
            for item in queue
            if str(_signal_from_queue_item(item).get("symbol") or "").upper() not in excluded_symbols
        ]
    selected_queue: list[dict[str, Any]] = []
    live_scan_count = 0
    for item in queue:
        open_gate = item.get("open_order_gate") if isinstance(item.get("open_order_gate"), dict) else {}
        will_scan_live = bool(open_gate.get("allowed")) and str(item.get("machine_state") or "") == "candidate_ready"
        if max_candidates > 0 and will_scan_live and live_scan_count >= max_candidates:
            continue
        selected_queue.append(item)
        if will_scan_live:
            live_scan_count += 1
    queue = selected_queue

    results: list[CandidateScanResult] = []
    for item in queue:
        try:
            results.append(
                _scan_candidate_live_readiness(
                    item=item,
                    strategy_config=strategy_config,
                    market=market,
                    limit=limit,
                    margin_notional_usdt=margin_notional_usdt,
                    execution_mode=execution_mode,
                )
            )
        except BinanceAPIError as exc:
            results.append(_scan_error_result(item, f"binance-private-api-auth-or-symbol-failed: {exc}"))

    denial_journal_path = root_dir / "readiness-denials.jsonl"
    denial_journal_count = _record_readiness_denials(
        results=results,
        path=denial_journal_path,
        execution_mode=execution_mode,
    )

    selected = next((item for item in results if item.allowed), None)
    ready_after_global_unlock = [
        _unlock_candidate_summary(item)
        for item in results
        if _is_ready_after_global_unlock(item)
    ]
    execution_ticket = _build_execution_ticket(
        selected=selected,
        strategy_config=strategy_config,
        market=market,
        limit=limit,
        margin_notional_usdt=margin_notional_usdt,
        execution_mode=execution_mode,
    )
    hard_blocker_taxonomy: dict[str, list[str]] = {}
    for result in results:
        for key, values in result.blocker_taxonomy.items():
            hard_blocker_taxonomy.setdefault(key, [])
            hard_blocker_taxonomy[key].extend(values)
    hard_blocker_taxonomy = {
        key: list(dict.fromkeys(values))
        for key, values in hard_blocker_taxonomy.items()
        if values
    }

    machine_action_queue = _build_machine_action_queue(results=results, selected=selected)
    research_candidate_report = _build_research_candidate_report(results)

    if not queue:
        next_machine_action = "repair_alpha_gate_or_hermes_candidate_queue"
    elif machine_action_queue:
        next_machine_action = str(machine_action_queue[0].get("action") or "hold_until_candidate_blockers_clear")
    else:
        next_machine_action = "hold_until_candidate_blockers_clear"

    payload = _json_safe({
        "generated_at": _utc_now().isoformat(),
        "mode": "hermes_ai_readiness_scanner_v1",
        "safety": {
            "opens_orders": False,
            "writes_execution_config": False,
            "mainnet_live_allowed": False,
            "execution_mode": execution_mode,
            "requires_operator_execute": True,
        },
        "strategy_config": str(strategy_config),
        "blueprint_config": str(blueprint_config),
        "candidate_count": len(queue),
        "excluded_symbols": sorted(excluded_symbols),
        "scanned_count": sum(1 for item in results if item.scanned),
        "allowed_count": sum(1 for item in results if item.allowed),
        "selected_ready_candidate": selected.to_dict() if selected else None,
        "ready_after_global_unlock_count": len(ready_after_global_unlock),
        "selected_after_global_unlock": ready_after_global_unlock[0] if ready_after_global_unlock else None,
        "ready_after_global_unlock_candidates": ready_after_global_unlock,
        "execution_ticket": execution_ticket,
        "next_machine_action": next_machine_action,
        "machine_action_queue": machine_action_queue,
        "research_candidate_report": research_candidate_report,
        "hard_blocker_taxonomy": hard_blocker_taxonomy,
        "denial_journal_path": str(denial_journal_path),
        "denial_journal_count": denial_journal_count,
        "scan_results": [item.to_dict() for item in results],
        "hermes_ai_trader_report": hermes_payload.get("report_path"),
        "risk_combo_matrix_report": risk_combo_matrix_path or None,
        "risk_combo_matrix_candidate_count": len(risk_combo_queue),
    })
    report_path = root_dir / f"{_stamp()}-hermes-readiness-scan.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    payload["report_path"] = str(report_path)
    return payload
