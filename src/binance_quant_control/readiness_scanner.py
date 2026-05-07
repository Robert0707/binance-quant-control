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
from .strategy import load_strategy_config

READINESS_SCAN_DIR = STATE_DIR / "hermes-readiness-scan"
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
        elif "quarantined" in lowered or "route/side" in lowered or "historical" in lowered:
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
    if not pre_gate_allowed:
        return "repair_hermes_gate_before_live_readiness"
    if not scanned:
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
    if max_candidates > 0:
        queue = queue[:max_candidates]

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
        "scanned_count": sum(1 for item in results if item.scanned),
        "allowed_count": sum(1 for item in results if item.allowed),
        "selected_ready_candidate": selected.to_dict() if selected else None,
        "ready_after_global_unlock_count": len(ready_after_global_unlock),
        "selected_after_global_unlock": ready_after_global_unlock[0] if ready_after_global_unlock else None,
        "ready_after_global_unlock_candidates": ready_after_global_unlock,
        "execution_ticket": execution_ticket,
        "next_machine_action": next_machine_action,
        "machine_action_queue": machine_action_queue,
        "hard_blocker_taxonomy": hard_blocker_taxonomy,
        "scan_results": [item.to_dict() for item in results],
        "hermes_ai_trader_report": hermes_payload.get("report_path"),
    })
    report_path = root_dir / f"{_stamp()}-hermes-readiness-scan.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    payload["report_path"] = str(report_path)
    return payload
