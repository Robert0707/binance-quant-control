from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ai_market_sentinel import run_ai_market_sentinel
from .asset_routing import normalize_symbol, resolve_symbol_route
from .config import CONFIG_DIR, STATE_DIR, ensure_runtime_dirs
from .readiness_scanner import run_ai_readiness_scan
from .risk_combo_sweep import build_risk_combo_matrix_report, run_risk_combo_sweep

DEFAULT_STRATEGY_CONFIG = CONFIG_DIR / "strategy-live-pilot.yaml"
DEFAULT_BLUEPRINT_CONFIG = CONFIG_DIR / "professional-system-blueprint.default.yaml"
NEW_SYMBOL_WORKFLOW_DIR = STATE_DIR / "new-symbol-workflow"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _workflow_commands(
    *,
    symbol: str,
    intervals: list[str],
    sides: list[str],
    research_depth: str,
    max_readiness_candidates: int,
) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = [
        {
            "stage": "route",
            "command": f"openclaw-quantctl route-symbol {symbol}",
            "opens_orders": False,
        },
        {
            "stage": "market_watch",
            "command": f"openclaw-quantctl ai-market-sentinel --symbols {symbol} --skip-readiness --compact",
            "opens_orders": False,
        },
    ]
    for side in sides:
        for interval in intervals:
            commands.append(
                {
                    "stage": "research_sweep",
                    "side": side,
                    "interval": interval,
                    "command": (
                        "openclaw-quantctl risk-combo-sweep "
                        f"--symbols {symbol} --target-side {side} --target-interval {interval} "
                        "--limit 600 --grid-mode fast --min-test-trades 10 "
                        "--target-profit-factor 1.0 --min-expectancy-r 0.0 "
                        "--max-stop-loss-ratio 55 --max-configs 8 "
                        "--max-walk-forward-validations 1 --top-n 5 --skip-news --compact"
                    ),
                    "opens_orders": False,
                    "depth": research_depth,
                }
            )
    commands.extend(
        [
            {
                "stage": "research_matrix",
                "command": "openclaw-quantctl risk-combo-matrix --latest-sweeps 6 --compact",
                "opens_orders": False,
            },
            {
                "stage": "readiness",
                "command": (
                    "openclaw-quantctl ai-readiness-scan "
                    f"--execution-mode testnet_exploration --max-candidates {max_readiness_candidates} --compact"
                ),
                "opens_orders": False,
            },
            {
                "stage": "dashboard",
                "command": "openclaw-quantctl operator-dashboard --compact",
                "opens_orders": False,
            },
        ]
    )
    return commands


def _candidate_symbol(item: dict[str, Any]) -> str:
    return str(item.get("symbol") or "").upper()


def _candidate_side(item: dict[str, Any]) -> str:
    return str(item.get("side") or "").upper()


def _candidate_interval(item: dict[str, Any]) -> str:
    return str(item.get("interval") or "")


def _candidate_key(item: dict[str, Any]) -> str:
    return ":".join(part for part in [_candidate_symbol(item), _candidate_side(item), _candidate_interval(item)] if part)


def _matching_candidates(report: dict[str, Any], symbol: str) -> dict[str, list[dict[str, Any]]]:
    research = report.get("research_candidate_report") if isinstance(report.get("research_candidate_report"), dict) else {}
    top = [item for item in (research.get("top_candidates") or []) if isinstance(item, dict) and _candidate_symbol(item) == symbol]
    near_ready = [
        item
        for item in (research.get("near_ready_candidates") or [])
        if isinstance(item, dict) and _candidate_symbol(item) == symbol
    ]
    ready = []
    selected = report.get("selected_ready_candidate")
    if isinstance(selected, dict) and _candidate_symbol(selected) == symbol:
        ready.append(selected)
    return {"top": top, "near_ready": near_ready, "ready": ready}


def _symbol_outcome(
    *,
    symbol: str,
    route_ok: bool,
    plan_only: bool,
    matching: dict[str, list[dict[str, Any]]],
) -> str:
    if not route_ok:
        return "reject"
    if plan_only:
        return "research_candidate"
    if matching["ready"]:
        return "testnet_ready_candidate"
    if matching["near_ready"]:
        return "near_ready_market_only"
    if matching["top"]:
        return "research_candidate"
    return "reject"


def _outcome_priority(outcome: str) -> int:
    return {
        "testnet_ready_candidate": 0,
        "near_ready_market_only": 1,
        "research_candidate": 2,
        "reject": 3,
    }.get(outcome, 99)


def _compact_sweep(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": payload.get("status"),
        "aggregate": payload.get("aggregate"),
        "best_by_route": payload.get("best_by_route"),
        "recovery_candidates": payload.get("recovery_candidates"),
        "robust_recovery_candidates": payload.get("robust_recovery_candidates"),
        "dataset_errors": payload.get("dataset_errors"),
        "report_path": payload.get("report_path"),
    }


def run_new_symbol_workflow(
    *,
    symbols: list[str] | tuple[str, ...],
    intervals: list[str] | tuple[str, ...] | None = None,
    sides: list[str] | tuple[str, ...] | None = None,
    research_depth: str = "smoke",
    plan_only: bool = False,
    output_dir: str | Path | None = None,
    strategy_config: str | Path = DEFAULT_STRATEGY_CONFIG,
    blueprint_config: str | Path = DEFAULT_BLUEPRINT_CONFIG,
    max_readiness_candidates: int = 6,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    symbol_list = _unique([normalize_symbol(item) for item in symbols if str(item).strip()])
    if not symbol_list:
        raise ValueError("At least one symbol is required.")
    interval_list = _unique([str(item).strip() for item in (intervals or ["15m", "4h", "1d"]) if str(item).strip()])
    side_list = _unique([str(item).strip().upper() for item in (sides or ["BUY", "SELL"]) if str(item).strip()])
    side_list = [item for item in side_list if item in {"BUY", "SELL"}]
    if not interval_list:
        raise ValueError("At least one interval is required.")
    if not side_list:
        raise ValueError("At least one side is required: BUY or SELL.")
    research_depth = str(research_depth or "smoke").strip().lower()
    if research_depth not in {"none", "smoke", "focused"}:
        raise ValueError("research_depth must be none, smoke, or focused.")

    route_results: dict[str, dict[str, Any]] = {}
    for symbol in symbol_list:
        try:
            route = resolve_symbol_route(symbol)
            route_results[symbol] = {
                "status": "ok",
                "route": route.to_dict(),
                "route_id": route.route_id,
                "simulation_mode": route.simulation_mode,
                "validation_summary": route.validation.to_dict(),
            }
        except (OSError, ValueError) as exc:
            route_results[symbol] = {"status": "error", "error": str(exc)}

    sentinel = run_ai_market_sentinel(
        symbols=symbol_list,
        interval="15m",
        limit=160,
        market="futures",
        strategy_config=strategy_config,
        blueprint_config=blueprint_config,
        skip_readiness=True,
        send_telegram=False,
        max_readiness_candidates=0,
    )

    sweep_reports: list[str] = []
    sweeps: list[dict[str, Any]] = []
    if not plan_only and research_depth != "none":
        for side in side_list:
            for interval in interval_list:
                sweep = run_risk_combo_sweep(
                    symbols=symbol_list,
                    limit=600 if research_depth == "smoke" else 1500,
                    grid_mode="fast" if research_depth == "smoke" else "focused",
                    target_side=side,
                    target_interval=interval,
                    target_profit_factor=1.0,
                    min_test_trades=10 if research_depth == "smoke" else 30,
                    max_stop_loss_ratio=55.0,
                    min_expectancy_r=0.0,
                    min_payoff_ratio=0.0 if research_depth == "smoke" else 1.0,
                    max_configs=8 if research_depth == "smoke" else 30,
                    max_walk_forward_validations=1 if research_depth == "smoke" else 6,
                    include_all_route_symbols=False,
                    skip_news=True,
                    top_n=5,
                )
                report_path = sweep.get("report_path")
                if report_path:
                    sweep_reports.append(str(report_path))
                sweeps.append(
                    {
                        "side": side,
                        "interval": interval,
                        "summary": _compact_sweep(sweep),
                    }
                )

    matrix: dict[str, Any] | None = None
    if sweep_reports:
        matrix = build_risk_combo_matrix_report(report_paths=sweep_reports)

    readiness = run_ai_readiness_scan(
        blueprint_config=blueprint_config,
        strategy_config=strategy_config,
        market="futures",
        limit=0,
        margin_notional_usdt=None,
        execution_mode="testnet_exploration",
        max_candidates=max_readiness_candidates,
    )

    symbol_reports: list[dict[str, Any]] = []
    for symbol in symbol_list:
        route_status = route_results.get(symbol) or {}
        matching = _matching_candidates(readiness, symbol)
        outcome = _symbol_outcome(
            symbol=symbol,
            route_ok=route_status.get("status") == "ok",
            plan_only=plan_only,
            matching=matching,
        )
        selected = matching["ready"][0] if matching["ready"] else None
        near_ready = matching["near_ready"][0] if matching["near_ready"] else None
        research = matching["top"][0] if matching["top"] else None
        next_command = None
        if selected:
            side = _candidate_side(selected)
            interval = _candidate_interval(selected) or "4h"
            next_command = (
                "openclaw-quantctl live-readiness "
                f"--symbol {symbol} --side {side} --interval {interval} "
                "--execution-mode testnet_exploration --compact"
            )
        elif near_ready:
            side = _candidate_side(near_ready)
            interval = _candidate_interval(near_ready) or "4h"
            next_command = (
                "openclaw-quantctl live-readiness "
                f"--symbol {symbol} --side {side} --interval {interval} "
                "--execution-mode testnet_exploration --compact"
            )
        elif outcome == "research_candidate":
            next_command = (
                "openclaw-quantctl new-symbol-workflow "
                f"--symbols {symbol} --research-depth focused --compact"
            )
        symbol_reports.append(
            {
                "symbol": symbol,
                "outcome": outcome,
                "route": route_status,
                "candidate_keys": [_candidate_key(item) for item in matching["top"]],
                "near_ready_candidate_keys": [_candidate_key(item) for item in matching["near_ready"]],
                "ready_candidate_keys": [_candidate_key(item) for item in matching["ready"]],
                "selected_candidate": selected,
                "near_ready_candidate": near_ready,
                "research_candidate": research,
                "next_command": next_command,
                "commands": _workflow_commands(
                    symbol=symbol,
                    intervals=interval_list,
                    sides=side_list,
                    research_depth=research_depth,
                    max_readiness_candidates=max_readiness_candidates,
                ),
            }
        )

    overall_outcome = sorted((item["outcome"] for item in symbol_reports), key=_outcome_priority)[0]
    payload: dict[str, Any] = {
        "generated_at": _utc_now().isoformat(),
        "mode": "new_symbol_workflow_v1",
        "objective": "fixed_no_code_new_symbol_analysis_to_testnet_readiness_workflow",
        "safety": {
            "opens_orders": False,
            "cancels_orders": False,
            "closes_positions": False,
            "writes_execution_config": False,
            "mainnet_live_allowed": False,
            "requires_operator_execute_for_testnet": True,
            "risk_ceiling_pct": 0.025,
        },
        "inputs": {
            "symbols": symbol_list,
            "intervals": interval_list,
            "sides": side_list,
            "research_depth": research_depth,
            "plan_only": plan_only,
            "max_readiness_candidates": max_readiness_candidates,
        },
        "outcome": overall_outcome,
        "status_counts": {
            "reject": sum(1 for item in symbol_reports if item["outcome"] == "reject"),
            "research_candidate": sum(1 for item in symbol_reports if item["outcome"] == "research_candidate"),
            "near_ready_market_only": sum(1 for item in symbol_reports if item["outcome"] == "near_ready_market_only"),
            "testnet_ready_candidate": sum(1 for item in symbol_reports if item["outcome"] == "testnet_ready_candidate"),
        },
        "symbols": symbol_reports,
        "sentinel": {
            "report_path": sentinel.get("report_path"),
            "errors": sentinel.get("errors"),
            "trend_symbols": sorted((sentinel.get("trend_state") or {}).keys()),
            "opens_orders": (sentinel.get("safety") or {}).get("opens_orders"),
        },
        "research_sweeps": sweeps,
        "risk_combo_matrix": {
            "report_path": (matrix or {}).get("report_path"),
            "surface_count": (matrix or {}).get("surface_count"),
            "promising_surface_count": (matrix or {}).get("promising_surface_count"),
            "best_surface": (matrix or {}).get("best_surface"),
        }
        if matrix is not None
        else None,
        "readiness": {
            "candidate_count": readiness.get("candidate_count"),
            "allowed_count": readiness.get("allowed_count"),
            "near_ready_count": (readiness.get("research_candidate_report") or {}).get("near_ready_count")
            if isinstance(readiness.get("research_candidate_report"), dict)
            else 0,
            "execution_ticket": readiness.get("execution_ticket"),
            "next_machine_action": readiness.get("next_machine_action"),
            "hard_blocker_classes": sorted((readiness.get("hard_blocker_taxonomy") or {}).keys()),
            "denial_journal_path": readiness.get("denial_journal_path"),
            "denial_journal_count": readiness.get("denial_journal_count"),
            "report_path": readiness.get("report_path"),
        },
        "promotion_boundary": {
            "research_outputs_do_not_authorize_orders": True,
            "testnet_requires_execution_ticket": True,
            "mainnet_live_allowed": False,
        },
    }

    report_dir = Path(output_dir).expanduser().resolve() if output_dir else NEW_SYMBOL_WORKFLOW_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{_stamp()}-new-symbol-workflow.json"
    payload["report_path"] = str(report_path)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload
