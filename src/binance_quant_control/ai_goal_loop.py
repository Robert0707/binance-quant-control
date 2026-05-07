from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ai_expectancy_upgrade import run_ai_expectancy_upgrade
from .ai_surface_audit import run_ai_surface_audit
from .config import STATE_DIR, ensure_runtime_dirs, load_settings
from .order_journal import summarize_closed_trade_reviews
from .readiness_scanner import run_ai_readiness_scan

AI_GOAL_LOOP_DIR = STATE_DIR / "ai-goal-loop"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _closed_trade_feedback(summary: dict[str, Any]) -> dict[str, Any]:
    count = _int(summary.get("count"))
    expectancy = summary.get("expectancy") if isinstance(summary.get("expectancy"), dict) else {}
    profit_factor = _float(summary.get("profit_factor"))
    expectancy_r = _float(expectancy.get("expectancy_r") or expectancy.get("expectancy"))
    if count >= 30 and profit_factor >= 1.0 and expectancy_r >= 0.0:
        sample_status = "forward_evidence_ready"
    elif count >= 30:
        sample_status = "forward_evidence_negative_or_mixed"
    else:
        sample_status = "insufficient_forward_evidence"
    return {
        "sample_status": sample_status,
        "closed_review_count": count,
        "profit_factor": profit_factor,
        "expectancy_r": expectancy_r,
        "summary": summary,
    }


def _score(
    *,
    surface_audit: dict[str, Any],
    expectancy_upgrade: dict[str, Any],
    readiness: dict[str, Any],
    feedback: dict[str, Any],
) -> dict[str, Any]:
    maturity = expectancy_upgrade.get("maturity_score") if isinstance(expectancy_upgrade.get("maturity_score"), dict) else {}
    base_score = _float(maturity.get("score_10"), 0.0)
    if surface_audit.get("status") != "passed" or _int(surface_audit.get("blocker_count")) > 0:
        base_score = min(base_score, 6.0)
    if _int(readiness.get("allowed_count")) > 0:
        base_score = max(base_score, 8.8)
    if feedback.get("sample_status") == "forward_evidence_ready":
        base_score += 0.2
    elif feedback.get("sample_status") == "insufficient_forward_evidence":
        base_score = min(base_score, 8.7)
    score_10 = round(min(base_score, 10.0), 2)
    return {
        "score_10": score_10,
        "score_100": round(score_10 * 10.0, 2),
        "missing_to_9": round(max(9.0 - score_10, 0.0), 2),
    }


def _recommended_commands(
    *,
    readiness: dict[str, Any],
    feedback: dict[str, Any],
    smoke: bool,
) -> tuple[str, list[str]]:
    allowed_count = _int(readiness.get("allowed_count"))
    hard_blockers = readiness.get("hard_blocker_taxonomy") if isinstance(readiness.get("hard_blocker_taxonomy"), dict) else {}
    if allowed_count > 0:
        return (
            "start_testnet_forward_evidence",
            [
                "openclaw-quantctl hermes-trade cycle --force --compact",
                "openclaw-quantctl review-closed-trades --compact",
                "openclaw-quantctl ai-goal-loop --compact",
            ],
        )
    if "exchange_constraints" in hard_blockers:
        action = "repair_exchange_sizing_or_margin"
        commands = [
            "repair_exchange_sizing_or_margin",
            "openclaw-quantctl ai-readiness-scan --max-candidates 6 --execution-mode testnet_exploration --margin-notional-usdt 25 --compact",
        ]
    elif "strategy_performance" in hard_blockers:
        action = "continue_expectancy_research"
        commands = []
    elif "market_state" in hard_blockers:
        action = "wait_for_market_state"
        commands = ["openclaw-quantctl ai-readiness-scan --max-candidates 6 --execution-mode testnet_exploration --compact"]
    else:
        action = str(readiness.get("next_machine_action") or "continue_expectancy_research")
        commands = []
    full_command = (
        "openclaw-quantctl ai-expectancy-upgrade --universe-limit 20 --limit 8000 "
        "--sweep-limit 5000 --max-configs 80 --max-walk-forward-validations 12 "
        "--max-readiness-candidates 6 --compact"
    )
    smoke_command = (
        "openclaw-quantctl ai-expectancy-upgrade --limit 120 --sweep-limit 280 "
        "--max-configs 2 --max-walk-forward-validations 1 --universe-limit 8 "
        "--max-readiness-candidates 6 --compact"
    )
    if feedback.get("sample_status") == "insufficient_forward_evidence":
        commands.append("openclaw-quantctl review-closed-trades --compact")
    commands.append("run_full_expectancy_upgrade")
    commands.append(smoke_command if smoke else full_command)
    return action, list(dict.fromkeys(commands))


def run_readiness_sizing_scout(
    *,
    output_dir: str | Path,
    margin_candidates: list[float] | None = None,
    max_candidates: int = 6,
) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    margins = margin_candidates or [25.0, 35.0, 50.0, 75.0, 100.0]
    attempts: list[dict[str, Any]] = []
    selected_margin: float | None = None
    status = "sizing_constraints_blocked"
    for margin in margins:
        readiness = run_ai_readiness_scan(
            output_dir=root / f"margin-{margin:g}",
            execution_mode="testnet_exploration",
            max_candidates=max_candidates,
            margin_notional_usdt=margin,
        )
        taxonomy = readiness.get("hard_blocker_taxonomy") if isinstance(readiness.get("hard_blocker_taxonomy"), dict) else {}
        exchange_blockers = taxonomy.get("exchange_constraints") or []
        attempts.append(
            {
                "margin_notional_usdt": margin,
                "candidate_count": readiness.get("candidate_count"),
                "allowed_count": readiness.get("allowed_count"),
                "exchange_blocker_count": len(exchange_blockers),
                "blocker_classes": sorted(taxonomy.keys()),
                "report_path": readiness.get("report_path"),
            }
        )
        if not exchange_blockers:
            selected_margin = margin
            status = "ready_candidate_found" if _int(readiness.get("allowed_count")) > 0 else "sizing_constraints_clear"
            break
    return {
        "mode": "readiness_sizing_scout_v1",
        "opens_orders": False,
        "mainnet_live_allowed": False,
        "status": status,
        "selected_margin_notional_usdt": selected_margin,
        "attempts": attempts,
    }


def run_ai_goal_loop(
    *,
    output_dir: str | Path | None = None,
    goal: str = "maximize_stable_expectancy",
    symbols: list[str] | None = None,
    discovery_symbols: list[str] | None = None,
    limit: int = 8000,
    sweep_limit: int = 5000,
    max_configs: int = 80,
    max_walk_forward_validations: int = 12,
    universe_limit: int = 20,
    max_readiness_candidates: int = 6,
    margin_notional_usdt: float | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    root = Path(output_dir).expanduser().resolve() if output_dir else AI_GOAL_LOOP_DIR / _stamp()
    root.mkdir(parents=True, exist_ok=True)
    settings = load_settings()

    surface_audit = run_ai_surface_audit(output_dir=root / "surface-audit")
    expectancy_upgrade = run_ai_expectancy_upgrade(
        output_dir=root / "expectancy-upgrade",
        symbols=symbols,
        discovery_symbols=discovery_symbols,
        limit=limit,
        sweep_limit=sweep_limit,
        max_configs=max_configs,
        max_walk_forward_validations=max_walk_forward_validations,
        universe_limit=universe_limit,
        max_readiness_candidates=max_readiness_candidates,
        readiness_execution_mode="testnet_exploration",
    )
    readiness = run_ai_readiness_scan(
        output_dir=root / "readiness-scan",
        execution_mode="testnet_exploration",
        max_candidates=max_readiness_candidates,
        margin_notional_usdt=margin_notional_usdt,
    )
    readiness_taxonomy = readiness.get("hard_blocker_taxonomy") if isinstance(readiness.get("hard_blocker_taxonomy"), dict) else {}
    sizing_scout = None
    if "exchange_constraints" in readiness_taxonomy:
        sizing_scout = run_readiness_sizing_scout(
            output_dir=root / "readiness-sizing-scout",
            max_candidates=max_readiness_candidates,
        )
    feedback = _closed_trade_feedback(summarize_closed_trade_reviews())
    score = _score(
        surface_audit=surface_audit,
        expectancy_upgrade=expectancy_upgrade,
        readiness=readiness,
        feedback=feedback,
    )
    next_action, commands = _recommended_commands(
        readiness=readiness,
        feedback=feedback,
        smoke=smoke,
    )
    payload = {
        "generated_at": _utc_now().isoformat(),
        "mode": "ai_goal_loop_v1",
        "goal": goal,
        "safety": {
            "opens_orders": False,
            "writes_execution_config": False,
            "mainnet_live_allowed": False,
            "uses_testnet_for_private_channels": settings.use_testnet,
            "live_trading_enabled": settings.live_trading_enabled,
        },
        "score": score,
        "surface_audit": {
            "status": surface_audit.get("status"),
            "blocker_count": surface_audit.get("blocker_count"),
            "report_path": surface_audit.get("report_path"),
        },
        "expectancy_upgrade": {
            "score": expectancy_upgrade.get("maturity_score"),
            "final_machine_decision": expectancy_upgrade.get("final_machine_decision"),
            "report_path": expectancy_upgrade.get("report_path"),
        },
        "readiness": {
            "candidate_count": readiness.get("candidate_count"),
            "allowed_count": readiness.get("allowed_count"),
            "next_machine_action": readiness.get("next_machine_action"),
            "hard_blocker_taxonomy": readiness.get("hard_blocker_taxonomy"),
            "execution_ticket": readiness.get("execution_ticket"),
            "report_path": readiness.get("report_path"),
        },
        "readiness_sizing_scout": sizing_scout,
        "closed_trade_feedback": feedback,
        "next_machine_action": next_action,
        "recommended_commands": commands,
        "reports": {
            "surface_audit": surface_audit.get("report_path"),
            "expectancy_upgrade": expectancy_upgrade.get("report_path"),
            "readiness": readiness.get("report_path"),
        },
    }
    report_path = root / "ai-goal-loop.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    payload["reports"]["goal_loop"] = str(report_path)
    payload["report_path"] = str(report_path)
    return payload
