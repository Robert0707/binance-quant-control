from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import STATE_DIR, ensure_runtime_dirs
from .event_bus import EventBus, TradingEvent, default_plugin_lifecycle
from .feature_registry import build_feature_manifest
from .hailo_entry_gate import evaluate_hailo_entry_gate
from .hailo_trading_plan import build_hailo_trading_plan
from .model_registry import model_registry_payload
from .payoff_objective import PayoffObjectiveTargets, payoff_objective_sort_key
from .portfolio_construction import (
    PortfolioConstructionPolicy,
    build_portfolio_risk_snapshot,
    build_portfolio_target,
)
from .professional_system_audit import DEFAULT_BLUEPRINT_PATH, run_professional_system_audit
from .route_risk_control import route_quarantine_status
from .side_risk_policy import evaluate_route_side_risk
from .signal_api import append_trading_signal, signal_api_contract
from .signal_schema import TradingSignal, build_signal_id
from .skipped_signal_journal import append_skipped_signal
from .trading_committee import run_structured_committee

HERMES_AI_TRADER_DIR = STATE_DIR / "hermes-ai-trader"

DEFAULT_OPEN_ORDER_TARGETS = {
    "min_trades": 100,
    "min_profit_factor": 1.25,
    "min_expectancy_r": 0.05,
    "min_payoff_ratio": 1.20,
}


@dataclass(frozen=True, slots=True)
class OpenOrderGate:
    allowed: bool
    blockers: list[str]
    required_sequence: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateQueueItem:
    rank: int
    signal: dict[str, Any]
    portfolio_target: dict[str, Any]
    committee_decision: str
    open_order_gate: dict[str, Any]
    blocker_taxonomy: dict[str, list[str]]
    next_action: str
    machine_state: str
    machine_directive: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MachineStrategyDirective:
    directive: str
    objective: str
    priority_score: float
    reason: str
    allowed_surface: str
    blocked_surface: str
    next_commands: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _row_score(row: dict[str, Any]) -> float:
    if row.get("ranking_score") not in (None, ""):
        return _float(row.get("ranking_score"))
    return payoff_objective_sort_key(row, targets=PayoffObjectiveTargets())[0]


def _side_from_row(row: dict[str, Any]) -> str:
    symbol_strategy = row.get("symbol_strategy") if isinstance(row.get("symbol_strategy"), dict) else {}
    interval = str(row.get("interval") or "")
    family = str(row.get("strategy_family") or "")
    side_map = symbol_strategy.get("interval_family_sides") if isinstance(symbol_strategy, dict) else {}
    if isinstance(side_map, dict):
        interval_sides = side_map.get(interval)
        if isinstance(interval_sides, dict):
            sides = interval_sides.get(family)
            if isinstance(sides, list) and sides:
                first = str(sides[0]).upper()
                if first in {"BUY", "SELL"}:
                    return first
    return "BUY"


def _row_route_id(row: dict[str, Any]) -> str:
    symbol_strategy = row.get("symbol_strategy") if isinstance(row.get("symbol_strategy"), dict) else {}
    return str(symbol_strategy.get("route_id") or row.get("route_id") or row.get("cohort_id") or "unrouted")


def _best_alpha_row(report: dict[str, Any]) -> dict[str, Any] | None:
    rows = [row for row in (report.get("rows") or []) if isinstance(row, dict)]
    if not rows:
        rows = [row for row in (report.get("top") or []) if isinstance(row, dict)]
    if not rows:
        return None
    traded_rows = [row for row in rows if _int(row.get("trade_count")) > 0]
    if traded_rows:
        rows = traded_rows
    return max(
        rows,
        key=lambda row: (
            bool(row.get("promotion_eligible")),
            _row_score(row),
            _float(row.get("expectancy_r")),
            _float(row.get("payoff_ratio")),
            _float(row.get("profit_factor")),
            _int(row.get("trade_count")),
        ),
    )


def _best_market_bot_row(gate: dict[str, Any]) -> dict[str, Any] | None:
    rows = [row for row in (gate.get("accepted") or []) if isinstance(row, dict)]
    if not rows:
        rows = [row for row in (gate.get("rows") or []) if isinstance(row, dict)]
    rows = [row for row in rows if _int(row.get("trade_count")) > 0]
    if not rows:
        best = gate.get("best")
        return best if isinstance(best, dict) else None
    return max(
        rows,
        key=lambda row: (
            bool(row.get("accepted")),
            _row_score(row),
            _float(row.get("expectancy_r")),
            _float(row.get("payoff_ratio")),
            _float(row.get("profit_factor")),
            _int(row.get("trade_count")),
        ),
    )


def _market_bot_rows(gate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in (gate.get("accepted") or []) if isinstance(row, dict)]
    if not rows:
        rows = [row for row in (gate.get("rows") or []) if isinstance(row, dict)]
    rows = [row for row in rows if _int(row.get("trade_count")) > 0]
    return sorted(
        rows,
        key=lambda row: (
            bool(row.get("accepted")),
            _row_score(row),
            _float(row.get("expectancy_r")),
            _float(row.get("payoff_ratio")),
            _float(row.get("profit_factor")),
            _int(row.get("trade_count")),
        ),
        reverse=True,
    )


def _targets_from_market_bot_gate(gate: dict[str, Any]) -> dict[str, float]:
    targets = dict(DEFAULT_OPEN_ORDER_TARGETS)
    raw_targets = gate.get("targets") if isinstance(gate.get("targets"), dict) else {}
    for key in targets:
        if key in raw_targets:
            targets[key] = _float(raw_targets.get(key), targets[key])
    return targets


def _signal_from_market_bot_row(
    row: dict[str, Any],
    *,
    gate: dict[str, Any],
    targets: dict[str, float],
) -> TradingSignal:
    blockers = list(row.get("blockers") or [])
    portfolio_gate = gate.get("portfolio_gate") if isinstance(gate.get("portfolio_gate"), dict) else {}
    if not gate.get("safe_to_open_new_entries"):
        blockers.append("market-bot-gate-not-safe")
    if portfolio_gate.get("enabled") and not portfolio_gate.get("passed"):
        blockers.append("market-bot-portfolio-gate-failed")
    if _int(row.get("trade_count")) < int(targets["min_trades"]):
        blockers.append("trade-count-below-market-bot-floor")
    if _float(row.get("expectancy_r")) < targets["min_expectancy_r"]:
        blockers.append("expectancy-r-below-market-bot-floor")
    if _float(row.get("payoff_ratio")) < targets["min_payoff_ratio"]:
        blockers.append("payoff-ratio-below-market-bot-floor")
    if _float(row.get("profit_factor")) < targets["min_profit_factor"]:
        blockers.append("profit-factor-below-market-bot-floor")
    symbol = str(row.get("symbol") or "UNRESOLVED")
    interval = str(row.get("interval") or "unknown")
    family = str(row.get("strategy_family") or "unknown")
    side = _side_from_row(row)
    route_id = _row_route_id(row)
    route_quarantine = route_quarantine_status(route_id)
    if route_quarantine.get("quarantined"):
        reasons = "; ".join(str(item) for item in (route_quarantine.get("reasons") or []))
        blockers.append(f"route-quarantined:{route_id}:{reasons or 'manual-review-required'}")
    route_side_risk = evaluate_route_side_risk(route_id=route_id, side=side)
    if not route_side_risk.allowed:
        blockers.extend(route_side_risk.reasons)
    return TradingSignal(
        signal_id=build_signal_id(symbol, interval, family, side),
        symbol=symbol,
        side=side,
        interval=interval,
        strategy_family=family,
        route_id=route_id,
        status="candidate" if not blockers else "rejected",
        signal_score=_float(row.get("market_bot_score") or _row_score(row)),
        expectancy_r=_float(row.get("expectancy_r")),
        payoff_ratio=_float(row.get("payoff_ratio")),
        profit_factor=_float(row.get("profit_factor")),
        win_rate=_float(row.get("win_rate")),
        stop_loss_ratio=_float(row.get("stop_loss_ratio"), 100.0),
        trade_count=_int(row.get("trade_count")),
        blockers=list(dict.fromkeys(blockers)),
        metadata={
            "source": "market_bot_gate.accepted",
            "market_bot_gate_path": gate.get("path"),
            "cohort_id": row.get("cohort_id"),
            "accepted_symbols": gate.get("accepted_symbols") or [],
            "accepted_count": gate.get("accepted_count"),
            "targets": targets,
            "research_state": row.get("research_state"),
            "route_quarantine": route_quarantine,
            "route_side_risk": route_side_risk.to_dict(),
            "machine_only": True,
        },
    )


def _load_alpha_report(alpha: dict[str, Any]) -> dict[str, Any] | None:
    path = alpha.get("path")
    if not path:
        return None
    candidate = Path(str(path)).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if not candidate.exists():
        return None
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _metrics_from_alpha(alpha: dict[str, Any]) -> dict[str, Any]:
    report = _load_alpha_report(alpha)
    row = _best_alpha_row(report or {}) if report else None
    if not row:
        return {
            "metrics": dict(alpha),
            "symbol": str(alpha.get("symbol") or "UNRESOLVED"),
            "side": "BUY",
            "interval": str(alpha.get("interval") or "unknown"),
            "family": str(alpha.get("strategy_family") or "unknown"),
            "route_id": str(alpha.get("route_id") or "unrouted"),
            "metadata": {"source": "professional_system_audit.alpha_report"},
        }
    symbol_strategy = row.get("symbol_strategy") if isinstance(row.get("symbol_strategy"), dict) else {}
    return {
        "metrics": row,
        "symbol": str(row.get("symbol") or "UNRESOLVED"),
        "side": _side_from_row(row),
        "interval": str(row.get("interval") or "unknown"),
        "family": str(row.get("strategy_family") or "unknown"),
        "route_id": str(symbol_strategy.get("route_id") or row.get("route_id") or row.get("cohort_id") or "unrouted"),
        "metadata": {
            "source": "alpha_research.rows",
            "alpha_report_path": alpha.get("path"),
            "cohort_id": row.get("cohort_id"),
            "promotion_eligible": bool(row.get("promotion_eligible")),
            "robustness_status": row.get("robustness_status"),
            "entry_veto_reasons": row.get("entry_veto_reasons") or {},
            "route_side_gate": row.get("route_side_gate") or {},
        },
    }


def _best_signal_from_alpha(alpha: dict[str, Any]) -> TradingSignal:
    symbol = "UNRESOLVED"
    side = "BUY"
    interval = "unknown"
    family = "unknown"
    route_id = "unrouted"
    metrics = dict(alpha)
    blockers = ["alpha-report-missing"]
    status = "rejected"
    if alpha.get("available"):
        selected = _metrics_from_alpha(alpha)
        blockers = []
        metrics = dict(selected["metrics"])
        symbol = str(selected["symbol"])
        side = str(selected["side"])
        interval = str(selected["interval"])
        family = str(selected["family"])
        route_id = str(selected["route_id"])
        status = "candidate"
        if _int(alpha.get("promotion_eligible_count")) <= 0:
            blockers.append("no-promotion-eligible-cohort")
        if symbol == "UNRESOLVED":
            blockers.append("symbol-unresolved")
        if _float(metrics.get("expectancy_r") or metrics.get("weighted_expectancy_r")) <= 0.0:
            blockers.append("expectancy-r-not-positive")
        if _float(metrics.get("payoff_ratio") or metrics.get("weighted_payoff_ratio")) < 1.15:
            blockers.append("payoff-ratio-below-floor")
        if _float(metrics.get("profit_factor") or metrics.get("finite_avg_profit_factor")) < 1.5:
            blockers.append("profit-factor-below-floor")
        if _int(metrics.get("trade_count") or alpha.get("trade_count")) < 100:
            blockers.append("trade-count-below-floor")
        route_quarantine = route_quarantine_status(route_id)
        if route_quarantine.get("quarantined"):
            reasons = "; ".join(str(item) for item in (route_quarantine.get("reasons") or []))
            blockers.append(f"route-quarantined:{route_id}:{reasons or 'manual-review-required'}")
            selected["metadata"]["route_quarantine"] = route_quarantine
        route_side_risk = evaluate_route_side_risk(route_id=route_id, side=side)
        if not route_side_risk.allowed:
            blockers.extend(route_side_risk.reasons)
        selected["metadata"]["route_side_risk"] = route_side_risk.to_dict()
    return TradingSignal(
        signal_id=build_signal_id(symbol, interval, family, side),
        symbol=symbol,
        side=side,
        interval=interval,
        strategy_family=family,
        route_id=route_id,
        status=status if not blockers else "rejected",
        signal_score=_row_score(metrics),
        expectancy_r=_float(metrics.get("expectancy_r") or metrics.get("weighted_expectancy_r")),
        payoff_ratio=_float(metrics.get("payoff_ratio") or metrics.get("weighted_payoff_ratio")),
        profit_factor=_float(metrics.get("profit_factor") or metrics.get("finite_avg_profit_factor")),
        win_rate=_float(metrics.get("win_rate") or metrics.get("weighted_win_rate")),
        stop_loss_ratio=_float(metrics.get("stop_loss_ratio") or metrics.get("weighted_stop_loss_ratio"), 100.0),
        trade_count=_int(metrics.get("trade_count")),
        blockers=blockers,
        metadata=selected["metadata"] if alpha.get("available") else {"source": "professional_system_audit.alpha_report"},
    )


def _signal_from_market_bot_gate(gate: dict[str, Any]) -> TradingSignal | None:
    if not gate.get("available"):
        return None
    row = _best_market_bot_row(gate)
    if not row:
        return None
    targets = _targets_from_market_bot_gate(gate)
    return _signal_from_market_bot_row(row, gate=gate, targets=targets)


def _blocker_taxonomy(blockers: list[str]) -> dict[str, list[str]]:
    taxonomy = {
        "alpha": [],
        "portfolio": [],
        "route_history": [],
        "committee": [],
        "architecture": [],
        "readiness": [],
        "execution": [],
    }
    for blocker in blockers:
        lowered = blocker.lower()
        if "route-quarantined" in lowered or "route-side" in lowered or "historical" in lowered:
            taxonomy["route_history"].append(blocker)
        elif "alpha" in lowered or "expectancy" in lowered or "payoff" in lowered or "profit-factor" in lowered:
            taxonomy["alpha"].append(blocker)
        elif "portfolio" in lowered or "symbol-open-risk" in lowered or "correlation" in lowered:
            taxonomy["portfolio"].append(blocker)
        elif "committee" in lowered:
            taxonomy["committee"].append(blocker)
        elif "audit" in lowered or "architecture" in lowered or ":partial" in lowered or ":missing" in lowered:
            taxonomy["architecture"].append(blocker)
        elif "live" in lowered or "readiness" in lowered or "gate-not-safe" in lowered:
            taxonomy["readiness"].append(blocker)
        else:
            taxonomy["execution"].append(blocker)
    return {key: value for key, value in taxonomy.items() if value}


def _candidate_next_action(gate: OpenOrderGate, taxonomy: dict[str, list[str]]) -> tuple[str, str]:
    if gate.allowed:
        return "ready_for_live_readiness_scan", "candidate_ready"
    if taxonomy.get("alpha"):
        return "return_to_alpha_research_or_expand_sample", "research_blocked"
    if taxonomy.get("portfolio"):
        return "reduce_portfolio_or_correlation_risk_before_scan", "portfolio_blocked"
    if taxonomy.get("route_history"):
        return "repair_route_history_or_wait_for_quarantine_clear", "route_history_blocked"
    if taxonomy.get("committee"):
        return "hold_for_structured_review", "committee_blocked"
    if taxonomy.get("architecture"):
        return "repair_architecture_or_missing_evidence", "architecture_blocked"
    if taxonomy.get("readiness"):
        return "wait_for_live_readiness_conditions", "readiness_blocked"
    return "skip_until_blockers_clear", "blocked"


def _machine_priority_score(signal: TradingSignal) -> float:
    return round(
        signal.signal_score
        + (signal.expectancy_r * 100.0)
        + (signal.payoff_ratio * 10.0)
        + (signal.profit_factor * 5.0)
        + min(signal.trade_count, 500) / 10.0,
        4,
    )


def _research_command_for_signal(signal: TradingSignal, *, mode: str) -> str:
    symbols = signal.symbol if signal.symbol != "UNRESOLVED" else "BTCUSDT,ETHUSDT"
    base = (
        "openclaw-quantctl risk-combo-sweep "
        f"--symbols {symbols} --limit 5000 --grid-mode focused "
        "--min-test-trades 100 --min-win-rate 45 --max-stop-loss-ratio 55 "
        "--target-profit-factor 1.25 --min-expectancy-r 0.05 "
        "--min-payoff-ratio 1.20 --max-configs 80 "
        "--max-walk-forward-validations 12 --skip-news --compact"
    )
    if mode == "sample":
        return base.replace("--limit 5000", "--limit 15000")
    return base


def _machine_strategy_directive(
    *,
    signal: TradingSignal,
    gate: OpenOrderGate,
    taxonomy: dict[str, list[str]],
) -> MachineStrategyDirective:
    priority = _machine_priority_score(signal)
    if gate.allowed:
        return MachineStrategyDirective(
            directive="exploit",
            objective="scan_testnet_readiness_without_opening_mainnet",
            priority_score=priority,
            reason="all machine gates are clean for the candidate surface",
            allowed_surface="live_readiness_scan,testnet_review,paper_forward_validation",
            blocked_surface="mainnet_order_submission",
            next_commands=[
                f"openclaw-quantctl live-readiness --symbol {signal.symbol} --execution-mode testnet_exploration --compact"
            ],
        )
    if taxonomy.get("architecture"):
        return MachineStrategyDirective(
            directive="repair",
            objective="restore_reproducible_evidence_and_control_plane_before_strategy_search",
            priority_score=priority,
            reason="architecture or audit evidence is incomplete",
            allowed_surface="repository_audit,professional_system_audit,feature_manifest_rebuild",
            blocked_surface="candidate_promotion",
            next_commands=[
                "openclaw-quantctl repository-audit --compact",
                "openclaw-quantctl professional-system-audit --compact",
            ],
        )
    if signal.expectancy_r <= 0.0 or signal.profit_factor < 1.0:
        return MachineStrategyDirective(
            directive="quarantine",
            objective="remove_negative_expectancy_surface_from_candidate_selection",
            priority_score=priority,
            reason="expectancy or profit factor is non-productive after current evidence",
            allowed_surface="loss_diagnostics,route_side_veto,feature_research",
            blocked_surface="paper_order,testnet_order,live_readiness_scan",
            next_commands=[
                "openclaw-quantctl loss-diagnostics --min-bucket-trades 5 --top-n 20 --compact"
            ],
        )
    if taxonomy.get("route_history"):
        return MachineStrategyDirective(
            directive="quarantine",
            objective="keep_negative_route_history_out_of_candidate_selection",
            priority_score=priority,
            reason="route is quarantined or route-side history is not cleared for promotion",
            allowed_surface="loss_diagnostics,route_risk_status,risk_combo_sweep,manual_quarantine_review",
            blocked_surface="paper_order,testnet_order,live_readiness_scan",
            next_commands=[
                f"openclaw-quantctl route-risk-status --route-id {signal.route_id} --compact",
                f"openclaw-quantctl risk-combo-sweep --routes {signal.route_id} --compact",
            ],
        )
    if taxonomy.get("portfolio"):
        return MachineStrategyDirective(
            directive="rebalance_search",
            objective="expand_to_uncorrelated_positive_expectancy_cohorts",
            priority_score=priority,
            reason="candidate edge exists but portfolio gate is not ready",
            allowed_surface="six_symbol_discovery,correlation_review,portfolio_gate",
            blocked_surface="single_cohort_live_promotion",
            next_commands=[
                "openclaw-quantctl alpha-research --config config/market-bot-six-symbol-discovery.default.yaml --limit 8000 --output-dir state/market-bot-six-symbol-payoff-l8000 --compact",
                "openclaw-quantctl market-bot-gate --alpha-report state/market-bot-six-symbol-payoff-l8000/alpha-research-ranking.json --compact",
            ],
        )
    if signal.trade_count < int(DEFAULT_OPEN_ORDER_TARGETS["min_trades"]):
        return MachineStrategyDirective(
            directive="harvest_data",
            objective="increase_sample_count_before_any_promotion_decision",
            priority_score=priority,
            reason="edge is not disproven but sample count is below machine floor",
            allowed_surface="paper_forward_validation,expanded_history_replay,walk_forward",
            blocked_surface="live_promotion",
            next_commands=[_research_command_for_signal(signal, mode="sample")],
        )
    if signal.payoff_ratio < DEFAULT_OPEN_ORDER_TARGETS["min_payoff_ratio"]:
        return MachineStrategyDirective(
            directive="explore_exit_surface",
            objective="raise_payoff_ratio_before_adding_more_entry_filters",
            priority_score=priority,
            reason="entry surface has value but reward-to-risk geometry is weak",
            allowed_surface="exit_profile_sweep,triple_barrier_replay,risk_combo_sweep",
            blocked_surface="new_indicator_stacking",
            next_commands=[_research_command_for_signal(signal, mode="payoff")],
        )
    if taxonomy.get("readiness"):
        return MachineStrategyDirective(
            directive="wait",
            objective="rescan_when_market_and_execution_conditions_change",
            priority_score=priority,
            reason="research can be acceptable while live-readiness conditions are not",
            allowed_surface="readiness_rescan,position_monitoring",
            blocked_surface="new_order_creation",
            next_commands=["openclaw-quantctl live-readiness --strategy-config config/strategy-live-pilot.yaml --compact"],
        )
    return MachineStrategyDirective(
        directive="explore",
        objective="search_adjacent_feature_exit_space_under_same_risk_ceiling",
        priority_score=priority,
        reason="candidate is blocked but not clearly negative expectancy",
        allowed_surface="bounded_research,paper_replay,walk_forward",
        blocked_surface="mainnet_order_submission",
        next_commands=[_research_command_for_signal(signal, mode="payoff")],
    )


def _build_candidate_queue(
    *,
    audit: dict[str, Any],
    market_bot_gate: dict[str, Any],
    fallback_signal: TradingSignal,
) -> list[CandidateQueueItem]:
    rows = _market_bot_rows(market_bot_gate) if market_bot_gate.get("available") else []
    targets = _targets_from_market_bot_gate(market_bot_gate)
    signals = [
        _signal_from_market_bot_row(row, gate=market_bot_gate, targets=targets)
        for row in rows
    ] or [fallback_signal]
    queue: list[CandidateQueueItem] = []
    for rank, item_signal in enumerate(signals, start=1):
        portfolio_target = build_portfolio_target(
            {
                "symbol": item_signal.symbol,
                "side": item_signal.side,
                "route_id": item_signal.route_id,
                "target_risk_pct": 0.006,
                "signal_score": item_signal.signal_score,
                "expectancy_r": item_signal.expectancy_r,
                "payoff_ratio": item_signal.payoff_ratio,
            },
            policy=PortfolioConstructionPolicy(),
        )
        committee = run_structured_committee(item_signal.to_dict(), audit.get("evidence") or {})
        gate = _open_order_gate(
            audit=audit,
            signal=item_signal,
            portfolio_target=portfolio_target.to_dict(),
            committee=committee,
        )
        taxonomy = _blocker_taxonomy(gate.blockers)
        next_action, machine_state = _candidate_next_action(gate, taxonomy)
        machine_directive = _machine_strategy_directive(
            signal=item_signal,
            gate=gate,
            taxonomy=taxonomy,
        )
        queue.append(
            CandidateQueueItem(
                rank=rank,
                signal=item_signal.to_dict(),
                portfolio_target=portfolio_target.to_dict(),
                committee_decision=str(committee.get("decision") or ""),
                open_order_gate=gate.to_dict(),
                blocker_taxonomy=taxonomy,
                next_action=next_action,
                machine_state=machine_state,
                machine_directive=machine_directive.to_dict(),
            )
        )
    return queue


def _build_machine_strategy_policy(candidate_queue: list[CandidateQueueItem]) -> dict[str, Any]:
    queue = [item.to_dict() for item in candidate_queue]
    ready = [item for item in candidate_queue if item.machine_directive.get("directive") == "exploit"]
    harvest = [item for item in candidate_queue if item.machine_directive.get("directive") == "harvest_data"]
    exit_explore = [
        item for item in candidate_queue if item.machine_directive.get("directive") == "explore_exit_surface"
    ]
    quarantine = [item for item in candidate_queue if item.machine_directive.get("directive") == "quarantine"]
    research_backlog = sorted(
        [item for item in candidate_queue if item.machine_directive.get("directive") != "exploit"],
        key=lambda item: _float(item.machine_directive.get("priority_score")),
        reverse=True,
    )
    next_commands: list[str] = []
    for item in ready + harvest + exit_explore + research_backlog:
        for command in item.machine_directive.get("next_commands") or []:
            if command not in next_commands:
                next_commands.append(command)
        if len(next_commands) >= 5:
            break
    return {
        "mode": "ai_trader_machine_strategy_router",
        "objective": "allocate_compute_to_exploit_harvest_explore_or_quarantine_without_mainnet_execution",
        "queue_size": len(queue),
        "exploit_symbols": [item.signal.get("symbol") for item in ready],
        "harvest_symbols": [item.signal.get("symbol") for item in harvest],
        "exit_explore_symbols": [item.signal.get("symbol") for item in exit_explore],
        "quarantine_symbols": [item.signal.get("symbol") for item in quarantine],
        "research_backlog": [
            {
                "rank": item.rank,
                "symbol": item.signal.get("symbol"),
                "directive": item.machine_directive.get("directive"),
                "priority_score": item.machine_directive.get("priority_score"),
            }
            for item in research_backlog[:10]
        ],
        "next_commands": next_commands,
        "hard_boundaries": [
            "opens_orders=false",
            "mainnet_live_allowed=false",
            "operator_execute_required=true",
            "max_per_trade_risk_pct<=2.5_when_promoted",
        ],
    }


def _open_order_gate(
    *,
    audit: dict[str, Any],
    signal: TradingSignal,
    portfolio_target: dict[str, Any],
    committee: dict[str, Any],
    hailo_gate: dict[str, Any] | None = None,
) -> OpenOrderGate:
    blockers: list[str] = []
    if not audit.get("trade_ready"):
        blockers.append("professional-system-audit-not-trade-ready")
    blockers.extend(audit.get("critical_blockers") or [])
    blockers.extend(signal.blockers)
    if not portfolio_target.get("accepted"):
        blockers.extend(portfolio_target.get("blockers") or [])
    if committee.get("decision") == "reject":
        blockers.append("structured-committee-reject")
    if hailo_gate and not hailo_gate.get("allowed", True):
        blockers.extend(hailo_gate.get("blockers") or ["hailo-entry-gate-blocked"])
    unique_blockers = list(dict.fromkeys(blockers))
    return OpenOrderGate(
        allowed=not unique_blockers,
        blockers=unique_blockers,
        required_sequence=[
            "universe_pass",
            "feature_manifest_live_safe",
            "alpha_positive_expectancy",
            "portfolio_target_accepted",
            "pre_trade_risk_pass",
            "structured_committee_not_reject",
            "live_readiness_allowed",
            "operator_execute",
        ],
    )


def run_hermes_ai_trader(
    *,
    blueprint_config: str | Path = DEFAULT_BLUEPRINT_PATH,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    bus = EventBus()
    emitted: list[dict[str, Any]] = []
    bus.subscribe("*", lambda event: emitted.append(event.to_dict()))
    audit = run_professional_system_audit(config_path=blueprint_config)
    bus.publish(TradingEvent("audit.completed", {"trade_ready": audit.get("trade_ready")}))
    feature_manifest = build_feature_manifest()
    bus.publish(
        TradingEvent(
            "features.ready",
            {
                "manifest_hash": feature_manifest.get("manifest_hash"),
                "live_safe": feature_manifest.get("live_safe"),
            },
        )
    )
    model_registry = model_registry_payload()
    hailo_plan = build_hailo_trading_plan()
    hailo_entry_gate = evaluate_hailo_entry_gate({"returncode": 0, "response": {}})
    plugin_lifecycle = [plugin.to_dict() for plugin in default_plugin_lifecycle()]
    evidence = audit.get("evidence") or {}
    market_bot_gate = evidence.get("market_bot_gate") or {}
    alpha = evidence.get("alpha_report") or {}
    signal = _signal_from_market_bot_gate(market_bot_gate) or _best_signal_from_alpha(alpha)
    candidate_queue = _build_candidate_queue(
        audit=audit,
        market_bot_gate=market_bot_gate,
        fallback_signal=signal,
    )
    machine_strategy = _build_machine_strategy_policy(candidate_queue)
    bus.publish(TradingEvent("signal.created", {"signal_id": signal.signal_id, "status": signal.status}))
    portfolio_target = build_portfolio_target(
        {
            "symbol": signal.symbol,
            "side": signal.side,
            "route_id": signal.route_id,
            "target_risk_pct": 0.006,
            "signal_score": signal.signal_score,
            "expectancy_r": signal.expectancy_r,
            "payoff_ratio": signal.payoff_ratio,
        },
        policy=PortfolioConstructionPolicy(),
    )
    portfolio_risk = build_portfolio_risk_snapshot(policy=PortfolioConstructionPolicy())
    bus.publish(
        TradingEvent(
            "portfolio.targeted",
            {
                "accepted": portfolio_target.accepted,
                "blockers": portfolio_target.blockers,
            },
        )
    )
    committee = run_structured_committee(signal.to_dict(), audit.get("evidence") or {})
    bus.publish(TradingEvent("committee.reviewed", {"decision": committee.get("decision")}))
    gate = _open_order_gate(
        audit=audit,
        signal=signal,
        portfolio_target=portfolio_target.to_dict(),
        committee=committee,
        hailo_gate=hailo_entry_gate,
    )
    signal_ledger_path = append_trading_signal(signal, gate=gate.to_dict())
    bus.publish(TradingEvent("gate.evaluated", gate.to_dict()))
    if not gate.allowed:
        append_skipped_signal(
            symbol=signal.symbol,
            side=signal.side,
            route_id=signal.route_id,
            strategy_family=signal.strategy_family,
            gate="hermes_ai_trader_open_order_gate",
            blockers=gate.blockers,
            signal_score=signal.signal_score,
            expectancy_r=signal.expectancy_r,
            payoff_ratio=signal.payoff_ratio,
            metadata={"signal_id": signal.signal_id},
        )
    payload = {
        "generated_at": _utc_now().isoformat(),
        "mode": "hermes_ai_trader_v2",
        "safety": {
            "opens_orders": False,
            "writes_execution_config": False,
            "mainnet_live_allowed": False,
            "requires_operator_execute": True,
        },
        "open_order_gate": gate.to_dict(),
        "signal": signal.to_dict(),
        "candidate_queue": [item.to_dict() for item in candidate_queue],
        "machine_strategy": machine_strategy,
        "machine_policy": {
            "selection": "rank_all_market_bot_candidates_then_scan_live_readiness",
            "primary_sort": "market_bot_score_expectancy_payoff_pf_sample",
            "do_not_execute_on": [
                "kill_switch",
                "live_readiness_allowed_false",
                "professional_entry_gate_failed",
                "exchange_min_notional_failed",
                "portfolio_risk_cap_failed",
            ],
            "next_scan_symbols": [
                item.signal.get("symbol")
                for item in candidate_queue
                if item.machine_state == "candidate_ready"
            ],
            "strategy_router_mode": machine_strategy.get("mode"),
        },
        "portfolio_target": portfolio_target.to_dict(),
        "portfolio_risk": portfolio_risk.to_dict(),
        "committee": committee,
        "feature_manifest": feature_manifest,
        "model_registry": model_registry,
        "signal_api": signal_api_contract() | {"latest_signal_ledger": str(signal_ledger_path)},
        "hailo_plan": hailo_plan,
        "hailo_entry_gate": hailo_entry_gate,
        "plugin_lifecycle": plugin_lifecycle,
        "architecture_audit": {
            "trade_ready": audit.get("trade_ready"),
            "execution_recommendation": audit.get("execution_recommendation"),
            "critical_blockers": audit.get("critical_blockers"),
            "layer_summary": audit.get("layer_summary"),
            "report_path": audit.get("report_path"),
        },
        "events": emitted,
    }
    root_dir = Path(output_dir).expanduser().resolve() if output_dir else HERMES_AI_TRADER_DIR
    root_dir.mkdir(parents=True, exist_ok=True)
    report_path = root_dir / f"{_stamp()}-hermes-ai-trader.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    payload["report_path"] = str(report_path)
    return payload
