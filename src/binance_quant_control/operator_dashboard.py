from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .binance_api import BinanceAPIError, BinanceClient
from .config import STATE_DIR, Settings, ensure_runtime_dirs
from .decision_audit import run_decision_audit
from .loss_diagnostics import run_loss_diagnostics
from .order_journal import (
    read_live_orders,
    summarize_closed_trade_reviews,
    summarize_live_orders,
)

OPERATOR_DASHBOARD_DIR = STATE_DIR / "operator-dashboard"
N8N_DIGEST_DIR = STATE_DIR / "n8n-digests"
RISK_COMBO_MATRIX_DIR = STATE_DIR / "risk-combo-matrix"
RISK_COMBO_SWEEP_DIR = STATE_DIR / "risk-combo-sweeps"
READINESS_SCAN_DIR = STATE_DIR / "hermes-readiness-scan"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _order_quantity(order: dict[str, Any]) -> float:
    for key in ("quantity", "origQty"):
        value = order.get(key)
        if value is not None:
            return _float(value)
    return 0.0


def _order_trigger(order: dict[str, Any]) -> float:
    return _float(order.get("triggerPrice", order.get("stopPrice")))


def _tp_ladder_health(
    *,
    position_qty: float,
    take_profit_orders: list[dict[str, Any]],
    trailing_count: int,
    step_size: float = 0.0,
) -> dict[str, Any]:
    quantities = [_order_quantity(item) for item in take_profit_orders]
    triggers = [_order_trigger(item) for item in take_profit_orders]
    total_tp_qty = sum(quantities)
    largest_tp_qty = max(quantities or [0.0])
    first_tp_qty = quantities[0] if quantities else 0.0
    first_tp_ratio = (first_tp_qty / position_qty) if position_qty > 0 else 0.0
    runner_qty = max(0.0, position_qty - total_tp_qty)
    micro_full_tp_fallback = (
        step_size > 0
        and position_qty <= step_size * 1.01
        and len(take_profit_orders) == 1
        and total_tp_qty >= position_qty * 0.98
    )
    issues: list[str] = []
    if not take_profit_orders:
        issues.append("missing_tp")
    if position_qty > 0 and first_tp_ratio >= 0.9 and not micro_full_tp_fallback:
        issues.append("tp1_full_position")
    if position_qty > 0 and largest_tp_qty >= position_qty * 0.9 and len(take_profit_orders) > 1:
        issues.append("oversized_tp_level")
    if position_qty > 0 and total_tp_qty > position_qty * 1.02:
        issues.append("tp_quantity_exceeds_position")
    if trailing_count > 0 and total_tp_qty >= position_qty * 0.98:
        issues.append("trailing_has_no_runner")
    if trailing_count <= 0 and position_qty > 0 and 0.0 < total_tp_qty < position_qty * 0.98:
        issues.append("runner_waiting_for_trailing")
    return {
        "level_count": len(take_profit_orders),
        "quantities": [round(item, 8) for item in quantities],
        "trigger_prices": [round(item, 8) for item in triggers],
        "total_tp_quantity": round(total_tp_qty, 8),
        "largest_tp_quantity": round(largest_tp_qty, 8),
        "first_tp_ratio": round(first_tp_ratio, 4),
        "runner_quantity": round(runner_qty, 8),
        "micro_full_tp_fallback": micro_full_tp_fallback,
        "issues": issues,
        "status": "attention" if any(item != "runner_waiting_for_trailing" for item in issues) else "ok",
    }


def _symbol_step_size(client: BinanceClient, symbol: str) -> float:
    try:
        exchange_info = client.exchange_info(symbol, "futures")
    except (AttributeError, BinanceAPIError):
        return 0.0
    row = next(
        (item for item in (exchange_info.get("symbols") or []) if item.get("symbol") == symbol.upper()),
        {},
    )
    filters = {item.get("filterType"): item for item in row.get("filters", []) if isinstance(item, dict)}
    lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE") or {}
    return _float(lot.get("stepSize"))


def _load_positions(settings: Settings) -> tuple[list[dict[str, Any]], str]:
    try:
        with BinanceClient(settings) as client:
            raw_positions = client.positions()
    except BinanceAPIError as exc:
        return [], str(exc)

    positions: list[dict[str, Any]] = []
    for item in raw_positions:
        qty = _float(item.get("positionAmt"))
        if abs(qty) <= 0:
            continue
        symbol = str(item.get("symbol") or "").upper()
        side = "LONG" if qty > 0 else "SHORT"
        entry_price = _float(item.get("entryPrice"))
        mark_price = _float(item.get("markPrice"))
        pnl = _float(item.get("unRealizedProfit"))
        leverage = int(_float(item.get("leverage")))
        margin = abs(qty * entry_price) / leverage if leverage > 0 and entry_price > 0 else 0.0
        pnl_pct_on_margin = (pnl / margin * 100.0) if margin > 0 else 0.0
        positions.append(
            {
                "symbol": symbol,
                "side": side,
                "quantity": round(abs(qty), 8),
                "entry_price": round(entry_price, 8),
                "mark_price": round(mark_price, 8),
                "leverage": leverage,
                "margin_usdt": round(margin, 6),
                "unrealized_pnl_usdt": round(pnl, 6),
                "unrealized_pnl_pct_on_margin": round(pnl_pct_on_margin, 4),
            }
        )
    return positions, ""


def _protective_summary_for_open_positions(
    settings: Settings,
    positions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    summaries: list[dict[str, Any]] = []
    try:
        with BinanceClient(settings) as client:
            for position in positions:
                symbol = str(position.get("symbol") or "").upper()
                algo_orders = client.open_algo_orders(symbol)
                stop_count = sum(
                    1 for item in algo_orders if str(item.get("orderType") or "").upper() == "STOP_MARKET"
                )
                take_profit_count = sum(
                    1 for item in algo_orders if str(item.get("orderType") or "").upper() == "TAKE_PROFIT_MARKET"
                )
                trailing_count = sum(
                    1 for item in algo_orders if str(item.get("orderType") or "").upper() == "TRAILING_STOP_MARKET"
                )
                take_profit_orders = [
                    item
                    for item in algo_orders
                    if str(item.get("orderType") or "").upper() == "TAKE_PROFIT_MARKET"
                ]
                step_size = _symbol_step_size(client, symbol)
                tp_health = _tp_ladder_health(
                    position_qty=_float(position.get("quantity")),
                    take_profit_orders=take_profit_orders,
                    trailing_count=trailing_count,
                    step_size=step_size,
                )
                missing = []
                if stop_count <= 0 and trailing_count <= 0:
                    missing.append("stop_or_trailing")
                if take_profit_count <= 0:
                    missing.append("take_profit")
                hard_tp_issues = [
                    str(item)
                    for item in tp_health["issues"]
                    if str(item) != "runner_waiting_for_trailing"
                ]
                if hard_tp_issues:
                    missing.extend(hard_tp_issues)
                if missing:
                    warnings.append(f"{symbol} missing protective coverage: {','.join(missing)}")
                summaries.append(
                    {
                        "symbol": symbol,
                        "open_algo_order_count": len(algo_orders),
                        "stop_loss_count": stop_count,
                        "take_profit_count": take_profit_count,
                        "trailing_stop_count": trailing_count,
                        "take_profit_ladder": tp_health,
                        "coverage": "ok" if not missing else "attention",
                    }
                )
    except BinanceAPIError as exc:
        warnings.append(f"protective-order-check-unavailable: {exc}")
    return summaries, warnings


def _recent_live_orders_for_positions(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    symbols = {str(item.get("symbol") or "").upper() for item in positions}
    rows: list[dict[str, Any]] = []
    for item in reversed(read_live_orders()):
        symbol = str(item.get("symbol") or "").upper()
        if symbol not in symbols:
            continue
        rows.append(
            {
                "timestamp": item.get("timestamp"),
                "symbol": symbol,
                "side": item.get("side"),
                "leverage": item.get("leverage"),
                "notional_usdt": item.get("notional_usdt"),
                "gross_notional_usdt": item.get("gross_notional_usdt"),
                "route_id": item.get("route_id"),
                "analysis_score": item.get("analysis_score"),
                "analysis_convergence": item.get("analysis_convergence"),
            }
        )
        if len(rows) >= len(symbols):
            break
    return list(reversed(rows))


def _load_latest_digest_summary(digest_dir: Path = N8N_DIGEST_DIR) -> dict[str, Any]:
    candidates = sorted(
        digest_dir.glob("*-daily-digest.json") if digest_dir.exists() else [],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {
            "available": False,
            "reason": "no-digest-report-found",
            "automation": "digest timer should generate state/n8n-digests/*-daily-digest.json",
        }
    latest = candidates[0]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "reason": f"latest-digest-unreadable: {exc}",
            "path": str(latest),
        }
    decision = payload.get("decision") or {}
    selected = decision.get("selected") or {}
    news = payload.get("news") or {}
    whale = payload.get("whale") or {}
    strategy_analysis = payload.get("strategy_analysis") or {}
    return {
        "available": True,
        "path": str(latest),
        "generated_at": payload.get("generated_at"),
        "decision": {
            "action": decision.get("action"),
            "reason": decision.get("reason"),
            "selected": {
                "symbol": selected.get("symbol"),
                "direction": selected.get("direction"),
                "adjusted_score": selected.get("adjusted_score"),
                "context_notes": selected.get("context_notes") or [],
                "route_side_feedback": selected.get("route_side_feedback") or {},
            }
            if selected
            else None,
        },
        "news": {
            "risk": (news.get("risk") or {}).get("risk_level"),
            "bias": (news.get("risk") or {}).get("bias"),
            "high_impact_count": (news.get("risk") or {}).get("high_impact_count"),
        },
        "whale": {
            "enabled": whale.get("enabled"),
            "available": whale.get("available"),
            "reason": whale.get("reason"),
            "signal": whale.get("signal"),
            "exchange_inflow_usd": whale.get("exchange_inflow_usd"),
            "exchange_outflow_usd": whale.get("exchange_outflow_usd"),
            "transaction_count": whale.get("transaction_count"),
        },
        "strategy_analysis": {
            "enabled": strategy_analysis.get("enabled"),
            "available": strategy_analysis.get("available"),
            "reason": strategy_analysis.get("reason"),
            "source": strategy_analysis.get("source"),
        },
    }


def _compact_risk_combo_surface(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "surface": row.get("surface"),
        "target_side": row.get("target_side"),
        "target_interval": row.get("target_interval"),
        "route_id": row.get("route_id"),
        "symbol": row.get("symbol"),
        "research_status": row.get("research_status"),
        "recovery_gate_passed": row.get("recovery_gate_passed"),
        "robust_recovery_gate_passed": row.get("robust_recovery_gate_passed"),
        "full": row.get("full"),
        "test": row.get("test"),
        "walk_forward": row.get("walk_forward"),
        "gate_reasons": row.get("gate_reasons"),
        "source_report_path": row.get("source_report_path"),
    }


def _compact_validation_plan_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "surface": item.get("surface"),
        "symbol": item.get("symbol"),
        "target_side": item.get("target_side"),
        "target_interval": item.get("target_interval"),
        "purpose": item.get("purpose"),
        "interactive_probe_command": item.get("interactive_probe_command"),
        "offline_validation_command": item.get("offline_validation_command"),
        "runtime_guidance": item.get("runtime_guidance"),
        "promotion_boundary": item.get("promotion_boundary"),
    }


def _default_matrix_risk_boundary() -> dict[str, Any]:
    return {
        "max_per_trade_risk_pct": 0.025,
        "max_per_trade_risk_percent": 2.5,
        "risk_ceiling_source": "operator_dashboard_default_for_legacy_matrix",
        "applies_to": "all_research_candidates_before_any_promotion",
        "changes_position_sizing": False,
        "opens_orders": False,
        "writes_execution_config": False,
        "mainnet_live_allowed": False,
    }


def _repair_plan_with_risk_boundary(item: dict[str, Any]) -> dict[str, Any]:
    row = dict(item)
    guardrails = row.get("guardrails") if isinstance(row.get("guardrails"), dict) else {}
    row["guardrails"] = {
        **guardrails,
        "max_per_trade_risk_pct": guardrails.get("max_per_trade_risk_pct", 0.025),
        "max_per_trade_risk_percent": guardrails.get("max_per_trade_risk_percent", 2.5),
        "mainnet_live_allowed": bool(guardrails.get("mainnet_live_allowed")),
    }
    return row


def _risk_combo_matrix_freshness(matrix_path: Path, *, sweep_dir: Path | None = None) -> dict[str, Any]:
    sweep_dir = sweep_dir or RISK_COMBO_SWEEP_DIR
    sweep_paths = sorted(
        sweep_dir.glob("*-risk-combo-sweep.json") if sweep_dir.exists() else [],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not sweep_paths:
        return {
            "status": "no_sweep_reports_found",
            "latest_sweep_path": None,
            "newer_sweep_count": 0,
            "newer_sweep_paths": [],
            "action": "run risk-combo-sweep before relying on the matrix for research coverage",
        }
    matrix_mtime = matrix_path.stat().st_mtime
    newer_paths = [path for path in sweep_paths if path.stat().st_mtime > matrix_mtime]
    latest_sweep = sweep_paths[0]
    return {
        "status": "stale_after_new_sweeps" if newer_paths else "current",
        "matrix_path": str(matrix_path),
        "matrix_mtime": datetime.fromtimestamp(matrix_mtime, timezone.utc).replace(microsecond=0).isoformat(),
        "latest_sweep_path": str(latest_sweep),
        "latest_sweep_mtime": datetime.fromtimestamp(latest_sweep.stat().st_mtime, timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "newer_sweep_count": len(newer_paths),
        "newer_sweep_paths": [str(path) for path in newer_paths[:5]],
        "action": (
            "rebuild risk-combo-matrix with the newer sweep reports before judging BUY/SELL coverage"
            if newer_paths
            else "matrix includes all sweep reports newer than or equal to its build time"
        ),
    }


def _load_latest_risk_combo_matrix_summary(
    matrix_dir: Path | None = None,
    *,
    plan_limit: int = 2,
    sweep_dir: Path | None = None,
) -> dict[str, Any]:
    matrix_dir = matrix_dir or RISK_COMBO_MATRIX_DIR
    candidates = sorted(
        matrix_dir.glob("*-risk-combo-matrix.json") if matrix_dir.exists() else [],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {
            "available": False,
            "reason": "no-risk-combo-matrix-report-found",
            "automation": "run risk-combo-matrix after research sweeps to summarize BUY/SELL x interval evidence",
            "mainnet_live_allowed": False,
        }
    latest = candidates[0]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "reason": f"latest-risk-combo-matrix-unreadable: {exc}",
            "path": str(latest),
            "mainnet_live_allowed": False,
        }

    best_surface = payload.get("best_surface") if isinstance(payload.get("best_surface"), dict) else {}
    validation_plan = payload.get("validation_plan") if isinstance(payload.get("validation_plan"), list) else []
    repair_plan = (
        payload.get("negative_surface_repair_plan")
        if isinstance(payload.get("negative_surface_repair_plan"), list)
        else []
    )
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    promotion_boundary = (
        payload.get("promotion_boundary") if isinstance(payload.get("promotion_boundary"), dict) else {}
    )
    risk_boundary = (
        payload.get("risk_boundary") if isinstance(payload.get("risk_boundary"), dict) else _default_matrix_risk_boundary()
    )
    promotion_boundary = {
        **promotion_boundary,
        "requires_robust_recovery_gate": promotion_boundary.get("requires_robust_recovery_gate", True),
        "requires_sufficient_test_trades": promotion_boundary.get("requires_sufficient_test_trades", True),
        "max_per_trade_risk_pct": promotion_boundary.get("max_per_trade_risk_pct", 0.025),
        "max_per_trade_risk_percent": promotion_boundary.get("max_per_trade_risk_percent", 2.5),
        "mainnet_live_allowed": bool(promotion_boundary.get("mainnet_live_allowed")),
    }
    robust_surface_count = int(_float(payload.get("robust_surface_count")))
    promising_surface_count = int(_float(payload.get("promising_surface_count")))
    if robust_surface_count > 0:
        status = "robust_research_candidate_found"
    elif promising_surface_count > 0:
        status = "promising_research_only"
    else:
        status = "no_promising_surface"

    return {
        "available": True,
        "status": status,
        "path": str(latest),
        "freshness": _risk_combo_matrix_freshness(latest, sweep_dir=sweep_dir),
        "generated_at": payload.get("generated_at"),
        "mode": payload.get("mode"),
        "input_report_count": int(_float(payload.get("input_report_count"))),
        "skipped_input_report_count": int(_float(payload.get("skipped_input_report_count"))),
        "skipped_input_reports": payload.get("skipped_input_reports")
        if isinstance(payload.get("skipped_input_reports"), list)
        else [],
        "surface_count": int(_float(payload.get("surface_count"))),
        "promising_surface_count": promising_surface_count,
        "emerging_positive_lead_count": int(_float(payload.get("emerging_positive_lead_count"))),
        "superseded_emerging_positive_lead_count": int(
            _float(payload.get("superseded_emerging_positive_lead_count"))
        ),
        "recent_failed_repair_identity_count": int(_float(payload.get("recent_failed_repair_identity_count"))),
        "robust_surface_count": robust_surface_count,
        "side_summary": payload.get("side_summary") if isinstance(payload.get("side_summary"), dict) else {},
        "horizon_summary": payload.get("horizon_summary") if isinstance(payload.get("horizon_summary"), dict) else {},
        "completion_audit": payload.get("completion_audit")
        if isinstance(payload.get("completion_audit"), dict)
        else {},
        "objective_scorecard": payload.get("objective_scorecard")
        if isinstance(payload.get("objective_scorecard"), dict)
        else {},
        "prompt_to_artifact_checklist": payload.get("prompt_to_artifact_checklist")
        if isinstance(payload.get("prompt_to_artifact_checklist"), dict)
        else {},
        "risk_boundary": risk_boundary,
        "safety": {
            "opens_orders": bool(safety.get("opens_orders")),
            "writes_execution_config": bool(safety.get("writes_execution_config")),
            "clears_route_quarantine": bool(safety.get("clears_route_quarantine")),
            "mainnet_live_allowed": bool(safety.get("mainnet_live_allowed")),
        },
        "mainnet_live_allowed": bool(
            safety.get("mainnet_live_allowed") or promotion_boundary.get("mainnet_live_allowed")
        ),
        "best_surface": _compact_risk_combo_surface(best_surface) if best_surface else None,
        "emerging_positive_leads": [
            _compact_risk_combo_surface(item)
            for item in (payload.get("emerging_positive_leads") or [])[: max(int(plan_limit), 0)]
            if isinstance(item, dict)
        ],
        "superseded_emerging_positive_leads": [
            _compact_risk_combo_surface(item)
            for item in (payload.get("superseded_emerging_positive_leads") or [])[: max(int(plan_limit), 0)]
            if isinstance(item, dict)
        ],
        "validation_plan": [
            _compact_validation_plan_item(item)
            for item in validation_plan[: max(int(plan_limit), 0)]
            if isinstance(item, dict)
        ],
        "negative_surface_repair_plan": [
            _repair_plan_with_risk_boundary(item)
            for item in repair_plan[: max(int(plan_limit), 0)]
            if isinstance(item, dict)
        ],
        "next_research_actions": payload.get("next_research_actions") or [],
        "promotion_boundary": promotion_boundary,
    }


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


def _horizon_candidate_status(
    horizon: str,
    *,
    matrix_summary: dict[str, Any],
    readiness_report: dict[str, Any],
) -> dict[str, Any]:
    matrix_horizons = matrix_summary.get("horizon_summary") if isinstance(matrix_summary.get("horizon_summary"), dict) else {}
    matrix_row = matrix_horizons.get(horizon) if isinstance(matrix_horizons.get(horizon), dict) else {}
    research_report = (
        readiness_report.get("research_candidate_report")
        if isinstance(readiness_report.get("research_candidate_report"), dict)
        else {}
    )
    reviewable_horizons = (
        research_report.get("reviewable_horizon_counts")
        if isinstance(research_report.get("reviewable_horizon_counts"), dict)
        else {}
    )
    horizon_candidates = [
        item
        for item in (research_report.get("top_candidates") or [])
        if isinstance(item, dict) and str(item.get("horizon") or "") == horizon
    ]
    trade_ready = next((item for item in horizon_candidates if item.get("trade_readiness_allowed")), None)
    reviewable = int(_float(reviewable_horizons.get(horizon)))
    promising = int(_float(matrix_row.get("promising_surface_count")))
    emerging = int(_float(matrix_row.get("emerging_positive_lead_count")))
    robust = int(_float(matrix_row.get("robust_surface_count")))
    best_surface = matrix_row.get("best_surface") if isinstance(matrix_row.get("best_surface"), dict) else None
    representative = trade_ready or (horizon_candidates[0] if horizon_candidates else None)

    if trade_ready:
        status = "testnet_ready_candidate"
        repair_action = "run_trade_decision_then_decision_audit"
    elif robust > 0:
        status = "candidate"
        repair_action = "run_ai_readiness_scan_for_robust_candidate"
    elif promising > 0 or reviewable > 0:
        status = "candidate"
        repair_action = "expand_walk_forward_and_readiness_validation"
    elif emerging > 0:
        status = "blocked"
        repair_action = "expand_sample_before_promotion"
    else:
        status = "blocked"
        repair_action = f"run_{horizon}_horizon_research_sweep"

    return {
        "horizon": horizon,
        "status": status,
        "trade_ready": bool(trade_ready),
        "reviewable_candidate_count": reviewable,
        "promising_surface_count": promising,
        "emerging_positive_lead_count": emerging,
        "robust_surface_count": robust,
        "best_surface": _compact_risk_combo_surface(best_surface) if best_surface else None,
        "readiness_candidate": {
            "symbol": representative.get("symbol"),
            "side": representative.get("side"),
            "interval": representative.get("interval"),
            "route_id": representative.get("route_id"),
            "research_status": representative.get("research_status"),
            "next_action": representative.get("readiness_next_action"),
        }
        if isinstance(representative, dict)
        else None,
        "repair_action": repair_action,
    }


def _build_candidate_pool(
    *,
    risk_combo_matrix: dict[str, Any],
    product_readiness: dict[str, Any],
    decision_audit: dict[str, Any],
    readiness_dir: Path | None = None,
) -> dict[str, Any]:
    readiness_payload, readiness_path = _latest_json_report(
        readiness_dir or READINESS_SCAN_DIR,
        "*-hermes-readiness-scan.json",
    )
    readiness_allowed = int(_float(readiness_payload.get("allowed_count"))) if readiness_payload else 0
    decision_status = str(decision_audit.get("status") or "")
    product_testnet_ready = bool(product_readiness.get("testnet_trade_ready"))
    horizons = {
        horizon: _horizon_candidate_status(
            horizon,
            matrix_summary=risk_combo_matrix,
            readiness_report=readiness_payload,
        )
        for horizon in ("short", "medium", "long")
    }
    missing_horizons = [
        horizon
        for horizon, row in horizons.items()
        if row["status"] == "blocked" and row["promising_surface_count"] <= 0 and row["reviewable_candidate_count"] <= 0
    ]
    ready_horizons = [horizon for horizon, row in horizons.items() if row["trade_ready"]]
    simulation_allowed = (
        readiness_allowed > 0
        and bool(ready_horizons)
        and decision_status == "passed"
        and product_testnet_ready
    )
    if simulation_allowed:
        next_action = "run_trade_decision_then_operator_approved_testnet_execution"
    elif readiness_allowed <= 0:
        next_action = "continue_scan_research_and_readiness_repairs"
    elif decision_status != "passed":
        next_action = "repair_decision_audit_before_testnet_execution"
    else:
        next_action = "repair_product_readiness_before_testnet_execution"
    return {
        "mode": "short_medium_long_candidate_pool_v1",
        "simulation_trade_allowed": simulation_allowed,
        "readiness_allowed_count": readiness_allowed,
        "ready_horizons": ready_horizons,
        "missing_horizons": missing_horizons,
        "horizons": horizons,
        "latest_readiness_scan_path": readiness_path or None,
        "decision_audit_status": decision_status,
        "product_testnet_trade_ready": product_testnet_ready,
        "next_action": next_action,
        "guardrails": {
            "mainnet_live_allowed": False,
            "requires_decision_audit_passed": True,
            "requires_readiness_allowed_candidate": True,
            "hold_is_valid_when_no_candidate": True,
        },
    }


def _build_customer_feedback(
    *,
    position_count: int,
    unrealized_pnl: float,
    realized_pnl: float,
    live_order_journal_count: int,
    loss_findings: list[str],
    protective_warnings: list[str],
    digest_summary: dict[str, Any],
) -> list[str]:
    feedback: list[str] = []
    if position_count <= 0:
        feedback.append("No open testnet positions; explorer should keep scanning until an allowed candidate appears.")
        if live_order_journal_count > 0:
            feedback.append(
                "Live/testnet order journal has historical entries, but the exchange position check is flat; "
                "treat journal count as audit history, not current exposure."
            )
    elif unrealized_pnl > 0:
        feedback.append("Current open basket is profitable; keep guardian active and let TP/trailing rules work.")
    elif unrealized_pnl < 0:
        feedback.append("Current open basket is negative; avoid adding size until guardian confirms protective coverage.")
    else:
        feedback.append("Current open basket is flat; continue collecting exchange feedback.")

    if realized_pnl < 0:
        feedback.append("Historical realized PnL is negative; prioritize loss-cause feedback over raising leverage globally.")
    if protective_warnings:
        feedback.append("Protective coverage needs attention before increasing leverage.")
    if any("stop-loss-dominant" in item for item in loss_findings):
        feedback.append("Stop-loss exits dominate the loss history; tighten entry quality or arm trailing only after profit activation.")
    if any("short-lane" in item for item in loss_findings):
        feedback.append("Short-side history remains weak; prefer long-biased testnet exploration until short buckets improve.")
    if any("fast-stop-cluster" in item for item in loss_findings):
        feedback.append("Many losses are being stopped quickly; require better flow confirmation or wider volatility-adjusted stops before increasing leverage.")
    if digest_summary.get("available"):
        whale = digest_summary.get("whale") or {}
        if whale.get("enabled") is False:
            feedback.append("News observation is automatic, but whale wallet data is neutral/unavailable until WHALE_ALERT_API_KEY is configured.")
    else:
        feedback.append("Digest automation has no recent report; check the quant-research/testnet explorer timers before trusting external context.")
    if not feedback:
        feedback.append("No major operator findings; continue automated testnet iteration.")
    return feedback


def _build_product_readiness(
    *,
    settings: Settings,
    position_error: str,
    position_count: int,
    protective_warnings: list[str],
    closed_summary: dict[str, Any],
    loss_summary: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    strengths: list[str] = []

    profit_factor = _float(loss_summary.get("profit_factor"), _float(closed_summary.get("profit_factor")))
    avg_r = _float(loss_summary.get("avg_r"))
    stop_loss_ratio = _float(loss_summary.get("stop_loss_ratio"), _float(closed_summary.get("pure_stop_loss_ratio")))
    closed_count = int(_float(closed_summary.get("count")))
    realized_pnl = _float(closed_summary.get("total_realized_pnl_usdt"))

    if position_error:
        blockers.append(f"exchange_position_check_unavailable:{position_error}")
    if settings.live_trading_enabled:
        warnings.append("mainnet_live_trading_enabled; verify explicit operator approval and readiness before customer use")
    else:
        blockers.append("mainnet_live_trading_disabled")
    if closed_count <= 0:
        blockers.append("no_closed_trade_evidence")
    if realized_pnl < 0:
        blockers.append(f"negative_realized_pnl:{realized_pnl:.4f}USDT")
    if profit_factor < 1.0:
        blockers.append(f"profit_factor_below_breakeven:{profit_factor:.4f}")
    if avg_r < 0.0:
        blockers.append(f"negative_average_r:{avg_r:.4f}")
    if stop_loss_ratio > 65.0:
        blockers.append(f"stop_loss_ratio_high:{stop_loss_ratio:.2f}%")
    if protective_warnings:
        blockers.append("protective_coverage_attention")

    if settings.use_testnet:
        strengths.append("testnet_mode_enabled")
    if not settings.live_trading_enabled:
        strengths.append("mainnet_safety_lock_enabled")
    if position_count == 0:
        strengths.append("no_current_exchange_exposure")
    if position_count > 0 and not protective_warnings:
        strengths.append("open_positions_have_protective_coverage")

    profitability_evidence = (
        closed_count >= 100
        and realized_pnl > 0
        and profit_factor >= 1.2
        and avg_r > 0.05
        and stop_loss_ratio <= 55.0
    )
    testnet_trade_ready = (
        not position_error
        and not protective_warnings
        and settings.testnet_trading_enabled
        and profit_factor >= 0.85
        and avg_r >= -0.05
        and stop_loss_ratio <= 65.0
    )
    mainnet_customer_ready = settings.live_trading_enabled and profitability_evidence and not blockers
    if mainnet_customer_ready:
        stage = "customer_trade_ready"
        status = "ready"
        next_action = "operate_with_guardian_and_closed_trade_review"
    elif testnet_trade_ready:
        stage = "testnet_exploration_only"
        status = "conditional"
        next_action = "continue_testnet_with_readiness_scan_and_operator_approval"
    else:
        stage = "watch_only_research"
        status = "blocked"
        next_action = "repair_expectancy_before_enabling_entries"

    return {
        "status": status,
        "stage": stage,
        "mainnet_customer_ready": mainnet_customer_ready,
        "testnet_trade_ready": testnet_trade_ready,
        "profitability_evidence": profitability_evidence,
        "metrics": {
            "closed_trade_count": closed_count,
            "realized_pnl_usdt": round(realized_pnl, 6),
            "profit_factor": round(profit_factor, 4),
            "avg_r": round(avg_r, 4),
            "stop_loss_ratio": round(stop_loss_ratio, 2),
        },
        "blockers": blockers,
        "warnings": warnings,
        "strengths": strengths,
        "next_action": next_action,
    }


def build_operator_dashboard(
    settings: Settings,
    *,
    min_bucket_trades: int = 10,
    top_n: int = 5,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    OPERATOR_DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    positions, position_error = _load_positions(settings)
    protective, protective_warnings = _protective_summary_for_open_positions(settings, positions)
    closed_summary = summarize_closed_trade_reviews()
    live_order_summary = summarize_live_orders()
    loss_report = run_loss_diagnostics(min_bucket_trades=min_bucket_trades, top_n=top_n)
    decision_audit = run_decision_audit(
        output_dir=OPERATOR_DASHBOARD_DIR / "decision-audit",
        since_contract=True,
    )
    loss_summary = loss_report.get("summary") if isinstance(loss_report.get("summary"), dict) else {}
    digest_summary = _load_latest_digest_summary()
    risk_combo_matrix = _load_latest_risk_combo_matrix_summary()
    unrealized_pnl = round(sum(_float(item.get("unrealized_pnl_usdt")) for item in positions), 6)
    realized_pnl = _float(closed_summary.get("total_realized_pnl_usdt"))
    live_order_journal_count = int(live_order_summary.get("count") or 0)
    product_readiness = _build_product_readiness(
        settings=settings,
        position_error=position_error,
        position_count=len(positions),
        protective_warnings=protective_warnings,
        closed_summary=closed_summary,
        loss_summary=loss_summary,
    )
    candidate_pool = _build_candidate_pool(
        risk_combo_matrix=risk_combo_matrix,
        product_readiness=product_readiness,
        decision_audit=decision_audit,
    )

    payload = {
        "generated_at": _utc_now().isoformat(),
        "status": "ok" if not position_error else "degraded",
        "position_error": position_error or None,
        "mode": {
            "use_testnet": settings.use_testnet,
            "live_trading_enabled": settings.live_trading_enabled,
            "testnet_trading_enabled": settings.testnet_trading_enabled,
        },
        "customer_summary": {
            "open_position_count": len(positions),
            "open_unrealized_pnl_usdt": unrealized_pnl,
            "closed_review_count": closed_summary.get("count", 0),
            "closed_realized_pnl_usdt": round(realized_pnl, 6),
            "live_order_count": live_order_journal_count,
            "live_order_count_meaning": "append_only_live_testnet_order_journal_records_not_current_open_orders",
        },
        "positions": positions,
        "protective_orders": protective,
        "recent_entry_orders": _recent_live_orders_for_positions(positions),
        "execution_journal": {
            "record_count": live_order_journal_count,
            "buy_count": live_order_summary.get("buy_count", 0),
            "sell_count": live_order_summary.get("sell_count", 0),
            "latest": live_order_summary.get("latest"),
            "meaning": "append-only order audit history; current exposure comes from positions/protective_orders",
        },
        "product_readiness": product_readiness,
        "candidate_pool": candidate_pool,
        "decision_artifact_audit": {
            "status": decision_audit.get("status"),
            "scope": decision_audit.get("scope"),
            "summary": decision_audit.get("summary"),
            "invalid_artifacts": decision_audit.get("invalid_artifacts"),
            "report_path": decision_audit.get("report_path"),
        },
        "risk_combo_matrix": risk_combo_matrix,
        "loss_diagnostics": {
            "summary": loss_summary,
            "findings": loss_report.get("findings", [])[:top_n],
            "worst_buckets": loss_report.get("worst_buckets", [])[:top_n],
            "root_cause_recommendations": loss_report.get("root_cause_recommendations", [])[:top_n],
            "report_path": loss_report.get("report_path"),
        },
        "external_context_automation": digest_summary,
        "operator_feedback": _build_customer_feedback(
            position_count=len(positions),
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            live_order_journal_count=live_order_journal_count,
            loss_findings=list(loss_report.get("findings", [])),
            protective_warnings=protective_warnings,
            digest_summary=digest_summary,
        ),
        "automation_expectation": {
            "local_first": True,
            "llm_usage": "decision-level only; dashboard uses local state and Binance read-only checks",
            "next_actions": [
                "Keep testnet explorer running until position cap is reached.",
                "Keep guardian running until each position exits by TP, SL, or armed trailing stop.",
                "Run closed-trade review after exits, then feed loss buckets back into sizing and route filters.",
            ],
        },
    }
    report_path = OPERATOR_DASHBOARD_DIR / f"{_stamp()}-operator-dashboard.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload
