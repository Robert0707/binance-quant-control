from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .binance_api import BinanceAPIError, BinanceClient
from .config import STATE_DIR, Settings, ensure_runtime_dirs
from .loss_diagnostics import run_loss_diagnostics
from .order_journal import (
    read_live_orders,
    summarize_closed_trade_reviews,
    summarize_live_orders,
)

OPERATOR_DASHBOARD_DIR = STATE_DIR / "operator-dashboard"
N8N_DIGEST_DIR = STATE_DIR / "n8n-digests"


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


def _build_customer_feedback(
    *,
    position_count: int,
    unrealized_pnl: float,
    realized_pnl: float,
    loss_findings: list[str],
    protective_warnings: list[str],
    digest_summary: dict[str, Any],
) -> list[str]:
    feedback: list[str] = []
    if position_count <= 0:
        feedback.append("No open testnet positions; explorer should keep scanning until an allowed candidate appears.")
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
    loss_report = run_loss_diagnostics(min_bucket_trades=min_bucket_trades, top_n=top_n)
    digest_summary = _load_latest_digest_summary()
    unrealized_pnl = round(sum(_float(item.get("unrealized_pnl_usdt")) for item in positions), 6)
    realized_pnl = _float(closed_summary.get("total_realized_pnl_usdt"))

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
            "live_order_count": summarize_live_orders().get("count", 0),
        },
        "positions": positions,
        "protective_orders": protective,
        "recent_entry_orders": _recent_live_orders_for_positions(positions),
        "loss_diagnostics": {
            "summary": loss_report.get("summary"),
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
