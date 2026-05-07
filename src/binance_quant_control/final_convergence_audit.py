from __future__ import annotations

import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .config import STATE_DIR, ensure_runtime_dirs
from .order_journal import read_closed_trade_reviews, read_live_orders, read_paper_orders
from .strategy_optimizer import run_strategy_optimizer

FINAL_AUDIT_STATE_DIR = STATE_DIR / "final-convergence-audit"
CORE_ROUTES = {"btc-core", "eth-core", "xau-macro"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _cohort_breakdown(reviews: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for item in reviews:
        cohort_id = str(item.get("cohort_id") or "uncohorted")
        row = rows.setdefault(
            cohort_id,
            {
                "route_id": str(item.get("route_id") or "unrouted"),
                "asset_class": str(item.get("asset_class") or "unknown"),
                "strategy_profile": str(item.get("strategy_profile") or ""),
                "count": 0,
                "wins": 0,
                "losses": 0,
                "sources": Counter(),
            },
        )
        pnl = float(item.get("realized_pnl_usdt") or 0.0)
        row["count"] += 1
        row["wins"] += 1 if pnl > 0 else 0
        row["losses"] += 1 if pnl < 0 else 0
        note = str(item.get("note") or "")
        source = "live-or-manual"
        if "binance_public_history" in note:
            source = "binance-public-history"
        elif "market_replay" in note:
            source = "market-replay"
        row["sources"][source] += 1
    return {
        cohort_id: {
            **row,
            "sources": dict(row["sources"]),
        }
        for cohort_id, row in rows.items()
    }


def _one_in_one_out_findings(live_orders: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> list[str]:
    findings: list[str] = []
    missing_context = [
        item for item in live_orders
        if not item.get("cohort_id") or not item.get("route_id") or not item.get("entry_reason_snapshot")
    ]
    if missing_context:
        findings.append(f"{len(missing_context)} live order(s) still lack route/cohort context")
    reviewed_order_ids = {
        str(item.get("source_order_id"))
        for item in reviews
        if item.get("source_order_id") is not None
    }
    open_regular_by_symbol = Counter(
        str(item.get("symbol") or "")
        for item in live_orders
        if str(item.get("status") or "").upper() in {"NEW", "PARTIALLY_FILLED"}
        and str(item.get("order_id")) not in reviewed_order_ids
    )
    repeated = {symbol: count for symbol, count in open_regular_by_symbol.items() if symbol and count > 1}
    if repeated:
        findings.append(f"multiple unreviewed live-order journal entries by symbol: {repeated}")
    return findings


def _run_hailo_once() -> dict[str, Any]:
    command = ["/home/robert/python/bin/openclaw-hailo-triage", "--once"]
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "command": command, "error": str(exc)}
    stdout = (completed.stdout or "").strip()
    response = None
    if stdout:
        try:
            response = json.loads(stdout)
        except json.JSONDecodeError:
            response = None
    return {
        "available": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "response": response,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": (completed.stderr or "").strip()[-2000:],
    }


def run_final_convergence_audit(*, run_hailo: bool = True) -> dict[str, Any]:
    ensure_runtime_dirs()
    FINAL_AUDIT_STATE_DIR.mkdir(parents=True, exist_ok=True)
    reviews = read_closed_trade_reviews()
    live_orders = read_live_orders()
    paper_orders = read_paper_orders()
    cohort_breakdown = _cohort_breakdown(reviews)
    route_counts = Counter(str(item.get("route_id") or "unrouted") for item in reviews)
    source_counts = Counter()
    for item in reviews:
        note = str(item.get("note") or "")
        if "binance_public_history" in note:
            source_counts["binance-public-history"] += 1
        elif "market_replay" in note:
            source_counts["market-replay"] += 1
        else:
            source_counts["live-or-manual"] += 1

    core_status = {}
    for route_id in CORE_ROUTES:
        route_cohorts = {
            cohort_id: row
            for cohort_id, row in cohort_breakdown.items()
            if row["route_id"] == route_id
        }
        core_status[route_id] = {
            "review_count": sum(int(row["count"]) for row in route_cohorts.values()),
            "cohorts": route_cohorts,
            "has_validation_cohort": any(int(row["count"]) >= 50 for row in route_cohorts.values()),
        }

    optimizer = run_strategy_optimizer()
    findings: list[str] = []
    for route_id, status in core_status.items():
        if not status["has_validation_cohort"]:
            findings.append(f"{route_id} has no cohort with >=50 closed reviews")
    findings.extend(_one_in_one_out_findings(live_orders, reviews))
    if optimizer.get("promotion_decision") == "reject":
        findings.append("optimizer still rejects promotion")
    hailo = _run_hailo_once() if run_hailo else {"available": None, "skipped": True}
    if run_hailo and not hailo.get("available"):
        findings.append("hailo triage did not complete cleanly")

    payload = {
        "generated_at": _utc_now().isoformat(),
        "status": "ready" if not findings else "needs-attention",
        "review_database": {
            "closed_review_count": len(reviews),
            "paper_order_count": len(paper_orders),
            "live_order_count": len(live_orders),
            "route_counts": dict(route_counts),
            "source_counts": dict(source_counts),
            "cohort_breakdown": cohort_breakdown,
            "core_status": core_status,
        },
        "one_in_one_out": {
            "findings": _one_in_one_out_findings(live_orders, reviews),
        },
        "optimizer": {
            "status": optimizer.get("status"),
            "review_count": optimizer.get("review_count"),
            "screening_status": optimizer.get("screening_status"),
            "validation_status": optimizer.get("validation_status"),
            "promotion_decision": optimizer.get("promotion_decision"),
            "report_path": optimizer.get("report_path"),
        },
        "hailo": hailo,
        "findings": findings,
    }
    report_path = FINAL_AUDIT_STATE_DIR / f"{_stamp()}-final-convergence-audit.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload
