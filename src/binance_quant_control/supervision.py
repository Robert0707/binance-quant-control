from __future__ import annotations

import json
import math
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .asset_routing import resolve_symbol_route
from .config import PROJECT_ROOT, STATE_DIR, ensure_runtime_dirs, load_settings
from .convergence import (
    calculate_loss_streak,
    calculate_max_drawdown_pct,
    calculate_profit_factor,
)
from .daily_digest import build_digest
from .daily_digest import load_config as load_digest_config
from .final_convergence_audit import run_final_convergence_audit
from .mission_control import run_trading_mission
from .order_journal import read_closed_trade_reviews, read_paper_orders
from .route_risk_control import update_route_quarantine_from_snapshot
from .strategy_optimizer import evaluate_optimizer_live_gate, run_strategy_optimizer

DEFAULT_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "XAUTUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "TRXUSDT",
    "UNIUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "DOGEUSDT",
    "1000PEPEUSDT",
    "WIFUSDT",
)
SUPERVISION_STATE_DIR = STATE_DIR / "delivery-supervision"


@dataclass(frozen=True, slots=True)
class SupervisorPolicy:
    cycles: int
    training_rounds: int
    symbols: tuple[str, ...]
    mission_symbols_per_cycle: int
    target_return_pct: float
    max_leverage: float
    margin_notional_usdt: float
    optimize_every: int
    max_recent_loss_usdt: float
    max_route_loss_streak: int
    min_route_profit_factor: float
    route_lookback: int
    build_digest_every: int
    audit_every: int
    stop_on_optimizer_promotion: bool


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_float(value: float) -> float | str:
    return round(value, 4) if math.isfinite(value) else "inf"


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _rotate_symbols(symbols: tuple[str, ...], *, cycle_index: int, count: int) -> list[str]:
    if not symbols:
        return []
    count = max(1, min(count, len(symbols)))
    start = (cycle_index * count) % len(symbols)
    return [symbols[(start + offset) % len(symbols)] for offset in range(count)]


def _run_training_command(policy: SupervisorPolicy) -> dict[str, Any]:
    command = [
        "python3",
        "scripts/run_demo_training.py",
        "--rounds",
        str(policy.training_rounds),
        "--symbols",
        ",".join(policy.symbols),
        "--target-return-pct",
        str(policy.target_return_pct),
        "--max-leverage",
        str(policy.max_leverage),
        "--margin-notional-usdt",
        str(policy.margin_notional_usdt),
        "--optimize-every",
        str(policy.optimize_every),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=max(900, policy.training_rounds * 180),
        check=False,
    )
    stdout = (completed.stdout or "").strip()
    payload: dict[str, Any] | None = None
    if stdout:
        try:
            raw = json.loads(stdout)
            if isinstance(raw, dict):
                payload = raw
        except json.JSONDecodeError:
            payload = None
    return {
        "command": command,
        "returncode": completed.returncode,
        "response": payload,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": (completed.stderr or "").strip()[-4000:],
    }


def _recent_review_pnl(reviews: list[dict[str, Any]], *, hours: float = 24.0) -> float:
    cutoff = _utc_now() - timedelta(hours=hours)
    total = 0.0
    for item in reviews:
        timestamp = _parse_datetime(item.get("reviewed_at") or item.get("closed_at"))
        if timestamp is None or timestamp < cutoff:
            continue
        total += _safe_float(item.get("realized_pnl_usdt"))
    return round(total, 8)


def _route_risk_snapshot(
    reviews: list[dict[str, Any]],
    *,
    route_lookback: int,
    min_route_profit_factor: float,
    max_route_loss_streak: int,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in reviews:
        route_id = str(item.get("route_id") or "unrouted")
        grouped.setdefault(route_id, []).append(item)

    routes: dict[str, Any] = {}
    quarantined: list[str] = []
    for route_id, items in grouped.items():
        recent = items[-route_lookback:] if route_lookback > 0 else items
        pnls = [_safe_float(item.get("realized_pnl_usdt")) for item in recent]
        wins = sum(1 for item in pnls if item > 0.0)
        losses = sum(1 for item in pnls if item < 0.0)
        profit_factor = calculate_profit_factor(pnls)
        loss_streak = calculate_loss_streak(pnls)
        drawdown = calculate_max_drawdown_pct(pnls)
        reasons: list[str] = []
        if len(recent) >= 10 and profit_factor < min_route_profit_factor:
            reasons.append(
                f"profit-factor {profit_factor:.3f} below floor {min_route_profit_factor:.3f}"
            )
        if loss_streak >= max_route_loss_streak:
            reasons.append(
                f"loss-streak {loss_streak} reached ceiling {max_route_loss_streak}"
            )
        if reasons:
            quarantined.append(route_id)
        routes[route_id] = {
            "count": len(recent),
            "wins": wins,
            "losses": losses,
            "win_rate": round((wins / len(recent)) * 100.0, 4) if recent else 0.0,
            "profit_factor": _json_float(profit_factor),
            "max_drawdown_pct": round(drawdown, 4),
            "loss_streak": loss_streak,
            "quarantined": bool(reasons),
            "quarantine_reasons": reasons,
        }
    return {"routes": routes, "quarantined_routes": sorted(quarantined)}


def _database_snapshot() -> dict[str, Any]:
    reviews = read_closed_trade_reviews()
    paper_orders = read_paper_orders()
    symbols = sorted({str(item.get("symbol") or "").upper() for item in reviews if item.get("symbol")})
    route_counts = Counter(str(item.get("route_id") or "unrouted") for item in reviews)
    return {
        "closed_review_count": len(reviews),
        "closed_review_symbols": len(symbols),
        "paper_order_count": len(paper_orders),
        "route_counts": dict(route_counts),
        "symbols": symbols,
        "recent_24h_realized_pnl_usdt": _recent_review_pnl(reviews),
    }


def _latest_digest_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    return {
        "output_path": payload.get("output_path"),
        "decision": payload.get("decision"),
        "news_risk": ((payload.get("news") or {}).get("risk") or {}),
        "strategy_analysis": payload.get("strategy_analysis"),
    }


def run_delivery_supervisor(policy: SupervisorPolicy) -> dict[str, Any]:
    ensure_runtime_dirs()
    SUPERVISION_STATE_DIR.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    started_at = _utc_now()
    cycles: list[dict[str, Any]] = []
    stop_reasons: list[str] = []
    digest_payload: dict[str, Any] | None = None

    for cycle_index in range(policy.cycles):
        cycle_errors: list[str] = []
        if policy.build_digest_every > 0 and cycle_index % policy.build_digest_every == 0:
            try:
                digest_payload = build_digest(
                    load_digest_config(PROJECT_ROOT / "config" / "n8n-daily-digest.default.json")
                )
            except Exception as exc:  # noqa: BLE001 - supervision must keep the audit trail alive.
                cycle_errors.append(f"digest-failed:{exc}")

        mission_symbols = _rotate_symbols(
            policy.symbols,
            cycle_index=cycle_index,
            count=policy.mission_symbols_per_cycle,
        )
        try:
            mission = run_trading_mission(
                symbols=mission_symbols,
                target_return_pct=policy.target_return_pct,
                max_leverage=policy.max_leverage,
                execute_live=False,
            )
        except Exception as exc:  # noqa: BLE001
            cycle_errors.append(f"mission-failed:{exc}")
            mission = {"report_path": None, "mission_actions": [], "system_findings": [str(exc)]}
        training = _run_training_command(policy)
        try:
            optimizer = run_strategy_optimizer()
        except Exception as exc:  # noqa: BLE001
            cycle_errors.append(f"optimizer-failed:{exc}")
            optimizer = {"status": "error", "promotion_decision": None, "report_path": None}
        reviews = read_closed_trade_reviews()
        route_risk = _route_risk_snapshot(
            reviews,
            route_lookback=policy.route_lookback,
            min_route_profit_factor=policy.min_route_profit_factor,
            max_route_loss_streak=policy.max_route_loss_streak,
        )
        route_risk_control = update_route_quarantine_from_snapshot(
            route_risk,
            updated_by="delivery-supervisor",
        )
        database = _database_snapshot()
        optimizer_gate = evaluate_optimizer_live_gate()
        audit = None
        if policy.audit_every > 0 and ((cycle_index + 1) % policy.audit_every == 0):
            try:
                audit = run_final_convergence_audit(run_hailo=True)
            except Exception as exc:  # noqa: BLE001
                cycle_errors.append(f"audit-failed:{exc}")
                audit = {"status": "error", "findings": [str(exc)], "report_path": None}

        cycle_stop_reasons: list[str] = []
        if cycle_errors:
            cycle_stop_reasons.append("supervision-cycle-error")
        if training["returncode"] != 0:
            cycle_stop_reasons.append("demo-training-command-failed")
        if database["recent_24h_realized_pnl_usdt"] <= -abs(policy.max_recent_loss_usdt):
            cycle_stop_reasons.append("recent-24h-simulated-loss-limit-hit")
        if route_risk["quarantined_routes"]:
            cycle_stop_reasons.append("one-or-more-routes-quarantined-by-performance")
        if policy.stop_on_optimizer_promotion and optimizer_gate["allowed"]:
            cycle_stop_reasons.append("optimizer-promoted-stop-for-human-review-before-live")

        cycles.append(
            {
                "cycle": cycle_index + 1,
                "generated_at": _utc_now().isoformat(),
                "settings": {
                    "use_testnet": settings.use_testnet,
                    "live_trading_enabled": settings.live_trading_enabled,
                },
                "digest": _latest_digest_summary(digest_payload),
                "mission": {
                    "symbols": mission_symbols,
                    "report_path": mission.get("report_path"),
                    "actions": mission.get("mission_actions"),
                    "selected_symbol": ((mission.get("selected_candidate") or {}).get("symbol")),
                    "simulation_status": ((mission.get("simulation") or {}).get("status")),
                    "system_findings": mission.get("system_findings"),
                },
                "training": {
                    "returncode": training["returncode"],
                    "response": training.get("response"),
                    "stderr_tail": training.get("stderr_tail"),
                },
                "optimizer": {
                    "status": optimizer.get("status"),
                    "review_count": optimizer.get("review_count"),
                    "promotion_decision": optimizer.get("promotion_decision"),
                    "report_path": optimizer.get("report_path"),
                },
                "optimizer_live_gate": optimizer_gate,
                "route_risk": route_risk,
                "route_risk_control": {
                    "active_quarantined_routes": route_risk_control.get("active_quarantined_routes"),
                    "path": str(STATE_DIR / "route-risk-control.json"),
                },
                "database": database,
                "audit": {
                    "status": audit.get("status"),
                    "findings": audit.get("findings"),
                    "report_path": audit.get("report_path"),
                }
                if audit
                else None,
                "cycle_errors": cycle_errors,
                "cycle_stop_reasons": cycle_stop_reasons,
            }
        )
        stop_reasons.extend(cycle_stop_reasons)
        if cycle_stop_reasons:
            break

    payload = {
        "generated_at": _utc_now().isoformat(),
        "started_at": started_at.isoformat(),
        "status": "stopped-for-risk-review" if stop_reasons else "ok",
        "mode": "paper_demo_supervision_only",
        "policy": asdict(policy),
        "cycles_completed": len(cycles),
        "stop_reasons": sorted(set(stop_reasons)),
        "live_guardrail": {
            "live_trading_enabled": settings.live_trading_enabled,
            "real_orders_sent_by_supervisor": False,
            "reason": "Delivery supervisor never calls live execution; it only writes paper/demo validation records.",
        },
        "cycles": cycles,
    }
    report_path = SUPERVISION_STATE_DIR / f"{_stamp()}-delivery-supervision.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload


def build_supervisor_policy(
    *,
    cycles: int,
    training_rounds: int,
    symbols: list[str] | None = None,
    mission_symbols_per_cycle: int = 6,
    target_return_pct: float = 5.0,
    max_leverage: float = 3.0,
    margin_notional_usdt: float = 3.0,
    optimize_every: int = 10,
    max_recent_loss_usdt: float = 5.0,
    max_route_loss_streak: int = 5,
    min_route_profit_factor: float = 0.8,
    route_lookback: int = 40,
    build_digest_every: int = 1,
    audit_every: int = 1,
    stop_on_optimizer_promotion: bool = True,
) -> SupervisorPolicy:
    normalized_symbols = tuple(
        str(item).strip().upper()
        for item in (symbols or list(DEFAULT_SYMBOLS))
        if str(item).strip()
    ) or DEFAULT_SYMBOLS
    for symbol in normalized_symbols:
        resolve_symbol_route(symbol)
    return SupervisorPolicy(
        cycles=max(1, int(cycles)),
        training_rounds=max(1, int(training_rounds)),
        symbols=normalized_symbols,
        mission_symbols_per_cycle=max(1, int(mission_symbols_per_cycle)),
        target_return_pct=float(target_return_pct),
        max_leverage=float(max_leverage),
        margin_notional_usdt=float(margin_notional_usdt),
        optimize_every=max(1, int(optimize_every)),
        max_recent_loss_usdt=max(0.0, float(max_recent_loss_usdt)),
        max_route_loss_streak=max(1, int(max_route_loss_streak)),
        min_route_profit_factor=max(0.0, float(min_route_profit_factor)),
        route_lookback=max(1, int(route_lookback)),
        build_digest_every=max(0, int(build_digest_every)),
        audit_every=max(0, int(audit_every)),
        stop_on_optimizer_promotion=bool(stop_on_optimizer_promotion),
    )
