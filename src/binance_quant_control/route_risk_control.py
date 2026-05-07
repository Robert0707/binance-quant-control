from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from .config import STATE_DIR

ROUTE_RISK_CONTROL_PATH = STATE_DIR / "route-risk-control.json"


@dataclass(frozen=True, slots=True)
class RouteRiskStatus:
    route_id: str
    quarantined: bool
    reasons: list[str]
    metrics: dict[str, Any]
    updated_at: str
    updated_by: str
    manual_review_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _default_state() -> dict[str, Any]:
    return {
        "generated_at": _utc_now_iso(),
        "updated_by": "route-risk-control",
        "routes": {},
        "active_quarantined_routes": [],
    }


def load_route_risk_state() -> dict[str, Any]:
    if not ROUTE_RISK_CONTROL_PATH.exists():
        return _default_state()
    try:
        payload = json.loads(ROUTE_RISK_CONTROL_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_state()
    if not isinstance(payload, dict):
        return _default_state()
    payload.setdefault("routes", {})
    payload.setdefault("active_quarantined_routes", [])
    return payload


def save_route_risk_state(payload: dict[str, Any]) -> dict[str, Any]:
    ROUTE_RISK_CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload["generated_at"] = _utc_now_iso()
    payload["active_quarantined_routes"] = sorted(
        route_id
        for route_id, row in (payload.get("routes") or {}).items()
        if bool((row or {}).get("quarantined", False))
    )
    ROUTE_RISK_CONTROL_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def update_route_quarantine_from_snapshot(
    snapshot: dict[str, Any],
    *,
    updated_by: str,
) -> dict[str, Any]:
    state = load_route_risk_state()
    routes = dict(state.get("routes") or {})
    now = _utc_now_iso()
    snapshot_routes = snapshot.get("routes") or {}
    if not isinstance(snapshot_routes, dict):
        snapshot_routes = {}
    for route_id, raw in snapshot_routes.items():
        row = raw if isinstance(raw, dict) else {}
        reasons = [str(item) for item in (row.get("quarantine_reasons") or []) if str(item)]
        quarantined = bool(row.get("quarantined", False))
        previous = routes.get(route_id) if isinstance(routes.get(route_id), dict) else {}
        if quarantined:
            routes[route_id] = RouteRiskStatus(
                route_id=str(route_id),
                quarantined=True,
                reasons=reasons,
                metrics={
                    key: row.get(key)
                    for key in (
                        "count",
                        "wins",
                        "losses",
                        "win_rate",
                        "profit_factor",
                        "max_drawdown_pct",
                        "loss_streak",
                    )
                },
                updated_at=now,
                updated_by=updated_by,
                manual_review_required=True,
            ).to_dict()
        elif previous:
            # A route that recovers is not auto-released for live trading. Keep
            # the historical status visible and require an explicit review.
            recovered = dict(previous)
            recovered["quarantined"] = bool(previous.get("quarantined", False))
            recovered["latest_recovery_candidate"] = {
                "checked_at": now,
                "updated_by": updated_by,
                "metrics": {
                    key: row.get(key)
                    for key in (
                        "count",
                        "wins",
                        "losses",
                        "win_rate",
                        "profit_factor",
                        "max_drawdown_pct",
                        "loss_streak",
                    )
                },
            }
            routes[route_id] = recovered
    state["updated_by"] = updated_by
    state["routes"] = routes
    return save_route_risk_state(state)


def route_quarantine_status(route_id: str) -> dict[str, Any]:
    state = load_route_risk_state()
    row = (state.get("routes") or {}).get(route_id)
    if not isinstance(row, dict):
        return {
            "route_id": route_id,
            "quarantined": False,
            "reasons": [],
            "metrics": {},
            "updated_at": "",
            "updated_by": "",
            "manual_review_required": False,
        }
    return {
        "route_id": route_id,
        "quarantined": bool(row.get("quarantined", False)),
        "reasons": [str(item) for item in (row.get("reasons") or [])],
        "metrics": row.get("metrics") or {},
        "updated_at": str(row.get("updated_at") or ""),
        "updated_by": str(row.get("updated_by") or ""),
        "manual_review_required": bool(row.get("manual_review_required", True)),
    }


def clear_route_quarantine(
    route_id: str,
    *,
    reason: str,
    updated_by: str,
) -> dict[str, Any]:
    state = load_route_risk_state()
    routes = dict(state.get("routes") or {})
    row = dict(routes.get(route_id) or {})
    row.update(
        {
            "route_id": route_id,
            "quarantined": False,
            "manual_review_required": False,
            "cleared_at": _utc_now_iso(),
            "cleared_by": updated_by,
            "clear_reason": reason.strip() or "manual route risk review cleared",
        }
    )
    routes[route_id] = row
    state["updated_by"] = updated_by
    state["routes"] = routes
    return save_route_risk_state(state)
