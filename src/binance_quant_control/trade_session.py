from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import STATE_DIR, load_settings
from .hermes_trade_loop import start_hermes_trade_loop, stop_hermes_trade_loop
from .route_risk_control import load_route_risk_state
from .trading_control import load_trading_control_state, set_trading_paused

TRADE_SESSION_STATE_PATH = STATE_DIR / "trade-session-control.json"
TRADE_SESSION_ACTOR = "openclaw-quantctl trade-session"
USER_SYSTEMD_DIR = Path.home() / ".config" / "systemd" / "user"
SYSTEMD_SOURCE_DIR = Path("/home/robert/python/ops/automation/systemd")
LOW_TOKEN_TRADE_TIMERS = [
    "ai-market-sentinel.timer",
    "openclaw-binance-position-guardian.timer",
    "openclaw-binance-operator-dashboard.timer",
    "openclaw-binance-quant-research.timer",
    "openclaw-binance-strategy-optimizer.timer",
]
HEAVY_ENTRY_TIMERS = [
    "openclaw-binance-testnet-explorer.timer",
    "openclaw-binance-autonomy.timer",
    "openclaw-binance-live-lane.timer",
]
MAX_CONCURRENT_POSITIONS = 4


@dataclass(frozen=True, slots=True)
class TradeSessionState:
    enabled: bool = False
    dry_run_only: bool = True
    started_at: str = ""
    stopped_at: str = ""
    updated_at: str = ""
    updated_by: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run_systemctl(args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        ["systemctl", "--user", *args],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    return {
        "returncode": completed.returncode,
        "stdout": (completed.stdout or "").strip(),
        "stderr": (completed.stderr or "").strip(),
        "args": args,
    }


def _ensure_user_timer_units(units: list[str]) -> dict[str, Any]:
    linked: list[str] = []
    existing: list[str] = []
    missing: list[str] = []
    errors: list[dict[str, str]] = []
    USER_SYSTEMD_DIR.mkdir(parents=True, exist_ok=True)
    for unit in units:
        target = USER_SYSTEMD_DIR / unit
        if target.exists() or target.is_symlink():
            existing.append(unit)
            continue
        source = SYSTEMD_SOURCE_DIR / unit
        if not source.exists():
            missing.append(unit)
            continue
        try:
            target.symlink_to(source)
            linked.append(unit)
        except OSError as exc:
            errors.append({"unit": unit, "error": str(exc)})
    daemon_reload = _run_systemctl(["daemon-reload"]) if linked else None
    return {
        "linked": linked,
        "existing": existing,
        "missing": missing,
        "errors": errors,
        "daemon_reload": daemon_reload,
    }


def load_trade_session_state() -> TradeSessionState:
    if not TRADE_SESSION_STATE_PATH.exists():
        return TradeSessionState()
    try:
        payload = json.loads(TRADE_SESSION_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return TradeSessionState()
    if not isinstance(payload, dict):
        return TradeSessionState()
    return TradeSessionState(
        enabled=bool(payload.get("enabled", False)),
        dry_run_only=bool(payload.get("dry_run_only", True)),
        started_at=str(payload.get("started_at") or ""),
        stopped_at=str(payload.get("stopped_at") or ""),
        updated_at=str(payload.get("updated_at") or ""),
        updated_by=str(payload.get("updated_by") or ""),
        reason=str(payload.get("reason") or ""),
    )


def save_trade_session_state(state: TradeSessionState) -> TradeSessionState:
    TRADE_SESSION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRADE_SESSION_STATE_PATH.write_text(
        json.dumps(state.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return state


def _timer_status(unit: str) -> dict[str, Any]:
    active = _run_systemctl(["is-active", unit])
    enabled = _run_systemctl(["is-enabled", unit])
    return {
        "unit": unit,
        "active": active.get("stdout") == "active",
        "enabled": enabled.get("stdout") == "enabled",
        "active_raw": active.get("stdout") or active.get("stderr"),
        "enabled_raw": enabled.get("stdout") or enabled.get("stderr"),
    }


def _positions_compact() -> dict[str, Any]:
    from .binance_api import BinanceAPIError, BinanceClient

    settings = load_settings()
    positions: list[dict[str, Any]] = []
    try:
        with BinanceClient(settings) as client:
            raw_positions = client.positions(None)
    except BinanceAPIError as exc:
        return {"count": 0, "positions": [], "error": str(exc)}
    for item in raw_positions if isinstance(raw_positions, list) else []:
        amount = float(item.get("positionAmt", 0) or 0)
        if amount == 0.0:
            continue
        positions.append(
            {
                "symbol": str(item.get("symbol") or "").upper(),
                "side": "LONG" if amount > 0 else "SHORT",
                "qty": abs(amount),
                "entry": float(item.get("entryPrice", 0) or 0),
                "pnl": float(item.get("unRealizedProfit", 0) or 0),
                "leverage": int(float(item.get("leverage", 1) or 1)),
            }
        )
    return {"count": len(positions), "positions": positions}


def _readiness_summary() -> dict[str, Any]:
    blockers: list[str] = []
    trading_control = load_trading_control_state()
    positions = _positions_compact()
    route_risk = load_route_risk_state()
    active_quarantines = [str(item) for item in (route_risk.get("active_quarantined_routes") or [])]
    if trading_control.paused:
        blockers.append("trading-control-paused")
    open_position_count = int(positions.get("count") or 0)
    if open_position_count >= MAX_CONCURRENT_POSITIONS:
        blockers.append("max-concurrent-positions-reached")
    if active_quarantines:
        blockers.append("active-negative-expectancy-route-quarantine")
    return {
        "can_start_positive_expectancy": not blockers,
        "blockers": blockers,
        "trading_control": trading_control.to_dict(),
        "positions": positions,
        "max_concurrent_positions": MAX_CONCURRENT_POSITIONS,
        "active_quarantined_routes": active_quarantines,
    }


def start_trade_session(*, dry_run_only: bool = True, reason: str = "operator start trading") -> dict[str, Any]:
    now = _utc_now_iso()
    units = _ensure_user_timer_units([*LOW_TOKEN_TRADE_TIMERS, *HEAVY_ENTRY_TIMERS])
    start_low_token = _run_systemctl(["start", *LOW_TOKEN_TRADE_TIMERS])
    stop_heavy = _run_systemctl(["stop", *HEAVY_ENTRY_TIMERS])
    hermes = start_hermes_trade_loop(
        execute_testnet_entries=not dry_run_only,
        note=reason,
        release_own_pause=True,
    )
    trading_control = load_trading_control_state()
    released_pause: dict[str, Any] | None = None
    if trading_control.paused and trading_control.updated_by == f"{TRADE_SESSION_ACTOR} stop":
        released_pause = set_trading_paused(
            paused=False,
            reason=reason,
            updated_by=f"{TRADE_SESSION_ACTOR} start",
        ).to_dict()
    state = save_trade_session_state(
        TradeSessionState(
            enabled=True,
            dry_run_only=bool(dry_run_only),
            started_at=now,
            stopped_at="",
            updated_at=now,
            updated_by=f"{TRADE_SESSION_ACTOR} start",
            reason=reason,
        )
    )
    return {
        "status": "started",
        "state": state.to_dict(),
        "safety": {
            "mainnet_live_allowed": False,
            "testnet_execution_enabled": not dry_run_only,
            "heavy_entry_timers_enabled": False,
        },
        "systemd": {
            "units": units,
            "low_token_start": start_low_token,
            "heavy_stop": stop_heavy,
        },
        "hermes_trade": hermes,
        "released_pause": released_pause,
        "readiness_summary": _readiness_summary(),
    }


def stop_trade_session(*, reason: str = "operator end trading") -> dict[str, Any]:
    now = _utc_now_iso()
    units = _ensure_user_timer_units([*LOW_TOKEN_TRADE_TIMERS, *HEAVY_ENTRY_TIMERS])
    stop_low_token = _run_systemctl(["stop", *LOW_TOKEN_TRADE_TIMERS])
    stop_heavy = _run_systemctl(["stop", *HEAVY_ENTRY_TIMERS])
    hermes = stop_hermes_trade_loop(reason=reason)
    trading_control = set_trading_paused(
        paused=True,
        reason=reason,
        updated_by=f"{TRADE_SESSION_ACTOR} stop",
    )
    state = save_trade_session_state(
        TradeSessionState(
            enabled=False,
            dry_run_only=True,
            started_at=load_trade_session_state().started_at,
            stopped_at=now,
            updated_at=now,
            updated_by=f"{TRADE_SESSION_ACTOR} stop",
            reason=reason,
        )
    )
    return {
        "status": "stopped",
        "state": state.to_dict(),
        "systemd": {
            "units": units,
            "low_token_stop": stop_low_token,
            "heavy_stop": stop_heavy,
        },
        "hermes_trade": hermes,
        "trading_control": trading_control.to_dict(),
    }


def trade_session_status() -> dict[str, Any]:
    return {
        "status": "enabled" if load_trade_session_state().enabled else "stopped",
        "state": load_trade_session_state().to_dict(),
        "timers": {
            "low_token": [_timer_status(unit) for unit in LOW_TOKEN_TRADE_TIMERS],
            "heavy_entry": [_timer_status(unit) for unit in HEAVY_ENTRY_TIMERS],
        },
        "readiness_summary": _readiness_summary(),
    }
