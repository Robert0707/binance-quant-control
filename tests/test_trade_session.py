from __future__ import annotations

from pathlib import Path

import binance_quant_control.trade_session as session
from binance_quant_control.trading_control import TradingControlState


def test_start_trade_session_enables_local_trading_timers(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(session, "TRADE_SESSION_STATE_PATH", tmp_path / "trade-session.json")
    commands: list[list[str]] = []
    monkeypatch.setattr(session, "_run_systemctl", lambda args: commands.append(args) or {"returncode": 0})
    monkeypatch.setattr(session, "_ensure_user_timer_units", lambda units: {"linked": [], "missing": []})
    monkeypatch.setattr(
        session,
        "start_hermes_trade_loop",
        lambda **kwargs: {"status": "started", "state": {"enabled": True}},
    )
    monkeypatch.setattr(
        session,
        "load_trading_control_state",
        lambda: TradingControlState(paused=True, updated_by="operator"),
    )
    monkeypatch.setattr(session, "_positions_compact", lambda: {"count": 0, "positions": []})
    monkeypatch.setattr(session, "load_route_risk_state", lambda: {"active_quarantined_routes": []})

    payload = session.start_trade_session(dry_run_only=True)

    assert payload["status"] == "started"
    assert payload["state"]["enabled"] is True
    assert ["start", *session.LOW_TOKEN_TRADE_TIMERS] in commands
    assert ["stop", *session.HEAVY_ENTRY_TIMERS] in commands
    assert payload["state"]["dry_run_only"] is True


def test_start_trade_session_releases_own_pause(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(session, "TRADE_SESSION_STATE_PATH", tmp_path / "trade-session.json")
    monkeypatch.setattr(session, "_run_systemctl", lambda args: {"returncode": 0})
    monkeypatch.setattr(session, "_ensure_user_timer_units", lambda units: {"linked": [], "missing": []})
    monkeypatch.setattr(
        session,
        "start_hermes_trade_loop",
        lambda **kwargs: {"status": "started", "state": {"enabled": True}},
    )
    monkeypatch.setattr(
        session,
        "load_trading_control_state",
        lambda: TradingControlState(
            paused=True,
            reason="operator end",
            updated_by=f"{session.TRADE_SESSION_ACTOR} stop",
        ),
    )
    pauses: list[dict] = []
    monkeypatch.setattr(
        session,
        "set_trading_paused",
        lambda **kwargs: pauses.append(kwargs) or TradingControlState(paused=kwargs["paused"]),
    )
    monkeypatch.setattr(session, "_positions_compact", lambda: {"count": 0, "positions": []})
    monkeypatch.setattr(session, "load_route_risk_state", lambda: {"active_quarantined_routes": []})

    payload = session.start_trade_session(dry_run_only=True)

    assert payload["released_pause"]["paused"] is False
    assert pauses[-1]["paused"] is False
    assert pauses[-1]["updated_by"] == f"{session.TRADE_SESSION_ACTOR} start"


def test_stop_trade_session_disables_trading_timers_and_pauses(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(session, "TRADE_SESSION_STATE_PATH", tmp_path / "trade-session.json")
    commands: list[list[str]] = []
    pauses: list[dict] = []
    monkeypatch.setattr(session, "_run_systemctl", lambda args: commands.append(args) or {"returncode": 0})
    monkeypatch.setattr(session, "_ensure_user_timer_units", lambda units: {"linked": [], "missing": []})
    monkeypatch.setattr(session, "stop_hermes_trade_loop", lambda **kwargs: {"status": "stopped"})
    monkeypatch.setattr(
        session,
        "set_trading_paused",
        lambda **kwargs: pauses.append(kwargs) or TradingControlState(paused=kwargs["paused"]),
    )

    payload = session.stop_trade_session(reason="operator end")

    assert payload["status"] == "stopped"
    assert ["stop", *session.LOW_TOKEN_TRADE_TIMERS] in commands
    assert ["stop", *session.HEAVY_ENTRY_TIMERS] in commands
    assert pauses[-1]["paused"] is True
    assert payload["state"]["enabled"] is False


def test_trade_session_status_reports_not_ready_when_blocked(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(session, "TRADE_SESSION_STATE_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(session, "_timer_status", lambda name: {"unit": name, "active": False, "enabled": False})
    monkeypatch.setattr(
        session,
        "load_trading_control_state",
        lambda: TradingControlState(paused=True, reason="protect exposure"),
    )
    monkeypatch.setattr(session, "_positions_compact", lambda: {"count": 1, "positions": [{"symbol": "BTCUSDT"}]})
    monkeypatch.setattr(
        session,
        "load_route_risk_state",
        lambda: {"active_quarantined_routes": ["btc-core"], "routes": {}},
    )

    payload = session.trade_session_status()

    assert payload["readiness_summary"]["can_start_positive_expectancy"] is False
    assert "trading-control-paused" in payload["readiness_summary"]["blockers"]
    assert "open-position-management-priority" not in payload["readiness_summary"]["blockers"]
