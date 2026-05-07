from __future__ import annotations

from pathlib import Path

import binance_quant_control.hermes_trade_loop as loop


def _patch_state_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(loop, "HERMES_TRADE_STATE_PATH", tmp_path / "hermes-trade-control.json")
    monkeypatch.setattr(loop, "HERMES_TRADE_REPORT_DIR", tmp_path / "hermes-trade-loop")


def test_hermes_trade_start_and_stop_manage_state_and_pause(monkeypatch, tmp_path: Path) -> None:
    _patch_state_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(loop, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(loop, "load_trading_control_state", lambda: loop.TradingControlState())
    pauses: list[dict] = []
    monkeypatch.setattr(
        loop,
        "set_trading_paused",
        lambda **kwargs: pauses.append(kwargs)
        or loop.TradingControlState(
            paused=bool(kwargs["paused"]),
            reason=str(kwargs["reason"]),
            updated_by=str(kwargs["updated_by"]),
        ),
    )

    started = loop.start_hermes_trade_loop(config_path="config/hermes-trade-loop.default.yaml")
    stopped = loop.stop_hermes_trade_loop(reason="operator said stop")

    assert started["status"] == "started"
    assert started["state"]["enabled"] is True
    assert started["state"]["execute_testnet_entries"] is True
    assert stopped["status"] == "stopped"
    assert stopped["state"]["enabled"] is False
    assert pauses[-1]["paused"] is True
    assert "operator said stop" in pauses[-1]["reason"]


def test_hermes_trade_cycle_stays_stopped_until_started(monkeypatch, tmp_path: Path) -> None:
    _patch_state_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(loop, "ensure_runtime_dirs", lambda: None)

    payload = loop.run_hermes_trade_cycle(config_path="config/hermes-trade-loop.default.yaml")

    assert payload["status"] == "stopped"
    assert "start" in payload["reason"].lower()
    assert Path(payload["report_path"]).exists()


def test_hermes_trade_cycle_uses_readiness_ticket_without_executing_when_dry_run_only(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_state_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(loop, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        loop,
        "load_trading_control_state",
        lambda: loop.TradingControlState(paused=False, reason="", updated_by=""),
    )
    loop.save_hermes_trade_state(
        loop.HermesTradeControlState(enabled=True, execute_testnet_entries=False)
    )
    monkeypatch.setattr(loop, "_run_auto_pause", lambda config: {"actions": ["no-action"]})
    monkeypatch.setattr(loop, "_optimizer_due", lambda config, state: False)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        loop,
        "_run_json_command",
        lambda command, timeout=900: commands.append(command)
        or {"returncode": 0, "response": {"allowed": True}},
    )
    monkeypatch.setattr(
        loop,
        "run_ai_readiness_scan",
        lambda **kwargs: {
            "candidate_count": 1,
            "allowed_count": 1,
            "selected_ready_candidate": {"symbol": "SOLUSDT", "side": "BUY"},
            "next_machine_action": "execute_ready_dry_run_only",
            "hard_blocker_taxonomy": {},
            "execution_ticket": {
                "state": "ready_for_operator_testnet_execution",
                "preflight_command": ".venv/bin/binance-quant-control live-readiness --symbol SOLUSDT --compact",
                "operator_testnet_execute_command": ".venv/bin/binance-quant-control live-pilot --symbol SOLUSDT --execute --compact",
            },
            "report_path": str(tmp_path / "scan.json"),
        },
    )

    payload = loop.run_hermes_trade_cycle(config_path="config/hermes-trade-loop.default.yaml")

    assert payload["status"] == "ready"
    assert payload["execution_gate"]["allowed"] is False
    assert "testnet-execution-disabled" in payload["execution_gate"]["blockers"]
    assert all("live-pilot" not in command for command in commands)


def test_hermes_trade_cycle_executes_testnet_ticket_after_preflight(monkeypatch, tmp_path: Path) -> None:
    _patch_state_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(loop, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        loop,
        "load_trading_control_state",
        lambda: loop.TradingControlState(paused=False, reason="", updated_by=""),
    )
    loop.save_hermes_trade_state(
        loop.HermesTradeControlState(enabled=True, execute_testnet_entries=True)
    )
    monkeypatch.setattr(loop, "_run_auto_pause", lambda config: {"actions": ["no-action"]})
    monkeypatch.setattr(loop, "_optimizer_due", lambda config, state: False)
    commands: list[list[str]] = []

    def fake_command(command, timeout=900):
        commands.append(command)
        if "live-readiness" in command:
            return {"returncode": 0, "response": {"allowed": True}}
        if "live-pilot" in command:
            return {"returncode": 0, "response": {"status": "ok"}}
        return {"returncode": 0, "response": {}}

    monkeypatch.setattr(loop, "_run_json_command", fake_command)
    monkeypatch.setattr(
        loop,
        "run_ai_readiness_scan",
        lambda **kwargs: {
            "candidate_count": 1,
            "allowed_count": 1,
            "selected_ready_candidate": {"symbol": "SOLUSDT", "side": "BUY"},
            "next_machine_action": "execute_ready_dry_run_only",
            "hard_blocker_taxonomy": {},
            "execution_ticket": {
                "state": "ready_for_operator_testnet_execution",
                "preflight_command": ".venv/bin/binance-quant-control live-readiness --symbol SOLUSDT --compact",
                "operator_testnet_execute_command": ".venv/bin/binance-quant-control live-pilot --symbol SOLUSDT --execute --compact",
            },
            "report_path": str(tmp_path / "scan.json"),
        },
    )

    payload = loop.run_hermes_trade_cycle(config_path="config/hermes-trade-loop.default.yaml")

    assert payload["status"] == "testnet_executed"
    assert payload["executed"] is True
    assert any("live-readiness" in command for command in commands)
    assert any("live-pilot" in command for command in commands)


def test_hermes_trade_cycle_lets_hailo_veto_testnet_execution(monkeypatch, tmp_path: Path) -> None:
    _patch_state_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(loop, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        loop,
        "load_trading_control_state",
        lambda: loop.TradingControlState(paused=False, reason="", updated_by=""),
    )
    loop.save_hermes_trade_state(
        loop.HermesTradeControlState(enabled=True, execute_testnet_entries=True)
    )
    monkeypatch.setattr(loop, "_run_auto_pause", lambda config: {"actions": ["no-action"]})
    monkeypatch.setattr(loop, "_optimizer_due", lambda config, state: False)
    monkeypatch.setattr(loop, "_run_review_closed_trades", lambda config: {"returncode": 0, "response": {}})
    monkeypatch.setattr(loop, "_run_position_scan", lambda: {"returncode": 0, "response": {"count": 0, "positions": []}})
    monkeypatch.setattr(loop, "_run_external_context", lambda config: {"returncode": 0, "response": {}})
    monkeypatch.setattr(
        loop,
        "_run_hailo_triage",
        lambda config: {
            "returncode": 0,
            "response": {
                "raw_event_count": 1,
                "output_event_count": 1,
                "events": [
                    {
                        "source": "quant_runtime",
                        "event_type": "system_review",
                        "priority": "high",
                        "labels": ["professional_gate_failed"],
                    }
                ],
            },
        },
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(
        loop,
        "_run_json_command",
        lambda command, timeout=900: commands.append(command) or {"returncode": 0, "response": {}},
    )
    monkeypatch.setattr(
        loop,
        "run_ai_readiness_scan",
        lambda **kwargs: {
            "candidate_count": 1,
            "allowed_count": 1,
            "selected_ready_candidate": {"symbol": "SOLUSDT", "side": "BUY"},
            "execution_ticket": {
                "state": "ready_for_operator_testnet_execution",
                "preflight_command": ".venv/bin/binance-quant-control live-readiness --symbol SOLUSDT --compact",
                "operator_testnet_execute_command": ".venv/bin/binance-quant-control live-pilot --symbol SOLUSDT --execute --compact",
            },
        },
    )

    payload = loop.run_hermes_trade_cycle(config_path="config/hermes-trade-loop.default.yaml")

    assert payload["execution_gate"]["allowed"] is False
    assert "hailo-veto:professional_gate_failed" in payload["execution_gate"]["blockers"]
    assert payload["hailo_entry_gate"]["decision"] == "veto"
    assert all("live-pilot" not in command for command in commands)


def test_hermes_trade_cycle_manages_open_position_before_new_entry(monkeypatch, tmp_path: Path) -> None:
    _patch_state_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(loop, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        loop,
        "load_trading_control_state",
        lambda: loop.TradingControlState(paused=False, reason="", updated_by=""),
    )
    loop.save_hermes_trade_state(
        loop.HermesTradeControlState(enabled=True, execute_testnet_entries=True)
    )
    monkeypatch.setattr(loop, "_run_auto_pause", lambda config: {"actions": ["no-action"]})
    monkeypatch.setattr(loop, "_run_market_sentinel", lambda config: {"status": "ok", "response": {"mode": "ai_market_sentinel_v1"}})
    monkeypatch.setattr(loop, "_optimizer_due", lambda config, state: False)
    monkeypatch.setattr(loop, "_run_review_closed_trades", lambda config: {"returncode": 0, "response": {}})
    monkeypatch.setattr(loop, "_run_external_context", lambda config: {"returncode": 0, "response": {}})
    monkeypatch.setattr(loop, "_run_hailo_triage", lambda config: {"returncode": 0, "response": {}})
    guardian_calls = []
    monkeypatch.setattr(
        loop,
        "_run_position_guardian",
        lambda config: guardian_calls.append(config.path) or {"returncode": 0, "response": {"status": "ok"}},
    )
    scans = [
        {
            "returncode": 0,
            "response": {
                "count": 1,
                "positions": [{"symbol": "ETHUSDT", "side": "LONG", "qty": 0.1, "entry": 3000.0}],
            },
        },
        {
            "returncode": 0,
            "response": {
                "count": 1,
                "positions": [{"symbol": "ETHUSDT", "side": "LONG", "qty": 0.1, "entry": 3000.0}],
            },
        },
    ]
    monkeypatch.setattr(loop, "_run_position_scan", lambda: scans.pop(0))
    readiness_calls = []
    monkeypatch.setattr(
        loop,
        "run_ai_readiness_scan",
        lambda **kwargs: readiness_calls.append(kwargs) or {},
    )

    payload = loop.run_hermes_trade_cycle(config_path="config/hermes-trade-loop.default.yaml")

    assert payload["steps"]["market_sentinel"]["response"]["mode"] == "ai_market_sentinel_v1"
    assert payload["status"] == "managing_position"
    assert payload["position_loop"]["mode"] == "manage_and_seek_new_entry"
    assert payload["position_loop"]["next_sleep_seconds"] == 45.0
    assert payload["execution_gate"]["allowed"] is False
    assert "no-execution-ticket" in payload["execution_gate"]["blockers"]
    assert guardian_calls
    assert readiness_calls


def test_hermes_trade_cycle_scans_new_symbol_with_open_positions_below_cap(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_state_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(loop, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        loop,
        "load_trading_control_state",
        lambda: loop.TradingControlState(paused=False, reason="", updated_by=""),
    )
    loop.save_hermes_trade_state(
        loop.HermesTradeControlState(enabled=True, execute_testnet_entries=True)
    )
    monkeypatch.setattr(loop, "_run_auto_pause", lambda config: {"actions": ["no-action"]})
    monkeypatch.setattr(loop, "_run_market_sentinel", lambda config: {"status": "ok", "response": {}})
    monkeypatch.setattr(loop, "_optimizer_due", lambda config, state: False)
    monkeypatch.setattr(loop, "_run_review_closed_trades", lambda config: {"returncode": 0, "response": {}})
    monkeypatch.setattr(loop, "_run_external_context", lambda config: {"returncode": 0, "response": {}})
    monkeypatch.setattr(loop, "_run_hailo_triage", lambda config: {"returncode": 0, "response": {}})
    monkeypatch.setattr(loop, "_run_position_guardian", lambda config: {"returncode": 0, "response": {}})
    scans = [
        {
            "returncode": 0,
            "response": {
                "count": 1,
                "positions": [{"symbol": "BTCUSDT", "side": "LONG", "qty": 0.001, "entry": 80000.0}],
            },
        },
        {
            "returncode": 0,
            "response": {
                "count": 2,
                "positions": [
                    {"symbol": "BTCUSDT", "side": "LONG", "qty": 0.001, "entry": 80000.0},
                    {"symbol": "SOLUSDT", "side": "LONG", "qty": 0.1, "entry": 180.0},
                ],
            },
        },
    ]
    monkeypatch.setattr(loop, "_run_position_scan", lambda: scans.pop(0))
    readiness_calls = []
    monkeypatch.setattr(
        loop,
        "run_ai_readiness_scan",
        lambda **kwargs: readiness_calls.append(kwargs)
        or {
            "candidate_count": 1,
            "allowed_count": 1,
            "selected_ready_candidate": {"symbol": "SOLUSDT", "side": "BUY"},
            "execution_ticket": {
                "state": "ready_for_operator_testnet_execution",
                "preflight_command": ".venv/bin/binance-quant-control live-readiness --symbol SOLUSDT --compact",
                "operator_testnet_execute_command": ".venv/bin/binance-quant-control live-pilot --symbol SOLUSDT --execute --compact",
            },
        },
    )
    commands: list[list[str]] = []

    def fake_command(command, timeout=900):
        commands.append(command)
        if "live-readiness" in command:
            return {"returncode": 0, "response": {"allowed": True}}
        return {"returncode": 0, "response": {"status": "ok"}}

    monkeypatch.setattr(loop, "_run_json_command", fake_command)

    payload = loop.run_hermes_trade_cycle(config_path="config/hermes-trade-loop.default.yaml")

    assert readiness_calls
    assert payload["position_loop"]["mode"] == "manage_and_seek_new_entry"
    assert payload["execution_gate"]["allowed"] is True
    assert "open-position-management-priority" not in payload["execution_gate"]["blockers"]
    assert payload["status"] == "testnet_executed"


def test_hermes_trade_cycle_blocks_new_symbol_at_four_positions(monkeypatch, tmp_path: Path) -> None:
    _patch_state_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(loop, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        loop,
        "load_trading_control_state",
        lambda: loop.TradingControlState(paused=False, reason="", updated_by=""),
    )
    loop.save_hermes_trade_state(
        loop.HermesTradeControlState(enabled=True, execute_testnet_entries=True)
    )
    monkeypatch.setattr(loop, "_run_auto_pause", lambda config: {"actions": ["no-action"]})
    monkeypatch.setattr(loop, "_optimizer_due", lambda config, state: False)
    monkeypatch.setattr(loop, "_run_review_closed_trades", lambda config: {"returncode": 0, "response": {}})
    monkeypatch.setattr(loop, "_run_market_sentinel", lambda config: {"returncode": 0, "response": {}})
    monkeypatch.setattr(loop, "_run_external_context", lambda config: {"returncode": 0, "response": {}})
    monkeypatch.setattr(loop, "_run_hailo_triage", lambda config: {"returncode": 0, "response": {}})
    monkeypatch.setattr(loop, "_run_position_guardian", lambda config: {"returncode": 0, "response": {}})
    open_positions = [
        {"symbol": "BTCUSDT", "side": "LONG", "qty": 0.001, "entry": 80000.0},
        {"symbol": "ETHUSDT", "side": "LONG", "qty": 0.01, "entry": 3000.0},
        {"symbol": "SOLUSDT", "side": "LONG", "qty": 0.1, "entry": 180.0},
        {"symbol": "BNBUSDT", "side": "LONG", "qty": 0.1, "entry": 600.0},
    ]
    monkeypatch.setattr(
        loop,
        "_run_position_scan",
        lambda: {"returncode": 0, "response": {"count": 4, "positions": open_positions}},
    )
    readiness_calls = []
    monkeypatch.setattr(loop, "run_ai_readiness_scan", lambda **kwargs: readiness_calls.append(kwargs) or {})

    payload = loop.run_hermes_trade_cycle(config_path="config/hermes-trade-loop.default.yaml")

    assert readiness_calls == []
    assert payload["readiness"]["reason"] == "max-concurrent-positions-reached"
    assert "max-concurrent-positions-reached" in payload["execution_gate"]["blockers"]


def test_hermes_trade_cycle_detects_closed_position_and_fast_reentry_scan(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_state_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(loop, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        loop,
        "load_trading_control_state",
        lambda: loop.TradingControlState(paused=False, reason="", updated_by=""),
    )
    loop.save_hermes_trade_state(
        loop.HermesTradeControlState(
            enabled=True,
            execute_testnet_entries=False,
            last_open_position_keys=("ETHUSDT:LONG:0.10000000:3000.00000000",),
        )
    )
    monkeypatch.setattr(loop, "_run_auto_pause", lambda config: {"actions": ["no-action"]})
    monkeypatch.setattr(loop, "_optimizer_due", lambda config, state: False)
    review_calls = []
    optimizer_calls = []
    monkeypatch.setattr(
        loop,
        "_run_review_closed_trades",
        lambda config: review_calls.append(config.closed_trade_review_limit)
        or {"returncode": 0, "response": {"new_review_count": 1}},
    )
    monkeypatch.setattr(loop, "_run_external_context", lambda config: {"returncode": 0, "response": {}})
    monkeypatch.setattr(loop, "_run_hailo_triage", lambda config: {"returncode": 0, "response": {}})
    monkeypatch.setattr(
        loop,
        "_run_strategy_optimizer",
        lambda config: optimizer_calls.append(config.path) or {"returncode": 0, "response": {}},
    )
    monkeypatch.setattr(
        loop,
        "_run_position_scan",
        lambda: {"returncode": 0, "response": {"count": 0, "positions": []}},
    )
    monkeypatch.setattr(
        loop,
        "run_ai_readiness_scan",
        lambda **kwargs: {
            "candidate_count": 1,
            "allowed_count": 1,
            "selected_ready_candidate": {"symbol": "SOLUSDT", "side": "BUY"},
            "execution_ticket": {
                "state": "ready_for_operator_testnet_execution",
                "preflight_command": ".venv/bin/binance-quant-control live-readiness --symbol SOLUSDT --compact",
                "operator_testnet_execute_command": ".venv/bin/binance-quant-control live-pilot --symbol SOLUSDT --execute --compact",
            },
        },
    )

    payload = loop.run_hermes_trade_cycle(config_path="config/hermes-trade-loop.default.yaml")

    assert payload["status"] == "ready"
    assert payload["position_loop"]["closed_position_detected"] is True
    assert payload["position_loop"]["closed_position_keys"] == [
        "ETHUSDT:LONG:0.10000000:3000.00000000"
    ]
    assert payload["position_loop"]["next_sleep_seconds"] == 5.0
    assert review_calls == [50]
    assert optimizer_calls
    state = loop.load_hermes_trade_state()
    assert state.last_open_position_keys == ()
    assert state.last_closed_position_keys == ("ETHUSDT:LONG:0.10000000:3000.00000000",)


def test_hermes_trade_cycle_only_marks_context_timestamps_on_success(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_state_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(loop, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        loop,
        "load_trading_control_state",
        lambda: loop.TradingControlState(paused=False, reason="", updated_by=""),
    )
    loop.save_hermes_trade_state(
        loop.HermesTradeControlState(enabled=True, execute_testnet_entries=False)
    )
    monkeypatch.setattr(loop, "_run_auto_pause", lambda config: {"actions": ["no-action"]})
    monkeypatch.setattr(loop, "_optimizer_due", lambda config, state: False)
    monkeypatch.setattr(loop, "_run_review_closed_trades", lambda config: {"returncode": 0, "response": {}})
    monkeypatch.setattr(loop, "_run_position_scan", lambda: {"returncode": 0, "response": {"count": 0, "positions": []}})
    monkeypatch.setattr(loop, "_run_external_context", lambda config: {"returncode": 1, "response": None})
    monkeypatch.setattr(loop, "_run_hailo_triage", lambda config: {"returncode": 1, "response": None})
    monkeypatch.setattr(loop, "run_ai_readiness_scan", lambda **kwargs: {})

    payload = loop.run_hermes_trade_cycle(config_path="config/hermes-trade-loop.default.yaml")

    assert payload["steps"]["external_context"]["returncode"] == 1
    assert payload["steps"]["hailo_triage"]["returncode"] == 1
    state = loop.load_hermes_trade_state()
    assert state.last_external_context_at == ""
    assert state.last_hailo_triage_at == ""
