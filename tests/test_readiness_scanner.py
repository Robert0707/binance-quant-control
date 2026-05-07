from __future__ import annotations

from pathlib import Path

import binance_quant_control.readiness_scanner as scanner


def _queue_item(rank: int, symbol: str) -> dict[str, object]:
    return {
        "rank": rank,
        "signal": {
            "symbol": symbol,
            "side": "BUY",
            "interval": "4h",
            "strategy_family": "ai_family_router",
            "route_id": f"{symbol.lower()}-route",
        },
        "machine_state": "candidate_ready",
        "open_order_gate": {"allowed": True, "blockers": []},
    }


def _plan(
    *,
    symbol: str,
    allowed: bool,
    violations: list[str] | None = None,
    warnings: list[str] | None = None,
) -> object:
    return type(
        "Plan",
        (),
        {
            "to_dict": lambda self: {
                "allowed": allowed,
                "symbol": symbol,
                "market": "futures",
                "side": "BUY",
                "quantity": 10.0,
                "price": 1.0,
                "leverage": 3,
                "margin_notional_usdt": 2.0,
                "gross_notional_usdt": 6.0,
                "min_notional_usdt": 5.0,
                "planned_account_risk_pct": 0.0025,
                "analysis_score": 80,
                "analysis_convergence": 0.82,
                "adx_value": 24.0,
                "execution_mode": "testnet_exploration",
                "violations": violations or [],
                "warnings": warnings or [],
                "professional_entry_gate": {"passed": allowed},
                "market_bot_gate": {
                    "allowed": True,
                    "report_path": "state/market-bot-gate.json",
                    "feature_manifest_hash": "abc123",
                    "matched_row": {"cohort_id": f"{symbol}:4h:ai_family_router"},
                },
                "challenge": {
                    "market_bot_gate": {"allowed": True},
                    "route_quarantine": {"quarantined": False, "reasons": []},
                    "route_side_risk": {"allowed": True, "reasons": []},
                    "historical_signal_risk": {"allowed": True, "reasons": []},
                },
            }
        },
    )()


def test_ai_readiness_scan_selects_second_candidate_when_first_is_blocked(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(scanner, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        scanner,
        "run_hermes_ai_trader",
        lambda **_kwargs: {
            "report_path": str(tmp_path / "hermes.json"),
            "candidate_queue": [_queue_item(1, "DOGEUSDT"), _queue_item(2, "ETHUSDT")],
        },
    )
    monkeypatch.setattr(scanner, "load_settings", lambda: object())
    monkeypatch.setattr(scanner, "load_strategy_config", lambda _path: type(
        "Strategy",
        (),
        {
            "defaults": type(
                "Defaults",
                (),
                {
                    "market": "futures",
                    "interval": "4h",
                    "limit": 600,
                    "use_blave": False,
                },
            )()
        },
    )())
    monkeypatch.setattr(
        scanner,
        "run_analysis",
        lambda _settings, **kwargs: (
            {
                "symbol": kwargs["symbol"],
                "market": kwargs["market"],
                "analysis": {"score": 80, "convergence": 0.82},
                "latest": {"close": 1.0},
                "trade_plan": {"long": {}, "short": {}},
            },
            None,
        ),
    )

    def fake_plan(_settings, _strategy, analysis, **_kwargs):
        symbol = analysis["symbol"]
        if symbol == "DOGEUSDT":
            return _plan(
                symbol=symbol,
                allowed=False,
                violations=["Trading is paused by kill-switch: consecutive losses reached 2."],
            )
        return _plan(symbol=symbol, allowed=True)

    monkeypatch.setattr(scanner, "build_live_execution_plan", fake_plan)

    payload = scanner.run_ai_readiness_scan(output_dir=tmp_path)

    assert payload["candidate_count"] == 2
    assert payload["scanned_count"] == 2
    assert payload["allowed_count"] == 1
    assert payload["selected_ready_candidate"]["symbol"] == "ETHUSDT"
    assert payload["ready_after_global_unlock_count"] == 1
    assert payload["selected_after_global_unlock"]["symbol"] == "DOGEUSDT"
    assert payload["next_machine_action"] == "execute_ready_dry_run_only"
    assert payload["machine_action_queue"][0]["action"] == "execute_ready_dry_run_only"
    assert payload["execution_ticket"]["symbol"] == "ETHUSDT"
    assert payload["execution_ticket"]["opens_orders"] is False
    assert "--execute" in payload["execution_ticket"]["operator_testnet_execute_command"]
    assert payload["scan_results"][0]["next_action"] == "wait_for_kill_switch_clear"
    assert Path(payload["report_path"]).exists()


def test_ai_readiness_scan_classifies_blockers_when_all_candidates_fail(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(scanner, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        scanner,
        "run_hermes_ai_trader",
        lambda **_kwargs: {
            "report_path": str(tmp_path / "hermes.json"),
            "candidate_queue": [_queue_item(1, "DOGEUSDT")],
        },
    )
    monkeypatch.setattr(scanner, "load_settings", lambda: object())
    monkeypatch.setattr(scanner, "load_strategy_config", lambda _path: type(
        "Strategy",
        (),
        {
            "defaults": type(
                "Defaults",
                (),
                {
                    "market": "futures",
                    "interval": "4h",
                    "limit": 600,
                    "use_blave": False,
                },
            )()
        },
    )())
    monkeypatch.setattr(
        scanner,
        "run_analysis",
        lambda _settings, **kwargs: (
            {
                "symbol": kwargs["symbol"],
                "market": kwargs["market"],
                "analysis": {"score": 55, "convergence": 0.5},
                "latest": {"close": 1.0},
                "trade_plan": {"long": {}, "short": {}},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        scanner,
        "build_live_execution_plan",
        lambda _settings, _strategy, analysis, **_kwargs: _plan(
            symbol=analysis["symbol"],
            allowed=False,
            violations=[
                "Volume z-score is below floor.",
                "Order notional 4.9000 USDT is below exchange minimum 5.0000.",
                "Recent DOGEUSDT BUY profit factor below live threshold.",
            ],
        ),
    )

    payload = scanner.run_ai_readiness_scan(output_dir=tmp_path)

    assert payload["allowed_count"] == 0
    assert payload["selected_ready_candidate"] is None
    assert payload["next_machine_action"] == "repair_exchange_sizing_or_margin"
    assert payload["machine_action_queue"][0]["action"] == "repair_exchange_sizing_or_margin"
    taxonomy = payload["hard_blocker_taxonomy"]
    assert "market_state" in taxonomy
    assert "exchange_constraints" in taxonomy
    assert "strategy_performance" in taxonomy


def test_ai_readiness_scan_action_queue_keeps_post_kill_switch_work_visible(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(scanner, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        scanner,
        "run_hermes_ai_trader",
        lambda **_kwargs: {
            "report_path": str(tmp_path / "hermes.json"),
            "candidate_queue": [_queue_item(1, "DOGEUSDT"), _queue_item(2, "ETHUSDT")],
        },
    )
    monkeypatch.setattr(scanner, "load_settings", lambda: object())
    monkeypatch.setattr(scanner, "load_strategy_config", lambda _path: type(
        "Strategy",
        (),
        {
            "defaults": type(
                "Defaults",
                (),
                {
                    "market": "futures",
                    "interval": "4h",
                    "limit": 600,
                    "use_blave": False,
                },
            )()
        },
    )())
    monkeypatch.setattr(
        scanner,
        "run_analysis",
        lambda _settings, **kwargs: (
            {
                "symbol": kwargs["symbol"],
                "market": kwargs["market"],
                "analysis": {"score": 55, "convergence": 0.5},
                "latest": {"close": 1.0},
                "trade_plan": {"long": {}, "short": {}},
            },
            None,
        ),
    )

    def fake_plan(_settings, _strategy, analysis, **_kwargs):
        symbol = analysis["symbol"]
        extra = (
            "Order notional 4.9000 USDT is below exchange minimum 5.0000."
            if symbol == "ETHUSDT"
            else "Recent PF 0.53 is below 0.85."
        )
        return _plan(
            symbol=symbol,
            allowed=False,
            violations=[
                "Trading is paused by kill-switch: consecutive losses reached 2.",
                extra,
            ],
        )

    monkeypatch.setattr(scanner, "build_live_execution_plan", fake_plan)

    payload = scanner.run_ai_readiness_scan(output_dir=tmp_path)

    assert payload["next_machine_action"] == "wait_for_kill_switch_clear"
    assert payload["ready_after_global_unlock_count"] == 0
    assert [item["action"] for item in payload["machine_action_queue"]] == [
        "wait_for_kill_switch_clear",
        "repair_exchange_sizing_or_margin",
        "repair_strategy_performance_or_route_history",
    ]
    kill_row = payload["machine_action_queue"][0]
    assert kill_row["unlock_ready_candidate_count"] == 0


def test_ai_readiness_scan_marks_candidates_ready_after_global_unlock(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(scanner, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        scanner,
        "run_hermes_ai_trader",
        lambda **_kwargs: {
            "report_path": str(tmp_path / "hermes.json"),
            "candidate_queue": [_queue_item(1, "ETHUSDT"), _queue_item(2, "DOGEUSDT")],
        },
    )
    monkeypatch.setattr(scanner, "load_settings", lambda: object())
    monkeypatch.setattr(scanner, "load_strategy_config", lambda _path: type(
        "Strategy",
        (),
        {
            "defaults": type(
                "Defaults",
                (),
                {
                    "market": "futures",
                    "interval": "4h",
                    "limit": 600,
                    "use_blave": False,
                },
            )()
        },
    )())
    monkeypatch.setattr(
        scanner,
        "run_analysis",
        lambda _settings, **kwargs: (
            {
                "symbol": kwargs["symbol"],
                "market": kwargs["market"],
                "analysis": {"score": 80, "convergence": 0.82},
                "latest": {"close": 1.0},
                "trade_plan": {"long": {}, "short": {}},
            },
            None,
        ),
    )

    def fake_plan(_settings, _strategy, analysis, **_kwargs):
        symbol = analysis["symbol"]
        violations = ["Trading is paused by kill-switch: consecutive losses reached 2."]
        if symbol == "DOGEUSDT":
            violations.append("Volume z-score is below floor.")
        return _plan(symbol=symbol, allowed=False, violations=violations)

    monkeypatch.setattr(scanner, "build_live_execution_plan", fake_plan)

    payload = scanner.run_ai_readiness_scan(output_dir=tmp_path)

    assert payload["allowed_count"] == 0
    assert payload["next_machine_action"] == "wait_for_kill_switch_clear"
    assert payload["ready_after_global_unlock_count"] == 1
    assert payload["selected_after_global_unlock"]["symbol"] == "ETHUSDT"
    assert payload["ready_after_global_unlock_candidates"][0]["next_action_after_global_unlock"] == "execute_ready_dry_run_only"
    kill_row = payload["machine_action_queue"][0]
    assert kill_row["unlock_ready_candidate_count"] == 1
    assert kill_row["unlock_ready_candidates"] == ["ETHUSDT:BUY"]


def test_ai_readiness_scan_writes_json_when_plan_contains_infinite_values(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(scanner, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        scanner,
        "run_hermes_ai_trader",
        lambda **_kwargs: {
            "report_path": str(tmp_path / "hermes.json"),
            "candidate_queue": [_queue_item(1, "DOGEUSDT")],
        },
    )
    monkeypatch.setattr(scanner, "load_settings", lambda: object())
    monkeypatch.setattr(scanner, "load_strategy_config", lambda _path: type(
        "Strategy",
        (),
        {
            "defaults": type(
                "Defaults",
                (),
                {
                    "market": "futures",
                    "interval": "4h",
                    "limit": 600,
                    "use_blave": False,
                },
            )()
        },
    )())
    monkeypatch.setattr(
        scanner,
        "run_analysis",
        lambda _settings, **kwargs: (
            {
                "symbol": kwargs["symbol"],
                "market": kwargs["market"],
                "analysis": {"score": 80, "convergence": 0.82},
                "latest": {"close": 1.0},
                "trade_plan": {"long": {}, "short": {}},
            },
            None,
        ),
    )

    def fake_plan(_settings, _strategy, analysis, **_kwargs):
        plan = _plan(symbol=analysis["symbol"], allowed=True)
        original = plan.to_dict
        return type(
            "InfPlan",
            (),
            {
                "to_dict": lambda self: {
                    **original(),
                    "challenge": {
                        **original()["challenge"],
                        "historical_signal_risk": {"profit_factor": float("inf")},
                    },
                }
            },
        )()

    monkeypatch.setattr(scanner, "build_live_execution_plan", fake_plan)

    payload = scanner.run_ai_readiness_scan(output_dir=tmp_path)

    report_text = Path(payload["report_path"]).read_text(encoding="utf-8")
    assert '"profit_factor": "inf"' in report_text
