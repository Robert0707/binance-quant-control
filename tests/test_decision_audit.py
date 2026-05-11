from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import binance_quant_control.cli as cli
from binance_quant_control.decision_audit import run_decision_audit


def _write_decision(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _base_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "decision": "HOLD",
        "direction": "neutral",
        "confidence": 0,
        "regime": "unknown",
        "entry_reason": ["No complete entry setup."],
        "blocked_reasons": ["Conditions are incomplete."],
        "invalid_if": {"price_crosses": None, "source": "test", "notes": []},
        "entry": None,
        "stop_loss": None,
        "take_profit": [],
        "risk_pct": 0.0,
        "risk_reward_ratio": None,
        "expected_value": {"positive": False},
        "position_size": {
            "quantity": None,
            "margin_notional_usdt": None,
            "gross_notional_usdt": None,
            "leverage": None,
        },
        "hailo_task_allocation": {
            "mode": "hailo_trading_plan",
            "tasks": [
                {"name": "chart-regime-triage", "status": "eligible"},
                {"name": "candlestick-image-anomaly-veto", "status": "eligible-after-training"},
                {"name": "order-execution-decision", "status": "not_allowed"},
            ],
            "hard_rule": "Hailo can veto or triage; it cannot approve trades without alpha/risk gates.",
        },
        "decision_answers": {
            "why_long_now": ["Not a long decision under current gates."],
            "why_short_now": ["Not a short decision under current gates."],
            "why_no_trade_now": ["Conditions are incomplete."],
            "where_stop_if_wrong": None,
            "where_take_profit_if_right": [],
            "max_loss": "Max planned account risk is 0.0000%.",
            "long_term_expected_value": {"positive": False},
        },
        "hard_gates": {
            "entry_factor_gates_passed": False,
            "ai_direct_order_allowed": False,
        },
        "opens_orders": False,
        "writes_execution_config": False,
        "decision_contract_validation": {"valid": True, "failed": []},
    }
    payload.update(overrides)
    return payload


def _passed_entry_evidence() -> dict[str, object]:
    return {
        "all_passed": True,
        "failed": [],
        "gates": [
            {"name": "regime_policy", "passed": True},
            {"name": "trend_direction", "passed": True},
            {"name": "momentum", "passed": True},
            {"name": "volume", "passed": True},
            {"name": "volatility", "passed": True},
            {"name": "support_resistance", "passed": True},
            {"name": "risk_reward", "passed": True},
            {"name": "stop_distance", "passed": True},
        ],
    }


def _passed_lifecycle_evidence() -> dict[str, object]:
    return {
        "all_passed": True,
        "failed": [],
        "gates": [
            {"name": "data_check", "passed": True},
            {"name": "backtest_or_performance_evidence", "passed": True},
            {"name": "trading_cost_check", "passed": True},
            {"name": "max_drawdown_check", "passed": True},
            {"name": "dry_run_check", "passed": True},
            {"name": "testnet_forward_check", "passed": True},
        ],
    }


def test_decision_audit_passes_valid_read_only_artifacts(tmp_path: Path) -> None:
    input_dir = tmp_path / "decisions"
    input_dir.mkdir()
    _write_decision(
        input_dir / "20260510T000000Z-aaaaaa-btcusdt-hold-trade-decision.json",
        _base_payload(),
    )

    payload = run_decision_audit(input_dir=input_dir, output_dir=tmp_path / "audit")

    assert payload["status"] == "passed"
    assert payload["summary"]["artifact_count"] == 1
    assert payload["summary"]["invalid_count"] == 0
    assert payload["summary"]["decision_counts"] == {"HOLD": 1}
    assert Path(payload["report_path"]).exists()


def test_decision_audit_flags_missing_required_contract_fields(tmp_path: Path) -> None:
    input_dir = tmp_path / "decisions"
    input_dir.mkdir()
    incomplete = _base_payload()
    incomplete.pop("invalid_if")
    incomplete["decision_answers"] = {"why_no_trade_now": ["missing answers"]}
    _write_decision(
        input_dir / "20260510T000000Z-aaaaaa-btcusdt-hold-trade-decision.json",
        incomplete,
    )

    payload = run_decision_audit(input_dir=input_dir, output_dir=tmp_path / "audit")

    assert payload["status"] == "failed"
    violations = payload["invalid_artifacts"][0]["violations"]
    assert any(str(item).startswith("missing-required-fields:invalid_if") for item in violations)
    assert any(str(item).startswith("missing-decision-answer-fields:") for item in violations)


def test_decision_audit_flags_invalid_value_shapes(tmp_path: Path) -> None:
    input_dir = tmp_path / "decisions"
    input_dir.mkdir()
    _write_decision(
        input_dir / "20260510T000000Z-aaaaaa-btcusdt-buy-trade-decision.json",
        _base_payload(
            decision="BUY",
            direction="bullish",
            confidence=101,
            entry=100.0,
            stop_loss=96.0,
            take_profit="106",
            risk_pct=-0.01,
            risk_reward_ratio=0.0,
            expected_value=[],
            position_size=[],
            invalid_if=[],
            hard_gates=[],
        ),
    )

    payload = run_decision_audit(input_dir=input_dir, output_dir=tmp_path / "audit")

    assert payload["status"] == "failed"
    violations = payload["invalid_artifacts"][0]["violations"]
    assert "confidence-out-of-range" in violations
    assert "risk-pct-invalid" in violations
    assert "entry-risk-reward-invalid" in violations
    assert "take-profit-not-list" in violations
    assert "invalid-if-not-object" in violations
    assert "expected-value-not-object" in violations
    assert "position-size-not-object" in violations
    assert "hard-gates-not-object" in violations


def test_decision_audit_flags_hailo_order_execution_approval(tmp_path: Path) -> None:
    input_dir = tmp_path / "decisions"
    input_dir.mkdir()
    payload = _base_payload()
    payload["hailo_task_allocation"] = {
        "mode": "hailo_trading_plan",
        "tasks": [
            {"name": "chart-regime-triage", "status": "eligible"},
            {"name": "order-execution-decision", "status": "eligible"},
        ],
        "hard_rule": "bad boundary",
    }
    _write_decision(
        input_dir / "20260510T000000Z-aaaaaa-btcusdt-hold-trade-decision.json",
        payload,
    )

    payload = run_decision_audit(input_dir=input_dir, output_dir=tmp_path / "audit")

    assert payload["status"] == "failed"
    violations = payload["invalid_artifacts"][0]["violations"]
    assert "hailo-order-execution-not-forbidden" in violations


def test_decision_audit_flags_unsafe_entry_artifacts(tmp_path: Path) -> None:
    input_dir = tmp_path / "decisions"
    input_dir.mkdir()
    _write_decision(
        input_dir / "20260510T000000Z-aaaaaa-btcusdt-buy-trade-decision.json",
        _base_payload(
            decision="BUY",
            direction="bullish",
            risk_pct=0.031,
            expected_value={"positive": False},
            hard_gates={
                "risk_ceiling_passed": False,
                "readiness_allowed": False,
                "entry_factor_gates_passed": False,
                "ai_direct_order_allowed": True,
            },
            opens_orders=True,
            decision_contract_validation={"valid": False, "failed": ["entry_risk_ceiling"]},
        ),
    )

    payload = run_decision_audit(input_dir=input_dir, output_dir=tmp_path / "audit")

    assert payload["status"] == "failed"
    assert payload["summary"]["invalid_count"] == 1
    violations = payload["invalid_artifacts"][0]["violations"]
    assert "entry-risk-exceeds-2.5pct" in violations
    assert "entry-without-risk-ceiling-gate" in violations
    assert "entry-without-readiness-allowance" in violations
    assert "entry-without-entry-factor-gates" in violations
    assert "entry-without-positive-expected-value" in violations
    assert "entry-without-positive-position-size" in violations
    assert "ai-direct-order-not-explicitly-forbidden" in violations
    assert "decision-artifact-opens-orders-not-false" in violations
    assert "embedded-contract-validation-failed" in violations


def test_decision_audit_flags_entry_without_execution_readiness(tmp_path: Path) -> None:
    input_dir = tmp_path / "decisions"
    input_dir.mkdir()
    _write_decision(
        input_dir / "20260510T000000Z-aaaaaa-btcusdt-buy-trade-decision.json",
        _base_payload(
            decision="BUY",
            direction="bullish",
            entry=100.0,
            stop_loss=96.0,
            take_profit=[106.0],
            risk_pct=0.006,
            risk_reward_ratio=1.5,
            expected_value={"positive": True},
            position_size={
                "quantity": 0.0,
                "margin_notional_usdt": 125.0,
                "gross_notional_usdt": 250.0,
                "leverage": 2,
            },
            hard_gates={
                "risk_ceiling_passed": False,
                "readiness_allowed": False,
                "entry_factor_gates_passed": True,
                "ai_direct_order_allowed": False,
            },
        ),
    )

    payload = run_decision_audit(input_dir=input_dir, output_dir=tmp_path / "audit")

    assert payload["status"] == "failed"
    violations = payload["invalid_artifacts"][0]["violations"]
    assert "entry-without-risk-ceiling-gate" in violations
    assert "entry-without-readiness-allowance" in violations
    assert "entry-without-positive-position-size" in violations


def test_decision_audit_flags_entry_without_required_gate_evidence(tmp_path: Path) -> None:
    input_dir = tmp_path / "decisions"
    input_dir.mkdir()
    _write_decision(
        input_dir / "20260510T000000Z-aaaaaa-btcusdt-buy-trade-decision.json",
        _base_payload(
            decision="BUY",
            direction="bullish",
            entry=100.0,
            stop_loss=96.0,
            take_profit=[106.0],
            risk_pct=0.006,
            risk_reward_ratio=1.5,
            expected_value={"positive": True},
            position_size={
                "quantity": 1.25,
                "margin_notional_usdt": 125.0,
                "gross_notional_usdt": 250.0,
                "leverage": 2,
            },
            hard_gates={
                "risk_ceiling_passed": True,
                "readiness_allowed": True,
                "entry_factor_gates_passed": True,
                "ai_direct_order_allowed": False,
            },
            entry_gate_evidence={
                "all_passed": False,
                "failed": ["momentum"],
                "gates": [{"name": "trend_direction", "passed": True}],
            },
            lifecycle_gate_evidence={
                "all_passed": False,
                "failed": ["testnet_forward_check"],
                "gates": [{"name": "data_check", "passed": True}],
            },
        ),
    )

    payload = run_decision_audit(input_dir=input_dir, output_dir=tmp_path / "audit")

    assert payload["status"] == "failed"
    violations = payload["invalid_artifacts"][0]["violations"]
    assert "entry-without-passed-entry-gate-evidence" in violations
    assert any(str(item).startswith("entry-missing-required-gate-evidence:") for item in violations)
    assert "entry-without-passed-lifecycle-evidence" in violations
    assert any(str(item).startswith("entry-missing-required-lifecycle-evidence:") for item in violations)


def test_decision_audit_flags_entry_gate_rows_that_did_not_pass(tmp_path: Path) -> None:
    input_dir = tmp_path / "decisions"
    input_dir.mkdir()
    entry_evidence = _passed_entry_evidence()
    entry_evidence["failed"] = ["momentum"]
    entry_evidence["gates"] = [
        {**item, "passed": False} if item["name"] == "momentum" else item
        for item in entry_evidence["gates"]  # type: ignore[index]
    ]
    lifecycle_evidence = _passed_lifecycle_evidence()
    lifecycle_evidence["failed"] = ["trading_cost_check"]
    lifecycle_evidence["gates"] = [
        {**item, "passed": False} if item["name"] == "trading_cost_check" else item
        for item in lifecycle_evidence["gates"]  # type: ignore[index]
    ]
    _write_decision(
        input_dir / "20260510T000000Z-aaaaaa-btcusdt-buy-trade-decision.json",
        _base_payload(
            decision="BUY",
            direction="bullish",
            entry=100.0,
            stop_loss=96.0,
            take_profit=[106.0],
            risk_pct=0.006,
            risk_reward_ratio=1.5,
            expected_value={"positive": True},
            position_size={
                "quantity": 1.25,
                "margin_notional_usdt": 125.0,
                "gross_notional_usdt": 250.0,
                "leverage": 2,
            },
            hard_gates={
                "risk_ceiling_passed": True,
                "readiness_allowed": True,
                "entry_factor_gates_passed": True,
                "ai_direct_order_allowed": False,
            },
            entry_gate_evidence=entry_evidence,
            lifecycle_gate_evidence=lifecycle_evidence,
        ),
    )

    payload = run_decision_audit(input_dir=input_dir, output_dir=tmp_path / "audit")

    assert payload["status"] == "failed"
    violations = payload["invalid_artifacts"][0]["violations"]
    assert "entry-required-gates-not-passed:momentum" in violations
    assert "entry-gate-evidence-has-failures:momentum" in violations
    assert "entry-required-lifecycle-gates-not-passed:trading_cost_check" in violations
    assert "entry-lifecycle-evidence-has-failures:trading_cost_check" in violations


def test_decision_audit_accepts_long_alias_with_valid_structure(tmp_path: Path) -> None:
    input_dir = tmp_path / "decisions"
    input_dir.mkdir()
    _write_decision(
        input_dir / "20260510T000000Z-aaaaaa-btcusdt-long-trade-decision.json",
        _base_payload(
            decision="LONG",
            direction="bullish",
            entry=100.0,
            stop_loss=96.0,
            take_profit=[106.0],
            risk_pct=0.006,
            risk_reward_ratio=1.5,
            expected_value={"positive": True},
            position_size={
                "quantity": 1.25,
                "margin_notional_usdt": 125.0,
                "gross_notional_usdt": 250.0,
                "leverage": 2,
            },
            hard_gates={
                "risk_ceiling_passed": True,
                "readiness_allowed": True,
                "entry_factor_gates_passed": True,
                "ai_direct_order_allowed": False,
            },
            entry_gate_evidence=_passed_entry_evidence(),
            lifecycle_gate_evidence=_passed_lifecycle_evidence(),
        ),
    )

    payload = run_decision_audit(input_dir=input_dir, output_dir=tmp_path / "audit")

    assert payload["status"] == "passed"
    assert payload["summary"]["decision_counts"] == {"LONG": 1}


def test_decision_audit_flags_direction_and_structure_mismatch(tmp_path: Path) -> None:
    input_dir = tmp_path / "decisions"
    input_dir.mkdir()
    _write_decision(
        input_dir / "20260510T000000Z-aaaaaa-btcusdt-short-trade-decision.json",
        _base_payload(
            decision="SHORT",
            direction="bullish",
            entry=100.0,
            stop_loss=96.0,
            take_profit=[106.0],
            risk_pct=0.006,
            risk_reward_ratio=1.5,
            expected_value={"positive": True},
            position_size={
                "quantity": 1.25,
                "margin_notional_usdt": 125.0,
                "gross_notional_usdt": 250.0,
                "leverage": 2,
            },
            hard_gates={
                "risk_ceiling_passed": True,
                "readiness_allowed": True,
                "entry_factor_gates_passed": True,
                "ai_direct_order_allowed": False,
            },
            entry_gate_evidence=_passed_entry_evidence(),
            lifecycle_gate_evidence=_passed_lifecycle_evidence(),
        ),
    )

    payload = run_decision_audit(input_dir=input_dir, output_dir=tmp_path / "audit")

    assert payload["status"] == "failed"
    violations = payload["invalid_artifacts"][0]["violations"]
    assert "decision-direction-mismatch" in violations
    assert "entry-stop-take-profit-orientation-invalid" in violations


def test_decision_audit_accepts_valid_reduce_only_exit(tmp_path: Path) -> None:
    input_dir = tmp_path / "decisions"
    input_dir.mkdir()
    _write_decision(
        input_dir / "20260510T000000Z-aaaaaa-btcusdt-exit-trade-decision.json",
        _base_payload(
            decision="EXIT",
            direction="bullish",
            confidence=100,
            entry_reason=["Confirmed adaptive reversal; close existing position."],
            blocked_reasons=[],
            invalid_if={
                "price_crosses": 99000.0,
                "source": "adaptive_exit_reference_stop",
                "notes": ["Open position is invalid if the reference stop is touched."],
            },
            entry=100000.0,
            stop_loss=99000.0,
            take_profit=[],
            risk_pct=0.01,
            risk_reward_ratio=None,
            expected_value={
                "positive": False,
                "reason_code": "confirmed-adaptive-reversal",
                "unrealized_r": 0.6,
            },
            position_size={
                "quantity": 0.01,
                "margin_notional_usdt": None,
                "gross_notional_usdt": 1000.0,
                "leverage": 1,
            },
            hard_gates={
                "risk_ceiling_passed": True,
                "readiness_allowed": True,
                "reduce_only_exit": True,
                "ai_direct_order_allowed": False,
            },
        ),
    )

    payload = run_decision_audit(input_dir=input_dir, output_dir=tmp_path / "audit")

    assert payload["status"] == "passed"
    assert payload["summary"]["decision_counts"] == {"EXIT": 1}


def test_decision_audit_flags_unsafe_exit_artifacts(tmp_path: Path) -> None:
    input_dir = tmp_path / "decisions"
    input_dir.mkdir()
    _write_decision(
        input_dir / "20260510T000000Z-aaaaaa-btcusdt-exit-trade-decision.json",
        _base_payload(
            decision="EXIT",
            direction="bearish",
            entry=None,
            stop_loss=None,
            risk_pct=0.0,
            expected_value={"positive": False},
            position_size={
                "quantity": 0.0,
                "margin_notional_usdt": None,
                "gross_notional_usdt": None,
                "leverage": 1,
            },
            hard_gates={
                "readiness_allowed": False,
                "reduce_only_exit": False,
                "ai_direct_order_allowed": False,
            },
        ),
    )

    payload = run_decision_audit(input_dir=input_dir, output_dir=tmp_path / "audit")

    assert payload["status"] == "failed"
    violations = payload["invalid_artifacts"][0]["violations"]
    assert "exit-not-reduce-only" in violations
    assert "exit-without-readiness-allowance" in violations
    assert "exit-without-positive-position-size" in violations
    assert "exit-without-active-position-entry" in violations
    assert "exit-without-reference-stop-loss" in violations
    assert "exit-without-reason-code" in violations


def test_decision_audit_since_contract_ignores_legacy_artifacts(tmp_path: Path) -> None:
    input_dir = tmp_path / "decisions"
    input_dir.mkdir()
    legacy = _base_payload()
    legacy.pop("decision_contract_validation")
    _write_decision(
        input_dir / "20260510T000000Z-aaaaaa-btcusdt-hold-trade-decision.json",
        legacy,
    )
    _write_decision(
        input_dir / "20260510T000001Z-bbbbbb-btcusdt-hold-trade-decision.json",
        _base_payload(),
    )

    payload = run_decision_audit(
        input_dir=input_dir,
        output_dir=tmp_path / "audit",
        since_contract=True,
    )

    assert payload["status"] == "passed"
    assert payload["scope"] == "since_contract"
    assert payload["summary"]["artifact_count"] == 1
    assert payload["summary"]["invalid_count"] == 0


def test_cmd_decision_audit_compact_summarizes_results(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "run_decision_audit",
        lambda **kwargs: {
            "mode": "decision_audit_v1",
            "status": "passed",
            "scope": "since_contract",
            "summary": {"artifact_count": 2, "invalid_count": 0},
            "invalid_artifacts": [],
            "artifacts": [{"path": "state/trade-decisions/a.json"}],
            "report_path": "state/decision-audit/report.json",
            "kwargs": kwargs,
        },
    )
    monkeypatch.setattr(
        cli,
        "print_json",
        lambda payload, compact=False: captured.update({"payload": payload, "compact": compact}),
    )

    cli.cmd_decision_audit(
        SimpleNamespace(
            input_dir=str(tmp_path / "decisions"),
            output_dir=str(tmp_path / "audit"),
            since_contract=True,
            compact=True,
        )
    )

    payload = captured["payload"]  # type: ignore[assignment]
    assert captured["compact"] is True
    assert payload["status"] == "passed"  # type: ignore[index]
    assert payload["scope"] == "since_contract"  # type: ignore[index]
    assert payload["summary"]["artifact_count"] == 2  # type: ignore[index]
    assert "artifacts" not in payload  # type: ignore[operator]
