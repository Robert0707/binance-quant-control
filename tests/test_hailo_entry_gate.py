from __future__ import annotations

from binance_quant_control.hailo_entry_gate import evaluate_hailo_entry_gate


def test_hailo_entry_gate_vetoes_high_priority_local_quant_risk() -> None:
    payload = {
        "returncode": 0,
        "response": {
            "raw_event_count": 3,
            "output_event_count": 1,
            "events": [
                {
                    "source": "quant_runtime",
                    "event_type": "system_review",
                    "priority": "high",
                    "labels": ["professional_gate_failed", "hailo_event_classifier"],
                    "reason": "local_rules_sufficient",
                }
            ],
        },
    }

    gate = evaluate_hailo_entry_gate(payload)

    assert gate["allowed"] is False
    assert "hailo-veto:professional_gate_failed" in gate["blockers"]
    assert gate["decision"] == "veto"


def test_hailo_entry_gate_does_not_veto_high_priority_quality_setup() -> None:
    payload = {
        "returncode": 0,
        "response": {
            "events": [
                {
                    "source": "strategy",
                    "event_type": "trade_candidate",
                    "priority": "high",
                    "labels": ["entry_ready", "actionable_signal", "high_quality_setup"],
                }
            ],
        },
    }

    gate = evaluate_hailo_entry_gate(payload)

    assert gate["allowed"] is True
    assert gate["blockers"] == []


def test_hailo_entry_gate_vetoes_event_risk_and_critical_errors() -> None:
    payload = {
        "returncode": 0,
        "response": {
            "events": [
                {
                    "source": "news",
                    "event_type": "news_event",
                    "priority": "high",
                    "labels": ["high_event_risk"],
                },
                {
                    "source": "error",
                    "event_type": "system_error",
                    "priority": "critical",
                    "labels": ["needs_root_cause_review"],
                },
            ],
        },
    }

    gate = evaluate_hailo_entry_gate(payload)

    assert gate["allowed"] is False
    assert "hailo-veto:high_event_risk" in gate["blockers"]
    assert "hailo-veto:priority_critical" in gate["blockers"]


def test_hailo_entry_gate_is_observability_only_when_no_output_events() -> None:
    payload = {
        "returncode": 0,
        "response": {
            "raw_event_count": 1,
            "output_event_count": 0,
            "retained_existing_output": True,
        },
    }

    gate = evaluate_hailo_entry_gate(payload)

    assert gate["allowed"] is True
    assert gate["decision"] == "allow"
    assert gate["blockers"] == []


def test_hailo_entry_gate_blocks_when_triage_fails_closed() -> None:
    gate = evaluate_hailo_entry_gate({"returncode": 1, "stderr": "hailo unavailable"})

    assert gate["allowed"] is False
    assert gate["decision"] == "unavailable"
    assert gate["blockers"] == ["hailo-triage-unavailable"]
