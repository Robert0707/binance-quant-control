from __future__ import annotations

from binance_quant_control.hailo_trading_plan import build_hailo_trading_plan
from binance_quant_control.hermes_ai_trader import _open_order_gate
from binance_quant_control.model_registry import model_registry_payload
from binance_quant_control.signal_schema import TradingSignal


def test_model_registry_marks_hailo_as_optional_veto_layer() -> None:
    payload = model_registry_payload()

    assert payload["hailo_eligible"][0]["name"] == "chart-regime-triage"
    assert payload["training_contract"]["requires_dataset_hash"] is True
    assert any(item["name"] == "feature-label-dataset-builder" for item in payload["models"])
    assert "cannot bypass" in payload["hard_rule"]


def test_hailo_plan_refuses_backtest_acceleration_and_execution_approval() -> None:
    payload = build_hailo_trading_plan()
    tasks = {item["name"]: item for item in payload["tasks"]}

    assert tasks["chart-regime-triage"]["status"] == "eligible"
    assert tasks["pandas-backtest-acceleration"]["status"] == "not_eligible"
    assert tasks["order-execution-decision"]["status"] == "not_allowed"


def test_hermes_ai_trader_open_order_gate_includes_hailo_veto() -> None:
    signal = TradingSignal(
        signal_id="sig-1",
        symbol="ETHUSDT",
        side="BUY",
        interval="1h",
        route_id="eth-core",
        strategy_family="trend",
        source="test",
        status="candidate",
        signal_score=90.0,
        expectancy_r=0.08,
        payoff_ratio=1.4,
        trade_count=150,
        blockers=[],
        created_at="2026-05-07T00:00:00+00:00",
    )

    gate = _open_order_gate(
        audit={"trade_ready": True, "critical_blockers": []},
        signal=signal,
        portfolio_target={"accepted": True},
        committee={"decision": "approve"},
        hailo_gate={"allowed": False, "blockers": ["hailo-veto:market_state_failed"]},
    )

    assert gate.allowed is False
    assert "hailo-veto:market_state_failed" in gate.blockers
