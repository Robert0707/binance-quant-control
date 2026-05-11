from __future__ import annotations

from types import SimpleNamespace

import binance_quant_control.cli as cli
from binance_quant_control.decision_output import (
    _validate_decision_contract,
    build_ai_exit_decision_output,
    build_ai_trade_decision_output,
    build_blocked_trade_decision_output,
)
from binance_quant_control.position_manager import AdaptiveExitPlan, PositionManagementPlan


def _analysis_payload(action: str = "BUY") -> dict[str, object]:
    return {
        "symbol": "ETHUSDT",
        "market": "futures",
        "analysis": {
            "recommended_action": action,
            "bias": "long-bias" if action == "BUY" else "short-bias" if action == "SELL" else "neutral",
            "score": 76,
            "convergence": 0.82,
            "regime": "trend-up",
            "entry_ready": action in {"BUY", "SELL"},
            "entry_blockers": [] if action in {"BUY", "SELL"} else ["Score does not meet entry thresholds."],
            "decision_notes": ["Trend and momentum are aligned."],
            "selected_strategy_family": {"family": "ai_family_router"},
        },
        "latest": {
            "realized_vol_20": 0.8,
            "volume_zscore_20": 0.4,
        },
        "market_context": {"spread_bps": 3.0},
        "trade_plan": {
            "long": {
                "entry_reference": 100.0,
                "invalidation": 96.0,
                "invalidation_source": "trend-line",
                "take_profit_levels": [106.0, 110.0, 116.0],
                "strategy_family": "ai_family_router",
            },
            "short": {
                "entry_reference": 100.0,
                "invalidation": 104.0,
                "invalidation_source": "trend-line",
                "take_profit_levels": [94.0, 90.0, 84.0],
                "strategy_family": "ai_family_router",
            },
        },
    }


def _live_plan(*, allowed: bool = True, side: str = "BUY", risk_pct: float = 0.006) -> dict[str, object]:
    return {
        "allowed": allowed,
        "symbol": "ETHUSDT",
        "market": "futures",
        "side": side,
        "price": 100.0,
        "stop_price": 96.0 if side == "BUY" else 104.0,
        "take_profit_prices": [106.0, 110.0, 116.0] if side == "BUY" else [94.0, 90.0, 84.0],
        "planned_account_risk_pct": risk_pct,
        "quantity": 1.25,
        "margin_notional_usdt": 25.0,
        "gross_notional_usdt": 100.0,
        "leverage": 4,
        "execution_mode": "testnet_exploration",
        "violations": [] if allowed else ["Recent expectancy is below threshold."],
        "challenge": {
            "route_side_risk": {
                "allowed": True,
                "profit_factor": 1.15,
                "net_pnl_usdt": 3.2,
            },
            "historical_signal_risk": {
                "buckets": [],
            },
            "optimizer_live_gate": {"allowed": True, "reasons": []},
            "market_bot_gate": {"allowed": False, "reasons": []},
            "status": "active",
            "current_drawdown_pct": 1.0,
            "max_drawdown_pct": 20.0,
        },
        "professional_entry_gate": {
            "passed": allowed,
            "layers": {
                "regime_policy": {
                    "passed": allowed,
                    "regime": "trend",
                    "strategy_family": "trend_continuation",
                    "allowed_families": ["trend_continuation", "trend_pullback", "breakout"],
                },
                "multi_timeframe_trend": {
                    "passed": allowed,
                    "expected_bias": "long" if side == "BUY" else "short",
                    "bias": "long" if side == "BUY" else "short",
                    "alignment": "strong",
                    "confidence": 0.82,
                },
                "market_state": {
                    "passed": allowed,
                    "realized_vol_20": 0.8,
                    "volume_zscore_20": 0.4,
                    "obv_zscore_20": 0.7 if side == "BUY" else -0.7,
                    "spread_bps": 3.0,
                },
                "signal_quality": {
                    "passed": allowed,
                    "analysis_score": 76,
                    "analysis_convergence": 0.82,
                    "adx_value": 24.0,
                    "obv_zscore_20": 0.7 if side == "BUY" else -0.7,
                    "price_structure_score": 72.0,
                },
                "execution_quality": {
                    "passed": allowed,
                    "reward_risk": 1.5,
                    "tp1_reward_risk": 1.5,
                },
                "strategy_performance": {
                    "passed": allowed,
                    "count": 32,
                    "scope": "route_side",
                    "profit_factor": 1.22,
                    "expectancy_r": 0.18,
                    "payoff_ratio": 1.6,
                    "avg_r_multiple": 0.12,
                    "stop_loss_ratio": 0.38,
                },
            },
        },
    }


def test_decision_output_emits_required_buy_contract() -> None:
    payload = build_ai_trade_decision_output(
        analysis_payload=_analysis_payload("BUY"),
        live_plan=_live_plan(),
    )

    assert payload["decision"] == "BUY"
    assert payload["direction"] == "bullish"
    assert payload["confidence"] == 76
    assert payload["regime"] == "trend"
    assert payload["entry"] == 100.0
    assert payload["stop_loss"] == 96.0
    assert payload["take_profit"] == [106.0, 110.0, 116.0]
    assert payload["risk_pct"] == 0.006
    assert payload["risk_reward_ratio"] == 1.5
    assert payload["expected_value"]["positive"] is True
    assert payload["expected_value"]["strategy_performance"]["profit_factor"] == 1.22
    assert payload["hard_gates"]["entry_factor_gates_passed"] is True
    assert payload["lifecycle_gate_evidence"]["gates"][0]["name"] == "data_check"
    assert payload["position_size"]["quantity"] == 1.25
    assert payload["hard_gates"]["ai_direct_order_allowed"] is False
    assert payload["entry_gate_evidence"]["all_passed"] is True
    assert payload["entry_gate_evidence"]["gates"][0]["name"] == "regime_policy"
    assert payload["decision_answers"]["why_long_now"]
    assert payload["decision_answers"]["why_short_now"] == ["Not a short decision under current gates."]
    assert payload["decision_answers"]["where_stop_if_wrong"] == 96.0
    assert payload["decision_answers"]["where_take_profit_if_right"] == [106.0, 110.0, 116.0]
    assert payload["decision_answers"]["long_term_expected_value"]["positive"] is True
    assert payload["decision_answers"]["next_actions"] == []
    assert payload["hailo_task_allocation"]["hard_rule"]
    hailo_tasks = {task["name"]: task for task in payload["hailo_task_allocation"]["tasks"]}
    assert hailo_tasks["chart-regime-triage"]["status"] == "eligible"
    assert hailo_tasks["candlestick-image-anomaly-veto"]["status"] == "eligible-after-training"
    assert hailo_tasks["order-execution-decision"]["status"] == "not_allowed"
    assert payload["opens_orders"] is False
    assert payload["writes_execution_config"] is False
    assert payload["decision_contract_validation"]["valid"] is True
    assert payload["decision_contract_validation"]["failed"] == []


def test_decision_output_emits_required_sell_contract() -> None:
    payload = build_ai_trade_decision_output(
        analysis_payload=_analysis_payload("SELL"),
        live_plan=_live_plan(side="SELL"),
    )

    assert payload["decision"] == "SELL"
    assert payload["direction"] == "bearish"
    assert payload["confidence"] == 76
    assert payload["entry"] == 100.0
    assert payload["stop_loss"] == 104.0
    assert payload["take_profit"] == [94.0, 90.0, 84.0]
    assert payload["risk_reward_ratio"] == 1.5
    assert payload["entry_gate_evidence"]["all_passed"] is True
    assert payload["entry_gate_evidence"]["gates"][1]["evidence"]["side"] == "SELL"
    assert payload["decision_answers"]["why_short_now"]
    assert payload["decision_answers"]["why_long_now"] == ["Not a long decision under current gates."]
    assert payload["decision_answers"]["where_stop_if_wrong"] == 104.0
    assert payload["decision_answers"]["where_take_profit_if_right"] == [94.0, 90.0, 84.0]
    assert payload["decision_answers"]["next_actions"] == []
    assert payload["decision_contract_validation"]["valid"] is True


def test_decision_output_forces_hold_when_risk_exceeds_ceiling() -> None:
    payload = build_ai_trade_decision_output(
        analysis_payload=_analysis_payload("BUY"),
        live_plan=_live_plan(risk_pct=0.031),
    )

    assert payload["decision"] == "HOLD"
    assert payload["hard_gates"]["risk_ceiling_passed"] is False
    assert any("2.5%" in item for item in payload["blocked_reasons"])
    assert payload["decision_contract_validation"]["valid"] is True


def test_decision_output_keeps_hold_as_legal_decision_when_blocked() -> None:
    payload = build_ai_trade_decision_output(
        analysis_payload=_analysis_payload("HOLD"),
        live_plan=_live_plan(allowed=False),
    )

    assert payload["decision"] == "HOLD"
    assert payload["direction"] == "neutral"
    assert payload["blocked_reasons"]
    assert payload["stop_loss"] == 96.0
    assert payload["take_profit"]
    assert payload["entry_gate_evidence"]["all_passed"] is False
    assert payload["decision_answers"]["why_no_trade_now"]
    assert payload["decision_answers"]["failed_entry_gates"]
    next_actions = {item["action"] for item in payload["decision_answers"]["next_actions"]}
    assert "repair_expectancy_before_enabling_entries" in next_actions


def test_decision_output_forces_hold_when_any_entry_factor_fails() -> None:
    live_plan = _live_plan()
    live_plan["professional_entry_gate"]["layers"]["market_state"]["passed"] = False  # type: ignore[index]
    live_plan["professional_entry_gate"]["layers"]["market_state"]["volume_zscore_20"] = -1.5  # type: ignore[index]

    payload = build_ai_trade_decision_output(
        analysis_payload=_analysis_payload("BUY"),
        live_plan=live_plan,
    )

    assert payload["decision"] == "HOLD"
    assert "volume" in payload["entry_gate_evidence"]["failed"]
    assert any("Entry factor gate failed: volume" in item for item in payload["blocked_reasons"])
    next_actions = {item["action"] for item in payload["decision_answers"]["next_actions"]}
    assert "wait_for_liquidity_and_volume_confirmation" in next_actions


def test_blocked_short_candidate_keeps_short_rationale() -> None:
    live_plan = _live_plan(side="SELL")
    live_plan["professional_entry_gate"]["layers"]["multi_timeframe_trend"]["passed"] = False  # type: ignore[index]
    live_plan["professional_entry_gate"]["layers"]["multi_timeframe_trend"]["bias"] = "long"  # type: ignore[index]
    live_plan["violations"] = ["Multi-timeframe trend conflicts with SELL."]

    payload = build_ai_trade_decision_output(
        analysis_payload=_analysis_payload("SELL"),
        live_plan=live_plan,
    )

    assert payload["decision"] == "HOLD"
    assert payload["direction"] == "bearish"
    assert payload["decision_answers"]["why_short_now"]
    assert payload["decision_answers"]["why_long_now"] == ["Not a long decision under current gates."]
    assert "trend_direction" in payload["decision_answers"]["failed_entry_gates"]


def test_blocked_conflicting_short_candidate_does_not_claim_approval() -> None:
    analysis_payload = _analysis_payload("SELL")
    analysis_payload["analysis"]["bias"] = "long-bias"  # type: ignore[index]
    live_plan = _live_plan(side="SELL")
    live_plan["professional_entry_gate"]["layers"]["multi_timeframe_trend"]["passed"] = False  # type: ignore[index]
    live_plan["professional_entry_gate"]["layers"]["multi_timeframe_trend"]["bias"] = "long"  # type: ignore[index]
    live_plan["violations"] = ["Multi-timeframe trend conflicts with SELL."]

    payload = build_ai_trade_decision_output(
        analysis_payload=analysis_payload,
        live_plan=live_plan,
        requested_action="SELL",
    )

    assert payload["decision"] == "HOLD"
    assert any("SELL candidate was evaluated" in item for item in payload["entry_reason"])
    assert not any("SELL is supported" in item for item in payload["entry_reason"])
    assert not any("Trend and momentum are aligned." == item for item in payload["entry_reason"])


def test_hold_entry_evidence_uses_price_structure_candidate_side() -> None:
    payload = build_ai_trade_decision_output(
        analysis_payload=_analysis_payload("HOLD"),
        live_plan=_live_plan(allowed=False, side="HOLD"),
    )

    trend_gate = payload["entry_gate_evidence"]["gates"][1]
    assert trend_gate["name"] == "trend_direction"
    assert trend_gate["evidence"]["side"] == "SELL"
    assert payload["risk_reward_ratio"] == 1.5


def test_live_decision_forces_hold_when_lifecycle_gate_fails() -> None:
    live_plan = _live_plan()
    live_plan["execution_mode"] = "live"
    live_plan["challenge"]["optimizer_live_gate"]["allowed"] = False  # type: ignore[index]
    live_plan["professional_entry_gate"]["layers"]["strategy_performance"]["count"] = 4  # type: ignore[index]
    live_plan["professional_entry_gate"]["layers"]["strategy_performance"]["profit_factor"] = 0.6  # type: ignore[index]
    live_plan["professional_entry_gate"]["layers"]["strategy_performance"]["expectancy_r"] = -0.2  # type: ignore[index]

    payload = build_ai_trade_decision_output(
        analysis_payload=_analysis_payload("BUY"),
        live_plan=live_plan,
    )

    assert payload["decision"] == "HOLD"
    assert payload["hard_gates"]["live_promotion_lifecycle_passed"] is False
    assert "backtest_or_performance_evidence" in payload["lifecycle_gate_evidence"]["failed"]
    assert "testnet_forward_check" in payload["lifecycle_gate_evidence"]["failed"]
    assert any("Live promotion lifecycle gate failed" in item for item in payload["blocked_reasons"])
    next_actions = {item["action"] for item in payload["decision_answers"]["next_actions"]}
    assert "run_backtest_or_repair_negative_expectancy" in next_actions
    assert "collect_testnet_forward_evidence" in next_actions


def _position_plan() -> PositionManagementPlan:
    return PositionManagementPlan(
        allowed=True,
        symbol="BTCUSDT",
        market="futures",
        side="BUY",
        quantity=0.01,
        entry_price=100000.0,
        mark_price=100600.0,
        unrealized_pnl_usdt=6.0,
        leverage=3,
        existing_open_orders=[],
        existing_algo_orders=[],
        step_size=0.001,
        tick_size=0.1,
        quantity_precision=3,
        price_precision=1,
        proposed_stop_price=None,
        proposed_take_profit_price=None,
        trailing_activation_price=None,
        trailing_callback_pct=None,
        trailing_quantity=None,
        cancel_existing_algo_orders=False,
        preserve_existing_take_profits=False,
        violations=[],
        warnings=[],
        actions=[],
    )


def _exit_plan(*, allowed: bool = True) -> AdaptiveExitPlan:
    return AdaptiveExitPlan(
        allowed=allowed,
        symbol="BTCUSDT",
        market="futures",
        side="BUY",
        exit_side="SELL",
        quantity=0.01,
        action="close_position" if allowed else "hold",
        reason_code="profit-protection-reversal" if allowed else "no-confirmed-reversal",
        reasons=["Score-model bias flipped short.", "Unrealized profit is 0.60R, enough to protect on reversal."],
        warnings=[],
        unrealized_r=0.6,
        reversal_score=8.5,
        confidence=1.0,
        risk_distance=1000.0,
        reference_stop_price=99000.0,
        entry_price=100000.0,
        mark_price=100600.0,
        analysis_bias="short",
        recommended_action="SELL",
        selected_family="trend_continuation",
        selected_family_bias="short",
    )


def test_exit_decision_output_emits_exit_for_confirmed_reversal() -> None:
    payload = build_ai_exit_decision_output(
        position_plan=_position_plan(),
        exit_plan=_exit_plan(),
        analysis_payload=_analysis_payload("SELL"),
    )

    assert payload["decision"] == "EXIT"
    assert payload["direction"] == "bullish"
    assert payload["confidence"] == 100
    assert payload["entry"] == 100000.0
    assert payload["stop_loss"] == 99000.0
    assert payload["position_size"]["quantity"] == 0.01
    assert payload["hard_gates"]["reduce_only_exit"] is True
    assert payload["hard_gates"]["ai_direct_order_allowed"] is False
    assert payload["decision_answers"]["where_stop_if_wrong"] == 99000.0
    assert payload["decision_answers"]["max_loss"] == "Max planned account risk is 1.0000%."
    assert payload["decision_contract_validation"]["valid"] is True


def test_exit_decision_output_holds_when_reversal_is_unconfirmed() -> None:
    payload = build_ai_exit_decision_output(
        position_plan=_position_plan(),
        exit_plan=_exit_plan(allowed=False),
        analysis_payload=_analysis_payload("BUY"),
    )

    assert payload["decision"] == "HOLD"
    assert payload["hard_gates"]["readiness_allowed"] is False
    assert any("no-confirmed-reversal" in item for item in payload["blocked_reasons"])


def test_decision_output_maps_pump_and_crash_regimes() -> None:
    pump_payload = _analysis_payload("BUY")
    pump_payload["analysis"]["regime"] = ""  # type: ignore[index]
    pump_payload["latest"] = {  # type: ignore[index]
        "close": 110.0,
        "ema_slow": 100.0,
        "macd_hist": 2.0,
        "realized_vol_20": 2.1,
        "volume_zscore_20": 3.0,
    }
    crash_payload = _analysis_payload("SELL")
    crash_payload["analysis"]["regime"] = ""  # type: ignore[index]
    crash_payload["latest"] = {  # type: ignore[index]
        "close": 90.0,
        "ema_slow": 100.0,
        "macd_hist": -2.0,
        "realized_vol_20": 2.1,
        "volume_zscore_20": 3.0,
    }

    assert build_ai_trade_decision_output(analysis_payload=pump_payload, live_plan=_live_plan())["regime"] == "pump"
    assert build_ai_trade_decision_output(
        analysis_payload=crash_payload,
        live_plan=_live_plan(side="SELL"),
    )["regime"] == "crash"


def test_decision_contract_validation_flags_unsafe_entry_mutations() -> None:
    payload = build_ai_trade_decision_output(
        analysis_payload=_analysis_payload("BUY"),
        live_plan=_live_plan(),
    )

    payload["hard_gates"]["ai_direct_order_allowed"] = True
    payload["opens_orders"] = True
    payload["stop_loss"] = 102.0
    payload["risk_pct"] = 0.03

    validation = _validate_decision_contract(payload)

    assert validation["valid"] is False
    assert "ai_cannot_directly_order" in validation["failed"]
    assert "read_only_decision_surface" in validation["failed"]
    assert "entry_stop_take_profit_orientation" in validation["failed"]
    assert "entry_risk_ceiling" in validation["failed"]


def test_decision_contract_validation_requires_positive_expectancy_for_entries() -> None:
    payload = build_ai_trade_decision_output(
        analysis_payload=_analysis_payload("BUY"),
        live_plan=_live_plan(),
    )
    payload["expected_value"]["positive"] = False

    validation = _validate_decision_contract(payload)

    assert validation["valid"] is False
    assert "positive_expected_value_required" in validation["failed"]


def test_decision_contract_validation_requires_hailo_and_answers() -> None:
    payload = build_ai_trade_decision_output(
        analysis_payload=_analysis_payload("BUY"),
        live_plan=_live_plan(),
    )
    payload["hailo_task_allocation"]["tasks"] = [
        {"name": "order-execution-decision", "status": "eligible"},
    ]
    payload["decision_answers"] = {"why_no_trade_now": []}

    validation = _validate_decision_contract(payload)

    assert validation["valid"] is False
    assert "hailo_cannot_execute_orders" in validation["failed"]
    assert "decision_answers_present" in validation["failed"]


def test_decision_contract_validation_requires_reduce_only_exit_boundary() -> None:
    payload = build_ai_exit_decision_output(
        position_plan=_position_plan(),
        exit_plan=_exit_plan(),
        analysis_payload=_analysis_payload("SELL"),
    )
    payload["hard_gates"]["reduce_only_exit"] = False
    payload["position_size"]["quantity"] = 0.0
    payload["entry"] = None
    payload["stop_loss"] = None

    validation = _validate_decision_contract(payload)

    assert validation["valid"] is False
    assert "exit_reduce_only_required" in validation["failed"]
    assert "exit_active_position_required" in validation["failed"]


def test_blocked_trade_decision_output_still_validates_contract() -> None:
    payload = build_blocked_trade_decision_output(
        reason="Private exchange checks are unavailable.",
        blockers=["binance-private-api-auth-failed:invalid-key"],
    )

    assert payload["decision"] == "HOLD"
    assert payload["opens_orders"] is False
    assert payload["writes_execution_config"] is False
    assert payload["hard_gates"]["ai_direct_order_allowed"] is False
    assert payload["decision_contract_validation"]["valid"] is True
    assert payload["decision_answers"]["why_no_trade_now"] == ["binance-private-api-auth-failed:invalid-key"]


def test_cmd_trade_decision_outputs_schema_without_execution(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "_build_live_response",
        lambda args: {
            "analysis_payload": _analysis_payload("BUY"),
            "live_plan": _live_plan(),
            "artifacts": {"report_json": "reports/analysis.json"},
        },
    )
    monkeypatch.setattr(cli, "load_strategy_config", lambda path: None)
    monkeypatch.setattr(cli, "_write_trade_decision_report", lambda payload: "state/trade-decisions/test.json")
    monkeypatch.setattr(
        cli,
        "print_json",
        lambda payload, compact=False: captured.update({"payload": payload, "compact": compact}),
    )

    cli.cmd_trade_decision(
        SimpleNamespace(
            strategy_config="config/strategy-live-pilot.yaml",
            action="",
            compact=True,
        )
    )

    payload = captured["payload"]  # type: ignore[assignment]
    assert captured["compact"] is True
    assert payload["decision"] == "BUY"  # type: ignore[index]
    assert payload["opens_orders"] is False  # type: ignore[index]
    assert payload["writes_execution_config"] is False  # type: ignore[index]
    assert payload["decision_report_path"] == "state/trade-decisions/test.json"  # type: ignore[index]


def test_cmd_trade_decision_exit_outputs_schema_without_execution(monkeypatch) -> None:
    captured: dict[str, object] = {}
    strategy = SimpleNamespace(
        profile="test",
        path="config/strategy-live-pilot.yaml",
        defaults=SimpleNamespace(
            symbol="BTCUSDT",
            market="futures",
            interval="4h",
            limit=240,
            use_blave=False,
            render_chart=False,
        ),
    )
    monkeypatch.setattr(cli, "load_settings", lambda: object())
    monkeypatch.setattr(cli, "load_strategy_config", lambda path: strategy)
    monkeypatch.setattr(cli, "run_analysis", lambda *args, **kwargs: (_analysis_payload("SELL"), SimpleNamespace(run_id="run-1")))
    monkeypatch.setattr(cli, "build_position_management_plan", lambda *args, **kwargs: _position_plan())
    monkeypatch.setattr(cli, "build_adaptive_exit_plan", lambda position_plan, analysis: _exit_plan())
    monkeypatch.setattr(cli, "_write_trade_decision_report", lambda payload: "state/trade-decisions/exit.json")
    monkeypatch.setattr(
        cli,
        "print_json",
        lambda payload, compact=False: captured.update({"payload": payload, "compact": compact}),
    )

    cli.cmd_trade_decision(
        SimpleNamespace(
            strategy_config="config/strategy-live-pilot.yaml",
            symbol="",
            market="",
            interval="",
            limit=0,
            use_blave=False,
            render_chart=False,
            action="EXIT",
            execution_mode="testnet_exploration",
            compact=True,
        )
    )

    payload = captured["payload"]  # type: ignore[assignment]
    assert captured["compact"] is True
    assert payload["decision"] == "EXIT"  # type: ignore[index]
    assert payload["opens_orders"] is False  # type: ignore[index]
    assert payload["writes_execution_config"] is False  # type: ignore[index]
    assert payload["decision_report_path"] == "state/trade-decisions/exit.json"  # type: ignore[index]
