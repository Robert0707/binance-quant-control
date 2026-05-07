from __future__ import annotations

from types import SimpleNamespace

import binance_quant_control.ai_goal_loop as goal_loop


def test_ai_goal_loop_routes_blocked_readiness_to_sizing_and_research_actions(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(goal_loop, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        goal_loop,
        "load_settings",
        lambda: SimpleNamespace(use_testnet=True, live_trading_enabled=False),
    )
    monkeypatch.setattr(
        goal_loop,
        "run_ai_surface_audit",
        lambda **_kwargs: {"status": "passed", "blocker_count": 0, "report_path": "audit.json"},
    )
    monkeypatch.setattr(
        goal_loop,
        "run_ai_expectancy_upgrade",
        lambda **_kwargs: {
            "maturity_score": {"score_10": 8.4, "target_for_9_plus": {"missing_points": 6.0}},
            "final_machine_decision": {"next_surface": "continue_expectancy_research"},
            "readiness_scan": {"allowed_count": 0},
            "report_path": "upgrade.json",
        },
    )
    monkeypatch.setattr(
        goal_loop,
        "run_ai_readiness_scan",
        lambda **_kwargs: {
            "candidate_count": 6,
            "allowed_count": 0,
            "next_machine_action": "repair_exchange_sizing_or_margin",
            "hard_blocker_taxonomy": {
                "exchange_constraints": ["Order notional below exchange minimum."],
                "strategy_performance": ["Recent PF below floor."],
            },
            "report_path": "readiness.json",
        },
    )
    monkeypatch.setattr(
        goal_loop,
        "summarize_closed_trade_reviews",
        lambda: {"count": 8, "profit_factor": 0.7, "expectancy": {"expectancy_r": -0.1}},
    )

    payload = goal_loop.run_ai_goal_loop(
        output_dir=tmp_path,
        goal="maximize_stable_expectancy",
        smoke=True,
    )

    assert payload["mode"] == "ai_goal_loop_v1"
    assert payload["safety"]["opens_orders"] is False
    assert payload["safety"]["mainnet_live_allowed"] is False
    assert payload["score"]["score_10"] < 9
    assert payload["next_machine_action"] == "repair_exchange_sizing_or_margin"
    assert "repair_exchange_sizing_or_margin" in payload["recommended_commands"][0]
    assert "run_full_expectancy_upgrade" in payload["recommended_commands"]
    assert payload["closed_trade_feedback"]["sample_status"] == "insufficient_forward_evidence"


def test_readiness_sizing_scout_returns_first_margin_without_exchange_blockers(monkeypatch, tmp_path) -> None:
    calls: list[float | None] = []

    def fake_readiness(**kwargs):
        margin = kwargs.get("margin_notional_usdt")
        calls.append(margin)
        if margin == 50.0:
            return {
                "candidate_count": 2,
                "allowed_count": 0,
                "hard_blocker_taxonomy": {"strategy_performance": ["Recent PF below floor."]},
                "report_path": f"{margin}.json",
            }
        return {
            "candidate_count": 2,
            "allowed_count": 0,
            "hard_blocker_taxonomy": {"exchange_constraints": ["Order notional below exchange minimum."]},
            "report_path": f"{margin}.json",
        }

    monkeypatch.setattr(goal_loop, "run_ai_readiness_scan", fake_readiness)

    payload = goal_loop.run_readiness_sizing_scout(
        output_dir=tmp_path,
        margin_candidates=[25.0, 50.0, 100.0],
        max_candidates=2,
    )

    assert calls == [25.0, 50.0]
    assert payload["selected_margin_notional_usdt"] == 50.0
    assert payload["status"] == "sizing_constraints_clear"
    assert payload["opens_orders"] is False


def test_ai_goal_loop_routes_allowed_readiness_to_testnet_forward_evidence(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(goal_loop, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        goal_loop,
        "load_settings",
        lambda: SimpleNamespace(use_testnet=True, live_trading_enabled=False),
    )
    monkeypatch.setattr(
        goal_loop,
        "run_ai_surface_audit",
        lambda **_kwargs: {"status": "passed", "blocker_count": 0, "report_path": "audit.json"},
    )
    monkeypatch.setattr(
        goal_loop,
        "run_ai_expectancy_upgrade",
        lambda **_kwargs: {
            "maturity_score": {"score_10": 9.1, "target_for_9_plus": {"missing_points": 0.0}},
            "final_machine_decision": {"next_surface": "testnet_forward_evidence"},
            "readiness_scan": {"allowed_count": 1},
            "report_path": "upgrade.json",
        },
    )
    monkeypatch.setattr(
        goal_loop,
        "run_ai_readiness_scan",
        lambda **_kwargs: {
            "candidate_count": 3,
            "allowed_count": 1,
            "next_machine_action": "operator_execute_testnet_ticket",
            "execution_ticket": {"operator_testnet_execute_command": "openclaw-quantctl live-pilot --execute --compact"},
            "hard_blocker_taxonomy": {},
            "report_path": "readiness.json",
        },
    )
    monkeypatch.setattr(
        goal_loop,
        "summarize_closed_trade_reviews",
        lambda: {"count": 42, "profit_factor": 1.2, "expectancy": {"expectancy_r": 0.05}},
    )

    payload = goal_loop.run_ai_goal_loop(
        output_dir=tmp_path,
        goal="maximize_stable_expectancy",
        smoke=True,
    )

    assert payload["next_machine_action"] == "start_testnet_forward_evidence"
    assert payload["recommended_commands"][0] == "openclaw-quantctl hermes-trade cycle --force --compact"
    assert payload["closed_trade_feedback"]["sample_status"] == "forward_evidence_ready"
