from __future__ import annotations

from argparse import Namespace

import binance_quant_control.cli as cli


def test_cmd_operator_dashboard_compact_summarizes_execution_journal(monkeypatch) -> None:
    captured: dict[str, object] = {}
    payload = {
        "status": "ok",
        "mode": {"use_testnet": True, "live_trading_enabled": False, "testnet_trading_enabled": True},
        "customer_summary": {
            "open_position_count": 0,
            "live_order_count": 21,
            "live_order_count_meaning": "append_only_live_testnet_order_journal_records_not_current_open_orders",
        },
        "positions": [],
        "protective_orders": [],
        "execution_journal": {
            "record_count": 21,
            "buy_count": 21,
            "sell_count": 0,
            "latest": {
                "timestamp": "2026-05-05T05:23:04+00:00",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "status": "NEW",
                "route_id": "btc-core",
                "simulation_mode": "demo_testnet",
                "binance_response": {"entry": {"orderId": 123}},
            },
            "meaning": "append-only order audit history; current exposure comes from positions/protective_orders",
        },
        "product_readiness": {
            "status": "blocked",
            "stage": "watch_only_research",
            "mainnet_customer_ready": False,
        },
        "decision_artifact_audit": {
            "status": "passed",
            "scope": "since_contract",
            "summary": {"artifact_count": 2, "invalid_count": 0},
            "invalid_artifacts": [],
            "report_path": "state/operator-dashboard/decision-audit/report.json",
        },
            "risk_combo_matrix": {
                "available": True,
                "status": "promising_research_only",
                "promising_surface_count": 2,
                "robust_surface_count": 0,
                "objective_scorecard": {
                    "buy_sell_stability": {"score": 66, "baseline_score": 45},
                    "long_term_expectancy": {"score": 36, "baseline_score": 20},
                    "live_readiness": {"score": 0, "required_score_until_promotion": 0},
                },
                "prompt_to_artifact_checklist": {
                    "status": "incomplete",
                    "missing_requirements": ["buy_and_sell_directional_evidence"],
                    "items": [{"requirement": "candidate_not_zero", "passed": True}],
                    "mainnet_live_allowed": False,
                },
                "risk_boundary": {
                    "max_per_trade_risk_pct": 0.025,
                    "writes_execution_config": False,
                    "mainnet_live_allowed": False,
                },
                "side_summary": {
                "buy": {"promising_surface_count": 2, "robust_surface_count": 0},
                "sell": {"promising_surface_count": 0, "robust_surface_count": 0},
            },
            "horizon_summary": {
                "short": {"promising_surface_count": 0},
                "medium": {"promising_surface_count": 1},
                "long": {"promising_surface_count": 1},
            },
            "completion_audit": {
                "status": "incomplete",
                "missing_requirements": ["robust_promotion_gate_passed"],
                "mainnet_live_allowed": False,
            },
            "mainnet_live_allowed": False,
            "best_surface": {
                "surface": "buy_4h",
                "target_side": "BUY",
                "target_interval": "4h",
                "full": {"profit_factor": 1.262, "expectancy_r": 0.0405},
            },
            "validation_plan": [
                {
                    "surface": "buy_4h",
                    "interactive_probe_command": "openclaw-quantctl risk-combo-sweep --target-side BUY --target-interval 4h --compact",
                }
            ],
            "negative_surface_repair_plan": [
                {
                    "coverage_type": "side",
                    "coverage_key": "sell",
                    "source_surface": "sell_15m",
                    "scout_command": "openclaw-quantctl risk-combo-sweep --target-side SELL --target-interval 15m --limit 600 --compact",
                    "cross_symbol_scout_command": "openclaw-quantctl risk-combo-sweep --symbols TRXUSDT,ADAUSDT --target-side SELL --target-interval 15m --compact",
                    "cross_symbol_scout_symbols": ["TRXUSDT", "ADAUSDT"],
                        "interactive_probe_command": "openclaw-quantctl risk-combo-sweep --target-side SELL --target-interval 15m --compact",
                        "runtime_guidance": "Run scout_command first during chat.",
                        "guardrails": {
                            "mainnet_live_allowed": False,
                            "does_not_lower_promotion_gates": True,
                            "max_per_trade_risk_pct": 0.025,
                        },
                    }
                ],
            "promotion_boundary": {"mainnet_live_allowed": False},
        },
        "loss_diagnostics": {"summary": {}, "findings": [], "root_cause_recommendations": []},
        "external_context_automation": {},
        "operator_feedback": [],
        "report_path": "state/operator-dashboard/report.json",
    }

    monkeypatch.setattr(cli, "load_settings", lambda: object())
    monkeypatch.setattr(cli, "build_operator_dashboard", lambda settings, min_bucket_trades, top_n: payload)
    monkeypatch.setattr(
        cli,
        "print_json",
        lambda output, compact=False: captured.update({"payload": output, "compact": compact}),
    )

    cli.cmd_operator_dashboard(Namespace(min_bucket_trades=10, top_n=5, compact=True))

    output = captured["payload"]  # type: ignore[assignment]
    assert captured["compact"] is True
    latest = output["execution_journal"]["latest"]  # type: ignore[index]
    assert latest == {
        "timestamp": "2026-05-05T05:23:04+00:00",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "status": "NEW",
        "route_id": "btc-core",
        "simulation_mode": "demo_testnet",
    }
    assert "binance_response" not in latest
    assert output["product_readiness"]["status"] == "blocked"  # type: ignore[index]
    assert output["decision_artifact_audit"]["status"] == "passed"  # type: ignore[index]
    assert output["risk_combo_matrix"]["promising_surface_count"] == 2  # type: ignore[index]
    assert output["risk_combo_matrix"]["robust_surface_count"] == 0  # type: ignore[index]
    assert output["risk_combo_matrix"]["mainnet_live_allowed"] is False  # type: ignore[index]
    assert output["risk_combo_matrix"]["best_surface"]["surface"] == "buy_4h"  # type: ignore[index]
    assert output["risk_combo_matrix"]["side_summary"]["sell"]["promising_surface_count"] == 0  # type: ignore[index]
    assert output["risk_combo_matrix"]["horizon_summary"]["long"]["promising_surface_count"] == 1  # type: ignore[index]
    assert output["risk_combo_matrix"]["completion_audit"]["status"] == "incomplete"  # type: ignore[index]
    assert output["risk_combo_matrix"]["completion_audit"]["mainnet_live_allowed"] is False  # type: ignore[index]
    assert output["risk_combo_matrix"]["objective_scorecard"]["buy_sell_stability"]["score"] == 66  # type: ignore[index]
    assert output["risk_combo_matrix"]["objective_scorecard"]["long_term_expectancy"]["score"] == 36  # type: ignore[index]
    assert output["risk_combo_matrix"]["objective_scorecard"]["live_readiness"]["score"] == 0  # type: ignore[index]
    assert output["risk_combo_matrix"]["prompt_to_artifact_checklist"]["status"] == "incomplete"  # type: ignore[index]
    assert "buy_and_sell_directional_evidence" in output["risk_combo_matrix"]["prompt_to_artifact_checklist"]["missing_requirements"]  # type: ignore[index]
    assert output["risk_combo_matrix"]["risk_boundary"]["max_per_trade_risk_pct"] == 0.025  # type: ignore[index]
    assert output["risk_combo_matrix"]["negative_surface_repair_plan"][0]["coverage_key"] == "sell"  # type: ignore[index]
    assert "ADAUSDT" in output["risk_combo_matrix"]["negative_surface_repair_plan"][0]["cross_symbol_scout_symbols"]  # type: ignore[index]
    assert output["risk_combo_matrix"]["negative_surface_repair_plan"][0]["guardrails"]["max_per_trade_risk_pct"] == 0.025  # type: ignore[index]
    assert output["risk_combo_matrix"]["negative_surface_repair_plan"][0]["guardrails"]["mainnet_live_allowed"] is False  # type: ignore[index]


def test_cmd_risk_combo_matrix_accepts_latest_sweeps_without_explicit_reports(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_build_risk_combo_matrix_report(*, report_paths, output_dir, latest_sweeps):
        captured["call"] = {
            "report_paths": report_paths,
            "output_dir": output_dir,
            "latest_sweeps": latest_sweeps,
        }
        return {
            "mode": "risk_combo_side_interval_matrix_v1",
            "safety": {"mainnet_live_allowed": False},
            "input_report_count": 3,
            "skipped_input_report_count": 0,
            "surface_count": 2,
            "promising_surface_count": 1,
            "robust_surface_count": 0,
            "report_path": "state/risk-combo-matrix/report.json",
        }

    monkeypatch.setattr(cli, "build_risk_combo_matrix_report", fake_build_risk_combo_matrix_report)
    monkeypatch.setattr(
        cli,
        "print_json",
        lambda output, compact=False: captured.update({"payload": output, "compact": compact}),
    )

    cli.cmd_risk_combo_matrix(
        Namespace(
            sweep_report=None,
            latest_sweeps=12,
            output_dir="",
            compact=True,
        )
    )

    assert captured["call"] == {
        "report_paths": [],
        "output_dir": None,
        "latest_sweeps": 12,
    }
    assert captured["compact"] is True
    assert captured["payload"]["input_report_count"] == 3  # type: ignore[index]
    assert captured["payload"]["safety"]["mainnet_live_allowed"] is False  # type: ignore[index]
