from __future__ import annotations

import json
import os
from pathlib import Path

import binance_quant_control.operator_dashboard as dashboard
from binance_quant_control.config import Settings


def _settings() -> Settings:
    return Settings(
        use_testnet=True,
        live_trading_enabled=False,
        testnet_trading_enabled=True,
        recv_window_ms=5000,
        default_symbol="BTCUSDT",
        default_market="futures",
        binance_api_key="",
        binance_secret_key="",
        binance_testnet_api_key="key",
        binance_testnet_secret_key="secret",
        blave_api_key="",
        blave_secret_key="",
        whale_alert_api_key="",
        max_leverage=5,
        max_notional_pct=0.5,
        max_daily_trades=5,
        min_balance_usdt=2.0,
        min_convergence=0.6,
        cooldown_hours=4.0,
    )


class FakeClient:
    def __init__(self, settings):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def positions(self):
        return [
            {
                "symbol": "APTUSDT",
                "positionAmt": "30",
                "entryPrice": "1.0",
                "markPrice": "1.01",
                "unRealizedProfit": "0.3",
                "leverage": "5",
            }
        ]

    def open_algo_orders(self, symbol):
        return [
            {"orderType": "STOP_MARKET", "quantity": "30", "triggerPrice": "0.95"},
            {"orderType": "TAKE_PROFIT_MARKET", "quantity": "9", "triggerPrice": "1.03"},
            {"orderType": "TAKE_PROFIT_MARKET", "quantity": "16.5", "triggerPrice": "1.08"},
        ]

    def exchange_info(self, symbol, market):
        return {
            "symbols": [
                {
                    "symbol": symbol,
                    "filters": [
                        {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.1"},
                    ],
                }
            ]
        }


def _fake_decision_audit(tmp_path: Path) -> dict[str, object]:
    return {
        "status": "passed",
        "scope": "since_contract",
        "summary": {
            "artifact_count": 2,
            "valid_count": 2,
            "invalid_count": 0,
            "decision_counts": {"HOLD": 2},
            "opens_orders_false_count": 2,
            "writes_execution_config_false_count": 2,
        },
        "invalid_artifacts": [],
        "report_path": str(tmp_path / "decision-audit.json"),
    }


def test_operator_dashboard_builds_customer_feedback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dashboard, "OPERATOR_DASHBOARD_DIR", tmp_path)
    monkeypatch.setattr(dashboard, "RISK_COMBO_MATRIX_DIR", tmp_path / "risk-combo-matrix")
    monkeypatch.setattr(dashboard, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(dashboard, "BinanceClient", lambda settings: FakeClient(settings))
    monkeypatch.setattr(
        dashboard,
        "summarize_closed_trade_reviews",
        lambda: {"count": 12, "total_realized_pnl_usdt": -2.5},
    )
    monkeypatch.setattr(dashboard, "summarize_live_orders", lambda: {"count": 3})
    monkeypatch.setattr(
        dashboard,
        "read_live_orders",
        lambda: [
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "symbol": "APTUSDT",
                "side": "BUY",
                "leverage": 5,
                "notional_usdt": 6.0,
                "gross_notional_usdt": 30.0,
                "route_id": "major-alt-trend",
                "analysis_score": 100,
                "analysis_convergence": 0.9,
            }
        ],
    )
    monkeypatch.setattr(
        dashboard,
        "run_loss_diagnostics",
        lambda min_bucket_trades, top_n: {
            "summary": {"count": 12, "profit_factor": 0.7},
            "findings": [
                "stop-loss-dominant: stop_loss_ratio=70.0%",
                "fast-stop-cluster:major-alt-trend/BUY/hold-lt-1h stop_loss_ratio=80.0% PF=0.4",
            ],
            "worst_buckets": [],
            "root_cause_recommendations": [
                {"type": "fast-stop-cluster", "action": "require-stronger-entry-confirmation"}
            ],
            "report_path": str(tmp_path / "loss.json"),
        },
    )
    monkeypatch.setattr(
        dashboard,
        "_load_latest_digest_summary",
        lambda: {
            "available": True,
            "news": {"risk": "high", "bias": "bearish", "high_impact_count": 2},
            "whale": {"enabled": False, "available": False, "reason": "WHALE_ALERT_API_KEY not configured"},
            "decision": {"action": "watchlist_only", "selected": {"symbol": "BTCUSDT"}},
        },
    )
    monkeypatch.setattr(
        dashboard,
        "run_decision_audit",
        lambda **kwargs: _fake_decision_audit(tmp_path) | {"kwargs": kwargs},
    )

    payload = dashboard.build_operator_dashboard(_settings())

    assert payload["customer_summary"]["open_position_count"] == 1
    assert payload["customer_summary"]["open_unrealized_pnl_usdt"] == 0.3
    assert (
        payload["customer_summary"]["live_order_count_meaning"]
        == "append_only_live_testnet_order_journal_records_not_current_open_orders"
    )
    assert payload["execution_journal"]["record_count"] == 3
    assert "current exposure" in payload["execution_journal"]["meaning"]
    assert payload["product_readiness"]["status"] == "blocked"
    assert payload["product_readiness"]["stage"] == "watch_only_research"
    assert payload["product_readiness"]["mainnet_customer_ready"] is False
    assert any(
        str(item).startswith("profit_factor_below_breakeven")
        for item in payload["product_readiness"]["blockers"]
    )
    assert payload["protective_orders"][0]["coverage"] == "ok"
    assert payload["protective_orders"][0]["take_profit_ladder"]["quantities"] == [9.0, 16.5]
    assert payload["protective_orders"][0]["take_profit_ladder"]["first_tp_ratio"] == 0.3
    assert payload["protective_orders"][0]["take_profit_ladder"]["runner_quantity"] == 4.5
    assert any("profitable" in item for item in payload["operator_feedback"])
    assert any("whale wallet" in item for item in payload["operator_feedback"])
    assert any("stopped quickly" in item for item in payload["operator_feedback"])
    assert payload["external_context_automation"]["available"] is True
    assert payload["decision_artifact_audit"]["status"] == "passed"
    assert payload["decision_artifact_audit"]["summary"]["invalid_count"] == 0
    assert payload["risk_combo_matrix"]["available"] is False
    assert payload["risk_combo_matrix"]["mainnet_live_allowed"] is False
    assert payload["loss_diagnostics"]["root_cause_recommendations"][0]["type"] == "fast-stop-cluster"
    assert Path(payload["report_path"]).exists()


def test_operator_dashboard_explains_flat_position_with_historical_order_journal(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FlatClient(FakeClient):
        def positions(self):
            return []

    monkeypatch.setattr(dashboard, "OPERATOR_DASHBOARD_DIR", tmp_path)
    monkeypatch.setattr(dashboard, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(dashboard, "BinanceClient", lambda settings: FlatClient(settings))
    monkeypatch.setattr(
        dashboard,
        "summarize_closed_trade_reviews",
        lambda: {"count": 12, "total_realized_pnl_usdt": -2.5},
    )
    monkeypatch.setattr(
        dashboard,
        "summarize_live_orders",
        lambda: {
            "count": 21,
            "buy_count": 11,
            "sell_count": 10,
            "latest": {"symbol": "BTCUSDT", "side": "BUY"},
        },
    )
    monkeypatch.setattr(dashboard, "read_live_orders", lambda: [])
    monkeypatch.setattr(
        dashboard,
        "run_loss_diagnostics",
        lambda min_bucket_trades, top_n: {
            "summary": {"count": 12, "profit_factor": 0.7},
            "findings": ["stop-loss-dominant: stop_loss_ratio=70.0%"],
            "worst_buckets": [],
            "root_cause_recommendations": [],
            "report_path": str(tmp_path / "loss.json"),
        },
    )
    monkeypatch.setattr(dashboard, "_load_latest_digest_summary", lambda: {"available": False})
    monkeypatch.setattr(dashboard, "run_decision_audit", lambda **kwargs: _fake_decision_audit(tmp_path))

    payload = dashboard.build_operator_dashboard(_settings())

    assert payload["customer_summary"]["open_position_count"] == 0
    assert payload["customer_summary"]["live_order_count"] == 21
    assert payload["execution_journal"]["record_count"] == 21
    assert any("journal count as audit history" in item for item in payload["operator_feedback"])


def test_operator_dashboard_marks_strong_testnet_evidence_conditional(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FlatClient(FakeClient):
        def positions(self):
            return []

    monkeypatch.setattr(dashboard, "OPERATOR_DASHBOARD_DIR", tmp_path)
    monkeypatch.setattr(dashboard, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(dashboard, "BinanceClient", lambda settings: FlatClient(settings))
    monkeypatch.setattr(
        dashboard,
        "summarize_closed_trade_reviews",
        lambda: {"count": 140, "total_realized_pnl_usdt": 7.5, "profit_factor": 1.18},
    )
    monkeypatch.setattr(dashboard, "summarize_live_orders", lambda: {"count": 0})
    monkeypatch.setattr(dashboard, "read_live_orders", lambda: [])
    monkeypatch.setattr(
        dashboard,
        "run_loss_diagnostics",
        lambda min_bucket_trades, top_n: {
            "summary": {"count": 140, "profit_factor": 1.18, "avg_r": 0.04, "stop_loss_ratio": 54.0},
            "findings": [],
            "worst_buckets": [],
            "root_cause_recommendations": [],
            "report_path": str(tmp_path / "loss.json"),
        },
    )
    monkeypatch.setattr(dashboard, "_load_latest_digest_summary", lambda: {"available": False})
    monkeypatch.setattr(dashboard, "run_decision_audit", lambda **kwargs: _fake_decision_audit(tmp_path))

    payload = dashboard.build_operator_dashboard(_settings())

    assert payload["product_readiness"]["status"] == "conditional"
    assert payload["product_readiness"]["stage"] == "testnet_exploration_only"
    assert payload["product_readiness"]["testnet_trade_ready"] is True
    assert payload["product_readiness"]["mainnet_customer_ready"] is False
    assert "mainnet_live_trading_disabled" in payload["product_readiness"]["blockers"]


class BadTpClient(FakeClient):
    def open_algo_orders(self, symbol):
        return [
            {"orderType": "STOP_MARKET", "quantity": "30", "triggerPrice": "0.95"},
            {"orderType": "TAKE_PROFIT_MARKET", "quantity": "30", "triggerPrice": "1.03"},
        ]


def test_operator_dashboard_flags_full_position_tp1(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dashboard, "OPERATOR_DASHBOARD_DIR", tmp_path)
    monkeypatch.setattr(dashboard, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(dashboard, "BinanceClient", lambda settings: BadTpClient(settings))
    monkeypatch.setattr(
        dashboard,
        "summarize_closed_trade_reviews",
        lambda: {"count": 12, "total_realized_pnl_usdt": -2.5},
    )
    monkeypatch.setattr(dashboard, "summarize_live_orders", lambda: {"count": 3})
    monkeypatch.setattr(dashboard, "read_live_orders", lambda: [])
    monkeypatch.setattr(
        dashboard,
        "run_loss_diagnostics",
        lambda min_bucket_trades, top_n: {
            "summary": {"count": 12, "profit_factor": 0.7},
            "findings": ["stop-loss-dominant: stop_loss_ratio=70.0%"],
            "worst_buckets": [],
            "root_cause_recommendations": [],
            "report_path": str(tmp_path / "loss.json"),
        },
    )
    monkeypatch.setattr(dashboard, "_load_latest_digest_summary", lambda: {"available": False})
    monkeypatch.setattr(dashboard, "run_decision_audit", lambda **kwargs: _fake_decision_audit(tmp_path))

    payload = dashboard.build_operator_dashboard(_settings())

    ladder = payload["protective_orders"][0]["take_profit_ladder"]
    assert payload["protective_orders"][0]["coverage"] == "attention"
    assert ladder["status"] == "attention"
    assert "tp1_full_position" in ladder["issues"]


class MicroFullTpClient(FakeClient):
    def positions(self):
        return [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.001",
                "entryPrice": "80883.5",
                "markPrice": "81000.0",
                "unRealizedProfit": "0.1165",
                "leverage": "3",
            }
        ]

    def open_algo_orders(self, symbol):
        return [
            {"orderType": "STOP_MARKET", "quantity": "0.001", "triggerPrice": "79536.5"},
            {"orderType": "TAKE_PROFIT_MARKET", "quantity": "0.001", "triggerPrice": "86655.9"},
        ]

    def exchange_info(self, symbol, market):
        return {
            "symbols": [
                {
                    "symbol": symbol,
                    "filters": [
                        {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.001"},
                    ],
                }
            ]
        }


def test_operator_dashboard_allows_full_tp_for_minimum_step_micro_position(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dashboard, "OPERATOR_DASHBOARD_DIR", tmp_path)
    monkeypatch.setattr(dashboard, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(dashboard, "BinanceClient", lambda settings: MicroFullTpClient(settings))
    monkeypatch.setattr(
        dashboard,
        "summarize_closed_trade_reviews",
        lambda: {"count": 12, "total_realized_pnl_usdt": -2.5},
    )
    monkeypatch.setattr(dashboard, "summarize_live_orders", lambda: {"count": 3})
    monkeypatch.setattr(dashboard, "read_live_orders", lambda: [])
    monkeypatch.setattr(
        dashboard,
        "run_loss_diagnostics",
        lambda min_bucket_trades, top_n: {
            "summary": {"count": 12, "profit_factor": 0.7},
            "findings": ["stop-loss-dominant: stop_loss_ratio=70.0%"],
            "worst_buckets": [],
            "root_cause_recommendations": [],
            "report_path": str(tmp_path / "loss.json"),
        },
    )
    monkeypatch.setattr(dashboard, "_load_latest_digest_summary", lambda: {"available": False})
    monkeypatch.setattr(dashboard, "run_decision_audit", lambda **kwargs: _fake_decision_audit(tmp_path))

    payload = dashboard.build_operator_dashboard(_settings())

    ladder = payload["protective_orders"][0]["take_profit_ladder"]
    assert payload["protective_orders"][0]["coverage"] == "ok"
    assert ladder["micro_full_tp_fallback"] is True
    assert ladder["issues"] == []


def test_operator_dashboard_embeds_latest_risk_combo_matrix_summary(monkeypatch, tmp_path: Path) -> None:
    class FlatClient(FakeClient):
        def positions(self):
            return []

    matrix_dir = tmp_path / "risk-combo-matrix"
    sweep_dir = tmp_path / "risk-combo-sweeps"
    matrix_dir.mkdir()
    sweep_dir.mkdir()
    matrix_path = matrix_dir / "20260510T080706Z-risk-combo-matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-10T08:07:06+00:00",
                "mode": "risk_combo_side_interval_matrix_v1",
                "safety": {
                    "opens_orders": False,
                    "writes_execution_config": False,
                    "clears_route_quarantine": False,
                    "mainnet_live_allowed": False,
                },
                "surface_count": 6,
                "promising_surface_count": 2,
                "robust_surface_count": 0,
                "side_summary": {
                    "buy": {
                        "surface_count": 3,
                        "promising_surface_count": 2,
                        "robust_surface_count": 0,
                        "status": "promising_research_only",
                    },
                    "sell": {
                        "surface_count": 3,
                        "promising_surface_count": 0,
                        "robust_surface_count": 0,
                        "status": "missing_or_negative_expectancy",
                    },
                },
                "horizon_summary": {
                    "short": {
                        "surface_count": 2,
                        "promising_surface_count": 0,
                        "robust_surface_count": 0,
                        "status": "missing_or_negative_expectancy",
                    },
                    "medium": {
                        "surface_count": 2,
                        "promising_surface_count": 1,
                        "robust_surface_count": 0,
                        "status": "promising_research_only",
                    },
                    "long": {
                        "surface_count": 2,
                        "promising_surface_count": 1,
                        "robust_surface_count": 0,
                        "status": "promising_research_only",
                    },
                },
                "completion_audit": {
                    "status": "incomplete",
                    "missing_requirements": [
                        "buy_and_sell_have_backtested_promising_surfaces",
                        "short_medium_long_have_promising_surfaces",
                        "robust_promotion_gate_passed",
                    ],
                    "mainnet_live_allowed": False,
                },
                "objective_scorecard": {
                    "buy_sell_stability": {"score": 66, "baseline_score": 45},
                    "long_term_expectancy": {"score": 36, "baseline_score": 20},
                    "live_readiness": {"score": 0, "required_score_until_promotion": 0},
                },
                "prompt_to_artifact_checklist": {
                    "status": "incomplete",
                    "missing_requirements": ["buy_and_sell_directional_evidence"],
                    "items": [
                        {"requirement": "candidate_not_zero", "passed": True},
                        {"requirement": "buy_and_sell_directional_evidence", "passed": False},
                    ],
                    "mainnet_live_allowed": False,
                },
                "best_surface": {
                    "surface": "buy_4h",
                    "target_side": "BUY",
                    "target_interval": "4h",
                    "route_id": "trx-mean-reversion",
                    "symbol": "TRXUSDT",
                    "research_status": "promising_but_under_validated",
                    "recovery_gate_passed": False,
                    "robust_recovery_gate_passed": False,
                    "full": {
                        "trade_count": 27,
                        "profit_factor": 1.262,
                        "expectancy_r": 0.0405,
                        "max_drawdown_pct": 0.7662,
                        "stop_loss_ratio": 44.44,
                    },
                    "test": {"trade_count": 5, "profit_factor": 46.9305, "expectancy_r": 0.2863},
                    "walk_forward": {
                        "window_count": 3,
                        "positive_expectancy_window_count": 2,
                        "min_profit_factor": 0.0311,
                        "min_expectancy_r": -0.2675,
                    },
                    "gate_reasons": {
                        "recovery": ["test-trade-count-too-low"],
                        "robust": [
                            "test-trade-count-too-low",
                            "walk-forward-min-profit-factor-below-target",
                        ],
                    },
                    "source_report_path": "state/risk-combo-sweeps/example.json",
                },
                "validation_plan": [
                    {
                        "surface": "buy_4h",
                        "symbol": "TRXUSDT",
                        "target_side": "BUY",
                        "target_interval": "4h",
                        "purpose": "interactive_probe_then_offline_validation_before_promotion",
                        "interactive_probe_command": "openclaw-quantctl risk-combo-sweep --symbols TRXUSDT --target-side BUY --target-interval 4h --compact",
                        "offline_validation_command": "openclaw-quantctl risk-combo-sweep --symbols TRXUSDT --target-side BUY --target-interval 4h --limit 5000 --compact",
                        "runtime_guidance": "Run interactive_probe_command during chat only.",
                        "promotion_boundary": "research_only_does_not_change_live_readiness_or_mainnet_permission",
                    },
                    {
                        "surface": "buy_1d",
                        "symbol": "TRXUSDT",
                        "target_side": "BUY",
                        "target_interval": "1d",
                    },
                    {
                        "surface": "sell_4h",
                        "symbol": "TRXUSDT",
                        "target_side": "SELL",
                        "target_interval": "4h",
                    },
                ],
                "negative_surface_repair_plan": [
                    {
                        "coverage_type": "side",
                        "coverage_key": "sell",
                        "status": "missing_or_negative_expectancy",
                        "source_surface": "sell_15m",
                        "target_side": "SELL",
                        "target_interval": "15m",
                        "current_metrics": {
                            "trade_count": 5,
                            "profit_factor": 0.1644,
                            "expectancy_r": -0.1207,
                            "max_drawdown_pct": 0.4325,
                            "stop_loss_ratio": 80.0,
                        },
                        "failure_reasons": ["full-profit-factor-below-target"],
                        "repair_objective": "find_positive_expectancy_without_relaxing_recovery_or_live_gates",
                        "scout_command": "openclaw-quantctl risk-combo-sweep --symbols TRXUSDT --target-side SELL --target-interval 15m --limit 600 --compact",
                        "cross_symbol_scout_command": "openclaw-quantctl risk-combo-sweep --symbols TRXUSDT,ADAUSDT --target-side SELL --target-interval 15m --compact",
                        "cross_symbol_scout_symbols": ["TRXUSDT", "ADAUSDT"],
                        "interactive_probe_command": "openclaw-quantctl risk-combo-sweep --symbols TRXUSDT --target-side SELL --target-interval 15m --compact",
                        "offline_validation_command": "openclaw-quantctl risk-combo-sweep --symbols TRXUSDT --target-side SELL --target-interval 15m --limit 5000 --compact",
                        "runtime_guidance": "Run scout_command first during chat.",
                        "guardrails": {
                            "does_not_open_orders": True,
                            "does_not_write_execution_config": True,
                            "does_not_clear_route_quarantine": True,
                            "does_not_lower_promotion_gates": True,
                            "mainnet_live_allowed": False,
                        },
                    }
                ],
                "next_research_actions": [
                    "expand_sample_and_walk_forward_for_promising_surfaces",
                    "keep_mainnet_blocked_until_robust_gate_passes",
                ],
                "promotion_boundary": {
                    "requires_robust_recovery_gate": True,
                    "requires_sufficient_test_trades": True,
                    "mainnet_live_allowed": False,
                },
            }
        ),
        encoding="utf-8",
    )
    sweep_path = sweep_dir / "20260510T080000Z-risk-combo-sweep.json"
    sweep_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    os.utime(sweep_path, (1_778_378_400, 1_778_378_400))
    os.utime(matrix_path, (1_778_378_800, 1_778_378_800))
    readiness_dir = tmp_path / "hermes-readiness-scan"
    readiness_dir.mkdir()
    readiness_path = readiness_dir / "20260510T080500Z-hermes-readiness-scan.json"
    readiness_path.write_text(
        json.dumps(
            {
                "mode": "hermes_ai_readiness_scanner_v1",
                "allowed_count": 0,
                "research_candidate_report": {
                    "reviewable_horizon_counts": {"medium": 1},
                    "top_candidates": [
                        {
                            "symbol": "TRXUSDT",
                            "side": "BUY",
                            "interval": "4h",
                            "route_id": "trx-mean-reversion",
                            "horizon": "medium",
                            "research_status": "reviewable_signal",
                            "trade_readiness_allowed": False,
                            "readiness_next_action": "expand_walk_forward_and_readiness_validation",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(dashboard, "OPERATOR_DASHBOARD_DIR", tmp_path / "operator-dashboard")
    monkeypatch.setattr(dashboard, "RISK_COMBO_MATRIX_DIR", matrix_dir)
    monkeypatch.setattr(dashboard, "RISK_COMBO_SWEEP_DIR", sweep_dir)
    monkeypatch.setattr(dashboard, "READINESS_SCAN_DIR", readiness_dir)
    monkeypatch.setattr(dashboard, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(dashboard, "BinanceClient", lambda settings: FlatClient(settings))
    monkeypatch.setattr(
        dashboard,
        "summarize_closed_trade_reviews",
        lambda: {"count": 12, "total_realized_pnl_usdt": -2.5},
    )
    monkeypatch.setattr(dashboard, "summarize_live_orders", lambda: {"count": 0})
    monkeypatch.setattr(dashboard, "read_live_orders", lambda: [])
    monkeypatch.setattr(
        dashboard,
        "run_loss_diagnostics",
        lambda min_bucket_trades, top_n: {
            "summary": {"count": 12, "profit_factor": 0.7},
            "findings": [],
            "worst_buckets": [],
            "root_cause_recommendations": [],
            "report_path": str(tmp_path / "loss.json"),
        },
    )
    monkeypatch.setattr(dashboard, "_load_latest_digest_summary", lambda: {"available": False})
    monkeypatch.setattr(dashboard, "run_decision_audit", lambda **kwargs: _fake_decision_audit(tmp_path))

    payload = dashboard.build_operator_dashboard(_settings())

    matrix = payload["risk_combo_matrix"]
    assert matrix["available"] is True
    assert matrix["status"] == "promising_research_only"
    assert matrix["path"] == str(matrix_path)
    assert matrix["freshness"]["status"] == "current"
    assert matrix["freshness"]["latest_sweep_path"] == str(sweep_path)
    assert matrix["promising_surface_count"] == 2
    assert matrix["robust_surface_count"] == 0
    assert matrix["side_summary"]["buy"]["promising_surface_count"] == 2
    assert matrix["side_summary"]["sell"]["status"] == "missing_or_negative_expectancy"
    assert matrix["horizon_summary"]["long"]["promising_surface_count"] == 1
    assert matrix["completion_audit"]["status"] == "incomplete"
    assert "robust_promotion_gate_passed" in matrix["completion_audit"]["missing_requirements"]
    assert matrix["completion_audit"]["mainnet_live_allowed"] is False
    assert matrix["objective_scorecard"]["buy_sell_stability"]["score"] == 66
    assert matrix["objective_scorecard"]["long_term_expectancy"]["score"] == 36
    assert matrix["objective_scorecard"]["live_readiness"]["score"] == 0
    assert matrix["prompt_to_artifact_checklist"]["status"] == "incomplete"
    assert "buy_and_sell_directional_evidence" in matrix["prompt_to_artifact_checklist"]["missing_requirements"]
    assert matrix["risk_boundary"]["max_per_trade_risk_pct"] == 0.025
    assert matrix["risk_boundary"]["writes_execution_config"] is False
    assert matrix["mainnet_live_allowed"] is False
    assert matrix["safety"]["opens_orders"] is False
    assert matrix["best_surface"]["surface"] == "buy_4h"
    assert matrix["best_surface"]["full"]["profit_factor"] == 1.262
    assert matrix["best_surface"]["gate_reasons"]["robust"] == [
        "test-trade-count-too-low",
        "walk-forward-min-profit-factor-below-target",
    ]
    assert len(matrix["validation_plan"]) == 2
    assert matrix["negative_surface_repair_plan"][0]["coverage_key"] == "sell"
    assert "ADAUSDT" in matrix["negative_surface_repair_plan"][0]["cross_symbol_scout_symbols"]
    assert matrix["negative_surface_repair_plan"][0]["guardrails"]["does_not_open_orders"] is True
    assert matrix["negative_surface_repair_plan"][0]["guardrails"]["max_per_trade_risk_pct"] == 0.025
    assert matrix["negative_surface_repair_plan"][0]["guardrails"]["mainnet_live_allowed"] is False
    assert "--target-side BUY" in matrix["validation_plan"][0]["interactive_probe_command"]
    assert matrix["promotion_boundary"]["mainnet_live_allowed"] is False
    assert matrix["promotion_boundary"]["max_per_trade_risk_pct"] == 0.025
    pool = payload["candidate_pool"]
    assert pool["mode"] == "short_medium_long_candidate_pool_v1"
    assert pool["simulation_trade_allowed"] is False
    assert pool["readiness_allowed_count"] == 0
    assert pool["latest_readiness_scan_path"] == str(readiness_path)
    assert pool["horizons"]["short"]["status"] == "blocked"
    assert pool["horizons"]["short"]["repair_action"] == "run_short_horizon_research_sweep"
    assert pool["horizons"]["medium"]["status"] == "candidate"
    assert pool["horizons"]["medium"]["readiness_candidate"]["symbol"] == "TRXUSDT"
    assert pool["horizons"]["long"]["status"] == "candidate"
    assert pool["missing_horizons"] == ["short"]
    assert pool["next_action"] == "continue_scan_research_and_readiness_repairs"
    assert pool["guardrails"]["hold_is_valid_when_no_candidate"] is True
    assert pool["guardrails"]["mainnet_live_allowed"] is False


def test_operator_dashboard_candidate_pool_allows_only_audited_readiness_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FlatClient(FakeClient):
        def positions(self):
            return []

    matrix_dir = tmp_path / "risk-combo-matrix"
    sweep_dir = tmp_path / "risk-combo-sweeps"
    readiness_dir = tmp_path / "hermes-readiness-scan"
    matrix_dir.mkdir()
    sweep_dir.mkdir()
    readiness_dir.mkdir()
    matrix_path = matrix_dir / "20260510T080706Z-risk-combo-matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-10T08:07:06+00:00",
                "mode": "risk_combo_side_interval_matrix_v1",
                "safety": {"opens_orders": False, "writes_execution_config": False, "mainnet_live_allowed": False},
                "surface_count": 3,
                "promising_surface_count": 3,
                "robust_surface_count": 1,
                "horizon_summary": {
                    "short": {"promising_surface_count": 1, "robust_surface_count": 0},
                    "medium": {"promising_surface_count": 1, "robust_surface_count": 1},
                    "long": {"promising_surface_count": 1, "robust_surface_count": 0},
                },
            }
        ),
        encoding="utf-8",
    )
    (sweep_dir / "20260510T080000Z-risk-combo-sweep.json").write_text("{}", encoding="utf-8")
    readiness_path = readiness_dir / "20260510T080500Z-hermes-readiness-scan.json"
    readiness_path.write_text(
        json.dumps(
            {
                "mode": "hermes_ai_readiness_scanner_v1",
                "allowed_count": 1,
                "research_candidate_report": {
                    "reviewable_horizon_counts": {"short": 1, "medium": 1, "long": 1},
                    "top_candidates": [
                        {
                            "symbol": "TRXUSDT",
                            "side": "BUY",
                            "interval": "4h",
                            "route_id": "trx-mean-reversion",
                            "horizon": "medium",
                            "research_status": "reviewable_signal",
                            "trade_readiness_allowed": True,
                            "readiness_next_action": "execute_ready_dry_run_only",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(dashboard, "OPERATOR_DASHBOARD_DIR", tmp_path / "operator-dashboard")
    monkeypatch.setattr(dashboard, "RISK_COMBO_MATRIX_DIR", matrix_dir)
    monkeypatch.setattr(dashboard, "RISK_COMBO_SWEEP_DIR", sweep_dir)
    monkeypatch.setattr(dashboard, "READINESS_SCAN_DIR", readiness_dir)
    monkeypatch.setattr(dashboard, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(dashboard, "BinanceClient", lambda settings: FlatClient(settings))
    monkeypatch.setattr(
        dashboard,
        "summarize_closed_trade_reviews",
        lambda: {"count": 140, "total_realized_pnl_usdt": 7.5, "profit_factor": 1.18},
    )
    monkeypatch.setattr(dashboard, "summarize_live_orders", lambda: {"count": 0})
    monkeypatch.setattr(dashboard, "read_live_orders", lambda: [])
    monkeypatch.setattr(
        dashboard,
        "run_loss_diagnostics",
        lambda min_bucket_trades, top_n: {
            "summary": {"count": 140, "profit_factor": 1.18, "avg_r": 0.04, "stop_loss_ratio": 54.0},
            "findings": [],
            "worst_buckets": [],
            "root_cause_recommendations": [],
            "report_path": str(tmp_path / "loss.json"),
        },
    )
    monkeypatch.setattr(dashboard, "_load_latest_digest_summary", lambda: {"available": False})
    monkeypatch.setattr(dashboard, "run_decision_audit", lambda **kwargs: _fake_decision_audit(tmp_path))

    payload = dashboard.build_operator_dashboard(_settings())

    assert payload["product_readiness"]["testnet_trade_ready"] is True
    pool = payload["candidate_pool"]
    assert pool["simulation_trade_allowed"] is True
    assert pool["ready_horizons"] == ["medium"]
    assert pool["missing_horizons"] == []
    assert pool["horizons"]["medium"]["status"] == "testnet_ready_candidate"
    assert pool["horizons"]["medium"]["readiness_candidate"]["symbol"] == "TRXUSDT"
    assert pool["next_action"] == "run_trade_decision_then_operator_approved_testnet_execution"
    assert pool["guardrails"]["mainnet_live_allowed"] is False


def test_operator_dashboard_flags_stale_risk_combo_matrix(monkeypatch, tmp_path: Path) -> None:
    matrix_dir = tmp_path / "risk-combo-matrix"
    sweep_dir = tmp_path / "risk-combo-sweeps"
    matrix_dir.mkdir()
    sweep_dir.mkdir()
    matrix_path = matrix_dir / "20260510T080706Z-risk-combo-matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-10T08:07:06+00:00",
                "mode": "risk_combo_side_interval_matrix_v1",
                "safety": {"mainnet_live_allowed": False},
                "surface_count": 1,
                "promising_surface_count": 0,
                "robust_surface_count": 0,
            }
        ),
        encoding="utf-8",
    )
    stale_sweep = sweep_dir / "20260510T080000Z-risk-combo-sweep.json"
    stale_sweep.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    newer_sweep = sweep_dir / "20260510T081000Z-risk-combo-sweep.json"
    newer_sweep.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    old_time = 1_000_000_000
    new_time = old_time + 60
    newer_time = new_time + 60
    import os

    os.utime(stale_sweep, (old_time, old_time))
    os.utime(matrix_path, (new_time, new_time))
    os.utime(newer_sweep, (newer_time, newer_time))

    summary = dashboard._load_latest_risk_combo_matrix_summary(
        matrix_dir=matrix_dir,
        sweep_dir=sweep_dir,
    )

    assert summary["available"] is True
    assert summary["freshness"]["status"] == "stale_after_new_sweeps"
    assert summary["freshness"]["newer_sweep_count"] == 1
    assert summary["freshness"]["newer_sweep_paths"] == [str(newer_sweep)]
    assert "rebuild risk-combo-matrix" in summary["freshness"]["action"]
