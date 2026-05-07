from __future__ import annotations

import json
from pathlib import Path

import binance_quant_control.hermes_ai_trader as trader


def test_hermes_ai_trader_blocks_negative_alpha_and_records_skip(tmp_path, monkeypatch) -> None:
    skipped: dict[str, object] = {}
    signal_rows: list[dict[str, object]] = []

    monkeypatch.setattr(trader, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        trader,
        "run_professional_system_audit",
        lambda **_kwargs: {
            "trade_ready": False,
            "critical_blockers": ["alpha-evidence:no-promotion-eligible-cohort"],
            "execution_recommendation": "block_new_entries_and_rebuild_edge",
            "layer_summary": {"total": 13},
            "report_path": "state/audit.json",
            "evidence": {
                "alpha_report": {
                    "available": True,
                    "row_count": 6,
                    "trade_count": 5,
                    "promotion_eligible_count": 0,
                    "weighted_expectancy_r": -0.0641,
                    "weighted_payoff_ratio": 0.3437,
                    "finite_avg_profit_factor": 0.5156,
                    "weighted_win_rate": 60.0,
                    "weighted_stop_loss_ratio": 40.0,
                },
                "high_win_iteration": {"safe_to_open_new_entries": False},
            },
        },
    )
    monkeypatch.setattr(
        trader,
        "append_skipped_signal",
        lambda **kwargs: skipped.update(kwargs) or Path("state/journals/skipped.jsonl"),
    )
    monkeypatch.setattr(
        trader,
        "append_trading_signal",
        lambda signal, gate=None: signal_rows.append({"signal": signal.to_dict(), "gate": gate}) or Path(
            "state/signals/trading-signals.jsonl"
        ),
    )

    payload = trader.run_hermes_ai_trader(output_dir=tmp_path)

    assert payload["open_order_gate"]["allowed"] is False
    assert "expectancy-r-not-positive" in payload["open_order_gate"]["blockers"]
    assert payload["committee"]["decision"] == "reject"
    assert payload["feature_manifest"]["manifest_hash"]
    assert payload["signal_api"]["schema"] == "binance_quant_control.trading_signal.v1"
    assert signal_rows[0]["signal"]["symbol"] == "UNRESOLVED"
    assert skipped["gate"] == "hermes_ai_trader_open_order_gate"
    assert Path(payload["report_path"]).exists()


def test_hermes_ai_trader_can_allow_when_all_gates_are_clean(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(trader, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        trader,
        "run_professional_system_audit",
        lambda **_kwargs: {
            "trade_ready": True,
            "critical_blockers": [],
            "execution_recommendation": "paper_or_testnet_readiness_review",
            "layer_summary": {"total": 13},
            "report_path": "state/audit.json",
            "evidence": {
                "alpha_report": {
                    "available": True,
                    "symbol": "BTCUSDT",
                    "interval": "4h",
                    "strategy_family": "trend_pullback",
                    "trade_count": 120,
                    "promotion_eligible_count": 1,
                    "weighted_expectancy_r": 0.2,
                    "weighted_payoff_ratio": 1.4,
                    "finite_avg_profit_factor": 1.7,
                    "weighted_win_rate": 68.0,
                    "weighted_stop_loss_ratio": 25.0,
                },
                "high_win_iteration": {"safe_to_open_new_entries": True},
            },
        },
    )

    payload = trader.run_hermes_ai_trader(output_dir=tmp_path)

    assert payload["open_order_gate"]["allowed"] is True
    assert payload["signal"]["status"] == "candidate"
    assert payload["portfolio_target"]["accepted"] is True
    assert payload["portfolio_risk"]["remaining_total_risk_pct"] == 0.03


def test_hermes_ai_trader_prefers_market_bot_gate_candidate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(trader, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(trader, "append_trading_signal", lambda signal, gate=None: tmp_path / "signals.jsonl")
    monkeypatch.setattr(trader, "append_skipped_signal", lambda **_kwargs: tmp_path / "skipped.jsonl")
    monkeypatch.setattr(
        trader,
        "run_professional_system_audit",
        lambda **_kwargs: {
            "trade_ready": True,
            "critical_blockers": [],
            "execution_recommendation": "paper_or_testnet_readiness_review",
            "layer_summary": {"total": 13},
            "report_path": "state/audit.json",
            "evidence": {
                "market_bot_gate": {
                    "available": True,
                    "path": "state/market-bot-gate.json",
                    "safe_to_open_new_entries": True,
                    "accepted_count": 6,
                    "accepted_symbols": ["DOGEUSDT", "ETHUSDT"],
                    "targets": {
                        "min_trades": 100,
                        "min_profit_factor": 1.25,
                        "min_expectancy_r": 0.05,
                        "min_payoff_ratio": 1.2,
                    },
                    "portfolio_gate": {"enabled": True, "passed": True},
                    "accepted": [
                        {
                            "symbol": "DOGEUSDT",
                            "interval": "4h",
                            "strategy_family": "ai_family_router",
                            "route_id": "doge-meme-high-beta",
                            "cohort_id": "DOGEUSDT:4h:ai_family_router",
                            "accepted": True,
                            "trade_count": 203,
                            "market_bot_score": 226.3,
                            "expectancy_r": 0.263,
                            "payoff_ratio": 3.3836,
                            "profit_factor": 1.4878,
                            "win_rate": 30.54,
                            "stop_loss_ratio": 50.25,
                        }
                    ],
                },
                "alpha_report": {
                    "available": True,
                    "symbol": "TRXUSDT",
                    "trade_count": 5,
                    "promotion_eligible_count": 0,
                    "weighted_expectancy_r": -0.0641,
                    "weighted_payoff_ratio": 0.3437,
                    "finite_avg_profit_factor": 0.5156,
                },
                "high_win_iteration": {"safe_to_open_new_entries": False},
            },
        },
    )

    payload = trader.run_hermes_ai_trader(output_dir=tmp_path)

    assert payload["open_order_gate"]["allowed"] is True
    assert payload["signal"]["symbol"] == "DOGEUSDT"
    assert payload["signal"]["route_id"] == "doge-meme-high-beta"
    assert payload["signal"]["metadata"]["source"] == "market_bot_gate.accepted"
    assert payload["candidate_queue"][0]["signal"]["symbol"] == "DOGEUSDT"
    assert payload["candidate_queue"][0]["next_action"] == "ready_for_live_readiness_scan"
    assert payload["candidate_queue"][0]["machine_directive"]["directive"] == "exploit"
    assert payload["machine_strategy"]["exploit_symbols"] == ["DOGEUSDT"]
    assert payload["machine_policy"]["next_scan_symbols"] == ["DOGEUSDT"]
    assert payload["committee"]["decision"] == "approve_for_paper_or_testnet_review"


def test_hermes_ai_trader_builds_machine_candidate_queue_for_all_market_bot_rows(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(trader, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(trader, "append_trading_signal", lambda signal, gate=None: tmp_path / "signals.jsonl")
    monkeypatch.setattr(trader, "append_skipped_signal", lambda **_kwargs: tmp_path / "skipped.jsonl")
    accepted = [
        {
            "symbol": "DOGEUSDT",
            "interval": "4h",
            "strategy_family": "ai_family_router",
            "route_id": "doge-meme-high-beta",
            "cohort_id": "DOGEUSDT:4h:ai_family_router",
            "accepted": True,
            "trade_count": 203,
            "market_bot_score": 226.3,
            "expectancy_r": 0.263,
            "payoff_ratio": 3.3836,
            "profit_factor": 1.4878,
            "win_rate": 30.54,
            "stop_loss_ratio": 50.25,
        },
        {
            "symbol": "ETHUSDT",
            "interval": "4h",
            "strategy_family": "ai_family_router",
            "route_id": "eth-core",
            "cohort_id": "ETHUSDT:4h:ai_family_router",
            "accepted": True,
            "trade_count": 118,
            "market_bot_score": 219.8,
            "expectancy_r": 0.2407,
            "payoff_ratio": 3.1092,
            "profit_factor": 1.4202,
            "win_rate": 31.36,
            "stop_loss_ratio": 54.24,
        },
        {
            "symbol": "BTCUSDT",
            "interval": "4h",
            "strategy_family": "ai_family_router",
            "route_id": "btc-core",
            "cohort_id": "BTCUSDT:4h:ai_family_router",
            "accepted": True,
            "trade_count": 154,
            "market_bot_score": 201.4,
            "expectancy_r": 0.1408,
            "payoff_ratio": 2.4427,
            "profit_factor": 1.3571,
            "win_rate": 35.71,
            "stop_loss_ratio": 46.75,
        },
    ]
    monkeypatch.setattr(
        trader,
        "run_professional_system_audit",
        lambda **_kwargs: {
            "trade_ready": True,
            "critical_blockers": [],
            "execution_recommendation": "paper_or_testnet_readiness_review",
            "layer_summary": {"total": 13},
            "report_path": "state/audit.json",
            "evidence": {
                "market_bot_gate": {
                    "available": True,
                    "path": "state/market-bot-gate.json",
                    "safe_to_open_new_entries": True,
                    "accepted_count": 3,
                    "accepted_symbols": ["BTCUSDT", "DOGEUSDT", "ETHUSDT"],
                    "targets": {
                        "min_trades": 100,
                        "min_profit_factor": 1.25,
                        "min_expectancy_r": 0.05,
                        "min_payoff_ratio": 1.2,
                    },
                    "portfolio_gate": {"enabled": True, "passed": True},
                    "accepted": accepted,
                },
                "alpha_report": {"available": False},
            },
        },
    )

    payload = trader.run_hermes_ai_trader(output_dir=tmp_path)

    assert [item["signal"]["symbol"] for item in payload["candidate_queue"]] == [
        "DOGEUSDT",
        "ETHUSDT",
        "BTCUSDT",
    ]
    assert all(item["machine_state"] == "candidate_ready" for item in payload["candidate_queue"])
    assert [item["machine_directive"]["directive"] for item in payload["candidate_queue"]] == [
        "exploit",
        "exploit",
        "exploit",
    ]
    assert payload["machine_policy"]["next_scan_symbols"] == ["DOGEUSDT", "ETHUSDT", "BTCUSDT"]
    assert payload["machine_strategy"]["exploit_symbols"] == ["DOGEUSDT", "ETHUSDT", "BTCUSDT"]


def test_hermes_ai_trader_selects_real_alpha_row_instead_of_aggregate(tmp_path, monkeypatch) -> None:
    alpha_report = tmp_path / "alpha-research-ranking.json"
    alpha_report.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "symbol": "ETHUSDT",
                        "interval": "1h",
                        "strategy_family": "trend_pullback",
                        "cohort_id": "ETHUSDT:1h:trend_pullback",
                        "trade_count": 20,
                        "expectancy_r": -0.1,
                        "payoff_ratio": 0.7,
                        "profit_factor": 0.8,
                        "win_rate": 45.0,
                        "stop_loss_ratio": 55.0,
                    },
                    {
                        "symbol": "TRXUSDT",
                        "interval": "4h",
                        "strategy_family": "mean_reversion",
                        "cohort_id": "TRXUSDT:4h:mean_reversion",
                        "trade_count": 5,
                        "expectancy_r": -0.0641,
                        "payoff_ratio": 0.3437,
                        "profit_factor": 0.5156,
                        "win_rate": 60.0,
                        "stop_loss_ratio": 40.0,
                        "symbol_strategy": {
                            "route_id": "trx-mean-reversion",
                            "interval_family_sides": {"4h": {"mean_reversion": ["SELL"]}},
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(trader, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(trader, "append_trading_signal", lambda signal, gate=None: tmp_path / "signals.jsonl")
    monkeypatch.setattr(trader, "append_skipped_signal", lambda **_kwargs: tmp_path / "skipped.jsonl")
    monkeypatch.setattr(
        trader,
        "run_professional_system_audit",
        lambda **_kwargs: {
            "trade_ready": False,
            "critical_blockers": ["alpha-evidence:no-promotion-eligible-cohort"],
            "execution_recommendation": "block_new_entries_and_rebuild_edge",
            "layer_summary": {"total": 13},
            "report_path": "state/audit.json",
            "evidence": {
                "alpha_report": {
                    "available": True,
                    "path": str(alpha_report),
                    "row_count": 2,
                    "trade_count": 25,
                    "promotion_eligible_count": 0,
                    "weighted_expectancy_r": -0.08,
                    "weighted_payoff_ratio": 0.5,
                    "finite_avg_profit_factor": 0.7,
                    "weighted_win_rate": 50.0,
                    "weighted_stop_loss_ratio": 50.0,
                },
                "high_win_iteration": {"safe_to_open_new_entries": False},
            },
        },
    )

    payload = trader.run_hermes_ai_trader(output_dir=tmp_path)

    assert payload["signal"]["symbol"] == "TRXUSDT"
    assert payload["signal"]["side"] == "SELL"
    assert payload["signal"]["route_id"] == "trx-mean-reversion"
    assert payload["signal"]["metadata"]["source"] == "alpha_research.rows"


def test_hermes_ai_trader_routes_negative_surface_to_quarantine(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(trader, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(trader, "append_trading_signal", lambda signal, gate=None: tmp_path / "signals.jsonl")
    monkeypatch.setattr(trader, "append_skipped_signal", lambda **_kwargs: tmp_path / "skipped.jsonl")
    monkeypatch.setattr(
        trader,
        "run_professional_system_audit",
        lambda **_kwargs: {
            "trade_ready": True,
            "critical_blockers": [],
            "execution_recommendation": "paper_or_testnet_readiness_review",
            "layer_summary": {"total": 13},
            "report_path": "state/audit.json",
            "evidence": {
                "market_bot_gate": {
                    "available": True,
                    "path": "state/market-bot-gate.json",
                    "safe_to_open_new_entries": False,
                    "accepted_count": 0,
                    "accepted_symbols": [],
                    "targets": {
                        "min_trades": 100,
                        "min_profit_factor": 1.25,
                        "min_expectancy_r": 0.05,
                        "min_payoff_ratio": 1.2,
                    },
                    "portfolio_gate": {"enabled": True, "passed": False},
                    "rows": [
                        {
                            "symbol": "TRXUSDT",
                            "interval": "4h",
                            "strategy_family": "trend_continuation",
                            "route_id": "trx-high-beta",
                            "cohort_id": "TRXUSDT:4h:trend_continuation",
                            "trade_count": 358,
                            "market_bot_score": 12.0,
                            "expectancy_r": -0.0011,
                            "payoff_ratio": 0.98,
                            "profit_factor": 0.9972,
                            "win_rate": 42.0,
                            "stop_loss_ratio": 56.0,
                        }
                    ],
                },
                "alpha_report": {"available": False},
                "high_win_iteration": {"safe_to_open_new_entries": False},
            },
        },
    )

    payload = trader.run_hermes_ai_trader(output_dir=tmp_path)

    directive = payload["candidate_queue"][0]["machine_directive"]
    assert directive["directive"] == "quarantine"
    assert directive["blocked_surface"] == "paper_order,testnet_order,live_readiness_scan"
    assert payload["machine_strategy"]["quarantine_symbols"] == ["TRXUSDT"]
    assert "loss-diagnostics" in payload["machine_strategy"]["next_commands"][0]
