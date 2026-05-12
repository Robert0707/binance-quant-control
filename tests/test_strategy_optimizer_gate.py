from __future__ import annotations

import json
from datetime import datetime, timezone

import binance_quant_control.strategy_optimizer as optimizer


def test_optimizer_live_gate_blocks_reject(monkeypatch, tmp_path) -> None:
    state_dir = tmp_path / "strategy-optimizer"
    state_dir.mkdir(parents=True)
    report = state_dir / "20260428T000000Z-strategy-optimizer.json"
    report.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "promotion_decision": "reject",
                "screening_status": "failed",
                "validation_status": "failed",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(optimizer, "OPTIMIZER_STATE_DIR", state_dir)

    result = optimizer.evaluate_optimizer_live_gate()

    assert result["allowed"] is False
    assert result["promotion_decision"] == "reject"
    assert any("not promoted" in item for item in result["reasons"])


def test_optimizer_live_gate_allows_fresh_promotion(monkeypatch, tmp_path) -> None:
    state_dir = tmp_path / "strategy-optimizer"
    state_dir.mkdir(parents=True)
    report = state_dir / "20260428T000000Z-strategy-optimizer.json"
    report.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "promotion_decision": "promote",
                "screening_status": "passed",
                "validation_status": "passed",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(optimizer, "OPTIMIZER_STATE_DIR", state_dir)

    result = optimizer.evaluate_optimizer_live_gate()

    assert result["allowed"] is True
    assert result["reasons"] == []


def test_market_bot_live_gate_allows_symbol_from_safe_gate(monkeypatch, tmp_path) -> None:
    state_dir = tmp_path / "state"
    gate_dir = state_dir / "market-bot-gate"
    gate_dir.mkdir(parents=True)
    report = gate_dir / "20260428T000000Z-market-bot-gate.json"
    report.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "safe_to_open_new_entries": True,
                "accepted_count": 6,
                "feature_manifest_hash": "abc123",
                "portfolio_gate": {"enabled": True, "passed": True},
                "accepted": [
                    {
                        "symbol": "DOGEUSDT",
                        "route_id": "meme-high-beta",
                        "trade_count": 203,
                        "profit_factor": 1.48,
                        "expectancy_r": 0.26,
                        "payoff_ratio": 3.38,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(optimizer, "STATE_DIR", state_dir)

    result = optimizer.evaluate_market_bot_live_gate(symbol="DOGEUSDT", route_id="meme-high-beta")

    assert result["allowed"] is True
    assert result["matched_row"]["symbol"] == "DOGEUSDT"  # type: ignore[index]
    assert result["reasons"] == []


def test_market_bot_live_gate_blocks_unaccepted_symbol(monkeypatch, tmp_path) -> None:
    state_dir = tmp_path / "state"
    gate_dir = state_dir / "market-bot-gate"
    gate_dir.mkdir(parents=True)
    report = gate_dir / "20260428T000000Z-market-bot-gate.json"
    report.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "safe_to_open_new_entries": True,
                "portfolio_gate": {"enabled": True, "passed": True},
                "accepted": [{"symbol": "ETHUSDT", "route_id": "eth-core"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(optimizer, "STATE_DIR", state_dir)

    result = optimizer.evaluate_market_bot_live_gate(symbol="DOGEUSDT", route_id="meme-high-beta")

    assert result["allowed"] is False
    assert any("no accepted row" in item for item in result["reasons"])


def test_market_bot_live_gate_uses_latest_generated_report_across_state_tree(
    monkeypatch, tmp_path
) -> None:
    state_dir = tmp_path / "state"
    older_dir = state_dir / "market-bot-six-symbol-router-profile-4h-gate-l6000-v5-current"
    newer_dir = state_dir / "market-bot-six-symbol-router-profile-4h-gate-l6000-v7-current"
    older_dir.mkdir(parents=True)
    newer_dir.mkdir(parents=True)
    older = older_dir / "20260428T000000Z-market-bot-gate.json"
    newer = newer_dir / "20260429T000000Z-market-bot-gate.json"
    older.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-28T00:00:00+00:00",
                "safe_to_open_new_entries": True,
                "portfolio_gate": {"enabled": True, "passed": True},
                "accepted": [{"symbol": "DOGEUSDT", "route_id": "meme-high-beta"}],
            }
        ),
        encoding="utf-8",
    )
    newer.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "safe_to_open_new_entries": True,
                "portfolio_gate": {"enabled": True, "passed": True},
                "accepted": [{"symbol": "ETHUSDT", "route_id": "eth-core"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(optimizer, "STATE_DIR", state_dir)

    result = optimizer.evaluate_market_bot_live_gate(symbol="ETHUSDT", route_id="eth-core")

    assert result["allowed"] is True
    assert result["report_path"] == str(newer)
    assert result["matched_row"]["symbol"] == "ETHUSDT"  # type: ignore[index]


def test_risk_combo_live_gate_allows_matching_robust_surface(monkeypatch, tmp_path) -> None:
    matrix_dir = tmp_path / "risk-combo-matrix"
    matrix_dir.mkdir(parents=True)
    report = matrix_dir / "20260511T010203Z-risk-combo-matrix.json"
    report.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "safety": {
                    "opens_orders": False,
                    "writes_execution_config": False,
                    "mainnet_live_allowed": False,
                },
                "robust_surface_count": 1,
                "promising_surface_count": 1,
                "surfaces": [
                    {
                        "surface": "buy_1d",
                        "symbol": "TRXUSDT",
                        "route_id": "trx-mean-reversion",
                        "target_side": "BUY",
                        "target_interval": "1d",
                        "promotion_eligible": True,
                        "recovery_gate_passed": True,
                        "robust_recovery_gate_passed": True,
                        "full": {
                            "trade_count": 76,
                            "profit_factor": 1.99,
                            "expectancy_r": 0.46,
                            "stop_loss_ratio": 43.42,
                        },
                        "test": {
                            "trade_count": 15,
                            "profit_factor": 2.96,
                            "expectancy_r": 0.69,
                        },
                        "walk_forward": {
                            "window_count": 3,
                            "positive_expectancy_window_count": 3,
                            "min_profit_factor": 1.36,
                            "min_expectancy_r": 0.2,
                        },
                        "source_report_path": "state/risk-combo-sweeps/trx.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(optimizer, "RISK_COMBO_MATRIX_DIR", matrix_dir)

    result = optimizer.evaluate_risk_combo_live_gate(
        symbol="TRXUSDT",
        route_id="trx-mean-reversion",
        side="BUY",
        interval="1d",
    )

    assert result["allowed"] is True
    assert result["matched_surface"]["surface"] == "buy_1d"  # type: ignore[index]
    assert result["reasons"] == []


def test_risk_combo_live_gate_requires_interval_match(monkeypatch, tmp_path) -> None:
    matrix_dir = tmp_path / "risk-combo-matrix"
    matrix_dir.mkdir(parents=True)
    report = matrix_dir / "20260511T010203Z-risk-combo-matrix.json"
    report.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "safety": {"opens_orders": False, "writes_execution_config": False, "mainnet_live_allowed": False},
                "surfaces": [
                    {
                        "symbol": "TRXUSDT",
                        "route_id": "trx-mean-reversion",
                        "target_side": "BUY",
                        "target_interval": "1d",
                        "promotion_eligible": True,
                        "recovery_gate_passed": True,
                        "robust_recovery_gate_passed": True,
                        "full": {"trade_count": 76, "profit_factor": 1.99, "expectancy_r": 0.46, "stop_loss_ratio": 43.42},
                        "test": {"trade_count": 15, "profit_factor": 2.96, "expectancy_r": 0.69},
                        "walk_forward": {
                            "window_count": 3,
                            "positive_expectancy_window_count": 3,
                            "min_profit_factor": 1.36,
                            "min_expectancy_r": 0.2,
                        },
                        "source_report_path": "state/risk-combo-sweeps/trx.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(optimizer, "RISK_COMBO_MATRIX_DIR", matrix_dir)

    result = optimizer.evaluate_risk_combo_live_gate(
        symbol="TRXUSDT",
        route_id="trx-mean-reversion",
        side="BUY",
        interval="4h",
    )

    assert result["allowed"] is False
    assert result["matched_surface"] is None
    assert any("no matching robust surface" in item for item in result["reasons"])


def test_risk_combo_live_gate_allows_positive_small_sample_for_testnet_exploration(
    monkeypatch, tmp_path
) -> None:
    matrix_dir = tmp_path / "risk-combo-matrix"
    matrix_dir.mkdir(parents=True)
    report = matrix_dir / "20260511T010203Z-risk-combo-matrix.json"
    report.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "safety": {"opens_orders": False, "writes_execution_config": False, "mainnet_live_allowed": False},
                "surfaces": [
                    {
                        "surface": "buy_30m",
                        "symbol": "WIFUSDT",
                        "route_id": "meme-high-beta",
                        "target_side": "BUY",
                        "target_interval": "30m",
                        "promotion_eligible": False,
                        "recovery_gate_passed": False,
                        "robust_recovery_gate_passed": False,
                        "research_lead_only": True,
                        "full": {
                            "trade_count": 20,
                            "profit_factor": 1.94,
                            "expectancy_r": 0.4,
                            "stop_loss_ratio": 37.5,
                        },
                        "test": {
                            "trade_count": 3,
                            "profit_factor": 2.09,
                            "expectancy_r": 0.39,
                        },
                        "walk_forward": {
                            "window_count": 0,
                            "positive_expectancy_window_count": 0,
                            "min_profit_factor": 0.0,
                            "min_expectancy_r": 0.0,
                        },
                        "source_report_path": "state/risk-combo-sweeps/wif.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(optimizer, "RISK_COMBO_MATRIX_DIR", matrix_dir)

    normal = optimizer.evaluate_risk_combo_live_gate(
        symbol="WIFUSDT",
        route_id="meme-high-beta",
        side="BUY",
        interval="30m",
    )
    exploratory = optimizer.evaluate_risk_combo_live_gate(
        symbol="WIFUSDT",
        route_id="meme-high-beta",
        side="BUY",
        interval="30m",
        execution_mode="testnet_exploration",
    )

    assert normal["allowed"] is False
    assert normal["exploration_allowed"] is False
    assert any("not promotion eligible" in item for item in normal["reasons"])
    assert exploratory["allowed"] is True
    assert exploratory["exploration_allowed"] is True
    assert exploratory["reasons"] == []
    assert exploratory["strict_reasons"]
    assert exploratory["promotion_required_for_mainnet"] is True


def test_risk_combo_testnet_exploration_still_blocks_negative_expectancy_lead(
    monkeypatch, tmp_path
) -> None:
    matrix_dir = tmp_path / "risk-combo-matrix"
    matrix_dir.mkdir(parents=True)
    report = matrix_dir / "20260511T010203Z-risk-combo-matrix.json"
    report.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "safety": {"opens_orders": False, "writes_execution_config": False, "mainnet_live_allowed": False},
                "surfaces": [
                    {
                        "symbol": "DOGEUSDT",
                        "route_id": "meme-high-beta",
                        "target_side": "BUY",
                        "target_interval": "30m",
                        "promotion_eligible": False,
                        "recovery_gate_passed": False,
                        "robust_recovery_gate_passed": False,
                        "full": {"trade_count": 20, "profit_factor": 0.8, "expectancy_r": -0.1, "stop_loss_ratio": 50.0},
                        "test": {"trade_count": 4, "profit_factor": 1.1, "expectancy_r": 0.05},
                        "walk_forward": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(optimizer, "RISK_COMBO_MATRIX_DIR", matrix_dir)

    result = optimizer.evaluate_risk_combo_live_gate(
        symbol="DOGEUSDT",
        route_id="meme-high-beta",
        side="BUY",
        interval="30m",
        execution_mode="testnet_exploration",
    )

    assert result["allowed"] is False
    assert result["exploration_allowed"] is False
    assert any("expectancy" in item for item in result["exploration_reasons"])
