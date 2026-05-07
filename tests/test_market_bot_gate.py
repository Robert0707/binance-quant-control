from __future__ import annotations

import json
from pathlib import Path

import yaml

from binance_quant_control.market_bot_gate import evaluate_market_bot_gate


def _write_config(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "targets": {
                    "min_trades": 100,
                    "min_profit_factor": 1.25,
                    "min_expectancy_r": 0.05,
                    "min_payoff_ratio": 1.2,
                    "min_win_rate": 45.0,
                    "max_stop_loss_ratio": 55.0,
                    "min_walk_forward_stability": 0.6,
                    "min_slippage_resilience": 0.7,
                    "min_accepted_symbols": 1,
                    "required_symbols": [],
                    "require_feature_manifest_hash": True,
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_market_bot_gate_accepts_positive_expectancy_mature_row(tmp_path, monkeypatch) -> None:
    alpha = tmp_path / "alpha-research-ranking.json"
    alpha.write_text(
        json.dumps(
            {
                "feature_manifest": {"manifest_hash": "abc123"},
                "rows": [
                    {
                        "symbol": "BTCUSDT",
                        "interval": "4h",
                        "strategy_family": "trend_pullback",
                        "cohort_id": "BTCUSDT:4h:trend_pullback",
                        "trade_count": 140,
                        "profit_factor": 1.55,
                        "expectancy_r": 0.14,
                        "payoff_ratio": 1.8,
                        "win_rate": 52.0,
                        "stop_loss_ratio": 38.0,
                        "walk_forward_stability": 0.75,
                        "slippage_resilience": 0.82,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("binance_quant_control.market_bot_gate.ensure_runtime_dirs", lambda: None)

    payload = evaluate_market_bot_gate(
        alpha_report=alpha,
        config_path=_write_config(tmp_path / "gate.yaml"),
        output_dir=tmp_path / "out",
    )

    assert payload["safe_to_open_new_entries"] is True
    assert payload["accepted_count"] == 1
    assert payload["best"]["accepted"] is True
    assert payload["portfolio_gate"]["passed"] is True


def test_market_bot_gate_treats_win_rate_as_metric_only_when_payoff_edge_is_positive(
    tmp_path,
    monkeypatch,
) -> None:
    alpha = tmp_path / "alpha-research-ranking.json"
    alpha.write_text(
        json.dumps(
            {
                "feature_manifest": {"manifest_hash": "abc123"},
                "rows": [
                    {
                        "symbol": "SOLUSDT",
                        "interval": "4h",
                        "strategy_family": "trend_continuation",
                        "cohort_id": "SOLUSDT:4h:trend_continuation",
                        "trade_count": 188,
                        "profit_factor": 1.3851,
                        "expectancy_r": 0.2037,
                        "payoff_ratio": 2.8836,
                        "win_rate": 32.45,
                        "stop_loss_ratio": 48.94,
                        "walk_forward_stability": 0.75,
                        "slippage_resilience": 0.82,
                        "expectancy_edge_points": 6.6978,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("binance_quant_control.market_bot_gate.ensure_runtime_dirs", lambda: None)

    payload = evaluate_market_bot_gate(
        alpha_report=alpha,
        config_path=_write_config(tmp_path / "gate.yaml"),
        output_dir=tmp_path / "out",
    )

    assert payload["safe_to_open_new_entries"] is True
    assert payload["best"]["accepted"] is True
    assert payload["best"]["target_gaps"]["win_rate_points_needed"] > 0
    assert "win-rate-below-screening-floor" not in payload["best"]["blockers"]
    assert "advisory_warnings" not in payload["best"]
    assert payload["best"]["primary_gap"] == "none"


def test_market_bot_gate_blocks_positive_full_sample_when_oos_tail_is_negative(
    tmp_path,
    monkeypatch,
) -> None:
    alpha = tmp_path / "alpha-research-ranking.json"
    alpha.write_text(
        json.dumps(
            {
                "feature_manifest": {"manifest_hash": "abc123"},
                "rows": [
                    {
                        "symbol": "SOLUSDT",
                        "interval": "4h",
                        "strategy_family": "trend_continuation",
                        "cohort_id": "SOLUSDT:4h:trend_continuation",
                        "trade_count": 188,
                        "profit_factor": 1.3851,
                        "expectancy_r": 0.2037,
                        "payoff_ratio": 2.8836,
                        "out_of_sample_total_return_pct": -4.1106,
                        "win_rate": 32.45,
                        "stop_loss_ratio": 48.94,
                        "walk_forward_stability": 0.75,
                        "slippage_resilience": 0.82,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("binance_quant_control.market_bot_gate.ensure_runtime_dirs", lambda: None)

    payload = evaluate_market_bot_gate(
        alpha_report=alpha,
        config_path=_write_config(tmp_path / "gate.yaml"),
        output_dir=tmp_path / "out",
    )

    assert payload["safe_to_open_new_entries"] is False
    assert payload["best"]["research_state"] == "out_of_sample_failed"
    assert "out-of-sample-return-below-floor" in payload["best"]["blockers"]
    assert payload["out_of_sample_failed"][0]["cohort_id"] == "SOLUSDT:4h:trend_continuation"


def test_market_bot_gate_blocks_missing_feature_manifest_and_bad_payoff(tmp_path, monkeypatch) -> None:
    alpha = tmp_path / "alpha-research-ranking.json"
    alpha.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "symbol": "TRXUSDT",
                        "interval": "4h",
                        "strategy_family": "mean_reversion",
                        "cohort_id": "TRXUSDT:4h:mean_reversion",
                        "trade_count": 5,
                        "profit_factor": 0.5156,
                        "expectancy_r": -0.0641,
                        "payoff_ratio": 0.3437,
                        "win_rate": 60.0,
                        "stop_loss_ratio": 40.0,
                        "walk_forward_stability": 0.0,
                        "slippage_resilience": 0.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("binance_quant_control.market_bot_gate.ensure_runtime_dirs", lambda: None)

    payload = evaluate_market_bot_gate(
        alpha_report=alpha,
        config_path=_write_config(tmp_path / "gate.yaml"),
        output_dir=tmp_path / "out",
    )

    assert payload["safe_to_open_new_entries"] is False
    assert "feature-manifest-hash-missing" in payload["best"]["blockers"]
    assert "expectancy-r-below-market-bot-floor" in payload["best"]["blockers"]


def test_market_bot_gate_prefers_real_traded_row_over_zero_trade_placeholder(tmp_path, monkeypatch) -> None:
    alpha = tmp_path / "alpha-research-ranking.json"
    alpha.write_text(
        json.dumps(
            {
                "feature_manifest": {"manifest_hash": "abc123"},
                "rows": [
                    {
                        "symbol": "ETHUSDT",
                        "interval": "4h",
                        "strategy_family": "trend_pullback",
                        "cohort_id": "ETHUSDT:4h:trend_pullback",
                        "trade_count": 0,
                        "profit_factor": 0.0,
                        "expectancy_r": 0.0,
                        "payoff_ratio": 0.0,
                        "win_rate": 0.0,
                        "stop_loss_ratio": 0.0,
                    },
                    {
                        "symbol": "TRXUSDT",
                        "interval": "4h",
                        "strategy_family": "mean_reversion",
                        "cohort_id": "TRXUSDT:4h:mean_reversion",
                        "trade_count": 4,
                        "profit_factor": 1.2588,
                        "expectancy_r": 0.0175,
                        "payoff_ratio": 0.4195,
                        "win_rate": 75.0,
                        "stop_loss_ratio": 25.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("binance_quant_control.market_bot_gate.ensure_runtime_dirs", lambda: None)

    payload = evaluate_market_bot_gate(
        alpha_report=alpha,
        config_path=_write_config(tmp_path / "gate.yaml"),
        output_dir=tmp_path / "out",
    )

    assert payload["best"]["cohort_id"] == "TRXUSDT:4h:mean_reversion"


def test_market_bot_gate_marks_short_sample_edge_for_expansion(tmp_path, monkeypatch) -> None:
    alpha = tmp_path / "alpha-research-ranking.json"
    alpha.write_text(
        json.dumps(
            {
                "feature_manifest": {"manifest_hash": "abc123"},
                "rows": [
                    {
                        "symbol": "TRXUSDT",
                        "interval": "4h",
                        "strategy_family": "trend_continuation",
                        "cohort_id": "TRXUSDT:4h:trend_continuation",
                        "trade_count": 49,
                        "profit_factor": 2.3123,
                        "expectancy_r": 0.3159,
                        "payoff_ratio": 2.614,
                        "win_rate": 46.94,
                        "stop_loss_ratio": 46.94,
                        "walk_forward_stability": 0.5,
                        "slippage_resilience": 0.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("binance_quant_control.market_bot_gate.ensure_runtime_dirs", lambda: None)

    payload = evaluate_market_bot_gate(
        alpha_report=alpha,
        config_path=_write_config(tmp_path / "gate.yaml"),
        output_dir=tmp_path / "out",
    )

    assert payload["safe_to_open_new_entries"] is False
    assert payload["best"]["research_state"] == "expand_sample_before_promotion"
    assert payload["best"]["primary_gap"] == "sample_size"
    assert payload["best"]["target_gaps"]["trades_needed"] == 51
    assert payload["expansion_candidates"][0]["cohort_id"] == "TRXUSDT:4h:trend_continuation"
    assert payload["diagnostics"]["gap_counts"]["sample_size"] == 1
    assert any("rerun larger sample" in item for item in payload["best"]["next_actions"])


def test_market_bot_gate_rejects_expanded_route_regression(tmp_path, monkeypatch) -> None:
    alpha = tmp_path / "alpha-research-ranking.json"
    alpha.write_text(
        json.dumps(
            {
                "feature_manifest": {"manifest_hash": "abc123"},
                "rows": [
                    {
                        "symbol": "TRXUSDT",
                        "interval": "4h",
                        "strategy_family": "breakout",
                        "cohort_id": "TRXUSDT:4h:breakout",
                        "trade_count": 221,
                        "profit_factor": 1.0614,
                        "expectancy_r": 0.0261,
                        "payoff_ratio": 1.4077,
                        "win_rate": 42.99,
                        "stop_loss_ratio": 56.56,
                        "walk_forward_stability": 0.35,
                        "slippage_resilience": 0.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("binance_quant_control.market_bot_gate.ensure_runtime_dirs", lambda: None)

    payload = evaluate_market_bot_gate(
        alpha_report=alpha,
        config_path=_write_config(tmp_path / "gate.yaml"),
        output_dir=tmp_path / "out",
    )

    assert payload["safe_to_open_new_entries"] is False
    assert payload["best"]["research_state"] == "reject_expanded_route_regressed"
    assert payload["regressed_routes"][0]["cohort_id"] == "TRXUSDT:4h:breakout"
    assert any("reject TRXUSDT:4h:breakout" in item for item in payload["best"]["next_actions"])


def test_market_bot_gate_blocks_portfolio_until_six_symbols_are_accepted(tmp_path, monkeypatch) -> None:
    alpha = tmp_path / "alpha-research-ranking.json"
    rows = []
    for symbol in ("BTCUSDT", "ETHUSDT"):
        rows.append(
            {
                "symbol": symbol,
                "interval": "4h",
                "strategy_family": "trend_pullback",
                "cohort_id": f"{symbol}:4h:trend_pullback",
                "trade_count": 140,
                "profit_factor": 1.55,
                "expectancy_r": 0.14,
                "payoff_ratio": 1.8,
                "win_rate": 52.0,
                "stop_loss_ratio": 38.0,
                "walk_forward_stability": 0.75,
                "slippage_resilience": 0.82,
            }
        )
    alpha.write_text(
        json.dumps({"feature_manifest": {"manifest_hash": "abc123"}, "rows": rows}),
        encoding="utf-8",
    )
    config = tmp_path / "gate.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "targets": {
                    "min_trades": 100,
                    "min_profit_factor": 1.25,
                    "min_expectancy_r": 0.05,
                    "min_payoff_ratio": 1.2,
                    "min_win_rate": 45.0,
                    "max_stop_loss_ratio": 55.0,
                    "min_walk_forward_stability": 0.6,
                    "min_slippage_resilience": 0.7,
                    "min_accepted_symbols": 6,
                    "required_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                    "require_feature_manifest_hash": True,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("binance_quant_control.market_bot_gate.ensure_runtime_dirs", lambda: None)

    payload = evaluate_market_bot_gate(alpha_report=alpha, config_path=config, output_dir=tmp_path / "out")

    assert payload["accepted_count"] == 2
    assert payload["safe_to_open_new_entries"] is False
    assert payload["portfolio_gate"]["passed"] is False
    assert payload["portfolio_gate"]["accepted_symbol_count"] == 2
    assert "SOLUSDT" in payload["portfolio_gate"]["missing_required_symbols"]
    assert "accepted-symbol-count-below-portfolio-floor" in payload["portfolio_gate"]["blockers"]


def test_market_bot_gate_blocks_when_accepted_symbols_share_one_beta_group(tmp_path, monkeypatch) -> None:
    alpha = tmp_path / "alpha-research-ranking.json"
    rows = []
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        rows.append(
            {
                "symbol": symbol,
                "interval": "4h",
                "strategy_family": "trend_pullback",
                "cohort_id": f"{symbol}:4h:trend_pullback",
                "trade_count": 140,
                "profit_factor": 1.55,
                "expectancy_r": 0.14,
                "payoff_ratio": 1.8,
                "win_rate": 52.0,
                "stop_loss_ratio": 38.0,
                "walk_forward_stability": 0.75,
                "slippage_resilience": 0.82,
                "correlation_group": "crypto_beta",
                "beta_group": "btc_beta",
            }
        )
    alpha.write_text(
        json.dumps({"feature_manifest": {"manifest_hash": "abc123"}, "rows": rows}),
        encoding="utf-8",
    )
    config = tmp_path / "gate.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "targets": {
                    "min_trades": 100,
                    "min_profit_factor": 1.25,
                    "min_expectancy_r": 0.05,
                    "min_payoff_ratio": 1.2,
                    "min_win_rate": 45.0,
                    "max_stop_loss_ratio": 55.0,
                    "min_walk_forward_stability": 0.6,
                    "min_slippage_resilience": 0.7,
                    "min_accepted_symbols": 3,
                    "min_correlation_groups": 2,
                    "min_beta_groups": 2,
                    "require_feature_manifest_hash": True,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("binance_quant_control.market_bot_gate.ensure_runtime_dirs", lambda: None)

    payload = evaluate_market_bot_gate(alpha_report=alpha, config_path=config, output_dir=tmp_path / "out")

    assert payload["accepted_count"] == 3
    assert payload["safe_to_open_new_entries"] is False
    assert payload["portfolio_gate"]["accepted_correlation_group_count"] == 1
    assert payload["portfolio_gate"]["accepted_beta_group_count"] == 1
    assert "accepted-correlation-group-count-below-floor" in payload["portfolio_gate"]["blockers"]
    assert "accepted-beta-group-count-below-floor" in payload["portfolio_gate"]["blockers"]
