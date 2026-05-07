from __future__ import annotations

import json
from pathlib import Path

import yaml

import binance_quant_control.professional_system_audit as audit


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_professional_system_audit_blocks_when_required_layer_missing(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    (root / "src" / "binance_quant_control").mkdir(parents=True)
    existing = root / "src" / "binance_quant_control" / "risk_guard.py"
    existing.write_text("", encoding="utf-8")
    config = tmp_path / "blueprint.yaml"
    alpha = root / "state" / "alpha" / "alpha-research-ranking.json"
    iteration = root / "state" / "high-win-iteration" / "20260504T000000Z-high-win-iteration.json"
    _write_json(
        alpha,
        {
            "performance_summary": {
                "row_count": 1,
                "trade_count": 5,
                "promotion_eligible_count": 0,
                "weighted_win_rate": 60.0,
                "weighted_stop_loss_ratio": 40.0,
                "finite_avg_profit_factor": 0.5,
                "weighted_expectancy_r": -0.1,
                "weighted_payoff_ratio": 0.3,
            },
            "execution_recommendation": "block_new_entries_and_continue_research",
        },
    )
    _write_json(
        iteration,
        {
            "safe_to_open_new_entries": False,
            "promotion_allowed": False,
            "best_alpha_gate": {"passed": False, "blockers": ["aggregate-expectancy-r-below-floor"]},
            "portfolio_gate": {"enabled": True, "passed": False, "blockers": ["promoted-symbol-count-below-floor"]},
        },
    )
    config.write_text(
        yaml.safe_dump(
            {
                "meta": {"status": "test"},
                "evidence": {"latest_alpha_report": str(alpha)},
                "layers": [
                    {
                        "id": "risk",
                        "name": "Risk",
                        "critical": True,
                        "trade_required": True,
                        "status_if_present": "ready",
                        "required_paths": [str(existing)],
                    },
                    {
                        "id": "portfolio",
                        "name": "Portfolio",
                        "critical": True,
                        "trade_required": True,
                        "status_if_present": "ready",
                        "required_paths": [str(root / "src" / "binance_quant_control" / "portfolio.py")],
                        "rebuild": ["build portfolio engine"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(audit, "PROJECT_ROOT", root)
    monkeypatch.setattr(audit, "STATE_DIR", root / "state")
    monkeypatch.setattr(audit, "PROFESSIONAL_SYSTEM_AUDIT_DIR", tmp_path / "audit")
    monkeypatch.setattr(audit, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(audit, "summarize_closed_trade_reviews", lambda: {"count": 0})
    monkeypatch.setattr(audit, "summarize_live_orders", lambda: {"count": 0})

    payload = audit.run_professional_system_audit(config_path=config, output_dir=tmp_path / "out")

    assert payload["trade_ready"] is False
    assert "portfolio:missing" in payload["critical_blockers"]
    assert "promotion-gate:safe-to-open-new-entries-false" in payload["critical_blockers"]
    assert payload["evidence"]["alpha_report"]["promotion_eligible_count"] == 0
    assert Path(payload["report_path"]).exists()


def test_professional_system_audit_can_pass_architecture_and_gate(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    layer_path = root / "src" / "binance_quant_control" / "portfolio.py"
    layer_path.parent.mkdir(parents=True)
    layer_path.write_text("", encoding="utf-8")
    alpha = root / "state" / "alpha" / "alpha-research-ranking.json"
    iteration = root / "state" / "high-win-iteration" / "20260504T010000Z-high-win-iteration.json"
    _write_json(
        alpha,
        {
            "performance_summary": {
                "row_count": 10,
                "trade_count": 1000,
                "promotion_eligible_count": 10,
                "weighted_win_rate": 68.0,
                "weighted_stop_loss_ratio": 25.0,
                "finite_avg_profit_factor": 1.7,
                "weighted_expectancy_r": 0.2,
                "weighted_payoff_ratio": 1.3,
            },
            "execution_recommendation": "paper_or_testnet_readiness_review",
        },
    )
    _write_json(
        iteration,
        {
            "mode": "expectancy_research_iteration",
            "safe_to_open_new_entries": True,
            "promotion_allowed": True,
            "best_alpha_gate": {"passed": True, "blockers": [], "targets": {"min_trades": 100}},
            "portfolio_gate": {
                "enabled": True,
                "passed": True,
                "promoted_symbol_count": 10,
                "missing_required_symbols": [],
                "blockers": [],
            },
        },
    )
    config = tmp_path / "blueprint.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "meta": {"status": "test"},
                "evidence": {"latest_alpha_report": str(alpha)},
                "layers": [
                    {
                        "id": "portfolio",
                        "name": "Portfolio",
                        "critical": True,
                        "trade_required": True,
                        "status_if_present": "ready",
                        "required_paths": [str(layer_path)],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(audit, "PROJECT_ROOT", root)
    monkeypatch.setattr(audit, "STATE_DIR", root / "state")
    monkeypatch.setattr(audit, "PROFESSIONAL_SYSTEM_AUDIT_DIR", tmp_path / "audit")
    monkeypatch.setattr(audit, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(audit, "summarize_closed_trade_reviews", lambda: {"count": 1000})
    monkeypatch.setattr(audit, "summarize_live_orders", lambda: {"count": 0})

    payload = audit.run_professional_system_audit(config_path=config, output_dir=tmp_path / "out")

    assert payload["trade_ready"] is True
    assert payload["critical_blockers"] == []
    assert payload["execution_recommendation"] == "paper_or_testnet_readiness_review"


def test_professional_system_audit_accepts_market_bot_gate_without_high_win_iteration(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "repo"
    layer_path = root / "src" / "binance_quant_control" / "portfolio.py"
    layer_path.parent.mkdir(parents=True)
    layer_path.write_text("", encoding="utf-8")
    alpha = root / "state" / "alpha" / "alpha-research-ranking.json"
    market_gate = root / "state" / "market-bot" / "20260504T010000Z-market-bot-gate.json"
    _write_json(
        alpha,
        {
            "performance_summary": {
                "row_count": 6,
                "trade_count": 988,
                "promotion_eligible_count": 3,
                "finite_avg_profit_factor": 1.37,
                "weighted_expectancy_r": 0.17,
                "weighted_payoff_ratio": 2.95,
            },
            "execution_recommendation": "paper_or_testnet_candidate_available",
        },
    )
    _write_json(
        market_gate,
        {
            "mode": "market_bot_expectancy_gate",
            "safe_to_open_new_entries": True,
            "execution_recommendation": "eligible_for_hermes_ai_trader_and_live_readiness",
            "accepted_count": 6,
            "targets": {"min_accepted_symbols": 6},
            "feature_manifest_hash": "abc123",
            "portfolio_gate": {
                "enabled": True,
                "passed": True,
                "accepted_symbol_count": 6,
                "blockers": [],
            },
            "accepted": [{"symbol": "BTCUSDT", "trade_count": 154}],
        },
    )
    config = tmp_path / "blueprint.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "meta": {"status": "test"},
                "promotion_gate": {"min_accepted_symbols": 6},
                "evidence": {
                    "latest_alpha_report": str(alpha),
                    "latest_market_bot_gate": str(market_gate),
                },
                "layers": [
                    {
                        "id": "portfolio",
                        "name": "Portfolio",
                        "critical": True,
                        "trade_required": True,
                        "status_if_present": "partial",
                        "required_paths": [str(layer_path)],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(audit, "PROJECT_ROOT", root)
    monkeypatch.setattr(audit, "STATE_DIR", root / "state")
    monkeypatch.setattr(audit, "PROFESSIONAL_SYSTEM_AUDIT_DIR", tmp_path / "audit")
    monkeypatch.setattr(audit, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(audit, "summarize_closed_trade_reviews", lambda: {"count": 0})
    monkeypatch.setattr(audit, "summarize_live_orders", lambda: {"count": 0})

    payload = audit.run_professional_system_audit(config_path=config, output_dir=tmp_path / "out")

    assert payload["trade_ready"] is True
    assert payload["critical_blockers"] == []
    assert payload["architecture_warnings"] == ["portfolio:partial"]
    assert payload["evidence"]["market_bot_gate"]["safe_to_open_new_entries"] is True
    assert payload["evidence"]["market_bot_gate"]["accepted_count"] == 6
