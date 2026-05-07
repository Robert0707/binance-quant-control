from __future__ import annotations

from types import SimpleNamespace

import yaml

import binance_quant_control.ai_expectancy_upgrade as upgrade
from binance_quant_control.candidate_universe import UniverseSymbol


def test_machine_symbol_allocation_combines_expectancy_and_liquidity(monkeypatch) -> None:
    monkeypatch.setattr(
        upgrade,
        "fetch_top_futures_symbols",
        lambda *_args, **_kwargs: [
            UniverseSymbol("SOLUSDT", 900.0, 1, "pytest-volume"),
            UniverseSymbol("DOGEUSDT", 800.0, 2, "pytest-volume"),
            UniverseSymbol("ETHUSDT", 700.0, 3, "pytest-volume"),
            UniverseSymbol("BTCUSDT", 600.0, 4, "pytest-volume"),
            UniverseSymbol("BNBUSDT", 500.0, 5, "pytest-volume"),
            UniverseSymbol("TRXUSDT", 400.0, 6, "pytest-volume"),
            UniverseSymbol("LINKUSDT", 300.0, 7, "pytest-volume"),
        ],
    )
    hermes = {
        "candidate_queue": [
            {
                "signal": {"symbol": "DOGEUSDT"},
                "machine_directive": {"priority_score": 100.0},
            },
            {
                "signal": {"symbol": "ETHUSDT"},
                "machine_directive": {"priority_score": 90.0},
            },
        ]
    }

    payload = upgrade._machine_symbol_allocation(
        SimpleNamespace(),
        hermes,
        requested_symbols=None,
        requested_discovery_symbols=None,
        universe_limit=7,
    )

    assert payload["exploit_symbols"] == ["DOGEUSDT", "ETHUSDT"]
    assert payload["explore_symbols"][:2] == ["SOLUSDT", "BTCUSDT"]
    assert payload["portfolio_symbols"] == [
        "DOGEUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BTCUSDT",
        "BNBUSDT",
        "TRXUSDT",
    ]
    assert payload["btc_eth_symbols"] == ["BTCUSDT", "ETHUSDT"]


def test_write_machine_research_config_enforces_ai_gates(tmp_path, monkeypatch) -> None:
    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        yaml.safe_dump(
            {
                "universe": {"symbols": ["BTCUSDT"], "intervals": ["1h"], "limit": 100},
                "research_entry_gate": {
                    "route_side_veto": False,
                    "shadow_route_side_veto": True,
                    "historical_signal_veto": False,
                },
                "feature_label_gate": {
                    "enabled": False,
                    "allow_if_insufficient_samples": True,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(upgrade, "DEFAULT_DISCOVERY_CONFIG", base_config)

    config_path = upgrade._write_machine_research_config(
        root=tmp_path,
        symbols=["DOGEUSDT", "ETHUSDT"],
        intervals=["30m", "1h"],
        limit=8000,
        dataset_path="state/feature-dataset.jsonl",
    )

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert payload["universe"]["symbols"] == ["DOGEUSDT", "ETHUSDT"]
    assert payload["universe"]["intervals"] == ["30m", "1h"]
    assert payload["research_entry_gate"]["route_side_veto"] is True
    assert payload["research_entry_gate"]["shadow_route_side_veto"] is False
    assert payload["research_entry_gate"]["historical_signal_veto"] is True
    assert payload["feature_label_gate"]["enabled"] is True
    assert payload["feature_label_gate"]["allow_if_insufficient_samples"] is False
    assert payload["feature_label_gate"]["dataset_path"] == "state/feature-dataset.jsonl"


def test_maturity_score_requires_more_than_dataset_for_nine_plus() -> None:
    payload = upgrade._maturity_score(
        steps=[
            {
                "name": "machine_ml_dataset_and_gate_config",
                "status": "passed",
                "machine_read": {"row_count": 1000},
            },
            {
                "name": "loss_diagnostics_side_veto",
                "status": "ok",
                "machine_read": {"summary": {"profit_factor": 0.8}},
            },
            {
                "name": "six_symbol_portfolio_gate",
                "status": "partial",
                "machine_read": {"accepted_count": 3},
            },
        ],
        final_decision={"safe_to_open_new_entries": False},
    )

    assert payload["score_10"] < 9
    assert payload["target_for_9_plus"]["missing_points"] > 0


def test_run_ai_expectancy_upgrade_attaches_readiness_scan(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(upgrade, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        upgrade,
        "load_settings",
        lambda: SimpleNamespace(use_testnet=True, live_trading_enabled=False),
    )
    monkeypatch.setattr(upgrade, "run_ai_surface_audit", lambda **_kwargs: {"status": "passed", "blocker_count": 0})
    monkeypatch.setattr(
        upgrade,
        "run_hermes_ai_trader",
        lambda **_kwargs: {
            "candidate_queue": [{"rank": 1, "signal": {"symbol": "BTCUSDT"}, "machine_directive": {"priority_score": 10}}],
            "machine_strategy": {"mode": "pytest"},
            "report_path": "hermes.json",
        },
    )
    monkeypatch.setattr(upgrade, "fetch_top_futures_symbols", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        upgrade,
        "build_feature_dataset",
        lambda *_args, **_kwargs: {
            "row_count": 2160,
            "dataset_hash": "dataset",
            "dataset_path": "dataset.jsonl",
            "report_path": "dataset.json",
            "feature_manifest": {"manifest_hash": "manifest"},
            "errors": [],
        },
    )
    monkeypatch.setattr(upgrade, "_write_machine_research_config", lambda **kwargs: tmp_path / "machine.yaml")
    monkeypatch.setattr(
        upgrade,
        "run_risk_combo_sweep",
        lambda **_kwargs: {"aggregate": {"robust_recovery_candidate_count": 1, "recovery_candidate_count": 1}},
    )
    monkeypatch.setattr(
        upgrade,
        "run_aggressive_alpha_research",
        lambda *_args, output_dir, **_kwargs: {"report_path": str(output_dir / "alpha.json")},
    )
    monkeypatch.setattr(
        upgrade,
        "evaluate_market_bot_gate",
        lambda **_kwargs: {
            "safe_to_open_new_entries": True,
            "accepted_count": 6,
            "portfolio_gate": {"accepted_symbols": ["BTCUSDT", "ETHUSDT"]},
            "best": {"symbol": "BTCUSDT", "expectancy_r": 0.2, "payoff_ratio": 1.4},
            "report_path": "gate.json",
        },
    )
    monkeypatch.setattr(
        upgrade,
        "run_loss_diagnostics",
        lambda **_kwargs: {"status": "ok", "summary": {"loss_cluster_count": 1}},
    )
    readiness_calls = []
    monkeypatch.setattr(
        upgrade,
        "run_ai_readiness_scan",
        lambda **kwargs: readiness_calls.append(kwargs)
        or {
            "candidate_count": 2,
            "allowed_count": 1,
            "selected_ready_candidate": {"symbol": "BTCUSDT"},
            "next_machine_action": "operator_execute_testnet_ticket",
            "hard_blocker_taxonomy": {},
            "execution_ticket": {"symbol": "BTCUSDT"},
            "report_path": "readiness.json",
        },
    )

    payload = upgrade.run_ai_expectancy_upgrade(
        output_dir=tmp_path,
        symbols=["BTCUSDT", "ETHUSDT"],
        discovery_symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "TRXUSDT"],
        limit=120,
        sweep_limit=120,
        max_configs=1,
        max_walk_forward_validations=1,
        max_readiness_candidates=2,
    )

    assert readiness_calls
    assert readiness_calls[0]["max_candidates"] == 2
    assert readiness_calls[0]["execution_mode"] == "testnet_exploration"
    assert payload["readiness_scan"]["allowed_count"] == 1
    assert payload["final_machine_decision"]["next_surface"] == "testnet_forward_evidence"
    assert payload["maturity_score"]["by_dimension"]["readiness_scan"] == 8.0
