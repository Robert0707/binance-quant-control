from __future__ import annotations

from argparse import Namespace

import binance_quant_control.cli as cli


def test_cmd_new_symbol_workflow_compact_outputs_fixed_no_order_surface(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_workflow(**kwargs):
        captured["kwargs"] = kwargs
        return {
            "mode": "new_symbol_workflow_v1",
            "objective": "fixed_no_code_new_symbol_analysis_to_testnet_readiness_workflow",
            "safety": {
                "opens_orders": False,
                "mainnet_live_allowed": False,
                "requires_operator_execute_for_testnet": True,
                "risk_ceiling_pct": 0.025,
            },
            "inputs": {
                "symbols": ["SOLUSDT"],
                "intervals": ["15m", "4h"],
                "sides": ["BUY", "SELL"],
                "research_depth": "smoke",
                "plan_only": True,
            },
            "outcome": "research_candidate",
            "status_counts": {
                "reject": 0,
                "research_candidate": 1,
                "near_ready_market_only": 0,
                "testnet_ready_candidate": 0,
            },
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "outcome": "research_candidate",
                    "route": {"route": {"route_id": "sol-high-beta"}},
                    "candidate_keys": ["SOLUSDT:BUY:4h"],
                    "near_ready_candidate_keys": [],
                    "ready_candidate_keys": [],
                    "next_command": "openclaw-quantctl new-symbol-workflow --symbols SOLUSDT --research-depth focused --compact",
                }
            ],
            "sentinel": {"opens_orders": False, "report_path": "state/sentinel.json"},
            "research_sweeps": [],
            "risk_combo_matrix": None,
            "readiness": {"allowed_count": 0, "execution_ticket": None},
            "promotion_boundary": {"mainnet_live_allowed": False},
            "report_path": "state/new-symbol-workflow/report.json",
        }

    monkeypatch.setattr(cli, "run_new_symbol_workflow", fake_workflow)
    monkeypatch.setattr(
        cli,
        "print_json",
        lambda payload, compact=False: captured.update({"payload": payload, "compact": compact}),
    )

    cli.cmd_new_symbol_workflow(
        Namespace(
            symbols="solusdt",
            intervals="15m,4h",
            sides="BUY,SELL",
            research_depth="smoke",
            plan_only=True,
            output_dir="",
            strategy_config="config/strategy-live-pilot.yaml",
            blueprint_config="config/professional-system-blueprint.default.yaml",
            max_readiness_candidates=2,
            compact=True,
        )
    )

    assert captured["kwargs"] == {
        "symbols": ["solusdt"],
        "intervals": ["15m", "4h"],
        "sides": ["BUY", "SELL"],
        "research_depth": "smoke",
        "plan_only": True,
        "output_dir": None,
        "strategy_config": "config/strategy-live-pilot.yaml",
        "blueprint_config": "config/professional-system-blueprint.default.yaml",
        "max_readiness_candidates": 2,
    }
    assert captured["compact"] is True
    payload = captured["payload"]  # type: ignore[assignment]
    assert payload["safety"]["opens_orders"] is False  # type: ignore[index]
    assert payload["safety"]["mainnet_live_allowed"] is False  # type: ignore[index]
    assert payload["outcome"] == "research_candidate"  # type: ignore[index]
    assert payload["status_counts"]["research_candidate"] == 1  # type: ignore[index]
    assert payload["symbols"][0]["symbol"] == "SOLUSDT"  # type: ignore[index]
    assert payload["symbols"][0]["next_command"].startswith("openclaw-quantctl new-symbol-workflow")  # type: ignore[index]
    assert payload["readiness"]["execution_ticket"] is None  # type: ignore[index]
