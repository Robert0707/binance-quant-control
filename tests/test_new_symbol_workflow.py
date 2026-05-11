from __future__ import annotations

from pathlib import Path

import binance_quant_control.new_symbol_workflow as workflow


def test_new_symbol_workflow_plan_only_accepts_arbitrary_symbol_without_orders(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(workflow, "ensure_runtime_dirs", lambda: None)

    class FakeValidation:
        def to_dict(self):
            return {"validation_min_profit_factor": 1.2}

    class FakeRoute:
        route_id = "defensive-unknown"
        simulation_mode = "paper"
        validation = FakeValidation()

        def to_dict(self):
            return {"route_id": self.route_id, "simulation_mode": self.simulation_mode}

    monkeypatch.setattr(workflow, "resolve_symbol_route", lambda symbol: FakeRoute())
    monkeypatch.setattr(
        workflow,
        "run_ai_market_sentinel",
        lambda **_kwargs: {
            "safety": {"opens_orders": False},
            "trend_state": {"NEWCOINUSDT": {"bias": "long"}},
            "errors": [],
            "report_path": str(tmp_path / "sentinel.json"),
        },
    )
    monkeypatch.setattr(
        workflow,
        "run_ai_readiness_scan",
        lambda **_kwargs: {
            "candidate_count": 0,
            "allowed_count": 0,
            "execution_ticket": None,
            "next_machine_action": "repair_alpha_gate_or_hermes_candidate_queue",
            "hard_blocker_taxonomy": {},
            "denial_journal_path": str(tmp_path / "denials.jsonl"),
            "denial_journal_count": 0,
            "research_candidate_report": {"near_ready_count": 0},
            "report_path": str(tmp_path / "readiness.json"),
        },
    )

    payload = workflow.run_new_symbol_workflow(
        symbols=["newcoinusdt"],
        intervals=["15m"],
        sides=["BUY", "SELL"],
        plan_only=True,
        output_dir=tmp_path,
    )

    assert payload["safety"]["opens_orders"] is False
    assert payload["safety"]["mainnet_live_allowed"] is False
    assert payload["outcome"] == "research_candidate"
    assert payload["symbols"][0]["symbol"] == "NEWCOINUSDT"
    assert payload["symbols"][0]["route"]["status"] == "ok"
    assert payload["symbols"][0]["commands"][0]["command"] == "openclaw-quantctl route-symbol NEWCOINUSDT"
    assert any("--target-side SELL" in item["command"] for item in payload["symbols"][0]["commands"])
    assert payload["research_sweeps"] == []
    assert payload["readiness"]["execution_ticket"] is None
    assert Path(payload["report_path"]).exists()


def test_new_symbol_workflow_marks_near_ready_market_only(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(workflow, "ensure_runtime_dirs", lambda: None)

    class FakeValidation:
        def to_dict(self):
            return {}

    class FakeRoute:
        route_id = "trx-mean-reversion"
        simulation_mode = "paper"
        validation = FakeValidation()

        def to_dict(self):
            return {"route_id": self.route_id, "simulation_mode": self.simulation_mode}

    monkeypatch.setattr(workflow, "resolve_symbol_route", lambda symbol: FakeRoute())
    monkeypatch.setattr(
        workflow,
        "run_ai_market_sentinel",
        lambda **_kwargs: {
            "safety": {"opens_orders": False},
            "trend_state": {"TRXUSDT": {"bias": "mixed"}},
            "errors": [],
            "report_path": str(tmp_path / "sentinel.json"),
        },
    )
    monkeypatch.setattr(
        workflow,
        "run_risk_combo_sweep",
        lambda **_kwargs: {"status": "ok", "aggregate": {}, "report_path": str(tmp_path / "sweep.json")},
    )
    monkeypatch.setattr(
        workflow,
        "build_risk_combo_matrix_report",
        lambda **_kwargs: {
            "report_path": str(tmp_path / "matrix.json"),
            "surface_count": 1,
            "promising_surface_count": 1,
            "best_surface": {"symbol": "TRXUSDT"},
        },
    )
    monkeypatch.setattr(
        workflow,
        "run_ai_readiness_scan",
        lambda **_kwargs: {
            "candidate_count": 1,
            "allowed_count": 0,
            "execution_ticket": None,
            "next_machine_action": "wait_for_market_state",
            "hard_blocker_taxonomy": {"market_state": ["Volume z-score below floor."]},
            "denial_journal_path": str(tmp_path / "denials.jsonl"),
            "denial_journal_count": 1,
            "research_candidate_report": {
                "near_ready_count": 1,
                "top_candidates": [
                    {
                        "symbol": "TRXUSDT",
                        "side": "BUY",
                        "interval": "1d",
                        "near_ready_market_only": True,
                    }
                ],
                "near_ready_candidates": [
                    {
                        "symbol": "TRXUSDT",
                        "side": "BUY",
                        "interval": "1d",
                        "near_ready_market_only": True,
                    }
                ],
            },
            "report_path": str(tmp_path / "readiness.json"),
        },
    )

    payload = workflow.run_new_symbol_workflow(
        symbols=["TRXUSDT"],
        intervals=["1d"],
        sides=["BUY"],
        plan_only=False,
        output_dir=tmp_path,
    )

    assert payload["outcome"] == "near_ready_market_only"
    assert payload["status_counts"]["near_ready_market_only"] == 1
    symbol = payload["symbols"][0]
    assert symbol["near_ready_candidate_keys"] == ["TRXUSDT:BUY:1d"]
    assert symbol["next_command"] == (
        "openclaw-quantctl live-readiness --symbol TRXUSDT --side BUY --interval 1d "
        "--execution-mode testnet_exploration --compact"
    )
    assert payload["readiness"]["allowed_count"] == 0
    assert payload["promotion_boundary"]["testnet_requires_execution_ticket"] is True
