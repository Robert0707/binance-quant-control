from __future__ import annotations

from argparse import Namespace

import binance_quant_control.cli as cli


def test_cmd_ai_readiness_scan_compact_summarizes_results(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "run_ai_readiness_scan",
        lambda **_kwargs: {
            "mode": "hermes_ai_readiness_scanner_v1",
            "safety": {"opens_orders": False},
            "candidate_count": 2,
            "scanned_count": 2,
            "allowed_count": 1,
            "selected_ready_candidate": {
                "rank": 2,
                "symbol": "ETHUSDT",
                "side": "BUY",
                "route_id": "eth-core",
                "next_action": "execute_ready_dry_run_only",
                "live_plan": {"violations": []},
            },
            "ready_after_global_unlock_count": 1,
            "selected_after_global_unlock": {
                "rank": 1,
                "symbol": "DOGEUSDT",
                "side": "BUY",
                "route_id": "doge-meme-high-beta",
                "next_action_after_global_unlock": "execute_ready_dry_run_only",
            },
            "next_machine_action": "execute_ready_dry_run_only",
            "machine_action_queue": [{"action": "execute_ready_dry_run_only", "candidates": ["ETHUSDT:BUY"]}],
            "execution_ticket": {
                "state": "ready_for_operator_testnet_execution",
                "symbol": "ETHUSDT",
                "side": "BUY",
                "opens_orders": False,
                "operator_testnet_execute_command": ".venv/bin/binance-quant-control live-pilot --symbol ETHUSDT --execute",
            },
            "research_candidate_report": {
                "mode": "research_candidate_report_v1",
                "candidate_count": 2,
                "reviewable_candidate_count": 2,
                "trade_allowed_count": 1,
                "promotion_boundary": {"mainnet_live_allowed": False},
            },
            "hard_blocker_taxonomy": {
                "kill_switch": ["Trading is paused by kill-switch."],
                "market_state": ["Volume z-score is below floor."],
            },
            "denial_journal_path": "state/hermes-readiness-scan/readiness-denials.jsonl",
            "denial_journal_count": 1,
            "scan_results": [
                {
                    "rank": 1,
                    "symbol": "DOGEUSDT",
                    "side": "BUY",
                    "route_id": "doge-meme-high-beta",
                    "allowed": False,
                    "next_action": "wait_for_kill_switch_clear",
                    "blocker_taxonomy": {"kill_switch": ["Trading is paused by kill-switch."]},
                    "live_plan": {
                        "analysis_score": 69,
                        "analysis_convergence": 1.0,
                        "planned_account_risk_pct": 0.000035,
                        "violations": ["this should not appear in compact"],
                    },
                },
                {
                    "rank": 2,
                    "symbol": "ETHUSDT",
                    "side": "BUY",
                    "route_id": "eth-core",
                    "allowed": True,
                    "next_action": "execute_ready_dry_run_only",
                    "blocker_taxonomy": {},
                    "live_plan": {
                        "analysis_score": 78,
                        "analysis_convergence": 1.0,
                        "planned_account_risk_pct": 0.000039,
                    },
                },
            ],
            "hermes_ai_trader_report": "state/hermes.json",
            "report_path": "state/scan.json",
        },
    )
    monkeypatch.setattr(
        cli,
        "print_json",
        lambda payload, compact=False: captured.update({"payload": payload, "compact": compact}),
    )

    cli.cmd_ai_readiness_scan(
        Namespace(
            blueprint_config="config/professional-system-blueprint.default.yaml",
            strategy_config="config/strategy-live-pilot.yaml",
            output_dir="",
            market="futures",
            limit=0,
            margin_notional_usdt=0.0,
            execution_mode="testnet_exploration",
            max_candidates=0,
            compact=True,
        )
    )

    assert captured["compact"] is True
    payload = captured["payload"]  # type: ignore[assignment]
    assert payload["allowed_count"] == 1  # type: ignore[index]
    assert payload["selected_ready_candidate"]["symbol"] == "ETHUSDT"  # type: ignore[index]
    assert payload["ready_after_global_unlock_count"] == 1  # type: ignore[index]
    assert payload["selected_after_global_unlock"]["symbol"] == "DOGEUSDT"  # type: ignore[index]
    assert payload["execution_ticket"]["symbol"] == "ETHUSDT"  # type: ignore[index]
    assert payload["research_candidate_report"]["candidate_count"] == 2  # type: ignore[index]
    assert payload["research_candidate_report"]["promotion_boundary"]["mainnet_live_allowed"] is False  # type: ignore[index]
    assert payload["machine_action_queue"][0]["action"] == "execute_ready_dry_run_only"  # type: ignore[index]
    assert payload["hard_blocker_classes"] == ["kill_switch", "market_state"]  # type: ignore[index]
    assert payload["denial_journal_count"] == 1  # type: ignore[index]
    assert payload["denial_journal_path"] == "state/hermes-readiness-scan/readiness-denials.jsonl"  # type: ignore[index]
    assert payload["scan_summary"][0]["blocker_classes"] == ["kill_switch"]  # type: ignore[index]
    assert "live_plan" not in payload["scan_summary"][0]  # type: ignore[index]
