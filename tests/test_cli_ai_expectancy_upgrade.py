from __future__ import annotations

from argparse import Namespace

import binance_quant_control.cli as cli


def test_cmd_ai_expectancy_upgrade_compact_is_machine_readable(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "run_ai_expectancy_upgrade",
        lambda **_kwargs: {
            "mode": "ai_expectancy_upgrade_v1",
            "safety": {"opens_orders": False, "mainnet_live_allowed": False},
            "objective": "increase fixed-risk expectancy",
            "symbol_allocation": {"mode": "liquidity_plus_expectancy_symbol_allocator"},
            "selected_symbols": {"btc_eth": ["BTCUSDT", "ETHUSDT"]},
            "hermes_machine_strategy": {"mode": "ai_trader_machine_strategy_router"},
            "steps": [{"priority": "P1", "status": "blocked"}],
            "readiness_scan": {"allowed_count": 0},
            "final_machine_decision": {"next_surface": "continue_expectancy_research"},
            "report_path": "state/ai-expectancy-upgrade/report.json",
        },
    )
    monkeypatch.setattr(
        cli,
        "print_json",
        lambda payload, compact=False: captured.update({"payload": payload, "compact": compact}),
    )

    cli.cmd_ai_expectancy_upgrade(
        Namespace(
            output_dir="",
            symbols="BTCUSDT,ETHUSDT",
            discovery_symbols="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,DOGEUSDT,TRXUSDT",
            limit=8000,
            sweep_limit=5000,
            max_configs=80,
            max_walk_forward_validations=12,
            universe_limit=12,
            max_readiness_candidates=6,
            readiness_execution_mode="testnet_exploration",
            dry_run=False,
            compact=True,
        )
    )

    assert captured["compact"] is True
    payload = captured["payload"]  # type: ignore[assignment]
    assert payload["mode"] == "ai_expectancy_upgrade_v1"  # type: ignore[index]
    assert payload["symbol_allocation"]["mode"] == "liquidity_plus_expectancy_symbol_allocator"  # type: ignore[index]
    assert payload["machine_strategy"]["mode"] == "ai_trader_machine_strategy_router"  # type: ignore[index]
    assert payload["steps"][0]["priority"] == "P1"  # type: ignore[index]
    assert payload["readiness_scan"]["allowed_count"] == 0  # type: ignore[index]
    assert payload["final_machine_decision"]["next_surface"] == "continue_expectancy_research"  # type: ignore[index]


def test_cmd_ai_goal_loop_compact_is_machine_readable(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "run_ai_goal_loop",
        lambda **_kwargs: {
            "mode": "ai_goal_loop_v1",
            "goal": "maximize_stable_expectancy",
            "safety": {"opens_orders": False, "mainnet_live_allowed": False},
            "score": {"score_10": 8.6},
            "readiness_sizing_scout": {"selected_margin_notional_usdt": 50.0},
            "next_machine_action": "repair_exchange_sizing_or_margin",
            "recommended_commands": ["repair_exchange_sizing_or_margin"],
            "reports": {"goal_loop": "state/ai-goal-loop/report.json"},
        },
    )
    monkeypatch.setattr(
        cli,
        "print_json",
        lambda payload, compact=False: captured.update({"payload": payload, "compact": compact}),
    )

    cli.cmd_ai_goal_loop(
        Namespace(
            output_dir="",
            goal="maximize_stable_expectancy",
            symbols="BTCUSDT,ETHUSDT",
            discovery_symbols="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,DOGEUSDT,TRXUSDT",
            limit=120,
            sweep_limit=280,
            max_configs=2,
            max_walk_forward_validations=1,
            universe_limit=8,
            max_readiness_candidates=6,
            margin_notional_usdt=25.0,
            smoke=True,
            compact=True,
        )
    )

    assert captured["compact"] is True
    payload = captured["payload"]  # type: ignore[assignment]
    assert payload["mode"] == "ai_goal_loop_v1"  # type: ignore[index]
    assert payload["next_machine_action"] == "repair_exchange_sizing_or_margin"  # type: ignore[index]
    assert payload["score"]["score_10"] == 8.6  # type: ignore[index]
    assert payload["readiness_sizing_scout"]["selected_margin_notional_usdt"] == 50.0  # type: ignore[index]
