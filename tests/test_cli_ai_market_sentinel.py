from __future__ import annotations

from argparse import Namespace

import binance_quant_control.cli as cli


def test_cmd_ai_market_sentinel_compact_is_machine_readable(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "run_ai_market_sentinel",
        lambda **_kwargs: {
            "mode": "ai_market_sentinel_v1",
            "safety": {"opens_orders": False},
            "trading_control": {"paused": True},
            "position_state": {"open_position_count": 1},
            "trend_state": {"BTCUSDT": {"bias": "short"}},
            "route_risk": {"active_quarantined_routes": ["btc-core"]},
            "readiness": {"allowed_count": 0},
            "expansion_gate": {"allowed": False},
            "conditional_order_alert": {"should_notify": False, "reason": "no-readiness-approved-candidate"},
            "telegram": {"sent": False, "reason": "send_telegram disabled"},
            "machine_action_queue": [{"action": "run_position_guardian"}],
            "errors": [],
            "report_path": "state/ai-market-sentinel/report.json",
        },
    )
    monkeypatch.setattr(
        cli,
        "print_json",
        lambda payload, compact=False: captured.update({"payload": payload, "compact": compact}),
    )

    cli.cmd_ai_market_sentinel(
        Namespace(
            symbols="BTCUSDT,ETHUSDT",
            interval="15m",
            limit=160,
            market="futures",
            strategy_config="config/strategy-live-pilot.yaml",
            blueprint_config="config/professional-system-blueprint.default.yaml",
            output_dir="",
            skip_readiness=False,
            send_telegram=False,
            max_readiness_candidates=6,
            compact=True,
        )
    )

    assert captured["compact"] is True
    payload = captured["payload"]  # type: ignore[assignment]
    assert payload["mode"] == "ai_market_sentinel_v1"  # type: ignore[index]
    assert payload["position_state"]["open_position_count"] == 1  # type: ignore[index]
    assert payload["conditional_order_alert"]["should_notify"] is False  # type: ignore[index]
    assert payload["telegram"]["sent"] is False  # type: ignore[index]
    assert payload["machine_action_queue"][0]["action"] == "run_position_guardian"  # type: ignore[index]


def test_cmd_hermes_trade_compact_includes_market_sentinel(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "run_hermes_trade_cycle",
        lambda **_kwargs: {
            "status": "managing_position",
            "state_after": {"enabled": True},
            "steps": {
                "market_sentinel": {
                    "response": {
                        "position_state": {"open_position_count": 1},
                        "expansion_gate": {"allowed": False},
                        "machine_action_queue": [{"action": "run_position_guardian"}],
                        "report_path": "state/sentinel.json",
                    }
                }
            },
        },
    )
    monkeypatch.setattr(
        cli,
        "print_json",
        lambda payload, compact=False: captured.update({"payload": payload, "compact": compact}),
    )

    cli.cmd_hermes_trade(
        Namespace(
            action="cycle",
            config="config/hermes-trade-loop.default.yaml",
            force=True,
            dry_run_only=True,
            set_execution_mode=True,
            compact=True,
        )
    )

    payload = captured["payload"]  # type: ignore[assignment]
    assert payload["market_sentinel"]["position_state"]["open_position_count"] == 1  # type: ignore[index]
    assert payload["market_sentinel"]["machine_action_queue"][0]["action"] == "run_position_guardian"  # type: ignore[index]
