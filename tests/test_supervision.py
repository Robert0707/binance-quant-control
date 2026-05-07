from __future__ import annotations

from types import SimpleNamespace

import binance_quant_control.supervision as supervision


def test_route_risk_snapshot_quarantines_weak_route() -> None:
    reviews = [
        {"route_id": "btc-core", "realized_pnl_usdt": -1.0},
        {"route_id": "btc-core", "realized_pnl_usdt": -0.5},
        {"route_id": "btc-core", "realized_pnl_usdt": 0.2},
        {"route_id": "btc-core", "realized_pnl_usdt": -0.3},
        {"route_id": "btc-core", "realized_pnl_usdt": -0.2},
        {"route_id": "btc-core", "realized_pnl_usdt": -0.1},
        {"route_id": "btc-core", "realized_pnl_usdt": -0.4},
        {"route_id": "btc-core", "realized_pnl_usdt": -0.3},
        {"route_id": "btc-core", "realized_pnl_usdt": -0.2},
        {"route_id": "btc-core", "realized_pnl_usdt": -0.1},
    ]

    snapshot = supervision._route_risk_snapshot(
        reviews,
        route_lookback=20,
        min_route_profit_factor=0.8,
        max_route_loss_streak=5,
    )

    assert snapshot["quarantined_routes"] == ["btc-core"]
    assert snapshot["routes"]["btc-core"]["quarantined"] is True


def test_build_supervisor_policy_normalizes_symbols(monkeypatch) -> None:
    monkeypatch.setattr(
        supervision,
        "resolve_symbol_route",
        lambda symbol: SimpleNamespace(route_id="route", symbol=symbol),
    )

    policy = supervision.build_supervisor_policy(
        cycles=0,
        training_rounds=0,
        symbols=["btc", " eth "],
    )

    assert policy.cycles == 1
    assert policy.training_rounds == 1
    assert policy.symbols == ("BTC", "ETH")


def test_delivery_supervisor_uses_paper_demo_only(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(supervision, "SUPERVISION_STATE_DIR", tmp_path / "supervision")
    monkeypatch.setattr(supervision, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        supervision,
        "load_settings",
        lambda: SimpleNamespace(use_testnet=True, live_trading_enabled=False),
    )
    monkeypatch.setattr(supervision, "build_digest", lambda config: {"decision": {}, "news": {"risk": {}}})
    monkeypatch.setattr(supervision, "load_digest_config", lambda path: {})
    monkeypatch.setattr(
        supervision,
        "run_trading_mission",
        lambda **kwargs: {
            "report_path": "/tmp/mission.json",
            "mission_actions": ["paper-order-recorded"],
            "selected_candidate": {"symbol": "BTCUSDT"},
            "simulation": {"status": "recorded_for_market_validation"},
            "system_findings": [],
        },
    )
    monkeypatch.setattr(
        supervision,
        "_run_training_command",
        lambda policy: {
            "returncode": 0,
            "response": {"recorded_review_count": 1, "wins": 1, "losses": 0},
            "stderr_tail": "",
        },
    )
    monkeypatch.setattr(
        supervision,
        "run_strategy_optimizer",
        lambda: {
            "status": "ok",
            "review_count": 1,
            "promotion_decision": "reject",
            "report_path": "/tmp/optimizer.json",
        },
    )
    monkeypatch.setattr(
        supervision,
        "evaluate_optimizer_live_gate",
        lambda: {"allowed": False, "promotion_decision": "reject", "reasons": ["not promoted"]},
    )
    monkeypatch.setattr(supervision, "read_closed_trade_reviews", lambda: [])
    monkeypatch.setattr(supervision, "read_paper_orders", lambda: [])
    monkeypatch.setattr(
        supervision,
        "update_route_quarantine_from_snapshot",
        lambda snapshot, updated_by: {"active_quarantined_routes": []},
    )

    payload = supervision.run_delivery_supervisor(
        supervision.SupervisorPolicy(
            cycles=1,
            training_rounds=1,
            symbols=("BTCUSDT",),
            mission_symbols_per_cycle=1,
            target_return_pct=5.0,
            max_leverage=3.0,
            margin_notional_usdt=3.0,
            optimize_every=1,
            max_recent_loss_usdt=5.0,
            max_route_loss_streak=5,
            min_route_profit_factor=0.8,
            route_lookback=40,
            build_digest_every=1,
            audit_every=0,
            stop_on_optimizer_promotion=True,
        )
    )

    assert payload["status"] == "ok"
    assert payload["live_guardrail"]["real_orders_sent_by_supervisor"] is False
    assert payload["cycles"][0]["mission"]["actions"] == ["paper-order-recorded"]
