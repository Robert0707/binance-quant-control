from __future__ import annotations

from argparse import Namespace

import binance_quant_control.cli as cli
import binance_quant_control.trade_console as console


def _settings() -> object:
    return type(
        "Settings",
        (),
        {
            "use_testnet": True,
            "live_trading_enabled": False,
            "testnet_trading_enabled": True,
        },
    )()


def test_trade_console_snapshot_combines_dashboard_session_and_equity(monkeypatch) -> None:
    monkeypatch.setattr(console, "build_operator_dashboard", lambda settings: {
        "status": "ok",
        "customer_summary": {"open_position_count": 1, "open_unrealized_pnl_usdt": 0.25},
        "positions": [{"symbol": "TRXUSDT"}],
        "protective_orders": [{"symbol": "TRXUSDT", "coverage": "ok"}],
        "execution_journal": {"record_count": 1},
        "product_readiness": {"status": "blocked"},
        "candidate_pool": {"readiness_allowed_count": 1},
        "risk_combo_matrix": {"status": "promising_research_only"},
        "operator_feedback": ["hold until ticket exists"],
        "report_path": "state/operator-dashboard/report.json",
    })
    monkeypatch.setattr(console, "trade_session_status", lambda: {"status": "enabled"})
    monkeypatch.setattr(console, "hermes_trade_status", lambda: {"status": "enabled"})
    monkeypatch.setattr(console, "read_closed_trade_reviews", lambda: [
        {
            "closed_at": "2026-01-01T00:00:00+00:00",
            "symbol": "TRXUSDT",
            "side": "BUY",
            "realized_pnl_usdt": 1.5,
            "exit_reason": "take_profit",
        },
        {
            "closed_at": "2026-01-01T01:00:00+00:00",
            "symbol": "WIFUSDT",
            "side": "BUY",
            "realized_pnl_usdt": -0.5,
            "exit_reason": "stop_loss",
        },
    ])

    payload = console.build_trade_console_snapshot(_settings())

    assert payload["mode"]["use_testnet"] is True
    assert payload["mode"]["mainnet_live_allowed"] is False
    assert payload["dashboard"]["candidate_pool"]["readiness_allowed_count"] == 1
    assert payload["dashboard"]["positions"][0]["symbol"] == "TRXUSDT"
    assert payload["trade_session"]["status"] == "enabled"
    assert payload["hermes_trade"]["status"] == "enabled"
    assert payload["equity_curve"][-1]["cumulative_pnl_usdt"] == 1.0
    assert payload["controls"]["close_position"]["requires"] == ["symbol", "confirm=true"]


def test_trade_console_close_position_requires_confirm() -> None:
    payload = console.close_position_from_console(symbol="TRXUSDT", confirm=False, settings=_settings())

    assert payload["status"] == "blocked"
    assert payload["reason"] == "close-position-requires-confirm=true"


def test_trade_console_close_position_submits_reduce_only_market_close(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, settings):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def positions(self, symbol):
            return [{"symbol": symbol, "positionAmt": "12.5"}]

        def new_order(self, symbol, side, order_type, **kwargs):
            calls.append({"symbol": symbol, "side": side, "order_type": order_type, **kwargs})
            return {"status": "NEW"}

        def cancel_all_algo_orders(self, symbol):
            return [{"symbol": symbol, "status": "cancelled"}]

    monkeypatch.setattr(console, "BinanceClient", lambda settings: FakeClient(settings))

    payload = console.close_position_from_console(symbol="TRXUSDT", confirm=True, settings=_settings())

    assert payload["status"] == "submitted"
    assert payload["side"] == "SELL"
    assert calls == [
        {
            "symbol": "TRXUSDT",
            "side": "SELL",
            "order_type": "MARKET",
            "quantity": 12.5,
            "reduce_only": True,
            "market": "futures",
        }
    ]


def test_cmd_trade_console_builds_server_with_requested_controls(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeServer:
        def serve_forever(self):
            captured["served"] = True

        def server_close(self):
            captured["closed"] = True

    def fake_server(config):
        captured["config"] = config
        return FakeServer()

    monkeypatch.setattr(cli, "run_trade_console_server", fake_server)
    monkeypatch.setattr(
        cli,
        "print_json",
        lambda payload, compact=False: captured.update({"payload": payload, "compact": compact}),
    )

    cli.cmd_trade_console(
        Namespace(host="127.0.0.1", port=9999, allow_order_actions=True, compact=True)
    )

    config = captured["config"]
    assert config.host == "127.0.0.1"
    assert config.port == 9999
    assert config.allow_order_actions is True
    assert captured["payload"]["url"] == "http://127.0.0.1:9999/"
    assert captured["payload"]["mainnet_live_allowed"] is False
    assert captured["served"] is True
    assert captured["closed"] is True
