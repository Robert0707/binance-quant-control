from __future__ import annotations

from pathlib import Path

import binance_quant_control.operator_dashboard as dashboard
from binance_quant_control.config import Settings


def _settings() -> Settings:
    return Settings(
        use_testnet=True,
        live_trading_enabled=False,
        testnet_trading_enabled=True,
        recv_window_ms=5000,
        default_symbol="BTCUSDT",
        default_market="futures",
        binance_api_key="",
        binance_secret_key="",
        binance_testnet_api_key="key",
        binance_testnet_secret_key="secret",
        blave_api_key="",
        blave_secret_key="",
        whale_alert_api_key="",
        max_leverage=5,
        max_notional_pct=0.5,
        max_daily_trades=5,
        min_balance_usdt=2.0,
        min_convergence=0.6,
        cooldown_hours=4.0,
    )


class FakeClient:
    def __init__(self, settings):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def positions(self):
        return [
            {
                "symbol": "APTUSDT",
                "positionAmt": "30",
                "entryPrice": "1.0",
                "markPrice": "1.01",
                "unRealizedProfit": "0.3",
                "leverage": "5",
            }
        ]

    def open_algo_orders(self, symbol):
        return [
            {"orderType": "STOP_MARKET", "quantity": "30", "triggerPrice": "0.95"},
            {"orderType": "TAKE_PROFIT_MARKET", "quantity": "9", "triggerPrice": "1.03"},
            {"orderType": "TAKE_PROFIT_MARKET", "quantity": "16.5", "triggerPrice": "1.08"},
        ]

    def exchange_info(self, symbol, market):
        return {
            "symbols": [
                {
                    "symbol": symbol,
                    "filters": [
                        {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.1"},
                    ],
                }
            ]
        }


def test_operator_dashboard_builds_customer_feedback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dashboard, "OPERATOR_DASHBOARD_DIR", tmp_path)
    monkeypatch.setattr(dashboard, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(dashboard, "BinanceClient", lambda settings: FakeClient(settings))
    monkeypatch.setattr(
        dashboard,
        "summarize_closed_trade_reviews",
        lambda: {"count": 12, "total_realized_pnl_usdt": -2.5},
    )
    monkeypatch.setattr(dashboard, "summarize_live_orders", lambda: {"count": 3})
    monkeypatch.setattr(
        dashboard,
        "read_live_orders",
        lambda: [
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "symbol": "APTUSDT",
                "side": "BUY",
                "leverage": 5,
                "notional_usdt": 6.0,
                "gross_notional_usdt": 30.0,
                "route_id": "major-alt-trend",
                "analysis_score": 100,
                "analysis_convergence": 0.9,
            }
        ],
    )
    monkeypatch.setattr(
        dashboard,
        "run_loss_diagnostics",
        lambda min_bucket_trades, top_n: {
            "summary": {"count": 12, "profit_factor": 0.7},
            "findings": [
                "stop-loss-dominant: stop_loss_ratio=70.0%",
                "fast-stop-cluster:major-alt-trend/BUY/hold-lt-1h stop_loss_ratio=80.0% PF=0.4",
            ],
            "worst_buckets": [],
            "root_cause_recommendations": [
                {"type": "fast-stop-cluster", "action": "require-stronger-entry-confirmation"}
            ],
            "report_path": str(tmp_path / "loss.json"),
        },
    )
    monkeypatch.setattr(
        dashboard,
        "_load_latest_digest_summary",
        lambda: {
            "available": True,
            "news": {"risk": "high", "bias": "bearish", "high_impact_count": 2},
            "whale": {"enabled": False, "available": False, "reason": "WHALE_ALERT_API_KEY not configured"},
            "decision": {"action": "watchlist_only", "selected": {"symbol": "BTCUSDT"}},
        },
    )

    payload = dashboard.build_operator_dashboard(_settings())

    assert payload["customer_summary"]["open_position_count"] == 1
    assert payload["customer_summary"]["open_unrealized_pnl_usdt"] == 0.3
    assert payload["protective_orders"][0]["coverage"] == "ok"
    assert payload["protective_orders"][0]["take_profit_ladder"]["quantities"] == [9.0, 16.5]
    assert payload["protective_orders"][0]["take_profit_ladder"]["first_tp_ratio"] == 0.3
    assert payload["protective_orders"][0]["take_profit_ladder"]["runner_quantity"] == 4.5
    assert any("profitable" in item for item in payload["operator_feedback"])
    assert any("whale wallet" in item for item in payload["operator_feedback"])
    assert any("stopped quickly" in item for item in payload["operator_feedback"])
    assert payload["external_context_automation"]["available"] is True
    assert payload["loss_diagnostics"]["root_cause_recommendations"][0]["type"] == "fast-stop-cluster"
    assert Path(payload["report_path"]).exists()


class BadTpClient(FakeClient):
    def open_algo_orders(self, symbol):
        return [
            {"orderType": "STOP_MARKET", "quantity": "30", "triggerPrice": "0.95"},
            {"orderType": "TAKE_PROFIT_MARKET", "quantity": "30", "triggerPrice": "1.03"},
        ]


def test_operator_dashboard_flags_full_position_tp1(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dashboard, "OPERATOR_DASHBOARD_DIR", tmp_path)
    monkeypatch.setattr(dashboard, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(dashboard, "BinanceClient", lambda settings: BadTpClient(settings))
    monkeypatch.setattr(
        dashboard,
        "summarize_closed_trade_reviews",
        lambda: {"count": 12, "total_realized_pnl_usdt": -2.5},
    )
    monkeypatch.setattr(dashboard, "summarize_live_orders", lambda: {"count": 3})
    monkeypatch.setattr(dashboard, "read_live_orders", lambda: [])
    monkeypatch.setattr(
        dashboard,
        "run_loss_diagnostics",
        lambda min_bucket_trades, top_n: {
            "summary": {"count": 12, "profit_factor": 0.7},
            "findings": ["stop-loss-dominant: stop_loss_ratio=70.0%"],
            "worst_buckets": [],
            "root_cause_recommendations": [],
            "report_path": str(tmp_path / "loss.json"),
        },
    )
    monkeypatch.setattr(dashboard, "_load_latest_digest_summary", lambda: {"available": False})

    payload = dashboard.build_operator_dashboard(_settings())

    ladder = payload["protective_orders"][0]["take_profit_ladder"]
    assert payload["protective_orders"][0]["coverage"] == "attention"
    assert ladder["status"] == "attention"
    assert "tp1_full_position" in ladder["issues"]


class MicroFullTpClient(FakeClient):
    def positions(self):
        return [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.001",
                "entryPrice": "80883.5",
                "markPrice": "81000.0",
                "unRealizedProfit": "0.1165",
                "leverage": "3",
            }
        ]

    def open_algo_orders(self, symbol):
        return [
            {"orderType": "STOP_MARKET", "quantity": "0.001", "triggerPrice": "79536.5"},
            {"orderType": "TAKE_PROFIT_MARKET", "quantity": "0.001", "triggerPrice": "86655.9"},
        ]

    def exchange_info(self, symbol, market):
        return {
            "symbols": [
                {
                    "symbol": symbol,
                    "filters": [
                        {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.001"},
                    ],
                }
            ]
        }


def test_operator_dashboard_allows_full_tp_for_minimum_step_micro_position(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dashboard, "OPERATOR_DASHBOARD_DIR", tmp_path)
    monkeypatch.setattr(dashboard, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(dashboard, "BinanceClient", lambda settings: MicroFullTpClient(settings))
    monkeypatch.setattr(
        dashboard,
        "summarize_closed_trade_reviews",
        lambda: {"count": 12, "total_realized_pnl_usdt": -2.5},
    )
    monkeypatch.setattr(dashboard, "summarize_live_orders", lambda: {"count": 3})
    monkeypatch.setattr(dashboard, "read_live_orders", lambda: [])
    monkeypatch.setattr(
        dashboard,
        "run_loss_diagnostics",
        lambda min_bucket_trades, top_n: {
            "summary": {"count": 12, "profit_factor": 0.7},
            "findings": ["stop-loss-dominant: stop_loss_ratio=70.0%"],
            "worst_buckets": [],
            "root_cause_recommendations": [],
            "report_path": str(tmp_path / "loss.json"),
        },
    )
    monkeypatch.setattr(dashboard, "_load_latest_digest_summary", lambda: {"available": False})

    payload = dashboard.build_operator_dashboard(_settings())

    ladder = payload["protective_orders"][0]["take_profit_ladder"]
    assert payload["protective_orders"][0]["coverage"] == "ok"
    assert ladder["micro_full_tp_fallback"] is True
    assert ladder["issues"] == []
