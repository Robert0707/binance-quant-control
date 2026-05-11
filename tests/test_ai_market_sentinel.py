from __future__ import annotations

from pathlib import Path

import binance_quant_control.ai_market_sentinel as sentinel
from binance_quant_control.trading_control import TradingControlState


class FakeSettings:
    pass


class FakeClient:
    def __init__(self, settings):
        self.settings = settings

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def positions(self, symbol=None):
        return [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.001",
                "entryPrice": "80000",
                "unRealizedProfit": "1.25",
                "leverage": "3",
            }
        ]

    def klines(self, symbol, interval, limit, market):
        rows = []
        price = 79000.0
        for idx in range(limit):
            price += 100.0
            rows.append(
                [
                    idx,
                    str(price - 50.0),
                    str(price + 100.0),
                    str(price - 100.0),
                    str(price),
                    "1000",
                    idx + 1,
                    "0",
                    1,
                    "0",
                    "0",
                    "0",
                ]
            )
        return rows


def test_ai_market_sentinel_prioritizes_position_guardian_and_blocks_expansion(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sentinel, "STATE_DIR", tmp_path)
    monkeypatch.setattr(sentinel, "load_settings", lambda: FakeSettings())
    monkeypatch.setattr(sentinel, "BinanceClient", FakeClient)
    monkeypatch.setattr(
        sentinel,
        "load_trading_control_state",
        lambda: TradingControlState(paused=True, reason="protect exposure", updated_by="pytest"),
    )
    monkeypatch.setattr(
        sentinel,
        "load_route_risk_state",
        lambda: {
            "active_quarantined_routes": ["btc-core"],
            "routes": {
                "btc-core": {
                    "quarantined": True,
                    "reasons": ["profit-factor below floor"],
                    "metrics": {"profit_factor": 0.37},
                }
            },
        },
    )
    monkeypatch.setattr(
        sentinel,
        "run_ai_readiness_scan",
        lambda **kwargs: {
            "candidate_count": 3,
            "allowed_count": 0,
            "next_machine_action": "repair_exchange_sizing_or_margin",
            "hard_blocker_taxonomy": {"strategy_performance": ["Recent expectancy -0.16R."]},
            "report_path": "state/readiness.json",
        },
    )

    payload = sentinel.run_ai_market_sentinel(symbols=["BTCUSDT"], limit=80, output_dir=tmp_path)

    assert payload["safety"]["opens_orders"] is False
    assert payload["position_state"]["open_position_count"] == 1
    assert payload["readiness"]["allowed_count"] == 0
    assert payload["expansion_gate"]["allowed"] is False
    assert "trading-control-paused" in payload["expansion_gate"]["blockers"]
    assert "no-readiness-approved-candidate" in payload["expansion_gate"]["blockers"]
    assert payload["machine_action_queue"][0]["action"] == "run_position_guardian"
    assert payload["trend_state"]["BTCUSDT"]["bias"] == "long"
    assert Path(payload["report_path"]).exists()


def test_ai_market_sentinel_allows_expansion_when_positions_below_cap(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sentinel, "STATE_DIR", tmp_path)
    monkeypatch.setattr(sentinel, "load_settings", lambda: FakeSettings())
    monkeypatch.setattr(sentinel, "BinanceClient", FakeClient)
    monkeypatch.setattr(sentinel, "load_trading_control_state", lambda: TradingControlState())
    monkeypatch.setattr(sentinel, "load_route_risk_state", lambda: {"active_quarantined_routes": [], "routes": {}})
    monkeypatch.setattr(
        sentinel,
        "run_ai_readiness_scan",
        lambda **kwargs: {
            "candidate_count": 3,
            "allowed_count": 1,
            "selected_ready_candidate": {
                "symbol": "SOLUSDT",
                "side": "BUY",
                "route_id": "sol-core",
                "strategy_family": "trend",
                "live_plan": {
                    "price": 180.0,
                    "stop_price": 174.0,
                    "take_profit_prices": [186.0, 192.0, 204.0],
                    "take_profit_quantities": [0.1, 0.1, 0.1],
                    "take_profit_weights": [0.3, 0.35, 0.35],
                    "take_profit_runner_quantity": 0.0,
                    "leverage": 3,
                    "quantity": 0.3,
                    "margin_notional_usdt": 18.0,
                    "gross_notional_usdt": 54.0,
                    "planned_account_risk_pct": 0.018,
                    "analysis_score": 72,
                    "analysis_convergence": 0.7,
                    "sizing": {"signal_scores": {"composite_quality_score": 71.5}},
                    "professional_entry_gate": {
                        "layers": {
                            "strategy_performance": {
                                "profit_factor": 1.24,
                                "expectancy_r": 0.08,
                                "payoff_ratio": 1.3,
                            }
                        }
                    },
                },
            },
            "next_machine_action": "execute_ready_dry_run_only",
            "hard_blocker_taxonomy": {},
            "denial_journal_path": "state/hermes-readiness-scan/readiness-denials.jsonl",
            "denial_journal_count": 0,
            "execution_ticket": {
                "symbol": "SOLUSDT",
                "side": "BUY",
                "market": "futures",
                "interval": "15m",
                "risk_snapshot": {
                    "price": 180.0,
                    "leverage": 3,
                    "margin_notional_usdt": 18.0,
                    "gross_notional_usdt": 54.0,
                    "planned_account_risk_pct": 0.018,
                    "analysis_score": 72,
                    "analysis_convergence": 0.7,
                },
                "expectancy_evidence": {
                    "profit_factor": 1.24,
                    "expectancy_r": 0.08,
                    "payoff_ratio": 1.3,
                },
                "preflight_command": "openclaw-quantctl live-readiness --symbol SOLUSDT --compact",
                "operator_testnet_execute_command": "openclaw-quantctl live-pilot --symbol SOLUSDT --execute --compact",
            },
            "report_path": "state/readiness.json",
        },
    )

    payload = sentinel.run_ai_market_sentinel(symbols=["BTCUSDT"], limit=80, output_dir=tmp_path)

    assert payload["position_state"]["open_position_count"] == 1
    assert payload["expansion_gate"]["allowed"] is True
    assert "open-position-management-priority" not in payload["expansion_gate"]["blockers"]
    assert payload["machine_action_queue"][0]["action"] == "run_position_guardian"
    assert any(item["action"] == "operator_testnet_preflight" for item in payload["machine_action_queue"])
    assert payload["readiness"]["denial_journal_count"] == 0
    alert = payload["conditional_order_alert"]
    assert alert["should_notify"] is True
    assert alert["symbol"] == "SOLUSDT"
    assert alert["action"] == "做多"
    assert alert["condition_entry_price"] == 180.0
    assert alert["stop_loss_price"] == 174.0
    assert alert["take_profit_prices"] == [186.0, 192.0, 204.0]
    assert alert["max_safe_leverage"] == 3
    assert alert["execution_boundary"] == "notification_only_no_orders_sent"
    assert "AI Trader 條件單候選" in payload["telegram_text"]
    assert "條件進場價: 180" in payload["telegram_text"]


def test_ai_market_sentinel_blocks_expansion_at_four_positions(monkeypatch, tmp_path: Path) -> None:
    class FourPositionClient(FakeClient):
        def positions(self, symbol=None):
            return [
                {"symbol": "BTCUSDT", "positionAmt": "0.001", "entryPrice": "80000", "unRealizedProfit": "1", "leverage": "3"},
                {"symbol": "ETHUSDT", "positionAmt": "0.01", "entryPrice": "3000", "unRealizedProfit": "1", "leverage": "3"},
                {"symbol": "SOLUSDT", "positionAmt": "0.1", "entryPrice": "180", "unRealizedProfit": "1", "leverage": "3"},
                {"symbol": "BNBUSDT", "positionAmt": "0.1", "entryPrice": "600", "unRealizedProfit": "1", "leverage": "3"},
            ]

    monkeypatch.setattr(sentinel, "STATE_DIR", tmp_path)
    monkeypatch.setattr(sentinel, "load_settings", lambda: FakeSettings())
    monkeypatch.setattr(sentinel, "BinanceClient", FourPositionClient)
    monkeypatch.setattr(sentinel, "load_trading_control_state", lambda: TradingControlState())
    monkeypatch.setattr(sentinel, "load_route_risk_state", lambda: {"active_quarantined_routes": [], "routes": {}})
    monkeypatch.setattr(
        sentinel,
        "run_ai_readiness_scan",
        lambda **kwargs: {"candidate_count": 3, "allowed_count": 1, "execution_ticket": {"symbol": "DOGEUSDT"}},
    )

    payload = sentinel.run_ai_market_sentinel(symbols=["BTCUSDT"], limit=80, output_dir=tmp_path)

    assert payload["position_state"]["open_position_count"] == 4
    assert payload["expansion_gate"]["allowed"] is False
    assert "max-concurrent-positions-reached" in payload["expansion_gate"]["blockers"]
    assert not any(item["action"] == "operator_testnet_preflight" for item in payload["machine_action_queue"])


def test_ai_market_sentinel_allows_research_loop_when_flat_and_no_allowed_candidate(
    monkeypatch, tmp_path: Path
) -> None:
    class FlatClient(FakeClient):
        def positions(self, symbol=None):
            return []

    monkeypatch.setattr(sentinel, "STATE_DIR", tmp_path)
    monkeypatch.setattr(sentinel, "load_settings", lambda: FakeSettings())
    monkeypatch.setattr(sentinel, "BinanceClient", FlatClient)
    monkeypatch.setattr(sentinel, "load_trading_control_state", lambda: TradingControlState())
    monkeypatch.setattr(sentinel, "load_route_risk_state", lambda: {"active_quarantined_routes": [], "routes": {}})
    monkeypatch.setattr(
        sentinel,
        "run_ai_readiness_scan",
        lambda **kwargs: {
            "candidate_count": 6,
            "allowed_count": 0,
            "next_machine_action": "continue_expectancy_research",
            "hard_blocker_taxonomy": {"strategy_performance": ["Recent PF below floor."]},
            "report_path": "state/readiness.json",
        },
    )

    payload = sentinel.run_ai_market_sentinel(symbols=["BTCUSDT"], limit=80, output_dir=tmp_path)

    assert payload["position_state"]["open_position_count"] == 0
    assert payload["expansion_gate"]["allowed"] is False
    assert payload["conditional_order_alert"]["should_notify"] is False
    assert payload["machine_action_queue"][0]["action"] == "run_ai_expectancy_upgrade"
    assert payload["machine_action_queue"][0]["reason"] == "no-readiness-approved-candidate"


def test_ai_market_sentinel_sends_telegram_only_for_ready_candidate(monkeypatch, tmp_path: Path) -> None:
    class FlatClient(FakeClient):
        def positions(self, symbol=None):
            return []

    sent: dict[str, str] = {}
    monkeypatch.setattr(sentinel, "STATE_DIR", tmp_path)
    monkeypatch.setattr(sentinel, "load_settings", lambda: FakeSettings())
    monkeypatch.setattr(sentinel, "BinanceClient", FlatClient)
    monkeypatch.setattr(sentinel, "load_trading_control_state", lambda: TradingControlState())
    monkeypatch.setattr(sentinel, "load_route_risk_state", lambda: {"active_quarantined_routes": [], "routes": {}})

    def fake_send_telegram(text: str) -> dict[str, object]:
        sent["text"] = text
        return {"sent": True}

    monkeypatch.setattr(
        sentinel,
        "send_telegram_text",
        fake_send_telegram,
    )
    monkeypatch.setattr(
        sentinel,
        "run_ai_readiness_scan",
        lambda **kwargs: {
            "candidate_count": 1,
            "allowed_count": 1,
            "selected_ready_candidate": {
                "symbol": "ETHUSDT",
                "side": "SELL",
                "route_id": "eth-core",
                "strategy_family": "breakdown",
                "live_plan": {
                    "price": 3000.0,
                    "stop_price": 3060.0,
                    "take_profit_prices": [2940.0, 2880.0],
                    "leverage": 2,
                    "quantity": 0.02,
                    "margin_notional_usdt": 30.0,
                    "gross_notional_usdt": 60.0,
                    "planned_account_risk_pct": 0.012,
                    "analysis_score": -72,
                    "analysis_convergence": 0.66,
                },
            },
            "execution_ticket": {
                "symbol": "ETHUSDT",
                "side": "SELL",
                "market": "futures",
                "interval": "15m",
                "risk_snapshot": {"leverage": 2},
            },
        },
    )

    payload = sentinel.run_ai_market_sentinel(
        symbols=["ETHUSDT"],
        limit=80,
        output_dir=tmp_path,
        send_telegram=True,
    )

    assert payload["telegram"]["sent"] is True
    assert "ETHUSDT 做空" in sent["text"]
    assert "止損價: 3060" in sent["text"]


def test_ai_market_sentinel_notifies_near_ready_market_watch(monkeypatch, tmp_path: Path) -> None:
    class FlatClient(FakeClient):
        def positions(self, symbol=None):
            return []

    sent: dict[str, str] = {}
    monkeypatch.setattr(sentinel, "STATE_DIR", tmp_path)
    monkeypatch.setattr(sentinel, "load_settings", lambda: FakeSettings())
    monkeypatch.setattr(sentinel, "BinanceClient", FlatClient)
    monkeypatch.setattr(sentinel, "load_trading_control_state", lambda: TradingControlState())
    monkeypatch.setattr(sentinel, "load_route_risk_state", lambda: {"active_quarantined_routes": [], "routes": {}})

    def fake_send_telegram(text: str) -> dict[str, object]:
        sent["text"] = text
        return {"sent": True}

    monkeypatch.setattr(sentinel, "send_telegram_text", fake_send_telegram)
    monkeypatch.setattr(
        sentinel,
        "run_ai_readiness_scan",
        lambda **kwargs: {
            "candidate_count": 4,
            "allowed_count": 0,
            "selected_ready_candidate": None,
            "execution_ticket": None,
            "next_machine_action": "wait_for_market_state",
            "research_candidate_report": {
                "near_ready_count": 1,
                "near_ready_candidates": [
                    {
                        "symbol": "TRXUSDT",
                        "side": "BUY",
                        "interval": "1d",
                        "route_id": "trx-mean-reversion",
                        "blocker_classes": ["market_state"],
                        "expectancy_metrics": {
                            "profit_factor": 1.9907,
                            "expectancy_r": 0.4666,
                            "payoff_ratio": 2.8897,
                            "sample_count": 76,
                        },
                        "risk_metrics": {"reward_risk": 2.94},
                    }
                ],
            },
        },
    )

    payload = sentinel.run_ai_market_sentinel(
        symbols=["TRXUSDT"],
        limit=80,
        output_dir=tmp_path,
        send_telegram=True,
        max_readiness_candidates=2,
    )

    assert payload["readiness"]["allowed_count"] == 0
    assert payload["readiness"]["near_ready_count"] == 1
    assert payload["conditional_order_alert"]["alert_type"] == "near_ready_market_state_watch"
    assert payload["telegram"]["sent"] is True
    assert "TRXUSDT 做多" in sent["text"]
    assert "目前阻擋: market_state" in sent["text"]
    assert payload["machine_action_queue"][0]["action"] == "monitor_near_ready_market_state"
