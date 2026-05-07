from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from binance_quant_control.binance_api import BinanceAPIError
from binance_quant_control.challenge import ChallengeState
from binance_quant_control.live_execution import (
    _split_reduce_only_quantities,
    _take_profit_weights,
    build_live_execution_plan,
    execute_live_order,
)
from binance_quant_control.strategy import load_strategy_config
from binance_quant_control.trading_control import TradingControlState


class FakeClient:
    def __init__(self, settings):
        self.settings = settings

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def ticker_price(self, symbol: str, market: str) -> float:
        return 1.416

    def balance(self, market: str):
        return [
            {
                "asset": "USDT",
                "balance": "5.31893002",
                "availableBalance": "2.16795127",
                "crossUnPnl": "0.00000000",
            }
        ]

    def open_orders(self, symbol: str, market: str = "futures"):
        return []

    def open_algo_orders(self, symbol: str | None = None):
        return []

    def positions(self, symbol: str | None = None):
        return [{"symbol": symbol or "NEARUSDT", "positionAmt": "0"}]

    def exchange_info(self, symbol: str, market: str = "futures"):
        return {
            "symbols": [
                {
                    "symbol": symbol,
                    "quantityPrecision": 0,
                    "pricePrecision": 4,
                    "filters": [
                        {"filterType": "MARKET_LOT_SIZE", "minQty": "1", "stepSize": "1"},
                        {"filterType": "MIN_NOTIONAL", "notional": "5"},
                    ],
                }
            ]
        }

    def order_book(self, symbol: str, market: str = "futures", limit: int = 20):
        return {
            "bids": [["1.4155", "50"], ["1.4150", "100"]],
            "asks": [["1.4165", "50"], ["1.4170", "100"]],
        }


def _allow_historical_signal_risk(monkeypatch) -> None:
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_historical_signal_risk",
        lambda route_id, symbol, side, score, convergence: type(
            "HistoricalSignalRisk",
            (),
            {
                "allowed": True,
                "reasons": [],
                "to_dict": lambda self: {
                    "allowed": True,
                    "route_id": route_id,
                    "symbol": symbol,
                    "side": side,
                    "score_bin": "score-081-100",
                    "convergence_bin": "conv-080-089",
                    "min_samples": 20,
                    "threshold_profit_factor": 0.8,
                    "reasons": [],
                    "buckets": [],
                },
            },
        )(),
    )


def test_build_live_execution_plan_allows_micro_near_pilot(monkeypatch):
    strategy = load_strategy_config(Path("config/strategy-live-pilot.yaml"))

    class FakeSettings:
        live_trading_enabled = False

    analysis_payload = {
        "symbol": "ETHUSDT",
        "market": "futures",
        "analysis": {
            "score": 88,
            "bias": "long-bias",
            "convergence": 0.82,
        },
        "latest": {
            "close": 1.416,
        },
        "trade_plan": {
            "long": {"invalidation": 1.36, "take_profit_1": 1.47},
            "short": {"invalidation": 1.45, "take_profit_1": 1.37},
        },
    }

    _allow_historical_signal_risk(monkeypatch)
    monkeypatch.setattr("binance_quant_control.live_execution.BinanceClient", FakeClient)
    monkeypatch.setattr(
        "binance_quant_control.live_execution.route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": False, "reasons": []},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_optimizer_live_gate",
        lambda: {"allowed": True, "reasons": [], "promotion_decision": "promote"},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_market_bot_live_gate",
        lambda symbol, route_id: {"allowed": False, "reasons": ["not promoted"], "matched_row": None},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.record_balance_snapshot",
        lambda payload, market, note="", scope=None: (
            type(
                "Snapshot",
                (),
                {
                    "wallet_balance_usdt": 5.31893002,
                    "available_balance_usdt": 2.16795127,
                    "unrealized_pnl_usdt": 0.0,
                    "equity_usdt": 5.31893002,
                },
            )(),
            ChallengeState(),
        ),
    )
    plan = build_live_execution_plan(FakeSettings(), strategy, analysis_payload)

    assert plan.allowed is False
    assert plan.side == "BUY"
    assert any("below exchange minimum" in item for item in plan.violations)
    assert plan.challenge["enabled"] is False
    assert plan.spread_bps > 0.0
    assert plan.professional_entry_gate["layers"]["execution_quality"]["tp1_reward_risk"] < 1.0
    assert any("Reward/risk" in item for item in plan.violations)


def test_live_execution_plan_splits_staged_take_profit_quantities(monkeypatch):
    strategy = load_strategy_config(Path("config/strategy-major-alt-trend.yaml"))

    class FakeSettings:
        live_trading_enabled = False

    analysis_payload = {
        "symbol": "NEARUSDT",
        "market": "futures",
        "analysis": {
            "score": 92,
            "bias": "long-bias",
            "convergence": 0.91,
        },
        "latest": {
            "close": 1.416,
            "adx": 25.0,
        },
        "trade_plan": {
            "long": {
                "invalidation": 1.36,
                "take_profit_1": 1.47,
                "take_profit_2": 1.52,
                "take_profit_3": 1.58,
            },
            "short": {
                "invalidation": 1.45,
                "take_profit_1": 1.37,
                "take_profit_2": 1.32,
                "take_profit_3": 1.28,
            },
        },
    }

    _allow_historical_signal_risk(monkeypatch)
    monkeypatch.setattr("binance_quant_control.live_execution.BinanceClient", FakeClient)
    monkeypatch.setattr(
        "binance_quant_control.live_execution.route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": False, "reasons": []},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_optimizer_live_gate",
        lambda: {"allowed": True, "reasons": [], "promotion_decision": "promote"},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.record_balance_snapshot",
        lambda payload, market, note="", scope=None: (
            type(
                "Snapshot",
                (),
                {
                    "wallet_balance_usdt": 5000.0,
                    "available_balance_usdt": 5000.0,
                    "unrealized_pnl_usdt": 0.0,
                    "equity_usdt": 5000.0,
                },
            )(),
            ChallengeState(),
        ),
    )

    plan = build_live_execution_plan(FakeSettings(), strategy, analysis_payload, margin_notional_usdt=4.0)

    assert plan.take_profit_prices == [1.47, 1.52, 1.58]
    assert len(plan.take_profit_quantities) == 3
    assert round(sum(plan.take_profit_quantities) + plan.take_profit_runner_quantity, 8) == plan.quantity
    assert plan.take_profit_weights == [0.2, 0.28, 0.32]
    assert plan.take_profit_runner_quantity > 0
    assert plan.take_profit_quantities[-1] >= plan.take_profit_quantities[0]
    assert plan.sizing["signal_scores"]["event_risk_score"] == 50.0


def test_live_execution_plan_reduces_sizing_for_high_news_and_weak_flow(monkeypatch):
    strategy = load_strategy_config(Path("config/strategy-meme-momentum.yaml"))

    class FakeSettings:
        live_trading_enabled = False
        use_testnet = True
        testnet_trading_enabled = True

    analysis_payload = {
        "symbol": "DOGEUSDT",
        "market": "futures",
        "analysis": {
            "score": 86,
            "bias": "long-bias",
            "convergence": 0.9,
        },
        "latest": {
            "close": 1.416,
            "adx": 24.0,
            "realized_vol_20": 0.8,
            "volume_zscore_20": -1.6,
            "obv_zscore_20": -0.8,
            "taker_buy_sell_ratio": 0.92,
            "order_book_imbalance": -0.3,
        },
        "trade_plan": {
            "long": {"invalidation": 1.36, "take_profit_1": 1.47},
            "short": {"invalidation": 1.45, "take_profit_1": 1.37},
        },
    }

    _allow_historical_signal_risk(monkeypatch)
    monkeypatch.setattr("binance_quant_control.live_execution.BinanceClient", FakeClient)
    monkeypatch.setattr(
        "binance_quant_control.live_execution.route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": False, "reasons": []},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_optimizer_live_gate",
        lambda: {"allowed": True, "reasons": [], "promotion_decision": "promote"},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_route_side_risk",
        lambda route_id, side: type(
            "SideRisk",
            (),
            {
                "allowed": True,
                "reasons": [],
                "to_dict": lambda self: {
                    "allowed": True,
                    "route_id": route_id,
                    "side": side,
                    "sample_count": 169,
                    "profit_factor": 0.491,
                    "net_pnl_usdt": -5.155,
                    "loss_streak": 1,
                    "threshold_profit_factor": 0.8,
                    "min_samples": 30,
                    "reasons": [],
                },
            },
        )(),
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.record_balance_snapshot",
        lambda payload, market, note="", scope=None: (
            type(
                "Snapshot",
                (),
                {
                    "wallet_balance_usdt": 5000.0,
                    "available_balance_usdt": 5000.0,
                    "unrealized_pnl_usdt": 0.0,
                    "equity_usdt": 5000.0,
                },
            )(),
            ChallengeState(),
        ),
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.load_trading_control_state",
        lambda: TradingControlState(paused=False, reason="", updated_at="", updated_by="test"),
    )

    plan = build_live_execution_plan(
        FakeSettings(),
        strategy,
        analysis_payload,
        margin_notional_usdt=8.0,
        execution_mode="testnet_exploration",
        news_risk={"risk_level": "high", "bias": "bearish"},
    )

    assert plan.sizing["signal_scores"]["flow_score"] < 40.0
    assert plan.sizing["recommended_leverage"] <= 2
    assert any("Weak flow" in item for item in plan.warnings)


def test_take_profit_weights_use_risk_adjusted_two_stage_split() -> None:
    assert _take_profit_weights(
        parts=2,
        route_id="major-alt-trend",
        confidence=0.75,
        news_risk={"risk_level": "normal"},
    ) == [0.3, 0.7]
    assert _take_profit_weights(
        parts=3,
        route_id="trx-mean-reversion",
        confidence=0.9,
        news_risk={"risk_level": "normal"},
        strategy_family="mean_reversion",
    ) == [0.55, 0.3, 0.15]
    assert _take_profit_weights(
        parts=2,
        route_id="meme-high-beta",
        confidence=0.75,
        news_risk={"risk_level": "high"},
    ) == [0.4, 0.6]
    assert _take_profit_weights(
        parts=2,
        route_id="doge-meme-high-beta",
        confidence=0.75,
        news_risk={"risk_level": "normal"},
    ) == [0.4, 0.6]


def test_split_reduce_only_quantities_follow_weights() -> None:
    quantities = _split_reduce_only_quantities(30.1, 0.1, 1, [0.3, 0.7])
    assert quantities == [9.0, 21.1]


def test_split_reduce_only_quantities_can_reserve_runner() -> None:
    quantities = _split_reduce_only_quantities(30.1, 0.1, 1, [0.255, 0.2975, 0.2975])
    assert quantities == [7.6, 8.9, 9.0]
    assert round(30.1 - sum(quantities), 1) == 4.6


def test_build_live_execution_plan_blocks_on_challenge_drawdown(monkeypatch):
    strategy = load_strategy_config(Path("config/strategy-live-pilot.yaml"))

    class FakeSettings:
        live_trading_enabled = True

    analysis_payload = {
        "symbol": "NEARUSDT",
        "market": "futures",
        "analysis": {
            "score": 88,
            "bias": "long-bias",
            "convergence": 0.82,
        },
        "latest": {
            "close": 1.416,
        },
        "trade_plan": {
            "long": {"invalidation": 1.36, "take_profit_1": 1.47},
            "short": {"invalidation": 1.45, "take_profit_1": 1.37},
        },
    }

    _allow_historical_signal_risk(monkeypatch)
    monkeypatch.setattr("binance_quant_control.live_execution.BinanceClient", FakeClient)
    monkeypatch.setattr(
        "binance_quant_control.live_execution.route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": False, "reasons": []},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_optimizer_live_gate",
        lambda: {"allowed": True, "reasons": [], "promotion_decision": "promote"},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.record_balance_snapshot",
        lambda payload, market, note="", scope=None: (
            type(
                "Snapshot",
                (),
                {
                    "wallet_balance_usdt": 4.0,
                    "available_balance_usdt": 2.0,
                    "unrealized_pnl_usdt": -0.4,
                    "equity_usdt": 3.6,
                },
            )(),
            ChallengeState(
                enabled=True,
                profile="micro-account-pilot",
                symbol="NEARUSDT",
                market="futures",
                started_at="2026-04-25T00:00:00+00:00",
                start_balance_usdt=5.0,
                target_balance_usdt=10.0,
                target_multiple=2.0,
                max_drawdown_pct=20.0,
                stop_balance_usdt=4.0,
                highest_balance_usdt=5.2,
                latest_balance_usdt=3.6,
                latest_snapshot_at="2026-04-25T00:00:00+00:00",
                status="drawdown-stop",
            ),
        ),
    )
    plan = build_live_execution_plan(FakeSettings(), strategy, analysis_payload)

    assert plan.allowed is False
    assert any("Challenge drawdown stop" in item for item in plan.violations)


def test_build_live_execution_plan_caps_risk_for_tight_profile(monkeypatch):
    strategy = load_strategy_config(Path("config/strategy-tight-risk.yaml"))

    class FakeSettings:
        live_trading_enabled = False

    analysis_payload = {
        "symbol": "NEARUSDT",
        "market": "futures",
        "analysis": {
            "score": 92,
            "bias": "long-bias",
            "convergence": 0.91,
        },
        "latest": {
            "close": 1.413,
            "adx": 26.0,
        },
        "trade_plan": {
            "long": {"invalidation": 1.4001, "take_profit_1": 1.4324},
            "short": {"invalidation": 1.4259, "take_profit_1": 1.3936},
        },
    }

    _allow_historical_signal_risk(monkeypatch)
    monkeypatch.setattr("binance_quant_control.live_execution.BinanceClient", FakeClient)
    monkeypatch.setattr(
        "binance_quant_control.live_execution.route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": False, "reasons": []},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_optimizer_live_gate",
        lambda: {"allowed": True, "reasons": [], "promotion_decision": "promote"},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.record_balance_snapshot",
        lambda payload, market, note="", scope=None: (
            type(
                "Snapshot",
                (),
                {
                    "wallet_balance_usdt": 5.16565357,
                    "available_balance_usdt": 3.28312818,
                    "unrealized_pnl_usdt": 0.0,
                    "equity_usdt": 5.16565357,
                },
            )(),
            ChallengeState(),
        ),
    )
    plan = build_live_execution_plan(FakeSettings(), strategy, analysis_payload)

    assert plan.allowed is False
    assert 1.0 <= plan.quantity <= 4.0
    assert plan.planned_account_risk_pct <= 0.01
    assert plan.trailing_stop_enabled is True
    assert any("below exchange minimum" in item for item in plan.violations)


def test_build_live_execution_plan_blocks_when_adx_is_too_low(monkeypatch):
    strategy = load_strategy_config(Path("config/strategy-tight-risk.yaml"))

    class FakeSettings:
        live_trading_enabled = False

    analysis_payload = {
        "symbol": "NEARUSDT",
        "market": "futures",
        "analysis": {
            "score": 92,
            "bias": "long-bias",
            "convergence": 0.91,
        },
        "latest": {
            "close": 1.413,
            "adx": 12.5,
        },
        "trade_plan": {
            "long": {"invalidation": 1.4001, "take_profit_1": 1.4324},
            "short": {"invalidation": 1.4259, "take_profit_1": 1.3936},
        },
    }

    _allow_historical_signal_risk(monkeypatch)
    monkeypatch.setattr("binance_quant_control.live_execution.BinanceClient", FakeClient)
    monkeypatch.setattr(
        "binance_quant_control.live_execution.route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": False, "reasons": []},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_optimizer_live_gate",
        lambda: {"allowed": True, "reasons": [], "promotion_decision": "promote"},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.record_balance_snapshot",
        lambda payload, market, note="", scope=None: (
            type(
                "Snapshot",
                (),
                {
                    "wallet_balance_usdt": 5.16565357,
                    "available_balance_usdt": 3.28312818,
                    "unrealized_pnl_usdt": 0.0,
                    "equity_usdt": 5.16565357,
                },
            )(),
            ChallengeState(),
        ),
    )
    plan = build_live_execution_plan(FakeSettings(), strategy, analysis_payload)

    assert plan.allowed is False
    assert any("ADX" in item for item in plan.violations)


def test_build_live_execution_plan_blocks_when_kill_switch_is_paused(monkeypatch):
    strategy = load_strategy_config(Path("config/strategy-live-pilot.yaml"))

    class FakeSettings:
        live_trading_enabled = True

    analysis_payload = {
        "symbol": "NEARUSDT",
        "market": "futures",
        "analysis": {
            "score": 88,
            "bias": "long-bias",
            "convergence": 0.82,
        },
        "latest": {
            "close": 1.416,
        },
        "trade_plan": {
            "long": {"invalidation": 1.36, "take_profit_1": 1.47},
            "short": {"invalidation": 1.45, "take_profit_1": 1.37},
        },
    }

    _allow_historical_signal_risk(monkeypatch)
    monkeypatch.setattr("binance_quant_control.live_execution.BinanceClient", FakeClient)
    monkeypatch.setattr(
        "binance_quant_control.live_execution.route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": False, "reasons": []},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_optimizer_live_gate",
        lambda: {"allowed": True, "reasons": [], "promotion_decision": "promote"},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.record_balance_snapshot",
        lambda payload, market, note="", scope=None: (
            type(
                "Snapshot",
                (),
                {
                    "wallet_balance_usdt": 5.31893002,
                    "available_balance_usdt": 2.16795127,
                    "unrealized_pnl_usdt": 0.0,
                    "equity_usdt": 5.31893002,
                },
            )(),
            ChallengeState(),
        ),
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.load_trading_control_state",
        lambda: TradingControlState(paused=True, reason="manual stop", updated_at="", updated_by="test"),
    )

    plan = build_live_execution_plan(FakeSettings(), strategy, analysis_payload)

    assert plan.allowed is False
    assert any("kill-switch" in item for item in plan.violations)


def test_build_live_execution_plan_blocks_research_only_invalid_futures_symbol(monkeypatch):
    strategy = load_strategy_config(Path("config/strategy-xau-macro.yaml"))

    class FakeSettings:
        live_trading_enabled = False

    class InvalidSymbolClient(FakeClient):
        def ticker_price(self, symbol: str, market: str) -> float:
            return 3300.0

        def open_orders(self, symbol: str, market: str = "futures"):
            raise BinanceAPIError("Binance API error status=400 code=-1121 msg=Invalid symbol.")

        def exchange_info(self, symbol: str, market: str = "futures"):
            return {
                "symbols": [
                    {
                        "symbol": symbol,
                        "quantityPrecision": 3,
                        "pricePrecision": 2,
                        "filters": [
                            {"filterType": "MARKET_LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"},
                            {"filterType": "MIN_NOTIONAL", "notional": "5"},
                        ],
                    }
                ]
            }

        def order_book(self, symbol: str, market: str = "futures", limit: int = 20):
            return {
                "bids": [["3299.0", "2"]],
                "asks": [["3301.0", "2"]],
            }

    analysis_payload = {
        "symbol": "XAUTUSDT",
        "market": "futures",
        "analysis": {
            "score": 82,
            "bias": "long-bias",
            "convergence": 0.78,
        },
        "latest": {
            "close": 3300.0,
            "adx": 22.0,
        },
        "trade_plan": {
            "long": {"invalidation": 3230.0, "take_profit_1": 3405.0},
            "short": {"invalidation": 3370.0, "take_profit_1": 3195.0},
        },
    }

    _allow_historical_signal_risk(monkeypatch)
    monkeypatch.setattr("binance_quant_control.live_execution.BinanceClient", InvalidSymbolClient)
    monkeypatch.setattr(
        "binance_quant_control.live_execution.route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": False, "reasons": []},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_optimizer_live_gate",
        lambda: {"allowed": True, "reasons": [], "promotion_decision": "promote"},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.record_balance_snapshot",
        lambda payload, market, note="", scope=None: (
            type(
                "Snapshot",
                (),
                {
                    "wallet_balance_usdt": 1000.0,
                    "available_balance_usdt": 1000.0,
                    "unrealized_pnl_usdt": 0.0,
                    "equity_usdt": 1000.0,
                },
            )(),
            ChallengeState(),
        ),
    )

    plan = build_live_execution_plan(FakeSettings(), strategy, analysis_payload, margin_notional_usdt=10.0)

    assert plan.allowed is False
    assert any("research-only" in item for item in plan.violations)


def test_execute_live_order_records_route_metadata(monkeypatch, tmp_path):
    journal_path = tmp_path / "live-orders.jsonl"
    monkeypatch.setattr("binance_quant_control.order_journal.LIVE_ORDERS_FILE", journal_path)
    monkeypatch.setattr(
        "binance_quant_control.live_execution.append_live_order",
        lambda record: (
            Path(journal_path).write_text(json.dumps(asdict(record), ensure_ascii=False) + "\n", encoding="utf-8"),
            journal_path,
        )[1],
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.load_trade_state",
        lambda: type(
            "State",
            (),
            {
                "daily_trade_count": 0,
                "consecutive_losses": 0,
                "last_loss_datetime": None,
                "record_trade": lambda self: None,
            },
        )(),
    )
    monkeypatch.setattr("binance_quant_control.live_execution.save_trade_state", lambda state: None)
    monkeypatch.setattr(
        "binance_quant_control.live_execution.load_trading_control_state",
        lambda: TradingControlState(paused=False, reason="", updated_at="", updated_by="test"),
    )

    class ExecClient(FakeClient):
        def set_margin_type(self, symbol: str, margin_type: str):
            return {"msg": "ok"}

        def set_leverage(self, symbol: str, leverage: int):
            return {"symbol": symbol, "leverage": leverage}

        def new_order(self, symbol: str, side: str, order_type: str, **kwargs):
            return {"orderId": 123456, "status": "NEW", "symbol": symbol, "side": side}

        def new_algo_order(self, symbol: str, side: str, order_type: str, **kwargs):
            return {
                "algoId": 99 + len(getattr(self, "algo_calls", [])),
                "symbol": symbol,
                "side": side,
                "type": order_type,
                **kwargs,
            }

    monkeypatch.setattr("binance_quant_control.live_execution.BinanceClient", ExecClient)
    _allow_historical_signal_risk(monkeypatch)
    monkeypatch.setattr(
        "binance_quant_control.live_execution.route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": False, "reasons": []},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_optimizer_live_gate",
        lambda: {"allowed": True, "reasons": [], "promotion_decision": "promote"},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_route_side_risk",
        lambda route_id, side: type(
            "SideRisk",
            (),
            {
                "allowed": True,
                "reasons": [],
                "to_dict": lambda self: {
                    "allowed": True,
                    "route_id": route_id,
                    "side": side,
                    "sample_count": 0,
                    "profit_factor": 0.0,
                    "net_pnl_usdt": 0.0,
                    "loss_streak": 0,
                    "threshold_profit_factor": 0.8,
                    "min_samples": 30,
                    "reasons": [],
                },
            },
        )(),
    )
    monkeypatch.setattr(
        "binance_quant_control.professional_entry_gate.read_closed_trade_reviews",
        lambda: [
            {
                "symbol": "ETHUSDT",
                "side": "BUY",
                "route_id": "eth-core",
                "strategy_profile": "eth-trend",
                "exit_reason": "take_profit",
                "realized_pnl_usdt": 2.0,
                "realized_r_multiple": 2.0,
            }
            for _ in range(6)
        ],
    )

    analysis_payload = {
        "symbol": "ETHUSDT",
        "market": "futures",
        "analysis": {"score": 84, "bias": "long-bias", "convergence": 0.82},
        "latest": {
            "close": 1.416,
            "adx": 25.0,
            "realized_vol_20": 0.8,
            "volume_zscore_20": 0.4,
            "obv_zscore_20": 0.4,
            "bb_bandwidth": 0.08,
        },
        "trade_plan": {
            "long": {"invalidation": 1.36, "take_profit_1": 1.47, "take_profit_2": 1.52, "take_profit_3": 1.58},
            "short": {"invalidation": 1.45, "take_profit_1": 1.37, "take_profit_2": 1.32, "take_profit_3": 1.28},
        },
        "market_context": {},
    }

    class FakeSettings:
        live_trading_enabled = True

    monkeypatch.setattr(
        "binance_quant_control.live_execution.record_balance_snapshot",
        lambda payload, market, note="", scope=None: (
            type(
                "Snapshot",
                (),
                {
                    "wallet_balance_usdt": 5000.0,
                    "available_balance_usdt": 5000.0,
                    "unrealized_pnl_usdt": 0.0,
                    "equity_usdt": 5000.0,
                },
            )(),
            ChallengeState(),
        ),
    )
    plan = build_live_execution_plan(
        FakeSettings(),
        load_strategy_config(Path("config/strategy-eth-trend.yaml")),
        analysis_payload,
        margin_notional_usdt=4.0,
        execution_mode="testnet_exploration",
    )

    response = execute_live_order(
        FakeSettings(),
        load_strategy_config(Path("config/strategy-eth-trend.yaml")),
        plan,
        entry_reason_snapshot={"bias": "long-bias", "score": 84, "convergence": 0.82, "interval": "4h"},
        signal_scores={"composite_convergence_score": 70.0},
    )

    assert response["entry_order"]["orderId"] == 123456
    rows = journal_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    assert '"route_id": "eth-core"' in rows[0]
    assert '"cohort_id": "eth_core:eth-trend:futures:4h"' in rows[0]
    assert len(response["protective_orders"]["take_profits"]) == 3


def test_build_live_execution_plan_blocks_when_optimizer_rejects(monkeypatch):
    strategy = load_strategy_config(Path("config/strategy-major-alt-trend.yaml"))

    class FakeSettings:
        live_trading_enabled = True

    analysis_payload = {
        "symbol": "NEARUSDT",
        "market": "futures",
        "analysis": {
            "score": 92,
            "bias": "long-bias",
            "convergence": 0.91,
        },
        "latest": {
            "close": 1.416,
            "adx": 25.0,
        },
        "trade_plan": {
            "long": {"invalidation": 1.36, "take_profit_1": 1.47},
            "short": {"invalidation": 1.45, "take_profit_1": 1.37},
        },
    }

    _allow_historical_signal_risk(monkeypatch)
    monkeypatch.setattr("binance_quant_control.live_execution.BinanceClient", FakeClient)
    monkeypatch.setattr(
        "binance_quant_control.live_execution.route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": False, "reasons": []},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_optimizer_live_gate",
        lambda: {
            "allowed": False,
            "promotion_decision": "reject",
            "reasons": ["Global strategy optimizer has not promoted this system."],
        },
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.record_balance_snapshot",
        lambda payload, market, note="", scope=None: (
            type(
                "Snapshot",
                (),
                {
                    "wallet_balance_usdt": 5000.0,
                    "available_balance_usdt": 5000.0,
                    "unrealized_pnl_usdt": 0.0,
                    "equity_usdt": 5000.0,
                },
            )(),
            ChallengeState(),
        ),
    )

    plan = build_live_execution_plan(FakeSettings(), strategy, analysis_payload, margin_notional_usdt=4.0)

    assert plan.allowed is False
    assert any("optimizer" in item.lower() for item in plan.violations)
    assert plan.challenge["optimizer_live_gate"]["promotion_decision"] == "reject"


def test_build_live_execution_plan_blocks_quarantined_route(monkeypatch):
    strategy = load_strategy_config(Path("config/strategy-major-alt-trend.yaml"))

    class FakeSettings:
        live_trading_enabled = True

    analysis_payload = {
        "symbol": "NEARUSDT",
        "market": "futures",
        "analysis": {"score": 92, "bias": "long-bias", "convergence": 0.91},
        "latest": {"close": 1.416, "adx": 25.0},
        "trade_plan": {
            "long": {"invalidation": 1.36, "take_profit_1": 1.47},
            "short": {"invalidation": 1.45, "take_profit_1": 1.37},
        },
    }

    _allow_historical_signal_risk(monkeypatch)
    monkeypatch.setattr("binance_quant_control.live_execution.BinanceClient", FakeClient)
    monkeypatch.setattr(
        "binance_quant_control.live_execution.route_quarantine_status",
        lambda route_id: {
            "route_id": route_id,
            "quarantined": True,
            "reasons": ["profit-factor 0.5 below floor"],
        },
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_optimizer_live_gate",
        lambda: {"allowed": True, "reasons": [], "promotion_decision": "promote"},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.record_balance_snapshot",
        lambda payload, market, note="", scope=None: (
            type(
                "Snapshot",
                (),
                {
                    "wallet_balance_usdt": 5000.0,
                    "available_balance_usdt": 5000.0,
                    "unrealized_pnl_usdt": 0.0,
                    "equity_usdt": 5000.0,
                },
            )(),
            ChallengeState(),
        ),
    )

    plan = build_live_execution_plan(FakeSettings(), strategy, analysis_payload, margin_notional_usdt=4.0)

    assert plan.allowed is False
    assert any("quarantined" in item for item in plan.violations)
    assert plan.challenge["route_quarantine"]["quarantined"] is True


def test_testnet_exploration_softens_strategy_gates(monkeypatch):
    strategy = load_strategy_config(Path("config/strategy-meme-momentum.yaml"))

    class FakeSettings:
        live_trading_enabled = False
        use_testnet = True
        testnet_trading_enabled = True

    analysis_payload = {
        "symbol": "DOGEUSDT",
        "market": "futures",
        "analysis": {"score": 78, "bias": "long-bias", "convergence": 0.72},
        "latest": {"close": 1.416, "adx": 16.0, "realized_vol_20": 1.0},
        "trade_plan": {
            "long": {"invalidation": 1.36, "take_profit_1": 1.47},
            "short": {"invalidation": 1.45, "take_profit_1": 1.37},
        },
    }

    _allow_historical_signal_risk(monkeypatch)
    monkeypatch.setattr("binance_quant_control.live_execution.BinanceClient", FakeClient)
    monkeypatch.setattr(
        "binance_quant_control.live_execution.route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": True, "reasons": ["loss-streak"]},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_optimizer_live_gate",
        lambda: {"allowed": False, "reasons": ["Global strategy optimizer has not promoted this system."], "promotion_decision": "reject"},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_route_side_risk",
        lambda route_id, side: type(
            "SideRisk",
            (),
            {
                "allowed": False,
                "reasons": ["route/side profit factor 0.50 below threshold"],
                "to_dict": lambda self: {"allowed": False, "reasons": ["route/side profit factor 0.50 below threshold"]},
            },
        )(),
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.record_balance_snapshot",
        lambda payload, market, note="", scope=None: (
            type(
                "Snapshot",
                (),
                {
                    "wallet_balance_usdt": 5000.0,
                    "available_balance_usdt": 5000.0,
                    "unrealized_pnl_usdt": 0.0,
                    "equity_usdt": 5000.0,
                },
            )(),
            ChallengeState(),
        ),
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.load_trading_control_state",
        lambda: TradingControlState(paused=False, reason="", updated_at="", updated_by="test"),
    )

    plan = build_live_execution_plan(
        FakeSettings(),
        strategy,
        analysis_payload,
        margin_notional_usdt=6.0,
        execution_mode="testnet_exploration",
    )

    assert plan.execution_mode == "testnet_exploration"
    assert not any("quarantined" in item for item in plan.violations)
    assert any("quarantined" in item for item in plan.warnings)
    assert plan.sizing["recommended_leverage"] >= 1


def test_testnet_exploration_blocks_paper_only_route_and_traces_decision(monkeypatch):
    strategy = load_strategy_config(Path("config/strategy-defensive-default.yaml"))

    class FakeSettings:
        live_trading_enabled = False
        use_testnet = True
        testnet_trading_enabled = True

    analysis_payload = {
        "symbol": "BRUSDT",
        "market": "futures",
        "analysis": {"score": 90, "bias": "long-bias", "convergence": 0.9},
        "latest": {"close": 1.416, "adx": 25.0},
        "trade_plan": {
            "long": {"invalidation": 1.36, "take_profit_1": 1.47},
            "short": {"invalidation": 1.45, "take_profit_1": 1.37},
        },
    }

    _allow_historical_signal_risk(monkeypatch)
    monkeypatch.setattr("binance_quant_control.live_execution.BinanceClient", FakeClient)
    monkeypatch.setattr(
        "binance_quant_control.live_execution.route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": False, "reasons": []},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_optimizer_live_gate",
        lambda: {"allowed": True, "reasons": [], "promotion_decision": "promote"},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_route_side_risk",
        lambda route_id, side: type(
            "SideRisk",
            (),
            {
                "allowed": True,
                "reasons": [],
                "to_dict": lambda self: {"allowed": True, "route_id": route_id, "side": side, "reasons": []},
            },
        )(),
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.record_balance_snapshot",
        lambda payload, market, note="", scope=None: (
            type(
                "Snapshot",
                (),
                {
                    "wallet_balance_usdt": 5000.0,
                    "available_balance_usdt": 5000.0,
                    "unrealized_pnl_usdt": 0.0,
                    "equity_usdt": 5000.0,
                },
            )(),
            ChallengeState(),
        ),
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.load_trading_control_state",
        lambda: TradingControlState(paused=False, reason="", updated_at="", updated_by="test"),
    )

    plan = build_live_execution_plan(
        FakeSettings(),
        strategy,
        analysis_payload,
        margin_notional_usdt=6.0,
        execution_mode="testnet_exploration",
    )

    by_layer = {step["layer"]: step for step in plan.decision_trace}

    assert plan.allowed is False
    assert any("need at least 30 reviews and PF > 1.0" in item for item in plan.violations)
    assert by_layer["route_mode"]["allowed"] is False
    assert by_layer["final_plan"]["allowed"] is False
    assert by_layer["route_mode"]["data"]["route_id"] == "defensive-unknown"


def test_testnet_exploration_uses_market_bot_gate_as_promotion_bridge(monkeypatch):
    strategy = load_strategy_config(Path("config/strategy-meme-momentum.yaml"))

    class FakeSettings:
        live_trading_enabled = False
        use_testnet = True
        testnet_trading_enabled = True

    analysis_payload = {
        "symbol": "DOGEUSDT",
        "market": "futures",
        "analysis": {"score": 90, "bias": "long-bias", "convergence": 0.9},
        "latest": {
            "close": 1.416,
            "adx": 28.0,
            "realized_vol_20": 0.8,
            "volume_zscore_20": 0.4,
            "obv_zscore_20": 0.7,
            "bb_bandwidth": 0.08,
        },
        "trade_plan": {
            "long": {
                "invalidation": 1.36,
                "take_profit_1": 1.49,
                "take_profit_2": 1.55,
                "take_profit_3": 1.62,
            },
            "short": {"invalidation": 1.45, "take_profit_1": 1.37},
        },
    }

    _allow_historical_signal_risk(monkeypatch)
    monkeypatch.setattr("binance_quant_control.live_execution.BinanceClient", FakeClient)
    monkeypatch.setattr(
        "binance_quant_control.live_execution.route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": False, "reasons": []},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_optimizer_live_gate",
        lambda: {
            "allowed": False,
            "reasons": ["Global strategy optimizer has not promoted this system."],
            "promotion_decision": "reject",
        },
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_market_bot_live_gate",
        lambda symbol, route_id: {
            "allowed": True,
            "safe_to_open_new_entries": True,
            "accepted_count": 6,
            "accepted_symbols": ["DOGEUSDT"],
            "matched_row": {
                "symbol": symbol,
                "route_id": route_id,
                "profit_factor": 1.48,
                "expectancy_r": 0.26,
                "payoff_ratio": 3.38,
            },
            "reasons": [],
        },
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_route_side_risk",
        lambda route_id, side: type(
            "SideRisk",
            (),
            {
                "allowed": True,
                "reasons": [],
                "to_dict": lambda self: {"allowed": True, "route_id": route_id, "side": side, "reasons": []},
            },
        )(),
    )
    monkeypatch.setattr(
        "binance_quant_control.professional_entry_gate.read_closed_trade_reviews",
        lambda: [
            {
                "symbol": "DOGEUSDT",
                "side": "BUY",
                "route_id": "meme-high-beta",
                "exit_reason": "take_profit",
                "realized_pnl_usdt": 2.0,
                "realized_r_multiple": 2.0,
            }
            for _ in range(8)
        ],
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.record_balance_snapshot",
        lambda payload, market, note="", scope=None: (
            type(
                "Snapshot",
                (),
                {
                    "wallet_balance_usdt": 5000.0,
                    "available_balance_usdt": 5000.0,
                    "unrealized_pnl_usdt": 0.0,
                    "equity_usdt": 5000.0,
                },
            )(),
            ChallengeState(),
        ),
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.load_trading_control_state",
        lambda: TradingControlState(paused=False, reason="", updated_at="", updated_by="test"),
    )

    plan = build_live_execution_plan(
        FakeSettings(),
        strategy,
        analysis_payload,
        margin_notional_usdt=6.0,
        execution_mode="testnet_exploration",
    )

    by_layer = {step["layer"]: step for step in plan.decision_trace}

    assert by_layer["market_bot_gate"]["allowed"] is True
    assert by_layer["route_mode"]["allowed"] is True
    assert by_layer["optimizer_gate"]["allowed"] is True
    assert not any("unknown/paper-only routes" in item for item in plan.violations)
    assert any("Market-bot gate is accepted" in item for item in plan.warnings)


def test_build_live_execution_plan_blocks_weak_route_side_history(monkeypatch):
    strategy = load_strategy_config(Path("config/strategy-major-alt-trend.yaml"))

    class FakeSettings:
        live_trading_enabled = True

    analysis_payload = {
        "symbol": "NEARUSDT",
        "market": "futures",
        "analysis": {"score": 92, "bias": "long-bias", "convergence": 0.91},
        "latest": {"close": 1.416, "adx": 25.0},
        "trade_plan": {
            "long": {"invalidation": 1.36, "take_profit_1": 1.47},
            "short": {"invalidation": 1.45, "take_profit_1": 1.37},
        },
    }

    _allow_historical_signal_risk(monkeypatch)
    monkeypatch.setattr("binance_quant_control.live_execution.BinanceClient", FakeClient)
    monkeypatch.setattr(
        "binance_quant_control.live_execution.route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": False, "reasons": []},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_optimizer_live_gate",
        lambda: {"allowed": True, "reasons": [], "promotion_decision": "promote"},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_route_side_risk",
        lambda route_id, side: type(
            "SideRisk",
            (),
            {
                "allowed": False,
                "reasons": [f"Route-side historical PF 0.5523 is below 0.8000 for {route_id}/{side}."],
                "to_dict": lambda self: {
                    "allowed": False,
                    "route_id": route_id,
                    "side": side,
                    "sample_count": 150,
                    "profit_factor": 0.5523,
                    "net_pnl_usdt": -7.5299,
                    "loss_streak": 1,
                    "threshold_profit_factor": 0.8,
                    "min_samples": 30,
                    "reasons": self.reasons,
                },
            },
        )(),
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.record_balance_snapshot",
        lambda payload, market, note="", scope=None: (
            type(
                "Snapshot",
                (),
                {
                    "wallet_balance_usdt": 5000.0,
                    "available_balance_usdt": 5000.0,
                    "unrealized_pnl_usdt": 0.0,
                    "equity_usdt": 5000.0,
                },
            )(),
            ChallengeState(),
        ),
    )

    plan = build_live_execution_plan(FakeSettings(), strategy, analysis_payload, margin_notional_usdt=4.0)

    assert plan.allowed is False
    assert any("Route-side historical PF" in item for item in plan.violations)
    assert plan.challenge["route_side_risk"]["allowed"] is False


def test_build_live_execution_plan_blocks_weak_historical_signal_bucket(monkeypatch):
    strategy = load_strategy_config(Path("config/strategy-major-alt-trend.yaml"))

    class FakeSettings:
        live_trading_enabled = True

    analysis_payload = {
        "symbol": "NEARUSDT",
        "market": "futures",
        "analysis": {"score": 92, "bias": "long-bias", "convergence": 0.91},
        "latest": {"close": 1.416, "adx": 25.0},
        "trade_plan": {
            "long": {"invalidation": 1.36, "take_profit_1": 1.47},
            "short": {"invalidation": 1.45, "take_profit_1": 1.37},
        },
    }

    monkeypatch.setattr("binance_quant_control.live_execution.BinanceClient", FakeClient)
    monkeypatch.setattr(
        "binance_quant_control.live_execution.route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": False, "reasons": []},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_optimizer_live_gate",
        lambda: {"allowed": True, "reasons": [], "promotion_decision": "promote"},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_route_side_risk",
        lambda route_id, side: type(
            "SideRisk",
            (),
            {
                "allowed": True,
                "reasons": [],
                "to_dict": lambda self: {
                    "allowed": True,
                    "route_id": route_id,
                    "side": side,
                    "sample_count": 0,
                    "profit_factor": 0.0,
                    "net_pnl_usdt": 0.0,
                    "loss_streak": 0,
                    "threshold_profit_factor": 0.8,
                    "min_samples": 30,
                    "reasons": [],
                },
            },
        )(),
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_historical_signal_risk",
        lambda route_id, symbol, side, score, convergence: type(
            "HistoricalSignalRisk",
            (),
            {
                "allowed": False,
                "reasons": [
                    "Historical feedback bucket route-side-score "
                    f"{route_id}/{side}/score-081-100 has PF 0.5372."
                ],
                "to_dict": lambda self: {
                    "allowed": False,
                    "route_id": route_id,
                    "symbol": symbol,
                    "side": side,
                    "score_bin": "score-081-100",
                    "convergence_bin": "conv-090-100",
                    "min_samples": 20,
                    "threshold_profit_factor": 0.8,
                    "reasons": self.reasons,
                    "buckets": [],
                },
            },
        )(),
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.record_balance_snapshot",
        lambda payload, market, note="", scope=None: (
            type(
                "Snapshot",
                (),
                {
                    "wallet_balance_usdt": 5000.0,
                    "available_balance_usdt": 5000.0,
                    "unrealized_pnl_usdt": 0.0,
                    "equity_usdt": 5000.0,
                },
            )(),
            ChallengeState(),
        ),
    )

    plan = build_live_execution_plan(FakeSettings(), strategy, analysis_payload, margin_notional_usdt=4.0)

    assert plan.allowed is False
    assert any("Historical feedback bucket" in item for item in plan.violations)
    assert plan.challenge["historical_signal_risk"]["allowed"] is False


def test_live_execution_plan_blocks_poor_professional_payoff(monkeypatch):
    strategy = load_strategy_config(Path("config/strategy-live-pilot.yaml"))

    class FakeSettings:
        live_trading_enabled = False

    analysis_payload = {
        "symbol": "NEARUSDT",
        "market": "futures",
        "analysis": {
            "score": 88,
            "bias": "long-bias",
            "convergence": 0.82,
        },
        "latest": {
            "close": 1.416,
            "adx": 28,
            "realized_vol_20": 0.8,
            "volume_zscore_20": 0.4,
            "obv_zscore_20": 0.4,
            "bb_bandwidth": 0.08,
        },
        "trade_plan": {
            "long": {"invalidation": 1.36, "take_profit_1": 1.425},
            "short": {"invalidation": 1.45, "take_profit_1": 1.37},
        },
    }

    _allow_historical_signal_risk(monkeypatch)
    monkeypatch.setattr("binance_quant_control.live_execution.BinanceClient", FakeClient)
    monkeypatch.setattr(
        "binance_quant_control.live_execution.route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": False, "reasons": []},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.evaluate_optimizer_live_gate",
        lambda: {"allowed": True, "reasons": [], "promotion_decision": "promote"},
    )
    monkeypatch.setattr(
        "binance_quant_control.live_execution.record_balance_snapshot",
        lambda payload, market, note="", scope=None: (
            type(
                "Snapshot",
                (),
                {
                    "wallet_balance_usdt": 50.0,
                    "available_balance_usdt": 50.0,
                    "unrealized_pnl_usdt": 0.0,
                    "equity_usdt": 50.0,
                },
            )(),
            ChallengeState(),
        ),
    )
    plan = build_live_execution_plan(
        FakeSettings(),
        strategy,
        analysis_payload,
        margin_notional_usdt=4.0,
        execution_mode="testnet_exploration",
    )

    assert plan.allowed is False
    assert plan.professional_entry_gate["passed"] is False
    assert any("Reward/risk" in item for item in plan.violations)
