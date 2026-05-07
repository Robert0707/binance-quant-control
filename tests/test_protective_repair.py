from __future__ import annotations

import binance_quant_control.protective_repair as repair
from binance_quant_control.strategy import load_strategy_config


class FakeClient:
    def __init__(self, settings):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def positions(self, symbol):
        return [{"symbol": symbol, "positionAmt": "30.1"}]

    def exchange_info(self, symbol, market):
        return {
            "symbols": [
                {
                    "symbol": symbol,
                    "quantityPrecision": 1,
                    "pricePrecision": 2,
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                        {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.1"},
                    ],
                }
            ]
        }

    def open_algo_orders(self, symbol):
        return [{"algoId": 1, "orderType": "TAKE_PROFIT_MARKET"}]

    def cancel_algo_order(self, symbol, algo_id):
        return {"algoId": algo_id, "status": "cancelled"}

    def new_algo_order(self, symbol, side, order_type, **kwargs):
        self.calls.append({"order_type": order_type, "kwargs": kwargs})
        return {"status": "ok", "type": order_type, **kwargs}


def test_staged_take_profit_repair_plan_uses_weighted_quantities(monkeypatch) -> None:
    strategy = load_strategy_config("config/strategy-major-alt-trend.yaml")
    monkeypatch.setattr(repair, "BinanceClient", lambda settings: FakeClient(settings))

    plan = repair.build_staged_take_profit_repair_plan(
        object(),
        strategy,
        symbol="APTUSDT",
        side_plan={
            "invalidation": 0.97,
            "take_profit_1": 1.03,
            "take_profit_2": 1.07,
            "take_profit_3": 1.11,
        },
        confidence=0.75,
        route_id="major-alt-trend",
        news_risk={"risk_level": "normal"},
    )

    assert plan.allowed is True
    assert plan.take_profit_weights == [0.255, 0.2975, 0.2975]
    assert plan.take_profit_quantities == [7.6, 8.9, 9.0]
    assert plan.take_profit_runner_quantity == 4.6
    assert round(sum(plan.take_profit_quantities) + plan.take_profit_runner_quantity, 1) == 30.1


class MicroBtcClient(FakeClient):
    def positions(self, symbol):
        return [{"symbol": symbol, "positionAmt": "0.001"}]

    def exchange_info(self, symbol, market):
        return {
            "symbols": [
                {
                    "symbol": symbol,
                    "quantityPrecision": 3,
                    "pricePrecision": 2,
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                        {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.001"},
                    ],
                }
            ]
        }


def test_staged_take_profit_repair_rounds_prices_and_falls_back_for_micro_position(monkeypatch) -> None:
    strategy = load_strategy_config("config/strategy-live-pilot.yaml")
    monkeypatch.setattr(repair, "BinanceClient", lambda settings: MicroBtcClient(settings))

    plan = repair.build_staged_take_profit_repair_plan(
        object(),
        strategy,
        symbol="BTCUSDT",
        side_plan={
            "invalidation": 79536.591421,
            "take_profit_1": 82548.630295,
            "take_profit_2": 84191.560589,
            "take_profit_3": 86655.956031,
        },
        confidence=0.75,
        route_id="btc-core",
        news_risk={"risk_level": "normal"},
    )

    assert plan.allowed is True
    assert plan.stop_price == 79536.5
    assert plan.take_profit_prices == [86655.9]
    assert plan.take_profit_quantities == [0.001]
    assert plan.take_profit_runner_quantity == 0.0
    assert plan.take_profit_weights == [1.0]
    assert any("too small for a staged" in item for item in plan.warnings)
