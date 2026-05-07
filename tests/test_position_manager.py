from __future__ import annotations

from binance_quant_control.position_manager import (
    AdaptiveExitPlan,
    PositionManagementPlan,
    build_adaptive_exit_plan,
    build_position_management_plan,
    execute_adaptive_exit_plan,
    execute_position_management_plan,
)
from binance_quant_control.trading_control import TradingControlState


class _FakeClient:
    def __init__(self, settings):
        self.algo_calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def open_orders(self, symbol, market):
        return []

    def cancel_all_algo_orders(self, symbol):
        return []

    def open_algo_orders(self, symbol):
        return []

    def exchange_info(self, symbol, market):
        return {
            "symbols": [
                {
                    "symbol": symbol,
                    "quantityPrecision": 8,
                    "pricePrecision": 2,
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                        {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.00000001"},
                    ],
                }
            ]
        }

    def cancel_algo_order(self, symbol, algo_id):
        return {"algoId": algo_id, "status": "cancelled"}

    def new_algo_order(self, symbol, side, order_type, **kwargs):
        self.algo_calls.append({"symbol": symbol, "side": side, "order_type": order_type, "kwargs": kwargs})
        return {"status": "ok", "type": order_type, **kwargs}

    def new_order(self, symbol, side, order_type, **kwargs):
        self.algo_calls.append({"symbol": symbol, "side": side, "order_type": order_type, "kwargs": kwargs})
        return {"status": "ok", "type": order_type, **kwargs}


def test_execute_position_management_plan_uses_algo_api_for_trailing(monkeypatch) -> None:
    fake_client = _FakeClient(None)
    monkeypatch.setattr(
        "binance_quant_control.position_manager.BinanceClient",
        lambda settings: fake_client,
    )
    monkeypatch.setattr(
        "binance_quant_control.position_manager.load_trading_control_state",
        lambda: TradingControlState(paused=False, reason="", updated_at="", updated_by="test"),
    )
    plan = PositionManagementPlan(
        allowed=True,
        symbol="BTCUSDT",
        market="futures",
        side="BUY",
        quantity=0.01,
        entry_price=100000.0,
        mark_price=101000.0,
        unrealized_pnl_usdt=10.0,
        leverage=3,
        existing_open_orders=[],
        existing_algo_orders=[],
        step_size=0.00000001,
        tick_size=0.01,
        quantity_precision=8,
        price_precision=2,
        proposed_stop_price=None,
        proposed_take_profit_price=None,
        trailing_activation_price=100500.0,
        trailing_callback_pct=0.7,
        trailing_quantity=None,
        cancel_existing_algo_orders=False,
        preserve_existing_take_profits=False,
        violations=[],
        warnings=[],
        actions=["place trailing"],
    )
    result = execute_position_management_plan(object(), plan)
    assert result["submitted"]["trailing_stop"]["type"] == "TRAILING_STOP_MARKET"
    assert fake_client.algo_calls[0]["kwargs"]["callback_rate"] == 0.7


def test_execute_position_management_plan_blocks_when_paused(monkeypatch) -> None:
    monkeypatch.setattr(
        "binance_quant_control.position_manager.load_trading_control_state",
        lambda: TradingControlState(paused=True, reason="manual stop", updated_at="", updated_by="test"),
    )
    plan = PositionManagementPlan(
        allowed=True,
        symbol="BTCUSDT",
        market="futures",
        side="BUY",
        quantity=0.01,
        entry_price=100000.0,
        mark_price=101000.0,
        unrealized_pnl_usdt=10.0,
        leverage=3,
        existing_open_orders=[],
        existing_algo_orders=[],
        step_size=0.00000001,
        tick_size=0.01,
        quantity_precision=8,
        price_precision=2,
        proposed_stop_price=99500.0,
        proposed_take_profit_price=None,
        trailing_activation_price=None,
        trailing_callback_pct=None,
        trailing_quantity=None,
        cancel_existing_algo_orders=False,
        preserve_existing_take_profits=False,
        violations=[],
        warnings=[],
        actions=["place stop"],
    )
    try:
        execute_position_management_plan(object(), plan)
    except RuntimeError as exc:
        assert "kill-switch" in str(exc)
    else:
        raise AssertionError("paused trading should block position management writes")


def test_execute_position_management_plan_allows_auto_pause_protective_updates(monkeypatch) -> None:
    fake_client = _FakeClient(None)
    monkeypatch.setattr(
        "binance_quant_control.position_manager.BinanceClient",
        lambda settings: fake_client,
    )
    monkeypatch.setattr(
        "binance_quant_control.position_manager.load_trading_control_state",
        lambda: TradingControlState(
            paused=True,
            reason="position timeout",
            updated_at="",
            updated_by="openclaw-quantctl auto-pause-trading",
        ),
    )
    plan = PositionManagementPlan(
        allowed=True,
        symbol="BTCUSDT",
        market="futures",
        side="BUY",
        quantity=0.01,
        entry_price=100000.0,
        mark_price=101000.0,
        unrealized_pnl_usdt=10.0,
        leverage=3,
        existing_open_orders=[],
        existing_algo_orders=[],
        step_size=0.00000001,
        tick_size=0.01,
        quantity_precision=8,
        price_precision=2,
        proposed_stop_price=99500.0,
        proposed_take_profit_price=None,
        trailing_activation_price=None,
        trailing_callback_pct=None,
        trailing_quantity=None,
        cancel_existing_algo_orders=False,
        preserve_existing_take_profits=False,
        violations=[],
        warnings=[],
        actions=["place stop"],
    )

    result = execute_position_management_plan(object(), plan)

    assert result["submitted"]["stop_loss"]["type"] == "STOP_MARKET"


def test_trailing_plan_does_not_replace_protection_before_activation(monkeypatch) -> None:
    class FakeReadClient:
        def __init__(self, settings):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def positions(self, symbol=None):
            return [
                {
                    "symbol": "BTCUSDT",
                    "positionAmt": "0.01",
                    "entryPrice": "100000",
                    "markPrice": "100500",
                    "unRealizedProfit": "5",
                    "leverage": "3",
                }
            ]

        def open_orders(self, symbol, market):
            return []

        def open_algo_orders(self, symbol):
            return [
                {
                    "algoId": 1,
                    "orderType": "STOP_MARKET",
                    "triggerPrice": "99000",
                }
            ]

        def exchange_info(self, symbol, market):
            return {
                "symbols": [
                    {
                        "symbol": symbol,
                        "quantityPrecision": 8,
                        "pricePrecision": 2,
                        "filters": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                            {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.001"},
                        ],
                    }
                ]
            }

    monkeypatch.setattr(
        "binance_quant_control.position_manager.BinanceClient",
        lambda settings: FakeReadClient(settings),
    )

    plan = build_position_management_plan(
        object(),
        symbol="BTCUSDT",
        market="futures",
        enable_trailing_stop=True,
        trailing_callback_pct=0.7,
        trailing_activation_price=101000.0,
    )

    assert plan.allowed is True
    assert plan.cancel_existing_algo_orders is False
    assert plan.actions == []
    assert any("not armed yet" in warning for warning in plan.warnings)


def test_trailing_plan_preserves_existing_take_profit_ladder(monkeypatch) -> None:
    class FakeReadClient:
        def __init__(self, settings):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def positions(self, symbol=None):
            return [
                {
                    "symbol": "APTUSDT",
                    "positionAmt": "30.1",
                    "entryPrice": "1.0",
                    "markPrice": "1.04",
                    "unRealizedProfit": "1.2",
                    "leverage": "5",
                }
            ]

        def open_orders(self, symbol, market):
            return []

        def open_algo_orders(self, symbol):
            return [
                {"algoId": 1, "orderType": "STOP_MARKET", "triggerPrice": "0.97", "quantity": "30.1"},
                {"algoId": 2, "orderType": "TAKE_PROFIT_MARKET", "triggerPrice": "1.05", "quantity": "9.0"},
                {"algoId": 3, "orderType": "TAKE_PROFIT_MARKET", "triggerPrice": "1.09", "quantity": "12.0"},
            ]

        def exchange_info(self, symbol, market):
            return {
                "symbols": [
                    {
                        "symbol": symbol,
                        "quantityPrecision": 1,
                        "pricePrecision": 4,
                        "filters": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0.0001"},
                            {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.1"},
                        ],
                    }
                ]
            }

    monkeypatch.setattr(
        "binance_quant_control.position_manager.BinanceClient",
        lambda settings: FakeReadClient(settings),
    )

    plan = build_position_management_plan(
        object(),
        symbol="APTUSDT",
        market="futures",
        enable_trailing_stop=True,
        trailing_callback_pct=0.7,
        trailing_activation_price=1.02,
    )

    assert plan.preserve_existing_take_profits is True
    assert plan.cancel_existing_algo_orders is False
    assert plan.trailing_quantity == 9.1
    assert plan.proposed_take_profit_price is None
    assert any("runner quantity" in action for action in plan.actions)


def test_trailing_plan_skips_when_full_tp_micro_position_has_no_runner(monkeypatch) -> None:
    class FakeReadClient:
        def __init__(self, settings):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def positions(self, symbol=None):
            return [
                {
                    "symbol": "BTCUSDT",
                    "positionAmt": "0.001",
                    "entryPrice": "80883.5",
                    "markPrice": "82000.0",
                    "unRealizedProfit": "1.1165",
                    "leverage": "3",
                }
            ]

        def open_orders(self, symbol, market):
            return []

        def open_algo_orders(self, symbol):
            return [
                {"algoId": 1, "orderType": "STOP_MARKET", "triggerPrice": "79536.5", "quantity": "0.001"},
                {"algoId": 2, "orderType": "TAKE_PROFIT_MARKET", "triggerPrice": "86655.9", "quantity": "0.001"},
            ]

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

    monkeypatch.setattr(
        "binance_quant_control.position_manager.BinanceClient",
        lambda settings: FakeReadClient(settings),
    )

    plan = build_position_management_plan(
        object(),
        symbol="BTCUSDT",
        market="futures",
        enable_trailing_stop=True,
        trailing_callback_pct=0.7,
        trailing_activation_price=81000.0,
    )

    assert plan.allowed is True
    assert plan.trailing_quantity is None
    assert plan.actions == []
    assert any("cover the full position" in warning for warning in plan.warnings)


def test_execute_position_management_preserves_take_profit_orders_when_trailing(monkeypatch) -> None:
    fake_client = _FakeClient(None)
    fake_client.open_algo_orders = lambda symbol: [
        {"algoId": 1, "orderType": "STOP_MARKET", "quantity": "30.1"},
        {"algoId": 2, "orderType": "TAKE_PROFIT_MARKET", "quantity": "9.0"},
        {"algoId": 3, "orderType": "TAKE_PROFIT_MARKET", "quantity": "12.0"},
    ]
    monkeypatch.setattr(
        "binance_quant_control.position_manager.BinanceClient",
        lambda settings: fake_client,
    )
    monkeypatch.setattr(
        "binance_quant_control.position_manager.load_trading_control_state",
        lambda: TradingControlState(paused=False, reason="", updated_at="", updated_by="test"),
    )
    plan = PositionManagementPlan(
        allowed=True,
        symbol="APTUSDT",
        market="futures",
        side="BUY",
        quantity=30.1,
        entry_price=1.0,
        mark_price=1.04,
        unrealized_pnl_usdt=1.2,
        leverage=5,
        existing_open_orders=[],
        existing_algo_orders=[],
        step_size=0.1,
        tick_size=0.0001,
        quantity_precision=1,
        price_precision=4,
        proposed_stop_price=None,
        proposed_take_profit_price=None,
        trailing_activation_price=1.02,
        trailing_callback_pct=0.7,
        trailing_quantity=9.1,
        cancel_existing_algo_orders=True,
        preserve_existing_take_profits=True,
        violations=[],
        warnings=[],
        actions=["replace stop/trailing"],
    )

    result = execute_position_management_plan(object(), plan)

    assert result["cancelled_algo_orders"] == []
    assert fake_client.algo_calls[0]["order_type"] == "TRAILING_STOP_MARKET"
    assert fake_client.algo_calls[0]["kwargs"]["quantity"] == 9.1


def test_execute_position_management_replaces_only_old_trailing_when_preserving_ladder(monkeypatch) -> None:
    fake_client = _FakeClient(None)
    fake_client.open_algo_orders = lambda symbol: [
        {"algoId": 1, "orderType": "STOP_MARKET", "quantity": "30.1"},
        {"algoId": 2, "orderType": "TAKE_PROFIT_MARKET", "quantity": "9.0"},
        {"algoId": 3, "orderType": "TAKE_PROFIT_MARKET", "quantity": "12.0"},
        {"algoId": 4, "orderType": "TRAILING_STOP_MARKET", "quantity": "9.1"},
    ]
    monkeypatch.setattr(
        "binance_quant_control.position_manager.BinanceClient",
        lambda settings: fake_client,
    )
    monkeypatch.setattr(
        "binance_quant_control.position_manager.load_trading_control_state",
        lambda: TradingControlState(paused=False, reason="", updated_at="", updated_by="test"),
    )
    plan = PositionManagementPlan(
        allowed=True,
        symbol="APTUSDT",
        market="futures",
        side="BUY",
        quantity=30.1,
        entry_price=1.0,
        mark_price=1.04,
        unrealized_pnl_usdt=1.2,
        leverage=5,
        existing_open_orders=[],
        existing_algo_orders=[],
        step_size=0.1,
        tick_size=0.0001,
        quantity_precision=1,
        price_precision=4,
        proposed_stop_price=None,
        proposed_take_profit_price=None,
        trailing_activation_price=1.02,
        trailing_callback_pct=0.7,
        trailing_quantity=9.1,
        cancel_existing_algo_orders=True,
        preserve_existing_take_profits=True,
        violations=[],
        warnings=[],
        actions=["replace old trailing"],
    )

    result = execute_position_management_plan(object(), plan)

    assert result["cancelled_algo_orders"] == [{"algoId": 4, "status": "cancelled"}]
    assert fake_client.algo_calls[0]["order_type"] == "TRAILING_STOP_MARKET"


def _active_long_plan() -> PositionManagementPlan:
    return PositionManagementPlan(
        allowed=True,
        symbol="BTCUSDT",
        market="futures",
        side="BUY",
        quantity=0.01,
        entry_price=100000.0,
        mark_price=100600.0,
        unrealized_pnl_usdt=6.0,
        leverage=3,
        existing_open_orders=[],
        existing_algo_orders=[{"algoId": 1, "orderType": "STOP_MARKET", "triggerPrice": "99000", "quantity": "0.01"}],
        step_size=0.001,
        tick_size=0.1,
        quantity_precision=3,
        price_precision=1,
        proposed_stop_price=None,
        proposed_take_profit_price=None,
        trailing_activation_price=None,
        trailing_callback_pct=None,
        trailing_quantity=None,
        cancel_existing_algo_orders=False,
        preserve_existing_take_profits=False,
        violations=[],
        warnings=[],
        actions=[],
    )


def test_adaptive_exit_closes_profitable_long_on_confirmed_reversal() -> None:
    analysis_payload = {
        "analysis": {
            "bias": "short",
            "recommended_action": "SELL",
            "selected_strategy_family": {"family": "trend_continuation", "bias": "short"},
        },
        "latest": {
            "close": 100400.0,
            "ema_fast": 100300.0,
            "ema_slow": 100500.0,
            "macd_hist": -25.0,
            "adx": 22.0,
            "plus_di": 12.0,
            "minus_di": 25.0,
            "supertrend_direction": -1,
            "trend_magic_direction": -1,
            "follow_line_direction": -1,
            "chandelier_direction": -1,
            "qqe_direction": -1,
            "psar_direction": -1,
            "ichimoku_direction": -1,
        },
    }

    plan = build_adaptive_exit_plan(_active_long_plan(), analysis_payload)

    assert plan.allowed is True
    assert plan.action == "close_position"
    assert plan.reason_code == "profit-protection-reversal"
    assert plan.exit_side == "SELL"
    assert plan.unrealized_r == 0.6


def test_adaptive_exit_holds_when_reversal_is_not_confirmed() -> None:
    analysis_payload = {
        "analysis": {
            "bias": "long",
            "recommended_action": "BUY",
            "selected_strategy_family": {"family": "trend_continuation", "bias": "long"},
        },
        "latest": {
            "close": 100700.0,
            "ema_fast": 100650.0,
            "ema_slow": 100200.0,
            "macd_hist": 15.0,
            "adx": 20.0,
            "plus_di": 22.0,
            "minus_di": 13.0,
            "supertrend_direction": 1,
            "trend_magic_direction": 1,
            "follow_line_direction": 1,
        },
    }

    plan = build_adaptive_exit_plan(_active_long_plan(), analysis_payload)

    assert plan.allowed is False
    assert plan.action == "hold"
    assert plan.reason_code == "no-confirmed-reversal"


def test_execute_adaptive_exit_plan_submits_reduce_only_market_close(monkeypatch) -> None:
    fake_client = _FakeClient(None)
    monkeypatch.setattr(
        "binance_quant_control.position_manager.BinanceClient",
        lambda settings: fake_client,
    )
    monkeypatch.setattr(
        "binance_quant_control.position_manager.load_trading_control_state",
        lambda: TradingControlState(paused=False, reason="", updated_at="", updated_by="test"),
    )
    plan = AdaptiveExitPlan(
        allowed=True,
        symbol="BTCUSDT",
        market="futures",
        side="BUY",
        exit_side="SELL",
        quantity=0.01,
        action="close_position",
        reason_code="profit-protection-reversal",
        reasons=["test"],
        warnings=[],
        unrealized_r=0.6,
        reversal_score=8.5,
        confidence=1.0,
        risk_distance=1000.0,
        reference_stop_price=99000.0,
        entry_price=100000.0,
        mark_price=100600.0,
        analysis_bias="short",
        recommended_action="SELL",
        selected_family="trend_continuation",
        selected_family_bias="short",
    )

    result = execute_adaptive_exit_plan(object(), plan)

    assert result["submitted"]["market_close"]["type"] == "MARKET"
    assert fake_client.algo_calls[0]["side"] == "SELL"
    assert fake_client.algo_calls[0]["kwargs"]["reduce_only"] is True
