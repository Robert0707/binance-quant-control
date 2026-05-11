from __future__ import annotations

from binance_quant_control.asset_routing import resolve_symbol_route
from binance_quant_control.signal_scoring import build_signal_scores, score_flow


def test_flow_score_is_side_aware() -> None:
    latest = {
        "volume_zscore_20": 0.8,
        "obv_zscore_20": -1.2,
        "taker_buy_sell_ratio": 0.75,
        "order_book_imbalance": -0.35,
        "whale_hunter_24h_oi": -0.2,
    }

    assert score_flow(latest, side="SELL") > score_flow(latest, side="BUY")


def test_execution_quality_uses_short_trade_plan_for_sell() -> None:
    route = resolve_symbol_route("ETHUSDT")
    latest = {
        "close": 100.0,
        "adx": 28.0,
        "volume_zscore_20": 0.5,
        "obv_zscore_20": -0.8,
        "taker_buy_sell_ratio": 0.8,
        "order_book_imbalance": -0.2,
    }
    analysis = {"score": 35, "convergence": 0.82}
    trade_plan = {
        "long": {"invalidation": 98.0, "take_profit_1": 101.0},
        "short": {"invalidation": 103.0, "take_profit_1": 94.0, "take_profit_2": 90.0},
    }

    buy_scores = build_signal_scores(
        route=route,
        latest=latest,
        analysis=analysis,
        trade_plan=trade_plan,
        side="BUY",
    )
    sell_scores = build_signal_scores(
        route=route,
        latest=latest,
        analysis=analysis,
        trade_plan=trade_plan,
        side="SELL",
    )

    assert sell_scores["flow_score"] > buy_scores["flow_score"]
    assert sell_scores["execution_quality_score"] > buy_scores["execution_quality_score"]
