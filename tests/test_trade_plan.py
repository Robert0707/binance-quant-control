from __future__ import annotations

import pandas as pd

from binance_quant_control.analysis import build_trade_plan, indicator_trade_plan_side
from binance_quant_control.strategy import load_strategy_config


def _latest_row(**overrides) -> pd.Series:
    row = {
        "close": 100.0,
        "atr_14": 2.0,
        "supertrend": 97.0,
        "trend_magic": 95.0,
        "follow_line": 96.0,
        "chandelier_long_stop": 94.0,
        "chandelier_short_stop": 106.0,
        "psar": 96.5,
        "donchian_lower": 92.0,
        "donchian_upper": 108.0,
        "keltner_lower": 93.0,
        "keltner_upper": 107.0,
        "bb_lower": 91.0,
        "bb_basis": 99.0,
        "bb_upper": 109.0,
    }
    row.update(overrides)
    return pd.Series(row)


def test_indicator_trade_plan_uses_trend_line_for_long_stop() -> None:
    strategy = load_strategy_config("config/strategy-aggressive-alpha-research.yaml")

    plan = indicator_trade_plan_side(
        _latest_row(),
        side="BUY",
        strategy=strategy,
        family="trend_continuation",
    )

    assert plan["invalidation_source"] == "psar_buffered"
    assert plan["invalidation"] == 96.2
    assert plan["risk_distance_source"] == "indicator"
    assert plan["take_profit_1"] == 105.32


def test_indicator_trade_plan_uses_short_trend_line_symmetrically() -> None:
    strategy = load_strategy_config("config/strategy-aggressive-alpha-research.yaml")

    plan = indicator_trade_plan_side(
        _latest_row(supertrend=103.0, trend_magic=104.0, follow_line=105.0, psar=103.5),
        side="SELL",
        strategy=strategy,
        family="trend_continuation",
    )

    assert plan["invalidation_source"] == "psar_buffered"
    assert plan["invalidation"] == 103.8
    assert plan["risk_distance_source"] == "indicator"
    assert plan["take_profit_1"] == 94.68


def test_mean_reversion_prefers_bollinger_structure_stop() -> None:
    strategy = load_strategy_config("config/strategy-aggressive-alpha-research.yaml")

    plan = build_trade_plan(
        _latest_row(bb_lower=97.2, keltner_lower=95.0),
        "long-bias",
        "futures",
        strategy=strategy,
        analysis={"selected_strategy_family": {"family": "mean_reversion"}},
    )

    assert plan["long"]["strategy_family"] == "mean_reversion"
    assert plan["long"]["invalidation_source"] == "bb_lower_buffered"
    assert plan["long"]["invalidation"] == 96.9
    assert plan["long"]["take_profit_1"] == 104.34


def test_trend_pullback_prefers_fibonacci_structure_stop() -> None:
    strategy = load_strategy_config("config/strategy-aggressive-alpha-research.yaml")

    plan = build_trade_plan(
        _latest_row(fib_786_long_89=96.8, fib_swing_low_89=93.0, trend_magic=94.0, follow_line=95.0),
        "long-bias",
        "futures",
        strategy=strategy,
        analysis={"selected_strategy_family": {"family": "trend_pullback"}},
    )

    assert plan["long"]["strategy_family"] == "trend_pullback"
    assert plan["long"]["invalidation_source"] == "fib_786_long_89_buffered"
    assert plan["long"]["risk_distance_source"] == "indicator"
