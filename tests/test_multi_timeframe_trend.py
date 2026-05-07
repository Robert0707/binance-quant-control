from __future__ import annotations

import pandas as pd

from binance_quant_control.analysis import (
    evaluate_multi_timeframe_structure,
    evaluate_multi_timeframe_trend,
)


def test_multi_timeframe_trend_favors_aligned_direction():
    htf = pd.DataFrame({"close": [100, 102, 104, 106, 108]})
    ltf = pd.DataFrame({"close": [107, 108, 109, 110, 111]})

    result = evaluate_multi_timeframe_trend(ltf, htf)

    assert result.bias == "long"
    assert result.confidence > 0.5


def test_multi_timeframe_trend_favors_short_when_both_timeframes_roll_over():
    htf = pd.DataFrame({"close": [108, 106, 104, 102, 100]})
    ltf = pd.DataFrame({"close": [101, 100, 99, 98, 97]})

    result = evaluate_multi_timeframe_trend(ltf, htf)

    assert result.bias == "short"
    assert result.confidence > 0.5


def test_multi_timeframe_trend_is_neutral_when_context_is_mixed():
    htf = pd.DataFrame({"close": [100, 101, 102, 103, 104]})
    ltf = pd.DataFrame({"close": [104, 103, 104, 103, 104]})

    result = evaluate_multi_timeframe_trend(ltf, htf)

    assert result.bias == "neutral"
    assert result.confidence <= 0.6


def test_multi_timeframe_structure_uses_ema_7_25_89_alignment() -> None:
    base = pd.DataFrame(
        {
            "open": [float(100 + idx) for idx in range(140)],
            "high": [float(101 + idx) for idx in range(140)],
            "low": [float(99 + idx) for idx in range(140)],
            "close": [float(100 + idx) for idx in range(140)],
            "volume": [1000.0 + idx for idx in range(140)],
        }
    )

    result = evaluate_multi_timeframe_structure(
        {
            "15m": base,
            "4h": base.assign(close=base["close"] + 10.0),
            "1d": base.assign(close=base["close"] + 20.0),
        }
    )

    assert result.bias == "long"
    assert result.alignment == "strong"
    assert result.structures[0].ema7 > result.structures[0].ema25 > result.structures[0].ema89
