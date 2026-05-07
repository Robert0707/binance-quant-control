from __future__ import annotations

from binance_quant_control.strategy_analyzer import analyze_strategy_payload


def test_analyze_strategy_payload_returns_compact_verdict() -> None:
    payload = {
        "selected": {"symbol": "BTCUSDT", "direction": "long", "composite_score": 88.0},
        "decision": {"action": "pre_trade_notify", "should_notify": True},
        "news": {"risk": {"risk_level": "normal"}},
        "whale": {"signal": "bullish"},
        "doctor": {"overall": "ok"},
    }

    result = analyze_strategy_payload(payload)

    assert result["status"] == "ok"
    assert result["symbol"] == "BTCUSDT"
    assert result["result"]["verdict"] == "approve"
    assert result["result"]["bias"] == "long"
    assert 0.6 <= result["result"]["confidence"] <= 1.0
