from binance_quant_control.signals import SignalResult, combine_signals, decide_trade_action


def test_combine_signals_prefers_high_confidence_bias():
    signals = [
        SignalResult(name="trend", bias="long", score=0.8, confidence=0.9),
        SignalResult(name="structure", bias="short", score=-0.2, confidence=0.4),
    ]

    result = combine_signals(signals)

    assert result.bias == "long"


def test_combine_signals_is_deterministic_on_weight_tie():
    signals = [
        SignalResult(name="a", bias="long", score=1.0, confidence=0.5),
        SignalResult(name="b", bias="short", score=-1.0, confidence=0.5),
    ]

    result = combine_signals(signals)

    assert result.bias == "long"
    assert result.confidence == 0.5


def test_signal_result_defaults_to_neutral_when_not_combined():
    result = SignalResult(name="risk", bias="neutral", score=0.0, confidence=0.0)

    assert result.bias == "neutral"
    assert result.score == 0.0


def test_decide_trade_action_blocks_range_entries_even_with_high_score():
    decision = decide_trade_action(
        market="futures",
        score=82,
        convergence=0.88,
        adx_value=29.0,
        regime="range",
        min_score_long=70,
        max_score_short=30,
        min_convergence=0.75,
        min_adx=20.0,
    )

    assert decision.action == "HOLD"
    assert decision.directional_bias == "long-bias"
    assert any("Range regime" in item for item in decision.blockers)


def test_decide_trade_action_allows_short_only_in_futures():
    decision = decide_trade_action(
        market="futures",
        score=25,
        convergence=0.81,
        adx_value=31.0,
        regime="trend-down",
        min_score_long=70,
        max_score_short=30,
        min_convergence=0.75,
        min_adx=20.0,
    )

    assert decision.allowed is True
    assert decision.action == "SELL"
