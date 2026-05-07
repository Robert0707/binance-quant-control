# BTCUSDT Futures Quant Control v2 Implementation Plan

> **For Hermes:** Use `subagent-driven-development` to implement this plan task-by-task.

**Goal:** Upgrade the Binance quant control plane from a single-timeframe indicator stack into a multi-layer trading control system with stronger regime filters, market structure signals, derivative sentiment inputs, and explicit strategy routing.

**Architecture:**
Keep the current analysis pipeline as the base layer, then add separate signal layers for trend, structure, volatility, derivatives, and event risk. Each layer should emit a small, typed signal object that is easy to test in isolation. The final score should be assembled from these layers with configurable weights, so the strategy can switch between trend-following, breakout, mean-reversion, and risk-off modes without hardcoding everything into one monolithic function.

**Tech Stack:**
- Python 3.11
- pandas / numpy
- pytest
- existing `binance_quant_control` package
- existing OpenClaw workflow runner and wrapper scripts

---

## Current baseline

The control plane already has:
- `doctor`, `analyze`, `account`, `positions`, `paper-order`, `build-analysis-spec`, `submit-analysis`
- core indicators in `src/binance_quant_control/indicators.py`
- strategy scoring and report generation in `src/binance_quant_control/analysis.py`
- OpenClaw routing via `openclaw_hostd.py`
- testnet-first defaults in `src/binance_quant_control/config.py`

This plan builds on that baseline rather than replacing it.

---

## Target upgrades

1. Multi-timeframe trend filter
2. Market structure / liquidity sweep signals
3. Volatility squeeze / expansion regime
4. Derivatives sentiment inputs when available
5. Event/news risk veto layer
6. Strategy router that can select:
   - trend-following
   - breakout
   - mean-reversion
   - risk-off / no-trade
7. Clear test coverage for each layer

---

## Task 1: Add a signal-layer module skeleton

**Objective:** Create a home for typed signal objects and signal-layer assembly so analysis logic stops growing inside one function.

**Files:**
- Create: `src/binance_quant_control/signals.py`
- Modify: `src/binance_quant_control/__init__.py`
- Test: `tests/test_signals.py`

**Step 1: Write failing test**

```python
from binance_quant_control.signals import SignalResult, combine_signals


def test_combine_signals_prefers_high_confidence_bias():
    signals = [
        SignalResult(name="trend", bias="long", score=0.8, confidence=0.9),
        SignalResult(name="structure", bias="short", score=-0.2, confidence=0.4),
    ]
    result = combine_signals(signals)
    assert result.bias == "long"
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_signals.py -v`
Expected: fail because module and types do not exist yet.

**Step 3: Write minimal implementation**

Implement a small dataclass for `SignalResult` and a deterministic `combine_signals()` helper that uses confidence-weighted voting.

**Step 4: Run test to verify pass**

Run: `pytest tests/test_signals.py -v`
Expected: pass.

---

## Task 2: Add multi-timeframe trend filter

**Objective:** Use 4h and 1d context to prevent 1h entries that fight the higher-timeframe direction.

**Files:**
- Modify: `src/binance_quant_control/analysis.py`
- Modify: `src/binance_quant_control/indicators.py`
- Test: `tests/test_multi_timeframe_trend.py`

**Step 1: Write failing test**

```python
import pandas as pd
from binance_quant_control.analysis import evaluate_multi_timeframe_trend


def test_multi_timeframe_trend_favors_aligned_direction():
    htf = pd.DataFrame({"close": [100, 102, 104, 106, 108]})
    ltf = pd.DataFrame({"close": [107, 108, 109, 110, 111]})
    result = evaluate_multi_timeframe_trend(ltf, htf)
    assert result.bias == "long"
    assert result.confidence > 0.5
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_multi_timeframe_trend.py -v`
Expected: fail until helper exists.

**Step 3: Minimal implementation**

Add a helper that compares higher-timeframe EMA slope, price location vs EMA200, and lower-timeframe alignment.

**Step 4: Run test to verify pass**

Run: `pytest tests/test_multi_timeframe_trend.py -v`
Expected: pass.

---

## Task 3: Add market structure signals

**Objective:** Detect break of structure, sweep, and reclaim patterns so the strategy can differentiate true breakouts from trap moves.

**Files:**
- Modify: `src/binance_quant_control/analysis.py`
- Modify: `src/binance_quant_control/indicators.py`
- Test: `tests/test_market_structure.py`

**Signals to add:**
- recent swing high / swing low
- break of structure
- liquidity sweep above/below swing
- reclaim / rejection after sweep

**Step 1: Write failing test**

```python
import pandas as pd
from binance_quant_control.analysis import detect_market_structure


def test_detect_market_structure_finds_bullish_reclaim():
    df = pd.DataFrame(
        {
            "high": [10, 11, 12, 11, 13, 14],
            "low": [8, 9, 10, 9, 11, 12],
            "close": [9, 10, 11, 10, 12, 13],
            "volume": [100, 110, 120, 115, 140, 150],
        }
    )
    result = detect_market_structure(df)
    assert result.has_bos or result.has_reclaim
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_market_structure.py -v`
Expected: fail until the detector is implemented.

**Step 3: Minimal implementation**

Use a simple swing-high / swing-low window and emit a structure object with boolean flags and a short explanation.

**Step 4: Run test to verify pass**

Run: `pytest tests/test_market_structure.py -v`
Expected: pass.

---

## Task 4: Add volatility squeeze / expansion regime

**Objective:** Only allow breakout-style strategies when the market has actually compressed first.

**Files:**
- Modify: `src/binance_quant_control/analysis.py`
- Test: `tests/test_volatility_regime.py`

**Step 1: Write failing test**

```python
import pandas as pd
from binance_quant_control.analysis import classify_volatility_regime


def test_classify_volatility_regime_detects_squeeze():
    series = pd.Series([100, 100.2, 100.1, 100.25, 100.3, 100.28, 100.31])
    result = classify_volatility_regime(series)
    assert result.regime in {"squeeze", "quiet"}
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_volatility_regime.py -v`
Expected: fail until function exists.

**Step 3: Minimal implementation**

Base the regime on Bollinger bandwidth, ATR compression, and rolling return dispersion.

**Step 4: Run test to verify pass**

Run: `pytest tests/test_volatility_regime.py -v`
Expected: pass.

---

## Task 5: Add derivatives sentiment adapter

**Objective:** Make funding rate, open interest, and long/short ratio optional inputs so futures analysis can react to positioning risk.

**Files:**
- Create: `src/binance_quant_control/derivatives.py`
- Modify: `src/binance_quant_control/analysis.py`
- Test: `tests/test_derivatives_adapter.py`

**Step 1: Write failing test**

```python
from binance_quant_control.derivatives import score_derivatives_sentiment


def test_score_derivatives_sentiment_handles_extreme_funding():
    payload = {"funding_rate": 0.01, "open_interest_change": 0.2, "long_short_ratio": 1.8}
    result = score_derivatives_sentiment(payload)
    assert result.bias in {"long", "short", "neutral"}
    assert result.risk_level in {"low", "medium", "high"}
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_derivatives_adapter.py -v`
Expected: fail until adapter exists.

**Step 3: Minimal implementation**

Return a small scored object with a risk veto when funding is extreme or open interest spikes without price confirmation.

**Step 4: Run test to verify pass**

Run: `pytest tests/test_derivatives_adapter.py -v`
Expected: pass.

---

## Task 6: Add event-risk veto input

**Objective:** Make the news digest useful for trading by turning it into a simple no-trade veto and risk state.

**Files:**
- Create: `src/binance_quant_control/event_risk.py`
- Modify: `src/binance_quant_control/analysis.py`
- Test: `tests/test_event_risk.py`

**Step 1: Write failing test**

```python
from binance_quant_control.event_risk import evaluate_event_risk


def test_event_risk_blocks_high_impact_macro_event():
    events = [
        {"category": "macro", "impact": "high", "headline": "CPI release in 30 minutes"}
    ]
    result = evaluate_event_risk(events)
    assert result.allow_new_entries is False
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_event_risk.py -v`
Expected: fail until helper exists.

**Step 3: Minimal implementation**

Return a simple allow/block decision plus a compact reason string.

**Step 4: Run test to verify pass**

Run: `pytest tests/test_event_risk.py -v`
Expected: pass.

---

## Task 7: Add strategy router

**Objective:** Route each market state to one of four strategies instead of using a single score for every environment.

**Files:**
- Modify: `src/binance_quant_control/analysis.py`
- Create: `src/binance_quant_control/strategy_router.py`
- Test: `tests/test_strategy_router.py`

**Strategies:**
- `trend_following`
- `breakout`
- `mean_reversion`
- `risk_off`

**Step 1: Write failing test**

```python
from binance_quant_control.strategy_router import route_strategy


def test_route_strategy_uses_risk_off_when_event_risk_high():
    result = route_strategy(
        trend_bias="long",
        volatility_regime="squeeze",
        structure_bias="neutral",
        event_risk="high",
    )
    assert result.strategy == "risk_off"
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_strategy_router.py -v`
Expected: fail until router exists.

**Step 3: Minimal implementation**

Hardcode the first version as a small rule engine with explicit precedence:
1. risk_off
2. breakout
3. trend_following
4. mean_reversion

**Step 4: Run test to verify pass**

Run: `pytest tests/test_strategy_router.py -v`
Expected: pass.

---

## Task 8: Wire the new layers into `analyze`

**Objective:** Merge the new signal layers into the report output without breaking the current CLI.

**Files:**
- Modify: `src/binance_quant_control/analysis.py`
- Modify: `src/binance_quant_control/cli.py`
- Test: `tests/test_analyze_payload.py`

**Expected payload additions:**
- `multi_timeframe_trend`
- `market_structure`
- `volatility_regime`
- `derivatives_sentiment`
- `event_risk`
- `strategy_route`

**Step 1: Write failing test**

```python
from binance_quant_control.analysis import build_analysis_payload


def test_analysis_payload_contains_strategy_route():
    payload = build_analysis_payload(...)
    assert "strategy_route" in payload
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_analyze_payload.py -v`
Expected: fail until payload is extended.

**Step 3: Minimal implementation**

Keep the legacy fields intact, add new signal sections, and ensure report rendering includes them.

**Step 4: Run test to verify pass**

Run: `pytest tests/test_analyze_payload.py -v`
Expected: pass.

---

## Task 9: Add strategy config example

**Objective:** Make the strategy weights editable without code changes.

**Files:**
- Create: `config/strategy-v2.example.yaml`
- Modify: `src/binance_quant_control/config.py`
- Test: `tests/test_strategy_config.py`

**Config fields to include:**
- layer weights
- regime thresholds
- event veto thresholds
- allowed strategies by market type
- default leverage caps per regime

**Step 1: Write failing test**

```python
from binance_quant_control.config import load_strategy_config


def test_strategy_config_loads_defaults():
    cfg = load_strategy_config()
    assert cfg.layer_weights["trend"] > 0
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_strategy_config.py -v`
Expected: fail until loader exists.

**Step 3: Minimal implementation**

Load YAML if present, otherwise fall back to safe defaults.

**Step 4: Run test to verify pass**

Run: `pytest tests/test_strategy_config.py -v`
Expected: pass.

---

## Task 10: Refresh documentation and workflow spec

**Objective:** Make the new strategy layers visible in the control-plane docs and workflow artifacts.

**Files:**
- Modify: `PROJECT.md`
- Modify: `src/binance_quant_control/cli.py`
- Modify: `~/.openclaw/runtime/binance-quant/task-specs/*.json` generation path via code
- Create: `docs/strategy-v2.md`

**Include:**
- What each strategy is for
- When to use it
- When to block trading
- What each layer contributes
- Safe-default wording that live trading remains off unless explicitly enabled

**Verification:**
- Run `openclaw-quantctl build-analysis-spec BTCUSDT --market futures --interval 1h --render-chart`
- Confirm the generated spec references the new strategy fields or version tag
- Run `openclaw-quantctl submit-analysis BTCUSDT --market futures --interval 1h --render-chart --run-now`
- Confirm the workflow still completes and artifacts are produced

---

## Final verification checklist

- [ ] All new tests pass
- [ ] `openclaw-quantctl doctor` still reports testnet connectivity ok
- [ ] `openclaw-quantctl analyze BTCUSDT --market futures --interval 1h --render-chart` still works
- [ ] `submit-analysis ... --run-now` still completes
- [ ] Existing paper journal behavior still only journals, never executes live orders
- [ ] Live trading remains disabled by default

---

## Implementation order recommendation

1. Signal layer skeleton
2. Multi-timeframe trend filter
3. Market structure detector
4. Volatility regime classifier
5. Derivatives adapter
6. Event-risk veto
7. Strategy router
8. Wire into `analyze`
9. Strategy config example
10. Docs and workflow refresh

---

## Notes

- Keep every new detector deterministic and cheap to run.
- Prefer simple rule engines first; do not jump to ML models unless the rule-based version is validated.
- Keep all live-trading paths disabled unless explicitly turned on in a separate review.
- If any layer reduces clarity or creates noisy output, remove it rather than stacking more indicators on top.
