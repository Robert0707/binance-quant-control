from __future__ import annotations

from dataclasses import replace

import pandas as pd

from binance_quant_control.analysis import enrich_indicators
from binance_quant_control.asset_routing import resolve_symbol_route
from binance_quant_control.backtest import (
    audit_backtest_robustness,
    run_backtest,
    simulate_backtest,
)
from binance_quant_control.strategy import load_strategy_config


def test_simulate_backtest_produces_trades_on_trending_frame():
    strategy = load_strategy_config("config/strategy-live-pilot.yaml")
    rows = []
    price = 1.0
    for idx in range(320):
        open_price = price
        close_price = price + 0.01
        high_price = close_price + 0.01
        low_price = open_price - 0.005
        rows.append(
            {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": 1000 + idx,
            }
        )
        price += 0.01

    df = pd.DataFrame(rows, index=pd.date_range("2026-01-01", periods=len(rows), freq="4h", tz="UTC"))
    enriched = enrich_indicators(df, "4h", strategy=strategy)
    result = simulate_backtest(enriched, "futures", strategy)

    assert result["trade_count"] > 0
    assert result["win_rate"] >= 0.0
    assert "total_return_pct" in result


def test_simulate_backtest_entry_filter_can_veto_entries():
    strategy = load_strategy_config("config/strategy-live-pilot.yaml")
    rows = []
    price = 1.0
    for idx in range(320):
        open_price = price
        close_price = price + 0.01
        high_price = close_price + 0.01
        low_price = open_price - 0.005
        rows.append(
            {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": 1000 + idx,
            }
        )
        price += 0.01

    df = pd.DataFrame(rows, index=pd.date_range("2026-01-01", periods=len(rows), freq="4h", tz="UTC"))
    enriched = enrich_indicators(df, "4h", strategy=strategy)
    result = simulate_backtest(
        enriched,
        "futures",
        strategy,
        entry_filter=lambda *_args: (False, "pytest-risk-veto"),
    )

    assert result["trade_count"] == 0
    assert result["entry_veto_count"] > 0
    assert result["entry_veto_reasons"] == {"pytest-risk-veto": result["entry_veto_count"]}


def test_run_backtest_reuses_enriched_frame_cache(monkeypatch, tmp_path):
    strategy = load_strategy_config("config/strategy-live-pilot.yaml")
    raw_klines = []
    open_time = 1_767_225_600_000
    price = 1.0
    for idx in range(260):
        raw_klines.append(
            [
                open_time + idx * 3_600_000,
                str(price),
                str(price + 0.02),
                str(price - 0.01),
                str(price + 0.01),
                str(1000 + idx),
                open_time + (idx + 1) * 3_600_000 - 1,
                str((1000 + idx) * price),
                100,
                str((500 + idx) * price),
                str((500 + idx) * price),
                "0",
            ]
        )
        price += 0.01
    calls = {"klines": 0}

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def klines(self, *_args, **_kwargs):
            calls["klines"] += 1
            return raw_klines

    monkeypatch.setattr("binance_quant_control.backtest.BinanceClient", FakeClient)
    cache = {}

    first = run_backtest(
        None,
        strategy=strategy,
        symbol="BTCUSDT",
        market="futures",
        interval="1h",
        limit=260,
        output_dir=tmp_path / "first",
        strategy_family="trend_continuation",
        frame_cache=cache,
    )
    second = run_backtest(
        None,
        strategy=strategy,
        symbol="BTCUSDT",
        market="futures",
        interval="1h",
        limit=260,
        output_dir=tmp_path / "second",
        strategy_family="breakout",
        frame_cache=cache,
    )

    assert calls["klines"] == 1
    assert first["symbol"] == second["symbol"] == "BTCUSDT"
    assert first["feature_manifest"]["manifest_hash"]


def test_simulate_backtest_reuses_market_context_cache(monkeypatch):
    strategy = load_strategy_config("config/strategy-live-pilot.yaml")
    rows = []
    price = 1.0
    for idx in range(260):
        rows.append(
            {
                "open": price,
                "high": price + 0.02,
                "low": price - 0.01,
                "close": price + 0.01,
                "volume": 1000 + idx,
            }
        )
        price += 0.01
    df = pd.DataFrame(rows, index=pd.date_range("2026-01-01", periods=len(rows), freq="4h", tz="UTC"))
    enriched = enrich_indicators(df, "4h", strategy=strategy)
    enriched["supertrend_direction"] = 1
    enriched["trend_magic_direction"] = 1
    enriched["follow_line_direction"] = 1
    enriched["jumbo_power"] = 55.0
    enriched["jumbo_power_ma"] = 35.0
    enriched["adx"] = 25.0
    calls = {"context": 0}

    def fake_context(*_args, **_kwargs):
        calls["context"] += 1
        return {}

    def family_once(previous, *_args, **_kwargs):
        if previous.name == enriched.index[220]:
            return {
                "entry_ready": True,
                "recommended_action": "BUY",
                "score": 90,
                "convergence": 0.9,
                "strategy_family": "trend_continuation",
                "selected_strategy_family": {"family": "trend_continuation"},
            }
        return {"entry_ready": False, "recommended_action": "HOLD", "score": 50, "convergence": 0.0}

    monkeypatch.setattr("binance_quant_control.backtest._backtest_market_context", fake_context)
    monkeypatch.setattr("binance_quant_control.backtest.strategy_family_trade_decision", family_once)
    cache: dict[int, dict[str, object]] = {}

    simulate_backtest(enriched, "futures", strategy, strategy_family="trend_continuation", market_context_cache=cache)
    first_call_count = calls["context"]
    simulate_backtest(enriched, "futures", strategy, strategy_family="breakout", market_context_cache=cache)

    assert first_call_count > 0
    assert calls["context"] == first_call_count


def test_simulate_backtest_does_not_replace_open_position(monkeypatch):
    strategy = load_strategy_config("config/strategy-core-high-win-research.yaml")
    rows = []
    price = 100.0
    for idx in range(240):
        rows.append(
            {
                "open": price,
                "high": price + 0.05,
                "low": price - 0.05,
                "close": price + 0.01,
                "volume": 1000 + idx,
            }
        )
        price += 0.01
    rows.append({"open": 102.4, "high": 102.45, "low": 102.35, "close": 102.4, "volume": 1500})
    rows.append({"open": 102.4, "high": 102.45, "low": 102.35, "close": 102.4, "volume": 1500})
    rows.append({"open": 102.4, "high": 104.2, "low": 102.35, "close": 104.0, "volume": 1500})
    df = pd.DataFrame(rows, index=pd.date_range("2026-01-01", periods=len(rows), freq="4h", tz="UTC"))
    enriched = enrich_indicators(df, "4h", strategy=strategy)
    enriched["atr_14"] = 1.0
    enriched["supertrend_direction"] = 1
    enriched["trend_magic_direction"] = 1
    enriched["follow_line_direction"] = 1
    enriched["chandelier_direction"] = 1
    enriched["ichimoku_direction"] = 1
    enriched["psar_direction"] = 1
    enriched["qqe_direction"] = 1
    enriched["jumbo_power"] = 50.0
    enriched["jumbo_power_ma"] = 20.0
    enriched["adx"] = 25.0
    enriched["plus_di"] = 30.0
    enriched["minus_di"] = 10.0
    enriched["squeeze_on"] = False
    enriched["squeeze_released"] = True
    seen_previous: set[pd.Timestamp] = set()

    def score_every_bar(previous, *_args, **_kwargs):
        seen_previous.add(previous.name)
        if previous.name in {enriched.index[239], enriched.index[240]}:
            return {
                "entry_ready": True,
                "recommended_action": "BUY",
                "score": 90,
                "convergence": 0.9,
                "strategy_family": "trend_continuation",
                "selected_strategy_family": {"family": "trend_continuation"},
            }
        return {"entry_ready": False, "recommended_action": "HOLD", "score": 50, "convergence": 0.0}

    monkeypatch.setattr("binance_quant_control.backtest.score_bias", score_every_bar)

    result = simulate_backtest(enriched, "futures", strategy)

    assert result["trade_count"] == 1
    assert result["trades"][0]["entry_time"] == str(enriched.index[240])
    assert result["trades"][0]["exit_reason"] in {"partial_tp_then_stop", "partial_tp_then_end", "staged_take_profit"}
    assert enriched.index[240] not in seen_previous


def test_simulate_backtest_exits_stale_position_on_time_limit(monkeypatch):
    strategy = load_strategy_config("config/strategy-market-bot-payoff-research.yaml")
    strategy = replace(
        strategy,
        risk=replace(
            strategy.risk,
            time_limit_bars=3,
            take_profit_r_multiples=(5.0,),
            trailing_stop_enabled=False,
        ),
    )
    rows = []
    price = 100.0
    for idx in range(230):
        rows.append(
            {
                "open": price,
                "high": price + 0.05,
                "low": price - 0.05,
                "close": price + 0.01,
                "volume": 1000 + idx,
            }
        )
        price += 0.01
    rows.extend(
        [
            {"open": 102.4, "high": 102.5, "low": 102.3, "close": 102.45, "volume": 1500},
            {"open": 102.45, "high": 102.55, "low": 102.35, "close": 102.46, "volume": 1500},
            {"open": 102.46, "high": 102.56, "low": 102.36, "close": 102.47, "volume": 1500},
            {"open": 102.47, "high": 102.57, "low": 102.37, "close": 102.48, "volume": 1500},
            {"open": 102.48, "high": 102.58, "low": 102.38, "close": 102.49, "volume": 1500},
        ]
    )
    df = pd.DataFrame(rows, index=pd.date_range("2026-01-01", periods=len(rows), freq="4h", tz="UTC"))
    enriched = enrich_indicators(df, "4h", strategy=strategy)
    enriched["atr_14"] = 1.0
    enriched["supertrend_direction"] = 1
    enriched["trend_magic_direction"] = 1
    enriched["follow_line_direction"] = 1
    enriched["chandelier_direction"] = 1
    enriched["ichimoku_direction"] = 1
    enriched["psar_direction"] = 1
    enriched["qqe_direction"] = 1
    enriched["jumbo_power"] = 50.0
    enriched["jumbo_power_ma"] = 20.0
    enriched["adx"] = 25.0
    enriched["plus_di"] = 30.0
    enriched["minus_di"] = 10.0
    enriched["squeeze_on"] = False
    enriched["squeeze_released"] = True

    def score_once(previous, *_args, **_kwargs):
        if previous.name == enriched.index[221]:
            return {
                "entry_ready": True,
                "recommended_action": "BUY",
                "score": 90,
                "convergence": 0.9,
                "strategy_family": "trend_continuation",
                "selected_strategy_family": {"family": "trend_continuation"},
                "signal_groups": [
                    {"name": "momentum_power", "bias": "long"},
                    {"name": "volume_flow", "bias": "long"},
                ],
            }
        return {"entry_ready": False, "recommended_action": "HOLD", "score": 50, "convergence": 0.0}

    monkeypatch.setattr("binance_quant_control.backtest.score_bias", score_once)

    result = simulate_backtest(enriched, "futures", strategy)

    assert result["trade_count"] == 1
    assert result["trades"][0]["exit_reason"] == "time_limit"


def test_simulate_backtest_quality_filter_blocks_unaligned_entries(monkeypatch):
    strategy = load_strategy_config("config/strategy-live-pilot.yaml")
    rows = []
    price = 1.0
    for idx in range(320):
        rows.append(
            {
                "open": price,
                "high": price + 0.02,
                "low": price - 0.01,
                "close": price + 0.01,
                "volume": 1000 + idx,
            }
        )
        price += 0.01
    df = pd.DataFrame(rows, index=pd.date_range("2026-01-01", periods=len(rows), freq="4h", tz="UTC"))
    enriched = enrich_indicators(df, "4h", strategy=strategy)
    enriched["supertrend_direction"] = -1
    enriched["trend_magic_direction"] = -1
    enriched["follow_line_direction"] = -1

    def always_long(*_args, **_kwargs):
        return {
            "entry_ready": True,
            "recommended_action": "BUY",
            "score": 90,
            "convergence": 1.0,
        }

    monkeypatch.setattr("binance_quant_control.backtest.score_bias", always_long)

    result = simulate_backtest(enriched, "futures", strategy)

    assert result["trade_count"] == 0
    assert result["entry_veto_reasons"]["trend-filter-not-long"] > 0


def test_simulate_backtest_quality_filter_blocks_unreleased_squeeze(monkeypatch):
    strategy = load_strategy_config("config/strategy-live-pilot.yaml")
    rows = []
    price = 1.0
    for idx in range(320):
        rows.append(
            {
                "open": price,
                "high": price + 0.02,
                "low": price - 0.01,
                "close": price + 0.01,
                "volume": 1000 + idx,
            }
        )
        price += 0.01
    df = pd.DataFrame(rows, index=pd.date_range("2026-01-01", periods=len(rows), freq="4h", tz="UTC"))
    enriched = enrich_indicators(df, "4h", strategy=strategy)
    enriched["squeeze_on"] = True
    enriched["squeeze_released"] = False
    enriched["supertrend_direction"] = 1
    enriched["trend_magic_direction"] = 1
    enriched["follow_line_direction"] = 1

    def always_long(*_args, **_kwargs):
        return {
            "entry_ready": True,
            "recommended_action": "BUY",
            "score": 90,
            "convergence": 1.0,
            "strategy_family": "trend_continuation",
        }

    monkeypatch.setattr("binance_quant_control.backtest.score_bias", always_long)

    result = simulate_backtest(enriched, "futures", strategy)

    assert result["trade_count"] == 0
    assert result["entry_veto_reasons"]["squeeze-not-released-for-long"] > 0


def test_simulate_backtest_can_use_one_strategy_family(monkeypatch):
    strategy = load_strategy_config("config/strategy-live-pilot.yaml")
    rows = []
    price = 1.0
    for idx in range(260):
        rows.append(
            {
                "open": price,
                "high": price + 0.02,
                "low": price - 0.01,
                "close": price + 0.01,
                "volume": 1000 + idx,
            }
        )
        price += 0.002
    df = pd.DataFrame(rows, index=pd.date_range("2026-01-01", periods=len(rows), freq="4h", tz="UTC"))
    enriched = enrich_indicators(df, "4h", strategy=strategy)
    enriched["supertrend_direction"] = 1
    enriched["trend_magic_direction"] = 1
    enriched["follow_line_direction"] = 1
    enriched["jumbo_power"] = 55.0
    enriched["jumbo_power_ma"] = 35.0
    enriched["adx"] = 25.0

    calls: list[str] = []

    def family_only(previous, *, market, family, **_kwargs):
        calls.append(family)
        return {
            "entry_ready": True,
            "recommended_action": "BUY",
            "score": 90,
            "convergence": 0.9,
            "selected_strategy_family": {"family": family},
        }

    monkeypatch.setattr("binance_quant_control.backtest.strategy_family_trade_decision", family_only)
    def confirming_score_model(*_args, **_kwargs):
        return {
            "entry_ready": True,
            "recommended_action": "BUY",
            "score": 82,
            "convergence": 0.8,
            "signal_groups": [
                {"name": "volume_flow", "bias": "long"},
                {"name": "momentum_power", "bias": "long"},
            ],
        }

    monkeypatch.setattr("binance_quant_control.backtest.score_bias", confirming_score_model)

    result = simulate_backtest(enriched, "futures", strategy, strategy_family="breakout")

    assert result["trade_count"] > 0
    assert calls
    assert set(calls) == {"breakout"}


def test_strategy_family_backtest_requires_score_model_confirmation(monkeypatch):
    strategy = load_strategy_config("config/strategy-live-pilot.yaml")
    rows = []
    price = 1.0
    for idx in range(260):
        rows.append(
            {
                "open": price,
                "high": price + 0.02,
                "low": price - 0.01,
                "close": price + 0.01,
                "volume": 1000 + idx,
            }
        )
        price += 0.002
    df = pd.DataFrame(rows, index=pd.date_range("2026-01-01", periods=len(rows), freq="4h", tz="UTC"))
    enriched = enrich_indicators(df, "4h", strategy=strategy)
    enriched["supertrend_direction"] = 1
    enriched["trend_magic_direction"] = 1
    enriched["follow_line_direction"] = 1
    enriched["jumbo_power"] = 55.0
    enriched["jumbo_power_ma"] = 35.0
    enriched["adx"] = 25.0
    enriched["squeeze_released"] = True
    enriched["squeeze_momentum"] = 1.0

    def breakout_long(*_args, **_kwargs):
        return {
            "entry_ready": True,
            "recommended_action": "BUY",
            "score": 90,
            "convergence": 0.9,
            "strategy_family": "breakout",
            "selected_strategy_family": {"family": "breakout"},
        }

    def rejecting_score_model(*_args, **_kwargs):
        return {
            "entry_ready": False,
            "recommended_action": "HOLD",
            "score": 55,
            "convergence": 0.5,
            "signal_groups": [
                {"name": "volume_flow", "bias": "neutral"},
                {"name": "momentum_power", "bias": "long"},
            ],
        }

    monkeypatch.setattr("binance_quant_control.backtest.strategy_family_trade_decision", breakout_long)
    monkeypatch.setattr("binance_quant_control.backtest.score_bias", rejecting_score_model)

    result = simulate_backtest(enriched, "futures", strategy, strategy_family="breakout")

    assert result["trade_count"] == 0
    assert result["entry_veto_reasons"]["score-model-not-long"] > 0


def test_breakout_family_requires_volume_flow_confirmation(monkeypatch):
    strategy = load_strategy_config("config/strategy-live-pilot.yaml")
    rows = []
    price = 1.0
    for idx in range(260):
        rows.append(
            {
                "open": price,
                "high": price + 0.02,
                "low": price - 0.01,
                "close": price + 0.01,
                "volume": 1000 + idx,
            }
        )
        price += 0.002
    df = pd.DataFrame(rows, index=pd.date_range("2026-01-01", periods=len(rows), freq="4h", tz="UTC"))
    enriched = enrich_indicators(df, "4h", strategy=strategy)
    enriched["supertrend_direction"] = 1
    enriched["trend_magic_direction"] = 1
    enriched["follow_line_direction"] = 1
    enriched["jumbo_power"] = 55.0
    enriched["jumbo_power_ma"] = 35.0
    enriched["adx"] = 25.0
    enriched["squeeze_released"] = True
    enriched["squeeze_momentum"] = 1.0

    def breakout_long(*_args, **_kwargs):
        return {
            "entry_ready": True,
            "recommended_action": "BUY",
            "score": 90,
            "convergence": 0.9,
            "strategy_family": "breakout",
            "selected_strategy_family": {"family": "breakout"},
        }

    def no_volume_confirmation(*_args, **_kwargs):
        return {
            "entry_ready": True,
            "recommended_action": "BUY",
            "score": 82,
            "convergence": 0.8,
            "signal_groups": [
                {"name": "volume_flow", "bias": "neutral"},
                {"name": "momentum_power", "bias": "long"},
            ],
        }

    monkeypatch.setattr("binance_quant_control.backtest.strategy_family_trade_decision", breakout_long)
    monkeypatch.setattr("binance_quant_control.backtest.score_bias", no_volume_confirmation)

    result = simulate_backtest(enriched, "futures", strategy, strategy_family="breakout")

    assert result["trade_count"] == 0
    assert result["entry_veto_reasons"]["breakout-volume-flow-not-long"] > 0


def test_mean_reversion_family_is_not_blocked_by_trend_entry_gate(monkeypatch):
    strategy = load_strategy_config("config/strategy-live-pilot.yaml")
    rows = []
    price = 1.0
    for idx in range(260):
        rows.append(
            {
                "open": price,
                "high": price + 0.02,
                "low": price - 0.01,
                "close": price + 0.001,
                "volume": 1000 + idx,
            }
        )
        price += 0.0005
    df = pd.DataFrame(rows, index=pd.date_range("2026-01-01", periods=len(rows), freq="4h", tz="UTC"))
    enriched = enrich_indicators(df, "4h", strategy=strategy)
    enriched["supertrend_direction"] = -1
    enriched["trend_magic_direction"] = -1
    enriched["follow_line_direction"] = -1
    enriched["chandelier_direction"] = -1
    enriched["ichimoku_direction"] = -1
    enriched["psar_direction"] = -1
    enriched["qqe_direction"] = -1
    enriched["jumbo_power"] = -40.0
    enriched["jumbo_power_ma"] = 10.0
    enriched["adx"] = 12.0
    enriched["ema_fast"] = 1.2
    enriched["ema_slow"] = 1.0
    enriched["macd_hist"] = 0.1
    enriched["plus_di"] = 30.0
    enriched["minus_di"] = 10.0
    enriched["bb_percent_b"] = 0.08
    enriched["mfi_14"] = 30.0

    def mean_reversion_long(*_args, **_kwargs):
        return {
            "entry_ready": True,
            "recommended_action": "BUY",
            "score": 88,
            "convergence": 0.95,
            "strategy_family": "mean_reversion",
            "selected_strategy_family": {"family": "mean_reversion"},
        }

    monkeypatch.setattr("binance_quant_control.backtest.strategy_family_trade_decision", mean_reversion_long)

    result = simulate_backtest(enriched, "futures", strategy, strategy_family="mean_reversion")

    assert result["entry_veto_reasons"].get("trend-filter-not-long", 0) == 0
    assert result["entry_veto_reasons"].get("tradingview-stack-opposes-long", 0) == 0
    assert result["trade_count"] > 0


def test_mean_reversion_family_blocks_strong_opposing_trend_stack(monkeypatch):
    strategy = load_strategy_config("config/strategy-live-pilot.yaml")
    rows = []
    price = 1.0
    for idx in range(260):
        rows.append(
            {
                "open": price,
                "high": price + 0.02,
                "low": price - 0.01,
                "close": price + 0.003,
                "volume": 1000 + idx,
            }
        )
        price += 0.001
    df = pd.DataFrame(rows, index=pd.date_range("2026-01-01", periods=len(rows), freq="4h", tz="UTC"))
    enriched = enrich_indicators(df, "4h", strategy=strategy)
    enriched["supertrend_direction"] = -1
    enriched["trend_magic_direction"] = -1
    enriched["follow_line_direction"] = -1
    enriched["chandelier_direction"] = -1
    enriched["ichimoku_direction"] = -1
    enriched["psar_direction"] = -1
    enriched["qqe_direction"] = -1
    enriched["jumbo_power"] = -40.0
    enriched["jumbo_power_ma"] = 10.0
    enriched["adx"] = 28.0
    enriched["ema_fast"] = 1.2
    enriched["ema_slow"] = 1.0
    enriched["macd_hist"] = 0.1
    enriched["plus_di"] = 30.0
    enriched["minus_di"] = 10.0
    enriched["bb_percent_b"] = 0.08
    enriched["mfi_14"] = 30.0

    def mean_reversion_long(*_args, **_kwargs):
        return {
            "entry_ready": True,
            "recommended_action": "BUY",
            "score": 88,
            "convergence": 0.95,
            "strategy_family": "mean_reversion",
            "selected_strategy_family": {"family": "mean_reversion"},
        }

    monkeypatch.setattr("binance_quant_control.backtest.strategy_family_trade_decision", mean_reversion_long)

    result = simulate_backtest(enriched, "futures", strategy, strategy_family="mean_reversion")

    assert result["trade_count"] == 0
    assert result["entry_veto_reasons"]["mean-reversion-trend-stack-opposes-long"] > 0


def test_symbol_mean_reversion_filters_known_regime_failures(monkeypatch):
    strategy = load_strategy_config("config/strategy-core-high-win-research.yaml")
    rows = []
    price = 1.0
    for idx in range(260):
        rows.append(
            {
                "open": price,
                "high": price + 0.02,
                "low": price - 0.01,
                "close": price + 0.001,
                "volume": 1000 + idx,
            }
        )
        price += 0.0005
    df = pd.DataFrame(rows, index=pd.date_range("2026-01-01", periods=len(rows), freq="4h", tz="UTC"))
    enriched = enrich_indicators(df, "4h", strategy=strategy)
    enriched["adx"] = 15.0
    enriched["rsi_14"] = 32.0
    enriched["bb_percent_b"] = 0.05
    enriched["mfi_14"] = 30.0
    enriched["bb_bandwidth"] = 0.045
    enriched["stoch_rsi_k"] = 10.0

    def mean_reversion_long(*_args, **_kwargs):
        return {
            "entry_ready": True,
            "recommended_action": "BUY",
            "score": 88,
            "convergence": 0.95,
            "strategy_family": "mean_reversion",
            "selected_strategy_family": {"family": "mean_reversion"},
        }

    monkeypatch.setattr("binance_quant_control.backtest.strategy_family_trade_decision", mean_reversion_long)

    result = simulate_backtest(enriched, "futures", strategy, symbol="XRPUSDT", interval="4h", strategy_family="mean_reversion")

    assert result["trade_count"] == 0
    assert result["entry_veto_reasons"]["xrp-mean-reversion-long-bandwidth-too-low"] > 0


def test_simulate_backtest_labels_partial_tp_then_stop(monkeypatch):
    strategy = load_strategy_config("config/strategy-live-pilot.yaml")
    rows = []
    price = 100.0
    for idx in range(230):
        rows.append(
            {
                "open": price,
                "high": price + 0.2,
                "low": price - 0.2,
                "close": price + 0.1,
                "volume": 1000 + idx,
            }
        )
        price += 0.1
    rows.append({"open": 123.0, "high": 123.6, "low": 122.8, "close": 123.5, "volume": 5000})
    rows.append({"open": 123.5, "high": 124.2, "low": 122.5, "close": 123.8, "volume": 4500})
    rows.append({"open": 123.8, "high": 123.9, "low": 121.0, "close": 121.5, "volume": 4200})
    rows.extend(
        {
            "open": 121.5,
            "high": 121.6,
            "low": 121.4,
            "close": 121.5,
            "volume": 1000,
        }
        for _ in range(5)
    )
    df = pd.DataFrame(rows, index=pd.date_range("2026-01-01", periods=len(rows), freq="4h", tz="UTC"))
    enriched = enrich_indicators(df, "4h", strategy=strategy)
    enriched["atr_14"] = 0.5

    def score_once(previous, *_args, **_kwargs):
        if previous.name == enriched.index[229]:
            return {
                "entry_ready": True,
                "recommended_action": "BUY",
                "score": 90,
                "convergence": 0.9,
            }
        return {"entry_ready": False, "recommended_action": "HOLD", "score": 50, "convergence": 0.0}

    monkeypatch.setattr("binance_quant_control.backtest.score_bias", score_once)

    result = simulate_backtest(enriched, "futures", strategy)

    assert result["trade_count"] == 1
    assert result["trades"][0]["exit_reason"] == "partial_tp_then_stop"
    assert result["trades"][0]["pnl_pct"] > -2.5


def test_simulate_backtest_moves_runner_stop_to_small_profit_after_second_target(monkeypatch):
    strategy = load_strategy_config("config/strategy-core-high-win-research.yaml")
    rows = []
    price = 100.0
    for idx in range(230):
        rows.append(
            {
                "open": price,
                "high": price + 0.2,
                "low": price - 0.2,
                "close": price + 0.05,
                "volume": 1000 + idx,
            }
        )
        price += 0.05
    rows.append({"open": 111.5, "high": 114.2, "low": 111.4, "close": 114.0, "volume": 5000})
    rows.append({"open": 114.0, "high": 114.1, "low": 112.05, "close": 112.2, "volume": 4500})
    rows.extend(
        {
            "open": 112.2,
            "high": 112.3,
            "low": 110.9,
            "close": 111.0,
            "volume": 1000,
        }
        for _ in range(5)
    )
    df = pd.DataFrame(rows, index=pd.date_range("2026-01-01", periods=len(rows), freq="4h", tz="UTC"))
    enriched = enrich_indicators(df, "4h", strategy=strategy)
    enriched["atr_14"] = 1.0
    enriched["bb_lower"] = enriched["close"] - 1.35
    enriched["bb_upper"] = enriched["close"] + 1.35
    enriched["keltner_lower"] = enriched["close"] - 1.35
    enriched["keltner_upper"] = enriched["close"] + 1.35

    def score_once(previous, *_args, **_kwargs):
        if previous.name == enriched.index[229]:
            return {
                "entry_ready": True,
                "recommended_action": "BUY",
                "score": 90,
                "convergence": 0.9,
                "strategy_family": "mean_reversion",
                "selected_strategy_family": {"family": "mean_reversion"},
            }
        return {"entry_ready": False, "recommended_action": "HOLD", "score": 50, "convergence": 0.0}

    monkeypatch.setattr("binance_quant_control.backtest.score_bias", score_once)

    result = simulate_backtest(enriched, "futures", strategy)

    assert result["trade_count"] == 1
    assert result["trades"][0]["exit_reason"] == "partial_tp_then_stop"
    assert result["trades"][0]["pnl_pct"] > 0.0


def test_robustness_audit_rejects_one_lucky_time_slice() -> None:
    route = resolve_symbol_route("BTCUSDT")
    trades = []
    for idx, pnl in enumerate([-1.0, -1.0, -1.0, -1.0, 6.0, 6.0, 6.0, 6.0], start=1):
        trades.append(
            {
                "entry_time": f"2026-01-{idx:02d}T00:00:00+00:00",
                "exit_time": f"2026-01-{idx:02d}T04:00:00+00:00",
                "pnl_pct": pnl,
                "pnl_r": pnl,
                "exit_reason": "take_profit" if pnl > 0 else "stop_loss",
            }
        )
    summary = {
        "trade_count": len(trades),
        "profit_factor": 6.0,
        "max_drawdown_pct": 4.0,
        "loss_streak": 0,
        "trades": trades,
    }

    audit = audit_backtest_robustness(
        summary,
        route.validation,
        folds=4,
        min_trades_per_fold=2,
    )

    assert audit["passed"] is False
    assert any("too-many-folds-below-1pf" in reason for reason in audit["reasons"])


def test_run_backtest_payload_includes_robustness(monkeypatch, tmp_path) -> None:
    strategy = load_strategy_config("config/strategy-live-pilot.yaml")
    rows = []
    price = 1.0
    for idx in range(320):
        rows.append(
            [
                idx * 60_000,
                str(price),
                str(price + 0.02),
                str(price - 0.005),
                str(price + 0.01),
                str(1000 + idx),
                (idx + 1) * 60_000,
                "0",
                0,
                "0",
                "0",
                "0",
            ]
        )
        price += 0.01

    class FakeClient:
        def __init__(self, settings):
            self.settings = settings

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def klines(self, symbol: str, interval: str, limit: int, market: str):
            return rows[-limit:]

    monkeypatch.setattr("binance_quant_control.backtest.BinanceClient", FakeClient)

    payload = run_backtest(
        object(),
        strategy=strategy,
        symbol="BTCUSDT",
        market="futures",
        interval="4h",
        limit=320,
        output_dir=tmp_path,
        strategy_family="trend_continuation",
    )

    assert "robustness" in payload
    assert payload["strategy_family"] == "trend_continuation"
    assert payload["robustness"]["applied_principles"]
    assert (tmp_path / "backtest.md").read_text(encoding="utf-8").count("Robustness gate") == 1
