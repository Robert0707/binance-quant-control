from __future__ import annotations

import json

import pandas as pd

from binance_quant_control.feature_label_gate import (
    FeatureLabelGateConfig,
    build_feature_label_entry_gate,
    load_feature_label_gate_index,
)


def _row(timestamp: str, label_r: float, *, plus_di: float = 30.0, minus_di: float = 10.0) -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "interval": "4h",
        "timestamp": timestamp,
        "adx": 24.0,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "obv_zscore_20": 0.1,
        "volume_zscore_20": 0.2,
        "ml_vol_ratio_20_80": 1.4,
        "ml_liquidity_pressure": 0.2,
        "ml_turbulence_20": 1.7,
        "ml_payoff_potential_long": 0.4,
        "ml_payoff_potential_short": 0.4,
        "ml_volatility_regime": "high",
        "ml_trend_regime": "positive",
        "ml_liquidity_regime": "mid",
        "ml_turbulence_regime": "high",
        "label_long_outcome": "take_profit" if label_r > 0 else "stop_loss",
        "label_long_r": label_r,
        "label_long_bars": 1,
        "label_short_outcome": "stop_loss",
        "label_short_r": -1.0,
        "label_short_bars": 1,
    }


def test_feature_label_gate_uses_only_observable_past_labels(tmp_path) -> None:
    dataset = tmp_path / "feature-dataset.jsonl"
    rows = [
        _row("2026-01-01 00:00:00+00:00", 1.5),
        _row("2026-01-01 04:00:00+00:00", -1.0),
        _row("2026-01-01 08:00:00+00:00", 1.5),
        _row("2026-01-01 12:00:00+00:00", -1.0),
        _row("2026-01-03 00:00:00+00:00", -1.0),
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    config = FeatureLabelGateConfig.from_mapping(
        {
            "enabled": True,
            "dataset_path": str(dataset),
            "min_samples": 4,
            "min_expectancy_r": 0.2,
            "min_profit_factor": 1.2,
            "allow_if_insufficient_samples": False,
        }
    )
    index = load_feature_label_gate_index(config)
    assert index is not None

    gate, metadata = build_feature_label_entry_gate(
        symbol="BTCUSDT",
        interval="4h",
        config=config,
        index=index,
    )

    assert metadata["example_count"] == 10
    assert gate is not None
    previous = pd.Series(
        {
            "adx": 24.0,
            "plus_di": 30.0,
            "minus_di": 10.0,
            "obv_zscore_20": 0.1,
            "volume_zscore_20": 0.2,
        },
        name=pd.Timestamp("2026-01-02 00:00:00+00:00"),
    )

    allowed, reason = gate(previous, previous, {"recommended_action": "BUY"}, 10)

    assert allowed is True
    assert reason == ""

    later = previous.copy()
    later.name = pd.Timestamp("2026-01-04 00:00:00+00:00")
    allowed, reason = gate(later, later, {"recommended_action": "BUY"}, 20)

    assert allowed is False
    assert reason.startswith("feature-label-gate-veto:")


def test_feature_label_gate_uses_ml_meta_signal_buckets(tmp_path) -> None:
    dataset = tmp_path / "feature-dataset.jsonl"
    rows = [
        _row("2026-01-01 00:00:00+00:00", -1.0),
        _row("2026-01-01 04:00:00+00:00", -1.0),
        _row("2026-01-01 08:00:00+00:00", -1.0),
        _row("2026-01-01 12:00:00+00:00", -1.0),
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    config = FeatureLabelGateConfig.from_mapping(
        {
            "enabled": True,
            "dataset_path": str(dataset),
            "min_samples": 4,
            "min_expectancy_r": 0.0,
            "min_profit_factor": 1.0,
            "allow_if_insufficient_samples": False,
            "use_ml_meta_features": True,
        }
    )
    index = load_feature_label_gate_index(config)
    assert index is not None
    gate, metadata = build_feature_label_entry_gate(
        symbol="BTCUSDT",
        interval="4h",
        config=config,
        index=index,
    )
    assert gate is not None
    assert "ml-regime-and-payoff-buckets-before-textual-context" in metadata["applied_principles"]

    previous = pd.Series(
        {
            "adx": 24.0,
            "plus_di": 30.0,
            "minus_di": 10.0,
            "obv_zscore_20": 0.1,
            "volume_zscore_20": 0.2,
            "ml_vol_ratio_20_80": 1.4,
            "ml_liquidity_pressure": 0.2,
            "ml_turbulence_20": 1.7,
            "ml_payoff_potential_long": 0.4,
        },
        name=pd.Timestamp("2026-01-02 00:00:00+00:00"),
    )

    allowed, reason = gate(previous, previous, {"recommended_action": "BUY"}, 10)

    assert allowed is False
    assert reason.startswith("feature-label-gate-veto:label-expectancy-below-floor")
