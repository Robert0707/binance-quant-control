from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    family: str
    source: str
    lookback_bars: int
    live_safe: bool
    leakage_notes: str = ""
    offline_online_parity: str = "same_source"
    replay_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LabelSpec:
    name: str
    method: str
    horizon_bars: int
    take_profit_r: float
    stop_loss_r: float
    time_limit_bars: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("ema_trend", "trend", "indicators.py", 200, True, replay_key="symbol:interval:close"),
    FeatureSpec(
        "adx_directional_strength",
        "trend",
        "indicators.py",
        14,
        True,
        replay_key="symbol:interval:ohlc",
    ),
    FeatureSpec("rsi_14", "mean_reversion", "indicators.py", 14, True, replay_key="symbol:interval:close"),
    FeatureSpec(
        "fib_ote_zone",
        "trend_pullback",
        "analysis.py",
        120,
        True,
        replay_key="symbol:interval:ohlc",
    ),
    FeatureSpec(
        "liquidity_sweep_reclaim",
        "liquidity_reclaim",
        "volume_structure.py",
        20,
        True,
        replay_key="symbol:interval:ohlcv",
    ),
    FeatureSpec(
        "rolling_vwap_reclaim",
        "vwap_reclaim",
        "vwap_features.py",
        48,
        True,
        replay_key="symbol:interval:ohlcv",
    ),
    FeatureSpec(
        "obv_volume_zscore",
        "volume_confirmation",
        "volume_structure.py",
        20,
        True,
        replay_key="symbol:interval:ohlcv",
    ),
    FeatureSpec(
        "ml_regime_state",
        "ai_meta_filter",
        "ml_feature_engine.py",
        80,
        True,
        replay_key="symbol:interval:ohlcv",
    ),
    FeatureSpec(
        "ml_execution_quality",
        "ai_meta_filter",
        "ml_feature_engine.py",
        20,
        True,
        replay_key="symbol:interval:ohlcv",
    ),
    FeatureSpec(
        "ml_payoff_potential",
        "ai_exit_quality",
        "ml_feature_engine.py",
        20,
        True,
        replay_key="symbol:interval:ohlcv",
    ),
)

DEFAULT_LABELS: tuple[LabelSpec, ...] = (
    LabelSpec(
        name="triple_barrier_r_multiple",
        method="take_profit_stop_loss_time_limit",
        horizon_bars=72,
        take_profit_r=1.5,
        stop_loss_r=1.0,
        time_limit_bars=72,
    ),
)


def _manifest_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_feature_manifest() -> dict[str, Any]:
    features = [item.to_dict() for item in DEFAULT_FEATURES]
    labels = [item.to_dict() for item in DEFAULT_LABELS]
    feature_sets: dict[str, list[str]] = {}
    for item in DEFAULT_FEATURES:
        feature_sets.setdefault(item.family, []).append(item.name)
    core = {
        "mode": "feature_label_manifest",
        "live_safe": all(item.live_safe for item in DEFAULT_FEATURES),
        "feature_count": len(features),
        "label_count": len(labels),
        "features": features,
        "labels": labels,
        "feature_sets": dict(sorted(feature_sets.items())),
        "pipeline_contract": {
            "offline_online_parity_required": True,
            "point_in_time_required": True,
            "lookahead_allowed": False,
            "replay_unit": "symbol_interval_bar_timestamp",
        },
        "required_next_steps": [
            "persist feature manifest hash in every research report",
            "record label parameters beside each backtest result",
            "reject features with lookahead or live/offline mismatch",
        ],
    }
    return core | {"manifest_hash": _manifest_hash(core)}
