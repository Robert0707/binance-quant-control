from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PROJECT_ROOT


@dataclass(frozen=True, slots=True)
class FeatureLabelGateConfig:
    enabled: bool = False
    dataset_path: str = ""
    min_samples: int = 30
    max_history_rows: int = 1200
    min_expectancy_r: float = 0.02
    min_profit_factor: float = 1.05
    min_payoff_ratio: float = 0.75
    max_stop_loss_ratio: float = 70.0
    allow_if_insufficient_samples: bool = True
    use_ml_meta_features: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "FeatureLabelGateConfig":
        data = raw or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            dataset_path=str(data.get("dataset_path") or ""),
            min_samples=max(int(data.get("min_samples") or 30), 1),
            max_history_rows=max(int(data.get("max_history_rows") or 1200), 1),
            min_expectancy_r=float(data.get("min_expectancy_r") or 0.02),
            min_profit_factor=float(data.get("min_profit_factor") or 1.05),
            min_payoff_ratio=float(data.get("min_payoff_ratio") or 0.75),
            max_stop_loss_ratio=float(data.get("max_stop_loss_ratio") or 70.0),
            allow_if_insufficient_samples=bool(data.get("allow_if_insufficient_samples", True)),
            use_ml_meta_features=bool(data.get("use_ml_meta_features", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureLabelExample:
    symbol: str
    interval: str
    side: str
    timestamp: pd.Timestamp
    label_available_at: pd.Timestamp
    r_multiple: float
    outcome: str
    features: dict[str, float]


@dataclass(frozen=True, slots=True)
class FeatureLabelGateDecision:
    allowed: bool
    reason: str
    bucket_level: str
    sample_count: int
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_dataset_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _interval_delta(interval: str) -> timedelta:
    raw = str(interval).strip().lower()
    if not raw:
        return timedelta(hours=1)
    unit = raw[-1]
    try:
        amount = int(raw[:-1])
    except ValueError:
        amount = 1
    if unit == "m":
        return timedelta(minutes=amount)
    if unit == "h":
        return timedelta(hours=amount)
    if unit == "d":
        return timedelta(days=amount)
    if unit == "w":
        return timedelta(weeks=amount)
    return timedelta(hours=1)


def _pick(row: Any, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if hasattr(row, "get"):
            value = row.get(key)
        else:
            value = None
        if value not in (None, ""):
            return _float(value, default)
    return default


def _feature_snapshot(row: Any, side: str) -> dict[str, float]:
    plus_di = _pick(row, "plus_di")
    minus_di = _pick(row, "minus_di")
    signed_di = plus_di - minus_di if side == "BUY" else minus_di - plus_di
    if side == "BUY":
        fib_zone = _bool(row.get("fib_ote_long_zone", row.get("fib_ote_long", False))) or _bool(
            row.get("fib_pullback_long_zone", False)
        )
        liquidity_reclaim = _bool(row.get("liquidity_reclaim_long_20", row.get("sweep_low_reclaim_long", False)))
        vwap_reclaim = _bool(row.get("vwap_reclaim_long_48", row.get("vwap_reclaim_long", False))) or _bool(
            row.get("vwap_mid_reclaim_long_48", False)
        )
    else:
        fib_zone = _bool(row.get("fib_ote_short_zone", row.get("fib_ote_short", False))) or _bool(
            row.get("fib_pullback_short_zone", False)
        )
        liquidity_reclaim = _bool(row.get("liquidity_reclaim_short_20", row.get("sweep_high_reclaim_short", False)))
        vwap_reclaim = _bool(row.get("vwap_reclaim_short_48", row.get("vwap_reclaim_short", False))) or _bool(
            row.get("vwap_mid_reclaim_short_48", False)
        )
    payoff_potential = _pick(
        row,
        "ml_payoff_potential_long" if side == "BUY" else "ml_payoff_potential_short",
    )
    regime_map = {
        "low": -1.0,
        "asia": -1.0,
        "neutral": 0.0,
        "mid": 0.0,
        "europe": 0.0,
        "positive": 1.0,
        "negative": -1.0,
        "high": 1.0,
        "us": 1.0,
    }

    def category(name: str) -> float:
        return regime_map.get(str(row.get(name) or "").strip().lower(), 0.0)

    return {
        "adx": _pick(row, "adx"),
        "signed_di": signed_di,
        "obv_zscore": _pick(row, "obv_zscore_20"),
        "volume_zscore": _pick(row, "volume_zscore_20"),
        "fib_zone": 1.0 if fib_zone else 0.0,
        "liquidity_reclaim": 1.0 if liquidity_reclaim else 0.0,
        "vwap_reclaim": 1.0 if vwap_reclaim else 0.0,
        "ml_vol_ratio": _pick(row, "ml_vol_ratio_20_80", default=1.0),
        "ml_trend_slope": _pick(row, "ml_trend_slope_80"),
        "ml_liquidity_pressure": _pick(row, "ml_liquidity_pressure"),
        "ml_turbulence": _pick(row, "ml_turbulence_20"),
        "ml_payoff_potential": payoff_potential,
        "ml_volatility_regime": category("ml_volatility_regime"),
        "ml_trend_regime": category("ml_trend_regime"),
        "ml_liquidity_regime": category("ml_liquidity_regime"),
        "ml_turbulence_regime": category("ml_turbulence_regime"),
    }


def _bucket(value: float, cuts: tuple[float, ...], names: tuple[str, ...]) -> str:
    for cut, name in zip(cuts, names, strict=False):
        if value < cut:
            return name
    return names[-1]


def _bucket_signature(features: dict[str, float], side: str, level: str) -> tuple[str, ...]:
    signed_di = _bucket(
        features["signed_di"],
        (0.0, 5.0, 12.0),
        ("opposed", "weak", "confirmed", "strong"),
    )
    obv = _bucket(
        features["obv_zscore"],
        (-1.5, 1.5),
        ("obv-washout", "obv-normal", "obv-crowded"),
    )
    volume = _bucket(
        features["volume_zscore"],
        (-0.5, 1.5),
        ("quiet", "normal-volume", "volume-spike"),
    )
    adx = _bucket(
        features["adx"],
        (18.0, 28.0),
        ("low-adx", "trend-adx", "high-adx"),
    )
    vol = _bucket(
        features.get("ml_vol_ratio", 1.0),
        (0.85, 1.25),
        ("low-vol", "mid-vol", "high-vol"),
    )
    liquidity = _bucket(
        features.get("ml_liquidity_pressure", 0.0),
        (-0.5, 1.0),
        ("thin", "normal-liquidity", "high-liquidity"),
    )
    turbulence = _bucket(
        features.get("ml_turbulence", 0.0),
        (0.8, 1.5),
        ("calm", "normal-turbulence", "high-turbulence"),
    )
    payoff = _bucket(
        features.get("ml_payoff_potential", 0.0),
        (0.75, 1.5, 3.0),
        ("weak-payoff", "fair-payoff", "good-payoff", "large-payoff"),
    )
    if level == "ml_meta_state":
        return (side, vol, liquidity, turbulence, payoff)
    if level == "ml_regime":
        return (side, vol, liquidity, turbulence)
    if level == "ml_payoff":
        return (side, payoff)
    if level == "directional_flow":
        return (side, signed_di, obv, volume)
    if level == "directional_obv":
        return (side, signed_di, obv)
    if level == "directional_strength":
        return (side, signed_di)
    if level == "side_regime":
        return (side, adx)
    return (side,)


def _label_metrics(examples: list[FeatureLabelExample]) -> dict[str, Any]:
    r_values = [item.r_multiple for item in examples]
    wins = [value for value in r_values if value > 0.0]
    losses = [value for value in r_values if value <= 0.0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = -sum(losses) / len(losses) if losses else 0.0
    stop_count = sum(1 for item in examples if item.outcome == "stop_loss" or item.r_multiple <= -1.0)
    return {
        "sample_count": len(examples),
        "win_rate": round((len(wins) / len(examples)) * 100.0, 2) if examples else 0.0,
        "expectancy_r": round(sum(r_values) / len(r_values), 4) if r_values else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0.0 else 9999.0,
        "payoff_ratio": round(avg_win / avg_loss, 4) if avg_loss > 0.0 else 9999.0,
        "stop_loss_ratio": round((stop_count / len(examples)) * 100.0, 2) if examples else 0.0,
    }


class FeatureLabelGateIndex:
    def __init__(self, examples: list[FeatureLabelExample]) -> None:
        grouped: dict[tuple[str, str], list[FeatureLabelExample]] = {}
        for example in examples:
            grouped.setdefault((example.symbol, example.interval), []).append(example)
        self._grouped = {
            key: sorted(items, key=lambda item: item.label_available_at)
            for key, items in grouped.items()
        }

    @property
    def example_count(self) -> int:
        return sum(len(items) for items in self._grouped.values())

    @property
    def symbol_intervals(self) -> list[str]:
        return [f"{symbol}:{interval}" for symbol, interval in sorted(self._grouped)]

    def evaluate(
        self,
        *,
        symbol: str,
        interval: str,
        side: str,
        row: pd.Series,
        timestamp: pd.Timestamp,
        config: FeatureLabelGateConfig,
    ) -> FeatureLabelGateDecision:
        features = _feature_snapshot(row, side)
        candidates = [
            item
            for item in self._grouped.get((symbol.upper(), interval), [])
            if item.side == side and item.label_available_at < timestamp
        ]
        if config.max_history_rows > 0:
            candidates = candidates[-config.max_history_rows :]
        levels = (
            "ml_meta_state",
            "ml_regime",
            "ml_payoff",
            "directional_flow",
            "directional_obv",
            "directional_strength",
            "side_regime",
            "side",
        ) if config.use_ml_meta_features else (
            "directional_flow",
            "directional_obv",
            "directional_strength",
            "side_regime",
            "side",
        )
        for level in levels:
            signature = _bucket_signature(features, side, level)
            bucketed = [
                item
                for item in candidates
                if _bucket_signature(item.features, side, level) == signature
            ]
            if len(bucketed) < config.min_samples:
                continue
            metrics = _label_metrics(bucketed)
            blockers = []
            if float(metrics["expectancy_r"]) < config.min_expectancy_r:
                blockers.append("label-expectancy-below-floor")
            if float(metrics["profit_factor"]) < config.min_profit_factor:
                blockers.append("label-profit-factor-below-floor")
            if float(metrics["payoff_ratio"]) < config.min_payoff_ratio:
                blockers.append("label-payoff-below-floor")
            if float(metrics["stop_loss_ratio"]) > config.max_stop_loss_ratio:
                blockers.append("label-stop-loss-ratio-above-ceiling")
            return FeatureLabelGateDecision(
                allowed=not blockers,
                reason=";".join(blockers),
                bucket_level=level,
                sample_count=len(bucketed),
                metrics=metrics,
            )
        return FeatureLabelGateDecision(
            allowed=config.allow_if_insufficient_samples,
            reason="insufficient-feature-label-samples",
            bucket_level="none",
            sample_count=len(candidates),
            metrics={"sample_count": len(candidates)},
        )


def load_feature_label_gate_index(config: FeatureLabelGateConfig) -> FeatureLabelGateIndex | None:
    if not config.enabled:
        return None
    if not config.dataset_path:
        raise ValueError("feature_label_gate.dataset_path is required when the gate is enabled.")
    path = _resolve_dataset_path(config.dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature label gate dataset not found: {path}")
    examples: list[FeatureLabelExample] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            symbol = str(row.get("symbol") or "").upper()
            interval = str(row.get("interval") or "")
            if not symbol or not interval:
                continue
            timestamp = pd.Timestamp(row.get("timestamp"))
            delta = _interval_delta(interval)
            for label_side, prefix in (("BUY", "long"), ("SELL", "short")):
                outcome = str(row.get(f"label_{prefix}_outcome") or "")
                r_multiple = _float(row.get(f"label_{prefix}_r"), math.nan)
                if outcome in {"", "unavailable"} or math.isnan(r_multiple):
                    continue
                bars = max(int(_float(row.get(f"label_{prefix}_bars"), 0.0)), 1)
                if outcome == "time_limit":
                    outcome = "take_profit" if r_multiple > 0.0 else "stop_loss"
                examples.append(
                    FeatureLabelExample(
                        symbol=symbol,
                        interval=interval,
                        side=label_side,
                        timestamp=timestamp,
                        label_available_at=timestamp + (delta * bars),
                        r_multiple=r_multiple,
                        outcome=outcome,
                        features=_feature_snapshot(row, label_side),
                    )
                )
    return FeatureLabelGateIndex(examples)


def build_feature_label_entry_gate(
    *,
    symbol: str,
    interval: str,
    config: FeatureLabelGateConfig,
    index: FeatureLabelGateIndex | None,
) -> tuple[Any | None, dict[str, Any]]:
    metadata = {
        "enabled": config.enabled,
        "symbol": symbol.upper(),
        "interval": interval,
        "config": config.to_dict(),
        "dataset_loaded": index is not None,
        "example_count": index.example_count if index is not None else 0,
        "symbol_intervals": index.symbol_intervals if index is not None else [],
        "applied_principles": [
            "point-in-time-feature-label-veto",
            "triple-barrier-labels-as-entry-quality-proxy",
            "labels-only-count-after-event-is-observable",
            "ml-regime-and-payoff-buckets-before-textual-context",
        ],
    }
    if not config.enabled or index is None:
        return None, metadata

    def entry_gate(
        previous: pd.Series,
        current: pd.Series,
        analysis: dict[str, Any],
        idx: int,
    ) -> tuple[bool, str]:
        del current, idx
        action = str((analysis or {}).get("recommended_action") or "").upper()
        if action not in {"BUY", "SELL"}:
            return True, ""
        timestamp = pd.Timestamp(previous.name)
        decision = index.evaluate(
            symbol=symbol.upper(),
            interval=interval,
            side=action,
            row=previous,
            timestamp=timestamp,
            config=config,
        )
        if decision.allowed:
            return True, ""
        return False, f"feature-label-gate-veto:{decision.reason or decision.bucket_level}"

    return entry_gate, metadata
