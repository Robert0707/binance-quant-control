from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analysis import enrich_indicators, prepare_klines_frame
from .binance_api import BinanceClient
from .config import STATE_DIR, Settings, ensure_runtime_dirs
from .feature_registry import build_feature_manifest
from .historical_klines import fetch_recent_klines
from .ml_feature_engine import build_ai_ml_features
from .strategy import load_strategy_config

FEATURE_DATASET_DIR = STATE_DIR / "feature-datasets"


@dataclass(frozen=True, slots=True)
class FeatureDatasetSpec:
    symbols: list[str]
    intervals: list[str]
    limit: int
    strategy_config: str
    market: str = "futures"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _dataset_hash(rows: list[dict[str, Any]], manifest_hash: str) -> str:
    basis = {
        "manifest_hash": manifest_hash,
        "row_count": len(rows),
        "first": rows[0] if rows else {},
        "last": rows[-1] if rows else {},
    }
    raw = json.dumps(basis, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _feature_columns(manifest: dict[str, Any]) -> list[str]:
    names = [str(item.get("name")) for item in manifest.get("features") or [] if item.get("name")]
    mapping = {
        "ema_trend": ["ema_fast", "ema_slow", "ema_trend"],
        "adx_directional_strength": ["adx", "plus_di", "minus_di"],
        "rsi_14": ["rsi_14"],
        "fib_ote_zone": [
            "fib_pullback_long_zone",
            "fib_pullback_short_zone",
            "fib_ote_long_zone",
            "fib_ote_short_zone",
        ],
        "liquidity_sweep_reclaim": [
            "liquidity_sweep_low_20",
            "liquidity_sweep_high_20",
            "liquidity_reclaim_long_20",
            "liquidity_reclaim_short_20",
            "liquidity_close_position",
        ],
        "rolling_vwap_reclaim": [
            "vwap_rolling_48",
            "vwap_distance_pct_48",
            "vwap_reclaim_long_48",
            "vwap_reclaim_short_48",
            "vwap_mid_reclaim_long_48",
            "vwap_mid_reclaim_short_48",
        ],
        "obv_volume_zscore": ["obv", "obv_zscore_20", "volume_zscore_20"],
        "ml_regime_state": [
            "ml_return_1",
            "ml_return_5",
            "ml_return_20",
            "ml_atr_pct_14",
            "ml_realized_vol_20",
            "ml_realized_vol_80",
            "ml_vol_ratio_20_80",
            "ml_trend_slope_20",
            "ml_trend_slope_80",
            "ml_trend_efficiency_20",
            "ml_range_position_80",
            "ml_volatility_regime",
            "ml_trend_regime",
            "ml_turbulence_regime",
            "ml_session_bucket",
        ],
        "ml_execution_quality": [
            "ml_volume_z_20",
            "ml_quote_volume_z_20",
            "ml_candle_body_pct",
            "ml_candle_range_pct",
            "ml_wick_imbalance",
            "ml_turbulence_20",
            "ml_liquidity_pressure",
            "ml_liquidity_regime",
        ],
        "ml_payoff_potential": [
            "ml_payoff_potential_long",
            "ml_payoff_potential_short",
        ],
    }
    columns: list[str] = []
    for name in names:
        columns.extend(mapping.get(name, [name]))
    return list(dict.fromkeys(columns))


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def _json_safe_value(value: Any) -> Any:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return value if value is not None else ""
    if math.isnan(parsed) or math.isinf(parsed):
        return ""
    return round(parsed, 8)


def _triple_barrier_label(
    df: Any,
    *,
    index: int,
    side: str,
    atr_stop_multiple: float,
    take_profit_r: float,
    time_limit_bars: int,
) -> dict[str, Any]:
    row = df.iloc[index]
    entry = _float_value(row.get("close"))
    atr = _float_value(row.get("atr_14"))
    if entry <= 0.0 or atr <= 0.0:
        return {
            "label_side": side,
            "label_outcome": "unavailable",
            "label_r_multiple": 0.0,
            "label_bars_to_event": 0,
        }
    risk_distance = max(atr * atr_stop_multiple, entry * 0.0005)
    if side == "BUY":
        stop = entry - risk_distance
        target = entry + risk_distance * take_profit_r
    else:
        stop = entry + risk_distance
        target = entry - risk_distance * take_profit_r
    end = min(len(df) - 1, index + max(int(time_limit_bars), 1))
    last_close = entry
    for future_idx in range(index + 1, end + 1):
        future = df.iloc[future_idx]
        high = _float_value(future.get("high"))
        low = _float_value(future.get("low"))
        close = _float_value(future.get("close"), last_close)
        last_close = close
        if side == "BUY":
            if low <= stop:
                return {
                    "label_side": side,
                    "label_outcome": "stop_loss",
                    "label_r_multiple": -1.0,
                    "label_bars_to_event": future_idx - index,
                }
            if high >= target:
                return {
                    "label_side": side,
                    "label_outcome": "take_profit",
                    "label_r_multiple": round(float(take_profit_r), 4),
                    "label_bars_to_event": future_idx - index,
                }
        else:
            if high >= stop:
                return {
                    "label_side": side,
                    "label_outcome": "stop_loss",
                    "label_r_multiple": -1.0,
                    "label_bars_to_event": future_idx - index,
                }
            if low <= target:
                return {
                    "label_side": side,
                    "label_outcome": "take_profit",
                    "label_r_multiple": round(float(take_profit_r), 4),
                    "label_bars_to_event": future_idx - index,
                }
    raw_r = (last_close - entry) / risk_distance
    if side == "SELL":
        raw_r *= -1.0
    return {
        "label_side": side,
        "label_outcome": "time_limit",
        "label_r_multiple": round(raw_r, 4),
        "label_bars_to_event": max(end - index, 0),
    }


def _dual_side_labels(df: Any, index: int, strategy: Any, label_spec: dict[str, Any]) -> dict[str, Any]:
    take_profit_r = _float_value(label_spec.get("take_profit_r"), 1.5)
    time_limit_bars = int(label_spec.get("time_limit_bars") or strategy.risk.time_limit_bars or 72)
    atr_stop_multiple = _float_value(strategy.risk.atr_stop_multiple, 1.0)
    long_label = _triple_barrier_label(
        df,
        index=index,
        side="BUY",
        atr_stop_multiple=atr_stop_multiple,
        take_profit_r=take_profit_r,
        time_limit_bars=time_limit_bars,
    )
    short_label = _triple_barrier_label(
        df,
        index=index,
        side="SELL",
        atr_stop_multiple=atr_stop_multiple,
        take_profit_r=take_profit_r,
        time_limit_bars=time_limit_bars,
    )
    return {
        "label_long_outcome": long_label["label_outcome"],
        "label_long_r": long_label["label_r_multiple"],
        "label_long_bars": long_label["label_bars_to_event"],
        "label_short_outcome": short_label["label_outcome"],
        "label_short_r": short_label["label_r_multiple"],
        "label_short_bars": short_label["label_bars_to_event"],
    }


def build_feature_dataset(
    settings: Settings | None,
    *,
    spec: FeatureDatasetSpec,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    strategy = load_strategy_config(spec.strategy_config)
    manifest = build_feature_manifest()
    columns = _feature_columns(manifest)
    label_spec = (manifest.get("labels") or [{}])[0]
    root = Path(output_dir).expanduser().resolve() if output_dir else FEATURE_DATASET_DIR / _stamp()
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with BinanceClient(settings) as client:
        for symbol in spec.symbols:
            for interval in spec.intervals:
                try:
                    raw_klines = fetch_recent_klines(client, symbol, interval, spec.limit, spec.market)
                    df = enrich_indicators(
                        prepare_klines_frame(raw_klines),
                        interval,
                        strategy=strategy,
                    )
                    ml_features = build_ai_ml_features(df, interval)
                    df = df.join(ml_features)
                    tail_start = max(len(df) - spec.limit, 0)
                    for index, (timestamp, row) in enumerate(df.iloc[tail_start:].iterrows(), start=tail_start):
                        item: dict[str, Any] = {
                            "symbol": symbol.upper(),
                            "interval": interval,
                            "timestamp": str(timestamp),
                            "open": float(row.get("open", 0.0) or 0.0),
                            "high": float(row.get("high", 0.0) or 0.0),
                            "low": float(row.get("low", 0.0) or 0.0),
                            "close": float(row.get("close", 0.0) or 0.0),
                            "volume": float(row.get("volume", 0.0) or 0.0),
                        }
                        for column in columns:
                            item[column] = _json_safe_value(row.get(column))
                        item.update(_dual_side_labels(df, index, strategy, label_spec))
                        rows.append(item)
                except Exception as exc:
                    errors.append({"symbol": symbol.upper(), "interval": interval, "error": str(exc)})
    dataset_hash = _dataset_hash(rows, str(manifest.get("manifest_hash") or ""))
    dataset_path = root / "feature-dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    payload = {
        "generated_at": _utc_now().isoformat(),
        "mode": "feature_dataset",
        "spec": spec.to_dict(),
        "feature_manifest": manifest,
        "dataset_hash": dataset_hash,
        "row_count": len(rows),
        "dataset_path": str(dataset_path),
        "errors": errors,
        "replay_contract": {
            "unit": "symbol_interval_timestamp",
            "point_in_time": True,
            "lookahead_allowed": False,
        },
    }
    report_path = root / "feature-dataset-summary.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload
