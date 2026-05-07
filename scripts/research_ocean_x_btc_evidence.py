#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import sys
import zipfile
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _project_python import ensure_project_python  # noqa: E402

ensure_project_python(PROJECT_ROOT)

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from binance_quant_control.analysis import enrich_indicators, prepare_klines_frame  # noqa: E402

BINANCE_PUBLIC_ROOT = "https://data.binance.vision/data"
TRADINGVIEW_L5_URL = "https://www.tradingview.com/script/6kRPcRVr-blackcat-L5-Whales-Jump-Out-of-Ocean-X/"
TRADINGVIEW_L3_URL = "https://www.tradingview.com/script/791WkWcm-blackcat-L3-Banker-Fund-Flow-Trend-Oscillator/"
BINANCE_PUBLIC_DATA_URL = "https://github.com/binance/binance-public-data"
BINANCE_FAPI_KLINES_URL = (
    "https://developers.binance.com/docs/derivatives/"
    "usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data"
)
CORE_WHALE_JUMP_SYMBOLS = ("BTCUSDT", "ETHUSDT", "XAUTUSDT")
BTC_ETH_TRADINGVIEW_SYMBOLS = ("BTCUSDT", "ETHUSDT")
REGIME_FILTERS = ("none", "trend", "pullback", "liquidity", "range", "strong_flow")
TRADINGVIEW_SIGNAL_FAMILIES = (
    "tv_supertrend_macd",
    "tv_stoch_rsi_pullback",
    "tv_vwap_trend",
    "tv_range_rsi",
    "banker_flow_proxy",
)
GATE_MODES = ("strict_win_rate", "expectancy")


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_type: str
    url: str
    checksum_url: str
    filename: str
    status_code: int
    rows: int
    sha256: str
    checksum_sha256: str
    checksum_ok: bool | None
    error: str = ""


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def _add_month(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def _previous_month(value: date) -> date:
    if value.month == 1:
        return date(value.year - 1, 12, 1)
    return date(value.year, value.month - 1, 1)


def _month_range(start: date, end: date) -> list[str]:
    current = _month_start(start)
    last = _month_start(end)
    months: list[str] = []
    while current <= last:
        months.append(f"{current.year:04d}-{current.month:02d}")
        current = _add_month(current)
    return months


def _date_range(start: date, end_exclusive: date) -> list[date]:
    days: list[date] = []
    current = start
    while current < end_exclusive:
        days.append(current)
        current += timedelta(days=1)
    return days


def _archive_url(*, market: str, symbol: str, interval: str, period: str, source_type: str) -> str:
    market_path = "futures/um" if market == "futures" else "spot"
    filename = f"{symbol}-{interval}-{period}.zip"
    return f"{BINANCE_PUBLIC_ROOT}/{market_path}/{source_type}/klines/{symbol}/{interval}/{filename}"


def _parse_checksum(text: str) -> str:
    token = text.strip().split()[0] if text.strip() else ""
    if len(token) == 64 and all(char.lower() in "0123456789abcdef" for char in token):
        return token.lower()
    return ""


def _read_zip_rows(blob: bytes) -> list[list[Any]]:
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        csv_name = next((name for name in archive.namelist() if name.endswith(".csv")), "")
        if not csv_name:
            return []
        with archive.open(csv_name) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8")
            rows = [row for row in csv.reader(text) if row]
    if rows and rows[0][0].lower() in {"open_time", "open time"}:
        rows = rows[1:]
    return rows


def _fetch_archive(
    client: httpx.Client,
    *,
    market: str,
    symbol: str,
    interval: str,
    period: str,
    source_type: str,
) -> tuple[list[list[Any]], SourceRecord]:
    url = _archive_url(
        market=market,
        symbol=symbol,
        interval=interval,
        period=period,
        source_type=source_type,
    )
    checksum_url = f"{url}.CHECKSUM"
    filename = url.rsplit("/", 1)[-1]
    try:
        response = client.get(url)
        if response.status_code == 404:
            return [], SourceRecord(
                source_type=source_type,
                url=url,
                checksum_url=checksum_url,
                filename=filename,
                status_code=404,
                rows=0,
                sha256="",
                checksum_sha256="",
                checksum_ok=None,
                error="not-found",
            )
        response.raise_for_status()
        blob = response.content
        sha256 = hashlib.sha256(blob).hexdigest()
        checksum = ""
        checksum_ok: bool | None = None
        checksum_response = client.get(checksum_url)
        if checksum_response.status_code == 200:
            checksum = _parse_checksum(checksum_response.text)
            checksum_ok = bool(checksum) and checksum == sha256
        elif checksum_response.status_code == 404:
            checksum_ok = None
        else:
            checksum_response.raise_for_status()
        rows = _read_zip_rows(blob)
        return rows, SourceRecord(
            source_type=source_type,
            url=url,
            checksum_url=checksum_url,
            filename=filename,
            status_code=response.status_code,
            rows=len(rows),
            sha256=sha256,
            checksum_sha256=checksum,
            checksum_ok=checksum_ok,
        )
    except Exception as exc:  # pragma: no cover - exercised by live network runs.
        return [], SourceRecord(
            source_type=source_type,
            url=url,
            checksum_url=checksum_url,
            filename=filename,
            status_code=0,
            rows=0,
            sha256="",
            checksum_sha256="",
            checksum_ok=False,
            error=str(exc),
        )


def _source_periods(start: date, end_exclusive: date) -> list[tuple[str, str]]:
    periods: list[tuple[str, str]] = []
    end_month = _month_start(end_exclusive)
    last_full_month = _previous_month(end_month)
    if _month_start(start) <= last_full_month:
        periods.extend(("monthly", month) for month in _month_range(start, last_full_month))

    daily_start = max(start, end_month)
    periods.extend(("daily", day.isoformat()) for day in _date_range(daily_start, end_exclusive))
    return periods


def fetch_history(
    *,
    market: str,
    symbol: str,
    interval: str,
    start: date,
    end_exclusive: date,
    timeout: float,
) -> tuple[pd.DataFrame, list[SourceRecord]]:
    rows: list[list[Any]] = []
    sources: list[SourceRecord] = []
    periods = _source_periods(start, end_exclusive)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for source_type, period in periods:
            archive_rows, record = _fetch_archive(
                client,
                market=market,
                symbol=symbol,
                interval=interval,
                period=period,
                source_type=source_type,
            )
            sources.append(record)
            rows.extend(archive_rows)
    if not rows:
        raise RuntimeError(f"No public rows fetched for {symbol} {market} {interval}.")

    frame = prepare_klines_frame(rows)
    start_ts = pd.Timestamp(datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc))
    end_ts = pd.Timestamp(datetime.combine(end_exclusive, datetime.min.time(), tzinfo=timezone.utc))
    frame = frame[(frame.index >= start_ts) & (frame.index < end_ts)]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    if frame.empty:
        raise RuntimeError(f"No rows remained after date filtering for {symbol} {interval}.")
    return frame, sources


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def build_proxy_base_features(
    frame: pd.DataFrame,
    *,
    interval: str,
) -> pd.DataFrame:
    enriched = enrich_indicators(frame, interval).copy()
    quote_volume = pd.to_numeric(enriched["quote_asset_volume"], errors="coerce").replace(0, np.nan)
    taker_quote = pd.to_numeric(enriched["taker_buy_quote_volume"], errors="coerce")
    taker_buy_share = (taker_quote / quote_volume).astype(float)
    taker_sell_quote = (quote_volume - taker_quote).where((quote_volume - taker_quote) > 0, np.nan)
    enriched["taker_buy_share"] = taker_buy_share
    enriched["taker_buy_sell_ratio"] = (taker_quote / taker_sell_quote).astype(float)
    enriched["taker_flow_imbalance"] = ((taker_buy_share - 0.5) * 2.0).astype(float)
    enriched["candle_return_pct"] = ((enriched["close"] / enriched["open"]) - 1.0) * 100.0
    return enriched


def build_proxy_features(
    frame: pd.DataFrame,
    *,
    interval: str,
    volume_z: float,
    extreme_volume_z: float,
    taker_share: float,
) -> pd.DataFrame:
    enriched = build_proxy_base_features(frame, interval=interval)
    volume_spike = (enriched["volume_zscore_20"] >= volume_z) | (enriched["volume_ratio_20"] >= 1.45)
    extreme_volume = (enriched["volume_zscore_20"] >= extreme_volume_z) | (
        enriched["volume_ratio_20"] >= 2.0
    )
    long_pressure = (
        volume_spike
        & (enriched["taker_buy_share"] >= taker_share)
        & (enriched["close"] > enriched["open"])
        & (enriched["mfi_14"] >= 50.0)
        & (enriched["jumbo_power"] >= enriched["jumbo_power_ma"])
    )
    short_pressure = (
        volume_spike
        & (enriched["taker_buy_share"] <= (1.0 - taker_share))
        & (enriched["close"] < enriched["open"])
        & (enriched["mfi_14"] <= 50.0)
        & (enriched["jumbo_power"] <= enriched["jumbo_power_ma"])
    )
    long_structure = (
        enriched["fib_pullback_long_zone"]
        | enriched["fib_ote_long_zone"]
        | enriched["liquidity_reclaim_long_20"]
        | enriched["jumbo_long_signal"]
    )
    short_structure = (
        enriched["fib_pullback_short_zone"]
        | enriched["fib_ote_short_zone"]
        | enriched["liquidity_reclaim_short_20"]
        | enriched["jumbo_short_signal"]
    )

    enriched["ocean_proxy_l"] = long_pressure
    enriched["ocean_proxy_s"] = short_pressure
    enriched["ocean_proxy_xl"] = long_pressure & extreme_volume & long_structure
    enriched["ocean_proxy_xs"] = short_pressure & extreme_volume & short_structure
    enriched["ocean_proxy_signal"] = ""
    enriched.loc[enriched["ocean_proxy_l"], "ocean_proxy_signal"] = "L"
    enriched.loc[enriched["ocean_proxy_s"], "ocean_proxy_signal"] = "S"
    enriched.loc[enriched["ocean_proxy_xl"], "ocean_proxy_signal"] = "XL"
    enriched.loc[enriched["ocean_proxy_xs"], "ocean_proxy_signal"] = "XS"
    return enriched


def _direction_for_signal(signal: str) -> int:
    return 1 if signal in {"L", "XL"} else -1


def _horizon_stats(values: list[float]) -> dict[str, Any]:
    series = pd.Series(values, dtype="float64").dropna()
    if series.empty:
        return {"n": 0, "win_rate": None, "avg_pct": None, "median_pct": None}
    return {
        "n": int(series.count()),
        "win_rate": round(float((series > 0.0).mean()), 4),
        "avg_pct": round(float(series.mean()), 4),
        "median_pct": round(float(series.median()), 4),
    }


@dataclass(frozen=True, slots=True)
class OceanProxyParams:
    volume_z: float
    volume_ratio: float
    extreme_volume_z: float
    extreme_volume_ratio: float
    taker_share: float
    mfi_long: float
    mfi_short: float
    min_abs_jumbo_delta: float
    min_adx: float
    require_structure: bool
    allow_regular_signals: bool
    stop_loss_pct: float
    take_profit_pct: float
    max_hold_bars: int
    fee_bps: float
    slippage_bps: float
    regime_filter: str = "none"

    def key(self) -> str:
        return (
            f"vz{self.volume_z:g}-vr{self.volume_ratio:g}-ez{self.extreme_volume_z:g}-"
            f"er{self.extreme_volume_ratio:g}-ts{self.taker_share:g}-"
            f"ml{self.mfi_long:g}-ms{self.mfi_short:g}-jd{self.min_abs_jumbo_delta:g}-"
            f"adx{self.min_adx:g}-struct{int(self.require_structure)}-reg{int(self.allow_regular_signals)}-"
            f"sl{self.stop_loss_pct:g}-tp{self.take_profit_pct:g}-hold{self.max_hold_bars}-"
            f"regime{self.regime_filter}"
        )


@dataclass(frozen=True, slots=True)
class TradingViewConvergenceParams:
    family: str
    side: str
    stop_loss_pct: float
    take_profit_pct: float
    max_hold_bars: int
    fee_bps: float
    slippage_bps: float
    regime_filter: str = "none"
    min_adx: float = 16.0
    max_adx: float = 28.0
    min_trend_votes: int = 2
    min_volume_z: float = 0.0
    min_volume_ratio: float = 1.0
    min_abs_taker_flow: float = 0.0
    min_abs_jumbo_delta: float = 0.0
    rsi_low: float = 35.0
    rsi_high: float = 65.0
    stoch_low: float = 20.0
    stoch_high: float = 80.0
    bb_low: float = 0.25
    bb_high: float = 0.75
    require_vwap: bool = False
    require_structure: bool = False
    require_liquidity_reclaim: bool = False
    require_mfi: bool = False

    def key(self) -> str:
        return (
            f"tv-{self.family}-{self.side}-sl{self.stop_loss_pct:g}-tp{self.take_profit_pct:g}-"
            f"hold{self.max_hold_bars}-adx{self.min_adx:g}-{self.max_adx:g}-"
            f"votes{self.min_trend_votes}-vz{self.min_volume_z:g}-vr{self.min_volume_ratio:g}-"
            f"tf{self.min_abs_taker_flow:g}-jd{self.min_abs_jumbo_delta:g}-"
            f"rsi{self.rsi_low:g}-{self.rsi_high:g}-stoch{self.stoch_low:g}-{self.stoch_high:g}-"
            f"bb{self.bb_low:g}-{self.bb_high:g}-vwap{int(self.require_vwap)}-"
            f"struct{int(self.require_structure)}-liq{int(self.require_liquidity_reclaim)}-"
            f"mfi{int(self.require_mfi)}-regime{self.regime_filter}"
        )


def _default_param_grid() -> list[OceanProxyParams]:
    params: list[OceanProxyParams] = []
    for (
        volume_z,
        taker_share,
        min_abs_jumbo_delta,
        min_adx,
        require_structure,
        allow_regular_signals,
        stop_loss_pct,
        take_profit_pct,
        max_hold_bars,
    ) in itertools.product(
        (1.3, 1.6, 1.9, 2.2),
        (0.56, 0.58, 0.60, 0.62),
        (0.0, 4.0, 8.0),
        (12.0, 16.0, 20.0),
        (False, True),
        (False, True),
        (0.8, 1.1, 1.5),
        (0.8, 1.2, 1.8, 2.4),
        (12, 24, 36),
    ):
        params.append(
            OceanProxyParams(
                volume_z=volume_z,
                volume_ratio=max(1.2, 1.0 + volume_z * 0.22),
                extreme_volume_z=volume_z + 0.5,
                extreme_volume_ratio=max(1.7, 1.0 + (volume_z + 0.5) * 0.26),
                taker_share=taker_share,
                mfi_long=50.0,
                mfi_short=50.0,
                min_abs_jumbo_delta=min_abs_jumbo_delta,
                min_adx=min_adx,
                require_structure=require_structure,
                allow_regular_signals=allow_regular_signals,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                max_hold_bars=max_hold_bars,
                fee_bps=4.0,
                slippage_bps=2.0,
            )
        )
    return params


def _limit_param_grid(params: list[OceanProxyParams], max_configs: int) -> list[OceanProxyParams]:
    if max_configs <= 0 or max_configs >= len(params):
        return params
    if max_configs == 1:
        return [params[0]]
    last = len(params) - 1
    selected_indexes = sorted({round(idx * last / (max_configs - 1)) for idx in range(max_configs)})
    return [params[idx] for idx in selected_indexes]


def _default_tradingview_param_grid(
    *,
    max_per_trade_risk_pct: float = 2.5,
) -> list[TradingViewConvergenceParams]:
    exit_profiles = (
        (0.9, 0.55, 10),
        (1.1, 0.70, 12),
        (1.3, 0.90, 18),
        (1.5, 1.10, 24),
        (1.8, 1.30, 30),
        (2.2, 1.55, 36),
        (0.7, 1.4, 18),
        (0.9, 1.8, 24),
        (1.1, 2.2, 36),
        (1.2, 3.0, 48),
        (1.5, 3.0, 60),
        (1.8, 3.6, 72),
    )
    family_profiles: dict[str, tuple[dict[str, Any], ...]] = {
        "tv_supertrend_macd": (
            {"min_adx": 16.0, "min_trend_votes": 4},
            {"min_adx": 20.0, "min_trend_votes": 4, "min_volume_z": 0.4},
            {"min_adx": 20.0, "min_trend_votes": 5, "min_abs_taker_flow": 0.06},
            {"min_adx": 24.0, "min_trend_votes": 5, "require_vwap": True},
        ),
        "tv_stoch_rsi_pullback": (
            {"min_adx": 14.0, "min_trend_votes": 3, "rsi_low": 38.0, "rsi_high": 58.0, "stoch_low": 18.0, "stoch_high": 55.0},
            {"min_adx": 16.0, "min_trend_votes": 3, "rsi_low": 40.0, "rsi_high": 62.0, "stoch_low": 20.0, "stoch_high": 65.0},
            {"min_adx": 18.0, "min_trend_votes": 4, "rsi_low": 42.0, "rsi_high": 66.0, "stoch_low": 25.0, "stoch_high": 70.0, "require_structure": True},
            {"min_adx": 20.0, "min_trend_votes": 4, "rsi_low": 35.0, "rsi_high": 60.0, "stoch_low": 12.0, "stoch_high": 58.0, "min_abs_jumbo_delta": 1.0},
        ),
        "tv_vwap_trend": (
            {"min_adx": 15.0, "min_trend_votes": 3, "require_vwap": True},
            {"min_adx": 18.0, "min_trend_votes": 4, "require_vwap": True, "min_abs_taker_flow": 0.06},
            {"min_adx": 20.0, "min_trend_votes": 4, "require_vwap": True, "min_volume_ratio": 1.12},
            {"min_adx": 22.0, "min_trend_votes": 5, "require_vwap": True, "min_volume_z": 0.4, "min_abs_taker_flow": 0.08},
        ),
        "tv_range_rsi": (
            {"max_adx": 24.0, "rsi_low": 34.0, "rsi_high": 66.0, "stoch_low": 18.0, "stoch_high": 82.0, "bb_low": 0.22, "bb_high": 0.78},
            {"max_adx": 26.0, "rsi_low": 36.0, "rsi_high": 64.0, "stoch_low": 22.0, "stoch_high": 78.0, "bb_low": 0.25, "bb_high": 0.75},
            {"max_adx": 28.0, "rsi_low": 38.0, "rsi_high": 62.0, "stoch_low": 25.0, "stoch_high": 75.0, "bb_low": 0.30, "bb_high": 0.70, "require_liquidity_reclaim": True},
            {"max_adx": 30.0, "rsi_low": 32.0, "rsi_high": 68.0, "stoch_low": 15.0, "stoch_high": 85.0, "bb_low": 0.20, "bb_high": 0.80, "min_abs_taker_flow": 0.04},
        ),
        "banker_flow_proxy": (
            {"min_adx": 12.0, "min_trend_votes": 2, "min_volume_z": 0.4, "min_abs_taker_flow": 0.06, "require_mfi": True},
            {"min_adx": 14.0, "min_trend_votes": 3, "min_volume_ratio": 1.15, "min_abs_taker_flow": 0.08, "min_abs_jumbo_delta": 1.0, "require_mfi": True},
            {"min_adx": 16.0, "min_trend_votes": 3, "min_volume_z": 0.7, "min_abs_taker_flow": 0.10, "min_abs_jumbo_delta": 2.0, "require_structure": True, "require_mfi": True},
            {"min_adx": 18.0, "min_trend_votes": 4, "min_volume_ratio": 1.25, "min_abs_taker_flow": 0.12, "min_abs_jumbo_delta": 3.0, "require_liquidity_reclaim": True, "require_mfi": True},
        ),
    }
    params: list[TradingViewConvergenceParams] = []
    for family, profiles in family_profiles.items():
        for side, profile, (stop_loss_pct, take_profit_pct, max_hold_bars) in itertools.product(
            ("long", "short"),
            profiles,
            exit_profiles,
        ):
            if stop_loss_pct > max_per_trade_risk_pct:
                continue
            params.append(
                TradingViewConvergenceParams(
                    family=family,
                    side=side,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                    max_hold_bars=max_hold_bars,
                    fee_bps=4.0,
                    slippage_bps=2.0,
                    **profile,
                )
            )
    return params


def _limit_tradingview_param_grid(
    params: list[TradingViewConvergenceParams],
    max_configs: int,
) -> list[TradingViewConvergenceParams]:
    if max_configs <= 0 or max_configs >= len(params):
        return params
    if max_configs == 1:
        return [params[0]]
    last = len(params) - 1
    selected_indexes = sorted({round(idx * last / (max_configs - 1)) for idx in range(max_configs)})
    return [params[idx] for idx in selected_indexes]


def _parse_csv_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_regime_filters(regime_filters: list[str] | None) -> list[str]:
    selected = regime_filters or ["none"]
    normalized: list[str] = []
    for item in selected:
        mode = item.strip().lower().replace("-", "_")
        if mode == "all":
            return list(REGIME_FILTERS)
        if mode not in REGIME_FILTERS:
            raise ValueError(f"Unsupported regime filter {item!r}; choose from {', '.join(REGIME_FILTERS)}.")
        if mode not in normalized:
            normalized.append(mode)
    return normalized or ["none"]


def _bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index)
    return frame[column].fillna(default).astype(bool)


def _num_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def apply_regime_filter(features: pd.DataFrame, *, signal: str, regime_filter: str) -> pd.DataFrame:
    mode = _normalize_regime_filters([regime_filter])[0]
    if mode == "none":
        return features

    filtered = features.copy()
    signal_mask = filtered["ocean_proxy_signal"] == signal
    if not signal_mask.any():
        return filtered

    long_side = signal in {"L", "XL"}
    close = _num_series(filtered, "close")
    ema_fast = _num_series(filtered, "ema_fast")
    ema_slow = _num_series(filtered, "ema_slow")
    sma_200 = _num_series(filtered, "sma_200")
    macd_hist = _num_series(filtered, "macd_hist")
    plus_di = _num_series(filtered, "plus_di")
    minus_di = _num_series(filtered, "minus_di")
    supertrend_direction = _num_series(filtered, "supertrend_direction")
    adx_value = _num_series(filtered, "adx")
    bb_percent_b = _num_series(filtered, "bb_percent_b", 0.5)
    rsi_value = _num_series(filtered, "rsi_14", 50.0)
    jumbo_power = _num_series(filtered, "jumbo_power")
    jumbo_ma = _num_series(filtered, "jumbo_power_ma")
    taker_flow = _num_series(filtered, "taker_flow_imbalance")
    long_structure = _bool_series(filtered, "fib_pullback_long_zone") | _bool_series(
        filtered, "fib_ote_long_zone"
    )
    short_structure = _bool_series(filtered, "fib_pullback_short_zone") | _bool_series(
        filtered, "fib_ote_short_zone"
    )

    if mode == "trend":
        if long_side:
            keep = (
                signal_mask
                & (close > ema_slow)
                & (ema_fast > ema_slow)
                & (macd_hist >= 0.0)
                & (plus_di >= minus_di)
                & (supertrend_direction >= 0.0)
            )
        else:
            keep = (
                signal_mask
                & (close < ema_slow)
                & (ema_fast < ema_slow)
                & (macd_hist <= 0.0)
                & (minus_di >= plus_di)
                & (supertrend_direction <= 0.0)
            )
    elif mode == "pullback":
        if long_side:
            keep = (
                signal_mask
                & (ema_fast > ema_slow)
                & ((sma_200 <= 0.0) | (close > sma_200))
                & (long_structure | bb_percent_b.between(0.25, 0.72))
                & rsi_value.between(42.0, 68.0)
            )
        else:
            keep = (
                signal_mask
                & (ema_fast < ema_slow)
                & ((sma_200 <= 0.0) | (close < sma_200))
                & (short_structure | bb_percent_b.between(0.28, 0.75))
                & rsi_value.between(32.0, 58.0)
            )
    elif mode == "liquidity":
        keep = signal_mask & (
            _bool_series(filtered, "liquidity_reclaim_long_20")
            if long_side
            else _bool_series(filtered, "liquidity_reclaim_short_20")
        )
    elif mode == "range":
        if long_side:
            keep = signal_mask & (adx_value <= 28.0) & (bb_percent_b <= 0.68) & (rsi_value <= 62.0)
        else:
            keep = signal_mask & (adx_value <= 28.0) & (bb_percent_b >= 0.32) & (rsi_value >= 38.0)
    elif mode == "strong_flow":
        if long_side:
            keep = signal_mask & (taker_flow >= 0.18) & (jumbo_power >= jumbo_ma)
        else:
            keep = signal_mask & (taker_flow <= -0.18) & (jumbo_power <= jumbo_ma)
    else:  # pragma: no cover - _normalize_regime_filters guards this.
        keep = signal_mask

    filtered.loc[signal_mask & ~keep, "ocean_proxy_signal"] = ""
    return filtered


def _profit_factor_value(value: Any) -> float:
    return 9999.0 if value == "inf" else _safe_float(value)


def _normalize_gate_mode(value: str) -> str:
    mode = value.strip().lower().replace("-", "_")
    if mode in {"strict", "win_rate", "high_win"}:
        return "strict_win_rate"
    if mode in {"expectancy", "expected_value", "ev"}:
        return "expectancy"
    if mode not in GATE_MODES:
        raise ValueError(f"Unsupported gate mode {value!r}; choose from {', '.join(GATE_MODES)}.")
    return mode


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return float("inf") if numerator > 0.0 else 0.0
    return numerator / denominator


def _sample_sharpe(pnls: list[float]) -> float:
    series = pd.Series(pnls, dtype="float64").dropna()
    if len(series) < 2:
        return 0.0
    std = float(series.std(ddof=1))
    if std <= 0.0 or not math.isfinite(std):
        return 9999.0 if float(series.mean()) > 0.0 else 0.0
    return float(series.mean()) / std * math.sqrt(len(series))


def _max_loss_streak(pnls: list[float]) -> int:
    current = 0
    max_streak = 0
    for pnl in pnls:
        if pnl <= 0.0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _dataset_quality(
    frame: pd.DataFrame,
    *,
    requested_start: date,
    requested_end: date,
    min_coverage_ratio: float,
    min_dataset_bars: int,
) -> dict[str, Any]:
    first = frame.index.min()
    last = frame.index.max()
    requested_days = max((requested_end - requested_start).days, 1)
    covered_days = max((last.date() - first.date()).days, 0)
    coverage_ratio = min(1.0, covered_days / requested_days)
    bars = int(len(frame))
    return {
        "bars": bars,
        "first": first.isoformat(),
        "last": last.isoformat(),
        "covered_days": covered_days,
        "requested_days": requested_days,
        "coverage_ratio": round(coverage_ratio, 4),
        "min_coverage_ratio": min_coverage_ratio,
        "min_dataset_bars": min_dataset_bars,
        "mature": bars >= min_dataset_bars and coverage_ratio >= min_coverage_ratio,
    }


def build_proxy_features_with_params(
    frame: pd.DataFrame | None = None,
    *,
    interval: str,
    params: OceanProxyParams,
    base_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if base_features is None:
        if frame is None:
            raise ValueError("frame or base_features is required")
        enriched = build_proxy_base_features(frame, interval=interval)
    else:
        enriched = base_features.copy()
    volume_spike = (enriched["volume_zscore_20"] >= params.volume_z) | (
        enriched["volume_ratio_20"] >= params.volume_ratio
    )
    extreme_volume = (enriched["volume_zscore_20"] >= params.extreme_volume_z) | (
        enriched["volume_ratio_20"] >= params.extreme_volume_ratio
    )
    long_structure = (
        enriched["fib_pullback_long_zone"]
        | enriched["fib_ote_long_zone"]
        | enriched["liquidity_reclaim_long_20"]
        | enriched["jumbo_long_signal"]
    )
    short_structure = (
        enriched["fib_pullback_short_zone"]
        | enriched["fib_ote_short_zone"]
        | enriched["liquidity_reclaim_short_20"]
        | enriched["jumbo_short_signal"]
    )
    jumbo_delta = enriched["jumbo_power"] - enriched["jumbo_power_ma"]
    base_long = (
        volume_spike
        & (enriched["taker_buy_share"] >= params.taker_share)
        & (enriched["close"] > enriched["open"])
        & (enriched["mfi_14"] >= params.mfi_long)
        & (jumbo_delta >= params.min_abs_jumbo_delta)
        & (enriched["adx"] >= params.min_adx)
    )
    base_short = (
        volume_spike
        & (enriched["taker_buy_share"] <= (1.0 - params.taker_share))
        & (enriched["close"] < enriched["open"])
        & (enriched["mfi_14"] <= params.mfi_short)
        & (jumbo_delta <= -params.min_abs_jumbo_delta)
        & (enriched["adx"] >= params.min_adx)
    )
    if params.require_structure:
        base_long &= long_structure
        base_short &= short_structure
    xl = base_long & extreme_volume & long_structure
    xs = base_short & extreme_volume & short_structure
    long_signal = base_long & (params.allow_regular_signals | xl)
    short_signal = base_short & (params.allow_regular_signals | xs)
    enriched["ocean_proxy_l"] = long_signal
    enriched["ocean_proxy_s"] = short_signal
    enriched["ocean_proxy_xl"] = xl
    enriched["ocean_proxy_xs"] = xs
    enriched["ocean_proxy_signal"] = ""
    enriched.loc[long_signal, "ocean_proxy_signal"] = "L"
    enriched.loc[short_signal, "ocean_proxy_signal"] = "S"
    enriched.loc[xl, "ocean_proxy_signal"] = "XL"
    enriched.loc[xs, "ocean_proxy_signal"] = "XS"
    return enriched


def _trend_votes(features: pd.DataFrame, *, side: str) -> pd.Series:
    close = _num_series(features, "close")
    ema_fast = _num_series(features, "ema_fast")
    ema_slow = _num_series(features, "ema_slow")
    macd_hist = _num_series(features, "macd_hist")
    plus_di = _num_series(features, "plus_di")
    minus_di = _num_series(features, "minus_di")
    supertrend_direction = _num_series(features, "supertrend_direction")
    trend_magic_direction = _num_series(features, "trend_magic_direction")
    follow_line_direction = _num_series(features, "follow_line_direction")
    if side == "long":
        votes = [
            close > ema_slow,
            ema_fast > ema_slow,
            macd_hist >= 0.0,
            plus_di >= minus_di,
            supertrend_direction > 0.0,
            trend_magic_direction > 0.0,
            follow_line_direction > 0.0,
        ]
    else:
        votes = [
            close < ema_slow,
            ema_fast < ema_slow,
            macd_hist <= 0.0,
            minus_di >= plus_di,
            supertrend_direction < 0.0,
            trend_magic_direction < 0.0,
            follow_line_direction < 0.0,
        ]
    return sum(vote.fillna(False).astype(int) for vote in votes)


def _tradingview_common_masks(
    features: pd.DataFrame,
    params: TradingViewConvergenceParams,
) -> dict[str, pd.Series]:
    side = params.side
    close = _num_series(features, "close")
    ema_fast = _num_series(features, "ema_fast")
    ema_slow = _num_series(features, "ema_slow")
    vwap_value = _num_series(features, "vwap")
    adx_value = _num_series(features, "adx")
    volume_z = _num_series(features, "volume_zscore_20")
    volume_ratio = _num_series(features, "volume_ratio_20", 1.0)
    taker_flow = _num_series(features, "taker_flow_imbalance")
    jumbo_delta = _num_series(features, "jumbo_power") - _num_series(features, "jumbo_power_ma")
    mfi_value = _num_series(features, "mfi_14", 50.0)
    trend_votes = _trend_votes(features, side=side)
    long_structure = (
        _bool_series(features, "fib_pullback_long_zone")
        | _bool_series(features, "fib_ote_long_zone")
        | _bool_series(features, "jumbo_long_signal")
    )
    short_structure = (
        _bool_series(features, "fib_pullback_short_zone")
        | _bool_series(features, "fib_ote_short_zone")
        | _bool_series(features, "jumbo_short_signal")
    )
    if side == "long":
        vwap_aligned = close >= vwap_value
        flow_aligned = taker_flow >= params.min_abs_taker_flow
        jumbo_aligned = jumbo_delta >= params.min_abs_jumbo_delta
        structure = long_structure
        liquidity = _bool_series(features, "liquidity_reclaim_long_20")
        mfi_aligned = mfi_value >= max(50.0, params.rsi_low)
        price_ema_aligned = (close >= ema_slow) & (ema_fast >= ema_slow)
    else:
        vwap_aligned = close <= vwap_value
        flow_aligned = taker_flow <= -params.min_abs_taker_flow
        jumbo_aligned = jumbo_delta <= -params.min_abs_jumbo_delta
        structure = short_structure
        liquidity = _bool_series(features, "liquidity_reclaim_short_20")
        mfi_aligned = mfi_value <= min(50.0, params.rsi_high)
        price_ema_aligned = (close <= ema_slow) & (ema_fast <= ema_slow)
    volume_ok = (volume_z >= params.min_volume_z) | (volume_ratio >= params.min_volume_ratio)
    common = (trend_votes >= params.min_trend_votes) & volume_ok & flow_aligned & jumbo_aligned
    if params.require_vwap:
        common &= vwap_aligned
    if params.require_structure:
        common &= structure
    if params.require_liquidity_reclaim:
        common &= liquidity
    if params.require_mfi:
        common &= mfi_aligned
    return {
        "common": common.fillna(False),
        "trend_votes": trend_votes,
        "adx": adx_value,
        "vwap_aligned": vwap_aligned.fillna(False),
        "price_ema_aligned": price_ema_aligned.fillna(False),
        "structure": structure.fillna(False),
        "liquidity": liquidity.fillna(False),
    }


def build_tradingview_signal_features(
    base_features: pd.DataFrame,
    *,
    params: TradingViewConvergenceParams,
) -> pd.DataFrame:
    if params.family not in TRADINGVIEW_SIGNAL_FAMILIES:
        raise ValueError(
            f"Unsupported TradingView signal family {params.family!r}; "
            f"choose from {', '.join(TRADINGVIEW_SIGNAL_FAMILIES)}."
        )
    if params.side not in {"long", "short"}:
        raise ValueError("TradingView convergence side must be 'long' or 'short'.")

    features = base_features.copy()
    close = _num_series(features, "close")
    ema_fast = _num_series(features, "ema_fast")
    ema_slow = _num_series(features, "ema_slow")
    macd_hist = _num_series(features, "macd_hist")
    adx_value = _num_series(features, "adx")
    rsi_value = _num_series(features, "rsi_14", 50.0)
    stoch_rsi_k = _num_series(features, "stoch_rsi_k", 50.0)
    bb_percent_b = _num_series(features, "bb_percent_b", 0.5)
    taker_flow = _num_series(features, "taker_flow_imbalance")
    jumbo_delta = _num_series(features, "jumbo_power") - _num_series(features, "jumbo_power_ma")
    masks = _tradingview_common_masks(features, params)
    common = masks["common"]
    side = params.side

    if params.family == "tv_supertrend_macd":
        if side == "long":
            signal_mask = common & (adx_value >= params.min_adx) & (close >= ema_slow) & (macd_hist >= 0.0)
        else:
            signal_mask = common & (adx_value >= params.min_adx) & (close <= ema_slow) & (macd_hist <= 0.0)
    elif params.family == "tv_stoch_rsi_pullback":
        trend_ok = (adx_value >= params.min_adx) & masks["price_ema_aligned"]
        if side == "long":
            pullback_ok = rsi_value.between(params.rsi_low, params.rsi_high) & stoch_rsi_k.between(
                params.stoch_low,
                params.stoch_high,
            )
            momentum_ok = (macd_hist >= -0.05) & (jumbo_delta >= -abs(params.min_abs_jumbo_delta))
        else:
            pullback_ok = rsi_value.between(100.0 - params.rsi_high, 100.0 - params.rsi_low) & stoch_rsi_k.between(
                100.0 - params.stoch_high,
                100.0 - params.stoch_low,
            )
            momentum_ok = (macd_hist <= 0.05) & (jumbo_delta <= abs(params.min_abs_jumbo_delta))
        signal_mask = common & trend_ok & pullback_ok & momentum_ok
    elif params.family == "tv_vwap_trend":
        if side == "long":
            signal_mask = common & (adx_value >= params.min_adx) & masks["vwap_aligned"] & (close >= ema_fast)
        else:
            signal_mask = common & (adx_value >= params.min_adx) & masks["vwap_aligned"] & (close <= ema_fast)
    elif params.family == "tv_range_rsi":
        range_ok = adx_value <= params.max_adx
        if side == "long":
            location_ok = (bb_percent_b <= params.bb_low) | (stoch_rsi_k <= params.stoch_low)
            oscillator_ok = rsi_value <= params.rsi_low
            flow_not_hostile = taker_flow >= -max(params.min_abs_taker_flow, 0.08)
        else:
            location_ok = (bb_percent_b >= params.bb_high) | (stoch_rsi_k >= params.stoch_high)
            oscillator_ok = rsi_value >= params.rsi_high
            flow_not_hostile = taker_flow <= max(params.min_abs_taker_flow, 0.08)
        signal_mask = common & range_ok & location_ok & oscillator_ok & flow_not_hostile
    else:
        if side == "long":
            flow_ok = (taker_flow >= params.min_abs_taker_flow) & (jumbo_delta >= params.min_abs_jumbo_delta)
            oscillator_ok = rsi_value.between(params.rsi_low, params.rsi_high + 10.0)
        else:
            flow_ok = (taker_flow <= -params.min_abs_taker_flow) & (jumbo_delta <= -params.min_abs_jumbo_delta)
            oscillator_ok = rsi_value.between(params.rsi_low - 10.0, params.rsi_high)
        signal_mask = common & (adx_value >= params.min_adx) & flow_ok & oscillator_ok

    features["tv_family"] = params.family
    features["tv_trend_votes"] = masks["trend_votes"]
    features["ocean_proxy_l"] = False
    features["ocean_proxy_s"] = False
    features["ocean_proxy_xl"] = False
    features["ocean_proxy_xs"] = False
    features["ocean_proxy_signal"] = ""
    signal = "L" if params.side == "long" else "S"
    if signal == "L":
        features.loc[signal_mask.fillna(False), "ocean_proxy_l"] = True
    else:
        features.loc[signal_mask.fillna(False), "ocean_proxy_s"] = True
    features.loc[signal_mask.fillna(False), "ocean_proxy_signal"] = signal
    return features


def _signal_count(features: pd.DataFrame, signal: str) -> int:
    if "ocean_proxy_signal" not in features:
        return 0
    return int((features["ocean_proxy_signal"] == signal).sum())


def _split_signal_counts(
    features: pd.DataFrame,
    *,
    signal: str,
    start: date,
    end_exclusive: date,
    train_ratio: float,
) -> dict[str, int]:
    start_ts = pd.Timestamp(datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc))
    end_ts = pd.Timestamp(datetime.combine(end_exclusive, datetime.min.time(), tzinfo=timezone.utc))
    split_ts = start_ts + (end_ts - start_ts) * train_ratio
    train_features = _slice_by_time(features, start_ts, split_ts)
    test_features = _slice_by_time(features, split_ts, end_ts)
    return {
        "full_signals": _signal_count(features, signal),
        "train_signals": _signal_count(train_features, signal),
        "test_signals": _signal_count(test_features, signal),
    }


def _signal_side(signal: str) -> str:
    return "long" if signal in {"L", "XL"} else "short"


def _simulate_signal_trades(
    features: pd.DataFrame,
    *,
    signal: str,
    params: OceanProxyParams | TradingViewConvergenceParams,
) -> list[dict[str, Any]]:
    signal_mask = features["ocean_proxy_signal"] == signal
    trades: list[dict[str, Any]] = []
    if not signal_mask.any():
        return trades
    side = _signal_side(signal)
    direction = 1 if side == "long" else -1
    trade_cost_pct = ((params.fee_bps * 2.0) + (params.slippage_bps * 2.0)) / 100.0
    signal_indexes = np.flatnonzero(signal_mask.to_numpy(dtype=bool, na_value=False))
    open_values = pd.to_numeric(features["open"], errors="coerce").to_numpy(dtype=float)
    high_values = pd.to_numeric(features["high"], errors="coerce").to_numpy(dtype=float)
    low_values = pd.to_numeric(features["low"], errors="coerce").to_numpy(dtype=float)
    close_values = pd.to_numeric(features["close"], errors="coerce").to_numpy(dtype=float)
    timestamps = features.index
    last_exit_idx = -1
    for signal_idx in signal_indexes:
        entry_idx = int(signal_idx) + 1
        if entry_idx >= len(features) or entry_idx <= last_exit_idx:
            continue
        entry_price = float(open_values[entry_idx])
        if not math.isfinite(entry_price) or entry_price <= 0.0:
            continue
        stop_price = entry_price * (1.0 - params.stop_loss_pct / 100.0) if direction > 0 else entry_price * (
            1.0 + params.stop_loss_pct / 100.0
        )
        tp_price = entry_price * (1.0 + params.take_profit_pct / 100.0) if direction > 0 else entry_price * (
            1.0 - params.take_profit_pct / 100.0
        )
        max_exit_idx = min(len(features) - 1, entry_idx + params.max_hold_bars)
        exit_idx = max_exit_idx
        exit_price = float(close_values[max_exit_idx])
        if not math.isfinite(exit_price) or exit_price <= 0.0:
            exit_price = entry_price
        exit_reason = "max_hold"
        for idx in range(entry_idx, max_exit_idx + 1):
            high = float(high_values[idx])
            low = float(low_values[idx])
            if not math.isfinite(high) or not math.isfinite(low):
                continue
            if direction > 0:
                stop_hit = low <= stop_price
                tp_hit = high >= tp_price
            else:
                stop_hit = high >= stop_price
                tp_hit = low <= tp_price
            if stop_hit and tp_hit:
                exit_idx = idx
                exit_price = stop_price
                exit_reason = "stop_priority_same_bar"
                break
            if stop_hit:
                exit_idx = idx
                exit_price = stop_price
                exit_reason = "stop_loss"
                break
            if tp_hit:
                exit_idx = idx
                exit_price = tp_price
                exit_reason = "take_profit"
                break
        raw_return = ((exit_price / entry_price) - 1.0) * 100.0
        pnl_pct = (raw_return * direction) - trade_cost_pct
        trades.append(
            {
                "signal_time": timestamps[int(signal_idx)].isoformat(),
                "entry_time": timestamps[entry_idx].isoformat(),
                "exit_time": timestamps[exit_idx].isoformat(),
                "side": side,
                "signal": signal,
                "entry_price": round(entry_price, 8),
                "exit_price": round(exit_price, 8),
                "pnl_pct": round(pnl_pct, 4),
                "raw_return_pct": round(raw_return * direction, 4),
                "exit_reason": exit_reason,
                "bars_held": int(exit_idx - entry_idx + 1),
            }
        )
        last_exit_idx = exit_idx
    return trades


def _trade_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [_safe_float(trade.get("pnl_pct")) for trade in trades]
    wins = sum(1 for pnl in pnls if pnl > 0.0)
    losses = len(pnls) - wins
    gross_profit = sum(pnl for pnl in pnls if pnl > 0.0)
    gross_loss = abs(sum(pnl for pnl in pnls if pnl <= 0.0))
    avg_win = gross_profit / wins if wins else 0.0
    avg_loss = gross_loss / losses if losses else 0.0
    win_rate_decimal = (wins / len(pnls)) if pnls else 0.0
    expectancy_pct = (win_rate_decimal * avg_win) - ((1.0 - win_rate_decimal) * avg_loss)
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for pnl in pnls:
        equity *= 1.0 + pnl / 100.0
        peak = max(peak, equity)
        if peak:
            max_dd = max(max_dd, (peak - equity) / peak)
    stop_loss_count = sum(1 for trade in trades if str(trade.get("exit_reason")) in {"stop_loss", "stop_priority_same_bar"})
    return {
        "trade_count": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round((wins / len(pnls)) * 100.0, 2) if pnls else 0.0,
        "expectancy_pct": round(expectancy_pct, 4) if pnls else 0.0,
        "avg_win_pct": round(avg_win, 4),
        "avg_loss_pct": round(avg_loss, 4),
        "payoff_ratio": round(_safe_ratio(avg_win, avg_loss), 4) if pnls else 0.0,
        "avg_pnl_pct": round(sum(pnls) / len(pnls), 4) if pnls else 0.0,
        "median_pnl_pct": round(float(pd.Series(pnls, dtype="float64").median()), 4) if pnls else 0.0,
        "total_return_pct": round((equity - 1.0) * 100.0, 4),
        "max_drawdown_pct": round(max_dd * 100.0, 4),
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss else ("inf" if gross_profit else 0.0),
        "stop_loss_ratio": round((stop_loss_count / len(trades)) * 100.0, 2) if trades else 0.0,
        "max_loss_streak": _max_loss_streak(pnls),
        "sharpe_like": round(_sample_sharpe(pnls), 4),
    }


def _slice_by_time(features: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return features[(features.index >= start) & (features.index < end)].copy()


def _evaluate_candidate(
    features: pd.DataFrame,
    *,
    symbol: str,
    interval: str,
    signal: str,
    params: OceanProxyParams,
    start: date,
    end_exclusive: date,
    train_ratio: float,
    walk_forward_windows: int,
    min_train_trades: int,
    min_test_trades: int,
    target_win_rate: float,
    min_profit_factor: float,
    min_stop_loss_ratio: float | None = None,
) -> dict[str, Any]:
    start_ts = pd.Timestamp(datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc))
    end_ts = pd.Timestamp(datetime.combine(end_exclusive, datetime.min.time(), tzinfo=timezone.utc))
    split_ts = start_ts + (end_ts - start_ts) * train_ratio
    train_features = _slice_by_time(features, start_ts, split_ts)
    test_features = _slice_by_time(features, split_ts, end_ts)
    full_trades = _simulate_signal_trades(features, signal=signal, params=params)
    train_trades = _simulate_signal_trades(train_features, signal=signal, params=params)
    test_trades = _simulate_signal_trades(test_features, signal=signal, params=params)
    full = _trade_summary(full_trades)
    train = _trade_summary(train_trades)
    test = _trade_summary(test_trades)

    window_results: list[dict[str, Any]] = []
    window_seconds = (end_ts - start_ts).total_seconds() / max(walk_forward_windows, 1)
    for window_idx in range(walk_forward_windows):
        window_start = start_ts + pd.Timedelta(seconds=window_seconds * window_idx)
        window_end = end_ts if window_idx == walk_forward_windows - 1 else start_ts + pd.Timedelta(
            seconds=window_seconds * (window_idx + 1)
        )
        window_features = _slice_by_time(features, window_start, window_end)
        window_trades = _simulate_signal_trades(window_features, signal=signal, params=params)
        window_summary = _trade_summary(window_trades)
        window_results.append(
            {
                "window": window_idx + 1,
                "start": window_start.isoformat(),
                "end": window_end.isoformat(),
                **window_summary,
            }
        )
    positive_windows = sum(1 for item in window_results if _safe_float(item.get("total_return_pct")) > 0.0)
    windows_with_sample = [item for item in window_results if int(item.get("trade_count") or 0) > 0]
    min_window_win_rate = min((_safe_float(item.get("win_rate")) for item in windows_with_sample), default=0.0)
    min_window_profit_factor = min(
        (
            _safe_float(item.get("profit_factor"), 9999.0 if item.get("profit_factor") == "inf" else 0.0)
            for item in windows_with_sample
        ),
        default=0.0,
    )
    passed = (
        train["trade_count"] >= min_train_trades
        and test["trade_count"] >= min_test_trades
        and train["win_rate"] >= target_win_rate
        and test["win_rate"] >= target_win_rate
        and _safe_float(train["profit_factor"], 9999.0 if train["profit_factor"] == "inf" else 0.0) >= min_profit_factor
        and _safe_float(test["profit_factor"], 9999.0 if test["profit_factor"] == "inf" else 0.0) >= min_profit_factor
        and positive_windows >= math.ceil(walk_forward_windows * 0.67)
        and min_window_win_rate >= max(55.0, target_win_rate - 15.0)
        and (min_stop_loss_ratio is None or test["stop_loss_ratio"] <= min_stop_loss_ratio)
    )
    blockers: list[str] = []
    if train["trade_count"] < min_train_trades:
        blockers.append("train-trade-count-below-floor")
    if test["trade_count"] < min_test_trades:
        blockers.append("test-trade-count-below-floor")
    if train["win_rate"] < target_win_rate:
        blockers.append("train-win-rate-below-target")
    if test["win_rate"] < target_win_rate:
        blockers.append("test-win-rate-below-target")
    if _safe_float(train["profit_factor"], 9999.0 if train["profit_factor"] == "inf" else 0.0) < min_profit_factor:
        blockers.append("train-profit-factor-below-floor")
    if _safe_float(test["profit_factor"], 9999.0 if test["profit_factor"] == "inf" else 0.0) < min_profit_factor:
        blockers.append("test-profit-factor-below-floor")
    if positive_windows < math.ceil(walk_forward_windows * 0.67):
        blockers.append("walk-forward-positive-window-count-too-low")
    if min_window_win_rate < max(55.0, target_win_rate - 15.0):
        blockers.append("walk-forward-min-win-rate-too-low")
    if min_stop_loss_ratio is not None and test["stop_loss_ratio"] > min_stop_loss_ratio:
        blockers.append("test-stop-loss-ratio-above-floor")
    return {
        "symbol": symbol,
        "interval": interval,
        "signal": signal,
        "side": _signal_side(signal),
        "params": asdict(params),
        "param_key": params.key(),
        "train": train,
        "test": test,
        "full": full,
        "walk_forward": {
            "window_count": walk_forward_windows,
            "positive_windows": positive_windows,
            "sampled_windows": len(windows_with_sample),
            "mean_window_win_rate": round(mean(item["win_rate"] for item in windows_with_sample), 2)
            if windows_with_sample
            else 0.0,
            "min_window_win_rate": round(min_window_win_rate, 2),
            "min_window_profit_factor": round(min_window_profit_factor, 4),
            "windows": window_results,
        },
        "gate": {
            "passed": passed,
            "blockers": blockers,
            "target_win_rate": target_win_rate,
            "min_train_trades": min_train_trades,
            "min_test_trades": min_test_trades,
            "min_profit_factor": min_profit_factor,
            "max_stop_loss_ratio": min_stop_loss_ratio,
        },
        "sample_trades": full_trades[:20],
    }


def _filter_trades_by_signal_time(
    trades: list[dict[str, Any]],
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for trade in trades:
        try:
            signal_ts = pd.Timestamp(str(trade.get("signal_time")))
        except ValueError:
            continue
        if start_ts <= signal_ts < end_ts:
            selected.append(trade)
    return selected


def _gate_thresholds(
    *,
    gate_mode: str,
    target_win_rate: float,
    min_train_trades: int,
    min_test_trades: int,
    min_profit_factor: float,
    min_stop_loss_ratio: float | None,
    min_expectancy_pct: float,
    min_payoff_ratio: float,
    max_drawdown_pct: float,
    max_loss_streak: int,
) -> dict[str, Any]:
    return {
        "gate_mode": gate_mode,
        "target_win_rate": target_win_rate,
        "min_train_trades": min_train_trades,
        "min_test_trades": min_test_trades,
        "min_profit_factor": min_profit_factor,
        "max_stop_loss_ratio": min_stop_loss_ratio,
        "min_expectancy_pct": min_expectancy_pct,
        "min_payoff_ratio": min_payoff_ratio,
        "max_drawdown_pct": max_drawdown_pct,
        "max_loss_streak": max_loss_streak,
    }


def _evaluate_gate(
    *,
    train: dict[str, Any],
    test: dict[str, Any],
    full: dict[str, Any],
    window_results: list[dict[str, Any]],
    walk_forward_windows: int,
    gate_mode: str,
    target_win_rate: float,
    min_train_trades: int,
    min_test_trades: int,
    min_profit_factor: float,
    min_stop_loss_ratio: float | None,
    min_expectancy_pct: float,
    min_payoff_ratio: float,
    max_drawdown_pct: float,
    max_loss_streak: int,
) -> tuple[bool, list[str], dict[str, Any]]:
    mode = _normalize_gate_mode(gate_mode)
    positive_windows = sum(1 for item in window_results if _safe_float(item.get("total_return_pct")) > 0.0)
    windows_with_sample = [item for item in window_results if int(item.get("trade_count") or 0) > 0]
    min_window_win_rate = min((_safe_float(item.get("win_rate")) for item in windows_with_sample), default=0.0)
    min_window_profit_factor = min(
        (
            _safe_float(item.get("profit_factor"), 9999.0 if item.get("profit_factor") == "inf" else 0.0)
            for item in windows_with_sample
        ),
        default=0.0,
    )
    min_window_expectancy = min(
        (_safe_float(item.get("expectancy_pct")) for item in windows_with_sample),
        default=0.0,
    )
    positive_expectancy_windows = sum(1 for item in windows_with_sample if _safe_float(item.get("expectancy_pct")) > 0.0)
    full_profit_factor = _profit_factor_value(full.get("profit_factor"))
    train_profit_factor = _profit_factor_value(train.get("profit_factor"))
    test_profit_factor = _profit_factor_value(test.get("profit_factor"))

    blockers: list[str] = []
    if train["trade_count"] < min_train_trades:
        blockers.append("train-trade-count-below-floor")
    if test["trade_count"] < min_test_trades:
        blockers.append("test-trade-count-below-floor")
    if train_profit_factor < min_profit_factor:
        blockers.append("train-profit-factor-below-floor")
    if test_profit_factor < min_profit_factor:
        blockers.append("test-profit-factor-below-floor")
    if full_profit_factor < min_profit_factor:
        blockers.append("full-profit-factor-below-floor")
    if positive_windows < math.ceil(walk_forward_windows * 0.67):
        blockers.append("walk-forward-positive-window-count-too-low")
    if min_stop_loss_ratio is not None and test["stop_loss_ratio"] > min_stop_loss_ratio:
        blockers.append("test-stop-loss-ratio-above-floor")

    if mode == "strict_win_rate":
        if train["win_rate"] < target_win_rate:
            blockers.append("train-win-rate-below-target")
        if test["win_rate"] < target_win_rate:
            blockers.append("test-win-rate-below-target")
        if min_window_win_rate < max(55.0, target_win_rate - 15.0):
            blockers.append("walk-forward-min-win-rate-too-low")
    else:
        if train["expectancy_pct"] < min_expectancy_pct:
            blockers.append("train-expectancy-below-floor")
        if test["expectancy_pct"] < min_expectancy_pct:
            blockers.append("test-expectancy-below-floor")
        if full["expectancy_pct"] < min_expectancy_pct:
            blockers.append("full-expectancy-below-floor")
        if test["payoff_ratio"] < min_payoff_ratio:
            blockers.append("test-payoff-ratio-below-floor")
        if full["payoff_ratio"] < min_payoff_ratio:
            blockers.append("full-payoff-ratio-below-floor")
        if test["max_drawdown_pct"] > max_drawdown_pct:
            blockers.append("test-drawdown-above-ceiling")
        if full["max_drawdown_pct"] > max_drawdown_pct:
            blockers.append("full-drawdown-above-ceiling")
        if test["max_loss_streak"] > max_loss_streak:
            blockers.append("test-loss-streak-above-ceiling")
        if full["max_loss_streak"] > max_loss_streak:
            blockers.append("full-loss-streak-above-ceiling")
        if positive_expectancy_windows < math.ceil(walk_forward_windows * 0.67):
            blockers.append("walk-forward-positive-expectancy-window-count-too-low")
        if min_window_expectancy < 0.0:
            blockers.append("walk-forward-min-expectancy-negative")

    thresholds = _gate_thresholds(
        gate_mode=mode,
        target_win_rate=target_win_rate,
        min_train_trades=min_train_trades,
        min_test_trades=min_test_trades,
        min_profit_factor=min_profit_factor,
        min_stop_loss_ratio=min_stop_loss_ratio,
        min_expectancy_pct=min_expectancy_pct,
        min_payoff_ratio=min_payoff_ratio,
        max_drawdown_pct=max_drawdown_pct,
        max_loss_streak=max_loss_streak,
    )
    diagnostics = {
        "positive_windows": positive_windows,
        "positive_expectancy_windows": positive_expectancy_windows,
        "windows_with_sample": len(windows_with_sample),
        "min_window_win_rate": round(min_window_win_rate, 2),
        "min_window_profit_factor": round(min_window_profit_factor, 4),
        "min_window_expectancy_pct": round(min_window_expectancy, 4),
        "thresholds": thresholds,
    }
    return not blockers, blockers, diagnostics


def _evaluate_candidate_from_full_trades(
    features: pd.DataFrame,
    *,
    symbol: str,
    interval: str,
    signal: str,
    params: OceanProxyParams | TradingViewConvergenceParams,
    start: date,
    end_exclusive: date,
    train_ratio: float,
    walk_forward_windows: int,
    min_train_trades: int,
    min_test_trades: int,
    target_win_rate: float,
    min_profit_factor: float,
    min_stop_loss_ratio: float | None = None,
    gate_mode: str = "strict_win_rate",
    min_expectancy_pct: float = 0.05,
    min_payoff_ratio: float = 1.2,
    max_drawdown_pct: float = 20.0,
    max_loss_streak: int = 8,
) -> dict[str, Any]:
    start_ts = pd.Timestamp(datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc))
    end_ts = pd.Timestamp(datetime.combine(end_exclusive, datetime.min.time(), tzinfo=timezone.utc))
    split_ts = start_ts + (end_ts - start_ts) * train_ratio
    full_trades = _simulate_signal_trades(features, signal=signal, params=params)
    train_trades = _filter_trades_by_signal_time(full_trades, start_ts=start_ts, end_ts=split_ts)
    test_trades = _filter_trades_by_signal_time(full_trades, start_ts=split_ts, end_ts=end_ts)
    full = _trade_summary(full_trades)
    train = _trade_summary(train_trades)
    test = _trade_summary(test_trades)

    window_results: list[dict[str, Any]] = []
    window_seconds = (end_ts - start_ts).total_seconds() / max(walk_forward_windows, 1)
    for window_idx in range(walk_forward_windows):
        window_start = start_ts + pd.Timedelta(seconds=window_seconds * window_idx)
        window_end = end_ts if window_idx == walk_forward_windows - 1 else start_ts + pd.Timedelta(
            seconds=window_seconds * (window_idx + 1)
        )
        window_trades = _filter_trades_by_signal_time(full_trades, start_ts=window_start, end_ts=window_end)
        window_summary = _trade_summary(window_trades)
        window_results.append(
            {
                "window": window_idx + 1,
                "start": window_start.isoformat(),
                "end": window_end.isoformat(),
                **window_summary,
            }
        )
    windows_with_sample = [item for item in window_results if int(item.get("trade_count") or 0) > 0]
    passed, blockers, gate_diagnostics = _evaluate_gate(
        train=train,
        test=test,
        full=full,
        window_results=window_results,
        walk_forward_windows=walk_forward_windows,
        gate_mode=gate_mode,
        target_win_rate=target_win_rate,
        min_train_trades=min_train_trades,
        min_test_trades=min_test_trades,
        min_profit_factor=min_profit_factor,
        min_stop_loss_ratio=min_stop_loss_ratio,
        min_expectancy_pct=min_expectancy_pct,
        min_payoff_ratio=min_payoff_ratio,
        max_drawdown_pct=max_drawdown_pct,
        max_loss_streak=max_loss_streak,
    )
    return {
        "symbol": symbol,
        "interval": interval,
        "signal": signal,
        "side": _signal_side(signal),
        "params": asdict(params),
        "param_key": params.key(),
        "train": train,
        "test": test,
        "full": full,
        "walk_forward": {
            "window_count": walk_forward_windows,
            "positive_windows": gate_diagnostics["positive_windows"],
            "positive_expectancy_windows": gate_diagnostics["positive_expectancy_windows"],
            "sampled_windows": len(windows_with_sample),
            "mean_window_win_rate": round(mean(item["win_rate"] for item in windows_with_sample), 2)
            if windows_with_sample
            else 0.0,
            "min_window_win_rate": gate_diagnostics["min_window_win_rate"],
            "min_window_profit_factor": gate_diagnostics["min_window_profit_factor"],
            "min_window_expectancy_pct": gate_diagnostics["min_window_expectancy_pct"],
            "windows": window_results,
        },
        "gate": {
            "passed": passed,
            "blockers": blockers,
            **gate_diagnostics["thresholds"],
            "partitioned_from_full_simulation": True,
        },
        "sample_trades": full_trades[:20],
    }


def optimize_core_whale_jump_proxy(
    *,
    symbols: list[str],
    market: str,
    interval: str,
    start: date,
    end_exclusive: date,
    timeout: float,
    output_dir: Path | None,
    target_win_rate: float,
    min_train_trades: int,
    min_test_trades: int,
    min_profit_factor: float,
    max_configs: int,
    regime_filters: list[str] | None = None,
    max_stop_loss_ratio: float | None = None,
    min_dataset_bars: int = 1000,
    min_coverage_ratio: float = 0.65,
) -> dict[str, Any]:
    run_id = f"{_utc_stamp()}-core-whale-jump-optimizer"
    report_dir = output_dir or (PROJECT_ROOT / "reports" / run_id)
    report_dir.mkdir(parents=True, exist_ok=True)
    param_grid = _limit_param_grid(_default_param_grid(), max_configs)
    selected_regime_filters = _normalize_regime_filters(regime_filters)
    source_manifest: dict[str, Any] = {
        "run_id": run_id,
        "symbols": symbols,
        "market": market,
        "interval": interval,
        "start_date": start.isoformat(),
        "end_date": end_exclusive.isoformat(),
        "sources": [],
    }
    ranked: list[dict[str, Any]] = []
    dataset_summaries: dict[str, Any] = {}
    for symbol in symbols:
        frame, sources = fetch_history(
            market=market,
            symbol=symbol,
            interval=interval,
            start=start,
            end_exclusive=end_exclusive,
            timeout=timeout,
        )
        source_manifest["sources"].extend(asdict(source) | {"symbol": symbol, "interval": interval} for source in sources)
        dataset_quality = _dataset_quality(
            frame,
            requested_start=start,
            requested_end=end_exclusive,
            min_coverage_ratio=min_coverage_ratio,
            min_dataset_bars=min_dataset_bars,
        )
        dataset_summaries[symbol] = {
            **dataset_quality,
            "source_files": sum(1 for source in sources if source.status_code == 200),
            "checksum_ok_files": sum(1 for source in sources if source.checksum_ok is True),
            "missing_files": [source.filename for source in sources if source.status_code == 404],
        }
        base_features = build_proxy_base_features(frame, interval=interval)
        for params in param_grid:
            base_proxy_features = build_proxy_features_with_params(
                interval=interval,
                params=params,
                base_features=base_features,
            )
            for signal in ("L", "XL", "S", "XS"):
                if not bool((base_proxy_features["ocean_proxy_signal"] == signal).any()):
                    continue
                for regime_filter in selected_regime_filters:
                    candidate_params = replace(params, regime_filter=regime_filter)
                    features = apply_regime_filter(
                        base_proxy_features,
                        signal=signal,
                        regime_filter=regime_filter,
                    )
                    if not bool((features["ocean_proxy_signal"] == signal).any()):
                        continue
                    ranked.append(
                        _evaluate_candidate(
                            features,
                            symbol=symbol,
                            interval=interval,
                            signal=signal,
                            params=candidate_params,
                            start=start,
                            end_exclusive=end_exclusive,
                            train_ratio=0.70,
                            walk_forward_windows=4,
                            min_train_trades=min_train_trades,
                            min_test_trades=min_test_trades,
                            target_win_rate=target_win_rate,
                            min_profit_factor=min_profit_factor,
                            min_stop_loss_ratio=max_stop_loss_ratio,
                        )
                    )
    ranked.sort(
        key=lambda item: (
            bool((item.get("gate") or {}).get("passed")),
            _safe_float((item.get("test") or {}).get("win_rate")),
            _profit_factor_value((item.get("test") or {}).get("profit_factor")),
            _safe_float((item.get("walk_forward") or {}).get("mean_window_win_rate")),
            _safe_float((item.get("full") or {}).get("total_return_pct")),
            int((item.get("test") or {}).get("trade_count") or 0),
        ),
        reverse=True,
    )
    candidates = [item for item in ranked if (item.get("gate") or {}).get("passed")]
    sample_leaders = [
        item
        for item in ranked
        if int((item.get("train") or {}).get("trade_count") or 0) >= min_train_trades
        and int((item.get("test") or {}).get("trade_count") or 0) >= min_test_trades
    ]
    sample_leaders.sort(
        key=lambda item: (
            _safe_float((item.get("test") or {}).get("win_rate")),
            _safe_float((item.get("walk_forward") or {}).get("mean_window_win_rate")),
            _profit_factor_value((item.get("test") or {}).get("profit_factor")),
            _safe_float((item.get("full") or {}).get("total_return_pct")),
            -_safe_float((item.get("test") or {}).get("max_drawdown_pct")),
        ),
        reverse=True,
    )
    near_gate_candidates = [
        item
        for item in sample_leaders
        if _safe_float((item.get("test") or {}).get("win_rate")) >= max(55.0, target_win_rate - 20.0)
    ][:30]
    best_by_symbol: dict[str, Any] = {}
    for symbol in symbols:
        symbol_rows = [item for item in ranked if item.get("symbol") == symbol]
        best_by_symbol[symbol] = symbol_rows[0] if symbol_rows else None
    best_sample_by_symbol: dict[str, Any] = {}
    for symbol in symbols:
        symbol_rows = [item for item in sample_leaders if item.get("symbol") == symbol]
        best_sample_by_symbol[symbol] = symbol_rows[0] if symbol_rows else None
    mature_candidates = [
        item for item in candidates if (dataset_summaries.get(str(item.get("symbol"))) or {}).get("mature")
    ]
    manifest_path = report_dir / "source_manifest.json"
    manifest_path.write_text(json.dumps(source_manifest, indent=2, sort_keys=True), encoding="utf-8")
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "mode": "core_whale_jump_proxy_optimizer",
        "safety": {
            "mainnet_live_allowed": False,
            "writes_execution_config": False,
            "opens_orders": False,
            "research_gate_only": True,
        },
        "symbols": symbols,
        "market": market,
        "interval": interval,
        "start_date": start.isoformat(),
        "end_date": end_exclusive.isoformat(),
        "targets": {
            "target_win_rate": target_win_rate,
            "min_train_trades": min_train_trades,
            "min_test_trades": min_test_trades,
            "min_profit_factor": min_profit_factor,
            "max_stop_loss_ratio": max_stop_loss_ratio,
            "min_dataset_bars": min_dataset_bars,
            "min_coverage_ratio": min_coverage_ratio,
        },
        "grid": {
            "configs": len(param_grid),
            "regime_filters": selected_regime_filters,
            "evaluations": len(ranked),
        },
        "dataset_summaries": dataset_summaries,
        "candidate_count": len(candidates),
        "mature_candidate_count": len(mature_candidates),
        "promotion_allowed": False,
        "execution_recommendation": "continue_research_do_not_wire_live",
        "best_by_symbol": best_by_symbol,
        "best_sample_by_symbol": best_sample_by_symbol,
        "top_candidates": ranked[:30],
        "sample_leaders": sample_leaders[:50],
        "near_gate_candidates": near_gate_candidates,
        "next_iteration": _optimizer_next_iteration(ranked, target_win_rate=target_win_rate),
        "artifacts": {
            "report_dir": str(report_dir),
            "summary_json": str(report_dir / "optimizer_summary.json"),
            "source_manifest": str(manifest_path),
            "research_md": str(report_dir / "optimizer.md"),
        },
    }
    summary_path = report_dir / "optimizer_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_optimizer_markdown(report_dir / "optimizer.md", payload)
    return payload


def optimize_btc_eth_ocean_proxy(**kwargs: Any) -> dict[str, Any]:
    return optimize_core_whale_jump_proxy(**kwargs)


def _tradingview_sources() -> dict[str, str]:
    return {
        "tradingview_l5_whales_jump": TRADINGVIEW_L5_URL,
        "tradingview_l3_banker_fund_flow": TRADINGVIEW_L3_URL,
        "tradingview_supertrend": "https://www.tradingview.com/support/solutions/43000634738-supertrend/",
        "tradingview_stoch_rsi": "https://www.tradingview.com/support/solutions/43000502333-stochastic-rsi-stoch-rsi/",
        "binance_public_data": BINANCE_PUBLIC_DATA_URL,
        "binance_fapi_klines": BINANCE_FAPI_KLINES_URL,
    }


def _tradingview_family_notes() -> dict[str, str]:
    return {
        "tv_supertrend_macd": "Trend-following resonance: Supertrend/TrendMagic/FollowLine, EMA, DI, and MACD alignment.",
        "tv_stoch_rsi_pullback": "Trend-continuation pullback: StochRSI/RSI reset inside an EMA-aligned trend.",
        "tv_vwap_trend": "VWAP trend continuation: price/VWAP/EMA alignment with volume and taker-flow confirmation.",
        "tv_range_rsi": "Range mean-reversion: low-ADX Bollinger/StochRSI/RSI extremes with hostile-flow veto.",
        "banker_flow_proxy": "Open-source Banker Fund Flow proxy: MFI, taker-flow, volume, JUMBO delta, and structure.",
    }


def _tradingview_pre_screen_floor(min_trades: int) -> int:
    return max(1, math.floor(min_trades * 0.55))


def optimize_btc_eth_tradingview_convergence(
    *,
    symbols: list[str],
    market: str,
    interval: str,
    start: date,
    end_exclusive: date,
    timeout: float,
    output_dir: Path | None,
    target_win_rate: float,
    min_train_trades: int,
    min_test_trades: int,
    min_profit_factor: float,
    max_configs: int,
    regime_filters: list[str] | None = None,
    max_stop_loss_ratio: float | None = None,
    min_dataset_bars: int = 1000,
    min_coverage_ratio: float = 0.65,
    max_per_trade_risk_pct: float = 2.5,
    max_full_evaluations: int = 600,
    gate_mode: str = "strict_win_rate",
    min_expectancy_pct: float = 0.05,
    min_payoff_ratio: float = 1.2,
    max_drawdown_pct: float = 20.0,
    max_loss_streak: int = 8,
) -> dict[str, Any]:
    run_id = f"{_utc_stamp()}-btc-eth-tradingview-convergence"
    report_dir = output_dir or (PROJECT_ROOT / "reports" / run_id)
    report_dir.mkdir(parents=True, exist_ok=True)
    clean_symbols = [symbol.upper() for symbol in symbols if symbol.upper() in BTC_ETH_TRADINGVIEW_SYMBOLS]
    if not clean_symbols:
        clean_symbols = list(BTC_ETH_TRADINGVIEW_SYMBOLS)
    param_grid = _limit_tradingview_param_grid(
        _default_tradingview_param_grid(max_per_trade_risk_pct=max_per_trade_risk_pct),
        max_configs,
    )
    selected_regime_filters = _normalize_regime_filters(regime_filters)
    selected_gate_mode = _normalize_gate_mode(gate_mode)
    source_manifest: dict[str, Any] = {
        "run_id": run_id,
        "symbols": clean_symbols,
        "market": market,
        "interval": interval,
        "start_date": start.isoformat(),
        "end_date": end_exclusive.isoformat(),
        "concept_sources": _tradingview_sources(),
        "sources": [],
    }
    ranked: list[dict[str, Any]] = []
    pre_screen_rows: list[dict[str, Any]] = []
    dataset_summaries: dict[str, Any] = {}
    train_signal_floor = _tradingview_pre_screen_floor(min_train_trades)
    test_signal_floor = _tradingview_pre_screen_floor(min_test_trades)
    for symbol in clean_symbols:
        frame, sources = fetch_history(
            market=market,
            symbol=symbol,
            interval=interval,
            start=start,
            end_exclusive=end_exclusive,
            timeout=timeout,
        )
        source_manifest["sources"].extend(asdict(source) | {"symbol": symbol, "interval": interval} for source in sources)
        dataset_quality = _dataset_quality(
            frame,
            requested_start=start,
            requested_end=end_exclusive,
            min_coverage_ratio=min_coverage_ratio,
            min_dataset_bars=min_dataset_bars,
        )
        dataset_summaries[symbol] = {
            **dataset_quality,
            "source_files": sum(1 for source in sources if source.status_code == 200),
            "checksum_ok_files": sum(1 for source in sources if source.checksum_ok is True),
            "missing_files": [source.filename for source in sources if source.status_code == 404],
        }
        base_features = build_proxy_base_features(frame, interval=interval)
        symbol_shortlist: list[dict[str, Any]] = []
        for params in param_grid:
            signal = "L" if params.side == "long" else "S"
            base_signal_features = build_tradingview_signal_features(base_features, params=params)
            for regime_filter in selected_regime_filters:
                candidate_params = replace(params, regime_filter=regime_filter)
                features = apply_regime_filter(
                    base_signal_features,
                    signal=signal,
                    regime_filter=regime_filter,
                )
                counts = _split_signal_counts(
                    features,
                    signal=signal,
                    start=start,
                    end_exclusive=end_exclusive,
                    train_ratio=0.70,
                )
                pre_screen = {
                    "symbol": symbol,
                    "interval": interval,
                    "signal": signal,
                    "side": params.side,
                    "family": params.family,
                    "param_key": candidate_params.key(),
                    "params": asdict(candidate_params),
                    **counts,
                    "pre_screen_passed": counts["train_signals"] >= train_signal_floor
                    and counts["test_signals"] >= test_signal_floor,
                }
                pre_screen_rows.append(pre_screen)
                if pre_screen["pre_screen_passed"]:
                    symbol_shortlist.append(
                        {
                            "sort_key": (
                                counts["test_signals"],
                                counts["train_signals"],
                                counts["full_signals"],
                                -candidate_params.stop_loss_pct,
                                candidate_params.take_profit_pct,
                            ),
                            "signal": signal,
                            "params": candidate_params,
                            "pre_screen": pre_screen,
                        }
                    )
        symbol_shortlist.sort(key=lambda item: item["sort_key"], reverse=True)
        if max_full_evaluations > 0:
            symbol_budget = max(1, math.ceil(max_full_evaluations / len(clean_symbols)))
            symbol_shortlist = symbol_shortlist[:symbol_budget]
        for item in symbol_shortlist:
            base_signal_features = build_tradingview_signal_features(base_features, params=item["params"])
            features = apply_regime_filter(
                base_signal_features,
                signal=item["signal"],
                regime_filter=item["params"].regime_filter,
            )
            ranked.append(
                _evaluate_candidate_from_full_trades(
                    features,
                    symbol=symbol,
                    interval=interval,
                    signal=item["signal"],
                    params=item["params"],
                    start=start,
                    end_exclusive=end_exclusive,
                    train_ratio=0.70,
                    walk_forward_windows=4,
                    min_train_trades=min_train_trades,
                    min_test_trades=min_test_trades,
                    target_win_rate=target_win_rate,
                    min_profit_factor=min_profit_factor,
                    min_stop_loss_ratio=max_stop_loss_ratio,
                    gate_mode=selected_gate_mode,
                    min_expectancy_pct=min_expectancy_pct,
                    min_payoff_ratio=min_payoff_ratio,
                    max_drawdown_pct=max_drawdown_pct,
                    max_loss_streak=max_loss_streak,
                )
                | {
                    "tradingview_family": item["params"].family,
                    "pre_screen": item["pre_screen"],
                }
            )
    ranked.sort(
        key=lambda item: (
            bool((item.get("gate") or {}).get("passed")),
            _safe_float((item.get("test") or {}).get("expectancy_pct")),
            _profit_factor_value((item.get("test") or {}).get("profit_factor")),
            _safe_float((item.get("full") or {}).get("expectancy_pct")),
            _safe_float((item.get("walk_forward") or {}).get("min_window_expectancy_pct")),
            _safe_float((item.get("test") or {}).get("payoff_ratio")),
            -_safe_float((item.get("test") or {}).get("stop_loss_ratio")),
            -_safe_float((item.get("test") or {}).get("max_drawdown_pct")),
            int((item.get("test") or {}).get("trade_count") or 0),
        ),
        reverse=True,
    )
    candidates = [item for item in ranked if (item.get("gate") or {}).get("passed")]
    mature_candidates = [
        item for item in candidates if (dataset_summaries.get(str(item.get("symbol"))) or {}).get("mature")
    ]
    sample_leaders = [
        item
        for item in ranked
        if int((item.get("train") or {}).get("trade_count") or 0) >= min_train_trades
        and int((item.get("test") or {}).get("trade_count") or 0) >= min_test_trades
    ]
    sample_leaders.sort(
        key=lambda item: (
            _safe_float((item.get("test") or {}).get("expectancy_pct")),
            _profit_factor_value((item.get("test") or {}).get("profit_factor")),
            _safe_float((item.get("full") or {}).get("expectancy_pct")),
            _safe_float((item.get("walk_forward") or {}).get("min_window_expectancy_pct")),
            _safe_float((item.get("test") or {}).get("payoff_ratio")),
            -_safe_float((item.get("test") or {}).get("stop_loss_ratio")),
            -_safe_float((item.get("test") or {}).get("max_drawdown_pct")),
        ),
        reverse=True,
    )
    best_by_symbol: dict[str, Any] = {}
    best_sample_by_symbol: dict[str, Any] = {}
    for symbol in clean_symbols:
        symbol_rows = [item for item in ranked if item.get("symbol") == symbol]
        best_by_symbol[symbol] = symbol_rows[0] if symbol_rows else None
        symbol_sample_rows = [item for item in sample_leaders if item.get("symbol") == symbol]
        best_sample_by_symbol[symbol] = symbol_sample_rows[0] if symbol_sample_rows else None
    pre_screen_passed = [item for item in pre_screen_rows if item["pre_screen_passed"]]
    pre_screen_rows.sort(
        key=lambda item: (
            bool(item.get("pre_screen_passed")),
            int(item.get("test_signals") or 0),
            int(item.get("train_signals") or 0),
            int(item.get("full_signals") or 0),
        ),
        reverse=True,
    )
    manifest_path = report_dir / "source_manifest.json"
    manifest_path.write_text(json.dumps(source_manifest, indent=2, sort_keys=True), encoding="utf-8")
    pre_screen_path = report_dir / "pre_screen_summary.json"
    pre_screen_path.write_text(json.dumps(pre_screen_rows[:500], indent=2, sort_keys=True), encoding="utf-8")
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "mode": "btc_eth_tradingview_convergence_optimizer",
        "safety": {
            "mainnet_live_allowed": False,
            "writes_execution_config": False,
            "opens_orders": False,
            "research_gate_only": True,
            "max_per_trade_risk_pct": max_per_trade_risk_pct,
        },
        "symbols": clean_symbols,
        "market": market,
        "interval": interval,
        "start_date": start.isoformat(),
        "end_date": end_exclusive.isoformat(),
        "targets": {
            "target_win_rate": target_win_rate,
            "min_train_trades": min_train_trades,
            "min_test_trades": min_test_trades,
            "min_profit_factor": min_profit_factor,
            "max_stop_loss_ratio": max_stop_loss_ratio,
            "max_per_trade_risk_pct": max_per_trade_risk_pct,
            "gate_mode": selected_gate_mode,
            "min_expectancy_pct": min_expectancy_pct,
            "min_payoff_ratio": min_payoff_ratio,
            "max_drawdown_pct": max_drawdown_pct,
            "max_loss_streak": max_loss_streak,
            "min_dataset_bars": min_dataset_bars,
            "min_coverage_ratio": min_coverage_ratio,
        },
        "grid": {
            "configs": len(param_grid),
            "signal_families": list(TRADINGVIEW_SIGNAL_FAMILIES),
            "regime_filters": selected_regime_filters,
            "pre_screen_rows": len(pre_screen_rows),
            "pre_screen_passed": len(pre_screen_passed),
            "evaluations": len(ranked),
            "max_full_evaluations": max_full_evaluations,
            "train_signal_floor": train_signal_floor,
            "test_signal_floor": test_signal_floor,
        },
        "concept_sources": _tradingview_sources(),
        "family_notes": _tradingview_family_notes(),
        "dataset_summaries": dataset_summaries,
        "candidate_count": len(candidates),
        "mature_candidate_count": len(mature_candidates),
        "promotion_allowed": False,
        "execution_recommendation": "continue_research_do_not_wire_live",
        "best_by_symbol": best_by_symbol,
        "best_sample_by_symbol": best_sample_by_symbol,
        "top_candidates": ranked[:30],
        "sample_leaders": sample_leaders[:50],
        "near_gate_candidates": [
            item
            for item in sample_leaders
            if _safe_float((item.get("test") or {}).get("expectancy_pct")) >= min_expectancy_pct
        ][:30],
        "pre_screen_leaders": pre_screen_rows[:80],
        "next_iteration": _tradingview_next_iteration(
            ranked,
            pre_screen_rows,
            target_win_rate=target_win_rate,
            gate_mode=selected_gate_mode,
        ),
        "artifacts": {
            "report_dir": str(report_dir),
            "summary_json": str(report_dir / "optimizer_summary.json"),
            "source_manifest": str(manifest_path),
            "pre_screen_summary": str(pre_screen_path),
            "research_md": str(report_dir / "optimizer.md"),
        },
    }
    summary_path = report_dir / "optimizer_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_tradingview_optimizer_markdown(report_dir / "optimizer.md", payload)
    return payload


def _optimizer_next_iteration(ranked: list[dict[str, Any]], *, target_win_rate: float) -> dict[str, Any]:
    best = ranked[0] if ranked else {}
    blockers = ((best.get("gate") or {}).get("blockers") or []) if isinstance(best, dict) else []
    suggestions: list[str] = []
    if "train-trade-count-below-floor" in blockers:
        suggestions.append("Expand public-history coverage or lower selectivity only after train sample is adequate.")
    if "test-trade-count-below-floor" in blockers:
        suggestions.append("Lower only one selectivity control at a time or expand history before trusting win rate.")
    if "test-win-rate-below-target" in blockers:
        suggestions.append("Increase taker-share, ADX, structure, or jumbo-delta filters; do not widen stops first.")
    if "walk-forward-min-win-rate-too-low" in blockers:
        suggestions.append("Prefer regimes with stable window performance over the highest single test win rate.")
    if "test-stop-loss-ratio-above-floor" in blockers:
        suggestions.append("Tighten regime filters before changing stop distance; stop-loss drag is still too high.")
    if not suggestions:
        suggestions.append("Run the same grid on a newer data cut and then a fee/slippage stress grid.")
    return {
        "target_win_rate": target_win_rate,
        "best_gap_win_rate_points": round(target_win_rate - _safe_float((best.get("test") or {}).get("win_rate")), 4)
        if isinstance(best, dict)
        else target_win_rate,
        "suggestions": suggestions,
    }


def _tradingview_next_iteration(
    ranked: list[dict[str, Any]],
    pre_screen_rows: list[dict[str, Any]],
    *,
    target_win_rate: float,
    gate_mode: str = "strict_win_rate",
) -> dict[str, Any]:
    best = ranked[0] if ranked else {}
    blockers = ((best.get("gate") or {}).get("blockers") or []) if isinstance(best, dict) else []
    suggestions: list[str] = []
    if not ranked:
        suggestions.append("Pre-screen produced no full-gate rows; widen one signal family at a time before TP/SL tuning.")
    if "train-trade-count-below-floor" in blockers or "test-trade-count-below-floor" in blockers:
        suggestions.append("Increase sample by using 15m or 30m public data in a background run; keep the same BTC/ETH-only gate.")
    if "test-win-rate-below-target" in blockers and gate_mode == "strict_win_rate":
        suggestions.append("Switch to expectancy gate or tighten family-specific flow before treating win rate as decisive.")
    if "test-expectancy-below-floor" in blockers or "full-expectancy-below-floor" in blockers:
        suggestions.append("Improve net expectancy before any paper/testnet promotion; do not rescue a negative-EV signal with more filters.")
    if "test-payoff-ratio-below-floor" in blockers or "full-payoff-ratio-below-floor" in blockers:
        suggestions.append("Prefer 1:2 / 1:3 exit profiles and stop accepting small-profit/high-win rows as trade-ready.")
    if "test-profit-factor-below-floor" in blockers:
        suggestions.append("Reject tiny take-profit profiles when profit factor collapses; prefer fewer but positive-expectancy trades.")
    if "test-stop-loss-ratio-above-floor" in blockers:
        suggestions.append("Reduce stop-loss frequency with stronger trend/liquidity confirmation; do not widen risk above 2.5%.")
    if "walk-forward-min-expectancy-negative" in blockers or "walk-forward-positive-expectancy-window-count-too-low" in blockers:
        suggestions.append("Favor the most stable positive-expectancy family across walk-forward windows, not the best single test slice.")
    elif "walk-forward-min-win-rate-too-low" in blockers:
        suggestions.append("Favor stable walk-forward performance instead of the highest one-window test win rate.")
    family_counts: dict[str, int] = {}
    for row in pre_screen_rows:
        if not row.get("pre_screen_passed"):
            continue
        family = str(row.get("family") or "")
        family_counts[family] = family_counts.get(family, 0) + 1
    if family_counts:
        best_family = max(family_counts.items(), key=lambda item: item[1])[0]
        suggestions.append(f"Continue with `{best_family}` first; it produced the broadest pre-screen sample.")
    if not suggestions:
        suggestions.append("Run the same BTC/ETH lane on 15m with a capped background budget and then stress fees/slippage.")
    return {
        "target_win_rate": target_win_rate,
        "gate_mode": gate_mode,
        "best_gap_win_rate_points": round(target_win_rate - _safe_float((best.get("test") or {}).get("win_rate")), 4)
        if isinstance(best, dict)
        else target_win_rate,
        "best_test_expectancy_pct": round(_safe_float((best.get("test") or {}).get("expectancy_pct")), 4)
        if isinstance(best, dict)
        else 0.0,
        "pre_screen_passed_by_family": family_counts,
        "suggestions": suggestions,
    }


def _write_optimizer_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# BTC/ETH/XAUT Whale Jump Optimizer",
        "",
        "Research only. This run does not modify live strategy config, live execution, or order code.",
        "",
        "## Scope",
        "",
        f"- Symbols: `{', '.join(payload['symbols'])}`",
        f"- Market: `{payload['market']}`",
        f"- Interval: `{payload['interval']}`",
        f"- Window: `{payload['start_date']}` to `{payload['end_date']}` UTC, end-exclusive",
        f"- Target win rate: `{payload['targets']['target_win_rate']}%`",
        f"- Min train/test trades: `{payload['targets']['min_train_trades']}` / `{payload['targets']['min_test_trades']}`",
        f"- Regime filters: `{', '.join(payload['grid']['regime_filters'])}`",
        f"- Evaluations: `{payload['grid']['evaluations']}`",
        "",
        "## Dataset",
        "",
    ]
    for symbol, summary in payload["dataset_summaries"].items():
        lines.extend(
            [
                f"- `{symbol}` bars: `{summary['bars']}`",
                f"- `{symbol}` first/last: `{summary['first']}` / `{summary['last']}`",
                f"- `{symbol}` coverage: `{summary['coverage_ratio']}` mature `{summary['mature']}`",
                f"- `{symbol}` checksum ok files: `{summary['checksum_ok_files']}`",
                f"- `{symbol}` missing files: `{summary['missing_files']}`",
            ]
        )
    lines.extend(["", "## Best By Symbol", ""])
    for symbol, row in payload["best_by_symbol"].items():
        if not row:
            lines.append(f"- `{symbol}`: no sample")
            continue
        lines.append(
            f"- `{symbol}` `{row['signal']}`: test win `{row['test']['win_rate']}%`, "
            f"test trades `{row['test']['trade_count']}`, PF `{row['test']['profit_factor']}`, "
            f"WF mean win `{row['walk_forward']['mean_window_win_rate']}%`, "
            f"regime `{row['params']['regime_filter']}`, "
            f"gate passed `{row['gate']['passed']}`"
        )
    lines.extend(["", "## Best With Sample Floor", ""])
    for symbol, row in payload["best_sample_by_symbol"].items():
        if not row:
            lines.append(f"- `{symbol}`: no parameter set met train/test sample floors")
            continue
        lines.append(
            f"- `{symbol}` `{row['signal']}`: test win `{row['test']['win_rate']}%`, "
            f"test trades `{row['test']['trade_count']}`, train trades `{row['train']['trade_count']}`, "
            f"PF `{row['test']['profit_factor']}`, WF mean win `{row['walk_forward']['mean_window_win_rate']}%`, "
            f"regime `{row['params']['regime_filter']}`, "
            f"gate passed `{row['gate']['passed']}`"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Candidate count over gate: `{payload['candidate_count']}`",
            f"- Mature candidate count over gate: `{payload['mature_candidate_count']}`",
            f"- Promotion allowed: `{payload['promotion_allowed']}`",
            f"- Execution recommendation: `{payload['execution_recommendation']}`",
            "",
            "Next iteration:",
        ]
    )
    lines.extend(f"- {item}" for item in payload["next_iteration"]["suggestions"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_tradingview_optimizer_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# BTC/ETH TradingView Convergence Optimizer",
        "",
        "Research only. This run does not modify live strategy config, live execution, or order code.",
        "",
        "## Scope",
        "",
        f"- Symbols: `{', '.join(payload['symbols'])}`",
        f"- Market: `{payload['market']}`",
        f"- Interval: `{payload['interval']}`",
        f"- Window: `{payload['start_date']}` to `{payload['end_date']}` UTC, end-exclusive",
        f"- Target win rate: `{payload['targets']['target_win_rate']}%`",
        f"- Gate mode: `{payload['targets'].get('gate_mode', 'strict_win_rate')}`",
        f"- Min train/test trades: `{payload['targets']['min_train_trades']}` / `{payload['targets']['min_test_trades']}`",
        f"- Min expectancy / payoff: `{payload['targets'].get('min_expectancy_pct')}%` / `{payload['targets'].get('min_payoff_ratio')}`",
        f"- Max per-trade risk: `{payload['targets']['max_per_trade_risk_pct']}%`",
        f"- Regime filters: `{', '.join(payload['grid']['regime_filters'])}`",
        f"- Pre-screen rows / passed: `{payload['grid']['pre_screen_rows']}` / `{payload['grid']['pre_screen_passed']}`",
        f"- Full gate evaluations: `{payload['grid']['evaluations']}`",
        "",
        "## TradingView Concept Sources",
        "",
    ]
    for name, url in payload["concept_sources"].items():
        lines.append(f"- `{name}`: {url}")
    lines.extend(
        [
            "",
            "The Whale Jump L5 page is treated only as a public description source because the script is invite-only. "
            "The optimizer uses transparent local proxies built from public klines, volume, taker flow, MFI, JUMBO, "
            "Supertrend-style trend votes, VWAP, RSI/StochRSI, and Bollinger location.",
            "",
            "## Signal Families",
            "",
        ]
    )
    for family, note in payload["family_notes"].items():
        lines.append(f"- `{family}`: {note}")
    lines.extend(["", "## Dataset", ""])
    for symbol, summary in payload["dataset_summaries"].items():
        lines.extend(
            [
                f"- `{symbol}` bars: `{summary['bars']}`",
                f"- `{symbol}` first/last: `{summary['first']}` / `{summary['last']}`",
                f"- `{symbol}` coverage: `{summary['coverage_ratio']}` mature `{summary['mature']}`",
                f"- `{symbol}` checksum ok files: `{summary['checksum_ok_files']}`",
                f"- `{symbol}` missing files: `{summary['missing_files']}`",
            ]
        )
    lines.extend(["", "## Best By Symbol", ""])
    for symbol, row in payload["best_by_symbol"].items():
        if not row:
            lines.append(f"- `{symbol}`: no full-gate evaluation after pre-screen")
            continue
        lines.append(
            f"- `{symbol}` `{row['tradingview_family']}` `{row['signal']}`: test win `{row['test']['win_rate']}%`, "
            f"test trades `{row['test']['trade_count']}`, PF `{row['test']['profit_factor']}`, "
            f"expectancy `{row['test']['expectancy_pct']}%`, payoff `{row['test']['payoff_ratio']}`, "
            f"stop-loss ratio `{row['test']['stop_loss_ratio']}%`, "
            f"WF mean win `{row['walk_forward']['mean_window_win_rate']}%`, "
            f"regime `{row['params']['regime_filter']}`, "
            f"gate passed `{row['gate']['passed']}`"
        )
    lines.extend(["", "## Best With Sample Floor", ""])
    for symbol, row in payload["best_sample_by_symbol"].items():
        if not row:
            lines.append(f"- `{symbol}`: no parameter set met train/test sample floors")
            continue
        lines.append(
            f"- `{symbol}` `{row['tradingview_family']}` `{row['signal']}`: test win `{row['test']['win_rate']}%`, "
            f"test trades `{row['test']['trade_count']}`, train trades `{row['train']['trade_count']}`, "
            f"PF `{row['test']['profit_factor']}`, expectancy `{row['test']['expectancy_pct']}%`, "
            f"payoff `{row['test']['payoff_ratio']}`, stop-loss ratio `{row['test']['stop_loss_ratio']}%`, "
            f"WF mean win `{row['walk_forward']['mean_window_win_rate']}%`, "
            f"regime `{row['params']['regime_filter']}`, "
            f"gate passed `{row['gate']['passed']}`"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Candidate count over gate: `{payload['candidate_count']}`",
            f"- Mature candidate count over gate: `{payload['mature_candidate_count']}`",
            f"- Promotion allowed: `{payload['promotion_allowed']}`",
            f"- Execution recommendation: `{payload['execution_recommendation']}`",
            "",
            "Next iteration:",
        ]
    )
    lines.extend(f"- {item}" for item in payload["next_iteration"]["suggestions"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_events(
    features: pd.DataFrame,
    *,
    interval: str,
    horizons: list[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    events: list[dict[str, Any]] = []
    index = list(features.index)
    for row_number, (_, row) in enumerate(features.iterrows()):
        signal = str(row.get("ocean_proxy_signal") or "")
        if not signal:
            continue
        direction = _direction_for_signal(signal)
        entry_price = _safe_float(row.get("close"))
        event: dict[str, Any] = {
            "open_time": index[row_number].isoformat(),
            "interval": interval,
            "signal": signal,
            "direction": "long" if direction > 0 else "short",
            "entry_price": round(entry_price, 8),
            "volume_zscore_20": round(_safe_float(row.get("volume_zscore_20")), 4),
            "volume_ratio_20": round(_safe_float(row.get("volume_ratio_20"), 1.0), 4),
            "taker_buy_share": round(_safe_float(row.get("taker_buy_share"), 0.5), 4),
            "taker_flow_imbalance": round(_safe_float(row.get("taker_flow_imbalance")), 4),
            "mfi_14": round(_safe_float(row.get("mfi_14"), 50.0), 4),
            "jumbo_power": round(_safe_float(row.get("jumbo_power")), 4),
            "fib_ote_long_zone": bool(row.get("fib_ote_long_zone", False)),
            "fib_ote_short_zone": bool(row.get("fib_ote_short_zone", False)),
            "liquidity_reclaim_long_20": bool(row.get("liquidity_reclaim_long_20", False)),
            "liquidity_reclaim_short_20": bool(row.get("liquidity_reclaim_short_20", False)),
        }
        for horizon in horizons:
            future_idx = row_number + horizon
            if future_idx >= len(features):
                event[f"ret_{horizon}b_pct"] = None
                event[f"mfe_{horizon}b_pct"] = None
                event[f"mae_{horizon}b_pct"] = None
                continue
            window = features.iloc[row_number + 1 : future_idx + 1]
            future_close = _safe_float(features.iloc[future_idx].get("close"))
            high = _safe_float(window["high"].max())
            low = _safe_float(window["low"].min())
            raw_return = ((future_close / entry_price) - 1.0) * 100.0 if entry_price else 0.0
            if direction > 0:
                directional_return = raw_return
                mfe = ((high / entry_price) - 1.0) * 100.0 if entry_price else 0.0
                mae = ((low / entry_price) - 1.0) * 100.0 if entry_price else 0.0
            else:
                directional_return = -raw_return
                mfe = ((entry_price / low) - 1.0) * 100.0 if low else 0.0
                mae = ((entry_price / high) - 1.0) * 100.0 if high else 0.0
            event[f"ret_{horizon}b_pct"] = round(directional_return, 4)
            event[f"mfe_{horizon}b_pct"] = round(mfe, 4)
            event[f"mae_{horizon}b_pct"] = round(mae, 4)
        events.append(event)

    events_frame = pd.DataFrame(events)
    summary: dict[str, Any] = {
        "interval": interval,
        "bars": int(len(features)),
        "date_start": features.index.min().isoformat(),
        "date_end": features.index.max().isoformat(),
        "signals": {},
        "baseline": {},
    }
    for horizon in horizons:
        future_return = ((features["close"].shift(-horizon) / features["close"]) - 1.0) * 100.0
        summary["baseline"][f"{horizon}b_long"] = _horizon_stats(future_return.dropna().tolist())
        summary["baseline"][f"{horizon}b_short"] = _horizon_stats((-future_return).dropna().tolist())

    if events_frame.empty:
        return events_frame, summary

    for signal in ("L", "XL", "S", "XS"):
        signal_events = events_frame[events_frame["signal"] == signal]
        signal_summary: dict[str, Any] = {"count": int(len(signal_events)), "horizons": {}}
        for horizon in horizons:
            signal_summary["horizons"][f"{horizon}b"] = _horizon_stats(
                signal_events[f"ret_{horizon}b_pct"].dropna().tolist()
            )
            signal_summary["horizons"][f"{horizon}b"]["avg_mfe_pct"] = round(
                float(signal_events[f"mfe_{horizon}b_pct"].dropna().mean()),
                4,
            ) if signal_events[f"mfe_{horizon}b_pct"].dropna().any() else None
            signal_summary["horizons"][f"{horizon}b"]["avg_mae_pct"] = round(
                float(signal_events[f"mae_{horizon}b_pct"].dropna().mean()),
                4,
            ) if signal_events[f"mae_{horizon}b_pct"].dropna().any() else None
        summary["signals"][signal] = signal_summary
    return events_frame, summary


def _horizon_label(interval: str, bars: int) -> str:
    if interval.endswith("h"):
        hours = int(interval[:-1]) * bars
        if hours % 24 == 0:
            return f"{bars} bars / {hours // 24}d"
        return f"{bars} bars / {hours}h"
    return f"{bars} bars"


def _best_signal_notes(interval_summary: dict[str, Any], horizons: list[int]) -> list[str]:
    notes: list[str] = []
    main_horizon = horizons[min(2, len(horizons) - 1)]
    horizon_key = f"{main_horizon}b"
    for signal, payload in interval_summary.get("signals", {}).items():
        stats = payload.get("horizons", {}).get(horizon_key, {})
        if not stats or not stats.get("n"):
            continue
        notes.append(
            (
                f"{interval_summary['interval']} {signal}: n={stats['n']}, "
                f"{_horizon_label(interval_summary['interval'], main_horizon)} "
                f"win={stats['win_rate']}, avg={stats['avg_pct']}%, "
                f"median={stats['median_pct']}%"
            )
        )
    return notes


def _research_verdict(interval_summaries: dict[str, Any]) -> dict[str, Any]:
    notes: list[str] = []
    for summary in interval_summaries.values():
        horizons = summary.get("horizons") or []
        notes.extend(_best_signal_notes(summary, horizons))

    viable = False
    for summary in interval_summaries.values():
        horizons = summary.get("horizons") or []
        if not horizons:
            continue
        horizon = horizons[min(2, len(horizons) - 1)]
        horizon_key = f"{horizon}b"
        for payload in summary.get("signals", {}).values():
            stats = payload.get("horizons", {}).get(horizon_key, {})
            if (stats.get("n") or 0) >= 20 and (stats.get("win_rate") or 0.0) >= 0.54:
                if (stats.get("avg_pct") or 0.0) > 0.0 and (stats.get("median_pct") or 0.0) >= -0.05:
                    viable = True
    return {
        "status": "research_candidate" if viable else "not_ready",
        "live_integration_allowed": False,
        "notes": notes,
        "required_next_steps": [
            "Run walk-forward and out-of-sample tests before any strategy wiring.",
            "Compare against existing BTC route gates and exchange costs.",
            "Keep 2.5% per-trade risk ceiling if this graduates past research.",
        ],
    }


def _write_markdown(
    *,
    path: Path,
    payload: dict[str, Any],
    source_manifest_path: Path,
    events_path: Path,
) -> None:
    lines = [
        "# Ocean X BTC Evidence Research",
        "",
        "This is a research artifact only. It does not modify live execution, strategy YAML, "
        "or order submission code.",
        "",
        "## Scope",
        "",
        f"- Symbol: `{payload['symbol']}`",
        f"- Market: `{payload['market']}`",
        f"- Window: `{payload['start_date']}` to `{payload['end_date']}` UTC, end-exclusive",
        f"- Intervals: `{', '.join(payload['intervals'])}`",
        "",
        "## Source Evidence",
        "",
        f"- TradingView L5 page: {TRADINGVIEW_L5_URL}",
        f"- TradingView L3 open-source reference page: {TRADINGVIEW_L3_URL}",
        f"- Binance public data repository: {BINANCE_PUBLIC_DATA_URL}",
        f"- Binance USD-M futures kline endpoint docs: {BINANCE_FAPI_KLINES_URL}",
        f"- Source manifest: `{source_manifest_path.name}`",
        f"- Event sample: `{events_path.name}`",
        "",
        "The L5 indicator page states that the original script is invite-only and closed-source. "
        "This run therefore uses a transparent proxy based on public BTCUSDT futures klines, "
        "taker-flow, volume spikes, MFI, local JUMBO-style composite fields, liquidity sweeps, "
        "and Fibonacci pullback zones.",
        "",
        "## Data Integrity",
        "",
    ]
    for interval, summary in payload["interval_summaries"].items():
        lines.extend(
            [
                f"- `{interval}` bars: `{summary['bars']}`",
                f"- `{interval}` first/last: `{summary['date_start']}` / `{summary['date_end']}`",
                f"- `{interval}` fetched files: `{summary['source_files']}`",
                f"- `{interval}` checksum ok files: `{summary['checksum_ok_files']}`",
                f"- `{interval}` missing files: `{summary['missing_files']}`",
            ]
        )
    lines.extend(["", "## Signal Evidence", ""])
    for interval, summary in payload["interval_summaries"].items():
        horizons = summary["horizons"]
        lines.append(f"### {interval}")
        for signal, signal_payload in summary["signals"].items():
            lines.append(f"- `{signal}` count: `{signal_payload['count']}`")
            for horizon in horizons:
                horizon_key = f"{horizon}b"
                stats = signal_payload["horizons"][horizon_key]
                lines.append(
                    f"  - {_horizon_label(interval, horizon)}: n=`{stats['n']}`, "
                    f"win=`{stats['win_rate']}`, avg=`{stats['avg_pct']}%`, "
                    f"median=`{stats['median_pct']}%`, "
                    f"avg MFE=`{stats['avg_mfe_pct']}%`, avg MAE=`{stats['avg_mae_pct']}%`"
                )
        lines.append("")
    verdict = payload["verdict"]
    lines.extend(
        [
            "## Verdict",
            "",
            f"- Status: `{verdict['status']}`",
            f"- Live integration allowed: `{verdict['live_integration_allowed']}`",
            "",
            "Notes:",
        ]
    )
    lines.extend(f"- {note}" for note in verdict["notes"])
    lines.extend(["", "Required next steps:"])
    lines.extend(f"- {step}" for step in verdict["required_next_steps"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_research(
    *,
    symbol: str,
    market: str,
    intervals: list[str],
    start: date,
    end_exclusive: date,
    timeout: float,
    volume_z: float,
    extreme_volume_z: float,
    taker_share: float,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    run_id = f"{_utc_stamp()}-{symbol.lower()}-{market}-ocean-x-evidence"
    report_dir = output_dir or (PROJECT_ROOT / "reports" / run_id)
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "symbol": symbol,
        "market": market,
        "start_date": start.isoformat(),
        "end_date": end_exclusive.isoformat(),
        "sources": [],
    }
    interval_summaries: dict[str, Any] = {}
    all_events: list[pd.DataFrame] = []
    for interval in intervals:
        horizons = [6, 12, 24, 48] if interval == "1h" else [3, 6, 12, 24]
        frame, sources = fetch_history(
            market=market,
            symbol=symbol,
            interval=interval,
            start=start,
            end_exclusive=end_exclusive,
            timeout=timeout,
        )
        features = build_proxy_features(
            frame,
            interval=interval,
            volume_z=volume_z,
            extreme_volume_z=extreme_volume_z,
            taker_share=taker_share,
        )
        events, summary = evaluate_events(features, interval=interval, horizons=horizons)
        if not events.empty:
            all_events.append(events)

        fetched_sources = [source for source in sources if source.status_code == 200]
        checksum_ok_files = [source for source in fetched_sources if source.checksum_ok is True]
        missing_files = [source.filename for source in sources if source.status_code == 404]
        summary.update(
            {
                "horizons": horizons,
                "source_files": len(fetched_sources),
                "checksum_ok_files": len(checksum_ok_files),
                "missing_files": missing_files,
            }
        )
        interval_summaries[interval] = summary
        manifest["sources"].extend(asdict(source) | {"interval": interval} for source in sources)

        feature_path = report_dir / f"{symbol}-{market}-{interval}-features.csv"
        compact_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "taker_buy_quote_volume",
            "volume_zscore_20",
            "volume_ratio_20",
            "taker_buy_share",
            "taker_flow_imbalance",
            "mfi_14",
            "jumbo_power",
            "jumbo_power_ma",
            "fib_ote_long_zone",
            "fib_ote_short_zone",
            "liquidity_reclaim_long_20",
            "liquidity_reclaim_short_20",
            "ocean_proxy_signal",
        ]
        features[compact_columns].to_csv(feature_path)

    events_frame = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    events_path = report_dir / "ocean_proxy_events.csv"
    events_frame.to_csv(events_path, index=False)
    source_manifest_path = report_dir / "source_manifest.json"
    source_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    payload: dict[str, Any] = {
        "run_id": run_id,
        "symbol": symbol,
        "market": market,
        "start_date": start.isoformat(),
        "end_date": end_exclusive.isoformat(),
        "intervals": intervals,
        "thresholds": {
            "volume_z": volume_z,
            "extreme_volume_z": extreme_volume_z,
            "taker_share": taker_share,
        },
        "interval_summaries": interval_summaries,
        "verdict": _research_verdict(interval_summaries),
        "artifacts": {
            "report_dir": str(report_dir),
            "source_manifest": str(source_manifest_path),
            "events_csv": str(events_path),
            "summary_json": str(report_dir / "summary.json"),
            "research_md": str(report_dir / "research.md"),
        },
    }
    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(
        path=report_dir / "research.md",
        payload=payload,
        source_manifest_path=source_manifest_path,
        events_path=events_path,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build BTCUSDT Ocean-X-inspired evidence without touching live strategy code."
    )
    parser.add_argument(
        "--optimize-btc-eth",
        action="store_true",
        help="Backward-compatible alias for the research-only core whale-jump optimizer.",
    )
    parser.add_argument("--optimize-core-whale-jump", action="store_true")
    parser.add_argument("--optimize-btc-eth-tradingview", action="store_true")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--market", choices=("futures", "spot"), default="futures")
    parser.add_argument("--intervals", default="1h,4h")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--start", default="2024-05-03")
    parser.add_argument("--end", default="2026-05-03")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--volume-z", type=float, default=1.8)
    parser.add_argument("--extreme-volume-z", type=float, default=2.4)
    parser.add_argument("--taker-share", type=float, default=0.58)
    parser.add_argument("--target-win-rate", type=float, default=80.0)
    parser.add_argument("--min-train-trades", type=int, default=40)
    parser.add_argument("--min-test-trades", type=int, default=15)
    parser.add_argument("--min-profit-factor", type=float, default=1.5)
    parser.add_argument("--max-stop-loss-ratio", type=float, default=20.0)
    parser.add_argument("--max-per-trade-risk-pct", type=float, default=2.5)
    parser.add_argument("--gate-mode", choices=GATE_MODES, default="strict_win_rate")
    parser.add_argument("--min-expectancy-pct", type=float, default=0.05)
    parser.add_argument("--min-payoff-ratio", type=float, default=1.2)
    parser.add_argument("--max-drawdown-pct", type=float, default=20.0)
    parser.add_argument("--max-loss-streak", type=int, default=8)
    parser.add_argument("--min-dataset-bars", type=int, default=1000)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.65)
    parser.add_argument("--regime-filters", default="none,trend,pullback,liquidity,range,strong_flow")
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--max-full-evaluations", type=int, default=600)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    if args.optimize_btc_eth_tradingview:
        symbols = [item.upper() for item in _parse_csv_items(args.symbols)]
        if not symbols:
            symbols = list(BTC_ETH_TRADINGVIEW_SYMBOLS)
        payload = optimize_btc_eth_tradingview_convergence(
            symbols=symbols,
            market=args.market,
            interval=args.interval,
            start=_parse_date(args.start),
            end_exclusive=_parse_date(args.end),
            timeout=args.timeout,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            target_win_rate=args.target_win_rate,
            min_train_trades=args.min_train_trades,
            min_test_trades=args.min_test_trades,
            min_profit_factor=args.min_profit_factor,
            max_configs=args.max_configs,
            regime_filters=_parse_csv_items(args.regime_filters),
            max_stop_loss_ratio=args.max_stop_loss_ratio,
            min_dataset_bars=args.min_dataset_bars,
            min_coverage_ratio=args.min_coverage_ratio,
            max_per_trade_risk_pct=args.max_per_trade_risk_pct,
            max_full_evaluations=args.max_full_evaluations,
            gate_mode=args.gate_mode,
            min_expectancy_pct=args.min_expectancy_pct,
            min_payoff_ratio=args.min_payoff_ratio,
            max_drawdown_pct=args.max_drawdown_pct,
            max_loss_streak=args.max_loss_streak,
        )
        print(
            json.dumps(
                {
                    "run_id": payload["run_id"],
                    "candidate_count": payload["candidate_count"],
                    "mature_candidate_count": payload["mature_candidate_count"],
                    "promotion_allowed": payload["promotion_allowed"],
                    "execution_recommendation": payload["execution_recommendation"],
                    "report_dir": payload["artifacts"]["report_dir"],
                    "summary_json": payload["artifacts"]["summary_json"],
                    "research_md": payload["artifacts"]["research_md"],
                    "pre_screen_passed": payload["grid"]["pre_screen_passed"],
                    "evaluations": payload["grid"]["evaluations"],
                    "best_by_symbol": {
                        symbol: {
                            "family": (row or {}).get("tradingview_family"),
                            "signal": (row or {}).get("signal"),
                            "test_win_rate": ((row or {}).get("test") or {}).get("win_rate"),
                            "test_trades": ((row or {}).get("test") or {}).get("trade_count"),
                            "profit_factor": ((row or {}).get("test") or {}).get("profit_factor"),
                            "expectancy_pct": ((row or {}).get("test") or {}).get("expectancy_pct"),
                            "payoff_ratio": ((row or {}).get("test") or {}).get("payoff_ratio"),
                            "stop_loss_ratio": ((row or {}).get("test") or {}).get("stop_loss_ratio"),
                            "regime_filter": ((row or {}).get("params") or {}).get("regime_filter"),
                            "gate_passed": ((row or {}).get("gate") or {}).get("passed"),
                        }
                        for symbol, row in payload["best_by_symbol"].items()
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.optimize_btc_eth or args.optimize_core_whale_jump:
        symbols = [item.upper() for item in _parse_csv_items(args.symbols)]
        if not symbols:
            symbols = list(CORE_WHALE_JUMP_SYMBOLS if args.optimize_core_whale_jump else ("BTCUSDT", "ETHUSDT"))
        payload = optimize_core_whale_jump_proxy(
            symbols=symbols,
            market=args.market,
            interval=args.interval,
            start=_parse_date(args.start),
            end_exclusive=_parse_date(args.end),
            timeout=args.timeout,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            target_win_rate=args.target_win_rate,
            min_train_trades=args.min_train_trades,
            min_test_trades=args.min_test_trades,
            min_profit_factor=args.min_profit_factor,
            max_configs=args.max_configs,
            regime_filters=_parse_csv_items(args.regime_filters),
            max_stop_loss_ratio=args.max_stop_loss_ratio,
            min_dataset_bars=args.min_dataset_bars,
            min_coverage_ratio=args.min_coverage_ratio,
        )
        print(
            json.dumps(
                {
                    "run_id": payload["run_id"],
                    "candidate_count": payload["candidate_count"],
                    "mature_candidate_count": payload["mature_candidate_count"],
                    "promotion_allowed": payload["promotion_allowed"],
                    "execution_recommendation": payload["execution_recommendation"],
                    "report_dir": payload["artifacts"]["report_dir"],
                    "summary_json": payload["artifacts"]["summary_json"],
                    "research_md": payload["artifacts"]["research_md"],
                    "best_by_symbol": {
                        symbol: {
                            "signal": (row or {}).get("signal"),
                            "test_win_rate": ((row or {}).get("test") or {}).get("win_rate"),
                            "test_trades": ((row or {}).get("test") or {}).get("trade_count"),
                            "regime_filter": ((row or {}).get("params") or {}).get("regime_filter"),
                            "gate_passed": ((row or {}).get("gate") or {}).get("passed"),
                        }
                        for symbol, row in payload["best_by_symbol"].items()
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    payload = run_research(
        symbol=args.symbol.upper(),
        market=args.market,
        intervals=[item.strip() for item in args.intervals.split(",") if item.strip()],
        start=_parse_date(args.start),
        end_exclusive=_parse_date(args.end),
        timeout=args.timeout,
        volume_z=args.volume_z,
        extreme_volume_z=args.extreme_volume_z,
        taker_share=args.taker_share,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(
        json.dumps(
            {
                "run_id": payload["run_id"],
                "status": payload["verdict"]["status"],
                "live_integration_allowed": payload["verdict"]["live_integration_allowed"],
                "report_dir": payload["artifacts"]["report_dir"],
                "summary_json": payload["artifacts"]["summary_json"],
                "research_md": payload["artifacts"]["research_md"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
