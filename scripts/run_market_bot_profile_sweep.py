from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from binance_quant_control.analysis import enrich_indicators, prepare_klines_frame
from binance_quant_control.asset_routing import resolve_symbol_route
from binance_quant_control.backtest import audit_backtest_robustness, simulate_backtest
from binance_quant_control.binance_api import BinanceClient
from binance_quant_control.config import PROJECT_ROOT, load_settings
from binance_quant_control.historical_klines import fetch_recent_klines
from binance_quant_control.strategy import load_strategy_config

DEFAULT_SYMBOLS = "SOLUSDT,TRXUSDT,XRPUSDT,ETHUSDT"
DEFAULT_INTERVAL = "4h"
DEFAULT_LIMIT = 6000
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "state" / "market-bot-profile-exit-sweep-4h-l6000-v3"
DEFAULT_STRATEGY_FAMILIES = ("ai_family_router",)
MARKET_BOT_TARGETS = {
    "min_trades": 100,
    "min_profit_factor": 1.25,
    "min_expectancy_r": 0.05,
    "min_payoff_ratio": 1.20,
    "min_out_of_sample_return_pct": 0.0,
    "max_stop_loss_ratio": 55.0,
    "min_walk_forward_stability": 0.60,
    "min_slippage_resilience": 0.70,
}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def _compound_return_pct(pnls: list[float]) -> float:
    equity = 1.0
    for pnl in pnls:
        equity *= 1.0 + (pnl / 100.0)
    return round((equity - 1.0) * 100.0, 4)


def _out_of_sample_return_pct(summary: dict[str, Any]) -> float:
    trades = list(summary.get("trades") or [])
    if not trades:
        return _float(summary.get("total_return_pct"))
    start = max(0, int(len(trades) * 0.7))
    return _compound_return_pct([_float(item.get("pnl_pct")) for item in trades[start:]])


def _fold_stability(robustness: dict[str, Any]) -> float:
    folds = list(robustness.get("folds") or [])
    if not folds:
        return 0.0
    positive = sum(1 for item in folds if _float(item.get("total_return_pct")) > 0.0)
    pf_ok = sum(1 for item in folds if _float(item.get("profit_factor")) >= 1.0)
    return round((positive / len(folds)) * 0.6 + (pf_ok / len(folds)) * 0.4, 4)


def _win_rate(values: list[float]) -> float:
    if not values:
        return 0.0
    return round((sum(1 for value in values if value > 0.0) / len(values)) * 100.0, 2)


def _profit_factor(values: list[float]) -> float:
    gains = sum(value for value in values if value > 0.0)
    losses = abs(sum(value for value in values if value <= 0.0))
    if losses <= 0.0:
        return 9999.0 if gains > 0.0 else 0.0
    return round(gains / losses, 4)


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _trade_breakdown(summary: dict[str, Any]) -> dict[str, Any]:
    trades = list(summary.get("trades") or [])
    by_side: dict[str, dict[str, Any]] = {}
    by_routed_family: dict[str, dict[str, Any]] = {}
    for key, field in (("by_side", "side"), ("by_routed_family", "routed_strategy_family")):
        groups: dict[str, list[float]] = {}
        for trade in trades:
            label = str(trade.get(field) or "unknown")
            groups.setdefault(label, []).append(_float(trade.get("pnl_r")))
        target = by_side if key == "by_side" else by_routed_family
        for label, values in groups.items():
            target[label] = {
                "trade_count": len(values),
                "win_rate": _win_rate(values),
                "profit_factor": _profit_factor(values),
                "expectancy_r": _avg(values),
            }
    return {
        "by_side": by_side,
        "by_routed_family": by_routed_family,
        "exit_reasons": dict(Counter(str(item.get("exit_reason") or "unknown") for item in trades)),
    }


def _signed_di(row: Any, side: str) -> float:
    plus_di = _float(row.get("plus_di"))
    minus_di = _float(row.get("minus_di"))
    return plus_di - minus_di if side == "BUY" else minus_di - plus_di


def _entry_filter(config: dict[str, Any]) -> Any:
    def gate(previous: Any, _current: Any, analysis: dict[str, Any], _idx: int) -> tuple[bool, str]:
        side = str(analysis.get("recommended_action") or "").upper()
        if side not in {"BUY", "SELL"}:
            return True, ""
        signed_di = _signed_di(previous, side)
        obv_zscore = _float(previous.get("obv_zscore_20"))
        volume_zscore = _float(previous.get("volume_zscore_20"))
        adx = _float(previous.get("adx"))
        bb_percent_b = _float(previous.get("bb_percent_b"), 0.5)
        bb_bandwidth = _float(previous.get("bb_bandwidth"))
        rsi = _float(previous.get("rsi_14"), 50.0)
        mfi = _float(previous.get("mfi_14"), 50.0)
        stoch_k = _float(previous.get("stoch_rsi_k"), 50.0)
        vwap_distance = _float(previous.get("vwap_distance_pct_48"))
        liquidity_close_position = _float(previous.get("liquidity_close_position"), 0.5)
        trend_votes = int(_float(previous.get("supertrend_direction"))) + int(
            _float(previous.get("trend_magic_direction"))
        ) + int(_float(previous.get("follow_line_direction")))
        allowed_sides = tuple(str(item).upper() for item in config.get("allowed_sides") or ())
        blocked_routed_families = tuple(str(item) for item in config.get("blocked_routed_families") or ())
        required_routed_families = tuple(str(item) for item in config.get("required_routed_families") or ())
        routed_family = str(analysis.get("routed_strategy_family") or analysis.get("strategy_family") or "")
        if allowed_sides and side not in allowed_sides:
            return False, "sweep-side-policy"
        if blocked_routed_families and routed_family in blocked_routed_families:
            return False, "sweep-blocked-routed-family"
        if required_routed_families and routed_family not in required_routed_families:
            return False, "sweep-required-routed-family"
        if "min_signed_di" in config and signed_di < float(config["min_signed_di"]):
            return False, "sweep-min-signed-di"
        if "max_signed_di" in config and signed_di > float(config["max_signed_di"]):
            return False, "sweep-max-signed-di"
        if "min_obv_zscore" in config and obv_zscore < float(config["min_obv_zscore"]):
            return False, "sweep-min-obv-zscore"
        if "max_obv_zscore" in config and obv_zscore > float(config["max_obv_zscore"]):
            return False, "sweep-max-obv-zscore"
        if "max_abs_obv_zscore" in config and abs(obv_zscore) > float(config["max_abs_obv_zscore"]):
            return False, "sweep-max-abs-obv-zscore"
        if "min_volume_zscore" in config and volume_zscore < float(config["min_volume_zscore"]):
            return False, "sweep-min-volume-zscore"
        if "max_volume_zscore" in config and volume_zscore > float(config["max_volume_zscore"]):
            return False, "sweep-max-volume-zscore"
        if "min_adx" in config and adx < float(config["min_adx"]):
            return False, "sweep-min-adx"
        if "max_adx" in config and adx > float(config["max_adx"]):
            return False, "sweep-max-adx"
        if "min_bb_percent_b" in config and bb_percent_b < float(config["min_bb_percent_b"]):
            return False, "sweep-min-bb-percent-b"
        if "max_bb_percent_b" in config and bb_percent_b > float(config["max_bb_percent_b"]):
            return False, "sweep-max-bb-percent-b"
        if "min_bb_bandwidth" in config and bb_bandwidth < float(config["min_bb_bandwidth"]):
            return False, "sweep-min-bb-bandwidth"
        if "max_bb_bandwidth" in config and bb_bandwidth > float(config["max_bb_bandwidth"]):
            return False, "sweep-max-bb-bandwidth"
        if "min_rsi" in config and rsi < float(config["min_rsi"]):
            return False, "sweep-min-rsi"
        if "max_rsi" in config and rsi > float(config["max_rsi"]):
            return False, "sweep-max-rsi"
        if "min_mfi" in config and mfi < float(config["min_mfi"]):
            return False, "sweep-min-mfi"
        if "max_mfi" in config and mfi > float(config["max_mfi"]):
            return False, "sweep-max-mfi"
        if "min_stoch_k" in config and stoch_k < float(config["min_stoch_k"]):
            return False, "sweep-min-stoch-k"
        if "max_stoch_k" in config and stoch_k > float(config["max_stoch_k"]):
            return False, "sweep-max-stoch-k"
        if "min_vwap_distance_pct" in config and vwap_distance < float(config["min_vwap_distance_pct"]):
            return False, "sweep-min-vwap-distance"
        if "max_vwap_distance_pct" in config and vwap_distance > float(config["max_vwap_distance_pct"]):
            return False, "sweep-max-vwap-distance"
        if "min_liquidity_close_position" in config and liquidity_close_position < float(
            config["min_liquidity_close_position"]
        ):
            return False, "sweep-min-liquidity-close-position"
        if "max_liquidity_close_position" in config and liquidity_close_position > float(
            config["max_liquidity_close_position"]
        ):
            return False, "sweep-max-liquidity-close-position"
        if "min_trend_votes" in config and trend_votes < int(config["min_trend_votes"]):
            return False, "sweep-min-trend-votes"
        if "max_trend_votes" in config and trend_votes > int(config["max_trend_votes"]):
            return False, "sweep-max-trend-votes"
        return True, ""

    return gate


def _profiles(symbol: str) -> list[dict[str, Any]]:
    profiles = {
        "SOLUSDT": [
            {"min_signed_di": value}
            for value in (10.0, 12.0, 13.5954, 15.0, 18.0)
        ]
        + [
            {"min_signed_di": 13.5954, "min_volume_zscore": value}
            for value in (-1.0, -0.5, 0.0)
        ],
        "TRXUSDT": [
            {"min_obv_zscore": low, "max_obv_zscore": high}
            for low, high in (
                (-1.9, 1.4),
                (-1.5816, 1.4232),
                (-1.2, 1.4),
                (-1.5816, 1.0),
                (-1.5816, 0.8),
                (-1.0, 1.0),
            )
        ]
        + [
            {"max_adx": 26.1715, "min_obv_zscore": -1.5816},
            {"min_adx": 17.2087, "max_volume_zscore": 0.5318},
            {"allowed_sides": ("BUY",), "min_obv_zscore": -1.5816, "max_obv_zscore": 1.4232},
            {"allowed_sides": ("SELL",), "min_obv_zscore": -1.5816, "max_obv_zscore": 1.4232},
            {"required_routed_families": ("vwap_reclaim",), "min_obv_zscore": -1.5816},
            {"required_routed_families": ("liquidity_reclaim",), "min_obv_zscore": -1.5816},
            {"blocked_routed_families": ("breakout",), "min_obv_zscore": -1.5816, "max_obv_zscore": 1.4232},
            {"max_bb_percent_b": 0.78, "min_obv_zscore": -1.5816},
            {"min_bb_percent_b": 0.22, "max_obv_zscore": 1.4232},
        ],
        "XRPUSDT": [
            {"min_volume_zscore": volume, "min_signed_di": signed_di}
            for volume, signed_di in (
                (-0.5, 8.0),
                (-0.3658, 10.2),
                (-0.2416, 10.2037),
                (-0.1251, 8.1612),
                (0.0, 8.0),
                (0.0, 10.0),
            )
        ]
        + [
            {"allowed_sides": ("BUY",), "min_volume_zscore": -0.2416, "min_signed_di": 10.2037},
            {"allowed_sides": ("SELL",), "min_volume_zscore": -0.2416, "min_signed_di": 10.2037},
            {"min_bb_bandwidth": 0.06, "min_volume_zscore": -0.2416, "min_signed_di": 8.0},
            {"min_bb_bandwidth": 0.075, "min_volume_zscore": -0.2416, "min_signed_di": 8.0},
            {"max_adx": 24.0, "min_volume_zscore": -0.2416, "min_signed_di": 8.0},
            {"max_adx": 22.0, "min_volume_zscore": -0.2416, "min_signed_di": 8.0},
            {"blocked_routed_families": ("breakout",), "min_volume_zscore": -0.2416, "min_signed_di": 8.0},
            {"required_routed_families": ("liquidity_reclaim",), "min_volume_zscore": -0.2416},
            {"required_routed_families": ("vwap_reclaim",), "min_volume_zscore": -0.2416},
            {"max_obv_zscore": 1.2, "min_volume_zscore": -0.2416, "min_signed_di": 8.0},
            {"min_obv_zscore": -1.2, "max_obv_zscore": 1.2, "min_signed_di": 8.0},
        ],
        "ETHUSDT": [
            {"max_obv_zscore": obv, "min_signed_di": signed_di}
            for obv, signed_di in (
                (1.3163, 5.1712),
                (1.7927, 8.6913),
                (1.7927, 9.7294),
                (0.8006, 1.4124),
                (1.3163, -1.4266),
            )
        ]
        + [
            {"max_adx": 25.5935, "max_obv_zscore": 1.7927},
            {"allowed_sides": ("BUY",), "max_obv_zscore": 1.3163, "min_signed_di": 5.1712},
            {"allowed_sides": ("SELL",), "max_obv_zscore": 1.3163, "min_signed_di": 5.1712},
            {"required_routed_families": ("trend_pullback",), "max_obv_zscore": 1.7927, "min_signed_di": 5.0},
            {"required_routed_families": ("vwap_reclaim",), "max_obv_zscore": 1.7927},
            {"required_routed_families": ("liquidity_reclaim",), "max_obv_zscore": 1.7927},
            {"blocked_routed_families": ("breakout",), "max_obv_zscore": 1.3163, "min_signed_di": 5.1712},
            {"min_adx": 18.0, "max_obv_zscore": 1.3163, "min_signed_di": 5.1712},
            {"min_adx": 20.0, "max_obv_zscore": 1.3163, "min_signed_di": 5.1712},
            {"min_trend_votes": 1, "max_obv_zscore": 1.3163, "min_signed_di": 5.1712},
            {"max_vwap_distance_pct": 0.65, "max_obv_zscore": 1.3163, "min_signed_di": 5.1712},
        ],
    }
    raw_profiles = profiles.get(symbol.upper(), [{}])
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for profile in raw_profiles:
        key = json.dumps(profile, sort_keys=True)
        if key not in seen:
            unique.append(profile)
            seen.add(key)
    return unique


def _exit_variants() -> list[dict[str, Any]]:
    return [
        {"name": "base", "atr": 1.35, "tp": (1.2, 2.4, 4.8), "trail": 0.45, "time_limit": 72},
        {"name": "wider", "atr": 1.55, "tp": (1.3, 2.6, 5.2), "trail": 0.55, "time_limit": 72},
        {"name": "faster", "atr": 1.20, "tp": (1.0, 2.0, 4.0), "trail": 0.40, "time_limit": 60},
        {"name": "runner", "atr": 1.35, "tp": (1.3, 2.8, 6.0), "trail": 0.65, "time_limit": 96},
        {"name": "quick_runner", "atr": 1.25, "tp": (1.1, 2.4, 5.2), "trail": 0.50, "time_limit": 48},
        {"name": "wide_runner", "atr": 1.70, "tp": (1.4, 3.0, 6.4), "trail": 0.70, "time_limit": 96},
        {"name": "fee_resilient", "atr": 1.45, "tp": (1.5, 3.0, 6.0), "trail": 0.60, "time_limit": 72},
    ]


def _strategy_families(args: argparse.Namespace) -> list[str]:
    raw = str(args.strategy_families or "").strip()
    if not raw:
        return list(DEFAULT_STRATEGY_FAMILIES)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _gate_passed(row: dict[str, Any]) -> bool:
    return (
        _float(row.get("trade_count")) >= MARKET_BOT_TARGETS["min_trades"]
        and _float(row.get("profit_factor")) >= MARKET_BOT_TARGETS["min_profit_factor"]
        and _float(row.get("expectancy_r")) >= MARKET_BOT_TARGETS["min_expectancy_r"]
        and _float(row.get("payoff_ratio")) >= MARKET_BOT_TARGETS["min_payoff_ratio"]
        and _float(row.get("out_of_sample_total_return_pct")) >= MARKET_BOT_TARGETS["min_out_of_sample_return_pct"]
        and _float(row.get("stop_loss_ratio")) <= MARKET_BOT_TARGETS["max_stop_loss_ratio"]
        and _float(row.get("walk_forward_stability")) >= MARKET_BOT_TARGETS["min_walk_forward_stability"]
        and _float(row.get("slippage_resilience")) >= MARKET_BOT_TARGETS["min_slippage_resilience"]
    )


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    symbols = [item.strip().upper() for item in str(args.symbols).split(",") if item.strip()]
    strategy_families = _strategy_families(args)
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    progress_path = root / "profile-sweep-progress.json"
    results_path = root / "profile-sweep-results.jsonl"
    final_path = root / "profile-sweep-summary.json"
    settings = load_settings()
    base_strategy = load_strategy_config(args.strategy_config)
    with results_path.open("w", encoding="utf-8") as results_file:
        with BinanceClient(settings) as client:
            frames = {}
            for symbol in symbols:
                raw = fetch_recent_klines(client, symbol, args.interval, args.limit, base_strategy.defaults.market)
                frames[symbol] = enrich_indicators(
                    prepare_klines_frame(raw),
                    args.interval,
                    strategy=base_strategy,
                )
        rows: list[dict[str, Any]] = []
        total = sum(len(strategy_families) * len(_profiles(symbol)) * len(_exit_variants()) for symbol in symbols)
        completed = 0
        for symbol in symbols:
            route = resolve_symbol_route(symbol)
            market_context_cache: dict[int, dict[str, Any]] = {}
            for strategy_family in strategy_families:
                for profile_index, profile in enumerate(_profiles(symbol)):
                    for exit_variant in _exit_variants():
                        completed += 1
                        _write_progress(
                            progress_path,
                            {
                                "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                                "completed": completed,
                                "total": total,
                                "current": {
                                    "symbol": symbol,
                                    "strategy_family": strategy_family,
                                    "profile_index": profile_index,
                                    "exit_variant": exit_variant["name"],
                                },
                                "row_count": len(rows),
                            },
                        )
                        strategy = replace(
                            base_strategy,
                            risk=replace(
                                base_strategy.risk,
                                atr_stop_multiple=float(exit_variant["atr"]),
                                take_profit_r_multiples=tuple(exit_variant["tp"]),
                                trailing_callback_pct=float(exit_variant["trail"]),
                                time_limit_bars=int(exit_variant["time_limit"]),
                            ),
                        )
                        summary = simulate_backtest(
                            frames[symbol],
                            base_strategy.defaults.market,
                            strategy,
                            symbol=symbol,
                            interval=args.interval,
                            strategy_family=strategy_family,
                            entry_filter=_entry_filter(profile),
                            market_context_cache=market_context_cache,
                            require_score_model_confirmation=not args.fast,
                            lightweight_market_context=args.fast,
                        )
                        if int(summary.get("trade_count") or 0) < args.min_trades:
                            continue
                        robustness = audit_backtest_robustness(summary, route.validation)
                        row: dict[str, Any] = {
                            "symbol": symbol,
                            "strategy_family": strategy_family,
                            "profile_index": profile_index,
                            "entry_filters": profile,
                            "exit_variant": exit_variant["name"],
                            "atr_stop_multiple": float(exit_variant["atr"]),
                            "take_profit_r_multiples": list(exit_variant["tp"]),
                            "trailing_callback_pct": float(exit_variant["trail"]),
                            "time_limit_bars": int(exit_variant["time_limit"]),
                            "trade_count": summary.get("trade_count"),
                            "profit_factor": summary.get("profit_factor"),
                            "expectancy_r": summary.get("expectancy_r"),
                            "payoff_ratio": summary.get("payoff_ratio"),
                            "win_rate": summary.get("win_rate"),
                            "stop_loss_ratio": summary.get("stop_loss_ratio"),
                            "partial_tp_then_stop_ratio": summary.get("partial_tp_then_stop_ratio"),
                            "out_of_sample_total_return_pct": _out_of_sample_return_pct(summary),
                            "total_return_pct": summary.get("total_return_pct"),
                            "max_drawdown_pct": summary.get("max_drawdown_pct"),
                            "walk_forward_stability": _fold_stability(robustness),
                            "robustness_status": robustness.get("status"),
                            "robustness_reasons": robustness.get("reasons") or [],
                            "breakdown": _trade_breakdown(summary),
                        }
                        if (
                            _float(row["profit_factor"]) >= 1.12
                            and _float(row["expectancy_r"]) >= 0.02
                            and _float(row["out_of_sample_total_return_pct"]) >= -2.0
                        ):
                            stressed_returns = []
                            for slippage_bps in (8.0, 15.0):
                                stressed_strategy = replace(
                                    strategy,
                                    execution=replace(strategy.execution, slippage_bps=slippage_bps),
                                )
                                stressed = simulate_backtest(
                                    frames[symbol],
                                    base_strategy.defaults.market,
                                    stressed_strategy,
                                    symbol=symbol,
                                    interval=args.interval,
                                    strategy_family=strategy_family,
                                    entry_filter=_entry_filter(profile),
                                    market_context_cache=market_context_cache,
                                    require_score_model_confirmation=not args.fast,
                                    lightweight_market_context=args.fast,
                                )
                                stressed_returns.append(_float(stressed.get("total_return_pct")))
                            total_return = _float(row["total_return_pct"])
                            row["stressed_returns"] = stressed_returns
                            row["slippage_resilience"] = (
                                round(max(0.0, min(stressed_returns)) / total_return, 4)
                                if total_return > 0.0
                                else 0.0
                            )
                        else:
                            row["stressed_returns"] = []
                            row["slippage_resilience"] = 0.0
                        row["gate_passed"] = _gate_passed(row)
                        rows.append(row)
                        results_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        results_file.flush()
    rows.sort(
        key=lambda row: (
            bool(row.get("gate_passed")),
            _float(row.get("slippage_resilience")),
            _float(row.get("profit_factor")),
            _float(row.get("expectancy_r")),
            _float(row.get("payoff_ratio")),
            _float(row.get("out_of_sample_total_return_pct")),
            _float(row.get("walk_forward_stability")),
        ),
        reverse=True,
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "mode": "market_bot_profile_exit_sweep",
        "symbols": symbols,
        "interval": args.interval,
        "limit": args.limit,
        "strategy_families": strategy_families,
        "fast_prefilter": bool(args.fast),
        "market_bot_targets": MARKET_BOT_TARGETS,
        "opens_orders": False,
        "mainnet_live_allowed": False,
        "progress_path": str(progress_path),
        "results_path": str(results_path),
        "top": rows[:20],
        "by_symbol": {
            symbol: [row for row in rows if row.get("symbol") == symbol][:5]
            for symbol in symbols
        },
    }
    final_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    payload["report_path"] = str(final_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run checkpointed market-bot profile/exit sweep.")
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--strategy-config", default="config/strategy-market-bot-payoff-research.yaml")
    parser.add_argument("--strategy-families", default=",".join(DEFAULT_STRATEGY_FAMILIES))
    parser.add_argument("--min-trades", type=int, default=95)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip heavy score-model confirmation for prefilter sweeps; validate survivors with alpha-research.",
    )
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    payload = run(args)
    if args.compact:
        print(
            json.dumps(
                {
                    "mode": payload["mode"],
                    "symbols": payload["symbols"],
                    "top": payload["top"][:8],
                    "by_symbol": payload["by_symbol"],
                    "report_path": payload["report_path"],
                    "progress_path": payload["progress_path"],
                    "results_path": payload["results_path"],
                    "opens_orders": payload["opens_orders"],
                    "mainnet_live_allowed": payload["mainnet_live_allowed"],
                },
                separators=(",", ":"),
            )
        )
        return 0
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
