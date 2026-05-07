from __future__ import annotations

import dataclasses
import itertools
import json
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean
from typing import Any

import httpx

from binance_quant_control.analysis import enrich_indicators, prepare_klines_frame
from binance_quant_control.asset_routing import resolve_symbol_route
from binance_quant_control.backtest import simulate_backtest
from binance_quant_control.config import REPORTS_DIR, ensure_runtime_dirs, load_settings
from binance_quant_control.convergence import ConvergenceMetrics, evaluate_convergence
from binance_quant_control.strategy import load_strategy_config

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "NEARUSDT",
    "AVAXUSDT",
    "APEUSDT",
    "AXSUSDT",
    "TRUMPUSDT",
    "HYPERUSDT",
    "API3USDT",
    "SOONUSDT",
    "1000PEPEUSDT",
    "SUIUSDT",
    "ORCAUSDT",
    "PENGUUSDT",
]
INTERVALS = ["1h", "4h"]
TOTAL_CANDLES = 1500
PAGE_SIZE = 1000

LONG_THRESHOLDS = [60, 65, 70, 75]
MIN_CONVERGENCES = [0.6, 0.7]
ATR_MULTIPLES = [1.8, 2.5]
TP_MULTIPLES = [0.5, 0.75, 1.0, 1.5]
WALK_FORWARD_WINDOWS = 3
MIN_WINDOW_CANDLES = 240


def base_url(market: str, use_testnet: bool = False) -> str:
    if market == "spot":
        return "https://testnet.binance.vision" if use_testnet else "https://api.binance.com"
    return "https://demo-fapi.binance.com" if use_testnet else "https://fapi.binance.com"


def fetch_klines_paginated(market: str, symbol: str, interval: str, total: int = TOTAL_CANDLES) -> list[list[Any]]:
    url = f"{base_url(market)}/fapi/v1/klines" if market == "futures" else f"{base_url(market)}/api/v3/klines"
    client = httpx.Client(timeout=30.0, headers={"User-Agent": "Mozilla/5.0"})
    all_rows: list[list[Any]] = []
    end_time: int | None = None
    try:
        while len(all_rows) < total:
            params = {"symbol": symbol, "interval": interval, "limit": min(PAGE_SIZE, total - len(all_rows))}
            if end_time is not None:
                params["endTime"] = end_time
            resp = client.get(url, params=params)
            resp.raise_for_status()
            batch = resp.json()
            if not isinstance(batch, list) or not batch:
                break
            # prepend older candles and keep chronological order
            all_rows = batch + all_rows
            first_open = int(batch[0][0])
            end_time = first_open - 1
            if len(batch) < params["limit"]:
                break
        if not all_rows:
            raise RuntimeError(f"No klines returned for {symbol} {interval}")
        return all_rows[-total:]
    finally:
        client.close()


def trade_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    trades = summary.get("trades", [])
    gross_profit = sum(t["pnl_pct"] for t in trades if t["pnl_pct"] > 0)
    gross_loss = abs(sum(t["pnl_pct"] for t in trades if t["pnl_pct"] < 0))
    profit_factor = round(gross_profit / gross_loss, 4) if gross_loss else (float("inf") if gross_profit else 0.0)
    loss_count = sum(1 for t in trades if t["pnl_pct"] <= 0)
    return {
        "profit_factor": profit_factor,
        "loss_count": loss_count,
        "all_trades_positive": loss_count == 0 and len(trades) > 0,
    }


def run_one(df, market: str, strategy) -> dict[str, Any]:
    summary = simulate_backtest(df, market, strategy)
    metrics = trade_metrics(summary)
    out = {**summary, **metrics}
    return out


def walk_forward_slices(df, windows: int = WALK_FORWARD_WINDOWS) -> list[tuple[int, int]]:
    if len(df) < MIN_WINDOW_CANDLES:
        return []
    window_size = len(df) // windows
    if window_size < MIN_WINDOW_CANDLES:
        return []
    slices: list[tuple[int, int]] = []
    for idx in range(windows):
        start = idx * window_size
        end = len(df) if idx == windows - 1 else (idx + 1) * window_size
        if end - start >= MIN_WINDOW_CANDLES:
            slices.append((start, end))
    return slices


def evaluate_walk_forward(df, market: str, strategy) -> dict[str, Any]:
    slices = walk_forward_slices(df)
    window_results: list[dict[str, Any]] = []
    for window_number, (start, end) in enumerate(slices, start=1):
        window_df = df.iloc[start:end].copy()
        window_summary = run_one(window_df, market, strategy)
        window_results.append(
            {
                "window": window_number,
                "start_index": start,
                "end_index": end,
                "trade_count": window_summary["trade_count"],
                "wins": window_summary["wins"],
                "losses": window_summary["losses"],
                "win_rate": window_summary["win_rate"],
                "profit_factor": window_summary["profit_factor"],
                "all_trades_positive": window_summary["all_trades_positive"],
                "total_return_pct": window_summary["total_return_pct"],
                "max_drawdown_pct": window_summary["max_drawdown_pct"],
            }
        )
    return {
        "window_count": len(window_results),
        "windows": window_results,
        "all_windows_positive": bool(window_results) and all(item["all_trades_positive"] for item in window_results),
        "all_windows_profitable": bool(window_results) and all(item["total_return_pct"] > 0 for item in window_results),
        "mean_window_win_rate": round(mean(item["win_rate"] for item in window_results), 2) if window_results else 0.0,
        "mean_window_profit_factor": round(
            mean(float(item["profit_factor"]) if item["profit_factor"] != float("inf") else 9999.0 for item in window_results),
            4,
        ) if window_results else 0.0,
    }


def main() -> None:
    ensure_runtime_dirs()
    settings = load_settings()
    base_strategy = load_strategy_config("config/strategy-live-pilot.yaml")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = REPORTS_DIR / f"{run_id}-large-backtest-sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[tuple[str, str], Any] = {}
    fetch_log: list[dict[str, Any]] = []

    for symbol, interval in itertools.product(SYMBOLS, INTERVALS):
        rows = fetch_klines_paginated("futures", symbol, interval, TOTAL_CANDLES)
        df = enrich_indicators(prepare_klines_frame(rows), interval, strategy=base_strategy)
        datasets[(symbol, interval)] = df
        fetch_log.append({"symbol": symbol, "interval": interval, "candles": len(df)})

    results: list[dict[str, Any]] = []

    for symbol, interval in itertools.product(SYMBOLS, INTERVALS):
        df = datasets[(symbol, interval)]
        split_idx = max(int(len(df) * 0.7), 250)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        for long_threshold, min_conv, atr_mult, tp_mult in itertools.product(
            LONG_THRESHOLDS, MIN_CONVERGENCES, ATR_MULTIPLES, TP_MULTIPLES
        ):
            short_threshold = 100 - long_threshold
            strategy = dataclasses.replace(
                base_strategy,
                risk=dataclasses.replace(
                    base_strategy.risk,
                    min_convergence=min_conv,
                    min_score_long=long_threshold,
                    max_score_short=short_threshold,
                    atr_stop_multiple=atr_mult,
                    take_profit_r_multiples=(tp_mult, max(tp_mult * 2.0, 2.0), max(tp_mult * 3.0, 3.0)),
                ),
            )
            full = run_one(df, "futures", strategy)
            train = run_one(train_df, "futures", strategy)
            test = run_one(test_df, "futures", strategy)
            walk_forward = evaluate_walk_forward(df, "futures", strategy)
            route = resolve_symbol_route(symbol)
            convergence = evaluate_convergence(
                ConvergenceMetrics(
                    trade_count=int(test["trade_count"]),
                    win_rate=float(test["win_rate"]),
                    profit_factor=float(test["profit_factor"]),
                    max_drawdown_pct=float(test["max_drawdown_pct"]),
                    loss_streak=int(test["loss_count"]),
                ),
                route.validation,
            )
            results.append(
                {
                    "symbol": symbol,
                    "route_id": route.route_id,
                    "asset_class": route.asset_class,
                    "interval": interval,
                    "params": {
                        "min_score_long": long_threshold,
                        "max_score_short": short_threshold,
                        "min_convergence": min_conv,
                        "atr_stop_multiple": atr_mult,
                        "primary_tp_multiple": tp_mult,
                    },
                    "full": {k: full[k] for k in ["trade_count", "wins", "losses", "win_rate", "avg_pnl_pct", "avg_r", "ending_equity", "total_return_pct", "max_drawdown_pct", "profit_factor", "all_trades_positive"]},
                    "train": {k: train[k] for k in ["trade_count", "wins", "losses", "win_rate", "avg_pnl_pct", "avg_r", "ending_equity", "total_return_pct", "max_drawdown_pct", "profit_factor", "all_trades_positive"]},
                    "test": {k: test[k] for k in ["trade_count", "wins", "losses", "win_rate", "avg_pnl_pct", "avg_r", "ending_equity", "total_return_pct", "max_drawdown_pct", "profit_factor", "all_trades_positive"]},
                    "walk_forward": walk_forward,
                    "convergence": convergence,
                }
            )

    # Rank by strictness first: zero losses on test, then higher win rate, PF, trade count.
    ranked = sorted(
        results,
        key=lambda r: (
            not r["test"]["all_trades_positive"],
            -r["test"]["win_rate"],
            -float(r["test"]["profit_factor"]) if r["test"]["profit_factor"] != float("inf") else float("-inf"),
            -r["test"]["trade_count"],
            -r["test"]["total_return_pct"],
        ),
    )

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked:
        by_dataset[f"{row['symbol']} {row['interval']}"] .append(row)

    best_per_dataset: list[dict[str, Any]] = []
    for dataset_key, rows in by_dataset.items():
        best_per_dataset.append({"dataset": dataset_key, "best": rows[0]})

    zero_loss_test = [r for r in ranked if r["test"]["all_trades_positive"]]
    zero_loss_full = [r for r in ranked if r["full"]["all_trades_positive"]]
    zero_loss_walk_forward = [r for r in ranked if r["walk_forward"]["all_windows_positive"]]

    aggregate = {
        "datasets": len(datasets),
        "configs_tested": len(results),
        "zero_loss_test_count": len(zero_loss_test),
        "zero_loss_full_count": len(zero_loss_full),
        "zero_loss_walk_forward_count": len(zero_loss_walk_forward),
        "best_test": ranked[0] if ranked else None,
        "best_full": max(results, key=lambda r: (r["full"]["win_rate"], r["full"]["profit_factor"], r["full"]["trade_count"], r["full"]["total_return_pct"]), default=None),
        "best_walk_forward": max(results, key=lambda r: (r["walk_forward"]["mean_window_win_rate"], r["walk_forward"]["mean_window_profit_factor"], r["walk_forward"]["window_count"]), default=None),
        "mean_test_win_rate": round(mean(r["test"]["win_rate"] for r in results), 2) if results else 0.0,
        "mean_test_profit_factor": round(mean(float(r["test"]["profit_factor"]) if r["test"]["profit_factor"] != float("inf") else 9999.0 for r in results), 4) if results else 0.0,
        "mean_walk_forward_window_win_rate": round(mean(r["walk_forward"]["mean_window_win_rate"] for r in results), 2) if results else 0.0,
        "promotion_decision_counts": {
            "elite_candidate": sum(1 for r in results if r["convergence"]["promotion_decision"] == "elite_candidate"),
            "promote": sum(1 for r in results if r["convergence"]["promotion_decision"] == "promote"),
            "watchlist": sum(1 for r in results if r["convergence"]["promotion_decision"] == "watchlist"),
            "reject": sum(1 for r in results if r["convergence"]["promotion_decision"] == "reject"),
        },
    }

    report = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "settings": {
            "use_testnet": settings.use_testnet,
            "live_trading_enabled": settings.live_trading_enabled,
        },
        "universe": SYMBOLS,
        "intervals": INTERVALS,
        "total_candles": TOTAL_CANDLES,
        "grid": {
            "long_thresholds": LONG_THRESHOLDS,
            "min_convergences": MIN_CONVERGENCES,
            "atr_multiples": ATR_MULTIPLES,
            "tp_multiples": TP_MULTIPLES,
        },
        "fetch_log": fetch_log,
        "aggregate": aggregate,
        "best_per_dataset": best_per_dataset,
        "ranked": ranked[:40],
    }

    report_path = output_dir / "sweep.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), "report_path": str(report_path), "aggregate": aggregate}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
