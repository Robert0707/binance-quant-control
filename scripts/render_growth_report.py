#!/usr/bin/env python3
"""Render an equity-growth report from challenge and snapshot state."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

from binance_quant_control.challenge import (
    challenge_scope_key,
    load_challenge_state,
    read_balance_snapshots,
)
from binance_quant_control.config import PROJECT_ROOT, REPORTS_DIR
from binance_quant_control.order_journal import summarize_live_orders
from binance_quant_control.strategy import load_strategy_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render challenge growth report")
    parser.add_argument(
        "--strategy-config",
        default="config/strategy-stable-risk.yaml",
        help="Strategy config used to scope the challenge state",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to reports/<timestamp>-growth-report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategy = load_strategy_config(args.strategy_config)
    scope = challenge_scope_key(strategy.profile, strategy.defaults.symbol, strategy.defaults.market)
    state = load_challenge_state(scope)
    snapshots = read_balance_snapshots()

    scoped = [
        item
        for item in snapshots
        if item.get("market") == strategy.defaults.market
    ]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else REPORTS_DIR / f"{timestamp}-growth-report"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    xs: list[datetime] = []
    ys: list[float] = []
    drawdowns: list[float] = []
    peak = 0.0
    for item in scoped:
        try:
            stamp = datetime.fromisoformat(str(item["timestamp"]))
            equity = float(item["equity_usdt"])
        except Exception:
            continue
        xs.append(stamp)
        ys.append(equity)
        peak = max(peak, equity)
        drawdowns.append(0.0 if peak <= 0 else (peak - equity) / peak * 100.0)

    summary = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "strategy_profile": strategy.profile,
        "symbol": strategy.defaults.symbol,
        "market": strategy.defaults.market,
        "interval": strategy.defaults.interval,
        "challenge": asdict(state),
        "snapshot_count": len(xs),
        "start_equity_usdt": ys[0] if ys else None,
        "latest_equity_usdt": ys[-1] if ys else None,
        "peak_equity_usdt": max(ys) if ys else None,
        "max_drawdown_pct": max(drawdowns) if drawdowns else None,
        "live_orders": summarize_live_orders(),
        "doctrine_reference": str(
            PROJECT_ROOT / "docs" / "workflows" / "github-inspired-trading-doctrine.md"
        ),
    }

    chart_path = output_dir / "growth.png"
    if xs and ys:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(xs, ys, color="#1f77b4", linewidth=2)
        axes[0].set_title(f"Equity Curve - {strategy.profile}")
        axes[0].set_ylabel("USDT")
        has_legend = False
        if state.target_balance_usdt > 0:
            axes[0].axhline(state.target_balance_usdt, color="#2ca02c", linestyle="--", label="target")
            has_legend = True
        if state.stop_balance_usdt > 0:
            axes[0].axhline(state.stop_balance_usdt, color="#d62728", linestyle="--", label="stop")
            has_legend = True
        if has_legend:
            axes[0].legend(loc="best")

        axes[1].plot(xs, drawdowns, color="#ff7f0e", linewidth=2)
        axes[1].set_title("Drawdown")
        axes[1].set_ylabel("%")
        axes[1].set_xlabel("Timestamp")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(chart_path, dpi=160)
        plt.close(fig)
        summary["chart_path"] = str(chart_path)
    else:
        summary["chart_path"] = None
        summary["note"] = "No usable balance snapshots were available for the requested scope."

    json_path = output_dir / "growth-report.json"
    md_path = output_dir / "growth-report.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        f"# Growth Report - {strategy.profile}",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Symbol: {strategy.defaults.symbol}",
        f"- Market: {strategy.defaults.market}",
        f"- Interval: {strategy.defaults.interval}",
        f"- Challenge status: {state.status}",
        f"- Snapshot count: {summary['snapshot_count']}",
        f"- Start equity: {summary['start_equity_usdt']}",
        f"- Latest equity: {summary['latest_equity_usdt']}",
        f"- Peak equity: {summary['peak_equity_usdt']}",
        f"- Max drawdown pct: {summary['max_drawdown_pct']}",
        f"- Live order count: {summary['live_orders']['count']}",
        "",
        "## Context",
        "",
        f"- Doctrine: `{summary['doctrine_reference']}`",
        f"- Chart: `{summary['chart_path']}`",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "report": str(json_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
