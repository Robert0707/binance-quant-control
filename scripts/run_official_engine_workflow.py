#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from binance_quant_control.daily_digest import build_digest, load_config

ROOT = Path("/home/robert/python")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = PROJECT_ROOT / "state" / "n8n-digests"
REPORT_ROOT = PROJECT_ROOT / "reports"
FREQTRADECTL = ROOT / "bin" / "openclaw-freqtradectl"
FREQTRADE_CONFIG = ROOT / "external" / "freqtrade" / "user_data" / "config.openclaw.json"
BACKTEST_DIR = ROOT / "external" / "freqtrade" / "user_data" / "backtest_results"


def run(cmd: list[str], *, timeout: int = 900) -> dict[str, object]:
    completed = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        cwd=str(ROOT),
    )
    return {
        "cmd": cmd,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-8000:].strip(),
        "stderr_tail": completed.stderr[-8000:].strip(),
    }


def load_whitelist(config_path: Path) -> list[str]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return list(payload.get("exchange", {}).get("pair_whitelist", []))


def latest_backtest_zip(previous: set[Path]) -> Path:
    current = sorted(BACKTEST_DIR.glob("*.zip"), key=lambda item: item.stat().st_mtime, reverse=True)
    for path in current:
        if path not in previous:
            return path
    if not current:
        raise SystemExit(f"No backtest zip files found under {BACKTEST_DIR}")
    return current[0]


def parse_backtest_zip(path: Path, strategy: str) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        json_name = next(
            name for name in archive.namelist() if name.endswith(".json") and "_config" not in name
        )
        payload = json.loads(archive.read(json_name))
    strategy_payload = payload["strategy"][strategy]
    wins = int(strategy_payload.get("wins", 0))
    losses = int(strategy_payload.get("losses", 0))
    draws = int(strategy_payload.get("draws", 0))
    total = max(1, wins + losses + draws)
    return {
        "backtest_zip": str(path),
        "strategy": strategy,
        "total_trades": int(strategy_payload.get("total_trades", 0)),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate_pct": round((wins / total) * 100, 2),
        "final_balance": float(strategy_payload.get("final_balance", 0.0)),
        "profit_total_pct": round(float(strategy_payload.get("profit_total", 0.0)) * 100, 4),
        "profit_total_abs": float(strategy_payload.get("profit_total_abs", 0.0)),
        "max_drawdown_account_pct": round(
            float(strategy_payload.get("max_drawdown_account", 0.0)) * 100, 4
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Binance digest plus official Freqtrade engine workflow."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "n8n-daily-digest.default.json"),
    )
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--strategy", default="SampleStrategy")
    args = parser.parse_args()

    generated_at = datetime.now(timezone.utc)
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    report_dir = REPORT_ROOT / f"{stamp}-official-engine-workflow"
    report_dir.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config).expanduser().resolve()
    digest_payload = build_digest(load_config(config_path))
    digest_path = STATE_DIR / f"{stamp}-daily-digest.json"
    digest_path.write_text(json.dumps(digest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    steps: dict[str, object] = {}
    failures: list[str] = []

    steps["sync_whitelist"] = run([str(FREQTRADECTL), "sync-whitelist"], timeout=120)
    if steps["sync_whitelist"]["returncode"] != 0:
        failures.append("sync_whitelist")

    whitelist = load_whitelist(FREQTRADE_CONFIG)
    if not whitelist:
        failures.append("empty_whitelist")

    if not failures:
        steps["download_data"] = run(
            [
                str(FREQTRADECTL),
                "download-data",
                "--pairs",
                *whitelist,
                "-t",
                args.timeframe,
                "--days",
                str(args.days),
            ],
            timeout=900,
        )
        if steps["download_data"]["returncode"] != 0:
            failures.append("download_data")

    existing_zips = set(BACKTEST_DIR.glob("*.zip"))
    backtest_summary: dict[str, object] | None = None
    timerange_end = generated_at.strftime("%Y%m%d")
    timerange_start = (generated_at - timedelta(days=args.days)).strftime("%Y%m%d")
    timerange = f"{timerange_start}-{timerange_end}"

    if not failures:
        steps["backtesting"] = run(
            [
                str(FREQTRADECTL),
                "backtesting",
                "--timerange",
                timerange,
                "--export",
                "trades",
            ],
            timeout=900,
        )
        if steps["backtesting"]["returncode"] != 0:
            failures.append("backtesting")
        else:
            backtest_zip = latest_backtest_zip(existing_zips)
            backtest_summary = parse_backtest_zip(backtest_zip, args.strategy)

    steps["growth_report"] = run(
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            str(PROJECT_ROOT / "scripts" / "render_growth_report.py"),
            "--strategy-config",
            str(PROJECT_ROOT / "config" / "strategy-stable-risk.yaml"),
            "--output-dir",
            str(report_dir / "growth-report"),
        ],
        timeout=180,
    )
    if steps["growth_report"]["returncode"] != 0:
        failures.append("growth_report")

    summary = {
        "generated_at": generated_at.replace(microsecond=0).isoformat(),
        "status": "ok" if not failures else "warn",
        "failures": failures,
        "digest_path": str(digest_path),
        "freqtrade_config": str(FREQTRADE_CONFIG),
        "pair_whitelist": whitelist,
        "timeframe": args.timeframe,
        "days": args.days,
        "timerange": timerange,
        "backtest_summary": backtest_summary,
        "steps": steps,
    }

    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "report_path": str(summary_path), "failures": failures, "backtest_summary": backtest_summary}, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
