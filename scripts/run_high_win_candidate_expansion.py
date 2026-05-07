from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = PROJECT_ROOT / "state" / "high-win-candidate-expansion"
DEFAULT_SYMBOLS = "NAORISUSDT,APEUSDT"
DEFAULT_INTERVALS = "1h,4h"
DEFAULT_LIMIT = 26500


def _stamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")


def _metadata_path(run_dir: Path) -> Path:
    return run_dir / "metadata.json"


def _read_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    return Path(f"/proc/{pid}").exists()


def _latest_run() -> Path | None:
    if not STATE_DIR.exists():
        return None
    runs = [path for path in STATE_DIR.iterdir() if path.is_dir()]
    return max(runs, key=lambda path: path.name) if runs else None


def _command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    return [
        str(PROJECT_ROOT / ".venv" / "bin" / "binance-quant-control"),
        "alpha-research",
        "--config",
        "config/core-replacement-scout.default.yaml",
        "--symbols",
        args.symbols,
        "--intervals",
        args.intervals,
        "--limit",
        str(args.limit),
        "--output-dir",
        str(run_dir / "output"),
        "--compact",
    ]


def start(args: argparse.Namespace) -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = STATE_DIR / _stamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    command = _command(args, run_dir)
    log_path = run_dir / "run.log"
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    log_file.close()
    metadata = {
        "started_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "pid": process.pid,
        "command": command,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "output_dir": str(run_dir / "output"),
        "symbols": args.symbols,
        "intervals": args.intervals,
        "limit": args.limit,
        "opens_orders": False,
        "mainnet_live_allowed": False,
        "status": "running",
    }
    _metadata_path(run_dir).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, separators=(",", ":")))
    return 0


def status(_args: argparse.Namespace) -> int:
    run_dir = _latest_run()
    if run_dir is None:
        print(json.dumps({"status": "not_started"}, separators=(",", ":")))
        return 0
    metadata = _read_metadata(_metadata_path(run_dir))
    pid = int(metadata.get("pid") or 0)
    report_path = run_dir / "output" / "alpha-research-ranking.json"
    payload = {
        **metadata,
        "running": _is_running(pid),
        "report_path": str(report_path) if report_path.exists() else "",
        "report_exists": report_path.exists(),
        "log_tail": "",
    }
    log_path = Path(str(metadata.get("log_path") or run_dir / "run.log"))
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        payload["log_tail"] = "\n".join(lines[-5:])
    if not payload["running"] and not payload["report_exists"]:
        payload["status"] = "exited_without_report"
    elif payload["report_exists"]:
        payload["status"] = "report_ready"
    else:
        payload["status"] = "running"
    print(json.dumps(payload, separators=(",", ":")))
    return 0


def evaluate(_args: argparse.Namespace) -> int:
    run_dir = _latest_run()
    if run_dir is None:
        print(json.dumps({"status": "not_started"}, separators=(",", ":")))
        return 1
    report_path = run_dir / "output" / "alpha-research-ranking.json"
    if not report_path.exists():
        print(
            json.dumps(
                {"status": "report_not_ready", "run_dir": str(run_dir), "report_path": str(report_path)},
                separators=(",", ":"),
            )
        )
        return 2
    command = [
        str(PROJECT_ROOT / ".venv" / "bin" / "binance-quant-control"),
        "high-win-iteration",
        "--alpha-report",
        ",".join(
            [
                "state/core-10-trend-pullback-80-20-l5000/alpha-research-ranking.json",
                "state/replacement-scout-trend-pullback-80-20-l5000/alpha-research-ranking.json",
                str(report_path),
            ]
        ),
        "--no-write-pid-state",
        "--compact",
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr.strip(), file=sys.stderr)
        print(completed.stdout.strip())
        return completed.returncode
    print(completed.stdout.strip())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run focused high-win candidate expansion safely.")
    parser.add_argument("action", choices=("start", "status", "evaluate"))
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--intervals", default=DEFAULT_INTERVALS)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args(argv)
    if args.action == "start":
        return start(args)
    if args.action == "evaluate":
        return evaluate(args)
    return status(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
