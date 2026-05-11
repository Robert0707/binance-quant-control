from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = PROJECT_ROOT / "state" / "offline-risk-validation"
BINANCE_QUANT = PROJECT_ROOT / ".venv" / "bin" / "binance-quant-control"


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
    return pid > 0 and Path(f"/proc/{pid}").exists()


def _child_processes(pid: int) -> list[dict[str, str]]:
    if pid <= 0:
        return []
    completed = subprocess.run(
        ["ps", "--ppid", str(pid), "-o", "pid=,stat=,etime=,pcpu=,pmem=,cmd="],
        text=True,
        capture_output=True,
        check=False,
    )
    children: list[dict[str, str]] = []
    for line in (completed.stdout or "").splitlines():
        parts = line.strip().split(None, 5)
        if len(parts) < 6:
            continue
        children.append(
            {
                "pid": parts[0],
                "stat": parts[1],
                "elapsed": parts[2],
                "pcpu": parts[3],
                "pmem": parts[4],
                "cmd": parts[5],
            }
        )
    return children


def _latest_run() -> Path | None:
    if not STATE_DIR.exists():
        return None
    runs = [path for path in STATE_DIR.iterdir() if path.is_dir()]
    return max(runs, key=lambda path: path.name) if runs else None


def _sweep_command(args: argparse.Namespace) -> list[str]:
    return [
        str(BINANCE_QUANT),
        "risk-combo-sweep",
        "--symbols",
        args.symbols,
        "--target-side",
        args.target_side,
        "--target-interval",
        args.target_interval,
        "--limit",
        str(args.limit),
        "--grid-mode",
        args.grid_mode,
        "--min-test-trades",
        str(args.min_test_trades),
        "--target-profit-factor",
        str(args.target_profit_factor),
        "--min-expectancy-r",
        str(args.min_expectancy_r),
        "--max-stop-loss-ratio",
        str(args.max_stop_loss_ratio),
        "--max-configs",
        str(args.max_configs),
        "--max-walk-forward-validations",
        str(args.max_walk_forward_validations),
        "--top-n",
        str(args.top_n),
        "--skip-news",
        "--compact",
    ]


def _worker_script(run_dir: Path, sweep_command: list[str], latest_sweeps: int, step_timeout_seconds: int) -> str:
    summary_path = run_dir / "summary.json"
    progress_path = run_dir / "progress.json"
    return f"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

project_root = Path({str(PROJECT_ROOT)!r})
run_dir = Path({str(run_dir)!r})
binary = Path({str(BINANCE_QUANT)!r})
summary_path = Path({str(summary_path)!r})
progress_path = Path({str(progress_path)!r})


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_progress(name, command, status):
    progress_path.write_text(json.dumps({{
        "updated_at": now_iso(),
        "step": name,
        "status": status,
        "command": command,
        "opens_orders": False,
        "writes_execution_config": False,
        "mainnet_live_allowed": False,
    }}, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")


def run_step(name, command):
    write_progress(name, command, "running")
    output_path = run_dir / f"{{name}}.stdout.log"
    error_path = run_dir / f"{{name}}.stderr.log"
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
            timeout={step_timeout_seconds},
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\\ntimeout after {step_timeout_seconds} seconds"
        returncode = 124
    output_path.write_text(stdout, encoding="utf-8")
    error_path.write_text(stderr, encoding="utf-8")
    response = None
    if stdout:
        lines = [line for line in stdout.splitlines() if line.strip()]
        if lines:
            try:
                response = json.loads(lines[-1])
            except json.JSONDecodeError:
                response = None
    step = {{
        "name": name,
        "command": command,
        "returncode": returncode,
        "stdout_path": str(output_path),
        "stderr_path": str(error_path),
        "timed_out": timed_out,
        "response": response,
    }}
    write_progress(name, command, "ok" if returncode == 0 else ("timeout" if timed_out else "error"))
    return step


steps = []
sweep_command = {sweep_command!r}
steps.append(run_step("risk_combo_sweep", sweep_command))
steps.append(run_step("risk_combo_matrix", [
    str(binary), "risk-combo-matrix", "--latest-sweeps", {str(latest_sweeps)!r}, "--compact"
]))
steps.append(run_step("ai_readiness_scan", [
    str(binary), "ai-readiness-scan", "--execution-mode", "testnet_exploration",
    "--max-candidates", "6", "--compact"
]))

status = "ok" if all(step["returncode"] == 0 for step in steps) else "error"
summary = {{
    "status": status,
    "started_at": None,
    "finished_at": now_iso(),
    "run_dir": str(run_dir),
    "opens_orders": False,
    "writes_execution_config": False,
    "mainnet_live_allowed": False,
    "steps": steps,
}}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
raise SystemExit(0 if status == "ok" else 1)
"""


def start(args: argparse.Namespace) -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = STATE_DIR / f"{_stamp()}-{args.symbols.lower().replace(',', '-')}-{args.target_side.lower()}-{args.target_interval}"
    run_dir.mkdir(parents=True, exist_ok=False)
    sweep_command = _sweep_command(args)
    worker_path = run_dir / "worker.py"
    log_path = run_dir / "run.log"
    worker_path.write_text(
        _worker_script(run_dir, sweep_command, args.latest_sweeps, args.step_timeout_seconds),
        encoding="utf-8",
    )
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(worker_path)],
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
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "worker_path": str(worker_path),
        "sweep_command": sweep_command,
        "latest_sweeps": args.latest_sweeps,
        "step_timeout_seconds": args.step_timeout_seconds,
        "opens_orders": False,
        "writes_execution_config": False,
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
    summary_path = run_dir / "summary.json"
    payload = {
        **metadata,
        "running": _is_running(pid),
        "summary_path": str(summary_path),
        "summary_exists": summary_path.exists(),
        "summary": None,
        "progress": None,
        "child_processes": _child_processes(pid),
        "log_tail": "",
    }
    progress_path = run_dir / "progress.json"
    if progress_path.exists():
        payload["progress"] = _read_metadata(progress_path)
    if summary_path.exists():
        payload["summary"] = _read_metadata(summary_path)
        payload["status"] = str((payload["summary"] or {}).get("status") or "completed")
    elif payload["running"]:
        payload["status"] = "running"
    else:
        payload["status"] = "exited_without_summary"
    log_path = Path(str(metadata.get("log_path") or run_dir / "run.log"))
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        payload["log_tail"] = "\n".join(lines[-10:])
    print(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline risk-combo validation without opening orders.")
    sub = parser.add_subparsers(dest="action", required=True)

    start_parser = sub.add_parser("start")
    start_parser.add_argument("--symbols", default="TRXUSDT")
    start_parser.add_argument("--target-side", choices=("BUY", "SELL"), default="BUY")
    start_parser.add_argument("--target-interval", default="1d")
    start_parser.add_argument("--limit", type=int, default=5000)
    start_parser.add_argument("--grid-mode", choices=("fast", "focused", "standard"), default="focused")
    start_parser.add_argument("--min-test-trades", type=int, default=30)
    start_parser.add_argument("--target-profit-factor", type=float, default=1.0)
    start_parser.add_argument("--min-expectancy-r", type=float, default=0.0)
    start_parser.add_argument("--max-stop-loss-ratio", type=float, default=55.0)
    start_parser.add_argument("--max-configs", type=int, default=40)
    start_parser.add_argument("--max-walk-forward-validations", type=int, default=6)
    start_parser.add_argument("--top-n", type=int, default=10)
    start_parser.add_argument("--latest-sweeps", type=int, default=12)
    start_parser.add_argument("--step-timeout-seconds", type=int, default=7200)

    sub.add_parser("status")
    args = parser.parse_args(argv)
    if args.action == "start":
        return start(args)
    return status(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
