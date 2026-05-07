from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    strategy_path = PROJECT_ROOT / "config" / "strategy-stable-risk.yaml"
    command = [
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        "-m",
        "binance_quant_control.cli",
        "stability-workflow",
        "--strategy-config",
        str(strategy_path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.stdout:
        print(completed.stdout.strip())
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr.strip(), file=sys.stderr)
        raise SystemExit(completed.returncode)

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit("stability workflow returned invalid JSON") from exc
    if payload.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
