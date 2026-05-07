#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from _project_python import ensure_project_python  # noqa: E402

ensure_project_python(PROJECT_ROOT)


def main() -> int:
    from binance_quant_control.autonomy import DEFAULT_CONFIG_PATH, run_autonomy_cycle

    parser = argparse.ArgumentParser(
        description="Run the Binance autonomous trader controller with local-first safety gates."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    payload = run_autonomy_cycle(args.config)
    summary = {
        "status": payload.get("status"),
        "autonomy_mode": payload.get("autonomy_mode"),
        "report_path": payload.get("report_path"),
        "execution": payload.get("execution"),
        "entry_gate": payload.get("entry_gate"),
        "candidate": payload.get("candidate"),
        "candidate_route": (payload.get("candidate") or {}).get("route"),
        "live_plan": payload.get("live_plan"),
        "simulation": payload.get("simulation"),
        "digest_decision": (payload.get("digest") or {}).get("decision"),
    }
    if args.compact:
        selected = (summary.get("digest_decision") or {}).get("selected") or {}
        live_plan = summary.get("live_plan") or {}
        execution = summary.get("execution") or {}
        summary = {
            "status": payload.get("status"),
            "autonomy_mode": payload.get("autonomy_mode"),
            "symbol": (payload.get("candidate") or {}).get("symbol"),
            "side": (payload.get("candidate") or {}).get("side_override"),
            "execution_mode": live_plan.get("execution_mode"),
            "executed_mode": execution.get("mode"),
            "allowed": live_plan.get("allowed"),
            "sizing": live_plan.get("sizing"),
            "violations": live_plan.get("violations"),
            "warnings": live_plan.get("warnings"),
            "digest_action": (payload.get("entry_gate") or {}).get("digest_action"),
            "selected_adjusted_score": selected.get("adjusted_score"),
            "report_path": payload.get("report_path"),
        }
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":") if args.compact else None, indent=None if args.compact else 2))
    return 0 if payload.get("status") != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
