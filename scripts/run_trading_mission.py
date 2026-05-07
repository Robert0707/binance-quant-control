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
    from binance_quant_control.mission_control import (
        DEFAULT_MISSION_CONFIG_PATH,
        run_trading_mission,
    )

    parser = argparse.ArgumentParser(
        description="Run one-command mission control for strategy convergence and optional live entry."
    )
    parser.add_argument("--symbols", required=True, help="Comma-separated symbol list.")
    parser.add_argument("--target-return-pct", type=float, default=0.0)
    parser.add_argument("--max-leverage", type=float, default=3.0)
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--config", default=str(DEFAULT_MISSION_CONFIG_PATH))
    args = parser.parse_args()

    payload = run_trading_mission(
        symbols=[item.strip() for item in args.symbols.split(",") if item.strip()],
        target_return_pct=args.target_return_pct,
        max_leverage=args.max_leverage,
        execute_live=args.execute_live,
        config_path=args.config,
    )
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "report_path": payload.get("report_path"),
                "selected_candidate": payload.get("selected_candidate"),
                "simulation": payload.get("simulation"),
                "live": payload.get("live"),
                "system_findings": payload.get("system_findings"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
