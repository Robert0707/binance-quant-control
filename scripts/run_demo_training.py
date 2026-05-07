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
    from binance_quant_control.training import run_demo_training

    parser = argparse.ArgumentParser(
        description="Run replay-backed Binance demo training rounds and auto-write closed-trade review samples."
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--symbols", default="", help="Comma-separated symbol list.")
    parser.add_argument("--target-return-pct", type=float, default=5.0)
    parser.add_argument("--max-leverage", type=float, default=3.0)
    parser.add_argument("--margin-notional-usdt", type=float, default=4.0)
    parser.add_argument("--optimize-every", type=int, default=10)
    args = parser.parse_args()

    payload = run_demo_training(
        rounds=args.rounds,
        symbols=[item.strip() for item in args.symbols.split(",") if item.strip()] if args.symbols else None,
        target_return_pct=args.target_return_pct,
        max_leverage=args.max_leverage,
        margin_notional_usdt=args.margin_notional_usdt,
        optimize_every=args.optimize_every,
    )
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "mode": payload.get("mode"),
                "rounds_requested": payload.get("rounds_requested"),
                "recorded_review_count": payload.get("recorded_review_count"),
                "wins": payload.get("wins"),
                "losses": payload.get("losses"),
                "total_realized_pnl_usdt": payload.get("total_realized_pnl_usdt"),
                "findings": payload.get("findings"),
                "optimizer_reports": payload.get("optimizer_reports"),
                "report_path": payload.get("report_path"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
