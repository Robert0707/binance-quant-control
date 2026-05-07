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
    from binance_quant_control.public_history_training import run_public_history_training

    parser = argparse.ArgumentParser(
        description="Import Binance public historical klines into strategy training reviews."
    )
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,PAXGUSDT")
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--end-month", default="")
    parser.add_argument("--max-reviews-per-symbol", type=int, default=100)
    parser.add_argument("--optimize-every", type=int, default=250)
    args = parser.parse_args()

    payload = run_public_history_training(
        symbols=[item.strip() for item in args.symbols.split(",") if item.strip()],
        months=args.months,
        end_month=(args.end_month or None),
        max_reviews_per_symbol=args.max_reviews_per_symbol,
        optimize_every=args.optimize_every,
    )
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "mode": payload.get("mode"),
                "inserted_review_count": payload.get("inserted_review_count"),
                "skipped_duplicate_count": payload.get("skipped_duplicate_count"),
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
