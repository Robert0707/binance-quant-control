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
    from binance_quant_control.strategy_optimizer import (
        DEFAULT_CONFIG_PATH,
        run_strategy_optimizer,
    )

    parser = argparse.ArgumentParser(
        description="Auto-tune strategy parameters from closed-trade reviews."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    payload = run_strategy_optimizer(args.config)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "review_count": payload.get("review_count"),
                "screening_status": payload.get("screening_status"),
                "validation_status": payload.get("validation_status"),
                "elite_status": payload.get("elite_status"),
                "promotion_decision": payload.get("promotion_decision"),
                "report_path": payload.get("report_path"),
                "output_strategy_config": payload.get("output_strategy_config"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
