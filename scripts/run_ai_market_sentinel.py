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

from binance_quant_control.ai_market_sentinel import run_ai_market_sentinel  # noqa: E402


def _split_symbols(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI market sentinel once.")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,DOGEUSDT,TRXUSDT")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--limit", type=int, default=160)
    parser.add_argument("--market", choices=("spot", "futures"), default="futures")
    parser.add_argument("--skip-readiness", action="store_true")
    parser.add_argument("--send-telegram", action="store_true")
    parser.add_argument("--max-readiness-candidates", type=int, default=6)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    payload = run_ai_market_sentinel(
        symbols=_split_symbols(args.symbols),
        interval=args.interval,
        limit=args.limit,
        market=args.market,
        skip_readiness=bool(args.skip_readiness),
        send_telegram=bool(args.send_telegram),
        max_readiness_candidates=args.max_readiness_candidates,
    )
    if args.compact:
        payload = {
            "mode": payload.get("mode"),
            "position_state": payload.get("position_state"),
            "expansion_gate": payload.get("expansion_gate"),
            "conditional_order_alert": payload.get("conditional_order_alert"),
            "telegram": payload.get("telegram"),
            "machine_action_queue": payload.get("machine_action_queue"),
            "errors": payload.get("errors"),
            "report_path": payload.get("report_path"),
        }
    print(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) if args.compact else json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
