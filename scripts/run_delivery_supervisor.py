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
    from binance_quant_control.supervision import build_supervisor_policy, run_delivery_supervisor

    parser = argparse.ArgumentParser(
        description="Run risk-first paper/demo delivery supervision cycles."
    )
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--training-rounds", type=int, default=10)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--mission-symbols-per-cycle", type=int, default=6)
    parser.add_argument("--target-return-pct", type=float, default=5.0)
    parser.add_argument("--max-leverage", type=float, default=3.0)
    parser.add_argument("--margin-notional-usdt", type=float, default=3.0)
    parser.add_argument("--optimize-every", type=int, default=10)
    parser.add_argument("--max-recent-loss-usdt", type=float, default=5.0)
    parser.add_argument("--max-route-loss-streak", type=int, default=5)
    parser.add_argument("--min-route-profit-factor", type=float, default=0.8)
    parser.add_argument("--route-lookback", type=int, default=40)
    parser.add_argument("--build-digest-every", type=int, default=1)
    parser.add_argument("--audit-every", type=int, default=1)
    parser.add_argument("--no-stop-on-optimizer-promotion", action="store_true")
    args = parser.parse_args()

    payload = run_delivery_supervisor(
        build_supervisor_policy(
            cycles=args.cycles,
            training_rounds=args.training_rounds,
            symbols=[item.strip() for item in args.symbols.split(",") if item.strip()]
            if args.symbols
            else None,
            mission_symbols_per_cycle=args.mission_symbols_per_cycle,
            target_return_pct=args.target_return_pct,
            max_leverage=args.max_leverage,
            margin_notional_usdt=args.margin_notional_usdt,
            optimize_every=args.optimize_every,
            max_recent_loss_usdt=args.max_recent_loss_usdt,
            max_route_loss_streak=args.max_route_loss_streak,
            min_route_profit_factor=args.min_route_profit_factor,
            route_lookback=args.route_lookback,
            build_digest_every=args.build_digest_every,
            audit_every=args.audit_every,
            stop_on_optimizer_promotion=not args.no_stop_on_optimizer_promotion,
        )
    )
    latest_cycle = payload["cycles"][-1] if payload.get("cycles") else {}
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "mode": payload.get("mode"),
                "cycles_completed": payload.get("cycles_completed"),
                "stop_reasons": payload.get("stop_reasons"),
                "latest_training": (latest_cycle.get("training") or {}).get("response"),
                "latest_optimizer": latest_cycle.get("optimizer"),
                "latest_database": latest_cycle.get("database"),
                "quarantined_routes": ((latest_cycle.get("route_risk") or {}).get("quarantined_routes")),
                "report_path": payload.get("report_path"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
