from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import STATE_DIR, ensure_runtime_dirs
from .signal_schema import TradingSignal

SIGNAL_LEDGER_FILE = STATE_DIR / "signals" / "trading-signals.jsonl"


def append_trading_signal(
    signal: TradingSignal,
    *,
    gate: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> Path:
    ensure_runtime_dirs()
    target = Path(path).expanduser().resolve() if path else SIGNAL_LEDGER_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema": "binance_quant_control.trading_signal.v1",
        "signal": signal.to_dict(),
        "gate": gate or {},
        "sync": {
            "dashboard_ready": True,
            "copy_trading_ready": False,
            "api_ready": True,
        },
    }
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return target


def read_trading_signals(path: str | Path | None = None, *, limit: int = 0) -> list[dict[str, Any]]:
    target = Path(path).expanduser().resolve() if path else SIGNAL_LEDGER_FILE
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:] if limit > 0 else rows


def signal_api_contract() -> dict[str, Any]:
    return {
        "schema": "binance_quant_control.trading_signal.v1",
        "transport": "local_jsonl_now_openapi_later",
        "ledger_path": str(SIGNAL_LEDGER_FILE),
        "required_fields": [
            "signal_id",
            "symbol",
            "side",
            "interval",
            "strategy_family",
            "route_id",
            "status",
            "expectancy_r",
            "payoff_ratio",
            "profit_factor",
            "trade_count",
            "blockers",
        ],
        "copy_sync_boundary": "blocked until live-readiness and execution approval pass",
    }
