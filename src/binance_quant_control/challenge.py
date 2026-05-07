from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import STATE_DIR

BALANCE_SNAPSHOTS_FILE = STATE_DIR / "balance-snapshots.jsonl"
CHALLENGE_STATE_FILE = STATE_DIR / "live-challenge.json"


@dataclass(slots=True)
class BalanceSnapshot:
    timestamp: str
    market: str
    wallet_balance_usdt: float
    available_balance_usdt: float
    unrealized_pnl_usdt: float
    equity_usdt: float
    note: str = ""


@dataclass(slots=True)
class ChallengeState:
    enabled: bool = False
    profile: str = ""
    symbol: str = ""
    market: str = "futures"
    started_at: str = ""
    start_balance_usdt: float = 0.0
    target_balance_usdt: float = 0.0
    target_multiple: float = 2.0
    max_drawdown_pct: float = 20.0
    stop_balance_usdt: float = 0.0
    highest_balance_usdt: float = 0.0
    latest_balance_usdt: float = 0.0
    latest_snapshot_at: str = ""
    status: str = "inactive"
    note: str = ""

    @property
    def progress_pct(self) -> float:
        if self.start_balance_usdt <= 0 or self.target_balance_usdt <= self.start_balance_usdt:
            return 0.0
        progress = (self.latest_balance_usdt - self.start_balance_usdt) / (
            self.target_balance_usdt - self.start_balance_usdt
        )
        return max(0.0, min(progress * 100.0, 100.0))

    @property
    def drawdown_pct(self) -> float:
        if self.highest_balance_usdt <= 0:
            return 0.0
        drawdown = (self.highest_balance_usdt - self.latest_balance_usdt) / self.highest_balance_usdt
        return max(0.0, drawdown * 100.0)

    @property
    def remaining_to_target_usdt(self) -> float:
        return max(0.0, self.target_balance_usdt - self.latest_balance_usdt)


def challenge_scope_key(profile: str, symbol: str, market: str) -> str:
    raw = f"{profile}-{symbol.upper()}-{market.lower()}"
    return "".join(char.lower() if char.isalnum() else "-" for char in raw)


def challenge_state_path(scope: str | None = None) -> Path:
    if not scope:
        return CHALLENGE_STATE_FILE
    return STATE_DIR / f"live-challenge-{scope}.json"


def _extract_futures_usdt_row(payload: Any) -> dict[str, Any]:
    for item in payload:
        if item.get("asset") == "USDT":
            return item
    return {}


def snapshot_from_balance_payload(payload: Any, market: str, *, note: str = "") -> BalanceSnapshot:
    if market != "futures":
        raise ValueError("challenge tracking currently supports futures balances only")
    row = _extract_futures_usdt_row(payload)
    wallet_balance = float(row.get("balance", 0.0))
    available_balance = float(row.get("availableBalance", wallet_balance))
    unrealized_pnl = float(row.get("crossUnPnl", 0.0))
    equity = wallet_balance + unrealized_pnl
    return BalanceSnapshot(
        timestamp=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        market=market,
        wallet_balance_usdt=wallet_balance,
        available_balance_usdt=available_balance,
        unrealized_pnl_usdt=unrealized_pnl,
        equity_usdt=equity,
        note=note,
    )


def append_balance_snapshot(snapshot: BalanceSnapshot) -> Path:
    BALANCE_SNAPSHOTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with BALANCE_SNAPSHOTS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(snapshot), ensure_ascii=False) + "\n")
    return BALANCE_SNAPSHOTS_FILE


def read_balance_snapshots(limit: int = 0) -> list[dict[str, Any]]:
    if not BALANCE_SNAPSHOTS_FILE.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in BALANCE_SNAPSHOTS_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit > 0:
        return records[-limit:]
    return records


def load_challenge_state(scope: str | None = None) -> ChallengeState:
    path = challenge_state_path(scope)
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return ChallengeState(
                enabled=bool(raw.get("enabled", False)),
                profile=str(raw.get("profile", "")),
                symbol=str(raw.get("symbol", "")),
                market=str(raw.get("market", "futures")),
                started_at=str(raw.get("started_at", "")),
                start_balance_usdt=float(raw.get("start_balance_usdt", 0.0)),
                target_balance_usdt=float(raw.get("target_balance_usdt", 0.0)),
                target_multiple=float(raw.get("target_multiple", 2.0)),
                max_drawdown_pct=float(raw.get("max_drawdown_pct", 20.0)),
                stop_balance_usdt=float(raw.get("stop_balance_usdt", 0.0)),
                highest_balance_usdt=float(raw.get("highest_balance_usdt", 0.0)),
                latest_balance_usdt=float(raw.get("latest_balance_usdt", 0.0)),
                latest_snapshot_at=str(raw.get("latest_snapshot_at", "")),
                status=str(raw.get("status", "inactive")),
                note=str(raw.get("note", "")),
            )
        except (json.JSONDecodeError, ValueError):
            pass
    return ChallengeState()


def save_challenge_state(state: ChallengeState, scope: str | None = None) -> Path:
    path = challenge_state_path(scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(state), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def initialize_challenge(
    *,
    profile: str,
    symbol: str,
    market: str,
    start_balance_usdt: float,
    target_multiple: float = 2.0,
    max_drawdown_pct: float = 20.0,
    note: str = "",
    scope: str | None = None,
) -> ChallengeState:
    target_balance = start_balance_usdt * target_multiple
    stop_balance = start_balance_usdt * (1.0 - max_drawdown_pct / 100.0)
    state = ChallengeState(
        enabled=True,
        profile=profile,
        symbol=symbol.upper(),
        market=market,
        started_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        start_balance_usdt=round(start_balance_usdt, 8),
        target_balance_usdt=round(target_balance, 8),
        target_multiple=float(target_multiple),
        max_drawdown_pct=float(max_drawdown_pct),
        stop_balance_usdt=round(stop_balance, 8),
        highest_balance_usdt=round(start_balance_usdt, 8),
        latest_balance_usdt=round(start_balance_usdt, 8),
        latest_snapshot_at="",
        status="active",
        note=note,
    )
    save_challenge_state(state, scope)
    return state


def update_challenge_state(state: ChallengeState, snapshot: BalanceSnapshot, scope: str | None = None) -> ChallengeState:
    if not state.enabled:
        return state
    latest_balance = round(snapshot.equity_usdt, 8)
    state.latest_balance_usdt = latest_balance
    state.latest_snapshot_at = snapshot.timestamp
    state.highest_balance_usdt = round(max(state.highest_balance_usdt, latest_balance), 8)
    if latest_balance >= state.target_balance_usdt > 0:
        state.status = "target-hit"
    elif latest_balance <= state.stop_balance_usdt and state.stop_balance_usdt > 0:
        state.status = "drawdown-stop"
    else:
        state.status = "active"
    save_challenge_state(state, scope)
    return state


def record_balance_snapshot(
    payload: Any,
    market: str,
    *,
    note: str = "",
    scope: str | None = None,
) -> tuple[BalanceSnapshot, ChallengeState]:
    snapshot = snapshot_from_balance_payload(payload, market, note=note)
    append_balance_snapshot(snapshot)
    state = update_challenge_state(load_challenge_state(scope), snapshot, scope)
    return snapshot, state


def challenge_summary_dict(state: ChallengeState) -> dict[str, Any]:
    return {
        "enabled": state.enabled,
        "profile": state.profile,
        "symbol": state.symbol,
        "market": state.market,
        "started_at": state.started_at,
        "start_balance_usdt": state.start_balance_usdt,
        "latest_balance_usdt": state.latest_balance_usdt,
        "highest_balance_usdt": state.highest_balance_usdt,
        "target_balance_usdt": state.target_balance_usdt,
        "target_multiple": state.target_multiple,
        "remaining_to_target_usdt": round(state.remaining_to_target_usdt, 8),
        "max_drawdown_pct": state.max_drawdown_pct,
        "current_drawdown_pct": round(state.drawdown_pct, 4),
        "stop_balance_usdt": state.stop_balance_usdt,
        "progress_pct": round(state.progress_pct, 4),
        "status": state.status,
        "latest_snapshot_at": state.latest_snapshot_at,
        "note": state.note,
    }
