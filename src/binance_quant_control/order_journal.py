"""Live order journal and state tracker.

Provides append-only recording of live orders, daily trade counting,
and consecutive loss tracking for the risk guard.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .config import STATE_DIR
from .convergence import (
    build_cohort_id,
    calculate_expectancy_stats,
    calculate_loss_streak,
    calculate_profit_factor,
)

LIVE_ORDERS_FILE = STATE_DIR / "live-orders.jsonl"
PAPER_ORDERS_FILE = STATE_DIR / "paper-orders.jsonl"
CLOSED_TRADE_REVIEWS_FILE = STATE_DIR / "closed-trade-reviews.jsonl"
TRADE_STATE_FILE = STATE_DIR / "live-trade-state.json"


@dataclass(slots=True)
class PaperOrderRecord:
    generated_at: str
    kind: str
    symbol: str
    market: str
    side: str
    margin_notional_usdt: float
    leverage: float
    gross_notional_usdt: float
    reference_price: float
    estimated_quantity: float
    analysis_bias: str
    analysis_score: int
    analysis_convergence: float
    cohort_id: str | None = None
    strategy_profile: str | None = None
    strategy_path: str | None = None
    asset_class: str | None = None
    route_id: str | None = None
    simulation_mode: str | None = None
    review_lane: str | None = None
    entry_reason_snapshot: dict[str, Any] | None = None
    signal_scores: dict[str, Any] | None = None
    analysis_report: str | None = None
    chart_path: str | None = None
    note: str = ""


@dataclass(slots=True)
class LiveOrderRecord:
    timestamp: str
    symbol: str
    market: str
    side: str
    order_type: str
    quantity: float
    price: float | None
    leverage: int
    notional_usdt: float
    gross_notional_usdt: float
    analysis_score: int
    analysis_bias: str
    analysis_convergence: float
    wallet_balance_usdt: float
    available_balance_usdt: float
    equity_usdt: float
    challenge_status: str
    challenge_target_usdt: float
    challenge_progress_pct: float
    order_id: int | str | None
    status: str
    cohort_id: str | None = None
    strategy_profile: str | None = None
    strategy_path: str | None = None
    asset_class: str | None = None
    route_id: str | None = None
    simulation_mode: str | None = None
    review_lane: str | None = None
    entry_reason_snapshot: dict[str, Any] | None = None
    signal_scores: dict[str, Any] | None = None
    binance_response: dict[str, Any] | None = None
    note: str = ""


@dataclass(slots=True)
class ClosedTradeReviewRecord:
    reviewed_at: str
    opened_at: str
    closed_at: str
    source_order_id: int | str | None
    symbol: str
    market: str
    side: str
    quantity: float
    leverage: int
    entry_price: float
    exit_price: float | None
    stop_loss_price: float | None
    take_profit_price: float | None
    exit_reason: str
    realized_pnl_usdt: float
    realized_pnl_pct: float
    realized_r_multiple: float | None
    analysis_score: int
    analysis_bias: str
    analysis_convergence: float
    challenge_status: str = "inactive"
    challenge_progress_pct: float = 0.0
    cohort_id: str | None = None
    strategy_profile: str | None = None
    strategy_path: str | None = None
    asset_class: str | None = None
    route_id: str | None = None
    review_lane: str | None = None
    entry_reason_snapshot: dict[str, Any] | None = None
    signal_scores: dict[str, Any] | None = None
    rule_compliant: bool | None = None
    false_positive_tag: str | None = None
    market_regime_tag: str | None = None
    note: str = ""
    source_hash: str = ""
    binance_context: dict[str, Any] | None = None


@dataclass(slots=True)
class TradeState:
    """Mutable trade state persisted as JSON."""

    daily_trade_count: int = 0
    daily_trade_date: str = ""
    consecutive_losses: int = 0
    last_loss_at: str | None = None
    total_live_trades: int = 0
    total_pnl_usdt: float = 0.0

    def today_str(self) -> str:
        return date.today().isoformat()

    def ensure_daily_reset(self) -> None:
        """Reset the daily counter if the date has changed."""
        today = self.today_str()
        if self.daily_trade_date != today:
            self.daily_trade_count = 0
            self.daily_trade_date = today

    def record_trade(self) -> None:
        self.ensure_daily_reset()
        self.daily_trade_count += 1
        self.total_live_trades += 1

    def record_loss(self, pnl: float) -> None:
        self.consecutive_losses += 1
        self.last_loss_at = datetime.now(timezone.utc).isoformat()
        self.total_pnl_usdt += pnl

    def record_win(self, pnl: float) -> None:
        self.consecutive_losses = 0
        self.last_loss_at = None
        self.total_pnl_usdt += pnl

    @property
    def last_loss_datetime(self) -> datetime | None:
        if self.last_loss_at:
            return datetime.fromisoformat(self.last_loss_at)
        return None


def load_trade_state() -> TradeState:
    """Load trade state from disk or return defaults."""
    if TRADE_STATE_FILE.exists():
        try:
            raw = json.loads(TRADE_STATE_FILE.read_text(encoding="utf-8"))
            state = TradeState(
                daily_trade_count=raw.get("daily_trade_count", 0),
                daily_trade_date=raw.get("daily_trade_date", ""),
                consecutive_losses=raw.get("consecutive_losses", 0),
                last_loss_at=raw.get("last_loss_at"),
                total_live_trades=raw.get("total_live_trades", 0),
                total_pnl_usdt=raw.get("total_pnl_usdt", 0.0),
            )
            state.ensure_daily_reset()
            return state
        except (json.JSONDecodeError, KeyError):
            pass
    return TradeState(daily_trade_date=date.today().isoformat())


def save_trade_state(state: TradeState) -> None:
    """Persist trade state to disk."""
    TRADE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRADE_STATE_FILE.write_text(
        json.dumps(asdict(state), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def append_live_order(record: LiveOrderRecord) -> Path:
    """Append a live order record to the journal file."""
    LIVE_ORDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LIVE_ORDERS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return LIVE_ORDERS_FILE


def append_paper_order(record: PaperOrderRecord) -> Path:
    """Append a paper/demo simulation order record to the journal file."""
    PAPER_ORDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PAPER_ORDERS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return PAPER_ORDERS_FILE


def append_closed_trade_review(record: ClosedTradeReviewRecord) -> Path:
    """Append a closed trade review record to the journal file."""
    CLOSED_TRADE_REVIEWS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CLOSED_TRADE_REVIEWS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return CLOSED_TRADE_REVIEWS_FILE


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def read_live_orders() -> list[dict[str, Any]]:
    """Read all live order records from the journal."""
    return _read_jsonl(LIVE_ORDERS_FILE)


def read_paper_orders() -> list[dict[str, Any]]:
    """Read all paper/demo simulation order records from the journal."""
    return _read_jsonl(PAPER_ORDERS_FILE)


def read_closed_trade_reviews() -> list[dict[str, Any]]:
    """Read all closed trade review records from the journal."""
    return _read_jsonl(CLOSED_TRADE_REVIEWS_FILE)


def _strategy_profile_from_note(note: str) -> str:
    if "strategy=" not in note:
        return ""
    return note.split("strategy=", 1)[1].split()[0].strip().strip(",;")


def _backfill_route_metadata(row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    from .asset_routing import resolve_symbol_route

    symbol = str(row.get("symbol") or "").upper()
    if not symbol:
        return row, False
    try:
        route = resolve_symbol_route(symbol)
    except ValueError:
        return row, False
    note = str(row.get("note") or "")
    strategy_profile = str(row.get("strategy_profile") or "").strip()
    if not strategy_profile:
        strategy_profile = _strategy_profile_from_note(note)
    if not strategy_profile:
        strategy_profile = route.strategy_config.stem.replace("strategy-", "")
    entry_reason = row.get("entry_reason_snapshot")
    if not isinstance(entry_reason, dict):
        entry_reason = {
            "bias": str(row.get("analysis_bias") or ""),
            "score": int(row.get("analysis_score") or 0),
            "convergence": float(row.get("analysis_convergence") or 0.0),
            "interval": route.interval,
        }
    desired = {
        "cohort_id": row.get("cohort_id")
        or build_cohort_id(
            asset_class=route.asset_class,
            strategy_profile=strategy_profile,
            market=str(row.get("market") or route.market),
            interval=str(entry_reason.get("interval") or route.interval),
        ),
        "strategy_profile": row.get("strategy_profile") or strategy_profile,
        "strategy_path": row.get("strategy_path") or str(route.strategy_config),
        "asset_class": row.get("asset_class") or route.asset_class,
        "route_id": row.get("route_id") or route.route_id,
        "simulation_mode": row.get("simulation_mode") or route.simulation_mode,
        "review_lane": row.get("review_lane") or route.review_lane,
        "entry_reason_snapshot": entry_reason,
    }
    updated = dict(row)
    before = dict(updated)
    updated.update(desired)
    return updated, updated != before


def backfill_closed_trade_review_metadata(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    updated: list[dict[str, Any]] = []
    changed = 0
    for item in records:
        row, did_change = _backfill_route_metadata(dict(item))
        changed += 1 if did_change else 0
        updated.append(row)
    return updated, changed


def backfill_live_order_metadata(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    updated: list[dict[str, Any]] = []
    changed = 0
    for item in records:
        row, did_change = _backfill_route_metadata(dict(item))
        changed += 1 if did_change else 0
        updated.append(row)
    return updated, changed


def write_live_orders(records: list[dict[str, Any]]) -> Path:
    LIVE_ORDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LIVE_ORDERS_FILE.open("w", encoding="utf-8") as fh:
        for item in records:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return LIVE_ORDERS_FILE


def write_closed_trade_reviews(records: list[dict[str, Any]]) -> Path:
    CLOSED_TRADE_REVIEWS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CLOSED_TRADE_REVIEWS_FILE.open("w", encoding="utf-8") as fh:
        for item in records:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return CLOSED_TRADE_REVIEWS_FILE


def count_today_trades() -> int:
    """Count the number of live trades made today."""
    state = load_trade_state()
    return state.daily_trade_count


def summarize_live_orders() -> dict[str, Any]:
    records = read_live_orders()
    buy_count = sum(1 for item in records if str(item.get("side", "")).upper() == "BUY")
    sell_count = sum(1 for item in records if str(item.get("side", "")).upper() == "SELL")
    latest = records[-1] if records else None
    return {
        "count": len(records),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "latest": latest,
    }


def summarize_paper_orders() -> dict[str, Any]:
    records = read_paper_orders()
    by_route: dict[str, int] = {}
    by_asset_class: dict[str, int] = {}
    by_cohort: dict[str, int] = {}
    for item in records:
        route_id = str(item.get("route_id") or "unrouted")
        asset_class = str(item.get("asset_class") or "unknown")
        cohort_id = str(item.get("cohort_id") or "uncohorted")
        by_route[route_id] = by_route.get(route_id, 0) + 1
        by_asset_class[asset_class] = by_asset_class.get(asset_class, 0) + 1
        by_cohort[cohort_id] = by_cohort.get(cohort_id, 0) + 1
    return {
        "count": len(records),
        "by_route": by_route,
        "by_asset_class": by_asset_class,
        "by_cohort": by_cohort,
        "latest": records[-1] if records else None,
    }


def summarize_closed_trade_reviews() -> dict[str, Any]:
    records = read_closed_trade_reviews()
    pnls = [float(item.get("realized_pnl_usdt", 0.0) or 0.0) for item in records]
    r_values = [
        float(item.get("realized_r_multiple"))
        for item in records
        if item.get("realized_r_multiple") not in (None, "")
    ]
    wins = [item for item in records if float(item.get("realized_pnl_usdt", 0.0) or 0.0) > 0.0]
    losses = [item for item in records if float(item.get("realized_pnl_usdt", 0.0) or 0.0) < 0.0]
    pure_stop_reasons = {"stop_loss", "stop_priority_same_bar"}
    pure_stops = [
        item
        for item in records
        if str(item.get("exit_reason") or "").lower() in pure_stop_reasons
    ]
    partial_tp_then_stops = [
        item
        for item in records
        if str(item.get("exit_reason") or "").lower() == "partial_tp_then_stop"
    ]
    latest = records[-1] if records else None
    total_realized = sum(pnls)
    by_route: dict[str, int] = {}
    by_asset_class: dict[str, int] = {}
    by_cohort: dict[str, int] = {}
    for item in records:
        route_id = str(item.get("route_id") or "unrouted")
        asset_class = str(item.get("asset_class") or "unknown")
        cohort_id = str(item.get("cohort_id") or "uncohorted")
        by_route[route_id] = by_route.get(route_id, 0) + 1
        by_asset_class[asset_class] = by_asset_class.get(asset_class, 0) + 1
        by_cohort[cohort_id] = by_cohort.get(cohort_id, 0) + 1
    return {
        "count": len(records),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round((len(wins) / len(records)) * 100.0, 2) if records else 0.0,
        "total_realized_pnl_usdt": round(total_realized, 8),
        "profit_factor": round(calculate_profit_factor(pnls), 4) if pnls else 0.0,
        "loss_streak": calculate_loss_streak(pnls),
        "pure_stop_loss_count": len(pure_stops),
        "pure_stop_loss_ratio": round((len(pure_stops) / len(records)) * 100.0, 2) if records else 0.0,
        "partial_tp_then_stop_count": len(partial_tp_then_stops),
        "partial_tp_then_stop_ratio": round((len(partial_tp_then_stops) / len(records)) * 100.0, 2) if records else 0.0,
        "expectancy": calculate_expectancy_stats(r_values),
        "by_route": by_route,
        "by_asset_class": by_asset_class,
        "by_cohort": by_cohort,
        "latest": latest,
    }
