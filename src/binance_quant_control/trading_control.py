from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .challenge import challenge_scope_key, load_challenge_state
from .config import REPORTS_DIR, STATE_DIR
from .order_journal import load_trade_state, read_live_orders

TRADING_CONTROL_STATE_PATH = STATE_DIR / "trading-control.json"
AUTO_PAUSE_ACTOR = "openclaw-quantctl auto-pause-trading"


@dataclass(frozen=True, slots=True)
class TradingControlState:
    paused: bool = False
    reason: str = ""
    updated_at: str = ""
    updated_by: str = ""

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AutoPausePolicy:
    consecutive_loss_threshold: int = 2
    loss_cooldown_hours: float = 0.0
    position_timeout_hours: float = 0.0
    reversal_min_score: int = 80
    reversal_min_convergence: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AutoPauseEvaluation:
    should_pause: bool
    reasons: list[str]
    warnings: list[str]
    checked_at: str
    trade_state: dict[str, Any]
    challenge_state: dict[str, Any]
    positions: list[dict[str, Any]]
    latest_reports: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_iso_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _safe_json_load(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _latest_report_path(symbol: str, market: str, interval: str) -> Path | None:
    pattern = f"*-{symbol.lower()}-{market.lower()}-{interval.lower()}/analysis.json"
    candidates = list(REPORTS_DIR.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _latest_live_order_for_symbol(symbol: str, side: str) -> dict[str, Any] | None:
    side_upper = side.upper()
    records = read_live_orders()
    filtered = [
        item
        for item in records
        if str(item.get("symbol", "")).upper() == symbol.upper()
        and str(item.get("side", "")).upper() == side_upper
    ]
    if not filtered:
        return None
    return filtered[-1]


def load_trading_control_state() -> TradingControlState:
    if not TRADING_CONTROL_STATE_PATH.exists():
        return TradingControlState()
    try:
        payload = json.loads(TRADING_CONTROL_STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return TradingControlState()
    if not isinstance(payload, dict):
        return TradingControlState()
    return TradingControlState(
        paused=bool(payload.get("paused", False)),
        reason=str(payload.get("reason", "")),
        updated_at=str(payload.get("updated_at", "")),
        updated_by=str(payload.get("updated_by", "")),
    )


def save_trading_control_state(state: TradingControlState) -> TradingControlState:
    TRADING_CONTROL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRADING_CONTROL_STATE_PATH.write_text(
        json.dumps(state.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return state


def set_trading_paused(*, paused: bool, reason: str = "", updated_by: str = "operator") -> TradingControlState:
    state = TradingControlState(
        paused=paused,
        reason=reason.strip(),
        updated_at=_utc_now_iso(),
        updated_by=updated_by.strip() or "operator",
    )
    return save_trading_control_state(state)


def evaluate_auto_pause_conditions(
    settings: Any,
    strategy: Any,
    *,
    policy: AutoPausePolicy | None = None,
) -> AutoPauseEvaluation:
    from .binance_api import BinanceAPIError, BinanceClient

    policy = policy or AutoPausePolicy()
    trade_state = load_trade_state()
    scope = challenge_scope_key(strategy.profile, strategy.defaults.symbol, strategy.defaults.market)
    challenge_state = load_challenge_state(scope)

    reasons: list[str] = []
    warnings: list[str] = []
    latest_reports: list[dict[str, Any]] = []

    if challenge_state.enabled:
        if challenge_state.status == "drawdown-stop":
            reasons.append(
                f"Challenge drawdown-stop is already active at {challenge_state.latest_balance_usdt:.4f} USDT "
                f"(floor {challenge_state.stop_balance_usdt:.4f})."
            )
        elif challenge_state.max_drawdown_pct > 0 and challenge_state.drawdown_pct >= challenge_state.max_drawdown_pct:
            reasons.append(
                f"Challenge drawdown {challenge_state.drawdown_pct:.2f}% reached the configured floor "
                f"of {challenge_state.max_drawdown_pct:.2f}%."
            )

    if trade_state.consecutive_losses >= policy.consecutive_loss_threshold:
        if policy.loss_cooldown_hours > 0.0:
            last_loss_at = _parse_iso_datetime(str(trade_state.last_loss_at or ""))
            if last_loss_at is None:
                reasons.append(
                    f"Consecutive losses reached {trade_state.consecutive_losses}, but no last-loss timestamp "
                    "is available; keeping auto-pause active as a precaution."
                )
            else:
                hours_since_loss = (datetime.now(timezone.utc) - last_loss_at).total_seconds() / 3600.0
                if hours_since_loss < policy.loss_cooldown_hours:
                    reasons.append(
                        f"Consecutive losses reached {trade_state.consecutive_losses}, which meets the "
                        f"auto-pause threshold of {policy.consecutive_loss_threshold}; last loss was "
                        f"{hours_since_loss:.1f}h ago and cooldown requires {policy.loss_cooldown_hours:.1f}h."
                    )
                else:
                    warnings.append(
                        f"Consecutive-loss auto-pause cooldown expired ({hours_since_loss:.1f}h since last loss; "
                        f"required {policy.loss_cooldown_hours:.1f}h)."
                    )
        else:
            reasons.append(
                f"Consecutive losses reached {trade_state.consecutive_losses}, which meets the auto-pause threshold "
                f"of {policy.consecutive_loss_threshold}."
            )

    try:
        with BinanceClient(settings) as client:
            raw_positions = client.positions(None)
    except BinanceAPIError as exc:
        warnings.append(f"Position scan skipped: {exc}")
        raw_positions = []

    open_positions = []
    for item in raw_positions:
        amount = float(item.get("positionAmt", 0.0))
        if amount == 0.0:
            continue
        symbol = str(item.get("symbol", "")).upper()
        side = "LONG" if amount > 0 else "SHORT"
        open_positions.append(
            {
                "symbol": symbol,
                "side": side,
                "qty": amount,
                "entry": float(item.get("entryPrice", 0.0)),
                "pnl": float(item.get("unRealizedProfit", 0.0)),
                "leverage": int(float(item.get("leverage", 1))),
            }
        )

        journal_entry = _latest_live_order_for_symbol(symbol, "BUY" if amount > 0 else "SELL")
        if policy.position_timeout_hours > 0 and journal_entry:
            opened_at = _parse_iso_datetime(str(journal_entry.get("timestamp", "")))
            if opened_at is not None:
                age_hours = (datetime.now(timezone.utc) - opened_at).total_seconds() / 3600.0
                if age_hours >= policy.position_timeout_hours:
                    reasons.append(
                        f"{symbol} {side} has been open for {age_hours:.1f}h, exceeding the timeout "
                        f"threshold of {policy.position_timeout_hours:.1f}h."
                    )

        report_path = _latest_report_path(symbol, "futures", strategy.defaults.interval)
        if report_path is not None:
            report = _safe_json_load(report_path)
            if report:
                analysis = report.get("analysis") or {}
                latest_reports.append(
                    {
                        "symbol": symbol,
                        "path": str(report_path),
                        "bias": analysis.get("bias"),
                        "score": analysis.get("score"),
                        "convergence": analysis.get("convergence"),
                        "generated_at": report.get("generated_at"),
                    }
                )
                bias = str(analysis.get("bias", ""))
                score = int(analysis.get("score", 0))
                convergence = float(analysis.get("convergence", 0.0))
                is_reverse = (
                    amount > 0 and bias == "short-bias"
                ) or (
                    amount < 0 and bias == "long-bias"
                )
                if is_reverse and score >= policy.reversal_min_score and convergence >= policy.reversal_min_convergence:
                    reasons.append(
                        f"{symbol} latest analysis flipped to {bias} (score {score}, convergence {convergence:.3f}) "
                        f"against the current {side} position."
                    )

    return AutoPauseEvaluation(
        should_pause=len(reasons) > 0,
        reasons=reasons,
        warnings=warnings,
        checked_at=_utc_now_iso(),
        trade_state={
            "daily_trade_count": trade_state.daily_trade_count,
            "daily_trade_date": trade_state.daily_trade_date,
            "consecutive_losses": trade_state.consecutive_losses,
            "last_loss_at": trade_state.last_loss_at,
            "total_live_trades": trade_state.total_live_trades,
            "total_pnl_usdt": trade_state.total_pnl_usdt,
        },
        challenge_state={
            "enabled": challenge_state.enabled,
            "profile": challenge_state.profile,
            "symbol": challenge_state.symbol,
            "market": challenge_state.market,
            "status": challenge_state.status,
            "latest_balance_usdt": challenge_state.latest_balance_usdt,
            "stop_balance_usdt": challenge_state.stop_balance_usdt,
            "drawdown_pct": round(challenge_state.drawdown_pct, 4),
            "max_drawdown_pct": challenge_state.max_drawdown_pct,
        },
        positions=open_positions,
        latest_reports=latest_reports,
    )
