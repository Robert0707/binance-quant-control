from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from .analysis import enrich_indicators, prepare_klines_frame
from .asset_routing import normalize_symbol, resolve_symbol_route
from .backtest import simulate_backtest
from .config import STATE_DIR, ensure_runtime_dirs
from .convergence import build_cohort_id
from .order_journal import (
    ClosedTradeReviewRecord,
    append_closed_trade_review,
    read_closed_trade_reviews,
)
from .strategy import load_strategy_config
from .strategy_optimizer import run_strategy_optimizer

BINANCE_PUBLIC_ROOT = "https://data.binance.vision/data"
PUBLIC_HISTORY_STATE_DIR = STATE_DIR / "public-history-training"


@dataclass(frozen=True, slots=True)
class PublicHistoryConfig:
    symbols: tuple[str, ...]
    months: tuple[str, ...]
    max_reviews_per_symbol: int
    optimize_every: int


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _month_range(end_month: str, count: int) -> tuple[str, ...]:
    year, month = (int(part) for part in end_month.split("-", 1))
    months: list[str] = []
    for _ in range(max(count, 1)):
        months.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month == 0:
            year -= 1
            month = 12
    return tuple(reversed(months))


def default_end_month() -> str:
    now = _utc_now()
    year = now.year
    month = now.month - 1
    if month == 0:
        year -= 1
        month = 12
    return f"{year:04d}-{month:02d}"


def _public_history_url(*, market: str, symbol: str, interval: str, month: str) -> str:
    if market == "spot":
        return f"{BINANCE_PUBLIC_ROOT}/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    return (
        f"{BINANCE_PUBLIC_ROOT}/futures/um/monthly/klines/"
        f"{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    )


def fetch_public_monthly_klines(
    *,
    market: str,
    symbol: str,
    interval: str,
    month: str,
    timeout: float = 60.0,
) -> list[list[Any]]:
    url = _public_history_url(market=market, symbol=symbol.upper(), interval=interval, month=month)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_name = next((name for name in archive.namelist() if name.endswith(".csv")), "")
        if not csv_name:
            return []
        with archive.open(csv_name) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8")
            rows = [row for row in csv.reader(text) if row]
    if rows and rows[0][0].lower() in {"open_time", "open time"}:
        rows = rows[1:]
    return rows


def fetch_public_history_frame(
    *,
    market: str,
    symbol: str,
    interval: str,
    months: tuple[str, ...],
    strategy: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    rows: list[list[Any]] = []
    fetch_log: list[dict[str, Any]] = []
    for month in months:
        monthly_rows = fetch_public_monthly_klines(
            market=market,
            symbol=symbol,
            interval=interval,
            month=month,
        )
        fetch_log.append(
            {
                "source": "binance-public-data",
                "url": _public_history_url(market=market, symbol=symbol, interval=interval, month=month),
                "symbol": symbol,
                "market": market,
                "interval": interval,
                "month": month,
                "rows": len(monthly_rows),
            }
        )
        rows.extend(monthly_rows)
    if not rows:
        raise RuntimeError(f"No public history rows found for {symbol} {market} {interval} {months}")
    frame = enrich_indicators(prepare_klines_frame(rows), interval, strategy=strategy)
    if len(frame) < 240:
        raise RuntimeError(f"Insufficient public history rows for {symbol} {market} {interval}: {len(frame)}")
    return frame, fetch_log


def _history_source_candidates(symbol: str) -> tuple[str, ...]:
    normalized = symbol.upper()
    if normalized == "XAUTUSDT":
        return ("XAUTUSDT", "PAXGUSDT")
    return (normalized,)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _review_from_public_trade(
    *,
    symbol: str,
    route: Any,
    strategy: Any,
    trade: dict[str, Any],
    source_index: int,
) -> ClosedTradeReviewRecord:
    opened_at = str(trade.get("entry_time") or "")
    closed_at = str(trade.get("exit_time") or "")
    pnl_pct = _safe_float(trade.get("pnl_pct"))
    gross_notional = 10.0
    realized_pnl_usdt = round(gross_notional * (pnl_pct / 100.0), 8)
    source_order_id = f"public-history-{symbol}-{route.interval}-{source_index:06d}"
    source_hash = ":".join(
        str(item)
        for item in (
            "public-history",
            symbol,
            route.route_id,
            route.market,
            route.interval,
            trade.get("side"),
            opened_at,
            closed_at,
            trade.get("entry_price"),
            trade.get("exit_price"),
            trade.get("exit_reason"),
        )
    )
    return ClosedTradeReviewRecord(
        reviewed_at=_utc_now().isoformat(),
        opened_at=opened_at,
        closed_at=closed_at,
        source_order_id=source_order_id,
        symbol=symbol,
        market=route.market,
        side=str(trade.get("side") or "").upper(),
        quantity=0.0,
        leverage=int(strategy.risk.default_leverage),
        entry_price=round(_safe_float(trade.get("entry_price")), 8),
        exit_price=round(_safe_float(trade.get("exit_price")), 8),
        stop_loss_price=None,
        take_profit_price=None,
        exit_reason=str(trade.get("exit_reason") or "backtest_exit"),
        realized_pnl_usdt=realized_pnl_usdt,
        realized_pnl_pct=round(pnl_pct, 4),
        realized_r_multiple=_safe_float(trade.get("pnl_r")) if trade.get("pnl_r") is not None else None,
        analysis_score=int(_safe_float(trade.get("analysis_score"))),
        analysis_bias=f"{str(trade.get('side') or '').lower()}-bias",
        analysis_convergence=_safe_float(trade.get("analysis_convergence")),
        challenge_status="public-history-training",
        challenge_progress_pct=0.0,
        cohort_id=build_cohort_id(
            asset_class=route.asset_class,
            strategy_profile=strategy.profile,
            market=route.market,
            interval=route.interval,
        ),
        strategy_profile=strategy.profile,
        strategy_path=str(strategy.path),
        asset_class=route.asset_class,
        route_id=route.route_id,
        review_lane=route.review_lane,
        entry_reason_snapshot={
            "bias": f"{str(trade.get('side') or '').lower()}-bias",
            "score": int(_safe_float(trade.get("analysis_score"))),
            "convergence": _safe_float(trade.get("analysis_convergence")),
            "interval": route.interval,
        },
        signal_scores=None,
        rule_compliant=True,
        false_positive_tag=(
            "public-history-high-conviction-loss"
            if realized_pnl_usdt < 0 and _safe_float(trade.get("analysis_convergence")) >= 0.8
            else None
        ),
        market_regime_tag="public-history",
        note=(
            "training_source=binance_public_history "
            f"strategy={strategy.profile} route={route.route_id}"
        ),
        source_hash=source_hash,
        binance_context={
            "training_mode": "binance_public_history",
            "source": "https://github.com/binance/binance-public-data",
            "trade": trade,
        },
    )


def run_public_history_training(
    *,
    symbols: list[str] | None = None,
    months: int = 24,
    end_month: str | None = None,
    max_reviews_per_symbol: int = 100,
    optimize_every: int = 250,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    PUBLIC_HISTORY_STATE_DIR.mkdir(parents=True, exist_ok=True)
    selected_symbols = tuple(normalize_symbol(item) for item in (symbols or ["BTCUSDT", "ETHUSDT", "PAXGUSDT"]) if item)
    selected_months = _month_range(end_month or default_end_month(), months)
    config = PublicHistoryConfig(
        symbols=selected_symbols,
        months=selected_months,
        max_reviews_per_symbol=max(max_reviews_per_symbol, 1),
        optimize_every=max(optimize_every, 1),
    )
    existing_hashes = {str(item.get("source_hash") or "") for item in read_closed_trade_reviews()}
    results: list[dict[str, Any]] = []
    optimizer_reports: list[dict[str, Any]] = []
    inserted = 0
    skipped_duplicates = 0
    fetch_log: list[dict[str, Any]] = []

    for symbol in config.symbols:
        route = resolve_symbol_route(symbol)
        strategy = load_strategy_config(route.strategy_config)
        last_error = ""
        frame = None
        symbol_fetch_log: list[dict[str, Any]] = []
        source_symbol = symbol
        for candidate_symbol in _history_source_candidates(symbol):
            try:
                frame, symbol_fetch_log = fetch_public_history_frame(
                    market=route.market,
                    symbol=candidate_symbol,
                    interval=route.interval,
                    months=config.months,
                    strategy=strategy,
                )
                source_symbol = candidate_symbol
                break
            except Exception as exc:
                last_error = str(exc)
        if frame is None:
            results.append(
                {
                    "symbol": symbol,
                    "route_id": route.route_id,
                    "status": "fetch_failed",
                    "error": last_error,
                }
            )
            continue
        fetch_log.extend(symbol_fetch_log)
        summary = simulate_backtest(frame, route.market, strategy)
        trades = list(summary.get("trades") or [])[-config.max_reviews_per_symbol :]
        symbol_inserted = 0
        for offset, trade in enumerate(trades):
            review = _review_from_public_trade(
                symbol=symbol,
                route=route,
                strategy=strategy,
                trade=trade,
                source_index=offset,
            )
            if review.source_hash in existing_hashes:
                skipped_duplicates += 1
                continue
            append_closed_trade_review(review)
            existing_hashes.add(review.source_hash)
            inserted += 1
            symbol_inserted += 1
            if inserted % config.optimize_every == 0:
                optimizer = run_strategy_optimizer()
                optimizer_reports.append(
                    {
                        "after_inserted_reviews": inserted,
                        "status": optimizer.get("status"),
                        "review_count": optimizer.get("review_count"),
                        "promotion_decision": optimizer.get("promotion_decision"),
                        "report_path": optimizer.get("report_path"),
                    }
                )
        results.append(
            {
                "symbol": symbol,
                "history_source_symbol": source_symbol,
                "route_id": route.route_id,
                "asset_class": route.asset_class,
                "strategy_profile": strategy.profile,
                "market": route.market,
                "interval": route.interval,
                "months": list(config.months),
                "candles": len(frame),
                "backtest": {
                    "trade_count": summary.get("trade_count"),
                    "wins": summary.get("wins"),
                    "losses": summary.get("losses"),
                    "win_rate": summary.get("win_rate"),
                    "profit_factor": summary.get("profit_factor"),
                    "total_return_pct": summary.get("total_return_pct"),
                    "max_drawdown_pct": summary.get("max_drawdown_pct"),
                },
                "reviews_inserted": symbol_inserted,
            }
        )

    optimizer = run_strategy_optimizer()
    optimizer_reports.append(
        {
            "after_inserted_reviews": inserted,
            "status": optimizer.get("status"),
            "review_count": optimizer.get("review_count"),
            "promotion_decision": optimizer.get("promotion_decision"),
            "report_path": optimizer.get("report_path"),
        }
    )
    payload = {
        "generated_at": _utc_now().isoformat(),
        "status": "ok",
        "mode": "binance_public_history_training",
        "source": {
            "name": "Binance Public Data",
            "url": "https://github.com/binance/binance-public-data",
            "archive_root": BINANCE_PUBLIC_ROOT,
        },
        "config": asdict(config),
        "inserted_review_count": inserted,
        "skipped_duplicate_count": skipped_duplicates,
        "results": results,
        "fetch_log": fetch_log,
        "optimizer_reports": optimizer_reports,
    }
    report_path = PUBLIC_HISTORY_STATE_DIR / f"{_stamp()}-public-history-training.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload
