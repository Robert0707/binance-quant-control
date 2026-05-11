from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .analysis import run_analysis
from .asset_routing import resolve_symbol_route
from .backtest import run_backtest
from .config import STATE_DIR, ensure_runtime_dirs, load_settings
from .convergence import build_cohort_id
from .live_execution import build_live_execution_plan
from .mission_control import mission_candidate_side
from .order_journal import (
    ClosedTradeReviewRecord,
    PaperOrderRecord,
    append_closed_trade_review,
    append_paper_order,
)
from .route_risk_control import route_quarantine_status
from .signal_scoring import build_signal_scores
from .strategy import load_strategy_config
from .strategy_optimizer import run_strategy_optimizer

DEFAULT_TRAINING_STATE_DIR = STATE_DIR / "training"
DEFAULT_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "NEARUSDT",
    "DOGEUSDT",
    "PENGUUSDT",
    "1000PEPEUSDT",
    "LINKUSDT",
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    symbols: tuple[str, ...]
    target_return_pct: float
    max_leverage: float
    margin_notional_usdt: float
    training_sample_size: int
    optimize_every: int
    include_research_only: bool


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _select_symbol(symbols: tuple[str, ...], idx: int) -> str:
    return symbols[idx % len(symbols)]


def _backtest_trade_matches_side(trade: dict[str, Any], candidate_side: str) -> bool:
    trade_side = str(trade.get("side") or "").upper()
    return bool(trade_side) and trade_side == candidate_side.upper()


def _paper_record_from_plan(
    *,
    generated_at: datetime,
    symbol: str,
    route: Any,
    strategy: Any,
    analysis_payload: dict[str, Any],
    side: str,
    plan_payload: dict[str, Any],
    signal_scores: dict[str, Any],
    note: str,
) -> PaperOrderRecord:
    price = _safe_float(plan_payload.get("price"))
    leverage = _safe_float(plan_payload.get("leverage"), float(strategy.risk.default_leverage))
    margin_notional_usdt = _safe_float(plan_payload.get("margin_notional_usdt"))
    gross_notional_usdt = _safe_float(plan_payload.get("gross_notional_usdt"))
    quantity = _safe_float(plan_payload.get("quantity"))
    return PaperOrderRecord(
        generated_at=generated_at.isoformat(),
        kind="paper-order",
        symbol=symbol,
        market=strategy.defaults.market,
        side=side,
        margin_notional_usdt=round(margin_notional_usdt, 6),
        leverage=round(leverage, 6),
        gross_notional_usdt=round(gross_notional_usdt, 6),
        reference_price=round(price, 8),
        estimated_quantity=round(quantity, 8),
        analysis_bias=str((analysis_payload.get("analysis") or {}).get("bias") or ""),
        analysis_score=int((analysis_payload.get("analysis") or {}).get("score") or 0),
        analysis_convergence=float((analysis_payload.get("analysis") or {}).get("convergence") or 0.0),
        cohort_id=build_cohort_id(
            asset_class=route.asset_class,
            strategy_profile=strategy.profile,
            market=strategy.defaults.market,
            interval=strategy.defaults.interval,
        ),
        strategy_profile=strategy.profile,
        strategy_path=str(strategy.path),
        asset_class=route.asset_class,
        route_id=route.route_id,
        simulation_mode=route.simulation_mode,
        review_lane=route.review_lane,
        entry_reason_snapshot={
            "bias": str((analysis_payload.get("analysis") or {}).get("bias") or ""),
            "score": int((analysis_payload.get("analysis") or {}).get("score") or 0),
            "convergence": float((analysis_payload.get("analysis") or {}).get("convergence") or 0.0),
            "interval": strategy.defaults.interval,
        },
        signal_scores=signal_scores,
        analysis_report=str((analysis_payload.get("artifacts") or {}).get("report_json") or ""),
        chart_path=(analysis_payload.get("artifacts") or {}).get("chart_path"),
        note=note,
    )


def _simulate_review_from_backtest(
    *,
    generated_at: datetime,
    idx: int,
    symbol: str,
    route: Any,
    strategy: Any,
    paper_record: PaperOrderRecord,
    plan_payload: dict[str, Any],
    trade: dict[str, Any],
) -> ClosedTradeReviewRecord:
    opened_at = trade.get("entry_time") or generated_at.isoformat()
    closed_at = trade.get("exit_time") or (generated_at + timedelta(hours=1)).isoformat()
    entry_price = _safe_float(trade.get("entry_price"), paper_record.reference_price)
    exit_price = _safe_float(trade.get("exit_price"), paper_record.reference_price)
    realized_pnl_pct = _safe_float(trade.get("pnl_pct"))
    gross_notional = max(_safe_float(plan_payload.get("gross_notional_usdt"), paper_record.gross_notional_usdt), 0.0)
    realized_pnl_usdt = round(gross_notional * (realized_pnl_pct / 100.0), 8)
    stop_price = _safe_float(plan_payload.get("stop_price"))
    take_profit_price = _safe_float(plan_payload.get("take_profit_price"))
    market_regime = "trend" if _safe_float(paper_record.analysis_convergence) >= 0.8 else "mixed"
    false_positive_tag = None
    if realized_pnl_usdt < 0 and paper_record.analysis_convergence >= 0.8:
        false_positive_tag = "training-high-conviction-loss"
    source_order_id = f"training-{generated_at.strftime('%Y%m%dT%H%M%SZ')}-{symbol}-{idx:04d}"
    return ClosedTradeReviewRecord(
        reviewed_at=generated_at.isoformat(),
        opened_at=str(opened_at),
        closed_at=str(closed_at),
        source_order_id=source_order_id,
        symbol=symbol,
        market=strategy.defaults.market,
        side=paper_record.side,
        quantity=paper_record.estimated_quantity,
        leverage=_safe_int(plan_payload.get("leverage"), int(round(paper_record.leverage or 0))),
        entry_price=round(entry_price, 8),
        exit_price=round(exit_price, 8),
        stop_loss_price=round(stop_price, 8) if stop_price > 0 else None,
        take_profit_price=round(take_profit_price, 8) if take_profit_price > 0 else None,
        exit_reason=str(trade.get("exit_reason") or "end_of_data"),
        realized_pnl_usdt=realized_pnl_usdt,
        realized_pnl_pct=round(realized_pnl_pct, 4),
        realized_r_multiple=_safe_float(trade.get("pnl_r")) if trade.get("pnl_r") is not None else None,
        analysis_score=paper_record.analysis_score,
        analysis_bias=paper_record.analysis_bias,
        analysis_convergence=paper_record.analysis_convergence,
        challenge_status="training",
        challenge_progress_pct=0.0,
        cohort_id=paper_record.cohort_id,
        strategy_profile=paper_record.strategy_profile,
        strategy_path=paper_record.strategy_path,
        asset_class=paper_record.asset_class,
        route_id=paper_record.route_id,
        review_lane=paper_record.review_lane,
        entry_reason_snapshot=paper_record.entry_reason_snapshot,
        signal_scores=paper_record.signal_scores,
        rule_compliant=True,
        false_positive_tag=false_positive_tag,
        market_regime_tag=market_regime,
        note=(
            "training_source=market_replay "
            f"strategy={strategy.profile} route={route.route_id} sample_index={idx}"
        ),
        source_hash=f"{source_order_id}:{trade.get('exit_reason')}:{trade.get('entry_time')}:{trade.get('exit_time')}",
        binance_context={
            "training_mode": "market_replay",
            "analysis_report": paper_record.analysis_report,
            "chart_path": paper_record.chart_path,
            "plan": plan_payload,
            "backtest_trade": trade,
        },
    )


def _build_training_iteration(
    *,
    idx: int,
    config: TrainingConfig,
    settings: Any,
) -> dict[str, Any]:
    symbol = _select_symbol(config.symbols, idx)
    route = resolve_symbol_route(symbol)
    quarantine = route_quarantine_status(route.route_id)
    strategy = load_strategy_config(route.strategy_config)
    analysis_payload, artifacts = run_analysis(
        settings,
        symbol=symbol,
        market=strategy.defaults.market,
        interval=strategy.defaults.interval,
        limit=max(strategy.defaults.limit, 240),
        use_blave=strategy.defaults.use_blave,
        render_chart_flag=strategy.defaults.render_chart,
        strategy=strategy,
    )
    backtest_payload = run_backtest(
        settings,
        strategy=strategy,
        symbol=symbol,
        market=strategy.defaults.market,
        interval=strategy.defaults.interval,
        limit=max(strategy.defaults.limit, 320),
        output_dir=artifacts.output_dir / f"training-backtest-{idx:04d}",
    )
    candidate_side = mission_candidate_side(analysis_payload)
    generated_at = _utc_now()
    signal_scores = build_signal_scores(
        route=route,
        latest=analysis_payload.get("latest") or {},
        analysis=analysis_payload.get("analysis") or {},
        trade_plan=analysis_payload.get("trade_plan") or {},
        side=candidate_side or "BUY",
    )
    live_plan = None
    if candidate_side:
        live_plan = build_live_execution_plan(
            settings,
            strategy,
            analysis_payload,
            side_override=candidate_side,
            margin_notional_usdt=config.margin_notional_usdt,
            require_optimizer_gate=False,
        ).to_dict()
    selected_trade = None
    trade_reason = "no-actionable-signal"
    paper_record = None
    paper_journal_path = None
    review_record = None
    review_journal_path = None
    plan_violations = [
        str(item)
        for item in (live_plan or {}).get("violations", [])
        if "Global strategy optimizer" not in str(item)
    ]
    if quarantine["quarantined"]:
        trade_reason = "route-quarantined-pending-manual-review"
    elif plan_violations:
        trade_reason = "live-plan-structure-blocked"
    elif candidate_side and live_plan and (_safe_float(live_plan.get("quantity")) > 0):
        trades = list((backtest_payload.get("summary") or {}).get("trades") or [])
        aligned_trades = [item for item in trades if _backtest_trade_matches_side(item, candidate_side)]
        min_convergence = float(getattr(strategy.risk, "min_convergence", 0.6))
        convergence_ok = float((analysis_payload.get("analysis") or {}).get("convergence") or 0.0) >= min_convergence
        if aligned_trades and convergence_ok:
            trade_index = idx % len(aligned_trades)
            selected_trade = aligned_trades[trade_index]
            trade_reason = "market-replay-sampled"
            paper_record = _paper_record_from_plan(
                generated_at=generated_at,
                symbol=symbol,
                route=route,
                strategy=strategy,
                analysis_payload=analysis_payload,
                side=candidate_side,
                plan_payload=live_plan,
                signal_scores=signal_scores,
                note=(
                    "Training simulation entry derived from live-plan structure and closed via market replay. "
                    "Journal only. No Binance order has been sent."
                ),
            )
            paper_journal_path = append_paper_order(paper_record)
            review_record = _simulate_review_from_backtest(
                generated_at=generated_at,
                idx=idx,
                symbol=symbol,
                route=route,
                strategy=strategy,
                paper_record=paper_record,
                plan_payload=live_plan,
                trade=selected_trade,
            )
            review_journal_path = append_closed_trade_review(review_record)
        elif trades and not aligned_trades:
            trade_reason = "backtest-side-mismatch"
        elif aligned_trades and not convergence_ok:
            trade_reason = "analysis-convergence-below-live-threshold"
        else:
            trade_reason = "backtest-produced-no-trades"
    return {
        "sample_index": idx,
        "symbol": symbol,
        "route_id": route.route_id,
        "route_quarantine": quarantine,
        "asset_class": route.asset_class,
        "strategy_profile": strategy.profile,
        "analysis_bias": (analysis_payload.get("analysis") or {}).get("bias"),
        "analysis_score": (analysis_payload.get("analysis") or {}).get("score"),
        "analysis_convergence": (analysis_payload.get("analysis") or {}).get("convergence"),
        "candidate_side": candidate_side,
        "live_plan_allowed": bool((live_plan or {}).get("allowed", False)),
        "live_plan_violations": list((live_plan or {}).get("violations") or []),
        "live_plan_warnings": list((live_plan or {}).get("warnings") or []),
        "backtest_trade_count": int((backtest_payload.get("summary") or {}).get("trade_count") or 0),
        "backtest_profit_factor": float((backtest_payload.get("summary") or {}).get("profit_factor") or 0.0),
        "backtest_total_return_pct": float((backtest_payload.get("summary") or {}).get("total_return_pct") or 0.0),
        "training_status": "review-recorded" if review_record else "skipped",
        "training_reason": trade_reason,
        "analysis_report": str((analysis_payload.get("artifacts") or {}).get("report_json") or ""),
        "backtest_report": str(((backtest_payload.get("artifacts") or {}).get("report_json")) or ""),
        "paper_order": asdict(paper_record) if paper_record else None,
        "paper_journal_path": str(paper_journal_path) if paper_journal_path else None,
        "review": asdict(review_record) if review_record else None,
        "review_journal_path": str(review_journal_path) if review_journal_path else None,
        "signal_scores": signal_scores,
        "selected_backtest_trade": selected_trade,
    }


def _derive_findings(results: list[dict[str, Any]]) -> list[str]:
    findings: list[str] = []
    if not results:
        return ["training-produced-no-results"]
    skipped = [item for item in results if item.get("training_status") != "review-recorded"]
    if skipped:
        findings.append(f"training-skip-rate={len(skipped)}/{len(results)}")
    blocked = [item for item in results if item.get("live_plan_violations")]
    if blocked:
        findings.append("live-plan-still-frequently-blocked-during-training")
    weak_pf = [
        item for item in results
        if float(item.get("backtest_profit_factor") or 0.0) < 1.2
    ]
    if weak_pf:
        findings.append("backtest-profit-factor-remains-the-main-strategic-bottleneck")
    low_signal = [item for item in results if not item.get("candidate_side")]
    if low_signal:
        findings.append("some-training-samples-never-formed-an-actionable-side")
    return findings


def run_demo_training(
    *,
    rounds: int,
    symbols: list[str] | None = None,
    target_return_pct: float = 5.0,
    max_leverage: float = 3.0,
    margin_notional_usdt: float = 4.0,
    optimize_every: int = 10,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    DEFAULT_TRAINING_STATE_DIR.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    training_symbols = tuple(item.strip().upper() for item in (symbols or list(DEFAULT_SYMBOLS)) if item.strip())
    if not training_symbols:
        training_symbols = DEFAULT_SYMBOLS
    config = TrainingConfig(
        symbols=training_symbols,
        target_return_pct=target_return_pct,
        max_leverage=max_leverage,
        margin_notional_usdt=margin_notional_usdt,
        training_sample_size=max(int(rounds), 1),
        optimize_every=max(int(optimize_every), 1),
        include_research_only=False,
    )
    results: list[dict[str, Any]] = []
    optimizer_reports: list[dict[str, Any]] = []
    for idx in range(config.training_sample_size):
        result = _build_training_iteration(idx=idx, config=config, settings=settings)
        results.append(result)
        if (idx + 1) % config.optimize_every == 0:
            optimizer_payload = run_strategy_optimizer()
            optimizer_reports.append(
                {
                    "after_round": idx + 1,
                    "status": optimizer_payload.get("status"),
                    "review_count": optimizer_payload.get("review_count"),
                    "promotion_decision": optimizer_payload.get("promotion_decision"),
                    "report_path": optimizer_payload.get("report_path"),
                }
            )
    if not optimizer_reports or optimizer_reports[-1].get("after_round") != config.training_sample_size:
        optimizer_payload = run_strategy_optimizer()
        optimizer_reports.append(
            {
                "after_round": config.training_sample_size,
                "status": optimizer_payload.get("status"),
                "review_count": optimizer_payload.get("review_count"),
                "promotion_decision": optimizer_payload.get("promotion_decision"),
                "report_path": optimizer_payload.get("report_path"),
            }
        )

    recorded = [item for item in results if item.get("training_status") == "review-recorded"]
    wins = sum(
        1
        for item in recorded
        if _safe_float(((item.get("review") or {}).get("realized_pnl_usdt"))) > 0.0
    )
    losses = sum(
        1
        for item in recorded
        if _safe_float(((item.get("review") or {}).get("realized_pnl_usdt"))) < 0.0
    )
    total_pnl = round(
        sum(_safe_float(((item.get("review") or {}).get("realized_pnl_usdt"))) for item in recorded),
        8,
    )
    payload = {
        "generated_at": _utc_now().isoformat(),
        "status": "ok",
        "mode": "demo_training_market_replay",
        "config": asdict(config),
        "rounds_requested": config.training_sample_size,
        "recorded_review_count": len(recorded),
        "wins": wins,
        "losses": losses,
        "total_realized_pnl_usdt": total_pnl,
        "findings": _derive_findings(results),
        "results": results,
        "optimizer_reports": optimizer_reports,
    }
    report_path = DEFAULT_TRAINING_STATE_DIR / f"{_stamp()}-demo-training.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload
