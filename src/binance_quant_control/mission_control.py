from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .analysis import run_analysis
from .asset_routing import normalize_symbol, resolve_symbol_route
from .backtest import run_backtest
from .config import PROJECT_ROOT, STATE_DIR, ensure_runtime_dirs, load_settings
from .convergence import build_cohort_id
from .daily_digest import DEFAULT_NEWS_FEEDS
from .daily_digest import load_config as load_digest_config
from .live_execution import build_live_execution_plan, execute_live_order
from .order_journal import PaperOrderRecord, append_paper_order, read_closed_trade_reviews
from .route_risk_control import route_quarantine_status
from .signal_scoring import build_signal_scores
from .strategy import load_strategy_config
from .strategy_baselines import baseline_for_route
from .strategy_optimizer import run_strategy_optimizer

DEFAULT_MISSION_CONFIG_PATH = PROJECT_ROOT / "config" / "mission-control.default.yaml"
MISSION_STATE_DIR = STATE_DIR / "mission-control"


@dataclass(frozen=True, slots=True)
class MissionExecutionConfig:
    paper_order_notional_usdt: float
    min_candidate_composite_score: float
    select_best_symbol_only: bool
    max_symbols_per_run: int
    force_simulation_after_analysis: bool
    require_backtest_screening_pass: bool
    run_optimizer_after_mission: bool
    run_review_after_mission: bool


@dataclass(frozen=True, slots=True)
class MissionScheduleConfig:
    scout_interval_minutes: int
    guardian_interval_minutes: int
    digest_interval_minutes: int
    optimizer_interval_hours: int
    review_interval_hours: int


@dataclass(frozen=True, slots=True)
class MissionBoundaryConfig:
    local_steps: tuple[str, ...]
    cloud_steps: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MissionConfig:
    path: Path
    execution: MissionExecutionConfig
    scheduling: MissionScheduleConfig
    boundaries: MissionBoundaryConfig


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def resolve_mission_config_path(path: str | Path | None = None) -> Path:
    candidate = Path(path or DEFAULT_MISSION_CONFIG_PATH).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()
    return (PROJECT_ROOT / "config" / candidate).resolve()


def load_mission_config(path: str | Path | None = None) -> MissionConfig:
    config_path = resolve_mission_config_path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    execution = payload.get("execution") or {}
    scheduling = payload.get("scheduling") or {}
    boundaries = payload.get("boundaries") or {}
    return MissionConfig(
        path=config_path,
        execution=MissionExecutionConfig(
            paper_order_notional_usdt=float(execution.get("paper_order_notional_usdt") or 3.0),
            min_candidate_composite_score=float(execution.get("min_candidate_composite_score") or 55.0),
            select_best_symbol_only=bool(execution.get("select_best_symbol_only", True)),
            max_symbols_per_run=int(execution.get("max_symbols_per_run") or 6),
            force_simulation_after_analysis=bool(execution.get("force_simulation_after_analysis", True)),
            require_backtest_screening_pass=bool(execution.get("require_backtest_screening_pass", True)),
            run_optimizer_after_mission=bool(execution.get("run_optimizer_after_mission", True)),
            run_review_after_mission=bool(execution.get("run_review_after_mission", True)),
        ),
        scheduling=MissionScheduleConfig(
            scout_interval_minutes=int(scheduling.get("scout_interval_minutes") or 30),
            guardian_interval_minutes=int(scheduling.get("guardian_interval_minutes") or 5),
            digest_interval_minutes=int(scheduling.get("digest_interval_minutes") or 240),
            optimizer_interval_hours=int(scheduling.get("optimizer_interval_hours") or 6),
            review_interval_hours=int(scheduling.get("review_interval_hours") or 2),
        ),
        boundaries=MissionBoundaryConfig(
            local_steps=tuple(str(item) for item in (boundaries.get("local_steps") or [])),
            cloud_steps=tuple(str(item) for item in (boundaries.get("cloud_steps") or [])),
        ),
    )


def mission_candidate_side(analysis_payload: dict[str, Any]) -> str | None:
    analysis = analysis_payload.get("analysis") or {}
    action = str(analysis.get("recommended_action") or "").upper()
    if action in {"BUY", "SELL"}:
        return action
    bias = str(analysis.get("bias") or "")
    if "long" in bias:
        return "BUY"
    if "short" in bias:
        return "SELL"
    return None


def _closed_trade_count() -> int:
    return len(read_closed_trade_reviews())


def _configured_news_feed_count() -> int:
    digest_config_path = PROJECT_ROOT / "config" / "n8n-daily-digest.default.json"
    try:
        digest_config = load_digest_config(digest_config_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return len(DEFAULT_NEWS_FEEDS)
    feeds = [str(url).strip() for url in digest_config.get("news_feeds") or DEFAULT_NEWS_FEEDS if str(url).strip()]
    return len(set(feeds))


def _build_paper_record(
    *,
    symbol: str,
    route: Any,
    strategy: Any,
    analysis_payload: dict[str, Any],
    side: str,
    margin_notional_usdt: float,
    leverage: float,
) -> PaperOrderRecord:
    price = float((analysis_payload.get("latest") or {}).get("close") or 0.0)
    gross_notional = margin_notional_usdt * leverage
    quantity = gross_notional / price if price > 0 else 0.0
    signal_scores = build_signal_scores(
        route=route,
        latest=analysis_payload.get("latest") or {},
        analysis=analysis_payload.get("analysis") or {},
        trade_plan=analysis_payload.get("trade_plan") or {},
    )
    return PaperOrderRecord(
        generated_at=_utc_now().isoformat(),
        kind="paper-order",
        symbol=symbol,
        market=strategy.defaults.market,
        side=side,
        margin_notional_usdt=round(margin_notional_usdt, 6),
        leverage=round(leverage, 6),
        gross_notional_usdt=round(gross_notional, 6),
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
        note="Mission-control simulation entry. Journal only. No live order has been sent.",
    )


def _promotion_allows_candidate(backtest_payload: dict[str, Any], *, require_screening_pass: bool) -> bool:
    convergence = backtest_payload.get("convergence") or {}
    robustness = backtest_payload.get("robustness") or {}
    if robustness and not bool(robustness.get("passed", False)):
        return False
    if require_screening_pass:
        return str(convergence.get("screening_status") or "") == "passed"
    return str(convergence.get("promotion_decision") or "") != "reject"


def _candidate_issues(
    *,
    symbol: str,
    route: Any,
    strategy: Any,
    analysis_payload: dict[str, Any],
    backtest_payload: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    latest = analysis_payload.get("latest") or {}
    analysis = analysis_payload.get("analysis") or {}
    if float(latest.get("adx") or 0.0) < float(strategy.risk.min_adx):
        issues.append("trend-strength-is-soft")
    if float(analysis.get("convergence") or 0.0) < 0.65:
        issues.append("convergence-is-weak")
    if float((backtest_payload.get("summary") or {}).get("profit_factor") or 0.0) < 1.2:
        issues.append("backtest-profit-factor-too-low")
    robustness = backtest_payload.get("robustness") or {}
    if robustness and not bool(robustness.get("passed", False)):
        issues.append("backtest-robustness-gate-failed")
    quarantine = route_quarantine_status(route.route_id)
    if quarantine["quarantined"]:
        issues.append("route-is-quarantined-pending-manual-review")
    if symbol == "PAXGUSDT":
        issues.append("xau-is-using-tokenized-gold-proxy-not-native-spot-xau")
    return issues


def _system_findings(normalized_symbols: list[str]) -> list[str]:
    findings: list[str] = []
    normalized_set = {symbol.upper() for symbol in normalized_symbols}
    if _configured_news_feed_count() < 3:
        findings.append("current-news-ingestion-has-fewer-than-three-rss-feeds-so-event-risk-is-still-thin")
    if normalized_set & {"PAXGUSDT", "XAUTUSDT", "XAUUSD", "XAU", "GOLD"}:
        findings.append("xau-needs-tokenized-proxy-because-native-metals-execution-is-not-in-this-binance-core")
    if _closed_trade_count() < 50:
        findings.append("closed-trade-cohort-is-still-below-50-review-validation-threshold")
    if len(normalized_symbols) > 1:
        findings.append("multi-symbol-mission-still-selects-one-best-candidate-for-automation-to-avoid-overtrading")
    return findings


def run_trading_mission(
    *,
    symbols: list[str],
    target_return_pct: float,
    max_leverage: float,
    execute_live: bool = False,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    MISSION_STATE_DIR.mkdir(parents=True, exist_ok=True)
    mission_config = load_mission_config(config_path)
    settings = load_settings()

    normalized_symbols = [normalize_symbol(symbol) for symbol in symbols if str(symbol).strip()]
    normalized_symbols = normalized_symbols[: mission_config.execution.max_symbols_per_run]
    stamp = _stamp()
    symbol_reports: list[dict[str, Any]] = []

    for requested_symbol, normalized_symbol in zip(symbols, normalized_symbols, strict=False):
        route = resolve_symbol_route(normalized_symbol)
        strategy = load_strategy_config(route.strategy_config)
        strategy = replace(
            strategy,
            defaults=replace(
                strategy.defaults,
                symbol=normalized_symbol,
                market=route.market,
                interval=route.interval,
            ),
            risk=replace(
                strategy.risk,
                default_leverage=min(strategy.risk.default_leverage, int(max_leverage)),
                max_leverage=min(strategy.risk.max_leverage, int(max_leverage)),
            ),
        )
        analysis_payload, artifacts = run_analysis(
            settings,
            symbol=normalized_symbol,
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
            symbol=normalized_symbol,
            market=strategy.defaults.market,
            interval=strategy.defaults.interval,
            limit=max(strategy.defaults.limit, 320),
            output_dir=artifacts.output_dir / "mission-backtest",
        )
        signal_scores = build_signal_scores(
            route=route,
            latest=analysis_payload.get("latest") or {},
            analysis=analysis_payload.get("analysis") or {},
            trade_plan=analysis_payload.get("trade_plan") or {},
        )
        symbol_reports.append(
            {
                "requested_symbol": requested_symbol,
                "symbol": normalized_symbol,
                "route": route.to_dict(),
                "strategy_profile": strategy.profile,
                "strategy_path": str(strategy.path),
                "baseline": (
                    baseline_for_route(route.route_id).to_dict()
                    if baseline_for_route(route.route_id) is not None
                    else None
                ),
                "analysis": {
                    "bias": (analysis_payload.get("analysis") or {}).get("bias"),
                    "recommended_action": (analysis_payload.get("analysis") or {}).get("recommended_action"),
                    "score": (analysis_payload.get("analysis") or {}).get("score"),
                    "convergence": (analysis_payload.get("analysis") or {}).get("convergence"),
                    "entry_ready": (analysis_payload.get("analysis") or {}).get("entry_ready"),
                },
                "signal_scores": signal_scores,
                "backtest_summary": backtest_payload.get("summary"),
                "backtest_convergence": backtest_payload.get("convergence"),
                "issues": _candidate_issues(
                    symbol=normalized_symbol,
                    route=route,
                    strategy=strategy,
                    analysis_payload=analysis_payload,
                    backtest_payload=backtest_payload,
                ),
            }
        )

    symbol_reports.sort(
        key=lambda item: (
            float((item.get("signal_scores") or {}).get("composite_convergence_score") or 0.0),
            float((item.get("backtest_summary") or {}).get("profit_factor") or 0.0),
            float((item.get("backtest_summary") or {}).get("total_return_pct") or 0.0),
        ),
        reverse=True,
    )
    selected = symbol_reports[0] if symbol_reports else None
    mission_actions: list[str] = []
    simulation: dict[str, Any] | None = None
    live: dict[str, Any] | None = None

    if selected:
        selected_symbol = str(selected["symbol"])
        selected_route = resolve_symbol_route(selected_symbol)
        selected_strategy = load_strategy_config(selected_route.strategy_config)
        selected_strategy = replace(
            selected_strategy,
            defaults=replace(
                selected_strategy.defaults,
                symbol=selected_symbol,
                market=selected_route.market,
                interval=selected_route.interval,
            ),
            risk=replace(
                selected_strategy.risk,
                default_leverage=min(selected_strategy.risk.default_leverage, int(max_leverage)),
                max_leverage=min(selected_strategy.risk.max_leverage, int(max_leverage)),
            ),
        )
        selected_analysis, selected_artifacts = run_analysis(
            settings,
            symbol=selected_symbol,
            market=selected_strategy.defaults.market,
            interval=selected_strategy.defaults.interval,
            limit=max(selected_strategy.defaults.limit, 240),
            use_blave=selected_strategy.defaults.use_blave,
            render_chart_flag=selected_strategy.defaults.render_chart,
            strategy=selected_strategy,
        )
        selected_backtest = run_backtest(
            settings,
            strategy=selected_strategy,
            symbol=selected_symbol,
            market=selected_strategy.defaults.market,
            interval=selected_strategy.defaults.interval,
            limit=max(selected_strategy.defaults.limit, 320),
            output_dir=selected_artifacts.output_dir / "mission-backtest",
        )
        candidate_side = mission_candidate_side(selected_analysis)
        summary = selected_backtest.get("summary") or {}
        total_return = float(summary.get("total_return_pct") or 0.0)
        composite = float((selected.get("signal_scores") or {}).get("composite_convergence_score") or 0.0)
        screening_ok = _promotion_allows_candidate(
            selected_backtest,
            require_screening_pass=mission_config.execution.require_backtest_screening_pass,
        )
        should_force_simulate = bool(mission_config.execution.force_simulation_after_analysis and candidate_side)
        passes_promotion_gate = bool(
            candidate_side
            and composite >= mission_config.execution.min_candidate_composite_score
            and screening_ok
            and total_return >= target_return_pct
        )
        if should_force_simulate:
            record = _build_paper_record(
                symbol=selected_symbol,
                route=selected_route,
                strategy=selected_strategy,
                analysis_payload=selected_analysis,
                side=candidate_side,
                margin_notional_usdt=mission_config.execution.paper_order_notional_usdt,
                leverage=float(min(selected_strategy.risk.default_leverage, int(max_leverage))),
            )
            record = replace(
                record,
                note=(
                    "Forced simulation after analysis for market validation. Journal only. No live order has been sent."
                    if not passes_promotion_gate
                    else record.note
                ),
            )
            journal_path = append_paper_order(record)
            simulation = {
                "status": "recorded_for_market_validation" if not passes_promotion_gate else "recorded",
                "forced_after_analysis": should_force_simulate,
                "passes_promotion_gate": passes_promotion_gate,
                "journal_path": str(journal_path),
                "paper_order": asdict(record),
            }
            mission_actions.append("paper-order-recorded")
            if execute_live and passes_promotion_gate:
                live_plan = build_live_execution_plan(
                    settings,
                    selected_strategy,
                    selected_analysis,
                    side_override=candidate_side,
                )
                live = {
                    "plan": live_plan.to_dict(),
                    "executed": False,
                }
                if live_plan.allowed:
                    live["execution"] = execute_live_order(
                        settings,
                        selected_strategy,
                        live_plan,
                        entry_reason_snapshot={
                            "bias": str((selected_analysis.get("analysis") or {}).get("bias") or ""),
                            "score": int((selected_analysis.get("analysis") or {}).get("score") or 0),
                            "convergence": float((selected_analysis.get("analysis") or {}).get("convergence") or 0.0),
                            "interval": selected_strategy.defaults.interval,
                        },
                        signal_scores=selected.get("signal_scores") or {},
                    )
                    live["executed"] = True
                    mission_actions.append("live-order-executed")
                else:
                    mission_actions.append("live-plan-blocked")
            elif execute_live and not passes_promotion_gate:
                mission_actions.append("live-skipped-promotion-gate-failed")
        else:
            mission_actions.append("candidate-kept-in-research-only")

    optimizer: dict[str, Any] | None = None
    if mission_config.execution.run_optimizer_after_mission:
        optimizer = run_strategy_optimizer()
        mission_actions.append("strategy-optimizer-ran")

    payload = {
        "generated_at": _utc_now().isoformat(),
        "status": "ok",
        "symbols_requested": symbols,
        "symbols_normalized": normalized_symbols,
        "target_return_pct": target_return_pct,
        "max_leverage": max_leverage,
        "execute_live": execute_live,
        "mission_config": str(mission_config.path),
        "scheduling": asdict(mission_config.scheduling),
        "boundaries": asdict(mission_config.boundaries),
        "system_findings": _system_findings(normalized_symbols),
        "mission_actions": mission_actions,
        "selected_candidate": selected,
        "symbol_reports": symbol_reports,
        "simulation": simulation,
        "live": live,
        "optimizer": {
            "status": optimizer.get("status"),
            "report_path": optimizer.get("report_path"),
        }
        if isinstance(optimizer, dict)
        else None,
    }
    report_path = MISSION_STATE_DIR / f"{stamp}-mission-control.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload
