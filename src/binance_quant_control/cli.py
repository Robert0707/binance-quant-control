from __future__ import annotations

import argparse
import json
import secrets
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ai_expectancy_upgrade import run_ai_expectancy_upgrade
from .ai_goal_loop import run_ai_goal_loop
from .ai_market_sentinel import run_ai_market_sentinel
from .ai_surface_audit import run_ai_surface_audit
from .alpha_research import run_aggressive_alpha_research
from .analysis import run_analysis
from .asset_routing import AssetRoute, resolve_symbol_route
from .backtest import run_backtest
from .binance_api import BinanceAPIError, BinanceClient, LiveTradingDisabledError
from .challenge import (
    challenge_scope_key,
    challenge_summary_dict,
    initialize_challenge,
    load_challenge_state,
    read_balance_snapshots,
    record_balance_snapshot,
)
from .config import (
    CONFIG_DIR,
    ENV_PATH,
    PROJECT_ROOT,
    REPORTS_DIR,
    RUN_DIR,
    STATE_DIR,
    TASK_SPEC_DIR,
    ensure_runtime_dirs,
    load_settings,
)
from .convergence import build_cohort_id
from .decision_audit import run_decision_audit
from .decision_output import (
    build_ai_exit_decision_output,
    build_ai_trade_decision_output,
    build_blocked_trade_decision_output,
)
from .external_context import build_external_context, external_context_key_status
from .feature_dataset import FeatureDatasetSpec, build_feature_dataset
from .final_convergence_audit import run_final_convergence_audit
from .hermes_ai_trader import run_hermes_ai_trader
from .hermes_trade_loop import (
    DEFAULT_HERMES_TRADE_CONFIG_PATH,
    hermes_trade_status,
    run_hermes_trade_cycle,
    run_hermes_trade_daemon,
    start_hermes_trade_loop,
    stop_hermes_trade_loop,
)
from .high_win_convergence import run_high_win_convergence_loop
from .high_win_iteration import compact_high_win_iteration, run_high_win_iteration
from .intent_router import resolve_operator_intent
from .live_execution import build_live_execution_plan, execute_live_order
from .loss_diagnostics import run_loss_diagnostics
from .market_bot_gate import DEFAULT_MARKET_BOT_GATE_CONFIG, evaluate_market_bot_gate
from .mission_control import run_trading_mission
from .new_symbol_workflow import run_new_symbol_workflow
from .operator_dashboard import build_operator_dashboard
from .order_journal import (
    ClosedTradeReviewRecord,
    PaperOrderRecord,
    append_closed_trade_review,
    append_paper_order,
    backfill_closed_trade_review_metadata,
    backfill_live_order_metadata,
    load_trade_state,
    read_closed_trade_reviews,
    read_live_orders,
    save_trade_state,
    summarize_closed_trade_reviews,
    summarize_live_orders,
    summarize_paper_orders,
    write_closed_trade_reviews,
    write_live_orders,
)
from .position_manager import (
    build_adaptive_exit_plan,
    build_position_management_plan,
    execute_position_management_plan,
)
from .professional_system_audit import run_professional_system_audit
from .protective_repair import (
    build_staged_take_profit_repair_plan,
    execute_staged_take_profit_repair,
)
from .public_history_training import run_public_history_training
from .readiness_scanner import run_ai_readiness_scan
from .repository_audit import run_repository_audit
from .risk_combo_sweep import build_risk_combo_matrix_report, run_risk_combo_sweep
from .route_risk_control import (
    clear_route_quarantine,
    load_route_risk_state,
    route_quarantine_status,
)
from .signal_scoring import build_signal_scores
from .strategy import load_strategy_config
from .supervision import build_supervisor_policy, run_delivery_supervisor
from .trade_session import start_trade_session, stop_trade_session, trade_session_status
from .trading_control import (
    AutoPausePolicy,
    evaluate_auto_pause_conditions,
    load_trading_control_state,
    set_trading_paused,
)
from .training import run_demo_training

HOSTCTL = Path("/usr/local/bin/openclaw-hostctl")
TASKCTL = Path("/home/robert/.openclaw/bin/openclaw-taskctl")


def print_json(payload: Any, *, compact: bool = False) -> None:
    if compact:
        print(json.dumps(payload, separators=(',', ':'), ensure_ascii=False))
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


def split_csv_arg(value: Any) -> list[str]:
    values = value if isinstance(value, (list, tuple)) else [value]
    items: list[str] = []
    for raw in values:
        if raw is None:
            continue
        for item in str(raw).split(","):
            stripped = item.strip()
            if stripped:
                items.append(stripped)
    return items


def run(cmd: list[str], *, timeout: int = 120) -> dict[str, Any]:
    import subprocess

    completed = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    return {
        "returncode": completed.returncode,
        "stdout": (completed.stdout or "").strip(),
        "stderr": (completed.stderr or "").strip(),
    }


def parse_json_result(result: dict[str, Any], context: str) -> Any:
    if result["returncode"] != 0:
        raise SystemExit(result["stderr"] or result["stdout"] or f"{context} failed")
    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{context} returned invalid JSON") from exc


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _iso_to_millis(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp() * 1000)


def _algo_status(algo_payload: dict[str, Any] | None) -> str:
    if not algo_payload:
        return "MISSING"
    return str(algo_payload.get("algoStatus") or algo_payload.get("status") or "UNKNOWN").upper()


def _algo_triggered(algo_payload: dict[str, Any] | None) -> bool:
    status = _algo_status(algo_payload)
    return status in {"TRIGGERED", "FILLED", "FINISHED", "SUCCESS", "CLOSED"}


def _algo_terminal(algo_payload: dict[str, Any] | None) -> bool:
    status = _algo_status(algo_payload)
    return status in {"TRIGGERED", "FILLED", "FINISHED", "SUCCESS", "CLOSED", "CANCELED", "CANCELLED", "EXPIRED", "REJECTED"}


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _build_paper_order_record(
    *,
    route: AssetRoute,
    symbol: str,
    market: str,
    interval: str,
    side: str,
    notional_usdt: float,
    leverage: float,
    price: float,
    analysis: dict[str, Any],
    strategy_profile: str | None,
    strategy_path: str | None,
    note: str,
) -> PaperOrderRecord:
    gross_notional = float(notional_usdt) * leverage
    quantity = gross_notional / price if price else 0.0
    latest = analysis.get("latest") or {}
    insight = analysis.get("analysis") or {}
    signal_scores = build_signal_scores(
        route=route,
        latest=latest,
        analysis=insight,
        trade_plan=analysis.get("trade_plan") or {},
        side=side,
    )
    cohort_id = build_cohort_id(
        asset_class=route.asset_class,
        strategy_profile=strategy_profile or "unprofiled",
        market=market,
        interval=interval,
    )
    return PaperOrderRecord(
        generated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        kind="paper-order",
        symbol=symbol,
        market=market,
        side=side.upper(),
        margin_notional_usdt=round(float(notional_usdt), 6),
        leverage=leverage,
        gross_notional_usdt=round(gross_notional, 6),
        reference_price=round(price, 6),
        estimated_quantity=round(quantity, 8),
        analysis_bias=str(insight.get("bias") or ""),
        analysis_score=int(insight.get("score") or 0),
        analysis_convergence=float(insight.get("convergence") or 0.0),
        cohort_id=cohort_id,
        strategy_profile=strategy_profile,
        strategy_path=strategy_path,
        asset_class=route.asset_class,
        route_id=route.route_id,
        simulation_mode=route.simulation_mode,
        review_lane=route.review_lane,
        entry_reason_snapshot={
            "bias": str(insight.get("bias") or ""),
            "score": int(insight.get("score") or 0),
            "convergence": float(insight.get("convergence") or 0.0),
            "interval": interval,
        },
        signal_scores=signal_scores,
        analysis_report=analysis["artifacts"]["report_json"],
        chart_path=analysis["artifacts"]["chart_path"],
        note=note,
    )


def _latest_update_timestamp(*payloads: dict[str, Any] | None) -> int | None:
    candidates = []
    for payload in payloads:
        if not payload:
            continue
        millis = _first_int(payload.get("updateTime"), payload.get("time"), payload.get("createTime"))
        if millis is not None:
            candidates.append(millis)
    return max(candidates) if candidates else None


def _build_closed_trade_review(
    client: BinanceClient,
    record: dict[str, Any],
    positions_by_symbol: dict[str, list[dict[str, Any]]],
) -> ClosedTradeReviewRecord | None:
    symbol = str(record.get("symbol", "")).upper()
    market = str(record.get("market", "futures"))
    if market != "futures" or not symbol:
        return None

    open_positions = positions_by_symbol.get(symbol, [])
    if any(abs(float(item.get("positionAmt", 0.0) or 0.0)) > 0.0 for item in open_positions):
        return None

    source_order_id = record.get("order_id")
    entry_payload = (((record.get("binance_response") or {}).get("entry")) or {})
    protective_orders = ((record.get("binance_response") or {}).get("protective_orders")) or {}
    stop_seed = protective_orders.get("stop_loss") or {}
    take_seed = protective_orders.get("take_profit") or {}

    try:
        entry_order = client.query_order(symbol, int(source_order_id), market="futures") if source_order_id is not None else {}
    except (BinanceAPIError, TypeError, ValueError):
        entry_order = {}

    try:
        algo_history = client.all_algo_orders(symbol, limit=100)
    except BinanceAPIError:
        algo_history = []
    algo_by_id = {int(item.get("algoId")): item for item in algo_history if item.get("algoId") is not None}
    stop_algo = algo_by_id.get(_first_int(stop_seed.get("algoId")) or -1)
    take_algo = algo_by_id.get(_first_int(take_seed.get("algoId")) or -1)

    if not any((_algo_terminal(stop_algo), _algo_terminal(take_algo), _algo_triggered(stop_algo), _algo_triggered(take_algo))):
        return None

    if _algo_triggered(take_algo):
        exit_reason = "take_profit"
        exit_payload = take_algo
    elif _algo_triggered(stop_algo):
        exit_reason = "stop_loss"
        exit_payload = stop_algo
    else:
        exit_reason = "manual_close"
        exit_payload = take_algo or stop_algo or {}

    opened_at = str(record.get("timestamp"))
    closed_ms = _latest_update_timestamp(exit_payload, entry_order, take_algo, stop_algo) or int(datetime.now(timezone.utc).timestamp() * 1000)
    closed_at = datetime.fromtimestamp(closed_ms / 1000, tz=timezone.utc).replace(microsecond=0).isoformat()
    entry_price = _first_float(
        entry_order.get("avgPrice"),
        entry_payload.get("avgPrice"),
        entry_order.get("price"),
        entry_payload.get("price"),
        record.get("price"),
    ) or 0.0
    exit_price = _first_float(
        exit_payload.get("actualPrice"),
        exit_payload.get("avgPrice"),
        exit_payload.get("price"),
        exit_payload.get("triggerPrice"),
    )
    quantity = _first_float(entry_order.get("executedQty"), entry_payload.get("executedQty"), record.get("quantity")) or 0.0
    leverage = int(record.get("leverage", 0) or 0)
    stop_loss_price = _first_float(stop_algo.get("triggerPrice") if stop_algo else None, stop_seed.get("triggerPrice"))
    take_profit_price = _first_float(take_algo.get("triggerPrice") if take_algo else None, take_seed.get("triggerPrice"))
    gross_notional = _first_float(record.get("gross_notional_usdt")) or 0.0
    risk_distance = abs(entry_price - stop_loss_price) if stop_loss_price is not None else 0.0
    risk_usdt = quantity * risk_distance if quantity > 0 and risk_distance > 0 else 0.0

    try:
        income_rows = client.income_history(
            symbol,
            income_type="REALIZED_PNL",
            start_time=max(_iso_to_millis(opened_at) - 60_000, 0),
            end_time=closed_ms + 900_000,
            limit=100,
        )
    except BinanceAPIError:
        income_rows = []
    realized_pnl_usdt = round(
        sum(
            float(item.get("income", 0.0) or 0.0)
            for item in income_rows
            if str(item.get("symbol", "")).upper() == symbol
        ),
        8,
    )
    realized_pnl_pct = round((realized_pnl_usdt / gross_notional) * 100.0, 4) if gross_notional > 0 else 0.0
    realized_r_multiple = round(realized_pnl_usdt / risk_usdt, 4) if risk_usdt > 0 else None
    symbol_route = resolve_symbol_route(symbol)
    strategy_profile = str(record.get("strategy_profile", "")) or symbol_route.strategy_config.stem
    entry_reason_snapshot = record.get("entry_reason_snapshot")
    if not isinstance(entry_reason_snapshot, dict):
        entry_reason_snapshot = None
    interval_tag = str((entry_reason_snapshot or {}).get("interval") or "unknown")
    cohort_id = str(
        record.get("cohort_id")
        or build_cohort_id(
            asset_class=symbol_route.asset_class,
            strategy_profile=strategy_profile,
            market=market,
            interval=interval_tag,
        )
    )
    rule_compliant = exit_reason in {"take_profit", "stop_loss", "manual_close"}
    false_positive_tag = None
    if realized_pnl_usdt < 0 and float(record.get("analysis_convergence", 0.0) or 0.0) >= 0.8:
        false_positive_tag = "high_conviction_loss"
    market_regime_tag = "trend"
    if symbol_route.asset_class == "meme_high_beta":
        market_regime_tag = "high_beta"
    elif symbol_route.asset_class == "defensive_unknown":
        market_regime_tag = "unknown"

    return ClosedTradeReviewRecord(
        reviewed_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        opened_at=opened_at,
        closed_at=closed_at,
        source_order_id=source_order_id,
        symbol=symbol,
        market=market,
        side=str(record.get("side", "")),
        quantity=round(quantity, 8),
        leverage=leverage,
        entry_price=round(entry_price, 8),
        exit_price=round(exit_price, 8) if exit_price is not None else None,
        stop_loss_price=round(stop_loss_price, 8) if stop_loss_price is not None else None,
        take_profit_price=round(take_profit_price, 8) if take_profit_price is not None else None,
        exit_reason=exit_reason,
        realized_pnl_usdt=realized_pnl_usdt,
        realized_pnl_pct=realized_pnl_pct,
        realized_r_multiple=realized_r_multiple,
        analysis_score=int(record.get("analysis_score", 0) or 0),
        analysis_bias=str(record.get("analysis_bias", "")),
        analysis_convergence=float(record.get("analysis_convergence", 0.0) or 0.0),
        cohort_id=cohort_id,
        strategy_profile=strategy_profile,
        strategy_path=str(record.get("strategy_path", "")) or None,
        asset_class=str(record.get("asset_class") or symbol_route.asset_class),
        route_id=str(record.get("route_id") or symbol_route.route_id),
        review_lane=str(record.get("review_lane") or symbol_route.review_lane),
        entry_reason_snapshot=entry_reason_snapshot,
        signal_scores=record.get("signal_scores") if isinstance(record.get("signal_scores"), dict) else None,
        rule_compliant=rule_compliant,
        false_positive_tag=false_positive_tag,
        market_regime_tag=market_regime_tag,
        challenge_status=str(record.get("challenge_status", "inactive")),
        challenge_progress_pct=float(record.get("challenge_progress_pct", 0.0) or 0.0),
        note=str(record.get("note", "")),
        source_hash=f"{symbol}:{source_order_id}:{closed_ms}:{exit_reason}",
        binance_context={
            "entry_order": entry_order,
            "stop_loss_algo": stop_algo or stop_seed,
            "take_profit_algo": take_algo or take_seed,
            "income_rows": income_rows,
        },
    )


def maybe_load_strategy(path: str) -> Any:
    if not path:
        return None
    return load_strategy_config(path)


def cmd_env_template(_args: argparse.Namespace) -> None:
    print(ENV_PATH.with_name(".env.example").read_text(encoding="utf-8"))


def cmd_validate_config(args: argparse.Namespace) -> None:
    settings = load_settings()
    strategy = load_strategy_config(args.strategy_config)
    payload = {
        "status": "ok",
        "settings": {
            "use_testnet": settings.use_testnet,
            "live_trading_enabled": settings.live_trading_enabled,
            "default_symbol": settings.default_symbol,
            "default_market": settings.default_market,
            "recv_window_ms": settings.recv_window_ms,
            "max_leverage": settings.max_leverage,
            "max_notional_pct": settings.max_notional_pct,
        },
        "strategy": {
            "profile": strategy.profile,
            "path": str(strategy.path),
            "symbol": strategy.defaults.symbol,
            "market": strategy.defaults.market,
            "interval": strategy.defaults.interval,
            "max_account_risk_pct": strategy.risk.max_account_risk_pct,
            "min_adx": strategy.risk.min_adx,
            "trailing_stop_enabled": strategy.risk.trailing_stop_enabled,
        },
    }
    print_json(payload, compact=getattr(args, "compact", False))


def cmd_stability_workflow(args: argparse.Namespace) -> None:
    project_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    steps: list[dict[str, Any]] = []

    def add_step(name: str, command: list[str], *, timeout: int = 600) -> None:
        result = run(command, timeout=timeout)
        steps.append(
            {
                "name": name,
                "ok": result["returncode"] == 0,
                "command": command,
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "returncode": result["returncode"],
            }
        )

    add_step(
        "validate-config",
        [
            str(project_python),
            "-m",
            "binance_quant_control.cli",
            "validate-config",
            "--strategy-config",
            args.strategy_config,
        ],
    )
    if not args.skip_ruff:
        add_step(
            "ruff-check",
            [
                str(project_python),
                "-m",
                "ruff",
                "check",
                "src/binance_quant_control",
                "tests",
                "scripts/run_stability_workflow.py",
            ],
        )
    if not args.skip_pytest:
        add_step("pytest", [str(project_python), "-m", "pytest", "-q"], timeout=1200)
    if args.include_doctor:
        add_step("doctor", [str(project_python), "-m", "binance_quant_control.cli", "doctor"], timeout=300)
    add_step(
        "live-readiness",
        [
            str(project_python),
            "-m",
            "binance_quant_control.cli",
            "live-readiness",
            "--strategy-config",
            args.strategy_config,
        ],
        timeout=600,
    )

    print_json(
        {
            "status": "ok" if all(step["ok"] for step in steps) else "failed",
            "strategy_config": args.strategy_config,
            "steps": steps,
        }
    )


def cmd_doctor(args: argparse.Namespace) -> None:
    settings = load_settings()
    ensure_runtime_dirs()
    private_auth_failures: list[tuple[str, str]] = []
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "identity": "OpenClaw Binance quant control plane",
        "project_root": str(PROJECT_ROOT),
        "env_path": str(ENV_PATH),
        "reports_dir": str(REPORTS_DIR),
        "task_spec_dir": str(TASK_SPEC_DIR),
        "run_dir": str(RUN_DIR),
        "settings": {
            "use_testnet": settings.use_testnet,
            "live_trading_enabled": settings.live_trading_enabled,
            "default_symbol": settings.default_symbol,
            "default_market": settings.default_market,
            "has_binance_credentials": settings.has_binance_credentials,
            "has_blave_credentials": settings.has_blave_credentials,
        },
        "connectivity": {},
        "warnings": [],
        "notes": [],
    }
    with BinanceClient(settings) as client:
        for market in ("spot", "futures"):
            try:
                payload["connectivity"][market] = {
                    "ok": True,
                    "server_time": client.ping(market),
                    "base_url_mode": "testnet" if settings.use_testnet else "mainnet",
                }
            except Exception as exc:  # pragma: no cover - runtime health path
                payload["connectivity"][market] = {"ok": False, "error": str(exc)}
                payload["warnings"].append(f"{market} connectivity check failed")
        if settings.has_binance_credentials:
            for market in ("spot", "futures"):
                try:
                    private_payload = client.balance(market) if market == "futures" else client.account(market)
                    payload["connectivity"].setdefault(market, {})
                    payload["connectivity"][market]["private_read_ok"] = True
                    payload["connectivity"][market]["private_usdt_balance"] = _extract_usdt_balance(private_payload, market)
                except Exception as exc:  # pragma: no cover - runtime health path
                    payload["connectivity"].setdefault(market, {})
                    payload["connectivity"][market]["private_read_ok"] = False
                    payload["connectivity"][market]["private_error"] = str(exc)
                    private_auth_failures.append((market, str(exc)))
            default_market_private_ok = bool(
                payload["connectivity"].get(settings.default_market, {}).get("private_read_ok")
            )
            for market, _error in private_auth_failures:
                if market != settings.default_market and default_market_private_ok:
                    payload["notes"].append(
                        f"{market} private API auth check failed but default market is "
                        f"{settings.default_market} and its private auth succeeded"
                    )
                else:
                    payload["warnings"].append(f"{market} private API auth check failed")
    if args.use_blave:
        if settings.has_blave_credentials:
            try:
                from .blave_api import BlaveClient

                with BlaveClient(settings) as blave:
                    row = blave.latest_snapshot(settings.default_symbol)
                payload["connectivity"]["blave"] = {"ok": True, "has_snapshot": bool(row)}
            except Exception as exc:  # pragma: no cover
                payload["connectivity"]["blave"] = {"ok": False, "error": str(exc)}
                payload["warnings"].append("Blave connectivity check failed")
        else:
            payload["connectivity"]["blave"] = {"ok": False, "error": "credentials missing"}
            payload["warnings"].append("Blave credentials are not configured")
    if settings.live_trading_enabled:
        payload["warnings"].append("Live trading flag is enabled; keep this off until testnet and paper validation are complete.")
    if not settings.has_binance_credentials:
        payload["notes"].append("Binance private commands stay unavailable until API keys are configured. Public analysis and workflow features are ready now.")
    payload["overall"] = "ok" if not payload["warnings"] else "warn"
    compact = getattr(args, "compact", False)
    if compact:
        payload = {
            "overall": payload["overall"],
            "live_trading_enabled": settings.live_trading_enabled,
            "spot_ok": payload["connectivity"].get("spot", {}).get("ok"),
            "futures_ok": payload["connectivity"].get("futures", {}).get("ok"),
            "warnings": payload["warnings"],
        }
    print_json(payload, compact=compact)


def cmd_analyze(args: argparse.Namespace) -> None:
    settings = load_settings()
    strategy = maybe_load_strategy(args.strategy_config)
    payload, artifacts = run_analysis(
        settings,
        symbol=args.symbol.upper(),
        market=args.market,
        interval=args.interval,
        limit=args.limit,
        use_blave=args.use_blave,
        render_chart_flag=args.render_chart,
        strategy=strategy,
        output_dir=Path(args.output_dir).expanduser() if args.output_dir else None,
    )
    payload["artifacts"]["run_id"] = artifacts.run_id
    compact = getattr(args, "compact", False)
    if compact:
        analysis = payload.get("analysis", {})
        payload = {
            "symbol": payload.get("symbol"),
            "market": payload.get("market"),
            "analysis": {
                "bias": analysis.get("bias"),
                "score": analysis.get("score"),
                "convergence": analysis.get("convergence"),
                "verdict": analysis.get("verdict"),
            },
            "trade_plan": payload.get("trade_plan"),
            "latest": {k: payload.get("latest", {}).get(k) for k in (
                "close", "ema_fast", "ema_slow", "sma_200", "rsi_14", "macd", "macd_signal", "macd_hist",
                "adx", "plus_di", "minus_di", "bb_percent_b", "bb_bandwidth", "vwap", "obv_zscore_20", "volume_zscore_20",
                "atr_14", "breakout_high_20", "breakout_low_20",
            ) if payload.get("latest", {}).get(k) is not None},
            "run_id": artifacts.run_id,
        }
    print_json(payload, compact=compact)


def _extract_usdt_balance(account_payload: Any, market: str) -> float | None:
    if market == "spot":
        balances = account_payload.get("balances") or []
        for item in balances:
            if item.get("asset") == "USDT":
                return float(item.get("free", 0)) + float(item.get("locked", 0))
        return None
    for item in account_payload:
        if item.get("asset") == "USDT":
            return float(item.get("balance", 0))
    return None


def cmd_account(args: argparse.Namespace) -> None:
    settings = load_settings()
    with BinanceClient(settings) as client:
        payload = client.balance(args.market) if args.market == "futures" else client.account(args.market)
    compact = getattr(args, "compact", False)
    response: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "market": args.market,
        "summary": {
            "usdt_balance": _extract_usdt_balance(payload, args.market),
        },
    }
    if not compact:
        response["raw"] = payload
    print_json(response, compact=compact)


def cmd_positions(args: argparse.Namespace) -> None:
    settings = load_settings()
    with BinanceClient(settings) as client:
        raw = client.positions(args.symbol.upper() if args.symbol else None)
    positions = []
    for item in raw:
        amount = float(item.get("positionAmt", 0))
        if args.all or amount != 0:
            positions.append(item)
    compact = getattr(args, "compact", False)
    if compact:
        slim_positions = []
        for p in positions:
            slim_positions.append({
                "symbol": p.get("symbol"),
                "side": "LONG" if float(p.get("positionAmt", 0)) > 0 else "SHORT",
                "qty": float(p.get("positionAmt", 0)),
                "entry": float(p.get("entryPrice", 0)),
                "pnl": float(p.get("unRealizedProfit", 0)),
                "leverage": int(p.get("leverage", 1)),
            })
        print_json({"count": len(slim_positions), "positions": slim_positions}, compact=True)
    else:
        print_json(
            {
                "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "symbol": args.symbol.upper() if args.symbol else None,
                "count": len(positions),
                "positions": positions,
            }
        )


def cmd_trading_control_status(args: argparse.Namespace) -> None:
    state = load_trading_control_state()
    print_json(
        {
            "status": "ok",
            "trading_control": state.to_dict(),
        },
        compact=getattr(args, "compact", False),
    )


def cmd_pause_trading(args: argparse.Namespace) -> None:
    settings = load_settings()
    ensure_runtime_dirs()
    reason = args.reason.strip() if args.reason else "manual kill-switch"
    regular_cancelled = 0
    algo_cancelled = 0
    errors: list[str] = []

    if args.cancel_open_orders:
        with BinanceClient(settings) as client:
            try:
                open_regular = client.open_orders(None, market="futures")
                for order in open_regular:
                    symbol = str(order.get("symbol", "")).upper()
                    order_id = order.get("orderId")
                    if not symbol or order_id is None:
                        continue
                    try:
                        client.cancel_order(symbol, int(order_id), market="futures")
                        regular_cancelled += 1
                    except (BinanceAPIError, ValueError, TypeError) as exc:
                        errors.append(f"regular:{symbol}:{order_id}:{exc}")

                open_algo = client.open_algo_orders(None)
                for order in open_algo:
                    symbol = str(order.get("symbol", "")).upper()
                    algo_id = order.get("algoId")
                    if not symbol or algo_id is None:
                        continue
                    try:
                        client.cancel_algo_order(symbol, int(algo_id))
                        algo_cancelled += 1
                    except (BinanceAPIError, ValueError, TypeError) as exc:
                        errors.append(f"algo:{symbol}:{algo_id}:{exc}")
            except LiveTradingDisabledError as exc:
                errors.append(str(exc))
            except BinanceAPIError as exc:
                errors.append(str(exc))

    state = set_trading_paused(paused=True, reason=reason, updated_by="openclaw-quantctl pause-trading")
    payload = {
        "status": "ok" if not errors else "partial",
        "trading_control": state.to_dict(),
        "cancel_summary": {
            "regular_orders_cancelled": regular_cancelled,
            "algo_orders_cancelled": algo_cancelled,
            "errors": errors,
        },
    }
    print_json(payload, compact=getattr(args, "compact", False))


def cmd_resume_trading(args: argparse.Namespace) -> None:
    reason = args.reason.strip() if args.reason else "manual resume"
    state = set_trading_paused(paused=False, reason=reason, updated_by="openclaw-quantctl resume-trading")
    print_json(
        {
            "status": "ok",
            "trading_control": state.to_dict(),
            "note": "Kill-switch released. New orders are allowed by policy gates.",
        },
        compact=getattr(args, "compact", False),
    )


def cmd_auto_pause_trading(args: argparse.Namespace) -> None:
    auto_pause_actor = "openclaw-quantctl auto-pause-trading"
    settings = load_settings()
    strategy = load_strategy_config(args.strategy_config)
    loss_cooldown_hours = getattr(args, "loss_cooldown_hours", None)
    if loss_cooldown_hours is None:
        loss_cooldown_hours = float(getattr(getattr(strategy, "risk", None), "cooldown_hours", 0.0) or 0.0)
    policy = AutoPausePolicy(
        consecutive_loss_threshold=args.consecutive_loss_threshold,
        loss_cooldown_hours=float(loss_cooldown_hours),
        position_timeout_hours=args.position_timeout_hours,
        reversal_min_score=args.reversal_min_score,
        reversal_min_convergence=args.reversal_min_convergence,
    )
    evaluation = evaluate_auto_pause_conditions(settings, strategy, policy=policy)
    current_state = load_trading_control_state()
    actions: list[str] = []
    cancel_errors: list[str] = []
    regular_cancelled = 0
    algo_cancelled = 0

    if evaluation.should_pause and not current_state.paused:
        if args.cancel_open_orders:
            with BinanceClient(settings) as client:
                try:
                    open_regular = client.open_orders(None, market="futures")
                    for order in open_regular:
                        symbol = str(order.get("symbol", "")).upper()
                        order_id = order.get("orderId")
                        if not symbol or order_id is None:
                            continue
                        try:
                            client.cancel_order(symbol, int(order_id), market="futures")
                            regular_cancelled += 1
                        except (BinanceAPIError, ValueError, TypeError) as exc:
                            cancel_errors.append(f"regular:{symbol}:{order_id}:{exc}")

                    open_algo = client.open_algo_orders(None)
                    for order in open_algo:
                        symbol = str(order.get("symbol", "")).upper()
                        algo_id = order.get("algoId")
                        if not symbol or algo_id is None:
                            continue
                        try:
                            client.cancel_algo_order(symbol, int(algo_id))
                            algo_cancelled += 1
                        except (BinanceAPIError, ValueError, TypeError) as exc:
                            cancel_errors.append(f"algo:{symbol}:{algo_id}:{exc}")
                except (LiveTradingDisabledError, BinanceAPIError) as exc:
                    cancel_errors.append(str(exc))

        state = set_trading_paused(
            paused=True,
            reason="; ".join(evaluation.reasons),
            updated_by=auto_pause_actor,
        )
        actions.append("paused")
    else:
        state = current_state
        if (
            not evaluation.should_pause
            and current_state.paused
            and current_state.updated_by == auto_pause_actor
        ):
            state = set_trading_paused(
                paused=False,
                reason="Auto-resumed after pause conditions cleared.",
                updated_by=auto_pause_actor,
            )
            actions.append("resumed")
        elif evaluation.should_pause and current_state.paused:
            actions.append("already-paused")
        else:
            actions.append("no-action")

    payload = {
        "status": "ok" if evaluation.should_pause else "idle",
        "actions": actions,
        "trading_control": state.to_dict(),
        "policy": policy.to_dict(),
        "evaluation": evaluation.to_dict(),
        "cancel_summary": {
            "regular_orders_cancelled": regular_cancelled,
            "algo_orders_cancelled": algo_cancelled,
            "errors": cancel_errors,
        },
    }
    print_json(payload, compact=getattr(args, "compact", False))


def cmd_paper_order(args: argparse.Namespace) -> None:
    settings = load_settings()
    ensure_runtime_dirs()
    market = args.market
    symbol = args.symbol.upper()
    route = resolve_symbol_route(symbol)
    strategy = maybe_load_strategy(args.strategy_config) or load_strategy_config(route.strategy_config)
    with BinanceClient(settings) as client:
        price = client.ticker_price(symbol, market)
    analysis, artifacts = run_analysis(
        settings,
        symbol=symbol,
        market=market,
        interval=args.interval,
        limit=max(args.limit, 240),
        use_blave=args.use_blave,
        render_chart_flag=args.render_chart,
        strategy=strategy,
    )
    leverage = float(args.leverage)
    if strategy and not args.leverage:
        leverage = float(strategy.risk.default_leverage)
    record = _build_paper_order_record(
        route=route,
        symbol=symbol,
        market=market,
        interval=args.interval,
        side=args.side,
        notional_usdt=float(args.notional_usdt),
        leverage=leverage,
        price=price,
        analysis=analysis,
        strategy_profile=analysis.get("strategy_profile"),
        strategy_path=str(strategy.path) if strategy else None,
        note="Journal only. No live or testnet order has been sent.",
    )
    journal = append_paper_order(record)
    entry = {
        **asdict(record),
        "route": route.to_dict(),
        "journal_path": str(journal),
    }
    entry["run_output_dir"] = str(artifacts.output_dir)
    print_json(entry)


def cmd_route_symbol(args: argparse.Namespace) -> None:
    route = resolve_symbol_route(args.symbol)
    strategy = load_strategy_config(route.strategy_config)
    print_json(
        {
            "status": "ok",
            "symbol": args.symbol.upper(),
            "route": route.to_dict(),
            "strategy_profile": strategy.profile,
            "strategy_defaults": {
                "symbol": strategy.defaults.symbol,
                "market": strategy.defaults.market,
                "interval": strategy.defaults.interval,
            },
            "validation_summary": route.validation.to_dict(),
        }
    )


def cmd_route_intent(args: argparse.Namespace) -> None:
    intent = resolve_operator_intent(args.message)
    print_json({"status": "ok", "message": args.message, "intent": intent.to_dict()})


def cmd_new_symbol_workflow(args: argparse.Namespace) -> None:
    payload = run_new_symbol_workflow(
        symbols=split_csv_arg(args.symbols),
        intervals=split_csv_arg(args.intervals),
        sides=split_csv_arg(args.sides),
        research_depth=args.research_depth,
        plan_only=bool(args.plan_only),
        output_dir=args.output_dir or None,
        strategy_config=args.strategy_config,
        blueprint_config=args.blueprint_config,
        max_readiness_candidates=args.max_readiness_candidates,
    )
    if getattr(args, "compact", False):
        print_json(
            {
                "mode": payload.get("mode"),
                "objective": payload.get("objective"),
                "safety": payload.get("safety"),
                "inputs": payload.get("inputs"),
                "outcome": payload.get("outcome"),
                "status_counts": payload.get("status_counts"),
                "symbols": [
                    {
                        "symbol": item.get("symbol"),
                        "outcome": item.get("outcome"),
                        "route_id": ((item.get("route") or {}).get("route") or {}).get("route_id")
                        if isinstance(item.get("route"), dict)
                        else None,
                        "candidate_keys": item.get("candidate_keys"),
                        "near_ready_candidate_keys": item.get("near_ready_candidate_keys"),
                        "ready_candidate_keys": item.get("ready_candidate_keys"),
                        "next_command": item.get("next_command"),
                    }
                    for item in payload.get("symbols") or []
                    if isinstance(item, dict)
                ],
                "sentinel": payload.get("sentinel"),
                "research_sweep_count": len(payload.get("research_sweeps") or []),
                "risk_combo_matrix": payload.get("risk_combo_matrix"),
                "readiness": payload.get("readiness"),
                "promotion_boundary": payload.get("promotion_boundary"),
                "report_path": payload.get("report_path"),
            },
            compact=True,
        )
        return
    print_json(payload)


def cmd_mission(args: argparse.Namespace) -> None:
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()]
    payload = run_trading_mission(
        symbols=symbols,
        target_return_pct=float(args.target_return_pct),
        max_leverage=float(args.max_leverage),
        execute_live=bool(args.execute_live),
        config_path=args.config,
    )
    print_json(payload, compact=getattr(args, "compact", False))


def build_analysis_spec(args: argparse.Namespace) -> dict[str, Any]:
    ensure_runtime_dirs()
    strategy = maybe_load_strategy(args.strategy_config)
    symbol = args.symbol.upper()
    market = args.market
    name = args.name or f"binance-{symbol.lower()}-{market}-{args.interval}"
    stamp = now_stamp()
    run_root = RUN_DIR / f"{stamp}-{symbol.lower()}-{market}-{args.interval}"
    run_root.mkdir(parents=True, exist_ok=True)
    steps = [
        {
            "id": "preflight-health",
            "run": [str(HOSTCTL), "health"],
            "timeout": 180,
            "retries": 1,
        },
        {
            "id": "quant-doctor",
            "run": ["/home/robert/.openclaw/bin/openclaw-quantctl", "doctor"],
            "timeout": 180,
            "retries": 0,
        },
        {
            "id": "market-analysis",
            "run": [
                "/home/robert/.openclaw/bin/openclaw-quantctl",
                "analyze",
                symbol,
                "--market",
                market,
                "--interval",
                args.interval,
                "--limit",
                str(args.limit),
                "--output-dir",
                str(run_root / "analysis"),
            ]
            + (["--strategy-config", str(strategy.path)] if strategy else [])
            + (["--use-blave"] if args.use_blave else [])
            + (["--render-chart"] if args.render_chart else []),
            "timeout": 600,
            "retries": 0,
        },
    ]
    if args.include_account_check:
        steps.append(
            {
                "id": "private-account-check",
                "run": ["/home/robert/.openclaw/bin/openclaw-quantctl", "account", "--market", market],
                "timeout": 180,
                "retries": 0,
            }
        )
    spec = {
        "name": name,
        "project": "binance-quant-control",
        "description": "Run a guarded Binance market analysis workflow with optional Blave enrichment.",
        "policy": {
            "allow_shell_strings": False,
            "allow_uncontrolled_execution": False,
            "allow_untrusted_cwd": False,
        },
        "memory": {
            "procedureOnSuccess": {
                "title": "Binance quant analysis workflow",
                "body": "OpenClaw workflow runs host preflight, quant doctor, and Binance market analysis through openclaw-quantctl.",
                "confidence": 0.9,
            },
            "contextOnSuccess": {
                "title": "Latest Binance quant workflow",
                "body": f"Latest workflow analyzed {symbol} on {market} {args.interval} and wrote artifacts under {run_root}.",
                "confidence": 0.85,
            },
        },
        "quant": {
            "symbol": symbol,
            "market": market,
            "interval": args.interval,
            "limit": args.limit,
            "use_blave": args.use_blave,
            "render_chart": args.render_chart,
            "run_root": str(run_root),
            "strategy_profile": strategy.profile if strategy else None,
            "strategy_path": str(strategy.path) if strategy else None,
        },
        "steps": steps,
    }
    spec_path = TASK_SPEC_DIR / f"{stamp}-{symbol.lower()}-{market}-{args.interval}.json"
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"spec": spec, "spec_path": spec_path, "run_root": run_root}


def cmd_build_analysis_spec(args: argparse.Namespace) -> None:
    payload = build_analysis_spec(args)
    print_json(
        {
            "status": "ok",
            "spec_path": str(payload["spec_path"]),
            "run_root": str(payload["run_root"]),
        }
    )


def cmd_submit_analysis(args: argparse.Namespace) -> None:
    payload = build_analysis_spec(args)
    lint = parse_json_result(run([str(TASKCTL), "lint", str(payload["spec_path"])]), "task lint")
    if not lint.get("ok"):
        raise SystemExit(json.dumps(lint, indent=2, ensure_ascii=False))
    submit = parse_json_result(run([str(TASKCTL), "submit", str(payload["spec_path"])]), "task submit")
    response = {
        "status": "queued",
        "spec_path": str(payload["spec_path"]),
        "run_root": str(payload["run_root"]),
        "lint": lint,
        "task": submit,
    }
    if args.run_now:
        execution = parse_json_result(
            run([str(TASKCTL), "resume", str(submit["task_id"])], timeout=1200),
            "task resume",
        )
        response["status"] = execution.get("status", "completed")
        response["execution"] = execution
    print_json(response)


def cmd_backtest(args: argparse.Namespace) -> None:
    settings = load_settings()
    strategy = load_strategy_config(args.strategy_config)
    payload = run_backtest(
        settings,
        strategy=strategy,
        symbol=args.symbol.upper() if args.symbol else strategy.defaults.symbol,
        market=args.market if args.market else strategy.defaults.market,
        interval=args.interval if args.interval else strategy.defaults.interval,
        limit=args.limit if args.limit else strategy.defaults.limit,
        output_dir=Path(args.output_dir).expanduser() if args.output_dir else None,
    )
    print_json(payload)


def cmd_backtest_sweep(_args: argparse.Namespace) -> None:
    script = PROJECT_ROOT / "scripts" / "backtest_sweep.py"
    python = PROJECT_ROOT / ".venv" / "bin" / "python"
    result = parse_json_result(run([str(python), str(script)], timeout=3600), "backtest sweep")
    print_json(result)


def cmd_alpha_research(args: argparse.Namespace) -> None:
    settings = load_settings()
    symbols = [item.strip().upper() for item in str(args.symbols).split(",") if item.strip()]
    intervals = [item.strip() for item in str(args.intervals).split(",") if item.strip()]
    strategy_families = getattr(args, "strategy_families", "")
    families = [item.strip() for item in str(strategy_families).split(",") if item.strip()]
    payload = run_aggressive_alpha_research(
        settings,
        config_path=args.config,
        output_dir=Path(args.output_dir).expanduser() if args.output_dir else None,
        symbol_overrides=symbols or None,
        interval_overrides=intervals or None,
        limit_override=args.limit if args.limit > 0 else None,
        family_overrides=families or None,
    )
    if getattr(args, "compact", False):
        print_json(
            {
                "status": "ok",
                "mode": payload.get("mode"),
                "mainnet_live_allowed": payload.get("mainnet_live_allowed"),
                "strategy_profile": payload.get("strategy_profile"),
                "symbols": payload.get("symbols"),
                "intervals": payload.get("intervals"),
                "ranking_method": payload.get("ranking_method"),
                "resolved_symbol_family_plan": payload.get("resolved_symbol_family_plan"),
                "progress_path": payload.get("progress_path"),
                "top": payload.get("top"),
                "performance_summary": payload.get("performance_summary"),
                "errors": payload.get("errors"),
                "skipped_symbol_intervals": payload.get("skipped_symbol_intervals"),
                "report_path": payload.get("report_path"),
            },
            compact=True,
        )
        return
    print_json(payload)


def cmd_external_context(args: argparse.Namespace) -> None:
    symbols = [item.strip().upper() for item in str(args.symbols).split(",") if item.strip()]
    payload = build_external_context(
        symbols,
        config_path=args.config,
        write_cache=not args.no_cache,
    )
    if getattr(args, "compact", False):
        payload = {
            "status": "ok",
            "combined_signal": payload.get("combined_signal"),
            "symbols": payload.get("symbols"),
            "available_sources": payload.get("available_sources"),
            "sources": {
                name: {
                    "enabled": item.get("enabled"),
                    "available": item.get("available"),
                    "signal": item.get("signal"),
                    "reason": item.get("reason"),
                }
                for name, item in (payload.get("sources") or {}).items()
            },
            "cache_path": payload.get("cache_path"),
        }
    print_json(payload, compact=getattr(args, "compact", False))


def cmd_external_context_key_status(args: argparse.Namespace) -> None:
    payload = external_context_key_status(config_path=args.config)
    if getattr(args, "compact", False):
        payload = {
            "status": "ok",
            "configured_count": payload.get("configured_count"),
            "missing": payload.get("missing"),
            "providers": {
                name: {
                    "enabled": item.get("enabled"),
                    "env_var": item.get("env_var"),
                    "configured": item.get("configured"),
                    "key_length": item.get("key_length"),
                    "key_suffix": item.get("key_suffix"),
                    "register_url": item.get("register_url"),
                }
                for name, item in (payload.get("providers") or {}).items()
            },
            "secret_policy": payload.get("secret_policy"),
        }
    print_json(payload, compact=getattr(args, "compact", False))


def cmd_challenge_init(args: argparse.Namespace) -> None:
    settings = load_settings()
    strategy = load_strategy_config(args.strategy_config)
    challenge_scope = challenge_scope_key(strategy.profile, strategy.defaults.symbol, strategy.defaults.market)
    start_balance = float(args.start_balance_usdt)
    snapshot_payload = None
    if args.from_account or start_balance <= 0:
        with BinanceClient(settings) as client:
            snapshot_payload = client.balance(strategy.defaults.market)
        snapshot, _ = record_balance_snapshot(
            snapshot_payload,
            strategy.defaults.market,
            note="challenge-init",
            scope=challenge_scope,
        )
        start_balance = snapshot.equity_usdt
    state = initialize_challenge(
        profile=strategy.profile,
        symbol=strategy.defaults.symbol,
        market=strategy.defaults.market,
        start_balance_usdt=start_balance,
        target_multiple=args.target_multiple if args.target_multiple > 0 else strategy.challenge.target_multiple,
        max_drawdown_pct=args.max_drawdown_pct if args.max_drawdown_pct > 0 else strategy.challenge.max_drawdown_pct,
        note=args.note,
        scope=challenge_scope,
    )
    print_json(
        {
            "status": "ok",
            "strategy_profile": strategy.profile,
            "challenge": challenge_summary_dict(state),
            "snapshot_from_account": bool(snapshot_payload is not None),
        }
    )


def cmd_challenge_status(args: argparse.Namespace) -> None:
    settings = load_settings()
    strategy = load_strategy_config(args.strategy_config)
    challenge_scope = challenge_scope_key(strategy.profile, strategy.defaults.symbol, strategy.defaults.market)
    state = load_challenge_state(challenge_scope)
    current_snapshot = None
    if args.refresh:
        with BinanceClient(settings) as client:
            payload = client.balance(strategy.defaults.market)
        snapshot, state = record_balance_snapshot(
            payload,
            strategy.defaults.market,
            note="challenge-status",
            scope=challenge_scope,
        )
        current_snapshot = {
            "wallet_balance_usdt": round(snapshot.wallet_balance_usdt, 8),
            "available_balance_usdt": round(snapshot.available_balance_usdt, 8),
            "unrealized_pnl_usdt": round(snapshot.unrealized_pnl_usdt, 8),
            "equity_usdt": round(snapshot.equity_usdt, 8),
            "timestamp": snapshot.timestamp,
        }
    compact = getattr(args, "compact", False)
    trade_state = load_trade_state()
    if compact:
        challenge = challenge_summary_dict(state)
        print_json(
            {
                "challenge": {
                    "status": challenge.get("status"),
                    "progress_pct": challenge.get("progress_pct"),
                    "latest_balance_usdt": challenge.get("latest_balance_usdt"),
                },
                "current_snapshot": current_snapshot,
                "trades": trade_state.total_live_trades,
                "consecutive_losses": trade_state.consecutive_losses,
                "pnl_usdt": trade_state.total_pnl_usdt,
            },
            compact=True,
        )
    else:
        journal = summarize_live_orders()
        print_json(
            {
                "status": "ok",
                "challenge": challenge_summary_dict(state),
                "current_snapshot": current_snapshot,
                "trade_state": {
                    "daily_trade_count": trade_state.daily_trade_count,
                    "daily_trade_date": trade_state.daily_trade_date,
                    "consecutive_losses": trade_state.consecutive_losses,
                    "total_live_trades": trade_state.total_live_trades,
                    "total_pnl_usdt": trade_state.total_pnl_usdt,
                },
                "journal": journal,
                "recent_balance_snapshots": read_balance_snapshots(limit=args.snapshot_limit),
            }
        )


def cmd_journal_summary(args: argparse.Namespace) -> None:
    trade_state = load_trade_state()
    strategy = load_strategy_config(str(CONFIG_DIR / "strategy-live-pilot.yaml"))
    challenge_scope = challenge_scope_key(strategy.profile, strategy.defaults.symbol, strategy.defaults.market)
    payload = {
        "status": "ok",
        "trade_state": {
            "daily_trade_count": trade_state.daily_trade_count,
            "daily_trade_date": trade_state.daily_trade_date,
            "consecutive_losses": trade_state.consecutive_losses,
            "total_live_trades": trade_state.total_live_trades,
            "total_pnl_usdt": trade_state.total_pnl_usdt,
        },
        "challenge": challenge_summary_dict(load_challenge_state(challenge_scope)),
        "journal": summarize_live_orders(),
        "paper_orders": summarize_paper_orders(),
        "closed_trade_reviews": summarize_closed_trade_reviews(),
        "recent_balance_snapshots": read_balance_snapshots(limit=5),
    }
    if getattr(args, "compact", False):
        payload = {
            "status": "ok",
            "daily_trade_count": trade_state.daily_trade_count,
            "consecutive_losses": trade_state.consecutive_losses,
            "live_order_count": payload["journal"]["count"],
            "paper_order_count": payload["paper_orders"]["count"],
            "closed_review_count": payload["closed_trade_reviews"]["count"],
            "total_realized_pnl_usdt": payload["closed_trade_reviews"]["total_realized_pnl_usdt"],
        }
    print_json(payload, compact=getattr(args, "compact", False))


def cmd_operator_dashboard(args: argparse.Namespace) -> None:
    settings = load_settings()
    payload = build_operator_dashboard(
        settings,
        min_bucket_trades=args.min_bucket_trades,
        top_n=args.top_n,
    )
    if args.compact:
        execution_journal = payload.get("execution_journal") or {}
        latest_entry = execution_journal.get("latest") if isinstance(execution_journal.get("latest"), dict) else {}
        compact_execution_journal = {
            "record_count": execution_journal.get("record_count", 0),
            "buy_count": execution_journal.get("buy_count", 0),
            "sell_count": execution_journal.get("sell_count", 0),
            "latest": {
                "timestamp": latest_entry.get("timestamp"),
                "symbol": latest_entry.get("symbol"),
                "side": latest_entry.get("side"),
                "status": latest_entry.get("status"),
                "route_id": latest_entry.get("route_id"),
                "simulation_mode": latest_entry.get("simulation_mode"),
            }
            if latest_entry
            else None,
            "meaning": execution_journal.get("meaning"),
        }
        payload = {
            "status": payload["status"],
            "mode": payload["mode"],
            "customer_summary": payload["customer_summary"],
            "positions": payload["positions"],
            "protective_orders": payload["protective_orders"],
            "execution_journal": compact_execution_journal,
            "product_readiness": payload.get("product_readiness"),
            "candidate_pool": payload.get("candidate_pool"),
            "decision_artifact_audit": payload.get("decision_artifact_audit"),
            "risk_combo_matrix": payload.get("risk_combo_matrix"),
            "loss_diagnostics": {
                "summary": payload.get("loss_diagnostics", {}).get("summary"),
                "findings": payload.get("loss_diagnostics", {}).get("findings"),
                "root_cause_recommendations": payload.get("loss_diagnostics", {}).get(
                    "root_cause_recommendations"
                ),
            },
            "external_context_automation": payload.get("external_context_automation"),
            "operator_feedback": payload["operator_feedback"],
            "report_path": payload["report_path"],
        }
    print_json(payload, compact=args.compact)


def cmd_review_closed_trades(args: argparse.Namespace) -> None:
    settings = load_settings()
    reviewed = read_closed_trade_reviews()
    reviewed_ids = {str(item.get("source_order_id")) for item in reviewed if item.get("source_order_id") is not None}
    live_orders = read_live_orders()
    new_reviews: list[dict[str, Any]] = []
    trade_state = load_trade_state()

    with BinanceClient(settings) as client:
        positions = client.positions()
        positions_by_symbol: dict[str, list[dict[str, Any]]] = {}
        for item in positions:
            positions_by_symbol.setdefault(str(item.get("symbol", "")).upper(), []).append(item)

        for record in live_orders[-int(args.limit):]:
            source_order_id = record.get("order_id")
            if source_order_id is None or str(source_order_id) in reviewed_ids:
                continue
            review = _build_closed_trade_review(client, record, positions_by_symbol)
            if review is None:
                continue
            append_closed_trade_review(review)
            reviewed_ids.add(str(source_order_id))
            if review.realized_pnl_usdt < 0:
                trade_state.record_loss(review.realized_pnl_usdt)
            elif review.realized_pnl_usdt > 0:
                trade_state.record_win(review.realized_pnl_usdt)
            new_reviews.append(
                {
                    "source_order_id": review.source_order_id,
                    "symbol": review.symbol,
                    "exit_reason": review.exit_reason,
                    "cohort_id": review.cohort_id,
                    "rule_compliant": review.rule_compliant,
                    "false_positive_tag": review.false_positive_tag,
                    "market_regime_tag": review.market_regime_tag,
                    "realized_pnl_usdt": review.realized_pnl_usdt,
                    "realized_pnl_pct": review.realized_pnl_pct,
                    "closed_at": review.closed_at,
                }
            )

    if new_reviews:
        save_trade_state(trade_state)

    payload = {
        "status": "ok",
        "new_review_count": len(new_reviews),
        "new_reviews": new_reviews,
        "closed_trade_reviews": summarize_closed_trade_reviews(),
        "trade_state": {
            "daily_trade_count": trade_state.daily_trade_count,
            "daily_trade_date": trade_state.daily_trade_date,
            "consecutive_losses": trade_state.consecutive_losses,
            "last_loss_at": trade_state.last_loss_at,
            "total_live_trades": trade_state.total_live_trades,
            "total_pnl_usdt": trade_state.total_pnl_usdt,
        },
    }
    if getattr(args, "compact", False):
        payload = {
            "status": payload["status"],
            "new_review_count": payload["new_review_count"],
            "closed_review_count": payload["closed_trade_reviews"]["count"],
            "total_realized_pnl_usdt": payload["closed_trade_reviews"]["total_realized_pnl_usdt"],
            "consecutive_losses": trade_state.consecutive_losses,
        }
    print_json(payload, compact=getattr(args, "compact", False))


def cmd_manage_position(args: argparse.Namespace) -> None:
    settings = load_settings()
    strategy = load_strategy_config(args.strategy_config) if args.strategy_config else None
    symbol = args.symbol.upper() if args.symbol else (strategy.defaults.symbol if strategy else "")
    market = args.market if args.market else (strategy.defaults.market if strategy else "futures")
    plan = build_position_management_plan(
        settings,
        symbol=symbol,
        market=market,
        stop_price=(args.stop_price if args.stop_price > 0 else None),
        take_profit_price=(args.take_profit_price if args.take_profit_price > 0 else None),
        strategy=strategy,
    )
    response = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "strategy_profile": strategy.profile if strategy else None,
        "strategy_path": str(strategy.path) if strategy else None,
        "plan": plan.to_dict(),
    }
    if args.execute:
        response["execution"] = execute_position_management_plan(settings, plan)
    print_json(response)


def cmd_trailing_update(args: argparse.Namespace) -> None:
    settings = load_settings()
    strategy = load_strategy_config(args.strategy_config)
    symbol = args.symbol.upper() if args.symbol else strategy.defaults.symbol
    market = args.market if args.market else strategy.defaults.market
    plan = build_position_management_plan(
        settings,
        symbol=symbol,
        market=market,
        take_profit_price=(args.take_profit_price if args.take_profit_price > 0 else None),
        strategy=strategy,
        enable_trailing_stop=True,
        trailing_callback_pct=(args.callback_pct if args.callback_pct > 0 else None),
        trailing_activation_price=(args.activation_price if args.activation_price > 0 else None),
    )
    response = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "strategy_profile": strategy.profile,
        "strategy_path": str(strategy.path),
        "plan": plan.to_dict(),
    }
    if args.execute:
        response["execution"] = execute_position_management_plan(settings, plan)
    print_json(response)


def cmd_repair_staged_tp(args: argparse.Namespace) -> None:
    settings = load_settings()
    strategy = load_strategy_config(args.strategy_config)
    symbol = args.symbol.upper() if args.symbol else strategy.defaults.symbol
    side = args.side.upper() if args.side else "BUY"
    analysis_payload, _ = run_analysis(
        settings,
        symbol=symbol,
        market=strategy.defaults.market,
        interval=strategy.defaults.interval,
        limit=max(strategy.defaults.limit, 240),
        use_blave=strategy.defaults.use_blave,
        render_chart_flag=False,
        strategy=strategy,
    )
    side_plan = analysis_payload["trade_plan"]["long" if side == "BUY" else "short"]
    confidence = float((analysis_payload.get("analysis") or {}).get("convergence") or 0.75)
    route = resolve_symbol_route(symbol)
    plan = build_staged_take_profit_repair_plan(
        settings,
        strategy,
        symbol=symbol,
        side_plan=side_plan,
        confidence=confidence,
        route_id=route.route_id,
        news_risk={"risk_level": args.news_risk},
    )
    response = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "strategy_profile": strategy.profile,
        "strategy_path": str(strategy.path),
        "plan": plan.to_dict(),
    }
    if args.execute:
        response["execution"] = execute_staged_take_profit_repair(settings, strategy, plan)
    print_json(response, compact=args.compact)


def _build_live_response(args: argparse.Namespace) -> dict[str, Any]:
    settings = load_settings()
    strategy = load_strategy_config(args.strategy_config)
    symbol = args.symbol.upper() if args.symbol else strategy.defaults.symbol
    market = args.market if args.market else strategy.defaults.market
    interval = args.interval if args.interval else strategy.defaults.interval
    limit = args.limit if args.limit else strategy.defaults.limit
    use_blave = args.use_blave or strategy.defaults.use_blave
    render_chart = args.render_chart or strategy.defaults.render_chart
    analysis, artifacts = run_analysis(
        settings,
        symbol=symbol,
        market=market,
        interval=interval,
        limit=max(limit, 240),
        use_blave=use_blave,
        render_chart_flag=render_chart,
        strategy=strategy,
    )
    plan = build_live_execution_plan(
        settings,
        strategy,
        analysis,
        side_override=args.side,
        margin_notional_usdt=(args.margin_notional_usdt if args.margin_notional_usdt > 0 else None),
        execution_mode=getattr(args, "execution_mode", "live"),
    )
    response = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "strategy_profile": strategy.profile,
        "strategy_path": str(strategy.path),
        "analysis": analysis["analysis"],
        "analysis_payload": analysis,
        "artifacts": analysis["artifacts"],
        "live_plan": plan.to_dict(),
    }
    if args.execute:
        route = resolve_symbol_route(symbol.upper())
        response["execution"] = execute_live_order(
            settings,
            strategy,
            plan,
            entry_reason_snapshot={
                "bias": str((analysis.get("analysis") or {}).get("bias") or ""),
                "score": int((analysis.get("analysis") or {}).get("score") or 0),
                "convergence": float((analysis.get("analysis") or {}).get("convergence") or 0.0),
                "interval": interval,
            },
            signal_scores=build_signal_scores(
                route=route,
                latest=analysis.get("latest") or {},
                analysis=analysis.get("analysis") or {},
                trade_plan=analysis.get("trade_plan") or {},
                side=plan.side,
            ),
        )
    return response


def _write_trade_decision_report(payload: dict[str, Any]) -> str:
    ensure_runtime_dirs()
    report_dir = STATE_DIR / "trade-decisions"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    symbol = str(payload.get("symbol") or "unknown").lower()
    decision = str(payload.get("decision") or "unknown").lower().replace("/", "-")
    path = report_dir / f"{stamp}-{secrets.token_hex(3)}-{symbol}-{decision}-trade-decision.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return str(path)


def cmd_trade_decision(args: argparse.Namespace) -> None:
    if str(getattr(args, "action", "")).upper() == "EXIT":
        settings = load_settings()
        strategy = load_strategy_config(args.strategy_config)
        symbol = args.symbol.upper() if args.symbol else strategy.defaults.symbol
        market = args.market if args.market else strategy.defaults.market
        interval = args.interval if args.interval else strategy.defaults.interval
        limit = args.limit if args.limit else strategy.defaults.limit
        use_blave = args.use_blave or strategy.defaults.use_blave
        analysis, artifacts = run_analysis(
            settings,
            symbol=symbol,
            market=market,
            interval=interval,
            limit=max(limit, 240),
            use_blave=use_blave,
            render_chart_flag=args.render_chart or strategy.defaults.render_chart,
            strategy=strategy,
        )
        position_plan = build_position_management_plan(
            settings,
            symbol=symbol,
            market=market,
            strategy=strategy,
        )
        exit_plan = build_adaptive_exit_plan(position_plan, analysis)
        payload = build_ai_exit_decision_output(
            position_plan=position_plan,
            exit_plan=exit_plan,
            analysis_payload=analysis,
        )
        payload["symbol"] = symbol
        payload["market"] = market
        payload["execution_mode"] = getattr(args, "execution_mode", "testnet_exploration")
        payload["opens_orders"] = False
        payload["writes_execution_config"] = False
        payload["report_artifacts"] = analysis.get("artifacts") or {"run_id": artifacts.run_id}
        payload["position_plan"] = position_plan.to_dict()
        payload["adaptive_exit_plan"] = exit_plan.to_dict()
        payload["decision_report_path"] = _write_trade_decision_report(payload)
        print_json(payload, compact=getattr(args, "compact", False))
        return

    try:
        response = _build_live_response(args)
    except BinanceAPIError as exc:
        payload = build_blocked_trade_decision_output(
            reason="Private exchange checks are unavailable.",
            blockers=[f"binance-private-api-auth-failed:{exc}"],
        )
        print_json(payload, compact=getattr(args, "compact", False))
        raise SystemExit(2) from exc

    strategy = load_strategy_config(args.strategy_config)
    payload = build_ai_trade_decision_output(
        analysis_payload=response.get("analysis_payload") or {},
        strategy=strategy,
        live_plan=response.get("live_plan") or {},
        requested_action=args.action or None,
    )
    live_plan = response.get("live_plan") or {}
    payload["symbol"] = live_plan.get("symbol")
    payload["market"] = live_plan.get("market")
    payload["execution_mode"] = live_plan.get("execution_mode")
    payload["opens_orders"] = False
    payload["writes_execution_config"] = False
    payload["report_artifacts"] = response.get("artifacts")
    payload["decision_report_path"] = _write_trade_decision_report(payload)
    print_json(payload, compact=getattr(args, "compact", False))


def _run_live_response(args: argparse.Namespace) -> None:
    try:
        payload = _build_live_response(args)
    except BinanceAPIError as exc:
        print_json(
            {
                "status": "blocked",
                "reason": "binance-private-api-auth-failed",
                "error": str(exc),
            }
        )
        raise SystemExit(2) from exc
    if getattr(args, "compact", False):
        plan = payload.get("live_plan") or {}
        challenge = plan.get("challenge") or {}
        print_json(
            {
                "generated_at": payload.get("generated_at"),
                "strategy_profile": payload.get("strategy_profile"),
                "symbol": plan.get("symbol"),
                "market": plan.get("market"),
                "side": plan.get("side"),
                "allowed": plan.get("allowed"),
                "analysis_score": plan.get("analysis_score"),
                "analysis_convergence": plan.get("analysis_convergence"),
                "adx_value": plan.get("adx_value"),
                "quantity": plan.get("quantity"),
                "margin_notional_usdt": plan.get("margin_notional_usdt"),
                "planned_account_risk_pct": plan.get("planned_account_risk_pct"),
                "execution_mode": plan.get("execution_mode"),
                "sizing": plan.get("sizing"),
                "violations": plan.get("violations") or [],
                "warnings": plan.get("warnings") or [],
                "optimizer_live_gate": challenge.get("optimizer_live_gate"),
                "route_quarantine": challenge.get("route_quarantine"),
                "route_side_risk": challenge.get("route_side_risk"),
                "historical_signal_risk": challenge.get("historical_signal_risk"),
                "professional_entry_gate": plan.get("professional_entry_gate"),
            },
            compact=True,
        )
        return
    print_json(payload)


def cmd_live_readiness(args: argparse.Namespace) -> None:
    _run_live_response(args)


def cmd_live_pilot(args: argparse.Namespace) -> None:
    _run_live_response(args)


def cmd_demo_train(args: argparse.Namespace) -> None:
    payload = run_demo_training(
        rounds=args.rounds,
        symbols=[item.strip() for item in args.symbols.split(",") if item.strip()] if args.symbols else None,
        target_return_pct=args.target_return_pct,
        max_leverage=args.max_leverage,
        margin_notional_usdt=args.margin_notional_usdt,
        optimize_every=args.optimize_every,
    )
    if getattr(args, "compact", False):
        print_json(
            {
                "status": payload.get("status"),
                "mode": payload.get("mode"),
                "rounds_requested": payload.get("rounds_requested"),
                "recorded_review_count": payload.get("recorded_review_count"),
                "wins": payload.get("wins"),
                "losses": payload.get("losses"),
                "total_realized_pnl_usdt": payload.get("total_realized_pnl_usdt"),
                "findings": payload.get("findings"),
                "report_path": payload.get("report_path"),
            },
            compact=True,
        )
        return
    print_json(payload)


def cmd_public_history_train(args: argparse.Namespace) -> None:
    payload = run_public_history_training(
        symbols=[item.strip() for item in args.symbols.split(",") if item.strip()] if args.symbols else None,
        months=args.months,
        end_month=(args.end_month or None),
        max_reviews_per_symbol=args.max_reviews_per_symbol,
        optimize_every=args.optimize_every,
    )
    summary = {
        "status": payload.get("status"),
        "mode": payload.get("mode"),
        "inserted_review_count": payload.get("inserted_review_count"),
        "skipped_duplicate_count": payload.get("skipped_duplicate_count"),
        "optimizer_reports": payload.get("optimizer_reports"),
        "report_path": payload.get("report_path"),
    }
    print_json(summary if getattr(args, "compact", False) else payload, compact=getattr(args, "compact", False))


def cmd_final_convergence_audit(args: argparse.Namespace) -> None:
    payload = run_final_convergence_audit(run_hailo=not args.skip_hailo)
    summary = {
        "status": payload.get("status"),
        "closed_review_count": (payload.get("review_database") or {}).get("closed_review_count"),
        "core_status": (payload.get("review_database") or {}).get("core_status"),
        "optimizer": payload.get("optimizer"),
        "hailo_available": (payload.get("hailo") or {}).get("available"),
        "findings": payload.get("findings"),
        "report_path": payload.get("report_path"),
    }
    print_json(summary if getattr(args, "compact", False) else payload, compact=getattr(args, "compact", False))


def cmd_route_risk_status(args: argparse.Namespace) -> None:
    if args.route_id:
        payload = {
            "status": "ok",
            "route": route_quarantine_status(args.route_id),
        }
    else:
        payload = {
            "status": "ok",
            "route_risk_control": load_route_risk_state(),
        }
    print_json(payload, compact=getattr(args, "compact", False))


def cmd_risk_combo_sweep(args: argparse.Namespace) -> None:
    payload = run_risk_combo_sweep(
        routes=[item.strip() for item in str(args.routes).split(",") if item.strip()],
        symbols=[item.strip() for item in str(args.symbols).split(",") if item.strip()],
        limit=args.limit,
        grid_mode=args.grid_mode,
        target_side=args.target_side,
        target_interval=args.target_interval,
        target_profit_factor=args.target_profit_factor,
        min_test_trades=args.min_test_trades,
        min_win_rate=args.min_win_rate,
        max_stop_loss_ratio=args.max_stop_loss_ratio,
        min_expectancy_r=args.min_expectancy_r,
        min_payoff_ratio=args.min_payoff_ratio,
        max_symbols_per_route=args.max_symbols_per_route,
        max_configs=args.max_configs,
        max_walk_forward_validations=args.max_walk_forward_validations,
        include_all_route_symbols=args.include_all_route_symbols,
        skip_news=args.skip_news,
        top_n=args.top_n,
    )
    if getattr(args, "compact", False):
        print_json(
            {
                "status": payload.get("status"),
                "mode": payload.get("mode"),
                "safety": payload.get("safety"),
                "aggregate": payload.get("aggregate"),
                "runtime_observability": payload.get("runtime_observability"),
                "best_by_route": payload.get("best_by_route"),
                "recovery_candidates": payload.get("recovery_candidates"),
                "robust_recovery_candidates": payload.get("robust_recovery_candidates"),
                "dataset_errors": payload.get("dataset_errors"),
                "report_path": payload.get("report_path"),
            },
            compact=True,
        )
        return
    print_json(payload)


def cmd_risk_combo_matrix(args: argparse.Namespace) -> None:
    payload = build_risk_combo_matrix_report(
        report_paths=split_csv_arg(args.sweep_report),
        output_dir=args.output_dir or None,
        latest_sweeps=args.latest_sweeps,
    )
    if getattr(args, "compact", False):
        print_json(
            {
                "mode": payload.get("mode"),
                "safety": payload.get("safety"),
                "input_report_count": payload.get("input_report_count"),
                "skipped_input_report_count": payload.get("skipped_input_report_count"),
                "skipped_input_reports": payload.get("skipped_input_reports"),
                "surface_count": payload.get("surface_count"),
                "promising_surface_count": payload.get("promising_surface_count"),
                "emerging_positive_lead_count": payload.get("emerging_positive_lead_count"),
                "superseded_emerging_positive_lead_count": payload.get("superseded_emerging_positive_lead_count"),
                "recent_failed_repair_identity_count": payload.get("recent_failed_repair_identity_count"),
                "robust_surface_count": payload.get("robust_surface_count"),
                "side_summary": payload.get("side_summary"),
                "horizon_summary": payload.get("horizon_summary"),
                "completion_audit": payload.get("completion_audit"),
                "objective_scorecard": payload.get("objective_scorecard"),
                "prompt_to_artifact_checklist": payload.get("prompt_to_artifact_checklist"),
                "risk_boundary": payload.get("risk_boundary"),
                "best_surface": payload.get("best_surface"),
                "emerging_positive_leads": payload.get("emerging_positive_leads"),
                "superseded_emerging_positive_leads": payload.get("superseded_emerging_positive_leads"),
                "validation_plan": payload.get("validation_plan"),
                "negative_surface_repair_plan": payload.get("negative_surface_repair_plan"),
                "next_research_actions": payload.get("next_research_actions"),
                "promotion_boundary": payload.get("promotion_boundary"),
                "report_path": payload.get("report_path"),
            },
            compact=True,
        )
        return
    print_json(payload)


def cmd_high_win_iteration(args: argparse.Namespace) -> None:
    payload = run_high_win_iteration(
        config_path=args.config,
        alpha_report_paths=split_csv_arg(args.alpha_report),
        sweep_report_paths=split_csv_arg(args.sweep_report),
        output_dir=args.output_dir or None,
        write_pid_state=not args.no_write_pid_state,
    )
    if getattr(args, "compact", False):
        print_json(compact_high_win_iteration(payload), compact=True)
        return
    print_json(payload)


def cmd_high_win_converge(args: argparse.Namespace) -> None:
    payload = run_high_win_convergence_loop(
        config_path=args.config,
        alpha_report_paths=split_csv_arg(args.alpha_report),
        sweep_report_paths=split_csv_arg(args.sweep_report),
        output_dir=args.output_dir or None,
        max_rounds=args.max_rounds,
        execute_research=args.execute_research,
        write_pid_state=not args.no_write_pid_state,
    )
    if getattr(args, "compact", False):
        print_json(
            {
                "mode": payload.get("mode"),
                "status": payload.get("status"),
                "execute_research": payload.get("execute_research"),
                "safety": payload.get("safety"),
                "policy": payload.get("policy"),
                "rounds_completed": len(payload.get("rounds") or []),
                "final_iteration": payload.get("final_iteration"),
                "promotion_allowed": payload.get("promotion_allowed"),
                "safe_to_open_new_entries": payload.get("safe_to_open_new_entries"),
                "execution_recommendation": payload.get("execution_recommendation"),
                "report_path": payload.get("report_path"),
            },
            compact=True,
        )
        return
    print_json(payload)


def cmd_repository_audit(args: argparse.Namespace) -> None:
    payload = run_repository_audit(
        root=args.root or None,
        include_generated=args.include_generated,
        output_dir=args.output_dir or None,
    )
    if getattr(args, "compact", False):
        print_json(
            {
                "mode": payload.get("mode"),
                "summary": payload.get("summary"),
                "largest_files": payload.get("largest_files"),
                "architecture_findings": payload.get("architecture_findings"),
                "report_path": payload.get("report_path"),
            },
            compact=True,
        )
        return
    print_json(payload)


def cmd_decision_audit(args: argparse.Namespace) -> None:
    payload = run_decision_audit(
        input_dir=args.input_dir or None,
        output_dir=args.output_dir or None,
        since_contract=bool(args.since_contract),
    )
    if getattr(args, "compact", False):
        print_json(
            {
                "mode": payload.get("mode"),
                "status": payload.get("status"),
                "scope": payload.get("scope"),
                "summary": payload.get("summary"),
                "invalid_artifacts": payload.get("invalid_artifacts"),
                "report_path": payload.get("report_path"),
            },
            compact=True,
        )
        return
    print_json(payload)


def cmd_professional_system_audit(args: argparse.Namespace) -> None:
    payload = run_professional_system_audit(
        config_path=args.config,
        output_dir=args.output_dir or None,
    )
    if getattr(args, "compact", False):
        print_json(
            {
                "mode": payload.get("mode"),
                "trade_ready": payload.get("trade_ready"),
                "execution_recommendation": payload.get("execution_recommendation"),
                "layer_summary": payload.get("layer_summary"),
                "critical_blockers": payload.get("critical_blockers"),
                "alpha_report": (payload.get("evidence") or {}).get("alpha_report"),
                "high_win_iteration": (payload.get("evidence") or {}).get("high_win_iteration"),
                "recommendations": payload.get("recommendations"),
                "report_path": payload.get("report_path"),
            },
            compact=True,
        )
        return
    print_json(payload)


def cmd_hermes_ai_trader(args: argparse.Namespace) -> None:
    payload = run_hermes_ai_trader(
        blueprint_config=args.blueprint_config,
        output_dir=args.output_dir or None,
    )
    if getattr(args, "compact", False):
        print_json(
            {
                "mode": payload.get("mode"),
                "safety": payload.get("safety"),
                "open_order_gate": payload.get("open_order_gate"),
                "signal": payload.get("signal"),
                "candidate_queue": payload.get("candidate_queue"),
                "machine_strategy": payload.get("machine_strategy"),
                "machine_policy": payload.get("machine_policy"),
                "portfolio_target": payload.get("portfolio_target"),
                "portfolio_risk": payload.get("portfolio_risk"),
                "committee_decision": (payload.get("committee") or {}).get("decision"),
                "feature_manifest_hash": (payload.get("feature_manifest") or {}).get("manifest_hash"),
                "signal_api": payload.get("signal_api"),
                "hailo_eligible": (payload.get("hailo_plan") or {}).get("tasks"),
                "architecture_audit": payload.get("architecture_audit"),
                "report_path": payload.get("report_path"),
            },
            compact=True,
        )
        return
    print_json(payload)


def cmd_ai_readiness_scan(args: argparse.Namespace) -> None:
    payload = run_ai_readiness_scan(
        blueprint_config=args.blueprint_config,
        strategy_config=args.strategy_config,
        output_dir=args.output_dir or None,
        market=args.market,
        limit=args.limit,
        margin_notional_usdt=(args.margin_notional_usdt if args.margin_notional_usdt > 0 else None),
        execution_mode=args.execution_mode,
        max_candidates=args.max_candidates,
    )
    if getattr(args, "compact", False):
        selected = payload.get("selected_ready_candidate")
        selected_summary = None
        if isinstance(selected, dict):
            selected_summary = {
                "rank": selected.get("rank"),
                "symbol": selected.get("symbol"),
                "side": selected.get("side"),
                "route_id": selected.get("route_id"),
                "next_action": selected.get("next_action"),
            }
        scan_summary = []
        for item in payload.get("scan_results") or []:
            if not isinstance(item, dict):
                continue
            taxonomy = item.get("blocker_taxonomy") if isinstance(item.get("blocker_taxonomy"), dict) else {}
            live_plan = item.get("live_plan") if isinstance(item.get("live_plan"), dict) else {}
            scan_summary.append(
                {
                    "rank": item.get("rank"),
                    "symbol": item.get("symbol"),
                    "side": item.get("side"),
                    "route_id": item.get("route_id"),
                    "allowed": item.get("allowed"),
                    "next_action": item.get("next_action"),
                    "blocker_classes": sorted(taxonomy.keys()),
                    "analysis_score": live_plan.get("analysis_score"),
                    "analysis_convergence": live_plan.get("analysis_convergence"),
                    "planned_account_risk_pct": live_plan.get("planned_account_risk_pct"),
                }
            )
        print_json(
            {
                "mode": payload.get("mode"),
                "safety": payload.get("safety"),
                "candidate_count": payload.get("candidate_count"),
                "scanned_count": payload.get("scanned_count"),
                "allowed_count": payload.get("allowed_count"),
                "selected_ready_candidate": selected_summary,
                "ready_after_global_unlock_count": payload.get("ready_after_global_unlock_count"),
                "selected_after_global_unlock": payload.get("selected_after_global_unlock"),
                "execution_ticket": payload.get("execution_ticket"),
                "next_machine_action": payload.get("next_machine_action"),
                "machine_action_queue": payload.get("machine_action_queue"),
                "research_candidate_report": payload.get("research_candidate_report"),
                "risk_combo_matrix_candidate_count": payload.get("risk_combo_matrix_candidate_count"),
                "risk_combo_matrix_report": payload.get("risk_combo_matrix_report"),
                "hard_blocker_classes": sorted((payload.get("hard_blocker_taxonomy") or {}).keys()),
                "denial_journal_path": payload.get("denial_journal_path"),
                "denial_journal_count": payload.get("denial_journal_count"),
                "scan_summary": scan_summary,
                "hermes_ai_trader_report": payload.get("hermes_ai_trader_report"),
                "report_path": payload.get("report_path"),
            },
            compact=True,
        )
        return
    print_json(payload)


def cmd_ai_expectancy_upgrade(args: argparse.Namespace) -> None:
    payload = run_ai_expectancy_upgrade(
        output_dir=args.output_dir or None,
        symbols=split_csv_arg(args.symbols),
        discovery_symbols=split_csv_arg(args.discovery_symbols),
        limit=args.limit,
        sweep_limit=args.sweep_limit,
        max_configs=args.max_configs,
        max_walk_forward_validations=args.max_walk_forward_validations,
        universe_limit=args.universe_limit,
        max_readiness_candidates=args.max_readiness_candidates,
        readiness_execution_mode=args.readiness_execution_mode,
        dry_run=args.dry_run,
    )
    if getattr(args, "compact", False):
        print_json(
            {
                "mode": payload.get("mode"),
                "dry_run": payload.get("dry_run"),
                "safety": payload.get("safety"),
                "objective": payload.get("objective"),
                "symbol_allocation": payload.get("symbol_allocation"),
                "selected_symbols": payload.get("selected_symbols"),
                "machine_strategy": payload.get("machine_strategy") or payload.get("hermes_machine_strategy"),
                "ai_surface_audit": payload.get("ai_surface_audit"),
                "steps": payload.get("steps") or payload.get("planned_steps"),
                "readiness_scan": payload.get("readiness_scan"),
                "final_machine_decision": payload.get("final_machine_decision"),
                "maturity_score": payload.get("maturity_score"),
                "report_path": payload.get("report_path"),
            },
            compact=True,
        )
        return
    print_json(payload)


def cmd_ai_goal_loop(args: argparse.Namespace) -> None:
    payload = run_ai_goal_loop(
        output_dir=args.output_dir or None,
        goal=args.goal,
        symbols=split_csv_arg(args.symbols),
        discovery_symbols=split_csv_arg(args.discovery_symbols),
        limit=args.limit,
        sweep_limit=args.sweep_limit,
        max_configs=args.max_configs,
        max_walk_forward_validations=args.max_walk_forward_validations,
        universe_limit=args.universe_limit,
        max_readiness_candidates=args.max_readiness_candidates,
        margin_notional_usdt=(args.margin_notional_usdt if args.margin_notional_usdt > 0 else None),
        smoke=args.smoke,
    )
    if getattr(args, "compact", False):
        print_json(
            {
                "mode": payload.get("mode"),
                "goal": payload.get("goal"),
                "safety": payload.get("safety"),
                "score": payload.get("score"),
                "surface_audit": payload.get("surface_audit"),
                "expectancy_upgrade": payload.get("expectancy_upgrade"),
                "readiness": payload.get("readiness"),
                "readiness_sizing_scout": payload.get("readiness_sizing_scout"),
                "closed_trade_feedback": payload.get("closed_trade_feedback"),
                "next_machine_action": payload.get("next_machine_action"),
                "recommended_commands": payload.get("recommended_commands"),
                "reports": payload.get("reports"),
            },
            compact=True,
        )
        return
    print_json(payload)


def cmd_ai_surface_audit(args: argparse.Namespace) -> None:
    payload = run_ai_surface_audit(output_dir=args.output_dir or None)
    if getattr(args, "compact", False):
        print_json(
            {
                "mode": payload.get("mode"),
                "status": payload.get("status"),
                "blocker_count": payload.get("blocker_count"),
                "allowed_count": payload.get("allowed_count"),
                "blockers": payload.get("blockers"),
                "report_path": payload.get("report_path"),
            },
            compact=True,
        )
        return
    print_json(payload)


def cmd_ai_market_sentinel(args: argparse.Namespace) -> None:
    payload = run_ai_market_sentinel(
        symbols=split_csv_arg(args.symbols),
        interval=args.interval,
        limit=args.limit,
        market=args.market,
        strategy_config=args.strategy_config,
        blueprint_config=args.blueprint_config,
        output_dir=args.output_dir or None,
        skip_readiness=bool(args.skip_readiness),
        send_telegram=bool(args.send_telegram),
        max_readiness_candidates=args.max_readiness_candidates,
    )
    if getattr(args, "compact", False):
        print_json(
            {
                "mode": payload.get("mode"),
                "safety": payload.get("safety"),
                "trading_control": payload.get("trading_control"),
                "position_state": payload.get("position_state"),
                "trend_state": payload.get("trend_state"),
                "route_risk": payload.get("route_risk"),
                "readiness": payload.get("readiness"),
                "expansion_gate": payload.get("expansion_gate"),
                "conditional_order_alert": payload.get("conditional_order_alert"),
                "telegram": payload.get("telegram"),
                "machine_action_queue": payload.get("machine_action_queue"),
                "errors": payload.get("errors"),
                "report_path": payload.get("report_path"),
            },
            compact=True,
        )
        return
    print_json(payload)


def cmd_hermes_trade(args: argparse.Namespace) -> None:
    action = str(args.action)
    if action == "start":
        payload = start_hermes_trade_loop(
            config_path=args.config,
            execute_testnet_entries=not bool(args.dry_run_only),
            note=args.note or "Hermes operator requested continuous trading.",
        )
    elif action == "stop":
        payload = stop_hermes_trade_loop(reason=args.reason or "Hermes operator requested stop.")
    elif action == "status":
        payload = hermes_trade_status(args.config)
    elif action == "cycle":
        payload = run_hermes_trade_cycle(
            config_path=args.config,
            force=bool(args.force),
            execute_testnet_entries=(not bool(args.dry_run_only)) if args.set_execution_mode else None,
        )
    elif action == "daemon":
        payload = run_hermes_trade_daemon(
            config_path=args.config,
            max_cycles=args.max_cycles,
            sleep_seconds=args.sleep_seconds,
        )
    else:
        raise ValueError(f"Unsupported hermes-trade action: {action}")
    if getattr(args, "compact", False):
        market_sentinel = ((payload.get("steps") or {}).get("market_sentinel") or {}).get("response")
        compact_payload = {
            "status": payload.get("status"),
            "action": action,
            "state": payload.get("state") or payload.get("state_after"),
            "trading_control": payload.get("trading_control"),
            "position_loop": payload.get("position_loop"),
            "market_sentinel": (
                {
                    "position_state": market_sentinel.get("position_state"),
                    "expansion_gate": market_sentinel.get("expansion_gate"),
                    "machine_action_queue": market_sentinel.get("machine_action_queue"),
                    "report_path": market_sentinel.get("report_path"),
                }
                if isinstance(market_sentinel, dict)
                else None
            ),
            "selected": ((payload.get("readiness") or {}).get("selected_ready_candidate")),
            "execution_gate": payload.get("execution_gate"),
            "external_context": ((payload.get("steps") or {}).get("external_context") or {}).get("response"),
            "hailo_triage": ((payload.get("steps") or {}).get("hailo_triage") or {}).get("response"),
            "executed": payload.get("executed"),
            "report_path": payload.get("report_path"),
            "cycle_count": payload.get("cycle_count"),
        }
        print_json(compact_payload, compact=True)
        return
    print_json(payload)


def cmd_trade_session(args: argparse.Namespace) -> None:
    action = str(args.action)
    if action == "start":
        payload = start_trade_session(
            dry_run_only=bool(args.dry_run_only),
            reason=args.reason or "operator start trading",
        )
    elif action == "stop":
        payload = stop_trade_session(reason=args.reason or "operator end trading")
    elif action == "status":
        payload = trade_session_status()
    else:
        raise ValueError(f"Unsupported trade-session action: {action}")
    if getattr(args, "compact", False):
        print_json(
            {
                "status": payload.get("status"),
                "action": action,
                "state": payload.get("state"),
                "safety": payload.get("safety"),
                "readiness_summary": payload.get("readiness_summary"),
                "trading_control": payload.get("trading_control")
                or (payload.get("readiness_summary") or {}).get("trading_control"),
            },
            compact=True,
        )
        return
    print_json(payload)


def cmd_feature_dataset(args: argparse.Namespace) -> None:
    settings = load_settings()
    payload = build_feature_dataset(
        settings,
        spec=FeatureDatasetSpec(
            symbols=split_csv_arg(args.symbols),
            intervals=split_csv_arg(args.intervals),
            limit=args.limit,
            strategy_config=args.strategy_config,
            market=args.market,
        ),
        output_dir=args.output_dir or None,
    )
    if getattr(args, "compact", False):
        print_json(
            {
                "mode": payload.get("mode"),
                "row_count": payload.get("row_count"),
                "dataset_hash": payload.get("dataset_hash"),
                "feature_manifest_hash": (payload.get("feature_manifest") or {}).get("manifest_hash"),
                "dataset_path": payload.get("dataset_path"),
                "errors": payload.get("errors"),
                "report_path": payload.get("report_path"),
            },
            compact=True,
        )
        return
    print_json(payload)


def cmd_market_bot_gate(args: argparse.Namespace) -> None:
    payload = evaluate_market_bot_gate(
        alpha_report=args.alpha_report,
        config_path=args.config,
        output_dir=args.output_dir or None,
    )
    if getattr(args, "compact", False):
        print_json(
            {
                "mode": payload.get("mode"),
                "safe_to_open_new_entries": payload.get("safe_to_open_new_entries"),
                "execution_recommendation": payload.get("execution_recommendation"),
                "accepted_count": payload.get("accepted_count"),
                "portfolio_gate": payload.get("portfolio_gate"),
                "diagnostics": payload.get("diagnostics"),
                "feature_manifest_hash": payload.get("feature_manifest_hash"),
                "targets": payload.get("targets"),
                "best": payload.get("best"),
                "expansion_candidates": payload.get("expansion_candidates"),
                "regressed_routes": payload.get("regressed_routes"),
                "stress_failed": payload.get("stress_failed"),
                "out_of_sample_failed": payload.get("out_of_sample_failed"),
                "report_path": payload.get("report_path"),
            },
            compact=True,
        )
        return
    print_json(payload)


def cmd_loss_diagnostics(args: argparse.Namespace) -> None:
    payload = run_loss_diagnostics(
        limit=args.limit,
        min_bucket_trades=args.min_bucket_trades,
        top_n=args.top_n,
    )
    if getattr(args, "compact", False):
        print_json(
            {
                "status": payload.get("status"),
                "input": payload.get("input"),
                "summary": payload.get("summary"),
                "findings": payload.get("findings"),
                "side_policy_recommendations": payload.get("side_policy_recommendations"),
                "root_cause_recommendations": payload.get("root_cause_recommendations"),
                "worst_buckets": payload.get("worst_buckets"),
                "best_buckets": payload.get("best_buckets"),
                "report_path": payload.get("report_path"),
            },
            compact=True,
        )
        return
    print_json(payload)


def cmd_clear_route_quarantine(args: argparse.Namespace) -> None:
    payload = clear_route_quarantine(
        args.route_id,
        reason=args.reason,
        updated_by="openclaw-quantctl clear-route-quarantine",
    )
    print_json(
        {
            "status": "ok",
            "route": route_quarantine_status(args.route_id),
            "active_quarantined_routes": payload.get("active_quarantined_routes"),
        },
        compact=getattr(args, "compact", False),
    )


def cmd_delivery_supervisor(args: argparse.Namespace) -> None:
    policy = build_supervisor_policy(
        cycles=args.cycles,
        training_rounds=args.training_rounds,
        symbols=[item.strip() for item in args.symbols.split(",") if item.strip()] if args.symbols else None,
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
    payload = run_delivery_supervisor(policy)
    if getattr(args, "compact", False):
        latest_cycle = payload["cycles"][-1] if payload.get("cycles") else {}
        print_json(
            {
                "status": payload.get("status"),
                "mode": payload.get("mode"),
                "cycles_completed": payload.get("cycles_completed"),
                "stop_reasons": payload.get("stop_reasons"),
                "live_guardrail": payload.get("live_guardrail"),
                "latest_training": (latest_cycle.get("training") or {}).get("response"),
                "latest_optimizer": latest_cycle.get("optimizer"),
                "latest_database": latest_cycle.get("database"),
                "quarantined_routes": ((latest_cycle.get("route_risk") or {}).get("quarantined_routes")),
                "report_path": payload.get("report_path"),
            },
            compact=True,
        )
        return
    print_json(payload)


def cmd_backfill_live_journal(args: argparse.Namespace) -> None:
    records = read_live_orders()
    updated, changed = backfill_live_order_metadata(records)
    if changed and not args.dry_run:
        write_live_orders(updated)
    print_json(
        {
            "status": "ok",
            "dry_run": bool(args.dry_run),
            "live_order_count": len(records),
            "changed_count": changed,
            "written": bool(changed and not args.dry_run),
        },
        compact=getattr(args, "compact", False),
    )


def cmd_backfill_closed_reviews(args: argparse.Namespace) -> None:
    records = read_closed_trade_reviews()
    updated, changed = backfill_closed_trade_review_metadata(records)
    if changed and not args.dry_run:
        write_closed_trade_reviews(updated)
    print_json(
        {
            "status": "ok",
            "dry_run": bool(args.dry_run),
            "closed_review_count": len(records),
            "changed_count": changed,
            "written": bool(changed and not args.dry_run),
        },
        compact=getattr(args, "compact", False),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenClaw Binance quant control plane")
    sub = parser.add_subparsers(dest="cmd", required=True)

    doctor = sub.add_parser("doctor", help="Audit Binance quant toolchain readiness")
    doctor.add_argument("--use-blave", action="store_true", help="Also probe optional Blave connectivity")
    doctor.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    doctor.set_defaults(func=cmd_doctor)

    env_template = sub.add_parser("env-template", help="Print the .env template")
    env_template.set_defaults(func=cmd_env_template)

    validate_config = sub.add_parser("validate-config", help="Validate settings and strategy configuration")
    validate_config.add_argument("--strategy-config", default=str(CONFIG_DIR / "strategy-stable-risk.yaml"))
    validate_config.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    validate_config.set_defaults(func=cmd_validate_config)

    analyze = sub.add_parser("analyze", help="Run market analysis and write artifacts")
    analyze.add_argument("symbol")
    analyze.add_argument("--market", choices=("spot", "futures"), default=load_settings().default_market)
    analyze.add_argument("--interval", default="1h")
    analyze.add_argument("--limit", type=int, default=500)
    analyze.add_argument("--use-blave", action="store_true")
    analyze.add_argument("--render-chart", action="store_true")
    analyze.add_argument("--output-dir", default="")
    analyze.add_argument("--strategy-config", default="")
    analyze.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    analyze.set_defaults(func=cmd_analyze)

    account = sub.add_parser("account", help="Read private account information")
    account.add_argument("--market", choices=("spot", "futures"), default=load_settings().default_market)
    account.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    account.set_defaults(func=cmd_account)

    positions = sub.add_parser("positions", help="Read futures positions")
    positions.add_argument("--symbol", default="")
    positions.add_argument("--all", action="store_true", help="Include flat positions too")
    positions.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    positions.set_defaults(func=cmd_positions)

    trading_control_status = sub.add_parser("trading-control-status", help="Show kill-switch / pause state")
    trading_control_status.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    trading_control_status.set_defaults(func=cmd_trading_control_status)

    pause_trading = sub.add_parser("pause-trading", help="Kill-switch: cancel open orders and block new entries")
    pause_trading.add_argument("--reason", default="", help="Operator note for audit trail")
    pause_trading.add_argument("--no-cancel-open-orders", dest="cancel_open_orders", action="store_false")
    pause_trading.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    pause_trading.set_defaults(func=cmd_pause_trading, cancel_open_orders=True)

    resume_trading = sub.add_parser("resume-trading", help="Release kill-switch and allow new entries")
    resume_trading.add_argument("--reason", default="", help="Operator note for audit trail")
    resume_trading.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    resume_trading.set_defaults(func=cmd_resume_trading)

    auto_pause = sub.add_parser("auto-pause-trading", help="Evaluate auto-stop rules and pause trading when triggered")
    auto_pause.add_argument("--strategy-config", default=str(CONFIG_DIR / "strategy-live-pilot.yaml"))
    auto_pause.add_argument("--consecutive-loss-threshold", type=int, default=2)
    auto_pause.add_argument(
        "--loss-cooldown-hours",
        type=float,
        default=None,
        help="Keep consecutive-loss auto-pause active for this many hours. Default uses strategy risk.cooldown_hours.",
    )
    auto_pause.add_argument(
        "--position-timeout-hours",
        type=float,
        default=0.0,
        help="Pause after a position has been open this many hours. Default 0 disables timeout pause.",
    )
    auto_pause.add_argument("--reversal-min-score", type=int, default=80)
    auto_pause.add_argument("--reversal-min-convergence", type=float, default=0.8)
    auto_pause.add_argument(
        "--cancel-open-orders",
        dest="cancel_open_orders",
        action="store_true",
        help=(
            "Emergency mode: cancel futures open/algo orders when auto-pause triggers. "
            "Default is false so existing TP/SL protection is preserved."
        ),
    )
    auto_pause.add_argument("--no-cancel-open-orders", dest="cancel_open_orders", action="store_false")
    auto_pause.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    auto_pause.set_defaults(func=cmd_auto_pause_trading, cancel_open_orders=False, loss_cooldown_hours=None)

    paper = sub.add_parser("paper-order", help="Create a paper order journal entry from current market state")
    paper.add_argument("symbol")
    paper.add_argument("--market", choices=("spot", "futures"), default=load_settings().default_market)
    paper.add_argument("--side", choices=("BUY", "SELL"), required=True)
    paper.add_argument("--notional-usdt", type=float, required=True)
    paper.add_argument("--leverage", type=float, default=0.0)
    paper.add_argument("--interval", default="1h")
    paper.add_argument("--limit", type=int, default=500)
    paper.add_argument("--use-blave", action="store_true")
    paper.add_argument("--render-chart", action="store_true")
    paper.add_argument("--strategy-config", default="")
    paper.set_defaults(func=cmd_paper_order)

    route_symbol = sub.add_parser("route-symbol", help="Classify a symbol and show its strategy lane")
    route_symbol.add_argument("symbol")
    route_symbol.set_defaults(func=cmd_route_symbol)

    route_intent = sub.add_parser("route-intent", help="Map an operator message into the trading workflow action lane")
    route_intent.add_argument("message")
    route_intent.set_defaults(func=cmd_route_intent)

    new_symbol_workflow = sub.add_parser(
        "new-symbol-workflow",
        help="Run the fixed no-code workflow from new symbol to research/testnet readiness without opening orders",
    )
    new_symbol_workflow.add_argument("--symbols", required=True, help="Comma-separated symbol list, e.g. SOLUSDT,TRXUSDT")
    new_symbol_workflow.add_argument("--intervals", default="15m,4h,1d")
    new_symbol_workflow.add_argument("--sides", default="BUY,SELL")
    new_symbol_workflow.add_argument("--research-depth", choices=("none", "smoke", "focused"), default="smoke")
    new_symbol_workflow.add_argument(
        "--plan-only",
        action="store_true",
        help="Emit the fixed workflow plan and cheap gates without running research sweeps.",
    )
    new_symbol_workflow.add_argument("--strategy-config", default=str(CONFIG_DIR / "strategy-live-pilot.yaml"))
    new_symbol_workflow.add_argument(
        "--blueprint-config",
        default=str(CONFIG_DIR / "professional-system-blueprint.default.yaml"),
    )
    new_symbol_workflow.add_argument("--max-readiness-candidates", type=int, default=6)
    new_symbol_workflow.add_argument("--output-dir", default="")
    new_symbol_workflow.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    new_symbol_workflow.set_defaults(func=cmd_new_symbol_workflow)

    mission = sub.add_parser("mission", help="Run one-command symbol-scoped strategy convergence automation")
    mission.add_argument("--symbols", required=True, help="Comma-separated symbol list, e.g. BTCUSDT,ETHUSDT,XAU")
    mission.add_argument("--target-return-pct", type=float, default=0.0)
    mission.add_argument("--max-leverage", type=float, default=3.0)
    mission.add_argument("--execute-live", action="store_true")
    mission.add_argument("--config", default=str(CONFIG_DIR / "mission-control.default.yaml"))
    mission.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    mission.set_defaults(func=cmd_mission)

    build = sub.add_parser(
        "build-analysis-spec",
        help="Generate an operator-only manual taskctl workflow spec",
    )
    build.add_argument("symbol")
    build.add_argument("--market", choices=("spot", "futures"), default=load_settings().default_market)
    build.add_argument("--interval", default="1h")
    build.add_argument("--limit", type=int, default=500)
    build.add_argument("--name", default="")
    build.add_argument("--use-blave", action="store_true")
    build.add_argument("--render-chart", action="store_true")
    build.add_argument("--include-account-check", action="store_true")
    build.add_argument("--strategy-config", default="")
    build.set_defaults(func=cmd_build_analysis_spec)

    submit = sub.add_parser(
        "submit-analysis",
        help="Generate, lint, and submit an operator-only manual analysis workflow",
    )
    submit.add_argument("symbol")
    submit.add_argument("--market", choices=("spot", "futures"), default=load_settings().default_market)
    submit.add_argument("--interval", default="1h")
    submit.add_argument("--limit", type=int, default=500)
    submit.add_argument("--name", default="")
    submit.add_argument("--use-blave", action="store_true")
    submit.add_argument("--render-chart", action="store_true")
    submit.add_argument("--include-account-check", action="store_true")
    submit.add_argument("--run-now", action="store_true")
    submit.add_argument("--strategy-config", default="")
    submit.set_defaults(func=cmd_submit_analysis)

    backtest = sub.add_parser("backtest", help="Run a deterministic backtest against Binance candles")
    backtest.add_argument("--strategy-config", default=str(CONFIG_DIR / "strategy-live-pilot.yaml"))
    backtest.add_argument("--symbol", default="")
    backtest.add_argument("--market", choices=("spot", "futures"), default="")
    backtest.add_argument("--interval", default="")
    backtest.add_argument("--limit", type=int, default=0)
    backtest.add_argument("--output-dir", default="")
    backtest.set_defaults(func=cmd_backtest)

    sweep = sub.add_parser("backtest-sweep", help="Run the large multi-symbol backtest sweep")
    sweep.set_defaults(func=cmd_backtest_sweep)

    alpha_research = sub.add_parser(
        "alpha-research",
        help="Run the demo/testnet-only aggressive alpha research ranking lane",
    )
    alpha_research.add_argument(
        "--config",
        default=str(CONFIG_DIR / "aggressive-alpha-research.default.yaml"),
    )
    alpha_research.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated symbol override. Defaults to config universe.",
    )
    alpha_research.add_argument(
        "--intervals",
        default="",
        help="Optional comma-separated interval override. Defaults to config universe.",
    )
    alpha_research.add_argument("--limit", type=int, default=0)
    alpha_research.add_argument(
        "--strategy-families",
        default="",
        help="Optional comma-separated strategy family override for targeted reruns.",
    )
    alpha_research.add_argument("--output-dir", default="")
    alpha_research.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    alpha_research.set_defaults(func=cmd_alpha_research)

    external_context = sub.add_parser(
        "external-context",
        help="Collect optional CMC/Arkham/DexScreener/CryptoPanic/Glassnode context",
    )
    external_context.add_argument(
        "--config",
        default=str(CONFIG_DIR / "external-context.default.yaml"),
    )
    external_context.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,XAUTUSDT,PAXGUSDT,BNBUSDT,SOLUSDT,XRPUSDT,LINKUSDT,AAVEUSDT,TRXUSDT",
    )
    external_context.add_argument("--no-cache", action="store_true")
    external_context.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    external_context.set_defaults(func=cmd_external_context)

    external_context_keys = sub.add_parser(
        "external-context-key-status",
        help="Show which optional external context API keys are configured without printing secrets",
    )
    external_context_keys.add_argument(
        "--config",
        default=str(CONFIG_DIR / "external-context.default.yaml"),
    )
    external_context_keys.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    external_context_keys.set_defaults(func=cmd_external_context_key_status)

    challenge_init = sub.add_parser("challenge-init", help="Initialize a funded challenge target with drawdown limits")
    challenge_init.add_argument("--strategy-config", default=str(CONFIG_DIR / "strategy-live-pilot.yaml"))
    challenge_init.add_argument("--start-balance-usdt", type=float, default=0.0)
    challenge_init.add_argument("--from-account", action="store_true")
    challenge_init.add_argument("--target-multiple", type=float, default=0.0)
    challenge_init.add_argument("--max-drawdown-pct", type=float, default=0.0)
    challenge_init.add_argument("--note", default="")
    challenge_init.set_defaults(func=cmd_challenge_init)

    challenge_status = sub.add_parser("challenge-status", help="Show funded challenge progress and recent balance snapshots")
    challenge_status.add_argument("--strategy-config", default=str(CONFIG_DIR / "strategy-live-pilot.yaml"))
    challenge_status.add_argument("--refresh", action="store_true")
    challenge_status.add_argument("--snapshot-limit", type=int, default=5)
    challenge_status.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    challenge_status.set_defaults(func=cmd_challenge_status)

    journal_summary = sub.add_parser("journal-summary", help="Summarize live journal, trade state, and challenge state")
    journal_summary.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    journal_summary.set_defaults(func=cmd_journal_summary)

    operator_dashboard = sub.add_parser(
        "operator-dashboard",
        help="Show customer-facing PnL, protection, and loss-cause feedback without using an LLM",
    )
    operator_dashboard.add_argument("--min-bucket-trades", type=int, default=10)
    operator_dashboard.add_argument("--top-n", type=int, default=5)
    operator_dashboard.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    operator_dashboard.set_defaults(func=cmd_operator_dashboard)

    review_closed = sub.add_parser(
        "review-closed-trades",
        help="Review closed TP/SL outcomes for strategy optimization without changing execution settings",
    )
    review_closed.add_argument("--limit", type=int, default=20)
    review_closed.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    review_closed.set_defaults(func=cmd_review_closed_trades)

    stability_workflow = sub.add_parser("stability-workflow", help="Run the local stability quality gates as one workflow")
    stability_workflow.add_argument("--strategy-config", default=str(CONFIG_DIR / "strategy-stable-risk.yaml"))
    stability_workflow.add_argument("--skip-ruff", action="store_true")
    stability_workflow.add_argument("--skip-pytest", action="store_true")
    stability_workflow.add_argument("--include-doctor", action="store_true")
    stability_workflow.set_defaults(func=cmd_stability_workflow)

    manage_position = sub.add_parser("manage-position", help="Dry-run or update protective orders for an open futures position")
    manage_position.add_argument("--strategy-config", default=str(CONFIG_DIR / "strategy-live-pilot.yaml"))
    manage_position.add_argument("--symbol", default="")
    manage_position.add_argument("--market", choices=("spot", "futures"), default="")
    manage_position.add_argument("--stop-price", type=float, default=0.0)
    manage_position.add_argument("--take-profit-price", type=float, default=0.0)
    manage_position.add_argument("--execute", action="store_true")
    manage_position.set_defaults(func=cmd_manage_position)

    trailing_update = sub.add_parser("trailing-update", help="Dry-run or arm a trailing stop for an open futures position")
    trailing_update.add_argument("--strategy-config", default=str(CONFIG_DIR / "strategy-live-pilot.yaml"))
    trailing_update.add_argument("--symbol", default="")
    trailing_update.add_argument("--market", choices=("spot", "futures"), default="")
    trailing_update.add_argument("--callback-pct", type=float, default=0.0)
    trailing_update.add_argument("--activation-price", type=float, default=0.0)
    trailing_update.add_argument("--take-profit-price", type=float, default=0.0)
    trailing_update.add_argument("--execute", action="store_true")
    trailing_update.set_defaults(func=cmd_trailing_update)

    repair_staged_tp = sub.add_parser(
        "repair-staged-tp",
        help="Rebuild futures protective orders as stop-loss plus staged take-profit quantities",
    )
    repair_staged_tp.add_argument("--strategy-config", default=str(CONFIG_DIR / "strategy-live-pilot.yaml"))
    repair_staged_tp.add_argument("--symbol", default="")
    repair_staged_tp.add_argument("--side", choices=("BUY", "SELL"), default="")
    repair_staged_tp.add_argument("--news-risk", choices=("normal", "elevated", "high"), default="normal")
    repair_staged_tp.add_argument("--execute", action="store_true")
    repair_staged_tp.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    repair_staged_tp.set_defaults(func=cmd_repair_staged_tp)

    live_readiness = sub.add_parser("live-readiness", help="Build a live-trading readiness plan without sending orders")
    live_readiness.add_argument("--strategy-config", default=str(CONFIG_DIR / "strategy-live-pilot.yaml"))
    live_readiness.add_argument("--symbol", default="")
    live_readiness.add_argument("--market", choices=("spot", "futures"), default="")
    live_readiness.add_argument("--interval", default="")
    live_readiness.add_argument("--limit", type=int, default=0)
    live_readiness.add_argument("--use-blave", action="store_true")
    live_readiness.add_argument("--render-chart", action="store_true")
    live_readiness.add_argument("--side", choices=("BUY", "SELL"), default="")
    live_readiness.add_argument("--margin-notional-usdt", type=float, default=0.0)
    live_readiness.add_argument("--execution-mode", choices=("live", "testnet_exploration"), default="live")
    live_readiness.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    live_readiness.set_defaults(func=cmd_live_readiness, execute=False)

    trade_decision = sub.add_parser(
        "trade-decision",
        help="Emit auditable BUY/SELL/HOLD/EXIT decision schema without sending orders",
    )
    trade_decision.add_argument("--strategy-config", default=str(CONFIG_DIR / "strategy-live-pilot.yaml"))
    trade_decision.add_argument("--symbol", default="")
    trade_decision.add_argument("--market", choices=("spot", "futures"), default="")
    trade_decision.add_argument("--interval", default="")
    trade_decision.add_argument("--limit", type=int, default=0)
    trade_decision.add_argument("--use-blave", action="store_true")
    trade_decision.add_argument("--render-chart", action="store_true")
    trade_decision.add_argument("--side", choices=("BUY", "SELL"), default="")
    trade_decision.add_argument("--action", choices=("BUY", "LONG", "SELL", "SHORT", "HOLD", "EXIT"), default="")
    trade_decision.add_argument("--margin-notional-usdt", type=float, default=0.0)
    trade_decision.add_argument("--execution-mode", choices=("live", "testnet_exploration"), default="testnet_exploration")
    trade_decision.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    trade_decision.set_defaults(func=cmd_trade_decision, execute=False)

    decision_audit = sub.add_parser(
        "decision-audit",
        help="Audit stored trade-decision JSON artifacts for contract, risk, and read-only violations",
    )
    decision_audit.add_argument("--input-dir", default="")
    decision_audit.add_argument("--output-dir", default="")
    decision_audit.add_argument(
        "--since-contract",
        action="store_true",
        help="Ignore pre-contract legacy artifacts that lack embedded decision_contract_validation",
    )
    decision_audit.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    decision_audit.set_defaults(func=cmd_decision_audit)

    live_pilot = sub.add_parser("live-pilot", help="Run the live pilot path; use --execute to actually place orders")
    live_pilot.add_argument("--strategy-config", default=str(CONFIG_DIR / "strategy-live-pilot.yaml"))
    live_pilot.add_argument("--symbol", default="")
    live_pilot.add_argument("--market", choices=("spot", "futures"), default="")
    live_pilot.add_argument("--interval", default="")
    live_pilot.add_argument("--limit", type=int, default=0)
    live_pilot.add_argument("--use-blave", action="store_true")
    live_pilot.add_argument("--render-chart", action="store_true")
    live_pilot.add_argument("--side", choices=("BUY", "SELL"), default="")
    live_pilot.add_argument("--margin-notional-usdt", type=float, default=0.0)
    live_pilot.add_argument("--execution-mode", choices=("live", "testnet_exploration"), default="live")
    live_pilot.add_argument("--execute", action="store_true")
    live_pilot.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    live_pilot.set_defaults(func=cmd_live_pilot)

    demo_train = sub.add_parser(
        "demo-train",
        help="Run replay-backed training rounds that convert analysis and live-plan structure into closed review samples",
    )
    demo_train.add_argument("--rounds", type=int, default=10)
    demo_train.add_argument("--symbols", default="")
    demo_train.add_argument("--target-return-pct", type=float, default=5.0)
    demo_train.add_argument("--max-leverage", type=float, default=3.0)
    demo_train.add_argument("--margin-notional-usdt", type=float, default=4.0)
    demo_train.add_argument("--optimize-every", type=int, default=10)
    demo_train.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    demo_train.set_defaults(func=cmd_demo_train)

    public_history_train = sub.add_parser(
        "public-history-train",
        help="Import Binance public historical klines into closed-trade training reviews",
    )
    public_history_train.add_argument("--symbols", default="BTCUSDT,ETHUSDT,PAXGUSDT")
    public_history_train.add_argument("--months", type=int, default=24)
    public_history_train.add_argument("--end-month", default="")
    public_history_train.add_argument("--max-reviews-per-symbol", type=int, default=100)
    public_history_train.add_argument("--optimize-every", type=int, default=250)
    public_history_train.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    public_history_train.set_defaults(func=cmd_public_history_train)

    final_audit = sub.add_parser(
        "final-convergence-audit",
        help="Run delivery-readiness audit across review DB, optimizer, one-in-one-out context, and Hailo triage",
    )
    final_audit.add_argument("--skip-hailo", action="store_true")
    final_audit.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    final_audit.set_defaults(func=cmd_final_convergence_audit)

    route_risk_status = sub.add_parser(
        "route-risk-status",
        help="Show route-level quarantine state from delivery supervision",
    )
    route_risk_status.add_argument("--route-id", default="")
    route_risk_status.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    route_risk_status.set_defaults(func=cmd_route_risk_status)

    risk_combo_sweep = sub.add_parser(
        "risk-combo-sweep",
        help="Sweep risk-control combinations for quarantined routes without writing training reviews",
    )
    risk_combo_sweep.add_argument(
        "--routes",
        default="",
        help="Comma-separated route IDs. Defaults to active route quarantines.",
    )
    risk_combo_sweep.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols. When set, routes are inferred from symbols.",
    )
    risk_combo_sweep.add_argument("--limit", type=int, default=1500)
    risk_combo_sweep.add_argument("--grid-mode", choices=("fast", "focused", "standard"), default="fast")
    risk_combo_sweep.add_argument(
        "--target-side",
        choices=("BUY", "SELL"),
        default="",
        help="Research-only side filter. Backtests only entries matching this direction.",
    )
    risk_combo_sweep.add_argument(
        "--target-interval",
        default="",
        help="Research-only kline interval override, e.g. 15m, 4h, 1d.",
    )
    risk_combo_sweep.add_argument("--target-profit-factor", type=float, default=0.8)
    risk_combo_sweep.add_argument("--min-test-trades", type=int, default=3)
    risk_combo_sweep.add_argument("--min-win-rate", type=float, default=0.0)
    risk_combo_sweep.add_argument("--max-stop-loss-ratio", type=float, default=100.0)
    risk_combo_sweep.add_argument("--min-expectancy-r", type=float, default=0.0)
    risk_combo_sweep.add_argument("--min-payoff-ratio", type=float, default=0.0)
    risk_combo_sweep.add_argument("--max-symbols-per-route", type=int, default=0)
    risk_combo_sweep.add_argument("--max-configs", type=int, default=0)
    risk_combo_sweep.add_argument("--max-walk-forward-validations", type=int, default=0)
    risk_combo_sweep.add_argument("--include-all-route-symbols", action="store_true")
    risk_combo_sweep.add_argument("--skip-news", action="store_true")
    risk_combo_sweep.add_argument("--top-n", type=int, default=20)
    risk_combo_sweep.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    risk_combo_sweep.set_defaults(func=cmd_risk_combo_sweep)

    risk_combo_matrix = sub.add_parser(
        "risk-combo-matrix",
        help="Aggregate risk-combo-sweep reports into an auditable BUY/SELL x interval research matrix",
    )
    risk_combo_matrix.add_argument(
        "--sweep-report",
        action="append",
        default=None,
        help="Risk-combo-sweep report path. May be repeated or comma-separated.",
    )
    risk_combo_matrix.add_argument(
        "--latest-sweeps",
        type=int,
        default=0,
        help="Automatically include the most recent N risk-combo-sweep reports.",
    )
    risk_combo_matrix.add_argument("--output-dir", default="")
    risk_combo_matrix.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    risk_combo_matrix.set_defaults(func=cmd_risk_combo_matrix)

    high_win_iteration = sub.add_parser(
        "high-win-iteration",
        help="Build the next expectancy-first research iteration plan without sending orders",
    )
    high_win_iteration.add_argument(
        "--config",
        default=str(CONFIG_DIR / "high-win-iteration.default.yaml"),
    )
    high_win_iteration.add_argument(
        "--alpha-report",
        action="append",
        default=None,
        help="Alpha-research report path. May be repeated or comma-separated. Defaults to config reports.",
    )
    high_win_iteration.add_argument(
        "--sweep-report",
        action="append",
        default=None,
        help="Risk-combo-sweep report path. May be repeated or comma-separated. Defaults to config reports.",
    )
    high_win_iteration.add_argument("--output-dir", default="")
    high_win_iteration.add_argument("--no-write-pid-state", action="store_true")
    high_win_iteration.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    high_win_iteration.set_defaults(func=cmd_high_win_iteration)

    high_win_converge = sub.add_parser(
        "high-win-converge",
        help="Run bounded strict high-win research convergence until promotion or stop condition",
    )
    high_win_converge.add_argument(
        "--config",
        default=str(CONFIG_DIR / "high-win-iteration.default.yaml"),
    )
    high_win_converge.add_argument(
        "--alpha-report",
        action="append",
        default=None,
        help="Starting alpha-research report path. May be repeated or comma-separated. Defaults to config reports.",
    )
    high_win_converge.add_argument(
        "--sweep-report",
        action="append",
        default=None,
        help="Starting risk-combo-sweep report path. May be repeated or comma-separated. Defaults to config reports.",
    )
    high_win_converge.add_argument("--output-dir", default="")
    high_win_converge.add_argument("--max-rounds", type=int, default=0)
    high_win_converge.add_argument(
        "--execute-research",
        action="store_true",
        help="Actually run bounded alpha-research / risk-combo backtest batches. Default is plan-only.",
    )
    high_win_converge.add_argument("--no-write-pid-state", action="store_true")
    high_win_converge.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    high_win_converge.set_defaults(func=cmd_high_win_converge)

    repository_audit = sub.add_parser(
        "repository-audit",
        help="Audit source/config/docs/scripts/tests structure without reading secrets",
    )
    repository_audit.add_argument("--root", default="")
    repository_audit.add_argument("--output-dir", default="")
    repository_audit.add_argument(
        "--include-generated",
        action="store_true",
        help="Include pycache, egg-info, logs, and local backups in the file list.",
    )
    repository_audit.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    repository_audit.set_defaults(func=cmd_repository_audit)

    professional_system_audit = sub.add_parser(
        "professional-system-audit",
        help="Audit the trading system against professional bot architecture gates without trading",
    )
    professional_system_audit.add_argument(
        "--config",
        default=str(CONFIG_DIR / "professional-system-blueprint.default.yaml"),
    )
    professional_system_audit.add_argument("--output-dir", default="")
    professional_system_audit.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    professional_system_audit.set_defaults(func=cmd_professional_system_audit)

    hermes_ai_trader = sub.add_parser(
        "hermes-ai-trader",
        help="Run the clean Hermes AI Trader v2 gate: signal schema, committee, portfolio, Hailo plan",
    )
    hermes_ai_trader.add_argument(
        "--blueprint-config",
        default=str(CONFIG_DIR / "professional-system-blueprint.default.yaml"),
    )
    hermes_ai_trader.add_argument("--output-dir", default="")
    hermes_ai_trader.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    hermes_ai_trader.set_defaults(func=cmd_hermes_ai_trader)

    ai_readiness_scan = sub.add_parser(
        "ai-readiness-scan",
        help="Scan Hermes AI Trader candidates through live-readiness dry-run without sending orders",
    )
    ai_readiness_scan.add_argument(
        "--blueprint-config",
        default=str(CONFIG_DIR / "professional-system-blueprint.default.yaml"),
    )
    ai_readiness_scan.add_argument("--strategy-config", default=str(CONFIG_DIR / "strategy-live-pilot.yaml"))
    ai_readiness_scan.add_argument("--market", choices=("spot", "futures"), default="futures")
    ai_readiness_scan.add_argument("--limit", type=int, default=0)
    ai_readiness_scan.add_argument("--margin-notional-usdt", type=float, default=0.0)
    ai_readiness_scan.add_argument("--execution-mode", choices=("live", "testnet_exploration"), default="testnet_exploration")
    ai_readiness_scan.add_argument("--max-candidates", type=int, default=0)
    ai_readiness_scan.add_argument("--output-dir", default="")
    ai_readiness_scan.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    ai_readiness_scan.set_defaults(func=cmd_ai_readiness_scan)

    ai_expectancy_upgrade = sub.add_parser(
        "ai-expectancy-upgrade",
        help="Run AI-trader expectancy upgrade loop without opening orders",
    )
    ai_expectancy_upgrade.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated primary symbols. Defaults to Hermes AI Trader priority queue.",
    )
    ai_expectancy_upgrade.add_argument(
        "--discovery-symbols",
        default="",
        help="Optional comma-separated portfolio discovery symbols. Defaults to AI-selected top six.",
    )
    ai_expectancy_upgrade.add_argument("--limit", type=int, default=8000)
    ai_expectancy_upgrade.add_argument("--sweep-limit", type=int, default=5000)
    ai_expectancy_upgrade.add_argument("--max-configs", type=int, default=80)
    ai_expectancy_upgrade.add_argument("--max-walk-forward-validations", type=int, default=12)
    ai_expectancy_upgrade.add_argument("--universe-limit", type=int, default=12)
    ai_expectancy_upgrade.add_argument("--max-readiness-candidates", type=int, default=6)
    ai_expectancy_upgrade.add_argument(
        "--readiness-execution-mode",
        choices=("live", "testnet_exploration"),
        default="testnet_exploration",
    )
    ai_expectancy_upgrade.add_argument("--output-dir", default="")
    ai_expectancy_upgrade.add_argument("--dry-run", action="store_true")
    ai_expectancy_upgrade.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    ai_expectancy_upgrade.set_defaults(func=cmd_ai_expectancy_upgrade)

    ai_goal_loop = sub.add_parser(
        "ai-goal-loop",
        help="Run a machine-only expectancy/readiness/feedback goal loop without opening orders",
    )
    ai_goal_loop.add_argument("--goal", default="maximize_stable_expectancy")
    ai_goal_loop.add_argument("--symbols", default="")
    ai_goal_loop.add_argument("--discovery-symbols", default="")
    ai_goal_loop.add_argument("--limit", type=int, default=8000)
    ai_goal_loop.add_argument("--sweep-limit", type=int, default=5000)
    ai_goal_loop.add_argument("--max-configs", type=int, default=80)
    ai_goal_loop.add_argument("--max-walk-forward-validations", type=int, default=12)
    ai_goal_loop.add_argument("--universe-limit", type=int, default=20)
    ai_goal_loop.add_argument("--max-readiness-candidates", type=int, default=6)
    ai_goal_loop.add_argument("--margin-notional-usdt", type=float, default=0.0)
    ai_goal_loop.add_argument("--output-dir", default="")
    ai_goal_loop.add_argument("--smoke", action="store_true")
    ai_goal_loop.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    ai_goal_loop.set_defaults(func=cmd_ai_goal_loop)

    ai_surface_audit = sub.add_parser(
        "ai-surface-audit",
        help="Audit AI decision surfaces for human narrative leakage",
    )
    ai_surface_audit.add_argument("--output-dir", default="")
    ai_surface_audit.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    ai_surface_audit.set_defaults(func=cmd_ai_surface_audit)

    ai_market_sentinel = sub.add_parser(
        "ai-market-sentinel",
        help="Run a read-only 24h AI market/position sentinel without opening orders",
    )
    ai_market_sentinel.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,DOGEUSDT,TRXUSDT",
    )
    ai_market_sentinel.add_argument("--interval", default="15m")
    ai_market_sentinel.add_argument("--limit", type=int, default=160)
    ai_market_sentinel.add_argument("--market", choices=("spot", "futures"), default="futures")
    ai_market_sentinel.add_argument("--strategy-config", default=str(CONFIG_DIR / "strategy-live-pilot.yaml"))
    ai_market_sentinel.add_argument(
        "--blueprint-config",
        default=str(CONFIG_DIR / "professional-system-blueprint.default.yaml"),
    )
    ai_market_sentinel.add_argument("--output-dir", default="")
    ai_market_sentinel.add_argument("--skip-readiness", action="store_true")
    ai_market_sentinel.add_argument("--max-readiness-candidates", type=int, default=6)
    ai_market_sentinel.add_argument(
        "--send-telegram",
        action="store_true",
        help="Send the readiness-approved conditional order candidate to Telegram",
    )
    ai_market_sentinel.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    ai_market_sentinel.set_defaults(func=cmd_ai_market_sentinel)

    hermes_trade = sub.add_parser(
        "hermes-trade",
        help="Start/stop/status/cycle the Hermes-controlled continuous testnet trade loop",
    )
    hermes_trade.add_argument("action", choices=("start", "stop", "status", "cycle", "daemon"))
    hermes_trade.add_argument("--config", default=str(DEFAULT_HERMES_TRADE_CONFIG_PATH))
    hermes_trade.add_argument("--note", default="")
    hermes_trade.add_argument("--reason", default="")
    hermes_trade.add_argument("--dry-run-only", action="store_true", help="Keep loop enabled but do not execute testnet tickets")
    hermes_trade.add_argument("--set-execution-mode", action="store_true", help="Apply --dry-run-only to stored execution state for cycle")
    hermes_trade.add_argument("--force", action="store_true", help="Run one cycle even if loop is currently disabled")
    hermes_trade.add_argument("--max-cycles", type=int, default=0)
    hermes_trade.add_argument("--sleep-seconds", type=float, default=None)
    hermes_trade.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    hermes_trade.set_defaults(func=cmd_hermes_trade)

    trade_session = sub.add_parser(
        "trade-session",
        help="Start/stop/status the local low-token trading session timers",
    )
    trade_session.add_argument("action", choices=("start", "stop", "status"))
    trade_session.add_argument("--dry-run-only", action="store_true", default=True)
    trade_session.add_argument("--execute-testnet", dest="dry_run_only", action="store_false")
    trade_session.add_argument("--reason", default="")
    trade_session.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    trade_session.set_defaults(func=cmd_trade_session)

    feature_dataset = sub.add_parser(
        "feature-dataset",
        help="Build replayable feature rows with manifest hash for research and model training",
    )
    feature_dataset.add_argument("--symbols", default="TRXUSDT,ETHUSDT,BTCUSDT")
    feature_dataset.add_argument("--intervals", default="1h,4h")
    feature_dataset.add_argument("--limit", type=int, default=5000)
    feature_dataset.add_argument("--market", choices=("spot", "futures"), default="futures")
    feature_dataset.add_argument(
        "--strategy-config",
        default=str(CONFIG_DIR / "strategy-core-high-win-research.yaml"),
    )
    feature_dataset.add_argument("--output-dir", default="")
    feature_dataset.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    feature_dataset.set_defaults(func=cmd_feature_dataset)

    market_bot_gate = sub.add_parser(
        "market-bot-gate",
        help="Evaluate alpha rows with mainstream bot expectancy/payoff/protection gates",
    )
    market_bot_gate.add_argument(
        "--config",
        default=str(DEFAULT_MARKET_BOT_GATE_CONFIG),
    )
    market_bot_gate.add_argument(
        "--alpha-report",
        default="state/recheck-core-alpha-mapped-l1500/alpha-research-ranking.json",
    )
    market_bot_gate.add_argument("--output-dir", default="")
    market_bot_gate.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    market_bot_gate.set_defaults(func=cmd_market_bot_gate)

    loss_diagnostics = sub.add_parser(
        "loss-diagnostics",
        help="Diagnose losing closed-trade buckets by route, side, source, and signal quality",
    )
    loss_diagnostics.add_argument("--limit", type=int, default=0)
    loss_diagnostics.add_argument("--min-bucket-trades", type=int, default=5)
    loss_diagnostics.add_argument("--top-n", type=int, default=20)
    loss_diagnostics.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    loss_diagnostics.set_defaults(func=cmd_loss_diagnostics)

    clear_quarantine = sub.add_parser(
        "clear-route-quarantine",
        help="Manually release a route quarantine after review",
    )
    clear_quarantine.add_argument("route_id")
    clear_quarantine.add_argument("--reason", default="")
    clear_quarantine.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    clear_quarantine.set_defaults(func=cmd_clear_route_quarantine)

    delivery_supervisor = sub.add_parser(
        "delivery-supervisor",
        help="Run paper/demo delivery supervision cycles with risk-first stop conditions",
    )
    delivery_supervisor.add_argument("--cycles", type=int, default=1)
    delivery_supervisor.add_argument("--training-rounds", type=int, default=10)
    delivery_supervisor.add_argument("--symbols", default="")
    delivery_supervisor.add_argument("--mission-symbols-per-cycle", type=int, default=6)
    delivery_supervisor.add_argument("--target-return-pct", type=float, default=5.0)
    delivery_supervisor.add_argument("--max-leverage", type=float, default=3.0)
    delivery_supervisor.add_argument("--margin-notional-usdt", type=float, default=3.0)
    delivery_supervisor.add_argument("--optimize-every", type=int, default=10)
    delivery_supervisor.add_argument("--max-recent-loss-usdt", type=float, default=5.0)
    delivery_supervisor.add_argument("--max-route-loss-streak", type=int, default=5)
    delivery_supervisor.add_argument("--min-route-profit-factor", type=float, default=0.8)
    delivery_supervisor.add_argument("--route-lookback", type=int, default=40)
    delivery_supervisor.add_argument("--build-digest-every", type=int, default=1)
    delivery_supervisor.add_argument("--audit-every", type=int, default=1)
    delivery_supervisor.add_argument("--no-stop-on-optimizer-promotion", action="store_true")
    delivery_supervisor.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    delivery_supervisor.set_defaults(func=cmd_delivery_supervisor)

    backfill_live = sub.add_parser(
        "backfill-live-journal",
        help="Backfill route/cohort metadata into historical live-order journal entries",
    )
    backfill_live.add_argument("--dry-run", action="store_true")
    backfill_live.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    backfill_live.set_defaults(func=cmd_backfill_live_journal)

    backfill_reviews = sub.add_parser(
        "backfill-closed-reviews",
        help="Backfill route/cohort metadata into historical closed-trade reviews",
    )
    backfill_reviews.add_argument("--dry-run", action="store_true")
    backfill_reviews.add_argument("--compact", action="store_true", help="Minimal output for AI agents")
    backfill_reviews.set_defaults(func=cmd_backfill_closed_reviews)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
