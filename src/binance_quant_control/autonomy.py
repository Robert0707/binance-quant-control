from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .analysis import run_analysis
from .asset_routing import DEFAULT_ROUTING_CONFIG_PATH, resolve_symbol_route
from .config import PROJECT_ROOT, STATE_DIR, ensure_runtime_dirs, load_settings
from .convergence import build_cohort_id
from .daily_digest import build_digest
from .daily_digest import load_config as load_digest_config
from .live_execution import build_live_execution_plan, execute_live_order
from .order_journal import PaperOrderRecord, append_paper_order
from .position_manager import (
    build_adaptive_exit_plan,
    build_position_management_plan,
    execute_adaptive_exit_plan,
    execute_position_management_plan,
)
from .professional_entry_gate import ProfessionalGatePolicy, evaluate_professional_entry_gate
from .protective_repair import (
    build_staged_take_profit_repair_plan,
    execute_staged_take_profit_repair,
)
from .signal_scoring import build_signal_scores
from .strategy import StrategyConfig, load_strategy_config
from .trading_control import load_trading_control_state

QUANTCTL = Path("/home/robert/.openclaw/bin/openclaw-quantctl")
HAILO_PROJECT_ROOT = PROJECT_ROOT.parent / "hailo-trading-triage"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "autonomous-trader.default.yaml"
AUTONOMY_STATE_DIR = STATE_DIR / "autonomy"


@dataclass(frozen=True, slots=True)
class AutonomyConfig:
    path: Path
    strategy_config: Path
    routing_config: Path
    digest_config: Path
    hailo_config: Path
    review_limit: int
    execute_live_entries: bool
    execute_testnet_entries: bool
    execute_simulated_entries: bool
    execute_position_protection: bool
    adaptive_exit_enabled: bool
    force_simulation_after_analysis: bool
    require_digest_action: str
    require_strategy_analyzer_approval: bool
    min_strategy_analyzer_confidence: float
    max_managed_positions: int
    allow_new_entries_with_open_positions: bool
    candidate_symbol_source: str
    execution_mode: str
    require_alpha_promotion: bool
    alpha_research_report: Path | None
    trailing_callback_pct: float | None
    adaptive_exit_min_profit_r: float
    adaptive_exit_max_loss_r: float
    adaptive_exit_min_reversal_score: float
    adaptive_exit_min_confidence: float
    margin_notional_usdt: float | None
    simulation_notional_usdt: float | None
    min_expected_profit_usdt: float
    require_professional_entry_gate: bool
    min_reward_risk: float
    min_net_profit_to_risk: float
    max_fee_profit_ratio: float
    max_slippage_profit_ratio: float
    max_volatility: float
    min_volume_zscore: float
    min_recent_reviews: int
    min_recent_win_rate: float
    min_recent_avg_r: float
    max_recent_stop_loss_ratio: float
    recent_lookback: int
    stop_loss_cooldown_hours: float
    hailo_enabled: bool
    digest_enabled: bool
    reuse_cached_digest: bool
    digest_min_interval_minutes: int
    skip_digest_when_positions_open: bool
    review_enabled: bool
    auto_pause_enabled: bool


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _run_json_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 900,
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
        check=False,
    )
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    payload: dict[str, Any] | None = None
    if stdout:
        try:
            raw = json.loads(stdout)
            if isinstance(raw, dict):
                payload = raw
        except json.JSONDecodeError:
            payload = None
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": stdout[-8000:],
        "stderr": stderr[-8000:],
        "response": payload,
    }


def _resolve_config_path(base: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        if not candidate.exists() and candidate.name.endswith(".auto.yaml"):
            fallback = candidate.with_name(candidate.name.replace(".auto.yaml", ".yaml"))
            if fallback.exists():
                return fallback.resolve()
        return candidate
    if candidate.exists():
        return candidate.resolve()
    resolved = (base / candidate).resolve()
    if not resolved.exists() and resolved.name.endswith(".auto.yaml"):
        fallback = resolved.with_name(resolved.name.replace(".auto.yaml", ".yaml"))
        if fallback.exists():
            return fallback.resolve()
    return resolved


def load_autonomy_config(path: str | Path | None = None) -> AutonomyConfig:
    config_path = Path(path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Autonomy config must be a mapping: {config_path}")

    base = config_path.parent
    strategy_cfg = payload.get("strategy") or {}
    automation_cfg = payload.get("automation") or {}
    integration_cfg = payload.get("integrations") or {}
    risk_cfg = payload.get("risk") or {}

    return AutonomyConfig(
        path=config_path,
        strategy_config=_resolve_config_path(
            base,
            str(strategy_cfg.get("strategy_config") or "strategy-hermes-pro.yaml"),
        ),
        routing_config=_resolve_config_path(
            base,
            str(strategy_cfg.get("routing_config") or DEFAULT_ROUTING_CONFIG_PATH.name),
        ),
        digest_config=_resolve_config_path(
            base,
            str(integration_cfg.get("digest_config") or "n8n-daily-digest.default.json"),
        ),
        hailo_config=_resolve_config_path(
            HAILO_PROJECT_ROOT,
            str(integration_cfg.get("hailo_config") or "config.yaml"),
        ),
        review_limit=int(automation_cfg.get("review_limit") or 20),
        execute_live_entries=bool(automation_cfg.get("execute_live_entries", False)),
        execute_testnet_entries=bool(automation_cfg.get("execute_testnet_entries", False)),
        execute_simulated_entries=bool(automation_cfg.get("execute_simulated_entries", False)),
        execute_position_protection=bool(automation_cfg.get("execute_position_protection", False)),
        adaptive_exit_enabled=bool(automation_cfg.get("adaptive_exit_enabled", False)),
        force_simulation_after_analysis=bool(automation_cfg.get("force_simulation_after_analysis", True)),
        require_digest_action=str(automation_cfg.get("require_digest_action") or "pre_trade_notify"),
        require_strategy_analyzer_approval=bool(
            automation_cfg.get("require_strategy_analyzer_approval", True)
        ),
        min_strategy_analyzer_confidence=float(
            automation_cfg.get("min_strategy_analyzer_confidence") or 0.62
        ),
        max_managed_positions=int(automation_cfg.get("max_managed_positions") or 3),
        allow_new_entries_with_open_positions=bool(
            automation_cfg.get("allow_new_entries_with_open_positions", False)
        ),
        candidate_symbol_source=str(automation_cfg.get("candidate_symbol_source") or "digest_selected"),
        execution_mode=str(automation_cfg.get("execution_mode") or "live"),
        require_alpha_promotion=bool(automation_cfg.get("require_alpha_promotion", False)),
        alpha_research_report=(
            _resolve_config_path(base, str(automation_cfg.get("alpha_research_report")))
            if automation_cfg.get("alpha_research_report")
            else None
        ),
        trailing_callback_pct=(
            float(risk_cfg["trailing_callback_pct"])
            if risk_cfg.get("trailing_callback_pct") is not None
            else None
        ),
        adaptive_exit_min_profit_r=float(risk_cfg.get("adaptive_exit_min_profit_r") or 0.35),
        adaptive_exit_max_loss_r=float(risk_cfg.get("adaptive_exit_max_loss_r") or -0.35),
        adaptive_exit_min_reversal_score=float(risk_cfg.get("adaptive_exit_min_reversal_score") or 5.0),
        adaptive_exit_min_confidence=float(risk_cfg.get("adaptive_exit_min_confidence") or 0.62),
        margin_notional_usdt=(
            float(risk_cfg["margin_notional_usdt"])
            if risk_cfg.get("margin_notional_usdt") is not None
            else None
        ),
        simulation_notional_usdt=(
            float(risk_cfg["simulation_notional_usdt"])
            if risk_cfg.get("simulation_notional_usdt") is not None
            else None
        ),
        min_expected_profit_usdt=float(risk_cfg.get("min_expected_profit_usdt") or 0.0),
        require_professional_entry_gate=bool(risk_cfg.get("require_professional_entry_gate", True)),
        min_reward_risk=float(risk_cfg.get("min_reward_risk") or 1.2),
        min_net_profit_to_risk=float(risk_cfg.get("min_net_profit_to_risk") or 0.8),
        max_fee_profit_ratio=float(risk_cfg.get("max_fee_profit_ratio") or 0.35),
        max_slippage_profit_ratio=float(risk_cfg.get("max_slippage_profit_ratio") or 0.25),
        max_volatility=float(risk_cfg.get("max_volatility") or 1.8),
        min_volume_zscore=float(risk_cfg.get("min_volume_zscore") or -0.8),
        min_recent_reviews=int(risk_cfg.get("min_recent_reviews") or 6),
        min_recent_win_rate=float(risk_cfg.get("min_recent_win_rate") or 0.42),
        min_recent_avg_r=float(risk_cfg.get("min_recent_avg_r") or 0.0),
        max_recent_stop_loss_ratio=float(risk_cfg.get("max_recent_stop_loss_ratio") or 0.55),
        recent_lookback=int(risk_cfg.get("recent_lookback") or 20),
        stop_loss_cooldown_hours=float(risk_cfg.get("stop_loss_cooldown_hours") or 6.0),
        hailo_enabled=bool(integration_cfg.get("hailo_enabled", True)),
        digest_enabled=bool(integration_cfg.get("digest_enabled", True)),
        reuse_cached_digest=bool(integration_cfg.get("reuse_cached_digest", True)),
        digest_min_interval_minutes=int(integration_cfg.get("digest_min_interval_minutes") or 240),
        skip_digest_when_positions_open=bool(
            integration_cfg.get("skip_digest_when_positions_open", True)
        ),
        review_enabled=bool(automation_cfg.get("review_closed_trades", True)),
        auto_pause_enabled=bool(automation_cfg.get("auto_pause", True)),
    )


def determine_candidate_symbol(
    config: AutonomyConfig,
    strategy: StrategyConfig,
    digest_payload: dict[str, Any] | None,
) -> str:
    if (
        config.candidate_symbol_source == "digest_selected"
        and digest_payload
        and isinstance(digest_payload.get("decision"), dict)
        and isinstance((digest_payload["decision"].get("selected") or {}), dict)
        and digest_payload["decision"]["selected"].get("symbol")
    ):
        return str(digest_payload["decision"]["selected"]["symbol"]).upper()
    return strategy.defaults.symbol.upper()


def determine_candidate_side(digest_payload: dict[str, Any] | None) -> str | None:
    if not digest_payload:
        return None
    selected = (digest_payload.get("decision") or {}).get("selected") or {}
    direction = str(selected.get("direction") or "").lower()
    if direction == "long":
        return "BUY"
    if direction == "short":
        return "SELL"
    return None


def testnet_execution_settings(settings: Any) -> Any:
    return replace(settings, use_testnet=True, live_trading_enabled=False, testnet_trading_enabled=True)


def position_protection_settings(config: AutonomyConfig, settings: Any) -> Any:
    if config.execution_mode == "testnet_exploration":
        return testnet_execution_settings(settings)
    return settings


def load_recent_digest(*, min_interval_minutes: int) -> dict[str, Any] | None:
    digest_dir = STATE_DIR / "n8n-digests"
    if not digest_dir.exists():
        return None
    candidates = sorted(
        digest_dir.glob("*-daily-digest.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    latest = candidates[0]
    age_seconds = _utc_now().timestamp() - latest.stat().st_mtime
    if age_seconds > (min_interval_minutes * 60):
        return None
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    strategy_analysis = payload.get("strategy_analysis") or {}
    if strategy_analysis and not bool(strategy_analysis.get("available", False)):
        return None
    return payload


def evaluate_entry_gate(
    config: AutonomyConfig,
    *,
    digest_payload: dict[str, Any] | None,
    live_plan: dict[str, Any],
    latest: dict[str, Any] | None = None,
    positions_count: int,
    trading_paused: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    decision = (digest_payload or {}).get("decision") or {}
    selected = decision.get("selected") or {}
    action = str(decision.get("action") or "")
    strategy_analysis = (digest_payload or {}).get("strategy_analysis") or {}
    strategy_result = strategy_analysis.get("result") or {}
    nested_result = strategy_result.get("result") or {}
    verdict = str(nested_result.get("verdict") or "")
    confidence = float(nested_result.get("confidence") or 0.0)

    if positions_count >= config.max_managed_positions:
        reasons.append(
            f"Max managed positions reached ({positions_count}/{config.max_managed_positions}); "
            "autonomy stays in manage-position mode."
        )
    elif positions_count > 0 and not config.allow_new_entries_with_open_positions:
        reasons.append("Open positions already exist; autonomy stays in manage-position mode.")
    if trading_paused:
        reasons.append("Trading is paused by kill-switch or auto-pause policy.")
    if config.digest_enabled:
        if action != config.require_digest_action:
            reasons.append(
                f"Digest action {action or 'none'} did not meet required action {config.require_digest_action}."
            )
        if not selected.get("symbol"):
            reasons.append("Digest did not select a concrete candidate symbol.")
    if config.require_strategy_analyzer_approval:
        if not strategy_analysis.get("available"):
            reasons.append("Strategy analyzer is unavailable, so cloud approval is missing.")
        elif verdict != "approve":
            reasons.append(f"Strategy analyzer verdict is {verdict or 'unknown'}, not approve.")
        elif confidence < config.min_strategy_analyzer_confidence:
            reasons.append(
                "Strategy analyzer confidence "
                f"{confidence:.3f} is below the minimum {config.min_strategy_analyzer_confidence:.3f}."
            )
    if not bool(live_plan.get("allowed", False)):
        reasons.extend(str(item) for item in (live_plan.get("violations") or []))

    price = float(live_plan.get("price") or 0.0)
    quantity = float(live_plan.get("quantity") or 0.0)
    take_profit_price = float(live_plan.get("take_profit_price") or 0.0)
    side = str(live_plan.get("side") or "")
    fee_bps = 4.0
    gross_target_profit = 0.0
    if price > 0 and quantity > 0 and take_profit_price > 0:
        if side == "BUY":
            gross_target_profit = max(0.0, (take_profit_price - price) * quantity)
        elif side == "SELL":
            gross_target_profit = max(0.0, (price - take_profit_price) * quantity)
    estimated_fees = (price * quantity) * (fee_bps / 10000.0) * 2.0 if price > 0 and quantity > 0 else 0.0
    expected_profit_after_fees = gross_target_profit - estimated_fees
    if config.min_expected_profit_usdt > 0 and expected_profit_after_fees < config.min_expected_profit_usdt:
        reasons.append(
            f"Expected TP1 net profit {expected_profit_after_fees:.4f} USDT is below the configured floor "
            f"{config.min_expected_profit_usdt:.4f} USDT."
        )

    professional_gate = evaluate_professional_entry_gate(
        side=side,
        latest=latest or {},
        live_plan={
            **live_plan,
            "fee_bps": live_plan.get("fee_bps", 4.0),
            "slippage_bps": live_plan.get("slippage_bps", 2.0),
        },
        policy=ProfessionalGatePolicy(
            min_reward_risk=config.min_reward_risk,
            min_net_profit_to_risk=config.min_net_profit_to_risk,
            max_fee_profit_ratio=config.max_fee_profit_ratio,
            max_slippage_profit_ratio=config.max_slippage_profit_ratio,
            max_volatility=config.max_volatility,
            min_volume_zscore=config.min_volume_zscore,
            min_recent_reviews=config.min_recent_reviews,
            min_recent_win_rate=config.min_recent_win_rate,
            min_recent_avg_r=config.min_recent_avg_r,
            max_recent_stop_loss_ratio=config.max_recent_stop_loss_ratio,
            recent_lookback=config.recent_lookback,
            stop_loss_cooldown_hours=config.stop_loss_cooldown_hours,
            require_professional_gate=config.require_professional_entry_gate,
        ),
    )
    if config.require_professional_entry_gate and not professional_gate.passed:
        reasons.extend(professional_gate.violations)

    return {
        "eligible": len(reasons) == 0,
        "reasons": reasons,
        "digest_action": action,
        "strategy_analyzer_verdict": verdict or None,
        "strategy_analyzer_confidence": round(confidence, 3),
        "expected_profit_after_fees_usdt": round(expected_profit_after_fees, 6),
        "professional_gate": professional_gate.to_dict(),
    }


def alpha_promotion_gate(
    config: AutonomyConfig,
    *,
    symbol: str,
    interval: str,
    strategy_family: str,
) -> dict[str, Any]:
    if not config.require_alpha_promotion:
        return {
            "required": False,
            "allowed": True,
            "reasons": [],
        }
    report_path = config.alpha_research_report
    if report_path is None:
        return {
            "required": True,
            "allowed": False,
            "reasons": ["alpha-promotion-report-not-configured"],
        }
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "required": True,
            "allowed": False,
            "report_path": str(report_path),
            "reasons": [f"alpha-promotion-report-unavailable:{exc}"],
        }
    cohort_id = f"{symbol.upper()}:{interval}:{strategy_family}"
    rows = payload.get("rows") or []
    row = next((item for item in rows if str(item.get("cohort_id") or "") == cohort_id), None)
    if row is None:
        return {
            "required": True,
            "allowed": False,
            "report_path": str(report_path),
            "cohort_id": cohort_id,
            "reasons": ["alpha-cohort-not-found"],
        }
    if bool(row.get("promotion_eligible", False)):
        return {
            "required": True,
            "allowed": True,
            "report_path": str(report_path),
            "cohort_id": cohort_id,
            "row": row,
            "reasons": [],
        }
    reasons = [
        "alpha-cohort-not-promotion-eligible",
        f"trade_count={row.get('trade_count')}",
        f"win_rate={row.get('win_rate')}",
        f"stop_loss_ratio={row.get('stop_loss_ratio')}",
        f"profit_factor={row.get('profit_factor')}",
        f"robustness_status={row.get('robustness_status')}",
    ]
    return {
        "required": True,
        "allowed": False,
        "report_path": str(report_path),
        "cohort_id": cohort_id,
        "row": row,
        "reasons": reasons,
    }


def should_execute_testnet_entry(
    config: AutonomyConfig,
    *,
    gate: dict[str, Any],
    digest_payload: dict[str, Any] | None,
    live_plan: dict[str, Any],
    candidate_side: str | None,
    trading_paused: bool,
    alpha_gate: dict[str, Any] | None = None,
) -> bool:
    return bool(
        config.execute_testnet_entries
        and config.execution_mode == "testnet_exploration"
        and bool(gate.get("eligible", False))
        and bool((alpha_gate or {"allowed": True}).get("allowed", True))
        and live_plan.get("allowed")
        and candidate_side
        and (
            not config.digest_enabled
            or str(((digest_payload or {}).get("decision") or {}).get("action") or "")
            == config.require_digest_action
        )
        and not trading_paused
    )


def _order_type(order: dict[str, Any]) -> str:
    return str(order.get("orderType", order.get("type", ""))).upper()


def _order_quantity(order: dict[str, Any]) -> float:
    value = order.get("quantity", order.get("origQty"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def protective_repair_reasons(plan: Any, strategy: StrategyConfig) -> list[str]:
    """Return why an open position needs stop/TP repair.

    This intentionally looks at exchange-side orders, not journal intent. A
    customer only cares whether the current position is actually protected.
    """

    quantity = float(getattr(plan, "quantity", 0.0) or 0.0)
    if quantity <= 0:
        return []
    algo_orders = list(getattr(plan, "existing_algo_orders", []) or [])
    stop_orders = [item for item in algo_orders if _order_type(item) in {"STOP", "STOP_MARKET"}]
    take_profit_orders = [
        item
        for item in algo_orders
        if _order_type(item) in {"TAKE_PROFIT", "TAKE_PROFIT_MARKET"}
    ]
    reasons: list[str] = []
    if not stop_orders:
        reasons.append("missing-stop-loss")

    configured_tp_levels = len(strategy.risk.take_profit_r_multiples or [])
    min_staged_tp_levels = min(2, configured_tp_levels) if configured_tp_levels > 1 else configured_tp_levels
    if configured_tp_levels and len(take_profit_orders) < min_staged_tp_levels:
        step_size = float(getattr(plan, "step_size", 0.0) or 0.0)
        micro_full_tp_fallback = (
            step_size > 0
            and quantity <= step_size * 1.01
            and len(take_profit_orders) == 1
            and _order_quantity(take_profit_orders[0]) >= quantity * 0.99
        )
        if not micro_full_tp_fallback:
            reasons.append("missing-staged-take-profits")

    take_profit_quantities = [_order_quantity(item) for item in take_profit_orders]
    largest_take_profit = max(take_profit_quantities or [0.0])
    total_take_profit = sum(take_profit_quantities)
    step_size = float(getattr(plan, "step_size", 0.0) or 0.0)
    micro_full_tp_fallback = (
        step_size > 0
        and quantity <= step_size * 1.01
        and len(take_profit_orders) == 1
        and largest_take_profit >= quantity * 0.99
    )
    if (
        len(take_profit_orders) <= 1
        and largest_take_profit >= quantity * 0.85
        and not micro_full_tp_fallback
    ):
        reasons.append("single-full-position-take-profit")
    if total_take_profit > quantity * 1.05:
        reasons.append("oversized-take-profit-ladder")
    return reasons


def _build_position_actions(
    config: AutonomyConfig,
    strategy: StrategyConfig,
) -> list[dict[str, Any]]:
    position_scan = _run_json_command([str(QUANTCTL), "positions", "--compact"], cwd=PROJECT_ROOT, timeout=300)
    positions = ((position_scan.get("response") or {}).get("positions") or [])[: config.max_managed_positions]
    if not isinstance(positions, list):
        return []

    settings = position_protection_settings(config, load_settings())
    plans: list[dict[str, Any]] = []
    for item in positions:
        symbol = str(item.get("symbol") or "").upper()
        if not symbol:
            continue
        callback_pct = config.trailing_callback_pct or strategy.risk.trailing_callback_pct
        route = resolve_symbol_route(symbol, config.routing_config)
        routed_strategy = load_strategy_config(route.strategy_config)
        routed_strategy = replace(
            routed_strategy,
            defaults=replace(
                routed_strategy.defaults,
                symbol=symbol,
                market=route.market,
                interval=route.interval,
            ),
        )
        plan = build_position_management_plan(
            settings,
            symbol=symbol,
            market=routed_strategy.defaults.market,
            strategy=routed_strategy,
            enable_trailing_stop=bool(routed_strategy.risk.trailing_stop_enabled),
            trailing_callback_pct=callback_pct if callback_pct > 0 else None,
        )
        repair_reasons = protective_repair_reasons(plan, routed_strategy)
        result: dict[str, Any] = {
            "symbol": symbol,
            "route": route.to_dict(),
            "plan": plan.to_dict(),
            "protective_repair_reasons": repair_reasons,
            "executed": False,
        }
        analysis_payload: dict[str, Any] | None = None
        if config.adaptive_exit_enabled and plan.allowed:
            analysis_payload, _ = run_analysis(
                settings,
                symbol=symbol,
                market=routed_strategy.defaults.market,
                interval=routed_strategy.defaults.interval,
                limit=max(routed_strategy.defaults.limit, 240),
                use_blave=routed_strategy.defaults.use_blave,
                render_chart_flag=False,
                strategy=routed_strategy,
            )
            adaptive_plan = build_adaptive_exit_plan(
                plan,
                analysis_payload,
                min_profit_r_for_reversal_exit=config.adaptive_exit_min_profit_r,
                max_loss_r_for_reversal_exit=config.adaptive_exit_max_loss_r,
                min_reversal_score=config.adaptive_exit_min_reversal_score,
                min_confidence=config.adaptive_exit_min_confidence,
            )
            result["adaptive_exit_plan"] = adaptive_plan.to_dict()
            if config.execute_position_protection and adaptive_plan.allowed:
                result["adaptive_exit_execution"] = execute_adaptive_exit_plan(settings, adaptive_plan)
                result["executed"] = True
                plans.append(result)
                continue
        if config.execute_position_protection and plan.allowed and repair_reasons:
            if analysis_payload is None:
                analysis_payload, _ = run_analysis(
                    settings,
                    symbol=symbol,
                    market=routed_strategy.defaults.market,
                    interval=routed_strategy.defaults.interval,
                    limit=max(routed_strategy.defaults.limit, 240),
                    use_blave=routed_strategy.defaults.use_blave,
                    render_chart_flag=False,
                    strategy=routed_strategy,
                )
            side_plan = analysis_payload["trade_plan"]["long" if plan.side == "BUY" else "short"]
            repair_plan = build_staged_take_profit_repair_plan(
                settings,
                routed_strategy,
                symbol=symbol,
                side_plan=side_plan,
                confidence=float((analysis_payload.get("analysis") or {}).get("convergence") or 0.75),
                route_id=route.route_id,
            )
            result["repair_plan"] = repair_plan.to_dict()
            if repair_plan.allowed:
                result["repair_execution"] = execute_staged_take_profit_repair(settings, routed_strategy, repair_plan)
                result["executed"] = True
        elif config.execute_position_protection and plan.allowed and plan.actions:
            result["execution"] = execute_position_management_plan(settings, plan)
            result["executed"] = True
        plans.append(result)
    return plans


def _extract_open_position_symbols(positions_response: dict[str, Any]) -> set[str]:
    positions = positions_response.get("positions") or []
    if not isinstance(positions, list):
        return set()
    symbols: set[str] = set()
    for item in positions:
        symbol = str((item or {}).get("symbol") or "").upper()
        if symbol:
            symbols.add(symbol)
    return symbols


def run_autonomy_cycle(path: str | Path | None = None) -> dict[str, Any]:
    config = load_autonomy_config(path)
    strategy = load_strategy_config(config.strategy_config)
    settings = load_settings()
    ensure_runtime_dirs()
    AUTONOMY_STATE_DIR.mkdir(parents=True, exist_ok=True)

    generated_at = _utc_now()
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    summary: dict[str, Any] = {
        "generated_at": generated_at.isoformat(),
        "config_path": str(config.path),
        "strategy_config": str(config.strategy_config),
        "routing_config": str(config.routing_config),
        "execute_live_entries": config.execute_live_entries,
        "execute_testnet_entries": config.execute_testnet_entries,
        "execute_simulated_entries": config.execute_simulated_entries,
        "execute_position_protection": config.execute_position_protection,
        "execution_mode": config.execution_mode,
        "steps": {},
    }

    if config.review_enabled:
        summary["steps"]["review_closed_trades"] = _run_json_command(
            [
                str(QUANTCTL),
                "review-closed-trades",
                "--limit",
                str(config.review_limit),
                "--compact",
            ],
            cwd=PROJECT_ROOT,
        )

    if config.auto_pause_enabled:
        summary["steps"]["auto_pause_trading"] = _run_json_command(
            [
                str(QUANTCTL),
                "auto-pause-trading",
                "--strategy-config",
                str(config.strategy_config),
                "--compact",
            ],
            cwd=PROJECT_ROOT,
        )

    summary["steps"]["positions"] = _run_json_command(
        [str(QUANTCTL), "positions", "--compact"],
        cwd=PROJECT_ROOT,
        timeout=300,
    )
    summary["steps"]["account"] = _run_json_command(
        [str(QUANTCTL), "account", "--market", strategy.defaults.market, "--compact"],
        cwd=PROJECT_ROOT,
        timeout=300,
    )
    summary["steps"]["journal_summary"] = _run_json_command(
        [str(QUANTCTL), "journal-summary"],
        cwd=PROJECT_ROOT,
        timeout=300,
    )

    positions_response = (summary["steps"]["positions"].get("response") or {}) if isinstance(
        summary["steps"]["positions"], dict
    ) else {}
    positions_count = int(positions_response.get("count") or 0)
    open_position_symbols = _extract_open_position_symbols(positions_response)
    entry_slot_available = positions_count == 0 or (
        config.allow_new_entries_with_open_positions and positions_count < config.max_managed_positions
    )
    trading_control = load_trading_control_state()
    summary["trading_control"] = trading_control.to_dict()

    if config.hailo_enabled:
        summary["steps"]["hailo_triage"] = _run_json_command(
            [
                sys.executable,
                str(HAILO_PROJECT_ROOT / "src" / "main.py"),
                "--config",
                str(config.hailo_config),
                "--once",
            ],
            cwd=HAILO_PROJECT_ROOT,
            timeout=300,
        )

    digest_payload: dict[str, Any] | None = None
    digest_skipped = positions_count > 0 and config.skip_digest_when_positions_open and not entry_slot_available
    reused_cached_digest = False
    if config.digest_enabled and not digest_skipped:
        if config.reuse_cached_digest:
            digest_payload = load_recent_digest(
                min_interval_minutes=config.digest_min_interval_minutes
            )
            reused_cached_digest = digest_payload is not None
        if digest_payload is None:
            digest_config = load_digest_config(config.digest_config)
            if open_position_symbols:
                configured_exclusions = {
                    str(symbol).upper() for symbol in digest_config.get("exclude_symbols") or []
                }
                digest_config["exclude_symbols"] = sorted(configured_exclusions | open_position_symbols)
            digest_payload = build_digest(digest_config)
        summary["digest"] = {
            "output_path": digest_payload.get("output_path"),
            "decision": digest_payload.get("decision"),
            "strategy_analysis": digest_payload.get("strategy_analysis"),
            "reused_cache": reused_cached_digest,
        }
    elif digest_skipped:
        summary["digest"] = {
            "status": "skipped",
            "reason": "positions-open-local-guardian-priority",
        }

    if positions_count > 0:
        management = _build_position_actions(config, strategy)
        summary["position_management"] = management

    if not entry_slot_available:
        summary["autonomy_mode"] = "manage-open-positions"
        summary["status"] = "ok"
    else:
        candidate_symbol = determine_candidate_symbol(config, strategy, digest_payload)
        if candidate_symbol in open_position_symbols:
            summary["autonomy_mode"] = "manage-open-positions"
            summary["entry_gate"] = {
                "eligible": False,
                "reasons": [f"{candidate_symbol} already has an open position; duplicate entries are blocked."],
            }
            summary["status"] = "ok"
            report_path = AUTONOMY_STATE_DIR / f"{stamp}-autonomous-cycle.json"
            report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            summary["report_path"] = str(report_path)
            return summary

        candidate_side = determine_candidate_side(digest_payload)
        route = resolve_symbol_route(candidate_symbol, config.routing_config)
        routed_strategy = load_strategy_config(route.strategy_config)
        routed_strategy = replace(
            routed_strategy,
            defaults=replace(
                routed_strategy.defaults,
                symbol=candidate_symbol,
                market=route.market,
                interval=route.interval,
            ),
        )
        analysis_payload, artifacts = run_analysis(
            settings,
            symbol=candidate_symbol,
            market=routed_strategy.defaults.market,
            interval=routed_strategy.defaults.interval,
            limit=max(routed_strategy.defaults.limit, 240),
            use_blave=routed_strategy.defaults.use_blave,
            render_chart_flag=routed_strategy.defaults.render_chart,
            strategy=routed_strategy,
        )
        live_plan = build_live_execution_plan(
            settings,
            routed_strategy,
            analysis_payload,
            side_override=candidate_side,
            margin_notional_usdt=config.margin_notional_usdt,
            execution_mode=config.execution_mode,
            news_risk=((digest_payload or {}).get("news") or {}).get("risk"),
        ).to_dict()
        gate = evaluate_entry_gate(
            config,
            digest_payload=digest_payload,
            live_plan=live_plan,
            latest=analysis_payload.get("latest") or {},
            positions_count=positions_count,
            trading_paused=trading_control.paused,
        )
        selected_family = (analysis_payload.get("analysis") or {}).get("selected_strategy_family") or {}
        alpha_gate = alpha_promotion_gate(
            config,
            symbol=candidate_symbol,
            interval=routed_strategy.defaults.interval,
            strategy_family=str(selected_family.get("family") or "mean_reversion"),
        )
        if alpha_gate.get("required") and not alpha_gate.get("allowed"):
            gate = {
                **gate,
                "eligible": False,
                "reasons": [*list(gate.get("reasons") or []), *list(alpha_gate.get("reasons") or [])],
            }
        summary["autonomy_mode"] = "entry-evaluation"
        summary["candidate"] = {
            "symbol": candidate_symbol,
            "side_override": candidate_side,
            "route": route.to_dict(),
            "strategy_profile": routed_strategy.profile,
            "strategy_path": str(routed_strategy.path),
            "analysis_report": analysis_payload["artifacts"]["report_json"],
            "chart_path": analysis_payload["artifacts"]["chart_path"],
            "run_output_dir": str(artifacts.output_dir),
        }
        summary["live_plan"] = live_plan
        summary["entry_gate"] = gate
        summary["alpha_promotion_gate"] = alpha_gate
        should_force_simulate = bool(config.force_simulation_after_analysis and candidate_side)
        should_execute_testnet = should_execute_testnet_entry(
            config,
            gate=gate,
            digest_payload=digest_payload,
            live_plan=live_plan,
            candidate_side=candidate_side,
            trading_paused=trading_control.paused,
            alpha_gate=alpha_gate,
        )
        if (gate["eligible"] and config.execute_live_entries) or should_execute_testnet:
            execution_settings = testnet_execution_settings(settings) if should_execute_testnet else settings
            news_risk = ((digest_payload or {}).get("news") or {}).get("risk")
            summary["execution"] = execute_live_order(
                execution_settings,
                routed_strategy,
                build_live_execution_plan(
                    execution_settings,
                    routed_strategy,
                    analysis_payload,
                    side_override=candidate_side,
                    margin_notional_usdt=config.margin_notional_usdt,
                    execution_mode=config.execution_mode,
                    news_risk=news_risk,
                ),
                entry_reason_snapshot={
                    "bias": str((analysis_payload.get("analysis") or {}).get("bias") or ""),
                    "score": int((analysis_payload.get("analysis") or {}).get("score") or 0),
                    "convergence": float((analysis_payload.get("analysis") or {}).get("convergence") or 0.0),
                    "interval": routed_strategy.defaults.interval,
                },
                signal_scores=build_signal_scores(
                    route=route,
                    latest=analysis_payload.get("latest") or {},
                    analysis=analysis_payload.get("analysis") or {},
                    trade_plan=analysis_payload.get("trade_plan") or {},
                    news_risk=news_risk,
                    side=(candidate_side or str(live_plan.get("side") or "BUY")).upper(),
                ),
            )
            summary["execution"]["mode"] = "testnet" if should_execute_testnet else "live"
            summary["status"] = "testnet_executed" if should_execute_testnet else "executed"
        elif (gate["eligible"] and config.execute_simulated_entries) or should_force_simulate:
            leverage = float(routed_strategy.risk.default_leverage)
            notional_usdt = (
                config.simulation_notional_usdt
                or routed_strategy.execution.margin_notional_usdt
                or config.margin_notional_usdt
                or 3.0
            )
            price = float(live_plan.get("price") or 0.0)
            gross_notional = float(notional_usdt) * leverage
            quantity = gross_notional / price if price > 0 else 0.0
            signal_scores = build_signal_scores(
                route=route,
                latest=analysis_payload.get("latest") or {},
                analysis=analysis_payload.get("analysis") or {},
                trade_plan=analysis_payload.get("trade_plan") or {},
                news_risk=news_risk,
                side=(candidate_side or str(live_plan.get("side") or "BUY")).upper(),
            )
            record = PaperOrderRecord(
                generated_at=generated_at.isoformat(),
                kind="paper-order",
                symbol=candidate_symbol,
                market=routed_strategy.defaults.market,
                side=(candidate_side or str(live_plan.get("side") or "BUY")).upper(),
                margin_notional_usdt=round(float(notional_usdt), 6),
                leverage=leverage,
                gross_notional_usdt=round(gross_notional, 6),
                reference_price=round(price, 8),
                estimated_quantity=round(quantity, 8),
                analysis_bias=str((analysis_payload.get("analysis") or {}).get("bias") or ""),
                analysis_score=int((analysis_payload.get("analysis") or {}).get("score") or 0),
                analysis_convergence=float((analysis_payload.get("analysis") or {}).get("convergence") or 0.0),
                cohort_id=build_cohort_id(
                    asset_class=route.asset_class,
                    strategy_profile=routed_strategy.profile,
                    market=routed_strategy.defaults.market,
                    interval=routed_strategy.defaults.interval,
                ),
                strategy_profile=routed_strategy.profile,
                strategy_path=str(routed_strategy.path),
                asset_class=route.asset_class,
                route_id=route.route_id,
                simulation_mode=route.simulation_mode,
                review_lane=route.review_lane,
                entry_reason_snapshot={
                    "bias": str((analysis_payload.get("analysis") or {}).get("bias") or ""),
                    "score": int((analysis_payload.get("analysis") or {}).get("score") or 0),
                    "convergence": float((analysis_payload.get("analysis") or {}).get("convergence") or 0.0),
                    "interval": routed_strategy.defaults.interval,
                },
                signal_scores=signal_scores,
                analysis_report=str(analysis_payload["artifacts"]["report_json"]),
                chart_path=analysis_payload["artifacts"]["chart_path"],
                note=(
                    "Forced simulation after analysis. Journal only. No live or testnet order has been sent."
                    if should_force_simulate and not gate["eligible"]
                    else "Autonomy simulation entry. Journal only. No live or testnet order has been sent."
                ),
            )
            journal_path = append_paper_order(record)
            summary["simulation"] = {
                "status": "recorded_for_market_validation" if should_force_simulate and not gate["eligible"] else "recorded",
                "forced_after_analysis": should_force_simulate,
                "journal_path": str(journal_path),
                "paper_order": {
                    **asdict(record),
                    "tags": list(route.tags),
                },
            }
            summary["status"] = "simulated"
        else:
            summary["status"] = "ok" if gate["eligible"] else "blocked"

    report_path = AUTONOMY_STATE_DIR / f"{stamp}-autonomous-cycle.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["report_path"] = str(report_path)
    return summary
