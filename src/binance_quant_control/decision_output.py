from __future__ import annotations

from typing import Any

from .hailo_trading_plan import build_hailo_trading_plan
from .position_manager import AdaptiveExitPlan, PositionManagementPlan
from .strategy import StrategyConfig

Decision = str

ALLOWED_DECISIONS = {"BUY", "LONG", "SELL", "SHORT", "HOLD", "EXIT"}
ENTRY_DECISIONS = {"BUY", "LONG", "SELL", "SHORT"}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_or_none(value: Any, digits: int = 6) -> float | None:
    if value in (None, ""):
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _direction_from_action(action: str, analysis: dict[str, Any]) -> str:
    normalized = action.upper()
    if normalized in {"BUY", "LONG"}:
        return "bullish"
    if normalized in {"SELL", "SHORT", "EXIT"}:
        return "bearish"
    bias = str(analysis.get("bias") or "").lower()
    if "long" in bias:
        return "bullish"
    if "short" in bias:
        return "bearish"
    return "neutral"


def _canonical_regime(analysis: dict[str, Any], latest: dict[str, Any], market_context: dict[str, Any]) -> str:
    raw = str(analysis.get("regime") or "").lower()
    if raw in {"trend-up", "trend-down", "crowded-long", "crowded-short"}:
        return "trend"
    if raw in {"range", "squeeze"}:
        return "range"

    volatility = _float(latest.get("realized_vol_20"))
    volume_z = _float(latest.get("volume_zscore_20"))
    spread_bps = _float(market_context.get("spread_bps"))
    if spread_bps > 12.0 or volume_z < -1.2:
        return "low_liquidity"
    close = _float(latest.get("close"))
    ema_slow = _float(latest.get("ema_slow"))
    macd_hist = _float(latest.get("macd_hist"))
    if volatility >= 1.8 or volume_z >= 2.5:
        if close > 0 and ema_slow > 0 and close >= ema_slow and macd_hist >= 0:
            return "pump"
        if close > 0 and ema_slow > 0 and close <= ema_slow and macd_hist <= 0:
            return "crash"
        return "abnormal_volatility"
    return raw or "unknown"


def _analysis_bias_supports_action(action: str, analysis: dict[str, Any]) -> bool:
    bias = str(analysis.get("bias") or "").lower()
    if not bias or "neutral" in bias:
        return True
    if action == "BUY":
        return "long" in bias or "bull" in bias
    if action == "SELL":
        return "short" in bias or "bear" in bias
    return True


def _entry_reason(action: str, analysis: dict[str, Any], side_plan: dict[str, Any], *, approved: bool) -> list[str]:
    reasons = []
    if approved or _analysis_bias_supports_action(action, analysis):
        reasons.extend(str(item) for item in (analysis.get("decision_notes") or []) if str(item))
    selected_family = analysis.get("selected_strategy_family") if isinstance(analysis.get("selected_strategy_family"), dict) else {}
    family = selected_family.get("family") or side_plan.get("strategy_family")
    if action in {"BUY", "SELL"}:
        if approved:
            reasons.append(
                f"{action} is approved by score={analysis.get('score')} and convergence={analysis.get('convergence')} after hard gates."
            )
        else:
            reasons.append(
                f"{action} candidate was evaluated with score={analysis.get('score')} and convergence={analysis.get('convergence')}, but hard gates did not approve entry."
            )
        if family:
            reasons.append(f"Active strategy family: {family}.")
    if not reasons:
        reasons.append("No complete entry setup; defaulting to HOLD until gates align.")
    return list(dict.fromkeys(reasons))


def _blocked_reasons(analysis: dict[str, Any], live_plan: dict[str, Any] | None, *, risk_pct: float) -> list[str]:
    reasons = [str(item) for item in (analysis.get("entry_blockers") or []) if str(item)]
    if live_plan:
        reasons.extend(str(item) for item in (live_plan.get("violations") or []) if str(item))
    if risk_pct > 0.025:
        reasons.append(f"Risk {risk_pct:.4%} exceeds the 2.5% per-trade ceiling.")
    return list(dict.fromkeys(reasons))


def _risk_reward(entry: float | None, stop_loss: float | None, take_profit: float | None, side: str) -> float | None:
    if entry in (None, 0.0) or stop_loss in (None, 0.0) or take_profit in (None, 0.0):
        return None
    risk = abs(float(entry) - float(stop_loss))
    reward = (float(take_profit) - float(entry)) if side == "BUY" else (float(entry) - float(take_profit))
    if risk <= 0:
        return None
    return round(max(0.0, reward / risk), 4)


def _candidate_side(
    side: str,
    *,
    entry: float | None,
    stop_loss: float | None,
    first_take_profit: float | None,
) -> str:
    normalized = side.upper()
    if normalized in {"BUY", "SELL"}:
        return normalized
    if entry is None or stop_loss is None or first_take_profit is None:
        return normalized
    if stop_loss < entry < first_take_profit:
        return "BUY"
    if first_take_profit < entry < stop_loss:
        return "SELL"
    return normalized


def _expected_value(analysis: dict[str, Any], live_plan: dict[str, Any] | None) -> dict[str, Any]:
    challenge = (live_plan or {}).get("challenge") if isinstance((live_plan or {}).get("challenge"), dict) else {}
    route_side = challenge.get("route_side_risk") if isinstance(challenge.get("route_side_risk"), dict) else {}
    historical = challenge.get("historical_signal_risk") if isinstance(challenge.get("historical_signal_risk"), dict) else {}
    buckets = historical.get("buckets") if isinstance(historical.get("buckets"), list) else []
    blocked_buckets = [item for item in buckets if isinstance(item, dict) and item.get("blocked")]
    performance = _layer(live_plan, "strategy_performance")
    route_pf = _round_or_none(route_side.get("profit_factor"), 4)
    route_net = _round_or_none(route_side.get("net_pnl_usdt"), 6)
    performance_sample_count = int(_float(performance.get("count"))) if performance else 0
    performance_pf = _round_or_none(performance.get("profit_factor"), 4)
    performance_expectancy = _round_or_none(performance.get("expectancy_r"), 4)
    performance_payoff = _round_or_none(performance.get("payoff_ratio"), 4)
    performance_avg_r = _round_or_none(performance.get("avg_r_multiple"), 4)
    performance_stop_loss_ratio = _round_or_none(performance.get("stop_loss_ratio"), 4)
    performance_passed = bool(performance.get("passed")) if performance else False
    return {
        "score": analysis.get("score"),
        "convergence": analysis.get("convergence"),
        "route_side_profit_factor": route_pf,
        "route_side_net_pnl_usdt": route_net,
        "historical_blocked_bucket_count": len(blocked_buckets),
        "strategy_performance": {
            "passed": performance_passed,
            "sample_count": performance_sample_count,
            "scope": performance.get("scope") if performance else None,
            "profit_factor": performance_pf,
            "expectancy_r": performance_expectancy,
            "payoff_ratio": performance_payoff,
            "avg_r": performance_avg_r,
            "stop_loss_ratio": performance_stop_loss_ratio,
        },
        "positive": bool(
            _float(analysis.get("convergence")) >= 0.6
            and (route_pf is None or route_pf >= 1.0)
            and (route_net is None or route_net >= 0.0)
            and not blocked_buckets
            and (performance_passed if performance else True)
        ),
    }


def _layer(live_plan: dict[str, Any] | None, name: str) -> dict[str, Any]:
    gate = (live_plan or {}).get("professional_entry_gate")
    if not isinstance(gate, dict):
        return {}
    layers = gate.get("layers")
    if not isinstance(layers, dict):
        return {}
    layer = layers.get(name)
    return layer if isinstance(layer, dict) else {}


def _gate_row(name: str, passed: bool, evidence: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "source": source,
        "evidence": evidence,
    }


def _entry_gate_evidence(
    *,
    analysis: dict[str, Any],
    latest: dict[str, Any],
    live_plan: dict[str, Any] | None,
    side: str,
    entry: float | None,
    stop_loss: float | None,
    first_take_profit: float | None,
    risk_reward_ratio: float | None,
    risk_pct: float,
) -> dict[str, Any]:
    side_upper = side.upper()
    mtf = _layer(live_plan, "multi_timeframe_trend")
    regime_policy = _layer(live_plan, "regime_policy")
    market = _layer(live_plan, "market_state")
    signal = _layer(live_plan, "signal_quality")
    execution = _layer(live_plan, "execution_quality")
    expected_bias = "long" if side_upper == "BUY" else "short" if side_upper == "SELL" else "neutral"
    analysis_bias = str(analysis.get("bias") or "").lower()
    trend_passed = bool(mtf.get("passed")) and (
        str(mtf.get("bias") or expected_bias) in {expected_bias, "neutral"}
    )
    momentum_score = _float(analysis.get("convergence"))
    adx_value = _float(signal.get("adx_value") or latest.get("adx"))
    momentum_passed = bool(signal.get("passed")) and momentum_score >= 0.6 and adx_value >= 18.0
    volume_z = _float(market.get("volume_zscore_20") if market else latest.get("volume_zscore_20"))
    obv_z = _float(market.get("obv_zscore_20") if market else latest.get("obv_zscore_20"))
    volume_passed = bool(market.get("passed")) and volume_z >= -0.8 and (
        (side_upper == "BUY" and obv_z >= -0.5)
        or (side_upper == "SELL" and obv_z <= 0.5)
        or side_upper not in {"BUY", "SELL"}
    )
    realized_vol = _float(market.get("realized_vol_20") if market else latest.get("realized_vol_20"))
    spread_bps = _float(market.get("spread_bps") if market else (live_plan or {}).get("spread_bps"))
    volatility_passed = bool(market.get("passed")) and realized_vol <= 1.8 and spread_bps <= 12.0
    structure_valid = False
    if entry is not None and stop_loss is not None and first_take_profit is not None:
        structure_valid = (
            side_upper == "BUY"
            and stop_loss < entry < first_take_profit
        ) or (
            side_upper == "SELL"
            and first_take_profit < entry < stop_loss
        )
    structure_score = _float(signal.get("price_structure_score"))
    support_resistance_passed = bool(signal.get("passed")) and structure_valid and structure_score >= 55.0
    execution_rr = _float(execution.get("reward_risk"), risk_reward_ratio or 0.0)
    risk_reward_passed = bool(execution.get("passed")) and execution_rr >= 1.2 and (risk_reward_ratio or 0.0) >= 1.2
    stop_distance = abs(float(entry) - float(stop_loss)) if entry is not None and stop_loss is not None else 0.0
    stop_distance_passed = stop_distance > 0.0 and risk_pct <= 0.025
    gates = [
        _gate_row(
            "regime_policy",
            bool(regime_policy.get("passed")) if regime_policy else False,
            {
                "regime": regime_policy.get("regime"),
                "strategy_family": regime_policy.get("strategy_family"),
                "allowed_families": regime_policy.get("allowed_families"),
            },
            "professional_entry_gate.regime_policy",
        ),
        _gate_row(
            "trend_direction",
            trend_passed,
            {
                "side": side_upper,
                "expected_bias": expected_bias,
                "analysis_bias": analysis_bias,
                "mtf_bias": mtf.get("bias"),
                "mtf_alignment": mtf.get("alignment"),
                "mtf_confidence": mtf.get("confidence"),
            },
            "professional_entry_gate.multi_timeframe_trend",
        ),
        _gate_row(
            "momentum",
            momentum_passed,
            {
                "analysis_score": analysis.get("score"),
                "analysis_convergence": analysis.get("convergence"),
                "adx_value": adx_value,
                "signal_layer_passed": signal.get("passed"),
            },
            "professional_entry_gate.signal_quality",
        ),
        _gate_row(
            "volume",
            volume_passed,
            {
                "volume_zscore_20": volume_z,
                "obv_zscore_20": obv_z,
                "market_layer_passed": market.get("passed"),
            },
            "professional_entry_gate.market_state",
        ),
        _gate_row(
            "volatility",
            volatility_passed,
            {
                "realized_vol_20": realized_vol,
                "spread_bps": spread_bps,
                "market_layer_passed": market.get("passed"),
            },
            "professional_entry_gate.market_state",
        ),
        _gate_row(
            "support_resistance",
            support_resistance_passed,
            {
                "entry": entry,
                "stop_loss": stop_loss,
                "first_take_profit": first_take_profit,
                "price_structure_score": structure_score,
                "structure_valid_for_side": structure_valid,
            },
            "trade_plan+professional_entry_gate.signal_quality",
        ),
        _gate_row(
            "risk_reward",
            risk_reward_passed,
            {
                "risk_reward_ratio": risk_reward_ratio,
                "professional_reward_risk": execution.get("reward_risk"),
                "tp1_reward_risk": execution.get("tp1_reward_risk"),
                "execution_layer_passed": execution.get("passed"),
            },
            "professional_entry_gate.execution_quality",
        ),
        _gate_row(
            "stop_distance",
            stop_distance_passed,
            {
                "entry": entry,
                "stop_loss": stop_loss,
                "stop_distance": round(stop_distance, 8),
                "risk_pct": round(risk_pct, 6),
                "risk_ceiling_pct": 0.025,
            },
            "live_plan_risk",
        ),
    ]
    failed = [row["name"] for row in gates if not row["passed"]]
    return {
        "all_passed": not failed,
        "failed": failed,
        "gates": gates,
    }


def _lifecycle_gate_evidence(
    *,
    analysis_payload: dict[str, Any],
    live_plan: dict[str, Any] | None,
    expected_value: dict[str, Any],
) -> dict[str, Any]:
    analysis = analysis_payload.get("analysis") if isinstance(analysis_payload.get("analysis"), dict) else {}
    latest = analysis_payload.get("latest") if isinstance(analysis_payload.get("latest"), dict) else {}
    trade_plan = analysis_payload.get("trade_plan") if isinstance(analysis_payload.get("trade_plan"), dict) else {}
    artifacts = analysis_payload.get("artifacts") if isinstance(analysis_payload.get("artifacts"), dict) else {}
    challenge = (live_plan or {}).get("challenge") if isinstance((live_plan or {}).get("challenge"), dict) else {}
    optimizer = challenge.get("optimizer_live_gate") if isinstance(challenge.get("optimizer_live_gate"), dict) else {}
    market_bot = challenge.get("market_bot_gate") if isinstance(challenge.get("market_bot_gate"), dict) else {}
    execution = _layer(live_plan, "execution_quality")
    performance = expected_value.get("strategy_performance") if isinstance(expected_value.get("strategy_performance"), dict) else {}
    execution_mode = str((live_plan or {}).get("execution_mode") or "")
    live_allowed = bool((live_plan or {}).get("allowed"))
    performance_sample_count = int(_float(performance.get("sample_count")))
    performance_pf = _float(performance.get("profit_factor"))
    performance_expectancy = _float(performance.get("expectancy_r"))
    backtest_passed = bool(optimizer.get("allowed") or market_bot.get("allowed")) or (
        performance_sample_count >= 30
        and performance_pf >= 1.0
        and performance_expectancy >= 0.0
    )
    challenge_status = str(challenge.get("status") or "")
    max_drawdown_pct = _round_or_none(challenge.get("max_drawdown_pct"), 4)
    current_drawdown_pct = _round_or_none(challenge.get("current_drawdown_pct"), 4)
    drawdown_passed = bool(challenge_status) and challenge_status != "drawdown-stop"
    if max_drawdown_pct is not None and current_drawdown_pct is not None:
        drawdown_passed = current_drawdown_pct <= max_drawdown_pct
    gates = [
        _gate_row(
            "data_check",
            bool(analysis and latest and trade_plan),
            {
                "has_analysis": bool(analysis),
                "has_latest_market_data": bool(latest),
                "has_trade_plan": bool(trade_plan),
                "analysis_report": artifacts.get("report_json"),
            },
            "analysis_payload",
        ),
        _gate_row(
            "backtest_or_performance_evidence",
            backtest_passed,
            {
                "optimizer_live_gate_allowed": optimizer.get("allowed"),
                "market_bot_gate_allowed": market_bot.get("allowed"),
                "strategy_performance_sample_count": performance_sample_count,
                "strategy_performance_profit_factor": performance.get("profit_factor"),
                "strategy_performance_expectancy_r": performance.get("expectancy_r"),
            },
            "optimizer_live_gate+market_bot_gate+strategy_performance",
        ),
        _gate_row(
            "trading_cost_check",
            bool(execution.get("passed")),
            {
                "execution_layer_passed": execution.get("passed"),
                "fee_profit_ratio": execution.get("fee_profit_ratio"),
                "slippage_profit_ratio": execution.get("slippage_profit_ratio"),
                "net_profit_to_risk": execution.get("net_profit_to_risk"),
            },
            "professional_entry_gate.execution_quality",
        ),
        _gate_row(
            "max_drawdown_check",
            drawdown_passed,
            {
                "challenge_status": challenge_status or None,
                "current_drawdown_pct": current_drawdown_pct,
                "max_drawdown_pct": max_drawdown_pct,
            },
            "challenge_state",
        ),
        _gate_row(
            "dry_run_check",
            True,
            {
                "command_surface": "trade-decision",
                "opens_orders": False,
                "writes_execution_config": False,
            },
            "trade_decision_cli",
        ),
        _gate_row(
            "testnet_forward_check",
            execution_mode == "testnet_exploration" and live_allowed,
            {
                "execution_mode": execution_mode,
                "live_plan_allowed": live_allowed,
            },
            "live_readiness_plan",
        ),
    ]
    failed = [row["name"] for row in gates if not row["passed"]]
    return {
        "all_passed": not failed,
        "failed": failed,
        "gates": gates,
    }


def _position_direction(side: str) -> str:
    normalized = side.upper()
    if normalized == "BUY":
        return "bullish"
    if normalized == "SELL":
        return "bearish"
    return "neutral"


def _max_loss_text(risk_pct: float) -> str:
    return f"Max planned account risk is {risk_pct:.4%}."


ENTRY_GATE_ACTIONS = {
    "regime_policy": {
        "action": "wait_for_regime_or_switch_strategy_family",
        "reason": "Current market regime does not allow the selected strategy family.",
    },
    "trend_direction": {
        "action": "wait_for_multi_timeframe_trend_alignment",
        "reason": "Trend structure conflicts with the candidate side.",
    },
    "momentum": {
        "action": "wait_for_momentum_confirmation",
        "reason": "Momentum strength is not sufficient for a new entry.",
    },
    "volume": {
        "action": "wait_for_liquidity_and_volume_confirmation",
        "reason": "Volume or OBV confirmation is too weak.",
    },
    "volatility": {
        "action": "wait_for_normalized_volatility",
        "reason": "Volatility or spread conditions are not safe enough.",
    },
    "support_resistance": {
        "action": "rebuild_entry_stop_take_profit_structure",
        "reason": "Entry, stop, and take-profit structure is not valid for the candidate side.",
    },
    "risk_reward": {
        "action": "wait_for_better_reward_risk_or_reprice_setup",
        "reason": "Reward/risk is not high enough after costs.",
    },
    "stop_distance": {
        "action": "reduce_size_or_wait_for_tighter_invalidation",
        "reason": "Stop distance or account risk is not acceptable.",
    },
}

LIFECYCLE_GATE_ACTIONS = {
    "data_check": {
        "action": "repair_market_data_or_analysis_artifacts",
        "reason": "Required analysis, latest data, or trade plan is missing.",
    },
    "backtest_or_performance_evidence": {
        "action": "run_backtest_or_repair_negative_expectancy",
        "reason": "Backtest or recent performance evidence is insufficient.",
    },
    "trading_cost_check": {
        "action": "wait_for_lower_costs_or_wider_profit_target",
        "reason": "Fees/slippage leave too little net profit versus risk.",
    },
    "max_drawdown_check": {
        "action": "pause_until_drawdown_recovers",
        "reason": "Drawdown state does not allow promotion.",
    },
    "dry_run_check": {
        "action": "rerun_read_only_trade_decision",
        "reason": "Dry-run/read-only decision surface did not validate.",
    },
    "testnet_forward_check": {
        "action": "collect_testnet_forward_evidence",
        "reason": "Testnet forward-readiness has not approved the candidate.",
    },
}


def _decision_next_actions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    entry_gates = payload.get("entry_gate_evidence") if isinstance(payload.get("entry_gate_evidence"), dict) else {}
    lifecycle = payload.get("lifecycle_gate_evidence") if isinstance(payload.get("lifecycle_gate_evidence"), dict) else {}
    expected = payload.get("expected_value") if isinstance(payload.get("expected_value"), dict) else {}
    if payload.get("decision") == "HOLD" and expected.get("positive") is False:
        actions.append(
            {
                "scope": "expected_value",
                "gate": "positive_expected_value",
                "action": "repair_expectancy_before_enabling_entries",
                "reason": "Long-term expected value is not positive.",
            }
        )
    for gate in [str(item) for item in entry_gates.get("failed") or [] if str(item)]:
        template = ENTRY_GATE_ACTIONS.get(
            gate,
            {"action": "inspect_entry_gate_evidence", "reason": "Entry gate failed."},
        )
        actions.append({"scope": "entry_gate", "gate": gate, **template})
    for gate in [str(item) for item in lifecycle.get("failed") or [] if str(item)]:
        template = LIFECYCLE_GATE_ACTIONS.get(
            gate,
            {"action": "inspect_lifecycle_gate_evidence", "reason": "Lifecycle gate failed."},
        )
        actions.append({"scope": "lifecycle_gate", "gate": gate, **template})
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in actions:
        key = (str(item.get("scope")), str(item.get("gate")), str(item.get("action")))
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _decision_answers(payload: dict[str, Any]) -> dict[str, Any]:
    decision = str(payload.get("decision") or "HOLD")
    direction = str(payload.get("direction") or "neutral")
    blocked = [str(item) for item in payload.get("blocked_reasons") or [] if str(item)]
    entry_reason = [str(item) for item in payload.get("entry_reason") or [] if str(item)]
    take_profit = payload.get("take_profit") if isinstance(payload.get("take_profit"), list) else []
    expected = payload.get("expected_value") if isinstance(payload.get("expected_value"), dict) else {}
    entry_gates = payload.get("entry_gate_evidence") if isinstance(payload.get("entry_gate_evidence"), dict) else {}
    lifecycle = payload.get("lifecycle_gate_evidence") if isinstance(payload.get("lifecycle_gate_evidence"), dict) else {}
    risk_pct = _float(payload.get("risk_pct"))
    why_not = blocked or ["All required gates passed."] if decision in {"BUY", "SELL", "EXIT"} else blocked
    if not why_not:
        failed = [str(item) for item in entry_gates.get("failed") or [] if str(item)]
        why_not = [f"Entry factor gate failed: {item}." for item in failed]
    why_long = entry_reason if decision == "BUY" or (decision == "HOLD" and direction == "bullish") else [
        "Not a long decision under current gates."
    ]
    why_short = entry_reason if decision == "SELL" or (decision == "HOLD" and direction == "bearish") else [
        "Not a short decision under current gates."
    ]
    return {
        "why_long_now": why_long,
        "why_short_now": why_short,
        "why_no_trade_now": why_not if decision == "HOLD" else ["Actionable decision was emitted; see entry_reason and hard_gates."],
        "where_stop_if_wrong": payload.get("stop_loss"),
        "where_take_profit_if_right": take_profit,
        "max_loss": _max_loss_text(risk_pct),
        "long_term_expected_value": {
            "positive": bool(expected.get("positive")),
            "strategy_performance": expected.get("strategy_performance"),
            "route_side_profit_factor": expected.get("route_side_profit_factor"),
            "route_side_net_pnl_usdt": expected.get("route_side_net_pnl_usdt"),
        },
        "failed_entry_gates": entry_gates.get("failed", []),
        "failed_lifecycle_gates": lifecycle.get("failed", []),
        "next_actions": _decision_next_actions(payload),
    }


def _validation_row(name: str, passed: bool, evidence: dict[str, Any], *, severity: str = "hard") -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "severity": severity,
        "evidence": evidence,
    }


def _valid_entry_structure(
    *,
    decision: str,
    entry: float | None,
    stop_loss: float | None,
    take_profit: list[float],
) -> bool:
    if decision not in ENTRY_DECISIONS:
        return True
    if entry is None or stop_loss is None or not take_profit:
        return False
    if decision in {"BUY", "LONG"}:
        return stop_loss < entry and all(level > entry for level in take_profit)
    if decision in {"SELL", "SHORT"}:
        return stop_loss > entry and all(level < entry for level in take_profit)
    return False


def _valid_direction_mapping(decision: str, direction: str) -> bool:
    if decision in {"BUY", "LONG"}:
        return direction == "bullish"
    if decision in {"SELL", "SHORT"}:
        return direction == "bearish"
    if decision == "HOLD":
        return direction in {"bullish", "bearish", "neutral"}
    if decision == "EXIT":
        return direction in {"bullish", "bearish", "neutral"}
    return False


def _validate_decision_contract(payload: dict[str, Any]) -> dict[str, Any]:
    decision = str(payload.get("decision") or "")
    direction = str(payload.get("direction") or "")
    required_fields = [
        "decision",
        "direction",
        "confidence",
        "regime",
        "entry_reason",
        "blocked_reasons",
        "invalid_if",
        "entry",
        "stop_loss",
        "take_profit",
        "risk_pct",
        "risk_reward_ratio",
        "expected_value",
        "position_size",
        "hailo_task_allocation",
        "decision_answers",
    ]
    hard_gates = payload.get("hard_gates") if isinstance(payload.get("hard_gates"), dict) else {}
    entry_evidence = payload.get("entry_gate_evidence") if isinstance(payload.get("entry_gate_evidence"), dict) else {}
    lifecycle = payload.get("lifecycle_gate_evidence") if isinstance(payload.get("lifecycle_gate_evidence"), dict) else {}
    expected = payload.get("expected_value") if isinstance(payload.get("expected_value"), dict) else {}
    position_size = payload.get("position_size") if isinstance(payload.get("position_size"), dict) else {}
    hailo_plan = payload.get("hailo_task_allocation") if isinstance(payload.get("hailo_task_allocation"), dict) else {}
    answers = payload.get("decision_answers") if isinstance(payload.get("decision_answers"), dict) else {}
    risk_pct = _float(payload.get("risk_pct"))
    take_profit = payload.get("take_profit") if isinstance(payload.get("take_profit"), list) else []
    entry = payload.get("entry")
    stop_loss = payload.get("stop_loss")
    execution_mode = str(payload.get("execution_mode") or hard_gates.get("execution_mode") or "")
    is_entry = decision in ENTRY_DECISIONS
    is_live_entry = is_entry and execution_mode == "live"
    is_exit = decision == "EXIT"
    hailo_tasks = hailo_plan.get("tasks") if isinstance(hailo_plan.get("tasks"), list) else []
    hailo_by_name = {str(item.get("name")): item for item in hailo_tasks if isinstance(item, dict)}
    answer_fields = [
        "why_long_now",
        "why_short_now",
        "why_no_trade_now",
        "where_stop_if_wrong",
        "where_take_profit_if_right",
        "max_loss",
        "long_term_expected_value",
    ]
    checks = [
        _validation_row(
            "allowed_decision",
            decision in ALLOWED_DECISIONS,
            {"decision": decision, "allowed": sorted(ALLOWED_DECISIONS)},
        ),
        _validation_row(
            "required_fields_present",
            all(field in payload for field in required_fields),
            {"missing": [field for field in required_fields if field not in payload]},
        ),
        _validation_row(
            "valid_direction_mapping",
            _valid_direction_mapping(decision, direction),
            {"decision": decision, "direction": direction},
        ),
        _validation_row(
            "entry_risk_ceiling",
            (not is_entry) or risk_pct <= 0.025,
            {"risk_pct": round(risk_pct, 6), "risk_ceiling_pct": 0.025},
        ),
        _validation_row(
            "entry_factor_gates_required",
            (not is_entry) or bool(hard_gates.get("entry_factor_gates_passed")),
            {
                "decision": decision,
                "entry_factor_gates_passed": hard_gates.get("entry_factor_gates_passed"),
                "failed_entry_gates": entry_evidence.get("failed", []),
            },
        ),
        _validation_row(
            "live_lifecycle_required",
            (not is_live_entry) or bool(hard_gates.get("live_promotion_lifecycle_passed")),
            {
                "execution_mode": execution_mode or None,
                "live_promotion_lifecycle_passed": hard_gates.get("live_promotion_lifecycle_passed"),
                "failed_lifecycle_gates": lifecycle.get("failed", []),
            },
        ),
        _validation_row(
            "positive_expected_value_required",
            (not is_entry) or bool(expected.get("positive")),
            {
                "decision": decision,
                "expected_value_positive": expected.get("positive"),
                "strategy_performance": expected.get("strategy_performance"),
            },
        ),
        _validation_row(
            "ai_cannot_directly_order",
            hard_gates.get("ai_direct_order_allowed") is False,
            {"ai_direct_order_allowed": hard_gates.get("ai_direct_order_allowed")},
        ),
        _validation_row(
            "hailo_cannot_execute_orders",
            hailo_by_name.get("order-execution-decision", {}).get("status") == "not_allowed",
            {
                "order_execution_status": hailo_by_name.get("order-execution-decision", {}).get("status"),
                "task_names": sorted(hailo_by_name),
            },
        ),
        _validation_row(
            "decision_answers_present",
            all(field in answers for field in answer_fields),
            {"missing": [field for field in answer_fields if field not in answers]},
        ),
        _validation_row(
            "read_only_decision_surface",
            payload.get("opens_orders") is False and payload.get("writes_execution_config") is False,
            {
                "opens_orders": payload.get("opens_orders"),
                "writes_execution_config": payload.get("writes_execution_config"),
            },
        ),
        _validation_row(
            "entry_stop_take_profit_orientation",
            _valid_entry_structure(
                decision=decision,
                entry=_round_or_none(entry, 8),
                stop_loss=_round_or_none(stop_loss, 8),
                take_profit=[item for item in (_round_or_none(level, 8) for level in take_profit) if item is not None],
            ),
            {
                "decision": decision,
                "entry": entry,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            },
        ),
        _validation_row(
            "entry_position_size_present",
            (not is_entry)
            or (
                _round_or_none(position_size.get("quantity"), 8) is not None
                and _float(position_size.get("quantity")) > 0.0
            ),
            {"quantity": position_size.get("quantity")},
        ),
        _validation_row(
            "exit_reduce_only_required",
            (not is_exit) or hard_gates.get("reduce_only_exit") is True,
            {"reduce_only_exit": hard_gates.get("reduce_only_exit")},
        ),
        _validation_row(
            "exit_active_position_required",
            (not is_exit)
            or (
                _round_or_none(position_size.get("quantity"), 8) is not None
                and _float(position_size.get("quantity")) > 0.0
                and _round_or_none(entry, 8) is not None
                and _round_or_none(stop_loss, 8) is not None
            ),
            {
                "quantity": position_size.get("quantity"),
                "entry": entry,
                "stop_loss": stop_loss,
            },
        ),
    ]
    failed = [check for check in checks if not check["passed"]]
    return {
        "schema_version": "decision-contract-v1",
        "valid": not failed,
        "failed": [check["name"] for check in failed],
        "checks": checks,
    }


def _decision_contract(
    *,
    decision: Decision,
    direction: str,
    confidence: int,
    regime: str,
    entry_reason: list[str],
    blocked_reasons: list[str],
    invalid_if: dict[str, Any],
    entry: float | None,
    stop_loss: float | None,
    take_profit: list[float],
    risk_pct: float,
    risk_reward_ratio: float | None,
    expected_value: dict[str, Any],
    position_size: dict[str, Any],
    hard_gates: dict[str, Any],
    entry_gate_evidence: dict[str, Any] | None = None,
    lifecycle_gate_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "decision": decision,
        "direction": direction,
        "confidence": confidence,
        "regime": regime,
        "entry_reason": entry_reason,
        "blocked_reasons": blocked_reasons,
        "invalid_if": invalid_if,
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_pct": round(risk_pct, 6),
        "risk_reward_ratio": risk_reward_ratio,
        "expected_value": expected_value,
        "position_size": position_size,
        "hard_gates": hard_gates,
        "hailo_task_allocation": build_hailo_trading_plan(),
        "opens_orders": False,
        "writes_execution_config": False,
    }
    if entry_gate_evidence is not None:
        payload["entry_gate_evidence"] = entry_gate_evidence
    if lifecycle_gate_evidence is not None:
        payload["lifecycle_gate_evidence"] = lifecycle_gate_evidence
    payload["decision_answers"] = _decision_answers(payload)
    payload["decision_contract_validation"] = _validate_decision_contract(payload)
    return payload


def build_ai_trade_decision_output(
    *,
    analysis_payload: dict[str, Any],
    strategy: StrategyConfig | None = None,
    live_plan: dict[str, Any] | None = None,
    requested_action: str | None = None,
) -> dict[str, Any]:
    """Build the auditable BUY/SELL/HOLD/EXIT decision schema.

    This layer does not execute orders. It normalizes existing analysis and
    readiness evidence into the operator-facing contract.
    """

    analysis = analysis_payload.get("analysis") if isinstance(analysis_payload.get("analysis"), dict) else {}
    latest = analysis_payload.get("latest") if isinstance(analysis_payload.get("latest"), dict) else {}
    trade_plan = analysis_payload.get("trade_plan") if isinstance(analysis_payload.get("trade_plan"), dict) else {}
    market_context = analysis_payload.get("market_context") if isinstance(analysis_payload.get("market_context"), dict) else {}
    raw_action = str(requested_action or analysis.get("recommended_action") or "HOLD").upper()
    if raw_action == "LONG":
        raw_action = "BUY"
    elif raw_action == "SHORT":
        raw_action = "SELL"
    if raw_action not in {"BUY", "SELL", "HOLD", "EXIT"}:
        raw_action = "HOLD"

    side = str((live_plan or {}).get("side") or raw_action)
    side = "BUY" if side == "LONG" else "SELL" if side == "SHORT" else side
    side_plan = trade_plan.get("short" if side == "SELL" else "long") if isinstance(trade_plan, dict) else {}
    side_plan = side_plan if isinstance(side_plan, dict) else {}
    entry = _round_or_none((live_plan or {}).get("price"), 6) or _round_or_none(side_plan.get("entry_reference"), 6)
    stop_loss = _round_or_none((live_plan or {}).get("stop_price"), 6) or _round_or_none(side_plan.get("invalidation"), 6)
    take_profit_levels = (live_plan or {}).get("take_profit_prices")
    if not isinstance(take_profit_levels, list) or not take_profit_levels:
        take_profit_levels = side_plan.get("take_profit_levels") if isinstance(side_plan.get("take_profit_levels"), list) else []
    take_profit = [_round_or_none(item, 6) for item in take_profit_levels]
    take_profit = [item for item in take_profit if item is not None]
    first_take_profit = take_profit[0] if take_profit else None
    evidence_side = _candidate_side(
        side,
        entry=entry,
        stop_loss=stop_loss,
        first_take_profit=first_take_profit,
    )
    risk_pct = _float((live_plan or {}).get("planned_account_risk_pct"), 0.0)
    blocked = _blocked_reasons(analysis, live_plan, risk_pct=risk_pct)
    live_allowed = bool((live_plan or {}).get("allowed")) if live_plan is not None else bool(analysis.get("entry_ready"))

    decision: Decision = raw_action if raw_action in {"BUY", "SELL", "EXIT"} else "HOLD"
    if decision in {"BUY", "SELL"} and (blocked or not live_allowed):
        decision = "HOLD"
    if risk_pct > 0.025:
        decision = "HOLD"

    confidence = int(max(0.0, min(100.0, _float(analysis.get("score"), 50.0))))
    rr = _risk_reward(entry, stop_loss, first_take_profit, evidence_side)
    entry_evidence = _entry_gate_evidence(
        analysis=analysis,
        latest=latest,
        live_plan=live_plan,
        side=evidence_side,
        entry=entry,
        stop_loss=stop_loss,
        first_take_profit=first_take_profit,
        risk_reward_ratio=rr,
        risk_pct=risk_pct,
    )
    expected = _expected_value(analysis, live_plan)
    lifecycle_evidence = _lifecycle_gate_evidence(
        analysis_payload=analysis_payload,
        live_plan=live_plan,
        expected_value=expected,
    )
    if decision in {"BUY", "SELL"} and not expected["positive"]:
        decision = "HOLD"
        blocked.append("Long-term expected value is not positive under current route/performance evidence.")
    if decision in {"BUY", "SELL"} and not entry_evidence["all_passed"]:
        decision = "HOLD"
        blocked.extend(f"Entry factor gate failed: {name}." for name in entry_evidence["failed"])
    if (
        decision in {"BUY", "SELL"}
        and str((live_plan or {}).get("execution_mode") or "") == "live"
        and not lifecycle_evidence["all_passed"]
    ):
        decision = "HOLD"
        blocked.extend(f"Live promotion lifecycle gate failed: {name}." for name in lifecycle_evidence["failed"])
    invalid_source = side_plan.get("invalidation_source")
    if decision == "HOLD" and not blocked:
        blocked.append("Conditions are incomplete or not explicitly approved by readiness gates.")

    return _decision_contract(
        decision=decision,
        direction=_direction_from_action(decision if decision != "HOLD" else raw_action, analysis),
        confidence=confidence,
        regime=_canonical_regime(analysis, latest, market_context),
        entry_reason=_entry_reason(
            decision if decision != "HOLD" else raw_action,
            analysis,
            side_plan,
            approved=decision in {"BUY", "SELL"},
        ),
        blocked_reasons=blocked,
        invalid_if={
            "price_crosses": stop_loss,
            "source": invalid_source or "live_plan_stop_loss",
            "notes": [
                "Setup is invalid if stop-loss is touched.",
                "Setup is invalid if trend/regime or hard risk gates fail before entry.",
            ],
        },
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_pct=risk_pct,
        risk_reward_ratio=rr,
        expected_value=expected,
        position_size={
            "quantity": _round_or_none((live_plan or {}).get("quantity"), 8),
            "margin_notional_usdt": _round_or_none((live_plan or {}).get("margin_notional_usdt"), 6),
            "gross_notional_usdt": _round_or_none((live_plan or {}).get("gross_notional_usdt"), 6),
            "leverage": (live_plan or {}).get("leverage"),
        },
        hard_gates={
            "risk_ceiling_pct": 0.025,
            "risk_ceiling_passed": risk_pct <= 0.025,
            "readiness_allowed": live_allowed,
            "entry_factor_gates_passed": bool(entry_evidence["all_passed"]),
            "live_promotion_lifecycle_passed": bool(lifecycle_evidence["all_passed"]),
            "ai_direct_order_allowed": False,
            "execution_mode": str((live_plan or {}).get("execution_mode") or ""),
        },
        entry_gate_evidence=entry_evidence,
        lifecycle_gate_evidence=lifecycle_evidence,
    )


def build_blocked_trade_decision_output(*, reason: str, blockers: list[str] | None = None) -> dict[str, Any]:
    """Build a read-only HOLD decision when analysis/readiness cannot run."""

    return _decision_contract(
        decision="HOLD",
        direction="neutral",
        confidence=0,
        regime="unknown",
        entry_reason=[reason],
        blocked_reasons=blockers or [reason],
        invalid_if={"price_crosses": None, "source": "unavailable", "notes": []},
        entry=None,
        stop_loss=None,
        take_profit=[],
        risk_pct=0.0,
        risk_reward_ratio=None,
        expected_value={"positive": False},
        position_size={
            "quantity": None,
            "margin_notional_usdt": None,
            "gross_notional_usdt": None,
            "leverage": None,
        },
        hard_gates={
            "risk_ceiling_pct": 0.025,
            "risk_ceiling_passed": False,
            "readiness_allowed": False,
            "ai_direct_order_allowed": False,
        },
    )


def build_ai_exit_decision_output(
    *,
    position_plan: PositionManagementPlan,
    exit_plan: AdaptiveExitPlan,
    analysis_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the auditable EXIT/HOLD schema for an existing position."""

    analysis = analysis_payload.get("analysis") if isinstance(analysis_payload.get("analysis"), dict) else {}
    latest = analysis_payload.get("latest") if isinstance(analysis_payload.get("latest"), dict) else {}
    market_context = analysis_payload.get("market_context") if isinstance(analysis_payload.get("market_context"), dict) else {}
    blocked = [str(item) for item in position_plan.violations + position_plan.warnings + exit_plan.warnings if str(item)]
    if not exit_plan.allowed:
        blocked.append(f"Adaptive exit gate blocked: {exit_plan.reason_code}.")
    blocked = list(dict.fromkeys(blocked))
    active_position = position_plan.side in {"BUY", "SELL"} and position_plan.quantity > 0
    decision: Decision = "EXIT" if exit_plan.allowed and exit_plan.action == "close_position" else "HOLD"
    confidence = int(max(0.0, min(100.0, round(exit_plan.confidence * 100.0))))
    stop_loss = _round_or_none(exit_plan.reference_stop_price, 6) if active_position else None
    risk_pct = 0.0
    if active_position and position_plan.entry_price > 0:
        risk_notional = exit_plan.risk_distance * position_plan.quantity
        gross_notional = abs(position_plan.entry_price * position_plan.quantity)
        if gross_notional > 0:
            risk_pct = risk_notional / gross_notional
    if risk_pct > 0.025 and decision == "EXIT":
        blocked.append(f"Current position risk {risk_pct:.4%} exceeds the 2.5% per-trade ceiling; exit remains risk-reducing.")

    return _decision_contract(
        decision=decision,
        direction=_position_direction(position_plan.side),
        confidence=confidence,
        regime=_canonical_regime(analysis, latest, market_context),
        entry_reason=exit_plan.reasons or ["No confirmed exit trigger; continue managing the open position."],
        blocked_reasons=blocked,
        invalid_if={
            "price_crosses": stop_loss,
            "source": "adaptive_exit_reference_stop",
            "notes": [
                "Open position is invalid if the reference stop is touched."
                if active_position
                else "No active position exists, so no exit stop is active.",
                "EXIT is allowed only when adaptive reversal evidence is confirmed.",
            ],
        },
        entry=_round_or_none(position_plan.entry_price, 6) if active_position else None,
        stop_loss=stop_loss,
        take_profit=[],
        risk_pct=risk_pct,
        risk_reward_ratio=None,
        expected_value={
            "positive": decision == "HOLD" and exit_plan.reason_code in {"no-confirmed-reversal", "inside-adaptive-exit-band"},
            "unrealized_r": exit_plan.unrealized_r,
            "reversal_score": exit_plan.reversal_score,
            "reason_code": exit_plan.reason_code,
            "analysis_bias": exit_plan.analysis_bias,
            "recommended_action": exit_plan.recommended_action,
            "selected_family": exit_plan.selected_family,
            "selected_family_bias": exit_plan.selected_family_bias,
        },
        position_size={
            "quantity": _round_or_none(position_plan.quantity, 8),
            "margin_notional_usdt": None,
            "gross_notional_usdt": _round_or_none(position_plan.entry_price * position_plan.quantity, 6)
            if active_position
            else None,
            "leverage": position_plan.leverage,
        },
        hard_gates={
            "risk_ceiling_pct": 0.025,
            "risk_ceiling_passed": risk_pct <= 0.025,
            "readiness_allowed": exit_plan.allowed,
            "ai_direct_order_allowed": False,
            "reduce_only_exit": decision == "EXIT",
        },
    )
