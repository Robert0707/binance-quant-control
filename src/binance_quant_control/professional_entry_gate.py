from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .convergence import calculate_expectancy_stats, calculate_loss_streak, calculate_profit_factor
from .order_journal import read_closed_trade_reviews

PURE_STOP_REASONS = {"stop_loss", "stop_priority_same_bar"}
STOP_AFTER_PROFIT_REASONS = {"partial_tp_then_stop"}
TREND_FAMILIES = {"trend_continuation", "trend_pullback", "breakout", "liquidity_reclaim", "vwap_reclaim", "ai_family_router"}
RANGE_FAMILIES = {"mean_reversion", "range_mean_reversion", "liquidity_reclaim", "vwap_reclaim", "ai_family_router"}
SURGE_FAMILIES = {"trend_continuation", "breakout", "liquidity_reclaim", "vwap_reclaim", "ai_family_router"}


@dataclass(frozen=True, slots=True)
class ProfessionalGatePolicy:
    min_reward_risk: float = 1.2
    min_net_profit_to_risk: float = 0.8
    max_fee_profit_ratio: float = 0.35
    max_slippage_profit_ratio: float = 0.25
    max_spread_bps: float = 12.0
    max_volatility: float = 1.8
    min_volume_zscore: float = -0.8
    min_obv_zscore_long: float = -0.5
    max_obv_zscore_short: float = 0.5
    min_composite_quality: float = 0.0
    min_price_structure_score: float = 0.0
    min_execution_quality_score: float = 0.0
    require_mtf_alignment: bool = True
    min_mtf_confidence: float = 0.55
    min_recent_reviews: int = 6
    min_recent_win_rate: float = 0.42
    min_recent_profit_factor: float = 1.0
    min_recent_avg_r: float = 0.0
    min_recent_expectancy_r: float = 0.0
    min_recent_payoff_ratio: float = 1.0
    max_recent_stop_loss_ratio: float = 0.55
    recent_lookback: int = 20
    stop_loss_cooldown_hours: float = 6.0
    require_professional_gate: bool = True
    allow_thin_scoped_history: bool = False
    allow_market_bot_evidence: bool = False
    allow_risk_combo_evidence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProfessionalGateResult:
    passed: bool
    layers: dict[str, dict[str, Any]]
    violations: list[str]
    warnings: list[str]
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "layers": self.layers,
            "violations": self.violations,
            "warnings": self.warnings,
            "stats": self.stats,
        }


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _row_side(row: dict[str, Any]) -> str:
    return str(row.get("side") or "").upper()


def _select_review_scope(
    reviews: list[dict[str, Any]],
    *,
    policy: ProfessionalGatePolicy,
    side: str,
    symbol: str = "",
    route_id: str = "",
    strategy_profile: str = "",
) -> tuple[list[dict[str, Any]], str, bool]:
    side_upper = side.upper()
    symbol_upper = symbol.upper()
    scopes: list[tuple[str, list[dict[str, Any]]]] = []
    if symbol_upper and side_upper:
        scopes.append(
            (
                "symbol_side",
                [
                    row
                    for row in reviews
                    if str(row.get("symbol") or "").upper() == symbol_upper
                    and _row_side(row) == side_upper
                ],
            )
        )
    if route_id and side_upper:
        scopes.append(
            (
                "route_side",
                [
                    row
                    for row in reviews
                    if str(row.get("route_id") or "") == route_id and _row_side(row) == side_upper
                ],
            )
        )
    if strategy_profile and side_upper:
        scopes.append(
            (
                "strategy_side",
                [
                    row
                    for row in reviews
                    if str(row.get("strategy_profile") or "") == strategy_profile
                    and _row_side(row) == side_upper
                ],
            )
        )
    if policy.allow_thin_scoped_history and scopes:
        for name, rows in scopes:
            if rows:
                return rows, name, len(rows) >= max(int(policy.min_recent_reviews), 1)
        return [], scopes[0][0], False
    if side_upper:
        scopes.append(("side", [row for row in reviews if _row_side(row) == side_upper]))
    scopes.append(("global", list(reviews)))

    min_reviews = max(int(policy.min_recent_reviews), 1)
    for name, rows in scopes:
        if len(rows) >= min_reviews:
            return rows, name, True
    for name, rows in scopes:
        if rows:
            return rows, name, False
    return [], scopes[0][0] if scopes else "global", False


def _recent_review_stats(
    policy: ProfessionalGatePolicy,
    *,
    side: str,
    symbol: str = "",
    route_id: str = "",
    strategy_profile: str = "",
) -> dict[str, Any]:
    reviews, scope, mature = _select_review_scope(
        read_closed_trade_reviews(),
        policy=policy,
        side=side,
        symbol=symbol,
        route_id=route_id,
        strategy_profile=strategy_profile,
    )
    recent = reviews[-policy.recent_lookback :] if policy.recent_lookback > 0 else reviews
    pnls = [_float(item.get("realized_pnl_usdt")) for item in recent]
    wins = [item for item in recent if _float(item.get("realized_pnl_usdt")) > 0.0]
    losses = [item for item in recent if _float(item.get("realized_pnl_usdt")) < 0.0]
    stop_losses = [
        item for item in recent if str(item.get("exit_reason", "")).lower() in PURE_STOP_REASONS
    ]
    stop_after_profit = [
        item for item in recent if str(item.get("exit_reason", "")).lower() in STOP_AFTER_PROFIT_REASONS
    ]
    r_values = [
        _float(item.get("realized_r_multiple"))
        for item in recent
        if item.get("realized_r_multiple") is not None
    ]
    last_stop_loss_at: datetime | None = None
    for item in reversed(recent):
        if str(item.get("exit_reason", "")).lower() in PURE_STOP_REASONS:
            last_stop_loss_at = _parse_datetime(item.get("closed_at") or item.get("reviewed_at"))
            break

    count = len(recent)
    expectancy_stats = calculate_expectancy_stats(r_values)
    return {
        "count": count,
        "scope": scope,
        "scope_mature": mature,
        "win_rate": round(len(wins) / count, 4) if count else 0.0,
        "loss_rate": round(len(losses) / count, 4) if count else 0.0,
        "stop_loss_ratio": round(len(stop_losses) / count, 4) if count else 0.0,
        "stop_after_profit_ratio": round(len(stop_after_profit) / count, 4) if count else 0.0,
        "profit_factor": round(calculate_profit_factor(pnls), 4) if pnls else 0.0,
        "avg_r_multiple": round(sum(r_values) / len(r_values), 4) if r_values else 0.0,
        "r_sample_count": len(r_values),
        "loss_streak": calculate_loss_streak(pnls),
        **expectancy_stats,
        "last_stop_loss_at": last_stop_loss_at.isoformat() if last_stop_loss_at else None,
    }


def _market_bot_review_stats(live_plan: dict[str, Any]) -> dict[str, Any] | None:
    gate = live_plan.get("market_bot_gate") if isinstance(live_plan.get("market_bot_gate"), dict) else {}
    row = gate.get("matched_row") if isinstance(gate.get("matched_row"), dict) else {}
    if not gate.get("allowed") or not row:
        return None
    trade_count = int(_float(row.get("trade_count")))
    if trade_count <= 0:
        return None
    win_rate = _float(row.get("win_rate")) / 100.0
    stop_loss_ratio = _float(row.get("stop_loss_ratio")) / 100.0
    expectancy_r = _float(row.get("expectancy_r"))
    payoff_ratio = _float(row.get("payoff_ratio"))
    profit_factor = _float(row.get("profit_factor"))
    loss_rate = max(0.0, 1.0 - win_rate)
    avg_loss_r = 1.0
    avg_win_r = payoff_ratio if payoff_ratio > 0.0 else 0.0
    break_even_win_rate = 100.0 / (payoff_ratio + 1.0) if payoff_ratio > 0.0 else 100.0
    return {
        "count": trade_count,
        "scope": "market_bot_gate",
        "scope_mature": True,
        "win_rate": round(win_rate, 4),
        "loss_rate": round(loss_rate, 4),
        "stop_loss_ratio": round(stop_loss_ratio, 4),
        "stop_after_profit_ratio": 0.0,
        "profit_factor": round(profit_factor, 4),
        "avg_r_multiple": round(expectancy_r, 4),
        "r_sample_count": trade_count,
        "loss_streak": 0,
        "expectancy_r": round(expectancy_r, 4),
        "avg_win_r": round(avg_win_r, 4),
        "avg_loss_r": avg_loss_r,
        "payoff_ratio": round(payoff_ratio, 4),
        "break_even_win_rate": round(break_even_win_rate, 4),
        "expectancy_edge_points": round((win_rate * 100.0) - break_even_win_rate, 4),
        "last_stop_loss_at": None,
        "source_report_path": gate.get("report_path"),
        "source_cohort_id": row.get("cohort_id"),
    }


def _risk_combo_review_stats(live_plan: dict[str, Any]) -> dict[str, Any] | None:
    gate = live_plan.get("risk_combo_gate") if isinstance(live_plan.get("risk_combo_gate"), dict) else {}
    surface = gate.get("matched_surface") if isinstance(gate.get("matched_surface"), dict) else {}
    if not gate.get("allowed") or not surface:
        return None
    metrics = surface.get("full") if isinstance(surface.get("full"), dict) else {}
    trade_count = int(_float(metrics.get("trade_count")))
    if trade_count <= 0:
        return None
    win_rate = _float(metrics.get("win_rate")) / 100.0
    stop_loss_ratio = _float(metrics.get("stop_loss_ratio")) / 100.0
    expectancy_r = _float(metrics.get("expectancy_r"))
    payoff_ratio = _float(metrics.get("payoff_ratio"))
    profit_factor = _float(metrics.get("profit_factor"))
    loss_rate = max(0.0, 1.0 - win_rate)
    avg_win_r = _float(metrics.get("avg_win_r"))
    avg_loss_r = _float(metrics.get("avg_loss_r"), 1.0)
    break_even_win_rate = 100.0 / (payoff_ratio + 1.0) if payoff_ratio > 0.0 else 100.0
    return {
        "count": trade_count,
        "scope": "risk_combo_matrix",
        "scope_mature": True,
        "win_rate": round(win_rate, 4),
        "loss_rate": round(loss_rate, 4),
        "stop_loss_ratio": round(stop_loss_ratio, 4),
        "stop_after_profit_ratio": _float(metrics.get("partial_tp_then_stop_ratio")) / 100.0,
        "profit_factor": round(profit_factor, 4),
        "avg_r_multiple": round(expectancy_r, 4),
        "r_sample_count": trade_count,
        "loss_streak": int(_float(metrics.get("loss_streak"))),
        "expectancy_r": round(expectancy_r, 4),
        "avg_win_r": round(avg_win_r, 4),
        "avg_loss_r": round(avg_loss_r, 4),
        "payoff_ratio": round(payoff_ratio, 4),
        "break_even_win_rate": round(break_even_win_rate, 4),
        "expectancy_edge_points": round((win_rate * 100.0) - break_even_win_rate, 4),
        "last_stop_loss_at": None,
        "source_report_path": gate.get("report_path"),
        "source_sweep_report_path": surface.get("source_report_path"),
        "source_surface": surface.get("surface"),
    }


def _planned_gross_profit(*, side: str, live_plan: dict[str, Any], price: float, quantity: float) -> float:
    side_upper = side.upper()
    if price <= 0.0 or quantity <= 0.0 or side_upper not in {"BUY", "SELL"}:
        return 0.0
    direction = 1.0 if side_upper == "BUY" else -1.0
    take_profit_prices = live_plan.get("take_profit_prices")
    if not isinstance(take_profit_prices, list) or not take_profit_prices:
        take_profit_prices = [live_plan.get("take_profit_price")]
    prices = [_float(item) for item in take_profit_prices if _float(item) > 0.0]
    quantities = live_plan.get("take_profit_quantities")
    if isinstance(quantities, list) and quantities:
        planned_quantities = [_float(item) for item in quantities]
    else:
        weights = live_plan.get("take_profit_weights")
        if isinstance(weights, list) and weights:
            planned_quantities = [quantity * _float(weight) for weight in weights]
        else:
            planned_quantities = [quantity]
    if len(planned_quantities) < len(prices):
        remaining = max(0.0, quantity - sum(planned_quantities))
        missing = len(prices) - len(planned_quantities)
        planned_quantities.extend([remaining / missing if missing > 0 else 0.0] * missing)
    gross_profit = 0.0
    for target_price, target_quantity in zip(prices, planned_quantities, strict=False):
        gross_profit += max(0.0, direction * (target_price - price)) * max(0.0, target_quantity)
    runner_quantity = _float(live_plan.get("take_profit_runner_quantity"))
    if runner_quantity > 0.0 and prices:
        gross_profit += max(0.0, direction * (prices[-1] - price)) * runner_quantity
    return gross_profit


def _planned_reward_distance(*, side: str, live_plan: dict[str, Any], price: float) -> float:
    side_upper = side.upper()
    if price <= 0.0 or side_upper not in {"BUY", "SELL"}:
        return 0.0
    direction = 1.0 if side_upper == "BUY" else -1.0
    take_profit_prices = live_plan.get("take_profit_prices")
    if not isinstance(take_profit_prices, list) or not take_profit_prices:
        take_profit_prices = [live_plan.get("take_profit_price")]
    prices = [_float(item) for item in take_profit_prices if _float(item) > 0.0]
    weights = live_plan.get("take_profit_weights")
    if isinstance(weights, list) and len(weights) >= len(prices):
        planned_weights = [_float(item) for item in weights]
    else:
        planned_weights = [1.0 / len(prices) for _ in prices] if prices else []
    weighted_reward = 0.0
    total_weight = 0.0
    for target_price, weight in zip(prices, planned_weights, strict=False):
        clean_weight = max(0.0, weight)
        weighted_reward += max(0.0, direction * (target_price - price)) * clean_weight
        total_weight += clean_weight
    return weighted_reward / total_weight if total_weight > 0.0 else 0.0


def _canonical_regime(latest: dict[str, Any], live_plan: dict[str, Any]) -> str:
    raw = str(live_plan.get("regime") or latest.get("regime") or "").lower()
    if raw in {"trend-up", "trend-down", "trend"}:
        return "trend"
    if raw in {"range", "squeeze"}:
        return "range"
    if raw in {"crash", "pump", "low_liquidity", "abnormal_volatility"}:
        return raw
    volume_z = _float(latest.get("volume_zscore_20"))
    spread_bps = _float(live_plan.get("spread_bps"))
    if spread_bps > 12.0 or volume_z < -1.2:
        return "low_liquidity"
    realized_vol = _float(latest.get("realized_vol_20"))
    close = _float(latest.get("close") or live_plan.get("price"))
    ema_slow = _float(latest.get("ema_slow"))
    macd_hist = _float(latest.get("macd_hist"))
    if realized_vol >= 1.8 or volume_z >= 2.5:
        if close > 0 and ema_slow > 0 and close >= ema_slow and macd_hist >= 0:
            return "pump"
        if close > 0 and ema_slow > 0 and close <= ema_slow and macd_hist <= 0:
            return "crash"
        return "abnormal_volatility"
    return raw or "unknown"


def _strategy_family(live_plan: dict[str, Any]) -> str:
    family = str(live_plan.get("strategy_family") or "").strip().lower()
    if family:
        return family
    selected = live_plan.get("selected_strategy_family")
    if isinstance(selected, dict):
        return str(selected.get("family") or "").strip().lower()
    return ""


def evaluate_professional_entry_gate(
    *,
    side: str,
    latest: dict[str, Any],
    live_plan: dict[str, Any],
    policy: ProfessionalGatePolicy | None = None,
    symbol: str = "",
    route_id: str = "",
    strategy_profile: str = "",
) -> ProfessionalGateResult:
    """Evaluate a professional trading-desk style entry checklist.

    This gate deliberately stays above the exchange execution layer. It blocks or
    warns based on market state, execution quality, signal quality, and recent
    strategy performance, but it never edits live-trading switches or order code.
    """

    active_policy = policy or ProfessionalGatePolicy()
    violations: list[str] = []
    warnings: list[str] = []
    layers: dict[str, dict[str, Any]] = {}

    side_upper = side.upper()
    regime = _canonical_regime(latest, live_plan)
    strategy_family = _strategy_family(live_plan)
    regime_passed = True
    allowed_families: set[str] = set()
    if regime == "low_liquidity":
        regime_passed = False
        violations.append("Low-liquidity regime blocks new entries until volume/spread normalize.")
    elif regime in {"pump", "crash", "abnormal_volatility"}:
        allowed_families = SURGE_FAMILIES
        if strategy_family and strategy_family not in allowed_families:
            regime_passed = False
            violations.append(
                f"Regime {regime} only allows surge/trend families; strategy family {strategy_family} is not allowed."
            )
    elif regime == "range":
        allowed_families = RANGE_FAMILIES
        if strategy_family and strategy_family not in allowed_families:
            regime_passed = False
            violations.append(
                f"Range regime only allows mean-reversion or reclaim families; strategy family {strategy_family} is not allowed."
            )
    elif regime == "trend":
        allowed_families = TREND_FAMILIES
        if strategy_family and strategy_family not in allowed_families:
            regime_passed = False
            violations.append(
                f"Trend regime requires trend/breakout/reclaim families; strategy family {strategy_family} is not allowed."
            )
    else:
        warnings.append("Market regime is unknown; professional gate relies on lower-level factor checks.")
    layers["regime_policy"] = {
        "passed": regime_passed,
        "regime": regime,
        "strategy_family": strategy_family or None,
        "allowed_families": sorted(allowed_families),
    }
    price = _float(live_plan.get("price"))
    quantity = _float(live_plan.get("quantity"))
    stop_price = _float(live_plan.get("stop_price"))
    take_profit_price = _float(live_plan.get("take_profit_price"))
    planned_risk = _float(live_plan.get("planned_account_risk_usdt"))
    stop_distance = abs(price - stop_price) if price > 0 and stop_price > 0 else 0.0
    target_distance = abs(take_profit_price - price) if price > 0 and take_profit_price > 0 else 0.0
    gross_risk = stop_distance * quantity
    gross_profit = _planned_gross_profit(side=side_upper, live_plan=live_plan, price=price, quantity=quantity)
    if gross_profit <= 0.0:
        gross_profit = target_distance * quantity
    notional = price * quantity
    fee_bps = _float(live_plan.get("fee_bps"), 4.0)
    slippage_bps = _float(live_plan.get("slippage_bps"), 2.0)
    estimated_fees = notional * (fee_bps / 10000.0) * 2.0
    estimated_slippage = notional * (slippage_bps / 10000.0) * 2.0
    net_profit = gross_profit - estimated_fees - estimated_slippage
    tp1_reward_risk = target_distance / stop_distance if stop_distance else 0.0
    planned_reward_distance = _planned_reward_distance(side=side_upper, live_plan=live_plan, price=price)
    reward_risk = (
        gross_profit / gross_risk
        if gross_risk > 0
        else planned_reward_distance / stop_distance
        if stop_distance > 0.0
        else 0.0
    )
    risk_denominator = max(planned_risk, gross_risk)
    net_profit_to_risk = net_profit / risk_denominator if risk_denominator > 0 else 0.0
    fee_profit_ratio = estimated_fees / gross_profit if gross_profit > 0 else 1.0
    slippage_profit_ratio = estimated_slippage / gross_profit if gross_profit > 0 else 1.0

    execution_passed = True
    if reward_risk < active_policy.min_reward_risk:
        execution_passed = False
        violations.append(
            f"Reward/risk {reward_risk:.2f}R is below professional floor "
            f"{active_policy.min_reward_risk:.2f}R."
        )
    if net_profit_to_risk < active_policy.min_net_profit_to_risk:
        execution_passed = False
        violations.append(
            f"Net TP1 profit/risk {net_profit_to_risk:.2f}R is below "
            f"{active_policy.min_net_profit_to_risk:.2f}R after fees/slippage."
        )
    if fee_profit_ratio > active_policy.max_fee_profit_ratio:
        execution_passed = False
        violations.append(
            f"Estimated fees consume {fee_profit_ratio:.1%} of gross TP1 profit."
        )
    if slippage_profit_ratio > active_policy.max_slippage_profit_ratio:
        execution_passed = False
        violations.append(
            f"Estimated slippage consumes {slippage_profit_ratio:.1%} of gross TP1 profit."
        )
    layers["execution_quality"] = {
        "passed": execution_passed,
        "reward_risk": round(reward_risk, 4),
        "tp1_reward_risk": round(tp1_reward_risk, 4),
        "planned_reward_distance": round(planned_reward_distance, 6),
        "net_profit_to_risk": round(net_profit_to_risk, 4),
        "gross_profit_usdt": round(gross_profit, 6),
        "net_profit_usdt": round(net_profit, 6),
        "gross_risk_usdt": round(gross_risk, 6),
        "estimated_fees_usdt": round(estimated_fees, 6),
        "estimated_slippage_usdt": round(estimated_slippage, 6),
        "fee_profit_ratio": round(fee_profit_ratio, 4),
        "slippage_profit_ratio": round(slippage_profit_ratio, 4),
    }

    realized_vol = _float(latest.get("realized_vol_20"))
    volume_z = _float(latest.get("volume_zscore_20"))
    obv_z = _float(latest.get("obv_zscore_20"))
    bb_bandwidth = _float(latest.get("bb_bandwidth"))
    spread_bps = _float(live_plan.get("spread_bps"), 0.0)
    market_passed = True
    if realized_vol > active_policy.max_volatility:
        market_passed = False
        violations.append(
            f"Realized volatility {realized_vol:.3f} is above ceiling "
            f"{active_policy.max_volatility:.3f}."
        )
    if volume_z < active_policy.min_volume_zscore:
        market_passed = False
        violations.append(
            f"Volume z-score {volume_z:.2f} is below liquidity floor "
            f"{active_policy.min_volume_zscore:.2f}."
        )
    if spread_bps > active_policy.max_spread_bps:
        market_passed = False
        violations.append(
            f"Spread {spread_bps:.2f} bps is above max {active_policy.max_spread_bps:.2f} bps."
        )
    if bb_bandwidth <= 0.0:
        warnings.append("Bollinger bandwidth is unavailable; volatility state is less reliable.")
    layers["market_state"] = {
        "passed": market_passed,
        "realized_vol_20": round(realized_vol, 6),
        "volume_zscore_20": round(volume_z, 4),
        "obv_zscore_20": round(obv_z, 4),
        "bb_bandwidth": round(bb_bandwidth, 6),
        "spread_bps": round(spread_bps, 4),
    }

    signal_passed = True
    signal_scores = live_plan.get("signal_scores") or (live_plan.get("sizing") or {}).get("signal_scores") or {}
    composite_quality = _float(signal_scores.get("composite_convergence_score"))
    price_structure_score = _float(signal_scores.get("price_structure_score"))
    execution_quality_score = _float(signal_scores.get("execution_quality_score"))
    if side_upper == "BUY" and obv_z < active_policy.min_obv_zscore_long:
        signal_passed = False
        violations.append(
            f"Long setup lacks accumulation confirmation: OBV z-score {obv_z:.2f}."
        )
    if side_upper == "SELL" and obv_z > active_policy.max_obv_zscore_short:
        signal_passed = False
        violations.append(
            f"Short setup lacks distribution confirmation: OBV z-score {obv_z:.2f}."
        )
    if active_policy.min_composite_quality > 0.0 and composite_quality <= 0.0:
        warnings.append("Composite signal quality is unavailable; professional gate relies on base signal fields.")
    elif composite_quality < active_policy.min_composite_quality:
        signal_passed = False
        violations.append(
            f"Composite signal quality {composite_quality:.2f} is below "
            f"{active_policy.min_composite_quality:.2f}."
        )
    if active_policy.min_price_structure_score > 0.0 and price_structure_score <= 0.0:
        warnings.append("Price-structure score is unavailable; professional gate relies on analysis score/convergence.")
    elif price_structure_score < active_policy.min_price_structure_score:
        signal_passed = False
        violations.append(
            f"Price-structure quality {price_structure_score:.2f} is below "
            f"{active_policy.min_price_structure_score:.2f}."
        )
    if active_policy.min_execution_quality_score > 0.0 and execution_quality_score <= 0.0:
        warnings.append("Execution-quality score is unavailable; professional gate relies on explicit R/R and cost checks.")
    elif execution_quality_score < active_policy.min_execution_quality_score:
        signal_passed = False
        violations.append(
            f"Execution-quality score {execution_quality_score:.2f} is below "
            f"{active_policy.min_execution_quality_score:.2f}."
        )
    layers["signal_quality"] = {
        "passed": signal_passed,
        "side": side_upper,
        "analysis_score": live_plan.get("analysis_score"),
        "analysis_convergence": live_plan.get("analysis_convergence"),
        "adx_value": live_plan.get("adx_value"),
        "obv_zscore_20": round(obv_z, 4),
        "composite_convergence_score": round(composite_quality, 4),
        "price_structure_score": round(price_structure_score, 4),
        "execution_quality_score": round(execution_quality_score, 4),
    }

    mtf = latest.get("multi_timeframe_structure") if isinstance(latest.get("multi_timeframe_structure"), dict) else {}
    mtf_bias = str(mtf.get("bias") or "neutral")
    mtf_alignment = str(mtf.get("alignment") or "unavailable")
    mtf_confidence = _float(mtf.get("confidence"))
    expected_mtf_bias = "long" if side_upper == "BUY" else "short" if side_upper == "SELL" else "neutral"
    mtf_passed = True
    if active_policy.require_mtf_alignment and mtf:
        if mtf_alignment == "conflicted" or (
            mtf_bias in {"long", "short"}
            and expected_mtf_bias in {"long", "short"}
            and mtf_bias != expected_mtf_bias
        ):
            mtf_passed = False
            violations.append(
                f"Multi-timeframe trend conflicts with {side_upper}: "
                f"bias={mtf_bias}, alignment={mtf_alignment}, confidence={mtf_confidence:.2f}."
            )
        elif mtf_alignment not in {"strong", "mixed"} or mtf_confidence < active_policy.min_mtf_confidence:
            mtf_passed = False
            violations.append(
                f"Multi-timeframe trend is not strong enough for {side_upper}: "
                f"bias={mtf_bias}, alignment={mtf_alignment}, confidence={mtf_confidence:.2f}."
            )
    elif active_policy.require_mtf_alignment:
        warnings.append("Multi-timeframe trend structure is unavailable; relying on base trend and flow gates.")
    layers["multi_timeframe_trend"] = {
        "passed": mtf_passed,
        "required": active_policy.require_mtf_alignment,
        "expected_bias": expected_mtf_bias,
        "bias": mtf_bias,
        "alignment": mtf_alignment,
        "confidence": round(mtf_confidence, 4),
    }

    review_stats = _recent_review_stats(
        active_policy,
        side=side_upper,
        symbol=symbol,
        route_id=route_id,
        strategy_profile=strategy_profile,
    )
    market_bot_stats = _market_bot_review_stats(live_plan) if active_policy.allow_market_bot_evidence else None
    if market_bot_stats is not None:
        review_stats = market_bot_stats
    risk_combo_stats = _risk_combo_review_stats(live_plan) if active_policy.allow_risk_combo_evidence else None
    if risk_combo_stats is not None:
        review_stats = risk_combo_stats
    performance_passed = True
    if review_stats["count"] < active_policy.min_recent_reviews:
        message = (
            f"Only {review_stats['count']} recent closed-trade reviews are available; "
            f"professional lane requires {active_policy.min_recent_reviews}."
        )
        if active_policy.allow_thin_scoped_history:
            warnings.append(message)
        else:
            performance_passed = False
            violations.append(message)
    if review_stats["count"] >= active_policy.min_recent_reviews:
        if review_stats["win_rate"] < active_policy.min_recent_win_rate:
            warnings.append(
                f"Recent win rate {review_stats['win_rate']:.1%} is below "
                f"{active_policy.min_recent_win_rate:.1%}; keeping this descriptive, not a hard gate."
            )
        if review_stats["profit_factor"] < active_policy.min_recent_profit_factor:
            performance_passed = False
            violations.append(
                f"Recent PF {review_stats['profit_factor']:.2f} is below "
                f"{active_policy.min_recent_profit_factor:.2f}."
            )
        if review_stats["avg_r_multiple"] < active_policy.min_recent_avg_r:
            performance_passed = False
            violations.append(
                f"Recent average R {review_stats['avg_r_multiple']:.2f} is below "
                f"{active_policy.min_recent_avg_r:.2f}."
            )
        if review_stats["r_sample_count"] <= 0:
            warnings.append("Recent reviews do not include R multiples; expectancy gate is less reliable.")
        elif review_stats["expectancy_r"] < active_policy.min_recent_expectancy_r:
            performance_passed = False
            violations.append(
                f"Recent expectancy {review_stats['expectancy_r']:.2f}R is below "
                f"{active_policy.min_recent_expectancy_r:.2f}R."
            )
        if review_stats["r_sample_count"] > 0 and review_stats["payoff_ratio"] < active_policy.min_recent_payoff_ratio:
            performance_passed = False
            violations.append(
                f"Recent payoff ratio {review_stats['payoff_ratio']:.2f} is below "
                f"{active_policy.min_recent_payoff_ratio:.2f}."
            )
        if review_stats["stop_loss_ratio"] > active_policy.max_recent_stop_loss_ratio:
            message = (
                f"Recent stop-loss ratio {review_stats['stop_loss_ratio']:.1%} exceeds "
                f"{active_policy.max_recent_stop_loss_ratio:.1%}."
            )
            if review_stats["profit_factor"] < 1.0 or review_stats["expectancy_r"] < 0.0:
                performance_passed = False
                violations.append(message)
            else:
                warnings.append(message)
    last_stop = _parse_datetime(review_stats.get("last_stop_loss_at"))
    if last_stop and active_policy.stop_loss_cooldown_hours > 0:
        cooldown_until = last_stop + timedelta(hours=active_policy.stop_loss_cooldown_hours)
        if datetime.now(timezone.utc) < cooldown_until:
            performance_passed = False
            violations.append(
                "Stop-loss cooldown is active after the latest closed trade review."
            )
    layers["strategy_performance"] = {
        "passed": performance_passed,
        **review_stats,
    }

    passed = all(layer.get("passed", False) for layer in layers.values())
    return ProfessionalGateResult(
        passed=passed,
        layers=layers,
        violations=violations if active_policy.require_professional_gate else [],
        warnings=warnings + ([] if active_policy.require_professional_gate else violations),
        stats={
            "policy": active_policy.to_dict(),
            "required": active_policy.require_professional_gate,
        },
    )
