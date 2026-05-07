from __future__ import annotations

from typing import Any

from .asset_routing import AssetRoute


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def score_price_structure(latest: dict[str, Any], analysis: dict[str, Any]) -> float:
    adx = float(latest.get("adx") or 0.0)
    convergence = float(analysis.get("convergence") or 0.0) * 100.0
    score = float(analysis.get("score") or 0.0)
    realized_vol = float(latest.get("realized_vol_20") or 0.0)
    price_structure = (score * 0.45) + (convergence * 0.35) + (min(adx, 35.0) / 35.0 * 20.0)
    if realized_vol > 2.2:
        price_structure -= 8.0
    return round(_clamp(price_structure), 3)


def score_flow(latest: dict[str, Any]) -> float:
    volume_z = float(latest.get("volume_zscore_20") or 0.0)
    obv_z = float(latest.get("obv_zscore_20") or 0.0)
    whale_pressure = float(latest.get("whale_hunter_24h_oi") or 0.0)
    taker_ratio = float(latest.get("taker_buy_sell_ratio") or 1.0)
    order_book_imbalance = float(latest.get("order_book_imbalance") or 0.0)
    flow = (
        50.0
        + (volume_z * 12.0)
        + (obv_z * 10.0)
        + (whale_pressure * 18.0)
        + ((taker_ratio - 1.0) * 30.0)
        + (order_book_imbalance * 25.0)
    )
    return round(_clamp(flow), 3)


def score_event_risk(news_risk: dict[str, Any] | None = None) -> float:
    if not news_risk:
        return 50.0
    risk_level = str(news_risk.get("risk_level") or "normal")
    bias = str(news_risk.get("bias") or "neutral")
    score = 70.0
    if risk_level == "high":
        score = 25.0
    elif risk_level == "elevated":
        score = 45.0
    if bias == "bearish":
        score -= 5.0
    elif bias == "bullish":
        score += 5.0
    return round(_clamp(score), 3)


def score_execution_quality(
    latest: dict[str, Any],
    trade_plan: dict[str, Any],
    *,
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0,
) -> float:
    price = float(latest.get("close") or latest.get("price") or 0.0)
    if price <= 0.0:
        return 35.0
    long_plan = trade_plan.get("long") or {}
    invalidation = float(long_plan.get("invalidation") or 0.0)
    take_profit = float(long_plan.get("take_profit_1") or 0.0)
    take_profit_levels = long_plan.get("take_profit_levels")
    if isinstance(take_profit_levels, list):
        tp_values = [float(item) for item in take_profit_levels if float(item or 0.0) > 0.0]
    else:
        tp_values = [
            float(long_plan[key])
            for key in ("take_profit_1", "take_profit_2", "take_profit_3")
            if key in long_plan and float(long_plan.get(key) or 0.0) > 0.0
        ]
    risk_distance = abs(price - invalidation) if invalidation > 0 else 0.0
    reward_distance = abs(take_profit - price) if take_profit > 0 else 0.0
    if len(tp_values) >= 3:
        weights = (0.30, 0.35, 0.35)
    elif len(tp_values) == 2:
        weights = (0.35, 0.65)
    else:
        weights = (1.0,)
    planned_reward_distance = sum(
        abs(target - price) * weights[index]
        for index, target in enumerate(tp_values[: len(weights)])
    )
    if planned_reward_distance > reward_distance:
        reward_distance = planned_reward_distance
    reward_risk = (reward_distance / risk_distance) if risk_distance > 0 else 0.0
    execution = 40.0 + min(reward_risk, 3.0) * 15.0
    total_cost_bps = fee_bps + slippage_bps
    spread_bps = float(latest.get("spread_bps") or 0.0)
    if reward_risk < 1.0:
        execution -= 18.0
    execution -= min(total_cost_bps / 10.0, 6.0)
    execution -= min(spread_bps / 6.0, 8.0)
    return round(_clamp(execution), 3)


def score_strategy_fit(route: AssetRoute, latest: dict[str, Any], analysis: dict[str, Any]) -> float:
    adx = float(latest.get("adx") or 0.0)
    volume_z = float(latest.get("volume_zscore_20") or 0.0)
    realized_vol = float(latest.get("realized_vol_20") or 0.0)
    score = float(analysis.get("score") or 0.0)
    fit = 55.0
    if route.asset_class in {"btc_core", "eth_core", "major_alt_trend"}:
        fit += min(adx, 25.0) * 0.8
        fit += min(max(volume_z, -1.0), 2.0) * 6.0
    elif route.asset_class == "meme_high_beta":
        fit += min(realized_vol, 2.0) * 8.0
        fit += min(max(volume_z, -1.0), 2.5) * 7.0
    else:
        fit += min(adx, 20.0) * 0.5
    fit += max(score - 60.0, 0.0) * 0.3
    return round(_clamp(fit), 3)


def build_signal_scores(
    *,
    route: AssetRoute,
    latest: dict[str, Any],
    analysis: dict[str, Any],
    trade_plan: dict[str, Any],
    news_risk: dict[str, Any] | None = None,
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0,
) -> dict[str, float]:
    price_structure_score = score_price_structure(latest, analysis)
    flow_score = score_flow(latest)
    event_risk_score = score_event_risk(news_risk)
    execution_quality_score = score_execution_quality(
        latest,
        trade_plan,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    strategy_fit_score = score_strategy_fit(route, latest, analysis)
    composite = (
        price_structure_score * 0.35
        + flow_score * 0.2
        + event_risk_score * 0.1
        + execution_quality_score * 0.2
        + strategy_fit_score * 0.15
    )
    return {
        "price_structure_score": round(price_structure_score, 3),
        "flow_score": round(flow_score, 3),
        "event_risk_score": round(event_risk_score, 3),
        "execution_quality_score": round(execution_quality_score, 3),
        "strategy_fit_score": round(strategy_fit_score, 3),
        "composite_convergence_score": round(_clamp(composite), 3),
    }
