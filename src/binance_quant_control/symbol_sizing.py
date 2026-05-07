from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .asset_routing import AssetRoute
from .strategy import StrategyConfig


@dataclass(frozen=True, slots=True)
class SymbolSizingPlan:
    symbol: str
    route_id: str
    asset_class: str
    tier: str
    max_leverage: int
    recommended_leverage: int
    max_margin_pct: float
    recommended_margin_pct: float
    max_account_risk_pct: float
    recommended_margin_usdt: float
    max_margin_usdt: float
    min_notional_usdt: float
    confidence: float
    reasons: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _base_rules(asset_class: str) -> tuple[int, float, float, str]:
    if asset_class == "btc_core":
        return 30, 0.5, 0.018, "core"
    if asset_class == "eth_core":
        return 24, 0.48, 0.017, "core"
    if asset_class == "major_alt_trend":
        return 18, 0.38, 0.014, "major-alt"
    if asset_class == "meme_high_beta":
        return 12, 0.26, 0.01, "high-beta"
    if asset_class == "xau_macro":
        return 10, 0.24, 0.009, "macro"
    return 8, 0.18, 0.008, "exploratory"


def _volume_tier(rank: int | None) -> tuple[str, float, int]:
    if rank is None or rank <= 0:
        return "unranked", 0.75, 0
    if rank <= 10:
        return "top-10", 1.12, 1
    if rank <= 30:
        return "top-30", 1.0, 0
    if rank <= 60:
        return "top-60", 0.82, -1
    return "outside-top-60", 0.65, -2


def build_symbol_sizing_plan(
    *,
    symbol: str,
    route: AssetRoute,
    strategy: StrategyConfig,
    latest: dict[str, Any],
    analysis: dict[str, Any],
    equity_usdt: float,
    available_balance_usdt: float,
    min_notional_usdt: float,
    news_risk: dict[str, Any] | None = None,
    volume_rank: int | None = None,
    manual_margin_cap_usdt: float | None = None,
    signal_scores: dict[str, Any] | None = None,
    route_side_risk: dict[str, Any] | None = None,
    market_bot_promoted: bool = False,
) -> SymbolSizingPlan:
    """Build per-symbol leverage and margin rules for testnet/live planning."""

    max_lev, base_margin_pct, base_risk_pct, class_tier = _base_rules(route.asset_class)
    tier, tier_multiplier, tier_leverage_delta = _volume_tier(volume_rank)
    reasons = [f"asset-class={route.asset_class}", f"liquidity-tier={tier}"]
    warnings: list[str] = []

    realized_vol = float(latest.get("realized_vol_20") or 0.0)
    adx = float(latest.get("adx") or 0.0)
    score = float(analysis.get("score") or 0.0)
    convergence = float(analysis.get("convergence") or 0.0)
    confidence = _clamp((score / 100.0 * 0.5) + (convergence * 0.35) + (min(adx, 35.0) / 35.0 * 0.15), 0.0, 1.0)

    dynamic_max_leverage = min(max_lev, strategy.risk.max_leverage)
    leverage = max(1, dynamic_max_leverage + tier_leverage_delta)
    margin_pct = min(base_margin_pct, strategy.risk.max_notional_pct) * tier_multiplier
    risk_pct = min(base_risk_pct, strategy.risk.max_account_risk_pct)

    if confidence >= 0.9 and adx >= 25 and 0.0 < realized_vol <= 0.9:
        leverage += 2
        margin_pct *= 1.16
        reasons.append("alpha-testnet-conviction")
    elif confidence >= 0.82 and adx >= 22:
        leverage += 1
        margin_pct *= 1.12
        reasons.append("strong-confidence")
    elif confidence < 0.55 or adx < 12:
        leverage -= 1
        dynamic_max_leverage = min(dynamic_max_leverage, 3)
        margin_pct *= 0.58
        risk_pct *= 0.7
        warnings.append("Weak confidence/trend; sizing reduced.")

    if realized_vol >= 2.2:
        leverage -= 2
        dynamic_max_leverage = min(dynamic_max_leverage, 2 if route.asset_class == "meme_high_beta" else 3)
        margin_pct *= 0.45
        risk_pct *= 0.55
        warnings.append("Extreme realized volatility; sizing reduced sharply.")
    elif realized_vol >= 1.5:
        leverage -= 1
        dynamic_max_leverage = min(dynamic_max_leverage, 5)
        margin_pct *= 0.7
        risk_pct *= 0.75
        warnings.append("Elevated realized volatility; sizing reduced.")
    elif 0.0 < realized_vol <= 0.7 and adx >= 18:
        margin_pct *= 1.08
        reasons.append("controlled-volatility")

    signal_scores = signal_scores or {}
    flow_score = float(signal_scores.get("flow_score") or 0.0)
    event_risk_score = float(signal_scores.get("event_risk_score") or 0.0)
    composite_signal_score = float(
        signal_scores.get("composite_convergence_score")
        or signal_scores.get("score")
        or 0.0
    )
    if signal_scores:
        if flow_score < 40.0:
            leverage -= 1
            dynamic_max_leverage = min(dynamic_max_leverage, 3)
            margin_pct *= 0.55
            risk_pct *= 0.65
            warnings.append("Weak flow confirmation; leverage and margin reduced.")
        elif flow_score < 55.0:
            margin_pct *= 0.78
            risk_pct *= 0.85
            warnings.append("Flow confirmation is only moderate; margin reduced.")
        elif flow_score >= 68.0 and composite_signal_score >= 72.0 and adx >= 18.0:
            margin_pct *= 1.08
            reasons.append("flow-confirmed")

        if event_risk_score and event_risk_score < 35.0:
            leverage -= 1
            dynamic_max_leverage = min(dynamic_max_leverage, 4)
            margin_pct *= 0.7
            risk_pct *= 0.8
            warnings.append("Macro/news event score is weak; sizing reduced.")
            if route.asset_class in {"defensive_unknown", "meme_high_beta"}:
                leverage -= 1
                dynamic_max_leverage = min(dynamic_max_leverage, 1)
                margin_pct *= 0.55
                risk_pct *= 0.65
                warnings.append(
                    "Weak event score on high-uncertainty symbol; forced to 1x exploratory sizing."
                )
        if composite_signal_score and composite_signal_score < 58.0:
            leverage -= 1
            dynamic_max_leverage = min(dynamic_max_leverage, 4)
            margin_pct *= 0.72
            risk_pct *= 0.8
            warnings.append("Composite signal quality is weak; sizing reduced.")
        elif (
            composite_signal_score >= 78.0
            and flow_score >= 62.0
            and (event_risk_score == 0.0 or event_risk_score >= 45.0)
            and adx >= 22.0
        ):
            leverage += 1
            margin_pct *= 1.08
            reasons.append("multi-factor-confirmation")

    risk_level = str((news_risk or {}).get("risk_level") or "normal")
    if risk_level == "high":
        leverage -= 1
        dynamic_max_leverage = min(dynamic_max_leverage, 4)
        if realized_vol >= 2.2:
            dynamic_max_leverage = min(dynamic_max_leverage, 1)
        margin_pct *= 0.55
        risk_pct *= 0.65
        warnings.append("High news risk; testnet sizing only.")
    elif risk_level == "elevated":
        dynamic_max_leverage = min(dynamic_max_leverage, 8)
        margin_pct *= 0.8
        risk_pct *= 0.85
        warnings.append("Elevated news risk; sizing reduced.")

    route_side_risk = route_side_risk or {}
    route_side_samples = int(route_side_risk.get("sample_count") or 0)
    route_side_pf = float(route_side_risk.get("profit_factor") or 0.0)
    route_side_net = float(route_side_risk.get("net_pnl_usdt") or 0.0)
    route_side_stop_ratio = float(route_side_risk.get("stop_loss_ratio") or 0.0)
    route_side_avg_r = float(route_side_risk.get("avg_r_multiple") or 0.0)
    if (
        route_side_samples >= 30
        and route_side_net < 0.0
        and route_side_pf < 0.8
        and not market_bot_promoted
    ):
        leverage -= 1
        dynamic_max_leverage = min(dynamic_max_leverage, 2)
        margin_pct *= 0.55
        risk_pct *= 0.65
        warnings.append(
            "Route/side historical feedback is negative; testnet sizing reduced."
        )
    elif (
        route_side_samples >= 30
        and route_side_net < 0.0
        and route_side_pf < 1.0
        and not market_bot_promoted
    ):
        dynamic_max_leverage = min(dynamic_max_leverage, 5)
        margin_pct *= 0.72
        risk_pct *= 0.8
        warnings.append(
            "Route/side historical feedback is still below breakeven; margin reduced."
        )
    elif route_side_samples >= 30 and route_side_net < 0.0 and route_side_pf < 1.0:
        margin_pct *= 0.9
        risk_pct *= 0.9
        warnings.append(
            "Route/side historical feedback is advisory because market-bot evidence promoted this candidate."
        )
    elif route_side_samples >= 30 and route_side_net > 0.0 and route_side_pf >= 1.2:
        leverage += 1
        margin_pct *= 1.1
        reasons.append("route-side-positive-feedback")

    if route_side_samples >= 30 and route_side_stop_ratio >= 70.0:
        leverage -= 1
        dynamic_max_leverage = min(dynamic_max_leverage, 3)
        margin_pct *= 0.72
        risk_pct *= 0.8
        warnings.append("Route/side stop-loss ratio is high; leverage reduced.")
    if route_side_samples >= 30 and route_side_avg_r < -0.2:
        margin_pct *= 0.82
        risk_pct *= 0.9
        warnings.append("Route/side average R is negative; risk budget reduced.")

    leverage = int(_clamp(leverage, 1, dynamic_max_leverage))
    margin_pct = _clamp(margin_pct, 0.02, min(strategy.risk.max_notional_pct, 0.6))
    risk_pct = _clamp(risk_pct, 0.0025, min(strategy.risk.max_account_risk_pct, 0.03))

    balance_cap = min(max(available_balance_usdt, 0.0), max(equity_usdt, 0.0) * margin_pct)
    if strategy.execution.margin_notional_usdt is not None:
        balance_cap = min(balance_cap, strategy.execution.margin_notional_usdt)
    if manual_margin_cap_usdt is not None:
        balance_cap = min(balance_cap, manual_margin_cap_usdt)

    minimum_margin = min_notional_usdt / leverage if leverage > 0 else min_notional_usdt
    if (
        min_notional_usdt > 0.0
        and available_balance_usdt > 0.0
        and leverage < dynamic_max_leverage
        and available_balance_usdt * leverage < min_notional_usdt
    ):
        required_leverage = int((min_notional_usdt / available_balance_usdt) + 0.999999)
        if required_leverage <= dynamic_max_leverage:
            leverage = required_leverage
            minimum_margin = min_notional_usdt / leverage
            reasons.append("leverage-raised-to-satisfy-exchange-min-notional")
    recommended_margin = max(balance_cap, minimum_margin) if available_balance_usdt >= minimum_margin else balance_cap
    recommended_margin = min(recommended_margin, available_balance_usdt)

    if recommended_margin < minimum_margin:
        warnings.append(
            f"Available balance cannot satisfy exchange minimum margin {minimum_margin:.4f} USDT."
        )

    return SymbolSizingPlan(
        symbol=symbol.upper(),
        route_id=route.route_id,
        asset_class=route.asset_class,
        tier=f"{class_tier}:{tier}",
        max_leverage=int(min(max_lev, strategy.risk.max_leverage)),
        recommended_leverage=leverage,
        max_margin_pct=round(min(strategy.risk.max_notional_pct, 0.6), 4),
        recommended_margin_pct=round(margin_pct, 4),
        max_account_risk_pct=round(risk_pct, 6),
        recommended_margin_usdt=round(recommended_margin, 6),
        max_margin_usdt=round(balance_cap, 6),
        min_notional_usdt=round(min_notional_usdt, 6),
        confidence=round(confidence, 4),
        reasons=reasons,
        warnings=warnings,
    )
