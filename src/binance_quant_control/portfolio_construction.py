from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PortfolioConstructionPolicy:
    max_total_open_risk_pct: float = 0.03
    max_symbol_open_risk_pct: float = 0.006
    max_group_open_risk_pct: float = 0.012
    max_same_beta_directional_risk_pct: float = 0.012
    max_concurrent_positions: int = 4
    min_signal_score: float = 0.0
    min_expectancy_r: float = 0.10
    min_payoff_ratio: float = 1.15

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PortfolioRiskSnapshot:
    total_open_risk_pct: float
    open_count: int
    by_symbol: dict[str, float]
    by_group: dict[str, float]
    by_beta_direction: dict[str, float]
    remaining_total_risk_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PortfolioTarget:
    symbol: str
    side: str
    route_id: str
    correlation_group: str
    target_risk_pct: float
    signal_score: float
    expectancy_r: float
    payoff_ratio: float
    accepted: bool
    blockers: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _open_position_risk(
    open_positions: list[dict[str, Any]],
) -> tuple[float, dict[str, float], dict[str, float], dict[str, float], int]:
    total = 0.0
    by_symbol: dict[str, float] = defaultdict(float)
    by_group: dict[str, float] = defaultdict(float)
    by_beta_direction: dict[str, float] = defaultdict(float)
    open_count = 0
    for row in open_positions:
        quantity = abs(_float(row.get("quantity") or row.get("qty") or row.get("positionAmt")))
        if quantity <= 0.0:
            continue
        symbol = str(row.get("symbol") or "").upper()
        side = str(row.get("side") or row.get("position_side") or "").upper()
        group = str(row.get("correlation_group") or row.get("route_id") or symbol or "unknown")
        beta_group = str(row.get("beta_group") or row.get("correlation_group") or row.get("route_id") or symbol or "unknown")
        risk = max(_float(row.get("open_risk_pct") or row.get("planned_account_risk_pct")), 0.0)
        total += risk
        by_symbol[symbol] += risk
        by_group[group] += risk
        if side in {"BUY", "SELL", "LONG", "SHORT"}:
            normalized_side = "BUY" if side in {"BUY", "LONG"} else "SELL"
            by_beta_direction[f"{beta_group}:{normalized_side}"] += risk
        open_count += 1
    return total, dict(by_symbol), dict(by_group), dict(by_beta_direction), open_count


def build_portfolio_risk_snapshot(
    open_positions: list[dict[str, Any]] | None = None,
    *,
    policy: PortfolioConstructionPolicy | None = None,
) -> PortfolioRiskSnapshot:
    policy = policy or PortfolioConstructionPolicy()
    total_risk, symbol_risk, group_risk, beta_direction_risk, open_count = _open_position_risk(open_positions or [])
    return PortfolioRiskSnapshot(
        total_open_risk_pct=round(total_risk, 6),
        open_count=open_count,
        by_symbol={key: round(value, 6) for key, value in sorted(symbol_risk.items())},
        by_group={key: round(value, 6) for key, value in sorted(group_risk.items())},
        by_beta_direction={key: round(value, 6) for key, value in sorted(beta_direction_risk.items())},
        remaining_total_risk_pct=round(max(policy.max_total_open_risk_pct - total_risk, 0.0), 6),
    )


def build_portfolio_target(
    signal: dict[str, Any],
    *,
    open_positions: list[dict[str, Any]] | None = None,
    policy: PortfolioConstructionPolicy | None = None,
) -> PortfolioTarget:
    policy = policy or PortfolioConstructionPolicy()
    open_positions = open_positions or []
    symbol = str(signal.get("symbol") or "").upper()
    side = str(signal.get("side") or "").upper()
    route_id = str(signal.get("route_id") or "unrouted")
    group = str(signal.get("correlation_group") or route_id or symbol or "unknown")
    beta_group = str(signal.get("beta_group") or group)
    requested_risk = max(_float(signal.get("target_risk_pct"), policy.max_symbol_open_risk_pct), 0.0)
    signal_score = _float(signal.get("signal_score"))
    expectancy_r = _float(signal.get("expectancy_r"))
    payoff_ratio = _float(signal.get("payoff_ratio"))
    total_risk, symbol_risk, group_risk, beta_direction_risk, open_count = _open_position_risk(open_positions)
    target_risk = min(requested_risk, policy.max_symbol_open_risk_pct)
    beta_direction_key = f"{beta_group}:{side}"
    blockers: list[str] = []
    if not symbol:
        blockers.append("symbol-missing")
    if side not in {"BUY", "SELL"}:
        blockers.append("side-invalid")
    if open_count >= policy.max_concurrent_positions:
        blockers.append("max-concurrent-positions-reached")
    if total_risk + target_risk > policy.max_total_open_risk_pct:
        blockers.append("portfolio-open-risk-above-cap")
    if symbol_risk.get(symbol, 0.0) + target_risk > policy.max_symbol_open_risk_pct:
        blockers.append("symbol-open-risk-above-cap")
    if group_risk.get(group, 0.0) + target_risk > policy.max_group_open_risk_pct:
        blockers.append("correlation-group-open-risk-above-cap")
    if beta_direction_risk.get(beta_direction_key, 0.0) + target_risk > policy.max_same_beta_directional_risk_pct:
        blockers.append("same-beta-directional-risk-above-cap")
    if signal_score < policy.min_signal_score:
        blockers.append("signal-score-below-floor")
    if expectancy_r < policy.min_expectancy_r:
        blockers.append("expectancy-r-below-floor")
    if payoff_ratio < policy.min_payoff_ratio:
        blockers.append("payoff-ratio-below-floor")
    return PortfolioTarget(
        symbol=symbol,
        side=side,
        route_id=route_id,
        correlation_group=group,
        target_risk_pct=round(target_risk, 6),
        signal_score=round(signal_score, 6),
        expectancy_r=round(expectancy_r, 6),
        payoff_ratio=round(payoff_ratio, 6),
        accepted=not blockers,
        blockers=blockers,
    )
