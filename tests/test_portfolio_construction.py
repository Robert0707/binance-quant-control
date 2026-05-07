from __future__ import annotations

from binance_quant_control.portfolio_construction import (
    PortfolioConstructionPolicy,
    build_portfolio_risk_snapshot,
    build_portfolio_target,
)


def test_portfolio_target_accepts_positive_expectancy_signal_under_risk_caps() -> None:
    target = build_portfolio_target(
        {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "route_id": "btc-core",
            "correlation_group": "majors",
            "target_risk_pct": 0.004,
            "signal_score": 1.2,
            "expectancy_r": 0.18,
            "payoff_ratio": 1.4,
        },
        policy=PortfolioConstructionPolicy(max_total_open_risk_pct=0.02, max_group_open_risk_pct=0.01),
    )

    assert target.accepted is True
    assert target.blockers == []
    assert target.target_risk_pct == 0.004


def test_portfolio_target_blocks_group_and_expectancy_risk() -> None:
    target = build_portfolio_target(
        {
            "symbol": "ETHUSDT",
            "side": "BUY",
            "route_id": "eth-core",
            "correlation_group": "majors",
            "target_risk_pct": 0.006,
            "signal_score": 1.0,
            "expectancy_r": -0.02,
            "payoff_ratio": 0.8,
        },
        open_positions=[
            {
                "symbol": "BTCUSDT",
                "quantity": 1.0,
                "open_risk_pct": 0.008,
                "correlation_group": "majors",
            }
        ],
        policy=PortfolioConstructionPolicy(max_group_open_risk_pct=0.01),
    )

    assert target.accepted is False
    assert "correlation-group-open-risk-above-cap" in target.blockers
    assert "expectancy-r-below-floor" in target.blockers
    assert "payoff-ratio-below-floor" in target.blockers


def test_portfolio_risk_snapshot_summarizes_open_risk_budget() -> None:
    snapshot = build_portfolio_risk_snapshot(
        [
            {"symbol": "BTCUSDT", "quantity": 1.0, "open_risk_pct": 0.006, "correlation_group": "majors"},
            {"symbol": "ETHUSDT", "quantity": 2.0, "open_risk_pct": 0.004, "correlation_group": "majors"},
            {"symbol": "SOLUSDT", "quantity": 0.0, "open_risk_pct": 0.004, "correlation_group": "alts"},
        ],
        policy=PortfolioConstructionPolicy(max_total_open_risk_pct=0.03),
    )

    assert snapshot.total_open_risk_pct == 0.01
    assert snapshot.open_count == 2
    assert snapshot.by_group["majors"] == 0.01
    assert snapshot.remaining_total_risk_pct == 0.02


def test_portfolio_target_blocks_same_beta_directional_exposure() -> None:
    target = build_portfolio_target(
        {
            "symbol": "ETHUSDT",
            "side": "BUY",
            "route_id": "eth-core",
            "correlation_group": "majors",
            "beta_group": "btc_beta",
            "target_risk_pct": 0.004,
            "signal_score": 1.4,
            "expectancy_r": 0.2,
            "payoff_ratio": 1.5,
        },
        open_positions=[
            {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 1.0,
                "open_risk_pct": 0.006,
                "correlation_group": "majors",
                "beta_group": "btc_beta",
            }
        ],
        policy=PortfolioConstructionPolicy(max_same_beta_directional_risk_pct=0.008),
    )

    assert target.accepted is False
    assert "same-beta-directional-risk-above-cap" in target.blockers
