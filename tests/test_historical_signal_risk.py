from __future__ import annotations

from binance_quant_control.historical_signal_risk import evaluate_historical_signal_risk


def test_historical_signal_risk_blocks_negative_route_side_score_bucket() -> None:
    reviews = [
        {
            "route_id": "major-alt-trend",
            "symbol": "NEARUSDT",
            "side": "SELL",
            "analysis_score": 12,
            "analysis_convergence": 0.94,
            "realized_pnl_usdt": -1.0,
        },
        {
            "route_id": "major-alt-trend",
            "symbol": "NEARUSDT",
            "side": "SELL",
            "analysis_score": 18,
            "analysis_convergence": 0.92,
            "realized_pnl_usdt": -1.0,
        },
        {
            "route_id": "major-alt-trend",
            "symbol": "NEARUSDT",
            "side": "SELL",
            "analysis_score": 8,
            "analysis_convergence": 0.91,
            "realized_pnl_usdt": 0.25,
        },
    ]

    result = evaluate_historical_signal_risk(
        route_id="major-alt-trend",
        symbol="NEARUSDT",
        side="SELL",
        score=10,
        convergence=0.95,
        min_samples=3,
        min_profit_factor=0.8,
        reviews=reviews,
    )

    assert result.allowed is False
    assert result.score_bin == "score-000-020"
    assert result.convergence_bin == "conv-090-100"
    assert any("route-side-score-convergence" in item for item in result.reasons)
    assert any("route-side-score" in item for item in result.reasons)
    assert any(bucket.blocked for bucket in result.buckets)


def test_historical_signal_risk_does_not_block_thin_bucket() -> None:
    reviews = [
        {
            "route_id": "btc-core",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "analysis_score": 90,
            "analysis_convergence": 0.85,
            "realized_pnl_usdt": -1.0,
        },
        {
            "route_id": "btc-core",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "analysis_score": 95,
            "analysis_convergence": 0.87,
            "realized_pnl_usdt": 0.1,
        },
    ]

    result = evaluate_historical_signal_risk(
        route_id="btc-core",
        symbol="BTCUSDT",
        side="BUY",
        score=92,
        convergence=0.86,
        min_samples=3,
        min_profit_factor=0.8,
        reviews=reviews,
    )

    assert result.allowed is True
    assert result.reasons == []


def test_historical_signal_risk_specific_positive_bucket_overrides_coarse_convergence() -> None:
    reviews = [
        *[
            {
                "route_id": "major-alt-trend",
                "symbol": "NEARUSDT",
                "side": "SELL",
                "analysis_score": 15,
                "analysis_convergence": 0.94,
                "realized_pnl_usdt": -1.0,
            }
            for _ in range(22)
        ],
        *[
            {
                "route_id": "major-alt-trend",
                "symbol": "NEARUSDT",
                "side": "SELL",
                "analysis_score": 30,
                "analysis_convergence": 0.85,
                "realized_pnl_usdt": 0.5,
            }
            for _ in range(12)
        ],
        *[
            {
                "route_id": "major-alt-trend",
                "symbol": "NEARUSDT",
                "side": "SELL",
                "analysis_score": 30,
                "analysis_convergence": 0.85,
                "realized_pnl_usdt": -0.25,
            }
            for _ in range(3)
        ],
        *[
            {
                "route_id": "major-alt-trend",
                "symbol": "NEARUSDT",
                "side": "SELL",
                "analysis_score": 18,
                "analysis_convergence": 0.85,
                "realized_pnl_usdt": -0.5,
            }
            for _ in range(20)
        ],
    ]

    result = evaluate_historical_signal_risk(
        route_id="major-alt-trend",
        symbol="NEARUSDT",
        side="SELL",
        score=30,
        convergence=0.85,
        min_samples=10,
        min_profit_factor=0.8,
        reviews=reviews,
    )

    assert result.allowed is True
    assert result.score_bin == "score-021-040"
    assert result.convergence_bin == "conv-080-089"
    assert result.reasons == []
