from __future__ import annotations

from binance_quant_control.side_risk_policy import evaluate_route_side_risk


def test_route_side_risk_blocks_enough_samples_below_pf_floor() -> None:
    reviews = [
        {"route_id": "btc-core", "side": "SELL", "realized_pnl_usdt": -1.0},
        {"route_id": "btc-core", "side": "SELL", "realized_pnl_usdt": -1.0},
        {"route_id": "btc-core", "side": "SELL", "realized_pnl_usdt": 0.5},
    ]

    result = evaluate_route_side_risk(
        route_id="btc-core",
        side="SELL",
        min_samples=3,
        min_profit_factor=0.8,
        reviews=reviews,
    )

    assert result.allowed is False
    assert result.sample_count == 3
    assert result.profit_factor == 0.25
    assert any("historical PF" in item for item in result.reasons)


def test_route_side_risk_does_not_block_thin_samples() -> None:
    reviews = [
        {"route_id": "meme-high-beta", "side": "BUY", "realized_pnl_usdt": -1.0},
        {"route_id": "meme-high-beta", "side": "BUY", "realized_pnl_usdt": 0.1},
    ]

    result = evaluate_route_side_risk(
        route_id="meme-high-beta",
        side="BUY",
        min_samples=3,
        min_profit_factor=0.8,
        reviews=reviews,
    )

    assert result.allowed is True
    assert result.sample_count == 2
    assert result.reasons == []


def test_route_side_risk_blocks_excessive_stop_loss_ratio() -> None:
    reviews = [
        {
            "route_id": "major-alt-trend",
            "side": "SELL",
            "realized_pnl_usdt": -1.0,
            "exit_reason": "stop_loss",
        },
        {
            "route_id": "major-alt-trend",
            "side": "SELL",
            "realized_pnl_usdt": -1.0,
            "exit_reason": "stop_loss",
        },
        {
            "route_id": "major-alt-trend",
            "side": "SELL",
            "realized_pnl_usdt": 0.1,
            "exit_reason": "take_profit",
        },
    ]

    result = evaluate_route_side_risk(
        route_id="major-alt-trend",
        side="SELL",
        min_samples=3,
        min_profit_factor=0.8,
        max_stop_loss_ratio=60.0,
        reviews=reviews,
    )

    assert result.allowed is False
    assert result.stop_loss_ratio == 66.66666666666666
    assert any("stop-loss ratio" in item for item in result.reasons)
