from __future__ import annotations

import pandas as pd

from binance_quant_control.research_entry_gate import (
    ResearchEntryGateConfig,
    build_research_entry_gate,
)


def test_research_entry_gate_blocks_weak_route_side_history() -> None:
    reviews = [
        {
            "route_id": "btc-core",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "analysis_score": 10,
            "analysis_convergence": 0.95,
            "realized_pnl_usdt": -1.0,
            "exit_reason": "stop_loss",
        },
        {
            "route_id": "btc-core",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "analysis_score": 12,
            "analysis_convergence": 0.94,
            "realized_pnl_usdt": -0.5,
            "exit_reason": "stop_loss",
        },
        {
            "route_id": "btc-core",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "analysis_score": 8,
            "analysis_convergence": 0.93,
            "realized_pnl_usdt": 0.1,
            "exit_reason": "take_profit",
        },
    ]
    entry_gate, metadata = build_research_entry_gate(
        route_id="btc-core",
        symbol="BTCUSDT",
        config=ResearchEntryGateConfig(
            enabled=True,
            route_side_min_samples=3,
            historical_signal_veto=False,
        ),
        reviews=reviews,
    )

    assert entry_gate is not None
    allowed, reason = entry_gate(
        pd.Series(),
        pd.Series(),
        {"recommended_action": "SELL", "score": 10, "convergence": 0.95},
        1,
    )

    assert allowed is False
    assert reason == "route-side-history-veto"
    assert metadata["route_side"]["SELL"]["allowed"] is False


def test_research_entry_gate_blocks_negative_signal_bucket() -> None:
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
    entry_gate, _metadata = build_research_entry_gate(
        route_id="major-alt-trend",
        symbol="NEARUSDT",
        config=ResearchEntryGateConfig(
            enabled=True,
            route_side_veto=False,
            historical_signal_min_samples=3,
        ),
        reviews=reviews,
    )

    assert entry_gate is not None
    allowed, reason = entry_gate(
        pd.Series(),
        pd.Series(),
        {"recommended_action": "SELL", "score": 10, "convergence": 0.95},
        1,
    )

    assert allowed is False
    assert reason == "historical-feedback-bucket-veto"


def test_research_entry_gate_shadow_mode_records_history_without_veto() -> None:
    reviews = [
        {
            "route_id": "btc-core",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "analysis_score": 10,
            "analysis_convergence": 0.95,
            "realized_pnl_usdt": -1.0,
            "exit_reason": "stop_loss",
        },
        {
            "route_id": "btc-core",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "analysis_score": 12,
            "analysis_convergence": 0.94,
            "realized_pnl_usdt": -0.5,
            "exit_reason": "stop_loss",
        },
        {
            "route_id": "btc-core",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "analysis_score": 8,
            "analysis_convergence": 0.93,
            "realized_pnl_usdt": 0.1,
            "exit_reason": "take_profit",
        },
    ]
    entry_gate, metadata = build_research_entry_gate(
        route_id="btc-core",
        symbol="BTCUSDT",
        config=ResearchEntryGateConfig(
            enabled=True,
            route_side_veto=False,
            historical_signal_veto=False,
            shadow_route_side_veto=True,
            shadow_historical_signal_veto=True,
            route_side_min_samples=3,
            historical_signal_min_samples=3,
        ),
        reviews=reviews,
    )

    assert entry_gate is not None
    allowed, reason = entry_gate(
        pd.Series(),
        pd.Series(),
        {"recommended_action": "SELL", "score": 10, "convergence": 0.95},
        1,
    )

    assert allowed is True
    assert reason == ""
    assert metadata["route_side"]["SELL"]["allowed"] is False
    assert metadata["historical_signal"]["shadow_only"] is True
