from __future__ import annotations

from binance_quant_control.professional_entry_gate import (
    ProfessionalGatePolicy,
    evaluate_professional_entry_gate,
)


def _live_plan() -> dict[str, float | int | str]:
    return {
        "side": "BUY",
        "price": 100.0,
        "quantity": 2.0,
        "stop_price": 98.0,
        "take_profit_price": 103.0,
        "take_profit_prices": [102.4, 104.8],
        "take_profit_quantities": [0.8, 1.2],
        "planned_account_risk_usdt": 4.0,
        "analysis_score": 82,
        "analysis_convergence": 0.82,
        "adx_value": 28.0,
        "fee_bps": 4.0,
        "slippage_bps": 2.0,
    }


def _market_bot_plan() -> dict[str, object]:
    plan = dict(_live_plan())
    plan["market_bot_gate"] = {
        "allowed": True,
        "report_path": "state/market-bot-gate.json",
        "matched_row": {
            "symbol": "DOGEUSDT",
            "cohort_id": "DOGEUSDT:4h:ai_family_router",
            "trade_count": 203,
            "win_rate": 30.54,
            "stop_loss_ratio": 50.25,
            "profit_factor": 1.4878,
            "expectancy_r": 0.263,
            "payoff_ratio": 3.3836,
        },
    }
    return plan


def _latest() -> dict[str, float]:
    return {
        "realized_vol_20": 0.8,
        "volume_zscore_20": 0.2,
        "obv_zscore_20": 0.7,
        "bb_bandwidth": 0.08,
    }


def test_professional_gate_passes_clean_setup(monkeypatch) -> None:
    monkeypatch.setattr(
        "binance_quant_control.professional_entry_gate.read_closed_trade_reviews",
        lambda: [
            {"exit_reason": "take_profit", "realized_pnl_usdt": 1, "realized_r_multiple": 1.2},
            {"exit_reason": "manual_close", "realized_pnl_usdt": 1, "realized_r_multiple": 0.8},
            {"exit_reason": "take_profit", "realized_pnl_usdt": 1, "realized_r_multiple": 1.5},
            {"exit_reason": "stop_loss", "realized_pnl_usdt": -1, "realized_r_multiple": -1.0},
            {"exit_reason": "take_profit", "realized_pnl_usdt": 1, "realized_r_multiple": 1.1},
            {"exit_reason": "manual_close", "realized_pnl_usdt": 1, "realized_r_multiple": 0.4},
        ],
    )

    result = evaluate_professional_entry_gate(
        side="BUY",
        latest=_latest(),
        live_plan=_live_plan(),
        policy=ProfessionalGatePolicy(stop_loss_cooldown_hours=0.0),
    )

    assert result.passed is True
    assert result.violations == []
    assert result.layers["execution_quality"]["passed"] is True
    assert result.layers["strategy_performance"]["win_rate"] > 0.5


def test_professional_gate_blocks_cost_inefficient_trade(monkeypatch) -> None:
    monkeypatch.setattr(
        "binance_quant_control.professional_entry_gate.read_closed_trade_reviews",
        lambda: [
            {"exit_reason": "take_profit", "realized_pnl_usdt": 1, "realized_r_multiple": 1.0}
            for _ in range(6)
        ],
    )
    plan = _live_plan()
    plan["take_profit_price"] = 100.6
    plan["take_profit_prices"] = [100.6]
    plan["take_profit_quantities"] = [2.0]

    result = evaluate_professional_entry_gate(
        side="BUY",
        latest=_latest(),
        live_plan=plan,
        policy=ProfessionalGatePolicy(stop_loss_cooldown_hours=0.0),
    )

    assert result.passed is False
    assert any("Reward/risk" in item for item in result.violations)


def test_professional_gate_blocks_weak_recent_strategy(monkeypatch) -> None:
    monkeypatch.setattr(
        "binance_quant_control.professional_entry_gate.read_closed_trade_reviews",
        lambda: [
            {"exit_reason": "stop_loss", "realized_pnl_usdt": -1, "realized_r_multiple": -1.0}
            for _ in range(6)
        ],
    )

    result = evaluate_professional_entry_gate(
        side="BUY",
        latest=_latest(),
        live_plan=_live_plan(),
        policy=ProfessionalGatePolicy(stop_loss_cooldown_hours=0.0),
    )

    assert result.passed is False
    assert any("PF" in item or "expectancy" in item or "payoff" in item for item in result.violations)


def test_professional_gate_uses_win_rate_as_descriptive_not_hard_gate(monkeypatch) -> None:
    monkeypatch.setattr(
        "binance_quant_control.professional_entry_gate.read_closed_trade_reviews",
        lambda: [
            {"exit_reason": "take_profit", "realized_pnl_usdt": 3, "realized_r_multiple": 3.0},
            {"exit_reason": "manual_close", "realized_pnl_usdt": -1, "realized_r_multiple": -1.0},
            {"exit_reason": "manual_close", "realized_pnl_usdt": -1, "realized_r_multiple": -1.0},
            {"exit_reason": "manual_close", "realized_pnl_usdt": -1, "realized_r_multiple": -1.0},
            {"exit_reason": "take_profit", "realized_pnl_usdt": 3, "realized_r_multiple": 3.0},
            {"exit_reason": "manual_close", "realized_pnl_usdt": -1, "realized_r_multiple": -1.0},
        ],
    )

    result = evaluate_professional_entry_gate(
        side="BUY",
        latest=_latest(),
        live_plan=_live_plan(),
        policy=ProfessionalGatePolicy(
            min_recent_win_rate=0.65,
            min_recent_profit_factor=1.0,
            min_recent_expectancy_r=0.0,
            min_recent_payoff_ratio=1.0,
            stop_loss_cooldown_hours=0.0,
        ),
    )

    assert result.passed is True
    assert any("win rate" in item.lower() for item in result.warnings)


def test_professional_gate_scopes_history_before_global(monkeypatch) -> None:
    monkeypatch.setattr(
        "binance_quant_control.professional_entry_gate.read_closed_trade_reviews",
        lambda: [
            {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "route_id": "btc-core",
                "exit_reason": "stop_loss",
                "realized_pnl_usdt": -1,
                "realized_r_multiple": -1.0,
            }
            for _ in range(10)
        ]
        + [
            {
                "symbol": "NEARUSDT",
                "side": "BUY",
                "route_id": "major-alt-trend",
                "exit_reason": "take_profit",
                "realized_pnl_usdt": 2,
                "realized_r_multiple": 2.0,
            }
            for _ in range(6)
        ],
    )

    result = evaluate_professional_entry_gate(
        side="BUY",
        latest=_latest(),
        live_plan=_live_plan(),
        symbol="NEARUSDT",
        route_id="major-alt-trend",
        policy=ProfessionalGatePolicy(stop_loss_cooldown_hours=0.0),
    )

    assert result.passed is True
    assert result.layers["strategy_performance"]["scope"] == "symbol_side"


def test_professional_gate_can_use_market_bot_evidence_for_promoted_testnet_candidate(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "binance_quant_control.professional_entry_gate.read_closed_trade_reviews",
        lambda: [
            {
                "symbol": "DOGEUSDT",
                "side": "BUY",
                "route_id": "doge-meme-high-beta",
                "exit_reason": "stop_loss",
                "realized_pnl_usdt": -1,
                "realized_r_multiple": -1.0,
            }
            for _ in range(6)
        ],
    )

    result = evaluate_professional_entry_gate(
        side="BUY",
        latest=_latest(),
        live_plan=_market_bot_plan(),
        symbol="DOGEUSDT",
        route_id="doge-meme-high-beta",
        policy=ProfessionalGatePolicy(
            min_recent_profit_factor=0.85,
            min_recent_avg_r=-0.05,
            min_recent_expectancy_r=-0.05,
            min_recent_payoff_ratio=0.9,
            max_recent_stop_loss_ratio=0.65,
            stop_loss_cooldown_hours=0.0,
            allow_market_bot_evidence=True,
        ),
    )

    assert result.passed is True
    assert result.layers["strategy_performance"]["scope"] == "market_bot_gate"
    assert result.layers["strategy_performance"]["count"] == 203


def test_professional_gate_still_blocks_bad_market_bot_evidence(monkeypatch) -> None:
    monkeypatch.setattr(
        "binance_quant_control.professional_entry_gate.read_closed_trade_reviews",
        lambda: [],
    )
    plan = _market_bot_plan()
    plan["market_bot_gate"]["matched_row"]["profit_factor"] = 0.5  # type: ignore[index]
    plan["market_bot_gate"]["matched_row"]["expectancy_r"] = -0.2  # type: ignore[index]

    result = evaluate_professional_entry_gate(
        side="BUY",
        latest=_latest(),
        live_plan=plan,
        policy=ProfessionalGatePolicy(
            min_recent_profit_factor=0.85,
            min_recent_avg_r=-0.05,
            min_recent_expectancy_r=-0.05,
            stop_loss_cooldown_hours=0.0,
            allow_market_bot_evidence=True,
        ),
    )

    assert result.passed is False
    assert any("Recent PF" in item for item in result.violations)
