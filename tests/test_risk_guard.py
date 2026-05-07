from __future__ import annotations

from binance_quant_control.risk_guard import check_order_allowed


def test_maturity_mode_allows_large_notional_when_liquidation_buffer_is_safe() -> None:
    result = check_order_allowed(
        side="BUY",
        margin_notional_usdt=100.0,
        leverage=25,
        account_balance_usdt=100.0,
        account_risk_pct=0.01,
        analysis_convergence=0.9,
        analysis_score=90,
        adx_value=30.0,
        daily_trade_count=0,
        consecutive_losses=0,
        last_loss_at=None,
        max_account_risk_pct=0.02,
        max_leverage=125,
        max_notional_pct=0.1,
        min_balance_usdt=1.0,
        min_convergence=0.7,
        min_score_long=70,
        max_score_short=30,
        min_adx=20.0,
        liquidation_buffer_pct=0.04,
        min_liquidation_buffer_pct=0.03,
    )

    assert result.allowed is True
    assert any("Maturity mode" in item for item in result.warnings)


def test_maturity_mode_blocks_when_liquidation_buffer_is_too_small() -> None:
    result = check_order_allowed(
        side="BUY",
        margin_notional_usdt=100.0,
        leverage=25,
        account_balance_usdt=100.0,
        account_risk_pct=0.01,
        analysis_convergence=0.9,
        analysis_score=90,
        adx_value=30.0,
        daily_trade_count=0,
        consecutive_losses=0,
        last_loss_at=None,
        max_account_risk_pct=0.02,
        max_leverage=125,
        max_notional_pct=0.1,
        min_balance_usdt=1.0,
        min_convergence=0.7,
        min_score_long=70,
        max_score_short=30,
        min_adx=20.0,
        liquidation_buffer_pct=0.005,
        min_liquidation_buffer_pct=0.03,
    )

    assert result.allowed is False
    assert any("Liquidation buffer" in item for item in result.violations)
