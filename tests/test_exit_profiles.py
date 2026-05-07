from __future__ import annotations

from binance_quant_control.exit_profiles import runner_stop_after_target, staged_take_profit_weights


def test_payoff_runner_leaves_more_size_for_late_targets_than_balanced_profile() -> None:
    balanced = staged_take_profit_weights(
        3,
        exit_profile="balanced",
        trailing_stop_enabled=True,
        confidence=0.9,
        strategy_family="trend_continuation",
    )
    payoff = staged_take_profit_weights(
        3,
        exit_profile="payoff_runner",
        trailing_stop_enabled=True,
        confidence=0.9,
        strategy_family="trend_continuation",
    )

    assert payoff[0] < balanced[0]
    assert (payoff[1] + payoff[2] + (1.0 - sum(payoff))) > (balanced[1] + balanced[2] + (1.0 - sum(balanced)))
    assert (1.0 - sum(payoff)) > (1.0 - sum(balanced))


def test_payoff_runner_does_not_move_stop_to_breakeven_after_first_target() -> None:
    stop = runner_stop_after_target(
        side="BUY",
        current_stop=98.0,
        entry_price=100.0,
        close_price=101.4,
        trailing_callback_pct=0.35,
        hit_count=1,
        exit_profile="payoff_runner",
        initial_risk_distance=2.0,
    )

    assert 99.7 <= stop < 100.0


def test_payoff_runner_locks_profit_after_second_target() -> None:
    stop = runner_stop_after_target(
        side="BUY",
        current_stop=98.0,
        entry_price=100.0,
        close_price=104.0,
        trailing_callback_pct=0.35,
        hit_count=2,
        exit_profile="payoff_runner",
        initial_risk_distance=2.0,
    )

    assert stop > 100.0


def test_asymmetric_payoff_keeps_more_runner_than_payoff_runner() -> None:
    payoff = staged_take_profit_weights(
        3,
        exit_profile="payoff_runner",
        trailing_stop_enabled=True,
        confidence=0.9,
        strategy_family="mean_reversion",
    )
    asymmetric = staged_take_profit_weights(
        3,
        exit_profile="asymmetric_payoff",
        trailing_stop_enabled=True,
        confidence=0.9,
        strategy_family="mean_reversion",
    )

    assert asymmetric[0] < payoff[0]
    assert (1.0 - sum(asymmetric)) > (1.0 - sum(payoff))


def test_asymmetric_payoff_waits_longer_before_locking_profit() -> None:
    stop = runner_stop_after_target(
        side="BUY",
        current_stop=98.0,
        entry_price=100.0,
        close_price=101.4,
        trailing_callback_pct=0.35,
        hit_count=1,
        exit_profile="asymmetric_payoff",
        initial_risk_distance=2.0,
    )

    assert stop < 100.0
