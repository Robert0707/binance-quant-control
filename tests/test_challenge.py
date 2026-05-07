from __future__ import annotations

from binance_quant_control.challenge import (
    ChallengeState,
    challenge_scope_key,
    initialize_challenge,
    load_challenge_state,
    snapshot_from_balance_payload,
    update_challenge_state,
)


def test_initialize_challenge_sets_target_and_stop(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_quant_control.challenge.CHALLENGE_STATE_FILE", tmp_path / "challenge.json")
    state = initialize_challenge(
        profile="micro-account-pilot",
        symbol="NEARUSDT",
        market="futures",
        start_balance_usdt=5.0,
        target_multiple=2.0,
        max_drawdown_pct=20.0,
    )

    loaded = load_challenge_state()
    assert state.target_balance_usdt == 10.0
    assert state.stop_balance_usdt == 4.0
    assert loaded.status == "active"


def test_update_challenge_state_marks_target_hit(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_quant_control.challenge.CHALLENGE_STATE_FILE", tmp_path / "challenge.json")
    state = ChallengeState(
        enabled=True,
        profile="micro-account-pilot",
        symbol="NEARUSDT",
        market="futures",
        started_at="2026-04-25T00:00:00+00:00",
        start_balance_usdt=5.0,
        target_balance_usdt=10.0,
        target_multiple=2.0,
        max_drawdown_pct=20.0,
        stop_balance_usdt=4.0,
        highest_balance_usdt=7.0,
        latest_balance_usdt=7.0,
        latest_snapshot_at="",
        status="active",
    )
    payload = [
        {
            "asset": "USDT",
            "balance": "9.7",
            "availableBalance": "4.0",
            "crossUnPnl": "0.4",
        }
    ]
    snapshot = snapshot_from_balance_payload(payload, "futures", note="test")
    updated = update_challenge_state(state, snapshot)

    assert updated.status == "target-hit"
    assert updated.latest_balance_usdt == 10.1


def test_challenge_scope_isolated_by_profile(monkeypatch, tmp_path):
    monkeypatch.setattr("binance_quant_control.challenge.STATE_DIR", tmp_path)
    scope_a = challenge_scope_key("micro-account-pilot", "NEARUSDT", "futures")
    scope_b = challenge_scope_key("stable-risk-micro", "NEARUSDT", "futures")

    initialize_challenge(
        profile="micro-account-pilot",
        symbol="NEARUSDT",
        market="futures",
        start_balance_usdt=5.0,
        scope=scope_a,
    )
    initialize_challenge(
        profile="stable-risk-micro",
        symbol="NEARUSDT",
        market="futures",
        start_balance_usdt=6.0,
        scope=scope_b,
    )

    loaded_a = load_challenge_state(scope_a)
    loaded_b = load_challenge_state(scope_b)

    assert loaded_a.profile == "micro-account-pilot"
    assert loaded_a.start_balance_usdt == 5.0
    assert loaded_b.profile == "stable-risk-micro"
    assert loaded_b.start_balance_usdt == 6.0
