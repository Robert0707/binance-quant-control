from __future__ import annotations

from dataclasses import replace

from binance_quant_control.autonomy import (
    AutonomyConfig,
    alpha_promotion_gate,
    determine_candidate_side,
    determine_candidate_symbol,
    evaluate_entry_gate,
    load_autonomy_config,
    position_protection_settings,
    protective_repair_reasons,
    should_execute_testnet_entry,
)
from binance_quant_control.config import Settings
from binance_quant_control.strategy import (
    StrategyChallenge,
    StrategyConfig,
    StrategyDefaults,
    StrategyExecution,
    StrategyRisk,
    StrategySignal,
)


def _config() -> AutonomyConfig:
    return AutonomyConfig(
        path=__file__,
        strategy_config=__file__,
        routing_config=__file__,
        digest_config=__file__,
        hailo_config=__file__,
        review_limit=20,
        execute_live_entries=False,
        execute_testnet_entries=False,
        execute_simulated_entries=False,
        execute_position_protection=False,
        adaptive_exit_enabled=False,
        force_simulation_after_analysis=True,
        require_digest_action="pre_trade_notify",
        require_strategy_analyzer_approval=True,
        min_strategy_analyzer_confidence=0.62,
        max_managed_positions=3,
        allow_new_entries_with_open_positions=False,
        candidate_symbol_source="digest_selected",
        execution_mode="live",
        require_alpha_promotion=False,
        alpha_research_report=None,
        trailing_callback_pct=None,
        adaptive_exit_min_profit_r=0.35,
        adaptive_exit_max_loss_r=-0.35,
        adaptive_exit_min_reversal_score=5.0,
        adaptive_exit_min_confidence=0.62,
        margin_notional_usdt=None,
        simulation_notional_usdt=None,
        min_expected_profit_usdt=2.0,
        require_professional_entry_gate=False,
        min_reward_risk=1.2,
        min_net_profit_to_risk=0.8,
        max_fee_profit_ratio=0.35,
        max_slippage_profit_ratio=0.25,
        max_volatility=1.8,
        min_volume_zscore=-0.8,
        min_recent_reviews=6,
        min_recent_win_rate=0.42,
        min_recent_avg_r=0.0,
        max_recent_stop_loss_ratio=0.55,
        recent_lookback=20,
        stop_loss_cooldown_hours=6.0,
        hailo_enabled=True,
        digest_enabled=True,
        reuse_cached_digest=True,
        digest_min_interval_minutes=240,
        skip_digest_when_positions_open=True,
        review_enabled=True,
        auto_pause_enabled=True,
    )


def _strategy() -> StrategyConfig:
    return StrategyConfig(
        profile="hermes-pro",
        description="test",
        defaults=StrategyDefaults(symbol="NEARUSDT"),
        risk=StrategyRisk(),
        signal=StrategySignal(),
        execution=StrategyExecution(),
        challenge=StrategyChallenge(),
        path=__file__,
    )


def test_digest_candidate_selection_prefers_digest_symbol() -> None:
    digest = {
        "decision": {
            "selected": {
                "symbol": "BTCUSDT",
                "direction": "long",
            }
        }
    }
    assert determine_candidate_symbol(_config(), _strategy(), digest) == "BTCUSDT"
    assert determine_candidate_side(digest) == "BUY"


def test_testnet_explorer_uses_core_high_win_strategy_profile() -> None:
    config = load_autonomy_config("config/autonomous-testnet-explorer.default.yaml")

    assert config.strategy_config.name == "strategy-core-high-win-research.yaml"
    assert config.execute_testnet_entries is True
    assert config.execute_live_entries is False
    assert config.margin_notional_usdt == 5.0
    assert config.trailing_callback_pct == 0.35
    assert config.max_recent_stop_loss_ratio == 0.45
    assert config.require_alpha_promotion is True
    assert config.alpha_research_report is not None
    assert config.alpha_research_report.name == "alpha-research-ranking.json"


def test_guardian_enables_adaptive_exit_controls() -> None:
    config = load_autonomy_config("config/autonomous-guardian.default.yaml")

    assert config.execute_position_protection is True
    assert config.adaptive_exit_enabled is True
    assert config.adaptive_exit_min_profit_r == 0.35
    assert config.adaptive_exit_max_loss_r == -0.35


def test_entry_gate_blocks_without_cloud_approval() -> None:
    digest = {
        "decision": {
            "action": "pre_trade_notify",
            "selected": {"symbol": "BTCUSDT", "direction": "long"},
        },
        "strategy_analysis": {
            "available": True,
            "result": {"result": {"verdict": "watch", "confidence": 0.81}},
        },
    }
    live_plan = {
        "allowed": True,
        "violations": [],
        "price": 100.0,
        "quantity": 1.0,
        "take_profit_price": 103.0,
        "side": "BUY",
    }
    gate = evaluate_entry_gate(
        _config(),
        digest_payload=digest,
        live_plan=live_plan,
        latest={},
        positions_count=0,
        trading_paused=False,
    )
    assert gate["eligible"] is False
    assert any("verdict is watch" in reason for reason in gate["reasons"])


def test_entry_gate_allows_approved_candidate() -> None:
    digest = {
        "decision": {
            "action": "pre_trade_notify",
            "selected": {"symbol": "BTCUSDT", "direction": "short"},
        },
        "strategy_analysis": {
            "available": True,
            "result": {"result": {"verdict": "approve", "confidence": 0.74}},
        },
    }
    live_plan = {
        "allowed": True,
        "violations": [],
        "price": 100.0,
        "quantity": 1.0,
        "take_profit_price": 103.0,
        "side": "SELL",
    }
    gate = evaluate_entry_gate(
        _config(),
        digest_payload=digest,
        live_plan=live_plan,
        latest={},
        positions_count=0,
        trading_paused=False,
    )
    assert gate["eligible"] is False
    assert any("below the configured floor" in reason for reason in gate["reasons"])
    assert gate["strategy_analyzer_verdict"] == "approve"


def test_entry_gate_allows_when_profit_floor_is_met() -> None:
    digest = {
        "decision": {
            "action": "pre_trade_notify",
            "selected": {"symbol": "BTCUSDT", "direction": "long"},
        },
        "strategy_analysis": {
            "available": True,
            "result": {"result": {"verdict": "approve", "confidence": 0.74}},
        },
    }
    live_plan = {
        "allowed": True,
        "violations": [],
        "price": 100.0,
        "quantity": 3.0,
        "take_profit_price": 101.0,
        "side": "BUY",
    }
    gate = evaluate_entry_gate(
        _config(),
        digest_payload=digest,
        live_plan=live_plan,
        latest={},
        positions_count=0,
        trading_paused=False,
    )
    assert gate["eligible"] is True


def test_testnet_config_does_not_execute_watchlist_only_candidate() -> None:
    config = _config()
    config = replace(
        config,
        execute_testnet_entries=True,
        execution_mode="testnet_exploration",
        require_strategy_analyzer_approval=False,
        min_expected_profit_usdt=0.0,
        require_professional_entry_gate=False,
    )
    digest = {
        "decision": {
            "action": "watchlist_only",
            "selected": {"symbol": "DOGEUSDT", "direction": "long"},
        },
        "strategy_analysis": {"available": False},
    }
    live_plan = {
        "allowed": True,
        "violations": [],
        "price": 0.1,
        "quantity": 100.0,
        "take_profit_price": 0.11,
        "side": "BUY",
    }

    gate = evaluate_entry_gate(
        config,
        digest_payload=digest,
        live_plan=live_plan,
        latest={},
        positions_count=0,
        trading_paused=False,
    )
    should_execute_testnet = should_execute_testnet_entry(
        config,
        gate=gate,
        digest_payload=digest,
        live_plan=live_plan,
        candidate_side=determine_candidate_side(digest),
        trading_paused=False,
    )

    assert gate["eligible"] is False
    assert should_execute_testnet is False


def test_testnet_execution_requires_entry_gate_eligible() -> None:
    config = replace(
        _config(),
        execute_testnet_entries=True,
        execution_mode="testnet_exploration",
        require_strategy_analyzer_approval=False,
        require_professional_entry_gate=True,
    )
    digest = {
        "decision": {
            "action": "pre_trade_notify",
            "selected": {"symbol": "DOGEUSDT", "direction": "long"},
        },
        "strategy_analysis": {"available": False},
    }
    live_plan = {
        "allowed": True,
        "violations": [],
        "price": 0.1,
        "quantity": 100.0,
        "take_profit_price": 0.11,
        "side": "BUY",
    }
    blocked_gate = {
        "eligible": False,
        "reasons": ["Recent stop-loss ratio 66.9% exceeds 55.0%."],
    }

    assert (
        should_execute_testnet_entry(
            config,
            gate=blocked_gate,
            digest_payload=digest,
            live_plan=live_plan,
            candidate_side=determine_candidate_side(digest),
            trading_paused=False,
        )
        is False
    )


def test_testnet_execution_requires_alpha_promotion_when_configured() -> None:
    config = replace(
        _config(),
        execute_testnet_entries=True,
        execution_mode="testnet_exploration",
        require_alpha_promotion=True,
    )
    digest = {
        "decision": {
            "action": "pre_trade_notify",
            "selected": {"symbol": "DOGEUSDT", "direction": "long"},
        }
    }
    live_plan = {
        "allowed": True,
        "violations": [],
        "price": 0.1,
        "quantity": 100.0,
        "take_profit_price": 0.11,
        "side": "BUY",
    }

    should_execute = should_execute_testnet_entry(
        config,
        gate={"eligible": True},
        digest_payload=digest,
        live_plan=live_plan,
        candidate_side="BUY",
        trading_paused=False,
        alpha_gate={"required": True, "allowed": False},
    )

    assert should_execute is False


def test_alpha_promotion_gate_blocks_unpromoted_research_row(tmp_path) -> None:
    report = tmp_path / "alpha-research-ranking.json"
    report.write_text(
        """
{
  "rows": [
    {
      "cohort_id": "ETHUSDT:4h:mean_reversion",
      "promotion_eligible": false,
      "trade_count": 4,
      "win_rate": 50.0,
      "stop_loss_ratio": 50.0,
      "profit_factor": 0.8,
      "robustness_status": "insufficient_sample"
    }
  ]
}
""",
        encoding="utf-8",
    )
    config = replace(_config(), require_alpha_promotion=True, alpha_research_report=report)

    gate = alpha_promotion_gate(
        config,
        symbol="ETHUSDT",
        interval="4h",
        strategy_family="mean_reversion",
    )

    assert gate["required"] is True
    assert gate["allowed"] is False
    assert "alpha-cohort-not-promotion-eligible" in gate["reasons"]


def test_entry_gate_can_allow_new_testnet_entry_below_position_cap() -> None:
    config = replace(
        _config(),
        allow_new_entries_with_open_positions=True,
        require_strategy_analyzer_approval=False,
        min_expected_profit_usdt=0.0,
        require_professional_entry_gate=False,
    )
    digest = {
        "decision": {
            "action": "pre_trade_notify",
            "selected": {"symbol": "ETHUSDT", "direction": "long"},
        },
        "strategy_analysis": {"available": False},
    }
    live_plan = {
        "allowed": True,
        "violations": [],
        "price": 100.0,
        "quantity": 1.0,
        "take_profit_price": 101.0,
        "side": "BUY",
    }

    gate = evaluate_entry_gate(
        config,
        digest_payload=digest,
        live_plan=live_plan,
        latest={},
        positions_count=1,
        trading_paused=False,
    )

    assert gate["eligible"] is True


def test_entry_gate_blocks_when_position_cap_is_reached() -> None:
    config = replace(
        _config(),
        allow_new_entries_with_open_positions=True,
        require_strategy_analyzer_approval=False,
        min_expected_profit_usdt=0.0,
        require_professional_entry_gate=False,
    )
    digest = {
        "decision": {
            "action": "pre_trade_notify",
            "selected": {"symbol": "ETHUSDT", "direction": "long"},
        },
        "strategy_analysis": {"available": False},
    }
    live_plan = {
        "allowed": True,
        "violations": [],
        "price": 100.0,
        "quantity": 1.0,
        "take_profit_price": 101.0,
        "side": "BUY",
    }

    gate = evaluate_entry_gate(
        config,
        digest_payload=digest,
        live_plan=live_plan,
        latest={},
        positions_count=3,
        trading_paused=False,
    )

    assert gate["eligible"] is False
    assert any("Max managed positions reached" in reason for reason in gate["reasons"])


def test_position_protection_settings_enable_testnet_without_mainnet_live() -> None:
    config = replace(_config(), execution_mode="testnet_exploration")
    settings = Settings(
        use_testnet=False,
        live_trading_enabled=False,
        testnet_trading_enabled=False,
        recv_window_ms=5000,
        default_symbol="BTCUSDT",
        default_market="futures",
        binance_api_key="",
        binance_secret_key="",
        binance_testnet_api_key="key",
        binance_testnet_secret_key="secret",
        blave_api_key="",
        blave_secret_key="",
        whale_alert_api_key="",
        max_leverage=3,
        max_notional_pct=0.5,
        max_daily_trades=3,
        min_balance_usdt=2.0,
        min_convergence=0.6,
        cooldown_hours=4.0,
    )

    protected = position_protection_settings(config, settings)

    assert protected.use_testnet is True
    assert protected.live_trading_enabled is False
    assert protected.testnet_trading_enabled is True


def test_protective_repair_reasons_detect_missing_ladder() -> None:
    strategy = _strategy()
    plan = type(
        "Plan",
        (),
        {
            "quantity": 10.0,
            "existing_algo_orders": [
                {"orderType": "TAKE_PROFIT_MARKET", "quantity": "10"},
            ],
        },
    )()

    reasons = protective_repair_reasons(plan, strategy)

    assert "missing-stop-loss" in reasons
    assert "missing-staged-take-profits" in reasons
    assert "single-full-position-take-profit" in reasons


def test_protective_repair_reasons_accept_healthy_staged_protection() -> None:
    strategy = _strategy()
    plan = type(
        "Plan",
        (),
        {
            "quantity": 10.0,
            "existing_algo_orders": [
                {"orderType": "STOP_MARKET", "quantity": "10"},
                {"orderType": "TAKE_PROFIT_MARKET", "quantity": "3"},
                {"orderType": "TAKE_PROFIT_MARKET", "quantity": "3"},
                {"orderType": "TAKE_PROFIT_MARKET", "quantity": "2"},
            ],
        },
    )()

    assert protective_repair_reasons(plan, strategy) == []


def test_protective_repair_reasons_accept_micro_full_tp_fallback() -> None:
    strategy = _strategy()
    plan = type(
        "Plan",
        (),
        {
            "quantity": 0.001,
            "step_size": 0.001,
            "existing_algo_orders": [
                {"orderType": "STOP_MARKET", "quantity": "0.001"},
                {"orderType": "TAKE_PROFIT_MARKET", "quantity": "0.001"},
            ],
        },
    )()

    assert protective_repair_reasons(plan, strategy) == []
