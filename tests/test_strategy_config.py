from pathlib import Path

from binance_quant_control.strategy import load_strategy_config


def test_load_live_pilot_strategy_uses_near_micro_profile():
    config = load_strategy_config(Path("config/strategy-live-pilot.yaml"))

    assert config.profile == "micro-account-pilot"
    assert config.defaults.symbol == "NEARUSDT"
    assert config.defaults.interval == "4h"
    assert config.risk.default_leverage == 6
    assert config.risk.max_leverage == 125
    assert config.execution.margin_notional_usdt == 2.0
    assert config.challenge.enabled is True
    assert config.challenge.target_multiple == 2.0


def test_load_tight_risk_strategy_enables_dynamic_risk_controls():
    config = load_strategy_config(Path("config/strategy-tight-risk.yaml"))

    assert config.profile == "tight-risk-micro"
    assert config.defaults.interval == "1h"
    assert config.risk.max_account_risk_pct == 0.01
    assert config.risk.min_adx == 20
    assert config.risk.trailing_stop_enabled is True
    assert config.execution.margin_notional_usdt is None


def test_load_stable_risk_strategy_uses_2_5_percent_cap():
    config = load_strategy_config(Path("config/strategy-stable-risk.yaml"))

    assert config.profile == "stable-risk-micro"
    assert config.defaults.interval == "1h"
    assert config.risk.max_account_risk_pct == 0.025
    assert config.risk.min_adx == 20
    assert config.risk.trailing_stop_enabled is True


def test_load_hermes_pro_strategy_relaxes_entry_thresholds_without_dropping_risk_caps():
    config = load_strategy_config(Path("config/strategy-hermes-pro.yaml"))

    assert config.profile == "hermes-pro"
    assert config.risk.max_account_risk_pct == 0.014
    assert config.risk.min_convergence == 0.74
    assert config.risk.min_score_long == 68
    assert config.risk.max_score_short == 32
    assert config.risk.min_adx == 20
    assert config.risk.trailing_stop_enabled is True


def test_load_multi_asset_route_strategies() -> None:
    btc = load_strategy_config(Path("config/strategy-btc-volatility.yaml"))
    eth = load_strategy_config(Path("config/strategy-eth-trend.yaml"))
    meme = load_strategy_config(Path("config/strategy-meme-momentum.yaml"))

    assert btc.profile == "btc-volatility"
    assert btc.defaults.symbol == "BTCUSDT"
    assert eth.profile == "eth-trend"
    assert eth.defaults.interval == "4h"
    assert meme.profile == "meme-momentum"
    assert meme.defaults.interval == "1h"


def test_load_core_high_win_strategy_is_low_risk_research_profile() -> None:
    config = load_strategy_config(Path("config/strategy-core-high-win-research.yaml"))

    assert config.profile == "core-high-win-research"
    assert config.defaults.symbol == "ETHUSDT"
    assert config.defaults.interval == "4h"
    assert config.defaults.limit == 1500
    assert config.risk.max_account_risk_pct == 0.006
    assert config.risk.default_leverage == 2
    assert config.risk.max_leverage == 4
    assert config.risk.trailing_stop_enabled is True
    assert config.risk.trailing_callback_pct == 0.35
    assert config.risk.exit_profile == "payoff_runner"
    assert config.risk.time_limit_bars == 72
    assert config.risk.take_profit_r_multiples == (1.0, 1.8, 3.2)


def test_load_market_bot_payoff_research_strategy_is_asymmetric() -> None:
    config = load_strategy_config(Path("config/strategy-market-bot-payoff-research.yaml"))

    assert config.profile == "market-bot-payoff-research"
    assert config.defaults.symbol == "BTCUSDT"
    assert config.defaults.interval == "4h"
    assert config.risk.max_account_risk_pct == 0.006
    assert config.risk.min_convergence == 0.66
    assert config.risk.min_adx == 14
    assert config.risk.trailing_stop_enabled is True
    assert config.risk.exit_profile == "asymmetric_payoff"
    assert config.risk.time_limit_bars == 72
    assert config.risk.take_profit_r_multiples == (1.2, 2.4, 4.8)
