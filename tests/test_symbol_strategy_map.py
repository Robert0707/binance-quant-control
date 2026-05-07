from __future__ import annotations

from binance_quant_control.symbol_strategy_map import (
    filter_symbol_families,
    filter_symbol_interval_families,
    load_symbol_strategy_map,
    resolve_symbol_interval_family_sides,
    resolve_symbol_strategy,
)


def test_core_symbol_strategy_map_routes_symbols_to_independent_families() -> None:
    specs = load_symbol_strategy_map("config/core-symbol-strategy-map.default.yaml")

    assert set(specs) == {
        "BTCUSDT",
        "ETHUSDT",
        "XAUTUSDT",
        "PAXGUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "LINKUSDT",
        "AAVEUSDT",
        "TRXUSDT",
    }
    assert specs["BTCUSDT"].primary_family == "breakout"
    assert specs["BTCUSDT"].allowed_families == (
        "vwap_reclaim",
        "breakout",
        "trend_pullback",
        "trend_continuation",
        "liquidity_reclaim",
    )
    assert "mean_reversion" in specs["BTCUSDT"].blocked_families
    assert specs["BTCUSDT"].interval_families["1h"] == ()
    assert specs["ETHUSDT"].allowed_families == (
        "vwap_reclaim",
        "trend_continuation",
        "breakout",
        "trend_pullback",
        "liquidity_reclaim",
        "mean_reversion",
    )
    assert specs["LINKUSDT"].interval_families["1h"] == ()
    assert specs["LINKUSDT"].interval_families["4h"] == ()
    assert specs["TRXUSDT"].primary_family == "mean_reversion"
    assert specs["TRXUSDT"].allowed_families == ("vwap_reclaim", "mean_reversion", "liquidity_reclaim")
    assert specs["SOLUSDT"].interval_families["15m"] == ()
    assert specs["SOLUSDT"].interval_families["4h"] == ()
    assert specs["XRPUSDT"].interval_families["4h"] == ("mean_reversion",)
    assert specs["XRPUSDT"].interval_family_sides["4h"]["mean_reversion"] == ("BUY",)
    assert specs["TRXUSDT"].interval_family_sides["4h"]["mean_reversion"] == ("SELL",)
    assert specs["BTCUSDT"].entry_filters == {}
    assert all(spec.execution_lane == "paper_research_only" for spec in specs.values())
    assert all(spec.allowed_families for spec in specs.values())
    assert all(spec.promotion.min_trades >= 100 for spec in specs.values())
    assert all(spec.promotion.min_win_rate >= 65.0 for spec in specs.values())
    assert all(spec.promotion.max_stop_loss_ratio <= 35.0 for spec in specs.values())
    assert all(spec.promotion.min_expectancy_r >= 0.10 for spec in specs.values())
    assert all(spec.promotion.min_payoff_ratio >= 1.15 for spec in specs.values())


def test_market_bot_router_map_loads_symbol_specific_entry_profiles() -> None:
    specs = load_symbol_strategy_map("config/market-bot-six-symbol-router-map.default.yaml")

    assert set(specs) == {"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT", "TRXUSDT"}
    assert specs["BTCUSDT"].allowed_families == ("ai_family_router",)
    assert specs["BTCUSDT"].entry_filters["min_obv_zscore"] == -0.879
    assert specs["ETHUSDT"].entry_filters["min_signed_di"] == 5.1712
    assert specs["ETHUSDT"].entry_filters["blocked_routed_families"] == ["trend_pullback"]
    assert specs["ETHUSDT"].strategy_overrides["risk"]["atr_stop_multiple"] == 1.70
    assert specs["SOLUSDT"].entry_filters["min_signed_di"] == 15.0
    assert specs["SOLUSDT"].strategy_overrides["risk"]["atr_stop_multiple"] == 1.55
    assert specs["SOLUSDT"].strategy_overrides["risk"]["take_profit_r_multiples"] == [1.3, 2.6, 5.2]
    assert specs["DOGEUSDT"].asset_class == "meme_high_beta"
    assert specs["DOGEUSDT"].entry_filters == {}
    assert specs["XRPUSDT"].execution_lane == "quarantined_research_only"
    assert "ai_family_router" in specs["XRPUSDT"].blocked_families
    assert specs["TRXUSDT"].entry_filters["max_obv_zscore"] == 1.4
    assert specs["TRXUSDT"].entry_filters["blocked_routed_families"] == [
        "liquidity_reclaim",
        "vwap_reclaim",
    ]
    assert specs["TRXUSDT"].strategy_overrides["risk"]["atr_stop_multiple"] == 1.25


def test_resolve_symbol_strategy_uses_alias_normalization() -> None:
    spec = resolve_symbol_strategy("xau", "config/core-symbol-strategy-map.default.yaml")

    assert spec.symbol == "XAUTUSDT"
    assert spec.asset_class == "xau_macro"
    assert spec.primary_family == "mean_reversion"


def test_filter_symbol_families_prevents_correlated_family_collisions() -> None:
    specs = load_symbol_strategy_map("config/core-symbol-strategy-map.default.yaml")
    candidates = [
        "trend_continuation",
        "breakout",
        "trend_pullback",
        "liquidity_reclaim",
        "vwap_reclaim",
        "mean_reversion",
    ]

    assert set(filter_symbol_families("BTCUSDT", candidates, specs)) == {
        "vwap_reclaim",
        "breakout",
        "trend_pullback",
        "trend_continuation",
        "liquidity_reclaim",
    }
    assert set(filter_symbol_families("TRXUSDT", candidates, specs)) == {
        "vwap_reclaim",
        "liquidity_reclaim",
        "mean_reversion",
    }
    assert filter_symbol_families("ETHUSDT", candidates, specs) == [
        "trend_continuation",
        "breakout",
        "trend_pullback",
        "liquidity_reclaim",
        "vwap_reclaim",
        "mean_reversion",
    ]


def test_filter_symbol_interval_families_applies_timeframe_specific_routes() -> None:
    specs = load_symbol_strategy_map("config/core-symbol-strategy-map.default.yaml")
    candidates = ["trend_continuation", "breakout", "trend_pullback", "liquidity_reclaim", "mean_reversion"]

    assert filter_symbol_interval_families("BTCUSDT", "15m", candidates, specs) == []
    assert filter_symbol_interval_families("BTCUSDT", "1h", candidates, specs) == []
    assert filter_symbol_interval_families("BTCUSDT", "4h", candidates, specs) == []
    assert filter_symbol_interval_families("BTCUSDT", "1d", candidates, specs) == []
    assert filter_symbol_interval_families("ETHUSDT", "15m", candidates, specs) == []
    assert filter_symbol_interval_families("ETHUSDT", "1h", candidates, specs) == []
    assert filter_symbol_interval_families("ETHUSDT", "4h", candidates, specs) == ["trend_pullback"]
    assert filter_symbol_interval_families("XAUTUSDT", "1h", candidates, specs) == []
    assert filter_symbol_interval_families("XAUTUSDT", "4h", candidates, specs) == ["trend_pullback"]
    assert filter_symbol_interval_families("SOLUSDT", "4h", candidates, specs) == []
    assert filter_symbol_interval_families("XRPUSDT", "4h", candidates, specs) == ["mean_reversion"]
    assert filter_symbol_interval_families("TRXUSDT", "1h", candidates, specs) == []
    assert resolve_symbol_interval_family_sides("XRPUSDT", "4h", "mean_reversion", specs) == ("BUY",)
    assert resolve_symbol_interval_family_sides("TRXUSDT", "4h", "mean_reversion", specs) == ("SELL",)
    assert resolve_symbol_interval_family_sides("ETHUSDT", "4h", "trend_pullback", specs) == ("BUY", "SELL")


def test_core_symbols_keep_only_backtest_supported_active_routes() -> None:
    specs = load_symbol_strategy_map("config/core-symbol-strategy-map.default.yaml")
    candidates = ["trend_continuation", "breakout", "trend_pullback", "liquidity_reclaim", "mean_reversion"]
    expected_active = {
        ("ETHUSDT", "4h"),
        ("XAUTUSDT", "4h"),
        ("PAXGUSDT", "1d"),
        ("BNBUSDT", "4h"),
        ("XRPUSDT", "4h"),
        ("TRXUSDT", "4h"),
    }

    for symbol in specs:
        for interval in ("15m", "1h", "4h", "1d"):
            active = bool(filter_symbol_interval_families(symbol, interval, candidates, specs))
            assert active is ((symbol, interval) in expected_active), (symbol, interval)
