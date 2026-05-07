from __future__ import annotations

from binance_quant_control.asset_routing import resolve_symbol_route
from binance_quant_control.mission_control import (
    _candidate_issues,
    _configured_news_feed_count,
    _promotion_allows_candidate,
    _system_findings,
)
from binance_quant_control.strategy import load_strategy_config


def test_candidate_issues_uses_strategy_adx_floor_instead_of_trade_count() -> None:
    route = resolve_symbol_route("BTCUSDT")
    strategy = load_strategy_config("config/strategy-btc-volatility.yaml")
    issues = _candidate_issues(
        symbol="BTCUSDT",
        route=route,
        strategy=strategy,
        analysis_payload={
            "latest": {"adx": strategy.risk.min_adx - 1.0},
            "analysis": {"convergence": 0.7},
        },
        backtest_payload={"summary": {"profit_factor": 1.3}},
    )
    assert "trend-strength-is-soft" in issues


def test_candidate_issues_surface_failed_backtest_robustness() -> None:
    route = resolve_symbol_route("BTCUSDT")
    strategy = load_strategy_config("config/strategy-btc-volatility.yaml")
    backtest_payload = {
        "summary": {"profit_factor": 1.3},
        "convergence": {"screening_status": "passed", "promotion_decision": "watchlist"},
        "robustness": {"passed": False, "reasons": ["too-many-folds-below-1pf:2/4"]},
    }

    issues = _candidate_issues(
        symbol="BTCUSDT",
        route=route,
        strategy=strategy,
        analysis_payload={
            "latest": {"adx": strategy.risk.min_adx + 1.0},
            "analysis": {"convergence": 0.7},
        },
        backtest_payload=backtest_payload,
    )

    assert "backtest-robustness-gate-failed" in issues
    assert _promotion_allows_candidate(backtest_payload, require_screening_pass=True) is False


def test_system_findings_only_adds_xau_note_for_xau_symbols() -> None:
    btc_findings = _system_findings(["BTCUSDT"])
    xau_findings = _system_findings(["PAXGUSDT"])

    assert "xau-needs-tokenized-proxy-because-native-metals-execution-is-not-in-this-binance-core" not in btc_findings
    assert "xau-needs-tokenized-proxy-because-native-metals-execution-is-not-in-this-binance-core" in xau_findings


def test_default_news_feed_count_is_no_longer_single_source() -> None:
    assert _configured_news_feed_count() >= 3
    assert "current-news-ingestion-has-fewer-than-three-rss-feeds-so-event-risk-is-still-thin" not in _system_findings(["BTCUSDT"])
