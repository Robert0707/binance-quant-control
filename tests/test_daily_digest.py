from __future__ import annotations

from unittest.mock import patch

import binance_quant_control.daily_digest as daily_digest
from binance_quant_control.candidate_universe import UniverseSymbol
from binance_quant_control.daily_digest import (
    adjusted_candidate_score,
    assess_news_risk,
    build_decision,
    candidate_is_tradeable,
    collect_news_items,
    rank_candidates,
    summarize_github_observability,
    summarize_whale_transactions,
)


def test_assess_news_risk_marks_high_impact_headlines() -> None:
    result = assess_news_risk(
        [
            {"title": "CPI release shakes crypto market"},
            {"title": "Fed minutes ahead of FOMC meeting"},
            {"title": "ETF inflow remains resilient"},
        ]
    )
    assert result["risk_level"] == "high"
    assert result["high_impact_count"] == 2


def test_collect_news_items_round_robins_multiple_feeds(monkeypatch) -> None:
    def fake_fetch(url: str, *, limit: int = 8) -> list[dict[str, object]]:
        return [
            {"title": f"{url}-1", "source": url},
            {"title": f"{url}-2", "source": url},
        ]

    monkeypatch.setattr(daily_digest, "fetch_rss_feed", fake_fetch)

    items = collect_news_items(["feed-a", "feed-b", "feed-c"], news_limit=5)

    assert [item["source"] for item in items] == ["feed-a", "feed-b", "feed-c", "feed-a", "feed-b"]


def test_summarize_whale_transactions_detects_exchange_inflow_bias() -> None:
    result = summarize_whale_transactions(
        [
            {
                "symbol": "btc",
                "amount_usd": 1000000,
                "from": {"owner": "unknown"},
                "to": {"owner": "binance"},
                "transaction_type": "transfer",
            },
            {
                "symbol": "eth",
                "amount_usd": 100000,
                "from": {"owner": "coinbase"},
                "to": {"owner": "unknown"},
                "transaction_type": "transfer",
            },
        ]
    )
    assert result["signal"] == "bearish"
    assert result["exchange_inflow_usd"] == 1000000.0


def test_whale_alert_failure_degrades_to_neutral(monkeypatch) -> None:
    monkeypatch.setattr(
        daily_digest,
        "http_get_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("upstream down")),
    )

    result = daily_digest.fetch_whale_alert_summary("key", min_value_usd=500000, limit=2)

    assert result["available"] is False
    assert result["signal"] == "neutral"
    assert result["exchange_inflow_usd"] == 0.0


def test_rank_candidates_orders_long_short_and_neutral_independently() -> None:
    ranked = rank_candidates(
        [
            {
                "symbol": "BTCUSDT",
                "direction": "long",
                "composite_score": 82.0,
                "analysis": {"convergence": 0.8},
                "latest": {"adx": 22.0, "realized_vol_20": 0.8},
                "candidate_quality": {"score": 80.0},
            },
            {
                "symbol": "ETHUSDT",
                "direction": "long",
                "composite_score": 61.0,
                "analysis": {"convergence": 0.7},
                "latest": {"adx": 18.0, "realized_vol_20": 0.8},
                "candidate_quality": {"score": 60.0},
            },
            {
                "symbol": "XRPUSDT",
                "direction": "short",
                "composite_score": 75.0,
                "analysis": {"convergence": 0.8},
                "latest": {"adx": 22.0, "realized_vol_20": 0.8},
                "candidate_quality": {"score": 75.0},
            },
            {"symbol": "BNBUSDT", "direction": "neutral", "composite_score": 45.0},
        ]
    )
    assert ranked["long"][0]["symbol"] == "BTCUSDT"
    assert ranked["short"][0]["symbol"] == "XRPUSDT"
    assert ranked["neutral"][0]["symbol"] == "BNBUSDT"
    assert ranked["long"][0]["candidate_quality"]["score"] >= ranked["long"][1]["candidate_quality"]["score"]


def test_candidate_quality_filter_blocks_chop_and_extreme_volatility() -> None:
    assert candidate_is_tradeable(
        {
            "direction": "long",
            "analysis": {"convergence": 0.7},
            "latest": {"adx": 20.0, "realized_vol_20": 0.8},
            "candidate_quality": {"score": 50.0},
        }
    )
    assert not candidate_is_tradeable(
        {
            "direction": "long",
            "analysis": {"convergence": 0.3},
            "latest": {"adx": 20.0, "realized_vol_20": 0.8},
            "candidate_quality": {"score": 50.0},
        }
    )
    assert not candidate_is_tradeable(
        {
            "direction": "short",
            "analysis": {"convergence": 0.7},
            "latest": {"adx": 20.0, "realized_vol_20": 3.0},
            "candidate_quality": {"score": 50.0},
        }
    )


def test_summarize_github_observability_marks_active_stack() -> None:
    summary = summarize_github_observability(
        [
            {"repo": "freqtrade/freqtrade", "stars": 49000, "recency_days": 1.2},
            {"repo": "hummingbot/hummingbot", "stars": 18000, "recency_days": 5.4},
            {"repo": "vectorbt", "stars": 7000, "recency_days": 40.0},
        ]
    )

    assert summary["signal"] == "active"
    assert summary["active_repo_count"] == 2
    assert summary["total_stars"] == 74000


def test_adjusted_candidate_score_penalizes_negative_route_side_feedback(monkeypatch) -> None:
    class FakeFeedback:
        sample_count = 60
        min_samples = 30
        net_pnl_usdt = -12.0
        profit_factor = 0.52
        stop_loss_ratio = 74.0
        loss_streak = 4

        def to_dict(self):
            return {
                "sample_count": self.sample_count,
                "min_samples": self.min_samples,
                "net_pnl_usdt": self.net_pnl_usdt,
                "profit_factor": self.profit_factor,
                "stop_loss_ratio": self.stop_loss_ratio,
                "loss_streak": self.loss_streak,
            }

    monkeypatch.setattr(daily_digest, "evaluate_route_side_risk", lambda **kwargs: FakeFeedback())

    adjusted, alignment, notes, feedback = adjusted_candidate_score(
        {
            "symbol": "SOLUSDT",
            "route_id": "major-alt-trend",
            "direction": "long",
            "composite_score": 88.0,
            "analysis": {"score": 90.0, "convergence": 0.9},
        },
        news_risk={"risk_level": "normal", "bias": "neutral"},
        whale_summary={"signal": "neutral"},
        github_summary={"signal": "active"},
    )

    assert adjusted < 80.0
    assert alignment == "neutral"
    assert "route-side-negative-expectancy" in notes
    assert "route-side-stop-loss-heavy" in notes
    assert feedback["profit_factor"] == 0.52


def test_adjusted_candidate_score_penalizes_early_negative_route_side_feedback(monkeypatch) -> None:
    class FakeFeedback:
        sample_count = 6
        min_samples = 30
        net_pnl_usdt = -7.75
        profit_factor = 0.03
        stop_loss_ratio = 33.33
        loss_streak = 1

        def to_dict(self):
            return {
                "sample_count": self.sample_count,
                "min_samples": self.min_samples,
                "net_pnl_usdt": self.net_pnl_usdt,
                "profit_factor": self.profit_factor,
                "stop_loss_ratio": self.stop_loss_ratio,
                "loss_streak": self.loss_streak,
            }

    monkeypatch.setattr(daily_digest, "evaluate_route_side_risk", lambda **kwargs: FakeFeedback())

    adjusted, alignment, notes, feedback = adjusted_candidate_score(
        {
            "symbol": "ENSOUSDT",
            "route_id": "defensive-unknown",
            "direction": "long",
            "composite_score": 93.2,
            "analysis": {"score": 100.0, "convergence": 0.947},
        },
        news_risk={"risk_level": "high", "bias": "neutral"},
        whale_summary={"signal": "neutral"},
        github_summary={"signal": "active"},
    )

    assert adjusted < 72.0
    assert alignment == "neutral"
    assert "route-side-early-negative-expectancy" in notes
    assert "high-news-risk" in notes
    assert feedback["sample_count"] == 6


def test_build_decision_stands_by_on_high_news_risk_and_early_negative_feedback(monkeypatch) -> None:
    class FakeFeedback:
        sample_count = 6
        min_samples = 30
        net_pnl_usdt = -7.75
        profit_factor = 0.03
        stop_loss_ratio = 33.33
        loss_streak = 1

        def to_dict(self):
            return {
                "sample_count": self.sample_count,
                "min_samples": self.min_samples,
                "net_pnl_usdt": self.net_pnl_usdt,
                "profit_factor": self.profit_factor,
                "stop_loss_ratio": self.stop_loss_ratio,
                "loss_streak": self.loss_streak,
            }

    monkeypatch.setattr(daily_digest, "evaluate_route_side_risk", lambda **kwargs: FakeFeedback())

    decision = build_decision(
        {
            "long": [
                {
                    "symbol": "ENSOUSDT",
                    "route_id": "defensive-unknown",
                    "asset_class": "defensive_unknown",
                    "direction": "long",
                    "composite_score": 93.2,
                    "analysis": {"score": 100.0, "convergence": 0.947},
                    "candidate_quality": {"score": 93.2},
                    "validation": {},
                }
            ],
            "short": [],
            "neutral": [],
        },
        news_risk={"risk_level": "high", "bias": "neutral"},
        whale_summary={"signal": "neutral"},
        github_summary={"signal": "active"},
        doctor={"warnings": []},
    )

    assert decision["action"] == "stand_by"
    assert decision["reason"] == "high_event_risk_and_negative_feedback"
    assert "route-side-early-negative-expectancy" in decision["selected"]["context_notes"]


def test_build_digest_uses_strategy_analyzer_url_and_compact_payload(tmp_path) -> None:
    config = {
        "symbols": ["BTCUSDT"],
        "market": "futures",
        "interval": "4h",
        "news_feeds": ["https://example.com/feed.xml"],
        "news_limit": 3,
        "whale_min_value_usd": 500000,
        "whale_limit": 2,
        "github_observability_repos": ["ccxt/ccxt"],
        "digest_root": str(tmp_path),
        "strategy_analyzer_url": "http://127.0.0.1:9999/analyze",
    }

    def fake_run_quant_analysis(symbol: str, market: str, interval: str) -> dict[str, object]:
        return {
            "symbol": symbol,
            "market": market,
            "interval": interval,
            "direction": "long",
            "composite_score": 88.0,
            "analysis": {"score": 88.0, "convergence": 0.92},
            "latest": {"adx": 23.0},
            "trade_plan": {},
            "run_id": "run-1",
        }

    with (
        patch.object(daily_digest, "fetch_rss_feed", return_value=[{"title": "ETF inflow remains resilient"}]),
        patch.object(
            daily_digest,
            "fetch_whale_alert_summary",
            return_value={"signal": "bullish", "exchange_inflow_usd": 0.0, "exchange_outflow_usd": 0.0},
        ),
        patch.object(
            daily_digest,
            "fetch_github_observability",
            return_value=[{"repo": "ccxt/ccxt", "stars": 35000, "recency_days": 2.0}],
        ),
        patch.object(daily_digest, "build_external_context", return_value={"combined_signal": "neutral"}),
        patch.object(daily_digest, "run_compact_json", return_value={"overall": "ok", "warnings": []}),
        patch.object(daily_digest, "run_quant_analysis", side_effect=fake_run_quant_analysis),
        patch.object(
            daily_digest,
            "fetch_strategy_analyzer_summary",
            return_value={
                "enabled": True,
                "available": True,
                "source": config["strategy_analyzer_url"],
                "result": {"result": {"confidence": 0.91, "verdict": "approve", "regime": "aligned", "notes": []}},
            },
        ) as analyzer,
    ):
        payload = daily_digest.build_digest(config)

    analyzer.assert_called_once()
    request_payload = analyzer.call_args.args[1]
    assert request_payload["selected"]["symbol"] == "BTCUSDT"
    assert request_payload["decision"]["action"] in {"pre_trade_notify", "watchlist_only", "stand_by", "no_trade"}
    assert payload["strategy_analysis"]["available"] is True
    assert payload["config"]["strategy_analyzer_url"] == config["strategy_analyzer_url"]
    assert payload["github_summary"]["signal"] in {"active", "mixed", "stale"}
    assert payload["external_context"]["combined_signal"] == "neutral"
    assert "price_structure_score" in payload["decision"]["selected"]
    assert "validation" in payload["decision"]["selected"]


def test_build_digest_can_expand_to_top_futures_volume(tmp_path) -> None:
    config = {
        "symbols": ["BTCUSDT"],
        "include_top_futures_volume": True,
        "top_futures_volume_limit": 2,
        "analysis_limit": 2,
        "market": "futures",
        "interval": "4h",
        "news_feeds": ["https://example.com/feed.xml"],
        "news_limit": 3,
        "github_observability_repos": ["ccxt/ccxt"],
        "digest_root": str(tmp_path),
        "strategy_analyzer_url": "",
    }

    def fake_run_quant_analysis(symbol: str, market: str, interval: str) -> dict[str, object]:
        return {
            "symbol": symbol,
            "market": market,
            "interval": interval,
            "direction": "long",
            "composite_score": 80.0,
            "analysis": {"score": 80.0, "convergence": 0.8},
            "latest": {"adx": 20.0, "realized_vol_20": 0.8},
            "trade_plan": {},
            "run_id": f"run-{symbol}",
        }

    with (
        patch.object(
            daily_digest,
            "fetch_top_futures_symbols",
            return_value=[
                UniverseSymbol("SOLUSDT", 100.0, 1, "binance-futures-24hr-volume"),
                UniverseSymbol("DOGEUSDT", 80.0, 2, "binance-futures-24hr-volume"),
            ],
        ),
        patch.object(daily_digest, "fetch_rss_feed", return_value=[{"title": "ETF inflow remains resilient"}]),
        patch.object(daily_digest, "fetch_whale_alert_summary", return_value={"signal": "neutral"}),
        patch.object(daily_digest, "fetch_github_observability", return_value=[]),
        patch.object(daily_digest, "build_external_context", return_value={"combined_signal": "neutral"}),
        patch.object(daily_digest, "run_compact_json", return_value={"overall": "ok", "warnings": []}),
        patch.object(daily_digest, "run_quant_analysis", side_effect=fake_run_quant_analysis),
    ):
        payload = daily_digest.build_digest(config)

    assert payload["config"]["symbols"] == ["SOLUSDT", "DOGEUSDT"]
    assert payload["config"]["include_top_futures_volume"] is True
    assert payload["config"]["universe_symbols"][0]["rank"] == 1


def test_build_digest_excludes_open_position_symbols(tmp_path) -> None:
    config = {
        "symbols": ["BTCUSDT"],
        "exclude_symbols": ["SOLUSDT"],
        "include_top_futures_volume": True,
        "top_futures_volume_limit": 3,
        "analysis_limit": 3,
        "market": "futures",
        "interval": "4h",
        "news_feeds": ["https://example.com/feed.xml"],
        "news_limit": 3,
        "github_observability_repos": ["ccxt/ccxt"],
        "digest_root": str(tmp_path),
        "strategy_analyzer_url": "",
    }

    analyzed_symbols: list[str] = []

    def fake_run_quant_analysis(symbol: str, market: str, interval: str) -> dict[str, object]:
        analyzed_symbols.append(symbol)
        return {
            "symbol": symbol,
            "market": market,
            "interval": interval,
            "direction": "long",
            "composite_score": 80.0,
            "analysis": {"score": 80.0, "convergence": 0.8},
            "latest": {"adx": 20.0, "realized_vol_20": 0.8},
            "trade_plan": {},
            "run_id": f"run-{symbol}",
        }

    with (
        patch.object(
            daily_digest,
            "fetch_top_futures_symbols",
            return_value=[
                UniverseSymbol("SOLUSDT", 100.0, 1, "binance-futures-24hr-volume"),
                UniverseSymbol("DOGEUSDT", 80.0, 2, "binance-futures-24hr-volume"),
                UniverseSymbol("XRPUSDT", 60.0, 3, "binance-futures-24hr-volume"),
            ],
        ),
        patch.object(daily_digest, "fetch_rss_feed", return_value=[{"title": "ETF inflow remains resilient"}]),
        patch.object(daily_digest, "fetch_whale_alert_summary", return_value={"signal": "neutral"}),
        patch.object(daily_digest, "fetch_github_observability", return_value=[]),
        patch.object(daily_digest, "build_external_context", return_value={"combined_signal": "neutral"}),
        patch.object(daily_digest, "run_compact_json", return_value={"overall": "ok", "warnings": []}),
        patch.object(daily_digest, "run_quant_analysis", side_effect=fake_run_quant_analysis),
    ):
        payload = daily_digest.build_digest(config)

    assert analyzed_symbols == ["DOGEUSDT", "XRPUSDT"]
    assert payload["config"]["symbols"] == ["DOGEUSDT", "XRPUSDT"]
    assert payload["config"]["exclude_symbols"] == ["SOLUSDT"]
