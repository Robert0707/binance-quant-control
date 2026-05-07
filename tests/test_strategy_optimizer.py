from __future__ import annotations

from binance_quant_control.asset_routing import resolve_symbol_route
from binance_quant_control.strategy import load_strategy_config
from binance_quant_control.strategy_optimizer import (
    TUNABLE_STRATEGY_PATHS,
    _balanced_recent_reviews,
    _build_strategy_payload,
    _cohort_convergence_report,
    _mutation_scope,
    _prepare_reviews,
    _review_robustness_report,
    load_optimizer_config,
    tune_strategy_from_reviews,
)


def test_tune_strategy_tightens_filters_on_stop_loss_cluster() -> None:
    strategy = load_strategy_config("config/strategy-hermes-pro.yaml")
    reviews = [
        {"exit_reason": "stop_loss", "realized_pnl_usdt": -1, "realized_pnl_pct": -1.4, "realized_r_multiple": -1.0},
        {"exit_reason": "stop_loss", "realized_pnl_usdt": -1, "realized_pnl_pct": -1.1, "realized_r_multiple": -0.9},
        {"exit_reason": "stop_loss", "realized_pnl_usdt": -1, "realized_pnl_pct": -1.2, "realized_r_multiple": -1.1},
        {"exit_reason": "take_profit", "realized_pnl_usdt": 1, "realized_pnl_pct": 0.7, "realized_r_multiple": 0.6},
        {"exit_reason": "manual_close", "realized_pnl_usdt": -1, "realized_pnl_pct": -0.8, "realized_r_multiple": -0.5},
        {"exit_reason": "stop_loss", "realized_pnl_usdt": -1, "realized_pnl_pct": -1.0, "realized_r_multiple": -1.0},
    ]
    tuned, notes, stats = tune_strategy_from_reviews(strategy, reviews)
    assert tuned["risk"]["min_adx"] > strategy.risk.min_adx
    assert tuned["risk"]["min_score_long"] > strategy.risk.min_score_long
    assert tuned["risk"]["max_score_short"] < strategy.risk.max_score_short
    assert stats["stop_loss_ratio"] > 0.45
    assert notes


def test_tune_strategy_lets_winners_run_on_healthy_profile() -> None:
    strategy = load_strategy_config("config/strategy-hermes-pro.yaml")
    reviews = [
        {"exit_reason": "take_profit", "realized_pnl_usdt": 1, "realized_pnl_pct": 1.6, "realized_r_multiple": 1.5},
        {"exit_reason": "take_profit", "realized_pnl_usdt": 1, "realized_pnl_pct": 1.8, "realized_r_multiple": 1.7},
        {"exit_reason": "take_profit", "realized_pnl_usdt": 1, "realized_pnl_pct": 1.4, "realized_r_multiple": 1.2},
        {"exit_reason": "manual_close", "realized_pnl_usdt": 1, "realized_pnl_pct": 1.1, "realized_r_multiple": 1.0},
        {"exit_reason": "stop_loss", "realized_pnl_usdt": -1, "realized_pnl_pct": -0.5, "realized_r_multiple": -0.4},
        {"exit_reason": "take_profit", "realized_pnl_usdt": 1, "realized_pnl_pct": 1.9, "realized_r_multiple": 1.8},
    ]
    tuned, _, stats = tune_strategy_from_reviews(strategy, reviews)
    assert tuned["risk"]["take_profit_r_multiples"][0] >= strategy.risk.take_profit_r_multiples[0]
    assert stats["win_rate"] > 0.6


def test_optimizer_mutation_scope_stays_within_strategy_allowlist() -> None:
    strategy = load_strategy_config("config/strategy-hermes-pro.yaml")
    reviews = [
        {"exit_reason": "stop_loss", "realized_pnl_usdt": -1, "realized_pnl_pct": -1.4, "realized_r_multiple": -1.0},
        {"exit_reason": "stop_loss", "realized_pnl_usdt": -1, "realized_pnl_pct": -1.1, "realized_r_multiple": -0.9},
        {"exit_reason": "take_profit", "realized_pnl_usdt": 1, "realized_pnl_pct": 0.9, "realized_r_multiple": 0.8},
        {"exit_reason": "manual_close", "realized_pnl_usdt": -1, "realized_pnl_pct": -0.8, "realized_r_multiple": -0.5},
        {"exit_reason": "stop_loss", "realized_pnl_usdt": -1, "realized_pnl_pct": -1.0, "realized_r_multiple": -1.0},
        {"exit_reason": "take_profit", "realized_pnl_usdt": 1, "realized_pnl_pct": 0.8, "realized_r_multiple": 0.7},
    ]
    tuned, _, _ = tune_strategy_from_reviews(strategy, reviews)
    base_payload = _build_strategy_payload(strategy, description_suffix="Auto-tuned from closed-trade reviews.")
    changed_paths, unexpected_paths = _mutation_scope(base_payload, tuned)
    assert changed_paths
    assert not unexpected_paths
    assert set(changed_paths).issubset(TUNABLE_STRATEGY_PATHS)


def test_tune_strategy_reports_loss_streak() -> None:
    strategy = load_strategy_config("config/strategy-hermes-pro.yaml")
    reviews = [
        {"exit_reason": "take_profit", "realized_pnl_usdt": 1, "realized_pnl_pct": 1.0, "realized_r_multiple": 1.0},
        {"exit_reason": "stop_loss", "realized_pnl_usdt": -1, "realized_pnl_pct": -1.0, "realized_r_multiple": -1.0},
        {"exit_reason": "manual_close", "realized_pnl_usdt": -1, "realized_pnl_pct": -0.5, "realized_r_multiple": -0.5},
    ]
    _, _, stats = tune_strategy_from_reviews(strategy, reviews)
    assert stats["loss_streak"] == 2


def test_cohort_convergence_report_marks_elite_candidate() -> None:
    route = resolve_symbol_route("BTCUSDT")
    config_path = "config/strategy-optimizer.default.yaml"
    from binance_quant_control.strategy_optimizer import load_optimizer_config

    config = load_optimizer_config(config_path)
    reviews = []
    for idx in range(100):
        pnl = 1.0
        if idx in {5, 18, 31, 44, 52, 67, 88}:
            pnl = -0.05
        reviews.append(
            {
                "cohort_id": "btc_core:btc-volatility:futures:4h",
                "route_id": route.route_id,
                "asset_class": route.asset_class,
                "strategy_profile": "btc-volatility",
                "review_lane": route.review_lane,
                "symbol": "BTCUSDT",
                "realized_pnl_usdt": pnl,
            }
        )
    report = _cohort_convergence_report(reviews, config)
    assert report[0]["promotion_decision"] in {"promote", "elite_candidate"}


def test_cohort_convergence_report_sorts_promotion_before_watchlist() -> None:
    config = load_optimizer_config("config/strategy-optimizer.default.yaml")
    reviews = []
    for idx in range(100):
        reviews.append(
            {
                "cohort_id": "btc_core:btc-volatility:futures:4h",
                "route_id": "btc-core",
                "asset_class": "core",
                "strategy_profile": "btc-volatility",
                "review_lane": "core",
                "symbol": "BTCUSDT",
                "realized_pnl_usdt": 1.8 if idx % 5 else -0.05,
                "realized_r_multiple": 1.8 if idx % 5 else -0.05,
            }
        )
        reviews.append(
            {
                "cohort_id": "eth_core:eth-trend:futures:4h",
                "route_id": "eth-core",
                "asset_class": "core",
                "strategy_profile": "eth-trend",
                "review_lane": "core",
                "symbol": "ETHUSDT",
                "realized_pnl_usdt": 1.0 if idx % 3 else -0.6,
                "realized_r_multiple": 1.0 if idx % 3 else -0.6,
            }
        )

    report = _cohort_convergence_report(reviews, config)

    assert report[0]["promotion_decision"] in {"promote", "elite_candidate"}
    assert report[0]["promotion_decision"] != "watchlist"


def test_prepare_reviews_backfills_route_and_filters_flat_manual_closes() -> None:
    config = load_optimizer_config("config/strategy-optimizer.default.yaml")
    reviews = [
        {
            "symbol": "PENGUUSDT",
            "market": "futures",
            "exit_reason": "manual_close",
            "realized_pnl_usdt": 0.0,
            "note": "strategy=micro-account-pilot",
        },
        {
            "symbol": "DOGEUSDT",
            "market": "futures",
            "exit_reason": "take_profit",
            "realized_pnl_usdt": 1.0,
            "note": "strategy=meme-momentum",
        },
    ]
    prepared, hygiene = _prepare_reviews(reviews, config)
    assert hygiene["input_reviews"] == 2
    assert hygiene["flat_manual_closes"] == 1
    assert len(prepared) == 1
    assert prepared[0]["route_id"] == "doge-meme-high-beta"
    assert prepared[0]["cohort_id"].startswith("meme_high_beta:")


def test_balanced_recent_reviews_keeps_multiple_routes_in_view() -> None:
    reviews = (
        [{"route_id": "btc-core", "seq": idx} for idx in range(3)]
        + [{"route_id": "eth-core", "seq": idx} for idx in range(3)]
        + [{"route_id": "meme-high-beta", "seq": idx} for idx in range(100)]
    )

    selected = _balanced_recent_reviews(reviews, per_route_limit=2, total_limit=10)

    assert [item["seq"] for item in selected if item["route_id"] == "btc-core"] == [1, 2]
    assert [item["seq"] for item in selected if item["route_id"] == "eth-core"] == [1, 2]
    assert [item["seq"] for item in selected if item["route_id"] == "meme-high-beta"] == [98, 99]


def test_review_robustness_report_flags_unstable_closed_review_folds() -> None:
    config = load_optimizer_config("config/strategy-optimizer.default.yaml")
    reviews = []
    for idx in range(40):
        pnl = 1.0 if idx >= 30 else -0.2
        reviews.append(
            {
                "symbol": "BTCUSDT",
                "route_id": "btc-core",
                "exit_reason": "take_profit" if pnl > 0 else "stop_loss",
                "realized_pnl_usdt": pnl,
                "realized_r_multiple": 1.0 if pnl > 0 else -1.0,
            }
        )

    report = _review_robustness_report(reviews, config)

    assert report["passed"] is False
    assert any("too-many-review-folds-below-1pf" in reason for reason in report["reasons"])
