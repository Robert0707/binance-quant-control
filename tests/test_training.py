from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import binance_quant_control.training as training
from binance_quant_control.order_journal import read_closed_trade_reviews, read_paper_orders


def test_demo_training_writes_replay_reviews(monkeypatch, tmp_path) -> None:
    state_dir = tmp_path / "state"
    reports_dir = tmp_path / "reports"
    training_dir = state_dir / "training"
    monkeypatch.setattr(training, "STATE_DIR", state_dir)
    monkeypatch.setattr(training, "DEFAULT_TRAINING_STATE_DIR", training_dir)
    monkeypatch.setattr("binance_quant_control.order_journal.STATE_DIR", state_dir)
    monkeypatch.setattr("binance_quant_control.order_journal.PAPER_ORDERS_FILE", state_dir / "paper-orders.jsonl")
    monkeypatch.setattr(
        "binance_quant_control.order_journal.CLOSED_TRADE_REVIEWS_FILE",
        state_dir / "closed-trade-reviews.jsonl",
    )
    monkeypatch.setattr(training, "ensure_runtime_dirs", lambda: (state_dir.mkdir(parents=True, exist_ok=True), reports_dir.mkdir(parents=True, exist_ok=True)))
    monkeypatch.setattr(training, "load_settings", lambda: SimpleNamespace())

    route = SimpleNamespace(
        route_id="meme-high-beta",
        asset_class="meme_high_beta",
        market="futures",
        interval="1h",
        simulation_mode="paper",
        review_lane="meme-beta-review",
        strategy_config=Path("config/strategy-meme-momentum.yaml"),
    )
    strategy = SimpleNamespace(
        profile="meme-momentum",
        path=Path("/tmp/strategy-meme-momentum.yaml"),
        defaults=SimpleNamespace(market="futures", interval="1h", limit=500, use_blave=False, render_chart=False),
        risk=SimpleNamespace(default_leverage=2, max_leverage=3),
    )

    monkeypatch.setattr(training, "resolve_symbol_route", lambda symbol: route)
    monkeypatch.setattr(
        training,
        "route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": False, "reasons": []},
    )
    monkeypatch.setattr(training, "load_strategy_config", lambda path: strategy)
    monkeypatch.setattr(training, "mission_candidate_side", lambda analysis_payload: "BUY")
    monkeypatch.setattr(
        training,
        "run_analysis",
        lambda settings, **kwargs: (
            {
                "symbol": kwargs["symbol"],
                "market": "futures",
                "analysis": {"bias": "long-bias", "score": 91, "convergence": 0.88},
                "latest": {"adx": 25.0},
                "trade_plan": {
                    "long": {"invalidation": 0.95, "take_profit_1": 1.08},
                    "short": {"invalidation": 1.05, "take_profit_1": 0.92},
                },
                "artifacts": {"report_json": "/tmp/analysis.json", "chart_path": None},
            },
            SimpleNamespace(output_dir=reports_dir / "sample"),
        ),
    )
    monkeypatch.setattr(
        training,
        "run_backtest",
        lambda settings, **kwargs: {
            "summary": {
                "trade_count": 2,
                "profit_factor": 1.7,
                "total_return_pct": 8.3,
                    "trades": [
                        {
                            "side": "BUY",
                            "entry_time": "2026-04-01T00:00:00+00:00",
                            "exit_time": "2026-04-01T01:00:00+00:00",
                            "entry_price": 1.0,
                        "exit_price": 1.08,
                        "pnl_pct": 7.5,
                        "pnl_r": 1.3,
                        "exit_reason": "take_profit",
                    }
                ],
            },
            "artifacts": {"report_json": "/tmp/backtest.json"},
        },
    )
    monkeypatch.setattr(
        training,
        "build_live_execution_plan",
        lambda settings, strategy, analysis_payload, **kwargs: SimpleNamespace(
            to_dict=lambda: {
                "allowed": True,
                "price": 1.0,
                "quantity": 8.0,
                "leverage": 2,
                "margin_notional_usdt": 4.0,
                "gross_notional_usdt": 8.0,
                "stop_price": 0.95,
                "take_profit_price": 1.08,
                "violations": [],
                "warnings": [],
                "challenge": {"optimizer_live_gate": {"required": False}},
            }
        ),
    )
    monkeypatch.setattr(training, "build_signal_scores", lambda **kwargs: {"composite_convergence_score": 77.0})
    monkeypatch.setattr(
        training,
        "run_strategy_optimizer",
        lambda: {
            "status": "ok",
            "review_count": 2,
            "promotion_decision": "watchlist",
            "report_path": "/tmp/optimizer.json",
        },
    )

    payload = training.run_demo_training(rounds=2, optimize_every=2)

    assert payload["status"] == "ok"
    assert payload["recorded_review_count"] == 2
    assert payload["wins"] == 2
    assert read_paper_orders()
    reviews = read_closed_trade_reviews()
    assert len(reviews) == 2
    assert all(item["route_id"] == "meme-high-beta" for item in reviews)
    assert all(item["challenge_status"] == "training" for item in reviews)


def test_demo_training_skips_backtest_side_mismatch(monkeypatch, tmp_path) -> None:
    state_dir = tmp_path / "state"
    reports_dir = tmp_path / "reports"
    training_dir = state_dir / "training"
    monkeypatch.setattr(training, "STATE_DIR", state_dir)
    monkeypatch.setattr(training, "DEFAULT_TRAINING_STATE_DIR", training_dir)
    monkeypatch.setattr("binance_quant_control.order_journal.STATE_DIR", state_dir)
    monkeypatch.setattr("binance_quant_control.order_journal.PAPER_ORDERS_FILE", state_dir / "paper-orders.jsonl")
    monkeypatch.setattr(
        "binance_quant_control.order_journal.CLOSED_TRADE_REVIEWS_FILE",
        state_dir / "closed-trade-reviews.jsonl",
    )
    monkeypatch.setattr(training, "ensure_runtime_dirs", lambda: (state_dir.mkdir(parents=True, exist_ok=True), reports_dir.mkdir(parents=True, exist_ok=True)))
    monkeypatch.setattr(training, "load_settings", lambda: SimpleNamespace())

    route = SimpleNamespace(
        route_id="eth-core",
        asset_class="eth_core",
        market="futures",
        interval="4h",
        simulation_mode="demo_testnet",
        review_lane="eth-trend-review",
        strategy_config=Path("config/strategy-eth-trend.yaml"),
    )
    strategy = SimpleNamespace(
        profile="eth-trend",
        path=Path("/tmp/strategy-eth-trend.yaml"),
        defaults=SimpleNamespace(market="futures", interval="4h", limit=500, use_blave=False, render_chart=False),
        risk=SimpleNamespace(default_leverage=3, max_leverage=3, min_convergence=0.74),
    )

    monkeypatch.setattr(training, "resolve_symbol_route", lambda symbol: route)
    monkeypatch.setattr(
        training,
        "route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": False, "reasons": []},
    )
    monkeypatch.setattr(training, "load_strategy_config", lambda path: strategy)
    monkeypatch.setattr(training, "mission_candidate_side", lambda analysis_payload: "SELL")
    monkeypatch.setattr(
        training,
        "run_analysis",
        lambda settings, **kwargs: (
            {
                "symbol": kwargs["symbol"],
                "market": "futures",
                "analysis": {"bias": "short-bias", "score": 70, "convergence": 0.8},
                "latest": {"adx": 22.0},
                "trade_plan": {
                    "long": {"invalidation": 0.95, "take_profit_1": 1.08},
                    "short": {"invalidation": 1.05, "take_profit_1": 0.92},
                },
                "artifacts": {"report_json": "/tmp/analysis.json", "chart_path": None},
            },
            SimpleNamespace(output_dir=reports_dir / "sample"),
        ),
    )
    monkeypatch.setattr(
        training,
        "run_backtest",
        lambda settings, **kwargs: {
            "summary": {
                "trade_count": 1,
                "profit_factor": 1.5,
                "total_return_pct": 5.0,
                "trades": [
                    {
                        "side": "BUY",
                        "entry_time": "2026-04-01T00:00:00+00:00",
                        "exit_time": "2026-04-01T04:00:00+00:00",
                        "entry_price": 1.0,
                        "exit_price": 1.1,
                        "pnl_pct": 5.0,
                        "pnl_r": 1.0,
                        "exit_reason": "take_profit",
                    }
                ],
            },
            "artifacts": {"report_json": "/tmp/backtest.json"},
        },
    )
    monkeypatch.setattr(
        training,
        "build_live_execution_plan",
        lambda settings, strategy, analysis_payload, **kwargs: SimpleNamespace(
            to_dict=lambda: {
                "allowed": True,
                "price": 1.0,
                "quantity": 10.0,
                "leverage": 3,
                "margin_notional_usdt": 4.0,
                "gross_notional_usdt": 12.0,
                "stop_price": 1.05,
                "take_profit_price": 0.92,
                "violations": [],
                "warnings": [],
                "challenge": {"optimizer_live_gate": {"required": False}},
            }
        ),
    )
    monkeypatch.setattr(training, "build_signal_scores", lambda **kwargs: {"composite_convergence_score": 65.0})
    monkeypatch.setattr(
        training,
        "run_strategy_optimizer",
        lambda: {"status": "ok", "review_count": 0, "promotion_decision": "reject", "report_path": "/tmp/optimizer.json"},
    )

    payload = training.run_demo_training(rounds=1, optimize_every=1)

    assert payload["recorded_review_count"] == 0
    assert payload["results"][0]["training_reason"] == "backtest-side-mismatch"
    assert read_closed_trade_reviews() == []


def test_demo_training_skips_when_live_plan_structure_is_blocked(monkeypatch, tmp_path) -> None:
    state_dir = tmp_path / "state"
    reports_dir = tmp_path / "reports"
    training_dir = state_dir / "training"
    monkeypatch.setattr(training, "STATE_DIR", state_dir)
    monkeypatch.setattr(training, "DEFAULT_TRAINING_STATE_DIR", training_dir)
    monkeypatch.setattr("binance_quant_control.order_journal.STATE_DIR", state_dir)
    monkeypatch.setattr("binance_quant_control.order_journal.PAPER_ORDERS_FILE", state_dir / "paper-orders.jsonl")
    monkeypatch.setattr(
        "binance_quant_control.order_journal.CLOSED_TRADE_REVIEWS_FILE",
        state_dir / "closed-trade-reviews.jsonl",
    )
    monkeypatch.setattr(
        training,
        "ensure_runtime_dirs",
        lambda: (state_dir.mkdir(parents=True, exist_ok=True), reports_dir.mkdir(parents=True, exist_ok=True)),
    )
    monkeypatch.setattr(training, "load_settings", lambda: SimpleNamespace())

    route = SimpleNamespace(
        route_id="major-alt-trend",
        asset_class="major_alt_trend",
        market="futures",
        interval="4h",
        simulation_mode="paper",
        review_lane="major-alt-review",
        strategy_config=Path("config/strategy-major-alt-trend.yaml"),
    )
    strategy = SimpleNamespace(
        profile="major-alt-trend",
        path=Path("/tmp/strategy-major-alt-trend.yaml"),
        defaults=SimpleNamespace(market="futures", interval="4h", limit=500, use_blave=False, render_chart=False),
        risk=SimpleNamespace(default_leverage=3, max_leverage=3, min_convergence=0.72),
    )

    monkeypatch.setattr(training, "resolve_symbol_route", lambda symbol: route)
    monkeypatch.setattr(
        training,
        "route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": False, "reasons": []},
    )
    monkeypatch.setattr(training, "load_strategy_config", lambda path: strategy)
    monkeypatch.setattr(training, "mission_candidate_side", lambda analysis_payload: "SELL")
    monkeypatch.setattr(
        training,
        "run_analysis",
        lambda settings, **kwargs: (
            {
                "symbol": kwargs["symbol"],
                "market": "futures",
                "analysis": {"bias": "short-bias", "score": 80, "convergence": 0.9},
                "latest": {"adx": 10.0},
                "trade_plan": {
                    "long": {"invalidation": 0.95, "take_profit_1": 1.08},
                    "short": {"invalidation": 1.05, "take_profit_1": 0.92},
                },
                "artifacts": {"report_json": "/tmp/analysis.json", "chart_path": None},
            },
            SimpleNamespace(output_dir=reports_dir / "sample"),
        ),
    )
    monkeypatch.setattr(
        training,
        "run_backtest",
        lambda settings, **kwargs: {
            "summary": {
                "trade_count": 1,
                "profit_factor": 1.5,
                "total_return_pct": 5.0,
                "trades": [
                    {
                        "side": "SELL",
                        "entry_time": "2026-04-01T00:00:00+00:00",
                        "exit_time": "2026-04-01T04:00:00+00:00",
                        "entry_price": 1.0,
                        "exit_price": 0.92,
                        "pnl_pct": 5.0,
                        "pnl_r": 1.0,
                        "exit_reason": "take_profit",
                    }
                ],
            },
            "artifacts": {"report_json": "/tmp/backtest.json"},
        },
    )
    monkeypatch.setattr(
        training,
        "build_live_execution_plan",
        lambda settings, strategy, analysis_payload, **kwargs: SimpleNamespace(
            to_dict=lambda: {
                "allowed": False,
                "price": 1.0,
                "quantity": 10.0,
                "leverage": 3,
                "margin_notional_usdt": 4.0,
                "gross_notional_usdt": 12.0,
                "stop_price": 1.05,
                "take_profit_price": 0.92,
                "violations": ["ADX 10.00 is below the minimum trend threshold 18.00."],
                "warnings": [],
            }
        ),
    )
    monkeypatch.setattr(training, "build_signal_scores", lambda **kwargs: {"composite_convergence_score": 65.0})
    monkeypatch.setattr(
        training,
        "run_strategy_optimizer",
        lambda: {"status": "ok", "review_count": 0, "promotion_decision": "reject", "report_path": "/tmp/optimizer.json"},
    )

    payload = training.run_demo_training(rounds=1, optimize_every=1)

    assert payload["recorded_review_count"] == 0
    assert payload["results"][0]["training_reason"] == "live-plan-structure-blocked"
    assert read_closed_trade_reviews() == []


def test_demo_training_skips_quarantined_route(monkeypatch, tmp_path) -> None:
    state_dir = tmp_path / "state"
    reports_dir = tmp_path / "reports"
    training_dir = state_dir / "training"
    monkeypatch.setattr(training, "STATE_DIR", state_dir)
    monkeypatch.setattr(training, "DEFAULT_TRAINING_STATE_DIR", training_dir)
    monkeypatch.setattr("binance_quant_control.order_journal.STATE_DIR", state_dir)
    monkeypatch.setattr("binance_quant_control.order_journal.PAPER_ORDERS_FILE", state_dir / "paper-orders.jsonl")
    monkeypatch.setattr(
        "binance_quant_control.order_journal.CLOSED_TRADE_REVIEWS_FILE",
        state_dir / "closed-trade-reviews.jsonl",
    )
    monkeypatch.setattr(
        training,
        "ensure_runtime_dirs",
        lambda: (state_dir.mkdir(parents=True, exist_ok=True), reports_dir.mkdir(parents=True, exist_ok=True)),
    )
    monkeypatch.setattr(training, "load_settings", lambda: SimpleNamespace())

    route = SimpleNamespace(
        route_id="btc-core",
        asset_class="btc_core",
        market="futures",
        interval="4h",
        simulation_mode="demo_testnet",
        review_lane="btc-volatility-review",
        strategy_config=Path("config/strategy-btc-volatility.yaml"),
    )
    strategy = SimpleNamespace(
        profile="btc-volatility",
        path=Path("/tmp/strategy-btc-volatility.yaml"),
        defaults=SimpleNamespace(market="futures", interval="4h", limit=500, use_blave=False, render_chart=False),
        risk=SimpleNamespace(default_leverage=3, max_leverage=3, min_convergence=0.78),
    )

    monkeypatch.setattr(training, "resolve_symbol_route", lambda symbol: route)
    monkeypatch.setattr(
        training,
        "route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": True, "reasons": ["loss-streak"]},
    )
    monkeypatch.setattr(training, "load_strategy_config", lambda path: strategy)
    monkeypatch.setattr(training, "mission_candidate_side", lambda analysis_payload: "BUY")
    monkeypatch.setattr(
        training,
        "run_analysis",
        lambda settings, **kwargs: (
            {
                "symbol": kwargs["symbol"],
                "market": "futures",
                "analysis": {"bias": "long-bias", "score": 90, "convergence": 0.9},
                "latest": {"adx": 30.0},
                "trade_plan": {
                    "long": {"invalidation": 0.95, "take_profit_1": 1.08},
                    "short": {"invalidation": 1.05, "take_profit_1": 0.92},
                },
                "artifacts": {"report_json": "/tmp/analysis.json", "chart_path": None},
            },
            SimpleNamespace(output_dir=reports_dir / "sample"),
        ),
    )
    monkeypatch.setattr(
        training,
        "run_backtest",
        lambda settings, **kwargs: {
            "summary": {
                "trade_count": 1,
                "profit_factor": 2.0,
                "total_return_pct": 5.0,
                "trades": [
                    {
                        "side": "BUY",
                        "entry_time": "2026-04-01T00:00:00+00:00",
                        "exit_time": "2026-04-01T04:00:00+00:00",
                        "entry_price": 1.0,
                        "exit_price": 1.08,
                        "pnl_pct": 5.0,
                        "pnl_r": 1.0,
                        "exit_reason": "take_profit",
                    }
                ],
            },
            "artifacts": {"report_json": "/tmp/backtest.json"},
        },
    )
    monkeypatch.setattr(
        training,
        "build_live_execution_plan",
        lambda settings, strategy, analysis_payload, **kwargs: SimpleNamespace(
            to_dict=lambda: {
                "allowed": True,
                "price": 1.0,
                "quantity": 10.0,
                "leverage": 3,
                "margin_notional_usdt": 4.0,
                "gross_notional_usdt": 12.0,
                "stop_price": 0.95,
                "take_profit_price": 1.08,
                "violations": [],
                "warnings": [],
            }
        ),
    )
    monkeypatch.setattr(training, "build_signal_scores", lambda **kwargs: {"composite_convergence_score": 80.0})
    monkeypatch.setattr(
        training,
        "run_strategy_optimizer",
        lambda: {"status": "ok", "review_count": 0, "promotion_decision": "reject", "report_path": "/tmp/optimizer.json"},
    )

    payload = training.run_demo_training(rounds=1, optimize_every=1)

    assert payload["recorded_review_count"] == 0
    assert payload["results"][0]["training_reason"] == "route-quarantined-pending-manual-review"
    assert read_closed_trade_reviews() == []
