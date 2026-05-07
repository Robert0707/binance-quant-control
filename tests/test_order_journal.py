from __future__ import annotations

from binance_quant_control.order_journal import (
    ClosedTradeReviewRecord,
    PaperOrderRecord,
    append_closed_trade_review,
    append_paper_order,
    backfill_closed_trade_review_metadata,
    backfill_live_order_metadata,
    read_closed_trade_reviews,
    read_paper_orders,
    summarize_closed_trade_reviews,
    summarize_paper_orders,
)


def test_closed_trade_review_round_trip(monkeypatch, tmp_path):
    journal_path = tmp_path / "closed-trade-reviews.jsonl"
    monkeypatch.setattr("binance_quant_control.order_journal.CLOSED_TRADE_REVIEWS_FILE", journal_path)

    append_closed_trade_review(
        ClosedTradeReviewRecord(
            reviewed_at="2026-04-27T10:00:00+00:00",
            opened_at="2026-04-27T09:00:00+00:00",
            closed_at="2026-04-27T09:30:00+00:00",
            source_order_id=123,
            symbol="BTCUSDT",
            market="futures",
            side="BUY",
            quantity=0.01,
            leverage=3,
            entry_price=100000.0,
            exit_price=101000.0,
            stop_loss_price=99000.0,
            take_profit_price=101000.0,
            exit_reason="take_profit",
            realized_pnl_usdt=10.0,
            realized_pnl_pct=1.0,
            realized_r_multiple=1.0,
            analysis_score=90,
            analysis_bias="long-bias",
            analysis_convergence=0.9,
            strategy_profile="btc-volatility",
            strategy_path="config/strategy-btc-volatility.yaml",
            asset_class="btc_core",
            route_id="btc-core",
            review_lane="btc-volatility-review",
            challenge_status="inactive",
            challenge_progress_pct=0.0,
        )
    )

    rows = read_closed_trade_reviews()
    assert len(rows) == 1
    assert rows[0]["symbol"] == "BTCUSDT"
    summary = summarize_closed_trade_reviews()
    assert summary["count"] == 1
    assert summary["wins"] == 1
    assert summary["total_realized_pnl_usdt"] == 10.0
    assert summary["by_route"]["btc-core"] == 1
    assert summary["by_asset_class"]["btc_core"] == 1


def test_paper_order_round_trip(monkeypatch, tmp_path):
    journal_path = tmp_path / "paper-orders.jsonl"
    monkeypatch.setattr("binance_quant_control.order_journal.PAPER_ORDERS_FILE", journal_path)

    append_paper_order(
        PaperOrderRecord(
            generated_at="2026-04-28T10:00:00+00:00",
            kind="paper-order",
            symbol="DOGEUSDT",
            market="futures",
            side="BUY",
            margin_notional_usdt=3.0,
            leverage=2.0,
            gross_notional_usdt=6.0,
            reference_price=0.12,
            estimated_quantity=50.0,
            analysis_bias="long-bias",
            analysis_score=78,
            analysis_convergence=0.81,
            cohort_id="meme_high_beta:meme-momentum:futures:1h",
            strategy_profile="meme-momentum",
            strategy_path="config/strategy-meme-momentum.yaml",
            asset_class="meme_high_beta",
            route_id="meme-high-beta",
            simulation_mode="paper",
            review_lane="meme-beta-review",
            entry_reason_snapshot={"bias": "long-bias", "interval": "1h"},
            signal_scores={"composite_convergence_score": 72.0},
            analysis_report="/tmp/report.json",
            chart_path=None,
            note="test",
        )
    )

    rows = read_paper_orders()
    assert len(rows) == 1
    assert rows[0]["route_id"] == "meme-high-beta"
    summary = summarize_paper_orders()
    assert summary["count"] == 1
    assert summary["by_route"]["meme-high-beta"] == 1
    assert summary["by_cohort"]["meme_high_beta:meme-momentum:futures:1h"] == 1


def test_backfill_live_order_metadata_infers_route_context():
    rows, changed = backfill_live_order_metadata(
        [
            {
                "timestamp": "2026-04-28T00:00:00+00:00",
                "symbol": "BTCUSDT",
                "market": "futures",
                "analysis_score": 81,
                "analysis_bias": "long-bias",
                "analysis_convergence": 0.83,
                "note": "strategy=btc-volatility",
            }
        ]
    )

    assert changed == 1
    assert rows[0]["route_id"] == "btc-core"
    assert rows[0]["cohort_id"] == "btc_core:btc-volatility:futures:4h"
    assert rows[0]["entry_reason_snapshot"]["interval"] == "4h"


def test_backfill_closed_trade_review_metadata_infers_route_context():
    rows, changed = backfill_closed_trade_review_metadata(
        [
            {
                "symbol": "DOGEUSDT",
                "market": "futures",
                "analysis_score": 77,
                "analysis_bias": "long-bias",
                "analysis_convergence": 0.76,
                "note": "strategy=meme-momentum",
            }
        ]
    )

    assert changed == 1
    assert rows[0]["route_id"] == "doge-meme-high-beta"
    assert rows[0]["asset_class"] == "meme_high_beta"
    assert rows[0]["cohort_id"] == "meme_high_beta:meme-momentum:futures:4h"
