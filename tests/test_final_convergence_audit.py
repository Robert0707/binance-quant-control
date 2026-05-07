from __future__ import annotations

from pathlib import Path

import binance_quant_control.final_convergence_audit as audit
from binance_quant_control.order_journal import ClosedTradeReviewRecord, append_closed_trade_review


def test_final_convergence_audit_reports_core_sample_gap(monkeypatch, tmp_path) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setattr(audit, "STATE_DIR", state_dir)
    monkeypatch.setattr(audit, "FINAL_AUDIT_STATE_DIR", state_dir / "final-convergence-audit")
    monkeypatch.setattr("binance_quant_control.order_journal.CLOSED_TRADE_REVIEWS_FILE", state_dir / "closed-trade-reviews.jsonl")
    monkeypatch.setattr("binance_quant_control.order_journal.LIVE_ORDERS_FILE", state_dir / "live-orders.jsonl")
    monkeypatch.setattr("binance_quant_control.order_journal.PAPER_ORDERS_FILE", state_dir / "paper-orders.jsonl")
    monkeypatch.setattr(audit, "ensure_runtime_dirs", lambda: state_dir.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(
        audit,
        "run_strategy_optimizer",
        lambda: {
            "status": "ok",
            "review_count": 1,
            "screening_status": "failed",
            "validation_status": "failed",
            "promotion_decision": "reject",
            "report_path": "/tmp/optimizer.json",
        },
    )

    append_closed_trade_review(
        ClosedTradeReviewRecord(
            reviewed_at="2026-04-28T00:00:00+00:00",
            opened_at="2026-04-27T00:00:00+00:00",
            closed_at="2026-04-27T04:00:00+00:00",
            source_order_id="x",
            symbol="BTCUSDT",
            market="futures",
            side="BUY",
            quantity=0.01,
            leverage=3,
            entry_price=100.0,
            exit_price=101.0,
            stop_loss_price=None,
            take_profit_price=None,
            exit_reason="take_profit",
            realized_pnl_usdt=1.0,
            realized_pnl_pct=1.0,
            realized_r_multiple=1.0,
            analysis_score=80,
            analysis_bias="long-bias",
            analysis_convergence=0.8,
            cohort_id="btc_core:btc-volatility:futures:4h",
            strategy_profile="btc-volatility",
            asset_class="btc_core",
            route_id="btc-core",
            review_lane="btc-volatility-review",
        )
    )

    payload = audit.run_final_convergence_audit(run_hailo=False)

    assert payload["status"] == "needs-attention"
    assert payload["review_database"]["closed_review_count"] == 1
    assert any("btc-core" in item for item in payload["findings"])
    assert Path(payload["report_path"]).exists()
