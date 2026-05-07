from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import binance_quant_control.public_history_training as pht
from binance_quant_control.order_journal import read_closed_trade_reviews


def test_public_history_training_writes_review_samples(monkeypatch, tmp_path) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setattr(pht, "STATE_DIR", state_dir)
    monkeypatch.setattr(pht, "PUBLIC_HISTORY_STATE_DIR", state_dir / "public-history-training")
    monkeypatch.setattr("binance_quant_control.order_journal.CLOSED_TRADE_REVIEWS_FILE", state_dir / "closed-trade-reviews.jsonl")
    monkeypatch.setattr(pht, "ensure_runtime_dirs", lambda: state_dir.mkdir(parents=True, exist_ok=True))

    route = SimpleNamespace(
        route_id="btc-core",
        asset_class="btc_core",
        market="futures",
        interval="4h",
        review_lane="btc-volatility-review",
        strategy_config=Path("config/strategy-btc-volatility.yaml"),
    )
    strategy = SimpleNamespace(
        profile="btc-volatility",
        path=Path("/tmp/strategy-btc-volatility.yaml"),
        risk=SimpleNamespace(default_leverage=3),
    )
    frame = pd.DataFrame({"close": [1.0, 1.1]})

    monkeypatch.setattr(pht, "resolve_symbol_route", lambda symbol: route)
    monkeypatch.setattr(pht, "load_strategy_config", lambda path: strategy)
    monkeypatch.setattr(
        pht,
        "fetch_public_history_frame",
        lambda **kwargs: (frame, [{"source": "binance-public-data", "rows": 2}]),
    )
    monkeypatch.setattr(
        pht,
        "simulate_backtest",
        lambda frame, market, strategy: {
            "trade_count": 1,
            "wins": 1,
            "losses": 0,
            "win_rate": 100.0,
            "profit_factor": 2.0,
            "total_return_pct": 1.2,
            "max_drawdown_pct": 0.0,
            "trades": [
                {
                    "side": "BUY",
                    "entry_time": "2025-01-01T00:00:00+00:00",
                    "exit_time": "2025-01-01T04:00:00+00:00",
                    "entry_price": 100.0,
                    "exit_price": 102.0,
                    "pnl_pct": 1.2,
                    "pnl_r": 0.8,
                    "exit_reason": "take_profit",
                    "analysis_score": 88,
                    "analysis_convergence": 0.86,
                }
            ],
        },
    )
    monkeypatch.setattr(
        pht,
        "run_strategy_optimizer",
        lambda: {
            "status": "ok",
            "review_count": 1,
            "promotion_decision": "watchlist",
            "report_path": "/tmp/optimizer.json",
        },
    )

    payload = pht.run_public_history_training(
        symbols=["BTCUSDT"],
        months=1,
        end_month="2025-01",
        max_reviews_per_symbol=10,
        optimize_every=10,
    )

    assert payload["inserted_review_count"] == 1
    rows = read_closed_trade_reviews()
    assert len(rows) == 1
    assert rows[0]["route_id"] == "btc-core"
    assert rows[0]["challenge_status"] == "public-history-training"
    assert "binance_public_history" in rows[0]["note"]
    assert "000000" not in rows[0]["source_hash"]
    assert rows[0]["source_hash"].startswith("public-history:BTCUSDT:btc-core:futures:4h")
