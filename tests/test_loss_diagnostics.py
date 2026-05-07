from __future__ import annotations

import json
from pathlib import Path

import binance_quant_control.loss_diagnostics as diagnostics


def _review(
    *,
    symbol: str,
    route_id: str,
    side: str,
    pnl: float,
    source_order_id: str,
    score: int = 88,
    convergence: float = 0.92,
    exit_reason: str = "stop_loss",
    signal_scores: dict[str, float] | None = None,
    leverage: int = 3,
    opened_at: str = "2026-01-01T00:00:00+00:00",
    closed_at: str = "2026-01-01T04:00:00+00:00",
) -> dict[str, object]:
    return {
        "reviewed_at": "2026-01-01T00:00:00+00:00",
        "opened_at": opened_at,
        "closed_at": closed_at,
        "source_order_id": source_order_id,
        "symbol": symbol,
        "market": "futures",
        "side": side,
        "leverage": leverage,
        "exit_reason": exit_reason,
        "realized_pnl_usdt": pnl,
        "realized_pnl_pct": pnl,
        "realized_r_multiple": pnl,
        "analysis_score": score,
        "analysis_bias": "short-bias" if side == "SELL" else "long-bias",
        "analysis_convergence": convergence,
        "route_id": route_id,
        "asset_class": route_id.replace("-", "_"),
        "strategy_profile": "pytest",
        "market_regime_tag": "public-history",
        "signal_scores": signal_scores,
    }


def test_loss_diagnostics_flags_weak_short_lane(monkeypatch, tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    reviews = [
        _review(
            symbol="BTCUSDT",
            route_id="btc-core",
            side="SELL",
            pnl=-1.0,
            source_order_id=f"public-history-BTCUSDT-4h-{idx:06d}",
        )
        for idx in range(6)
    ]
    reviews.extend(
        [
            _review(
                symbol="BTCUSDT",
                route_id="btc-core",
                side="BUY",
                pnl=1.5,
                source_order_id=f"public-history-BTCUSDT-4h-win-{idx:06d}",
                exit_reason="take_profit",
            )
            for idx in range(3)
        ]
    )
    monkeypatch.setattr(diagnostics, "STATE_DIR", state_dir)
    monkeypatch.setattr(diagnostics, "LOSS_DIAGNOSTICS_DIR", state_dir / "loss-diagnostics")
    monkeypatch.setattr(diagnostics, "ensure_runtime_dirs", lambda: state_dir.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(diagnostics, "read_closed_trade_reviews", lambda: reviews)

    payload = diagnostics.run_loss_diagnostics(min_bucket_trades=3, top_n=5)

    assert payload["status"] == "ok"
    assert payload["summary"]["count"] == 9
    assert any("short-lane-underperforming" in item for item in payload["findings"])
    assert any(item["route_id"] == "btc-core" and item["side"] == "SELL" for item in payload["side_policy_recommendations"])
    cross_buckets = payload["buckets"]["by_route_side_score_convergence_bin"]
    assert any(
        item["dimensions"] == {
            "route_id": "btc-core",
            "side": "SELL",
            "score_bin": "score-081-100",
            "convergence_bin": "conv-090-100",
        }
        for item in cross_buckets
    )
    assert Path(payload["report_path"]).exists()
    report = json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))
    assert report["summary"]["losses"] == 6


def test_loss_diagnostics_adds_signal_and_fast_stop_root_causes(monkeypatch, tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    reviews = [
        _review(
            symbol="SOLUSDT",
            route_id="major-alt-trend",
            side="BUY",
            pnl=-0.8,
            source_order_id=f"training-SOLUSDT-loss-{idx:06d}",
            signal_scores={
                "flow_score": 34.0,
                "event_risk_score": 50.0,
                "execution_quality_score": 58.0,
                "composite_convergence_score": 49.0,
            },
            opened_at="2026-01-01T00:00:00+00:00",
            closed_at="2026-01-01T00:30:00+00:00",
        )
        for idx in range(4)
    ]
    reviews.append(
        _review(
            symbol="SOLUSDT",
            route_id="major-alt-trend",
            side="BUY",
            pnl=0.3,
            source_order_id="training-SOLUSDT-win-000001",
            exit_reason="take_profit",
            signal_scores={
                "flow_score": 34.0,
                "event_risk_score": 50.0,
                "execution_quality_score": 58.0,
                "composite_convergence_score": 49.0,
            },
            opened_at="2026-01-01T02:00:00+00:00",
            closed_at="2026-01-01T02:20:00+00:00",
        )
    )
    monkeypatch.setattr(diagnostics, "STATE_DIR", state_dir)
    monkeypatch.setattr(diagnostics, "LOSS_DIAGNOSTICS_DIR", state_dir / "loss-diagnostics")
    monkeypatch.setattr(diagnostics, "ensure_runtime_dirs", lambda: state_dir.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(diagnostics, "read_closed_trade_reviews", lambda: reviews)

    payload = diagnostics.run_loss_diagnostics(min_bucket_trades=3, top_n=8)

    assert any("fast-stop-cluster" in item for item in payload["findings"])
    assert any("weak-flow-loss-bucket" in item for item in payload["findings"])
    assert any(item["type"] == "fast-stop-cluster" for item in payload["root_cause_recommendations"])
    assert any(item["type"] == "weak-flow-confirmation" for item in payload["root_cause_recommendations"])
    assert "by_route_side_holding_time" in payload["buckets"]
    assert "by_route_side_flow_bin" in payload["buckets"]


def test_loss_diagnostics_respects_limit(monkeypatch, tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    reviews = [
        _review(
            symbol="ETHUSDT",
            route_id="eth-core",
            side="BUY",
            pnl=1.0,
            source_order_id=f"public-history-ETHUSDT-4h-{idx:06d}",
            exit_reason="take_profit",
        )
        for idx in range(4)
    ]
    monkeypatch.setattr(diagnostics, "STATE_DIR", state_dir)
    monkeypatch.setattr(diagnostics, "LOSS_DIAGNOSTICS_DIR", state_dir / "loss-diagnostics")
    monkeypatch.setattr(diagnostics, "ensure_runtime_dirs", lambda: state_dir.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(diagnostics, "read_closed_trade_reviews", lambda: reviews)

    payload = diagnostics.run_loss_diagnostics(limit=2, min_bucket_trades=1)

    assert payload["input"]["closed_review_count"] == 2
    assert payload["summary"]["count"] == 2
