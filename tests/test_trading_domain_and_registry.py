from __future__ import annotations

from pathlib import Path

from binance_quant_control.feature_registry import build_feature_manifest
from binance_quant_control.portfolio_construction import (
    PortfolioConstructionPolicy,
    build_portfolio_target,
)
from binance_quant_control.signal_api import append_trading_signal, read_trading_signals
from binance_quant_control.signal_schema import TradingSignal
from binance_quant_control.skipped_signal_journal import append_skipped_signal, read_skipped_signals
from binance_quant_control.trading_domain import (
    FillEvent,
    OrderIntent,
    OrderSnapshot,
    PositionSnapshot,
)


def test_trading_domain_serializes_order_fill_and_position() -> None:
    order = OrderIntent(symbol="BTCUSDT", side="BUY", order_kind="LIMIT", quantity=0.1, price=50000)
    order_snapshot = OrderSnapshot(
        order_id="abc",
        client_order_id="client-abc",
        symbol="BTCUSDT",
        side="BUY",
        order_kind="LIMIT",
        quantity=0.1,
        filled_quantity=0.04,
        status="PARTIALLY_FILLED",
        price=50000,
    )
    fill = FillEvent(symbol="BTCUSDT", side="BUY", quantity=0.1, price=50000, fee_usdt=2.0)
    position = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        quantity=0.1,
        entry_price=50000,
        mark_price=50500,
        unrealized_pnl_usdt=50.0,
    )

    assert order.notional_usdt == 5000
    assert order_snapshot.remaining_quantity == 0.06
    assert fill.gross_notional_usdt == 5000
    assert position.is_open is True
    assert position.gross_notional_usdt == 5050
    assert order.to_dict()["symbol"] == "BTCUSDT"


def test_feature_manifest_contains_live_safe_triple_barrier_label() -> None:
    manifest = build_feature_manifest()

    assert manifest["live_safe"] is True
    assert len(manifest["manifest_hash"]) == 16
    assert manifest["pipeline_contract"]["lookahead_allowed"] is False
    assert any(item["name"] == "rolling_vwap_reclaim" for item in manifest["features"])
    assert manifest["labels"][0]["method"] == "take_profit_stop_loss_time_limit"


def test_skipped_signal_journal_round_trips(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("binance_quant_control.skipped_signal_journal.ensure_runtime_dirs", lambda: None)
    path = tmp_path / "skipped.jsonl"

    written = append_skipped_signal(
        symbol="ethusdt",
        side="buy",
        route_id="eth-core",
        strategy_family="trend_pullback",
        gate="professional_entry_gate",
        blockers=["expectancy-r-below-floor"],
        expectancy_r=-0.1,
        path=path,
    )
    rows = read_skipped_signals(path)

    assert written == Path(path)
    assert rows[0]["symbol"] == "ETHUSDT"
    assert rows[0]["blockers"] == ["expectancy-r-below-floor"]


def test_signal_api_ledger_round_trips(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("binance_quant_control.signal_api.ensure_runtime_dirs", lambda: None)
    path = tmp_path / "signals.jsonl"
    signal = TradingSignal(
        signal_id="BTCUSDT:4h:trend_pullback:BUY",
        symbol="BTCUSDT",
        side="BUY",
        interval="4h",
        strategy_family="trend_pullback",
        route_id="btc-core",
        status="rejected",
        blockers=["expectancy-r-not-positive"],
    )

    written = append_trading_signal(signal, gate={"allowed": False}, path=path)
    rows = read_trading_signals(path)

    assert written == path
    assert rows[0]["schema"] == "binance_quant_control.trading_signal.v1"
    assert rows[0]["signal"]["symbol"] == "BTCUSDT"
    assert rows[0]["gate"]["allowed"] is False


def test_portfolio_policy_allows_fourth_position_but_blocks_fifth() -> None:
    open_positions = [
        {"symbol": "BTCUSDT", "side": "LONG", "qty": 0.001, "open_risk_pct": 0.003, "route_id": "btc-core"},
        {"symbol": "ETHUSDT", "side": "LONG", "qty": 0.01, "open_risk_pct": 0.003, "route_id": "eth-core"},
        {"symbol": "SOLUSDT", "side": "LONG", "qty": 0.1, "open_risk_pct": 0.003, "route_id": "sol-core"},
    ]
    signal = {
        "symbol": "BNBUSDT",
        "side": "BUY",
        "route_id": "bnb-core",
        "target_risk_pct": 0.003,
        "signal_score": 100,
        "expectancy_r": 0.2,
        "payoff_ratio": 1.5,
    }

    fourth = build_portfolio_target(signal, open_positions=open_positions)
    fifth = build_portfolio_target(
        {**signal, "symbol": "DOGEUSDT", "route_id": "doge-core"},
        open_positions=[*open_positions, {"symbol": "BNBUSDT", "side": "LONG", "qty": 0.1, "open_risk_pct": 0.003, "route_id": "bnb-core"}],
        policy=PortfolioConstructionPolicy(),
    )

    assert fourth.accepted is True
    assert "max-concurrent-positions-reached" in fifth.blockers
