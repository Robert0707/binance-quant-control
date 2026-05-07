from __future__ import annotations

from pathlib import Path

import binance_quant_control.ai_market_sentinel as sentinel
from binance_quant_control.trading_control import TradingControlState


class FakeSettings:
    pass


class FakeClient:
    def __init__(self, settings):
        self.settings = settings

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def positions(self, symbol=None):
        return [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.001",
                "entryPrice": "80000",
                "unRealizedProfit": "1.25",
                "leverage": "3",
            }
        ]

    def klines(self, symbol, interval, limit, market):
        rows = []
        price = 79000.0
        for idx in range(limit):
            price += 100.0
            rows.append(
                [
                    idx,
                    str(price - 50.0),
                    str(price + 100.0),
                    str(price - 100.0),
                    str(price),
                    "1000",
                    idx + 1,
                    "0",
                    1,
                    "0",
                    "0",
                    "0",
                ]
            )
        return rows


def test_ai_market_sentinel_prioritizes_position_guardian_and_blocks_expansion(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sentinel, "STATE_DIR", tmp_path)
    monkeypatch.setattr(sentinel, "load_settings", lambda: FakeSettings())
    monkeypatch.setattr(sentinel, "BinanceClient", FakeClient)
    monkeypatch.setattr(
        sentinel,
        "load_trading_control_state",
        lambda: TradingControlState(paused=True, reason="protect exposure", updated_by="pytest"),
    )
    monkeypatch.setattr(
        sentinel,
        "load_route_risk_state",
        lambda: {
            "active_quarantined_routes": ["btc-core"],
            "routes": {
                "btc-core": {
                    "quarantined": True,
                    "reasons": ["profit-factor below floor"],
                    "metrics": {"profit_factor": 0.37},
                }
            },
        },
    )
    monkeypatch.setattr(
        sentinel,
        "run_ai_readiness_scan",
        lambda **kwargs: {
            "candidate_count": 3,
            "allowed_count": 0,
            "next_machine_action": "repair_exchange_sizing_or_margin",
            "hard_blocker_taxonomy": {"strategy_performance": ["Recent expectancy -0.16R."]},
            "report_path": "state/readiness.json",
        },
    )

    payload = sentinel.run_ai_market_sentinel(symbols=["BTCUSDT"], limit=80, output_dir=tmp_path)

    assert payload["safety"]["opens_orders"] is False
    assert payload["position_state"]["open_position_count"] == 1
    assert payload["readiness"]["allowed_count"] == 0
    assert payload["expansion_gate"]["allowed"] is False
    assert "trading-control-paused" in payload["expansion_gate"]["blockers"]
    assert "no-readiness-approved-candidate" in payload["expansion_gate"]["blockers"]
    assert payload["machine_action_queue"][0]["action"] == "run_position_guardian"
    assert payload["trend_state"]["BTCUSDT"]["bias"] == "long"
    assert Path(payload["report_path"]).exists()


def test_ai_market_sentinel_allows_expansion_when_positions_below_cap(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sentinel, "STATE_DIR", tmp_path)
    monkeypatch.setattr(sentinel, "load_settings", lambda: FakeSettings())
    monkeypatch.setattr(sentinel, "BinanceClient", FakeClient)
    monkeypatch.setattr(sentinel, "load_trading_control_state", lambda: TradingControlState())
    monkeypatch.setattr(sentinel, "load_route_risk_state", lambda: {"active_quarantined_routes": [], "routes": {}})
    monkeypatch.setattr(
        sentinel,
        "run_ai_readiness_scan",
        lambda **kwargs: {
            "candidate_count": 3,
            "allowed_count": 1,
            "selected_ready_candidate": {"symbol": "SOLUSDT", "side": "BUY"},
            "next_machine_action": "execute_ready_dry_run_only",
            "hard_blocker_taxonomy": {},
            "execution_ticket": {"symbol": "SOLUSDT"},
            "report_path": "state/readiness.json",
        },
    )

    payload = sentinel.run_ai_market_sentinel(symbols=["BTCUSDT"], limit=80, output_dir=tmp_path)

    assert payload["position_state"]["open_position_count"] == 1
    assert payload["expansion_gate"]["allowed"] is True
    assert "open-position-management-priority" not in payload["expansion_gate"]["blockers"]
    assert payload["machine_action_queue"][0]["action"] == "run_position_guardian"
    assert any(item["action"] == "operator_testnet_preflight" for item in payload["machine_action_queue"])


def test_ai_market_sentinel_blocks_expansion_at_four_positions(monkeypatch, tmp_path: Path) -> None:
    class FourPositionClient(FakeClient):
        def positions(self, symbol=None):
            return [
                {"symbol": "BTCUSDT", "positionAmt": "0.001", "entryPrice": "80000", "unRealizedProfit": "1", "leverage": "3"},
                {"symbol": "ETHUSDT", "positionAmt": "0.01", "entryPrice": "3000", "unRealizedProfit": "1", "leverage": "3"},
                {"symbol": "SOLUSDT", "positionAmt": "0.1", "entryPrice": "180", "unRealizedProfit": "1", "leverage": "3"},
                {"symbol": "BNBUSDT", "positionAmt": "0.1", "entryPrice": "600", "unRealizedProfit": "1", "leverage": "3"},
            ]

    monkeypatch.setattr(sentinel, "STATE_DIR", tmp_path)
    monkeypatch.setattr(sentinel, "load_settings", lambda: FakeSettings())
    monkeypatch.setattr(sentinel, "BinanceClient", FourPositionClient)
    monkeypatch.setattr(sentinel, "load_trading_control_state", lambda: TradingControlState())
    monkeypatch.setattr(sentinel, "load_route_risk_state", lambda: {"active_quarantined_routes": [], "routes": {}})
    monkeypatch.setattr(
        sentinel,
        "run_ai_readiness_scan",
        lambda **kwargs: {"candidate_count": 3, "allowed_count": 1, "execution_ticket": {"symbol": "DOGEUSDT"}},
    )

    payload = sentinel.run_ai_market_sentinel(symbols=["BTCUSDT"], limit=80, output_dir=tmp_path)

    assert payload["position_state"]["open_position_count"] == 4
    assert payload["expansion_gate"]["allowed"] is False
    assert "max-concurrent-positions-reached" in payload["expansion_gate"]["blockers"]
    assert not any(item["action"] == "operator_testnet_preflight" for item in payload["machine_action_queue"])


def test_ai_market_sentinel_allows_research_loop_when_flat_and_no_allowed_candidate(
    monkeypatch, tmp_path: Path
) -> None:
    class FlatClient(FakeClient):
        def positions(self, symbol=None):
            return []

    monkeypatch.setattr(sentinel, "STATE_DIR", tmp_path)
    monkeypatch.setattr(sentinel, "load_settings", lambda: FakeSettings())
    monkeypatch.setattr(sentinel, "BinanceClient", FlatClient)
    monkeypatch.setattr(sentinel, "load_trading_control_state", lambda: TradingControlState())
    monkeypatch.setattr(sentinel, "load_route_risk_state", lambda: {"active_quarantined_routes": [], "routes": {}})
    monkeypatch.setattr(
        sentinel,
        "run_ai_readiness_scan",
        lambda **kwargs: {
            "candidate_count": 6,
            "allowed_count": 0,
            "next_machine_action": "continue_expectancy_research",
            "hard_blocker_taxonomy": {"strategy_performance": ["Recent PF below floor."]},
            "report_path": "state/readiness.json",
        },
    )

    payload = sentinel.run_ai_market_sentinel(symbols=["BTCUSDT"], limit=80, output_dir=tmp_path)

    assert payload["position_state"]["open_position_count"] == 0
    assert payload["expansion_gate"]["allowed"] is False
    assert payload["machine_action_queue"][0]["action"] == "run_ai_expectancy_upgrade"
    assert payload["machine_action_queue"][0]["reason"] == "no-readiness-approved-candidate"
