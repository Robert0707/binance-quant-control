from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from binance_quant_control.challenge import ChallengeState
from binance_quant_control.cli import cmd_auto_pause_trading
from binance_quant_control.order_journal import TradeState
from binance_quant_control.trading_control import (
    AutoPausePolicy,
    TradingControlState,
    evaluate_auto_pause_conditions,
    load_trading_control_state,
    set_trading_paused,
)


class FakeClientFlat:
    def __init__(self, settings):
        self.settings = settings

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def positions(self, symbol: str | None = None):
        return []


class FakeClientLong:
    def __init__(self, settings):
        self.settings = settings

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def positions(self, symbol: str | None = None):
        return [
            {
                "symbol": "NEARUSDT",
                "positionAmt": "4",
                "entryPrice": "1.4000",
                "unRealizedProfit": "0.10",
                "leverage": "3",
            }
        ]


class FakeSettings:
    live_trading_enabled = True


class FakeDefaults:
    symbol = "NEARUSDT"
    market = "futures"
    interval = "1h"


class FakeStrategy:
    profile = "micro-account-pilot"
    defaults = FakeDefaults()


def test_trading_control_defaults_when_file_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "binance_quant_control.trading_control.TRADING_CONTROL_STATE_PATH",
        tmp_path / "missing.json",
    )
    state = load_trading_control_state()
    assert state == TradingControlState()


def test_set_trading_paused_round_trip(monkeypatch, tmp_path: Path):
    state_path = tmp_path / "trading-control.json"
    monkeypatch.setattr("binance_quant_control.trading_control.TRADING_CONTROL_STATE_PATH", state_path)

    paused = set_trading_paused(paused=True, reason="operator stop", updated_by="pytest")
    assert paused.paused is True
    assert paused.reason == "operator stop"
    assert paused.updated_by == "pytest"

    loaded = load_trading_control_state()
    assert loaded.paused is True
    assert loaded.reason == "operator stop"

    resumed = set_trading_paused(paused=False, reason="operator resume", updated_by="pytest")
    assert resumed.paused is False
    assert resumed.reason == "operator resume"


def test_auto_pause_evaluation_triggers_on_loss_streak(monkeypatch):
    monkeypatch.setattr("binance_quant_control.binance_api.BinanceClient", FakeClientFlat)
    monkeypatch.setattr(
        "binance_quant_control.trading_control.load_trade_state",
        lambda: TradeState(
            daily_trade_count=1,
            daily_trade_date="2026-04-26",
            consecutive_losses=2,
            last_loss_at="2026-04-26T10:00:00+00:00",
            total_live_trades=3,
            total_pnl_usdt=-0.5,
        ),
    )
    monkeypatch.setattr(
        "binance_quant_control.trading_control.load_challenge_state",
        lambda scope=None: ChallengeState(enabled=False),
    )
    monkeypatch.setattr("binance_quant_control.trading_control.REPORTS_DIR", Path("/nonexistent"))

    evaluation = evaluate_auto_pause_conditions(FakeSettings(), FakeStrategy(), policy=AutoPausePolicy())

    assert evaluation.should_pause is True
    assert any("Consecutive losses" in reason for reason in evaluation.reasons)


def test_auto_pause_evaluation_keeps_recent_loss_streak_paused_during_cooldown(monkeypatch):
    monkeypatch.setattr("binance_quant_control.binance_api.BinanceClient", FakeClientFlat)
    last_loss_at = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    monkeypatch.setattr(
        "binance_quant_control.trading_control.load_trade_state",
        lambda: TradeState(
            daily_trade_count=1,
            daily_trade_date="2026-04-26",
            consecutive_losses=2,
            last_loss_at=last_loss_at,
            total_live_trades=3,
            total_pnl_usdt=-0.5,
        ),
    )
    monkeypatch.setattr(
        "binance_quant_control.trading_control.load_challenge_state",
        lambda scope=None: ChallengeState(enabled=False),
    )
    monkeypatch.setattr("binance_quant_control.trading_control.REPORTS_DIR", Path("/nonexistent"))

    evaluation = evaluate_auto_pause_conditions(
        FakeSettings(),
        FakeStrategy(),
        policy=AutoPausePolicy(loss_cooldown_hours=4.0),
    )

    assert evaluation.should_pause is True
    assert any("cooldown requires" in reason for reason in evaluation.reasons)


def test_auto_pause_evaluation_treats_expired_loss_cooldown_as_warning(monkeypatch):
    monkeypatch.setattr("binance_quant_control.binance_api.BinanceClient", FakeClientFlat)
    last_loss_at = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()
    monkeypatch.setattr(
        "binance_quant_control.trading_control.load_trade_state",
        lambda: TradeState(
            daily_trade_count=1,
            daily_trade_date="2026-04-26",
            consecutive_losses=2,
            last_loss_at=last_loss_at,
            total_live_trades=3,
            total_pnl_usdt=-0.5,
        ),
    )
    monkeypatch.setattr(
        "binance_quant_control.trading_control.load_challenge_state",
        lambda scope=None: ChallengeState(enabled=False),
    )
    monkeypatch.setattr("binance_quant_control.trading_control.REPORTS_DIR", Path("/nonexistent"))

    evaluation = evaluate_auto_pause_conditions(
        FakeSettings(),
        FakeStrategy(),
        policy=AutoPausePolicy(loss_cooldown_hours=4.0),
    )

    assert evaluation.should_pause is False
    assert not evaluation.reasons
    assert any("cooldown expired" in warning for warning in evaluation.warnings)


def test_auto_pause_evaluation_triggers_on_signal_reversal(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("binance_quant_control.binance_api.BinanceClient", FakeClientLong)
    monkeypatch.setattr(
        "binance_quant_control.trading_control.load_trade_state",
        lambda: TradeState(
            daily_trade_count=1,
            daily_trade_date="2026-04-26",
            consecutive_losses=0,
            last_loss_at=None,
            total_live_trades=1,
            total_pnl_usdt=0.0,
        ),
    )
    monkeypatch.setattr(
        "binance_quant_control.trading_control.load_challenge_state",
        lambda scope=None: ChallengeState(enabled=False),
    )
    monkeypatch.setattr("binance_quant_control.trading_control.REPORTS_DIR", tmp_path)
    report_dir = tmp_path / "20260426T102214Z-nearusdt-futures-1h"
    report_dir.mkdir(parents=True)
    report_path = report_dir / "analysis.json"
    report_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-26T10:22:14+00:00",
                "symbol": "NEARUSDT",
                "market": "futures",
                "interval": "1h",
                "analysis": {
                    "bias": "short-bias",
                    "score": 92,
                    "convergence": 0.91,
                },
            }
        ),
        encoding="utf-8",
    )
    journal_entry = {
        "timestamp": "2026-04-26T10:23:46+00:00",
        "symbol": "NEARUSDT",
        "side": "BUY",
    }
    monkeypatch.setattr("binance_quant_control.trading_control.read_live_orders", lambda: [journal_entry])

    evaluation = evaluate_auto_pause_conditions(
        FakeSettings(),
        FakeStrategy(),
        policy=AutoPausePolicy(position_timeout_hours=12.0, reversal_min_score=80, reversal_min_convergence=0.8),
    )

    assert evaluation.should_pause is True
    assert any("flipped" in reason for reason in evaluation.reasons)
    assert evaluation.positions and evaluation.latest_reports


def test_cmd_auto_pause_trading_auto_resumes_previous_auto_pause(monkeypatch) -> None:
    payloads: list[dict] = []

    monkeypatch.setattr("binance_quant_control.cli.load_settings", lambda: FakeSettings())
    monkeypatch.setattr("binance_quant_control.cli.load_strategy_config", lambda path: FakeStrategy())
    monkeypatch.setattr(
        "binance_quant_control.cli.evaluate_auto_pause_conditions",
        lambda settings, strategy, policy=None: type(
            "Eval",
            (),
            {
                "should_pause": False,
                "reasons": [],
                "warnings": [],
                "to_dict": lambda self=None: {"should_pause": False, "reasons": [], "warnings": []},
            },
        )(),
    )
    monkeypatch.setattr(
        "binance_quant_control.cli.load_trading_control_state",
        lambda: TradingControlState(
            paused=True,
            reason="old auto pause",
            updated_at="2026-04-28T00:00:00+00:00",
            updated_by="openclaw-quantctl auto-pause-trading",
        ),
    )
    monkeypatch.setattr(
        "binance_quant_control.cli.set_trading_paused",
        lambda **kwargs: TradingControlState(
            paused=bool(kwargs["paused"]),
            reason=str(kwargs["reason"]),
            updated_at="2026-04-28T00:05:00+00:00",
            updated_by=str(kwargs["updated_by"]),
        ),
    )
    monkeypatch.setattr("binance_quant_control.cli.print_json", lambda payload, compact=False: payloads.append(payload))

    cmd_auto_pause_trading(
        Namespace(
            strategy_config="config/strategy-hermes-pro.auto.yaml",
            consecutive_loss_threshold=2,
            loss_cooldown_hours=None,
            position_timeout_hours=12.0,
            reversal_min_score=80,
            reversal_min_convergence=0.8,
            cancel_open_orders=True,
            compact=True,
        )
    )

    assert payloads
    payload = payloads[0]
    assert payload["status"] == "idle"
    assert payload["actions"] == ["resumed"]
    assert payload["trading_control"]["paused"] is False


def test_cmd_auto_pause_trading_keeps_manual_pause_intact(monkeypatch) -> None:
    payloads: list[dict] = []

    monkeypatch.setattr("binance_quant_control.cli.load_settings", lambda: FakeSettings())
    monkeypatch.setattr("binance_quant_control.cli.load_strategy_config", lambda path: FakeStrategy())
    monkeypatch.setattr(
        "binance_quant_control.cli.evaluate_auto_pause_conditions",
        lambda settings, strategy, policy=None: type(
            "Eval",
            (),
            {
                "should_pause": False,
                "reasons": [],
                "warnings": [],
                "to_dict": lambda self=None: {"should_pause": False, "reasons": [], "warnings": []},
            },
        )(),
    )
    monkeypatch.setattr(
        "binance_quant_control.cli.load_trading_control_state",
        lambda: TradingControlState(
            paused=True,
            reason="manual stop",
            updated_at="2026-04-28T00:00:00+00:00",
            updated_by="operator",
        ),
    )
    monkeypatch.setattr(
        "binance_quant_control.cli.set_trading_paused",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("manual pause should not auto-resume")),
    )
    monkeypatch.setattr("binance_quant_control.cli.print_json", lambda payload, compact=False: payloads.append(payload))

    cmd_auto_pause_trading(
        Namespace(
            strategy_config="config/strategy-hermes-pro.auto.yaml",
            consecutive_loss_threshold=2,
            loss_cooldown_hours=None,
            position_timeout_hours=12.0,
            reversal_min_score=80,
            reversal_min_convergence=0.8,
            cancel_open_orders=True,
            compact=True,
        )
    )

    assert payloads
    payload = payloads[0]
    assert payload["status"] == "idle"
    assert payload["actions"] == ["no-action"]
    assert payload["trading_control"]["paused"] is True
