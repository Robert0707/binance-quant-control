from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import binance_quant_control.cli as cli
from binance_quant_control.binance_api import BinanceAPIError


class DummyClient:
    def __init__(self, settings: SimpleNamespace) -> None:
        self.settings = settings

    def __enter__(self) -> DummyClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def ping(self, market: str) -> dict[str, int]:
        return {"serverTime": 1}

    def account(self, market: str) -> dict[str, object]:
        if market == "spot":
            raise BinanceAPIError("spot auth failed")
        return {"balances": [{"asset": "USDT", "free": "10", "locked": "0"}]}

    def balance(self, market: str) -> list[dict[str, str]]:
        if market == "futures":
            return [{"asset": "USDT", "balance": "25"}]
        raise AssertionError(f"unexpected balance market: {market}")


def test_doctor_ignores_non_default_market_private_auth_failure_when_default_market_is_healthy(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    settings = SimpleNamespace(
        use_testnet=True,
        live_trading_enabled=False,
        default_symbol="BTCUSDT",
        default_market="futures",
        has_binance_credentials=True,
        has_blave_credentials=False,
    )

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(cli, "BinanceClient", DummyClient)
    monkeypatch.setattr(cli, "print_json", lambda payload, compact=False: captured.update(payload))

    cli.cmd_doctor(Namespace(use_blave=False, compact=True))

    assert captured["overall"] == "ok"
    assert captured["warnings"] == []
    assert captured["spot_ok"] is True
    assert captured["futures_ok"] is True


def test_doctor_warns_when_default_market_private_auth_fails(monkeypatch) -> None:
    captured: dict[str, object] = {}
    settings = SimpleNamespace(
        use_testnet=True,
        live_trading_enabled=False,
        default_symbol="BTCUSDT",
        default_market="spot",
        has_binance_credentials=True,
        has_blave_credentials=False,
    )

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(cli, "BinanceClient", DummyClient)
    monkeypatch.setattr(cli, "print_json", lambda payload, compact=False: captured.update(payload))

    cli.cmd_doctor(Namespace(use_blave=False, compact=True))

    assert captured["overall"] == "warn"
    assert "spot private API auth check failed" in captured["warnings"]
