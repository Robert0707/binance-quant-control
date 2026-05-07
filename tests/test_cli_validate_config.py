from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import binance_quant_control.cli as cli


def test_cmd_validate_config_accepts_compact(monkeypatch) -> None:
    captured: dict[str, object] = {}
    settings = SimpleNamespace(
        use_testnet=True,
        live_trading_enabled=False,
        default_symbol="BTCUSDT",
        default_market="futures",
        recv_window_ms=5000,
        max_leverage=30,
        max_notional_pct=0.5,
    )
    strategy = SimpleNamespace(
        profile="core-high-win-research",
        path="config/strategy-core-high-win-research.yaml",
        defaults=SimpleNamespace(symbol="ETHUSDT", market="futures", interval="4h"),
        risk=SimpleNamespace(
            max_account_risk_pct=0.006,
            min_adx=16.0,
            trailing_stop_enabled=True,
        ),
    )

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "load_strategy_config", lambda path: strategy)
    monkeypatch.setattr(
        cli,
        "print_json",
        lambda payload, compact=False: captured.update({"payload": payload, "compact": compact}),
    )

    cli.cmd_validate_config(
        Namespace(strategy_config="config/strategy-core-high-win-research.yaml", compact=True)
    )

    assert captured["compact"] is True
    assert captured["payload"]["status"] == "ok"  # type: ignore[index]
    assert captured["payload"]["strategy"]["profile"] == "core-high-win-research"  # type: ignore[index]
