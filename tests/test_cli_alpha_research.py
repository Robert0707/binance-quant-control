from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import binance_quant_control.cli as cli


def test_cmd_alpha_research_forwards_overrides_and_compact_payload(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    settings = SimpleNamespace(use_testnet=True, live_trading_enabled=False)

    def fake_runner(received_settings, **kwargs):
        captured["settings"] = received_settings
        captured["kwargs"] = kwargs
        return {
            "mode": "aggressive_alpha_research",
            "mainnet_live_allowed": False,
            "strategy_profile": "aggressive-alpha-research",
            "symbols": ["SOLUSDT", "ETHUSDT"],
            "intervals": ["1h", "4h"],
            "ranking_method": ["out-of-sample total return"],
            "resolved_symbol_family_plan": [
                {"symbol": "SOLUSDT", "interval": "1h", "families": ["breakout"]}
            ],
            "performance_summary": {"execution_recommendation": "block_new_entries_and_continue_research"},
            "top": [{"symbol": "SOLUSDT", "ranking_score": 12.3}],
            "errors": [],
            "skipped_symbol_intervals": [
                {"symbol": "BTCUSDT", "interval": "1h", "reason": "symbol-interval-quarantined"}
            ],
            "report_path": str(tmp_path / "alpha-research-ranking.json"),
        }

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "run_aggressive_alpha_research", fake_runner)
    monkeypatch.setattr(
        cli,
        "print_json",
        lambda payload, compact=False: captured.update({"payload": payload, "compact": compact}),
    )

    cli.cmd_alpha_research(
        Namespace(
            config="config/aggressive-alpha-research.default.yaml",
            symbols="solusdt, ethusdt",
            intervals="1h,4h",
            limit=320,
            output_dir=str(tmp_path),
            compact=True,
        )
    )

    assert captured["settings"] is settings
    assert captured["compact"] is True
    assert captured["payload"]["status"] == "ok"  # type: ignore[index]
    assert captured["payload"]["mainnet_live_allowed"] is False  # type: ignore[index]
    assert captured["payload"]["resolved_symbol_family_plan"] == [  # type: ignore[index]
        {"symbol": "SOLUSDT", "interval": "1h", "families": ["breakout"]}
    ]
    assert captured["payload"]["performance_summary"] == {  # type: ignore[index]
        "execution_recommendation": "block_new_entries_and_continue_research"
    }
    assert captured["payload"]["skipped_symbol_intervals"] == [  # type: ignore[index]
        {"symbol": "BTCUSDT", "interval": "1h", "reason": "symbol-interval-quarantined"}
    ]
    assert captured["kwargs"]["symbol_overrides"] == ["SOLUSDT", "ETHUSDT"]  # type: ignore[index]
    assert captured["kwargs"]["interval_overrides"] == ["1h", "4h"]  # type: ignore[index]
    assert captured["kwargs"]["limit_override"] == 320  # type: ignore[index]
