from __future__ import annotations

from argparse import Namespace

import binance_quant_control.cli as cli


def test_cmd_external_context_compact_payload(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_builder(symbols, **kwargs):
        captured["symbols"] = symbols
        captured["kwargs"] = kwargs
        return {
            "combined_signal": "neutral",
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "available_sources": [],
            "sources": {
                "coinmarketcap": {
                    "enabled": True,
                    "available": False,
                    "signal": "neutral",
                    "reason": "COINMARKETCAP_API_KEY not configured",
                }
            },
            "cache_path": "/tmp/external-context.json",
        }

    monkeypatch.setattr(cli, "build_external_context", fake_builder)
    monkeypatch.setattr(
        cli,
        "print_json",
        lambda payload, compact=False: captured.update({"payload": payload, "compact": compact}),
    )

    cli.cmd_external_context(
        Namespace(
            config="config/external-context.default.yaml",
            symbols="btcusdt, ethusdt",
            no_cache=False,
            compact=True,
        )
    )

    assert captured["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert captured["kwargs"]["write_cache"] is True  # type: ignore[index]
    assert captured["payload"]["status"] == "ok"  # type: ignore[index]
    assert captured["payload"]["sources"]["coinmarketcap"]["available"] is False  # type: ignore[index]
    assert captured["compact"] is True
