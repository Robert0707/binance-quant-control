from __future__ import annotations

from pathlib import Path

import binance_quant_control.external_context as external_context


def _write_config(path: Path) -> None:
    path.write_text(
        """
cache_root: cache
timeout_seconds: 5
symbols: [BTCUSDT, ETHUSDT]
env:
  coinmarketcap_api_key: CMC_KEY
  arkham_api_key: ARKHAM_KEY
  cryptopanic_api_key: PANIC_KEY
  glassnode_api_key: GLASSNODE_KEY
providers:
  coinmarketcap:
    enabled: true
  dexscreener:
    enabled: true
  cryptopanic:
    enabled: true
  glassnode:
    enabled: true
    metrics: [market/price_usd_close]
  arkham:
    enabled: true
""",
        encoding="utf-8",
    )


def test_external_context_degrades_to_neutral_without_keys(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "external.yaml"
    _write_config(config_path)
    for key in ("CMC_KEY", "ARKHAM_KEY", "PANIC_KEY", "GLASSNODE_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(external_context, "http_get_json", lambda *args, **kwargs: [])

    payload = external_context.build_external_context(
        ["btcusdt", "ethusdt"],
        config_path=config_path,
        write_cache=False,
    )

    assert payload["combined_signal"] == "neutral"
    assert payload["sources"]["coinmarketcap"]["available"] is False
    assert payload["sources"]["coinmarketcap"]["reason"] == "COINMARKETCAP_API_KEY not configured"
    assert payload["sources"]["cryptopanic"]["reason"] == "CRYPTOPANIC_API_KEY not configured"
    assert payload["sources"]["glassnode"]["reason"] == "GLASSNODE_API_KEY not configured"
    assert payload["sources"]["arkham"]["reason"] == "ARKHAM_API_KEY not configured"


def test_external_context_uses_optional_provider_payloads(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "external.yaml"
    _write_config(config_path)
    monkeypatch.setenv("CMC_KEY", "cmc")
    monkeypatch.setenv("PANIC_KEY", "panic")
    monkeypatch.setenv("GLASSNODE_KEY", "glassnode")
    monkeypatch.setenv("ARKHAM_KEY", "arkham")

    def fake_http_get_json(url: str, **_kwargs):
        if "global-metrics" in url:
            return {"data": {"btc_dominance": 48.5, "eth_dominance": 17.2, "active_cryptocurrencies": 10000}}
        if "quotes/latest" in url:
            return {"data": {"BTC": {}, "ETH": {}}}
        if "token-profiles" in url:
            return [{"chainId": "solana", "description": "SOL momentum token"}]
        if "cryptopanic" in url:
            return {"results": [{"title": "ETF inflow supports BTC and ETH"}]}
        if "glassnode" in url:
            assert "api_key=" not in url
            assert _kwargs["headers"]["X-Api-Key"] == "glassnode"
            return [{"t": 1, "v": 100.0}]
        if "api.arkm.com" in url:
            return [{"chain": "ethereum", "active": True}]
        raise AssertionError(url)

    monkeypatch.setattr(external_context, "http_get_json", fake_http_get_json)

    payload = external_context.build_external_context(
        ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        config_path=config_path,
        write_cache=True,
    )

    assert payload["combined_signal"] == "risk_on"
    assert set(payload["available_sources"]) == {
        "coinmarketcap",
        "dexscreener",
        "cryptopanic",
        "glassnode",
        "arkham",
    }
    assert payload["sources"]["coinmarketcap"]["signal"] == "risk_on"
    assert payload["sources"]["dexscreener"]["signal"] == "hot_onchain"
    assert payload["sources"]["cryptopanic"]["signal"] == "risk_on"
    assert payload["sources"]["glassnode"]["available"] is True
    assert payload["sources"]["arkham"]["available"] is True
    assert payload["sources"]["arkham"]["signal"] == "whale_context_available"
    assert Path(payload["cache_path"]).exists()


def test_external_context_key_status_never_prints_secrets(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "external.yaml"
    _write_config(config_path)
    monkeypatch.setenv("CMC_KEY", "cmc-secret")
    monkeypatch.setenv("PANIC_KEY", "panic-secret")
    monkeypatch.delenv("GLASSNODE_KEY", raising=False)
    monkeypatch.setenv("ARKHAM_KEY", "arkham-secret")

    payload = external_context.external_context_key_status(config_path)

    assert payload["configured_count"] == 4
    assert payload["missing"] == ["glassnode"]
    assert payload["providers"]["coinmarketcap"]["configured"] is True
    assert payload["providers"]["coinmarketcap"]["key_suffix"] == "cret"
    assert "cmc-secret" not in str(payload)
    assert payload["providers"]["glassnode"]["configured"] is False


def test_key_status_only_requires_enabled_keyed_providers(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "external.yaml"
    config_path.write_text(
        """
cache_root: cache
symbols: [BTCUSDT]
env:
  coinmarketcap_api_key: CMC_KEY
  arkham_api_key: ARKHAM_KEY
  cryptopanic_api_key: PANIC_KEY
  glassnode_api_key: GLASSNODE_KEY
providers:
  coinmarketcap:
    enabled: true
  dexscreener:
    enabled: true
  cryptopanic:
    enabled: false
  glassnode:
    enabled: false
  arkham:
    enabled: false
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CMC_KEY", "cmc-secret")
    for key in ("PANIC_KEY", "GLASSNODE_KEY", "ARKHAM_KEY"):
        monkeypatch.delenv(key, raising=False)

    payload = external_context.external_context_key_status(config_path)

    assert payload["missing"] == []
    assert payload["optional_missing"] == ["cryptopanic", "glassnode", "arkham"]
    assert payload["providers"]["dexscreener"]["configured"] is True
    assert payload["providers"]["dexscreener"]["auth"] == "none"
