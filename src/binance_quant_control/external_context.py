from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .asset_routing import normalize_symbol
from .config import CONFIG_DIR, STATE_DIR

DEFAULT_EXTERNAL_CONTEXT_CONFIG_PATH = CONFIG_DIR / "external-context.default.yaml"
DEFAULT_EXTERNAL_CONTEXT_ROOT = STATE_DIR / "external-context"
DEFAULT_USER_AGENT = "openclaw-binance-quant/0.1"
PROVIDER_SETUP_GUIDES: dict[str, dict[str, str]] = {
    "coinmarketcap": {
        "env_var": "COINMARKETCAP_API_KEY",
        "register_url": "https://pro.coinmarketcap.com/signup/",
        "docs_url": "https://coinmarketcap.com/api/documentation/guides/authentication",
        "auth": "X-CMC_PRO_API_KEY header",
        "cost_tier": "free-api-key",
        "role": "global crypto market-capital-flow and BTC/ETH dominance filter",
    },
    "cryptopanic": {
        "env_var": "CRYPTOPANIC_API_KEY",
        "register_url": "https://cryptopanic.com/developers/api/",
        "docs_url": "https://cryptopanic.com/developers/api/",
        "auth": "auth_token query parameter",
        "oauth": "not supported by the CryptoPanic posts API; use the account API auth token",
        "cost_tier": "paid-or-limited-optional",
        "role": "crypto news and event-risk filter",
    },
    "glassnode": {
        "env_var": "GLASSNODE_API_KEY",
        "register_url": "https://studio.glassnode.com/",
        "docs_url": "https://docs.glassnode.com/basic-api/api-key",
        "auth": "X-Api-Key header",
        "cost_tier": "paid-optional",
        "role": "BTC/ETH on-chain macro and market metrics filter",
    },
    "arkham": {
        "env_var": "ARKHAM_API_KEY",
        "register_url": "https://intel.arkm.com/api",
        "docs_url": "https://api-guide.intel.arkm.com/",
        "auth": "API-Key header",
        "cost_tier": "paid-optional",
        "role": "wallet/entity flow and whale-context filter",
    },
}
FREE_PUBLIC_PROVIDER_GUIDES: dict[str, dict[str, str]] = {
    "dexscreener": {
        "env_var": "",
        "register_url": "https://docs.dexscreener.com/api/reference",
        "docs_url": "https://docs.dexscreener.com/api/reference",
        "auth": "none",
        "cost_tier": "free-public-endpoint",
        "role": "public on-chain hot token rotation filter",
    },
}


@dataclass(frozen=True, slots=True)
class ExternalContextConfig:
    path: Path
    cache_root: Path
    timeout_seconds: int
    symbols: tuple[str, ...]
    providers: dict[str, dict[str, Any]]
    env: dict[str, str]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def resolve_external_context_config_path(path: str | Path | None = None) -> Path:
    candidate = Path(path or DEFAULT_EXTERNAL_CONTEXT_CONFIG_PATH).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0] == CONFIG_DIR.name:
        return (CONFIG_DIR.parent / candidate).resolve()
    return (CONFIG_DIR / candidate).resolve()


def load_external_context_config(path: str | Path | None = None) -> ExternalContextConfig:
    config_path = resolve_external_context_config_path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"External context config must be a mapping: {config_path}")
    cache_root = Path(payload.get("cache_root") or DEFAULT_EXTERNAL_CONTEXT_ROOT).expanduser()
    if not cache_root.is_absolute():
        cache_root = (config_path.parent / cache_root).resolve()
    providers = payload.get("providers") or {}
    if not isinstance(providers, dict):
        providers = {}
    env = payload.get("env") or {}
    if not isinstance(env, dict):
        env = {}
    return ExternalContextConfig(
        path=config_path,
        cache_root=cache_root.resolve(),
        timeout_seconds=int(payload.get("timeout_seconds") or 15),
        symbols=tuple(normalize_symbol(str(item)) for item in (payload.get("symbols") or [])),
        providers={
            str(name): dict(raw or {}) if isinstance(raw, dict) else {}
            for name, raw in providers.items()
        },
        env={str(key): str(value) for key, value in env.items()},
    )


def _provider_enabled(config: ExternalContextConfig, name: str) -> bool:
    raw = config.providers.get(name) or {}
    return bool(raw.get("enabled", False))


def _env_name(config: ExternalContextConfig, key: str, default: str) -> str:
    return str(config.env.get(key) or default)


def _env_value(config: ExternalContextConfig, key: str, default: str) -> str:
    return os.getenv(_env_name(config, key, default), "").strip()


def _masked_key_status(value: str) -> dict[str, Any]:
    if not value:
        return {"configured": False, "length": 0, "suffix": ""}
    return {"configured": True, "length": len(value), "suffix": value[-4:]}


def external_context_key_status(config_path: str | Path | None = None) -> dict[str, Any]:
    config = load_external_context_config(config_path)
    key_map = {
        "coinmarketcap": ("coinmarketcap_api_key", "COINMARKETCAP_API_KEY"),
        "cryptopanic": ("cryptopanic_api_key", "CRYPTOPANIC_API_KEY"),
        "glassnode": ("glassnode_api_key", "GLASSNODE_API_KEY"),
        "arkham": ("arkham_api_key", "ARKHAM_API_KEY"),
    }
    providers: dict[str, dict[str, Any]] = {}
    for name, (config_key, default_env) in key_map.items():
        env_var = _env_name(config, config_key, default_env)
        status = _masked_key_status(os.getenv(env_var, "").strip())
        guide = PROVIDER_SETUP_GUIDES[name]
        providers[name] = {
            "enabled": _provider_enabled(config, name),
            "env_var": env_var,
            "configured": status["configured"],
            "key_length": status["length"],
            "key_suffix": status["suffix"],
            "auth": guide["auth"],
            "cost_tier": guide["cost_tier"],
            "role": guide["role"],
            "register_url": guide["register_url"],
            "docs_url": guide["docs_url"],
        }
    for name, guide in FREE_PUBLIC_PROVIDER_GUIDES.items():
        providers[name] = {
            "enabled": _provider_enabled(config, name),
            "env_var": "",
            "configured": True,
            "key_length": 0,
            "key_suffix": "",
            "auth": guide["auth"],
            "cost_tier": guide["cost_tier"],
            "role": guide["role"],
            "register_url": guide["register_url"],
            "docs_url": guide["docs_url"],
        }
    configured_count = sum(1 for item in providers.values() if item["configured"])
    enabled_missing = [
        name
        for name, item in providers.items()
        if bool(item["enabled"]) and item["env_var"] and not bool(item["configured"])
    ]
    return {
        "generated_at": _utc_now().isoformat(),
        "config_path": str(config.path),
        "env_file": str(CONFIG_DIR.parent / ".env"),
        "configured_count": configured_count,
        "missing": enabled_missing,
        "optional_missing": [
            name
            for name, item in providers.items()
            if not bool(item["enabled"]) and item["env_var"] and not bool(item["configured"])
        ],
        "providers": providers,
        "secret_policy": "values are never printed; only length and last four characters are shown",
    }


def _has_keyed_context(config: ExternalContextConfig) -> bool:
    return any(
        _env_value(config, key, default)
        for key, default in (
            ("coinmarketcap_api_key", "COINMARKETCAP_API_KEY"),
            ("cryptopanic_api_key", "CRYPTOPANIC_API_KEY"),
            ("glassnode_api_key", "GLASSNODE_API_KEY"),
            ("arkham_api_key", "ARKHAM_API_KEY"),
        )
    )


def _status(
    *,
    provider: str,
    enabled: bool,
    available: bool = False,
    signal: str = "neutral",
    reason: str = "",
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "enabled": enabled,
        "available": available,
        "signal": signal,
        "reason": reason,
        "summary": summary or {},
    }


def http_get_json(url: str, *, headers: dict[str, str] | None = None, timeout: int = 15) -> Any:
    request_headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}
    request_headers.update(headers or {})
    request = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def _safe_fetch(provider: str, enabled: bool, fetcher: Any) -> dict[str, Any]:
    if not enabled:
        return _status(provider=provider, enabled=False, reason="provider-disabled")
    try:
        return fetcher()
    except Exception as exc:
        return _status(
            provider=provider,
            enabled=True,
            reason=f"provider-unavailable:{exc.__class__.__name__}",
        )


def _cmc_context(config: ExternalContextConfig, symbols: tuple[str, ...]) -> dict[str, Any]:
    enabled = _provider_enabled(config, "coinmarketcap")
    api_key = _env_value(config, "coinmarketcap_api_key", "COINMARKETCAP_API_KEY")

    def fetch() -> dict[str, Any]:
        if not api_key:
            return _status(
                provider="coinmarketcap",
                enabled=True,
                reason="COINMARKETCAP_API_KEY not configured",
            )
        headers = {"X-CMC_PRO_API_KEY": api_key, "Accept": "application/json"}
        global_payload = http_get_json(
            "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest",
            headers=headers,
            timeout=config.timeout_seconds,
        )
        bases = ",".join(sorted({symbol.removesuffix("USDT") for symbol in symbols}))
        quotes: Any = {}
        if bases:
            query = urllib.parse.urlencode({"symbol": bases, "convert": "USD"})
            quotes = http_get_json(
                f"https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest?{query}",
                headers=headers,
                timeout=config.timeout_seconds,
            )
        data = global_payload.get("data") if isinstance(global_payload, dict) else {}
        btc_dominance = float((data or {}).get("btc_dominance") or 0.0)
        signal = "risk_on" if btc_dominance < 50 else "defensive"
        return _status(
            provider="coinmarketcap",
            enabled=True,
            available=True,
            signal=signal,
            summary={
                "btc_dominance": round(btc_dominance, 4),
                "eth_dominance": round(float((data or {}).get("eth_dominance") or 0.0), 4),
                "active_cryptocurrencies": int((data or {}).get("active_cryptocurrencies") or 0),
                "quote_symbols": sorted({symbol.removesuffix("USDT") for symbol in symbols}),
                "quote_count": len((quotes.get("data") or {}) if isinstance(quotes, dict) else {}),
            },
        )

    return _safe_fetch("coinmarketcap", enabled, fetch)


def _dexscreener_context(config: ExternalContextConfig, symbols: tuple[str, ...]) -> dict[str, Any]:
    enabled = _provider_enabled(config, "dexscreener")

    def fetch() -> dict[str, Any]:
        raw_profiles = http_get_json(
            "https://api.dexscreener.com/token-profiles/latest/v1",
            timeout=config.timeout_seconds,
        )
        profiles = raw_profiles if isinstance(raw_profiles, list) else []
        watched = {symbol.removesuffix("USDT").upper() for symbol in symbols}
        matched = []
        for item in profiles[:50]:
            text = " ".join(str(item.get(key) or "") for key in ("tokenAddress", "chainId", "description", "url")).upper()
            if any(base and base in text for base in watched):
                matched.append(item)
        signal = "hot_onchain" if matched else "neutral"
        return _status(
            provider="dexscreener",
            enabled=True,
            available=True,
            signal=signal,
            summary={
                "latest_profile_count": len(profiles),
                "matched_watch_symbols": len(matched),
                "chains": sorted({str(item.get("chainId") or "") for item in profiles[:20] if item.get("chainId")}),
            },
        )

    return _safe_fetch("dexscreener", enabled, fetch)


def _cryptopanic_context(config: ExternalContextConfig, symbols: tuple[str, ...]) -> dict[str, Any]:
    enabled = _provider_enabled(config, "cryptopanic")
    api_key = _env_value(config, "cryptopanic_api_key", "CRYPTOPANIC_API_KEY")

    def fetch() -> dict[str, Any]:
        if not api_key:
            return _status(
                provider="cryptopanic",
                enabled=True,
                reason="CRYPTOPANIC_API_KEY not configured",
            )
        currencies = ",".join(sorted({symbol.removesuffix("USDT") for symbol in symbols}))
        query = urllib.parse.urlencode(
            {
                "auth_token": api_key,
                "public": "true",
                "currencies": currencies,
                "kind": "news",
            }
        )
        payload = http_get_json(
            f"https://cryptopanic.com/api/v1/posts/?{query}",
            headers={"Accept": "application/json"},
            timeout=config.timeout_seconds,
        )
        results = payload.get("results") if isinstance(payload, dict) else []
        titles = [str(item.get("title") or "").lower() for item in results if isinstance(item, dict)]
        negative = sum(1 for title in titles if any(word in title for word in ("hack", "lawsuit", "liquidation", "exploit", "sec")))
        positive = sum(1 for title in titles if any(word in title for word in ("etf", "inflow", "approval", "adoption")))
        signal = "risk_off" if negative > positive else "risk_on" if positive > negative else "neutral"
        return _status(
            provider="cryptopanic",
            enabled=True,
            available=True,
            signal=signal,
            summary={
                "post_count": len(results),
                "negative_keyword_count": negative,
                "positive_keyword_count": positive,
                "currencies": currencies,
            },
        )

    return _safe_fetch("cryptopanic", enabled, fetch)


def _glassnode_context(config: ExternalContextConfig, symbols: tuple[str, ...]) -> dict[str, Any]:
    enabled = _provider_enabled(config, "glassnode")
    api_key = _env_value(config, "glassnode_api_key", "GLASSNODE_API_KEY")
    raw = config.providers.get("glassnode") or {}
    metrics = [str(item) for item in (raw.get("metrics") or ["market/price_usd_close"])]
    interval = str(raw.get("interval") or "24h")

    def fetch() -> dict[str, Any]:
        if not api_key:
            return _status(
                provider="glassnode",
                enabled=True,
                reason="GLASSNODE_API_KEY not configured",
            )
        assets = [symbol.removesuffix("USDT") for symbol in symbols if symbol.startswith(("BTC", "ETH"))]
        snapshots: dict[str, Any] = {}
        for asset in sorted(set(assets)):
            for metric in metrics:
                query = urllib.parse.urlencode({"a": asset, "i": interval})
                url = f"https://api.glassnode.com/v1/metrics/{metric}?{query}"
                payload = http_get_json(
                    url,
                    headers={"X-Api-Key": api_key},
                    timeout=config.timeout_seconds,
                )
                if isinstance(payload, list) and payload:
                    snapshots[f"{asset}:{metric}"] = payload[-1]
        return _status(
            provider="glassnode",
            enabled=True,
            available=bool(snapshots),
            signal="onchain_available" if snapshots else "neutral",
            summary={"metric_count": len(snapshots), "metrics": sorted(snapshots)},
        )

    return _safe_fetch("glassnode", enabled, fetch)


def _arkham_context(config: ExternalContextConfig, symbols: tuple[str, ...]) -> dict[str, Any]:
    enabled = _provider_enabled(config, "arkham")
    api_key = _env_value(config, "arkham_api_key", "ARKHAM_API_KEY")

    def fetch() -> dict[str, Any]:
        if not api_key:
            return _status(provider="arkham", enabled=True, reason="ARKHAM_API_KEY not configured")
        payload = http_get_json(
            "https://api.arkm.com/networks/status",
            headers={"API-Key": api_key},
            timeout=config.timeout_seconds,
        )
        networks = payload if isinstance(payload, list) else []
        return _status(
            provider="arkham",
            enabled=True,
            available=bool(networks),
            signal="whale_context_available" if networks else "neutral",
            reason="" if networks else "arkham-network-status-empty",
            summary={
                "network_count": len(networks),
                "active_networks": [
                    str(item.get("chain") or "")
                    for item in networks
                    if isinstance(item, dict) and bool(item.get("active", False))
                ],
                "symbols": list(symbols),
                "whale_flow_note": "configure Arkham entities or addresses before using wallet-flow vetoes",
            },
        )

    return _safe_fetch("arkham", enabled, fetch)


def _combined_signal(sources: dict[str, dict[str, Any]]) -> str:
    signals = [str(item.get("signal") or "neutral") for item in sources.values()]
    if any(signal in {"risk_off", "defensive"} for signal in signals):
        return "risk_off"
    if any(signal in {"risk_on", "hot_onchain"} for signal in signals):
        return "risk_on"
    return "neutral"


def build_external_context(
    symbols: list[str] | tuple[str, ...],
    *,
    config_path: str | Path | None = None,
    write_cache: bool = True,
) -> dict[str, Any]:
    config = load_external_context_config(config_path)
    normalized_symbols = tuple(dict.fromkeys(normalize_symbol(symbol) for symbol in (symbols or config.symbols)))
    sources = {
        "coinmarketcap": _cmc_context(config, normalized_symbols),
        "dexscreener": _dexscreener_context(config, normalized_symbols),
        "cryptopanic": _cryptopanic_context(config, normalized_symbols),
        "glassnode": _glassnode_context(config, normalized_symbols),
        "arkham": _arkham_context(config, normalized_symbols),
    }
    payload = {
        "generated_at": _utc_now().isoformat(),
        "config_path": str(config.path),
        "symbols": list(normalized_symbols),
        "combined_signal": _combined_signal(sources),
        "available_sources": [
            name for name, item in sources.items() if bool(item.get("available", False))
        ],
        "sources": sources,
        "execution_note": "external-context-is-a-filter-not-a-standalone-entry-signal",
    }
    if write_cache:
        config.cache_root.mkdir(parents=True, exist_ok=True)
        path = config.cache_root / f"{_utc_now().strftime('%Y%m%dT%H%M%SZ')}-external-context.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        payload["cache_path"] = str(path)
    return payload
