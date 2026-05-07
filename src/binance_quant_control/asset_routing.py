from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import CONFIG_DIR

DEFAULT_ROUTING_CONFIG_PATH = CONFIG_DIR / "asset-routing.default.yaml"

SYMBOL_ALIASES = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "XAU": "XAUTUSDT",
    "XAUUSD": "XAUTUSDT",
    "GOLD": "XAUTUSDT",
    "PAXG": "PAXGUSDT",
    "XAUT": "XAUTUSDT",
}


@dataclass(frozen=True, slots=True)
class RouteValidationSpec:
    screening_min_win_rate: float
    screening_min_profit_factor: float
    screening_min_expectancy_r: float
    screening_min_payoff_ratio: float
    screening_min_trades: int
    validation_min_win_rate: float
    validation_min_profit_factor: float
    validation_min_expectancy_r: float
    validation_min_payoff_ratio: float
    validation_min_simulated_trades: int
    max_drawdown_pct: float
    max_loss_streak: int
    elite_enabled: bool
    elite_min_win_rate: float
    elite_min_profit_factor: float
    elite_min_trades: int

    def to_dict(self) -> dict[str, object]:
        return {
            "screening_min_win_rate": self.screening_min_win_rate,
            "screening_min_profit_factor": self.screening_min_profit_factor,
            "screening_min_expectancy_r": self.screening_min_expectancy_r,
            "screening_min_payoff_ratio": self.screening_min_payoff_ratio,
            "screening_min_trades": self.screening_min_trades,
            "validation_min_win_rate": self.validation_min_win_rate,
            "validation_min_profit_factor": self.validation_min_profit_factor,
            "validation_min_expectancy_r": self.validation_min_expectancy_r,
            "validation_min_payoff_ratio": self.validation_min_payoff_ratio,
            "validation_min_simulated_trades": self.validation_min_simulated_trades,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_loss_streak": self.max_loss_streak,
            "elite_enabled": self.elite_enabled,
            "elite_min_win_rate": self.elite_min_win_rate,
            "elite_min_profit_factor": self.elite_min_profit_factor,
            "elite_min_trades": self.elite_min_trades,
        }


@dataclass(frozen=True, slots=True)
class AssetRoute:
    route_id: str
    asset_class: str
    strategy_config: Path
    market: str
    interval: str
    simulation_mode: str
    review_lane: str
    rationale: str
    tags: tuple[str, ...]
    validation: RouteValidationSpec

    def to_dict(self) -> dict[str, object]:
        return {
            "route_id": self.route_id,
            "asset_class": self.asset_class,
            "strategy_config": str(self.strategy_config),
            "market": self.market,
            "interval": self.interval,
            "simulation_mode": self.simulation_mode,
            "review_lane": self.review_lane,
            "rationale": self.rationale,
            "tags": list(self.tags),
            "validation": self.validation.to_dict(),
        }


def resolve_routing_config_path(path: str | Path | None = None) -> Path:
    candidate = Path(path or DEFAULT_ROUTING_CONFIG_PATH).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (CONFIG_DIR / candidate).resolve()


def normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    return SYMBOL_ALIASES.get(normalized, normalized)


def classify_symbol(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    base = normalized.removesuffix("USDT").removesuffix("USDC").removesuffix("BUSD")
    if base == "BTC":
        return "btc_core"
    if base == "ETH":
        return "eth_core"
    if base in {"PAXG", "XAUT", "XAU", "GOLD"}:
        return "xau_macro"
    if base == "BNB":
        return "exchange_token_event_risk"
    if base == "SOL":
        return "high_beta_l1"
    if base == "XRP":
        return "event_liquidity_alt"
    if base == "TRX":
        return "stablecoin_settlement_alt"
    if base == "LINK":
        return "oracle_defi_beta"
    if base == "AAVE":
        return "defi_lending_beta"
    if base in {"NEAR", "AVAX", "APT", "ATOM", "ADA", "UNI", "LTC", "BCH", "FIL", "ARB", "OP", "INJ", "SUI"}:
        return "major_alt_trend"
    if base in {"DOGE", "PENGU", "PEPE", "1000PEPE", "1000LUNC", "TRUMP", "WIF", "ORCA", "HYPER", "SOON", "API3", "APE", "AXS"}:
        return "meme_high_beta"
    return "defensive_unknown"


def _resolve_project_path(raw: str) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()
    return (CONFIG_DIR / candidate).resolve()


def load_asset_routes(path: str | Path | None = None) -> dict[str, dict[str, object]]:
    config_path = resolve_routing_config_path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Asset routing config must be a mapping: {config_path}")
    routes = payload.get("routes") or {}
    if not isinstance(routes, dict):
        raise ValueError(f"Asset routing config routes must be a mapping: {config_path}")
    return routes


def _build_validation_spec(raw: dict[str, object]) -> RouteValidationSpec:
    validation = raw.get("validation") or {}
    if not isinstance(validation, dict):
        validation = {}
    return RouteValidationSpec(
        screening_min_win_rate=float(validation.get("screening_min_win_rate") or 80.0),
        screening_min_profit_factor=float(validation.get("screening_min_profit_factor") or 1.2),
        screening_min_expectancy_r=float(validation.get("screening_min_expectancy_r") or 0.05),
        screening_min_payoff_ratio=float(validation.get("screening_min_payoff_ratio") or 1.0),
        screening_min_trades=int(validation.get("screening_min_trades") or 100),
        validation_min_win_rate=float(validation.get("validation_min_win_rate") or 80.0),
        validation_min_profit_factor=float(validation.get("validation_min_profit_factor") or 1.5),
        validation_min_expectancy_r=float(validation.get("validation_min_expectancy_r") or 0.10),
        validation_min_payoff_ratio=float(validation.get("validation_min_payoff_ratio") or 1.15),
        validation_min_simulated_trades=int(validation.get("validation_min_simulated_trades") or 100),
        max_drawdown_pct=float(validation.get("max_drawdown_pct") or 15.0),
        max_loss_streak=int(validation.get("max_loss_streak") or 3),
        elite_enabled=bool(validation.get("elite_enabled", True)),
        elite_min_win_rate=float(validation.get("elite_min_win_rate") or 90.0),
        elite_min_profit_factor=float(validation.get("elite_min_profit_factor") or 1.5),
        elite_min_trades=int(validation.get("elite_min_trades") or 100),
    )


def resolve_symbol_route(symbol: str, path: str | Path | None = None) -> AssetRoute:
    normalized = normalize_symbol(symbol)
    routes = load_asset_routes(path)
    selected: tuple[str, dict[str, object]] | None = None
    for route_id, raw in routes.items():
        if not isinstance(raw, dict):
            continue
        symbols = [str(item).upper() for item in (raw.get("symbols") or [])]
        if normalized in symbols:
            selected = (route_id, raw)
            break
    if selected is None:
        asset_class = classify_symbol(normalized)
        for route_id, raw in routes.items():
            if not isinstance(raw, dict):
                continue
            if str(raw.get("asset_class") or "") == asset_class:
                selected = (route_id, raw)
                break
    if selected is None:
        raise ValueError(f"No asset route matched symbol {normalized}")
    route_id, raw = selected
    return AssetRoute(
        route_id=route_id,
        asset_class=str(raw.get("asset_class") or classify_symbol(normalized)),
        strategy_config=_resolve_project_path(str(raw.get("strategy_config") or "strategy-stable-risk.yaml")),
        market=str(raw.get("market") or "futures"),
        interval=str(raw.get("interval") or "4h"),
        simulation_mode=str(raw.get("simulation_mode") or "paper"),
        review_lane=str(raw.get("review_lane") or "strategy-review-only"),
        rationale=str(raw.get("rationale") or ""),
        tags=tuple(str(item) for item in (raw.get("tags") or [])),
        validation=_build_validation_spec(raw),
    )
