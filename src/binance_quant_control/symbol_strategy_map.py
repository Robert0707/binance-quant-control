from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .alpha_families import ACTIVE_STRATEGY_FAMILIES
from .asset_routing import normalize_symbol
from .config import CONFIG_DIR

DEFAULT_SYMBOL_STRATEGY_MAP_PATH = CONFIG_DIR / "core-symbol-strategy-map.default.yaml"


@dataclass(frozen=True, slots=True)
class SymbolPromotionSpec:
    min_trades: int
    min_profit_factor: float
    min_win_rate: float
    max_stop_loss_ratio: float
    min_expectancy_r: float
    min_payoff_ratio: float

    def to_dict(self) -> dict[str, object]:
        return {
            "min_trades": self.min_trades,
            "min_profit_factor": self.min_profit_factor,
            "min_win_rate": self.min_win_rate,
            "max_stop_loss_ratio": self.max_stop_loss_ratio,
            "min_expectancy_r": self.min_expectancy_r,
            "min_payoff_ratio": self.min_payoff_ratio,
        }


@dataclass(frozen=True, slots=True)
class SymbolStrategySpec:
    symbol: str
    primary_family: str
    allowed_families: tuple[str, ...]
    blocked_families: tuple[str, ...]
    interval_families: dict[str, tuple[str, ...]]
    interval_family_sides: dict[str, dict[str, tuple[str, ...]]]
    interval: str
    execution_lane: str
    route_id: str
    asset_class: str
    promotion: SymbolPromotionSpec
    thesis: str
    risk_filters: tuple[str, ...]
    entry_filters: dict[str, Any]
    strategy_overrides: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "primary_family": self.primary_family,
            "allowed_families": list(self.allowed_families),
            "blocked_families": list(self.blocked_families),
            "interval_families": {
                interval: list(families) for interval, families in sorted(self.interval_families.items())
            },
            "interval_family_sides": {
                interval: {family: list(sides) for family, sides in sorted(family_sides.items())}
                for interval, family_sides in sorted(self.interval_family_sides.items())
            },
            "interval": self.interval,
            "execution_lane": self.execution_lane,
            "route_id": self.route_id,
            "asset_class": self.asset_class,
            "promotion": self.promotion.to_dict(),
            "thesis": self.thesis,
            "risk_filters": list(self.risk_filters),
            "entry_filters": self.entry_filters,
            "strategy_overrides": self.strategy_overrides,
        }


def resolve_symbol_strategy_map_path(path: str | Path | None = None) -> Path:
    candidate = Path(path or DEFAULT_SYMBOL_STRATEGY_MAP_PATH).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0] == CONFIG_DIR.name:
        return (CONFIG_DIR.parent / candidate).resolve()
    return (CONFIG_DIR / candidate).resolve()


def _promotion_spec(raw: dict[str, Any], defaults: dict[str, Any]) -> SymbolPromotionSpec:
    merged = {**defaults, **(raw.get("promotion") or {})}
    return SymbolPromotionSpec(
        min_trades=int(merged.get("min_trades") or 100),
        min_profit_factor=float(merged.get("min_profit_factor") or 1.5),
        min_win_rate=float(merged.get("min_win_rate") or 90.0),
        max_stop_loss_ratio=float(merged.get("max_stop_loss_ratio") or 10.0),
        min_expectancy_r=float(merged.get("min_expectancy_r") or 0.10),
        min_payoff_ratio=float(merged.get("min_payoff_ratio") or 1.15),
    )


def _families(raw: Any, *, fallback: list[str] | None = None) -> tuple[str, ...]:
    if raw is None:
        source = fallback or []
    else:
        source = raw
    items = [str(item) for item in source]
    unique = tuple(dict.fromkeys(items))
    invalid = [item for item in unique if item not in ACTIVE_STRATEGY_FAMILIES]
    if invalid:
        valid = ", ".join(ACTIVE_STRATEGY_FAMILIES)
        raise ValueError(f"Unknown strategy families {invalid}; expected one of: {valid}")
    return unique


def _interval_families(raw: Any, *, fallback: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("intervals must be a mapping of interval -> family list or settings.")
    intervals: dict[str, tuple[str, ...]] = {}
    for interval, raw_interval_spec in raw.items():
        key = str(interval).strip()
        if not key:
            raise ValueError("interval family map contains an empty interval key.")
        if isinstance(raw_interval_spec, dict):
            families = _families(raw_interval_spec.get("allowed_families"), fallback=list(fallback))
        else:
            families = _families(raw_interval_spec)
        intervals[key] = families
    return intervals


def _sides(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    items = tuple(dict.fromkeys(str(item).upper() for item in raw if str(item).strip()))
    invalid = [item for item in items if item not in {"BUY", "SELL"}]
    if invalid:
        raise ValueError(f"Unknown strategy sides {invalid}; expected BUY or SELL")
    return items


def _interval_family_sides(raw: Any) -> dict[str, dict[str, tuple[str, ...]]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("intervals must be a mapping of interval -> family list or settings.")
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for interval, raw_interval_spec in raw.items():
        if not isinstance(raw_interval_spec, dict):
            continue
        raw_side_policy = raw_interval_spec.get("family_sides") or raw_interval_spec.get("side_policy") or {}
        if not raw_side_policy:
            continue
        if not isinstance(raw_side_policy, dict):
            raise ValueError("interval family side policy must be a mapping of family -> BUY/SELL list.")
        interval_key = str(interval).strip()
        family_sides: dict[str, tuple[str, ...]] = {}
        for family, sides in raw_side_policy.items():
            family_name = str(family).strip()
            if family_name not in ACTIVE_STRATEGY_FAMILIES:
                valid = ", ".join(ACTIVE_STRATEGY_FAMILIES)
                raise ValueError(f"Unknown side-policy family {family_name!r}; expected one of: {valid}")
            family_sides[family_name] = _sides(sides)
        result[interval_key] = family_sides
    return result


def _family_union(primary_family: str, allowed: tuple[str, ...], interval_families: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    merged: list[str] = list(allowed)
    for families in interval_families.values():
        merged.extend(families)
    if primary_family in ACTIVE_STRATEGY_FAMILIES:
        merged.append(primary_family)
    if not merged:
        merged = [primary_family]
    return tuple(dict.fromkeys(merged))


def load_symbol_strategy_map(path: str | Path | None = None) -> dict[str, SymbolStrategySpec]:
    config_path = resolve_symbol_strategy_map_path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Symbol strategy map must be a mapping: {config_path}")
    defaults = payload.get("defaults") or {}
    if not isinstance(defaults, dict):
        defaults = {}
    default_promotion = defaults.get("promotion") or {}
    symbols = payload.get("symbols") or {}
    if not isinstance(symbols, dict):
        raise ValueError(f"Symbol strategy map symbols must be a mapping: {config_path}")

    specs: dict[str, SymbolStrategySpec] = {}
    for raw_symbol, raw_spec in symbols.items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Symbol strategy spec for {raw_symbol} must be a mapping.")
        symbol = normalize_symbol(str(raw_symbol))
        primary_family = str(raw_spec.get("primary_family") or "").strip()
        if not primary_family:
            raise ValueError(f"Symbol strategy spec for {symbol} is missing primary_family.")
        allowed = _families(raw_spec.get("allowed_families"), fallback=[primary_family])
        intervals = _interval_families(raw_spec.get("intervals"), fallback=allowed)
        interval_sides = _interval_family_sides(raw_spec.get("intervals"))
        allowed = _family_union(primary_family, allowed, intervals)
        specs[symbol] = SymbolStrategySpec(
            symbol=symbol,
            primary_family=primary_family,
            allowed_families=allowed,
            blocked_families=_families(raw_spec.get("blocked_families")),
            interval_families=intervals,
            interval_family_sides=interval_sides,
            interval=str(raw_spec.get("interval") or defaults.get("interval") or "4h"),
            execution_lane=str(raw_spec.get("execution_lane") or defaults.get("execution_lane") or "paper"),
            route_id=str(raw_spec.get("route_id") or ""),
            asset_class=str(raw_spec.get("asset_class") or ""),
            promotion=_promotion_spec(raw_spec, default_promotion),
            thesis=str(raw_spec.get("thesis") or ""),
            risk_filters=tuple(str(item) for item in (raw_spec.get("risk_filters") or [])),
            entry_filters=dict(raw_spec.get("entry_filters") or {}),
            strategy_overrides=dict(raw_spec.get("strategy_overrides") or {}),
        )
    return specs


def resolve_symbol_strategy(
    symbol: str,
    path: str | Path | None = None,
) -> SymbolStrategySpec:
    normalized = normalize_symbol(symbol)
    specs = load_symbol_strategy_map(path)
    if normalized not in specs:
        raise KeyError(f"No symbol strategy spec configured for {normalized}")
    return specs[normalized]


def filter_symbol_families(
    symbol: str,
    candidate_families: list[str],
    strategy_specs: dict[str, SymbolStrategySpec],
) -> list[str]:
    spec = strategy_specs.get(normalize_symbol(symbol))
    if spec is None:
        return list(candidate_families)
    allowed = [family for family in candidate_families if family in spec.allowed_families]
    blocked = set(spec.blocked_families)
    return [family for family in allowed if family not in blocked]


def filter_symbol_interval_families(
    symbol: str,
    interval: str,
    candidate_families: list[str],
    strategy_specs: dict[str, SymbolStrategySpec],
) -> list[str]:
    spec = strategy_specs.get(normalize_symbol(symbol))
    if spec is None:
        return list(candidate_families)
    configured = spec.interval_families.get(str(interval), spec.allowed_families)
    allowed = [family for family in candidate_families if family in configured]
    blocked = set(spec.blocked_families)
    return [family for family in allowed if family not in blocked]


def resolve_symbol_interval_family_sides(
    symbol: str,
    interval: str,
    family: str,
    strategy_specs: dict[str, SymbolStrategySpec],
) -> tuple[str, ...]:
    spec = strategy_specs.get(normalize_symbol(symbol))
    if spec is None:
        return ("BUY", "SELL")
    family_sides = spec.interval_family_sides.get(str(interval), {})
    configured = family_sides.get(str(family), ())
    return configured or ("BUY", "SELL")
