from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import CONFIG_DIR, PROJECT_ROOT, STATE_DIR, ensure_runtime_dirs
from .payoff_objective import PayoffObjectiveTargets, payoff_objective_sort_key

MARKET_BOT_GATE_DIR = STATE_DIR / "market-bot-gate"
DEFAULT_MARKET_BOT_GATE_CONFIG = CONFIG_DIR / "market-bot-gate.default.yaml"


@dataclass(frozen=True, slots=True)
class MarketBotGateTargets:
    min_trades: int = 100
    min_profit_factor: float = 1.25
    min_expectancy_r: float = 0.05
    min_payoff_ratio: float = 1.20
    min_out_of_sample_return_pct: float = 0.0
    min_win_rate: float = 45.0
    max_stop_loss_ratio: float = 55.0
    min_walk_forward_stability: float = 0.60
    min_slippage_resilience: float = 0.70
    min_accepted_symbols: int = 1
    min_correlation_groups: int = 1
    min_beta_groups: int = 1
    required_symbols: tuple[str, ...] = ()
    require_feature_manifest_hash: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_symbols"] = list(self.required_symbols)
        return payload


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%S%fZ")


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    candidate = _resolve_path(path)
    return yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve_path(path).read_text(encoding="utf-8"))


def _float(value: Any, default: float = 0.0) -> float:
    if value in {"inf", "+inf"}:
        return 9999.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _side_from_alpha_row(row: dict[str, Any]) -> str:
    direct = str(row.get("side") or "").upper()
    if direct in {"BUY", "SELL"}:
        return direct
    symbol_strategy = row.get("symbol_strategy") if isinstance(row.get("symbol_strategy"), dict) else {}
    interval = str(row.get("interval") or "")
    family = str(row.get("strategy_family") or "")
    side_map = symbol_strategy.get("interval_family_sides") if isinstance(symbol_strategy, dict) else {}
    if isinstance(side_map, dict):
        interval_sides = side_map.get(interval)
        if isinstance(interval_sides, dict):
            sides = interval_sides.get(family)
            if isinstance(sides, list) and sides:
                candidate = str(sides[0]).upper()
                if candidate in {"BUY", "SELL"}:
                    return candidate
    return "BUY"


def _targets_from_config(config: dict[str, Any]) -> MarketBotGateTargets:
    raw = config.get("targets") or {}
    required_symbols = tuple(
        dict.fromkeys(str(item).strip().upper() for item in raw.get("required_symbols") or [] if str(item).strip())
    )
    return MarketBotGateTargets(
        min_trades=_int(raw.get("min_trades"), 100),
        min_profit_factor=_float(raw.get("min_profit_factor"), 1.25),
        min_expectancy_r=_float(raw.get("min_expectancy_r"), 0.05),
        min_payoff_ratio=_float(raw.get("min_payoff_ratio"), 1.20),
        min_out_of_sample_return_pct=_float(raw.get("min_out_of_sample_return_pct"), 0.0),
        min_win_rate=_float(raw.get("min_win_rate"), 45.0),
        max_stop_loss_ratio=_float(raw.get("max_stop_loss_ratio"), 55.0),
        min_walk_forward_stability=_float(raw.get("min_walk_forward_stability"), 0.60),
        min_slippage_resilience=_float(raw.get("min_slippage_resilience"), 0.70),
        min_accepted_symbols=max(_int(raw.get("min_accepted_symbols"), 1), 1),
        min_correlation_groups=max(_int(raw.get("min_correlation_groups"), 1), 1),
        min_beta_groups=max(_int(raw.get("min_beta_groups"), 1), 1),
        required_symbols=required_symbols,
        require_feature_manifest_hash=bool(raw.get("require_feature_manifest_hash", True)),
    )


def _row_blockers(row: dict[str, Any], targets: MarketBotGateTargets) -> list[str]:
    blockers: list[str] = []
    if _int(row.get("trade_count")) < targets.min_trades:
        blockers.append("trade-count-below-market-bot-floor")
    if _float(row.get("profit_factor")) < targets.min_profit_factor:
        blockers.append("profit-factor-below-market-bot-floor")
    if _float(row.get("expectancy_r")) < targets.min_expectancy_r:
        blockers.append("expectancy-r-below-market-bot-floor")
    if _float(row.get("payoff_ratio")) < targets.min_payoff_ratio:
        blockers.append("payoff-ratio-below-market-bot-floor")
    if (
        "out_of_sample_total_return_pct" in row
        and _float(row.get("out_of_sample_total_return_pct")) < targets.min_out_of_sample_return_pct
    ):
        blockers.append("out-of-sample-return-below-floor")
    if _float(row.get("stop_loss_ratio")) > targets.max_stop_loss_ratio:
        blockers.append("stop-loss-ratio-above-protection-ceiling")
    if _float(row.get("walk_forward_stability")) < targets.min_walk_forward_stability:
        blockers.append("walk-forward-stability-below-floor")
    if _float(row.get("slippage_resilience")) < targets.min_slippage_resilience:
        blockers.append("slippage-resilience-below-floor")
    return blockers


def _target_gaps(row: dict[str, Any], targets: MarketBotGateTargets) -> dict[str, float]:
    return {
        "trades_needed": max(targets.min_trades - _int(row.get("trade_count")), 0),
        "profit_factor_needed": round(max(targets.min_profit_factor - _float(row.get("profit_factor")), 0.0), 4),
        "expectancy_r_needed": round(max(targets.min_expectancy_r - _float(row.get("expectancy_r")), 0.0), 4),
        "payoff_ratio_needed": round(max(targets.min_payoff_ratio - _float(row.get("payoff_ratio")), 0.0), 4),
        "out_of_sample_return_needed": round(
            max(targets.min_out_of_sample_return_pct - _float(row.get("out_of_sample_total_return_pct")), 0.0),
            4,
        ),
        "win_rate_points_needed": round(max(targets.min_win_rate - _float(row.get("win_rate")), 0.0), 4),
        "stop_loss_ratio_points_to_cut": round(max(_float(row.get("stop_loss_ratio")) - targets.max_stop_loss_ratio, 0.0), 4),
        "walk_forward_stability_needed": round(
            max(targets.min_walk_forward_stability - _float(row.get("walk_forward_stability")), 0.0),
            4,
        ),
        "slippage_resilience_needed": round(
            max(targets.min_slippage_resilience - _float(row.get("slippage_resilience")), 0.0),
            4,
        ),
    }


def _primary_gap(row: dict[str, Any], targets: MarketBotGateTargets) -> str:
    gaps = _target_gaps(row, targets)
    if gaps["trades_needed"] > 0:
        return "sample_size"
    if gaps["slippage_resilience_needed"] > 0:
        return "slippage_resilience"
    if gaps["expectancy_r_needed"] > 0:
        return "expectancy"
    if gaps["profit_factor_needed"] > 0:
        return "profit_factor"
    if gaps["stop_loss_ratio_points_to_cut"] > 0:
        return "stop_loss_drag"
    if gaps["walk_forward_stability_needed"] > 0:
        return "walk_forward_stability"
    if gaps["payoff_ratio_needed"] > 0:
        return "payoff_ratio"
    if gaps["out_of_sample_return_needed"] > 0:
        return "out_of_sample"
    return "none"


def _research_state(row: dict[str, Any], blockers: list[str], targets: MarketBotGateTargets) -> str:
    if not blockers:
        return "tradable_candidate"
    trade_count = _int(row.get("trade_count"))
    expectancy_r = _float(row.get("expectancy_r"))
    profit_factor = _float(row.get("profit_factor"))
    payoff_ratio = _float(row.get("payoff_ratio"))
    stop_loss_ratio = _float(row.get("stop_loss_ratio"))
    walk_forward_stability = _float(row.get("walk_forward_stability"))
    slippage_resilience = _float(row.get("slippage_resilience"))
    has_positive_edge = (
        expectancy_r >= targets.min_expectancy_r
        and profit_factor >= targets.min_profit_factor
        and payoff_ratio >= targets.min_payoff_ratio
        and stop_loss_ratio <= targets.max_stop_loss_ratio
    )
    oos_failed = (
        "out_of_sample_total_return_pct" in row
        and _float(row.get("out_of_sample_total_return_pct")) < targets.min_out_of_sample_return_pct
    )
    if trade_count >= targets.min_trades and has_positive_edge and oos_failed:
        return "out_of_sample_failed"
    if trade_count < targets.min_trades and has_positive_edge:
        return "expand_sample_before_promotion"
    if trade_count >= targets.min_trades and (
        expectancy_r <= 0.0
        or profit_factor < targets.min_profit_factor
        or stop_loss_ratio > targets.max_stop_loss_ratio
    ):
        return "reject_expanded_route_regressed"
    if trade_count >= targets.min_trades and (
        walk_forward_stability < targets.min_walk_forward_stability
        or slippage_resilience < targets.min_slippage_resilience
    ):
        return "stress_or_walk_forward_failed"
    return "continue_research"


def _research_actions(row: dict[str, Any], state: str, targets: MarketBotGateTargets) -> list[str]:
    symbol = str(row.get("symbol") or "").upper()
    interval = str(row.get("interval") or "")
    family = str(row.get("strategy_family") or "")
    cohort = str(row.get("cohort_id") or f"{symbol}:{interval}:{family}")
    limit_hint = 15000 if interval in {"4h", "1d"} else 8000
    if state == "tradable_candidate":
        return [
            f"send {cohort} to hermes-ai-trader and live-readiness testnet gate",
            "keep fixed-risk sizing and portfolio exposure caps before any execution",
        ]
    if state == "expand_sample_before_promotion":
        return [
            (
                "rerun larger sample: "
                f"openclaw-quantctl alpha-research --config config/market-bot-discovery.default.yaml "
                f"--symbols {symbol} --intervals {interval} --limit {limit_hint} "
                f"--output-dir state/market-bot-{symbol.lower()}-{interval}-{family}-l{limit_hint} --compact"
            ),
            "do not promote until the same cohort reaches the trade floor and survives stress",
        ]
    if state == "reject_expanded_route_regressed":
        return [
            f"reject {cohort} as a current promotion route after expanded-sample regression",
            "shift work to exit/risk sweep or a different strategy family instead of lowering gates",
        ]
    if state == "stress_or_walk_forward_failed":
        return [
            (
                "rerun bounded slippage/walk-forward validation before promotion; "
                f"required stability>={targets.min_walk_forward_stability:.2f}, "
                f"slippage_resilience>={targets.min_slippage_resilience:.2f}"
            ),
            "if stress remains negative, quarantine the route from trading",
        ]
    if state == "out_of_sample_failed":
        return [
            f"reject {cohort} for now; full-sample edge did not survive the out-of-sample tail",
            "tighten regime/entry filters or test a different family before any promotion",
        ]
    return [
        "continue feature/route research; current row is not mature enough for execution",
        "prefer higher payoff with positive expectancy over indicator stacking",
    ]


def _rows_from_alpha(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in report.get("rows") or []:
        if not isinstance(row, dict):
            continue
        normalized = dict(row)
        symbol_strategy = normalized.get("symbol_strategy")
        if isinstance(symbol_strategy, dict) and not normalized.get("route_id"):
            normalized["route_id"] = symbol_strategy.get("route_id")
        normalized["side"] = _side_from_alpha_row(normalized)
        rows.append(normalized)
    return rows


def _feature_manifest_ok(report: dict[str, Any], targets: MarketBotGateTargets) -> tuple[bool, str]:
    if not targets.require_feature_manifest_hash:
        return True, ""
    manifest = report.get("feature_manifest") or {}
    manifest_hash = str(manifest.get("manifest_hash") or "")
    return bool(manifest_hash), manifest_hash


def _best_by_symbol(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        current = best.get(symbol)
        if current is None or (
            _float(row.get("market_bot_score")),
            _int(row.get("trade_count")),
            _float(row.get("expectancy_r")),
            _float(row.get("payoff_ratio")),
        ) > (
            _float(current.get("market_bot_score")),
            _int(current.get("trade_count")),
            _float(current.get("expectancy_r")),
            _float(current.get("payoff_ratio")),
        ):
            best[symbol] = row
    return best


def _portfolio_gate(
    *,
    evaluated: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    targets: MarketBotGateTargets,
) -> dict[str, Any]:
    accepted_by_symbol = _best_by_symbol(accepted)
    evaluated_by_symbol = _best_by_symbol(evaluated)
    accepted_symbols = sorted(accepted_by_symbol)
    accepted_correlation_groups = sorted({
        str(row.get("correlation_group") or row.get("route_id") or row.get("symbol") or "unknown")
        for row in accepted_by_symbol.values()
    })
    accepted_beta_groups = sorted({
        str(row.get("beta_group") or row.get("correlation_group") or row.get("route_id") or row.get("symbol") or "unknown")
        for row in accepted_by_symbol.values()
    })
    required_symbols = list(targets.required_symbols)
    missing_required_symbols = [
        symbol for symbol in required_symbols if symbol not in accepted_by_symbol
    ]
    blockers: list[str] = []
    if len(accepted_symbols) < targets.min_accepted_symbols:
        blockers.append("accepted-symbol-count-below-portfolio-floor")
    if missing_required_symbols:
        blockers.append("required-symbols-missing-positive-expectancy-cohort")
    if len(accepted_correlation_groups) < targets.min_correlation_groups:
        blockers.append("accepted-correlation-group-count-below-floor")
    if len(accepted_beta_groups) < targets.min_beta_groups:
        blockers.append("accepted-beta-group-count-below-floor")

    symbol_states: list[dict[str, Any]] = []
    all_symbols = sorted(set(evaluated_by_symbol) | set(required_symbols))
    for symbol in all_symbols:
        accepted_row = accepted_by_symbol.get(symbol)
        best_row = accepted_row or evaluated_by_symbol.get(symbol) or {}
        symbol_states.append(
            {
                "symbol": symbol,
                "accepted": accepted_row is not None,
                "best_cohort_id": best_row.get("cohort_id"),
                "research_state": best_row.get("research_state"),
                "trade_count": best_row.get("trade_count"),
                "profit_factor": best_row.get("profit_factor"),
                "expectancy_r": best_row.get("expectancy_r"),
                "payoff_ratio": best_row.get("payoff_ratio"),
                "out_of_sample_total_return_pct": best_row.get("out_of_sample_total_return_pct"),
                "win_rate": best_row.get("win_rate"),
                "stop_loss_ratio": best_row.get("stop_loss_ratio"),
                "walk_forward_stability": best_row.get("walk_forward_stability"),
                "slippage_resilience": best_row.get("slippage_resilience"),
                "correlation_group": best_row.get("correlation_group"),
                "beta_group": best_row.get("beta_group"),
                "blockers": best_row.get("blockers") or [],
            }
        )

    return {
        "enabled": targets.min_accepted_symbols > 1 or bool(required_symbols),
        "passed": not blockers,
        "min_accepted_symbols": targets.min_accepted_symbols,
        "min_correlation_groups": targets.min_correlation_groups,
        "min_beta_groups": targets.min_beta_groups,
        "accepted_symbol_count": len(accepted_symbols),
        "accepted_symbols": accepted_symbols,
        "accepted_correlation_group_count": len(accepted_correlation_groups),
        "accepted_correlation_groups": accepted_correlation_groups,
        "accepted_beta_group_count": len(accepted_beta_groups),
        "accepted_beta_groups": accepted_beta_groups,
        "required_symbols": required_symbols,
        "missing_required_symbols": missing_required_symbols,
        "blockers": blockers,
        "symbol_states": symbol_states,
        "principles": [
            "six-symbol-positive-expectancy-before-portfolio-promotion",
            "one-coin-short-sample-edge-cannot-authorize-portfolio-trading",
            "each accepted symbol needs mature sample, payoff, PF, stress, and walk-forward evidence",
            "accepted symbols must span enough correlation and beta groups to avoid fake diversification",
        ],
    }


def evaluate_market_bot_gate(
    *,
    alpha_report: str | Path,
    config_path: str | Path = DEFAULT_MARKET_BOT_GATE_CONFIG,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    config = _load_yaml(config_path)
    targets = _targets_from_config(config)
    report = _read_json(alpha_report)
    manifest_ok, manifest_hash = _feature_manifest_ok(report, targets)
    rows = _rows_from_alpha(report)
    target_obj = PayoffObjectiveTargets(
        min_trades=targets.min_trades,
        min_profit_factor=targets.min_profit_factor,
        min_expectancy_r=targets.min_expectancy_r,
        min_payoff_ratio=targets.min_payoff_ratio,
        min_win_rate=targets.min_win_rate,
        max_stop_loss_ratio=targets.max_stop_loss_ratio,
    )
    evaluated: list[dict[str, Any]] = []
    for row in rows:
        blockers = _row_blockers(row, targets)
        if not manifest_ok:
            blockers.append("feature-manifest-hash-missing")
        state = _research_state(row, blockers, targets)
        evaluated.append(
            {
                "symbol": row.get("symbol"),
                "interval": row.get("interval"),
                "strategy_family": row.get("strategy_family"),
                "route_id": row.get("route_id"),
                "correlation_group": row.get("correlation_group"),
                "beta_group": row.get("beta_group") or row.get("correlation_group"),
                "side": row.get("side"),
                "cohort_id": row.get("cohort_id"),
                "trade_count": row.get("trade_count"),
                "win_rate": row.get("win_rate"),
                "stop_loss_ratio": row.get("stop_loss_ratio"),
                "profit_factor": row.get("profit_factor"),
                "expectancy_r": row.get("expectancy_r"),
                "payoff_ratio": row.get("payoff_ratio"),
                "out_of_sample_total_return_pct": row.get("out_of_sample_total_return_pct"),
                "walk_forward_stability": row.get("walk_forward_stability"),
                "slippage_resilience": row.get("slippage_resilience"),
                "market_bot_score": payoff_objective_sort_key(row, targets=target_obj)[0],
                "target_gaps": _target_gaps(row, targets),
                "primary_gap": _primary_gap(row, targets),
                "research_state": state,
                "next_actions": _research_actions(row, state, targets),
                "accepted": not blockers,
                "blockers": blockers,
            }
        )
    evaluated.sort(
        key=lambda item: (
            bool(item.get("accepted")),
            _int(item.get("trade_count")) > 0,
            _float(item.get("market_bot_score")),
            _int(item.get("trade_count")),
            _float(item.get("expectancy_r")),
            _float(item.get("payoff_ratio")),
        ),
        reverse=True,
    )
    accepted = [row for row in evaluated if row.get("accepted")]
    portfolio_gate = _portfolio_gate(evaluated=evaluated, accepted=accepted, targets=targets)
    expansion_candidates = [
        row for row in evaluated if row.get("research_state") == "expand_sample_before_promotion"
    ]
    regressed_routes = [
        row for row in evaluated if row.get("research_state") == "reject_expanded_route_regressed"
    ]
    stress_failed = [
        row for row in evaluated if row.get("research_state") == "stress_or_walk_forward_failed"
    ]
    out_of_sample_failed = [
        row for row in evaluated if row.get("research_state") == "out_of_sample_failed"
    ]
    diagnostics = {
        "nearest_symbols": [
            {
                "symbol": row.get("symbol"),
                "cohort_id": row.get("cohort_id"),
                "primary_gap": row.get("primary_gap"),
                "target_gaps": row.get("target_gaps"),
                "research_state": row.get("research_state"),
                "market_bot_score": row.get("market_bot_score"),
                "next_actions": row.get("next_actions"),
            }
            for row in list(_best_by_symbol(evaluated).values())
        ],
        "gap_counts": {
            gap: sum(1 for row in evaluated if row.get("primary_gap") == gap)
            for gap in sorted({str(row.get("primary_gap") or "unknown") for row in evaluated})
        },
        "portfolio_gap": {
            "accepted_symbols_needed": max(targets.min_accepted_symbols - len(_best_by_symbol(accepted)), 0),
            "min_accepted_symbols": targets.min_accepted_symbols,
        },
    }
    root = Path(output_dir).expanduser().resolve() if output_dir else MARKET_BOT_GATE_DIR
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": _utc_now().replace(microsecond=0).isoformat(),
        "mode": "market_bot_expectancy_gate",
        "safety": {
            "opens_orders": False,
            "writes_execution_config": False,
            "mainnet_live_allowed": False,
        },
        "alpha_report": str(_resolve_path(alpha_report)),
        "targets": targets.to_dict(),
        "feature_manifest_hash": manifest_hash,
        "market_bot_references": {
            "freqtrade": "cooldown, stoploss guard, max drawdown style protections",
            "hummingbot": "triple-barrier stop/take-profit/time-limit thinking",
            "quantconnect": "universe, alpha, portfolio, risk, execution separation",
            "nautilustrader": "pre-trade risk before order creation",
        },
        "accepted_count": len(accepted),
        "portfolio_gate": portfolio_gate,
        "diagnostics": diagnostics,
        "safe_to_open_new_entries": bool(accepted) and bool(portfolio_gate.get("passed", True)),
        "execution_recommendation": (
            "eligible_for_hermes_ai_trader_and_live_readiness"
            if accepted and bool(portfolio_gate.get("passed", True))
            else "block_new_entries_and_continue_research"
        ),
        "best": evaluated[0] if evaluated else None,
        "accepted": accepted,
        "expansion_candidates": expansion_candidates[:10],
        "regressed_routes": regressed_routes[:10],
        "stress_failed": stress_failed[:10],
        "out_of_sample_failed": out_of_sample_failed[:10],
        "rows": evaluated,
        "next_research": [
            "increase real trade sample before trusting the row",
            "improve payoff ratio before adding more indicators",
            "only widen win-rate if payoff and expectancy remain positive",
            "rerun alpha-research with feature_manifest hash and slippage stress",
        ],
    }
    report_path = root / f"{_stamp()}-market-bot-gate.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload
