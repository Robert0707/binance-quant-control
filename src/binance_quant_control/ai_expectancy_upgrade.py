from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .ai_surface_audit import run_ai_surface_audit
from .alpha_research import run_aggressive_alpha_research
from .candidate_universe import fetch_top_futures_symbols
from .config import CONFIG_DIR, STATE_DIR, ensure_runtime_dirs, load_settings
from .feature_dataset import FeatureDatasetSpec, build_feature_dataset
from .hermes_ai_trader import run_hermes_ai_trader
from .loss_diagnostics import run_loss_diagnostics
from .market_bot_gate import DEFAULT_MARKET_BOT_GATE_CONFIG, evaluate_market_bot_gate
from .readiness_scanner import run_ai_readiness_scan
from .risk_combo_sweep import run_risk_combo_sweep

AI_EXPECTANCY_UPGRADE_DIR = STATE_DIR / "ai-expectancy-upgrade"
DEFAULT_DISCOVERY_CONFIG = CONFIG_DIR / "market-bot-six-symbol-discovery.default.yaml"


@dataclass(frozen=True, slots=True)
class UpgradeStep:
    priority: str
    name: str
    objective: str
    status: str
    report_path: str
    machine_read: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _top_symbols_from_hermes(payload: dict[str, Any], *, max_symbols: int = 6) -> list[str]:
    queue = [item for item in payload.get("candidate_queue") or [] if isinstance(item, dict)]
    ranked = sorted(
        queue,
        key=lambda item: _float((item.get("machine_directive") or {}).get("priority_score")),
        reverse=True,
    )
    symbols: list[str] = []
    for item in ranked:
        signal = item.get("signal") if isinstance(item.get("signal"), dict) else {}
        symbol = str(signal.get("symbol") or "").upper()
        if symbol and symbol != "UNRESOLVED" and symbol not in symbols:
            symbols.append(symbol)
        if len(symbols) >= max_symbols:
            break
    return symbols


def _machine_symbol_allocation(
    settings: Any,
    hermes: dict[str, Any],
    *,
    requested_symbols: list[str] | None,
    requested_discovery_symbols: list[str] | None,
    universe_limit: int,
) -> dict[str, Any]:
    hermes_symbols = _top_symbols_from_hermes(hermes, max_symbols=max(universe_limit, 6))
    universe_errors: list[str] = []
    top_volume_symbols: list[str] = []
    top_volume_rows: list[dict[str, Any]] = []
    if universe_limit > 0:
        try:
            rows = fetch_top_futures_symbols(
                settings,
                limit=universe_limit,
                include_symbols=list(dict.fromkeys(hermes_symbols + (requested_symbols or []) + (requested_discovery_symbols or []))),
            )
            top_volume_rows = [row.to_dict() for row in rows]
            top_volume_symbols = [row.symbol for row in rows]
        except Exception as exc:
            universe_errors.append(str(exc))
    exploit_symbols = list(dict.fromkeys(requested_symbols or hermes_symbols or ["BTCUSDT", "ETHUSDT"]))
    explore_symbols = [
        symbol
        for symbol in top_volume_symbols
        if symbol not in exploit_symbols
    ]
    portfolio_symbols = list(dict.fromkeys(requested_discovery_symbols or (exploit_symbols + explore_symbols)))[:6]
    if len(portfolio_symbols) < 6:
        for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "TRXUSDT"]:
            if symbol not in portfolio_symbols:
                portfolio_symbols.append(symbol)
            if len(portfolio_symbols) >= 6:
                break
    btc_eth_symbols = [symbol for symbol in ("BTCUSDT", "ETHUSDT") if symbol in set(exploit_symbols + portfolio_symbols)]
    return {
        "mode": "liquidity_plus_expectancy_symbol_allocator",
        "principle": "treat symbols as arms; exploit proven expectancy and explore liquid unresolved arms",
        "exploit_symbols": exploit_symbols,
        "explore_symbols": explore_symbols[: max(universe_limit - len(exploit_symbols), 0)],
        "portfolio_symbols": portfolio_symbols,
        "btc_eth_symbols": btc_eth_symbols or ["BTCUSDT", "ETHUSDT"],
        "top_volume_symbols": top_volume_symbols,
        "top_volume_rows": top_volume_rows,
        "errors": universe_errors,
    }


def _alpha_report_path(payload: dict[str, Any]) -> str:
    report_path = str(payload.get("report_path") or "")
    return str(Path(report_path).with_name("alpha-research-ranking.json")) if report_path else ""


def _best_gate_read(gate: dict[str, Any]) -> dict[str, Any]:
    best = gate.get("best") if isinstance(gate.get("best"), dict) else {}
    portfolio = gate.get("portfolio_gate") if isinstance(gate.get("portfolio_gate"), dict) else {}
    return {
        "safe_to_open_new_entries": gate.get("safe_to_open_new_entries"),
        "accepted_count": gate.get("accepted_count"),
        "accepted_symbols": portfolio.get("accepted_symbols") or [],
        "missing_required_symbols": portfolio.get("missing_required_symbols") or [],
        "best_symbol": best.get("symbol"),
        "best_cohort": best.get("cohort_id"),
        "best_expectancy_r": best.get("expectancy_r"),
        "best_payoff_ratio": best.get("payoff_ratio"),
        "best_profit_factor": best.get("profit_factor"),
        "best_trade_count": best.get("trade_count"),
        "best_primary_gap": best.get("primary_gap"),
        "report_path": gate.get("report_path"),
    }


def _step_status_from_gate(gate: dict[str, Any]) -> str:
    if gate.get("safe_to_open_new_entries"):
        return "passed"
    if _int(gate.get("accepted_count")) > 0:
        return "partial"
    return "blocked"


def _maturity_score(*, steps: list[dict[str, Any]], final_decision: dict[str, Any]) -> dict[str, Any]:
    points = 0.0
    max_points = 100.0
    by_dimension = {
        "machine_surface": 0.0,
        "ml_dataset": 0.0,
        "exit_payoff": 0.0,
        "lane_split": 0.0,
        "sample_expansion": 0.0,
        "side_veto": 0.0,
        "portfolio_gate": 0.0,
        "readiness_scan": 0.0,
    }
    for step in steps:
        status = str(step.get("status") or "")
        name = str(step.get("name") or "")
        read = step.get("machine_read") if isinstance(step.get("machine_read"), dict) else {}
        if name == "machine_ml_dataset_and_gate_config":
            by_dimension["ml_dataset"] = 15.0 if status == "passed" else 0.0
        elif name == "btc_eth_exit_payoff_sweep":
            by_dimension["exit_payoff"] = 15.0 if status == "passed" else 5.0 if status == "partial" else 0.0
        elif name == "btc_eth_lane_split":
            by_dimension["lane_split"] = 10.0 if status == "passed" else 4.0 if status == "partial" else 0.0
        elif name == "btc_eth_30m_sample_expansion":
            by_dimension["sample_expansion"] = 10.0 if status == "passed" else 4.0 if status == "partial" else 0.0
        elif name == "loss_diagnostics_side_veto":
            summary = read.get("summary") if isinstance(read.get("summary"), dict) else {}
            by_dimension["side_veto"] = 15.0 if summary else 0.0
        elif name == "six_symbol_portfolio_gate":
            accepted = _int(read.get("accepted_count"))
            by_dimension["portfolio_gate"] = 15.0 if final_decision.get("safe_to_open_new_entries") else min(accepted * 2.0, 10.0)
        elif name == "testnet_readiness_scan":
            allowed = _int(read.get("allowed_count"))
            candidates = _int(read.get("candidate_count"))
            by_dimension["readiness_scan"] = 8.0 if allowed > 0 else 4.0 if candidates > 0 else 0.0
    points = sum(by_dimension.values())
    # Surface audit is counted outside step list because it guards the whole process.
    by_dimension["machine_surface"] = 20.0
    points += by_dimension["machine_surface"]
    return {
        "score_100": round(min(points, max_points), 2),
        "score_10": round(min(points, max_points) / 10.0, 2),
        "by_dimension": by_dimension,
        "target_for_9_plus": {
            "required_score_100": 90,
            "missing_points": round(max(90.0 - min(points, max_points), 0.0), 2),
        },
    }


def _write_machine_research_config(
    *,
    root: Path,
    symbols: list[str],
    intervals: list[str],
    limit: int,
    dataset_path: str,
) -> Path:
    base = yaml.safe_load(DEFAULT_DISCOVERY_CONFIG.read_text(encoding="utf-8")) or {}
    base.setdefault("universe", {})
    base["universe"]["symbols"] = symbols
    base["universe"]["intervals"] = intervals
    base["universe"]["limit"] = limit
    base.setdefault("research_entry_gate", {})
    base["research_entry_gate"]["route_side_veto"] = True
    base["research_entry_gate"]["shadow_route_side_veto"] = False
    base["research_entry_gate"]["historical_signal_veto"] = True
    base["research_entry_gate"]["shadow_historical_signal_veto"] = False
    base.setdefault("feature_label_gate", {})
    base["feature_label_gate"]["enabled"] = True
    base["feature_label_gate"]["dataset_path"] = dataset_path
    base["feature_label_gate"]["allow_if_insufficient_samples"] = False
    base["feature_label_gate"]["use_ml_meta_features"] = True
    config_path = root / "machine-six-symbol-discovery.yaml"
    config_path.write_text(yaml.safe_dump(base, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return config_path


def run_ai_expectancy_upgrade(
    *,
    output_dir: str | Path | None = None,
    symbols: list[str] | None = None,
    discovery_symbols: list[str] | None = None,
    limit: int = 8000,
    sweep_limit: int = 5000,
    max_configs: int = 80,
    max_walk_forward_validations: int = 12,
    universe_limit: int = 12,
    max_readiness_candidates: int = 6,
    readiness_execution_mode: str = "testnet_exploration",
    dry_run: bool = False,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    root = Path(output_dir).expanduser().resolve() if output_dir else AI_EXPECTANCY_UPGRADE_DIR / _stamp()
    root.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    steps: list[UpgradeStep] = []

    surface_audit = run_ai_surface_audit(output_dir=root / "ai-surface-audit")
    hermes = run_hermes_ai_trader(output_dir=root / "hermes-ai-trader")
    symbol_allocation = _machine_symbol_allocation(
        settings,
        hermes,
        requested_symbols=symbols,
        requested_discovery_symbols=discovery_symbols,
        universe_limit=universe_limit,
    )
    btc_eth = list(symbol_allocation["btc_eth_symbols"])
    six_symbols = list(symbol_allocation["portfolio_symbols"])

    if dry_run:
        payload = {
            "generated_at": _utc_now().isoformat(),
            "mode": "ai_expectancy_upgrade_v1",
            "dry_run": True,
            "safety": {
                "opens_orders": False,
                "writes_execution_config": False,
                "mainnet_live_allowed": False,
            },
            "symbol_allocation": symbol_allocation,
            "selected_symbols": {"btc_eth": btc_eth, "six_symbol_portfolio": six_symbols},
            "planned_steps": [
                "P0 rebuild ML feature dataset and machine-only research config",
                "P1 exit/payoff sweep on BTC/ETH",
                "P2 split BTC and ETH alpha lanes",
                "P3 30m sample expansion",
                "P4 loss diagnostics side veto",
                "P5 six-symbol portfolio gate",
                "P6 testnet readiness scan",
            ],
            "hermes_report_path": hermes.get("report_path"),
            "machine_strategy": hermes.get("machine_strategy"),
            "ai_surface_audit": {
                "status": surface_audit.get("status"),
                "blocker_count": surface_audit.get("blocker_count"),
                "report_path": surface_audit.get("report_path"),
            },
        }
        report_path = root / "ai-expectancy-upgrade-plan.json"
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        payload["report_path"] = str(report_path)
        return payload

    feature_dataset = build_feature_dataset(
        settings,
        spec=FeatureDatasetSpec(
            symbols=six_symbols,
            intervals=["30m", "1h", "4h"],
            limit=limit,
            strategy_config=str(CONFIG_DIR / "strategy-market-bot-payoff-research.yaml"),
        ),
        output_dir=root / f"feature-dataset-six-symbol-l{limit}",
    )
    machine_config_path = _write_machine_research_config(
        root=root,
        symbols=six_symbols,
        intervals=["30m", "1h", "4h"],
        limit=limit,
        dataset_path=str(feature_dataset.get("dataset_path") or ""),
    )
    steps.append(
        UpgradeStep(
            priority="P0",
            name="machine_ml_dataset_and_gate_config",
            objective="rebuild point-in-time ML meta-label dataset and route-side veto config before research",
            status="passed" if not feature_dataset.get("errors") and feature_dataset.get("row_count") else "blocked",
            report_path=str(feature_dataset.get("report_path") or ""),
            machine_read={
                "row_count": feature_dataset.get("row_count"),
                "dataset_hash": feature_dataset.get("dataset_hash"),
                "feature_manifest_hash": (feature_dataset.get("feature_manifest") or {}).get("manifest_hash"),
                "dataset_path": feature_dataset.get("dataset_path"),
                "machine_config_path": str(machine_config_path),
                "errors": feature_dataset.get("errors"),
            },
        )
    )

    sweep = run_risk_combo_sweep(
        symbols=btc_eth,
        limit=sweep_limit,
        grid_mode="focused",
        target_profit_factor=1.25,
        min_test_trades=100,
        min_win_rate=45,
        max_stop_loss_ratio=55,
        min_expectancy_r=0.05,
        min_payoff_ratio=1.20,
        max_configs=max_configs,
        max_walk_forward_validations=max_walk_forward_validations,
        skip_news=True,
        top_n=20,
    )
    steps.append(
        UpgradeStep(
            priority="P1",
            name="btc_eth_exit_payoff_sweep",
            objective="maximize expectancy and payoff through exit/risk geometry before adding entries",
            status="passed" if (sweep.get("aggregate") or {}).get("robust_recovery_candidate_count") else "blocked",
            report_path=str(sweep.get("report_path") or ""),
            machine_read={
                "robust_recovery_candidate_count": (sweep.get("aggregate") or {}).get("robust_recovery_candidate_count"),
                "recovery_candidate_count": (sweep.get("aggregate") or {}).get("recovery_candidate_count"),
                "best_by_symbol": sweep.get("best_by_symbol"),
            },
        )
    )

    split_reports: list[dict[str, Any]] = []
    for symbol in btc_eth:
        alpha = run_aggressive_alpha_research(
            settings,
            config_path=machine_config_path,
            output_dir=root / f"alpha-{symbol.lower()}-split-l{limit}",
            symbol_overrides=[symbol],
            interval_overrides=["1h", "4h"],
            limit_override=limit,
        )
        gate = evaluate_market_bot_gate(
            alpha_report=_alpha_report_path(alpha),
            config_path=DEFAULT_MARKET_BOT_GATE_CONFIG,
            output_dir=root / f"gate-{symbol.lower()}-split-l{limit}",
        )
        split_reports.append({"symbol": symbol, "alpha": alpha, "gate": gate})
    steps.append(
        UpgradeStep(
            priority="P2",
            name="btc_eth_lane_split",
            objective="separate BTC and ETH cohorts so one symbol does not hide the other symbol's edge",
            status="passed" if any((item["gate"].get("accepted_count") or 0) for item in split_reports) else "blocked",
            report_path=str(root),
            machine_read={
                item["symbol"]: _best_gate_read(item["gate"])
                for item in split_reports
            },
        )
    )

    alpha_30m = run_aggressive_alpha_research(
        settings,
        config_path=machine_config_path,
        output_dir=root / f"alpha-btc-eth-30m-l{limit}",
        symbol_overrides=btc_eth,
        interval_overrides=["30m"],
        limit_override=limit,
    )
    gate_30m = evaluate_market_bot_gate(
        alpha_report=_alpha_report_path(alpha_30m),
        config_path=DEFAULT_MARKET_BOT_GATE_CONFIG,
        output_dir=root / f"gate-btc-eth-30m-l{limit}",
    )
    steps.append(
        UpgradeStep(
            priority="P3",
            name="btc_eth_30m_sample_expansion",
            objective="increase sample count without lowering expectancy/payoff gates",
            status=_step_status_from_gate(gate_30m),
            report_path=str(gate_30m.get("report_path") or ""),
            machine_read=_best_gate_read(gate_30m),
        )
    )

    diagnostics = run_loss_diagnostics(min_bucket_trades=5, top_n=20)
    steps.append(
        UpgradeStep(
            priority="P4",
            name="loss_diagnostics_side_veto",
            objective="convert losing route-side buckets into veto surfaces",
            status=str(diagnostics.get("status") or "ok"),
            report_path=str(diagnostics.get("report_path") or ""),
            machine_read={
                "summary": diagnostics.get("summary"),
                "side_policy_recommendations": diagnostics.get("side_policy_recommendations"),
                "root_cause_recommendations": diagnostics.get("root_cause_recommendations"),
            },
        )
    )

    alpha_six = run_aggressive_alpha_research(
        settings,
        config_path=machine_config_path,
        output_dir=root / f"alpha-six-symbol-l{limit}",
        symbol_overrides=six_symbols,
        interval_overrides=["1h", "4h"],
        limit_override=limit,
    )
    gate_six = evaluate_market_bot_gate(
        alpha_report=_alpha_report_path(alpha_six),
        config_path=DEFAULT_MARKET_BOT_GATE_CONFIG,
        output_dir=root / f"gate-six-symbol-l{limit}",
    )
    steps.append(
        UpgradeStep(
            priority="P5",
            name="six_symbol_portfolio_gate",
            objective="promote only a diversified positive-expectancy cohort set",
            status=_step_status_from_gate(gate_six),
            report_path=str(gate_six.get("report_path") or ""),
            machine_read=_best_gate_read(gate_six),
        )
    )

    readiness_scan = run_ai_readiness_scan(
        output_dir=root / "readiness-scan",
        execution_mode=readiness_execution_mode,
        max_candidates=max_readiness_candidates,
    )
    readiness_read = {
        "candidate_count": readiness_scan.get("candidate_count"),
        "scanned_count": readiness_scan.get("scanned_count"),
        "allowed_count": readiness_scan.get("allowed_count"),
        "selected_ready_candidate": readiness_scan.get("selected_ready_candidate"),
        "next_machine_action": readiness_scan.get("next_machine_action"),
        "hard_blocker_taxonomy": readiness_scan.get("hard_blocker_taxonomy"),
        "execution_ticket": readiness_scan.get("execution_ticket"),
        "report_path": readiness_scan.get("report_path"),
    }
    steps.append(
        UpgradeStep(
            priority="P6",
            name="testnet_readiness_scan",
            objective="convert promoted candidates into testnet-forward evidence tickets without opening mainnet orders",
            status="passed"
            if _int(readiness_scan.get("allowed_count")) > 0
            else "partial"
            if _int(readiness_scan.get("candidate_count")) > 0
            else "blocked",
            report_path=str(readiness_scan.get("report_path") or ""),
            machine_read=readiness_read,
        )
    )

    step_payloads = [step.to_dict() for step in steps]
    final_decision = {
        "safe_to_open_new_entries": gate_six.get("safe_to_open_new_entries"),
        "accepted_count": gate_six.get("accepted_count"),
        "accepted_symbols": (gate_six.get("portfolio_gate") or {}).get("accepted_symbols") or [],
        "next_surface": (
            "testnet_forward_evidence"
            if gate_six.get("safe_to_open_new_entries") and _int(readiness_scan.get("allowed_count")) > 0
            else "hermes_ai_trader_live_readiness"
            if gate_six.get("safe_to_open_new_entries")
            else "continue_expectancy_research"
        ),
        "readiness_allowed_count": readiness_scan.get("allowed_count"),
        "readiness_next_machine_action": readiness_scan.get("next_machine_action"),
    }
    payload = {
        "generated_at": _utc_now().isoformat(),
        "mode": "ai_expectancy_upgrade_v1",
        "safety": {
            "opens_orders": False,
            "writes_execution_config": False,
            "mainnet_live_allowed": False,
            "uses_testnet_for_private_channels": settings.use_testnet,
            "live_trading_enabled": settings.live_trading_enabled,
        },
        "objective": "increase fixed-risk expectancy and expected return by allocating compute to exploit, harvest, exit-surface exploration, side veto, and portfolio gating",
        "symbol_allocation": symbol_allocation,
        "selected_symbols": {"btc_eth": btc_eth, "six_symbol_portfolio": six_symbols},
        "hermes_machine_strategy": hermes.get("machine_strategy"),
        "ai_surface_audit": {
            "status": surface_audit.get("status"),
            "blocker_count": surface_audit.get("blocker_count"),
            "report_path": surface_audit.get("report_path"),
        },
        "steps": step_payloads,
        "readiness_scan": readiness_read,
        "final_machine_decision": final_decision,
        "maturity_score": _maturity_score(steps=step_payloads, final_decision=final_decision),
    }
    report_path = root / "ai-expectancy-upgrade.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload
