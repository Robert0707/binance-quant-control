from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .alpha_research import run_aggressive_alpha_research
from .config import STATE_DIR, ensure_runtime_dirs, load_settings
from .high_win_iteration import (
    HighWinTargets,
    _int,
    _load_yaml,
    _resolve_project_path,
    _stamp,
    _targets,
    _utc_now,
    compact_high_win_iteration,
    high_win_gap_score,
    run_high_win_iteration,
    suggested_next_limit,
)
from .risk_combo_sweep import run_risk_combo_sweep

HIGH_WIN_CONVERGENCE_DIR = STATE_DIR / "high-win-convergence"


@dataclass(frozen=True, slots=True)
class ConvergencePolicy:
    max_rounds: int = 3
    default_execute_research: bool = False
    stop_on_stagnation_rounds: int = 2
    run_core_research: bool = True
    run_replacement_scout: bool = True
    run_risk_combo_sweep: bool = True
    core_config: str = "config/core-high-win-research.default.yaml"
    replacement_config: str = "config/core-replacement-scout.default.yaml"
    risk_symbols: tuple[str, ...] = ("PAXGUSDT", "ETHUSDT", "XRPUSDT", "TRXUSDT")
    risk_limit: int = 5000
    risk_grid_mode: str = "focused"
    risk_skip_news: bool = True
    risk_top_n: int = 20
    risk_max_configs: int = 0
    risk_max_walk_forward_validations: int = 0


def _convergence_policy(raw: dict[str, Any]) -> ConvergencePolicy:
    convergence = raw.get("convergence") or {}
    batches = raw.get("research_batches") or {}
    core = batches.get("core") or {}
    replacement = batches.get("replacement_scout") or {}
    risk = batches.get("risk_combo_sweep") or {}
    risk_symbols = tuple(str(item).upper() for item in (risk.get("symbols") or []) if str(item).strip())
    return ConvergencePolicy(
        max_rounds=max(_int(convergence.get("max_rounds"), 3), 1),
        default_execute_research=bool(convergence.get("default_execute_research", False)),
        stop_on_stagnation_rounds=max(_int(convergence.get("stop_on_stagnation_rounds"), 2), 1),
        run_core_research=bool(core.get("enabled", True)),
        run_replacement_scout=bool(replacement.get("enabled", True)),
        run_risk_combo_sweep=bool(risk.get("enabled", True)),
        core_config=str(core.get("config") or "config/core-high-win-research.default.yaml"),
        replacement_config=str(replacement.get("config") or "config/core-replacement-scout.default.yaml"),
        risk_symbols=risk_symbols or ("PAXGUSDT", "ETHUSDT", "XRPUSDT", "TRXUSDT"),
        risk_limit=max(_int(risk.get("limit"), 5000), 500),
        risk_grid_mode=str(risk.get("grid_mode") or "focused"),
        risk_skip_news=bool(risk.get("skip_news", True)),
        risk_top_n=max(_int(risk.get("top_n"), 20), 1),
        risk_max_configs=max(_int(risk.get("max_configs"), 0), 0),
        risk_max_walk_forward_validations=max(_int(risk.get("max_walk_forward_validations"), 0), 0),
    )


def _selected_research_jobs(
    payload: dict[str, Any],
    *,
    policy: ConvergencePolicy,
) -> list[str]:
    if bool(payload.get("promotion_allowed")):
        return []
    blockers = set((payload.get("best_alpha_gate") or {}).get("blockers") or [])
    action_codes = {str(item.get("code") or "") for item in (payload.get("next_actions") or [])}
    jobs: list[str] = []
    if policy.run_core_research and (
        "aggregate-trade-count-below-floor" in blockers
        or "run-core-high-win-sample" in action_codes
        or "family-insufficient-sample" in action_codes
    ):
        jobs.append("core")
    if policy.run_replacement_scout and (
        "aggregate-profit-factor-below-floor" in blockers
        or "no-promotion-eligible-cohort" in blockers
        or "reject-low-pf-cohorts-and-scout-replacements" in action_codes
    ):
        jobs.append("replacement_scout")
    if policy.run_risk_combo_sweep and (
        "aggregate-win-rate-below-floor" in blockers
        or "aggregate-stop-loss-ratio-above-ceiling" in blockers
        or "aggregate-expectancy-r-below-floor" in blockers
        or "aggregate-payoff-ratio-below-floor" in blockers
        or "run-strict-risk-combo-sweep" in action_codes
        or "tighten-structure-and-stoploss-guard" in action_codes
        or "improve-risk-reward-expectancy" in action_codes
    ):
        jobs.append("risk_combo_sweep")
    return list(dict.fromkeys(jobs))


def _run_research_job(
    job: str,
    *,
    policy: ConvergencePolicy,
    targets: HighWinTargets,
    round_dir: Path,
    iteration_payload: dict[str, Any],
) -> dict[str, Any]:
    settings = load_settings()
    started_at = _utc_now()
    try:
        if job == "core":
            output_dir = round_dir / "core-alpha"
            payload = run_aggressive_alpha_research(
                settings,
                config_path=policy.core_config,
                output_dir=output_dir,
                limit_override=suggested_next_limit(iteration_payload) or None,
            )
            return {
                "job": job,
                "status": "ok",
                "started_at": started_at.isoformat(),
                "finished_at": _utc_now().isoformat(),
                "report_path": payload.get("report_path"),
                "summary": payload.get("performance_summary"),
            }
        if job == "replacement_scout":
            output_dir = round_dir / "replacement-alpha"
            payload = run_aggressive_alpha_research(
                settings,
                config_path=policy.replacement_config,
                output_dir=output_dir,
                limit_override=suggested_next_limit(iteration_payload) or None,
            )
            return {
                "job": job,
                "status": "ok",
                "started_at": started_at.isoformat(),
                "finished_at": _utc_now().isoformat(),
                "report_path": payload.get("report_path"),
                "summary": payload.get("performance_summary"),
            }
        if job == "risk_combo_sweep":
            payload = run_risk_combo_sweep(
                routes=[],
                symbols=list(policy.risk_symbols),
                limit=policy.risk_limit,
                grid_mode=policy.risk_grid_mode,
                target_profit_factor=targets.min_profit_factor,
                min_test_trades=targets.min_trades,
                min_win_rate=targets.min_win_rate,
                max_stop_loss_ratio=targets.max_stop_loss_ratio,
                min_expectancy_r=targets.min_expectancy_r,
                min_payoff_ratio=targets.min_payoff_ratio,
                max_symbols_per_route=0,
                include_all_route_symbols=False,
                skip_news=policy.risk_skip_news,
                top_n=policy.risk_top_n,
                max_configs=policy.risk_max_configs,
                max_walk_forward_validations=policy.risk_max_walk_forward_validations,
            )
            return {
                "job": job,
                "status": "ok",
                "started_at": started_at.isoformat(),
                "finished_at": _utc_now().isoformat(),
                "report_path": payload.get("report_path"),
                "summary": payload.get("aggregate"),
            }
        return {
            "job": job,
            "status": "skipped",
            "started_at": started_at.isoformat(),
            "finished_at": _utc_now().isoformat(),
            "error": "unknown-job",
        }
    except Exception as exc:
        return {
            "job": job,
            "status": "failed",
            "started_at": started_at.isoformat(),
            "finished_at": _utc_now().isoformat(),
            "error": str(exc),
        }


def run_high_win_convergence_loop(
    *,
    config_path: str | Path = "high-win-iteration.default.yaml",
    alpha_report_paths: list[str | Path] | None = None,
    sweep_report_paths: list[str | Path] | None = None,
    output_dir: str | Path | None = None,
    max_rounds: int | None = None,
    execute_research: bool | None = None,
    write_pid_state: bool = True,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    config = _load_yaml(config_path)
    targets = _targets(config)
    policy = _convergence_policy(config)
    rounds_to_run = max(int(max_rounds or policy.max_rounds), 1)
    should_execute = policy.default_execute_research if execute_research is None else bool(execute_research)
    root = _resolve_project_path(output_dir) if output_dir else HIGH_WIN_CONVERGENCE_DIR / _stamp()
    root.mkdir(parents=True, exist_ok=True)
    alpha_paths: list[str | Path] = list(alpha_report_paths or [])
    sweep_paths: list[str | Path] = list(sweep_report_paths or [])
    iteration = run_high_win_iteration(
        config_path=config_path,
        alpha_report_paths=alpha_paths,
        sweep_report_paths=sweep_paths,
        output_dir=root / "iteration-reports",
        write_pid_state=write_pid_state,
    )
    rounds: list[dict[str, Any]] = []
    status = "promotion_found" if bool(iteration.get("promotion_allowed")) else "plan_only"
    stagnant_rounds = 0
    previous_gap = high_win_gap_score(iteration)

    if should_execute and not bool(iteration.get("promotion_allowed")):
        status = "max_rounds_reached"
        for round_number in range(1, rounds_to_run + 1):
            selected_jobs = _selected_research_jobs(iteration, policy=policy)
            if not selected_jobs:
                status = "no_research_jobs_selected"
                rounds.append(
                    {
                        "round": round_number,
                        "before": compact_high_win_iteration(iteration),
                        "selected_jobs": [],
                        "job_results": [],
                    }
                )
                break
            round_dir = root / f"round-{round_number:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            job_results = [
                _run_research_job(
                    job,
                    policy=policy,
                    targets=targets,
                    round_dir=round_dir,
                    iteration_payload=iteration,
                )
                for job in selected_jobs
            ]
            for result in job_results:
                report_path = result.get("report_path")
                if not report_path:
                    continue
                if result.get("job") in {"core", "replacement_scout"}:
                    alpha_paths.append(str(report_path))
                elif result.get("job") == "risk_combo_sweep":
                    sweep_paths.append(str(report_path))
            next_iteration = run_high_win_iteration(
                config_path=config_path,
                alpha_report_paths=alpha_paths,
                sweep_report_paths=sweep_paths,
                output_dir=root / "iteration-reports",
                write_pid_state=write_pid_state,
            )
            next_gap = high_win_gap_score(next_iteration)
            improved = next_gap < previous_gap
            stagnant_rounds = 0 if improved else stagnant_rounds + 1
            rounds.append(
                {
                    "round": round_number,
                    "before": compact_high_win_iteration(iteration),
                    "selected_jobs": selected_jobs,
                    "job_results": job_results,
                    "after": compact_high_win_iteration(next_iteration),
                    "improved": improved,
                }
            )
            iteration = next_iteration
            previous_gap = next_gap
            if bool(iteration.get("promotion_allowed")):
                status = "promotion_found"
                break
            if not any(result.get("status") == "ok" for result in job_results):
                status = "research_jobs_failed"
                break
            if stagnant_rounds >= policy.stop_on_stagnation_rounds:
                status = "stagnated_requires_new_family_or_symbol_universe"
                break

    payload = {
        "generated_at": _utc_now().isoformat(),
        "mode": "high_win_convergence_loop",
        "status": status,
        "execute_research": should_execute,
        "safety": {
            "mainnet_live_allowed": False,
            "opens_orders": False,
            "writes_execution_config": False,
            "research_gate_only": True,
            "max_per_trade_risk_pct": targets.max_per_trade_risk_pct,
        },
        "policy": {
            "max_rounds": rounds_to_run,
            "stop_on_stagnation_rounds": policy.stop_on_stagnation_rounds,
            "run_core_research": policy.run_core_research,
            "run_replacement_scout": policy.run_replacement_scout,
            "run_risk_combo_sweep": policy.run_risk_combo_sweep,
            "core_config": policy.core_config,
            "replacement_config": policy.replacement_config,
            "risk_symbols": list(policy.risk_symbols),
            "risk_limit": policy.risk_limit,
            "risk_grid_mode": policy.risk_grid_mode,
            "risk_max_configs": policy.risk_max_configs,
            "risk_max_walk_forward_validations": policy.risk_max_walk_forward_validations,
        },
        "rounds": rounds,
        "final_iteration": compact_high_win_iteration(iteration),
        "promotion_allowed": bool(iteration.get("promotion_allowed")),
        "safe_to_open_new_entries": bool(iteration.get("safe_to_open_new_entries")),
        "execution_recommendation": iteration.get("execution_recommendation"),
        "alpha_reports": [str(path) for path in alpha_paths],
        "sweep_reports": [str(path) for path in sweep_paths],
        "operator_note": (
            "Run with execute_research=true only for bounded backtest/research batches. "
            "A promotion still requires live-readiness before any testnet entry."
        ),
    }
    report_path = root / f"{_stamp()}-high-win-convergence.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload
