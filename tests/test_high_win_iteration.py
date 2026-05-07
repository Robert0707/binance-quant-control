from __future__ import annotations

import json
from pathlib import Path

import yaml

import binance_quant_control.high_win_convergence as convergence
import binance_quant_control.high_win_iteration as iteration


def _write_report(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_config(tmp_path: Path, alpha_report: Path, extra_alpha_reports: list[Path] | None = None) -> Path:
    config = {
        "targets": {
            "min_trades": 100,
            "min_win_rate": 65.0,
            "max_stop_loss_ratio": 35.0,
            "min_profit_factor": 1.5,
            "min_expectancy_r": 0.10,
            "min_payoff_ratio": 1.15,
            "max_per_trade_risk_pct": 2.5,
        },
        "portfolio_gate": {"min_promoted_symbols": 0, "required_symbols": []},
        "reports": {
            "alpha_reports": [str(alpha_report), *[str(path) for path in (extra_alpha_reports or [])]],
            "sweep_reports": [],
        },
        "pid": {"state_path": str(tmp_path / "pid-state.json"), "max_limit": 10000},
        "candidate_expansion": {"enabled": True, "top_n": 8, "min_trade_count": 10, "max_limit": 30000},
        "research_batches": {
            "core": {"config": "config/core-high-win-research.default.yaml"},
            "replacement_scout": {"config": "config/core-replacement-scout.default.yaml"},
        },
        "commands": {
            "core_l5000": "openclaw-quantctl alpha-research --config config/core-high-win-research.default.yaml --compact",
            "replacement_scout_l5000": "openclaw-quantctl alpha-research --config config/core-replacement-scout.default.yaml --compact",
            "strict_risk_combo_sweep": "openclaw-quantctl risk-combo-sweep --min-win-rate 65 --max-stop-loss-ratio 35 --target-profit-factor 1.5 --compact",
        },
    }
    path = tmp_path / "high-win-iteration.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _write_config_with_sweep(tmp_path: Path, alpha_report: Path, sweep_report: Path) -> Path:
    path = _write_config(tmp_path, alpha_report)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["reports"]["sweep_reports"] = [str(sweep_report)]
    config["commands"][
        "strict_risk_combo_sweep"
    ] = "openclaw-quantctl risk-combo-sweep --symbols PAXGUSDT,ETHUSDT --limit 5000 --max-configs 80 --compact"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_high_win_iteration_blocks_current_like_smoke_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(iteration, "HIGH_WIN_ITERATION_DIR", tmp_path / "state")
    report_path = _write_report(
        tmp_path / "alpha-research-ranking.json",
        {
            "generated_at": "2026-05-03T01:34:15+00:00",
            "mainnet_live_allowed": False,
            "strategy_profile": "core-high-win-research",
            "limit": 1500,
            "execution_recommendation": "block_new_entries_and_continue_research",
            "strategy_families": ["trend_continuation", "liquidity_reclaim"],
            "rows": [
                {
                    "symbol": "PAXGUSDT",
                    "interval": "1d",
                    "strategy_family": "trend_continuation",
                    "ranking_score": 0.128,
                    "profit_factor": 0.5664,
                    "trade_count": 33,
                    "win_rate": 69.6994,
                    "stop_loss_ratio": 30.3006,
                    "promotion_eligible": False,
                    "total_return_pct": -1.0,
                }
            ],
        },
    )
    config_path = _write_config(tmp_path, report_path)

    payload = iteration.run_high_win_iteration(
        config_path=config_path,
        output_dir=tmp_path / "out",
        write_pid_state=False,
    )

    assert payload["promotion_allowed"] is False
    assert payload["safe_to_open_new_entries"] is False
    assert payload["execution_recommendation"] == "block_new_entries_and_continue_research"
    assert payload["best_alpha_gate"]["blockers"] == [
        "aggregate-trade-count-below-floor",
        "aggregate-profit-factor-below-floor",
        "aggregate-expectancy-r-below-floor",
        "aggregate-payoff-ratio-below-floor",
        "no-promotion-eligible-cohort",
    ]
    action_codes = {item["code"] for item in payload["next_actions"]}
    assert "reject-low-pf-cohorts-and-scout-replacements" in action_codes
    assert "improve-risk-reward-expectancy" in action_codes
    assert "family-insufficient-sample" in action_codes
    assert Path(payload["report_path"]).exists()


def test_high_win_iteration_allows_only_strict_passing_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(iteration, "HIGH_WIN_ITERATION_DIR", tmp_path / "state")
    report_path = _write_report(
        tmp_path / "alpha-research-ranking.json",
        {
            "generated_at": "2026-05-03T03:00:00+00:00",
            "mainnet_live_allowed": False,
            "strategy_profile": "core-high-win-research",
            "limit": 5000,
            "execution_recommendation": "paper_or_testnet_candidate_available",
            "rows": [
                {
                    "symbol": "ETHUSDT",
                    "interval": "4h",
                    "strategy_family": "liquidity_reclaim",
                    "ranking_score": 12.0,
                    "profit_factor": 1.8,
                    "trade_count": 120,
                    "win_rate": 84.0,
                    "stop_loss_ratio": 16.0,
                    "expectancy_r": 0.22,
                    "payoff_ratio": 1.45,
                    "promotion_eligible": True,
                    "total_return_pct": 9.0,
                }
            ],
        },
    )
    config_path = _write_config(tmp_path, report_path)

    payload = iteration.run_high_win_iteration(
        config_path=config_path,
        output_dir=tmp_path / "out",
        write_pid_state=False,
    )

    assert payload["promotion_allowed"] is True
    assert payload["safe_to_open_new_entries"] is True
    assert payload["execution_recommendation"] == "paper_or_testnet_candidate_available_require_live_readiness"
    assert payload["best_alpha_gate"] == {
        "passed": True,
        "blockers": [],
        "targets": {
            "min_trades": 100,
            "min_win_rate": 65.0,
            "max_stop_loss_ratio": 35.0,
            "min_profit_factor": 1.5,
            "min_expectancy_r": 0.1,
            "min_payoff_ratio": 1.15,
        },
    }


def test_high_win_iteration_blocks_until_required_symbols_are_promoted(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(iteration, "HIGH_WIN_ITERATION_DIR", tmp_path / "state")
    report_path = _write_report(
        tmp_path / "single-passing-alpha-research-ranking.json",
        {
            "generated_at": "2026-05-03T07:00:00+00:00",
            "mainnet_live_allowed": False,
            "strategy_profile": "core-high-win-research",
            "limit": 26500,
            "execution_recommendation": "paper_or_testnet_candidate_available",
            "performance_summary": {
                "row_count": 1,
                "trade_count": 120,
                "promotion_eligible_count": 1,
                "weighted_win_rate": 84.0,
                "weighted_stop_loss_ratio": 16.0,
                "finite_avg_profit_factor": 1.8,
                "weighted_expectancy_r": 0.22,
                "weighted_payoff_ratio": 1.45,
                "positive_row_count": 1,
            },
            "rows": [
                {
                    "symbol": "NAORISUSDT",
                    "interval": "1h",
                    "strategy_family": "mean_reversion",
                    "ranking_score": 12.0,
                    "profit_factor": 1.8,
                    "trade_count": 120,
                    "win_rate": 84.0,
                    "stop_loss_ratio": 16.0,
                    "expectancy_r": 0.22,
                    "payoff_ratio": 1.45,
                    "promotion_eligible": True,
                    "total_return_pct": 9.0,
                }
            ],
        },
    )
    config_path = _write_config(tmp_path, report_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["portfolio_gate"] = {
        "min_promoted_symbols": 2,
        "required_symbols": ["NAORISUSDT", "APEUSDT"],
    }
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    payload = iteration.run_high_win_iteration(
        config_path=config_path,
        output_dir=tmp_path / "out",
        write_pid_state=False,
    )

    assert payload["best_alpha_gate"]["passed"] is True
    assert payload["portfolio_gate"]["passed"] is False
    assert payload["portfolio_gate"]["blockers"] == [
        "promoted-symbol-count-below-floor",
        "required-promoted-symbols-missing",
    ]
    assert payload["promotion_allowed"] is False
    assert payload["safe_to_open_new_entries"] is False


def test_high_win_iteration_blocks_short_sample_that_regresses_on_expansion(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(iteration, "HIGH_WIN_ITERATION_DIR", tmp_path / "state")
    short_report = _write_report(
        tmp_path / "short-alpha-research-ranking.json",
        {
            "generated_at": "2026-05-03T05:00:00+00:00",
            "mainnet_live_allowed": False,
            "strategy_profile": "core-high-win-research",
            "limit": 900,
            "rows": [
                {
                    "cohort_id": "TRXUSDT:4h:mean_reversion",
                    "symbol": "TRXUSDT",
                    "interval": "4h",
                    "strategy_family": "mean_reversion",
                    "ranking_score": 4.0,
                    "profit_factor": 3.0,
                    "trade_count": 8,
                    "win_rate": 87.5,
                    "stop_loss_ratio": 12.5,
                    "expectancy_r": 0.32,
                    "payoff_ratio": 1.60,
                    "promotion_eligible": False,
                    "total_return_pct": 0.3,
                }
            ],
        },
    )
    expanded_report = _write_report(
        tmp_path / "expanded-alpha-research-ranking.json",
        {
            "generated_at": "2026-05-03T05:20:00+00:00",
            "mainnet_live_allowed": False,
            "strategy_profile": "core-high-win-research",
            "limit": 3000,
            "rows": [
                {
                    "cohort_id": "TRXUSDT:4h:mean_reversion",
                    "symbol": "TRXUSDT",
                    "interval": "4h",
                    "strategy_family": "mean_reversion",
                    "ranking_score": 1.0,
                    "profit_factor": 0.68,
                    "trade_count": 17,
                    "win_rate": 70.59,
                    "stop_loss_ratio": 29.41,
                    "expectancy_r": 0.02,
                    "payoff_ratio": 0.80,
                    "promotion_eligible": False,
                    "total_return_pct": -0.5,
                }
            ],
        },
    )
    config_path = _write_config(tmp_path, short_report, [expanded_report])

    payload = iteration.run_high_win_iteration(
        config_path=config_path,
        output_dir=tmp_path / "out",
        write_pid_state=False,
    )

    short_eval = next(
        item
        for item in payload["alpha_evaluations"]
        if item["path"] == str(short_report)
    )

    assert "sample-expansion-regression" in short_eval["gate"]["blockers"]
    assert short_eval["sample_expansion_regressions"][0]["cohort_id"] == "TRXUSDT:4h:mean_reversion"
    assert payload["promotion_allowed"] is False
    assert payload["safe_to_open_new_entries"] is False
    assert "reject-short-sample-regression" in {item["code"] for item in payload["next_actions"]}


def test_high_win_iteration_expands_target_shaped_under_sampled_candidates(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(iteration, "HIGH_WIN_ITERATION_DIR", tmp_path / "state")
    report_path = _write_report(
        tmp_path / "replacement-alpha-research-ranking.json",
        {
            "generated_at": "2026-05-03T06:00:00+00:00",
            "mainnet_live_allowed": False,
            "strategy_profile": "core-high-win-research",
            "limit": 5000,
            "rows": [
                {
                    "cohort_id": "NAORISUSDT:1h:mean_reversion",
                    "symbol": "NAORISUSDT",
                    "interval": "1h",
                    "strategy_family": "mean_reversion",
                    "ranking_score": 1.3,
                    "profit_factor": 3.891,
                    "trade_count": 23,
                    "win_rate": 82.61,
                    "stop_loss_ratio": 17.39,
                    "expectancy_r": 0.35,
                    "payoff_ratio": 1.60,
                    "promotion_eligible": False,
                    "total_return_pct": 6.23,
                },
                {
                    "cohort_id": "DOGEUSDT:1d:trend_pullback",
                    "symbol": "DOGEUSDT",
                    "interval": "1d",
                    "strategy_family": "trend_pullback",
                    "ranking_score": 1.1,
                    "profit_factor": "inf",
                    "trade_count": 4,
                    "win_rate": 100.0,
                    "stop_loss_ratio": 0.0,
                    "expectancy_r": 0.20,
                    "payoff_ratio": "inf",
                    "promotion_eligible": False,
                    "total_return_pct": 1.34,
                },
                {
                    "cohort_id": "SOLUSDT:4h:mean_reversion",
                    "symbol": "SOLUSDT",
                    "interval": "4h",
                    "strategy_family": "mean_reversion",
                    "ranking_score": 1.0,
                    "profit_factor": 0.369,
                    "trade_count": 12,
                    "win_rate": 41.67,
                    "stop_loss_ratio": 58.33,
                    "expectancy_r": -0.30,
                    "payoff_ratio": 0.70,
                    "promotion_eligible": False,
                    "total_return_pct": -2.38,
                },
            ],
        },
    )
    config_path = _write_config(tmp_path, report_path)

    payload = iteration.run_high_win_iteration(
        config_path=config_path,
        output_dir=tmp_path / "out",
        write_pid_state=False,
    )

    assert payload["promotion_allowed"] is False
    candidates = payload["cohort_expansion_candidates"]
    assert [item["cohort_id"] for item in candidates] == ["NAORISUSDT:1h:mean_reversion"]
    assert candidates[0]["suggested_next_limit"] == 26500
    action = next(
        item for item in payload["next_actions"] if item["code"] == "expand-target-shaped-under-sampled-cohorts"
    )
    assert action["priority"] == "critical"
    assert action["params"]["symbols"] == ["NAORISUSDT"]
    assert "config/core-high-win-research.default.yaml" in action["command"]
    assert "config/core-replacement-scout.default.yaml" not in action["command"]


def test_high_win_iteration_expands_payoff_shaped_sweep_candidates(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(iteration, "HIGH_WIN_ITERATION_DIR", tmp_path / "state")
    alpha_report = _write_report(
        tmp_path / "alpha-research-ranking.json",
        {
            "generated_at": "2026-05-04T11:00:00+00:00",
            "mainnet_live_allowed": False,
            "strategy_profile": "core-high-win-research",
            "limit": 1500,
            "rows": [],
        },
    )
    sweep_report = _write_report(
        tmp_path / "risk-combo-sweep.json",
        {
            "generated_at": "2026-05-04T11:27:45+00:00",
            "status": "ok",
            "aggregate": {
                "limit": 900,
                "recovery_candidate_count": 0,
                "robust_recovery_candidate_count": 0,
                "target_profit_factor": 1.5,
                "min_test_trades": 100,
                "min_win_rate": 65.0,
                "max_stop_loss_ratio": 35.0,
                "min_expectancy_r": 0.10,
                "min_payoff_ratio": 1.15,
            },
            "best_by_symbol": {
                "TRXUSDT": {
                    "route_id": "trx-mean-reversion",
                    "requested_symbol": "TRXUSDT",
                    "interval": "4h",
                    "strategy_profile": "core-high-win-research",
                    "params": {"exit_profile": "payoff_runner", "primary_tp_multiple": 1.5},
                    "full": {
                        "trade_count": 15,
                        "win_rate": 60.0,
                        "stop_loss_ratio": 40.0,
                        "profit_factor": 2.0127,
                        "expectancy_r": 0.1966,
                        "payoff_ratio": 1.3418,
                    },
                    "test": {
                        "trade_count": 3,
                        "win_rate": 100.0,
                        "stop_loss_ratio": 0.0,
                        "profit_factor": "inf",
                        "expectancy_r": 0.5349,
                        "payoff_ratio": "inf",
                    },
                    "robust_recovery_gate": {
                        "passed": False,
                        "reasons": [
                            "test-trade-count-too-low",
                            "initial-recovery-gate-not-passed",
                            "walk-forward-min-profit-factor-below-target",
                            "walk-forward-min-payoff-ratio-below-target",
                        ],
                    },
                }
            },
        },
    )
    config_path = _write_config_with_sweep(tmp_path, alpha_report, sweep_report)

    payload = iteration.run_high_win_iteration(
        config_path=config_path,
        output_dir=tmp_path / "out",
        write_pid_state=False,
    )

    candidates = payload["sweep_under_sampled_candidates"]
    assert [item["symbol"] for item in candidates] == ["TRXUSDT"]
    action = next(item for item in payload["next_actions"] if item["code"] == "expand-payoff-shaped-risk-sweep-candidates")
    assert action["priority"] == "critical"
    assert action["params"]["symbols"] == ["TRXUSDT"]
    assert "--symbols TRXUSDT" in action["command"]


def test_high_win_convergence_loop_defaults_to_plan_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(convergence, "HIGH_WIN_CONVERGENCE_DIR", tmp_path / "convergence")
    report_path = _write_report(
        tmp_path / "alpha-research-ranking.json",
        {
            "generated_at": "2026-05-03T01:34:15+00:00",
            "mainnet_live_allowed": False,
            "strategy_profile": "core-high-win-research",
            "limit": 1500,
            "rows": [
                {
                    "symbol": "PAXGUSDT",
                    "interval": "1d",
                    "strategy_family": "trend_continuation",
                    "ranking_score": 0.128,
                    "profit_factor": 0.5664,
                    "trade_count": 33,
                    "win_rate": 69.6994,
                    "stop_loss_ratio": 30.3006,
                    "promotion_eligible": False,
                }
            ],
        },
    )
    config_path = _write_config(tmp_path, report_path)

    payload = convergence.run_high_win_convergence_loop(
        config_path=config_path,
        output_dir=tmp_path / "out",
        write_pid_state=False,
    )

    assert payload["status"] == "plan_only"
    assert payload["execute_research"] is False
    assert payload["policy"]["replacement_config"] == "config/core-replacement-scout.default.yaml"
    assert payload["rounds"] == []
    assert payload["promotion_allowed"] is False
    assert payload["safe_to_open_new_entries"] is False
    assert Path(payload["report_path"]).exists()


def test_high_win_convergence_loop_runs_until_promotion_when_executed(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(convergence, "HIGH_WIN_CONVERGENCE_DIR", tmp_path / "convergence")
    report_path = _write_report(
        tmp_path / "alpha-research-ranking.json",
        {
            "generated_at": "2026-05-03T01:34:15+00:00",
            "mainnet_live_allowed": False,
            "strategy_profile": "core-high-win-research",
            "limit": 1500,
            "strategy_families": ["trend_continuation"],
            "rows": [
                {
                    "symbol": "PAXGUSDT",
                    "interval": "1d",
                    "strategy_family": "trend_continuation",
                    "ranking_score": 0.128,
                    "profit_factor": 0.5664,
                    "trade_count": 33,
                    "win_rate": 69.6994,
                    "stop_loss_ratio": 30.3006,
                    "promotion_eligible": False,
                }
            ],
        },
    )
    config_path = _write_config(tmp_path, report_path)
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(convergence, "load_settings", lambda: object())

    def fake_alpha_research(settings, **kwargs):
        calls.append({"job": "alpha", **kwargs})
        output_dir = Path(kwargs["output_dir"])
        report = {
            "generated_at": "2026-05-03T04:00:00+00:00",
            "mainnet_live_allowed": False,
            "strategy_profile": "core-high-win-research",
            "limit": kwargs.get("limit_override") or 5000,
            "rows": [
                {
                    "symbol": "ETHUSDT",
                    "interval": "4h",
                    "strategy_family": "liquidity_reclaim",
                    "ranking_score": 12.0,
                    "profit_factor": 1.8,
                    "trade_count": 120,
                    "win_rate": 84.0,
                    "stop_loss_ratio": 16.0,
                    "expectancy_r": 0.22,
                    "payoff_ratio": 1.45,
                    "promotion_eligible": True,
                    "total_return_pct": 9.0,
                }
            ],
        }
        path = _write_report(output_dir / "alpha-research-ranking.json", report)
        return {**report, "report_path": str(path), "performance_summary": {}}

    monkeypatch.setattr(convergence, "run_aggressive_alpha_research", fake_alpha_research)
    monkeypatch.setattr(
        convergence,
        "run_risk_combo_sweep",
        lambda **kwargs: {
            "report_path": str(_write_report(tmp_path / "sweep.json", {"aggregate": {}})),
            "aggregate": {},
        },
    )

    payload = convergence.run_high_win_convergence_loop(
        config_path=config_path,
        output_dir=tmp_path / "out",
        max_rounds=1,
        execute_research=True,
        write_pid_state=False,
    )

    assert payload["status"] == "promotion_found"
    assert payload["execute_research"] is True
    assert payload["promotion_allowed"] is True
    assert payload["safe_to_open_new_entries"] is True
    assert len(payload["rounds"]) == 1
    assert any(call["job"] == "alpha" for call in calls)
