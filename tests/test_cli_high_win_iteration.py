from __future__ import annotations

from argparse import Namespace

import binance_quant_control.cli as cli


def test_cmd_high_win_iteration_compact_uses_shared_summary(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "run_high_win_iteration",
        lambda **_kwargs: {
            "mode": "high_win_research_iteration",
            "safety": {"opens_orders": False},
            "targets": {"min_trades": 100},
            "best_alpha_path": "state/short/alpha-research-ranking.json",
            "best_alpha_gate": {"passed": False, "blockers": ["sample-expansion-regression"]},
            "promotion_allowed": False,
            "safe_to_open_new_entries": False,
            "execution_recommendation": "block_new_entries_and_continue_research",
            "alpha_evaluations": [
                {
                    "path": "state/short/alpha-research-ranking.json",
                    "summary": {"trade_count": 8, "weighted_win_rate": 87.5},
                    "target_gaps": {},
                    "gate": {"targets": {"min_trades": 100}},
                    "sample_expansion_regressions": [
                        {
                            "cohort_id": "TRXUSDT:4h:mean_reversion",
                            "short_sample_limit": 900,
                            "expanded_sample_limit": 3000,
                            "short_sample": {"trade_count": 8, "win_rate": 87.5},
                            "expanded_sample": {"trade_count": 17, "win_rate": 70.59},
                        }
                    ],
                }
            ],
            "cohort_expansion_candidates": [
                {
                    "cohort_id": "NAORISUSDT:1h:mean_reversion",
                    "symbol": "NAORISUSDT",
                    "interval": "1h",
                    "strategy_family": "mean_reversion",
                    "trade_count": 23,
                }
            ],
            "next_actions": [{"code": "reject-short-sample-regression"}],
            "report_path": "state/high-win-iteration/report.json",
        },
    )
    monkeypatch.setattr(
        cli,
        "print_json",
        lambda payload, compact=False: captured.update({"payload": payload, "compact": compact}),
    )

    cli.cmd_high_win_iteration(
        Namespace(
            config="config/high-win-iteration.default.yaml",
            alpha_report=[
                "state/short/alpha-research-ranking.json,state/expanded/alpha-research-ranking.json"
            ],
            sweep_report=None,
            output_dir="",
            no_write_pid_state=True,
            compact=True,
        )
    )

    payload = captured["payload"]  # type: ignore[assignment]
    assert captured["compact"] is True
    assert payload["best_alpha_path"] == "state/short/alpha-research-ranking.json"  # type: ignore[index]
    assert payload["cohort_expansion_candidate_count"] == 1  # type: ignore[index]
    assert payload["sample_expansion_regression_count"] == 1  # type: ignore[index]
    assert payload["sample_expansion_regressions"][0]["cohort_id"] == "TRXUSDT:4h:mean_reversion"  # type: ignore[index]
    assert payload["next_action_codes"] == ["reject-short-sample-regression"]  # type: ignore[index]


def test_split_csv_arg_accepts_repeated_and_comma_separated_values() -> None:
    assert cli.split_csv_arg(["a.json,b.json", " c.json "]) == ["a.json", "b.json", "c.json"]
    assert cli.split_csv_arg(None) == []
