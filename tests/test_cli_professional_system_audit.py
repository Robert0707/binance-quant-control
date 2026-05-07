from __future__ import annotations

from argparse import Namespace

import binance_quant_control.cli as cli


def test_cmd_professional_system_audit_compact(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "run_professional_system_audit",
        lambda **_kwargs: {
            "mode": "professional_system_audit",
            "trade_ready": False,
            "execution_recommendation": "block_new_entries_and_rebuild_edge",
            "layer_summary": {"total": 2, "counts": {"ready": 1, "missing": 1}},
            "critical_blockers": ["portfolio_construction:missing"],
            "evidence": {
                "alpha_report": {"promotion_eligible_count": 0},
                "high_win_iteration": {"safe_to_open_new_entries": False},
            },
            "recommendations": {"rebuild": ["portfolio construction"]},
            "report_path": "state/professional-system-audit/report.json",
        },
    )
    monkeypatch.setattr(
        cli,
        "print_json",
        lambda payload, compact=False: captured.update({"payload": payload, "compact": compact}),
    )

    cli.cmd_professional_system_audit(
        Namespace(
            config="config/professional-system-blueprint.default.yaml",
            output_dir="",
            compact=True,
        )
    )

    assert captured["compact"] is True
    payload = captured["payload"]  # type: ignore[assignment]
    assert payload["trade_ready"] is False  # type: ignore[index]
    assert payload["critical_blockers"] == ["portfolio_construction:missing"]  # type: ignore[index]
    assert payload["alpha_report"]["promotion_eligible_count"] == 0  # type: ignore[index]
