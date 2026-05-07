from __future__ import annotations

import binance_quant_control.ai_surface_audit as audit


def test_ai_surface_audit_reports_machine_only_status(tmp_path, monkeypatch) -> None:
    surface = tmp_path / "surface.py"
    surface.write_text(
        'requires_operator_execute = True\nnote = "operator_execute_required"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(audit, "DECISION_SURFACES", ("surface.py",))
    monkeypatch.setattr(audit, "ensure_runtime_dirs", lambda: None)

    payload = audit.run_ai_surface_audit(output_dir=tmp_path / "audit")

    assert payload["status"] == "passed"
    assert payload["blocker_count"] == 0
    assert payload["allowed_count"] == 0


def test_ai_surface_audit_blocks_human_advisory_leakage(tmp_path, monkeypatch) -> None:
    surface = tmp_path / "surface.py"
    surface.write_text('gate_input = "human advisory note"\n', encoding="utf-8")
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(audit, "DECISION_SURFACES", ("surface.py",))
    monkeypatch.setattr(audit, "ensure_runtime_dirs", lambda: None)

    payload = audit.run_ai_surface_audit(output_dir=tmp_path / "audit")

    assert payload["status"] == "blocked"
    assert payload["blocker_count"] == 2
    assert payload["blockers"][0]["token"] == "human"
