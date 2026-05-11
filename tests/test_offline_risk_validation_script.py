from __future__ import annotations

import json

import scripts.run_offline_risk_validation as offline


def test_offline_risk_validation_status_reads_completed_summary(monkeypatch, tmp_path, capsys) -> None:
    state_dir = tmp_path / "offline-risk-validation"
    run_dir = state_dir / "20260511T010203Z-trxusdt-buy-1d"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(offline, "STATE_DIR", state_dir)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "pid": 999999,
                "run_dir": str(run_dir),
                "log_path": str(run_dir / "run.log"),
                "opens_orders": False,
                "writes_execution_config": False,
                "mainnet_live_allowed": False,
                "status": "running",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "opens_orders": False,
                "writes_execution_config": False,
                "mainnet_live_allowed": False,
                "steps": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert offline.status(object()) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["running"] is False
    assert payload["summary_exists"] is True
    assert payload["summary"]["opens_orders"] is False
    assert payload["summary"]["writes_execution_config"] is False
    assert payload["summary"]["mainnet_live_allowed"] is False
