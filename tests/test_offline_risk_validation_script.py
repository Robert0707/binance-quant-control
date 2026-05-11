from __future__ import annotations

import argparse
import json

import scripts.run_offline_risk_validation as offline


def test_offline_risk_validation_status_reads_completed_summary(monkeypatch, tmp_path, capsys) -> None:
    state_dir = tmp_path / "offline-risk-validation"
    run_dir = state_dir / "20260511T010203Z-trxusdt-buy-1d"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(offline, "STATE_DIR", state_dir)
    monkeypatch.setattr(
        offline,
        "_child_processes",
        lambda pid: [
            {
                "pid": "123",
                "stat": "Rl",
                "elapsed": "00:10",
                "pcpu": "99.9",
                "pmem": "1.0",
                "cmd": "binance-quant-control risk-combo-sweep",
            }
        ],
    )
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
    (run_dir / "progress.json").write_text(
        json.dumps(
            {
                "step": "ai_readiness_scan",
                "status": "ok",
                "opens_orders": False,
                "writes_execution_config": False,
                "mainnet_live_allowed": False,
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
    assert payload["progress"]["step"] == "ai_readiness_scan"
    assert payload["progress"]["opens_orders"] is False
    assert payload["child_processes"][0]["pcpu"] == "99.9"


def test_offline_risk_validation_start_records_timeout(monkeypatch, tmp_path, capsys) -> None:
    state_dir = tmp_path / "offline-risk-validation"
    monkeypatch.setattr(offline, "STATE_DIR", state_dir)
    monkeypatch.setattr(offline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(offline, "BINANCE_QUANT", tmp_path / ".venv" / "bin" / "binance-quant-control")
    monkeypatch.setattr(offline, "_stamp", lambda: "20260511T010203Z")

    class FakePopen:
        def __init__(self, *args, **kwargs) -> None:
            self.pid = 12345

    monkeypatch.setattr(offline.subprocess, "Popen", FakePopen)
    args = argparse.Namespace(
        symbols="TRXUSDT",
        target_side="BUY",
        target_interval="1d",
        limit=5000,
        grid_mode="focused",
        min_test_trades=30,
        target_profit_factor=1.0,
        min_expectancy_r=0.0,
        max_stop_loss_ratio=55.0,
        max_configs=40,
        max_walk_forward_validations=6,
        top_n=10,
        latest_sweeps=12,
        step_timeout_seconds=123,
    )

    assert offline.start(args) == 0

    payload = json.loads(capsys.readouterr().out)
    run_dir = state_dir / "20260511T010203Z-trxusdt-buy-1d"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    worker = (run_dir / "worker.py").read_text(encoding="utf-8")
    assert payload["step_timeout_seconds"] == 123
    assert metadata["step_timeout_seconds"] == 123
    assert "timeout=123" in worker
    assert payload["opens_orders"] is False
