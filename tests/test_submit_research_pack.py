from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "submit_research_pack.py"


def load_submit_research_pack_module():
    spec = importlib.util.spec_from_file_location("submit_research_pack", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_config(path: Path, payload: dict[object, object]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_scheduled_strategy_only_blocks_analysis_profiles(tmp_path, monkeypatch) -> None:
    submit_research_pack = load_submit_research_pack_module()
    config_path = tmp_path / "research-pack.yaml"
    task_spec_dir = tmp_path / "task-specs"
    task_spec_dir.mkdir()
    before = {item.name for item in task_spec_dir.iterdir()}
    _write_config(
        config_path,
        {
            "meta": {"lane_mode": "strategy_review_only"},
            "guardrails": {
                "max_scheduled_limit": 240,
                "reject_run_now": True,
                "reject_analysis_profiles": True,
            },
            "scheduled_review": {"command": "review_closed_trades", "limit": 20, "compact": True},
            "defaults": {"limit": 500},
            "profiles": [{"name": "btcusdt-core-1h", "symbol": "BTCUSDT", "interval": "1h", "enabled": True}],
        },
    )
    monkeypatch.setattr(submit_research_pack, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(
        "sys.argv",
        ["submit_research_pack.py", "--scheduled", "--config", str(config_path)],
    )

    assert submit_research_pack.main() == 0
    output_files = sorted((tmp_path / "state").glob("*-submission.json"))
    assert output_files
    payload = json.loads(output_files[-1].read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert any("defaults.limit=500" in reason for reason in payload["blocked_reasons"])
    assert any("raw market-analysis profiles" in reason for reason in payload["blocked_reasons"])
    after = {item.name for item in task_spec_dir.iterdir()}
    assert after == before


def test_scheduled_strategy_only_runs_closed_trade_review(tmp_path, monkeypatch) -> None:
    submit_research_pack = load_submit_research_pack_module()
    config_path = tmp_path / "research-pack.yaml"
    _write_config(
        config_path,
        {
            "meta": {"lane_mode": "strategy_review_only"},
            "guardrails": {
                "max_scheduled_limit": 240,
                "reject_run_now": True,
                "reject_analysis_profiles": True,
            },
            "scheduled_review": {"command": "review_closed_trades", "limit": 7, "compact": True},
            "defaults": {"limit": 240},
            "profiles": [],
        },
    )
    monkeypatch.setattr(submit_research_pack, "STATE_ROOT", tmp_path / "state")

    captured: dict[str, object] = {}

    def fake_run_command(command: list[str]) -> dict[str, object]:
        captured["command"] = command
        return {
            "command": command,
            "returncode": 0,
            "stdout": '{"status":"ok","new_review_count":0}',
            "stderr": "",
            "response": {"status": "ok", "new_review_count": 0},
        }

    monkeypatch.setattr(submit_research_pack, "run_command", fake_run_command)
    monkeypatch.setattr(
        "sys.argv",
        ["submit_research_pack.py", "--scheduled", "--config", str(config_path)],
    )

    assert submit_research_pack.main() == 0
    assert captured["command"] == [
        str(submit_research_pack.QUANTCTL),
        "review-closed-trades",
        "--limit",
        "7",
        "--compact",
    ]
    output_files = sorted((tmp_path / "state").glob("*-submission.json"))
    assert output_files
    payload = json.loads(output_files[-1].read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["results"][0]["name"] == "scheduled-review"
