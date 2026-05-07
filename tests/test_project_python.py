from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "_project_python.py"
)
SPEC = importlib.util.spec_from_file_location("project_python_helper", MODULE_PATH)
assert SPEC and SPEC.loader
project_python = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(project_python)


def test_ensure_project_python_skips_when_current_python_matches(monkeypatch, tmp_path: Path) -> None:
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(project_python.sys, "executable", str(venv_python))

    called = {"execv": False}

    def fake_execv(path: str, argv: list[str]) -> None:
        called["execv"] = True

    monkeypatch.setattr(project_python.os, "execv", fake_execv)

    project_python.ensure_project_python(tmp_path)

    assert called["execv"] is False


def test_ensure_project_python_reexecs_into_project_venv(monkeypatch, tmp_path: Path) -> None:
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(project_python.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(project_python.sys, "argv", ["scripts/run_strategy_optimizer.py", "--config", "cfg.yaml"])

    called: dict[str, object] = {}

    def fake_execv(path: str, argv: list[str]) -> None:
        called["path"] = path
        called["argv"] = argv

    monkeypatch.setattr(project_python.os, "execv", fake_execv)

    project_python.ensure_project_python(tmp_path)

    assert called["path"] == str(venv_python)
    assert called["argv"] == [str(venv_python), "scripts/run_strategy_optimizer.py", "--config", "cfg.yaml"]
