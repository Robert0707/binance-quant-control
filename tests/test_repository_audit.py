from __future__ import annotations

from pathlib import Path

import binance_quant_control.repository_audit as audit


def test_repository_audit_skips_secrets_and_generated_by_default(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    (root / "src" / "binance_quant_control").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "state").mkdir()
    (root / "src" / "binance_quant_control" / "sample.py").write_text(
        "import json\n\nclass Demo:\n    pass\n\ndef run():\n    return json.dumps({})\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_sample.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    (root / ".env").write_text("SECRET=value\n", encoding="utf-8")
    (root / "src" / "binance_quant_control" / "__pycache__").mkdir()
    (root / "src" / "binance_quant_control" / "__pycache__" / "sample.pyc").write_bytes(b"cache")
    monkeypatch.setattr(audit, "REPOSITORY_AUDIT_DIR", tmp_path / "audit")
    monkeypatch.setattr(audit, "ensure_runtime_dirs", lambda: None)

    payload = audit.run_repository_audit(root=root, output_dir=tmp_path / "out")

    paths = {item["path"] for item in payload["files"]}
    assert "src/binance_quant_control/sample.py" in paths
    assert "tests/test_sample.py" in paths
    assert ".env" not in paths
    assert "src/binance_quant_control/__pycache__/sample.pyc" not in paths
    assert payload["summary"]["secret_file_count"] == 0
    assert payload["summary"]["generated_file_count"] == 0
    assert Path(payload["report_path"]).exists()


def test_repository_audit_can_include_generated_without_reading_secrets(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    (root / "src" / "binance_quant_control" / "__pycache__").mkdir(parents=True)
    (root / "src" / "binance_quant_control" / "__pycache__" / "sample.pyc").write_bytes(b"cache")
    (root / ".env.bak-20260503").write_text("SECRET=value\n", encoding="utf-8")
    monkeypatch.setattr(audit, "REPOSITORY_AUDIT_DIR", tmp_path / "audit")
    monkeypatch.setattr(audit, "ensure_runtime_dirs", lambda: None)

    payload = audit.run_repository_audit(
        root=root,
        include_generated=True,
        output_dir=tmp_path / "out",
    )

    by_path = {item["path"]: item for item in payload["files"]}
    assert by_path["src/binance_quant_control/__pycache__/sample.pyc"]["generated"] is True
    assert by_path[".env.bak-20260503"]["secret"] is True
    assert by_path[".env.bak-20260503"]["line_count"] is None
