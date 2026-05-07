from __future__ import annotations

import ast
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, STATE_DIR, ensure_runtime_dirs

REPOSITORY_AUDIT_DIR = STATE_DIR / "repository-audit"

SKIP_DIRS = {
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "reports",
    "state",
}
SECRET_NAMES = {".env"}
SECRET_PREFIXES = (".env.bak",)
GENERATED_PARTS = {"__pycache__"}
GENERATED_SUFFIXES = (".pyc", ".pyo")
GENERATED_NAMES = {"hailort.log"}
GENERATED_PATH_PARTS = ("binance_quant_control.egg-info",)
TEXT_SUFFIXES = {
    ".cfg",
    ".example",
    ".ini",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True, slots=True)
class FileAudit:
    path: str
    category: str
    size_bytes: int
    line_count: int | None
    generated: bool
    secret: bool
    backup: bool
    functions: int | None = None
    classes: int | None = None
    imports: int | None = None
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "category": self.category,
            "size_bytes": self.size_bytes,
            "line_count": self.line_count,
            "generated": self.generated,
            "secret": self.secret,
            "backup": self.backup,
            "functions": self.functions,
            "classes": self.classes,
            "imports": self.imports,
            "parse_error": self.parse_error,
        }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _is_secret(path: Path) -> bool:
    return path.name in SECRET_NAMES or path.name.startswith(SECRET_PREFIXES)


def _is_generated(path: Path) -> bool:
    parts = set(path.parts)
    return (
        bool(parts & GENERATED_PARTS)
        or path.suffix in GENERATED_SUFFIXES
        or path.name in GENERATED_NAMES
        or any(part in path.parts for part in GENERATED_PATH_PARTS)
    )


def _is_backup(path: Path) -> bool:
    return ".bak-" in path.name


def _category(path: Path) -> str:
    if _is_secret(path):
        return "secret-local"
    if _is_generated(path):
        return "generated"
    if _is_backup(path):
        return "local-backup"
    first = path.parts[0] if path.parts else ""
    if first == "src":
        return "source"
    if first == "tests":
        return "tests"
    if first == "config":
        if path.name.startswith("strategy-"):
            return "config-strategy"
        return "config"
    if first == "docs":
        return "docs"
    if first == "scripts":
        return "scripts"
    if first == "examples":
        return "examples"
    if path.name in {
        ".env.example",
        ".gitignore",
        ".pre-commit-config.yaml",
        "PROJECT.md",
        "pyproject.toml",
        "uv.lock",
    }:
        return "project-root"
    return "other"


def _should_skip_dir(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def _text_line_count(path: Path) -> tuple[int | None, str | None]:
    if path.suffix not in TEXT_SUFFIXES and path.name not in {"PROJECT.md", ".env.example", ".gitignore"}:
        return None, None
    try:
        return len(path.read_text(encoding="utf-8").splitlines()), None
    except UnicodeDecodeError:
        return None, "non-utf8-text"


def _python_shape(path: Path) -> tuple[int | None, int | None, int | None, str | None]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, None, None, str(exc)
    functions = 0
    classes = 0
    imports = 0
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions += 1
        elif isinstance(node, ast.ClassDef):
            classes += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports += 1
    return functions, classes, imports, None


def _audit_file(root: Path, path: Path) -> FileAudit:
    rel = path.relative_to(root)
    category = _category(rel)
    secret = _is_secret(rel)
    generated = _is_generated(rel)
    backup = _is_backup(rel)
    stat = path.stat()
    line_count: int | None = None
    parse_error: str | None = None
    functions: int | None = None
    classes: int | None = None
    imports: int | None = None
    if not secret and not generated:
        line_count, parse_error = _text_line_count(path)
        if path.suffix == ".py":
            functions, classes, imports, python_error = _python_shape(path)
            parse_error = parse_error or python_error
    return FileAudit(
        path=str(rel),
        category=category,
        size_bytes=stat.st_size,
        line_count=line_count,
        generated=generated,
        secret=secret,
        backup=backup,
        functions=functions,
        classes=classes,
        imports=imports,
        parse_error=parse_error,
    )


def _walk_files(root: Path, *, include_generated: bool) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.is_dir():
            continue
        if _should_skip_dir(rel):
            continue
        if not include_generated and (_is_generated(rel) or _is_secret(rel)):
            continue
        files.append(path)
    return files


def _architecture_findings(rows: list[FileAudit]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    source_rows = [row for row in rows if row.category == "source"]
    for row in sorted(source_rows, key=lambda item: item.line_count or 0, reverse=True):
        lines = row.line_count or 0
        if lines >= 1200:
            findings.append(
                {
                    "severity": "medium",
                    "code": "oversized-source-module",
                    "path": row.path,
                    "line_count": lines,
                    "recommendation": "Split by command surface, domain service, or report writer before adding more features.",
                }
            )
        elif lines >= 800:
            findings.append(
                {
                    "severity": "low",
                    "code": "large-source-module",
                    "path": row.path,
                    "line_count": lines,
                    "recommendation": "Keep new logic in focused modules and avoid growing this file further.",
                }
            )
    backups = [row for row in rows if row.backup]
    if backups:
        findings.append(
            {
                "severity": "low",
                "code": "local-backups-in-config-tree",
                "count": len(backups),
                "recommendation": "Keep backups out of active config listings; do not read or expose .env backups.",
            }
        )
    generated = [row for row in rows if row.generated]
    if generated:
        findings.append(
            {
                "severity": "low",
                "code": "generated-artifacts-in-project-tree",
                "count": len(generated),
                "recommendation": "Ignore __pycache__, egg-info, logs, reports, and state from architecture review.",
            }
        )
    return findings


def run_repository_audit(
    *,
    root: str | Path | None = None,
    include_generated: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    project_root = Path(root).expanduser().resolve() if root else PROJECT_ROOT
    files = _walk_files(project_root, include_generated=include_generated)
    rows = [_audit_file(project_root, path) for path in files]
    category_counts = Counter(row.category for row in rows)
    large_files = sorted(
        [row for row in rows if not row.secret],
        key=lambda item: item.line_count if item.line_count is not None else -1,
        reverse=True,
    )[:20]
    generated_count = sum(1 for row in rows if row.generated)
    secret_count = sum(1 for row in rows if row.secret)
    payload = {
        "generated_at": _utc_now().isoformat(),
        "mode": "repository_audit",
        "root": str(project_root),
        "safety": {
            "reads_secrets": False,
            "include_generated": include_generated,
            "skipped_dirs": sorted(SKIP_DIRS),
        },
        "summary": {
            "audited_file_count": len(rows),
            "category_counts": dict(sorted(category_counts.items())),
            "generated_file_count": generated_count,
            "secret_file_count": secret_count,
            "backup_file_count": sum(1 for row in rows if row.backup),
            "source_line_count": sum(row.line_count or 0 for row in rows if row.category == "source"),
            "test_line_count": sum(row.line_count or 0 for row in rows if row.category == "tests"),
        },
        "largest_files": [row.to_dict() for row in large_files],
        "config_backups": [row.path for row in rows if row.backup],
        "generated_artifacts": [row.path for row in rows if row.generated][:200],
        "architecture_findings": _architecture_findings(rows),
        "files": [row.to_dict() for row in rows],
    }
    root_dir = Path(output_dir).expanduser().resolve() if output_dir else REPOSITORY_AUDIT_DIR
    root_dir.mkdir(parents=True, exist_ok=True)
    report_path = root_dir / f"{_stamp()}-repository-audit.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload
