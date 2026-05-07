from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_project_python(project_root: Path) -> None:
    """Re-exec into the project virtualenv when available."""

    project_python = project_root / ".venv" / "bin" / "python"
    current_python = Path(sys.executable)
    if not project_python.exists() or current_python == project_python:
        return
    os.execv(str(project_python), [str(project_python), *sys.argv])
