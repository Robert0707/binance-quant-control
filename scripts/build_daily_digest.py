#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from _project_python import ensure_project_python  # noqa: E402

ensure_project_python(PROJECT_ROOT)


def main() -> int:
    from binance_quant_control.daily_digest import build_digest, load_config

    parser = argparse.ArgumentParser(description="Build a daily Binance quant digest for n8n.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "n8n-daily-digest.default.json"),
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    payload = build_digest(load_config(config_path))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
