#!/usr/bin/env python3
"""Sync the latest digest candidates into the Freqtrade whitelist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_DIGEST_DIR = Path("/home/robert/python/projects/binance-quant-control/state/n8n-digests")
DEFAULT_CONFIG = Path("/home/robert/python/external/freqtrade/user_data/config.openclaw.json")


def latest_digest(path: Path) -> Path:
    candidates = sorted(path.glob("*-daily-digest.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"No digest files found under {path}")
    return candidates[0]


def futures_pair(symbol: str) -> str:
    if not symbol.endswith("USDT"):
        raise ValueError(f"Unsupported symbol format: {symbol}")
    base = symbol[:-4]
    return f"{base}/USDT:USDT"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync top digest candidates into freqtrade whitelist")
    parser.add_argument("--digest-dir", default=str(DEFAULT_DIGEST_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--top", type=int, default=4)
    args = parser.parse_args()

    digest_path = latest_digest(Path(args.digest_dir))
    payload = json.loads(digest_path.read_text(encoding="utf-8"))
    long_candidates = payload.get("ranked", {}).get("long", [])
    neutral_candidates = payload.get("ranked", {}).get("neutral", [])
    chosen = []
    for item in [*long_candidates, *neutral_candidates]:
        symbol = item.get("symbol")
        if not symbol:
            continue
        pair = futures_pair(str(symbol))
        if pair not in chosen:
            chosen.append(pair)
        if len(chosen) >= args.top:
            break

    if not chosen:
        raise SystemExit(f"No candidates available in {digest_path}")

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.setdefault("exchange", {})
    config["exchange"]["pair_whitelist"] = chosen
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "digest": str(digest_path),
                "config": str(config_path),
                "pair_whitelist": chosen,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
