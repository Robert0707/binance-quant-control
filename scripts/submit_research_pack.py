#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "research-pack.default.yaml"
STATE_ROOT = PROJECT_ROOT / "state" / "scheduled-research"
QUANTCTL = Path("/home/robert/.openclaw/bin/openclaw-quantctl")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Config {path} must be a mapping.")
    return payload


def build_review_command(config: dict[str, Any]) -> list[str]:
    review = config.get("scheduled_review") or {}
    limit = int(review.get("limit", 20))
    command = [str(QUANTCTL), "review-closed-trades", "--limit", str(limit)]
    if bool(review.get("compact", True)):
        command.append("--compact")
    return command


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    result: dict[str, Any] = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": (completed.stdout or "").strip(),
        "stderr": (completed.stderr or "").strip(),
    }
    if result["stdout"]:
        try:
            result["response"] = json.loads(result["stdout"])
        except json.JSONDecodeError:
            result["response"] = None
    return result


def scheduled_policy_decision(config: dict[str, Any], *, run_now: bool) -> tuple[str, list[str]]:
    meta = config.get("meta") or {}
    guardrails = config.get("guardrails") or {}
    defaults = config.get("defaults") or {}
    profiles = config.get("profiles") or []
    scheduled_review = config.get("scheduled_review") or {}
    reasons: list[str] = []

    lane_mode = str(meta.get("lane_mode", "strategy_review_only"))
    if lane_mode != "strategy_review_only":
        reasons.append(f"scheduled policy requires lane_mode=strategy_review_only (got {lane_mode})")

    if run_now and bool(guardrails.get("reject_run_now", True)):
        reasons.append("scheduled strategy-only policy blocks --run-now")

    max_scheduled_limit = int(guardrails.get("max_scheduled_limit", 240))
    default_limit = int(defaults.get("limit", 0) or 0)
    if profiles and default_limit and default_limit > max_scheduled_limit:
        reasons.append(
            f"scheduled strategy-only policy blocks defaults.limit={default_limit} above cap {max_scheduled_limit}"
        )

    review_limit = int(scheduled_review.get("limit", 20) or 20)
    if review_limit > max_scheduled_limit:
        reasons.append(
            f"scheduled strategy-only policy blocks scheduled_review.limit={review_limit} above cap {max_scheduled_limit}"
        )

    if bool(guardrails.get("reject_analysis_profiles", True)) and profiles:
        reasons.append("scheduled strategy-only policy blocks raw market-analysis profiles")

    command = str(scheduled_review.get("command", "review_closed_trades"))
    if command != "review_closed_trades":
        reasons.append(
            f"scheduled strategy-only policy only allows review_closed_trades (got {command})"
        )

    return ("blocked", reasons) if reasons else ("allowed", [])


def write_payload(config_path: Path, config: dict[str, Any], payload: dict[str, Any]) -> Path:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = STATE_ROOT / f"{now_stamp()}-submission.json"
    document = {
        "generated_at": now_iso(),
        "config_path": str(config_path),
        "meta": config.get("meta") or {},
        **payload,
    }
    output_path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    document["output_path"] = str(output_path)
    print(json.dumps(document, indent=2, ensure_ascii=False))
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the scheduled strategy-review lane or block unsafe scheduled research."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-now", action="store_true", help="Rejected in scheduled strategy-only mode.")
    parser.add_argument("--dry-run", action="store_true", help="Print the review command without executing it.")
    parser.add_argument(
        "--scheduled",
        action="store_true",
        help="Apply scheduled policy guardrails and run only the closed-trade review lane.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)

    if args.scheduled:
        status, reasons = scheduled_policy_decision(config, run_now=args.run_now)
        review_command = build_review_command(config)
        if status == "blocked":
            write_payload(
                config_path,
                config,
                {
                    "status": "blocked",
                    "scheduled": True,
                    "dry_run": args.dry_run,
                    "run_now": args.run_now,
                    "blocked_reasons": reasons,
                    "results": [],
                },
            )
            return 0

        if args.dry_run:
            write_payload(
                config_path,
                config,
                {
                    "status": "dry-run",
                    "scheduled": True,
                    "dry_run": True,
                    "run_now": args.run_now,
                    "blocked_reasons": [],
                    "results": [{"name": "scheduled-review", "status": "dry-run", "command": review_command}],
                },
            )
            return 0

        result = run_command(review_command)
        result["name"] = "scheduled-review"
        overall_status = "ok" if result["returncode"] == 0 else "error"
        write_payload(
            config_path,
            config,
            {
                "status": overall_status,
                "scheduled": True,
                "dry_run": False,
                "run_now": args.run_now,
                "blocked_reasons": [],
                "results": [result],
            },
        )
        return 0 if result["returncode"] == 0 else 1

    write_payload(
        config_path,
        config,
        {
            "status": "blocked",
            "scheduled": False,
            "dry_run": args.dry_run,
            "run_now": args.run_now,
            "blocked_reasons": [
                "submit_research_pack.py is reserved for scheduled strategy review only",
                "manual market analysis must use openclaw-quantctl analyze or submit-analysis directly",
            ],
            "results": [],
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
