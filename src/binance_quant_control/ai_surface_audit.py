from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, STATE_DIR, ensure_runtime_dirs

AI_SURFACE_AUDIT_DIR = STATE_DIR / "ai-surface-audit"

DECISION_SURFACES = (
    "src/binance_quant_control/alpha_research.py",
    "src/binance_quant_control/market_bot_gate.py",
    "src/binance_quant_control/hermes_ai_trader.py",
    "src/binance_quant_control/feature_label_gate.py",
    "src/binance_quant_control/ai_expectancy_upgrade.py",
    "src/binance_quant_control/readiness_scanner.py",
    "src/binance_quant_control/live_execution.py",
    "config/market-bot-six-symbol-discovery.default.yaml",
    "config/market-bot-gate.default.yaml",
)

HUMAN_TOKENS = (
    "human",
    "operator_prompt",
    "manual",
    "advisory",
    "narrative",
    "senior-trader",
)

ALLOWED_PATTERNS = (
    "requires_operator_execute",
    "operator_execute_required",
    "operator_execute",
    "operator explicitly executes",
    "manual route risk review cleared",
    "manual_margin_cap",
    "manual cap",
    "manual kill-switch",
    "pending manual review",
    "market-bot gate is accepted; stale or legacy optimizer rejection",
)


@dataclass(frozen=True, slots=True)
class SurfaceHit:
    path: str
    line: int
    token: str
    text: str
    allowed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _is_allowed(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in ALLOWED_PATTERNS)


def run_ai_surface_audit(*, output_dir: str | Path | None = None) -> dict[str, Any]:
    ensure_runtime_dirs()
    root = Path(output_dir).expanduser().resolve() if output_dir else AI_SURFACE_AUDIT_DIR
    root.mkdir(parents=True, exist_ok=True)
    hits: list[SurfaceHit] = []
    for relative in DECISION_SURFACES:
        path = PROJECT_ROOT / relative
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            lowered = line.lower()
            for token in HUMAN_TOKENS:
                if token in lowered:
                    hits.append(
                        SurfaceHit(
                            path=relative,
                            line=line_no,
                            token=token,
                            text=line.strip(),
                            allowed=_is_allowed(line),
                        )
                    )
    blockers = [hit for hit in hits if not hit.allowed]
    payload = {
        "generated_at": _utc_now().isoformat(),
        "mode": "ai_surface_audit",
        "safety": {
            "opens_orders": False,
            "writes_execution_config": False,
            "mainnet_live_allowed": False,
        },
        "principle": "machine decision surfaces must use numeric gates, labels, features, and risk state instead of human narrative inputs",
        "decision_surfaces": list(DECISION_SURFACES),
        "human_influence_tokens": list(HUMAN_TOKENS),
        "status": "passed" if not blockers else "blocked",
        "blocker_count": len(blockers),
        "allowed_count": len(hits) - len(blockers),
        "blockers": [hit.to_dict() for hit in blockers],
        "allowed_hits": [hit.to_dict() for hit in hits if hit.allowed],
    }
    report_path = root / f"{_stamp()}-ai-surface-audit.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload
