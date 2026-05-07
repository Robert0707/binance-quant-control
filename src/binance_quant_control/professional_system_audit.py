from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import CONFIG_DIR, PROJECT_ROOT, STATE_DIR, ensure_runtime_dirs
from .order_journal import summarize_closed_trade_reviews, summarize_live_orders

PROFESSIONAL_SYSTEM_AUDIT_DIR = STATE_DIR / "professional-system-audit"
DEFAULT_BLUEPRINT_PATH = CONFIG_DIR / "professional-system-blueprint.default.yaml"

BLOCKING_STATUSES = {"missing", "blocked"}


@dataclass(frozen=True, slots=True)
class ProfessionalAuditLayer:
    layer_id: str
    name: str
    status: str
    critical: bool
    trade_required: bool
    model_references: list[str]
    evidence_paths: list[str]
    missing_paths: list[str]
    gaps: list[str]
    keep: list[str]
    refactor: list[str]
    rebuild: list[str]
    next_action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0] == CONFIG_DIR.name:
        return (PROJECT_ROOT / candidate).resolve()
    if candidate.parts and candidate.parts[0] == STATE_DIR.name:
        return (PROJECT_ROOT / candidate).resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    candidate = _resolve_project_path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"professional system blueprint not found: {candidate}")
    return yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}


def _path_label(path: str | Path) -> str:
    resolved = _resolve_project_path(path)
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _existing_paths(paths: list[Any]) -> list[str]:
    return [_path_label(path) for path in paths if _resolve_project_path(str(path)).exists()]


def _missing_paths(paths: list[Any]) -> list[str]:
    return [_path_label(path) for path in paths if not _resolve_project_path(str(path)).exists()]


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def _layer_status(raw: dict[str, Any], missing_required: list[str]) -> str:
    if missing_required:
        return "missing"
    status = str(raw.get("status_if_present") or "ready").strip().lower()
    if status not in {"ready", "partial", "missing", "blocked"}:
        return "ready"
    return status


def _audit_layer(raw: dict[str, Any]) -> ProfessionalAuditLayer:
    required_paths = _coerce_list(raw.get("required_paths"))
    optional_paths = _coerce_list(raw.get("optional_paths"))
    missing_required = _missing_paths(required_paths)
    existing = _existing_paths(required_paths + optional_paths)
    status = _layer_status(raw, missing_required)
    gaps = _coerce_list(raw.get("known_gaps"))
    if missing_required:
        gaps = [*gaps, "required implementation paths are missing"]
    return ProfessionalAuditLayer(
        layer_id=str(raw.get("id") or "unnamed"),
        name=str(raw.get("name") or raw.get("id") or "Unnamed Layer"),
        status=status,
        critical=bool(raw.get("critical", False)),
        trade_required=bool(raw.get("trade_required", False)),
        model_references=_coerce_list(raw.get("model_references")),
        evidence_paths=existing,
        missing_paths=missing_required,
        gaps=gaps,
        keep=_coerce_list(raw.get("keep")),
        refactor=_coerce_list(raw.get("refactor")),
        rebuild=_coerce_list(raw.get("rebuild")),
        next_action=str(raw.get("next_action") or ""),
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _latest_file(root: Path, pattern: str) -> Path | None:
    if not root.exists():
        return None
    paths = sorted(root.glob(pattern))
    return paths[-1] if paths else None


def _latest_file_recursive(root: Path, pattern: str) -> Path | None:
    if not root.exists():
        return None
    paths = sorted(root.glob(pattern))
    return paths[-1] if paths else None


def _slim_alpha_summary(report: dict[str, Any] | None, path: Path | None) -> dict[str, Any]:
    if not report:
        return {
            "path": str(path) if path else None,
            "available": False,
        }
    summary = report.get("performance_summary") or {}
    return {
        "path": str(path) if path else None,
        "available": True,
        "row_count": summary.get("row_count"),
        "trade_count": summary.get("trade_count"),
        "promotion_eligible_count": summary.get("promotion_eligible_count"),
        "weighted_win_rate": summary.get("weighted_win_rate"),
        "weighted_stop_loss_ratio": summary.get("weighted_stop_loss_ratio"),
        "finite_avg_profit_factor": summary.get("finite_avg_profit_factor"),
        "weighted_expectancy_r": summary.get("weighted_expectancy_r"),
        "weighted_payoff_ratio": summary.get("weighted_payoff_ratio"),
        "execution_recommendation": report.get("execution_recommendation"),
    }


def _slim_market_bot_gate_summary(report: dict[str, Any] | None, path: Path | None) -> dict[str, Any]:
    if not report:
        return {
            "path": str(path) if path else None,
            "available": False,
            "safe_to_open_new_entries": False,
            "blockers": ["market-bot-gate-report-missing"],
            "accepted": [],
        }
    portfolio_gate = report.get("portfolio_gate") or {}
    accepted = [row for row in (report.get("accepted") or []) if isinstance(row, dict)]
    return {
        "path": str(path) if path else None,
        "available": True,
        "mode": report.get("mode"),
        "safe_to_open_new_entries": bool(report.get("safe_to_open_new_entries")),
        "execution_recommendation": report.get("execution_recommendation"),
        "accepted_count": int(report.get("accepted_count") or len(accepted)),
        "accepted_symbols": sorted({str(row.get("symbol")) for row in accepted if row.get("symbol")}),
        "targets": report.get("targets") or {},
        "portfolio_gate": {
            "enabled": bool(portfolio_gate.get("enabled")),
            "passed": bool(portfolio_gate.get("passed", True)),
            "min_accepted_symbols": portfolio_gate.get("min_accepted_symbols"),
            "accepted_symbol_count": portfolio_gate.get("accepted_symbol_count"),
            "missing_required_symbols": portfolio_gate.get("missing_required_symbols") or [],
            "blockers": portfolio_gate.get("blockers") or [],
        },
        "feature_manifest_hash": report.get("feature_manifest_hash"),
        "blockers": portfolio_gate.get("blockers") or [],
        "best": report.get("best") or {},
        "accepted": accepted[:20],
    }


def _slim_iteration_summary(report: dict[str, Any] | None, path: Path | None) -> dict[str, Any]:
    if not report:
        return {
            "path": str(path) if path else None,
            "available": False,
            "safe_to_open_new_entries": False,
            "blockers": ["high-win-iteration-report-missing"],
        }
    best_gate = report.get("best_alpha_gate") or {}
    portfolio_gate = report.get("portfolio_gate") or {}
    blockers = list(best_gate.get("blockers") or [])
    blockers.extend(portfolio_gate.get("blockers") or [])
    return {
        "path": str(path) if path else None,
        "available": True,
        "mode": report.get("mode"),
        "promotion_allowed": bool(report.get("promotion_allowed")),
        "safe_to_open_new_entries": bool(report.get("safe_to_open_new_entries")),
        "execution_recommendation": report.get("execution_recommendation"),
        "best_alpha_gate": {
            "passed": bool(best_gate.get("passed")),
            "blockers": best_gate.get("blockers") or [],
            "targets": best_gate.get("targets") or {},
        },
        "portfolio_gate": {
            "enabled": bool(portfolio_gate.get("enabled")),
            "passed": bool(portfolio_gate.get("passed")),
            "promoted_symbol_count": portfolio_gate.get("promoted_symbol_count"),
            "missing_required_symbols": portfolio_gate.get("missing_required_symbols") or [],
            "blockers": portfolio_gate.get("blockers") or [],
        },
        "blockers": blockers,
    }


def _configured_alpha_report_path(blueprint: dict[str, Any]) -> Path | None:
    evidence = blueprint.get("evidence") or {}
    path = evidence.get("latest_alpha_report")
    if path:
        candidate = _resolve_project_path(str(path))
        if candidate.exists():
            return candidate
    return None


def _configured_market_bot_gate_path(blueprint: dict[str, Any]) -> Path | None:
    evidence = blueprint.get("evidence") or {}
    path = evidence.get("latest_market_bot_gate")
    if path:
        candidate = _resolve_project_path(str(path))
        if candidate.exists():
            return candidate
    return None


def _latest_alpha_report(blueprint: dict[str, Any]) -> Path | None:
    configured = _configured_alpha_report_path(blueprint)
    if configured:
        return configured
    return _latest_file(STATE_DIR, "*/alpha-research-ranking.json")


def _latest_market_bot_gate(blueprint: dict[str, Any]) -> Path | None:
    configured = _configured_market_bot_gate_path(blueprint)
    if configured:
        return configured
    return _latest_file_recursive(STATE_DIR, "*/*-market-bot-gate.json")


def _build_evidence(blueprint: dict[str, Any]) -> dict[str, Any]:
    alpha_path = _latest_alpha_report(blueprint)
    market_bot_gate_path = _latest_market_bot_gate(blueprint)
    iteration_path = _latest_file(STATE_DIR / "high-win-iteration", "*-high-win-iteration.json")
    return {
        "alpha_report": _slim_alpha_summary(
            _read_json(alpha_path) if alpha_path else None,
            alpha_path,
        ),
        "market_bot_gate": _slim_market_bot_gate_summary(
            _read_json(market_bot_gate_path) if market_bot_gate_path else None,
            market_bot_gate_path,
        ),
        "high_win_iteration": _slim_iteration_summary(
            _read_json(iteration_path) if iteration_path else None,
            iteration_path,
        ),
        "closed_trade_reviews": summarize_closed_trade_reviews(),
        "live_orders": summarize_live_orders(),
    }


def _critical_blockers(
    layers: list[ProfessionalAuditLayer],
    evidence: dict[str, Any],
    blueprint: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    promotion_gate = blueprint.get("promotion_gate") or {}
    block_partial_layers = bool(promotion_gate.get("partial_layers_block_trade", False))
    for layer in layers:
        if layer.trade_required and layer.status in BLOCKING_STATUSES:
            blockers.append(f"{layer.layer_id}:{layer.status}")
        elif layer.trade_required and layer.status == "partial" and block_partial_layers:
            blockers.append(f"{layer.layer_id}:partial")
    market_bot_gate = evidence.get("market_bot_gate") or {}
    if market_bot_gate.get("available"):
        if not market_bot_gate.get("safe_to_open_new_entries"):
            blockers.append("market-bot-gate:safe-to-open-new-entries-false")
        portfolio_gate = market_bot_gate.get("portfolio_gate") or {}
        if portfolio_gate.get("enabled") and not portfolio_gate.get("passed"):
            blockers.append("market-bot-gate:portfolio-gate-failed")
        min_symbols = int(
            promotion_gate.get("min_accepted_symbols")
            or promotion_gate.get("min_promoted_symbols")
            or (market_bot_gate.get("targets") or {}).get("min_accepted_symbols")
            or 0
        )
        if min_symbols > 0 and int(market_bot_gate.get("accepted_count") or 0) < min_symbols:
            blockers.append("market-bot-gate:accepted-symbol-count-below-floor")
        if not market_bot_gate.get("feature_manifest_hash"):
            blockers.append("market-bot-gate:feature-manifest-hash-missing")
        return blockers
    iteration = evidence.get("high_win_iteration") or {}
    if not iteration.get("safe_to_open_new_entries"):
        blockers.append("promotion-gate:safe-to-open-new-entries-false")
    alpha = evidence.get("alpha_report") or {}
    if alpha.get("available") and int(alpha.get("promotion_eligible_count") or 0) <= 0:
        blockers.append("alpha-evidence:no-promotion-eligible-cohort")
    return blockers


def _recommendations(layers: list[ProfessionalAuditLayer], blockers: list[str]) -> dict[str, Any]:
    preserve: list[str] = []
    refactor: list[str] = []
    rebuild: list[str] = []
    for layer in layers:
        if layer.status == "ready":
            preserve.extend(layer.keep or [layer.name])
        elif layer.status == "partial":
            refactor.extend(layer.refactor or [layer.name])
        else:
            rebuild.extend(layer.rebuild or [layer.name])
    return {
        "preserve": preserve,
        "refactor": refactor,
        "rebuild": rebuild,
        "first_order_fix": "repair-alpha-payoff-and-sample-evidence" if blockers else "paper-forward-test",
    }


def run_professional_system_audit(
    *,
    config_path: str | Path = DEFAULT_BLUEPRINT_PATH,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    blueprint = _load_yaml(config_path)
    layer_rows = blueprint.get("layers") or []
    layers = [_audit_layer(row) for row in layer_rows if isinstance(row, dict)]
    evidence = _build_evidence(blueprint)
    blockers = _critical_blockers(layers, evidence, blueprint)
    layer_counts = Counter(layer.status for layer in layers)
    architecture_warnings = [
        f"{layer.layer_id}:partial"
        for layer in layers
        if layer.trade_required and layer.status == "partial"
    ]
    trade_ready = not blockers
    payload = {
        "generated_at": _utc_now().isoformat(),
        "mode": "professional_system_audit",
        "safety": {
            "opens_orders": False,
            "writes_execution_config": False,
            "mainnet_live_allowed": False,
            "research_gate_only": True,
        },
        "trade_ready": trade_ready,
        "execution_recommendation": (
            "paper_or_testnet_readiness_review" if trade_ready else "block_new_entries_and_rebuild_edge"
        ),
        "blueprint": {
            "path": str(_resolve_project_path(config_path)),
            "status": (blueprint.get("meta") or {}).get("status"),
            "references": (blueprint.get("meta") or {}).get("references") or {},
            "promotion_gate": blueprint.get("promotion_gate") or {},
        },
        "layer_summary": {
            "total": len(layers),
            "counts": dict(sorted(layer_counts.items())),
            "trade_required": [layer.layer_id for layer in layers if layer.trade_required],
        },
        "layers": [layer.to_dict() for layer in layers],
        "evidence": evidence,
        "critical_blockers": blockers,
        "architecture_warnings": architecture_warnings,
        "recommendations": _recommendations(layers, blockers),
        "target_workflow": blueprint.get("target_workflow") or [],
    }
    root_dir = Path(output_dir).expanduser().resolve() if output_dir else PROFESSIONAL_SYSTEM_AUDIT_DIR
    root_dir.mkdir(parents=True, exist_ok=True)
    report_path = root_dir / f"{_stamp()}-professional-system-audit.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    payload["report_path"] = str(report_path)
    return payload
