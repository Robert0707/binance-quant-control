from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import STATE_DIR, ensure_runtime_dirs

TRADE_DECISION_DIR = STATE_DIR / "trade-decisions"
DECISION_AUDIT_DIR = STATE_DIR / "decision-audit"
ALLOWED_DECISIONS = {"BUY", "LONG", "SELL", "SHORT", "HOLD", "EXIT"}
ENTRY_DECISIONS = {"BUY", "LONG", "SELL", "SHORT"}
EXIT_DECISIONS = {"EXIT"}
REQUIRED_ENTRY_GATE_NAMES = {
    "regime_policy",
    "trend_direction",
    "momentum",
    "volume",
    "volatility",
    "support_resistance",
    "risk_reward",
    "stop_distance",
}
REQUIRED_LIFECYCLE_GATE_NAMES = {
    "data_check",
    "backtest_or_performance_evidence",
    "trading_cost_check",
    "max_drawdown_check",
    "dry_run_check",
    "testnet_forward_check",
}
REQUIRED_DECISION_FIELDS = (
    "decision",
    "direction",
    "confidence",
    "regime",
    "entry_reason",
    "blocked_reasons",
    "invalid_if",
    "entry",
    "stop_loss",
    "take_profit",
    "risk_pct",
    "risk_reward_ratio",
    "expected_value",
    "position_size",
    "hailo_task_allocation",
)
REQUIRED_ANSWER_FIELDS = (
    "why_long_now",
    "why_short_now",
    "why_no_trade_now",
    "where_stop_if_wrong",
    "where_take_profit_if_right",
    "max_loss",
    "long_term_expected_value",
)


@dataclass(frozen=True, slots=True)
class DecisionArtifactAudit:
    path: str
    decision: str
    direction: str
    valid: bool
    failed_checks: list[str]
    violations: list[str]
    risk_pct: float | None
    expected_value_positive: bool | None
    opens_orders: bool | None
    writes_execution_config: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "decision": self.decision,
            "direction": self.direction,
            "valid": self.valid,
            "failed_checks": self.failed_checks,
            "violations": self.violations,
            "risk_pct": self.risk_pct,
            "expected_value_positive": self.expected_value_positive,
            "opens_orders": self.opens_orders,
            "writes_execution_config": self.writes_execution_config,
        }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _valid_direction(decision: str, direction: str) -> bool:
    if decision in {"BUY", "LONG"}:
        return direction == "bullish"
    if decision in {"SELL", "SHORT"}:
        return direction == "bearish"
    if decision in {"HOLD", "EXIT"}:
        return direction in {"bullish", "bearish", "neutral"}
    return False


def _valid_entry_structure(
    *,
    decision: str,
    entry: float | None,
    stop_loss: float | None,
    take_profit: list[Any],
) -> bool:
    if decision not in ENTRY_DECISIONS:
        return True
    tp = [_float_or_none(level) for level in take_profit]
    tp = [level for level in tp if level is not None]
    if entry is None or stop_loss is None or not tp:
        return False
    if decision in {"BUY", "LONG"}:
        return stop_loss < entry and all(level > entry for level in tp)
    if decision in {"SELL", "SHORT"}:
        return stop_loss > entry and all(level < entry for level in tp)
    return False


def _append_shape_violations(payload: dict[str, Any], violations: list[str]) -> None:
    confidence = _float_or_none(payload.get("confidence"))
    risk_pct = _float_or_none(payload.get("risk_pct"))
    rr = _float_or_none(payload.get("risk_reward_ratio"))
    decision = str(payload.get("decision") or "")
    if confidence is None or not 0.0 <= confidence <= 100.0:
        violations.append("confidence-out-of-range")
    if risk_pct is None or risk_pct < 0.0:
        violations.append("risk-pct-invalid")
    if decision in ENTRY_DECISIONS and (rr is None or rr <= 0.0):
        violations.append("entry-risk-reward-invalid")
    if not isinstance(payload.get("entry_reason"), list):
        violations.append("entry-reason-not-list")
    if not isinstance(payload.get("blocked_reasons"), list):
        violations.append("blocked-reasons-not-list")
    if not isinstance(payload.get("take_profit"), list):
        violations.append("take-profit-not-list")
    if not isinstance(payload.get("invalid_if"), dict):
        violations.append("invalid-if-not-object")
    if not isinstance(payload.get("expected_value"), dict):
        violations.append("expected-value-not-object")
    if not isinstance(payload.get("position_size"), dict):
        violations.append("position-size-not-object")
    if not isinstance(payload.get("hard_gates"), dict):
        violations.append("hard-gates-not-object")


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"invalid-json:{exc}"
    if not isinstance(raw, dict):
        return None, "json-root-not-object"
    return raw, None


def _gate_names(evidence: Any) -> set[str]:
    if not isinstance(evidence, dict):
        return set()
    gates = evidence.get("gates")
    if not isinstance(gates, list):
        return set()
    return {str(item.get("name")) for item in gates if isinstance(item, dict) and str(item.get("name"))}


def _required_gate_failures(evidence: Any, required_names: set[str]) -> list[str]:
    if not isinstance(evidence, dict):
        return sorted(required_names)
    gates = evidence.get("gates")
    if not isinstance(gates, list):
        return sorted(required_names)
    passed_by_name = {
        str(item.get("name")): item.get("passed") is True
        for item in gates
        if isinstance(item, dict) and str(item.get("name"))
    }
    return sorted(name for name in required_names if name in passed_by_name and not passed_by_name[name])


def _failed_gate_list(evidence: Any) -> list[str]:
    if not isinstance(evidence, dict):
        return []
    failed = evidence.get("failed")
    if not isinstance(failed, list):
        return []
    return [str(item) for item in failed if str(item)]


def _audit_payload(path: Path, payload: dict[str, Any], *, root: Path) -> DecisionArtifactAudit:
    decision = str(payload.get("decision") or "")
    direction = str(payload.get("direction") or "")
    hard_gates = payload.get("hard_gates") if isinstance(payload.get("hard_gates"), dict) else {}
    entry_evidence = payload.get("entry_gate_evidence") if isinstance(payload.get("entry_gate_evidence"), dict) else {}
    lifecycle_evidence = (
        payload.get("lifecycle_gate_evidence")
        if isinstance(payload.get("lifecycle_gate_evidence"), dict)
        else {}
    )
    expected = payload.get("expected_value") if isinstance(payload.get("expected_value"), dict) else {}
    position_size = payload.get("position_size") if isinstance(payload.get("position_size"), dict) else {}
    validation = (
        payload.get("decision_contract_validation")
        if isinstance(payload.get("decision_contract_validation"), dict)
        else {}
    )
    risk_pct = _float_or_none(payload.get("risk_pct"))
    entry = _float_or_none(payload.get("entry"))
    stop_loss = _float_or_none(payload.get("stop_loss"))
    take_profit = payload.get("take_profit") if isinstance(payload.get("take_profit"), list) else []
    opens_orders = _bool_or_none(payload.get("opens_orders"))
    writes_execution_config = _bool_or_none(payload.get("writes_execution_config"))
    failed_checks = [str(item) for item in (validation.get("failed") or []) if str(item)]
    violations: list[str] = []
    missing_fields = [field for field in REQUIRED_DECISION_FIELDS if field not in payload]
    if missing_fields:
        violations.append(f"missing-required-fields:{','.join(missing_fields)}")
    _append_shape_violations(payload, violations)
    answers = payload.get("decision_answers") if isinstance(payload.get("decision_answers"), dict) else {}
    missing_answers = [field for field in REQUIRED_ANSWER_FIELDS if field not in answers]
    if missing_answers:
        violations.append(f"missing-decision-answer-fields:{','.join(missing_answers)}")
    hailo_plan = payload.get("hailo_task_allocation") if isinstance(payload.get("hailo_task_allocation"), dict) else {}
    hailo_tasks = hailo_plan.get("tasks") if isinstance(hailo_plan.get("tasks"), list) else []
    hailo_by_name = {str(item.get("name")): item for item in hailo_tasks if isinstance(item, dict)}
    order_task = hailo_by_name.get("order-execution-decision")
    if not hailo_plan:
        violations.append("missing-hailo-task-allocation")
    elif order_task is None:
        violations.append("missing-hailo-order-execution-boundary")
    elif order_task.get("status") != "not_allowed":
        violations.append("hailo-order-execution-not-forbidden")
    if decision not in ALLOWED_DECISIONS:
        violations.append(f"decision-not-allowed:{decision or 'missing'}")
    if not _valid_direction(decision, direction):
        violations.append("decision-direction-mismatch")
    if not _valid_entry_structure(
        decision=decision,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
    ):
        violations.append("entry-stop-take-profit-orientation-invalid")
    if decision in ENTRY_DECISIONS and risk_pct is not None and risk_pct > 0.025:
        violations.append("entry-risk-exceeds-2.5pct")
    if decision in ENTRY_DECISIONS and hard_gates.get("risk_ceiling_passed") is not True:
        violations.append("entry-without-risk-ceiling-gate")
    if decision in ENTRY_DECISIONS and hard_gates.get("readiness_allowed") is not True:
        violations.append("entry-without-readiness-allowance")
    if decision in ENTRY_DECISIONS and not bool(hard_gates.get("entry_factor_gates_passed")):
        violations.append("entry-without-entry-factor-gates")
    if decision in ENTRY_DECISIONS and entry_evidence.get("all_passed") is not True:
        violations.append("entry-without-passed-entry-gate-evidence")
    if decision in ENTRY_DECISIONS:
        missing_entry_gates = sorted(REQUIRED_ENTRY_GATE_NAMES - _gate_names(entry_evidence))
        if missing_entry_gates:
            violations.append(f"entry-missing-required-gate-evidence:{','.join(missing_entry_gates)}")
        failed_required_entry_gates = _required_gate_failures(entry_evidence, REQUIRED_ENTRY_GATE_NAMES)
        if failed_required_entry_gates:
            violations.append(f"entry-required-gates-not-passed:{','.join(failed_required_entry_gates)}")
        failed_entry_list = _failed_gate_list(entry_evidence)
        if failed_entry_list:
            violations.append(f"entry-gate-evidence-has-failures:{','.join(failed_entry_list)}")
    if decision in ENTRY_DECISIONS and not bool(expected.get("positive")):
        violations.append("entry-without-positive-expected-value")
    if decision in ENTRY_DECISIONS and lifecycle_evidence.get("all_passed") is not True:
        violations.append("entry-without-passed-lifecycle-evidence")
    if decision in ENTRY_DECISIONS:
        missing_lifecycle_gates = sorted(REQUIRED_LIFECYCLE_GATE_NAMES - _gate_names(lifecycle_evidence))
        if missing_lifecycle_gates:
            violations.append(f"entry-missing-required-lifecycle-evidence:{','.join(missing_lifecycle_gates)}")
        failed_required_lifecycle_gates = _required_gate_failures(
            lifecycle_evidence,
            REQUIRED_LIFECYCLE_GATE_NAMES,
        )
        if failed_required_lifecycle_gates:
            violations.append(f"entry-required-lifecycle-gates-not-passed:{','.join(failed_required_lifecycle_gates)}")
        failed_lifecycle_list = _failed_gate_list(lifecycle_evidence)
        if failed_lifecycle_list:
            violations.append(f"entry-lifecycle-evidence-has-failures:{','.join(failed_lifecycle_list)}")
    if decision in ENTRY_DECISIONS and _float_or_none(position_size.get("quantity")) is not None:
        quantity = _float_or_none(position_size.get("quantity"))
        if quantity is None or quantity <= 0.0:
            violations.append("entry-without-positive-position-size")
    elif decision in ENTRY_DECISIONS:
        violations.append("entry-without-positive-position-size")
    if decision in EXIT_DECISIONS and hard_gates.get("reduce_only_exit") is not True:
        violations.append("exit-not-reduce-only")
    if decision in EXIT_DECISIONS and hard_gates.get("readiness_allowed") is not True:
        violations.append("exit-without-readiness-allowance")
    if decision in EXIT_DECISIONS and _float_or_none(position_size.get("quantity")) is not None:
        quantity = _float_or_none(position_size.get("quantity"))
        if quantity is None or quantity <= 0.0:
            violations.append("exit-without-positive-position-size")
    elif decision in EXIT_DECISIONS:
        violations.append("exit-without-positive-position-size")
    if decision in EXIT_DECISIONS and entry is None:
        violations.append("exit-without-active-position-entry")
    if decision in EXIT_DECISIONS and stop_loss is None:
        violations.append("exit-without-reference-stop-loss")
    if decision in EXIT_DECISIONS and not str(expected.get("reason_code") or ""):
        violations.append("exit-without-reason-code")
    if hard_gates.get("ai_direct_order_allowed") is not False:
        violations.append("ai-direct-order-not-explicitly-forbidden")
    if opens_orders is not False:
        violations.append("decision-artifact-opens-orders-not-false")
    if writes_execution_config is not False:
        violations.append("decision-artifact-writes-config-not-false")
    if validation and validation.get("valid") is not True:
        violations.append("embedded-contract-validation-failed")
    if not validation:
        violations.append("missing-decision-contract-validation")
    if failed_checks:
        violations.extend(f"contract-check-failed:{item}" for item in failed_checks)
    return DecisionArtifactAudit(
        path=str(path.relative_to(root)),
        decision=decision,
        direction=direction,
        valid=not violations,
        failed_checks=failed_checks,
        violations=violations,
        risk_pct=risk_pct,
        expected_value_positive=_bool_or_none(expected.get("positive")),
        opens_orders=opens_orders,
        writes_execution_config=writes_execution_config,
    )


def _audit_file(path: Path, *, root: Path) -> DecisionArtifactAudit:
    payload, error = _load_json(path)
    if error:
        return DecisionArtifactAudit(
            path=str(path.relative_to(root)),
            decision="",
            direction="",
            valid=False,
            failed_checks=[],
            violations=[error],
            risk_pct=None,
            expected_value_positive=None,
            opens_orders=None,
            writes_execution_config=None,
        )
    assert payload is not None
    return _audit_payload(path, payload, root=root)


def run_decision_audit(
    *,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    since_contract: bool = False,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    source_dir = Path(input_dir).expanduser().resolve() if input_dir else TRADE_DECISION_DIR
    root_dir = Path(output_dir).expanduser().resolve() if output_dir else DECISION_AUDIT_DIR
    root_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(source_dir.glob("*-trade-decision.json")) if source_dir.exists() else []
    rows = [_audit_file(path, root=source_dir) for path in files]
    if since_contract:
        rows = [
            row
            for row in rows
            if "missing-decision-contract-validation" not in row.violations
        ]
    invalid_rows = [row for row in rows if not row.valid]
    decision_counts: dict[str, int] = {}
    for row in rows:
        key = row.decision or "unknown"
        decision_counts[key] = decision_counts.get(key, 0) + 1
    payload = {
        "mode": "decision_audit_v1",
        "generated_at": _utc_now().isoformat(),
        "input_dir": str(source_dir),
        "scope": "since_contract" if since_contract else "all_artifacts",
        "summary": {
            "artifact_count": len(rows),
            "valid_count": len(rows) - len(invalid_rows),
            "invalid_count": len(invalid_rows),
            "decision_counts": decision_counts,
            "opens_orders_false_count": sum(1 for row in rows if row.opens_orders is False),
            "writes_execution_config_false_count": sum(1 for row in rows if row.writes_execution_config is False),
        },
        "status": "passed" if not invalid_rows else "failed",
        "invalid_artifacts": [row.to_dict() for row in invalid_rows],
        "artifacts": [row.to_dict() for row in rows],
    }
    report_path = root_dir / f"{_stamp()}-decision-audit.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload
