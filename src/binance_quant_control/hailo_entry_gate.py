from __future__ import annotations

from typing import Any

HAILO_VETO_LABELS = {
    "entry_gate_blocked",
    "profit_floor_failed",
    "professional_gate_failed",
    "execution_quality_failed",
    "market_state_failed",
    "strategy_performance_failed",
    "loss_trade",
    "high_leverage",
    "low_convergence",
    "high_event_risk",
    "needs_root_cause_review",
}
HAILO_ADVISORY_LABELS = {
    "weak_win_rate",
}
TESTNET_CONTEXTUAL_LABELS = {
    "high_event_risk",
    "needs_root_cause_review",
}
TESTNET_CONTEXTUAL_EVENT_TYPES = {
    "news_event",
    "system_error",
}


def _response_from_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    response = payload.get("response")
    if isinstance(response, dict):
        return response
    return payload


def _events_from_response(response: dict[str, Any]) -> list[dict[str, Any]]:
    events = response.get("events")
    if isinstance(events, list):
        return [item for item in events if isinstance(item, dict)]
    output_events = response.get("output_events")
    if isinstance(output_events, list):
        return [item for item in output_events if isinstance(item, dict)]
    return []


def _candidate_symbols(candidate: dict[str, Any] | None) -> set[str]:
    if not isinstance(candidate, dict):
        return set()
    symbols: set[str] = set()
    for key in ("symbol", "base_symbol", "quote_symbol"):
        value = str(candidate.get(key) or "").strip().upper()
        if value:
            symbols.add(value)
    return symbols


def _event_symbol(event: dict[str, Any]) -> str:
    return str(event.get("symbol") or "").strip().upper()


def _testnet_contextual_advisory(
    *,
    execution_mode: str,
    event: dict[str, Any],
    matched: list[str],
    candidate_symbols: set[str],
) -> bool:
    if execution_mode != "testnet_exploration":
        return False
    event_type = str(event.get("event_type") or "")
    if event_type not in TESTNET_CONTEXTUAL_EVENT_TYPES:
        return False
    event_symbol = _event_symbol(event)
    if event_symbol and event_symbol in candidate_symbols:
        return False
    if event_symbol and candidate_symbols:
        return True
    return all(label in TESTNET_CONTEXTUAL_LABELS or label.startswith("priority_") for label in matched)


def evaluate_hailo_entry_gate(
    payload: dict[str, Any] | None,
    *,
    execution_mode: str = "strict",
    candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or int(payload.get("returncode", 0) or 0) != 0:
        return {
            "allowed": False,
            "decision": "unavailable",
            "blockers": ["hailo-triage-unavailable"],
            "reason": "fail_closed_without_hailo_triage",
            "events": [],
        }

    response = _response_from_payload(payload)
    events = _events_from_response(response)
    blockers: list[str] = []
    advisories: list[str] = []
    veto_events: list[dict[str, Any]] = []
    advisory_events: list[dict[str, Any]] = []
    contextual_advisory_events: list[dict[str, Any]] = []
    symbols = _candidate_symbols(candidate)
    for event in events:
        labels = [str(item) for item in (event.get("labels") or [])]
        priority = str(event.get("priority") or "low").lower()
        event_type = str(event.get("event_type") or "")
        matched = [label for label in labels if label in HAILO_VETO_LABELS]
        advisory = [label for label in labels if label in HAILO_ADVISORY_LABELS]
        if priority == "critical" or event_type == "system_error":
            matched.append(f"priority_{priority}")
        if not matched and not advisory:
            continue
        if matched and _testnet_contextual_advisory(
            execution_mode=execution_mode,
            event=event,
            matched=matched,
            candidate_symbols=symbols,
        ):
            contextual_advisory_events.append(event)
            advisory_events.append(event)
            for label in matched:
                warning = f"hailo-advisory:contextual_{label}"
                if warning not in advisories:
                    advisories.append(warning)
            continue
        if matched:
            veto_events.append(event)
        for label in matched:
            blocker = f"hailo-veto:{label}"
            if blocker not in blockers:
                blockers.append(blocker)
        if advisory:
            advisory_events.append(event)
        for label in advisory:
            warning = f"hailo-advisory:{label}"
            if warning not in advisories:
                advisories.append(warning)

    return {
        "allowed": not blockers,
        "decision": "veto" if blockers else "allow",
        "blockers": blockers,
        "advisories": advisories,
        "reason": "hailo_local_veto" if blockers else "hailo_observability_pass",
        "events": veto_events,
        "advisory_events": advisory_events,
        "contextual_advisory_events": contextual_advisory_events,
        "execution_mode": execution_mode,
        "candidate_symbols": sorted(symbols),
        "raw_event_count": response.get("raw_event_count"),
        "output_event_count": response.get("output_event_count"),
        "retained_existing_output": response.get("retained_existing_output"),
    }
