from __future__ import annotations

from binance_quant_control.trading_committee import run_structured_committee


def test_committee_rejects_when_research_gate_is_not_safe() -> None:
    payload = run_structured_committee(
        {
            "trade_count": 5,
            "expectancy_r": -0.1,
            "payoff_ratio": 0.4,
            "profit_factor": 0.5,
            "blockers": ["expectancy-r-not-positive"],
        },
        {"high_win_iteration": {"safe_to_open_new_entries": False}},
    )

    assert payload["decision"] == "reject"
    votes = {item["role"]: item for item in payload["votes"]}
    assert votes["risk_manager"]["decision"] == "reject"


def test_committee_can_approve_for_review_when_hard_gates_are_clean() -> None:
    payload = run_structured_committee(
        {
            "trade_count": 120,
            "expectancy_r": 0.2,
            "payoff_ratio": 1.4,
            "profit_factor": 1.7,
            "blockers": [],
        },
        {"high_win_iteration": {"safe_to_open_new_entries": True}},
    )

    assert payload["decision"] == "approve_for_paper_or_testnet_review"
