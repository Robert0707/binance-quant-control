from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CommitteeVote:
    role: str
    decision: str
    confidence: float
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def run_structured_committee(signal: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    blockers = list(signal.get("blockers") or [])
    expectancy_r = _float(signal.get("expectancy_r"))
    payoff_ratio = _float(signal.get("payoff_ratio"))
    profit_factor = _float(signal.get("profit_factor"))
    trade_count = int(_float(signal.get("trade_count")))
    market_bot_gate = evidence.get("market_bot_gate") or {}
    market_targets = market_bot_gate.get("targets") if isinstance(market_bot_gate.get("targets"), dict) else {}
    min_profit_factor = _float(market_targets.get("min_profit_factor"), 1.25)
    gate_safe = bool(
        market_bot_gate.get("safe_to_open_new_entries")
        or (evidence.get("high_win_iteration") or {}).get("safe_to_open_new_entries")
    )
    votes = [
        CommitteeVote(
            role="analyst",
            decision="reject" if trade_count < 100 else "watch",
            confidence=0.8,
            reasons=[f"sample_count={trade_count}", "need >=100 trades per promoted cohort"],
        ),
        CommitteeVote(
            role="bull_case",
            decision="support" if expectancy_r > 0 and payoff_ratio >= 1.15 else "reject",
            confidence=0.65,
            reasons=[f"expectancy_r={expectancy_r}", f"payoff_ratio={payoff_ratio}"],
        ),
        CommitteeVote(
            role="bear_case",
            decision="reject" if blockers else "watch",
            confidence=0.9,
            reasons=blockers or ["no signal-level blockers"],
        ),
        CommitteeVote(
            role="risk_manager",
            decision="reject" if not gate_safe else "support",
            confidence=0.95,
            reasons=[
                "safe_to_open_new_entries=false" if not gate_safe else "research gate allows entries"
            ],
        ),
        CommitteeVote(
            role="portfolio_manager",
            decision="reject" if profit_factor < min_profit_factor else "watch",
            confidence=0.8,
            reasons=[
                f"profit_factor={profit_factor}",
                f"min_profit_factor={min_profit_factor}",
                "portfolio promotion still requires core coverage",
            ],
        ),
    ]
    hard_reject = any(vote.decision == "reject" for vote in votes if vote.role != "bull_case")
    return {
        "mode": "structured_trading_committee",
        "decision": "reject" if hard_reject else "approve_for_paper_or_testnet_review",
        "votes": [vote.to_dict() for vote in votes],
        "hard_rule": "committee cannot override alpha, portfolio, risk, or execution gates",
    }
