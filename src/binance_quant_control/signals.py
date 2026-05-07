"""Typed signal objects and deterministic decision helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

SignalBias = Literal["long", "short", "neutral"]
TradeAction = Literal["BUY", "SELL", "HOLD"]


@dataclass(frozen=True, slots=True)
class SignalResult:
    name: str
    bias: SignalBias
    score: float
    confidence: float


@dataclass(frozen=True, slots=True)
class TradeDecision:
    action: TradeAction
    directional_bias: str
    blockers: tuple[str, ...] = field(default_factory=tuple)
    rationale: tuple[str, ...] = field(default_factory=tuple)

    @property
    def allowed(self) -> bool:
        return self.action in {"BUY", "SELL"} and not self.blockers


def combine_signals(signals: Sequence[SignalResult]) -> SignalResult:
    """Combine multiple signals into a single deterministic result.

    The combined bias is decided by confidence-weighted voting. If the weights
    tie, the highest-confidence individual signal wins deterministically.
    """

    if not signals:
        return SignalResult(name="combined", bias="neutral", score=0.0, confidence=0.0)

    total_confidence = sum(signal.confidence for signal in signals)
    weighted_score = (
        sum(signal.score * signal.confidence for signal in signals) / total_confidence
        if total_confidence
        else 0.0
    )

    long_weight = sum(signal.confidence for signal in signals if signal.bias == "long")
    short_weight = sum(signal.confidence for signal in signals if signal.bias == "short")

    if long_weight > short_weight:
        bias: SignalBias = "long"
    elif short_weight > long_weight:
        bias = "short"
    else:
        best_signal = max(signals, key=lambda signal: signal.confidence)
        bias = best_signal.bias if best_signal.bias != "neutral" else "neutral"

    return SignalResult(
        name="combined",
        bias=bias,
        score=weighted_score,
        confidence=total_confidence / len(signals),
    )


def decide_trade_action(
    *,
    market: str,
    score: int,
    convergence: float,
    adx_value: float | None,
    regime: str,
    min_score_long: int,
    max_score_short: int,
    min_convergence: float,
    min_adx: float,
) -> TradeDecision:
    """Apply one deterministic entry policy across analysis, backtests, and live."""

    if score >= min_score_long:
        directional_bias = "long-bias"
        candidate_action: TradeAction = "BUY"
    elif score <= max_score_short:
        directional_bias = "short-bias" if market == "futures" else "defensive"
        candidate_action = "SELL" if market == "futures" else "HOLD"
    else:
        directional_bias = "neutral"
        candidate_action = "HOLD"

    blockers: list[str] = []
    rationale: list[str] = []

    if candidate_action == "HOLD":
        rationale.append("Score does not meet the configured entry thresholds.")
    if convergence < min_convergence:
        blockers.append(
            f"Signal convergence {convergence:.3f} is below the required {min_convergence:.3f}."
        )
    if adx_value is not None and adx_value < min_adx:
        blockers.append(f"ADX {adx_value:.2f} is below the required {min_adx:.2f}.")
    if regime == "range":
        blockers.append("Range regime detected; professional mode skips new trend entries.")
    if candidate_action == "SELL" and market != "futures":
        blockers.append("Short setups are disabled outside the futures lane.")

    action: TradeAction = candidate_action if not blockers else "HOLD"
    return TradeDecision(
        action=action,
        directional_bias=directional_bias,
        blockers=tuple(blockers),
        rationale=tuple(rationale),
    )
