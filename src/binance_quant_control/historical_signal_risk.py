from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .asset_routing import normalize_symbol
from .convergence import calculate_loss_streak, calculate_profit_factor
from .order_journal import read_closed_trade_reviews


@dataclass(frozen=True, slots=True)
class HistoricalSignalBucketStats:
    bucket_type: str
    label: str
    sample_count: int
    profit_factor: float
    net_pnl_usdt: float
    loss_streak: int
    min_samples: int
    threshold_profit_factor: float
    blocked: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket_type": self.bucket_type,
            "label": self.label,
            "sample_count": self.sample_count,
            "profit_factor": round(self.profit_factor, 4),
            "net_pnl_usdt": round(self.net_pnl_usdt, 8),
            "loss_streak": self.loss_streak,
            "min_samples": self.min_samples,
            "threshold_profit_factor": round(self.threshold_profit_factor, 4),
            "blocked": self.blocked,
        }


@dataclass(frozen=True, slots=True)
class HistoricalSignalRiskEvaluation:
    allowed: bool
    route_id: str
    symbol: str
    side: str
    score_bin: str
    convergence_bin: str
    min_samples: int
    threshold_profit_factor: float
    reasons: list[str]
    buckets: list[HistoricalSignalBucketStats]

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "route_id": self.route_id,
            "symbol": self.symbol,
            "side": self.side,
            "score_bin": self.score_bin,
            "convergence_bin": self.convergence_bin,
            "min_samples": self.min_samples,
            "threshold_profit_factor": round(self.threshold_profit_factor, 4),
            "reasons": self.reasons,
            "buckets": [bucket.to_dict() for bucket in self.buckets],
        }


@dataclass(frozen=True, slots=True)
class HistoricalSignalRiskIndex:
    route_side_score_convergence: dict[tuple[str, str, str, str], list[float]]
    route_side_score: dict[tuple[str, str, str], list[float]]
    route_side_convergence: dict[tuple[str, str, str], list[float]]
    symbol_side: dict[tuple[str, str], list[float]]
    review_count: int


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _side(value: Any) -> str:
    normalized = str(value or "").upper()
    return normalized if normalized in {"BUY", "SELL"} else "UNKNOWN"


def score_bucket(score: Any) -> str:
    value = int(_float(score))
    if value <= 20:
        return "score-000-020"
    if value <= 40:
        return "score-021-040"
    if value <= 60:
        return "score-041-060"
    if value <= 80:
        return "score-061-080"
    return "score-081-100"


def convergence_bucket(convergence: Any) -> str:
    value = _float(convergence)
    if value < 0.7:
        return "conv-000-069"
    if value < 0.8:
        return "conv-070-079"
    if value < 0.9:
        return "conv-080-089"
    return "conv-090-100"


def _append_bucket(
    buckets: dict[Any, list[float]],
    key: Any,
    pnl: float,
) -> None:
    buckets.setdefault(key, []).append(pnl)


def build_historical_signal_risk_index(
    reviews: list[dict[str, Any]] | None = None,
) -> HistoricalSignalRiskIndex:
    rows = reviews if reviews is not None else read_closed_trade_reviews()
    route_side_score_convergence: dict[tuple[str, str, str, str], list[float]] = {}
    route_side_score: dict[tuple[str, str, str], list[float]] = {}
    route_side_convergence: dict[tuple[str, str, str], list[float]] = {}
    symbol_side: dict[tuple[str, str], list[float]] = {}

    for row in rows:
        route_id = str(row.get("route_id") or "")
        symbol = normalize_symbol(str(row.get("symbol") or ""))
        side = _side(row.get("side"))
        if not route_id or side not in {"BUY", "SELL"}:
            continue
        pnl = _float(row.get("realized_pnl_usdt"))
        score_bin = score_bucket(row.get("analysis_score"))
        convergence_bin = convergence_bucket(row.get("analysis_convergence"))
        _append_bucket(
            route_side_score_convergence,
            (route_id, side, score_bin, convergence_bin),
            pnl,
        )
        _append_bucket(
            route_side_score,
            (route_id, side, score_bin),
            pnl,
        )
        _append_bucket(
            route_side_convergence,
            (route_id, side, convergence_bin),
            pnl,
        )
        if symbol:
            _append_bucket(symbol_side, (symbol, side), pnl)

    return HistoricalSignalRiskIndex(
        route_side_score_convergence=route_side_score_convergence,
        route_side_score=route_side_score,
        route_side_convergence=route_side_convergence,
        symbol_side=symbol_side,
        review_count=len(rows),
    )


def _bucket_stats(
    *,
    bucket_type: str,
    label: str,
    pnls: list[float],
    min_samples: int,
    min_profit_factor: float,
) -> HistoricalSignalBucketStats:
    profit_factor = calculate_profit_factor(pnls)
    net_pnl = sum(pnls)
    blocked = len(pnls) >= min_samples and net_pnl < 0.0 and profit_factor < min_profit_factor
    return HistoricalSignalBucketStats(
        bucket_type=bucket_type,
        label=label,
        sample_count=len(pnls),
        profit_factor=profit_factor,
        net_pnl_usdt=net_pnl,
        loss_streak=calculate_loss_streak(pnls),
        min_samples=min_samples,
        threshold_profit_factor=min_profit_factor,
        blocked=blocked,
    )


def evaluate_historical_signal_risk(
    *,
    route_id: str,
    symbol: str,
    side: str,
    score: int | float,
    convergence: float,
    min_samples: int = 20,
    min_profit_factor: float = 0.8,
    reviews: list[dict[str, Any]] | None = None,
    index: HistoricalSignalRiskIndex | None = None,
) -> HistoricalSignalRiskEvaluation:
    normalized_route = str(route_id or "")
    normalized_symbol = normalize_symbol(str(symbol or ""))
    normalized_side = _side(side)
    normalized_score_bucket = score_bucket(score)
    normalized_convergence_bucket = convergence_bucket(convergence)
    min_samples = max(int(min_samples), 1)
    min_profit_factor = float(min_profit_factor)
    signal_index = index or build_historical_signal_risk_index(reviews)

    bucket_specs = (
        (
            "route-side-score-convergence",
            f"{normalized_route}/{normalized_side}/{normalized_score_bucket}/{normalized_convergence_bucket}",
            signal_index.route_side_score_convergence.get(
                (
                    normalized_route,
                    normalized_side,
                    normalized_score_bucket,
                    normalized_convergence_bucket,
                ),
                [],
            ),
        ),
        (
            "route-side-score",
            f"{normalized_route}/{normalized_side}/{normalized_score_bucket}",
            signal_index.route_side_score.get(
                (normalized_route, normalized_side, normalized_score_bucket),
                [],
            ),
        ),
        (
            "route-side-convergence",
            f"{normalized_route}/{normalized_side}/{normalized_convergence_bucket}",
            signal_index.route_side_convergence.get(
                (normalized_route, normalized_side, normalized_convergence_bucket),
                [],
            ),
        ),
        (
            "symbol-side",
            f"{normalized_symbol}/{normalized_side}",
            signal_index.symbol_side.get((normalized_symbol, normalized_side), []),
        ),
    )

    buckets = [
        _bucket_stats(
            bucket_type=bucket_type,
            label=label,
            pnls=list(pnls),
            min_samples=min_samples,
            min_profit_factor=min_profit_factor,
        )
        for bucket_type, label, pnls in bucket_specs
    ]
    bucket_by_type = {bucket.bucket_type: bucket for bucket in buckets}
    specific_bucket = bucket_by_type.get("route-side-score-convergence")
    specific_positive_override = (
        specific_bucket is not None
        and specific_bucket.sample_count >= min_samples
        and specific_bucket.net_pnl_usdt > 0.0
        and specific_bucket.profit_factor >= 1.0
    )
    blocked_buckets = [
        bucket
        for bucket in buckets
        if bucket.blocked
        and not (
            specific_positive_override
            and bucket.bucket_type in {"route-side-convergence", "symbol-side"}
        )
    ]
    reasons = [
        (
            f"Historical feedback bucket {bucket.bucket_type} {bucket.label} has PF "
            f"{bucket.profit_factor:.4f} and net PnL {bucket.net_pnl_usdt:.4f} USDT "
            f"over {bucket.sample_count} reviews; threshold is PF {min_profit_factor:.4f} "
            "with non-negative net PnL."
        )
        for bucket in blocked_buckets
    ]
    return HistoricalSignalRiskEvaluation(
        allowed=not reasons,
        route_id=normalized_route,
        symbol=normalized_symbol,
        side=normalized_side,
        score_bin=normalized_score_bucket,
        convergence_bin=normalized_convergence_bucket,
        min_samples=min_samples,
        threshold_profit_factor=min_profit_factor,
        reasons=reasons,
        buckets=buckets,
    )
