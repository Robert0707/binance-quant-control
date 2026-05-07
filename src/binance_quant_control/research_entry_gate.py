from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .asset_routing import normalize_symbol
from .historical_signal_risk import (
    HistoricalSignalRiskIndex,
    build_historical_signal_risk_index,
    evaluate_historical_signal_risk,
)
from .order_journal import read_closed_trade_reviews
from .side_risk_policy import SideRiskEvaluation, evaluate_route_side_risk

ResearchEntryFilter = Callable[
    [pd.Series, pd.Series, dict[str, Any], int],
    bool | tuple[bool, str],
]


@dataclass(frozen=True, slots=True)
class ResearchEntryGateConfig:
    enabled: bool = False
    route_side_veto: bool = True
    historical_signal_veto: bool = True
    shadow_route_side_veto: bool = False
    shadow_historical_signal_veto: bool = False
    route_side_min_samples: int = 30
    route_side_min_profit_factor: float = 0.8
    route_side_max_stop_loss_ratio: float = 70.0
    historical_signal_min_samples: int = 20
    historical_signal_min_profit_factor: float = 0.8

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "ResearchEntryGateConfig":
        data = raw or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            route_side_veto=bool(data.get("route_side_veto", True)),
            historical_signal_veto=bool(data.get("historical_signal_veto", True)),
            shadow_route_side_veto=bool(data.get("shadow_route_side_veto", False)),
            shadow_historical_signal_veto=bool(data.get("shadow_historical_signal_veto", False)),
            route_side_min_samples=max(int(data.get("route_side_min_samples") or 30), 1),
            route_side_min_profit_factor=float(data.get("route_side_min_profit_factor") or 0.8),
            route_side_max_stop_loss_ratio=float(data.get("route_side_max_stop_loss_ratio") or 70.0),
            historical_signal_min_samples=max(int(data.get("historical_signal_min_samples") or 20), 1),
            historical_signal_min_profit_factor=float(data.get("historical_signal_min_profit_factor") or 0.8),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "route_side_veto": self.route_side_veto,
            "historical_signal_veto": self.historical_signal_veto,
            "shadow_route_side_veto": self.shadow_route_side_veto,
            "shadow_historical_signal_veto": self.shadow_historical_signal_veto,
            "route_side_min_samples": self.route_side_min_samples,
            "route_side_min_profit_factor": round(self.route_side_min_profit_factor, 4),
            "route_side_max_stop_loss_ratio": round(self.route_side_max_stop_loss_ratio, 2),
            "historical_signal_min_samples": self.historical_signal_min_samples,
            "historical_signal_min_profit_factor": round(self.historical_signal_min_profit_factor, 4),
        }


def _signal_side(analysis: dict[str, Any]) -> str:
    action = str((analysis or {}).get("recommended_action") or "").upper()
    return action if action in {"BUY", "SELL"} else "UNKNOWN"


def _route_side_evaluations(
    *,
    route_id: str,
    config: ResearchEntryGateConfig,
    reviews: list[dict[str, Any]],
) -> dict[str, SideRiskEvaluation]:
    return {
        side: evaluate_route_side_risk(
            route_id=route_id,
            side=side,
            min_samples=config.route_side_min_samples,
            min_profit_factor=config.route_side_min_profit_factor,
            max_stop_loss_ratio=config.route_side_max_stop_loss_ratio,
            reviews=reviews,
        )
        for side in ("BUY", "SELL")
    }


def build_research_entry_gate(
    *,
    route_id: str,
    symbol: str,
    config: ResearchEntryGateConfig,
    reviews: list[dict[str, Any]] | None = None,
    historical_signal_index: HistoricalSignalRiskIndex | None = None,
) -> tuple[ResearchEntryFilter | None, dict[str, Any]]:
    normalized_route = str(route_id or "")
    normalized_symbol = normalize_symbol(str(symbol or ""))
    if not config.enabled:
        return None, {"enabled": False, "route_id": normalized_route, "symbol": normalized_symbol}

    review_rows = reviews if reviews is not None else read_closed_trade_reviews()
    route_side_enabled = config.route_side_veto or config.shadow_route_side_veto
    historical_enabled = config.historical_signal_veto or config.shadow_historical_signal_veto
    side_evaluations = (
        _route_side_evaluations(route_id=normalized_route, config=config, reviews=review_rows)
        if route_side_enabled
        else {}
    )
    signal_index = (
        historical_signal_index
        if historical_signal_index is not None
        else build_historical_signal_risk_index(review_rows)
        if historical_enabled
        else None
    )

    metadata = {
        "enabled": True,
        "route_id": normalized_route,
        "symbol": normalized_symbol,
        "review_count": len(review_rows),
        "config": config.to_dict(),
        "route_side": {
            side: evaluation.to_dict()
            for side, evaluation in sorted(side_evaluations.items())
        },
        "historical_signal": {
            "enabled": historical_enabled,
            "enforced": config.historical_signal_veto,
            "shadow_only": bool(config.shadow_historical_signal_veto and not config.historical_signal_veto),
            "review_count": signal_index.review_count if signal_index is not None else 0,
            "min_samples": config.historical_signal_min_samples,
            "threshold_profit_factor": round(config.historical_signal_min_profit_factor, 4),
        },
        "applied_principles": [
            "pre-trade-risk-before-backtest-entry",
            "quarantine-route-side-with-negative-history",
            "veto-known-losing-score-convergence-buckets",
        ],
    }

    def entry_gate(
        previous: pd.Series,
        current: pd.Series,
        analysis: dict[str, Any],
        idx: int,
    ) -> tuple[bool, str]:
        del previous, current, idx
        side = _signal_side(analysis)
        if side not in {"BUY", "SELL"}:
            return True, ""
        if config.route_side_veto:
            side_gate = side_evaluations.get(side)
            if side_gate is not None and not side_gate.allowed:
                return False, "route-side-history-veto"
        if config.historical_signal_veto and signal_index is not None:
            signal_gate = evaluate_historical_signal_risk(
                route_id=normalized_route,
                symbol=normalized_symbol,
                side=side,
                score=float((analysis or {}).get("score") or 0.0),
                convergence=float((analysis or {}).get("convergence") or 0.0),
                min_samples=config.historical_signal_min_samples,
                min_profit_factor=config.historical_signal_min_profit_factor,
                index=signal_index,
            )
            if not signal_gate.allowed:
                return False, "historical-feedback-bucket-veto"
        return True, ""

    return entry_gate, metadata
