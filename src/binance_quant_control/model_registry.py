from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    task: str
    runtime: str
    status: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    owner: str
    notes: str = ""
    retrain_trigger: str = "manual_review"
    promotion_gate: str = "cannot_approve_orders"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_model_registry() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="alpha-family-ranker",
            task="rank symbol/family/timeframe cohorts by expectancy and payoff",
            runtime="python",
            status="active",
            inputs=("alpha-research-ranking.json", "feature_manifest"),
            outputs=("promotion_score", "blockers"),
            owner="cpu",
            notes="Current alpha is negative; this model cannot approve trades by itself.",
            retrain_trigger="new_alpha_report_or_feature_manifest_hash_change",
            promotion_gate="positive_expectancy_and_portfolio_gate_required",
        ),
        ModelSpec(
            name="feature-label-dataset-builder",
            task="build replayable feature rows and triple-barrier labels",
            runtime="python",
            status="planned",
            inputs=("ohlcv", "feature_manifest", "label_spec"),
            outputs=("feature_matrix", "label_rows", "dataset_hash"),
            owner="cpu",
            notes="Required before any supervised model is allowed into promotion review.",
            retrain_trigger="new_candles_or_label_spec_change",
            promotion_gate="dataset_hash_and_leakage_report_required",
        ),
        ModelSpec(
            name="chart-regime-triage",
            task="image/chart regime classification",
            runtime="hailo-hef",
            status="optional",
            inputs=("rendered_chart_png",),
            outputs=("regime_tag", "chart_quality_warning"),
            owner="hailo",
            notes="Use only as veto/review assist unless trained and validated on trading labels.",
            retrain_trigger="new_labeled_chart_dataset",
            promotion_gate="veto_only_until_oos_validated",
        ),
        ModelSpec(
            name="news-risk-summarizer",
            task="event risk and narrative compression",
            runtime="llm-or-python",
            status="optional",
            inputs=("external_context",),
            outputs=("risk_level", "bias", "veto_reason"),
            owner="hermes",
            notes="Can veto entries; cannot override hard risk gates.",
            retrain_trigger="external_context_source_change",
            promotion_gate="veto_only",
        ),
    ]


def model_registry_payload() -> dict[str, Any]:
    rows = [item.to_dict() for item in default_model_registry()]
    return {
        "mode": "model_registry",
        "models": rows,
        "hailo_eligible": [row for row in rows if row["owner"] == "hailo"],
        "training_contract": {
            "requires_dataset_hash": True,
            "requires_feature_manifest_hash": True,
            "requires_out_of_sample_report": True,
            "requires_live_replay_match": True,
        },
        "hard_rule": "models may veto or rank candidates but cannot bypass expectancy, portfolio, or execution gates",
    }
