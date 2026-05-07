from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class HailoTask:
    name: str
    status: str
    reason: str
    input_artifacts: tuple[str, ...]
    output_artifacts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_hailo_trading_plan() -> dict[str, Any]:
    tasks = [
        HailoTask(
            name="chart-regime-triage",
            status="eligible",
            reason="Hailo is useful for compiled HEF image inference on rendered charts.",
            input_artifacts=("reports/*/chart.png",),
            output_artifacts=("regime_tag", "chart_quality_warning"),
        ),
        HailoTask(
            name="candlestick-image-anomaly-veto",
            status="eligible-after-training",
            reason="Can become a veto model only after labeled chart outcomes exist.",
            input_artifacts=("rendered_chart_png", "triple_barrier_label"),
            output_artifacts=("anomaly_veto",),
        ),
        HailoTask(
            name="pandas-backtest-acceleration",
            status="not_eligible",
            reason="Current indicator/backtest loops are pandas/NumPy CPU workloads, not HEF inference.",
            input_artifacts=("ohlcv_dataframe",),
            output_artifacts=(),
        ),
        HailoTask(
            name="order-execution-decision",
            status="not_allowed",
            reason="Hardware inference may assist review but cannot bypass risk or exchange gates.",
            input_artifacts=("live_signal",),
            output_artifacts=(),
        ),
    ]
    return {
        "mode": "hailo_trading_plan",
        "tasks": [task.to_dict() for task in tasks],
        "hard_rule": "Hailo can veto or triage; it cannot approve trades without alpha/risk gates.",
    }
