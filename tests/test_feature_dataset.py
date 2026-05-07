from __future__ import annotations

import json
from types import SimpleNamespace

from binance_quant_control.feature_dataset import FeatureDatasetSpec, build_feature_dataset


def test_feature_dataset_builds_replayable_rows(tmp_path, monkeypatch) -> None:
    raw_klines = []
    open_time = 1_767_225_600_000
    price = 100.0
    for idx in range(260):
        raw_klines.append(
            [
                open_time + idx * 3_600_000,
                str(price),
                str(price + 2.0),
                str(price - 1.0),
                str(price + 1.0),
                str(1000 + idx),
                open_time + (idx + 1) * 3_600_000 - 1,
                str((1000 + idx) * price),
                100,
                str((500 + idx) * price),
                str((500 + idx) * price),
                "0",
            ]
        )
        price += 1.0

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def klines(self, *_args, **_kwargs):
            return raw_klines

    monkeypatch.setattr("binance_quant_control.feature_dataset.ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr("binance_quant_control.feature_dataset.BinanceClient", FakeClient)

    payload = build_feature_dataset(
        SimpleNamespace(),
        spec=FeatureDatasetSpec(
            symbols=["BTCUSDT"],
            intervals=["1h"],
            limit=260,
            strategy_config="config/strategy-core-high-win-research.yaml",
        ),
        output_dir=tmp_path,
    )

    assert payload["row_count"] == 260
    assert payload["dataset_hash"]
    assert payload["feature_manifest"]["manifest_hash"]
    assert payload["errors"] == []
    rows = (tmp_path / "feature-dataset.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    assert first["label_long_outcome"] in {"take_profit", "stop_loss", "time_limit", "unavailable"}
    assert first["label_short_outcome"] in {"take_profit", "stop_loss", "time_limit", "unavailable"}
    assert "label_long_r" in first
    assert "label_short_r" in first
    assert "ml_volatility_regime" in first
    assert "ml_trend_regime" in first
    assert "ml_liquidity_regime" in first
    assert "ml_payoff_potential_long" in first
    assert "ml_payoff_potential_short" in first
