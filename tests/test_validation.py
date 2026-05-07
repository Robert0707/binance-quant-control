from __future__ import annotations

from pathlib import Path

import pytest

from binance_quant_control.config import load_settings
from binance_quant_control.strategy import load_strategy_config


def test_load_settings_rejects_invalid_market(monkeypatch):
    monkeypatch.setenv("BINANCE_DEFAULT_MARKET", "margin")

    with pytest.raises(ValueError, match="Settings validation failed"):
        load_settings()


def test_load_strategy_config_rejects_inverted_signal_thresholds(tmp_path):
    strategy_path = tmp_path / "invalid-strategy.yaml"
    strategy_path.write_text(
        """
profile: invalid
risk:
  min_score_long: 40
  max_score_short: 50
signal:
  ema_fast: 13
  ema_slow: 34
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Strategy config .* validation failed"):
        load_strategy_config(Path(strategy_path))
