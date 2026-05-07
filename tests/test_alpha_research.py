from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import binance_quant_control.alpha_research as alpha_research
from binance_quant_control.candidate_universe import UniverseSymbol
from binance_quant_control.strategy import load_strategy_config
from binance_quant_control.symbol_strategy_map import SymbolPromotionSpec, SymbolStrategySpec


def _fake_backtest_payload(
    symbol: str,
    interval: str,
    slippage_bps: float,
    strategy_family: str,
) -> dict[str, object]:
    drag = max(slippage_bps - 3.0, 0.0) * 0.2
    family_bonus = {
        "breakout": 3.0,
        "trend_pullback": 2.0,
        "liquidity_reclaim": 1.5,
        "vwap_reclaim": 1.7,
        "trend_continuation": 1.0,
        "mean_reversion": -2.0,
    }.get(
        strategy_family,
        0.0,
    )
    total_return = round(18.0 + family_bonus - drag, 4)
    trades = [
        {
            "side": "BUY",
            "entry_time": f"2026-01-{idx + 1:02d}T00:00:00+00:00",
            "exit_time": f"2026-01-{idx + 1:02d}T04:00:00+00:00",
            "pnl_pct": pnl,
            "pnl_r": pnl / 2.0,
            "exit_reason": "take_profit" if pnl > 0 else "stop_loss",
        }
        for idx, pnl in enumerate([1.2, 1.4, -0.6, 1.8, 2.2, -0.8, 2.4, 2.0, 1.7, 1.5])
    ]
    return {
        "symbol": symbol,
        "interval": interval,
        "strategy_family": strategy_family,
        "summary": {
            "trade_count": len(trades),
            "total_return_pct": total_return,
            "max_drawdown_pct": 3.0,
            "profit_factor": 2.4,
            "expectancy_r": 0.52,
            "payoff_ratio": 2.1,
            "win_rate": 80.0,
            "loss_streak": 1,
            "trades": trades,
        },
        "robustness": {
            "status": "passed",
            "passed": True,
            "folds": [
                {"total_return_pct": 2.0, "profit_factor": 1.4},
                {"total_return_pct": 3.0, "profit_factor": 1.5},
                {"total_return_pct": 4.0, "profit_factor": 1.6},
            ],
        },
        "artifacts": {"report_json": f"/tmp/{symbol}-{interval}-{slippage_bps}.json"},
    }


def test_aggressive_alpha_research_rejects_mainnet_live_config(tmp_path) -> None:
    config_path = tmp_path / "unsafe-alpha.yaml"
    config_path.write_text(
        """
strategy:
  strategy_config: strategy-aggressive-alpha-research.yaml
universe:
  symbols: [BTCUSDT]
ranking:
  slippage_bps_cases: [3.0]
safety:
  mainnet_live_allowed: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mainnet_live_allowed=false"):
        alpha_research.run_aggressive_alpha_research(SimpleNamespace(), config_path=config_path)


def test_resolve_alpha_research_config_accepts_config_prefixed_path() -> None:
    path = alpha_research.resolve_alpha_research_config_path(
        "config/aggressive-alpha-research.default.yaml"
    )

    assert path == (alpha_research.CONFIG_DIR / "aggressive-alpha-research.default.yaml").resolve()
    assert path.exists()


def test_default_alpha_research_config_targets_sixty_symbol_universe() -> None:
    config = alpha_research._load_alpha_research_config("config/aggressive-alpha-research.default.yaml")

    assert config["universe"]["include_top_futures_volume"] is True
    assert config["universe"]["top_volume_limit"] == 60
    assert config["ranking"]["stress_top_n"] == 20
    assert config["ranking"]["min_profit_factor"] == 1.2
    assert config["ranking"]["max_stop_loss_ratio"] == 55.0


def test_core_alpha_research_config_targets_core_symbols_only() -> None:
    config = alpha_research._load_alpha_research_config("config/core-alpha-research.default.yaml")

    assert config["strategy"]["strategy_config"] == "strategy-core-high-win-research.yaml"
    assert config["symbol_strategy_map"] == {
        "enabled": True,
        "path": "core-symbol-strategy-map.default.yaml",
    }
    assert config["research_entry_gate"]["enabled"] is True
    assert config["universe"]["include_top_futures_volume"] is False
    assert config["universe"]["symbols"] == [
        "BTCUSDT",
        "ETHUSDT",
        "XAUTUSDT",
        "PAXGUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "LINKUSDT",
        "AAVEUSDT",
        "TRXUSDT",
    ]
    assert config["universe"]["intervals"] == ["15m", "1h", "4h", "1d"]
    assert config["ranking"]["strategy_families"] == [
        "trend_continuation",
        "breakout",
        "trend_pullback",
        "liquidity_reclaim",
        "vwap_reclaim",
        "mean_reversion",
    ]
    assert config["ranking"]["min_trades"] == 100
    assert config["ranking"]["min_win_rate"] == 65.0
    assert config["ranking"]["max_stop_loss_ratio"] == 35.0
    assert config["ranking"]["min_expectancy_r"] == 0.10
    assert config["ranking"]["min_payoff_ratio"] == 1.15
    assert config["safety"]["mainnet_live_allowed"] is False


def test_market_bot_gate_config_uses_expectancy_payoff_thresholds() -> None:
    config = alpha_research._load_alpha_research_config("config/market-bot-gate.default.yaml")

    assert config["targets"]["min_profit_factor"] == 1.25
    assert config["targets"]["min_expectancy_r"] == 0.05
    assert config["targets"]["min_payoff_ratio"] == 1.20
    assert config["targets"]["min_out_of_sample_return_pct"] == 0.0
    assert config["targets"]["min_win_rate"] == 45.0
    assert config["targets"]["min_accepted_symbols"] == 6
    assert config["targets"]["require_feature_manifest_hash"] is True


def test_market_bot_discovery_config_expands_without_symbol_strategy_map() -> None:
    config = alpha_research._load_alpha_research_config("config/market-bot-discovery.default.yaml")

    assert "symbol_strategy_map" not in config
    assert config["strategy"]["strategy_config"] == "strategy-core-high-win-research.yaml"
    assert config["universe"]["symbols"] == ["TRXUSDT", "ETHUSDT", "BTCUSDT"]
    assert config["universe"]["intervals"] == ["1h", "4h"]
    assert config["ranking"]["min_trades"] == 100
    assert config["ranking"]["min_profit_factor"] == 1.25
    assert config["ranking"]["min_expectancy_r"] == 0.05
    assert config["ranking"]["min_payoff_ratio"] == 1.20
    assert config["ranking"]["min_win_rate"] == 45.0
    assert config["ranking"]["max_stop_loss_ratio"] == 55.0
    assert config["research_entry_gate"]["route_side_veto"] is False
    assert config["research_entry_gate"]["shadow_route_side_veto"] is True
    assert config["safety"]["mainnet_live_allowed"] is False


def test_market_bot_six_symbol_discovery_config_is_payoff_first() -> None:
    config = alpha_research._load_alpha_research_config(
        "config/market-bot-six-symbol-discovery.default.yaml"
    )

    assert config["strategy"]["strategy_config"] == "strategy-market-bot-payoff-research.yaml"
    assert config["universe"]["symbols"] == [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "TRXUSDT",
        "DOGEUSDT",
    ]
    assert config["universe"]["intervals"] == ["1h", "4h"]
    assert config["ranking"]["min_trades"] == 100
    assert config["ranking"]["min_profit_factor"] == 1.25
    assert config["ranking"]["min_expectancy_r"] == 0.05
    assert config["ranking"]["min_payoff_ratio"] == 1.20
    assert config["ranking"]["enforce_win_rate_gate"] is False
    assert config["ranking"]["strategy_families"][0] == "ai_family_router"
    assert config["ranking"]["weights"]["payoff_objective"] == 0.56
    assert config["research_entry_gate"]["route_side_veto"] is False
    assert config["research_entry_gate"]["shadow_route_side_veto"] is True
    assert config["safety"]["mainnet_live_allowed"] is False


def test_core_high_win_research_config_is_mean_reversion_first() -> None:
    config = alpha_research._load_alpha_research_config("config/core-high-win-research.default.yaml")

    assert config["symbol_strategy_map"] == {
        "enabled": True,
        "path": "core-symbol-strategy-map.default.yaml",
    }
    assert config["universe"]["include_top_futures_volume"] is False
    assert config["universe"]["symbols"] == [
        "BTCUSDT",
        "ETHUSDT",
        "XAUTUSDT",
        "PAXGUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "LINKUSDT",
        "AAVEUSDT",
        "TRXUSDT",
    ]
    assert config["universe"]["intervals"] == ["15m", "1h", "4h", "1d"]
    assert config["ranking"]["strategy_families"] == [
        "trend_continuation",
        "breakout",
        "trend_pullback",
        "liquidity_reclaim",
        "vwap_reclaim",
        "mean_reversion",
    ]
    assert config["ranking"]["target_metrics"] == {
        "win_rate": 65.0,
        "max_stop_loss_ratio": 35.0,
        "expectancy_r": 0.10,
        "payoff_ratio": 1.15,
    }
    assert config["ranking"]["min_trades"] == 100
    assert config["ranking"]["min_win_rate"] == 65.0
    assert config["ranking"]["max_stop_loss_ratio"] == 35.0
    assert config["ranking"]["min_expectancy_r"] == 0.10
    assert config["ranking"]["min_payoff_ratio"] == 1.15
    assert config["safety"]["mainnet_live_allowed"] is False


def test_alpha_research_applies_symbol_strategy_map(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "mapped-alpha.yaml"
    config_path.write_text(
        """
strategy:
  strategy_config: strategy-core-high-win-research.yaml
symbol_strategy_map:
  enabled: true
  path: core-symbol-strategy-map.default.yaml
universe:
  include_top_futures_volume: false
  symbols: [BTCUSDT, TRXUSDT, SOLUSDT]
  intervals: [1h, 4h, 1d]
  limit: 320
ranking:
  min_trades: 8
  strategy_families: [trend_continuation, breakout, trend_pullback, liquidity_reclaim, vwap_reclaim, mean_reversion]
  slippage_bps_cases: [3.0]
  stress_top_n: 0
  min_profit_factor: 1.1
  min_win_rate: 40
  max_stop_loss_ratio: 70
safety:
  mainnet_live_allowed: false
""",
        encoding="utf-8",
    )
    calls: list[tuple[str, str]] = []

    def fake_run_backtest(settings, *, symbol, interval, strategy, strategy_family=None, **_kwargs):
        calls.append((symbol, strategy_family or "none"))
        return _fake_backtest_payload(
            symbol,
            interval,
            strategy.execution.slippage_bps,
            strategy_family or "none",
        )

    monkeypatch.setattr(alpha_research, "run_backtest", fake_run_backtest)

    payload = alpha_research.run_aggressive_alpha_research(
        SimpleNamespace(),
        config_path=config_path,
        output_dir=tmp_path / "out",
    )

    assert payload["symbol_strategy_map"]["enabled"] is True
    assert payload["resolved_symbol_family_plan"] == [
        {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "families": [],
            "active": False,
            "inactive_reason": "symbol-interval-quarantined-by-strategy-map",
            "symbol_strategy_map_applied": True,
            "primary_family": "breakout",
            "execution_lane": "paper_research_only",
            "route_id": "btc-core",
        },
        {
            "symbol": "BTCUSDT",
            "interval": "4h",
            "families": [],
            "active": False,
            "inactive_reason": "symbol-interval-quarantined-by-strategy-map",
            "symbol_strategy_map_applied": True,
            "primary_family": "breakout",
            "execution_lane": "paper_research_only",
            "route_id": "btc-core",
        },
        {
            "symbol": "BTCUSDT",
            "interval": "1d",
            "families": [],
            "active": False,
            "inactive_reason": "symbol-interval-quarantined-by-strategy-map",
            "symbol_strategy_map_applied": True,
            "primary_family": "breakout",
            "execution_lane": "paper_research_only",
            "route_id": "btc-core",
        },
        {
            "symbol": "TRXUSDT",
            "interval": "1h",
            "families": [],
            "active": False,
            "inactive_reason": "symbol-interval-quarantined-by-strategy-map",
            "symbol_strategy_map_applied": True,
            "primary_family": "mean_reversion",
            "execution_lane": "paper_research_only",
            "route_id": "trx-mean-reversion",
        },
        {
            "symbol": "TRXUSDT",
            "interval": "4h",
            "families": ["mean_reversion"],
            "active": True,
            "inactive_reason": None,
            "symbol_strategy_map_applied": True,
            "primary_family": "mean_reversion",
            "execution_lane": "paper_research_only",
            "route_id": "trx-mean-reversion",
        },
        {
            "symbol": "TRXUSDT",
            "interval": "1d",
            "families": [],
            "active": False,
            "inactive_reason": "symbol-interval-quarantined-by-strategy-map",
            "symbol_strategy_map_applied": True,
            "primary_family": "mean_reversion",
            "execution_lane": "paper_research_only",
            "route_id": "trx-mean-reversion",
        },
        {
            "symbol": "SOLUSDT",
            "interval": "1h",
            "families": [],
            "active": False,
            "inactive_reason": "symbol-interval-quarantined-by-strategy-map",
            "symbol_strategy_map_applied": True,
            "primary_family": "breakout",
            "execution_lane": "paper_research_only",
            "route_id": "sol-volatility-breakout",
        },
        {
            "symbol": "SOLUSDT",
            "interval": "4h",
            "families": [],
            "active": False,
            "inactive_reason": "symbol-interval-quarantined-by-strategy-map",
            "symbol_strategy_map_applied": True,
            "primary_family": "breakout",
            "execution_lane": "paper_research_only",
            "route_id": "sol-volatility-breakout",
        },
        {
            "symbol": "SOLUSDT",
            "interval": "1d",
            "families": [],
            "active": False,
            "inactive_reason": "symbol-interval-quarantined-by-strategy-map",
            "symbol_strategy_map_applied": True,
            "primary_family": "breakout",
            "execution_lane": "paper_research_only",
            "route_id": "sol-volatility-breakout",
        },
    ]
    assert set(calls) == {
        ("TRXUSDT", "mean_reversion"),
    }
    trx_rows = [row for row in payload["rows"] if row["symbol"] == "TRXUSDT"]
    assert trx_rows[0]["symbol_strategy"]["primary_family"] == "mean_reversion"
    assert trx_rows[0]["symbol_strategy"]["interval_family_sides"]["4h"]["mean_reversion"] == ["SELL"]
    assert payload["skipped_symbol_intervals"] == [
        {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "family": "none",
            "reason": "symbol-interval-quarantined-by-strategy-map",
        },
        {
            "symbol": "BTCUSDT",
            "interval": "4h",
            "family": "none",
            "reason": "symbol-interval-quarantined-by-strategy-map",
        },
        {
            "symbol": "BTCUSDT",
            "interval": "1d",
            "family": "none",
            "reason": "symbol-interval-quarantined-by-strategy-map",
        },
        {
            "symbol": "TRXUSDT",
            "interval": "1h",
            "family": "none",
            "reason": "symbol-interval-quarantined-by-strategy-map",
        },
        {
            "symbol": "TRXUSDT",
            "interval": "1d",
            "family": "none",
            "reason": "symbol-interval-quarantined-by-strategy-map",
        },
        {
            "symbol": "SOLUSDT",
            "interval": "1h",
            "family": "none",
            "reason": "symbol-interval-quarantined-by-strategy-map",
        },
        {
            "symbol": "SOLUSDT",
            "interval": "4h",
            "family": "none",
            "reason": "symbol-interval-quarantined-by-strategy-map",
        },
        {
            "symbol": "SOLUSDT",
            "interval": "1d",
            "family": "none",
            "reason": "symbol-interval-quarantined-by-strategy-map",
        },
    ]


def test_aggressive_alpha_research_ranks_and_writes_report(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, str, float, str]] = []

    def fake_fetch_top_symbols(*_args, **_kwargs):
        return [
            UniverseSymbol("SOLUSDT", 3_000_000_000.0, 1, "pytest-volume"),
            UniverseSymbol("DOGEUSDT", 2_000_000_000.0, 2, "pytest-volume"),
        ]

    def fake_run_backtest(settings, *, strategy, symbol, interval, strategy_family=None, **_kwargs):
        calls.append((symbol, interval, strategy.execution.slippage_bps, strategy_family or "none"))
        return _fake_backtest_payload(
            symbol,
            interval,
            strategy.execution.slippage_bps,
            strategy_family or "none",
        )

    monkeypatch.setattr(alpha_research, "fetch_top_futures_symbols", fake_fetch_top_symbols)
    monkeypatch.setattr(alpha_research, "run_backtest", fake_run_backtest)

    payload = alpha_research.run_aggressive_alpha_research(
        SimpleNamespace(),
        output_dir=tmp_path,
        interval_overrides=["1h"],
    )

    assert payload["mainnet_live_allowed"] is False
    assert payload["symbols"] == ["SOLUSDT", "DOGEUSDT"]
    assert payload["strategy_families"] == [
        "trend_continuation",
        "breakout",
        "trend_pullback",
        "liquidity_reclaim",
        "vwap_reclaim",
        "mean_reversion",
    ]
    assert payload["top"]
    assert payload["top"][0]["strategy_family"] == "breakout"
    assert payload["top"][0]["cohort_id"] == "SOLUSDT:1h:breakout"
    assert payload["top"][0]["out_of_sample_total_return_pct"] > 0
    assert payload["top"][0]["promotion_eligible"] is True
    assert payload["top"][0]["stress_tested"] is True
    assert payload["top"][0]["universe_source"] == "pytest-volume"
    assert payload["stress_test"]["stressed_cohort_count"] > 0
    assert payload["universe_selection"]["alpha_sources"]
    assert payload["universe_selection"]["high_beta_proxy"][0]["symbol"] == "SOLUSDT"
    assert payload["performance_summary"]["weighted_win_rate"] > 0.0
    assert payload["performance_summary"]["by_family"][0]["strategy_family"] == "breakout"
    assert payload["performance_summary"]["by_interval"][0]["interval"] == "1h"
    assert payload["performance_summary"]["by_symbol_interval"][0]["symbol"] in {"SOLUSDT", "DOGEUSDT"}
    assert payload["execution_recommendation"] == "paper_or_testnet_candidate_available"
    assert Path(payload["report_path"]).exists()
    base_calls = [call for call in calls if call[2] == 3.0]
    stress_calls = [call for call in calls if call[2] != 3.0]
    assert len(base_calls) == len(payload["symbols"]) * len(payload["strategy_families"])
    assert stress_calls
    assert {call[2] for call in stress_calls} == {8.0, 15.0}
    assert {call[3] for call in calls} == {
        "trend_continuation",
        "breakout",
        "trend_pullback",
        "liquidity_reclaim",
        "vwap_reclaim",
        "mean_reversion",
    }


def test_alpha_research_requires_stress_before_promotion(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "stress-required.yaml"
    config_path.write_text(
        """
strategy:
  strategy_config: strategy-aggressive-alpha-research.yaml
universe:
  include_top_futures_volume: false
  symbols: [SOLUSDT]
  intervals: [1h]
  limit: 320
ranking:
  min_trades: 8
  strategy_families: [breakout, trend_continuation]
  slippage_bps_cases: [3.0, 8.0]
  stress_top_n: 1
  min_profit_factor: 1.2
  min_win_rate: 42
  max_stop_loss_ratio: 70
safety:
  mainnet_live_allowed: false
""",
        encoding="utf-8",
    )
    calls: list[tuple[str, float]] = []

    def fake_run_backtest(settings, *, strategy, strategy_family=None, **_kwargs):
        calls.append((strategy_family or "none", strategy.execution.slippage_bps))
        return _fake_backtest_payload(
            "SOLUSDT",
            "1h",
            strategy.execution.slippage_bps,
            strategy_family or "none",
        )

    monkeypatch.setattr(alpha_research, "run_backtest", fake_run_backtest)

    payload = alpha_research.run_aggressive_alpha_research(
        SimpleNamespace(),
        config_path=config_path,
        output_dir=tmp_path / "out",
    )

    assert payload["stress_test"]["stressed_cohort_count"] == 1
    stressed = [row for row in payload["rows"] if row["stress_tested"]]
    unstressed = [row for row in payload["rows"] if not row["stress_tested"]]
    assert len(stressed) == 1
    assert stressed[0]["promotion_eligible"] is True
    assert unstressed
    assert all(row["promotion_eligible"] is False for row in unstressed)
    assert any(row.get("promotion_blocker") == "awaiting-slippage-stress" for row in unstressed)
    assert calls.count(("breakout", 8.0)) == 1


def test_alpha_research_blocks_high_stop_loss_ratio_promotion(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "strict-stop.yaml"
    config_path.write_text(
        """
strategy:
  strategy_config: strategy-aggressive-alpha-research.yaml
universe:
  include_top_futures_volume: false
  symbols: [SOLUSDT]
  intervals: [1h]
  limit: 320
ranking:
  min_trades: 8
  strategy_families: [breakout]
  slippage_bps_cases: [3.0]
  stress_top_n: 0
  min_profit_factor: 1.2
  min_win_rate: 40
  max_stop_loss_ratio: 40
safety:
  mainnet_live_allowed: false
""",
        encoding="utf-8",
    )

    def high_stop_backtest(settings, *, strategy, strategy_family=None, **_kwargs):
        payload = _fake_backtest_payload(
            "SOLUSDT",
            "1h",
            strategy.execution.slippage_bps,
            strategy_family or "breakout",
        )
        trades = payload["summary"]["trades"]
        for item in trades[:6]:
            item["exit_reason"] = "stop_loss"
        for item in trades[6:]:
            item["exit_reason"] = "take_profit"
        payload["summary"]["win_rate"] = 70.0
        payload["summary"]["profit_factor"] = 2.0
        payload["summary"]["total_return_pct"] = 10.0
        return payload

    monkeypatch.setattr(alpha_research, "run_backtest", high_stop_backtest)

    payload = alpha_research.run_aggressive_alpha_research(
        SimpleNamespace(),
        config_path=config_path,
        output_dir=tmp_path / "out",
    )

    assert payload["rows"][0]["stop_loss_ratio"] == 60.0
    assert payload["rows"][0]["promotion_eligible"] is False


def test_alpha_research_family_override_and_progress_file(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "family-override.yaml"
    config_path.write_text(
        """
strategy:
  strategy_config: strategy-aggressive-alpha-research.yaml
universe:
  include_top_futures_volume: false
  symbols: [BNBUSDT]
  intervals: [4h]
  limit: 320
ranking:
  min_trades: 8
  strategy_families: [breakout, trend_continuation, mean_reversion]
  slippage_bps_cases: [3.0]
  stress_top_n: 0
  min_profit_factor: 1.2
  min_win_rate: 40
  max_stop_loss_ratio: 70
safety:
  mainnet_live_allowed: false
""",
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_run_backtest(settings, *, strategy, symbol, interval, strategy_family=None, **_kwargs):
        calls.append(str(strategy_family))
        return _fake_backtest_payload(
            symbol,
            interval,
            strategy.execution.slippage_bps,
            strategy_family or "none",
        )

    monkeypatch.setattr(alpha_research, "run_backtest", fake_run_backtest)

    payload = alpha_research.run_aggressive_alpha_research(
        SimpleNamespace(),
        config_path=config_path,
        output_dir=tmp_path / "out",
        family_overrides=["trend_continuation"],
    )

    assert payload["strategy_families"] == ["trend_continuation"]
    assert calls == ["trend_continuation"]
    progress_path = Path(payload["progress_path"])
    assert progress_path.exists()
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["phase"] == "complete"
    assert progress["completed_base_cohorts"] == 1
    assert progress["total_base_cohorts"] == 1


def test_research_summary_does_not_meet_target_without_promotion_sample() -> None:
    summary = alpha_research._research_performance_summary(
        [
            {
                "symbol": "SOLUSDT",
                "strategy_family": "mean_reversion",
                "trade_count": 1,
                "win_rate": 100.0,
                "stop_loss_ratio": 0.0,
                "partial_tp_then_stop_ratio": 100.0,
                "profit_factor": "inf",
                "total_return_pct": 0.25,
                "promotion_eligible": False,
            }
        ],
        thresholds={
            "min_trades": 30.0,
            "min_profit_factor": 1.35,
            "min_win_rate": 65.0,
            "max_stop_loss_ratio": 35.0,
        },
        target_metrics={
            "win_rate": 90.0,
            "max_stop_loss_ratio": 15.0,
        },
    )

    assert summary["weighted_win_rate"] == 100.0
    assert summary["weighted_stop_loss_ratio"] == 0.0
    assert summary["meets_target"] is False
    assert summary["execution_recommendation"] == "block_new_entries_and_continue_research"


def test_research_summary_can_use_advisory_win_rate_for_payoff_bot_lane() -> None:
    summary = alpha_research._research_performance_summary(
        [
            {
                "symbol": "SOLUSDT",
                "interval": "4h",
                "strategy_family": "trend_continuation",
                "trade_count": 188,
                "win_rate": 32.45,
                "stop_loss_ratio": 48.94,
                "profit_factor": 1.3851,
                "expectancy_r": 0.2037,
                "payoff_ratio": 2.8836,
                "total_return_pct": 24.5799,
                "promotion_eligible": True,
            }
        ],
        thresholds={
            "min_trades": 100.0,
            "min_profit_factor": 1.25,
            "min_win_rate": 45.0,
            "max_stop_loss_ratio": 55.0,
            "min_expectancy_r": 0.05,
            "min_payoff_ratio": 1.2,
        },
        target_metrics={
            "win_rate": 45.0,
            "max_stop_loss_ratio": 55.0,
            "expectancy_r": 0.05,
            "payoff_ratio": 1.2,
        },
        enforce_win_rate_gate=False,
    )

    assert summary["target_metrics"]["win_rate_mode"] == "advisory"
    assert summary["meets_target"] is True
    assert summary["by_symbol"][0]["health"] == "ok"


def test_research_summary_ignores_zero_trade_rows_in_pf_average() -> None:
    summary = alpha_research._research_performance_summary(
        [
            {
                "symbol": "TRXUSDT",
                "interval": "4h",
                "strategy_family": "mean_reversion",
                "trade_count": 8,
                "win_rate": 87.5,
                "stop_loss_ratio": 12.5,
                "partial_tp_then_stop_ratio": 87.5,
                "profit_factor": 3.0,
                "total_return_pct": 0.25,
                "promotion_eligible": False,
            },
            {
                "symbol": "TRXUSDT",
                "interval": "4h",
                "strategy_family": "liquidity_reclaim",
                "trade_count": 0,
                "win_rate": 0.0,
                "stop_loss_ratio": 0.0,
                "partial_tp_then_stop_ratio": 0.0,
                "profit_factor": 0.0,
                "total_return_pct": 0.0,
                "promotion_eligible": False,
            },
        ],
        thresholds={
            "min_trades": 100.0,
            "min_profit_factor": 1.5,
            "min_win_rate": 80.0,
            "max_stop_loss_ratio": 20.0,
        },
        target_metrics={
            "win_rate": 80.0,
            "max_stop_loss_ratio": 20.0,
        },
    )

    assert summary["finite_avg_profit_factor"] == 3.0
    assert summary["by_symbol"][0]["finite_avg_profit_factor"] == 3.0
    assert summary["by_family"][0]["strategy_family"] == "mean_reversion"
    assert summary["meets_target"] is False


def test_rank_score_penalizes_lucky_low_sample_cohort() -> None:
    lucky_low_sample = {
        "trade_count": 2,
        "total_return_pct": 8.0,
        "max_drawdown_pct": 0.0,
        "profit_factor": "inf",
        "trades": [{"pnl_pct": 4.0}, {"pnl_pct": 4.0}],
    }
    robust_sample = {
        "trade_count": 12,
        "total_return_pct": 6.0,
        "max_drawdown_pct": 2.0,
        "profit_factor": 1.8,
        "trades": [{"pnl_pct": 0.5} for _ in range(12)],
    }

    lucky_score = alpha_research._rank_score(
        summary=lucky_low_sample,
        robustness={"status": "insufficient_sample", "passed": False, "folds": []},
        slippage_resilience=0.0,
        weights={},
        min_trades=8,
    )
    robust_score = alpha_research._rank_score(
        summary=robust_sample,
        robustness={
            "status": "passed",
            "passed": True,
            "folds": [
                {"total_return_pct": 1.0, "profit_factor": 1.2},
                {"total_return_pct": 2.0, "profit_factor": 1.3},
            ],
        },
        slippage_resilience=0.8,
        weights={},
        min_trades=8,
    )

    assert lucky_score < robust_score


def test_slippage_variant_only_changes_execution_slippage() -> None:
    strategy = load_strategy_config("config/strategy-aggressive-alpha-research.yaml")

    variant = alpha_research._slippage_variant(strategy, 15.0)

    assert variant.execution.slippage_bps == 15.0
    assert strategy.execution.slippage_bps == 3.0
    assert replace(variant, execution=strategy.execution).execution.slippage_bps == 3.0


def test_symbol_entry_profile_can_block_weak_router_subfamilies() -> None:
    spec = SymbolStrategySpec(
        symbol="TRXUSDT",
        primary_family="ai_family_router",
        allowed_families=("ai_family_router",),
        blocked_families=(),
        interval_families={"4h": ("ai_family_router",)},
        interval_family_sides={},
        interval="4h",
        execution_lane="paper_research_only",
        route_id="trx-test",
        asset_class="alt",
        promotion=SymbolPromotionSpec(
            min_trades=100,
            min_profit_factor=1.25,
            min_win_rate=45.0,
            max_stop_loss_ratio=55.0,
            min_expectancy_r=0.05,
            min_payoff_ratio=1.2,
        ),
        thesis="test",
        risk_filters=(),
        entry_filters={"blocked_routed_families": ["liquidity_reclaim"]},
        strategy_overrides={},
    )
    gate = alpha_research._symbol_entry_profile_filter(symbol_strategy_spec=spec)

    allowed, reason = gate(
        {"plus_di": 30.0, "minus_di": 10.0, "obv_zscore_20": 0.0, "volume_zscore_20": 0.0, "adx": 20.0},
        {},
        {"recommended_action": "BUY", "routed_strategy_family": "liquidity_reclaim"},
        1,
    )

    assert allowed is False
    assert reason == "symbol-entry-profile-blocked-routed-family"
