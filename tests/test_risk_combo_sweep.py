from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import binance_quant_control.risk_combo_sweep as sweep
from binance_quant_control.asset_routing import resolve_symbol_route
from binance_quant_control.strategy import load_strategy_config


def test_risk_combo_sweep_reports_recovery_candidates_without_writing_reviews(
    monkeypatch,
    tmp_path,
) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setattr(sweep, "STATE_DIR", state_dir)
    monkeypatch.setattr(sweep, "RISK_COMBO_SWEEP_DIR", state_dir / "risk-combo-sweeps")
    monkeypatch.setattr(sweep, "ensure_runtime_dirs", lambda: state_dir.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(
        sweep,
        "load_settings",
        lambda: SimpleNamespace(live_trading_enabled=False, use_testnet=True),
    )
    monkeypatch.setattr(
        sweep,
        "_collect_news",
        lambda skip_news: {
            "available": True,
            "risk": {"risk_level": "normal", "bias": "neutral", "high_impact_count": 0},
            "items": [],
        },
    )
    monkeypatch.setattr(
        sweep,
        "route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": True, "reasons": ["pytest"]},
    )

    route = resolve_symbol_route("BTCUSDT")
    strategy = load_strategy_config(Path("config/strategy-btc-volatility.yaml"))
    frame = pd.DataFrame(
        {
            "open": [1.0] * 900,
            "high": [1.1] * 900,
            "low": [0.9] * 900,
            "close": [1.0] * 900,
            "volume_zscore_20": [0.0] * 900,
            "realized_vol_20": [0.5] * 900,
            "atr_14": [0.1] * 900,
        },
        index=pd.date_range("2026-01-01", periods=900, freq="4h", tz="UTC"),
    )
    dataset = sweep.SweepDataset(
        route=route,
        strategy=strategy,
        requested_symbol="BTCUSDT",
        source_symbol="BTCUSDT",
        market="futures",
        interval="4h",
        frame=frame,
        fetch_log=[{"status": "ok", "candles": len(frame)}],
    )
    monkeypatch.setattr(
        sweep,
        "fetch_dataset",
        lambda settings, **kwargs: dataset,
    )
    monkeypatch.setattr(
        sweep,
        "_grid_values",
        lambda base, mode="fast": {
            "min_adx": (20.0,),
            "min_convergence": (0.7,),
            "atr_stop_multiple": (1.8,),
            "primary_tp_multiple": (1.5,),
            "exit_profile": ("balanced",),
            "news_veto_mode": ("off",),
            "side_policy_mode": ("baseline",),
            "structure_policy_mode": ("baseline",),
            "historical_policy_mode": ("off",),
        },
    )
    monkeypatch.setattr(
        sweep,
        "simulate_backtest",
        lambda *args, **kwargs: {
            "trade_count": 4,
            "wins": 3,
            "losses": 1,
            "win_rate": 75.0,
            "stop_loss_ratio": 25.0,
            "partial_tp_then_stop_ratio": 0.0,
            "avg_pnl_pct": 0.2,
            "avg_r": 0.4,
            "ending_equity": 1.01,
            "total_return_pct": 1.0,
            "max_drawdown_pct": 1.5,
            "profit_factor": 1.25,
            "expectancy_r": 0.18,
            "avg_win_r": 1.2,
            "avg_loss_r": 0.8,
            "payoff_ratio": 1.5,
            "break_even_win_rate": 40.0,
            "expectancy_edge_points": 35.0,
            "loss_streak": 1,
            "entry_veto_count": 0,
            "entry_veto_reasons": {},
            "trades": [],
        },
    )

    payload = sweep.run_risk_combo_sweep(
        routes=["btc-core"],
        symbols=[],
        limit=900,
        target_profit_factor=0.8,
        min_test_trades=3,
        min_win_rate=70.0,
        max_stop_loss_ratio=30.0,
        min_expectancy_r=0.10,
        min_payoff_ratio=1.15,
        max_symbols_per_route=1,
        skip_news=True,
        top_n=5,
    )

    assert payload["status"] == "ok"
    assert payload["safety"]["writes_closed_trade_reviews"] is False
    assert payload["safety"]["clears_route_quarantine"] is False
    assert payload["aggregate"]["recovery_candidate_count"] >= 1
    assert payload["aggregate"]["robust_recovery_candidate_count"] >= 1
    assert payload["route_side_risk"]["modes"] == ["off", "route-side-veto"]
    runtime = payload["runtime_observability"]
    assert runtime["status"] == "completed"
    assert runtime["dataset_count"] == payload["aggregate"]["dataset_count"]
    assert runtime["configs_tested"] == payload["aggregate"]["configs_tested"]
    assert runtime["total_seconds"] >= 0.0
    assert runtime["fetch_seconds"] >= 0.0
    assert runtime["grid_evaluation_seconds"] >= 0.0
    assert runtime["walk_forward_validations"] == payload["aggregate"]["walk_forward_validations"]
    assert payload["best_by_route"]["btc-core"]["recovery_gate"]["passed"] is True
    assert payload["best_by_route"]["btc-core"]["robust_recovery_gate"]["passed"] is True
    assert Path(payload["report_path"]).exists()


def test_risk_combo_sweep_blocks_recovery_when_win_stop_gate_fails(
    monkeypatch,
    tmp_path,
) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setattr(sweep, "STATE_DIR", state_dir)
    monkeypatch.setattr(sweep, "RISK_COMBO_SWEEP_DIR", state_dir / "risk-combo-sweeps")
    monkeypatch.setattr(sweep, "ensure_runtime_dirs", lambda: state_dir.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(
        sweep,
        "load_settings",
        lambda: SimpleNamespace(live_trading_enabled=False, use_testnet=True),
    )
    monkeypatch.setattr(
        sweep,
        "_collect_news",
        lambda skip_news: {
            "available": False,
            "risk": {"risk_level": "unknown", "bias": "neutral", "high_impact_count": 0},
            "items": [],
        },
    )
    monkeypatch.setattr(
        sweep,
        "route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": True, "reasons": []},
    )

    route = resolve_symbol_route("BTCUSDT")
    strategy = load_strategy_config(Path("config/strategy-btc-volatility.yaml"))
    frame = pd.DataFrame(
        {
            "open": [1.0] * 900,
            "high": [1.1] * 900,
            "low": [0.9] * 900,
            "close": [1.0] * 900,
            "volume_zscore_20": [0.0] * 900,
            "realized_vol_20": [0.5] * 900,
            "atr_14": [0.1] * 900,
        },
        index=pd.date_range("2026-01-01", periods=900, freq="4h", tz="UTC"),
    )
    dataset = sweep.SweepDataset(
        route=route,
        strategy=strategy,
        requested_symbol="BTCUSDT",
        source_symbol="BTCUSDT",
        market="futures",
        interval="4h",
        frame=frame,
        fetch_log=[{"status": "ok", "candles": len(frame)}],
    )
    monkeypatch.setattr(sweep, "fetch_dataset", lambda settings, **kwargs: dataset)
    monkeypatch.setattr(
        sweep,
        "_grid_values",
        lambda base, mode="fast": {
            "min_adx": (20.0,),
            "min_convergence": (0.7,),
            "atr_stop_multiple": (1.8,),
            "primary_tp_multiple": (1.5,),
            "exit_profile": ("balanced",),
            "news_veto_mode": ("off",),
            "side_policy_mode": ("baseline",),
            "structure_policy_mode": ("baseline",),
            "historical_policy_mode": ("off",),
        },
    )
    monkeypatch.setattr(
        sweep,
        "simulate_backtest",
        lambda *args, **kwargs: {
            "trade_count": 120,
            "wins": 102,
            "losses": 18,
            "win_rate": 85.0,
            "stop_loss_ratio": 15.0,
            "partial_tp_then_stop_ratio": 0.0,
            "avg_pnl_pct": 0.2,
            "avg_r": 0.4,
            "ending_equity": 1.01,
            "total_return_pct": 1.0,
            "max_drawdown_pct": 1.5,
            "profit_factor": 2.0,
            "expectancy_r": 0.24,
            "avg_win_r": 1.2,
            "avg_loss_r": 0.8,
            "payoff_ratio": 1.5,
            "break_even_win_rate": 40.0,
            "expectancy_edge_points": 45.0,
            "loss_streak": 1,
            "entry_veto_count": 0,
            "entry_veto_reasons": {},
            "trades": [],
        },
    )

    payload = sweep.run_risk_combo_sweep(
        routes=["btc-core"],
        symbols=[],
        limit=900,
        target_profit_factor=1.5,
        min_test_trades=100,
        min_win_rate=90.0,
        max_stop_loss_ratio=10.0,
        max_symbols_per_route=1,
        skip_news=True,
        top_n=5,
    )

    best = payload["best_by_route"]["btc-core"]
    assert payload["aggregate"]["recovery_candidate_count"] == 0
    assert payload["aggregate"]["robust_recovery_candidate_count"] == 0
    assert best["recovery_gate"]["passed"] is False
    assert "test-win-rate-below-target" in best["recovery_gate"]["reasons"]
    assert "test-stop-loss-ratio-above-target" in best["recovery_gate"]["reasons"]


def test_risk_combo_sweep_blocks_high_win_low_payoff_recovery(
    monkeypatch,
    tmp_path,
) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setattr(sweep, "STATE_DIR", state_dir)
    monkeypatch.setattr(sweep, "RISK_COMBO_SWEEP_DIR", state_dir / "risk-combo-sweeps")
    monkeypatch.setattr(sweep, "ensure_runtime_dirs", lambda: state_dir.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(
        sweep,
        "load_settings",
        lambda: SimpleNamespace(live_trading_enabled=False, use_testnet=True),
    )
    monkeypatch.setattr(
        sweep,
        "_collect_news",
        lambda skip_news: {
            "available": False,
            "risk": {"risk_level": "unknown", "bias": "neutral", "high_impact_count": 0},
            "items": [],
        },
    )
    monkeypatch.setattr(
        sweep,
        "route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": True, "reasons": []},
    )

    route = resolve_symbol_route("BTCUSDT")
    strategy = load_strategy_config(Path("config/strategy-btc-volatility.yaml"))
    frame = pd.DataFrame(
        {
            "open": [1.0] * 900,
            "high": [1.1] * 900,
            "low": [0.9] * 900,
            "close": [1.0] * 900,
            "volume_zscore_20": [0.0] * 900,
            "realized_vol_20": [0.5] * 900,
            "atr_14": [0.1] * 900,
        },
        index=pd.date_range("2026-01-01", periods=900, freq="4h", tz="UTC"),
    )
    dataset = sweep.SweepDataset(
        route=route,
        strategy=strategy,
        requested_symbol="BTCUSDT",
        source_symbol="BTCUSDT",
        market="futures",
        interval="4h",
        frame=frame,
        fetch_log=[{"status": "ok", "candles": len(frame)}],
    )
    monkeypatch.setattr(sweep, "fetch_dataset", lambda settings, **kwargs: dataset)
    monkeypatch.setattr(
        sweep,
        "_grid_values",
        lambda base, mode="fast": {
            "min_adx": (20.0,),
            "min_convergence": (0.7,),
            "atr_stop_multiple": (1.8,),
            "primary_tp_multiple": (1.5,),
            "exit_profile": ("balanced",),
            "news_veto_mode": ("off",),
            "side_policy_mode": ("baseline",),
            "structure_policy_mode": ("baseline",),
            "historical_policy_mode": ("off",),
        },
    )
    monkeypatch.setattr(
        sweep,
        "simulate_backtest",
        lambda *args, **kwargs: {
            "trade_count": 120,
            "wins": 96,
            "losses": 24,
            "win_rate": 80.0,
            "stop_loss_ratio": 20.0,
            "partial_tp_then_stop_ratio": 0.0,
            "avg_pnl_pct": 0.02,
            "avg_r": 0.002,
            "expectancy_r": 0.002,
            "avg_win_r": 0.2563,
            "avg_loss_r": 1.0,
            "payoff_ratio": 0.2563,
            "break_even_win_rate": 79.6,
            "expectancy_edge_points": 0.4,
            "ending_equity": 1.001,
            "total_return_pct": 0.1,
            "max_drawdown_pct": 1.5,
            "profit_factor": 1.0252,
            "loss_streak": 1,
            "entry_veto_count": 0,
            "entry_veto_reasons": {},
            "trades": [],
        },
    )

    payload = sweep.run_risk_combo_sweep(
        routes=["btc-core"],
        symbols=[],
        limit=900,
        target_profit_factor=1.0,
        min_test_trades=100,
        min_win_rate=65.0,
        max_stop_loss_ratio=35.0,
        min_expectancy_r=0.10,
        min_payoff_ratio=1.15,
        max_symbols_per_route=1,
        skip_news=True,
        top_n=5,
    )

    best = payload["best_by_route"]["btc-core"]
    assert payload["aggregate"]["recovery_candidate_count"] == 0
    assert best["test"]["expectancy_r"] == 0.002
    assert best["test"]["payoff_ratio"] == 0.2563
    assert "test-expectancy-r-below-target" in best["recovery_gate"]["reasons"]
    assert "test-payoff-ratio-below-target" in best["recovery_gate"]["reasons"]


def test_risk_combo_sweep_honors_max_configs_budget(
    monkeypatch,
    tmp_path,
) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setattr(sweep, "STATE_DIR", state_dir)
    monkeypatch.setattr(sweep, "RISK_COMBO_SWEEP_DIR", state_dir / "risk-combo-sweeps")
    monkeypatch.setattr(sweep, "ensure_runtime_dirs", lambda: state_dir.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(
        sweep,
        "load_settings",
        lambda: SimpleNamespace(live_trading_enabled=False, use_testnet=True),
    )
    monkeypatch.setattr(
        sweep,
        "_collect_news",
        lambda skip_news: {
            "available": False,
            "risk": {"risk_level": "unknown", "bias": "neutral", "high_impact_count": 0},
            "items": [],
        },
    )
    monkeypatch.setattr(
        sweep,
        "route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": True, "reasons": []},
    )

    route = resolve_symbol_route("BTCUSDT")
    strategy = load_strategy_config(Path("config/strategy-btc-volatility.yaml"))
    frame = pd.DataFrame(
        {
            "open": [1.0] * 900,
            "high": [1.1] * 900,
            "low": [0.9] * 900,
            "close": [1.0] * 900,
            "volume_zscore_20": [0.0] * 900,
            "realized_vol_20": [0.5] * 900,
            "atr_14": [0.1] * 900,
        },
        index=pd.date_range("2026-01-01", periods=900, freq="4h", tz="UTC"),
    )
    dataset = sweep.SweepDataset(
        route=route,
        strategy=strategy,
        requested_symbol="BTCUSDT",
        source_symbol="BTCUSDT",
        market="futures",
        interval="4h",
        frame=frame,
        fetch_log=[{"status": "ok", "candles": len(frame)}],
    )
    monkeypatch.setattr(sweep, "fetch_dataset", lambda settings, **kwargs: dataset)
    monkeypatch.setattr(
        sweep,
        "_grid_values",
        lambda base, mode="fast": {
            "min_adx": (20.0, 24.0, 28.0),
            "min_convergence": (0.7, 0.8, 0.9),
            "atr_stop_multiple": (1.8,),
            "primary_tp_multiple": (1.5,),
            "exit_profile": ("balanced",),
            "news_veto_mode": ("off",),
            "side_policy_mode": ("baseline",),
            "structure_policy_mode": ("baseline",),
            "historical_policy_mode": ("off",),
        },
    )
    calls = 0

    def fake_backtest(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {
            "trade_count": 20,
            "wins": 14,
            "losses": 6,
            "win_rate": 70.0,
            "stop_loss_ratio": 30.0,
            "partial_tp_then_stop_ratio": 0.0,
            "avg_pnl_pct": 0.2,
            "avg_r": 0.2,
            "expectancy_r": 0.12,
            "avg_win_r": 1.2,
            "avg_loss_r": 0.8,
            "payoff_ratio": 1.5,
            "break_even_win_rate": 40.0,
            "expectancy_edge_points": 30.0,
            "ending_equity": 1.01,
            "total_return_pct": 1.0,
            "max_drawdown_pct": 1.5,
            "profit_factor": 1.6,
            "loss_streak": 1,
            "entry_veto_count": 0,
            "entry_veto_reasons": {},
            "trades": [],
        }

    monkeypatch.setattr(sweep, "simulate_backtest", fake_backtest)

    payload = sweep.run_risk_combo_sweep(
        routes=["btc-core"],
        symbols=[],
        limit=900,
        max_symbols_per_route=1,
        max_configs=2,
        skip_news=True,
        top_n=5,
    )

    assert payload["aggregate"]["configs_tested"] == 2
    assert payload["aggregate"]["max_configs_per_dataset"] == 2
    assert calls >= 6


def test_risk_combo_sweep_applies_target_side_and_interval_controls(
    monkeypatch,
    tmp_path,
) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setattr(sweep, "STATE_DIR", state_dir)
    monkeypatch.setattr(sweep, "RISK_COMBO_SWEEP_DIR", state_dir / "risk-combo-sweeps")
    monkeypatch.setattr(sweep, "ensure_runtime_dirs", lambda: state_dir.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(
        sweep,
        "load_settings",
        lambda: SimpleNamespace(live_trading_enabled=False, use_testnet=True),
    )
    monkeypatch.setattr(
        sweep,
        "_collect_news",
        lambda skip_news: {
            "available": False,
            "risk": {"risk_level": "unknown", "bias": "neutral", "high_impact_count": 0},
            "items": [],
        },
    )
    monkeypatch.setattr(
        sweep,
        "route_quarantine_status",
        lambda route_id: {"route_id": route_id, "quarantined": True, "reasons": []},
    )

    route = resolve_symbol_route("TRXUSDT")
    strategy = load_strategy_config(route.strategy_config)
    frame = pd.DataFrame(
        {
            "open": [1.0] * 900,
            "high": [1.1] * 900,
            "low": [0.9] * 900,
            "close": [1.0] * 900,
            "volume_zscore_20": [0.0] * 900,
            "realized_vol_20": [0.5] * 900,
            "atr_14": [0.1] * 900,
        },
        index=pd.date_range("2026-01-01", periods=900, freq="15min", tz="UTC"),
    )
    captured_fetch: dict[str, object] = {}

    def fake_fetch_dataset(settings, **kwargs):
        captured_fetch.update(kwargs)
        return sweep.SweepDataset(
            route=route,
            strategy=strategy,
            requested_symbol=kwargs["requested_symbol"],
            source_symbol=kwargs["requested_symbol"],
            market="futures",
            interval=kwargs["interval"],
            frame=frame,
            fetch_log=[{"status": "ok", "interval": kwargs["interval"], "candles": len(frame)}],
        )

    monkeypatch.setattr(sweep, "fetch_dataset", fake_fetch_dataset)
    monkeypatch.setattr(
        sweep,
        "_grid_values",
        lambda base, mode="fast": {
            "min_adx": (20.0,),
            "min_convergence": (0.7,),
            "atr_stop_multiple": (1.8,),
            "primary_tp_multiple": (1.5,),
            "exit_profile": ("balanced",),
            "news_veto_mode": ("off",),
            "side_policy_mode": ("baseline",),
            "structure_policy_mode": ("baseline",),
            "historical_policy_mode": ("off",),
        },
    )
    captured_filters = []

    def fake_backtest(_frame, _market, _strategy, *, entry_filter=None, **_kwargs):
        captured_filters.append(entry_filter)
        if entry_filter is not None:
            previous = pd.Series(
                {
                    "adx": 30.0,
                    "minus_di": 30.0,
                    "plus_di": 10.0,
                    "volume_zscore_20": 0.0,
                    "realized_vol_20": 0.8,
                    "atr_14": 1.0,
                    "high": 12.0,
                    "low": 11.0,
                }
            )
            buy_allowed, buy_reason = entry_filter(previous, previous, {"recommended_action": "BUY"}, 1)
            sell_allowed = entry_filter(previous, previous, {"recommended_action": "SELL"}, 2)
            assert buy_allowed is False
            assert buy_reason == "target-side-sell-only"
            assert sell_allowed is True
        return {
            "trade_count": 20,
            "wins": 12,
            "losses": 8,
            "win_rate": 60.0,
            "stop_loss_ratio": 40.0,
            "partial_tp_then_stop_ratio": 0.0,
            "avg_pnl_pct": 0.2,
            "avg_r": 0.2,
            "expectancy_r": 0.12,
            "avg_win_r": 1.2,
            "avg_loss_r": 0.8,
            "payoff_ratio": 1.5,
            "break_even_win_rate": 40.0,
            "expectancy_edge_points": 20.0,
            "ending_equity": 1.01,
            "total_return_pct": 1.0,
            "max_drawdown_pct": 1.5,
            "profit_factor": 1.6,
            "loss_streak": 1,
            "entry_veto_count": 0,
            "entry_veto_reasons": {},
            "trades": [],
        }

    monkeypatch.setattr(sweep, "simulate_backtest", fake_backtest)

    payload = sweep.run_risk_combo_sweep(
        symbols=["TRXUSDT"],
        limit=900,
        target_side="SELL",
        target_interval="15m",
        max_configs=1,
        max_walk_forward_validations=1,
        skip_news=True,
        top_n=1,
    )

    assert captured_fetch["interval"] == "15m"
    assert payload["config"]["target_side"] == "SELL"
    assert payload["config"]["target_interval"] == "15m"
    assert payload["aggregate"]["target_side"] == "SELL"
    assert payload["best_by_symbol"]["TRXUSDT"]["interval"] == "15m"
    assert payload["best_by_symbol"]["TRXUSDT"]["params"]["target_side"] == "SELL"
    assert captured_filters


def test_bounded_grid_keeps_exit_profile_coverage() -> None:
    grid = {
        "min_adx": (12.0, 16.0, 20.0),
        "min_convergence": (0.6, 0.7, 0.8),
        "atr_stop_multiple": (1.2,),
        "primary_tp_multiple": (1.0,),
        "exit_profile": ("balanced", "payoff_runner"),
        "news_veto_mode": ("off",),
        "side_policy_mode": ("baseline",),
        "structure_policy_mode": ("baseline",),
        "historical_policy_mode": ("off",),
        "route_side_policy_mode": ("off",),
    }

    combos = sweep._bounded_grid_combinations(grid, max_configs=4)
    profiles = {combo[4] for combo in combos}

    assert len(combos) == 4
    assert profiles == {"balanced", "payoff_runner"}


def test_risk_combo_matrix_report_marks_promising_surface_without_promotion(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setattr(sweep, "STATE_DIR", state_dir)
    monkeypatch.setattr(sweep, "RISK_COMBO_MATRIX_DIR", state_dir / "risk-combo-matrix")
    monkeypatch.setattr(sweep, "ensure_runtime_dirs", lambda: state_dir.mkdir(parents=True, exist_ok=True))
    good_report = tmp_path / "buy-4h.json"
    weak_report = tmp_path / "sell-4h.json"
    base_payload = {
        "status": "ok",
        "safety": {"live_trading_enabled": False},
        "best_by_route": {
            "trx-mean-reversion": {
                "route_id": "trx-mean-reversion",
                "requested_symbol": "TRXUSDT",
                "source_symbol": "TRXUSDT",
                "interval": "4h",
                "params": {"target_side": "BUY"},
                "full": {
                    "trade_count": 21,
                    "wins": 9,
                    "losses": 12,
                    "win_rate": 42.86,
                    "loss_streak": 0,
                    "profit_factor": 1.755,
                    "expectancy_r": 0.0948,
                    "avg_win_r": 1.9,
                    "avg_loss_r": 0.8,
                    "payoff_ratio": 2.375,
                    "break_even_win_rate": 29.63,
                    "expectancy_edge_points": 13.23,
                    "max_drawdown_pct": 0.7662,
                    "stop_loss_ratio": 38.1,
                    "partial_tp_then_stop_ratio": 20.0,
                },
                "train": {
                    "trade_count": 6,
                    "win_rate": 50.0,
                    "profit_factor": 3.5389,
                    "expectancy_r": 0.2898,
                    "payoff_ratio": 2.1,
                },
                "test": {
                    "trade_count": 5,
                    "wins": 4,
                    "losses": 1,
                    "win_rate": 80.0,
                    "profit_factor": 21.4049,
                    "expectancy_r": 0.2789,
                    "avg_win_r": 1.2,
                    "avg_loss_r": 0.4,
                    "payoff_ratio": 3.0,
                    "break_even_win_rate": 25.0,
                    "expectancy_edge_points": 55.0,
                    "max_drawdown_pct": 0.0406,
                    "stop_loss_ratio": 0.0,
                    "partial_tp_then_stop_ratio": 0.0,
                },
                "walk_forward": {"window_count": 0, "min_profit_factor": 0.0, "min_expectancy_r": 0.0},
                "recovery_gate": {"passed": False, "reasons": ["test-trade-count-too-low"]},
                "robust_recovery_gate": {"passed": False, "reasons": ["walk-forward-window-count-too-low"]},
            }
        },
        "aggregate": {"target_side": "BUY", "target_interval": "4h"},
    }
    good_report.write_text(json.dumps(base_payload), encoding="utf-8")
    weak_payload = json.loads(json.dumps(base_payload))
    weak_payload["aggregate"] = {"target_side": "SELL", "target_interval": "4h"}
    weak_candidate = weak_payload["best_by_route"]["trx-mean-reversion"]
    weak_candidate["params"] = {"target_side": "SELL"}
    weak_candidate["full"] = {
        "trade_count": 3,
        "profit_factor": 0.0,
        "expectancy_r": -0.3714,
        "max_drawdown_pct": 0.6671,
        "stop_loss_ratio": 100.0,
    }
    weak_candidate["test"] = {
        "trade_count": 0,
        "profit_factor": 0.0,
        "expectancy_r": 0.0,
        "max_drawdown_pct": 0.0,
        "stop_loss_ratio": 0.0,
    }
    weak_report.write_text(json.dumps(weak_payload), encoding="utf-8")

    payload = sweep.build_risk_combo_matrix_report(
        report_paths=[good_report, weak_report],
        output_dir=tmp_path / "matrix",
    )

    assert payload["safety"]["mainnet_live_allowed"] is False
    assert payload["promising_surface_count"] == 1
    assert payload["emerging_positive_lead_count"] == 0
    assert payload["robust_surface_count"] == 0
    assert payload["side_summary"]["buy"]["promising_surface_count"] == 1
    assert payload["side_summary"]["buy"]["emerging_positive_lead_count"] == 0
    assert payload["side_summary"]["buy"]["robust_surface_count"] == 0
    assert payload["side_summary"]["buy"]["status"] == "promising_research_only"
    assert payload["side_summary"]["sell"]["surface_count"] == 1
    assert payload["side_summary"]["sell"]["promising_surface_count"] == 0
    assert payload["side_summary"]["sell"]["emerging_positive_lead_count"] == 0
    assert payload["side_summary"]["sell"]["status"] == "missing_or_negative_expectancy"
    assert payload["horizon_summary"]["medium"]["promising_surface_count"] == 1
    assert payload["horizon_summary"]["medium"]["emerging_positive_lead_count"] == 0
    assert payload["horizon_summary"]["short"]["surface_count"] == 0
    assert payload["horizon_summary"]["long"]["surface_count"] == 0
    scorecard = payload["objective_scorecard"]
    assert scorecard["buy_sell_stability"]["baseline_score"] == 45
    assert scorecard["buy_sell_stability"]["score"] == 62
    assert "sell_promising_surface_missing" in scorecard["buy_sell_stability"]["blockers"]
    assert "robust_surface_missing" in scorecard["buy_sell_stability"]["blockers"]
    assert scorecard["long_term_expectancy"]["baseline_score"] == 20
    assert scorecard["long_term_expectancy"]["score"] == 32
    assert scorecard["long_term_expectancy"]["evidence"]["best_promising_full_profit_factor"] == 1.755
    assert scorecard["live_readiness"]["score"] == 0
    assert scorecard["live_readiness"]["evidence"]["mainnet_live_allowed"] is False
    checklist = payload["prompt_to_artifact_checklist"]
    assert checklist["status"] == "incomplete"
    checklist_items = {item["requirement"]: item for item in checklist["items"]}
    assert checklist_items["candidate_not_zero"]["passed"] is True
    assert checklist_items["buy_and_sell_directional_evidence"]["passed"] is False
    assert checklist_items["short_medium_long_horizon_coverage"]["passed"] is False
    assert checklist_items["positive_expectancy_metrics_not_win_rate"]["passed"] is True
    assert checklist_items["robust_promotion_gate_passed"]["passed"] is False
    assert checklist_items["robust_promotion_gate_passed"]["evidence"]["robust_surface_count"] == 0
    assert checklist_items["readiness_stays_zero_and_mainnet_blocked"]["passed"] is True
    assert checklist_items["per_trade_risk_ceiling_2_5pct"]["passed"] is True
    assert checklist_items["no_gate_relaxation_for_forced_trades"]["passed"] is True
    assert checklist_items["score_improvement_visible"]["evidence"]["buy_sell_stability_score"] == 62
    assert "buy_and_sell_directional_evidence" in checklist["missing_requirements"]
    assert "robust_promotion_gate_passed" in checklist["missing_requirements"]
    audit = payload["completion_audit"]
    assert audit["status"] == "incomplete"
    assert "candidate_signal_not_zero" not in audit["missing_requirements"]
    assert "buy_and_sell_have_backtested_promising_surfaces" in audit["missing_requirements"]
    assert "short_medium_long_have_promising_surfaces" in audit["missing_requirements"]
    assert "robust_promotion_gate_passed" in audit["missing_requirements"]
    assert audit["mainnet_live_allowed"] is False
    mainnet_check = next(item for item in audit["checks"] if item["requirement"] == "mainnet_live_blocked")
    assert mainnet_check["passed"] is True
    risk_check = next(item for item in audit["checks"] if item["requirement"] == "per_trade_risk_ceiling_preserved")
    assert risk_check["passed"] is True
    assert risk_check["evidence"]["max_per_trade_risk_pct"] == 0.025
    assert payload["risk_boundary"] == {
        "max_per_trade_risk_pct": 0.025,
        "max_per_trade_risk_percent": 2.5,
        "risk_ceiling_source": "user_policy_and_strategy_stable_risk",
        "applies_to": "all_research_candidates_before_any_promotion",
        "changes_position_sizing": False,
        "opens_orders": False,
        "writes_execution_config": False,
        "mainnet_live_allowed": False,
    }
    repair_plan = payload["negative_surface_repair_plan"]
    assert repair_plan
    sell_repair = next(item for item in repair_plan if item["coverage_type"] == "side" and item["coverage_key"] == "sell")
    assert sell_repair["source_surface"] == "sell_4h"
    assert sell_repair["repair_objective"] == "find_positive_expectancy_without_relaxing_recovery_or_live_gates"
    assert "--target-side SELL" in sell_repair["scout_command"]
    assert "--limit 600" in sell_repair["scout_command"]
    assert "--grid-mode fast" in sell_repair["scout_command"]
    assert sell_repair["cross_symbol_scout_symbols"] == [
        "TRXUSDT",
        "ADAUSDT",
        "AVAXUSDT",
        "NEARUSDT",
        "DOGEUSDT",
        "WIFUSDT",
    ]
    assert "--symbols TRXUSDT,ADAUSDT,AVAXUSDT,NEARUSDT,DOGEUSDT,WIFUSDT" in sell_repair[
        "cross_symbol_scout_command"
    ]
    assert "--target-side SELL" in sell_repair["cross_symbol_scout_command"]
    assert "--min-test-trades 10" in sell_repair["cross_symbol_scout_command"]
    assert "--target-profit-factor 1.0" in sell_repair["cross_symbol_scout_command"]
    assert "--min-expectancy-r 0.0" in sell_repair["cross_symbol_scout_command"]
    assert "--max-stop-loss-ratio 55" in sell_repair["cross_symbol_scout_command"]
    assert sell_repair["multi_interval_scout_intervals"] == ["15m", "30m", "1h"]
    multi_interval_commands = sell_repair["multi_interval_scout_commands"]
    assert [item["target_interval"] for item in multi_interval_commands] == ["15m", "30m", "1h"]
    assert [item["horizon"] for item in multi_interval_commands] == ["short", "short", "medium"]
    for scout in multi_interval_commands:
        assert scout["purpose"] == "scout_adjacent_timeframe_without_relaxing_gates"
        assert "--symbols TRXUSDT,ADAUSDT,AVAXUSDT,NEARUSDT,DOGEUSDT,WIFUSDT" in scout["command"]
        assert "--target-side SELL" in scout["command"]
        assert f"--target-interval {scout['target_interval']}" in scout["command"]
        assert "--limit 600" in scout["command"]
        assert "--grid-mode fast" in scout["command"]
        assert "--min-test-trades 10" in scout["command"]
        assert "--target-profit-factor 1.0" in scout["command"]
        assert "--min-expectancy-r 0.0" in scout["command"]
        assert "--max-stop-loss-ratio 55" in scout["command"]
    assert "--target-side SELL" in sell_repair["interactive_probe_command"]
    assert "--limit 900" in sell_repair["interactive_probe_command"]
    assert "--grid-mode fast" in sell_repair["interactive_probe_command"]
    assert "--min-test-trades 10" in sell_repair["interactive_probe_command"]
    assert "--grid-mode focused" in sell_repair["offline_validation_command"]
    assert "cross_symbol_scout_command" in sell_repair["runtime_guidance"]
    assert "multi_interval_scout_commands" in sell_repair["runtime_guidance"]
    assert sell_repair["guardrails"]["does_not_open_orders"] is True
    assert sell_repair["guardrails"]["does_not_lower_promotion_gates"] is True
    assert sell_repair["guardrails"]["max_per_trade_risk_pct"] == 0.025
    assert sell_repair["guardrails"]["mainnet_live_allowed"] is False
    assert payload["best_surface"]["surface"] == "buy_4h"
    assert payload["best_surface"]["research_status"] == "promising_but_under_validated"
    assert payload["best_surface"]["promotion_eligible"] is False
    assert payload["best_surface"]["full"]["win_rate"] == 42.86
    assert payload["best_surface"]["full"]["payoff_ratio"] == 2.375
    assert payload["best_surface"]["full"]["avg_win_r"] == 1.9
    assert payload["best_surface"]["test"]["payoff_ratio"] == 3.0
    assert payload["best_surface"]["gate_reasons"]["recovery"] == ["test-trade-count-too-low"]
    assert payload["validation_plan"][0]["surface"] == "buy_4h"
    assert "--target-side BUY" in payload["validation_plan"][0]["interactive_probe_command"]
    assert "--target-interval 4h" in payload["validation_plan"][0]["interactive_probe_command"]
    assert "--limit 900" in payload["validation_plan"][0]["interactive_probe_command"]
    assert "--grid-mode fast" in payload["validation_plan"][0]["interactive_probe_command"]
    assert "--min-test-trades 10" in payload["validation_plan"][0]["interactive_probe_command"]
    assert "--max-configs 8" in payload["validation_plan"][0]["interactive_probe_command"]
    assert "--max-walk-forward-validations 1" in payload["validation_plan"][0]["interactive_probe_command"]
    assert "--limit 5000" in payload["validation_plan"][0]["offline_validation_command"]
    assert "--grid-mode focused" in payload["validation_plan"][0]["offline_validation_command"]
    assert "--min-test-trades 30" in payload["validation_plan"][0]["offline_validation_command"]
    assert "--max-walk-forward-validations 6" in payload["validation_plan"][0]["offline_validation_command"]
    assert "scheduled" in payload["validation_plan"][0]["runtime_guidance"]
    assert payload["validation_plan"][0]["promotion_boundary"] == (
        "research_only_does_not_change_live_readiness_or_mainnet_permission"
    )
    assert payload["promotion_boundary"]["requires_robust_recovery_gate"] is True
    assert payload["promotion_boundary"]["max_per_trade_risk_pct"] == 0.025
    assert Path(payload["report_path"]).exists()
    persisted = json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))
    assert persisted["report_path"] == payload["report_path"]
    assert persisted["safety"]["mainnet_live_allowed"] is False
    assert persisted["side_summary"]["sell"]["status"] == "missing_or_negative_expectancy"
    assert persisted["completion_audit"]["status"] == "incomplete"
    assert persisted["risk_boundary"]["max_per_trade_risk_pct"] == 0.025
    assert persisted["negative_surface_repair_plan"][0]["guardrails"]["mainnet_live_allowed"] is False


def test_repair_symbol_basket_expands_from_route_symbols(monkeypatch) -> None:
    monkeypatch.setattr(
        sweep,
        "load_asset_routes",
        lambda: {
            "major-alt-trend": {
                "symbols": [
                    "ADAUSDT",
                    "AVAXUSDT",
                    "NEARUSDT",
                    "UNIUSDT",
                    "LTCUSDT",
                    "BCHUSDT",
                    "FILUSDT",
                    "ATOMUSDT",
                    "APTUSDT",
                    "INJUSDT",
                    "ARBUSDT",
                    "OPUSDT",
                    "SUIUSDT",
                ]
            }
        },
    )

    basket = sweep._repair_symbol_basket(
        {
            "target_side": "SELL",
            "target_interval": "15m",
            "symbol": "FILUSDT",
            "route_id": "major-alt-trend",
        }
    )

    assert basket[:7] == (
        "FILUSDT",
        "ADAUSDT",
        "AVAXUSDT",
        "NEARUSDT",
        "UNIUSDT",
        "LTCUSDT",
        "BCHUSDT",
    )
    assert "ATOMUSDT" in basket
    assert "DOGEUSDT" in basket
    assert len(basket) == len(set(basket))


def test_risk_combo_matrix_skips_no_dataset_reports_instead_of_ranking_them(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setattr(sweep, "STATE_DIR", state_dir)
    monkeypatch.setattr(sweep, "RISK_COMBO_MATRIX_DIR", state_dir / "risk-combo-matrix")
    monkeypatch.setattr(sweep, "ensure_runtime_dirs", lambda: state_dir.mkdir(parents=True, exist_ok=True))
    valid_report = tmp_path / "sell-15m.json"
    no_dataset_report = tmp_path / "sell-30m-no-datasets.json"
    valid_report.write_text(
        json.dumps(
            {
                "status": "ok",
                "aggregate": {
                    "dataset_count": 1,
                    "configs_tested": 1,
                    "target_side": "SELL",
                    "target_interval": "15m",
                },
                "ranked": [
                    {
                        "route_id": "major-alt-trend",
                        "requested_symbol": "ADAUSDT",
                        "interval": "15m",
                        "params": {"target_side": "SELL"},
                        "full": {
                            "trade_count": 9,
                            "profit_factor": 0.6833,
                            "expectancy_r": -0.154,
                            "max_drawdown_pct": 3.0883,
                            "stop_loss_ratio": 44.44,
                        },
                        "test": {
                            "trade_count": 2,
                            "profit_factor": "inf",
                            "expectancy_r": 0.067,
                            "max_drawdown_pct": 0,
                            "stop_loss_ratio": 0,
                        },
                        "recovery_gate": {"passed": False, "reasons": ["full-profit-factor-below-target"]},
                        "robust_recovery_gate": {"passed": False, "reasons": ["initial-recovery-gate-not-passed"]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    no_dataset_report.write_text(
        json.dumps(
            {
                "status": "no_datasets",
                "aggregate": {
                    "dataset_count": 0,
                    "configs_tested": 0,
                    "target_side": "SELL",
                    "target_interval": "30m",
                },
                "ranked": [],
                "best_by_route": {},
                "best_by_symbol": {},
                "dataset_errors": [{"symbol": "ADAUSDT", "error": "Temporary failure in name resolution"}],
            }
        ),
        encoding="utf-8",
    )

    payload = sweep.build_risk_combo_matrix_report(
        report_paths=[valid_report, no_dataset_report],
        output_dir=tmp_path / "matrix",
    )

    assert payload["input_report_count"] == 2
    assert payload["skipped_input_report_count"] == 1
    assert payload["skipped_input_reports"] == [
        {
            "report_path": str(no_dataset_report),
            "status": "no_datasets",
            "reason": "no_datasets",
            "dataset_count": 0,
            "configs_tested": 0,
            "dataset_error_count": 1,
            "target_side": "SELL",
            "target_interval": "30m",
        }
    ]
    assert payload["surface_count"] == 1
    assert [row["surface"] for row in payload["surfaces"]] == ["sell_15m"]


def test_risk_combo_matrix_can_include_latest_sweep_reports(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    sweep_dir = state_dir / "risk-combo-sweeps"
    sweep_dir.mkdir(parents=True)
    monkeypatch.setattr(sweep, "STATE_DIR", state_dir)
    monkeypatch.setattr(sweep, "RISK_COMBO_SWEEP_DIR", sweep_dir)
    monkeypatch.setattr(sweep, "RISK_COMBO_MATRIX_DIR", state_dir / "risk-combo-matrix")
    monkeypatch.setattr(sweep, "ensure_runtime_dirs", lambda: state_dir.mkdir(parents=True, exist_ok=True))
    older = sweep_dir / "20260510T080000Z-risk-combo-sweep.json"
    newer = sweep_dir / "20260510T081000Z-risk-combo-sweep.json"
    older.write_text(
        json.dumps(
            {
                "status": "ok",
                "aggregate": {"target_side": "BUY", "target_interval": "4h"},
                "best_by_route": {
                    "trx-mean-reversion": {
                        "route_id": "trx-mean-reversion",
                        "requested_symbol": "TRXUSDT",
                        "interval": "4h",
                        "params": {"target_side": "BUY"},
                        "full": {"trade_count": 10, "profit_factor": 0.8, "expectancy_r": -0.1},
                        "test": {"trade_count": 1, "profit_factor": 0.0, "expectancy_r": 0.0},
                        "recovery_gate": {"passed": False},
                        "robust_recovery_gate": {"passed": False},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    newer.write_text(
        json.dumps(
            {
                "status": "ok",
                "aggregate": {"target_side": "SELL", "target_interval": "15m"},
                "best_by_route": {
                    "major-alt-trend": {
                        "route_id": "major-alt-trend",
                        "requested_symbol": "FILUSDT",
                        "interval": "15m",
                        "params": {"target_side": "SELL"},
                        "full": {"trade_count": 5, "profit_factor": 0.7826, "expectancy_r": -0.1346},
                        "test": {"trade_count": 2, "profit_factor": "inf", "expectancy_r": 0.5227},
                        "recovery_gate": {"passed": False},
                        "robust_recovery_gate": {"passed": False},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    import os

    os.utime(older, (1_000_000_000, 1_000_000_000))
    os.utime(newer, (1_000_000_100, 1_000_000_100))

    payload = sweep.build_risk_combo_matrix_report(
        report_paths=[],
        output_dir=tmp_path / "matrix",
        latest_sweeps=1,
    )

    assert payload["input_report_count"] == 1
    assert payload["surface_count"] == 1
    assert payload["surfaces"][0]["surface"] == "sell_15m"
    assert payload["surfaces"][0]["source_report_path"] == str(newer)


def test_risk_combo_matrix_does_not_rank_tiny_infinite_pf_above_promising_surface(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setattr(sweep, "STATE_DIR", state_dir)
    monkeypatch.setattr(sweep, "RISK_COMBO_MATRIX_DIR", state_dir / "risk-combo-matrix")
    monkeypatch.setattr(sweep, "ensure_runtime_dirs", lambda: state_dir.mkdir(parents=True, exist_ok=True))
    promising_report = tmp_path / "buy-4h.json"
    tiny_report = tmp_path / "sell-1d.json"
    promising_report.write_text(
        json.dumps(
            {
                "best_by_route": {
                    "trx-mean-reversion": {
                        "route_id": "trx-mean-reversion",
                        "requested_symbol": "TRXUSDT",
                        "interval": "4h",
                        "params": {"target_side": "BUY"},
                        "full": {
                            "trade_count": 21,
                            "profit_factor": 1.755,
                            "expectancy_r": 0.0948,
                            "max_drawdown_pct": 0.7662,
                            "stop_loss_ratio": 38.1,
                        },
                        "train": {"trade_count": 6, "profit_factor": 3.5389, "expectancy_r": 0.2898},
                        "test": {
                            "trade_count": 5,
                            "profit_factor": 21.4049,
                            "expectancy_r": 0.2789,
                            "max_drawdown_pct": 0.0406,
                            "stop_loss_ratio": 0.0,
                        },
                        "walk_forward": {"window_count": 0},
                        "recovery_gate": {"passed": False, "min_test_trades": 10, "reasons": ["test-trade-count-too-low"]},
                        "robust_recovery_gate": {"passed": False, "reasons": ["walk-forward-window-count-too-low"]},
                    }
                },
                "aggregate": {"target_side": "BUY", "target_interval": "4h"},
            }
        ),
        encoding="utf-8",
    )
    tiny_report.write_text(
        json.dumps(
            {
                "best_by_route": {
                    "trx-mean-reversion": {
                        "route_id": "trx-mean-reversion",
                        "requested_symbol": "TRXUSDT",
                        "interval": "1d",
                        "params": {"target_side": "SELL"},
                        "full": {
                            "trade_count": 1,
                            "profit_factor": "inf",
                            "expectancy_r": 0.1257,
                            "max_drawdown_pct": 0.0,
                            "stop_loss_ratio": 0.0,
                        },
                        "train": {"trade_count": 0, "profit_factor": 0.0, "expectancy_r": 0.0},
                        "test": {
                            "trade_count": 0,
                            "profit_factor": 0.0,
                            "expectancy_r": 0.0,
                            "max_drawdown_pct": 0.0,
                            "stop_loss_ratio": 0.0,
                        },
                        "walk_forward": {"window_count": 0},
                        "recovery_gate": {"passed": False, "min_test_trades": 10, "reasons": ["test-trade-count-too-low"]},
                        "robust_recovery_gate": {"passed": False, "reasons": ["walk-forward-window-count-too-low"]},
                    }
                },
                "aggregate": {"target_side": "SELL", "target_interval": "1d"},
            }
        ),
        encoding="utf-8",
    )

    payload = sweep.build_risk_combo_matrix_report(
        report_paths=[tiny_report, promising_report],
        output_dir=tmp_path / "matrix",
    )

    assert payload["best_surface"]["surface"] == "buy_4h"
    assert payload["best_surface"]["research_status"] == "promising_but_under_validated"


def test_risk_combo_matrix_tracks_emerging_positive_leads_without_promoting(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setattr(sweep, "STATE_DIR", state_dir)
    monkeypatch.setattr(sweep, "RISK_COMBO_MATRIX_DIR", state_dir / "risk-combo-matrix")
    monkeypatch.setattr(sweep, "ensure_runtime_dirs", lambda: state_dir.mkdir(parents=True, exist_ok=True))
    report = tmp_path / "sell-30m-ranked.json"
    report.write_text(
        json.dumps(
            {
                "status": "ok",
                "aggregate": {
                    "dataset_count": 2,
                    "configs_tested": 10,
                    "target_side": "SELL",
                    "target_interval": "30m",
                },
                "ranked": [
                    {
                        "route_id": "major-alt-trend",
                        "requested_symbol": "FILUSDT",
                        "interval": "30m",
                        "params": {"target_side": "SELL"},
                        "full": {
                            "trade_count": 8,
                            "profit_factor": 0.2916,
                            "expectancy_r": -0.555,
                            "max_drawdown_pct": 8.4598,
                            "stop_loss_ratio": 75.0,
                        },
                        "test": {"trade_count": 2, "profit_factor": "inf", "expectancy_r": 0.9139},
                        "recovery_gate": {
                            "passed": False,
                            "min_test_trades": 10,
                            "reasons": ["test-trade-count-too-low", "full-profit-factor-below-target"],
                        },
                        "robust_recovery_gate": {"passed": False, "reasons": ["initial-recovery-gate-not-passed"]},
                    },
                    {
                        "route_id": "doge-meme-high-beta",
                        "requested_symbol": "DOGEUSDT",
                        "interval": "30m",
                        "params": {"target_side": "SELL"},
                        "full": {
                            "trade_count": 4,
                            "profit_factor": 1.6011,
                            "expectancy_r": 0.3216,
                            "max_drawdown_pct": 2.1287,
                            "stop_loss_ratio": 50.0,
                        },
                        "test": {
                            "trade_count": 2,
                            "profit_factor": 1.1349,
                            "expectancy_r": 0.0736,
                            "max_drawdown_pct": 1.0914,
                            "stop_loss_ratio": 50.0,
                        },
                        "recovery_gate": {
                            "passed": False,
                            "min_test_trades": 10,
                            "reasons": ["test-trade-count-too-low"],
                        },
                        "robust_recovery_gate": {"passed": False, "reasons": ["initial-recovery-gate-not-passed"]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = sweep.build_risk_combo_matrix_report(
        report_paths=[report],
        output_dir=tmp_path / "matrix",
    )

    assert payload["promising_surface_count"] == 0
    assert payload["emerging_positive_lead_count"] == 1
    assert payload["side_summary"]["sell"]["promising_surface_count"] == 0
    assert payload["side_summary"]["sell"]["emerging_positive_lead_count"] == 1
    assert payload["side_summary"]["sell"]["status"] == "emerging_positive_needs_sample"
    assert payload["horizon_summary"]["short"]["promising_surface_count"] == 0
    assert payload["horizon_summary"]["short"]["emerging_positive_lead_count"] == 1
    lead = payload["emerging_positive_leads"][0]
    assert lead["symbol"] == "DOGEUSDT"
    assert lead["research_status"] == "emerging_positive_research_lead"
    assert lead["promotion_eligible"] is False
    assert lead["research_lead_only"] is True
    assert payload["best_surface"]["symbol"] == "DOGEUSDT"
    assert payload["best_surface"]["research_status"] == "emerging_positive_research_lead"
    checklist = payload["prompt_to_artifact_checklist"]
    checklist_items = {item["requirement"]: item for item in checklist["items"]}
    assert checklist_items["candidate_not_zero"]["passed"] is False
    assert checklist_items["buy_and_sell_directional_evidence"]["evidence"]["sell_emerging_positive_leads"] == 1
    assert checklist_items["short_medium_long_horizon_coverage"]["evidence"]["short_emerging_positive_leads"] == 1
    audit = payload["completion_audit"]
    assert "candidate_signal_not_zero" in audit["missing_requirements"]
    repair_plan = payload["negative_surface_repair_plan"]
    assert repair_plan[0]["research_status"] == "emerging_positive_research_lead"
    assert repair_plan[0]["promotion_eligible"] is False
    assert "--symbols DOGEUSDT" in repair_plan[0]["scout_command"]
    assert payload["validation_plan"][0]["research_status"] == "emerging_positive_research_lead"
    assert payload["validation_plan"][0]["promotion_eligible"] is False
    assert payload["validation_plan"][0]["promotion_boundary"] == "not_promotion_eligible_until_sample_and_robust_gates_pass"
    persisted = json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))
    assert persisted["emerging_positive_lead_count"] == 1
    assert persisted["safety"]["mainnet_live_allowed"] is False


def test_risk_combo_matrix_supersedes_emerging_lead_after_new_failed_validation(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setattr(sweep, "STATE_DIR", state_dir)
    monkeypatch.setattr(sweep, "RISK_COMBO_MATRIX_DIR", state_dir / "risk-combo-matrix")
    monkeypatch.setattr(sweep, "ensure_runtime_dirs", lambda: state_dir.mkdir(parents=True, exist_ok=True))
    old_report = tmp_path / "old-doge-sell-30m.json"
    new_report = tmp_path / "new-doge-sell-30m.json"
    old_report.write_text(
        json.dumps(
            {
                "status": "ok",
                "aggregate": {"dataset_count": 1, "configs_tested": 1, "target_side": "SELL", "target_interval": "30m"},
                "ranked": [
                    {
                        "route_id": "doge-meme-high-beta",
                        "requested_symbol": "DOGEUSDT",
                        "interval": "30m",
                        "params": {"target_side": "SELL"},
                        "full": {
                            "trade_count": 4,
                            "profit_factor": 1.6011,
                            "expectancy_r": 0.3216,
                            "max_drawdown_pct": 2.1287,
                            "stop_loss_ratio": 50.0,
                        },
                        "test": {
                            "trade_count": 2,
                            "profit_factor": 1.1349,
                            "expectancy_r": 0.0736,
                            "max_drawdown_pct": 1.0914,
                            "stop_loss_ratio": 50.0,
                        },
                        "recovery_gate": {"passed": False, "min_test_trades": 10, "reasons": ["test-trade-count-too-low"]},
                        "robust_recovery_gate": {"passed": False, "reasons": ["walk-forward-validation-not-run"]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    new_report.write_text(
        json.dumps(
            {
                "status": "ok",
                "aggregate": {"dataset_count": 1, "configs_tested": 8, "target_side": "SELL", "target_interval": "30m"},
                "ranked": [
                    {
                        "route_id": "doge-meme-high-beta",
                        "requested_symbol": "DOGEUSDT",
                        "interval": "30m",
                        "params": {"target_side": "SELL"},
                        "full": {
                            "trade_count": 6,
                            "profit_factor": 1.0501,
                            "expectancy_r": 0.0273,
                            "max_drawdown_pct": 2.1595,
                            "stop_loss_ratio": 50.0,
                        },
                        "test": {
                            "trade_count": 2,
                            "profit_factor": 0.9478,
                            "expectancy_r": -0.0285,
                            "max_drawdown_pct": 1.0914,
                            "stop_loss_ratio": 50.0,
                        },
                        "recovery_gate": {
                            "passed": False,
                            "min_test_trades": 10,
                            "reasons": [
                                "test-trade-count-too-low",
                                "test-profit-factor-below-target",
                                "test-expectancy-r-below-target",
                            ],
                        },
                        "robust_recovery_gate": {
                            "passed": False,
                            "reasons": ["walk-forward-min-expectancy-r-below-target"],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    import os

    os.utime(old_report, (1_000_000_000, 1_000_000_000))
    os.utime(new_report, (1_000_000_100, 1_000_000_100))

    payload = sweep.build_risk_combo_matrix_report(
        report_paths=[old_report, new_report],
        output_dir=tmp_path / "matrix",
    )

    assert payload["promising_surface_count"] == 0
    assert payload["emerging_positive_lead_count"] == 0
    assert payload["superseded_emerging_positive_lead_count"] == 1
    assert payload["side_summary"]["sell"]["emerging_positive_lead_count"] == 0
    assert payload["side_summary"]["sell"]["status"] == "missing_or_negative_expectancy"
    assert payload["best_surface"]["symbol"] == "DOGEUSDT"
    assert payload["best_surface"]["research_status"] == "rejected_or_negative_expectancy"
    assert payload["recent_failed_repair_identity_count"] == 1
    superseded = payload["superseded_emerging_positive_leads"][0]
    assert superseded["symbol"] == "DOGEUSDT"
    assert superseded["research_status"] == "emerging_positive_research_lead"
    assert superseded["superseded_by_report_path"] == str(new_report)
    assert superseded["superseded_by_research_status"] == "rejected_or_negative_expectancy"
    assert superseded["superseded_by_metrics"]["test"]["expectancy_r"] == -0.0285
    assert payload["validation_plan"] == []
    assert payload["negative_surface_repair_plan"][0]["research_status"] == "rejected_or_negative_expectancy"
    assert "DOGEUSDT" in payload["negative_surface_repair_plan"][0]["excluded_recent_failed_symbols"]
    assert "DOGEUSDT" not in payload["negative_surface_repair_plan"][0]["cross_symbol_scout_symbols"]
    assert "--symbols TRXUSDT,ADAUSDT,AVAXUSDT,NEARUSDT,WIFUSDT" in payload["negative_surface_repair_plan"][0][
        "cross_symbol_scout_command"
    ]


def test_news_veto_filter_blocks_event_proxy_entries() -> None:
    previous = pd.Series(
        {
            "volume_zscore_20": 2.5,
            "realized_vol_20": 0.8,
            "atr_14": 1.0,
            "high": 12.0,
            "low": 9.0,
        }
    )
    entry_filter = sweep._news_veto_filter(
        "event-proxy",
        {"risk_level": "normal", "bias": "neutral", "high_impact_count": 0},
    )

    allowed, reason = entry_filter(previous, previous, {}, 1)

    assert allowed is False
    assert reason == "event-risk-proxy-veto"


def test_entry_filter_can_veto_short_side_policy() -> None:
    previous = pd.Series(
        {
            "adx": 40.0,
            "minus_di": 30.0,
            "plus_di": 10.0,
            "volume_zscore_20": 0.0,
            "realized_vol_20": 0.8,
            "atr_14": 1.0,
            "high": 12.0,
            "low": 11.0,
        }
    )
    entry_filter = sweep._entry_filter(
        "off",
        {"risk_level": "normal", "bias": "neutral", "high_impact_count": 0},
        side_policy_mode="long-only",
        min_adx=20.0,
        min_convergence=0.7,
    )

    allowed, reason = entry_filter(
        previous,
        previous,
        {"recommended_action": "SELL", "score": 10, "convergence": 0.95},
        1,
    )

    assert allowed is False
    assert reason == "side-policy-short-veto"


def test_entry_filter_can_veto_macro_misaligned_short() -> None:
    previous = pd.Series(
        {
            "close": 105.0,
            "sma_200": 100.0,
            "ema_fast": 103.0,
            "ema_slow": 101.0,
            "adx": 30.0,
            "bb_bandwidth": 0.12,
            "minus_di": 30.0,
            "plus_di": 10.0,
            "volume_zscore_20": 0.0,
            "realized_vol_20": 0.8,
            "atr_14": 1.0,
            "high": 106.0,
            "low": 104.0,
        }
    )
    entry_filter = sweep._entry_filter(
        "off",
        {"risk_level": "normal", "bias": "neutral", "high_impact_count": 0},
        side_policy_mode="baseline",
        structure_policy_mode="macro-aligned",
        min_adx=20.0,
        min_convergence=0.7,
    )

    allowed, reason = entry_filter(
        previous,
        previous,
        {"recommended_action": "SELL", "score": 10, "convergence": 0.95},
        1,
    )

    assert allowed is False
    assert reason == "structure-policy-short-above-sma200"


def test_entry_filter_can_veto_squeeze_regime() -> None:
    previous = pd.Series(
        {
            "close": 99.0,
            "sma_200": 100.0,
            "ema_fast": 98.0,
            "ema_slow": 99.0,
            "adx": 12.0,
            "bb_bandwidth": 0.03,
            "minus_di": 30.0,
            "plus_di": 10.0,
            "volume_zscore_20": 0.0,
            "realized_vol_20": 0.8,
            "atr_14": 1.0,
            "high": 100.0,
            "low": 98.0,
        }
    )
    entry_filter = sweep._entry_filter(
        "off",
        {"risk_level": "normal", "bias": "neutral", "high_impact_count": 0},
        side_policy_mode="baseline",
        structure_policy_mode="no-squeeze",
        min_adx=20.0,
        min_convergence=0.7,
    )

    allowed, reason = entry_filter(
        previous,
        previous,
        {"recommended_action": "SELL", "score": 10, "convergence": 0.95, "regime": "squeeze"},
        1,
    )

    assert allowed is False
    assert reason == "structure-policy-squeeze-regime-veto"


def test_entry_filter_can_veto_negative_historical_feedback_bucket() -> None:
    reviews = [
        {
            "route_id": "major-alt-trend",
            "symbol": "NEARUSDT",
            "side": "SELL",
            "analysis_score": 10,
            "analysis_convergence": 0.95,
            "realized_pnl_usdt": -1.0,
        },
        *[
            {
                "route_id": "major-alt-trend",
                "symbol": "NEARUSDT",
                "side": "SELL",
                "analysis_score": 12,
                "analysis_convergence": 0.94,
                "realized_pnl_usdt": -0.5,
            }
            for _ in range(19)
        ],
        *[
            {
                "route_id": "major-alt-trend",
                "symbol": "NEARUSDT",
                "side": "SELL",
                "analysis_score": 8,
                "analysis_convergence": 0.96,
                "realized_pnl_usdt": 0.1,
            }
            for _ in range(2)
        ],
    ]
    historical_index = sweep.build_historical_signal_risk_index(reviews)
    previous = pd.Series(
        {
            "close": 99.0,
            "sma_200": 100.0,
            "ema_fast": 98.0,
            "ema_slow": 99.0,
            "adx": 30.0,
            "bb_bandwidth": 0.12,
            "minus_di": 30.0,
            "plus_di": 10.0,
            "volume_zscore_20": 0.0,
            "realized_vol_20": 0.8,
            "atr_14": 1.0,
            "high": 100.0,
            "low": 98.0,
        }
    )
    entry_filter = sweep._entry_filter(
        "off",
        {"risk_level": "normal", "bias": "neutral", "high_impact_count": 0},
        side_policy_mode="baseline",
        structure_policy_mode="baseline",
        min_adx=20.0,
        min_convergence=0.7,
        historical_policy_mode="feedback-bucket-veto",
        historical_signal_index=historical_index,
        route_id="major-alt-trend",
        symbol="NEARUSDT",
    )

    allowed, reason = entry_filter(
        previous,
        previous,
        {"recommended_action": "SELL", "score": 10, "convergence": 0.95},
        1,
    )

    assert allowed is False
    assert reason == "historical-feedback-bucket-veto"


def test_entry_filter_can_veto_weak_route_side_history() -> None:
    previous = pd.Series(
        {
            "close": 99.0,
            "sma_200": 100.0,
            "ema_fast": 98.0,
            "ema_slow": 99.0,
            "adx": 30.0,
            "bb_bandwidth": 0.12,
            "minus_di": 30.0,
            "plus_di": 10.0,
            "volume_zscore_20": 0.0,
            "realized_vol_20": 0.8,
            "atr_14": 1.0,
            "high": 100.0,
            "low": 98.0,
        }
    )
    route_side_gate = sweep.evaluate_route_side_risk(
        route_id="major-alt-trend",
        side="SELL",
        min_samples=3,
        min_profit_factor=0.8,
        reviews=[
            {"route_id": "major-alt-trend", "side": "SELL", "realized_pnl_usdt": -1.0},
            {"route_id": "major-alt-trend", "side": "SELL", "realized_pnl_usdt": -1.0},
            {"route_id": "major-alt-trend", "side": "SELL", "realized_pnl_usdt": 0.1},
        ],
    )
    entry_filter = sweep._entry_filter(
        "off",
        {"risk_level": "normal", "bias": "neutral", "high_impact_count": 0},
        side_policy_mode="baseline",
        structure_policy_mode="baseline",
        min_adx=20.0,
        min_convergence=0.7,
        route_side_policy_mode="route-side-veto",
        route_side_evaluation=route_side_gate,
        route_id="major-alt-trend",
        symbol="NEARUSDT",
    )

    allowed, reason = entry_filter(
        previous,
        previous,
        {"recommended_action": "SELL", "score": 10, "convergence": 0.95},
        1,
    )

    assert allowed is False
    assert reason == "route-side-history-veto"


def test_focused_grid_keeps_structure_search_small() -> None:
    strategy = load_strategy_config(Path("config/strategy-btc-volatility.yaml"))

    grid = sweep._grid_values(strategy, mode="focused")

    assert grid["structure_policy_mode"] == ("baseline", "macro-trend-no-squeeze")
    assert grid["historical_policy_mode"] == ("feedback-bucket-veto",)
    assert grid["news_veto_mode"] == ("off",)
    assert "payoff_runner" in grid["exit_profile"]
    assert "asymmetric_payoff" in grid["exit_profile"]
    assert "focused" not in grid
    total = 1
    for values in grid.values():
        total *= len(values)
    assert total < 600
