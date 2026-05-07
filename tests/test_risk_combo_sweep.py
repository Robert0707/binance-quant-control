from __future__ import annotations

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
