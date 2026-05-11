from __future__ import annotations

import json
from pathlib import Path

import pytest

import binance_quant_control.readiness_scanner as scanner


@pytest.fixture(autouse=True)
def _isolated_risk_combo_matrix_dir(monkeypatch, tmp_path):
    matrix_dir = tmp_path / "risk-combo-matrix"
    matrix_dir.mkdir()
    monkeypatch.setattr(scanner, "RISK_COMBO_MATRIX_DIR", matrix_dir)


def _queue_item(rank: int, symbol: str) -> dict[str, object]:
    return {
        "rank": rank,
        "signal": {
            "symbol": symbol,
            "side": "BUY",
            "interval": "4h",
            "strategy_family": "ai_family_router",
            "route_id": f"{symbol.lower()}-route",
        },
        "machine_state": "candidate_ready",
        "open_order_gate": {"allowed": True, "blockers": []},
    }


def _blocked_queue_item(rank: int, symbol: str) -> dict[str, object]:
    item = _queue_item(rank, symbol)
    item["machine_state"] = "route_history_blocked"
    item["open_order_gate"] = {
        "allowed": False,
        "blockers": [f"route-quarantined:{symbol.lower()}-route:profit-factor below floor"],
    }
    return item


def _plan(
    *,
    symbol: str,
    allowed: bool,
    violations: list[str] | None = None,
    warnings: list[str] | None = None,
) -> object:
    return type(
        "Plan",
        (),
        {
            "to_dict": lambda self: {
                "allowed": allowed,
                "symbol": symbol,
                "market": "futures",
                "side": "BUY",
                "quantity": 10.0,
                "price": 1.0,
                "leverage": 3,
                "margin_notional_usdt": 2.0,
                "gross_notional_usdt": 6.0,
                "min_notional_usdt": 5.0,
                "planned_account_risk_pct": 0.0025,
                "analysis_score": 80,
                "analysis_convergence": 0.82,
                "adx_value": 24.0,
                "execution_mode": "testnet_exploration",
                "violations": violations or [],
                "warnings": warnings or [],
                "professional_entry_gate": {"passed": allowed},
                "market_bot_gate": {
                    "allowed": True,
                    "report_path": "state/market-bot-gate.json",
                    "feature_manifest_hash": "abc123",
                    "matched_row": {"cohort_id": f"{symbol}:4h:ai_family_router"},
                },
                "challenge": {
                    "market_bot_gate": {"allowed": True},
                    "route_quarantine": {"quarantined": False, "reasons": []},
                    "route_side_risk": {"allowed": True, "reasons": []},
                    "historical_signal_risk": {"allowed": True, "reasons": []},
                },
            }
        },
    )()


def test_ai_readiness_scan_selects_second_candidate_when_first_is_blocked(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(scanner, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        scanner,
        "run_hermes_ai_trader",
        lambda **_kwargs: {
            "report_path": str(tmp_path / "hermes.json"),
            "candidate_queue": [_queue_item(1, "DOGEUSDT"), _queue_item(2, "ETHUSDT")],
        },
    )
    monkeypatch.setattr(scanner, "load_settings", lambda: object())
    monkeypatch.setattr(scanner, "load_strategy_config", lambda _path: type(
        "Strategy",
        (),
        {
            "defaults": type(
                "Defaults",
                (),
                {
                    "market": "futures",
                    "interval": "4h",
                    "limit": 600,
                    "use_blave": False,
                },
            )()
        },
    )())
    monkeypatch.setattr(
        scanner,
        "run_analysis",
        lambda _settings, **kwargs: (
            {
                "symbol": kwargs["symbol"],
                "market": kwargs["market"],
                "analysis": {"score": 80, "convergence": 0.82},
                "latest": {"close": 1.0},
                "trade_plan": {"long": {}, "short": {}},
            },
            None,
        ),
    )

    def fake_plan(_settings, _strategy, analysis, **_kwargs):
        symbol = analysis["symbol"]
        if symbol == "DOGEUSDT":
            return _plan(
                symbol=symbol,
                allowed=False,
                violations=["Trading is paused by kill-switch: consecutive losses reached 2."],
            )
        return _plan(symbol=symbol, allowed=True)

    monkeypatch.setattr(scanner, "build_live_execution_plan", fake_plan)

    payload = scanner.run_ai_readiness_scan(output_dir=tmp_path)

    assert payload["candidate_count"] == 2
    assert payload["scanned_count"] == 2
    assert payload["allowed_count"] == 1
    assert payload["selected_ready_candidate"]["symbol"] == "ETHUSDT"
    assert payload["ready_after_global_unlock_count"] == 1
    assert payload["selected_after_global_unlock"]["symbol"] == "DOGEUSDT"
    assert payload["next_machine_action"] == "execute_ready_dry_run_only"
    assert payload["machine_action_queue"][0]["action"] == "execute_ready_dry_run_only"
    assert payload["execution_ticket"]["symbol"] == "ETHUSDT"
    assert payload["execution_ticket"]["opens_orders"] is False
    assert "--execute" in payload["execution_ticket"]["operator_testnet_execute_command"]
    assert payload["scan_results"][0]["next_action"] == "wait_for_kill_switch_clear"
    assert Path(payload["report_path"]).exists()


def test_ai_readiness_scan_classifies_blockers_when_all_candidates_fail(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(scanner, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        scanner,
        "run_hermes_ai_trader",
        lambda **_kwargs: {
            "report_path": str(tmp_path / "hermes.json"),
            "candidate_queue": [_queue_item(1, "DOGEUSDT")],
        },
    )
    monkeypatch.setattr(scanner, "load_settings", lambda: object())
    monkeypatch.setattr(scanner, "load_strategy_config", lambda _path: type(
        "Strategy",
        (),
        {
            "defaults": type(
                "Defaults",
                (),
                {
                    "market": "futures",
                    "interval": "4h",
                    "limit": 600,
                    "use_blave": False,
                },
            )()
        },
    )())
    monkeypatch.setattr(
        scanner,
        "run_analysis",
        lambda _settings, **kwargs: (
            {
                "symbol": kwargs["symbol"],
                "market": kwargs["market"],
                "analysis": {"score": 55, "convergence": 0.5},
                "latest": {"close": 1.0},
                "trade_plan": {"long": {}, "short": {}},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        scanner,
        "build_live_execution_plan",
        lambda _settings, _strategy, analysis, **_kwargs: _plan(
            symbol=analysis["symbol"],
            allowed=False,
            violations=[
                "Volume z-score is below floor.",
                "Order notional 4.9000 USDT is below exchange minimum 5.0000.",
                "Recent DOGEUSDT BUY profit factor below live threshold.",
            ],
        ),
    )

    payload = scanner.run_ai_readiness_scan(output_dir=tmp_path)

    assert payload["allowed_count"] == 0
    assert payload["selected_ready_candidate"] is None
    assert payload["next_machine_action"] == "repair_exchange_sizing_or_margin"
    assert payload["machine_action_queue"][0]["action"] == "repair_exchange_sizing_or_margin"
    taxonomy = payload["hard_blocker_taxonomy"]
    assert "market_state" in taxonomy
    assert "exchange_constraints" in taxonomy
    assert "strategy_performance" in taxonomy
    assert payload["denial_journal_count"] == 1
    report = payload["research_candidate_report"]
    assert report["candidate_count"] == 1
    assert report["reviewable_candidate_count"] == 1
    assert report["trade_allowed_count"] == 0
    assert report["side_counts"] == {"BUY": 1}
    assert report["horizon_counts"] == {"medium": 1}
    assert report["side_horizon_counts"] == {"buy_medium": 1}
    assert report["reviewable_side_horizon_counts"] == {"buy_medium": 1}
    assert report["expectancy_improvement_targets"]["risk_ceiling_pct"] == 0.025
    assert report["promotion_boundary"]["mainnet_live_allowed"] is False
    top = report["top_candidates"][0]
    assert top["symbol"] == "DOGEUSDT"
    assert top["research_status"] == "reviewable_signal"
    assert top["horizon"] == "medium"
    assert top["trade_readiness_allowed"] is False
    assert top["expectancy_metrics"]["profit_factor"] is None
    assert "profit_factor_below_1.0_or_missing" in top["positive_expectancy_gap"]
    assert top["promotion_boundary"] == "research_only_not_trade_permission"
    journal_path = Path(payload["denial_journal_path"])
    assert journal_path.exists()
    row = json.loads(journal_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["gate"] == "ai_readiness_live_plan_denial"
    assert row["symbol"] == "DOGEUSDT"
    assert "Order notional 4.9000 USDT is below exchange minimum 5.0000." in row["blockers"]
    assert row["metadata"]["next_action"] == "repair_exchange_sizing_or_margin"


def test_ai_readiness_scan_max_candidates_limits_live_scans_not_blocked_reporting(
    monkeypatch,
    tmp_path,
) -> None:
    scanned_symbols: list[str] = []
    monkeypatch.setattr(scanner, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        scanner,
        "run_hermes_ai_trader",
        lambda **_kwargs: {
            "report_path": str(tmp_path / "hermes.json"),
            "candidate_queue": [
                _blocked_queue_item(1, "BTCUSDT"),
                _queue_item(2, "DOGEUSDT"),
                _queue_item(3, "ETHUSDT"),
            ],
        },
    )
    monkeypatch.setattr(scanner, "load_settings", lambda: object())
    monkeypatch.setattr(scanner, "load_strategy_config", lambda _path: type(
        "Strategy",
        (),
        {
            "defaults": type(
                "Defaults",
                (),
                {
                    "market": "futures",
                    "interval": "4h",
                    "limit": 600,
                    "use_blave": False,
                },
            )()
        },
    )())
    monkeypatch.setattr(
        scanner,
        "run_analysis",
        lambda _settings, **kwargs: scanned_symbols.append(kwargs["symbol"]) or (
            {
                "symbol": kwargs["symbol"],
                "market": kwargs["market"],
                "analysis": {"score": 80, "convergence": 0.82},
                "latest": {"close": 1.0},
                "trade_plan": {"long": {}, "short": {}},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        scanner,
        "build_live_execution_plan",
        lambda _settings, _strategy, analysis, **_kwargs: _plan(
            symbol=analysis["symbol"],
            allowed=False,
            violations=["Volume z-score is below floor."],
        ),
    )

    payload = scanner.run_ai_readiness_scan(output_dir=tmp_path, max_candidates=1)

    assert payload["candidate_count"] == 2
    assert payload["scanned_count"] == 1
    assert scanned_symbols == ["DOGEUSDT"]
    assert [item["symbol"] for item in payload["scan_results"]] == ["BTCUSDT", "DOGEUSDT"]
    assert payload["scan_results"][0]["machine_state"] == "route_history_blocked"


def test_ai_readiness_scan_classifies_route_side_text_as_route_history(
    monkeypatch,
    tmp_path,
) -> None:
    blocked = _queue_item(1, "ETHUSDT")
    blocked["machine_state"] = "route_history_blocked"
    blocked["open_order_gate"] = {
        "allowed": False,
        "blockers": ["Route-side net PnL is still negative (-0.8314 USDT) for eth-core/BUY."],
    }
    monkeypatch.setattr(scanner, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        scanner,
        "run_hermes_ai_trader",
        lambda **_kwargs: {
            "report_path": str(tmp_path / "hermes.json"),
            "candidate_queue": [blocked],
        },
    )

    payload = scanner.run_ai_readiness_scan(output_dir=tmp_path)

    assert payload["scan_results"][0]["blocker_taxonomy"]["route_history"]
    assert payload["hard_blocker_taxonomy"]["route_history"]
    assert payload["machine_action_queue"][0]["action"] == "repair_route_history_or_wait_for_quarantine_clear"


def test_ai_readiness_scan_action_queue_keeps_post_kill_switch_work_visible(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(scanner, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        scanner,
        "run_hermes_ai_trader",
        lambda **_kwargs: {
            "report_path": str(tmp_path / "hermes.json"),
            "candidate_queue": [_queue_item(1, "DOGEUSDT"), _queue_item(2, "ETHUSDT")],
        },
    )
    monkeypatch.setattr(scanner, "load_settings", lambda: object())
    monkeypatch.setattr(scanner, "load_strategy_config", lambda _path: type(
        "Strategy",
        (),
        {
            "defaults": type(
                "Defaults",
                (),
                {
                    "market": "futures",
                    "interval": "4h",
                    "limit": 600,
                    "use_blave": False,
                },
            )()
        },
    )())
    monkeypatch.setattr(
        scanner,
        "run_analysis",
        lambda _settings, **kwargs: (
            {
                "symbol": kwargs["symbol"],
                "market": kwargs["market"],
                "analysis": {"score": 55, "convergence": 0.5},
                "latest": {"close": 1.0},
                "trade_plan": {"long": {}, "short": {}},
            },
            None,
        ),
    )

    def fake_plan(_settings, _strategy, analysis, **_kwargs):
        symbol = analysis["symbol"]
        extra = (
            "Order notional 4.9000 USDT is below exchange minimum 5.0000."
            if symbol == "ETHUSDT"
            else "Recent PF 0.53 is below 0.85."
        )
        return _plan(
            symbol=symbol,
            allowed=False,
            violations=[
                "Trading is paused by kill-switch: consecutive losses reached 2.",
                extra,
            ],
        )

    monkeypatch.setattr(scanner, "build_live_execution_plan", fake_plan)

    payload = scanner.run_ai_readiness_scan(output_dir=tmp_path)

    assert payload["next_machine_action"] == "wait_for_kill_switch_clear"
    assert payload["ready_after_global_unlock_count"] == 0
    assert [item["action"] for item in payload["machine_action_queue"]] == [
        "wait_for_kill_switch_clear",
        "repair_exchange_sizing_or_margin",
        "repair_strategy_performance_or_route_history",
    ]
    kill_row = payload["machine_action_queue"][0]
    assert kill_row["unlock_ready_candidate_count"] == 0


def test_ai_readiness_scan_marks_candidates_ready_after_global_unlock(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(scanner, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        scanner,
        "run_hermes_ai_trader",
        lambda **_kwargs: {
            "report_path": str(tmp_path / "hermes.json"),
            "candidate_queue": [_queue_item(1, "ETHUSDT"), _queue_item(2, "DOGEUSDT")],
        },
    )
    monkeypatch.setattr(scanner, "load_settings", lambda: object())
    monkeypatch.setattr(scanner, "load_strategy_config", lambda _path: type(
        "Strategy",
        (),
        {
            "defaults": type(
                "Defaults",
                (),
                {
                    "market": "futures",
                    "interval": "4h",
                    "limit": 600,
                    "use_blave": False,
                },
            )()
        },
    )())
    monkeypatch.setattr(
        scanner,
        "run_analysis",
        lambda _settings, **kwargs: (
            {
                "symbol": kwargs["symbol"],
                "market": kwargs["market"],
                "analysis": {"score": 80, "convergence": 0.82},
                "latest": {"close": 1.0},
                "trade_plan": {"long": {}, "short": {}},
            },
            None,
        ),
    )

    def fake_plan(_settings, _strategy, analysis, **_kwargs):
        symbol = analysis["symbol"]
        violations = ["Trading is paused by kill-switch: consecutive losses reached 2."]
        if symbol == "DOGEUSDT":
            violations.append("Volume z-score is below floor.")
        return _plan(symbol=symbol, allowed=False, violations=violations)

    monkeypatch.setattr(scanner, "build_live_execution_plan", fake_plan)

    payload = scanner.run_ai_readiness_scan(output_dir=tmp_path)

    assert payload["allowed_count"] == 0
    assert payload["next_machine_action"] == "wait_for_kill_switch_clear"
    assert payload["ready_after_global_unlock_count"] == 1
    assert payload["selected_after_global_unlock"]["symbol"] == "ETHUSDT"
    assert payload["ready_after_global_unlock_candidates"][0]["next_action_after_global_unlock"] == "execute_ready_dry_run_only"
    kill_row = payload["machine_action_queue"][0]
    assert kill_row["unlock_ready_candidate_count"] == 1
    assert kill_row["unlock_ready_candidates"] == ["ETHUSDT:BUY"]


def test_ai_readiness_scan_reports_blocked_research_candidates_by_side(
    monkeypatch,
    tmp_path,
) -> None:
    sell_item = _queue_item(1, "ETHUSDT")
    sell_item["signal"]["side"] = "SELL"  # type: ignore[index]
    sell_item["signal"]["route_id"] = "eth-core-short"  # type: ignore[index]
    buy_item = _queue_item(2, "SOLUSDT")
    monkeypatch.setattr(scanner, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        scanner,
        "run_hermes_ai_trader",
        lambda **_kwargs: {
            "report_path": str(tmp_path / "hermes.json"),
            "candidate_queue": [sell_item, buy_item],
        },
    )
    monkeypatch.setattr(scanner, "load_settings", lambda: object())
    monkeypatch.setattr(scanner, "load_strategy_config", lambda _path: type(
        "Strategy",
        (),
        {
            "defaults": type(
                "Defaults",
                (),
                {
                    "market": "futures",
                    "interval": "4h",
                    "limit": 600,
                    "use_blave": False,
                },
            )()
        },
    )())
    monkeypatch.setattr(
        scanner,
        "run_analysis",
        lambda _settings, **kwargs: (
            {
                "symbol": kwargs["symbol"],
                "market": kwargs["market"],
                "analysis": {"score": 72, "convergence": 0.9},
                "latest": {"close": 1.0},
                "trade_plan": {"long": {}, "short": {}},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        scanner,
        "build_live_execution_plan",
        lambda _settings, _strategy, analysis, **_kwargs: _plan(
            symbol=analysis["symbol"],
            allowed=False,
            violations=["Recent expectancy is below threshold."],
        ),
    )

    payload = scanner.run_ai_readiness_scan(output_dir=tmp_path)

    assert payload["allowed_count"] == 0
    report = payload["research_candidate_report"]
    assert report["candidate_count"] == 2
    assert report["reviewable_candidate_count"] == 2
    assert report["trade_allowed_count"] == 0
    assert report["side_counts"] == {"SELL": 1, "BUY": 1}
    assert report["reviewable_side_counts"] == {"SELL": 1, "BUY": 1}
    assert "missing_reviewable_sell_research_candidate" not in report["coverage_gaps"]
    assert "improve_expectancy_before_promotion" in report["research_next_actions"]
    assert {item["side"] for item in report["top_candidates"]} == {"BUY", "SELL"}
    assert all(item["promotion_boundary"] == "research_only_not_trade_permission" for item in report["top_candidates"])


def test_ai_readiness_scan_reports_research_candidates_by_horizon(
    monkeypatch,
    tmp_path,
) -> None:
    short_item = _queue_item(1, "DOGEUSDT")
    short_item["signal"]["interval"] = "15m"  # type: ignore[index]
    medium_item = _queue_item(2, "SOLUSDT")
    medium_item["signal"]["interval"] = "4h"  # type: ignore[index]
    long_item = _queue_item(3, "BTCUSDT")
    long_item["signal"]["interval"] = "1d"  # type: ignore[index]
    monkeypatch.setattr(scanner, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        scanner,
        "run_hermes_ai_trader",
        lambda **_kwargs: {
            "report_path": str(tmp_path / "hermes.json"),
            "candidate_queue": [short_item, medium_item, long_item],
        },
    )
    monkeypatch.setattr(scanner, "load_settings", lambda: object())
    monkeypatch.setattr(scanner, "load_strategy_config", lambda _path: type(
        "Strategy",
        (),
        {
            "defaults": type(
                "Defaults",
                (),
                {
                    "market": "futures",
                    "interval": "4h",
                    "limit": 600,
                    "use_blave": False,
                },
            )()
        },
    )())
    monkeypatch.setattr(
        scanner,
        "run_analysis",
        lambda _settings, **kwargs: (
            {
                "symbol": kwargs["symbol"],
                "market": kwargs["market"],
                "analysis": {"score": 72, "convergence": 0.9},
                "latest": {"close": 1.0},
                "trade_plan": {"long": {}, "short": {}},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        scanner,
        "build_live_execution_plan",
        lambda _settings, _strategy, analysis, **_kwargs: _plan(
            symbol=analysis["symbol"],
            allowed=False,
            violations=["Recent expectancy is below threshold."],
        ),
    )

    payload = scanner.run_ai_readiness_scan(output_dir=tmp_path)

    report = payload["research_candidate_report"]
    assert report["horizon_counts"] == {"short": 1, "medium": 1, "long": 1}
    assert report["reviewable_horizon_counts"] == {"short": 1, "medium": 1, "long": 1}
    assert not any(gap.endswith("_horizon_candidate") for gap in report["coverage_gaps"])
    assert "missing_reviewable_sell_short_research_candidate" in report["coverage_gaps"]
    assert "missing_reviewable_sell_long_research_candidate" in report["coverage_gaps"]
    assert "missing_reviewable_buy_long_research_candidate" not in report["coverage_gaps"]
    assert report["expectancy_improvement_targets"] == {
        "profit_factor_min": 1.0,
        "expectancy_r_min": 0.0,
        "sample_count_min": 30,
        "stop_loss_ratio_max": 0.55,
        "risk_ceiling_pct": 0.025,
    }
    assert {item["horizon"] for item in report["top_candidates"]} == {"short", "medium", "long"}


def test_ai_readiness_scan_adds_promising_risk_combo_matrix_surfaces(
    monkeypatch,
    tmp_path,
) -> None:
    matrix_dir = tmp_path / "risk-combo-matrix"
    matrix_dir.mkdir(exist_ok=True)
    matrix_path = matrix_dir / "20260511T010203Z-risk-combo-matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "mode": "risk_combo_side_interval_matrix_v1",
                "best_surface": {
                    "surface": "buy_1d",
                    "symbol": "TRXUSDT",
                    "target_side": "BUY",
                    "target_interval": "1d",
                    "route_id": "trx-mean-reversion",
                    "research_status": "promising_but_under_validated",
                    "source_report_path": "state/risk-combo-sweeps/trx-buy-1d.json",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(scanner, "RISK_COMBO_MATRIX_DIR", matrix_dir)
    monkeypatch.setattr(scanner, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        scanner,
        "run_hermes_ai_trader",
        lambda **_kwargs: {
            "report_path": str(tmp_path / "hermes.json"),
            "candidate_queue": [],
        },
    )
    monkeypatch.setattr(scanner, "load_settings", lambda: object())
    monkeypatch.setattr(scanner, "load_strategy_config", lambda _path: type(
        "Strategy",
        (),
        {
            "defaults": type(
                "Defaults",
                (),
                {
                    "market": "futures",
                    "interval": "4h",
                    "limit": 600,
                    "use_blave": False,
                },
            )()
        },
    )())
    monkeypatch.setattr(
        scanner,
        "run_analysis",
        lambda _settings, **kwargs: (
            {
                "symbol": kwargs["symbol"],
                "market": kwargs["market"],
                "analysis": {"score": 78, "convergence": 0.9},
                "latest": {"close": 1.0},
                "trade_plan": {"long": {}, "short": {}},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        scanner,
        "build_live_execution_plan",
        lambda _settings, _strategy, analysis, **_kwargs: _plan(
            symbol=analysis["symbol"],
            allowed=False,
            violations=["Recent expectancy is below threshold."],
        ),
    )

    payload = scanner.run_ai_readiness_scan(output_dir=tmp_path)

    assert payload["candidate_count"] == 1
    assert payload["risk_combo_matrix_candidate_count"] == 1
    assert payload["risk_combo_matrix_report"] == str(matrix_path)
    assert payload["scan_results"][0]["symbol"] == "TRXUSDT"
    assert payload["scan_results"][0]["interval"] == "1d"
    report = payload["research_candidate_report"]
    assert report["reviewable_horizon_counts"] == {"long": 1}
    assert report["top_candidates"][0]["horizon"] == "long"
    assert report["top_candidates"][0]["trade_readiness_allowed"] is False


def test_research_candidate_report_marks_market_only_near_ready_candidate() -> None:
    result = scanner.CandidateScanResult(
        rank=500,
        symbol="TRXUSDT",
        side="BUY",
        interval="1d",
        route_id="trx-mean-reversion",
        strategy_family="risk_combo_matrix",
        machine_state="candidate_ready",
        pre_gate_allowed=True,
        scanned=True,
        allowed=False,
        next_action="wait_for_market_state",
        blocker_taxonomy={"market_state": ["Volume z-score is below floor."]},
        warning_taxonomy={},
        live_plan={
            "analysis_score": 70,
            "analysis_convergence": 1.0,
            "adx_value": 47.7,
            "planned_account_risk_pct": 0.0001,
            "gross_notional_usdt": 15.0,
            "min_notional_usdt": 5.0,
            "professional_entry_gate": {
                "layers": {
                    "execution_quality": {"reward_risk": 2.8, "net_profit_to_risk": 2.7},
                    "strategy_performance": {
                        "count": 76,
                        "profit_factor": 1.9907,
                        "expectancy_r": 0.4666,
                        "payoff_ratio": 2.8897,
                        "stop_loss_ratio": 0.4342,
                    },
                }
            },
        },
        error="",
    )

    report = scanner._build_research_candidate_report([result])

    assert report["trade_allowed_count"] == 0
    assert report["near_ready_count"] == 1
    assert report["near_ready_candidates"][0]["symbol"] == "TRXUSDT"
    assert report["near_ready_candidates"][0]["positive_expectancy_gap"] == []
    assert report["near_ready_candidates"][0]["near_ready_market_only"] is True


def test_ai_readiness_scan_reports_research_coverage_gaps(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(scanner, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        scanner,
        "run_hermes_ai_trader",
        lambda **_kwargs: {
            "report_path": str(tmp_path / "hermes.json"),
            "candidate_queue": [_queue_item(1, "SOLUSDT")],
        },
    )
    monkeypatch.setattr(scanner, "load_settings", lambda: object())
    monkeypatch.setattr(scanner, "load_strategy_config", lambda _path: type(
        "Strategy",
        (),
        {
            "defaults": type(
                "Defaults",
                (),
                {
                    "market": "futures",
                    "interval": "4h",
                    "limit": 600,
                    "use_blave": False,
                },
            )()
        },
    )())
    monkeypatch.setattr(
        scanner,
        "run_analysis",
        lambda _settings, **kwargs: (
            {
                "symbol": kwargs["symbol"],
                "market": kwargs["market"],
                "analysis": {"score": 72, "convergence": 0.9},
                "latest": {"close": 1.0},
                "trade_plan": {"long": {}, "short": {}},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        scanner,
        "build_live_execution_plan",
        lambda _settings, _strategy, analysis, **_kwargs: _plan(
            symbol=analysis["symbol"],
            allowed=False,
            violations=["Recent expectancy is below threshold."],
        ),
    )

    payload = scanner.run_ai_readiness_scan(output_dir=tmp_path)

    report = payload["research_candidate_report"]
    assert report["coverage_gaps"] == [
        "missing_reviewable_sell_research_candidate",
        "missing_reviewable_short_horizon_candidate",
        "missing_reviewable_long_horizon_candidate",
        "missing_reviewable_buy_short_research_candidate",
        "missing_reviewable_buy_long_research_candidate",
        "missing_reviewable_sell_short_research_candidate",
        "missing_reviewable_sell_medium_research_candidate",
        "missing_reviewable_sell_long_research_candidate",
        "no_trade_readiness_allowed_candidate",
    ]
    assert report["research_next_actions"] == [
        "expand_short_and_long_interval_research_lanes",
        "repair_or_expand_short_side_candidate_generation",
        "improve_expectancy_before_promotion",
    ]
    expansion = report["research_expansion_plan"]
    surfaces = {item["surface"] for item in expansion}
    assert "sell_medium_research" in surfaces
    assert "buy_short_research" in surfaces
    assert "buy_long_research" in surfaces
    assert all(item["purpose"] == "research_only_candidate_generation" for item in expansion)
    assert all(item["promotion_boundary"] == "does_not_change_live_readiness_or_mainnet_permission" for item in expansion)
    assert all("--target-profit-factor 1.0" in item["command"] for item in expansion)
    assert all(f"--target-side {item['target_side']}" in item["command"] for item in expansion)
    assert all(f"--target-interval {item['target_interval']}" in item["command"] for item in expansion)
    assert all("--limit 600" in item["command"] for item in expansion)
    assert all("--grid-mode fast" in item["command"] for item in expansion)
    assert all("--min-test-trades 10" in item["command"] for item in expansion)
    assert all("--max-configs 8" in item["command"] for item in expansion)
    assert all("--max-walk-forward-validations 1" in item["command"] for item in expansion)
    assert all("--top-n 5" in item["command"] for item in expansion)
    assert all("--skip-news" in item["command"] for item in expansion)
    assert all("--side " not in item["command"] for item in expansion)
    assert all("--interval " not in item["command"] for item in expansion)
    assert all(item["smoke_sweep_budget"]["max_configs"] == 8 for item in expansion)
    assert all(item["smoke_sweep_budget"]["limit"] == 600 for item in expansion)
    assert {item["target_side"] for item in expansion} == {"BUY", "SELL"}
    assert {item["target_interval"] for item in expansion} == {"15m", "4h", "1d"}


def test_ai_readiness_scan_writes_json_when_plan_contains_infinite_values(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(scanner, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        scanner,
        "run_hermes_ai_trader",
        lambda **_kwargs: {
            "report_path": str(tmp_path / "hermes.json"),
            "candidate_queue": [_queue_item(1, "DOGEUSDT")],
        },
    )
    monkeypatch.setattr(scanner, "load_settings", lambda: object())
    monkeypatch.setattr(scanner, "load_strategy_config", lambda _path: type(
        "Strategy",
        (),
        {
            "defaults": type(
                "Defaults",
                (),
                {
                    "market": "futures",
                    "interval": "4h",
                    "limit": 600,
                    "use_blave": False,
                },
            )()
        },
    )())
    monkeypatch.setattr(
        scanner,
        "run_analysis",
        lambda _settings, **kwargs: (
            {
                "symbol": kwargs["symbol"],
                "market": kwargs["market"],
                "analysis": {"score": 80, "convergence": 0.82},
                "latest": {"close": 1.0},
                "trade_plan": {"long": {}, "short": {}},
            },
            None,
        ),
    )

    def fake_plan(_settings, _strategy, analysis, **_kwargs):
        plan = _plan(symbol=analysis["symbol"], allowed=True)
        original = plan.to_dict
        return type(
            "InfPlan",
            (),
            {
                "to_dict": lambda self: {
                    **original(),
                    "challenge": {
                        **original()["challenge"],
                        "historical_signal_risk": {"profit_factor": float("inf")},
                    },
                }
            },
        )()

    monkeypatch.setattr(scanner, "build_live_execution_plan", fake_plan)

    payload = scanner.run_ai_readiness_scan(output_dir=tmp_path)

    report_text = Path(payload["report_path"]).read_text(encoding="utf-8")
    assert '"profit_factor": "inf"' in report_text
