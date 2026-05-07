from __future__ import annotations

import binance_quant_control.route_risk_control as route_risk


def test_update_route_quarantine_from_snapshot_blocks_route(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(route_risk, "ROUTE_RISK_CONTROL_PATH", tmp_path / "route-risk.json")

    state = route_risk.update_route_quarantine_from_snapshot(
        {
            "routes": {
                "btc-core": {
                    "quarantined": True,
                    "quarantine_reasons": ["profit-factor 0.4 below floor"],
                    "count": 40,
                    "profit_factor": 0.4,
                    "loss_streak": 6,
                }
            }
        },
        updated_by="pytest",
    )

    assert state["active_quarantined_routes"] == ["btc-core"]
    status = route_risk.route_quarantine_status("btc-core")
    assert status["quarantined"] is True
    assert status["manual_review_required"] is True
    assert "profit-factor" in status["reasons"][0]


def test_clear_route_quarantine_requires_manual_action(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(route_risk, "ROUTE_RISK_CONTROL_PATH", tmp_path / "route-risk.json")
    route_risk.update_route_quarantine_from_snapshot(
        {"routes": {"major-alt-trend": {"quarantined": True, "quarantine_reasons": ["losses"]}}},
        updated_by="pytest",
    )

    route_risk.clear_route_quarantine(
        "major-alt-trend",
        reason="manual review completed",
        updated_by="pytest",
    )

    status = route_risk.route_quarantine_status("major-alt-trend")
    assert status["quarantined"] is False
    assert status["manual_review_required"] is False
