from pathlib import Path

from binance_quant_control.intent_router import resolve_operator_intent


def test_resolve_operator_intent_matches_simulation_upgrade() -> None:
    intent = resolve_operator_intent(
        "我加入了模擬單，測試新策略要先進去模擬單做測試",
        Path("config/operator-intent.default.yaml"),
    )

    assert intent.intent_id == "simulation-first-upgrade"
    assert "paper-order" in intent.actions


def test_resolve_operator_intent_falls_back_when_unmatched() -> None:
    intent = resolve_operator_intent(
        "just inspect the system",
        Path("config/operator-intent.default.yaml"),
    )

    assert intent.intent_id == "fallback"


def test_resolve_operator_intent_maps_start_and_end_trading() -> None:
    start = resolve_operator_intent("開始交易", Path("config/operator-intent.default.yaml"))
    stop = resolve_operator_intent("結束交易", Path("config/operator-intent.default.yaml"))

    assert start.intent_id == "hermes-start-continuous-testnet-trading"
    assert start.actions == ("trade-session start",)
    assert stop.intent_id == "hermes-stop-continuous-testnet-trading"
    assert stop.actions == ("trade-session stop",)


def test_resolve_operator_intent_maps_new_symbol_trade_pipeline() -> None:
    intent = resolve_operator_intent(
        "將任意幣丟進去，從新幣到交易做成流水線並完美嵌入Hermes",
        Path("config/operator-intent.default.yaml"),
    )

    assert intent.intent_id == "new-symbol-trade-pipeline"
    assert "new-symbol-workflow" in intent.actions
    assert "risk-combo-sweep" in intent.actions
    assert "docs/workflows/new-symbol-to-trade-pipeline.md" in intent.resources
