from __future__ import annotations

from binance_quant_control.event_bus import EventBus, TradingEvent, default_plugin_lifecycle


def test_event_bus_publishes_to_specific_and_wildcard_handlers() -> None:
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe("signal.created", lambda event: seen.append(event.event_type))
    bus.subscribe("*", lambda event: seen.append(f"all:{event.event_type}"))

    bus.publish(TradingEvent("signal.created", {"symbol": "BTCUSDT"}))

    assert seen == ["signal.created", "all:signal.created"]
    assert bus.history()[0].payload["symbol"] == "BTCUSDT"


def test_default_plugin_lifecycle_keeps_execution_disabled_until_gates_pass() -> None:
    plugins = {plugin.name: plugin for plugin in default_plugin_lifecycle()}

    assert plugins["alpha"].enabled is True
    assert plugins["alpha"].input_events == ("features.ready",)
    assert plugins["risk"].output_events == ("gate.evaluated",)
    assert plugins["execution"].enabled is False
