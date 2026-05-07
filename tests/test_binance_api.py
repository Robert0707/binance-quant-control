from __future__ import annotations

import httpx

from binance_quant_control.binance_api import BinanceAPIError, BinanceClient
from binance_quant_control.config import load_settings


def test_binance_private_error_surfaces_exchange_code_and_message(monkeypatch):
    settings = load_settings()

    def fake_request(*args, **kwargs):
        request = httpx.Request("GET", "https://fapi.binance.com/fapi/v2/balance")
        return httpx.Response(
            401,
            request=request,
            json={"code": -2015, "msg": "Invalid API-key, IP, or permissions for action"},
        )

    with BinanceClient(settings) as client:
        monkeypatch.setattr(client.client, "request", fake_request)
        try:
            client.balance("futures")
        except BinanceAPIError as exc:
            message = str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("Expected BinanceAPIError")

    assert "status=401" in message
    assert "code=-2015" in message
    assert "Invalid API-key, IP, or permissions for action" in message


def test_income_history_returns_list(monkeypatch):
    settings = load_settings()

    def fake_request(*args, **kwargs):
        request = httpx.Request("GET", "https://fapi.binance.com/fapi/v1/income")
        return httpx.Response(
            200,
            request=request,
            json=[{"symbol": "BTCUSDT", "income": "1.23", "incomeType": "REALIZED_PNL"}],
        )

    with BinanceClient(settings) as client:
        monkeypatch.setattr(client.client, "request", fake_request)
        rows = client.income_history("BTCUSDT", income_type="REALIZED_PNL", limit=10)

    assert len(rows) == 1
    assert rows[0]["symbol"] == "BTCUSDT"


def test_string_success_code_is_not_treated_as_error(monkeypatch):
    settings = load_settings()

    def fake_request(*args, **kwargs):
        request = httpx.Request("DELETE", "https://fapi.binance.com/fapi/v1/algoOrder")
        return httpx.Response(
            200,
            request=request,
            json={"code": "200", "msg": "success"},
        )

    with BinanceClient(settings) as client:
        monkeypatch.setattr(client.client, "request", fake_request)
        response = client.cancel_algo_order("DOGEUSDT", 123)

    assert response["code"] == "200"
