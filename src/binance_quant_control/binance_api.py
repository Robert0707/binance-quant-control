from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any
from urllib.parse import urlencode

import httpx

from .config import Settings

SPOT_BASE = "https://api.binance.com"
SPOT_TESTNET_BASE = "https://testnet.binance.vision"
FUTURES_BASE = "https://fapi.binance.com"
FUTURES_TESTNET_BASE = "https://demo-fapi.binance.com"


class BinanceAPIError(RuntimeError):
    pass


class LiveTradingDisabledError(RuntimeError):
    """Raised when a write operation is attempted while live trading is disabled."""

    def __init__(self) -> None:
        super().__init__(
            "Live trading is disabled. Set BINANCE_LIVE_TRADING_ENABLED=true "
            "in .env to enable real order execution, or enable "
            "BINANCE_TESTNET_TRADING_ENABLED=true while BINANCE_USE_TESTNET=true "
            "for Binance testnet execution."
        )


def _binance_error_detail(response: httpx.Response) -> str:
    detail = response.text.strip()
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        code = payload.get("code")
        msg = payload.get("msg")
        if code is not None or msg is not None:
            return f"Binance API error status={response.status_code} code={code} msg={msg}"
    if detail:
        return f"Binance API error status={response.status_code}: {detail}"
    return f"Binance API error status={response.status_code}"


class BinanceClient:
    def __init__(self, settings: Settings, timeout: float = 20.0) -> None:
        self.settings = settings
        self.client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "BinanceClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def base_url(self, market: str, *, signed: bool = False) -> str:
        market = market.lower()
        if market == "spot":
            if signed and self.settings.use_testnet:
                return SPOT_TESTNET_BASE
            return SPOT_BASE
        if market == "futures":
            if signed and self.settings.use_testnet:
                return FUTURES_TESTNET_BASE
            return FUTURES_BASE
        raise ValueError(f"unsupported market: {market}")

    def _require_live_trading(self) -> None:
        """Gate for all write operations."""
        if self.settings.live_trading_enabled:
            return
        testnet_write_allowed = bool(
            getattr(self.settings, "use_testnet", False)
            and getattr(self.settings, "testnet_trading_enabled", False)
        )
        if testnet_write_allowed:
            return
        raise LiveTradingDisabledError()

    def _request(
        self,
        method: str,
        market: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        signed: bool = False,
    ) -> Any:
        params = {key: value for key, value in (params or {}).items() if value is not None}
        headers: dict[str, str] = {}
        payload = params.copy()
        data = None
        request_params = None

        if signed:
            if not self.settings.has_binance_credentials:
                raise BinanceAPIError("Binance API credentials are missing.")
            payload["timestamp"] = int(time.time() * 1000)
            payload.setdefault("recvWindow", self.settings.recv_window_ms)
            query = urlencode(payload, doseq=True)
            signature = hmac.new(
                self.settings.active_binance_secret_key.encode("utf-8"),
                query.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            payload["signature"] = signature
            headers["X-MBX-APIKEY"] = self.settings.active_binance_api_key

        if method.upper() in {"POST", "PUT", "DELETE"}:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            data = urlencode(payload, doseq=True)
        else:
            request_params = payload

        response = self.client.request(
            method.upper(),
            f"{self.base_url(market, signed=signed)}{path}",
            params=request_params,
            data=data,
            headers=headers,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise BinanceAPIError(_binance_error_detail(exc.response)) from exc
        result = response.json()
        if isinstance(result, dict):
            code = result.get("code")
            try:
                numeric_code = int(code) if code is not None else None
            except (TypeError, ValueError):
                numeric_code = None
            if (
                code is not None
                and numeric_code not in (200, 0)
                and not (numeric_code is not None and numeric_code >= 200 and numeric_code < 300)
            ):
                raise BinanceAPIError(_binance_error_detail(response))
        return result

    # ── read-only endpoints ──────────────────────────────────────────

    def ping(self, market: str) -> Any:
        path = "/api/v3/time" if market == "spot" else "/fapi/v1/time"
        return self._request("GET", market, path)

    def klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        market: str,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> Any:
        path = "/api/v3/klines" if market == "spot" else "/fapi/v1/klines"
        return self._request(
            "GET",
            market,
            path,
            params={
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": int(limit),
                "startTime": start_time,
                "endTime": end_time,
            },
        )

    def ticker_price(self, symbol: str, market: str) -> float:
        path = "/api/v3/ticker/price" if market == "spot" else "/fapi/v1/ticker/price"
        payload = self._request("GET", market, path, params={"symbol": symbol.upper()})
        return float(payload["price"])

    def order_book(self, symbol: str, market: str = "futures", limit: int = 20) -> Any:
        path = "/api/v3/depth" if market == "spot" else "/fapi/v1/depth"
        return self._request(
            "GET",
            market,
            path,
            params={"symbol": symbol.upper(), "limit": int(limit)},
        )

    def funding_rate_history(self, symbol: str, limit: int = 2) -> list[Any]:
        result = self._request(
            "GET",
            "futures",
            "/fapi/v1/fundingRate",
            params={"symbol": symbol.upper(), "limit": int(limit)},
        )
        return result if isinstance(result, list) else []

    def open_interest_hist(self, symbol: str, period: str = "5m", limit: int = 2) -> list[Any]:
        result = self._request(
            "GET",
            "futures",
            "/futures/data/openInterestHist",
            params={"symbol": symbol.upper(), "period": period, "limit": int(limit)},
        )
        return result if isinstance(result, list) else []

    def account(self, market: str) -> Any:
        path = "/api/v3/account" if market == "spot" else "/fapi/v2/account"
        return self._request("GET", market, path, signed=True)

    def balance(self, market: str) -> Any:
        if market == "spot":
            return self.account("spot")
        return self._request("GET", market, "/fapi/v2/balance", signed=True)

    def positions(self, symbol: str | None = None) -> Any:
        params = {"symbol": symbol.upper()} if symbol else None
        return self._request("GET", "futures", "/fapi/v2/positionRisk", params=params, signed=True)

    def open_orders(self, symbol: str | None = None, market: str = "futures") -> Any:
        path = "/api/v3/openOrders" if market == "spot" else "/fapi/v1/openOrders"
        params = {"symbol": symbol.upper()} if symbol else {}
        return self._request("GET", market, path, params=params, signed=True)

    def income_history(
        self,
        symbol: str | None = None,
        *,
        income_type: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 100,
    ) -> list[Any]:
        params: dict[str, Any] = {"limit": limit}
        if symbol:
            params["symbol"] = symbol.upper()
        if income_type:
            params["incomeType"] = income_type
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        result = self._request("GET", "futures", "/fapi/v1/income", params=params, signed=True)
        return result if isinstance(result, list) else []

    def exchange_info(self, symbol: str, market: str = "futures") -> Any:
        path = "/api/v3/exchangeInfo" if market == "spot" else "/fapi/v1/exchangeInfo"
        params = {"symbol": symbol.upper()} if symbol else None
        return self._request("GET", market, path, params=params)

    def ticker_24hr(self, market: str = "futures") -> Any:
        path = "/api/v3/ticker/24hr" if market == "spot" else "/fapi/v1/ticker/24hr"
        return self._request("GET", market, path)

    # ── write endpoints (require live_trading_enabled) ───────────────

    def set_leverage(self, symbol: str, leverage: int) -> Any:
        """Set futures leverage for a symbol."""
        self._require_live_trading()
        return self._request(
            "POST",
            "futures",
            "/fapi/v1/leverage",
            params={"symbol": symbol.upper(), "leverage": leverage},
            signed=True,
        )

    def set_margin_type(self, symbol: str, margin_type: str) -> Any:
        """Set margin type (ISOLATED or CROSSED) for a symbol.

        Binance returns an error if the margin type is already set to
        the requested value, so we silently swallow code -4046.
        """
        self._require_live_trading()
        try:
            return self._request(
                "POST",
                "futures",
                "/fapi/v1/marginType",
                params={"symbol": symbol.upper(), "marginType": margin_type.upper()},
                signed=True,
            )
        except BinanceAPIError as exc:
            if "-4046" in str(exc):
                return {"msg": "No need to change margin type."}
            raise

    def new_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        *,
        quantity: float | None = None,
        price: float | None = None,
        stop_price: float | None = None,
        time_in_force: str | None = None,
        reduce_only: bool = False,
        close_position: bool = False,
        working_type: str | None = None,
        callback_rate: float | None = None,
        activation_price: float | None = None,
        market: str = "futures",
    ) -> Any:
        """Submit a new order.  Requires live trading to be enabled."""
        self._require_live_trading()
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
        }
        if quantity is not None:
            params["quantity"] = quantity
        if price is not None:
            params["price"] = price
        if stop_price is not None:
            params["stopPrice"] = stop_price
        if time_in_force:
            params["timeInForce"] = time_in_force
        if reduce_only:
            params["reduceOnly"] = "true"
        if close_position:
            params["closePosition"] = "true"
        if working_type:
            params["workingType"] = working_type
        if callback_rate is not None:
            params["callbackRate"] = callback_rate
        if activation_price is not None:
            params["activationPrice"] = activation_price

        path = "/api/v3/order" if market == "spot" else "/fapi/v1/order"
        return self._request("POST", market, path, params=params, signed=True)

    def new_algo_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        *,
        trigger_price: float,
        quantity: float | None = None,
        callback_rate: float | None = None,
        activation_price: float | None = None,
        close_position: bool = False,
        reduce_only: bool = False,
        working_type: str | None = None,
        price_protect: bool = False,
        position_side: str | None = None,
        market: str = "futures",
    ) -> Any:
        """Submit a futures algo order for STOP/TAKE_PROFIT style protection.

        Uses POST /fapi/v1/algoOrder with algoType=CONDITIONAL.
        The trigger_price maps to the 'triggerPrice' request parameter.
        closePosition and reduceOnly are mutually exclusive.
        """
        self._require_live_trading()
        if market != "futures":
            raise ValueError("algo orders are only supported for futures")
        params: dict[str, Any] = {
            "algoType": "CONDITIONAL",
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "triggerPrice": trigger_price,
        }
        if quantity is not None:
            params["quantity"] = quantity
        if callback_rate is not None:
            params["callbackRate"] = callback_rate
        if activation_price is not None:
            params["activationPrice"] = activation_price
        if close_position:
            params["closePosition"] = "true"
        elif reduce_only:
            # closePosition and reduceOnly cannot both be true
            params["reduceOnly"] = "true"
        if working_type:
            params["workingType"] = working_type
        if price_protect:
            params["priceProtect"] = "TRUE"
        if position_side:
            params["positionSide"] = position_side.upper()
        return self._request("POST", market, "/fapi/v1/algoOrder", params=params, signed=True)

    def cancel_algo_order(self, symbol: str, algo_id: int) -> Any:
        """Cancel a single algo order by ID."""
        self._require_live_trading()
        return self._request(
            "DELETE",
            "futures",
            "/fapi/v1/algoOrder",
            params={"symbol": symbol.upper(), "algoId": algo_id},
            signed=True,
        )

    def cancel_all_algo_orders(self, symbol: str) -> list[Any]:
        """Cancel ALL open algo orders for a symbol.  Returns list of cancel results."""
        self._require_live_trading()
        open_orders = self.open_algo_orders(symbol)
        results = []
        for order in open_orders:
            algo_id = order.get("algoId")
            if algo_id:
                try:
                    result = self.cancel_algo_order(symbol, algo_id)
                    results.append({"algoId": algo_id, "status": "cancelled", "response": result})
                except BinanceAPIError as exc:
                    results.append({"algoId": algo_id, "status": "error", "error": str(exc)})
        return results

    def open_algo_orders(self, symbol: str | None = None) -> list[Any]:
        """Query all currently open algo orders."""
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        result = self._request("GET", "futures", "/fapi/v1/openAlgoOrders", params=params, signed=True)
        return result if isinstance(result, list) else []

    def all_algo_orders(self, symbol: str | None = None, limit: int = 100) -> list[Any]:
        """Query all algo orders (open + closed + cancelled)."""
        params: dict[str, Any] = {"limit": limit}
        if symbol:
            params["symbol"] = symbol.upper()
        result = self._request("GET", "futures", "/fapi/v1/allAlgoOrders", params=params, signed=True)
        return result if isinstance(result, list) else []

    def cancel_order(self, symbol: str, order_id: int, market: str = "futures") -> Any:
        """Cancel an open order."""
        self._require_live_trading()
        path = "/api/v3/order" if market == "spot" else "/fapi/v1/order"
        return self._request(
            "DELETE",
            market,
            path,
            params={"symbol": symbol.upper(), "orderId": order_id},
            signed=True,
        )

    def query_order(self, symbol: str, order_id: int, market: str = "futures") -> Any:
        """Query an order by ID."""
        path = "/api/v3/order" if market == "spot" else "/fapi/v1/order"
        return self._request(
            "GET",
            market,
            path,
            params={"symbol": symbol.upper(), "orderId": order_id},
            signed=True,
        )
