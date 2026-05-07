from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .binance_api import BinanceClient
from .config import Settings
from .live_execution import (
    _reserve_runner_from_weights,
    _round_price_to_tick,
    _split_reduce_only_quantities,
    _take_profit_prices,
    _take_profit_weights,
    _trailing_runner_weight,
)
from .strategy import StrategyConfig
from .trading_control import AUTO_PAUSE_ACTOR, load_trading_control_state


@dataclass(frozen=True, slots=True)
class StagedTakeProfitRepairPlan:
    symbol: str
    side: str
    quantity: float
    stop_price: float
    take_profit_prices: list[float]
    take_profit_quantities: list[float]
    take_profit_weights: list[float]
    take_profit_runner_quantity: float
    allowed: bool
    violations: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _active_position(raw_positions: list[dict[str, Any]], symbol: str) -> dict[str, Any]:
    symbol_upper = symbol.upper()
    for item in raw_positions:
        if item.get("symbol") == symbol_upper and abs(float(item.get("positionAmt", 0.0))) > 0.0:
            return item
    return {}


def _cancel_all_algo_orders(client: BinanceClient, symbol: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for order in client.open_algo_orders(symbol):
        algo_id = order.get("algoId")
        if algo_id is None:
            continue
        results.append(client.cancel_algo_order(symbol, int(algo_id)))
    return results


def _cancel_replaced_algo_orders(
    client: BinanceClient,
    symbol: str,
    *,
    preserve_algo_ids: set[int],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for order in client.open_algo_orders(symbol):
        algo_id = order.get("algoId")
        if algo_id is None:
            continue
        normalized_algo_id = int(algo_id)
        if normalized_algo_id in preserve_algo_ids:
            continue
        results.append(client.cancel_algo_order(symbol, normalized_algo_id))
    return results


def _algo_id(response: Any) -> int | None:
    if not isinstance(response, dict):
        return None
    value = response.get("algoId")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_staged_take_profit_repair_plan(
    settings: Settings,
    strategy: StrategyConfig,
    *,
    symbol: str,
    side_plan: dict[str, Any],
    confidence: float = 0.75,
    route_id: str = "manual-repair",
    news_risk: dict[str, Any] | None = None,
) -> StagedTakeProfitRepairPlan:
    symbol_upper = symbol.upper()
    violations: list[str] = []
    warnings: list[str] = []
    with BinanceClient(settings) as client:
        position = _active_position(client.positions(symbol_upper), symbol_upper)
        rules_info = client.exchange_info(symbol_upper, strategy.defaults.market)

    if not position:
        violations.append(f"No active futures position exists for {symbol_upper}.")
        quantity = 0.0
        side = "HOLD"
    else:
        position_amt = float(position.get("positionAmt", 0.0))
        quantity = abs(position_amt)
        side = "BUY" if position_amt > 0 else "SELL"

    symbol_row = next((item for item in (rules_info.get("symbols") or []) if item.get("symbol") == symbol_upper), {})
    filters = {item["filterType"]: item for item in symbol_row.get("filters", [])}
    market_lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE") or {}
    price_filter = filters.get("PRICE_FILTER") or {}
    step_size = float(market_lot.get("stepSize", 0.0))
    tick_size = float(price_filter.get("tickSize", 0.0))
    quantity_precision = int(symbol_row.get("quantityPrecision", 8))
    price_precision = int(symbol_row.get("pricePrecision", 8))
    price_round_mode = "down" if side == "BUY" else "up" if side == "SELL" else "nearest"
    stop_price = float(side_plan.get("invalidation") or 0.0)
    if stop_price <= 0:
        violations.append("Side plan does not include a valid invalidation stop.")
    else:
        stop_price = _round_price_to_tick(
            stop_price,
            tick_size=tick_size,
            precision=price_precision,
            mode=price_round_mode,
        )
    take_profit_prices = _take_profit_prices(side_plan)
    if not take_profit_prices:
        violations.append("Side plan does not include take-profit levels.")
    take_profit_prices = [
        _round_price_to_tick(
            item,
            tick_size=tick_size,
            precision=price_precision,
            mode=price_round_mode,
        )
        for item in take_profit_prices
    ]
    weights = _take_profit_weights(
        parts=len(take_profit_prices),
        route_id=route_id,
        confidence=confidence,
        news_risk=news_risk,
    )
    runner_weight = _trailing_runner_weight(
        trailing_stop_enabled=strategy.risk.trailing_stop_enabled,
        parts=len(take_profit_prices),
        route_id=route_id,
        confidence=confidence,
        news_risk=news_risk,
    )
    weights = _reserve_runner_from_weights(weights, runner_weight)
    quantities = _split_reduce_only_quantities(quantity, step_size, quantity_precision, weights)
    if quantity > 0 and take_profit_prices and not quantities:
        take_profit_prices = [take_profit_prices[-1]]
        quantities = [round(quantity, quantity_precision)]
        weights = [1.0]
        warnings.append(
            "Position is too small for a staged take-profit ladder after exchange rounding; "
            "using a full-position high-payoff take-profit fallback."
        )
    runner_quantity = round(max(0.0, quantity - sum(quantities)), quantity_precision)
    if not strategy.risk.trailing_stop_enabled and quantities and round(sum(quantities), quantity_precision) != round(quantity, quantity_precision):
        warnings.append("Take-profit quantities do not sum exactly to position quantity after exchange rounding.")

    return StagedTakeProfitRepairPlan(
        symbol=symbol_upper,
        side=side,
        quantity=round(quantity, quantity_precision),
        stop_price=stop_price,
        take_profit_prices=take_profit_prices,
        take_profit_quantities=quantities,
        take_profit_weights=[round(item, 4) for item in weights],
        take_profit_runner_quantity=runner_quantity,
        allowed=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


def execute_staged_take_profit_repair(
    settings: Settings,
    strategy: StrategyConfig,
    plan: StagedTakeProfitRepairPlan,
) -> dict[str, Any]:
    if not plan.allowed:
        raise RuntimeError("Staged TP repair plan is blocked.")
    trading_control = load_trading_control_state()
    if trading_control.paused and trading_control.updated_by != AUTO_PAUSE_ACTOR:
        reason = trading_control.reason or "manual kill-switch"
        raise RuntimeError(f"Trading is paused by kill-switch: {reason}.")
    exit_side = "SELL" if plan.side == "BUY" else "BUY"
    with BinanceClient(settings) as client:
        stop_loss = client.new_algo_order(
            plan.symbol,
            exit_side,
            "STOP_MARKET",
            trigger_price=plan.stop_price,
            quantity=plan.quantity,
            reduce_only=True,
            working_type="MARK_PRICE",
            market=strategy.defaults.market,
        )
        preserve_algo_ids = {value for value in (_algo_id(stop_loss),) if value is not None}
        take_profits = []
        for index, (tp_price, tp_quantity) in enumerate(
            zip(plan.take_profit_prices, plan.take_profit_quantities, strict=False),
            start=1,
        ):
            take_profits.append(
                {
                    "level": index,
                    "trigger_price": tp_price,
                    "quantity": tp_quantity,
                    "response": client.new_algo_order(
                        plan.symbol,
                        exit_side,
                        "TAKE_PROFIT_MARKET",
                        trigger_price=tp_price,
                        quantity=tp_quantity,
                        reduce_only=True,
                        working_type="MARK_PRICE",
                        market=strategy.defaults.market,
                    ),
                }
            )
            algo_id = _algo_id(take_profits[-1]["response"])
            if algo_id is not None:
                preserve_algo_ids.add(algo_id)
        cancelled = _cancel_replaced_algo_orders(
            client,
            plan.symbol,
            preserve_algo_ids=preserve_algo_ids,
        )
    return {
        "cancelled_algo_orders": cancelled,
        "submitted": {
            "stop_loss": stop_loss,
            "take_profits": take_profits,
        },
    }
