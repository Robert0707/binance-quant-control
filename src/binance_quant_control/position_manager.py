from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .analysis import run_analysis
from .binance_api import BinanceAPIError, BinanceClient
from .config import Settings
from .live_execution import _round_price_to_tick
from .strategy import StrategyConfig
from .trading_control import AUTO_PAUSE_ACTOR, load_trading_control_state

PROTECTIVE_ORDER_TYPES = {
    "STOP",
    "STOP_MARKET",
    "TAKE_PROFIT",
    "TAKE_PROFIT_MARKET",
    "TRAILING_STOP_MARKET",
}


@dataclass(frozen=True, slots=True)
class PositionManagementPlan:
    allowed: bool
    symbol: str
    market: str
    side: str
    quantity: float
    entry_price: float
    mark_price: float
    unrealized_pnl_usdt: float
    leverage: int
    existing_open_orders: list[dict[str, Any]]
    existing_algo_orders: list[dict[str, Any]]
    step_size: float
    tick_size: float
    quantity_precision: int
    price_precision: int
    proposed_stop_price: float | None
    proposed_take_profit_price: float | None
    trailing_activation_price: float | None
    trailing_callback_pct: float | None
    trailing_quantity: float | None
    cancel_existing_algo_orders: bool
    preserve_existing_take_profits: bool
    violations: list[str]
    warnings: list[str]
    actions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AdaptiveExitPlan:
    allowed: bool
    symbol: str
    market: str
    side: str
    exit_side: str
    quantity: float
    action: str
    reason_code: str
    reasons: list[str]
    warnings: list[str]
    unrealized_r: float
    reversal_score: float
    confidence: float
    risk_distance: float
    reference_stop_price: float | None
    entry_price: float
    mark_price: float
    analysis_bias: str
    recommended_action: str
    selected_family: str
    selected_family_bias: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_position(raw_positions: list[dict[str, Any]], symbol: str) -> dict[str, Any]:
    symbol_upper = symbol.upper()
    for item in raw_positions:
        if item.get("symbol") == symbol_upper and abs(float(item.get("positionAmt", 0.0))) > 0.0:
            return item
    return {}


def _protective_regular_orders(open_orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in open_orders if str(item.get("type", "")).upper() in PROTECTIVE_ORDER_TYPES]


def _first_trigger(orders: list[dict[str, Any]], order_type: str) -> float | None:
    target = order_type.upper()
    for item in orders:
        if str(item.get("orderType", item.get("type", ""))).upper() == target:
            trigger = item.get("triggerPrice", item.get("stopPrice"))
            if trigger is not None:
                return float(trigger)
    return None


def _order_quantity(order: dict[str, Any]) -> float:
    for key in ("quantity", "origQty"):
        value = order.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _algo_id(order: dict[str, Any]) -> int | None:
    value = order.get("algoId")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _replaceable_protective_algo_orders(orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    replaceable_types = {"STOP_MARKET", "STOP", "TRAILING_STOP_MARKET"}
    return [
        item
        for item in orders
        if str(item.get("orderType", item.get("type", ""))).upper() in replaceable_types
    ]


def _trailing_algo_orders(orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in orders
        if str(item.get("orderType", item.get("type", ""))).upper() == "TRAILING_STOP_MARKET"
    ]


def _take_profit_algo_orders(orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in orders
        if str(item.get("orderType", item.get("type", ""))).upper() in {"TAKE_PROFIT", "TAKE_PROFIT_MARKET"}
    ]


def _opposite_side(side: str) -> str:
    return "SELL" if side.upper() == "BUY" else "BUY"


def _position_bias(side: str) -> str:
    return "long" if side.upper() == "BUY" else "short"


def _opposite_bias(side: str) -> str:
    return "short" if _position_bias(side) == "long" else "long"


def _symbol_rules(client: BinanceClient, symbol: str, market: str) -> dict[str, float | int]:
    try:
        exchange_info = client.exchange_info(symbol, market)
    except (AttributeError, BinanceAPIError, KeyError, TypeError, ValueError):
        return {"step_size": 0.0, "tick_size": 0.0, "quantity_precision": 8, "price_precision": 8}
    row = next(
        (item for item in (exchange_info.get("symbols") or []) if item.get("symbol") == symbol.upper()),
        {},
    )
    filters = {item.get("filterType"): item for item in row.get("filters", []) if isinstance(item, dict)}
    lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE") or {}
    price_filter = filters.get("PRICE_FILTER") or {}
    return {
        "step_size": float(lot.get("stepSize", 0.0)),
        "tick_size": float(price_filter.get("tickSize", 0.0)),
        "quantity_precision": int(row.get("quantityPrecision", 8)),
        "price_precision": int(row.get("pricePrecision", 8)),
    }


def build_position_management_plan(
    settings: Settings,
    *,
    symbol: str,
    market: str = "futures",
    stop_price: float | None = None,
    take_profit_price: float | None = None,
    strategy: StrategyConfig | None = None,
    enable_trailing_stop: bool = False,
    trailing_callback_pct: float | None = None,
    trailing_activation_price: float | None = None,
) -> PositionManagementPlan:
    symbol_upper = symbol.upper()
    violations: list[str] = []
    warnings: list[str] = []
    actions: list[str] = []

    with BinanceClient(settings) as client:
        raw_positions = client.positions(symbol_upper if market == "futures" else None) if market == "futures" else []
        open_orders = client.open_orders(symbol_upper, market)
        algo_orders = client.open_algo_orders(symbol_upper) if market == "futures" else []
        rules = _symbol_rules(client, symbol_upper, market) if market == "futures" else {
            "step_size": 0.0,
            "tick_size": 0.0,
            "quantity_precision": 8,
            "price_precision": 8,
        }
    step_size = float(rules["step_size"])
    tick_size = float(rules["tick_size"])
    quantity_precision = int(rules["quantity_precision"])
    price_precision = int(rules["price_precision"])

    position = _extract_position(raw_positions, symbol_upper)
    if not position:
        violations.append(f"No non-flat {market} position exists for {symbol_upper}.")
        return PositionManagementPlan(
            allowed=False,
            symbol=symbol_upper,
            market=market,
            side="HOLD",
            quantity=0.0,
            entry_price=0.0,
            mark_price=0.0,
            unrealized_pnl_usdt=0.0,
            leverage=0,
            existing_open_orders=_protective_regular_orders(open_orders),
            existing_algo_orders=algo_orders,
            step_size=step_size,
            tick_size=tick_size,
            quantity_precision=quantity_precision,
            price_precision=price_precision,
            proposed_stop_price=stop_price,
            proposed_take_profit_price=take_profit_price,
            trailing_activation_price=trailing_activation_price,
            trailing_callback_pct=trailing_callback_pct,
            trailing_quantity=None,
            cancel_existing_algo_orders=False,
            preserve_existing_take_profits=False,
            violations=violations,
            warnings=warnings,
            actions=actions,
        )

    position_amt = float(position.get("positionAmt", 0.0))
    side = "BUY" if position_amt > 0 else "SELL"
    quantity = abs(position_amt)
    entry_price = float(position.get("entryPrice", 0.0))
    mark_price = float(position.get("markPrice", 0.0))
    unrealized_pnl_usdt = float(position.get("unRealizedProfit", 0.0))
    leverage = int(float(position.get("leverage", 0)))

    protective_regular_orders = _protective_regular_orders(open_orders)
    existing_stop_price = _first_trigger(protective_regular_orders, "STOP_MARKET") or _first_trigger(algo_orders, "STOP_MARKET")
    existing_stop_orders = [
        item
        for item in algo_orders
        if str(item.get("orderType", item.get("type", ""))).upper() in {"STOP", "STOP_MARKET"}
    ]
    existing_take_profit_orders = _take_profit_algo_orders(algo_orders)
    existing_take_profit_quantity = sum(_order_quantity(item) for item in existing_take_profit_orders)
    replaceable_algo_orders = _replaceable_protective_algo_orders(algo_orders)
    existing_trailing_orders = _trailing_algo_orders(algo_orders)
    preserve_existing_take_profits = False
    trailing_quantity: float | None = None
    expected_ladder_count = len(existing_stop_orders) + len(existing_take_profit_orders) + len(existing_trailing_orders)
    unexpected_algo_count = max(0, len(algo_orders) - expected_ladder_count)
    if protective_regular_orders or unexpected_algo_count > 0 or len(existing_stop_orders) > 1 or len(existing_trailing_orders) > 1:
        warnings.append("Unexpected protective order mix exists; review before replacing protection.")

    if enable_trailing_stop and stop_price is not None:
        violations.append("Choose either a fixed stop update or a trailing stop, not both in one run.")

    if enable_trailing_stop:
        callback_pct = trailing_callback_pct
        if callback_pct is None and strategy is not None:
            callback_pct = strategy.risk.trailing_callback_pct
        if callback_pct is None or callback_pct <= 0:
            violations.append("Trailing stop requires a positive callback percentage.")

        activation_price = trailing_activation_price
        if activation_price is None and strategy is not None:
            reference_stop = existing_stop_price
            if reference_stop is None:
                analysis_payload, _ = run_analysis(
                    settings,
                    symbol=symbol_upper,
                    market=market,
                    interval=strategy.defaults.interval,
                    limit=max(strategy.defaults.limit, 240),
                    use_blave=strategy.defaults.use_blave,
                    render_chart_flag=False,
                    strategy=strategy,
                )
                trade_plan = analysis_payload["trade_plan"]
                if side == "BUY":
                    reference_stop = float(trade_plan["long"]["invalidation"])
                else:
                    reference_stop = float(trade_plan["short"]["invalidation"])
            risk_distance = abs(entry_price - float(reference_stop or entry_price))
            if risk_distance <= 0:
                violations.append("Unable to derive a trailing-stop activation price from the current position.")
            else:
                activation_multiple = strategy.risk.trailing_activation_r_multiple
                if side == "BUY":
                    activation_price = entry_price + (risk_distance * activation_multiple)
                else:
                    activation_price = entry_price - (risk_distance * activation_multiple)
        if activation_price is None:
            violations.append("Trailing stop requires an activation price or a strategy context.")
        else:
            trailing_armed = True
            if side == "BUY" and mark_price < activation_price:
                warnings.append(
                    f"Trailing stop is not armed yet. Mark {mark_price:.4f} is below activation {activation_price:.4f}."
                )
                trailing_armed = False
            if side == "SELL" and mark_price > activation_price:
                warnings.append(
                    f"Trailing stop is not armed yet. Mark {mark_price:.4f} is above activation {activation_price:.4f}."
                )
                trailing_armed = False
            if trailing_armed:
                preserve_existing_take_profits = bool(existing_take_profit_orders)
                if preserve_existing_take_profits:
                    trailing_quantity = round(max(0.0, quantity - existing_take_profit_quantity), quantity_precision)
                    if trailing_quantity <= 0 or (step_size > 0 and trailing_quantity < step_size):
                        trailing_quantity = None
                        warnings.append(
                            "Existing take-profit orders already cover the full position; trailing stop will not be added."
                        )
                    elif existing_trailing_orders:
                        actions.append("Replace old trailing protection and preserve the stop/TP ladder.")
                    else:
                        actions.append("Preserve the stop/TP ladder and place a trailing stop for the runner quantity.")
                else:
                    trailing_quantity = quantity
                    actions.append("Cancel existing protective orders and place a trailing stop for the open position.")
        trailing_activation_price = activation_price
        trailing_callback_pct = callback_pct
    else:
        trailing_activation_price = None
        trailing_callback_pct = None

    if trailing_activation_price is not None:
        round_mode = "down" if side == "BUY" else "up"
        trailing_activation_price = _round_price_to_tick(
            trailing_activation_price,
            tick_size=tick_size,
            precision=price_precision,
            mode=round_mode,
        )
    if stop_price is None and not enable_trailing_stop and existing_stop_price is None:
        warnings.append("No active stop-loss order is protecting the current position.")

    if stop_price is not None:
        actions.append(f"Place STOP_MARKET reduce-only at {stop_price:.4f}.")
    if take_profit_price is not None:
        actions.append(f"Place TAKE_PROFIT_MARKET reduce-only at {take_profit_price:.4f}.")

    if preserve_existing_take_profits:
        cancel_existing_algo_orders = bool(actions) and (
            bool(protective_regular_orders) or bool(existing_trailing_orders)
        )
    else:
        cancel_existing_algo_orders = bool(actions) and (
            bool(protective_regular_orders)
            or bool(replaceable_algo_orders)
            or bool(algo_orders)
        )
    if cancel_existing_algo_orders:
        actions.insert(0, "Cancel existing protective orders before replacing them.")
    if protective_regular_orders:
        warnings.append(
            f"{len(protective_regular_orders)} regular protective order(s) exist; they will be cancelled with algo orders."
        )

    return PositionManagementPlan(
        allowed=len(violations) == 0,
        symbol=symbol_upper,
        market=market,
        side=side,
        quantity=quantity,
        entry_price=round(entry_price, 6),
        mark_price=round(mark_price, 6),
        unrealized_pnl_usdt=round(unrealized_pnl_usdt, 6),
        leverage=leverage,
        existing_open_orders=protective_regular_orders,
        existing_algo_orders=algo_orders,
        step_size=step_size,
        tick_size=tick_size,
        quantity_precision=quantity_precision,
        price_precision=price_precision,
        proposed_stop_price=round(stop_price, 6) if stop_price is not None else None,
        proposed_take_profit_price=round(take_profit_price, 6) if take_profit_price is not None else None,
        trailing_activation_price=round(trailing_activation_price, 6) if trailing_activation_price is not None else None,
        trailing_callback_pct=round(trailing_callback_pct, 4) if trailing_callback_pct is not None else None,
        trailing_quantity=round(trailing_quantity, 8) if trailing_quantity is not None else None,
        cancel_existing_algo_orders=cancel_existing_algo_orders,
        preserve_existing_take_profits=preserve_existing_take_profits,
        violations=violations,
        warnings=warnings,
        actions=actions,
    )


def execute_position_management_plan(settings: Settings, plan: PositionManagementPlan) -> dict[str, Any]:
    if not plan.allowed:
        raise RuntimeError("Position management plan is blocked. Refusing to send exchange writes.")
    if plan.side not in {"BUY", "SELL"}:
        raise RuntimeError("Position management plan has no active position side.")
    trading_control = load_trading_control_state()
    if trading_control.paused and trading_control.updated_by != AUTO_PAUSE_ACTOR:
        reason = trading_control.reason or "manual kill-switch"
        raise RuntimeError(f"Trading is paused by kill-switch: {reason}.")

    results: dict[str, Any] = {
        "cancelled_open_orders": [],
        "cancelled_algo_orders": [],
        "submitted": {},
    }
    exit_side = _opposite_side(plan.side)

    with BinanceClient(settings) as client:
        if plan.cancel_existing_algo_orders:
            for order in _protective_regular_orders(client.open_orders(plan.symbol, plan.market)):
                order_id = order.get("orderId")
                if order_id:
                    results["cancelled_open_orders"].append(client.cancel_order(plan.symbol, int(order_id), plan.market))
            if plan.preserve_existing_take_profits:
                for order in _trailing_algo_orders(client.open_algo_orders(plan.symbol)):
                    algo_id = _algo_id(order)
                    if algo_id is not None:
                        results["cancelled_algo_orders"].append(client.cancel_algo_order(plan.symbol, algo_id))
            else:
                results["cancelled_algo_orders"] = client.cancel_all_algo_orders(plan.symbol)

        if plan.proposed_stop_price is not None:
            results["submitted"]["stop_loss"] = client.new_algo_order(
                plan.symbol,
                exit_side,
                "STOP_MARKET",
                trigger_price=plan.proposed_stop_price,
                quantity=plan.quantity,
                reduce_only=True,
                working_type="MARK_PRICE",
                market=plan.market,
            )
        if plan.proposed_take_profit_price is not None:
            results["submitted"]["take_profit"] = client.new_algo_order(
                plan.symbol,
                exit_side,
                "TAKE_PROFIT_MARKET",
                trigger_price=plan.proposed_take_profit_price,
                quantity=plan.quantity,
                reduce_only=True,
                working_type="MARK_PRICE",
                market=plan.market,
            )
        if plan.trailing_activation_price is not None and plan.trailing_callback_pct is not None:
            trailing_quantity = plan.trailing_quantity if plan.trailing_quantity is not None else plan.quantity
            if trailing_quantity <= 0:
                return results
            results["submitted"]["trailing_stop"] = client.new_algo_order(
                plan.symbol,
                exit_side,
                "TRAILING_STOP_MARKET",
                trigger_price=plan.trailing_activation_price,
                quantity=trailing_quantity,
                callback_rate=plan.trailing_callback_pct,
                reduce_only=True,
                working_type="MARK_PRICE",
                activation_price=plan.trailing_activation_price,
                market=plan.market,
            )

    return results


def _reference_stop_from_payload(plan: PositionManagementPlan, analysis_payload: dict[str, Any]) -> float | None:
    existing_stop = _first_trigger(plan.existing_open_orders, "STOP_MARKET") or _first_trigger(
        plan.existing_algo_orders,
        "STOP_MARKET",
    )
    if existing_stop is not None:
        return existing_stop
    side_key = "long" if plan.side == "BUY" else "short"
    side_plan = ((analysis_payload.get("trade_plan") or {}).get(side_key) or {})
    invalidation = _float(side_plan.get("invalidation"))
    return invalidation if invalidation > 0 else None


def _analysis_family(analysis: dict[str, Any]) -> tuple[str, str]:
    selected = analysis.get("selected_strategy_family") or {}
    if not isinstance(selected, dict):
        return "", ""
    return str(selected.get("family") or ""), str(selected.get("bias") or "")


def _directional_filter_score(latest: dict[str, Any], opposite: str) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    direction = 1 if opposite == "long" else -1
    filter_keys = (
        "supertrend_direction",
        "trend_magic_direction",
        "follow_line_direction",
        "chandelier_direction",
        "qqe_direction",
        "psar_direction",
        "ichimoku_direction",
    )
    filter_vote = sum(1 for key in filter_keys if int(_float(latest.get(key))) == direction)
    if filter_vote >= 4:
        score += 2.0
        reasons.append(f"{filter_vote} trend filters now point {opposite}.")
    elif filter_vote >= 3:
        score += 1.0
        reasons.append(f"{filter_vote} trend filters lean {opposite}.")

    close = _float(latest.get("close"))
    ema_fast = _float(latest.get("ema_fast"))
    ema_slow = _float(latest.get("ema_slow"))
    if close > 0 and ema_fast > 0 and ema_slow > 0:
        if opposite == "short" and close < ema_slow and ema_fast < ema_slow:
            score += 1.5
            reasons.append("Price and fast EMA are below slow EMA.")
        elif opposite == "long" and close > ema_slow and ema_fast > ema_slow:
            score += 1.5
            reasons.append("Price and fast EMA are above slow EMA.")

    macd_hist = _float(latest.get("macd_hist"))
    if (opposite == "short" and macd_hist < 0) or (opposite == "long" and macd_hist > 0):
        score += 1.0
        reasons.append(f"MACD histogram confirms {opposite} momentum.")

    adx_value = _float(latest.get("adx"))
    plus_di = _float(latest.get("plus_di"))
    minus_di = _float(latest.get("minus_di"))
    if adx_value >= 18:
        if opposite == "short" and minus_di > plus_di:
            score += 1.0
            reasons.append("ADX/DI confirms bearish pressure.")
        elif opposite == "long" and plus_di > minus_di:
            score += 1.0
            reasons.append("ADX/DI confirms bullish pressure.")

    if opposite == "short":
        reclaim_keys = ("liquidity_reclaim_short_20", "vwap_reclaim_short_48", "vwap_mid_reclaim_short_48")
    else:
        reclaim_keys = ("liquidity_reclaim_long_20", "vwap_reclaim_long_48", "vwap_mid_reclaim_long_48")
    reclaim_votes = sum(1 for key in reclaim_keys if bool(latest.get(key)))
    if reclaim_votes:
        score += float(reclaim_votes) * 0.5
        reasons.append(f"{reclaim_votes} liquidity/VWAP reclaim signal(s) point {opposite}.")

    return score, reasons


def build_adaptive_exit_plan(
    base_plan: PositionManagementPlan,
    analysis_payload: dict[str, Any],
    *,
    min_profit_r_for_reversal_exit: float = 0.35,
    max_loss_r_for_reversal_exit: float = -0.35,
    min_reversal_score: float = 5.0,
    min_confidence: float = 0.62,
) -> AdaptiveExitPlan:
    """Build an auditable early-exit plan from fresh market state.

    The plan never predicts profit. It only closes risk when the current
    position has enough opposite evidence and either profit to protect or loss
    to cut before the fixed stop.
    """

    warnings: list[str] = []
    reasons: list[str] = []
    analysis = analysis_payload.get("analysis") or {}
    latest = analysis_payload.get("latest") or {}
    position_bias = _position_bias(base_plan.side)
    opposite = _opposite_bias(base_plan.side)
    exit_side = _opposite_side(base_plan.side)
    family, family_bias = _analysis_family(analysis if isinstance(analysis, dict) else {})
    analysis_bias = str((analysis or {}).get("bias") or "neutral")
    recommended_action = str((analysis or {}).get("recommended_action") or "HOLD").upper()
    reference_stop = _reference_stop_from_payload(base_plan, analysis_payload)
    risk_distance = abs(base_plan.entry_price - float(reference_stop or base_plan.entry_price))

    if not base_plan.allowed or base_plan.side not in {"BUY", "SELL"} or base_plan.quantity <= 0:
        warnings.append("No active manageable position is available for adaptive exit.")
        risk_distance = max(risk_distance, 0.0)
        return AdaptiveExitPlan(
            allowed=False,
            symbol=base_plan.symbol,
            market=base_plan.market,
            side=base_plan.side,
            exit_side=exit_side,
            quantity=base_plan.quantity,
            action="hold",
            reason_code="no-manageable-position",
            reasons=reasons,
            warnings=warnings,
            unrealized_r=0.0,
            reversal_score=0.0,
            confidence=0.0,
            risk_distance=round(risk_distance, 8),
            reference_stop_price=reference_stop,
            entry_price=base_plan.entry_price,
            mark_price=base_plan.mark_price,
            analysis_bias=analysis_bias,
            recommended_action=recommended_action,
            selected_family=family,
            selected_family_bias=family_bias,
        )

    if risk_distance <= 0:
        warnings.append("Adaptive exit needs a stop-loss or trade-plan invalidation to define 1R.")
        return AdaptiveExitPlan(
            allowed=False,
            symbol=base_plan.symbol,
            market=base_plan.market,
            side=base_plan.side,
            exit_side=exit_side,
            quantity=base_plan.quantity,
            action="hold",
            reason_code="missing-risk-anchor",
            reasons=reasons,
            warnings=warnings,
            unrealized_r=0.0,
            reversal_score=0.0,
            confidence=0.0,
            risk_distance=0.0,
            reference_stop_price=reference_stop,
            entry_price=base_plan.entry_price,
            mark_price=base_plan.mark_price,
            analysis_bias=analysis_bias,
            recommended_action=recommended_action,
            selected_family=family,
            selected_family_bias=family_bias,
        )

    if base_plan.side == "BUY":
        unrealized_r = (base_plan.mark_price - base_plan.entry_price) / risk_distance
        opposite_action = "SELL"
    else:
        unrealized_r = (base_plan.entry_price - base_plan.mark_price) / risk_distance
        opposite_action = "BUY"

    reversal_score = 0.0
    if analysis_bias == opposite:
        reversal_score += 2.0
        reasons.append(f"Score-model bias flipped {opposite}.")
    elif analysis_bias == position_bias:
        reasons.append(f"Score-model still supports {position_bias}.")
    if recommended_action == opposite_action:
        reversal_score += 2.0
        reasons.append(f"Recommended action is {opposite_action}.")
    if family_bias == opposite:
        reversal_score += 1.5
        reasons.append(f"Selected strategy family flipped {opposite}.")
    multi_timeframe_bias = str(latest.get("multi_timeframe_bias") or "neutral")
    if multi_timeframe_bias == opposite:
        reversal_score += 1.0
        reasons.append(f"Multi-timeframe bias flipped {opposite}.")

    filter_score, filter_reasons = _directional_filter_score(latest if isinstance(latest, dict) else {}, opposite)
    reversal_score += filter_score
    reasons.extend(filter_reasons)

    confidence = min(1.0, reversal_score / max(min_reversal_score + 2.0, 1.0))
    if reversal_score < min_reversal_score or confidence < min_confidence:
        return AdaptiveExitPlan(
            allowed=False,
            symbol=base_plan.symbol,
            market=base_plan.market,
            side=base_plan.side,
            exit_side=exit_side,
            quantity=base_plan.quantity,
            action="hold",
            reason_code="no-confirmed-reversal",
            reasons=reasons,
            warnings=warnings,
            unrealized_r=round(unrealized_r, 4),
            reversal_score=round(reversal_score, 4),
            confidence=round(confidence, 4),
            risk_distance=round(risk_distance, 8),
            reference_stop_price=reference_stop,
            entry_price=base_plan.entry_price,
            mark_price=base_plan.mark_price,
            analysis_bias=analysis_bias,
            recommended_action=recommended_action,
            selected_family=family,
            selected_family_bias=family_bias,
        )

    if unrealized_r >= min_profit_r_for_reversal_exit:
        reason_code = "profit-protection-reversal"
        reasons.append(f"Unrealized profit is {unrealized_r:.2f}R, enough to protect on reversal.")
    elif unrealized_r <= max_loss_r_for_reversal_exit:
        reason_code = "loss-cut-reversal"
        reasons.append(f"Unrealized loss is {unrealized_r:.2f}R with confirmed reversal.")
    else:
        return AdaptiveExitPlan(
            allowed=False,
            symbol=base_plan.symbol,
            market=base_plan.market,
            side=base_plan.side,
            exit_side=exit_side,
            quantity=base_plan.quantity,
            action="hold",
            reason_code="inside-adaptive-exit-band",
            reasons=reasons,
            warnings=warnings,
            unrealized_r=round(unrealized_r, 4),
            reversal_score=round(reversal_score, 4),
            confidence=round(confidence, 4),
            risk_distance=round(risk_distance, 8),
            reference_stop_price=reference_stop,
            entry_price=base_plan.entry_price,
            mark_price=base_plan.mark_price,
            analysis_bias=analysis_bias,
            recommended_action=recommended_action,
            selected_family=family,
            selected_family_bias=family_bias,
        )

    return AdaptiveExitPlan(
        allowed=True,
        symbol=base_plan.symbol,
        market=base_plan.market,
        side=base_plan.side,
        exit_side=exit_side,
        quantity=base_plan.quantity,
        action="close_position",
        reason_code=reason_code,
        reasons=reasons,
        warnings=warnings,
        unrealized_r=round(unrealized_r, 4),
        reversal_score=round(reversal_score, 4),
        confidence=round(confidence, 4),
        risk_distance=round(risk_distance, 8),
        reference_stop_price=reference_stop,
        entry_price=base_plan.entry_price,
        mark_price=base_plan.mark_price,
        analysis_bias=analysis_bias,
        recommended_action=recommended_action,
        selected_family=family,
        selected_family_bias=family_bias,
    )


def execute_adaptive_exit_plan(settings: Settings, plan: AdaptiveExitPlan) -> dict[str, Any]:
    if not plan.allowed or plan.action != "close_position":
        raise RuntimeError("Adaptive exit plan is not executable.")
    if plan.side not in {"BUY", "SELL"} or plan.quantity <= 0:
        raise RuntimeError("Adaptive exit plan has no active position side or quantity.")
    trading_control = load_trading_control_state()
    if trading_control.paused and trading_control.updated_by != AUTO_PAUSE_ACTOR:
        reason = trading_control.reason or "manual kill-switch"
        raise RuntimeError(f"Trading is paused by kill-switch: {reason}.")

    results: dict[str, Any] = {
        "submitted": {},
        "cancelled_open_orders": [],
        "cancelled_algo_orders": [],
    }
    with BinanceClient(settings) as client:
        results["submitted"]["market_close"] = client.new_order(
            plan.symbol,
            plan.exit_side,
            "MARKET",
            quantity=plan.quantity,
            reduce_only=True,
            market=plan.market,
        )
        for order in _protective_regular_orders(client.open_orders(plan.symbol, plan.market)):
            order_id = order.get("orderId")
            if order_id:
                results["cancelled_open_orders"].append(client.cancel_order(plan.symbol, int(order_id), plan.market))
        if plan.market == "futures":
            results["cancelled_algo_orders"] = client.cancel_all_algo_orders(plan.symbol)
    return results
