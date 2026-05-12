from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from .asset_routing import resolve_symbol_route
from .binance_api import BinanceAPIError, BinanceClient
from .challenge import challenge_scope_key, challenge_summary_dict, record_balance_snapshot
from .config import Settings
from .convergence import build_cohort_id
from .decision_trace import trace_step
from .historical_signal_risk import evaluate_historical_signal_risk
from .market_context import estimate_slippage_from_order_book
from .order_journal import LiveOrderRecord, append_live_order, load_trade_state, save_trade_state
from .professional_entry_gate import ProfessionalGatePolicy, evaluate_professional_entry_gate
from .risk_guard import check_order_allowed
from .route_risk_control import route_quarantine_status
from .side_risk_policy import evaluate_route_side_risk
from .signal_scoring import build_signal_scores
from .strategy import StrategyConfig
from .strategy_optimizer import (
    evaluate_market_bot_live_gate,
    evaluate_optimizer_live_gate,
    evaluate_risk_combo_live_gate,
)
from .symbol_sizing import build_symbol_sizing_plan
from .trading_control import load_trading_control_state


@dataclass(frozen=True, slots=True)
class SymbolRules:
    min_qty: float
    step_size: float
    tick_size: float
    min_notional: float
    quantity_precision: int
    price_precision: int


@dataclass(frozen=True, slots=True)
class LiveExecutionPlan:
    allowed: bool
    symbol: str
    market: str
    side: str
    quantity: float
    price: float
    leverage: int
    wallet_balance_usdt: float
    available_balance_usdt: float
    equity_usdt: float
    margin_notional_usdt: float
    gross_notional_usdt: float
    min_notional_usdt: float
    strategy_profile: str
    analysis_score: int
    analysis_bias: str
    analysis_convergence: float
    adx_value: float
    stop_price: float
    stop_distance: float
    take_profit_price: float
    take_profit_prices: list[float]
    take_profit_quantities: list[float]
    take_profit_weights: list[float]
    take_profit_runner_quantity: float
    required_structured_exit_quantity: float
    planned_account_risk_usdt: float
    planned_account_risk_pct: float
    liquidation_buffer_pct: float
    min_liquidation_buffer_pct: float
    trailing_stop_enabled: bool
    trailing_callback_pct: float
    fee_bps: float
    slippage_bps: float
    spread_bps: float
    order_book_fill_ratio: float
    market_context: dict[str, Any]
    sizing: dict[str, Any]
    execution_mode: str
    challenge: dict[str, Any]
    decision_trace: list[dict[str, Any]]
    professional_entry_gate: dict[str, Any]
    violations: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_futures_balance(payload: Any) -> tuple[float, float]:
    total = 0.0
    available = 0.0
    for item in payload:
        if item.get("asset") == "USDT":
            total = float(item.get("balance", 0.0))
            available = float(item.get("availableBalance", total))
            break
    return total, available


def _nonzero_position(raw_positions: Any, symbol: str) -> bool:
    for item in raw_positions:
        if item.get("symbol") == symbol.upper():
            return abs(float(item.get("positionAmt", 0.0))) > 0.0
    return False


def _round_down_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step) * step


def _round_up_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.ceil(value / step) * step


def _round_price_to_tick(
    value: float,
    *,
    tick_size: float,
    precision: int,
    mode: str = "nearest",
) -> float:
    if value <= 0:
        return value
    if tick_size <= 0:
        return round(value, precision)
    epsilon = 1e-12
    if mode == "up":
        rounded = _round_up_step(value - epsilon, tick_size)
    elif mode == "down":
        rounded = _round_down_step(value + epsilon, tick_size)
    else:
        rounded = round(value / tick_size) * tick_size
    return round(rounded, precision)


def _risk_sized_quantity(
    *,
    price: float,
    stop_price: float,
    equity_usdt: float,
    available_balance_usdt: float,
    leverage: int,
    strategy: StrategyConfig,
    max_account_risk_pct: float | None = None,
    max_notional_pct: float | None = None,
    manual_margin_cap_usdt: float | None,
) -> tuple[float, float, float]:
    stop_distance = abs(price - stop_price)
    if price <= 0 or stop_distance <= 0 or leverage <= 0:
        return 0.0, 0.0, stop_distance

    effective_risk_pct = strategy.risk.max_account_risk_pct if max_account_risk_pct is None else max_account_risk_pct
    effective_notional_pct = strategy.risk.max_notional_pct if max_notional_pct is None else max_notional_pct
    max_risk_usdt = equity_usdt * effective_risk_pct
    quantity_from_risk = max_risk_usdt / stop_distance

    configured_margin_cap = strategy.execution.margin_notional_usdt
    maturity_mode = strategy.risk.max_leverage >= 125
    margin_cap = available_balance_usdt if maturity_mode else min(available_balance_usdt, equity_usdt * effective_notional_pct)
    if configured_margin_cap is not None:
        margin_cap = min(margin_cap, configured_margin_cap)
    if manual_margin_cap_usdt is not None:
        margin_cap = min(margin_cap, manual_margin_cap_usdt)

    gross_notional_cap = max(margin_cap, 0.0) * leverage
    quantity_from_notional = gross_notional_cap / price if price else 0.0
    quantity = min(quantity_from_risk, quantity_from_notional)
    return quantity, margin_cap, stop_distance


def _liquidation_buffer_pct(*, side: str, price: float, stop_price: float, leverage: int) -> float:
    if price <= 0 or leverage <= 0:
        return 0.0
    liquidation_distance = price / leverage
    liquidation_price = price - liquidation_distance if side == "BUY" else price + liquidation_distance
    stop_distance_from_liq = abs(stop_price - liquidation_price)
    return stop_distance_from_liq / price


def _max_leverage_for_liquidation_buffer(
    *,
    price: float,
    stop_price: float,
    min_buffer_pct: float,
    configured_leverage: int,
) -> int:
    if price <= 0 or stop_price <= 0 or min_buffer_pct <= 0 or configured_leverage <= 1:
        return configured_leverage
    stop_distance_pct = abs(price - stop_price) / price
    denominator = min_buffer_pct + stop_distance_pct
    if denominator <= 0:
        return configured_leverage
    buffer_safe_leverage = max(1, int(math.floor(1.0 / denominator)))
    return min(configured_leverage, buffer_safe_leverage)


def _soften_exploration_violations(
    violations: list[str],
    warnings: list[str],
    *,
    execution_mode: str,
    research_promoted: bool = False,
) -> list[str]:
    if execution_mode != "testnet_exploration":
        return violations
    soft_markers = (
        "optimizer",
        "quarantined",
        "route/side",
        "historical",
        "profit factor",
        "net pnl",
        "sample",
        "convergence",
        "analysis score",
        "adx",
        "trend threshold",
    )
    research_soft_markers = (
        "volume z-score",
        "lacks accumulation confirmation",
        "lacks distribution confirmation",
        "multi-timeframe trend is not strong enough",
    )
    hard: list[str] = []
    for item in violations:
        lowered = item.lower()
        if any(marker in lowered for marker in soft_markers) or (
            research_promoted and any(marker in lowered for marker in research_soft_markers)
        ):
            warnings.append(f"Testnet exploration override: {item}")
        else:
            hard.append(item)
    return hard


def fetch_symbol_rules(client: BinanceClient, symbol: str, market: str) -> SymbolRules:
    info = client.exchange_info(symbol, market)
    row = next((item for item in (info.get("symbols") or []) if item.get("symbol") == symbol.upper()), {})
    filters = {item["filterType"]: item for item in row.get("filters", [])}
    market_lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE") or {}
    min_notional = filters.get("MIN_NOTIONAL") or {}
    price_filter = filters.get("PRICE_FILTER") or {}
    return SymbolRules(
        min_qty=float(market_lot.get("minQty", 0.0)),
        step_size=float(market_lot.get("stepSize", 0.0)),
        tick_size=float(price_filter.get("tickSize", 0.0)),
        min_notional=float(min_notional.get("notional", 0.0)),
        quantity_precision=int(row.get("quantityPrecision", 8)),
        price_precision=int(row.get("pricePrecision", 8)),
    )


def _take_profit_prices(side_plan: dict[str, Any]) -> list[float]:
    levels = side_plan.get("take_profit_levels")
    if isinstance(levels, list):
        values = [float(item) for item in levels if float(item or 0.0) > 0.0]
    else:
        values = [
            float(side_plan[key])
            for key in ("take_profit_1", "take_profit_2", "take_profit_3")
            if key in side_plan and float(side_plan.get(key) or 0.0) > 0.0
        ]
    seen: list[float] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _take_profit_weights(
    *,
    parts: int,
    route_id: str,
    confidence: float,
    news_risk: dict[str, Any] | None,
    strategy_family: str = "",
) -> list[float]:
    if parts <= 0:
        return []
    if parts == 1:
        return [1.0]
    risk_level = str((news_risk or {}).get("risk_level") or "normal")
    high_beta = route_id in {"doge-meme-high-beta", "meme-high-beta", "defensive-unknown"}
    if parts == 2:
        if strategy_family == "mean_reversion":
            return [0.6, 0.4] if risk_level == "high" or high_beta else [0.55, 0.45]
        if risk_level == "high" or high_beta:
            return [0.4, 0.6]
        if confidence >= 0.86:
            return [0.25, 0.75]
        return [0.3, 0.7]
    if strategy_family == "mean_reversion":
        base = [0.60, 0.30, 0.10] if risk_level == "high" or high_beta else [0.55, 0.30, 0.15]
    elif risk_level == "high" or high_beta:
        base = [0.45, 0.35, 0.20]
    elif confidence >= 0.86:
        base = [0.25, 0.35, 0.40]
    else:
        base = [0.30, 0.35, 0.35]
    if parts <= len(base):
        selected = base[:parts]
    else:
        selected = [*base, *([0.0] * (parts - len(base)))]
        selected[-1] = max(0.0, 1.0 - sum(selected[:-1]))
    total = sum(selected)
    return [item / total for item in selected] if total > 0 else [1.0 / parts for _ in range(parts)]


def _trailing_runner_weight(
    *,
    trailing_stop_enabled: bool,
    parts: int,
    route_id: str,
    confidence: float,
    news_risk: dict[str, Any] | None,
    strategy_family: str = "",
) -> float:
    if not trailing_stop_enabled or parts <= 0:
        return 0.0
    risk_level = str((news_risk or {}).get("risk_level") or "normal")
    high_beta = route_id in {"doge-meme-high-beta", "meme-high-beta", "defensive-unknown"}
    if strategy_family == "mean_reversion":
        return 0.05
    if risk_level == "high" or high_beta:
        return 0.10
    if confidence >= 0.86:
        return 0.20
    return 0.15


def _reserve_runner_from_weights(weights: list[float], runner_weight: float) -> list[float]:
    if not weights:
        return []
    runner_weight = max(0.0, min(0.5, runner_weight))
    return [weight * (1.0 - runner_weight) for weight in weights]


def _split_reduce_only_quantities(
    quantity: float,
    step_size: float,
    precision: int,
    weights: list[float],
) -> list[float]:
    parts = len(weights)
    if quantity <= 0 or parts <= 0:
        return []
    if parts == 1:
        target_quantity = quantity * min(max(weights[0], 0.0), 1.0)
        return [round(_round_down_step(target_quantity, step_size), precision)]
    normalized_total = sum(weight for weight in weights if weight > 0.0)
    if normalized_total <= 0:
        return [round(quantity, precision)]
    if normalized_total > 1.0:
        normalized = [max(0.0, weight) / normalized_total for weight in weights]
        target_quantity = quantity
    else:
        normalized = [max(0.0, weight) for weight in weights]
        target_quantity = _round_down_step(quantity * normalized_total, step_size)
    quantities = [
        round(_round_down_step(quantity * weight, step_size), precision)
        for weight in normalized[:-1]
    ]
    remainder = round(_round_down_step(target_quantity - sum(quantities), step_size), precision)
    if remainder > 0:
        quantities.append(remainder)
    return [item for item in quantities if item > 0]


def _minimum_structured_exit_quantity(
    *,
    step_size: float,
    quantity_precision: int,
    take_profit_parts: int,
    trailing_stop_enabled: bool,
) -> float:
    if step_size <= 0 or take_profit_parts <= 0:
        return 0.0
    slices = take_profit_parts + (1 if trailing_stop_enabled else 0)
    return round(step_size * slices, quantity_precision)


def _fit_exit_structure_to_quantity(
    *,
    quantity: float,
    step_size: float,
    quantity_precision: int,
    take_profit_prices: list[float],
    trailing_stop_enabled: bool,
) -> tuple[list[float], bool, float]:
    selected_prices = list(take_profit_prices)
    runner_enabled = bool(trailing_stop_enabled)
    if quantity <= 0.0 or step_size <= 0.0:
        return selected_prices, runner_enabled, _minimum_structured_exit_quantity(
            step_size=step_size,
            quantity_precision=quantity_precision,
            take_profit_parts=len(selected_prices),
            trailing_stop_enabled=runner_enabled,
        )
    while selected_prices:
        required = _minimum_structured_exit_quantity(
            step_size=step_size,
            quantity_precision=quantity_precision,
            take_profit_parts=len(selected_prices),
            trailing_stop_enabled=runner_enabled,
        )
        if quantity + 1e-12 >= required:
            return selected_prices, runner_enabled, required
        if runner_enabled:
            runner_enabled = False
            continue
        if len(selected_prices) > 1:
            selected_prices = selected_prices[: len(selected_prices) - 1]
            runner_enabled = bool(trailing_stop_enabled)
            continue
        return selected_prices, False, required
    return selected_prices, False, 0.0


def build_live_execution_plan(
    settings: Settings,
    strategy: StrategyConfig,
    analysis_payload: dict[str, Any],
    *,
    side_override: str | None = None,
    margin_notional_usdt: float | None = None,
    require_optimizer_gate: bool = True,
    execution_mode: str = "live",
    news_risk: dict[str, Any] | None = None,
    volume_rank: int | None = None,
) -> LiveExecutionPlan:
    symbol = analysis_payload["symbol"].upper()
    market = analysis_payload["market"]
    analysis = analysis_payload["analysis"]
    latest = dict(analysis_payload["latest"])
    market_context_payload = analysis_payload.get("market_context") if isinstance(analysis_payload.get("market_context"), dict) else {}
    if "multi_timeframe_structure" not in latest and isinstance(market_context_payload.get("multi_timeframe_structure"), dict):
        latest["multi_timeframe_structure"] = market_context_payload["multi_timeframe_structure"]
    trade_plan = analysis_payload["trade_plan"]
    bias = analysis["bias"]
    if side_override:
        side = side_override.upper()
    else:
        side = str(analysis.get("recommended_action", "HOLD"))
        if side == "HOLD":
            if bias == "long-bias":
                side = "BUY"
            elif bias == "short-bias" and market == "futures":
                side = "SELL"

    violations: list[str] = []
    warnings: list[str] = []
    decision_trace: list[dict[str, Any]] = []

    trading_control = load_trading_control_state()
    if trading_control.paused:
        reason = trading_control.reason or "manual kill-switch"
        violations.append(f"Trading is paused by kill-switch: {reason}.")

    route = resolve_symbol_route(symbol)
    market_bot_gate = evaluate_market_bot_live_gate(symbol=symbol, route_id=route.route_id)
    analysis_interval = str(analysis_payload.get("interval") or strategy.defaults.interval)
    risk_combo_gate = evaluate_risk_combo_live_gate(
        symbol=symbol,
        route_id=route.route_id,
        side=side,
        interval=analysis_interval,
        execution_mode=execution_mode,
    )
    route_mode_reason = ""
    market_bot_promoted = bool(market_bot_gate.get("allowed"))
    risk_combo_promoted = bool(risk_combo_gate.get("allowed"))
    research_promoted = market_bot_promoted or risk_combo_promoted
    if execution_mode == "testnet_exploration" and route.simulation_mode in {
        "paper",
        "paper_research_only",
    } and not research_promoted:
        route_mode_reason = (
            f"Route {route.route_id} is {route.simulation_mode}; "
            "unknown/paper-only routes need accepted market-bot evidence or robust risk-combo evidence before testnet."
        )
        violations.append(route_mode_reason)
    elif execution_mode == "testnet_exploration" and route.simulation_mode in {
        "paper",
        "paper_research_only",
    }:
        if market_bot_promoted:
            warnings.append(
                f"Market-bot gate promotes {symbol}/{route.route_id}; paper route flag does not block testnet readiness."
            )
        if risk_combo_promoted:
            if risk_combo_gate.get("exploration_allowed"):
                warnings.append(
                    f"Risk-combo matrix allows exploratory testnet harvesting for {symbol}/{side}/{analysis_interval}; "
                    "this is not a mainnet promotion."
                )
            else:
                warnings.append(
                    f"Risk-combo matrix promotes {symbol}/{side}/{analysis_interval}; paper route flag does not block testnet readiness."
                )
    route_quarantine = route_quarantine_status(route.route_id)
    if route_quarantine["quarantined"]:
        reasons = "; ".join(route_quarantine["reasons"]) or "route performance quarantine"
        target = warnings if execution_mode == "testnet_exploration" else violations
        target.append(
            f"Route {route_quarantine['route_id']} is quarantined pending manual review: {reasons}."
        )

    optimizer_gate = evaluate_optimizer_live_gate()
    optimizer_gate = {**optimizer_gate, "required": require_optimizer_gate}
    if require_optimizer_gate and not optimizer_gate["allowed"]:
        if execution_mode == "testnet_exploration" and research_promoted:
            warnings.append(
                "Research promotion gate is accepted; stale or legacy optimizer rejection is advisory for testnet readiness."
            )
            warnings.extend(str(item) for item in optimizer_gate["reasons"])
        else:
            target = warnings if execution_mode == "testnet_exploration" else violations
            target.extend(str(item) for item in optimizer_gate["reasons"])

    side_risk_gate = evaluate_route_side_risk(route_id=route.route_id, side=side)
    if not side_risk_gate.allowed:
        target = warnings if execution_mode == "testnet_exploration" else violations
        target.extend(side_risk_gate.reasons)
    historical_signal_gate = evaluate_historical_signal_risk(
        route_id=route.route_id,
        symbol=symbol,
        side=side,
        score=int(analysis["score"]),
        convergence=float(analysis["convergence"]),
    )
    if not historical_signal_gate.allowed:
        target = warnings if execution_mode == "testnet_exploration" else violations
        target.extend(historical_signal_gate.reasons)

    decision_trace.extend(
        [
            trace_step(
                "trading_control",
                allowed=not trading_control.paused,
                reasons=[f"Trading is paused by kill-switch: {trading_control.reason}."] if trading_control.paused else [],
                data=trading_control.to_dict(),
            ),
            trace_step(
                "route_mode",
                allowed=not bool(route_mode_reason),
                reasons=[route_mode_reason] if route_mode_reason else [],
                data=route.to_dict(),
            ),
            trace_step(
                "market_bot_gate",
                allowed=market_bot_promoted,
                reasons=list(market_bot_gate.get("reasons") or []),
                data=market_bot_gate,
            ),
            trace_step(
                "risk_combo_gate",
                allowed=risk_combo_promoted,
                reasons=list(risk_combo_gate.get("reasons") or []),
                data=risk_combo_gate,
            ),
            trace_step(
                "optimizer_gate",
                allowed=not require_optimizer_gate
                or bool(optimizer_gate.get("allowed"))
                or bool(execution_mode == "testnet_exploration" and research_promoted),
                reasons=list(optimizer_gate.get("reasons") or []),
                data=optimizer_gate,
            ),
            trace_step(
                "route_quarantine",
                allowed=not bool(route_quarantine.get("quarantined")),
                reasons=list(route_quarantine.get("reasons") or []),
                data=route_quarantine,
            ),
            trace_step(
                "route_side_risk",
                allowed=side_risk_gate.allowed,
                reasons=side_risk_gate.reasons,
                data=side_risk_gate.to_dict(),
            ),
            trace_step(
                "historical_signal_risk",
                allowed=historical_signal_gate.allowed,
                reasons=historical_signal_gate.reasons,
                data=historical_signal_gate.to_dict(),
            ),
        ]
    )

    with BinanceClient(settings) as client:
        price = client.ticker_price(symbol, market)
        balance_payload = client.balance(market)
        challenge_scope = challenge_scope_key(strategy.profile, symbol, market)
        balance_snapshot, challenge_state = record_balance_snapshot(
            balance_payload,
            market,
            note=f"live-plan:{symbol}",
            scope=challenge_scope,
        )
        total_balance, available_balance = _extract_futures_balance(balance_payload)
        try:
            open_orders = client.open_orders(symbol, market) if market == "futures" else []
            open_algo = client.open_algo_orders(symbol) if market == "futures" else []
            raw_positions = client.positions(symbol if market == "futures" else None) if market == "futures" else []
        except BinanceAPIError as exc:
            if "Invalid symbol" not in str(exc):
                raise
            violations.append(
                f"Futures account channel rejected {symbol}: {exc}. "
                "Treat this symbol as research-only until the exchange lane supports it."
            )
            open_orders = []
            open_algo = []
            raw_positions = []
        rules = fetch_symbol_rules(client, symbol, market)
        order_book = client.order_book(symbol, market, limit=20)

    if side == "HOLD":
        blockers = tuple(str(item) for item in analysis.get("entry_blockers", []))
        violations.extend(blockers or ("Strategy policy does not allow a new live entry.",))
        if not blockers and analysis.get("decision_notes"):
            warnings.extend(str(item) for item in analysis.get("decision_notes", []))
    decision_trace.append(
        trace_step(
            "analysis_policy",
            allowed=side != "HOLD",
            reasons=list(analysis.get("entry_blockers") or []) if side == "HOLD" else [],
            warnings=list(analysis.get("decision_notes") or []),
            data={
                "score": analysis.get("score"),
                "convergence": analysis.get("convergence"),
                "recommended_action": analysis.get("recommended_action"),
                "selected_strategy_family": analysis.get("selected_strategy_family"),
            },
        )
    )

    if open_orders:
        violations.append(f"{len(open_orders)} open regular order(s) already exist for {symbol}.")
    if open_algo:
        warnings.append(
            f"{len(open_algo)} orphaned algo order(s) exist for {symbol}; "
            "they will be cancelled before entry."
        )
    if raw_positions and _nonzero_position(raw_positions, symbol):
        violations.append(f"A non-flat futures position already exists for {symbol}.")

    if side == "BUY":
        side_plan = trade_plan["long"]
        price_round_mode = "down"
    else:
        side_plan = trade_plan["short"]
        price_round_mode = "up"
    stop_price = float(side_plan["invalidation"])
    take_profit_prices = _take_profit_prices(side_plan)
    stop_price = _round_price_to_tick(
        stop_price,
        tick_size=rules.tick_size,
        precision=rules.price_precision,
        mode=price_round_mode,
    )
    take_profit_prices = [
        _round_price_to_tick(
            item,
            tick_size=rules.tick_size,
            precision=rules.price_precision,
            mode=price_round_mode,
        )
        for item in take_profit_prices
    ]
    take_profit_price = take_profit_prices[0] if take_profit_prices else 0.0
    selected_family = analysis.get("selected_strategy_family") or {}
    strategy_family = str(analysis.get("strategy_family") or selected_family.get("family") or "")

    equity_reference = max(balance_snapshot.equity_usdt, total_balance)
    available_reference = max(balance_snapshot.available_balance_usdt, available_balance)
    signal_scores = build_signal_scores(
        route=route,
        latest=latest,
        analysis=analysis,
        trade_plan=trade_plan,
        news_risk=news_risk,
        side=side,
        fee_bps=strategy.execution.fee_bps,
        slippage_bps=strategy.execution.slippage_bps,
    )
    sizing = build_symbol_sizing_plan(
        symbol=symbol,
        route=route,
        strategy=strategy,
        latest=latest,
        analysis=analysis,
        equity_usdt=equity_reference,
        available_balance_usdt=available_reference,
        min_notional_usdt=rules.min_notional,
        news_risk=news_risk,
        volume_rank=volume_rank,
        manual_margin_cap_usdt=margin_notional_usdt,
        signal_scores=signal_scores,
        route_side_risk=side_risk_gate.to_dict(),
        market_bot_promoted=research_promoted,
    )
    leverage = sizing.recommended_leverage
    effective_margin_cap = sizing.recommended_margin_usdt
    min_liquidation_buffer_pct = max(0.01, strategy.risk.max_account_risk_pct * 1.5)
    buffer_safe_leverage = _max_leverage_for_liquidation_buffer(
        price=price,
        stop_price=stop_price,
        min_buffer_pct=min_liquidation_buffer_pct,
        configured_leverage=leverage,
    )
    if buffer_safe_leverage < leverage:
        warnings.append(
            f"Leverage reduced from {leverage}x to {buffer_safe_leverage}x to keep stop-loss outside the liquidation buffer."
        )
        leverage = buffer_safe_leverage

    raw_quantity, margin_cap, stop_distance = _risk_sized_quantity(
        price=price,
        stop_price=stop_price,
        equity_usdt=equity_reference,
        available_balance_usdt=available_reference,
        leverage=leverage,
        strategy=strategy,
        max_account_risk_pct=sizing.max_account_risk_pct,
        max_notional_pct=sizing.recommended_margin_pct,
        manual_margin_cap_usdt=effective_margin_cap,
    )
    quantity = _round_down_step(raw_quantity, rules.step_size)
    quantity = round(quantity, rules.quantity_precision)
    if quantity <= 0.0 and raw_quantity > 0.0 and rules.min_qty > 0.0:
        quantity = round(rules.min_qty, rules.quantity_precision)
    original_take_profit_count = len(take_profit_prices)
    original_trailing_stop_enabled = bool(strategy.risk.trailing_stop_enabled)
    take_profit_prices, effective_trailing_stop_enabled, required_structured_exit_quantity = _fit_exit_structure_to_quantity(
        quantity=quantity,
        step_size=rules.step_size,
        quantity_precision=rules.quantity_precision,
        take_profit_prices=take_profit_prices,
        trailing_stop_enabled=original_trailing_stop_enabled,
    )
    if len(take_profit_prices) != original_take_profit_count or effective_trailing_stop_enabled != original_trailing_stop_enabled:
        warnings.append(
            "Exit structure adapted to exchange quantity step: "
            f"{original_take_profit_count} TP(s) + runner={original_trailing_stop_enabled} -> "
            f"{len(take_profit_prices)} TP(s) + runner={effective_trailing_stop_enabled}."
        )
    take_profit_price = take_profit_prices[0] if take_profit_prices else 0.0
    take_profit_weights = _take_profit_weights(
        parts=len(take_profit_prices),
        route_id=route.route_id,
        confidence=sizing.confidence,
        news_risk=news_risk,
        strategy_family=strategy_family,
    )
    runner_weight = _trailing_runner_weight(
        trailing_stop_enabled=effective_trailing_stop_enabled,
        parts=len(take_profit_prices),
        route_id=route.route_id,
        confidence=sizing.confidence,
        news_risk=news_risk,
        strategy_family=strategy_family,
    )
    take_profit_weights = _reserve_runner_from_weights(take_profit_weights, runner_weight)
    take_profit_quantities = _split_reduce_only_quantities(
        quantity,
        rules.step_size,
        rules.quantity_precision,
        take_profit_weights,
    )
    runner_quantity = round(max(0.0, quantity - sum(take_profit_quantities)), rules.quantity_precision)
    actual_notional = quantity * price
    requested_margin = actual_notional / leverage if leverage else 0.0
    planned_account_risk_usdt = quantity * stop_distance
    planned_account_risk_pct = planned_account_risk_usdt / equity_reference if equity_reference else 0.0
    liquidation_buffer_pct = _liquidation_buffer_pct(
        side=side,
        price=price,
        stop_price=stop_price,
        leverage=leverage,
    )

    if quantity < rules.min_qty:
        violations.append(
            f"Quantity {quantity} is below exchange minimum {rules.min_qty} for {symbol}."
        )
    if actual_notional < rules.min_notional:
        violations.append(
            f"Order notional {actual_notional:.4f} USDT is below exchange minimum {rules.min_notional:.4f}."
        )
    if requested_margin > available_reference:
        violations.append(
            f"Requested margin {requested_margin:.4f} exceeds available balance {available_reference:.4f}."
        )
    if margin_notional_usdt is not None and requested_margin > margin_notional_usdt:
        violations.append(
            f"Requested margin {requested_margin:.4f} exceeds the manual cap {margin_notional_usdt:.4f}."
        )
    if (
        effective_trailing_stop_enabled
        and len(take_profit_prices) >= 3
        and required_structured_exit_quantity > 0
    ):
        if len(take_profit_quantities) < 3 or runner_quantity < rules.step_size:
            violations.append(
                "Quantity cannot support three staged take-profits plus a trailing runner at exchange step size "
                f"{rules.step_size:g}; required at least {required_structured_exit_quantity:g}, planned {quantity:g}."
            )

    challenge_summary = challenge_summary_dict(challenge_state)
    execution_costs = estimate_slippage_from_order_book(
        order_book,
        side=side,
        target_notional_usdt=actual_notional,
        fallback_slippage_bps=strategy.execution.slippage_bps,
    )
    spread_bps = float(execution_costs.get("spread_bps", 0.0))
    dynamic_slippage_bps = float(execution_costs.get("estimated_slippage_bps", strategy.execution.slippage_bps))
    fill_ratio = float(execution_costs.get("fill_ratio", 0.0))
    if spread_bps >= 12.0:
        warnings.append(f"Spread is elevated at {spread_bps:.2f} bps.")
    if 0.0 < fill_ratio < 0.999:
        warnings.append(
            f"Visible book only covers {fill_ratio:.1%} of the planned notional; slippage fallback remains conservative."
        )
    if strategy.challenge.enabled and challenge_state.enabled:
        if challenge_state.status == "drawdown-stop" and strategy.challenge.pause_on_drawdown_breach:
            violations.append(
                f"Challenge drawdown stop is active at {challenge_state.latest_balance_usdt:.4f} USDT "
                f"(floor {challenge_state.stop_balance_usdt:.4f})."
            )
        elif challenge_state.status == "target-hit" and strategy.challenge.pause_on_target:
            violations.append(
                f"Challenge target already hit at {challenge_state.latest_balance_usdt:.4f} USDT "
                f"(target {challenge_state.target_balance_usdt:.4f})."
            )
        elif challenge_state.progress_pct >= 80.0:
            warnings.append(
                f"Challenge progress is already {challenge_state.progress_pct:.1f}% toward target."
            )

    state = load_trade_state()
    risk = check_order_allowed(
        side=side,
        margin_notional_usdt=requested_margin,
        leverage=leverage,
        account_balance_usdt=equity_reference,
        account_risk_pct=planned_account_risk_pct,
        analysis_convergence=float(analysis["convergence"]),
        analysis_score=int(analysis["score"]),
        adx_value=float(latest["adx"]) if "adx" in latest else None,
        daily_trade_count=state.daily_trade_count,
        consecutive_losses=state.consecutive_losses,
        last_loss_at=state.last_loss_datetime,
        max_account_risk_pct=sizing.max_account_risk_pct,
        max_leverage=sizing.max_leverage,
        max_notional_pct=sizing.recommended_margin_pct,
        max_daily_trades=strategy.risk.max_daily_trades,
        min_balance_usdt=strategy.risk.min_balance_usdt,
        min_convergence=strategy.risk.min_convergence,
        min_score_long=strategy.risk.min_score_long,
        max_score_short=strategy.risk.max_score_short,
        cooldown_hours=strategy.risk.cooldown_hours,
        min_adx=strategy.risk.min_adx,
        liquidation_buffer_pct=liquidation_buffer_pct,
        min_liquidation_buffer_pct=min_liquidation_buffer_pct,
    )
    violations.extend(risk.violations)
    warnings.extend(risk.warnings)
    warnings.extend(sizing.warnings)
    decision_trace.extend(
        [
            trace_step(
                "risk_guard",
                allowed=risk.allowed,
                reasons=risk.violations,
                warnings=risk.warnings,
                data={"planned_account_risk_pct": planned_account_risk_pct, "leverage": leverage},
            ),
            trace_step(
                "symbol_sizing",
                allowed=True,
                warnings=sizing.warnings,
                data=sizing.to_dict(),
            ),
            trace_step(
                "exchange_microstructure",
                allowed=True,
                warnings=[
                    item
                    for item in warnings
                    if "Spread" in item or "Visible book" in item or "slippage" in item.lower()
                ],
                data={
                    "spread_bps": spread_bps,
                    "estimated_slippage_bps": dynamic_slippage_bps,
                    "fill_ratio": fill_ratio,
                },
            ),
        ]
    )
    live_plan_snapshot = {
        "side": side,
        "price": price,
        "quantity": quantity,
        "stop_price": stop_price,
        "take_profit_price": take_profit_price,
        "take_profit_prices": take_profit_prices,
        "take_profit_quantities": take_profit_quantities,
        "take_profit_weights": take_profit_weights,
        "take_profit_runner_quantity": runner_quantity,
        "required_structured_exit_quantity": required_structured_exit_quantity,
        "planned_account_risk_usdt": planned_account_risk_usdt,
        "analysis_score": int(analysis["score"]),
        "analysis_convergence": float(analysis["convergence"]),
        "adx_value": float(latest.get("adx", 0.0)),
        "fee_bps": strategy.execution.fee_bps,
        "slippage_bps": dynamic_slippage_bps,
        "spread_bps": spread_bps,
        "sizing": {**sizing.to_dict(), "signal_scores": signal_scores},
        "market_bot_gate": market_bot_gate,
        "risk_combo_gate": risk_combo_gate,
        "regime": analysis.get("regime"),
        "strategy_family": strategy_family,
        "selected_strategy_family": selected_family,
    }
    professional_gate_policy = ProfessionalGatePolicy(
        min_reward_risk=1.2,
        min_net_profit_to_risk=0.35 if execution_mode == "testnet_exploration" else 0.8,
        max_fee_profit_ratio=0.35,
        max_slippage_profit_ratio=0.25,
        max_spread_bps=12.0,
        max_volatility=1.8,
        min_volume_zscore=-0.8,
        min_obv_zscore_long=-0.5,
        max_obv_zscore_short=0.5,
        min_composite_quality=50.0,
        min_price_structure_score=55.0,
        min_execution_quality_score=45.0,
        min_recent_reviews=6,
        min_recent_win_rate=0.42,
        min_recent_profit_factor=0.85 if execution_mode == "testnet_exploration" else 1.0,
        min_recent_avg_r=-0.05 if execution_mode == "testnet_exploration" else 0.0,
        min_recent_expectancy_r=-0.05 if execution_mode == "testnet_exploration" else 0.0,
        min_recent_payoff_ratio=0.9 if execution_mode == "testnet_exploration" else 1.0,
        max_recent_stop_loss_ratio=0.65 if execution_mode == "testnet_exploration" else 0.55,
        recent_lookback=20,
        stop_loss_cooldown_hours=2.0 if execution_mode == "testnet_exploration" else 6.0,
        require_professional_gate=True,
        allow_thin_scoped_history=execution_mode == "testnet_exploration",
        allow_market_bot_evidence=execution_mode == "testnet_exploration" and market_bot_promoted,
        allow_risk_combo_evidence=execution_mode == "testnet_exploration" and risk_combo_promoted,
    )
    professional_gate = evaluate_professional_entry_gate(
        side=side,
        latest=latest,
        live_plan=live_plan_snapshot,
        policy=professional_gate_policy,
        symbol=symbol,
        route_id=route.route_id,
        strategy_profile=strategy.profile,
    )
    if not professional_gate.passed:
        violations.extend(professional_gate.violations)
    warnings.extend(professional_gate.warnings)
    professional_gate_payload = professional_gate.to_dict()
    decision_trace.append(
        trace_step(
            "professional_entry_gate",
            allowed=professional_gate.passed,
            reasons=professional_gate.violations,
            warnings=professional_gate.warnings,
            data=professional_gate_payload,
        )
    )
    violations = _soften_exploration_violations(
        violations,
        warnings,
        execution_mode=execution_mode,
        research_promoted=research_promoted,
    )
    decision_trace.append(
        trace_step(
            "final_plan",
            allowed=len(violations) == 0,
            reasons=violations,
            warnings=warnings,
            data={"execution_mode": execution_mode, "side": side, "quantity": quantity},
        )
    )

    if total_balance < 10.0:
        warnings.append(
            f"Micro-account mode: total futures balance is only {total_balance:.4f} USDT."
        )

    return LiveExecutionPlan(
        allowed=len(violations) == 0,
        symbol=symbol,
        market=market,
        side=side,
        quantity=quantity,
        price=round(price, rules.price_precision),
        leverage=leverage,
        wallet_balance_usdt=round(balance_snapshot.wallet_balance_usdt, 6),
        available_balance_usdt=round(balance_snapshot.available_balance_usdt, 6),
        equity_usdt=round(balance_snapshot.equity_usdt, 6),
        margin_notional_usdt=round(requested_margin, 6),
        gross_notional_usdt=round(actual_notional, 6),
        min_notional_usdt=round(rules.min_notional, 6),
        strategy_profile=strategy.profile,
        analysis_score=int(analysis["score"]),
        analysis_bias=str(bias),
        analysis_convergence=float(analysis["convergence"]),
        adx_value=round(float(latest.get("adx", 0.0)), 4),
        stop_price=round(stop_price, rules.price_precision),
        stop_distance=round(stop_distance, rules.price_precision),
        take_profit_price=round(take_profit_price, rules.price_precision),
        take_profit_prices=[round(item, rules.price_precision) for item in take_profit_prices],
        take_profit_quantities=take_profit_quantities,
        take_profit_weights=[round(item, 4) for item in take_profit_weights],
        take_profit_runner_quantity=runner_quantity,
        required_structured_exit_quantity=required_structured_exit_quantity,
        planned_account_risk_usdt=round(planned_account_risk_usdt, 6),
        planned_account_risk_pct=round(planned_account_risk_pct, 6),
        liquidation_buffer_pct=round(liquidation_buffer_pct, 6),
        min_liquidation_buffer_pct=round(min_liquidation_buffer_pct, 6),
        trailing_stop_enabled=effective_trailing_stop_enabled,
        trailing_callback_pct=round(strategy.risk.trailing_callback_pct, 4),
        fee_bps=round(strategy.execution.fee_bps, 4),
        slippage_bps=round(dynamic_slippage_bps, 4),
        spread_bps=round(spread_bps, 4),
        order_book_fill_ratio=round(fill_ratio, 4),
        market_context=market_context_payload,
        sizing={**sizing.to_dict(), "signal_scores": signal_scores},
        execution_mode=execution_mode,
        challenge={
            **challenge_summary,
            "optimizer_live_gate": optimizer_gate,
            "market_bot_gate": market_bot_gate,
            "route_quarantine": route_quarantine,
            "route_side_risk": side_risk_gate.to_dict(),
            "historical_signal_risk": historical_signal_gate.to_dict(),
        },
        decision_trace=decision_trace,
        professional_entry_gate=professional_gate_payload,
        violations=violations,
        warnings=warnings,
    )


def execute_live_order(
    settings: Settings,
    strategy: StrategyConfig,
    plan: LiveExecutionPlan,
    *,
    entry_reason_snapshot: dict[str, Any] | None = None,
    signal_scores: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not plan.allowed:
        raise RuntimeError("Execution plan is blocked. Refusing to submit live order.")
    if plan.side not in {"BUY", "SELL"}:
        raise RuntimeError("Execution plan does not contain an actionable side.")

    trading_control = load_trading_control_state()
    if trading_control.paused:
        reason = trading_control.reason or "manual kill-switch"
        raise RuntimeError(f"Trading is paused by kill-switch: {reason}.")

    with BinanceClient(settings) as client:
        if plan.market == "futures":
            client.set_margin_type(plan.symbol, strategy.execution.margin_type)
            client.set_leverage(plan.symbol, plan.leverage)
            # Clean up any orphaned algo orders before placing new entry
            existing_algo = client.open_algo_orders(plan.symbol)
            if existing_algo:
                cleanup = client.cancel_all_algo_orders(plan.symbol)
                # Log but don't fail on cleanup errors
                for item in cleanup:
                    if item.get("status") == "error":
                        raise RuntimeError(
                            f"Failed to cancel orphaned algo order {item.get('algoId')}: "
                            f"{item.get('error')}"
                        )
        entry_response = client.new_order(
            plan.symbol,
            plan.side,
            strategy.execution.order_type,
            quantity=plan.quantity,
            market=plan.market,
        )
        protective_orders: dict[str, Any] = {}
        if plan.market == "futures":
            exit_side = "SELL" if plan.side == "BUY" else "BUY"
            # Use reduce_only with explicit quantity (not closePosition)
            # to avoid orphaned algo orders that have no matching position
            protective_orders["stop_loss"] = client.new_algo_order(
                plan.symbol,
                exit_side,
                "STOP_MARKET",
                trigger_price=plan.stop_price,
                quantity=plan.quantity,
                reduce_only=True,
                working_type="MARK_PRICE",
                market=plan.market,
            )
            tp_prices = plan.take_profit_prices or [plan.take_profit_price]
            tp_quantities = plan.take_profit_quantities or [plan.quantity]
            protective_orders["take_profits"] = []
            for index, (tp_price, tp_quantity) in enumerate(zip(tp_prices, tp_quantities, strict=False), start=1):
                protective_orders["take_profits"].append(
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
                            market=plan.market,
                        ),
                    }
                )
            protective_orders["take_profit"] = (
                protective_orders["take_profits"][0]["response"]
                if protective_orders["take_profits"]
                else None
            )

    state = load_trade_state()
    state.record_trade()
    save_trade_state(state)
    route = resolve_symbol_route(plan.symbol)
    normalized_entry_reason = dict(entry_reason_snapshot or {})
    if "interval" not in normalized_entry_reason:
        normalized_entry_reason["interval"] = strategy.defaults.interval
    live_signal_scores = dict(signal_scores or {})
    journal_path = append_live_order(
        LiveOrderRecord(
            timestamp=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            symbol=plan.symbol,
            market=plan.market,
            side=plan.side,
            order_type=strategy.execution.order_type,
            quantity=plan.quantity,
            price=plan.price,
            leverage=plan.leverage,
            wallet_balance_usdt=plan.wallet_balance_usdt,
            available_balance_usdt=plan.available_balance_usdt,
            equity_usdt=plan.equity_usdt,
            challenge_status=str(plan.challenge.get("status", "inactive")),
            challenge_target_usdt=float(plan.challenge.get("target_balance_usdt", 0.0)),
            challenge_progress_pct=float(plan.challenge.get("progress_pct", 0.0)),
            notional_usdt=plan.margin_notional_usdt,
            gross_notional_usdt=plan.gross_notional_usdt,
            analysis_score=plan.analysis_score,
            analysis_bias=plan.analysis_bias,
            analysis_convergence=plan.analysis_convergence,
            order_id=entry_response.get("orderId"),
            status=str(entry_response.get("status", "NEW")),
            cohort_id=build_cohort_id(
                asset_class=route.asset_class,
                strategy_profile=strategy.profile,
                market=plan.market,
                interval=strategy.defaults.interval,
            ),
            strategy_profile=strategy.profile,
            strategy_path=str(strategy.path),
            asset_class=route.asset_class,
            route_id=route.route_id,
            simulation_mode=route.simulation_mode,
            review_lane=route.review_lane,
            entry_reason_snapshot=normalized_entry_reason,
            signal_scores=live_signal_scores,
            binance_response={
                "entry": entry_response,
                "protective_orders": protective_orders,
            },
            note=f"strategy={plan.strategy_profile}",
        )
    )
    return {
        "entry_order": entry_response,
        "protective_orders": protective_orders,
        "journal_path": str(journal_path),
    }
