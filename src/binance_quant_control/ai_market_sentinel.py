from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analysis import enrich_indicators, prepare_klines_frame
from .binance_api import BinanceAPIError, BinanceClient
from .config import CONFIG_DIR, STATE_DIR, load_settings
from .readiness_scanner import run_ai_readiness_scan
from .route_risk_control import load_route_risk_state
from .trading_control import load_trading_control_state

DEFAULT_SENTINEL_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "TRXUSDT")
DEFAULT_STRATEGY_CONFIG = CONFIG_DIR / "strategy-live-pilot.yaml"
DEFAULT_BLUEPRINT_CONFIG = CONFIG_DIR / "professional-system-blueprint.default.yaml"
MAX_CONCURRENT_POSITIONS = 4


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _normalize_positions(raw_positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positions: list[dict[str, Any]] = []
    for item in raw_positions:
        amount = _float(item.get("positionAmt"))
        if amount == 0.0:
            continue
        entry = _float(item.get("entryPrice"))
        symbol = str(item.get("symbol") or "").upper()
        if not symbol:
            continue
        positions.append(
            {
                "symbol": symbol,
                "side": "LONG" if amount > 0 else "SHORT",
                "qty": abs(amount),
                "signed_qty": amount,
                "entry": entry,
                "pnl": _float(item.get("unRealizedProfit")),
                "leverage": _int(item.get("leverage"), 1),
            }
        )
    return positions


def _trend_from_klines(raw_klines: list[list[Any]], *, interval: str) -> dict[str, Any]:
    if len(raw_klines) < 60:
        return {"status": "insufficient_data", "bias": "unknown"}
    frame = enrich_indicators(prepare_klines_frame(raw_klines), interval)
    latest = frame.iloc[-1]
    previous = frame.iloc[-8] if len(frame) >= 8 else frame.iloc[-2]
    close = _float(latest.get("close"))
    ema_fast = _float(latest.get("ema_fast"))
    ema_slow = _float(latest.get("ema_slow"))
    ema_fast_prev = _float(previous.get("ema_fast"))
    rsi = _float(latest.get("rsi_14"), 50.0)
    adx = _float(latest.get("adx"))
    plus_di = _float(latest.get("plus_di"))
    minus_di = _float(latest.get("minus_di"))
    slope_pct = ((ema_fast - ema_fast_prev) / ema_fast_prev * 100.0) if ema_fast_prev else 0.0
    long_votes = 0
    short_votes = 0
    if close > ema_fast > ema_slow:
        long_votes += 2
    if close < ema_fast < ema_slow:
        short_votes += 2
    if slope_pct > 0:
        long_votes += 1
    elif slope_pct < 0:
        short_votes += 1
    if adx >= 18 and plus_di > minus_di:
        long_votes += 1
    if adx >= 18 and minus_di > plus_di:
        short_votes += 1
    if rsi >= 55:
        long_votes += 1
    elif rsi <= 45:
        short_votes += 1
    if long_votes >= short_votes + 2:
        bias = "long"
    elif short_votes >= long_votes + 2:
        bias = "short"
    else:
        bias = "mixed"
    return {
        "status": "ok",
        "bias": bias,
        "close": round(close, 6),
        "ema_fast": round(ema_fast, 6),
        "ema_slow": round(ema_slow, 6),
        "ema_fast_slope_pct_8": round(slope_pct, 4),
        "rsi_14": round(rsi, 4),
        "adx": round(adx, 4),
        "plus_di": round(plus_di, 4),
        "minus_di": round(minus_di, 4),
        "votes": {"long": long_votes, "short": short_votes},
    }


def _position_risk_overlay(
    positions: list[dict[str, Any]],
    trend_state: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    overlays: list[dict[str, Any]] = []
    for position in positions:
        symbol = str(position.get("symbol") or "")
        trend = trend_state.get(symbol) or {}
        side = str(position.get("side") or "")
        entry = _float(position.get("entry"))
        close = _float(trend.get("close"))
        distance_pct = ((close - entry) / entry * 100.0) if entry and close else 0.0
        aligned = (side == "LONG" and trend.get("bias") == "long") or (
            side == "SHORT" and trend.get("bias") == "short"
        )
        overlays.append(
            {
                "symbol": symbol,
                "side": side,
                "qty": position.get("qty"),
                "entry": entry,
                "mark_proxy": close,
                "unrealized_pnl_usdt": position.get("pnl"),
                "distance_from_entry_pct": round(distance_pct, 4),
                "trend_bias": trend.get("bias", "unknown"),
                "trend_aligned": aligned,
                "priority": "protect_or_trail" if aligned else "tighten_or_review",
            }
        )
    return overlays


def _build_expansion_gate(
    *,
    positions: list[dict[str, Any]],
    trading_control: dict[str, Any],
    readiness: dict[str, Any],
    active_quarantines: list[str],
) -> dict[str, Any]:
    blockers: list[str] = []
    if len(positions) >= MAX_CONCURRENT_POSITIONS:
        blockers.append("max-concurrent-positions-reached")
    if bool(trading_control.get("paused")):
        blockers.append("trading-control-paused")
    if int(readiness.get("allowed_count") or 0) <= 0:
        blockers.append("no-readiness-approved-candidate")
    if active_quarantines:
        blockers.append("active-negative-expectancy-route-quarantine")
    return {
        "allowed": not blockers,
        "blockers": blockers,
        "readiness_allowed_count": int(readiness.get("allowed_count") or 0),
        "active_quarantined_routes": active_quarantines,
        "open_position_count": len(positions),
        "max_concurrent_positions": MAX_CONCURRENT_POSITIONS,
    }


def _build_machine_action_queue(
    *,
    positions: list[dict[str, Any]],
    expansion_gate: dict[str, Any],
    readiness: dict[str, Any],
    position_overlays: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if positions:
        actions.append(
            {
                "priority": 10,
                "action": "run_position_guardian",
                "reason": "open-position-management-priority",
                "symbols": [str(item.get("symbol")) for item in positions],
                "command": "openclaw-quantctl hermes-trade cycle --force --dry-run-only --compact",
                "position_overlays": position_overlays,
            }
        )
    if "no-readiness-approved-candidate" in expansion_gate.get("blockers", []):
        actions.append(
            {
                "priority": 30,
                "action": "run_ai_expectancy_upgrade",
                "reason": "no-readiness-approved-candidate",
                "command": (
                    "openclaw-quantctl ai-expectancy-upgrade --universe-limit 20 --limit 8000 "
                    "--sweep-limit 5000 --max-configs 80 --max-walk-forward-validations 12 "
                    "--max-readiness-candidates 6 --compact"
                ),
            }
        )
    if "active-negative-expectancy-route-quarantine" in expansion_gate.get("blockers", []):
        actions.append(
            {
                "priority": 40,
                "action": "keep_route_quarantine",
                "reason": "negative-expectancy-routes-remain-active",
                "routes": expansion_gate.get("active_quarantined_routes", []),
            }
        )
    if expansion_gate.get("allowed"):
        ticket = readiness.get("execution_ticket")
        actions.append(
            {
                "priority": 20,
                "action": "operator_testnet_preflight",
                "reason": "readiness-approved-candidate-present",
                "ticket": ticket,
            }
        )
    return sorted(actions, key=lambda item: int(item.get("priority") or 999))


def run_ai_market_sentinel(
    *,
    symbols: list[str] | tuple[str, ...] | None = None,
    interval: str = "15m",
    limit: int = 160,
    market: str = "futures",
    strategy_config: str | Path = DEFAULT_STRATEGY_CONFIG,
    blueprint_config: str | Path = DEFAULT_BLUEPRINT_CONFIG,
    output_dir: str | Path | None = None,
    skip_readiness: bool = False,
) -> dict[str, Any]:
    symbol_list = tuple(str(item).upper() for item in (symbols or DEFAULT_SENTINEL_SYMBOLS) if str(item).strip())
    settings = load_settings()
    generated_at = _utc_now().isoformat()
    errors: list[str] = []
    positions: list[dict[str, Any]] = []
    trend_state: dict[str, dict[str, Any]] = {}
    with BinanceClient(settings) as client:
        try:
            raw_positions = client.positions()
            positions = _normalize_positions(raw_positions if isinstance(raw_positions, list) else [])
        except (BinanceAPIError, OSError, ValueError) as exc:
            errors.append(f"positions:{exc}")
        trend_symbols = sorted(set(symbol_list) | {str(item.get("symbol")) for item in positions if item.get("symbol")})
        for symbol in trend_symbols:
            try:
                trend_state[symbol] = _trend_from_klines(
                    client.klines(symbol, interval, int(limit), market),
                    interval=interval,
                )
            except (BinanceAPIError, OSError, ValueError) as exc:
                trend_state[symbol] = {"status": "error", "bias": "unknown", "error": str(exc)}
                errors.append(f"trend:{symbol}:{exc}")
    trading_control = load_trading_control_state().to_dict()
    route_risk = load_route_risk_state()
    active_quarantines = [str(item) for item in (route_risk.get("active_quarantined_routes") or [])]
    readiness: dict[str, Any]
    if skip_readiness:
        readiness = {
            "status": "skipped",
            "candidate_count": None,
            "allowed_count": 0,
            "next_machine_action": "readiness-scan-skipped",
            "hard_blocker_taxonomy": {},
            "report_path": None,
        }
    else:
        readiness = run_ai_readiness_scan(
            blueprint_config=Path(blueprint_config),
            strategy_config=Path(strategy_config),
            market=market,
            limit=0,
            margin_notional_usdt=None,
            execution_mode="testnet_exploration",
        )
    position_overlays = _position_risk_overlay(positions, trend_state)
    expansion_gate = _build_expansion_gate(
        positions=positions,
        trading_control=trading_control,
        readiness=readiness,
        active_quarantines=active_quarantines,
    )
    action_queue = _build_machine_action_queue(
        positions=positions,
        expansion_gate=expansion_gate,
        readiness=readiness,
        position_overlays=position_overlays,
    )
    payload: dict[str, Any] = {
        "generated_at": generated_at,
        "mode": "ai_market_sentinel_v1",
        "safety": {
            "opens_orders": False,
            "cancels_orders": False,
            "closes_positions": False,
            "writes_execution_config": False,
            "mainnet_live_allowed": False,
        },
        "symbols": list(symbol_list),
        "interval": interval,
        "market": market,
        "trading_control": trading_control,
        "position_state": {
            "open_position_count": len(positions),
            "positions": positions,
            "overlays": position_overlays,
        },
        "trend_state": trend_state,
        "route_risk": {
            "active_quarantined_routes": active_quarantines,
            "route_count": len(route_risk.get("routes") or {}),
        },
        "readiness": {
            "candidate_count": readiness.get("candidate_count"),
            "allowed_count": readiness.get("allowed_count"),
            "selected_ready_candidate": readiness.get("selected_ready_candidate"),
            "next_machine_action": readiness.get("next_machine_action"),
            "hard_blocker_taxonomy": readiness.get("hard_blocker_taxonomy"),
            "report_path": readiness.get("report_path"),
        },
        "expansion_gate": expansion_gate,
        "machine_action_queue": action_queue,
        "errors": errors,
    }
    report_dir = Path(output_dir).expanduser().resolve() if output_dir else STATE_DIR / "ai-market-sentinel"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{_stamp()}-ai-market-sentinel.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    payload["report_path"] = str(report_path)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload
