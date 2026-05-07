from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import Any

from .config import STATE_DIR, ensure_runtime_dirs
from .convergence import calculate_loss_streak, calculate_profit_factor
from .order_journal import read_closed_trade_reviews

LOSS_DIAGNOSTICS_DIR = STATE_DIR / "loss-diagnostics"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return number


def _metric_float(value: Any) -> float:
    if value == "inf":
        return 9999.0
    if value == "-inf":
        return -9999.0
    number = _float(value)
    if math.isinf(number):
        return 9999.0 if number > 0 else -9999.0
    return number


def _json_float(value: Any, digits: int = 4) -> float | str:
    number = _float(value)
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return round(number, digits)


def _text(value: Any, fallback: str) -> str:
    normalized = str(value or "").strip()
    return normalized if normalized else fallback


def _route_id(row: dict[str, Any]) -> str:
    return _text(row.get("route_id"), "unrouted")


def _side(row: dict[str, Any]) -> str:
    value = _text(row.get("side"), "UNKNOWN").upper()
    return value if value in {"BUY", "SELL"} else "UNKNOWN"


def _source_bucket(row: dict[str, Any]) -> str:
    source = str(row.get("source_order_id") or "")
    regime = str(row.get("market_regime_tag") or "")
    note = str(row.get("note") or "").lower()
    if source.startswith("public-history") or regime == "public-history":
        return "binance-public-history"
    if source.startswith("market-replay") or regime == "mixed":
        return "market-replay"
    if "training" in note or "demo" in note:
        return "demo-training"
    if source.isdigit():
        return "binance-demo-or-live"
    return "unknown-source"


def _score_bin(row: dict[str, Any]) -> str:
    score = int(_float(row.get("analysis_score"), 0.0))
    if score <= 20:
        return "score-000-020"
    if score <= 40:
        return "score-021-040"
    if score <= 60:
        return "score-041-060"
    if score <= 80:
        return "score-061-080"
    return "score-081-100"


def _convergence_bin(row: dict[str, Any]) -> str:
    convergence = _float(row.get("analysis_convergence"), 0.0)
    if convergence < 0.7:
        return "conv-000-069"
    if convergence < 0.8:
        return "conv-070-079"
    if convergence < 0.9:
        return "conv-080-089"
    return "conv-090-100"


def _signal_score(row: dict[str, Any], key: str) -> float:
    signal_scores = row.get("signal_scores")
    if isinstance(signal_scores, dict):
        return _float(signal_scores.get(key))
    return 0.0


def _quality_bin(value: float, *, prefix: str, missing_label: str | None = None) -> str:
    if value <= 0.0 and missing_label:
        return missing_label
    if value < 40.0:
        band = "000-039"
    elif value < 55.0:
        band = "040-054"
    elif value < 70.0:
        band = "055-069"
    elif value < 82.0:
        band = "070-081"
    else:
        band = "082-100"
    return f"{prefix}-{band}"


def _flow_bin(row: dict[str, Any]) -> str:
    return _quality_bin(
        _signal_score(row, "flow_score"),
        prefix="flow",
        missing_label="flow-missing",
    )


def _event_risk_bin(row: dict[str, Any]) -> str:
    return _quality_bin(
        _signal_score(row, "event_risk_score"),
        prefix="event",
        missing_label="event-missing",
    )


def _execution_quality_bin(row: dict[str, Any]) -> str:
    return _quality_bin(
        _signal_score(row, "execution_quality_score"),
        prefix="exec",
        missing_label="exec-missing",
    )


def _composite_signal_bin(row: dict[str, Any]) -> str:
    return _quality_bin(
        _signal_score(row, "composite_convergence_score"),
        prefix="composite",
        missing_label="composite-missing",
    )


def _leverage_bin(row: dict[str, Any]) -> str:
    leverage = _float(row.get("leverage"))
    if leverage <= 0:
        return "lev-missing"
    if leverage <= 1:
        return "lev-001"
    if leverage <= 3:
        return "lev-002-003"
    if leverage <= 6:
        return "lev-004-006"
    return "lev-007-plus"


def _holding_time_bin(row: dict[str, Any]) -> str:
    opened = _parse_datetime(row.get("opened_at"))
    closed = _parse_datetime(row.get("closed_at") or row.get("reviewed_at"))
    if opened is None or closed is None:
        return "hold-missing"
    hours = max(0.0, (closed - opened).total_seconds() / 3600.0)
    if hours < 1.0:
        return "hold-lt-1h"
    if hours < 4.0:
        return "hold-001-004h"
    if hours < 12.0:
        return "hold-004-012h"
    if hours < 48.0:
        return "hold-012-048h"
    return "hold-048h-plus"


def _r_bin(row: dict[str, Any]) -> str:
    value = row.get("realized_r_multiple")
    if value in (None, ""):
        return "r-unknown"
    r_value = _float(value)
    if r_value <= -1.0:
        return "r-loss-ge-1"
    if r_value < 0.0:
        return "r-loss-lt-1"
    if r_value < 1.0:
        return "r-win-lt-1"
    return "r-win-ge-1"


def _time_key(row: dict[str, Any]) -> str:
    return str(row.get("closed_at") or row.get("opened_at") or row.get("reviewed_at") or "")


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _pnl(row: dict[str, Any]) -> float:
    return _float(row.get("realized_pnl_usdt"), 0.0)


def _bucket_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [_pnl(item) for item in records]
    wins = [item for item in pnls if item > 0.0]
    losses = [item for item in pnls if item < 0.0]
    breakeven = [item for item in pnls if item == 0.0]
    r_values = [
        _float(item.get("realized_r_multiple"))
        for item in records
        if item.get("realized_r_multiple") not in (None, "")
    ]
    stop_loss_count = sum(1 for item in records if str(item.get("exit_reason") or "") == "stop_loss")
    take_profit_count = sum(1 for item in records if str(item.get("exit_reason") or "") == "take_profit")
    high_conviction_losses = sum(
        1
        for item in records
        if _pnl(item) < 0.0 and _float(item.get("analysis_convergence"), 0.0) >= 0.8
    )
    count = len(records)
    net_pnl = sum(pnls)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = calculate_profit_factor(pnls)
    return {
        "count": count,
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(breakeven),
        "win_rate": round((len(wins) / count) * 100.0, 2) if count else 0.0,
        "gross_profit_usdt": round(gross_profit, 8),
        "gross_loss_usdt": round(gross_loss, 8),
        "net_pnl_usdt": round(net_pnl, 8),
        "avg_pnl_usdt": round(net_pnl / count, 8) if count else 0.0,
        "avg_win_usdt": round(sum(wins) / len(wins), 8) if wins else 0.0,
        "avg_loss_usdt": round(sum(losses) / len(losses), 8) if losses else 0.0,
        "avg_r": round(sum(r_values) / len(r_values), 4) if r_values else 0.0,
        "profit_factor": _json_float(profit_factor),
        "loss_streak": calculate_loss_streak(pnls),
        "stop_loss_count": stop_loss_count,
        "take_profit_count": take_profit_count,
        "stop_loss_ratio": round((stop_loss_count / count) * 100.0, 2) if count else 0.0,
        "high_conviction_loss_count": high_conviction_losses,
        "symbols": sorted({_text(item.get("symbol"), "UNKNOWN") for item in records}),
    }


DimensionFn = Callable[[dict[str, Any]], str]


def _group_records(
    records: Iterable[dict[str, Any]],
    *,
    dimensions: dict[str, DimensionFn],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = tuple(func(row) for func in dimensions.values())
        grouped[key].append(row)

    rows: list[dict[str, Any]] = []
    dimension_names = tuple(dimensions.keys())
    for key, bucket_records in grouped.items():
        labels = dict(zip(dimension_names, key, strict=True))
        rows.append(
            {
                "label": " | ".join(key),
                "dimensions": labels,
                "metrics": _bucket_metrics(bucket_records),
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            _float((item.get("metrics") or {}).get("net_pnl_usdt")),
            _metric_float((item.get("metrics") or {}).get("profit_factor")),
        ),
    )


def _worst_bucket_rows(
    bucket_groups: dict[str, list[dict[str, Any]]],
    *,
    min_bucket_trades: int,
    top_n: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for group_name, rows in bucket_groups.items():
        for row in rows:
            metrics = row.get("metrics") or {}
            if int(metrics.get("count") or 0) < min_bucket_trades:
                continue
            if _float(metrics.get("net_pnl_usdt")) >= 0.0:
                continue
            candidates.append({"group": group_name, **row})
    return sorted(
        candidates,
        key=lambda item: (
            _float((item.get("metrics") or {}).get("net_pnl_usdt")),
            _metric_float((item.get("metrics") or {}).get("profit_factor")),
        ),
    )[:top_n]


def _best_bucket_rows(
    bucket_groups: dict[str, list[dict[str, Any]]],
    *,
    min_bucket_trades: int,
    top_n: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for group_name, rows in bucket_groups.items():
        for row in rows:
            metrics = row.get("metrics") or {}
            if int(metrics.get("count") or 0) < min_bucket_trades:
                continue
            if _float(metrics.get("net_pnl_usdt")) <= 0.0:
                continue
            candidates.append({"group": group_name, **row})
    return sorted(
        candidates,
        key=lambda item: (
            _metric_float((item.get("metrics") or {}).get("profit_factor")),
            _float((item.get("metrics") or {}).get("net_pnl_usdt")),
        ),
        reverse=True,
    )[:top_n]


def _build_findings(
    *,
    summary: dict[str, Any],
    bucket_groups: dict[str, list[dict[str, Any]]],
    min_bucket_trades: int,
) -> list[str]:
    findings: list[str] = []
    if _metric_float(summary.get("profit_factor")) < 1.0:
        findings.append(
            f"overall-negative-expectancy: PF={summary.get('profit_factor')} net_pnl={summary.get('net_pnl_usdt')}"
        )
    if _float(summary.get("stop_loss_ratio")) >= 60.0:
        findings.append(
            f"stop-loss-dominant: stop_loss_ratio={summary.get('stop_loss_ratio')}%"
        )

    side_rows = {
        (row.get("dimensions") or {}).get("side"): row
        for row in bucket_groups.get("by_side", [])
        if int(((row.get("metrics") or {}).get("count") or 0)) >= min_bucket_trades
    }
    buy_pf = _metric_float(((side_rows.get("BUY") or {}).get("metrics") or {}).get("profit_factor"))
    sell_pf = _metric_float(((side_rows.get("SELL") or {}).get("metrics") or {}).get("profit_factor"))
    if side_rows.get("SELL") and sell_pf < 0.8:
        findings.append(f"short-lane-underperforming: SELL_PF={sell_pf:.4f}")
    if side_rows.get("BUY") and buy_pf < 0.8:
        findings.append(f"long-lane-underperforming: BUY_PF={buy_pf:.4f}")
    if side_rows.get("BUY") and side_rows.get("SELL") and sell_pf < buy_pf:
        findings.append(f"short-lane-worse-than-long: SELL_PF={sell_pf:.4f} BUY_PF={buy_pf:.4f}")

    route_side_rows = bucket_groups.get("by_route_side", [])
    weak_route_sides = [
        row
        for row in route_side_rows
        if int(((row.get("metrics") or {}).get("count") or 0)) >= min_bucket_trades
        and _metric_float((row.get("metrics") or {}).get("profit_factor")) < 0.8
    ]
    for row in weak_route_sides[:6]:
        dimensions = row.get("dimensions") or {}
        metrics = row.get("metrics") or {}
        findings.append(
            "weak-route-side:"
            f"{dimensions.get('route_id')}/{dimensions.get('side')}"
            f" PF={metrics.get('profit_factor')} net_pnl={metrics.get('net_pnl_usdt')}"
        )
    fast_stop_rows = [
        row
        for row in bucket_groups.get("by_route_side_holding_time", [])
        if int(((row.get("metrics") or {}).get("count") or 0)) >= min_bucket_trades
        and str((row.get("dimensions") or {}).get("holding_time_bin")) in {"hold-lt-1h", "hold-001-004h"}
        and _float((row.get("metrics") or {}).get("stop_loss_ratio")) >= 70.0
    ]
    for row in fast_stop_rows[:4]:
        dimensions = row.get("dimensions") or {}
        metrics = row.get("metrics") or {}
        findings.append(
            "fast-stop-cluster:"
            f"{dimensions.get('route_id')}/{dimensions.get('side')}/{dimensions.get('holding_time_bin')}"
            f" stop_loss_ratio={metrics.get('stop_loss_ratio')}% PF={metrics.get('profit_factor')}"
        )

    weak_flow_rows = [
        row
        for row in bucket_groups.get("by_route_side_flow_bin", [])
        if int(((row.get("metrics") or {}).get("count") or 0)) >= min_bucket_trades
        and str((row.get("dimensions") or {}).get("flow_bin")) in {"flow-000-039", "flow-040-054"}
        and _metric_float((row.get("metrics") or {}).get("profit_factor")) < 0.8
    ]
    for row in weak_flow_rows[:4]:
        dimensions = row.get("dimensions") or {}
        metrics = row.get("metrics") or {}
        findings.append(
            "weak-flow-loss-bucket:"
            f"{dimensions.get('route_id')}/{dimensions.get('side')}/{dimensions.get('flow_bin')}"
            f" PF={metrics.get('profit_factor')} net_pnl={metrics.get('net_pnl_usdt')}"
        )
    return list(dict.fromkeys(findings))


def _build_side_policy_recommendations(
    bucket_groups: dict[str, list[dict[str, Any]]],
    *,
    min_bucket_trades: int,
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for row in bucket_groups.get("by_route_side", []):
        metrics = row.get("metrics") or {}
        dimensions = row.get("dimensions") or {}
        count = int(metrics.get("count") or 0)
        if count < min_bucket_trades:
            continue
        profit_factor = _metric_float(metrics.get("profit_factor"))
        net_pnl = _float(metrics.get("net_pnl_usdt"))
        side = str(dimensions.get("side") or "")
        if side == "SELL" and profit_factor < 0.8:
            action = "disable-shorts-or-require-extra-confirmation"
        elif profit_factor < 0.8:
            action = "keep-route-quarantined-and-tighten-entry-quality"
        else:
            continue
        recommendations.append(
            {
                "route_id": dimensions.get("route_id"),
                "side": side,
                "action": action,
                "evidence": {
                    "count": count,
                    "profit_factor": metrics.get("profit_factor"),
                    "net_pnl_usdt": net_pnl,
                    "stop_loss_ratio": metrics.get("stop_loss_ratio"),
                },
            }
        )
    return recommendations


def _build_root_cause_recommendations(
    bucket_groups: dict[str, list[dict[str, Any]]],
    *,
    min_bucket_trades: int,
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []

    for row in bucket_groups.get("by_route_side_holding_time", []):
        metrics = row.get("metrics") or {}
        dimensions = row.get("dimensions") or {}
        count = int(metrics.get("count") or 0)
        if count < min_bucket_trades:
            continue
        if (
            str(dimensions.get("holding_time_bin")) in {"hold-lt-1h", "hold-001-004h"}
            and _float(metrics.get("stop_loss_ratio")) >= 70.0
            and _metric_float(metrics.get("profit_factor")) < 0.8
        ):
            recommendations.append(
                {
                    "type": "fast-stop-cluster",
                    "route_id": dimensions.get("route_id"),
                    "side": dimensions.get("side"),
                    "action": "require-stronger-entry-confirmation-or-widen-volatility-adjusted-stop",
                    "evidence": {
                        "holding_time_bin": dimensions.get("holding_time_bin"),
                        "count": count,
                        "profit_factor": metrics.get("profit_factor"),
                        "stop_loss_ratio": metrics.get("stop_loss_ratio"),
                        "avg_r": metrics.get("avg_r"),
                    },
                }
            )

    for row in bucket_groups.get("by_route_side_flow_bin", []):
        metrics = row.get("metrics") or {}
        dimensions = row.get("dimensions") or {}
        count = int(metrics.get("count") or 0)
        flow_bin = str(dimensions.get("flow_bin") or "")
        if count < min_bucket_trades or flow_bin not in {"flow-000-039", "flow-040-054"}:
            continue
        if _float(metrics.get("net_pnl_usdt")) < 0.0 and _metric_float(metrics.get("profit_factor")) < 0.8:
            recommendations.append(
                {
                    "type": "weak-flow-confirmation",
                    "route_id": dimensions.get("route_id"),
                    "side": dimensions.get("side"),
                    "action": "downsize-or-skip-unless-price-structure-and-external-context-align",
                    "evidence": {
                        "flow_bin": flow_bin,
                        "count": count,
                        "profit_factor": metrics.get("profit_factor"),
                        "net_pnl_usdt": metrics.get("net_pnl_usdt"),
                        "stop_loss_ratio": metrics.get("stop_loss_ratio"),
                    },
                }
            )

    for row in bucket_groups.get("by_route_side_composite_signal_bin", []):
        metrics = row.get("metrics") or {}
        dimensions = row.get("dimensions") or {}
        count = int(metrics.get("count") or 0)
        composite_bin = str(dimensions.get("composite_signal_bin") or "")
        if count < min_bucket_trades or composite_bin in {"composite-missing", "composite-070-081", "composite-082-100"}:
            continue
        if _float(metrics.get("net_pnl_usdt")) < 0.0 and _metric_float(metrics.get("profit_factor")) < 0.8:
            recommendations.append(
                {
                    "type": "weak-composite-signal",
                    "route_id": dimensions.get("route_id"),
                    "side": dimensions.get("side"),
                    "action": "lower-leverage-and-keep-testnet-only-until-positive-bucket-emerges",
                    "evidence": {
                        "composite_signal_bin": composite_bin,
                        "count": count,
                        "profit_factor": metrics.get("profit_factor"),
                        "net_pnl_usdt": metrics.get("net_pnl_usdt"),
                    },
                }
            )

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in recommendations:
        key = (
            str(item.get("type") or ""),
            str(item.get("route_id") or ""),
            str(item.get("side") or ""),
            str(item.get("action") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique[:12]


def run_loss_diagnostics(
    *,
    limit: int = 0,
    min_bucket_trades: int = 5,
    top_n: int = 20,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    LOSS_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    records = sorted(read_closed_trade_reviews(), key=_time_key)
    if limit > 0:
        records = records[-int(limit):]
    min_bucket_trades = max(1, int(min_bucket_trades))
    top_n = max(1, int(top_n))

    bucket_groups = {
        "by_route": _group_records(records, dimensions={"route_id": _route_id}),
        "by_side": _group_records(records, dimensions={"side": _side}),
        "by_route_side": _group_records(
            records,
            dimensions={"route_id": _route_id, "side": _side},
        ),
        "by_symbol": _group_records(records, dimensions={"symbol": lambda row: _text(row.get("symbol"), "UNKNOWN")}),
        "by_source": _group_records(records, dimensions={"source": _source_bucket}),
        "by_route_source": _group_records(
            records,
            dimensions={"route_id": _route_id, "source": _source_bucket},
        ),
        "by_exit_reason": _group_records(
            records,
            dimensions={"exit_reason": lambda row: _text(row.get("exit_reason"), "unknown-exit")},
        ),
        "by_regime": _group_records(
            records,
            dimensions={"regime": lambda row: _text(row.get("market_regime_tag"), "unknown-regime")},
        ),
        "by_route_score_bin": _group_records(
            records,
            dimensions={"route_id": _route_id, "score_bin": _score_bin},
        ),
        "by_route_convergence_bin": _group_records(
            records,
            dimensions={"route_id": _route_id, "convergence_bin": _convergence_bin},
        ),
        "by_route_side_score_convergence_bin": _group_records(
            records,
            dimensions={
                "route_id": _route_id,
                "side": _side,
                "score_bin": _score_bin,
                "convergence_bin": _convergence_bin,
            },
        ),
        "by_route_side_flow_bin": _group_records(
            records,
            dimensions={"route_id": _route_id, "side": _side, "flow_bin": _flow_bin},
        ),
        "by_route_side_event_risk_bin": _group_records(
            records,
            dimensions={
                "route_id": _route_id,
                "side": _side,
                "event_risk_bin": _event_risk_bin,
            },
        ),
        "by_route_side_execution_quality_bin": _group_records(
            records,
            dimensions={
                "route_id": _route_id,
                "side": _side,
                "execution_quality_bin": _execution_quality_bin,
            },
        ),
        "by_route_side_composite_signal_bin": _group_records(
            records,
            dimensions={
                "route_id": _route_id,
                "side": _side,
                "composite_signal_bin": _composite_signal_bin,
            },
        ),
        "by_route_side_leverage_bin": _group_records(
            records,
            dimensions={"route_id": _route_id, "side": _side, "leverage_bin": _leverage_bin},
        ),
        "by_route_side_holding_time": _group_records(
            records,
            dimensions={
                "route_id": _route_id,
                "side": _side,
                "holding_time_bin": _holding_time_bin,
            },
        ),
        "by_route_r_bin": _group_records(
            records,
            dimensions={"route_id": _route_id, "r_bin": _r_bin},
        ),
    }
    summary = _bucket_metrics(records)
    payload = {
        "generated_at": _utc_now().isoformat(),
        "status": "ok" if records else "no_reviews",
        "input": {
            "closed_review_count": len(records),
            "limit": int(limit),
            "min_bucket_trades": min_bucket_trades,
            "top_n": top_n,
        },
        "summary": summary,
        "findings": _build_findings(
            summary=summary,
            bucket_groups=bucket_groups,
            min_bucket_trades=min_bucket_trades,
        ),
        "side_policy_recommendations": _build_side_policy_recommendations(
            bucket_groups,
            min_bucket_trades=min_bucket_trades,
        ),
        "root_cause_recommendations": _build_root_cause_recommendations(
            bucket_groups,
            min_bucket_trades=min_bucket_trades,
        ),
        "worst_buckets": _worst_bucket_rows(
            bucket_groups,
            min_bucket_trades=min_bucket_trades,
            top_n=top_n,
        ),
        "best_buckets": _best_bucket_rows(
            bucket_groups,
            min_bucket_trades=min_bucket_trades,
            top_n=top_n,
        ),
        "buckets": bucket_groups,
    }
    report_path = LOSS_DIAGNOSTICS_DIR / f"{_stamp()}-loss-diagnostics.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload
