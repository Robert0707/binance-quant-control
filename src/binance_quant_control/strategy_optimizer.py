from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .asset_routing import RouteValidationSpec, resolve_symbol_route
from .config import PROJECT_ROOT, STATE_DIR
from .convergence import (
    ConvergenceMetrics,
    build_cohort_id,
    calculate_expectancy_stats,
    calculate_loss_streak,
    calculate_max_drawdown_pct,
    calculate_profit_factor,
    evaluate_convergence,
)
from .order_journal import read_closed_trade_reviews
from .payoff_objective import (
    PayoffObjectiveTargets,
    payoff_objective_sort_key,
    promotion_decision_rank,
)
from .strategy import StrategyConfig, load_strategy_config

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "strategy-optimizer.default.yaml"
OPTIMIZER_STATE_DIR = STATE_DIR / "strategy-optimizer"
LIVE_PROMOTION_DECISIONS = frozenset({"promote", "elite_candidate"})
MARKET_BOT_GATE_PATTERNS = (
    "market-bot-gate/*-market-bot-gate.json",
    "market-bot-*/*-market-bot-gate.json",
)
RISK_COMBO_MATRIX_DIR = STATE_DIR / "risk-combo-matrix"
TUNABLE_STRATEGY_PATHS = frozenset(
    {
        "profile",
        "description",
        "risk.min_convergence",
        "risk.min_score_long",
        "risk.max_score_short",
        "risk.cooldown_hours",
        "risk.atr_stop_multiple",
        "risk.min_adx",
        "risk.trailing_callback_pct",
        "risk.take_profit_r_multiples",
    }
)


@dataclass(frozen=True, slots=True)
class StrategyOptimizerConfig:
    path: Path
    base_strategy_config: Path
    output_strategy_config: Path
    min_closed_reviews: int
    lookback_reviews: int
    lookback_reviews_per_route: int
    auto_apply: bool
    screening_min_win_rate: float
    screening_min_profit_factor: float
    screening_min_expectancy_r: float
    screening_min_payoff_ratio: float
    screening_min_trades: int
    validation_min_win_rate: float
    validation_min_profit_factor: float
    validation_min_expectancy_r: float
    validation_min_payoff_ratio: float
    validation_min_simulated_trades: int
    elite_min_win_rate: float
    elite_min_profit_factor: float
    max_drawdown_pct: float
    min_avg_r_multiple: float
    max_stop_loss_ratio: float
    max_loss_streak: int
    ignore_legacy_unrouted: bool
    exclude_flat_manual_closes: bool


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _parse_report_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def latest_optimizer_report(state_dir: Path | None = None) -> dict[str, Any] | None:
    active_state_dir = state_dir or OPTIMIZER_STATE_DIR
    candidates = sorted(
        active_state_dir.glob("*-strategy-optimizer.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payload.setdefault("report_path", str(path))
            return payload
    return None


def latest_market_bot_gate_report(state_dir: Path | None = None) -> dict[str, Any] | None:
    active_state_dir = state_dir or STATE_DIR
    candidates: set[Path] = set()
    for pattern in MARKET_BOT_GATE_PATTERNS:
        candidates.update(active_state_dir.glob(pattern))
    candidates.update(active_state_dir.rglob("*-market-bot-gate.json"))
    latest: tuple[datetime, float, dict[str, Any]] | None = None
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payload.setdefault("report_path", str(path))
            generated_at = _parse_report_datetime(payload.get("generated_at"))
            sort_time = generated_at or datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            candidate = (sort_time, path.stat().st_mtime, payload)
            if latest is None or candidate[:2] > latest[:2]:
                latest = candidate
    if latest is None:
        return None
    return latest[2]


def latest_risk_combo_matrix_report(state_dir: Path | None = None) -> dict[str, Any] | None:
    active_state_dir = state_dir or RISK_COMBO_MATRIX_DIR
    candidates = sorted(
        active_state_dir.glob("*-risk-combo-matrix.json") if active_state_dir.exists() else [],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    latest: tuple[datetime, float, dict[str, Any]] | None = None
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payload.setdefault("report_path", str(path))
            generated_at = _parse_report_datetime(payload.get("generated_at"))
            sort_time = generated_at or datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            candidate = (sort_time, path.stat().st_mtime, payload)
            if latest is None or candidate[:2] > latest[:2]:
                latest = candidate
    if latest is None:
        return None
    return latest[2]


def _risk_combo_surface_metric(surface: dict[str, Any], section: str, key: str) -> float:
    metrics = surface.get(section) if isinstance(surface.get(section), dict) else {}
    try:
        return float(metrics.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def evaluate_risk_combo_live_gate(
    *,
    symbol: str,
    route_id: str = "",
    side: str = "",
    interval: str = "",
    max_report_age_hours: float = 24.0,
) -> dict[str, Any]:
    report = latest_risk_combo_matrix_report()
    if report is None:
        return {
            "allowed": False,
            "source": "risk_combo_matrix",
            "report_path": None,
            "report_age_hours": None,
            "reasons": ["No risk-combo matrix report is available."],
            "matched_surface": None,
        }

    reasons: list[str] = []
    generated_at = _parse_report_datetime(report.get("generated_at"))
    age_hours = None
    if generated_at is None:
        reasons.append("Latest risk-combo matrix report has no parseable generated_at timestamp.")
    else:
        age_hours = (_utc_now() - generated_at).total_seconds() / 3600.0
        if max_report_age_hours > 0 and age_hours > max_report_age_hours:
            reasons.append(
                f"Latest risk-combo matrix report is stale at {age_hours:.1f}h "
                f"(max {max_report_age_hours:.1f}h)."
            )

    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    if safety.get("opens_orders") or safety.get("writes_execution_config") or safety.get("mainnet_live_allowed"):
        reasons.append("Risk-combo matrix safety boundary is not research-only.")

    normalized_symbol = str(symbol or "").upper()
    normalized_route = str(route_id or "")
    normalized_side = str(side or "").upper()
    normalized_interval = str(interval or "")
    surfaces = [row for row in (report.get("surfaces") or []) if isinstance(row, dict)]
    if not surfaces and isinstance(report.get("best_surface"), dict):
        surfaces = [report["best_surface"]]

    matched = next(
        (
            row
            for row in surfaces
            if str(row.get("symbol") or "").upper() == normalized_symbol
            and (not normalized_route or str(row.get("route_id") or "") == normalized_route)
            and (not normalized_side or str(row.get("target_side") or "").upper() == normalized_side)
            and (not normalized_interval or str(row.get("target_interval") or "") == normalized_interval)
        ),
        None,
    )
    if matched is None:
        matched = next(
            (
                row
                for row in surfaces
                if str(row.get("symbol") or "").upper() == normalized_symbol
                and (not normalized_side or str(row.get("target_side") or "").upper() == normalized_side)
                and (not normalized_interval or str(row.get("target_interval") or "") == normalized_interval)
            ),
            None,
        )
    if matched is None:
        reasons.append(f"Risk-combo matrix has no matching robust surface for {normalized_symbol}.")
    else:
        if not matched.get("promotion_eligible"):
            reasons.append("Risk-combo surface is not promotion eligible.")
        if not matched.get("recovery_gate_passed"):
            reasons.append("Risk-combo recovery gate has not passed.")
        if not matched.get("robust_recovery_gate_passed"):
            reasons.append("Risk-combo robust recovery gate has not passed.")
        full_trades = _risk_combo_surface_metric(matched, "full", "trade_count")
        test_trades = _risk_combo_surface_metric(matched, "test", "trade_count")
        full_pf = _risk_combo_surface_metric(matched, "full", "profit_factor")
        test_pf = _risk_combo_surface_metric(matched, "test", "profit_factor")
        full_expectancy = _risk_combo_surface_metric(matched, "full", "expectancy_r")
        test_expectancy = _risk_combo_surface_metric(matched, "test", "expectancy_r")
        stop_loss_ratio = _risk_combo_surface_metric(matched, "full", "stop_loss_ratio")
        walk_forward = matched.get("walk_forward") if isinstance(matched.get("walk_forward"), dict) else {}
        wf_windows = int(_risk_combo_surface_metric(matched, "walk_forward", "window_count"))
        wf_positive = int(_risk_combo_surface_metric(matched, "walk_forward", "positive_expectancy_window_count"))
        wf_min_pf = _risk_combo_surface_metric(matched, "walk_forward", "min_profit_factor")
        wf_min_expectancy = _risk_combo_surface_metric(matched, "walk_forward", "min_expectancy_r")
        if full_trades < 30:
            reasons.append(f"Risk-combo full sample has only {full_trades:.0f} trades; minimum is 30.")
        if test_trades < 10:
            reasons.append(f"Risk-combo test sample has only {test_trades:.0f} trades; minimum is 10.")
        if full_pf < 1.0 or test_pf < 1.0:
            reasons.append("Risk-combo full/test profit factor is below 1.0.")
        if full_expectancy < 0.0 or test_expectancy < 0.0:
            reasons.append("Risk-combo full/test expectancy is not positive.")
        if stop_loss_ratio > 55.0:
            reasons.append(f"Risk-combo stop-loss ratio {stop_loss_ratio:.2f}% exceeds 55.00%.")
        if wf_windows < 3 or wf_positive < wf_windows:
            reasons.append("Risk-combo walk-forward validation is not consistently positive.")
        if wf_min_pf < 1.0 or wf_min_expectancy < 0.0:
            reasons.append("Risk-combo walk-forward minimum PF/expectancy is below target.")
        if walk_forward and not matched.get("source_report_path"):
            reasons.append("Risk-combo surface has no source sweep report path.")

    return {
        "allowed": not reasons,
        "source": "risk_combo_matrix",
        "report_path": report.get("report_path"),
        "report_age_hours": round(age_hours, 4) if age_hours is not None else None,
        "robust_surface_count": int(report.get("robust_surface_count") or 0),
        "promising_surface_count": int(report.get("promising_surface_count") or 0),
        "reasons": reasons,
        "matched_surface": matched,
    }


def evaluate_market_bot_live_gate(
    *,
    symbol: str,
    route_id: str = "",
    max_report_age_hours: float = 24.0,
) -> dict[str, Any]:
    report = latest_market_bot_gate_report()
    if report is None:
        return {
            "allowed": False,
            "source": "market_bot_gate",
            "report_path": None,
            "report_age_hours": None,
            "reasons": ["No market-bot gate report is available."],
            "matched_row": None,
        }
    reasons: list[str] = []
    generated_at = _parse_report_datetime(report.get("generated_at"))
    age_hours = None
    if generated_at is None:
        reasons.append("Latest market-bot gate report has no parseable generated_at timestamp.")
    else:
        age_hours = (_utc_now() - generated_at).total_seconds() / 3600.0
        if max_report_age_hours > 0 and age_hours > max_report_age_hours:
            reasons.append(
                f"Latest market-bot gate report is stale at {age_hours:.1f}h "
                f"(max {max_report_age_hours:.1f}h)."
            )
    if not report.get("safe_to_open_new_entries"):
        reasons.append("Market-bot gate has not marked this portfolio safe to open new entries.")
    portfolio_gate = report.get("portfolio_gate") if isinstance(report.get("portfolio_gate"), dict) else {}
    if portfolio_gate.get("enabled") and not portfolio_gate.get("passed"):
        reasons.append("Market-bot portfolio gate failed.")
    normalized_symbol = str(symbol or "").upper()
    normalized_route = str(route_id or "")
    accepted = [row for row in (report.get("accepted") or []) if isinstance(row, dict)]
    matched = next(
        (
            row
            for row in accepted
            if str(row.get("symbol") or "").upper() == normalized_symbol
            and (not normalized_route or str(row.get("route_id") or "") == normalized_route)
        ),
        None,
    )
    if matched is None:
        matched = next(
            (
                row
                for row in accepted
                if str(row.get("symbol") or "").upper() == normalized_symbol
            ),
            None,
        )
    if matched is None:
        reasons.append(f"Market-bot gate has no accepted row for {normalized_symbol}.")
    return {
        "allowed": not reasons,
        "source": "market_bot_gate",
        "report_path": report.get("report_path"),
        "report_age_hours": round(age_hours, 4) if age_hours is not None else None,
        "safe_to_open_new_entries": bool(report.get("safe_to_open_new_entries")),
        "accepted_count": int(report.get("accepted_count") or len(accepted)),
        "accepted_symbols": sorted({str(row.get("symbol")) for row in accepted if row.get("symbol")}),
        "feature_manifest_hash": report.get("feature_manifest_hash"),
        "reasons": reasons,
        "matched_row": matched,
    }


def evaluate_optimizer_live_gate(*, max_report_age_hours: float = 6.0) -> dict[str, Any]:
    report = latest_optimizer_report()
    if report is None:
        return {
            "allowed": False,
            "promotion_decision": None,
            "report_path": None,
            "report_age_hours": None,
            "reasons": ["No strategy optimizer report is available; live entry is blocked."],
        }

    reasons: list[str] = []
    generated_at = _parse_report_datetime(report.get("generated_at"))
    age_hours = None
    if generated_at is None:
        reasons.append("Latest optimizer report has no parseable generated_at timestamp.")
    else:
        age_hours = (_utc_now() - generated_at).total_seconds() / 3600.0
        if max_report_age_hours > 0 and age_hours > max_report_age_hours:
            reasons.append(
                f"Latest optimizer report is stale at {age_hours:.1f}h "
                f"(max {max_report_age_hours:.1f}h)."
            )

    decision = str(report.get("promotion_decision") or "")
    if decision not in LIVE_PROMOTION_DECISIONS:
        reasons.append(
            "Global strategy optimizer has not promoted this system "
            f"(promotion_decision={decision or 'unknown'})."
        )

    return {
        "allowed": not reasons,
        "promotion_decision": decision or None,
        "screening_status": report.get("screening_status"),
        "validation_status": report.get("validation_status"),
        "report_path": report.get("report_path"),
        "report_age_hours": round(age_hours, 4) if age_hours is not None else None,
        "reasons": reasons,
    }


def _resolve_project_config(value: str, default_name: str) -> Path:
    candidate = Path(value or default_name).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()
    return (PROJECT_ROOT / "config" / candidate).resolve()


def load_optimizer_config(path: str | Path | None = None) -> StrategyOptimizerConfig:
    config_path = Path(path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Optimizer config must be a mapping: {config_path}")
    strategy_cfg = payload.get("strategy") or {}
    policy_cfg = payload.get("policy") or {}
    output_cfg = payload.get("output") or {}
    return StrategyOptimizerConfig(
        path=config_path,
        base_strategy_config=_resolve_project_config(
            str(strategy_cfg.get("base_strategy_config") or ""),
            "strategy-hermes-pro.yaml",
        ),
        output_strategy_config=_resolve_project_config(
            str(output_cfg.get("output_strategy_config") or ""),
            "strategy-hermes-pro.auto.yaml",
        ),
        min_closed_reviews=int(policy_cfg.get("min_closed_reviews") or 8),
        lookback_reviews=int(policy_cfg.get("lookback_reviews") or 40),
        lookback_reviews_per_route=int(policy_cfg.get("lookback_reviews_per_route") or 80),
        auto_apply=bool(output_cfg.get("auto_apply", True)),
        screening_min_win_rate=float(policy_cfg.get("screening_min_win_rate") or 80.0),
        screening_min_profit_factor=float(policy_cfg.get("screening_min_profit_factor") or 1.2),
        screening_min_expectancy_r=float(policy_cfg.get("screening_min_expectancy_r") or 0.05),
        screening_min_payoff_ratio=float(policy_cfg.get("screening_min_payoff_ratio") or 1.0),
        screening_min_trades=int(policy_cfg.get("screening_min_trades") or 100),
        validation_min_win_rate=float(policy_cfg.get("validation_min_win_rate") or 80.0),
        validation_min_profit_factor=float(policy_cfg.get("validation_min_profit_factor") or 1.5),
        validation_min_expectancy_r=float(policy_cfg.get("validation_min_expectancy_r") or 0.10),
        validation_min_payoff_ratio=float(policy_cfg.get("validation_min_payoff_ratio") or 1.15),
        validation_min_simulated_trades=int(policy_cfg.get("validation_min_simulated_trades") or 100),
        elite_min_win_rate=float(policy_cfg.get("elite_min_win_rate") or 90.0),
        elite_min_profit_factor=float(policy_cfg.get("elite_min_profit_factor") or 1.5),
        max_drawdown_pct=float(policy_cfg.get("max_drawdown_pct") or 15.0),
        min_avg_r_multiple=float(policy_cfg.get("min_avg_r_multiple") or 0.2),
        max_stop_loss_ratio=float(policy_cfg.get("max_stop_loss_ratio") or 0.5),
        max_loss_streak=int(policy_cfg.get("max_loss_streak") or 3),
        ignore_legacy_unrouted=bool(policy_cfg.get("ignore_legacy_unrouted", True)),
        exclude_flat_manual_closes=bool(policy_cfg.get("exclude_flat_manual_closes", True)),
    )


def _recent_reviews(limit: int) -> list[dict[str, Any]]:
    reviews = read_closed_trade_reviews()
    return reviews[-limit:] if limit > 0 else reviews


def _balanced_recent_reviews(
    reviews: list[dict[str, Any]],
    *,
    per_route_limit: int,
    total_limit: int,
) -> list[dict[str, Any]]:
    if not reviews:
        return []
    if per_route_limit <= 0 and total_limit <= 0:
        return list(reviews)
    by_route: dict[str, list[dict[str, Any]]] = {}
    for item in reviews:
        route_id = str(item.get("route_id") or item.get("asset_class") or "unrouted")
        by_route.setdefault(route_id, []).append(item)
    selected: list[dict[str, Any]] = []
    for route_items in by_route.values():
        selected.extend(route_items[-per_route_limit:] if per_route_limit > 0 else route_items)
    if total_limit > 0 and len(selected) > total_limit:
        selected = selected[-total_limit:]
    return selected


def _strategy_profile_from_review(item: dict[str, Any], route: Any | None) -> str:
    direct = str(item.get("strategy_profile") or "").strip()
    if direct:
        return direct
    note = str(item.get("note") or "")
    if "strategy=" in note:
        extracted = note.split("strategy=", 1)[1].split()[0].strip().strip(",;")
        if extracted:
            return extracted
    if route is not None:
        return route.strategy_config.stem.replace("strategy-", "")
    return "unknown-profile"


def _normalize_review(item: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(item)
    symbol = str(normalized.get("symbol") or "").upper()
    route = None
    try:
        route = resolve_symbol_route(symbol) if symbol else None
    except ValueError:
        route = None
    strategy_profile = _strategy_profile_from_review(normalized, route)
    market = str(normalized.get("market") or getattr(route, "market", "futures"))
    interval = str(
        ((normalized.get("entry_reason_snapshot") or {}).get("interval"))
        or getattr(getattr(route, "interval", None), "strip", lambda: "")()
        or getattr(route, "interval", "unknown")
    )
    inferred_route = False
    inferred_cohort = False
    if route is not None:
        if not item.get("route_id"):
            inferred_route = True
        normalized["route_id"] = str(normalized.get("route_id") or route.route_id)
        normalized["asset_class"] = str(normalized.get("asset_class") or route.asset_class)
        normalized["review_lane"] = str(normalized.get("review_lane") or route.review_lane)
    normalized["strategy_profile"] = strategy_profile
    if not normalized.get("cohort_id"):
        inferred_cohort = True
    normalized["cohort_id"] = str(
        normalized.get("cohort_id")
        or build_cohort_id(
            asset_class=str(normalized.get("asset_class") or getattr(route, "asset_class", "unknown")),
            strategy_profile=strategy_profile,
            market=market,
            interval=interval,
        )
    )
    normalized["is_legacy_unrouted"] = bool((not route and not item.get("route_id")) or (inferred_cohort and not inferred_route and not route))
    return normalized


def _is_flat_manual_close(item: dict[str, Any]) -> bool:
    return (
        str(item.get("exit_reason") or "").lower() == "manual_close"
        and abs(float(item.get("realized_pnl_usdt", 0.0) or 0.0)) < 1e-9
    )


def _prepare_reviews(reviews: list[dict[str, Any]], config: StrategyOptimizerConfig) -> tuple[list[dict[str, Any]], dict[str, int]]:
    normalized = [_normalize_review(item) for item in reviews]
    stats = {
        "input_reviews": len(reviews),
        "legacy_unrouted_reviews": sum(1 for item in normalized if bool(item.get("is_legacy_unrouted"))),
        "flat_manual_closes": sum(1 for item in normalized if _is_flat_manual_close(item)),
    }
    filtered = normalized
    if config.ignore_legacy_unrouted:
        filtered = [item for item in filtered if not bool(item.get("is_legacy_unrouted"))]
    if config.exclude_flat_manual_closes:
        filtered = [item for item in filtered if not _is_flat_manual_close(item)]
    stats["usable_reviews"] = len(filtered)
    return filtered, stats


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _review_pnl(item: dict[str, Any]) -> float:
    return float(item.get("realized_pnl_usdt", 0.0) or 0.0)


def _review_fold_metrics(
    reviews: list[dict[str, Any]],
    *,
    fold: int,
    start_index: int,
    end_index: int,
) -> dict[str, Any]:
    pnls = [_review_pnl(item) for item in reviews]
    r_values = [
        float(item.get("realized_r_multiple") or 0.0)
        for item in reviews
        if item.get("realized_r_multiple") is not None
    ]
    expectancy = calculate_expectancy_stats(r_values)
    wins = sum(1 for pnl in pnls if pnl > 0.0)
    stop_losses = sum(1 for item in reviews if str(item.get("exit_reason") or "").lower() == "stop_loss")
    return {
        "fold": fold,
        "start_index": start_index,
        "end_index": end_index,
        "trade_count": len(reviews),
        "wins": wins,
        "losses": sum(1 for pnl in pnls if pnl < 0.0),
        "win_rate": round((wins / len(reviews)) * 100.0, 2) if reviews else 0.0,
        "profit_factor": round(calculate_profit_factor(pnls), 4),
        **expectancy,
        "net_pnl_usdt": round(sum(pnls), 8),
        "avg_r_multiple": round(
            _avg(
                [
                    float(item.get("realized_r_multiple") or 0.0)
                    for item in reviews
                    if item.get("realized_r_multiple") is not None
                ]
            ),
            4,
        ),
        "max_drawdown_pct": round(calculate_max_drawdown_pct(pnls), 4),
        "loss_streak": calculate_loss_streak(pnls),
        "stop_loss_ratio": round((stop_losses / len(reviews)) * 100.0, 2) if reviews else 0.0,
    }


def _review_chronological_folds(
    reviews: list[dict[str, Any]],
    *,
    target_folds: int = 4,
    min_trades_per_fold: int = 10,
) -> list[dict[str, Any]]:
    if not reviews:
        return []
    max_folds = len(reviews) // max(1, min_trades_per_fold)
    fold_count = min(max(1, target_folds), max_folds)
    if fold_count <= 1:
        return []
    fold_size = max(1, len(reviews) // fold_count)
    folds: list[dict[str, Any]] = []
    for index in range(fold_count):
        start = index * fold_size
        end = len(reviews) if index == fold_count - 1 else (index + 1) * fold_size
        fold_reviews = reviews[start:end]
        if len(fold_reviews) < min_trades_per_fold:
            continue
        folds.append(
            _review_fold_metrics(
                fold_reviews,
                fold=index + 1,
                start_index=start,
                end_index=end,
            )
        )
    return folds


def _review_robustness_report(
    reviews: list[dict[str, Any]],
    config: StrategyOptimizerConfig,
) -> dict[str, Any]:
    folds = _review_chronological_folds(reviews)
    target_pf = max(1.0, float(config.screening_min_profit_factor))
    required_positive_folds = int((len(folds) * 0.67) + 0.9999) if folds else 0
    positive_folds = sum(1 for item in folds if float(item.get("net_pnl_usdt") or 0.0) > 0.0)
    weak_pf_folds = sum(1 for item in folds if float(item.get("profit_factor") or 0.0) < 1.0)
    reasons: list[str] = []
    if len(reviews) < config.screening_min_trades:
        reasons.append(
            f"review-count-below-screening-floor:{len(reviews)}/{config.screening_min_trades}"
        )
    if len(folds) < 2:
        reasons.append("insufficient-review-folds")
    if folds and positive_folds < required_positive_folds:
        reasons.append(f"positive-review-fold-count-too-low:{positive_folds}/{required_positive_folds}")
    allowed_weak_folds = 1 if len(folds) >= 4 else 0
    if folds and weak_pf_folds > allowed_weak_folds:
        reasons.append(f"too-many-review-folds-below-1pf:{weak_pf_folds}/{len(folds)}")
    if folds and max(float(item.get("max_drawdown_pct") or 0.0) for item in folds) > config.max_drawdown_pct:
        reasons.append("review-fold-drawdown-above-policy")
    if folds and max(int(item.get("loss_streak") or 0) for item in folds) > config.max_loss_streak:
        reasons.append("review-fold-loss-streak-above-policy")
    return {
        "status": "passed" if not reasons else "failed",
        "passed": not reasons,
        "fold_count": len(folds),
        "target_profit_factor": round(target_pf, 4),
        "positive_fold_count": positive_folds,
        "required_positive_fold_count": required_positive_folds,
        "folds_below_one_pf": weak_pf_folds,
        "reasons": reasons,
        "folds": folds,
        "applied_principles": [
            "rolling-closed-review-validation",
            "reject-single-period-profit-factor",
            "stop-loss-and-drawdown-aware-promotion",
        ],
    }


def _build_lane_breakdown(reviews: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lanes: dict[str, dict[str, Any]] = {}
    for item in reviews:
        route_id = str(item.get("route_id") or "unrouted")
        lane = lanes.setdefault(
            route_id,
            {
                "asset_class": str(item.get("asset_class") or "unknown"),
                "strategy_profile": str(item.get("strategy_profile") or ""),
                "review_lane": str(item.get("review_lane") or ""),
                "count": 0,
                "wins": 0,
                "losses": 0,
                "total_realized_pnl_usdt": 0.0,
            },
        )
        pnl = float(item.get("realized_pnl_usdt", 0.0) or 0.0)
        lane["count"] += 1
        lane["total_realized_pnl_usdt"] = round(float(lane["total_realized_pnl_usdt"]) + pnl, 8)
        if pnl > 0:
            lane["wins"] += 1
        elif pnl < 0:
            lane["losses"] += 1
    return lanes


def _config_validation_spec(config: StrategyOptimizerConfig) -> RouteValidationSpec:
    return RouteValidationSpec(
        screening_min_win_rate=config.screening_min_win_rate,
        screening_min_profit_factor=config.screening_min_profit_factor,
        screening_min_expectancy_r=config.screening_min_expectancy_r,
        screening_min_payoff_ratio=config.screening_min_payoff_ratio,
        screening_min_trades=config.screening_min_trades,
        validation_min_win_rate=config.validation_min_win_rate,
        validation_min_profit_factor=config.validation_min_profit_factor,
        validation_min_expectancy_r=config.validation_min_expectancy_r,
        validation_min_payoff_ratio=config.validation_min_payoff_ratio,
        validation_min_simulated_trades=config.validation_min_simulated_trades,
        max_drawdown_pct=config.max_drawdown_pct,
        max_loss_streak=config.max_loss_streak,
        elite_enabled=True,
        elite_min_win_rate=config.elite_min_win_rate,
        elite_min_profit_factor=config.elite_min_profit_factor,
        elite_min_trades=config.validation_min_simulated_trades,
    )


def _cohort_reviews(reviews: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in reviews:
        cohort_id = str(item.get("cohort_id") or "uncohorted")
        grouped.setdefault(cohort_id, []).append(item)
    return grouped


def _cohort_convergence_report(
    reviews: list[dict[str, Any]],
    config: StrategyOptimizerConfig,
) -> list[dict[str, Any]]:
    grouped = _cohort_reviews(reviews)
    report: list[dict[str, Any]] = []
    default_spec = _config_validation_spec(config)
    for cohort_id, items in grouped.items():
        latest = items[-1]
        route_id = str(latest.get("route_id") or "unrouted")
        symbol = str(latest.get("symbol") or "")
        try:
            route = resolve_symbol_route(symbol) if symbol else None
        except ValueError:
            route = None
        spec = route.validation if route is not None else default_spec
        pnls = [float(item.get("realized_pnl_usdt", 0.0) or 0.0) for item in items]
        r_values = [
            float(item.get("realized_r_multiple") or 0.0)
            for item in items
            if item.get("realized_r_multiple") is not None
        ]
        expectancy = calculate_expectancy_stats(r_values)
        wins = sum(1 for pnl in pnls if pnl > 0.0)
        metrics = ConvergenceMetrics(
            trade_count=len(items),
            win_rate=((wins / len(items)) * 100.0) if items else 0.0,
            profit_factor=calculate_profit_factor(pnls),
            max_drawdown_pct=calculate_max_drawdown_pct(pnls),
            loss_streak=calculate_loss_streak(pnls),
            expectancy_r=expectancy["expectancy_r"],
            payoff_ratio=expectancy["payoff_ratio"],
        )
        evaluation = evaluate_convergence(metrics, spec)
        report.append(
            {
                "cohort_id": cohort_id,
                "route_id": route_id,
                "asset_class": str(latest.get("asset_class") or getattr(route, "asset_class", "unknown")),
                "strategy_profile": str(latest.get("strategy_profile") or ""),
                "review_lane": str(latest.get("review_lane") or ""),
                "metrics": metrics.to_dict(),
                **evaluation,
            }
        )
    targets = PayoffObjectiveTargets(
        min_trades=config.validation_min_simulated_trades,
        min_profit_factor=config.validation_min_profit_factor,
        min_expectancy_r=config.validation_min_expectancy_r,
        min_payoff_ratio=config.validation_min_payoff_ratio,
        min_win_rate=config.validation_min_win_rate,
        max_stop_loss_ratio=config.max_stop_loss_ratio,
    )
    report.sort(
        key=lambda item: (
            promotion_decision_rank(item.get("promotion_decision")),
            *payoff_objective_sort_key(
                item.get("metrics") or {},
                targets=targets,
                min_trades=config.validation_min_simulated_trades,
            ),
        ),
        reverse=True,
    )
    return report


def _build_strategy_payload(strategy: StrategyConfig, *, description_suffix: str) -> dict[str, Any]:
    return {
        "profile": f"{strategy.profile}-auto",
        "description": f"{strategy.description} {description_suffix}".strip(),
        "defaults": {
            "symbol": strategy.defaults.symbol,
            "market": strategy.defaults.market,
            "interval": strategy.defaults.interval,
            "limit": strategy.defaults.limit,
            "use_blave": strategy.defaults.use_blave,
            "render_chart": strategy.defaults.render_chart,
        },
        "risk": {
            "max_account_risk_pct": strategy.risk.max_account_risk_pct,
            "default_leverage": strategy.risk.default_leverage,
            "max_leverage": strategy.risk.max_leverage,
            "max_notional_pct": strategy.risk.max_notional_pct,
            "max_daily_trades": strategy.risk.max_daily_trades,
            "min_balance_usdt": strategy.risk.min_balance_usdt,
            "min_convergence": strategy.risk.min_convergence,
            "min_score_long": strategy.risk.min_score_long,
            "max_score_short": strategy.risk.max_score_short,
            "cooldown_hours": strategy.risk.cooldown_hours,
            "atr_stop_multiple": strategy.risk.atr_stop_multiple,
            "min_adx": strategy.risk.min_adx,
            "trailing_stop_enabled": strategy.risk.trailing_stop_enabled,
            "trailing_activation_r_multiple": strategy.risk.trailing_activation_r_multiple,
            "trailing_callback_pct": strategy.risk.trailing_callback_pct,
            "take_profit_r_multiples": list(strategy.risk.take_profit_r_multiples),
        },
        "signal": {
            "ema_fast": strategy.signal.ema_fast,
            "ema_slow": strategy.signal.ema_slow,
            "rsi_length": strategy.signal.rsi_length,
            "macd_fast": strategy.signal.macd_fast,
            "macd_slow": strategy.signal.macd_slow,
            "macd_signal": strategy.signal.macd_signal,
            "breakout_length": strategy.signal.breakout_length,
        },
        "execution": {
            "order_type": strategy.execution.order_type,
            "margin_type": strategy.execution.margin_type,
            "reduce_only_close": strategy.execution.reduce_only_close,
            "fee_bps": strategy.execution.fee_bps,
            "slippage_bps": strategy.execution.slippage_bps,
            "margin_notional_usdt": strategy.execution.margin_notional_usdt,
        },
        "challenge": {
            "enabled": strategy.challenge.enabled,
            "target_multiple": strategy.challenge.target_multiple,
            "max_drawdown_pct": strategy.challenge.max_drawdown_pct,
            "pause_on_target": strategy.challenge.pause_on_target,
            "pause_on_drawdown_breach": strategy.challenge.pause_on_drawdown_breach,
        },
    }


def _freeze_payload_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_payload_value(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_payload_value(item) for item in value)
    return value


def _flatten_payload(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_payload(value, path))
            continue
        flattened[path] = _freeze_payload_value(value)
    return flattened


def _mutation_scope(
    base_payload: dict[str, Any],
    tuned_payload: dict[str, Any],
) -> tuple[list[str], list[str]]:
    base_flat = _flatten_payload(base_payload)
    tuned_flat = _flatten_payload(tuned_payload)
    changed_paths = sorted(
        {
            *base_flat.keys(),
            *tuned_flat.keys(),
        }
        - {
            path
            for path in { *base_flat.keys(), *tuned_flat.keys() }
            if base_flat.get(path) == tuned_flat.get(path)
        }
    )
    unexpected_paths = [path for path in changed_paths if path not in TUNABLE_STRATEGY_PATHS]
    return changed_paths, unexpected_paths


def tune_strategy_from_reviews(
    strategy: StrategyConfig,
    reviews: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    tuned = _build_strategy_payload(
        strategy,
        description_suffix="Auto-tuned from closed-trade reviews.",
    )
    notes: list[str] = []
    wins = [item for item in reviews if float(item.get("realized_pnl_usdt", 0.0) or 0.0) > 0.0]
    losses = [item for item in reviews if float(item.get("realized_pnl_usdt", 0.0) or 0.0) < 0.0]
    stop_losses = [item for item in reviews if str(item.get("exit_reason", "")).lower() == "stop_loss"]
    take_profits = [item for item in reviews if str(item.get("exit_reason", "")).lower() == "take_profit"]
    win_rate = len(wins) / len(reviews) if reviews else 0.0
    avg_win = _avg([float(item.get("realized_pnl_pct", 0.0) or 0.0) for item in wins])
    avg_loss = _avg([abs(float(item.get("realized_pnl_pct", 0.0) or 0.0)) for item in losses])
    avg_r = _avg(
        [
            float(item.get("realized_r_multiple", 0.0) or 0.0)
            for item in reviews
            if item.get("realized_r_multiple") is not None
        ]
    )
    r_values = [
        float(item.get("realized_r_multiple", 0.0) or 0.0)
        for item in reviews
        if item.get("realized_r_multiple") is not None
    ]
    expectancy = calculate_expectancy_stats(r_values)
    stop_ratio = len(stop_losses) / len(reviews) if reviews else 0.0
    tp_ratio = len(take_profits) / len(reviews) if reviews else 0.0
    loss_streak = 0
    for item in reversed(reviews):
        if float(item.get("realized_pnl_usdt", 0.0) or 0.0) < 0.0:
            loss_streak += 1
        else:
            break

    if stop_ratio >= 0.45:
        tuned["risk"]["min_adx"] = min(35.0, round(float(tuned["risk"]["min_adx"]) + 2.0, 2))
        tuned["risk"]["min_score_long"] = min(90, int(tuned["risk"]["min_score_long"]) + 3)
        tuned["risk"]["max_score_short"] = max(10, int(tuned["risk"]["max_score_short"]) - 3)
        tuned["risk"]["cooldown_hours"] = min(
            12.0, round(float(tuned["risk"]["cooldown_hours"]) + 1.0, 2)
        )
        notes.append("High stop-loss ratio: tightened entry filters and added cooldown.")

    if avg_loss > avg_win and losses:
        tuned["risk"]["atr_stop_multiple"] = min(
            3.0, round(float(tuned["risk"]["atr_stop_multiple"]) + 0.1, 2)
        )
        tuned["risk"]["trailing_callback_pct"] = max(
            0.4, round(float(tuned["risk"]["trailing_callback_pct"]) - 0.05, 2)
        )
        notes.append(
            "Average loss exceeds average win: widened invalidation slightly and tightened trailing callback."
        )

    if win_rate >= 0.62 and tp_ratio >= 0.45:
        tp_levels = [float(item) for item in tuned["risk"]["take_profit_r_multiples"]]
        tp_levels[0] = min(2.2, round(tp_levels[0] + 0.1, 2))
        if len(tp_levels) > 1:
            tp_levels[1] = min(3.8, round(tp_levels[1] + 0.1, 2))
        tuned["risk"]["take_profit_r_multiples"] = tp_levels
        notes.append("Healthy win profile: let winners run slightly further.")

    if avg_r < 0.8 and wins:
        tuned["risk"]["trailing_callback_pct"] = max(
            0.45, round(float(tuned["risk"]["trailing_callback_pct"]) - 0.05, 2)
        )
        notes.append("Average R is modest: tighten trailing callback to protect realized edge.")

    if win_rate < 0.45:
        tuned["risk"]["min_convergence"] = min(
            0.9, round(float(tuned["risk"]["min_convergence"]) + 0.03, 3)
        )
        notes.append("Win rate is weak: raised minimum convergence requirement.")

    stats = {
        "review_count": len(reviews),
        "win_rate": round(win_rate, 4),
        "avg_win_pct": round(avg_win, 4),
        "avg_loss_pct": round(avg_loss, 4),
        "avg_r_multiple": round(avg_r, 4),
        **expectancy,
        "stop_loss_ratio": round(stop_ratio, 4),
        "take_profit_ratio": round(tp_ratio, 4),
        "loss_streak": loss_streak,
    }
    if not notes:
        notes.append("Recent closed-trade profile is stable; no material strategy tightening required.")
    return tuned, notes, stats


def run_strategy_optimizer(path: str | Path | None = None) -> dict[str, Any]:
    config = load_optimizer_config(path)
    base_strategy = load_strategy_config(config.base_strategy_config)
    raw_reviews = _balanced_recent_reviews(
        read_closed_trade_reviews(),
        per_route_limit=config.lookback_reviews_per_route,
        total_limit=config.lookback_reviews,
    )
    reviews, review_hygiene = _prepare_reviews(raw_reviews, config)
    OPTIMIZER_STATE_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = _utc_now()
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    summary: dict[str, Any] = {
        "generated_at": generated_at.isoformat(),
        "config_path": str(config.path),
        "base_strategy_config": str(config.base_strategy_config),
        "output_strategy_config": str(config.output_strategy_config),
        "review_count": len(reviews),
        "review_hygiene": review_hygiene,
        "lane_breakdown": _build_lane_breakdown(reviews),
        "convergence_report": _cohort_convergence_report(reviews, config),
        "review_robustness": _review_robustness_report(reviews, config),
    }
    if len(reviews) < config.min_closed_reviews:
        summary["status"] = "skipped"
        summary["reason"] = (
            f"Need at least {config.min_closed_reviews} closed-trade reviews before retuning; got {len(reviews)}."
        )
    else:
        tuned_payload, notes, stats = tune_strategy_from_reviews(base_strategy, reviews)
        guardrail_notes: list[str] = []
        if stats["win_rate"] * 100.0 < config.screening_min_win_rate:
            tuned_payload["risk"]["min_convergence"] = min(
                0.92,
                round(float(tuned_payload["risk"]["min_convergence"]) + 0.02, 3),
            )
            guardrail_notes.append(
                f"Win rate {stats['win_rate']:.1%} below screening floor {config.screening_min_win_rate:.1f}%: tightened convergence."
            )
        if stats["avg_r_multiple"] < config.min_avg_r_multiple:
            tuned_payload["risk"]["trailing_callback_pct"] = max(
                0.45,
                round(float(tuned_payload["risk"]["trailing_callback_pct"]) - 0.05, 2),
            )
            guardrail_notes.append(
                f"Average R {stats['avg_r_multiple']:.2f} below policy floor {config.min_avg_r_multiple:.2f}: tightened trailing."
            )
        if stats["stop_loss_ratio"] > config.max_stop_loss_ratio:
            tuned_payload["risk"]["min_adx"] = min(
                35.0,
                round(float(tuned_payload["risk"]["min_adx"]) + 1.0, 2),
            )
            guardrail_notes.append(
                f"Stop-loss ratio {stats['stop_loss_ratio']:.1%} above policy ceiling {config.max_stop_loss_ratio:.1%}: raised ADX floor."
            )
        if stats["loss_streak"] >= config.max_loss_streak:
            tuned_payload["risk"]["cooldown_hours"] = min(
                12.0,
                round(float(tuned_payload["risk"]["cooldown_hours"]) + 2.0, 2),
            )
            guardrail_notes.append(
                f"Loss streak {stats['loss_streak']} reached policy ceiling {config.max_loss_streak}: extended cooldown."
            )
        notes.extend(guardrail_notes)
        base_payload = _build_strategy_payload(base_strategy, description_suffix="Auto-tuned from closed-trade reviews.")
        changed_paths, unexpected_paths = _mutation_scope(base_payload, tuned_payload)
        if unexpected_paths:
            raise ValueError(
                "Strategy optimizer attempted to change protected fields: "
                + ", ".join(unexpected_paths)
            )
        overall_metrics = ConvergenceMetrics(
            trade_count=len(reviews),
            win_rate=stats["win_rate"] * 100.0,
            profit_factor=calculate_profit_factor(
                [float(item.get("realized_pnl_usdt", 0.0) or 0.0) for item in reviews]
            ),
            max_drawdown_pct=calculate_max_drawdown_pct(
                [float(item.get("realized_pnl_usdt", 0.0) or 0.0) for item in reviews]
            ),
            loss_streak=stats["loss_streak"],
            expectancy_r=stats["expectancy_r"],
            payoff_ratio=stats["payoff_ratio"],
        )
        convergence_eval = evaluate_convergence(overall_metrics, _config_validation_spec(config))
        summary["status"] = "ok"
        summary["notes"] = notes
        summary["stats"] = stats
        summary["screening_status"] = convergence_eval["screening_status"]
        summary["validation_status"] = convergence_eval["validation_status"]
        summary["elite_status"] = convergence_eval["elite_status"]
        summary["promotion_decision"] = convergence_eval["promotion_decision"]
        summary["mutation_scope"] = {
            "allowed_paths": sorted(TUNABLE_STRATEGY_PATHS),
            "changed_paths": changed_paths,
            "unexpected_paths": unexpected_paths,
        }
        if config.auto_apply:
            config.output_strategy_config.write_text(
                yaml.safe_dump(tuned_payload, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            summary["applied"] = True
        else:
            summary["applied"] = False
        summary["tuned_profile"] = tuned_payload["profile"]
    report_path = OPTIMIZER_STATE_DIR / f"{stamp}-strategy-optimizer.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["report_path"] = str(report_path)
    return summary
