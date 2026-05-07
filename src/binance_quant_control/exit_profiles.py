from __future__ import annotations

from collections.abc import Sequence

EXIT_PROFILES = {"balanced", "payoff_runner", "asymmetric_payoff", "capital_preservation"}


def normalize_exit_profile(value: str | None) -> str:
    normalized = str(value or "balanced").strip().lower().replace("-", "_")
    return normalized if normalized in EXIT_PROFILES else "balanced"


def staged_take_profit_weights(
    parts: int,
    *,
    exit_profile: str,
    trailing_stop_enabled: bool,
    confidence: float,
    strategy_family: str = "",
) -> list[float]:
    if parts <= 0:
        return []
    if parts == 1:
        base = [1.0]
    else:
        base = _base_target_weights(
            parts,
            exit_profile=normalize_exit_profile(exit_profile),
            confidence=float(confidence),
            strategy_family=strategy_family,
        )
    selected = _fit_parts(base, parts)
    runner_weight = _runner_weight(
        exit_profile=normalize_exit_profile(exit_profile),
        trailing_stop_enabled=trailing_stop_enabled,
        confidence=float(confidence),
        strategy_family=strategy_family,
    )
    return [round(item * (1.0 - runner_weight), 10) for item in selected]


def runner_stop_after_target(
    *,
    side: str,
    current_stop: float,
    entry_price: float,
    close_price: float,
    trailing_callback_pct: float,
    hit_count: int,
    exit_profile: str,
    initial_risk_distance: float | None = None,
) -> float:
    profile = normalize_exit_profile(exit_profile)
    risk_distance = max(float(initial_risk_distance or 0.0), abs(float(entry_price) - float(current_stop)))
    callback_pct = max(float(trailing_callback_pct), 0.0) / 100.0
    if profile == "payoff_runner":
        return _payoff_runner_stop(
            side=side,
            current_stop=float(current_stop),
            entry_price=float(entry_price),
            close_price=float(close_price),
            callback_pct=callback_pct,
            hit_count=int(hit_count),
            risk_distance=risk_distance,
        )
    if profile == "asymmetric_payoff":
        return _asymmetric_payoff_stop(
            side=side,
            current_stop=float(current_stop),
            entry_price=float(entry_price),
            close_price=float(close_price),
            callback_pct=callback_pct,
            hit_count=int(hit_count),
            risk_distance=risk_distance,
        )
    if profile == "capital_preservation":
        return _balanced_stop(
            side=side,
            current_stop=float(current_stop),
            entry_price=float(entry_price),
            close_price=float(close_price),
            callback_pct=callback_pct,
            hit_count=int(hit_count),
            second_target_lock_r=0.15,
            risk_distance=risk_distance,
        )
    return _balanced_stop(
        side=side,
        current_stop=float(current_stop),
        entry_price=float(entry_price),
        close_price=float(close_price),
        callback_pct=callback_pct,
        hit_count=int(hit_count),
        second_target_lock_r=0.0,
        risk_distance=risk_distance,
    )


def _base_target_weights(
    parts: int,
    *,
    exit_profile: str,
    confidence: float,
    strategy_family: str,
) -> list[float]:
    family = str(strategy_family or "").strip().lower()
    if exit_profile == "payoff_runner":
        if family == "mean_reversion":
            return [0.34, 0.36] if parts == 2 else [0.30, 0.34, 0.36]
        return [0.28, 0.42] if parts == 2 else [0.24, 0.34, 0.42]
    if exit_profile == "asymmetric_payoff":
        if family == "mean_reversion":
            return [0.18, 0.30] if parts == 2 else [0.16, 0.26, 0.58]
        return [0.14, 0.30] if parts == 2 else [0.12, 0.24, 0.64]
    if exit_profile == "capital_preservation":
        if family == "mean_reversion":
            return [0.62, 0.38] if parts == 2 else [0.58, 0.28, 0.14]
        return [0.40, 0.60] if parts == 2 else [0.38, 0.34, 0.28]
    if family == "mean_reversion":
        return [0.55, 0.45] if parts == 2 else [0.55, 0.30, 0.15]
    if parts == 2:
        return [0.25, 0.75] if confidence >= 0.86 else [0.30, 0.70]
    return [0.25, 0.35, 0.40] if confidence >= 0.86 else [0.30, 0.35, 0.35]


def _fit_parts(base: Sequence[float], parts: int) -> list[float]:
    selected = list(base[:parts])
    if parts > len(selected):
        selected.extend([0.0] * (parts - len(selected)))
        selected[-1] = max(0.0, 1.0 - sum(selected[:-1]))
    total = sum(selected)
    if total <= 0.0:
        return [1.0 / parts for _ in range(parts)]
    return [item / total for item in selected]


def _runner_weight(
    *,
    exit_profile: str,
    trailing_stop_enabled: bool,
    confidence: float,
    strategy_family: str,
) -> float:
    if not trailing_stop_enabled:
        return 0.0
    family = str(strategy_family or "").strip().lower()
    if exit_profile == "payoff_runner":
        if family == "mean_reversion":
            return 0.18 if confidence >= 0.82 else 0.14
        return 0.28 if confidence >= 0.86 else 0.22
    if exit_profile == "asymmetric_payoff":
        if family == "mean_reversion":
            return 0.34 if confidence >= 0.82 else 0.28
        return 0.42 if confidence >= 0.86 else 0.36
    if exit_profile == "capital_preservation":
        return 0.04 if family == "mean_reversion" else 0.10
    return 0.05 if family == "mean_reversion" else 0.20 if confidence >= 0.86 else 0.15


def _balanced_stop(
    *,
    side: str,
    current_stop: float,
    entry_price: float,
    close_price: float,
    callback_pct: float,
    hit_count: int,
    second_target_lock_r: float,
    risk_distance: float,
) -> float:
    if side == "BUY":
        trailing_stop = close_price * (1.0 - callback_pct)
        stop = max(current_stop, entry_price, trailing_stop)
        if hit_count >= 2:
            stop = max(stop, entry_price + (risk_distance * second_target_lock_r), entry_price * 1.0005)
        return stop
    trailing_stop = close_price * (1.0 + callback_pct)
    stop = min(current_stop, entry_price, trailing_stop)
    if hit_count >= 2:
        stop = min(stop, entry_price - (risk_distance * second_target_lock_r), entry_price * 0.9995)
    return stop


def _payoff_runner_stop(
    *,
    side: str,
    current_stop: float,
    entry_price: float,
    close_price: float,
    callback_pct: float,
    hit_count: int,
    risk_distance: float,
) -> float:
    risk = max(float(risk_distance), 0.0)
    if side == "BUY":
        soft_floor = entry_price - (risk * 0.12)
        lock_after_second = entry_price + (risk * 0.30)
        lock_after_third = entry_price + (risk * 0.85)
        trailing_stop = close_price * (1.0 - callback_pct)
        stop = max(current_stop, soft_floor)
        if hit_count >= 2:
            stop = max(stop, lock_after_second)
        if hit_count >= 3:
            stop = max(stop, lock_after_third)
        return max(stop, trailing_stop) if hit_count >= 2 else stop
    soft_floor = entry_price + (risk * 0.12)
    lock_after_second = entry_price - (risk * 0.30)
    lock_after_third = entry_price - (risk * 0.85)
    trailing_stop = close_price * (1.0 + callback_pct)
    stop = min(current_stop, soft_floor)
    if hit_count >= 2:
        stop = min(stop, lock_after_second)
    if hit_count >= 3:
        stop = min(stop, lock_after_third)
    return min(stop, trailing_stop) if hit_count >= 2 else stop


def _asymmetric_payoff_stop(
    *,
    side: str,
    current_stop: float,
    entry_price: float,
    close_price: float,
    callback_pct: float,
    hit_count: int,
    risk_distance: float,
) -> float:
    risk = max(float(risk_distance), 0.0)
    if side == "BUY":
        soft_floor = entry_price - (risk * 0.35)
        lock_after_second = entry_price + (risk * 0.12)
        lock_after_third = entry_price + (risk * 0.70)
        trailing_stop = close_price * (1.0 - max(callback_pct, 0.002))
        stop = max(current_stop, soft_floor)
        if hit_count >= 2:
            stop = max(stop, lock_after_second)
        if hit_count >= 3:
            stop = max(stop, lock_after_third, trailing_stop)
        return stop
    soft_floor = entry_price + (risk * 0.35)
    lock_after_second = entry_price - (risk * 0.12)
    lock_after_third = entry_price - (risk * 0.70)
    trailing_stop = close_price * (1.0 + max(callback_pct, 0.002))
    stop = min(current_stop, soft_floor)
    if hit_count >= 2:
        stop = min(stop, lock_after_second)
    if hit_count >= 3:
        stop = min(stop, lock_after_third, trailing_stop)
    return stop
