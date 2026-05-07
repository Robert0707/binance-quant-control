"""Hard risk management guard for live order execution.

Every live order must pass through ``check_order_allowed()`` before
submission.  The guard is intentionally strict and deterministic: it
will reject any order that violates the configured risk envelope rather
than silently adjusting parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class RiskCheckResult:
    allowed: bool
    violations: list[str]
    warnings: list[str]

    @property
    def summary(self) -> str:
        if self.allowed:
            return "PASS"
        return "BLOCKED: " + "; ".join(self.violations)


def check_order_allowed(
    *,
    side: str,
    margin_notional_usdt: float,
    leverage: int,
    account_balance_usdt: float,
    account_risk_pct: float = 0.0,
    analysis_convergence: float,
    analysis_score: int,
    adx_value: float | None = None,
    daily_trade_count: int,
    consecutive_losses: int,
    last_loss_at: datetime | None,
    # Risk config
    max_account_risk_pct: float = 0.01,
    max_leverage: int = 2,
    max_notional_pct: float = 0.5,
    max_daily_trades: int = 3,
    min_balance_usdt: float = 2.0,
    min_convergence: float = 0.6,
    min_score_long: int = 60,
    max_score_short: int = 40,
    cooldown_hours: float = 4.0,
    max_consecutive_losses: int = 2,
    min_adx: float = 20.0,
    liquidation_buffer_pct: float = 0.0,
    min_liquidation_buffer_pct: float = 0.0,
) -> RiskCheckResult:
    """Run all risk checks and return a composite result.

    The function never raises; instead it collects violations and returns
    a ``RiskCheckResult`` with ``allowed=False`` if any check fails.
    """

    violations: list[str] = []
    warnings: list[str] = []

    maturity_mode = max_leverage >= 125

    # --- leverage cap ---
    if leverage > max_leverage:
        violations.append(
            f"Leverage {leverage}x exceeds hard cap {max_leverage}x."
        )

    # --- account balance floor ---
    if account_balance_usdt < min_balance_usdt:
        violations.append(
            f"Account balance {account_balance_usdt:.2f} USDT "
            f"is below the minimum {min_balance_usdt:.2f} USDT."
        )

    # --- notional vs account size ---
    side_upper = side.upper()
    if side_upper not in {"BUY", "SELL"}:
        violations.append(f"Unsupported execution side {side_upper}.")

    gross_notional = margin_notional_usdt * leverage
    max_allowed_notional = account_balance_usdt * max_notional_pct * max_leverage
    if not maturity_mode and gross_notional > max_allowed_notional:
        violations.append(
            f"Gross notional {gross_notional:.2f} USDT exceeds "
            f"the maximum allowed {max_allowed_notional:.2f} USDT "
            f"({max_notional_pct*100:.0f}% of balance × {max_leverage}x)."
        )
    elif maturity_mode:
        warnings.append(
            "Maturity mode: notional is governed by stop-loss risk and liquidation buffer."
        )

    # --- stop-loss risk cap ---
    if account_risk_pct > max_account_risk_pct:
        violations.append(
            f"Planned stop-loss risk {account_risk_pct*100:.2f}% exceeds "
            f"the maximum account risk {max_account_risk_pct*100:.2f}%."
        )

    if min_liquidation_buffer_pct > 0 and liquidation_buffer_pct < min_liquidation_buffer_pct:
        violations.append(
            f"Liquidation buffer {liquidation_buffer_pct*100:.2f}% is below "
            f"the required {min_liquidation_buffer_pct*100:.2f}%."
        )

    # --- daily trade count ---
    if daily_trade_count >= max_daily_trades:
        violations.append(
            f"Daily trade count {daily_trade_count} has reached "
            f"the maximum of {max_daily_trades}."
        )

    # --- signal convergence ---
    if analysis_convergence < min_convergence:
        violations.append(
            f"Signal convergence {analysis_convergence:.3f} is below "
            f"the minimum threshold {min_convergence:.3f}."
        )

    # --- score direction alignment ---
    if side_upper == "BUY" and analysis_score < min_score_long:
        violations.append(
            f"Analysis score {analysis_score} is too low for a BUY "
            f"(minimum {min_score_long})."
        )
    elif side_upper == "SELL" and analysis_score > max_score_short:
        violations.append(
            f"Analysis score {analysis_score} is too high for a SELL "
            f"(maximum {max_score_short})."
        )

    # --- trend strength guard ---
    if adx_value is not None and adx_value < min_adx:
        violations.append(
            f"ADX {adx_value:.2f} is below the minimum trend threshold {min_adx:.2f}."
        )

    # --- consecutive loss cooldown ---
    if consecutive_losses >= max_consecutive_losses:
        if last_loss_at is not None:
            now = datetime.now(timezone.utc)
            hours_since = (now - last_loss_at).total_seconds() / 3600
            if hours_since < cooldown_hours:
                violations.append(
                    f"Cooldown active: {consecutive_losses} consecutive losses, "
                    f"last loss was {hours_since:.1f}h ago "
                    f"(cooldown requires {cooldown_hours}h)."
                )
            else:
                warnings.append(
                    f"Cooldown expired ({hours_since:.1f}h since last loss). "
                    f"Consecutive loss counter will reset on next win."
                )
        else:
            violations.append(
                f"{consecutive_losses} consecutive losses recorded "
                f"but no timestamp available; blocking as precaution."
            )

    # --- micro account warning ---
    if account_balance_usdt < 10.0:
        warnings.append(
            f"Micro account ({account_balance_usdt:.2f} USDT). "
            f"Fees and slippage will have outsized impact."
        )

    return RiskCheckResult(
        allowed=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )
