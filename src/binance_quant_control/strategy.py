from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import CONFIG_DIR
from .validation import StrategyConfigModel, format_validation_error


@dataclass(frozen=True, slots=True)
class StrategyDefaults:
    symbol: str = "BTCUSDT"
    market: str = "futures"
    interval: str = "1h"
    limit: int = 500
    use_blave: bool = False
    render_chart: bool = False


@dataclass(frozen=True, slots=True)
class StrategyRisk:
    max_account_risk_pct: float = 0.01
    default_leverage: int = 2
    max_leverage: int = 2
    max_notional_pct: float = 0.5
    max_daily_trades: int = 3
    min_balance_usdt: float = 2.0
    min_convergence: float = 0.6
    min_score_long: int = 60
    max_score_short: int = 40
    cooldown_hours: float = 4.0
    atr_stop_multiple: float = 1.8
    min_adx: float = 20.0
    trailing_stop_enabled: bool = False
    trailing_activation_r_multiple: float = 1.0
    trailing_callback_pct: float = 0.8
    take_profit_r_multiples: tuple[float, ...] = (1.5, 2.5, 4.0)
    exit_profile: str = "balanced"
    time_limit_bars: int = 0


@dataclass(frozen=True, slots=True)
class StrategySignal:
    ema_fast: int = 21
    ema_slow: int = 55
    rsi_length: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    breakout_length: int = 20


@dataclass(frozen=True, slots=True)
class StrategyExecution:
    order_type: str = "MARKET"
    margin_type: str = "ISOLATED"
    reduce_only_close: bool = True
    fee_bps: float = 4.0
    slippage_bps: float = 2.0
    margin_notional_usdt: float | None = None


@dataclass(frozen=True, slots=True)
class StrategyChallenge:
    enabled: bool = False
    target_multiple: float = 2.0
    max_drawdown_pct: float = 20.0
    pause_on_target: bool = True
    pause_on_drawdown_breach: bool = True


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    profile: str
    description: str
    defaults: StrategyDefaults
    risk: StrategyRisk
    signal: StrategySignal
    execution: StrategyExecution
    challenge: StrategyChallenge
    path: Path

    @property
    def primary_tp_multiple(self) -> float:
        if self.risk.take_profit_r_multiples:
            return float(self.risk.take_profit_r_multiples[0])
        return 1.0


def resolve_strategy_path(path: str | Path | None) -> Path:
    if path is None or str(path).strip() == "":
        return (CONFIG_DIR / "strategy.example.yaml").resolve()
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (CONFIG_DIR / candidate).resolve()


def load_strategy_config(path: str | Path | None = None) -> StrategyConfig:
    config_path = resolve_strategy_path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Strategy config {config_path} must be a mapping.")
    try:
        validated = StrategyConfigModel.model_validate(raw)
    except Exception as exc:
        if exc.__class__.__name__ == "ValidationError":
            raise ValueError(format_validation_error(f"Strategy config {config_path}", exc)) from exc
        raise

    defaults = StrategyDefaults(
        symbol=validated.defaults.symbol,
        market=validated.defaults.market,
        interval=validated.defaults.interval,
        limit=validated.defaults.limit,
        use_blave=validated.defaults.use_blave,
        render_chart=validated.defaults.render_chart,
    )
    risk = StrategyRisk(
        max_account_risk_pct=validated.risk.max_account_risk_pct,
        default_leverage=validated.risk.default_leverage,
        max_leverage=validated.risk.max_leverage,
        max_notional_pct=validated.risk.max_notional_pct,
        max_daily_trades=validated.risk.max_daily_trades,
        min_balance_usdt=validated.risk.min_balance_usdt,
        min_convergence=validated.risk.min_convergence,
        min_score_long=validated.risk.min_score_long,
        max_score_short=validated.risk.max_score_short,
        cooldown_hours=validated.risk.cooldown_hours,
        atr_stop_multiple=validated.risk.atr_stop_multiple,
        min_adx=validated.risk.min_adx,
        trailing_stop_enabled=validated.risk.trailing_stop_enabled,
        trailing_activation_r_multiple=validated.risk.trailing_activation_r_multiple,
        trailing_callback_pct=validated.risk.trailing_callback_pct,
        take_profit_r_multiples=validated.risk.take_profit_r_multiples,
        exit_profile=validated.risk.exit_profile,
        time_limit_bars=validated.risk.time_limit_bars,
    )
    signal = StrategySignal(
        ema_fast=validated.signal.ema_fast,
        ema_slow=validated.signal.ema_slow,
        rsi_length=validated.signal.rsi_length,
        macd_fast=validated.signal.macd_fast,
        macd_slow=validated.signal.macd_slow,
        macd_signal=validated.signal.macd_signal,
        breakout_length=validated.signal.breakout_length,
    )
    execution = StrategyExecution(
        order_type=validated.execution.order_type,
        margin_type=validated.execution.margin_type,
        reduce_only_close=validated.execution.reduce_only_close,
        fee_bps=validated.execution.fee_bps,
        slippage_bps=validated.execution.slippage_bps,
        margin_notional_usdt=validated.execution.margin_notional_usdt,
    )
    challenge = StrategyChallenge(
        enabled=validated.challenge.enabled,
        target_multiple=validated.challenge.target_multiple,
        max_drawdown_pct=validated.challenge.max_drawdown_pct,
        pause_on_target=validated.challenge.pause_on_target,
        pause_on_drawdown_breach=validated.challenge.pause_on_drawdown_breach,
    )
    return StrategyConfig(
        profile=validated.profile,
        description=validated.description.strip(),
        defaults=defaults,
        risk=risk,
        signal=signal,
        execution=execution,
        challenge=challenge,
        path=config_path,
    )
