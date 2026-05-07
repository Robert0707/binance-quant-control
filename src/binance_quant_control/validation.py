from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class SettingsModel(_FrozenModel):
    use_testnet: bool = True
    live_trading_enabled: bool = False
    testnet_trading_enabled: bool = False
    recv_window_ms: int = Field(default=5000, ge=1000, le=60000)
    default_symbol: str = "BTCUSDT"
    default_market: Literal["spot", "futures"] = "futures"
    binance_api_key: str = ""
    binance_secret_key: str = ""
    binance_testnet_api_key: str = ""
    binance_testnet_secret_key: str = ""
    blave_api_key: str = ""
    blave_secret_key: str = ""
    whale_alert_api_key: str = ""
    max_leverage: int = Field(default=2, ge=1, le=125)
    max_notional_pct: float = Field(default=0.5, gt=0.0, le=1.0)
    max_daily_trades: int = Field(default=3, ge=1, le=100)
    min_balance_usdt: float = Field(default=2.0, ge=0.0)
    min_convergence: float = Field(default=0.6, ge=0.0, le=1.0)
    cooldown_hours: float = Field(default=4.0, ge=0.0, le=168.0)

    @field_validator("default_symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("default_symbol must not be empty")
        return normalized


class StrategyDefaultsModel(_FrozenModel):
    symbol: str = "BTCUSDT"
    market: Literal["spot", "futures"] = "futures"
    interval: str = "1h"
    limit: int = Field(default=500, ge=50, le=5000)
    use_blave: bool = False
    render_chart: bool = False

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("symbol must not be empty")
        return normalized


class StrategyRiskModel(_FrozenModel):
    max_account_risk_pct: float = Field(default=0.01, gt=0.0, le=0.25)
    default_leverage: int = Field(default=2, ge=1, le=125)
    max_leverage: int = Field(default=2, ge=1, le=125)
    max_notional_pct: float = Field(default=0.5, gt=0.0, le=1.0)
    max_daily_trades: int = Field(default=3, ge=1, le=100)
    min_balance_usdt: float = Field(default=2.0, ge=0.0)
    min_convergence: float = Field(default=0.6, ge=0.0, le=1.0)
    min_score_long: int = Field(default=60, ge=0, le=100)
    max_score_short: int = Field(default=40, ge=0, le=100)
    cooldown_hours: float = Field(default=4.0, ge=0.0, le=168.0)
    atr_stop_multiple: float = Field(default=1.8, gt=0.0, le=10.0)
    min_adx: float = Field(default=20.0, ge=0.0, le=100.0)
    trailing_stop_enabled: bool = False
    trailing_activation_r_multiple: float = Field(default=1.0, gt=0.0, le=10.0)
    trailing_callback_pct: float = Field(default=0.8, gt=0.0, le=10.0)
    take_profit_r_multiples: tuple[float, ...] = (1.5, 2.5, 4.0)
    exit_profile: Literal["balanced", "payoff_runner", "asymmetric_payoff", "capital_preservation"] = "balanced"
    time_limit_bars: int = Field(default=0, ge=0, le=1000)

    @field_validator("take_profit_r_multiples", mode="before")
    @classmethod
    def _normalize_tp_levels(cls, value: Any) -> tuple[float, ...]:
        if value is None:
            return (1.5, 2.5, 4.0)
        if not isinstance(value, (list, tuple)):
            raise ValueError("take_profit_r_multiples must be a list or tuple")
        levels = tuple(float(item) for item in value)
        if not levels:
            raise ValueError("take_profit_r_multiples must include at least one value")
        if any(item <= 0 for item in levels):
            raise ValueError("take_profit_r_multiples must all be positive")
        return levels

    @model_validator(mode="after")
    def _check_consistency(self) -> "StrategyRiskModel":
        if self.default_leverage > self.max_leverage:
            raise ValueError("default_leverage must be <= max_leverage")
        if self.min_score_long <= self.max_score_short:
            raise ValueError("min_score_long must be greater than max_score_short")
        return self


class StrategySignalModel(_FrozenModel):
    ema_fast: int = Field(default=21, ge=2, le=500)
    ema_slow: int = Field(default=55, ge=2, le=1000)
    rsi_length: int = Field(default=14, ge=2, le=200)
    macd_fast: int = Field(default=12, ge=2, le=200)
    macd_slow: int = Field(default=26, ge=2, le=400)
    macd_signal: int = Field(default=9, ge=2, le=200)
    breakout_length: int = Field(default=20, ge=2, le=500)

    @model_validator(mode="after")
    def _check_consistency(self) -> "StrategySignalModel":
        if self.ema_fast >= self.ema_slow:
            raise ValueError("ema_fast must be smaller than ema_slow")
        if self.macd_fast >= self.macd_slow:
            raise ValueError("macd_fast must be smaller than macd_slow")
        return self


class StrategyExecutionModel(_FrozenModel):
    order_type: str = "MARKET"
    margin_type: Literal["ISOLATED", "CROSSED"] = "ISOLATED"
    reduce_only_close: bool = True
    fee_bps: float = Field(default=4.0, ge=0.0, le=100.0)
    slippage_bps: float = Field(default=2.0, ge=0.0, le=100.0)
    margin_notional_usdt: float | None = Field(default=None, gt=0.0)

    @field_validator("order_type")
    @classmethod
    def _normalize_order_type(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("order_type must not be empty")
        return normalized


class StrategyChallengeModel(_FrozenModel):
    enabled: bool = False
    target_multiple: float = Field(default=2.0, gt=0.0, le=100.0)
    max_drawdown_pct: float = Field(default=20.0, ge=0.0, le=100.0)
    pause_on_target: bool = True
    pause_on_drawdown_breach: bool = True


class StrategyConfigModel(_FrozenModel):
    profile: str = "custom"
    description: str = ""
    defaults: StrategyDefaultsModel = Field(default_factory=StrategyDefaultsModel)
    risk: StrategyRiskModel = Field(default_factory=StrategyRiskModel)
    signal: StrategySignalModel = Field(default_factory=StrategySignalModel)
    execution: StrategyExecutionModel = Field(default_factory=StrategyExecutionModel)
    challenge: StrategyChallengeModel = Field(default_factory=StrategyChallengeModel)

    @field_validator("profile")
    @classmethod
    def _normalize_profile(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("profile must not be empty")
        return normalized


def format_validation_error(prefix: str, error: ValidationError) -> str:
    details = "; ".join(
        f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}"
        for issue in error.errors()
    )
    return f"{prefix} validation failed: {details}"
