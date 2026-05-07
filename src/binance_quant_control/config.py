from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .validation import SettingsModel, format_validation_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
REPORTS_DIR = PROJECT_ROOT / "reports"
STATE_DIR = PROJECT_ROOT / "state"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
ENV_PATH = PROJECT_ROOT / ".env"
OPENCLAW_RUNTIME = Path("/home/robert/.openclaw/runtime/binance-quant")
TASK_SPEC_DIR = OPENCLAW_RUNTIME / "task-specs"
RUN_DIR = OPENCLAW_RUNTIME / "runs"

load_dotenv(ENV_PATH)


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    use_testnet: bool
    live_trading_enabled: bool
    testnet_trading_enabled: bool
    recv_window_ms: int
    default_symbol: str
    default_market: str
    binance_api_key: str
    binance_secret_key: str
    binance_testnet_api_key: str
    binance_testnet_secret_key: str
    blave_api_key: str
    blave_secret_key: str
    whale_alert_api_key: str
    # Live trading risk parameters
    max_leverage: int
    max_notional_pct: float
    max_daily_trades: int
    min_balance_usdt: float
    min_convergence: float
    cooldown_hours: float

    @property
    def active_binance_api_key(self) -> str:
        if self.use_testnet and self.binance_testnet_api_key:
            return self.binance_testnet_api_key
        return self.binance_api_key

    @property
    def active_binance_secret_key(self) -> str:
        if self.use_testnet and self.binance_testnet_secret_key:
            return self.binance_testnet_secret_key
        return self.binance_secret_key

    @property
    def has_binance_credentials(self) -> bool:
        return bool(self.active_binance_api_key and self.active_binance_secret_key)

    @property
    def has_blave_credentials(self) -> bool:
        return bool(self.blave_api_key and self.blave_secret_key)

    @property
    def has_whale_alert_credentials(self) -> bool:
        return bool(self.whale_alert_api_key)


def load_settings() -> Settings:
    raw = {
        "use_testnet": env_bool("BINANCE_USE_TESTNET", True),
        "live_trading_enabled": env_bool("BINANCE_LIVE_TRADING_ENABLED", False),
        "testnet_trading_enabled": env_bool("BINANCE_TESTNET_TRADING_ENABLED", False),
        "recv_window_ms": int(os.getenv("BINANCE_RECV_WINDOW_MS", "5000")),
        "default_symbol": os.getenv("BINANCE_DEFAULT_SYMBOL", "BTCUSDT"),
        "default_market": os.getenv("BINANCE_DEFAULT_MARKET", "futures"),
        "binance_api_key": os.getenv("BINANCE_API_KEY", ""),
        "binance_secret_key": os.getenv("BINANCE_SECRET_KEY", ""),
        "binance_testnet_api_key": os.getenv("BINANCE_TESTNET_API_KEY", ""),
        "binance_testnet_secret_key": os.getenv("BINANCE_TESTNET_SECRET_KEY", ""),
        "blave_api_key": os.getenv("blave_api_key", ""),
        "blave_secret_key": os.getenv("blave_secret_key", ""),
        "whale_alert_api_key": os.getenv("WHALE_ALERT_API_KEY", ""),
        "max_leverage": int(os.getenv("BINANCE_MAX_LEVERAGE", "2")),
        "max_notional_pct": float(os.getenv("BINANCE_MAX_NOTIONAL_PCT", "0.5")),
        "max_daily_trades": int(os.getenv("BINANCE_MAX_DAILY_TRADES", "3")),
        "min_balance_usdt": float(os.getenv("BINANCE_MIN_BALANCE_USDT", "2.0")),
        "min_convergence": float(os.getenv("BINANCE_MIN_CONVERGENCE", "0.6")),
        "cooldown_hours": float(os.getenv("BINANCE_COOLDOWN_HOURS", "4.0")),
    }
    try:
        validated = SettingsModel.model_validate(raw)
    except Exception as exc:
        if exc.__class__.__name__ == "ValidationError":
            raise ValueError(format_validation_error("Settings", exc)) from exc
        raise
    return Settings(
        use_testnet=validated.use_testnet,
        live_trading_enabled=validated.live_trading_enabled,
        testnet_trading_enabled=validated.testnet_trading_enabled,
        recv_window_ms=validated.recv_window_ms,
        default_symbol=validated.default_symbol,
        default_market=validated.default_market,
        binance_api_key=validated.binance_api_key,
        binance_secret_key=validated.binance_secret_key,
        binance_testnet_api_key=validated.binance_testnet_api_key,
        binance_testnet_secret_key=validated.binance_testnet_secret_key,
        blave_api_key=validated.blave_api_key,
        blave_secret_key=validated.blave_secret_key,
        whale_alert_api_key=validated.whale_alert_api_key,
        max_leverage=validated.max_leverage,
        max_notional_pct=validated.max_notional_pct,
        max_daily_trades=validated.max_daily_trades,
        min_balance_usdt=validated.min_balance_usdt,
        min_convergence=validated.min_convergence,
        cooldown_hours=validated.cooldown_hours,
    )


def ensure_runtime_dirs() -> None:
    for path in (REPORTS_DIR, STATE_DIR, EXAMPLES_DIR, TASK_SPEC_DIR, RUN_DIR):
        path.mkdir(parents=True, exist_ok=True)
