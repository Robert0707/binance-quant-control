"""Binance quant analysis control plane for OpenClaw."""

from .signals import SignalResult, TradeDecision, combine_signals, decide_trade_action
from .strategy import StrategyChallenge, StrategyConfig, load_strategy_config

__all__ = [
    "__version__",
    "SignalResult",
    "TradeDecision",
    "StrategyChallenge",
    "StrategyConfig",
    "combine_signals",
    "decide_trade_action",
    "load_strategy_config",
]

__version__ = "0.1.0"
