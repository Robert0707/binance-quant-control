from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import CONFIG_DIR, PROJECT_ROOT, STATE_DIR, ensure_runtime_dirs, load_settings
from .hailo_entry_gate import evaluate_hailo_entry_gate
from .readiness_scanner import run_ai_readiness_scan
from .strategy import load_strategy_config
from .trading_control import (
    AUTO_PAUSE_ACTOR,
    AutoPausePolicy,
    TradingControlState,
    evaluate_auto_pause_conditions,
    load_trading_control_state,
    set_trading_paused,
)

DEFAULT_HERMES_TRADE_CONFIG_PATH = CONFIG_DIR / "hermes-trade-loop.default.yaml"
HERMES_TRADE_STATE_PATH = STATE_DIR / "hermes-trade-control.json"
HERMES_TRADE_REPORT_DIR = STATE_DIR / "hermes-trade-loop"
BINANCE_QUANT_CONTROL = PROJECT_ROOT / ".venv" / "bin" / "binance-quant-control"
HERMES_TRADE_START_ACTOR = "openclaw-quantctl hermes-trade start"
HERMES_TRADE_STOP_ACTOR = "openclaw-quantctl hermes-trade stop"


@dataclass(frozen=True, slots=True)
class HermesTradeLoopConfig:
    path: Path
    strategy_config: Path
    blueprint_config: Path
    market: str = "futures"
    limit: int = 0
    margin_notional_usdt: float | None = None
    execution_mode: str = "testnet_exploration"
    mainnet_live_allowed: bool = False
    execute_testnet_entries: bool = True
    allow_entries_while_positions_open: bool = False
    max_concurrent_positions: int = 4
    run_review_before_cycle: bool = True
    closed_trade_review_limit: int = 50
    run_auto_pause_before_cycle: bool = True
    run_position_guardian_before_cycle: bool = True
    run_market_sentinel_before_cycle: bool = True
    market_sentinel_command: tuple[str, ...] = (
        str(BINANCE_QUANT_CONTROL),
        "ai-market-sentinel",
        "--compact",
    )
    position_guardian_command: tuple[str, ...] = (
        "python3",
        "scripts/run_autonomous_trader.py",
        "--config",
        "config/autonomous-guardian.default.yaml",
        "--compact",
    )
    run_strategy_optimizer_before_cycle: bool = True
    run_strategy_optimizer_after_closed_position: bool = True
    strategy_optimizer_min_interval_minutes: int = 360
    strategy_optimizer_command: tuple[str, ...] = (
        "python3",
        "scripts/run_strategy_optimizer.py",
        "--config",
        "config/strategy-optimizer.default.yaml",
    )
    run_external_context_before_cycle: bool = True
    external_context_config: Path = CONFIG_DIR / "external-context.default.yaml"
    external_context_symbols: tuple[str, ...] = (
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "DOGEUSDT",
        "TRXUSDT",
    )
    external_context_min_interval_minutes: int = 10
    run_hailo_triage_before_cycle: bool = True
    hailo_triage_command: tuple[str, ...] = (
        "/home/robert/python/bin/openclaw-hailo-triage",
        "--once",
    )
    hailo_triage_min_interval_minutes: int = 5
    max_orders_per_cycle: int = 1
    max_cycles_per_run: int = 1
    sleep_seconds: float = 300.0
    position_poll_seconds: float = 45.0
    reentry_scan_seconds: float = 5.0
    stop_on_execution_error: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        payload["strategy_config"] = str(self.strategy_config)
        payload["blueprint_config"] = str(self.blueprint_config)
        payload["external_context_config"] = str(self.external_context_config)
        payload["external_context_symbols"] = list(self.external_context_symbols)
        payload["market_sentinel_command"] = list(self.market_sentinel_command)
        payload["position_guardian_command"] = list(self.position_guardian_command)
        payload["strategy_optimizer_command"] = list(self.strategy_optimizer_command)
        payload["hailo_triage_command"] = list(self.hailo_triage_command)
        return payload


@dataclass(frozen=True, slots=True)
class HermesTradeControlState:
    enabled: bool = False
    mode: str = "testnet"
    execute_testnet_entries: bool = False
    started_at: str = ""
    stopped_at: str = ""
    updated_at: str = ""
    updated_by: str = ""
    note: str = ""
    cycle_count: int = 0
    last_cycle_at: str = ""
    last_execution_at: str = ""
    last_parameter_adjustment_at: str = ""
    last_external_context_at: str = ""
    last_hailo_triage_at: str = ""
    last_open_position_keys: tuple[str, ...] = ()
    last_closed_position_at: str = ""
    last_closed_position_keys: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _resolve_config_path(base: Path, value: Any, default: Path) -> Path:
    raw = str(value or "").strip()
    if not raw:
        return default.resolve()
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()
    return (base / candidate).resolve()


def load_hermes_trade_loop_config(path: str | Path | None = None) -> HermesTradeLoopConfig:
    config_path = Path(path or DEFAULT_HERMES_TRADE_CONFIG_PATH).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Hermes trade loop config must be a mapping: {config_path}")
    strategy_cfg = payload.get("strategy") if isinstance(payload.get("strategy"), dict) else {}
    readiness_cfg = payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {}
    execution_cfg = payload.get("execution") if isinstance(payload.get("execution"), dict) else {}
    automation_cfg = payload.get("automation") if isinstance(payload.get("automation"), dict) else {}
    context_cfg = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    loop_cfg = payload.get("loop") if isinstance(payload.get("loop"), dict) else {}
    safety_cfg = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    guardian_command = automation_cfg.get("position_guardian_command") or [
        "python3",
        "scripts/run_autonomous_trader.py",
        "--config",
        "config/autonomous-guardian.default.yaml",
        "--compact",
    ]
    sentinel_command = automation_cfg.get("market_sentinel_command") or [
        str(BINANCE_QUANT_CONTROL),
        "ai-market-sentinel",
        "--compact",
    ]
    optimizer_command = automation_cfg.get("strategy_optimizer_command") or [
        "python3",
        "scripts/run_strategy_optimizer.py",
        "--config",
        "config/strategy-optimizer.default.yaml",
    ]
    hailo_command = context_cfg.get("hailo_triage_command") or [
        "/home/robert/python/bin/openclaw-hailo-triage",
        "--once",
    ]
    context_symbols = context_cfg.get("external_context_symbols") or [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "DOGEUSDT",
        "TRXUSDT",
    ]
    return HermesTradeLoopConfig(
        path=config_path,
        strategy_config=_resolve_config_path(
            config_path.parent,
            strategy_cfg.get("strategy_config"),
            CONFIG_DIR / "strategy-live-pilot.yaml",
        ),
        blueprint_config=_resolve_config_path(
            config_path.parent,
            readiness_cfg.get("blueprint_config"),
            CONFIG_DIR / "professional-system-blueprint.default.yaml",
        ),
        market=str(readiness_cfg.get("market") or "futures"),
        limit=int(readiness_cfg.get("limit") or 0),
        margin_notional_usdt=(
            float(execution_cfg["margin_notional_usdt"])
            if execution_cfg.get("margin_notional_usdt") is not None
            else None
        ),
        execution_mode=str(safety_cfg.get("execution_mode") or "testnet_exploration"),
        mainnet_live_allowed=bool(safety_cfg.get("mainnet_live_allowed", False)),
        execute_testnet_entries=bool(execution_cfg.get("execute_testnet_entries", True)),
        allow_entries_while_positions_open=bool(
            execution_cfg.get("allow_entries_while_positions_open", False)
        ),
        max_concurrent_positions=int(execution_cfg.get("max_concurrent_positions") or 4),
        run_review_before_cycle=bool(automation_cfg.get("review_closed_trades", True)),
        closed_trade_review_limit=int(automation_cfg.get("closed_trade_review_limit") or 50),
        run_auto_pause_before_cycle=bool(automation_cfg.get("auto_pause", True)),
        run_position_guardian_before_cycle=bool(
            automation_cfg.get("position_guardian_before_cycle", True)
        ),
        run_market_sentinel_before_cycle=bool(
            automation_cfg.get("market_sentinel_before_cycle", True)
        ),
        market_sentinel_command=tuple(str(item) for item in sentinel_command),
        position_guardian_command=tuple(str(item) for item in guardian_command),
        run_strategy_optimizer_before_cycle=bool(
            automation_cfg.get("strategy_optimizer_before_cycle", True)
        ),
        run_strategy_optimizer_after_closed_position=bool(
            automation_cfg.get("strategy_optimizer_after_closed_position", True)
        ),
        strategy_optimizer_min_interval_minutes=int(
            automation_cfg.get("strategy_optimizer_min_interval_minutes") or 360
        ),
        strategy_optimizer_command=tuple(str(item) for item in optimizer_command),
        run_external_context_before_cycle=bool(
            context_cfg.get("external_context_before_cycle", True)
        ),
        external_context_config=_resolve_config_path(
            config_path.parent,
            context_cfg.get("external_context_config"),
            CONFIG_DIR / "external-context.default.yaml",
        ),
        external_context_symbols=tuple(
            str(item).upper()
            for item in context_symbols
            if str(item or "").strip()
        ),
        external_context_min_interval_minutes=int(
            context_cfg.get("external_context_min_interval_minutes") or 10
        ),
        run_hailo_triage_before_cycle=bool(context_cfg.get("hailo_triage_before_cycle", True)),
        hailo_triage_command=tuple(str(item) for item in hailo_command),
        hailo_triage_min_interval_minutes=int(
            context_cfg.get("hailo_triage_min_interval_minutes") or 5
        ),
        max_orders_per_cycle=int(execution_cfg.get("max_orders_per_cycle") or 1),
        max_cycles_per_run=int(loop_cfg.get("max_cycles_per_run") or 1),
        sleep_seconds=float(loop_cfg.get("sleep_seconds") or 300.0),
        position_poll_seconds=float(loop_cfg.get("position_poll_seconds") or 45.0),
        reentry_scan_seconds=float(loop_cfg.get("reentry_scan_seconds") or 5.0),
        stop_on_execution_error=bool(loop_cfg.get("stop_on_execution_error", True)),
    )


def load_hermes_trade_state() -> HermesTradeControlState:
    if not HERMES_TRADE_STATE_PATH.exists():
        return HermesTradeControlState()
    try:
        payload = json.loads(HERMES_TRADE_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return HermesTradeControlState()
    if not isinstance(payload, dict):
        return HermesTradeControlState()
    return HermesTradeControlState(
        enabled=bool(payload.get("enabled", False)),
        mode=str(payload.get("mode") or "testnet"),
        execute_testnet_entries=bool(payload.get("execute_testnet_entries", False)),
        started_at=str(payload.get("started_at") or ""),
        stopped_at=str(payload.get("stopped_at") or ""),
        updated_at=str(payload.get("updated_at") or ""),
        updated_by=str(payload.get("updated_by") or ""),
        note=str(payload.get("note") or ""),
        cycle_count=int(payload.get("cycle_count") or 0),
        last_cycle_at=str(payload.get("last_cycle_at") or ""),
        last_execution_at=str(payload.get("last_execution_at") or ""),
        last_parameter_adjustment_at=str(payload.get("last_parameter_adjustment_at") or ""),
        last_external_context_at=str(payload.get("last_external_context_at") or ""),
        last_hailo_triage_at=str(payload.get("last_hailo_triage_at") or ""),
        last_open_position_keys=tuple(str(item) for item in payload.get("last_open_position_keys") or ()),
        last_closed_position_at=str(payload.get("last_closed_position_at") or ""),
        last_closed_position_keys=tuple(
            str(item) for item in payload.get("last_closed_position_keys") or ()
        ),
    )


def save_hermes_trade_state(state: HermesTradeControlState) -> HermesTradeControlState:
    HERMES_TRADE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    HERMES_TRADE_STATE_PATH.write_text(
        json.dumps(state.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return state


def start_hermes_trade_loop(
    *,
    config_path: str | Path | None = None,
    execute_testnet_entries: bool | None = None,
    note: str = "",
    release_own_pause: bool = True,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    config = load_hermes_trade_loop_config(config_path)
    now = _utc_now_iso()
    execute_testnet = config.execute_testnet_entries if execute_testnet_entries is None else execute_testnet_entries
    state = save_hermes_trade_state(
        replace(
            load_hermes_trade_state(),
            enabled=True,
            mode="testnet",
            execute_testnet_entries=bool(execute_testnet),
            started_at=now,
            stopped_at="",
            updated_at=now,
            updated_by=HERMES_TRADE_START_ACTOR,
            note=note or "Hermes trade loop started.",
        )
    )
    trading_control = load_trading_control_state()
    released_pause: dict[str, Any] | None = None
    if release_own_pause and trading_control.paused and trading_control.updated_by == HERMES_TRADE_STOP_ACTOR:
        released_pause = set_trading_paused(
            paused=False,
            reason="Hermes trade loop restarted.",
            updated_by=HERMES_TRADE_START_ACTOR,
        ).to_dict()
    return {
        "status": "started",
        "safety": {
            "mainnet_live_allowed": False,
            "execution_mode": config.execution_mode,
            "requires_testnet": True,
        },
        "config": config.to_dict(),
        "state": state.to_dict(),
        "released_pause": released_pause,
    }


def stop_hermes_trade_loop(*, reason: str = "Hermes stop command") -> dict[str, Any]:
    ensure_runtime_dirs()
    now = _utc_now_iso()
    state = save_hermes_trade_state(
        replace(
            load_hermes_trade_state(),
            enabled=False,
            execute_testnet_entries=False,
            stopped_at=now,
            updated_at=now,
            updated_by=HERMES_TRADE_STOP_ACTOR,
            note=reason,
        )
    )
    trading_control = set_trading_paused(
        paused=True,
        reason=reason,
        updated_by=HERMES_TRADE_STOP_ACTOR,
    )
    return {
        "status": "stopped",
        "state": state.to_dict(),
        "trading_control": trading_control.to_dict(),
    }


def hermes_trade_status(config_path: str | Path | None = None) -> dict[str, Any]:
    config = load_hermes_trade_loop_config(config_path)
    return {
        "status": "enabled" if load_hermes_trade_state().enabled else "stopped",
        "config": config.to_dict(),
        "state": load_hermes_trade_state().to_dict(),
        "trading_control": load_trading_control_state().to_dict(),
    }


def _run_json_command(command: list[str], *, timeout: int = 900) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        cwd=str(PROJECT_ROOT),
        timeout=timeout,
        check=False,
    )
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    response: dict[str, Any] | None = None
    if stdout:
        try:
            parsed = json.loads(stdout)
            if isinstance(parsed, dict):
                response = parsed
        except json.JSONDecodeError:
            response = None
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": stdout[-8000:],
        "stderr": stderr[-8000:],
        "response": response,
    }


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _optimizer_due(config: HermesTradeLoopConfig, state: HermesTradeControlState) -> bool:
    if not config.run_strategy_optimizer_before_cycle:
        return False
    last = _parse_datetime(state.last_parameter_adjustment_at)
    if last is None:
        return True
    age_minutes = (_utc_now() - last).total_seconds() / 60.0
    return age_minutes >= config.strategy_optimizer_min_interval_minutes


def _interval_due(last_value: str, *, minutes: int) -> bool:
    last = _parse_datetime(last_value)
    if last is None:
        return True
    return (_utc_now() - last).total_seconds() >= max(0, minutes) * 60


def _position_key(position: dict[str, Any]) -> str:
    symbol = str(position.get("symbol") or "").upper()
    side = str(position.get("side") or "").upper()
    qty = float(position.get("qty", position.get("quantity", 0.0)) or 0.0)
    entry = float(position.get("entry", position.get("entry_price", 0.0)) or 0.0)
    return f"{symbol}:{side}:{qty:.8f}:{entry:.8f}"


def _positions_from_scan(scan: dict[str, Any]) -> list[dict[str, Any]]:
    response = scan.get("response") if isinstance(scan.get("response"), dict) else {}
    positions = response.get("positions") if isinstance(response, dict) else []
    return positions if isinstance(positions, list) else []


def _position_keys(positions: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted(_position_key(item) for item in positions if str(item.get("symbol") or "").strip()))


def _closed_position_keys(
    previous: tuple[str, ...],
    current: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(sorted(set(previous) - set(current)))


def _run_position_scan() -> dict[str, Any]:
    return _run_json_command(
        [str(BINANCE_QUANT_CONTROL), "positions", "--compact"],
        timeout=300,
    )


def _run_review_closed_trades(config: HermesTradeLoopConfig) -> dict[str, Any]:
    return _run_json_command(
        [
            str(BINANCE_QUANT_CONTROL),
            "review-closed-trades",
            "--limit",
            str(config.closed_trade_review_limit),
            "--compact",
        ],
        timeout=600,
    )


def _run_external_context(config: HermesTradeLoopConfig) -> dict[str, Any]:
    return _run_json_command(
        [
            str(BINANCE_QUANT_CONTROL),
            "external-context",
            "--config",
            str(config.external_context_config),
            "--symbols",
            ",".join(config.external_context_symbols),
            "--compact",
        ],
        timeout=120,
    )


def _run_position_guardian(config: HermesTradeLoopConfig) -> dict[str, Any]:
    return _run_json_command(list(config.position_guardian_command), timeout=900)


def _run_market_sentinel(config: HermesTradeLoopConfig) -> dict[str, Any]:
    return _run_json_command(list(config.market_sentinel_command), timeout=900)


def _run_hailo_triage(config: HermesTradeLoopConfig) -> dict[str, Any]:
    return _run_json_command(list(config.hailo_triage_command), timeout=180)


def _run_strategy_optimizer(config: HermesTradeLoopConfig) -> dict[str, Any]:
    return _run_json_command(list(config.strategy_optimizer_command), timeout=1200)


def _run_auto_pause(config: HermesTradeLoopConfig) -> dict[str, Any]:
    settings = load_settings()
    strategy = load_strategy_config(config.strategy_config)
    policy = AutoPausePolicy(
        consecutive_loss_threshold=2,
        loss_cooldown_hours=float(strategy.risk.cooldown_hours),
    )
    evaluation = evaluate_auto_pause_conditions(settings, strategy, policy=policy)
    current = load_trading_control_state()
    actions: list[str] = []
    state: TradingControlState = current
    if evaluation.should_pause and not current.paused:
        state = set_trading_paused(
            paused=True,
            reason="; ".join(evaluation.reasons),
            updated_by=AUTO_PAUSE_ACTOR,
        )
        actions.append("paused")
    elif not evaluation.should_pause and current.paused and current.updated_by == AUTO_PAUSE_ACTOR:
        state = set_trading_paused(
            paused=False,
            reason="Auto-resumed after pause conditions cleared.",
            updated_by=AUTO_PAUSE_ACTOR,
        )
        actions.append("resumed")
    elif evaluation.should_pause and current.paused:
        actions.append("already-paused")
    else:
        actions.append("no-action")
    return {
        "actions": actions,
        "trading_control": state.to_dict(),
        "policy": policy.to_dict(),
        "evaluation": evaluation.to_dict(),
    }


def _command_string_to_list(command: Any) -> list[str]:
    if isinstance(command, list):
        return [str(item) for item in command]
    return shlex.split(str(command or ""))


def _safe_execution_allowed(
    *,
    config: HermesTradeLoopConfig,
    state: HermesTradeControlState,
    trading_control: TradingControlState,
    ticket: dict[str, Any] | None,
    open_position_count: int = 0,
    hailo_blockers: list[str] | None = None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not state.enabled:
        reasons.append("hermes-trade-loop-disabled")
    if not state.execute_testnet_entries or not config.execute_testnet_entries:
        reasons.append("testnet-execution-disabled")
    if config.mainnet_live_allowed:
        reasons.append("mainnet-live-is-not-allowed-in-hermes-loop")
    if config.execution_mode != "testnet_exploration":
        reasons.append(f"execution-mode-{config.execution_mode}-is-not-testnet-exploration")
    if trading_control.paused:
        reasons.append("trading-control-paused")
    if open_position_count >= config.max_concurrent_positions:
        reasons.append("max-concurrent-positions-reached")
    if open_position_count > 0 and not config.allow_entries_while_positions_open:
        reasons.append("open-position-management-priority")
    reasons.extend(hailo_blockers or [])
    if not ticket:
        reasons.append("no-execution-ticket")
    elif ticket.get("state") != "ready_for_operator_testnet_execution":
        reasons.append("execution-ticket-not-ready")
    return len(reasons) == 0, reasons


def run_hermes_trade_cycle(
    *,
    config_path: str | Path | None = None,
    force: bool = False,
    execute_testnet_entries: bool | None = None,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    HERMES_TRADE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_hermes_trade_loop_config(config_path)
    state = load_hermes_trade_state()
    if execute_testnet_entries is not None:
        state = save_hermes_trade_state(
            replace(
                state,
                execute_testnet_entries=bool(execute_testnet_entries),
                updated_at=_utc_now_iso(),
                updated_by="openclaw-quantctl hermes-trade cycle",
            )
        )
    generated_at = _utc_now()
    summary: dict[str, Any] = {
        "generated_at": generated_at.isoformat(),
        "mode": "hermes_trade_loop_v1",
        "safety": {
            "opens_orders": False,
            "mainnet_live_allowed": False,
            "execution_mode": config.execution_mode,
            "can_execute_testnet_when_enabled": bool(config.execute_testnet_entries),
        },
        "config": config.to_dict(),
        "state_before": state.to_dict(),
        "steps": {},
    }
    if not state.enabled and not force:
        summary["status"] = "stopped"
        summary["reason"] = "Hermes trade loop is disabled. Use hermes-trade start first."
        report_path = HERMES_TRADE_REPORT_DIR / f"{_stamp()}-hermes-trade-cycle.json"
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary["report_path"] = str(report_path)
        return summary

    positions_before = _run_position_scan()
    positions_before_rows = _positions_from_scan(positions_before)
    open_position_keys_before = _position_keys(positions_before_rows)
    closed_position_keys = _closed_position_keys(
        state.last_open_position_keys,
        open_position_keys_before,
    )
    closed_position_detected = bool(closed_position_keys)
    summary["steps"]["positions_before"] = positions_before
    summary["position_loop"] = {
        "open_position_count_before": len(open_position_keys_before),
        "open_position_keys_before": list(open_position_keys_before),
        "previous_open_position_keys": list(state.last_open_position_keys),
        "closed_position_detected": closed_position_detected,
        "closed_position_keys": list(closed_position_keys),
        "mode": (
            "manage_and_seek_new_entry"
            if open_position_keys_before and config.allow_entries_while_positions_open
            and len(open_position_keys_before) < config.max_concurrent_positions
            else "manage_positions"
            if open_position_keys_before
            else "seek_next_entry"
        ),
    }

    if config.run_review_before_cycle:
        summary["steps"]["review_closed_trades"] = _run_review_closed_trades(config)
    if config.run_auto_pause_before_cycle:
        summary["steps"]["auto_pause"] = _run_auto_pause(config)
    if config.run_market_sentinel_before_cycle:
        summary["steps"]["market_sentinel"] = _run_market_sentinel(config)
    if config.run_position_guardian_before_cycle and open_position_keys_before:
        summary["steps"]["position_guardian"] = _run_position_guardian(config)
    if config.run_external_context_before_cycle and _interval_due(
        state.last_external_context_at,
        minutes=config.external_context_min_interval_minutes,
    ):
        external_context = _run_external_context(config)
        summary["steps"]["external_context"] = external_context
        if external_context.get("returncode") == 0:
            state = replace(state, last_external_context_at=_utc_now_iso())
    if config.run_hailo_triage_before_cycle and _interval_due(
        state.last_hailo_triage_at,
        minutes=config.hailo_triage_min_interval_minutes,
    ):
        hailo_triage = _run_hailo_triage(config)
        summary["steps"]["hailo_triage"] = hailo_triage
        summary["hailo_entry_gate"] = evaluate_hailo_entry_gate(hailo_triage)
        if hailo_triage.get("returncode") == 0:
            state = replace(state, last_hailo_triage_at=_utc_now_iso())
    optimizer_due = _optimizer_due(config, state)
    optimizer_after_close = (
        closed_position_detected
        and config.run_strategy_optimizer_after_closed_position
        and config.run_strategy_optimizer_before_cycle
    )
    if optimizer_due or optimizer_after_close:
        summary["steps"]["strategy_optimizer"] = _run_strategy_optimizer(config)
        state = replace(state, last_parameter_adjustment_at=_utc_now_iso())

    trading_control = load_trading_control_state()
    summary["trading_control"] = trading_control.to_dict()
    readiness: dict[str, Any] = {}
    ticket: dict[str, Any] | None = None
    capacity_reached = len(open_position_keys_before) >= config.max_concurrent_positions
    if capacity_reached or (open_position_keys_before and not config.allow_entries_while_positions_open):
        summary["readiness"] = {
            "status": "skipped",
            "reason": "max-concurrent-positions-reached" if capacity_reached else "open-position-management-priority",
            "candidate_count": None,
            "allowed_count": None,
            "selected_ready_candidate": None,
            "execution_ticket": None,
        }
    else:
        readiness = run_ai_readiness_scan(
            blueprint_config=config.blueprint_config,
            strategy_config=config.strategy_config,
            market=config.market,
            limit=config.limit,
            margin_notional_usdt=config.margin_notional_usdt,
            execution_mode=config.execution_mode,
            exclude_symbols=[
                str(position.get("symbol") or "").upper()
                for position in positions_before_rows
                if str(position.get("symbol") or "").strip()
            ],
        )
        summary["readiness"] = {
            "candidate_count": readiness.get("candidate_count"),
            "excluded_symbols": readiness.get("excluded_symbols"),
            "allowed_count": readiness.get("allowed_count"),
            "selected_ready_candidate": readiness.get("selected_ready_candidate"),
            "next_machine_action": readiness.get("next_machine_action"),
            "hard_blocker_taxonomy": readiness.get("hard_blocker_taxonomy"),
            "execution_ticket": readiness.get("execution_ticket"),
            "report_path": readiness.get("report_path"),
        }
        ticket = readiness.get("execution_ticket") if isinstance(readiness.get("execution_ticket"), dict) else None
    allowed_to_execute, execution_blockers = _safe_execution_allowed(
        config=config,
        state=state,
        trading_control=trading_control,
        ticket=ticket,
        open_position_count=len(open_position_keys_before),
        hailo_blockers=(summary.get("hailo_entry_gate") or {}).get("blockers") or [],
    )
    summary["execution_gate"] = {
        "allowed": allowed_to_execute,
        "blockers": execution_blockers,
    }

    executed = False
    if allowed_to_execute and ticket and config.max_orders_per_cycle > 0:
        preflight = _run_json_command(
            _command_string_to_list(ticket.get("preflight_command")),
            timeout=600,
        )
        summary["steps"]["preflight"] = preflight
        preflight_response = preflight.get("response") if isinstance(preflight.get("response"), dict) else {}
        if preflight.get("returncode") == 0 and bool(preflight_response.get("allowed", False)):
            execution = _run_json_command(
                _command_string_to_list(ticket.get("operator_testnet_execute_command")),
                timeout=900,
            )
            summary["steps"]["testnet_execution"] = execution
            executed = execution.get("returncode") == 0
            summary["safety"]["opens_orders"] = executed
            summary["status"] = "testnet_executed" if executed else "execution_failed"
            if executed:
                state = replace(state, last_execution_at=_utc_now_iso())
            elif config.stop_on_execution_error:
                state = replace(
                    state,
                    enabled=False,
                    execute_testnet_entries=False,
                    stopped_at=_utc_now_iso(),
                    updated_by="openclaw-quantctl hermes-trade cycle",
                    note="Stopped after testnet execution command failed.",
                )
                set_trading_paused(
                    paused=True,
                    reason="Hermes trade loop stopped after execution command failed.",
                    updated_by="openclaw-quantctl hermes-trade cycle",
                )
        else:
            summary["status"] = "preflight_blocked"
    else:
        summary["status"] = "ready" if ticket else "blocked"

    summary["steps"]["positions_after"] = _run_position_scan()
    positions_after_rows = _positions_from_scan(summary["steps"]["positions_after"])
    open_position_keys_after = _position_keys(positions_after_rows)
    summary["position_loop"].update(
        {
            "open_position_count_after": len(open_position_keys_after),
            "open_position_keys_after": list(open_position_keys_after),
            "next_sleep_seconds": (
                config.position_poll_seconds
                if open_position_keys_after
                else config.reentry_scan_seconds
                if closed_position_detected
                else config.sleep_seconds
            ),
        }
    )
    if summary.get("status") == "blocked" and open_position_keys_before and not ticket:
        summary["status"] = "managing_position"
    state = save_hermes_trade_state(
        replace(
            state,
            cycle_count=state.cycle_count + 1,
            last_cycle_at=generated_at.isoformat(),
            updated_at=_utc_now_iso(),
            updated_by="openclaw-quantctl hermes-trade cycle",
            last_open_position_keys=open_position_keys_after,
            last_closed_position_at=_utc_now_iso() if closed_position_detected else state.last_closed_position_at,
            last_closed_position_keys=closed_position_keys if closed_position_detected else state.last_closed_position_keys,
        )
    )
    summary["state_after"] = state.to_dict()
    summary["executed"] = executed
    report_path = HERMES_TRADE_REPORT_DIR / f"{_stamp()}-hermes-trade-cycle.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["report_path"] = str(report_path)
    return summary


def run_hermes_trade_daemon(
    *,
    config_path: str | Path | None = None,
    max_cycles: int | None = None,
    sleep_seconds: float | None = None,
) -> dict[str, Any]:
    config = load_hermes_trade_loop_config(config_path)
    cycle_limit = config.max_cycles_per_run if max_cycles is None else int(max_cycles)
    delay = config.sleep_seconds if sleep_seconds is None else float(sleep_seconds)
    cycles: list[dict[str, Any]] = []
    index = 0
    while load_hermes_trade_state().enabled:
        if cycle_limit > 0 and index >= cycle_limit:
            break
        cycle = run_hermes_trade_cycle(config_path=config.path)
        cycles.append(
            {
                "status": cycle.get("status"),
                "executed": cycle.get("executed"),
                "position_loop": cycle.get("position_loop"),
                "selected": (cycle.get("readiness") or {}).get("selected_ready_candidate"),
                "report_path": cycle.get("report_path"),
            }
        )
        index += 1
        if cycle_limit > 0 and index >= cycle_limit:
            break
        if load_hermes_trade_state().enabled:
            position_loop = cycle.get("position_loop") if isinstance(cycle.get("position_loop"), dict) else {}
            next_delay = position_loop.get("next_sleep_seconds")
            if next_delay is None:
                next_delay = delay
            time.sleep(max(0.0, float(next_delay)))
    return {
        "status": "completed",
        "cycle_count": len(cycles),
        "state": load_hermes_trade_state().to_dict(),
        "cycles": cycles,
    }
