# Repository Map

Status: maintenance guide. This file is not a trading signal and does not
authorize execution.

## Active Surfaces

- `src/binance_quant_control/` contains the Python package.
- `config/` contains operator-editable research, strategy, routing, digest, and
  workflow defaults.
- `scripts/` contains thin executable wrappers around package functions.
- `tests/` mirrors the package by behavior area.
- `docs/workflows/` contains repeatable operating workflows.
- `docs/runbooks/` contains operational policies.
- `docs/architecture/` contains structural maps and architecture decisions.

## Generated Or Local Surfaces

- `.env` and `.env.bak-*` are local secret material. Do not audit, copy, or
  expose them.
- `state/` contains JSON reports and controller state.
- `reports/` contains analysis artifacts.
- `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `*.egg-info/`, and
  `hailort.log` are generated artifacts.
- `config/*.bak-*` are local backups, not active strategy configs.

## Module Boundaries

- Market data and indicators: `binance_api.py`, `historical_klines.py`,
  `indicators.py`, `volume_structure.py`, `market_context.py`.
- Analysis and signal construction: `analysis.py`, `alpha_families.py`,
  `signals.py`, `signal_scoring.py`, `side_risk_policy.py`.
- Strategy/routing/config: `strategy.py`, `asset_routing.py`,
  `symbol_strategy_map.py`, `strategy_baselines.py`, `validation.py`.
- Research/backtest/convergence: `backtest.py`, `alpha_research.py`,
  `risk_combo_sweep.py`, `high_win_iteration.py`, `high_win_convergence.py`,
  `convergence.py`, `repository_audit.py`.
- Execution and protection: `live_execution.py`, `risk_guard.py`,
  `position_manager.py`, `protective_repair.py`, `trading_control.py`.
- Journaling/review/training: `order_journal.py`, `loss_diagnostics.py`,
  `strategy_optimizer.py`, `training.py`, `public_history_training.py`,
  `historical_signal_risk.py`.
- Automation/workflows: `autonomy.py`, `mission_control.py`, `supervision.py`,
  `daily_digest.py`, `external_context.py`, `operator_dashboard.py`,
  `final_convergence_audit.py`.
- CLI and scripts: `cli.py` is the command router; avoid adding business logic
  there. Put new logic in focused modules and expose it through a thin command.

## Architecture Pressure Points

- `cli.py` is intentionally a router but is already large. New commands should
  call module-level services and keep compact output blocks short.
- `analysis.py` owns many indicator enrichment and report fields. New indicator
  families should prefer `indicators.py` / `alpha_families.py` unless they must
  enrich the shared frame.
- `risk_combo_sweep.py`, `high_win_iteration.py`, and
  `high_win_convergence.py` are research controllers. They must remain
  research-only and must not write live execution configs.
- `backtest.py` must stay aligned with live exit semantics: stop-loss, staged
  TP, runner/trailing, and time-stop behavior should not diverge silently.
- `professional-system-audit` is the professional trading-system blueprint gate:
  it checks architecture layers plus current alpha promotion evidence, and it
  does not trade.
- `hermes-ai-trader` is the clean v2 decision gate: signal schema, local signal
  ledger, event/plugin lifecycle, feature/model manifest hash, committee review,
  portfolio target/risk snapshot, Hailo boundaries, and skipped-signal logging.

## Audit Command

```bash
openclaw-quantctl repository-audit --compact
openclaw-quantctl professional-system-audit --compact
openclaw-quantctl hermes-ai-trader --compact
```

Use `--include-generated` only when diagnosing local clutter. The default audit
skips secrets and generated artifacts.
