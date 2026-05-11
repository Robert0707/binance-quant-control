# Binance Quant Control

Safe-by-default Binance quant control plane for research, paper trading, testnet validation, route gating, and AI-assisted strategy operations.

This repo is built for a machine-operated trading workflow: local evidence first, explicit risk gates, compact status commands, and no hidden live execution. It does not promise profit. It is designed to prevent unproven routes from reaching live trading until they pass readiness, risk, and forward evidence checks.

## Safety Defaults

- Mainnet live trading is off by default.
- Testnet is the default exchange mode.
- `.env`, `state/`, `reports/`, `.venv/`, logs, and caches are ignored by git.
- Live execution requires explicit environment switches and a separate readiness pass.
- Strategy optimization must respect the stable risk baseline, including the 2.5% per-trade account-risk ceiling.
- Hailo is used as a local triage / veto / compression layer, not as an alpha oracle.

## Repository Layout

```text
config/     Strategy, route, risk, AI trader, and workflow configs
docs/       Architecture notes, runbooks, workflows, and templates
scripts/    Local workflow wrappers and research runners
src/        Python package and CLI implementation
tests/      Unit and workflow coverage
state/      Local runtime state, ignored by git
reports/    Local evidence and backtest reports, ignored by git
```

## Setup

```bash
cd /home/robert/python/projects/binance-quant-control
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e '.[dev]'
cp .env.example .env
```

Fill `.env` locally. Do not commit it.

Important defaults in `.env.example`:

```bash
BINANCE_USE_TESTNET=true
BINANCE_LIVE_TRADING_ENABLED=false
BINANCE_TESTNET_TRADING_ENABLED=false
```

## Fast Verification

Use compact commands for routine work to avoid noisy logs and token burn.

```bash
openclaw-quantctl doctor --compact
openclaw-quantctl validate-config --strategy-config config/strategy-stable-risk.yaml --compact
.venv/bin/python -m pytest
.venv/bin/ruff check .
```

The stable baseline is:

```text
config/strategy-stable-risk.yaml
```

It is the conservative profile for survival-first testing and should be validated before broader workflows.

## Common Commands

Read-only local status:

```bash
openclaw-quantctl trading-control-status --compact
openclaw-quantctl positions --compact
openclaw-quantctl route-risk-status --compact
openclaw-quantctl operator-dashboard --compact
```

AI market sentinel:

```bash
openclaw-quantctl ai-market-sentinel --skip-readiness --compact
openclaw-quantctl ai-market-sentinel --compact
```

New symbol to trade pipeline:

```bash
openclaw-quantctl route-intent "把 SOLUSDT 從新幣審核到可交易"
openclaw-quantctl route-symbol SOLUSDT
openclaw-quantctl feature-dataset --symbols SOLUSDT --compact
openclaw-quantctl risk-combo-sweep --symbols SOLUSDT --compact
openclaw-quantctl risk-combo-matrix --latest-sweeps 4 --compact
openclaw-quantctl hermes-ai-trader --compact
openclaw-quantctl ai-readiness-scan --compact
openclaw-quantctl operator-dashboard --compact
```

Long risk-combo validation runs:

```bash
python3 scripts/run_offline_risk_validation.py start --symbols TRXUSDT --target-side BUY --target-interval 1d
python3 scripts/run_offline_risk_validation.py status
```

The offline runner only writes local research evidence under `state/`; it does not open orders, change execution config, or enable mainnet.

Hermes trade cycle dry run:

```bash
openclaw-quantctl hermes-trade cycle --force --dry-run-only --compact
```

Closed-trade review and strategy feedback:

```bash
openclaw-quantctl review-closed-trades --compact
openclaw-quantctl journal-summary --compact
```

Formal large-sample expectancy upgrade:

```bash
openclaw-quantctl ai-expectancy-upgrade \
  --universe-limit 20 \
  --limit 8000 \
  --sweep-limit 5000 \
  --max-configs 80 \
  --max-walk-forward-validations 12 \
  --max-readiness-candidates 6 \
  --compact
```

Goal loop:

```bash
openclaw-quantctl ai-goal-loop --compact
```

## Low-Token Local Schedule

Routine market watching should run locally, not through chat.

Recommended low-cost timers:

- `ai-market-sentinel.timer`: every 2 minutes, read-only watch loop.
- `openclaw-binance-position-guardian.timer`: every 5 minutes, open-position protection.
- `openclaw-binance-operator-dashboard.timer`: every 15 minutes, compact status summary.
- `openclaw-binance-quant-research.timer`: hourly closed-trade review.
- `openclaw-binance-strategy-optimizer.timer`: every 6 hours, whitelisted strategy parameter optimization.

Timer status:

```bash
systemctl --user list-timers --all --no-pager | rg 'ai-market|binance|quant|NEXT'
```

Keep heavier exploration timers disabled unless readiness and exposure gates are clean:

- `openclaw-binance-testnet-explorer.timer`
- `openclaw-binance-autonomy.timer`
- `openclaw-binance-live-lane.timer`

## Trading Boundary

The intended flow is:

```text
market data -> features -> strategy candidates -> risk gates -> readiness scan -> testnet/paper evidence -> closed-trade review -> promotion gate
```

New entries should stay blocked when:

- an existing exposure blocks the lane,
- trading control is paused,
- route quarantine is active,
- readiness has no allowed candidate,
- portfolio correlation or same-beta exposure is too high,
- protective stop / staged take-profit coverage is incomplete.

Do not loosen gates just to force trades. A blocked trade can be the correct output.

## Live Execution

Live execution is intentionally separate from research and readiness.

Useful readiness commands:

```bash
openclaw-quantctl live-readiness --strategy-config config/strategy-stable-risk.yaml --compact
openclaw-quantctl live-readiness --strategy-config config/strategy-hermes-pro.yaml --compact
```

Only use execution commands after explicit operator approval, valid credentials, clean readiness, and testnet/forward evidence.

## Evidence And Local State

Ignored local outputs:

```text
state/
reports/
hailort.log
```

These are intentionally not committed because they may contain local account state, runtime traces, or bulky evidence.

## Development

```bash
.venv/bin/ruff check .
.venv/bin/python -m pytest
```

Before pushing:

```bash
git status --short
git check-ignore -v .env state reports .venv hailort.log
```

## Key Docs

- `PROJECT.md`: detailed project inventory and operating boundary.
- `docs/runbooks/local-low-token-schedule.md`: local timer policy.
- `docs/runbooks/ai-market-sentinel-24h.md`: read-only 24h watch loop.
- `docs/runbooks/stoploss-takeprofit-policy.md`: stop-loss and staged take-profit policy.
- `docs/workflows/new-symbol-to-trade-pipeline.md`: full evidence-first path from a new symbol to paper/testnet eligibility.
- `docs/workflows/market-bot-expectancy-research-pipeline.md`: positive-expectancy research flow.
- `docs/architecture/hermes-ai-trader-v2.md`: AI trader control-plane architecture.
