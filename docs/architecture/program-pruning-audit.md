# Program Pruning Audit

Status: 2026-05-11 architecture cleanup pass.

## Objective

Reduce dead or duplicate execution surfaces without weakening the trading gates.
The control plane must stay evidence-first:

- research and scanning can run locally;
- BUY/SELL/HOLD decisions must pass decision audit;
- testnet exploration needs readiness approval;
- mainnet live remains outside this cleanup.

## Removed

- `scripts/targeted_volatile_sweep.py`

  Reason: old one-off volatile-symbol experiment with hardcoded symbol universe
  and bespoke grid logic. It had no repo references and is superseded by the
  auditable `alpha-research`, `risk-combo-sweep`, `risk-combo-matrix`, and
  `high-win-iteration` workflow.

## Retained Manual Or External Entrypoints

These scripts may look low-reference from inside the repo, but they are still
operator or external-integration surfaces:

- `scripts/build_daily_digest.py`: n8n/daily digest wrapper around
  `daily_digest`.
- `scripts/run_official_engine_workflow.py`: Freqtrade workflow wrapper.
- `scripts/run_strategy_analyzer_service.py`: local strategy-analyzer service
  launcher.
- `scripts/sync_freqtrade_whitelist.py`: external Freqtrade whitelist bridge.
- `scripts/run_market_bot_profile_sweep.py`: older market-bot profile research
  script; keep until `risk-combo-sweep` fully covers all profile-sweep evidence
  needed by historical reports.

Do not delete these without checking local wrappers, timers, n8n flows, and
operator runbooks.

## Architecture Pressure Points

Current `repository-audit` reports the main cleanup pressure as oversized
modules, not secret files or generated files:

- `src/binance_quant_control/cli.py`
- `src/binance_quant_control/risk_combo_sweep.py`
- `src/binance_quant_control/analysis.py`
- `src/binance_quant_control/high_win_iteration.py`
- `src/binance_quant_control/alpha_research.py`

Future changes should extract focused command handlers and report writers
instead of adding more logic to these files.

## Simulation Readiness Verdict

The system is configured for testnet exploration, but the current gate result
does not allow starting an autonomous simulation order loop.

Latest checked gate:

```bash
openclaw-quantctl live-readiness --strategy-config config/strategy-stable-risk.yaml --execution-mode testnet_exploration --compact
```

Observed result:

- `allowed=false`
- `execution_mode=testnet_exploration`
- current candidate: `NEARUSDT` `SELL`
- blockers include paper-only route, weak volume, multi-timeframe trend conflict,
  stale optimizer report, negative route-side historical PF, and negative
  historical signal bucket.

Allowed work now:

- run read-only scanning;
- run bounded paper/testnet readiness checks;
- run research sweeps and matrix rebuilds;
- generate HOLD/decision artifacts and audit them.

Blocked work now:

- autonomous testnet order placement from the current candidate;
- mainnet live trading;
- relaxing gates to force a trade.
