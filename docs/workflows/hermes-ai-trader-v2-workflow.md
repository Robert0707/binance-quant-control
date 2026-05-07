# Hermes AI Trader v2 Workflow

Status: repeatable gate. This workflow does not open orders.

## Run

```bash
openclaw-quantctl hermes-ai-trader --compact
```

The output includes:

- standardized signal schema,
- local signal ledger contract,
- portfolio target result,
- portfolio risk snapshot,
- structured committee votes,
- feature manifest hash,
- Hailo task boundary,
- architecture audit blockers,
- final open-order gate.

## If Blocked

Do not open entries. Use the blockers:

- `no-promotion-eligible-cohort`: rebuild alpha, not execution.
- `expectancy-r-not-positive`: improve payoff/cost/entry quality.
- `trade-count-below-floor`: do not promote short-sample rows.
- `portfolio_target.accepted=false`: reduce open risk or correlation.
- `structured-committee-reject`: fix evidence before retrying.
- `live_readiness.allowed=false`: fix kill-switch, route-side PF, notional, or
  protective order issues.

## If Allowed

Allowed means only "eligible for paper/testnet readiness review." The next
manual command is:

```bash
openclaw-quantctl live-readiness --strategy-config config/strategy-live-pilot.yaml --execution-mode testnet_exploration --compact
```

Actual order execution still requires an explicit operator command and live
execution gates.

## Clean Rebuild Rule

Do not delete `state/` or `reports/` as "old junk"; they are evidence. Clean
caches and backups, then quarantine legacy strategy pieces by routing them
through this v2 gate. A strategy becomes usable only when it leaves a replayable
feature manifest, alpha row, signal ledger record, portfolio decision, committee
decision, and readiness result.
