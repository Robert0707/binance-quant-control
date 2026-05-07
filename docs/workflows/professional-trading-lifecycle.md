# Professional Trading Lifecycle

Status: repeatable workflow. This does not authorize live entries.

## Goal

Use professional trading-bot boundaries from coin discovery to execution so the
project can mature without loosening gates just to force trades.

## Workflow

1. Audit architecture:

```bash
openclaw-quantctl professional-system-audit --compact
```

2. Validate the active research config:

```bash
openclaw-quantctl validate-config --strategy-config config/strategy-core-high-win-research.yaml --compact
```

3. Run mapped core alpha research:

```bash
openclaw-quantctl alpha-research --config config/core-alpha-research.default.yaml --limit 5000 --output-dir state/core-10-professional-l5000 --compact
```

4. Convert research into an operator gate:

```bash
openclaw-quantctl high-win-iteration --alpha-report state/core-10-professional-l5000/alpha-research-ranking.json --compact
```

5. If and only if the research gate passes, run live readiness in dry-run mode:

```bash
openclaw-quantctl live-readiness --strategy-config config/strategy-live-pilot.yaml --execution-mode testnet_exploration --compact
```

## Stop Rules

- If `professional-system-audit.trade_ready=false`, do not open new entries.
- If `safe_to_open_new_entries=false`, do not open new entries.
- If `promotion_eligible_count=0`, do not open new entries.
- If exchange min notional forces risk above sizing intent, skip the trade.
- If protective orders cannot be attached at entry, pause new entries.

## Next Build Priorities

1. Portfolio construction module for risk-budgeted targets.
2. Broker-neutral `Order`, `Position`, `Fill`, `Broker`, and `DataSource`
   contracts.
3. Feature manifest plus triple-barrier label registry.
4. Skipped-signal ledger for every gate denial.
5. Structured analyst/bull/bear/risk/portfolio promotion review.
