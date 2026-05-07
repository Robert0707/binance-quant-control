# AI Market Sentinel 24h

Purpose: keep a machine-readable watch loop over positions, trend state, route quarantines, and readiness blockers without opening orders.

## One-Shot Check

```bash
openclaw-quantctl ai-market-sentinel --compact
```

Safe properties:

- opens no orders
- cancels no orders
- closes no positions
- does not release the trading pause
- does not write execution config

## 24h Timer Pattern

Use a user timer or cron to run the sentinel every 1-5 minutes. The command should stay read-only:

```bash
cd /home/robert/python/projects/binance-quant-control
.venv/bin/binance-quant-control ai-market-sentinel --skip-readiness --compact
```

High-frequency timer mode skips readiness because readiness can be slower and
already runs inside Hermes trade cycles / expectancy loops. Use full sentinel
readiness manually when needed:

```bash
openclaw-quantctl ai-market-sentinel --compact
```

For the existing Hermes loop, each cycle now runs the sentinel before position guardian/readiness:

```bash
openclaw-quantctl hermes-trade cycle --force --dry-run-only --compact
```

Continuous testnet mode remains opt-in:

```bash
openclaw-quantctl hermes-trade start --dry-run-only --compact
openclaw-quantctl hermes-trade daemon --max-cycles 0 --compact
```

## Machine Actions

- `run_position_guardian`: open position exists; protect first and skip expansion.
- `run_ai_expectancy_upgrade`: no readiness-approved candidate exists; run formal large-sample expectancy research.
- `keep_route_quarantine`: negative-expectancy route remains blocked.
- `operator_testnet_preflight`: readiness has an approved testnet ticket; still requires execution boundary checks.

## Promotion Boundary

New entries stay blocked until:

- no open position blocks the lane,
- trading control is not paused,
- route quarantines are resolved by evidence, not manually loosened,
- readiness `allowed_count > 0`,
- the selected ticket passes fresh preflight.
