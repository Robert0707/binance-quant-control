# AI Market Sentinel 24h

Purpose: keep a machine-readable watch loop over positions, trend state, route quarantines, and readiness blockers without opening orders.

## One-Shot Check

```bash
openclaw-quantctl ai-market-sentinel --compact
```

When a readiness-approved candidate exists, the sentinel now builds a
notification-only conditional-order card with:

- symbol and long/short side
- conditional entry/reference price
- staged take-profit prices and quantities
- stop-loss price
- maximum safe leverage from the execution plan
- machine-readable reasons for entry
- the preflight and operator testnet execution commands

When no candidate is executable but a promoted candidate is blocked only by
market state, the sentinel emits a near-ready watch card instead. This is useful
for candidates such as `TRXUSDT BUY 1d`, where the research / performance gate
passes but `volume_zscore_20` is still below the liquidity floor.

Telegram delivery is opt-in and still does not open orders:

```bash
openclaw-quantctl ai-market-sentinel --max-readiness-candidates 2 --send-telegram --compact
```

If no candidate passes readiness and no near-ready market-state watch exists,
Telegram is not sent and the report keeps the hard blocker taxonomy.

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
already runs inside Hermes trade cycles / expectancy loops. Add a lower-frequency
near-ready watch when a promoted risk-combo candidate exists:

```bash
openclaw-quantctl ai-market-sentinel --max-readiness-candidates 2 --compact
```

Use full sentinel readiness manually when needed:

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
- `monitor_near_ready_market_state`: positive-expectancy candidate exists, but readiness is blocked only by market state.
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
