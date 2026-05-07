# Market-Bot Expectancy Research Pipeline

Status: research gate only. This workflow does not open orders.

## Principle

AI trading bots still use strategies and indicators, but mature bots do not
trade from indicator voting alone. The production shape is:

1. universe and route selection,
2. feature manifest and replayable dataset,
3. strategy family signal,
4. triple-barrier exit assumptions,
5. expectancy/PF/payoff/sample gate,
6. portfolio and pre-trade risk,
7. execution readiness.

The primary gate is fixed-risk expectancy and payoff after costs. Win rate is a
screening metric, not the objective.

The portfolio gate requires six accepted symbols by default. A single high-score
row can become an expansion candidate, but it cannot make
`safe_to_open_new_entries=true` unless the portfolio has at least six symbols
with mature positive-expectancy cohorts.

## Commands

Build replayable TRX/ETH/BTC feature rows:

```bash
openclaw-quantctl feature-dataset --symbols TRXUSDT,ETHUSDT,BTCUSDT --intervals 1h,4h --limit 5000 --compact
```

Run focused alpha research:

```bash
openclaw-quantctl alpha-research --config config/market-bot-discovery.default.yaml --symbols TRXUSDT,ETHUSDT,BTCUSDT --intervals 1h,4h --limit 5000 --output-dir state/market-bot-trx-eth-btc-discovery-l5000 --compact
```

Evaluate with the mainstream-bot gate:

```bash
openclaw-quantctl market-bot-gate --alpha-report state/market-bot-trx-eth-btc-discovery-l5000/alpha-research-ranking.json --compact
```

Run the six-symbol AI-trader discovery lane:

```bash
openclaw-quantctl alpha-research --config config/market-bot-six-symbol-discovery.default.yaml --limit 8000 --output-dir state/market-bot-six-symbol-payoff-l8000 --compact
openclaw-quantctl market-bot-gate --alpha-report state/market-bot-six-symbol-payoff-l8000/alpha-research-ranking.json --compact
```

If a row passes this gate, continue:

```bash
openclaw-quantctl hermes-ai-trader --compact
openclaw-quantctl live-readiness --strategy-config config/strategy-live-pilot.yaml --execution-mode testnet_exploration --compact
```

## Current Read

The first TRX/ETH/BTC l1200 smoke produced a valid manifest hash and a best row:

- `TRXUSDT:4h:mean_reversion`
- 4 trades
- win rate 75%
- PF 1.2588
- expectancy 0.0175R
- payoff ratio 0.4195

Decision: blocked. The next optimization is payoff structure and sample
expansion, not more entry indicators.

The 2026-05-04 l5000 protective-map run produced 30,000 feature rows but only
10 trades because the core map intentionally quarantined most routes. The
market-bot gate blocked it: `trade_count=10`, `payoff_ratio=0.9148`, and no
slippage resilience.

The 2026-05-04 discovery run widened TRX/ETH/BTC to all six strategy families.
It found a short-sample TRX 4h edge:

- `TRXUSDT:4h:trend_continuation`
- 49 trades
- PF 2.3123
- expectancy 0.3159R
- payoff ratio 2.614

Decision: blocked and tagged `expand_sample_before_promotion`.

The expanded TRX 4h l15000 follow-up rejected the short-sample illusion:

- `TRXUSDT:4h:breakout`: 221 trades, PF 1.0614, expectancy 0.0261R, stop-loss
  56.56%, stressed returns turned negative
- `TRXUSDT:4h:trend_continuation`: 358 trades, PF 0.9972, expectancy -0.0011R

Decision: blocked. Do not promote TRX 4h breakout/continuation until a new
route filter or exit/risk structure changes the expanded-sample evidence.

Use `risk-combo-sweep` to test exit profiles:

```bash
openclaw-quantctl risk-combo-sweep --symbols TRXUSDT --limit 5000 --grid-mode focused --min-test-trades 100 --min-win-rate 45 --max-stop-loss-ratio 55 --target-profit-factor 1.25 --min-expectancy-r 0.05 --min-payoff-ratio 1.20 --max-configs 80 --max-walk-forward-validations 12 --skip-news --compact
```

Gate state meanings:

- `tradable_candidate`: row may continue to Hermes and live-readiness gates.
- `expand_sample_before_promotion`: row has positive edge but is below the
  trade floor; expand the same cohort before changing live maps.
- `reject_expanded_route_regressed`: row reached the sample floor but PF,
  expectancy, or stop-loss regressed; change route/exit logic instead of
  lowering gates.
- `stress_or_walk_forward_failed`: base edge exists but robustness failed;
  rerun bounded stress or quarantine.
