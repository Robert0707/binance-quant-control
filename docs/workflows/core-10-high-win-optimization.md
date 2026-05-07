# Core-10 High-Win Optimization Workflow

Status: paper/testnet research only. Mainnet live stays disabled.

## Current Result

The new indicator stack is useful as a filter layer, but it is not a mature
strategy yet. The current promotion target is expectancy-first: at least 100
trades per symbol-interval-family cohort, `win_rate>=65%`,
`stop_loss_ratio<=35%`, `PF>=1.5`, `expectancy_r>=0.10`, and
`payoff_ratio>=1.15`. A win rate above 80% is an elite quality label, not the
basic permission to trade.

Latest long-sample core run with Fibonacci/OTE `trend_pullback`:
`state/core-10-trend-pullback-80-20-l5000/alpha-research-ranking.json`.

- 94 active trades across the active symbol map.
- 58.5095% weighted win rate.
- 41.4905% pure stop-loss ratio.
- finite average PF 0.6314.
- zero promoted cohorts.

The conclusion is strict: the new pullback lane is valid as a research family,
but it is not a production strategy. ETH 4h improved directionally but only
produced 4 trades. BTC 1h/4h stayed negative and is quarantined again.

2026-05-03 update: mainstream bot risk-boundary review added
`liquidity_reclaim` as a new research family. This is a failed-breakout /
sweep-reclaim structure lane, not a permission to open new testnet entries.
It must pass the same 100-trade expectancy/PF/payoff/slippage/walk-forward gate.

## Hard Promotion Gate

A symbol-interval-family cohort must pass all of these before testnet promotion:

- at least 100 completed trades in the cohort,
- route-side historical gate is not blocking the side,
- historical score/convergence feedback bucket is not blocking the entry class,
- win rate at least 65%,
- pure stop-loss ratio at most 35%,
- profit factor at least 1.50,
- expectancy at least `0.10R` per trade,
- payoff ratio at least `1.15`,
- positive out-of-sample return,
- walk-forward robustness pass,
- slippage stress remains positive,
- external context is not high-risk or whale/news opposed.

Short samples are now rejected even if they show 100% win rate.
Routes and sides with enough closed-review history but PF below `0.8`, negative
net PnL, or stop-loss ratio above `70%` are quarantined from the research entry
stream until a separate positive bucket emerges.

`risk-combo-sweep` uses the same recovery gate when the corresponding CLI
options are passed. A parameter combination is not a recovery candidate unless
its test and full slices also meet the trade-count, win-rate, stop-loss-ratio,
PF, expectancy, and payoff requirements.

## Strategy Families

Keep the official strategy surface small:

- `trend_continuation`: EMA/MACD/ADX/SuperTrend/Follow Line alignment, no duplicate voting.
- `breakout`: Bollinger expansion, volume/imbalance confirmation, no late chase without volume.
- `trend_pullback`: Fibonacci 0.382-0.786 pullback/OTE zone in an existing trend, with oscillator reset and JUMBO/volume as filters.
- `liquidity_reclaim`: 20-bar liquidity sweep, close back inside the range, volume/taker-flow confirmation, and funding/OI/event-risk vetoes.
- `mean_reversion`: Bollinger percent-B extremes, range ADX, ATR stop, route-side feedback.

JUMBO Power, Trend Magic, Volume Bubbles, volume profile, HTF imbalance, funding,
OI, news, whale flow, exchange filters, and route-side risk are filters or
vetoes, not independent votes.

## Commands

1. Validate config:

```bash
openclaw-quantctl validate-config --strategy-config config/strategy-core-high-win-research.yaml
```

2. Collect external context:

```bash
openclaw-quantctl external-context --config config/external-context.default.yaml --compact
```

3. Run the BTC/ETH/XAUT whale-jump proxy lane:

```bash
python3 scripts/research_ocean_x_btc_evidence.py --optimize-core-whale-jump --symbols BTCUSDT,ETHUSDT,XAUTUSDT --start 2024-05-03 --end 2026-05-03 --interval 1h --target-win-rate 80 --min-train-trades 70 --min-test-trades 30 --min-profit-factor 1.5 --max-stop-loss-ratio 20 --regime-filters none,trend,pullback,liquidity,range,strong_flow --max-configs 192
```

This is research only and does not write live strategy config or open orders.
Treat `candidate_count=0` as a valid protective result. The 2026-05-04
standard run found no mature candidate: BTC/ETH had mature data but failed the
PF / expectancy / payoff / sample-quality gate, while XAUT public history was
too short for the standard sample floor.

For the BTC/ETH-only TradingView convergence lane, run:

```bash
python3 scripts/research_ocean_x_btc_evidence.py --optimize-btc-eth-tradingview --symbols BTCUSDT,ETHUSDT --start 2024-05-03 --end 2026-05-03 --interval 1h --target-win-rate 80 --min-train-trades 70 --min-test-trades 30 --min-profit-factor 1.5 --max-stop-loss-ratio 20 --max-per-trade-risk-pct 2.5 --max-full-evaluations 0 --regime-filters none,trend,pullback,liquidity,range,strong_flow
```

This lane converts public TradingView concepts into transparent local filters
and still treats `candidate_count=0` as a protective result. The 2026-05-04 run
found no pass after all `552` pre-screened candidates: BTC best sample-floor row
was `tv_vwap_trend` long at `73.33%` test win over `30` trades, and ETH best was
`tv_supertrend_macd` short at `71.08%` test win over `83` trades. A BTC
`tv_supertrend_macd` short row reached `82.35%`, but only over `17` test trades,
so it remains rejected by the sample gate.

4. Run the core 10 multi-timeframe research lane:

```bash
openclaw-quantctl alpha-research --config config/core-high-win-research.default.yaml --output-dir state/core-10-high-win-l5000 --compact
```

5. Run replacement-symbol scout when the current 10-symbol map cannot pass the expectancy gate:

```bash
openclaw-quantctl alpha-research --config config/core-replacement-scout.default.yaml --output-dir state/replacement-scout-expectancy-l5000 --compact
```

This keeps BTC/ETH/XAUT as anchors and searches high-volume Binance futures for
symbols that fit the same strategy families better. Do not replace a core symbol
unless the replacement passes the same 100-trade, PF, expectancy, payoff,
win-rate, stop-loss, slippage, and walk-forward gates.

6. Run focused parameter sweeps only for surviving paper cohorts:

```bash
openclaw-quantctl risk-combo-sweep --symbols PAXGUSDT,ETHUSDT,XRPUSDT,TRXUSDT --limit 5000 --grid-mode focused --min-test-trades 100 --min-win-rate 65 --max-stop-loss-ratio 35 --target-profit-factor 1.5 --min-expectancy-r 0.10 --min-payoff-ratio 1.15 --max-configs 80 --max-walk-forward-validations 12 --skip-news --compact
```

For the current expectancy-first objective, pass the full gate explicitly:

```bash
openclaw-quantctl risk-combo-sweep --symbols PAXGUSDT,ETHUSDT,XRPUSDT,TRXUSDT --limit 5000 --grid-mode focused --min-test-trades 100 --min-win-rate 65 --max-stop-loss-ratio 35 --target-profit-factor 1.5 --min-expectancy-r 0.10 --min-payoff-ratio 1.15 --max-configs 80 --max-walk-forward-validations 12 --skip-news --compact
```

This is a long-running batch job. If it cannot complete interactively, run it as
a scheduled/background research task and use the report under
`state/risk-combo-sweeps/` as the only source of truth.
The core research profile uses `exit_profile: payoff_runner`, and the sweep
keeps exit-profile coverage even with `--max-configs`, so bounded runs still
test payoff-friendly exits instead of only the first lexicographic grid rows.

7. Build the next PID-like research iteration plan:

```bash
openclaw-quantctl high-win-iteration --alpha-report state/core-10-high-win-l5000/alpha-research-ranking.json --compact
```

When comparing a short smoke report against expanded samples, pass every report
into the same controller call. Both comma-separated and repeated flags are
accepted:

```bash
openclaw-quantctl high-win-iteration \
  --alpha-report state/core-10-route-side-gate-smoke-v2/alpha-research-ranking.json,state/trx-4h-route-side-gate-l3000/alpha-research-ranking.json,state/sol-4h-route-side-gate-l3000/alpha-research-ranking.json \
  --no-write-pid-state \
  --compact
```

If `reject-short-sample-regression` appears, the short report had a cohort that
looked target-shaped but failed after sample expansion. Treat the short-sample
cohort as rejected until the expanded report also passes the full 100-trade
expectancy/PF/payoff gate.

If `expand-target-shaped-under-sampled-cohorts` appears, those cohorts already
match the expectancy/PF/payoff shape but do not yet have 100 trades. Expand only those
symbol/interval/family cohorts first. This is the preferred convergence path
before another broad replacement scout.

For long focused expansions, use the background runner so the session does not
block on a 20k+ candle job:

```bash
python3 scripts/run_high_win_candidate_expansion.py start
python3 scripts/run_high_win_candidate_expansion.py status
python3 scripts/run_high_win_candidate_expansion.py evaluate
```

The default background job expands the current focused candidates
`NAORISUSDT:1h:mean_reversion` and `APEUSDT:4h:mean_reversion` to the estimated
100-trade sample size. It only runs alpha research and never opens orders.

This controller reads alpha / risk-combo reports, measures the gap to the
100-trade / >=65% win / <=35% pure stop-loss / PF>=1.5 /
expectancy_r>=0.10 / payoff_ratio>=1.15 gate, and emits the next
batch commands. It only changes research pressure: suggested sample size,
confirmation pressure, strict sweep focus, and replacement-scout priority. It
does not open orders, edit live execution settings, or enable mainnet.

8. Keep converging when the gate still fails:

Plan-only mode:

```bash
openclaw-quantctl high-win-converge --alpha-report state/core-10-high-win-l5000/alpha-research-ranking.json --max-rounds 1 --compact
```

Bounded research execution:

```bash
openclaw-quantctl high-win-converge --alpha-report state/core-10-high-win-l5000/alpha-research-ranking.json --max-rounds 1 --execute-research --compact
```

This command loops through the same strict controller. If no cohort passes, it
runs only bounded backtest/research batches: core sample expansion,
replacement-scout, and strict risk-combo sweep. It stops on promotion, max
rounds, failed research jobs, or stagnation. It does not open orders or change
live execution settings.

9. Re-check historical/live state:

```bash
openclaw-quantctl operator-dashboard --compact
openclaw-quantctl loss-diagnostics --compact
python3 scripts/run_strategy_optimizer.py --config config/strategy-optimizer.default.yaml
```

## External Context

The optional context layer is now configured in `config/external-context.default.yaml`:

- CoinMarketCap: global market metrics and capital-flow regime.
- Arkham: entity/wallet flow when `ARKHAM_API_KEY` is available.
- DexScreener: on-chain hot-token rotation and DEX attention.
- CryptoPanic: symbol-filtered news/event risk.
- Glassnode: BTC/ETH on-chain macro metrics.

Missing keys are neutral. External context can down-rank or veto a trade, but it
does not create an entry.

## Framework Lessons Applied

- Jesse: route each strategy/symbol/timeframe independently, then stress test
  with optimization and Monte Carlo-style robustness before promotion.
- NautilusTrader: keep research, paper, and live semantics aligned; do not let
  a backtest use an exit model that live execution does not use; risk checks
  must be pre-trade gates, not post-trade explanations.
- Freqtrade / Hummingbot / QuantConnect: cooldown, stoploss guard, max drawdown,
  kill switch, position limits, and triple-barrier exits are trading boundaries,
  not optional reports.
- FinRL: keep train/test/trade separated. No AI/RL model enters execution until
  it improves the deterministic baseline out of sample.
- TradingView/Fibonacci: 0.618/0.786 OTE zones are structure filters only. They
  require trend context, oscillator reset, and volume/JUMBO confirmation; they
  do not override the promotion gate.
- Binance Futures: exchangeInfo filters, funding/crowding context, quantitative
  rule warnings, and reduce-only protection are hard execution boundaries.

## Stop Rule

If no cohort reaches 100 trades while keeping positive expectancy, adequate
payoff, and PF, do not rescue it with a headline win-rate filter. Add more
history, find a better symbol that fits the same family, improve exits, or leave
the lane quarantined.
