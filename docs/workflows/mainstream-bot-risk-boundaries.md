# Mainstream Bot Risk Boundaries

Status: research and testnet gate only. This document does not authorize
mainnet live entries.

## Strict Conclusion

The current bot should trade only inside a small, auditable boundary:

- exchange filters pass before sizing,
- pre-trade risk decides before order creation,
- every entry has a stop, staged take-profit, runner/trailing, and time stop,
- drawdown, stop-loss streak, route-side PF, and missing protection can pause
  new entries,
- funding/open-interest crowding, news, whale flow, spread, and liquidity
  sweep failures are vetoes, not entry signals.

This keeps the system aligned with mainstream trading bots without pretending
that any public indicator can force a high win-rate edge. The production-style
gate is now expectancy-first: win rate is only useful when average winner,
average loser, PF, drawdown, and slippage all agree.

## Sources Checked

- Binance USD-M Futures exchange information exposes symbol status plus
  `PRICE_FILTER`, `LOT_SIZE`, `MIN_NOTIONAL`, and percent-price style filters;
  these must be sizing/order boundaries.
- Binance Futures quantitative rules expose account/order-rate risk indicators;
  live execution should treat exchange-side risk warnings as a block or
  cooldown, not as a log-only event.
- Binance funding-rate history makes funding a measurable crowding input.
- Freqtrade protections include cooldown, max-drawdown, stoploss guard, and
  low-profit pair style circuit breakers.
- Freqtrade stoploss/trailing docs support stop at entry and trailing only after
  a trade moves favorably.
- Hummingbot PositionExecutor/TripleBarrier practice supports take-profit,
  stop-loss, time-limit, and trailing as one exit contract.
- QuantConnect LEAN risk models include maximum drawdown and trailing stop
  style portfolio/position controls.
- NautilusTrader separates a pre-trade risk engine from execution, which is the
  correct production boundary for this repo too.

## Effective Loss-Control Boundaries

- Per-trade research risk: `0.6%` account risk in
  `strategy-core-high-win-research.yaml`.
- Absolute live ceiling: never above the user's `2.5%` per-trade risk ceiling.
- Testnet promotion: `100` completed trades, `>=65%` screening win rate
  (`>=70%` validation preference), `<=35%` pure stop-loss ratio, `PF>=1.5`,
  `expectancy_r>=0.10`, `payoff_ratio>=1.15`, positive OOS, walk-forward pass,
  and slippage stress pass.
- Elite label only: `>=80%` win rate is now an elite quality marker, not the
  basic permission to trade.
- Execution rejection: skip the trade if exchange min notional forces risk
  above the intended size.
- Portfolio pause: stop new entries on route-side PF failure, repeated
  stop-losses, daily drawdown, challenge drawdown, missing protection orders,
  high event risk, or unknown auth/exchange state.

## Profit-Maximizing Boundaries

Profit is maximized by staying alive through bad regimes:

- Use staged exits: TP1 reduces risk, TP2 pays the setup, runner only remains
  when trailing protection is active.
- Let only independent families compete: `trend_continuation`, `breakout`,
  `trend_pullback`, `liquidity_reclaim`, `mean_reversion`.
- Treat JUMBO, volume profile, Fib/OTE, funding, OI, news, whale flow, and BTC
  regime as filters or vetoes unless a family has its own 100-trade proof.
- Prefer symbol/timeframe cohorts that survive slippage and walk-forward over
  high headline win rates from small samples.

## New Family: Liquidity Reclaim

`liquidity_reclaim` is now a research family:

- Long setup: sweep below the prior 20-bar low, close back inside the range,
  close in the upper candle area, and confirm with volume or taker flow.
- Short setup: sweep above the prior 20-bar high, reject back inside the range,
  close in the lower candle area, and confirm with volume or taker flow.
- Veto: sweep without reclaim, low relative volume, strong trend against the
  reclaim, funding/OI crowding against the entry, or high event risk.

It is active only in paper research routes and still must pass the same
expectancy/PF/payoff/sample gate before any testnet candidate is allowed.

## Commands

```bash
openclaw-quantctl validate-config --strategy-config config/strategy-core-high-win-research.yaml
openclaw-quantctl alpha-research --config config/core-high-win-research.default.yaml --output-dir state/core-10-mainstream-boundary-l5000 --compact
openclaw-quantctl high-win-iteration --alpha-report state/core-10-mainstream-boundary-l5000/alpha-research-ranking.json --compact
openclaw-quantctl alpha-research --config config/core-replacement-scout.default.yaml --output-dir state/replacement-scout-mainstream-boundary-l5000 --compact
openclaw-quantctl operator-dashboard --compact
```

`high-win-iteration` is the control layer for the professional loop. It converts
the latest report into explicit blockers and next actions: collect more sample,
tighten structure/stoploss guard, run strict risk-combo sweep, or scout
replacement symbols. A passing strategy still requires live-readiness before
any testnet entry and never bypasses the mainnet live disabled default.

## Stop Rule

If the new family improves architecture but fails the 100-trade
expectancy/PF/payoff gate, do not open new paper/testnet entries. Keep existing
protected testnet positions guarded and continue research.
