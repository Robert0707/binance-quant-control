# Professional Trading System Blueprint

Status: architecture and promotion gate. This document does not authorize live
entries.

## Why This Exists

The old failure mode was treating trading as an indicator-vote problem. The new
system treats trading as a lifecycle:

1. choose the tradable universe,
2. ingest point-in-time data and exchange rules,
3. generate features and labels with offline/online parity,
4. research independent alpha families,
5. backtest with realistic cost and walk-forward checks,
6. convert signals into portfolio targets,
7. run pre-trade risk before order creation,
8. execute through a broker/exchange adapter,
9. protect every position with stop, staged TP, runner/trailing, and time stop,
10. monitor fills, skipped signals, PnL, expectancy, and drawdown,
11. require structured review before promotion.

## External Patterns Adopted

- TradingAgents: use a structured review committee idea: analyst, bull case,
  bear case, trader, risk, and portfolio manager. This should become a
  machine-readable promotion review, not a free-form chat opinion.
- Lumibot: move toward same-strategy backtest/live parity through broker, data,
  order, position, and fill abstractions.
- OctoBot: keep crypto-specific workflows productized: paper/live/backtest,
  exchange adapters, optimizer, monitoring, and clear plugin/event boundaries.
- intelligent-trading-bot: make feature generation, labels, training, predict,
  and signal export reproducible so research and live do not drift.
- AI-Trader: defer OpenAPI/signal/copy surfaces until core strategy evidence is
  positive, then expose decisions through a formal contract.
- QuantConnect: keep universe, alpha, portfolio construction, risk management,
  and execution as separate modules.
- NautilusTrader: pre-trade risk is allowed to deny orders before execution.
- Hummingbot: model exits as a triple-barrier contract: take-profit, stop-loss,
  time-limit, and trailing.
- Freqtrade: protections such as cooldown, stoploss guard, max drawdown, and
  pair locks are circuit breakers, not suggestions.
- Binance: exchange filters are hard boundaries for price, quantity, percent
  price, and notional sizing.

## Current Keep / Refactor / Rebuild

Keep:
- symbol strategy map and asset routing,
- independent alpha-family research,
- expectancy-first professional entry gate,
- kill-switch, route-side quarantine, and protective repair,
- order journal, loss diagnostics, operator dashboard.

Refactor:
- split Binance data access from broker execution,
- require fee/spread/slippage evidence in every promotion report,
- record every skipped signal and order denial with the exact gate reason,
- make feature generation a manifest-driven offline/online parity layer.

Rebuild:
- portfolio construction: risk-budgeted targets, correlation caps, same-side
  exposure limits, core-10 portfolio promotion gate,
- broker-neutral order domain: `Order`, `Position`, `Fill`, `Broker`,
  `DataSource`,
- structured promotion review: bull/bear/risk/portfolio decision record.

New skeletons added:
- `trading_domain.py`: broker-neutral `OrderIntent`, `FillEvent`,
  `PositionSnapshot`.
- `portfolio_construction.py`: risk-budgeted `PortfolioTarget` acceptance and
  blocker generation.
- `feature_registry.py`: explicit feature and triple-barrier label manifest.
- `skipped_signal_journal.py`: append-only record for denied or skipped signals.

## Gate Command

```bash
openclaw-quantctl professional-system-audit --compact
```

This command does not trade and does not change execution config. It reports
which layers are ready, partial, missing, or blocked, then combines that with
the latest alpha and high-win iteration evidence.

The current expected result is still protective:

```text
trade_ready=false
execution_recommendation=block_new_entries_and_rebuild_edge
```

That is correct until enough symbol/family cohorts show positive expectancy,
profit factor, payoff ratio, realistic cost tolerance, and portfolio-level
coverage.
