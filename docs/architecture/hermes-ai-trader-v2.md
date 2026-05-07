# Hermes AI Trader v2

Status: clean architecture gate. This document does not authorize live entries.

## Core Boundary

Hermes AI Trader v2 is not a bigger CLI wrapper. It is a clean decision system:

- domain: broker-neutral `Broker`, `DataSource`, `OrderIntent`, `FillEvent`,
  `OrderSnapshot`, `PositionSnapshot`,
- event lifecycle: universe, features, alpha, committee, portfolio, risk,
  execution, monitoring,
- feature system: manifest-driven features and triple-barrier labels,
- model registry: CPU/Python models, optional Hailo chart inference, optional
  Hermes narrative review,
- signal layer: standard `TradingSignal`,
- local API layer: append-only `trading_signal.v1` JSONL ledger for dashboard,
  skill, and later OpenAPI sync,
- committee: analyst, bull case, bear case, risk manager, portfolio manager,
- open-order gate: only approves when architecture, alpha, portfolio, risk,
  committee, and live-readiness all pass.

## What Can Open A Trade

The sequence is hard:

1. universe and symbol strategy map approve the symbol,
2. features are live-safe and reproducible,
3. alpha cohort has at least 100 trades,
4. expectancy is positive after cost,
5. profit factor and payoff ratio pass,
6. portfolio construction accepts the target risk,
7. pre-trade risk accepts the order before creation,
8. structured committee does not reject,
9. live-readiness says `allowed=true`,
10. operator explicitly executes.

Any single failure means the signal is written to the skipped-signal journal.

Win rate is no longer the primary open-order objective. It remains a screening
metric, while the gate is driven by fixed-risk net expectancy, payoff ratio,
profit factor, sample count, and portfolio risk budget.

## Hailo Use

Good Hailo tasks:

- chart image regime triage,
- candlestick image anomaly veto after labels exist,
- chart-quality screening for reports.

Bad Hailo tasks:

- pandas/NumPy indicator acceleration,
- backtest loop acceleration without a compiled model,
- final order approval.

Hailo can veto or triage. It cannot bypass alpha, risk, or exchange gates.

## Legacy Strategy Policy

Keep old strategy pieces only when they are useful evidence or reusable logic:

- keep independent alpha families,
- keep symbol-specific strategy map,
- keep risk gates and protective exits,
- keep closed-trade review history.

Quarantine:

- negative expectancy cohorts,
- tiny-sample PF-infinite rows,
- duplicate indicator voting,
- configs with no slippage/walk-forward evidence.

## Main Command

```bash
openclaw-quantctl hermes-ai-trader --compact
```

Expected current result:

```text
open_order_gate.allowed=false
reason=alpha still has no promotion-eligible cohort
```

## Current Alpha Truth

The current mapped l1500 evidence is not tradable: 6 rows, 5 trades,
`PF=0.5156`, `expectancy=-0.0641R`, payoff ratio `0.3437`, promotion eligible
count `0`. Hermes AI Trader v2 can organize and veto signals, but it must not
turn that evidence into an order.
