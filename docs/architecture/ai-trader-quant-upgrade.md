# AI Trader Quant Upgrade

Status: research and machine-gate architecture. This document does not
authorize live entries.

## Design Sources

- Qlib style: dataset, feature, model, strategy, and backtest are separate
  replayable stages.
- FinRL style: market state and turbulence belong in the agent state, not in a
  human chart note.
- QuantConnect / Nautilus style: universe, alpha, portfolio, risk, execution,
  and monitoring must remain separate gates.
- Hummingbot V2 style: strategy controllers decide intent; executors manage
  order lifecycle and protection.
- Freqtrade style: protections such as cooldown, stoploss guard, and max
  drawdown are first-class controls.

## Machine Objective

The AI trader optimizes fixed-risk expectancy, not indicator agreement.

Primary objective:

```text
maximize E[R] after cost, slippage, drawdown, loss-streak, and portfolio caps
```

Do not optimize for raw win rate unless payoff and expectancy remain positive.

## Feature Families

The feature dataset now includes three AI-native feature groups:

- `ml_regime_state`: volatility, trend, range position, turbulence, session.
- `ml_execution_quality`: liquidity pressure, volume/quote-volume z-score,
  candle range, wick imbalance.
- `ml_payoff_potential`: long/short distance-to-reward in ATR units.

These features are point-in-time and replayable. They are intended for
meta-labeling, route veto, symbol allocation, and exit/payoff optimization.

## Symbol Allocation

Symbols are treated as arms:

- exploit proven positive-expectancy arms,
- explore high-liquidity unresolved arms,
- quarantine negative-expectancy route-side arms,
- promote only portfolio-grade cohorts.

The command surface is:

```bash
openclaw-quantctl ai-expectancy-upgrade --universe-limit 20 --limit 8000 --sweep-limit 5000 --max-configs 80 --max-walk-forward-validations 12 --compact
```

## Promotion Rule

A strategy surface can move forward only when it leaves:

- feature manifest hash,
- triple-barrier labels,
- alpha row,
- market-bot gate result,
- machine directive,
- portfolio gate result,
- loss-diagnostics veto check,
- live-readiness dry-run.

Any missing stage keeps the surface in research, paper, or testnet review.
