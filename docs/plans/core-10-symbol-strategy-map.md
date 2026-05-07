# Core 10 Symbol Strategy Map

Status: paper/testnet research only. Mainnet live remains disabled.

## Research Basis

- Binance USD-M futures `exchangeInfo` confirms the current focus symbols are
  tradable perpetual contracts: BTCUSDT, ETHUSDT, XAUTUSDT, PAXGUSDT, BNBUSDT,
  SOLUSDT, XRPUSDT, LINKUSDT, AAVEUSDT, and TRXUSDT.
- TradingView-derived indicators are treated as filters inside one strategy
  family, not as duplicate votes. Bollinger, MACD, SuperTrend, Follow Line,
  Trend Magic, JUMBO Power, volume bubbles, and HTF imbalance signals should
  confirm trend/volatility/range state rather than all vote independently.
- Promotion has been raised to a hard 100-trade minimum per
  symbol-interval-family cohort. The 90% win / 10% stop-loss target is a gate,
  not a reported achievement.
- Local `core-10-mtf-strategy-map-refined-l1500` still does not meet the
  requested 90% win rate / 15% stop-loss-ratio target. Short-window improvement
  did not survive the longer 1500-bar check, so active lanes are now restricted
  to the small set that remains directionally acceptable.

## Symbol Routing

| Symbol | 15m | 1h | 4h | 1d | Reason |
| --- | --- | --- | --- | --- | --- |
| BTCUSDT | quarantined | quarantined | quarantined | quarantined | Current breakout/continuation lanes were negative; keep BTC as regime reference only. |
| ETHUSDT | quarantined | quarantined | quarantined | quarantined | 1h breakout failed the 1500-bar check; keep inactive until BTC-regime filter is added. |
| XAUTUSDT | quarantined | quarantined | quarantined | quarantined | 1h mean reversion failed the longer sample; needs macro/liquidity filters. |
| PAXGUSDT | quarantined | quarantined | quarantined | trend_continuation | Gold proxy; 1d trend was the only acceptable lane. |
| BNBUSDT | quarantined | quarantined | quarantined | quarantined | No active trade sample; keep symbol but require a new BNB-specific edge. |
| SOLUSDT | quarantined | quarantined | mean_reversion | quarantined | 4h mean reversion remains low-sample; breakout is quarantined. |
| XRPUSDT | quarantined | quarantined | mean_reversion | quarantined | 4h mean reversion remains below promotion but directionally acceptable. |
| LINKUSDT | quarantined | quarantined | quarantined | quarantined | 1h/1d candidates failed the longer sample; needs DeFi/oracle sector filter. |
| AAVEUSDT | quarantined | quarantined | quarantined | quarantined | 15m reversion failed the longer sample; needs DeFi risk filters. |
| TRXUSDT | quarantined | quarantined | mean_reversion | quarantined | Best local family remains 4h mean reversion, still sample-limited. |

## Current Backtest Reality

From `state/core-10-mtf-strategy-map-refined-l1500/alpha-research-ranking.json`:

- Long-sample refined map before final quarantine: 203 trades, 43.84% win,
  54.19% pure stop-loss ratio, and zero promotion-eligible cohorts.
- Survivors worth more research: PAXGUSDT 1d trend, TRXUSDT 4h mean reversion,
  XRPUSDT 4h mean reversion, and SOLUSDT 4h mean reversion.
- Failed long-sample lanes now quarantined: ETHUSDT 1h breakout, LINKUSDT 1h/1d,
  AAVEUSDT 15m, XAUTUSDT 1h, TRXUSDT 1d, and all BTC/BNB lanes.

The system should not open new testnet/live entries from this map until a
symbol-interval-family cohort passes promotion thresholds, slippage stress, and
walk-forward robustness. Quarantined lanes remain in the 10-symbol universe but
are skipped by alpha research.

## Operating Rule

One symbol-interval-family cohort is the unit of truth. A high score from
correlated technical indicators is not enough. Promotion requires:

- at least 100 trades in the exact symbol-interval-family cohort,
- PF above 1.50,
- win rate at least 90%,
- stop-loss ratio at most 10%,
- positive out-of-sample return,
- walk-forward pass,
- slippage-stress pass.
