# BTC/ETH TradingView Convergence Optimizer

Research only. This run does not modify live strategy config, live execution, or order code.

## Scope

- Symbols: `ETHUSDT`
- Market: `futures`
- Interval: `30m`
- Window: `2023-05-03` to `2026-05-03` UTC, end-exclusive
- Target win rate: `45.0%`
- Gate mode: `expectancy`
- Min train/test trades: `70` / `30`
- Min expectancy / payoff: `0.03%` / `1.2`
- Max per-trade risk: `2.5%`
- Regime filters: `strong_flow, quality_flow, trend_flow`
- Pre-screen rows / passed: `630` / `318`
- Full gate evaluations: `220`

## TradingView Concept Sources

- `tradingview_l5_whales_jump`: https://www.tradingview.com/script/6kRPcRVr-blackcat-L5-Whales-Jump-Out-of-Ocean-X/
- `tradingview_l3_banker_fund_flow`: https://www.tradingview.com/script/791WkWcm-blackcat-L3-Banker-Fund-Flow-Trend-Oscillator/
- `tradingview_supertrend`: https://www.tradingview.com/support/solutions/43000634738-supertrend/
- `tradingview_stoch_rsi`: https://www.tradingview.com/support/solutions/43000502333-stochastic-rsi-stoch-rsi/
- `binance_public_data`: https://github.com/binance/binance-public-data
- `binance_fapi_klines`: https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data

The Whale Jump L5 page is treated only as a public description source because the script is invite-only. The optimizer uses transparent local proxies built from public klines, volume, taker flow, MFI, JUMBO, Supertrend-style trend votes, VWAP, RSI/StochRSI, and Bollinger location.

## Signal Families

- `tv_supertrend_macd`: Trend-following resonance: Supertrend/TrendMagic/FollowLine, EMA, DI, and MACD alignment.
- `tv_stoch_rsi_pullback`: Trend-continuation pullback: StochRSI/RSI reset inside an EMA-aligned trend.
- `tv_vwap_trend`: VWAP trend continuation: price/VWAP/EMA alignment with volume and taker-flow confirmation.
- `tv_range_rsi`: Range mean-reversion: low-ADX Bollinger/StochRSI/RSI extremes with hostile-flow veto.
- `banker_flow_proxy`: Open-source Banker Fund Flow proxy: MFI, taker-flow, volume, JUMBO delta, and structure.

## Dataset

- `ETHUSDT` bars: `52608`
- `ETHUSDT` first/last: `2023-05-03T00:00:00+00:00` / `2026-05-02T23:30:00+00:00`
- `ETHUSDT` coverage: `0.9991` mature `True`
- `ETHUSDT` checksum ok files: `38`
- `ETHUSDT` missing files: `[]`

## Best By Symbol

- `ETHUSDT` `tv_vwap_trend` `L`: test win `40.62%`, test trades `32`, PF `1.8999`, expectancy `0.922%`, payoff `2.7767`, stop-loss ratio `46.88%`, WF mean win `40.3%`, regime `strong_flow`, gate passed `True`

## Best With Sample Floor

- `ETHUSDT` `tv_vwap_trend` `L`: test win `40.62%`, test trades `32`, train trades `76`, PF `1.8999`, expectancy `0.922%`, payoff `2.7767`, stop-loss ratio `46.88%`, WF mean win `40.3%`, regime `strong_flow`, gate passed `True`

## Decision

- Candidate count over gate: `4`
- Mature candidate count over gate: `4`
- Promotion allowed: `False`
- Execution recommendation: `continue_research_do_not_wire_live`

Next iteration:
- Continue with `tv_supertrend_macd` first; it produced the broadest pre-screen sample.
