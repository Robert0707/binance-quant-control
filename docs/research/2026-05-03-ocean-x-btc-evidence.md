# Ocean X BTC Evidence Research

Date: 2026-05-03

Status: research candidate only. Do not wire this into live trading yet.

## Request Boundary

The user asked to research an Ocean-X-style whale indicator for the trading
program, starting with the last two years of BTC, but explicitly said not to add
it to the trading program yet.

## Evidence Sources

- Original TradingView indicator page:
  https://www.tradingview.com/script/6kRPcRVr-blackcat-L5-Whales-Jump-Out-of-Ocean-X/
- Related open-source TradingView reference:
  https://www.tradingview.com/script/791WkWcm-blackcat-L3-Banker-Fund-Flow-Trend-Oscillator/
- Binance public data repository:
  https://github.com/binance/binance-public-data
- Binance USD-M futures kline docs:
  https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data

The L5 indicator is closed-source / invite-only, so the research uses a
transparent local proxy. It does not copy the closed-source formula.

## Local Artifacts

- Main report:
  `reports/20260503T025558Z-btcusdt-futures-ocean-x-evidence/research.md`
- Machine summary:
  `reports/20260503T025558Z-btcusdt-futures-ocean-x-evidence/summary.json`
- Source manifest and checksum proof:
  `reports/20260503T025558Z-btcusdt-futures-ocean-x-evidence/source_manifest.json`
- Event rows:
  `reports/20260503T025558Z-btcusdt-futures-ocean-x-evidence/ocean_proxy_events.csv`
- Repro script:
  `scripts/research_ocean_x_btc_evidence.py`

## Data Window

- Symbol: `BTCUSDT`
- Market: Binance USD-M futures public klines
- Requested window: 2024-05-03 to 2026-05-03 UTC, end-exclusive
- Actual latest rows:
  - `1h`: 2026-05-01 23:00 UTC
  - `4h`: 2026-05-01 20:00 UTC
- Missing official files at run time:
  - `BTCUSDT-1h-2026-05-02.zip`
  - `BTCUSDT-4h-2026-05-02.zip`
- Manifest result: 50 fetched files, 50 checksum matches, 2 expected 404s.

## Proxy Logic

The proxy combines:

- volume spike: `volume_zscore_20 >= 1.8` or `volume_ratio_20 >= 1.45`
- extreme volume: `volume_zscore_20 >= 2.4` or `volume_ratio_20 >= 2.0`
- taker flow: buyer or seller share threshold `0.58`
- money flow confirmation: `mfi_14`
- local transparent JUMBO-style composite fields
- Fibonacci 89 pullback / OTE zones
- liquidity reclaim events

Signals:

- `L`: long-side whale pressure
- `S`: short-side whale pressure
- `XL`: long pressure plus extreme volume and structure
- `XS`: short pressure plus extreme volume and structure

## Key Findings

`1h` had enough events to continue research:

- `L`: 122 events. At 24 bars / 1 day: win 49.59%, avg +0.0717%, median -0.0456%.
- `S`: 69 events. At 24 bars / 1 day: win 66.67%, avg +0.7649%, median +0.6512%.
- `XL`: only 5 events. Positive but too small to trust.
- `XS`: only 1 event. Ignore until more data or looser thresholds.

`4h` did not produce enough events:

- `L`: 1 event.
- `S` / `XL` / `XS`: 0 events.

## Decision

This is a research candidate, not a deployable trading module.

Best next step is a walk-forward / out-of-sample study for the `1h S` signal,
including fees, slippage, funding, existing BTC route gates, and the 2.5%
per-trade risk ceiling. Do not connect this to live execution until that passes.
