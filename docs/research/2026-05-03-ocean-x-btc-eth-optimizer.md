# BTC/ETH/XAUT Whale Jump Optimizer Iteration

Date: 2026-05-03

Status: no live candidate. Continue research only.

## Request Boundary

The requested strategy scope is now BTC, ETH, and XAUT, using the existing
Ocean-X / whale-jump proxy as the base. The 80% win-rate target is a research
gate, not a promise and not a live setting. No strategy YAML used by live
execution was modified.

## What Changed

- Extended `scripts/research_ocean_x_btc_evidence.py` with
  `--optimize-btc-eth`.
- 2026-05-04: added `--optimize-core-whale-jump`, BTC/ETH/XAUT defaults,
  regime filters, max stop-loss-ratio gate, and dataset-maturity checks.
- Added a research-only config:
  `config/ocean-x-btc-eth-optimizer.default.yaml`.
- Produced optimizer reports under:
  - `reports/20260503T124854Z-btc-eth-ocean-x-optimizer/`
  - `reports/20260503T171144Z-btc-eth-ocean-x-optimizer/`
  - `reports/20260504T-core-whale-jump-standard/`

## Data Evidence

Both optimizer runs used Binance public USD-M futures klines:

- Symbols: `BTCUSDT`, `ETHUSDT`
- Interval: `1h`
- Window: 2024-05-03 to 2026-05-03 UTC, end-exclusive
- Bars per symbol: `17520`
- Source files: `52`
- Checksum matches: `52`
- Missing files: none

External source references:

- Original TradingView L5 page:
  https://www.tradingview.com/script/6kRPcRVr-blackcat-L5-Whales-Jump-Out-of-Ocean-X/
- Binance public data:
  https://github.com/binance/binance-public-data
- Binance USD-M futures kline docs:
  https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data

## Iteration Results

Baseline sample sweep:

- Report: `reports/20260503T124854Z-btc-eth-ocean-x-optimizer/optimizer.md`
- Evaluations: `1104`
- Candidate count over gate: `0`
- BTC best with sample floor: `S`, test win `72.73%`, test trades `55`,
  train trades `119`, test PF `1.1193`, walk-forward mean win `72.11%`.
- ETH best with sample floor: `L`, test win `65.08%`, test trades `63`,
  train trades `114`, test PF `0.8308`, walk-forward mean win `61.23%`.

Uniform-space sweep:

- Report: `reports/20260503T171144Z-btc-eth-ocean-x-optimizer/optimizer.md`
- Evaluations: `567`
- Candidate count over gate: `0`
- BTC best with sample floor: `S`, test win `85.71%`, test trades `7`,
  train trades `20`, test PF `8.2623`, walk-forward mean win `56.6%`.
- ETH best with sample floor: `L`, test win `77.78%`, test trades `27`,
  train trades `40`, test PF `1.6618`, walk-forward mean win `68.09%`.

## 2026-05-04 Core Whale-Jump Standard Run

Command:

```bash
python3 scripts/research_ocean_x_btc_evidence.py --optimize-core-whale-jump --symbols BTCUSDT,ETHUSDT,XAUTUSDT --start 2024-05-03 --end 2026-05-03 --interval 1h --target-win-rate 80 --min-train-trades 70 --min-test-trades 30 --min-profit-factor 1.5 --max-stop-loss-ratio 20 --regime-filters none,trend,pullback,liquidity,range,strong_flow --max-configs 192 --output-dir reports/20260504T-core-whale-jump-standard
```

Result:

- Report: `reports/20260504T-core-whale-jump-standard/optimizer.md`
- Evaluations: `2370`
- Candidate count over gate: `0`
- Mature candidate count over gate: `0`
- BTC mature data: yes, `17520` 1h bars, coverage `0.9986`.
- ETH mature data: yes, `17520` 1h bars, coverage `0.9986`.
- XAUT mature data: no, `898` 1h bars, coverage `0.0507`; public history
  begins locally at `2026-03-26T14:00:00+00:00`.

Best sample-floor rows:

- BTCUSDT `L`, regime `none`: train `81` trades / `69.14%` win / PF `1.0381`;
  test `34` trades / `73.53%` win / PF `1.2717`; test stop-loss ratio
  `23.53%`. It fails win-rate, PF, and stop-loss-ratio gates.
- ETHUSDT `L`, regime `none`: train `76` trades / `50.0%` win / PF `0.6667`;
  test `44` trades / `59.09%` win / PF `1.037`; test stop-loss ratio
  `36.36%`. It fails win-rate, PF, walk-forward, and stop-loss-ratio gates.
- XAUTUSDT: no parameter set met train/test sample floors on 1h futures or
  spot. The blocker is both short public history and weak trade sample.

## Decision

Do not add this to the trading program yet.

The proxy can still produce attractive small-sample 100% rows, but the standard
gate rejects them. The best sample-floor BTC result is below 80% and PF `1.5`;
ETH is weaker; XAUT does not have mature public history yet. This is now a
finished research-only strategy artifact, not a live candidate.

## Next Iteration

- Split BTC short and ETH long into separate parameter lanes.
- Add regime filters before more TP/SL tuning.
- Keep XAUT in the run but treat it as data-immature until public history covers
  enough train/test/walk-forward windows.
- Run any 15m BTC/ETH expansion as a background research task; it is too slow
  for regular chat-loop execution.
- Stress any future candidate with fees, slippage, and funding.
- Keep the 2.5% per-trade risk ceiling if this ever graduates from research.

## 2026-05-04 BTC/ETH TradingView Convergence Run

The user narrowed the lane to BTC/ETH and asked to use public TradingView ideas
for further convergence. The new optimizer keeps the same research-only
boundary and adds transparent signal families instead of copying closed scripts:

- `tv_supertrend_macd`
- `tv_stoch_rsi_pullback`
- `tv_vwap_trend`
- `tv_range_rsi`
- `banker_flow_proxy`

Command:

```bash
python3 scripts/research_ocean_x_btc_evidence.py --optimize-btc-eth-tradingview --symbols BTCUSDT,ETHUSDT --start 2024-05-03 --end 2026-05-03 --interval 1h --target-win-rate 80 --min-train-trades 70 --min-test-trades 30 --min-profit-factor 1.5 --max-stop-loss-ratio 20 --max-per-trade-risk-pct 2.5 --max-full-evaluations 0 --regime-filters none,trend,pullback,liquidity,range,strong_flow --output-dir reports/20260504T-btc-eth-tradingview-convergence-full
```

Result:

- Report: `reports/20260504T-btc-eth-tradingview-convergence-full/optimizer.md`
- Pre-screen rows / passed: `2880` / `552`
- Full gate evaluations: `552`
- Candidate count over gate: `0`
- Mature candidate count over gate: `0`
- BTC/ETH data mature: yes, `17520` 1h bars each, coverage `0.9986`.

Best sample-floor rows:

- BTCUSDT `tv_vwap_trend` `L`, regime `trend`: train `103` trades /
  `56.31%` win / PF `0.9802`; test `30` trades / `73.33%` win / PF `1.5842`;
  test stop-loss ratio `26.67%`. It fails train/test win-rate, train PF,
  walk-forward, and stop-loss-ratio gates.
- ETHUSDT `tv_supertrend_macd` `S`, regime `none`: train `252` trades /
  `61.11%` win / PF `1.0012`; test `83` trades / `71.08%` win / PF `1.5853`;
  test stop-loss ratio `26.51%`. It fails win-rate, train PF, walk-forward,
  and stop-loss-ratio gates.

Highest win-rate low-sample row:

- BTCUSDT `tv_supertrend_macd` `S`, regime `strong_flow`: train `34` trades /
  `64.71%` win / PF `1.3761`; test `17` trades / `82.35%` win / PF `2.8681`;
  test stop-loss ratio `17.65%`. It is rejected because train/test sample
  counts are below `70` / `30`, train win-rate is below `80%`, and
  walk-forward min win-rate is too low.

Decision remains unchanged: no live wiring, no order opening, and no gate
relaxation. The next focused iteration should test 15m or 30m to expand sample
while tightening `tv_supertrend_macd` / `tv_vwap_trend` around flow and
liquidity confirmation.

## 2026-05-04 Expectancy-First Gate Revision

The high-win gate was replaced for the BTC/ETH TradingView lane. Win rate is
now descriptive; it is no longer the hard promotion condition. The finished
research gate is:

- `gate_mode=expectancy`
- `min_train_trades=70`
- `min_test_trades=30`
- `min_profit_factor=1.2`
- `min_expectancy_pct=0.03`
- `min_payoff_ratio=1.2`
- `max_drawdown_pct=20`
- `max_loss_streak=8`
- `max_stop_loss_ratio=55`
- `max_per_trade_risk_pct=2.5`

Command:

```bash
python3 scripts/research_ocean_x_btc_evidence.py --optimize-btc-eth-tradingview --symbols BTCUSDT,ETHUSDT --start 2024-05-03 --end 2026-05-03 --interval 1h --gate-mode expectancy --target-win-rate 45 --min-train-trades 70 --min-test-trades 30 --min-profit-factor 1.2 --min-expectancy-pct 0.03 --min-payoff-ratio 1.2 --max-drawdown-pct 20 --max-loss-streak 8 --max-stop-loss-ratio 55 --max-per-trade-risk-pct 2.5 --max-full-evaluations 0 --regime-filters none,trend,pullback,liquidity,range,strong_flow --output-dir reports/20260504T-btc-eth-tradingview-expectancy-rr-1h-full
```

Result:

- Report: `reports/20260504T-btc-eth-tradingview-expectancy-rr-1h-full/optimizer.md`
- Candidate count over expectancy gate: `0`
- Mature candidate count over expectancy gate: `0`
- BTC best visible row: `tv_supertrend_macd` short, test `17` trades /
  `82.35%` win / PF `2.8681` / expectancy `0.6329%`, but payoff only
  `0.6146` and sample is below floor.
- ETH best visible row: `banker_flow_proxy` long, test `23` trades /
  `73.91%` win / PF `1.6781` / expectancy `0.2239%`, but payoff only
  `0.5923` and sample is below floor.

Decision: the final program is now structurally correct and expectancy-first,
but current BTC/ETH public-data candidates are still not promotable. Do not
wire to live. The next valid step is paper/testnet candidate generation with
better payoff structure, not lowering the gate or restoring an 80% win-rate
rule.
