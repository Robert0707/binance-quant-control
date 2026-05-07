# Quant Stack Research - 2026-04-25

## Objective

Build a low-token, schedule-first crypto research stack that combines:

- existing `binance-quant-control` technical analysis
- whale and wallet monitoring
- macro/event veto logic based on current world conditions
- Hailo as a local accelerator only where it actually saves cloud/API cost

## Current Local Baseline

- `openclaw-quantctl doctor` succeeded on 2026-04-25.
- Current local sample run on 2026-04-25 for `BTCUSDT` futures `1h` produced:
  - bias: `short-bias`
  - regime: `range`
  - convergence: `0.667`
  - close: `77637.6`
- The existing project already supports:
  - public Binance analysis
  - optional Blave enrichment
  - workflow spec generation
  - paper-order journaling

## GitHub Tool Shortlist

### Tier 1: Keep the stack lean

1. [ccxt/ccxt](https://github.com/ccxt/ccxt)
   - Best use here: unified exchange API layer and adapter fallback.
   - Why it fits: broad exchange coverage and stable Python support.
   - Source note: CCXT describes itself as a crypto trading API for more than 100 exchanges.

2. [bmoscon/cryptofeed](https://github.com/bmoscon/cryptofeed)
   - Best use here: websocket market data collector without bolting live execution into the same process.
   - Why it fits: ideal for low-latency order book, trades, and funding watchers.
   - Source note: the project is explicitly a cryptocurrency exchange websocket data feed handler.

3. [freqtrade/freqtrade](https://github.com/freqtrade/freqtrade)
   - Best use here: dry-run, backtesting, hyperopt, and strategy harvesting from the community.
   - Why it fits: strong for paper trading and parameter search before anything touches live capital.
   - Source note: Freqtrade lists dry-run, backtesting, and strategy optimization, and supports Python 3.11+.

4. [polakowo/vectorbt](https://github.com/polakowo/vectorbt)
   - Best use here: fast offline research and parameter sweeps.
   - Why it fits: use it to kill weak ideas quickly before promoting them into the controlled workflow lane.
   - Source note: VectorBT emphasizes large-scale experimentation across assets and timeframes in a few lines of code.

### Tier 2: Add only after the base lane is stable

5. [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader)
   - Best use here: promotion path from research to deterministic event-driven execution.
   - Why it fits: suitable if this grows beyond single-bot analysis into a more formal execution engine.
   - Source note: NautilusTrader positions itself as a production-grade deterministic event-driven architecture.

6. [hummingbot/hummingbot](https://github.com/hummingbot/hummingbot)
   - Best use here: only if we intentionally move into market making or DEX routing.
   - Why it is not first choice: heavier operational surface area than we need for directional BTC research.

## Strategy Convergence Recommendation

### Recommended factor weights

- technical structure: `0.45`
- whale / wallet flow: `0.25`
- macro / event veto: `0.20`
- derivatives positioning: `0.10`

### Decision rules

- `risk-off / no-trade` if macro veto is high, even when technicals look clean.
- Promote `breakout` only when:
  - technical trend is aligned
  - whale data shows net exchange outflow or accumulation
  - derivatives are not showing crowded long risk
- Promote `mean-reversion` only when:
  - macro risk is low
  - volatility is compressed or range-like
  - whale data is neutral instead of aggressively directional
- Penalize longs when whale transfers are flowing into exchanges faster than the 30-day baseline.
- Penalize shorts when stablecoin mint or exchange outflow implies fresh buying power.

## World Regime Snapshot As Of 2026-04-25

### Macro

- U.S. March 2026 CPI was released on 2026-04-10:
  - CPI `+0.9%` month over month
  - CPI `+3.3%` year over year
  - energy `+10.9%` month over month
  - gasoline `+21.2%` month over month
- Fed March 18, 2026 minutes say the target range was left unchanged, and the next FOMC meeting is scheduled for `2026-04-28` to `2026-04-29`.
- The IEA April 2026 oil report says early-April Strait shipments stayed severely restricted around `3.8 mb/d` versus more than `20 mb/d` before the crisis, while alternative routes rose to `7.2 mb/d`.

### Crypto regime read

- The inflation impulse is energy-led, so leverage should stay smaller whenever crude and shipping risk are rising together.
- The calendar is event-heavy into the late-April FOMC window, so event-risk vetoes should matter more than normal this week.
- Binance Research's April 2026 market insights highlight both `BTC/ETH` relative resilience during the Middle East conflict and continued long-term holder accumulation, so the medium-term tape is not uniformly bearish even when 1h structure softens.

## Whale And Wallet Layer

### Priority sources

1. Existing Blave lane
   - Already partially integrated into this repo via `holder_concentration` and `whale_hunter`.
   - Keep this as the first optional enrichment if credentials are available.

2. [Whale Alert API](https://developer.whale-alert.io/documentation/)
   - Fit: large transfer stream from REST + websocket.
   - Best use: thresholded alerts only, not firehose ingestion into chat.

3. [Arkham Alerts](https://codex.arkm.com/the-intelligence-platform/alerts)
   - Fit: wallet-specific alerts with direct transaction links and webhook delivery.
   - Best use: label and watch known exchange wallets, ETF-related entities, custodians, and recurring whales.

### Signals that matter

- exchange inflow spike: bearish short-term
- exchange outflow spike: bullish or at least sell-pressure reduction
- stablecoin mint + exchange inflow: bullish liquidity
- unknown-to-unknown transfer: mostly noise unless repeated by labeled entities
- ETF/custodian accumulation wallets: medium-term structural bid

## Hailo / AI HAT Guidance

### What Hailo should do

- local batch summarization of scheduled macro docs and whale reports
- lightweight local classification of event severity
- OCR / screenshot / chart artifact tagging
- offline assistance when cloud LLM use should be minimized

### What Hailo should not do first

- direct candle-alpha prediction on the Pi as the primary signal engine
- pretending existing vision HEFs are ready-made financial models
- replacing structured market data with image inference

### Practical maximization path

- Keep technical analysis in Python on CPU.
- Use Hailo only for event-driven local summarization and document triage once `hailo-ollama` is installed.
- Trigger local AI only on schedule or thresholds:
  - FOMC / CPI / payroll / war escalation windows
  - whale transfer bursts above threshold
  - abnormal basis / funding / open-interest divergence

## Agent Framework Takeaways

### `my-claude-devteam` repo

- The `NYCU-Chung/my-claude-devteam` repo is best understood as a disciplined agent-team template:
  - specialized roles
  - review/checkpoint hooks
  - explicit planning and critic loops
- The best ideas to adopt here are not the raw number of agents, but the guardrails:
  - split research, execution, and review responsibilities
  - keep deterministic hooks around risky steps
  - make live trading depend on machine-checkable gates instead of conversational confidence

### BlockTempo article signal on 2026-04-21

- The BlockTempo piece on `agency-agents` highlights a much larger persona-heavy setup with `144` AI roles across `12` departments.
- That is useful as proof that role specialization is popular, but it is the wrong default for this project because:
  - too many personas increase token burn
  - coordination overhead can exceed the value for a single-account trading lane
  - quant execution benefits more from hard gates and narrow role ownership than from agent sprawl

### Converged recommendation for this repo

- Keep the operating model lean:
  - planner: strategy config and scheduling
  - analyst: deterministic analysis and backtest
  - executor: live-readiness and order path
  - critic: doctor, auth probe, and risk veto
- Prefer hook-style validation over extra chat loops:
  - `doctor` for environment and auth
  - `backtest` for reproducible historical sanity check
  - `live-readiness` for exchange constraints and risk gates
  - `live-pilot --execute` only after all previous stages pass

## Schedule-First Operating Model

### Low-token cadence

- hourly:
  - queue `BTCUSDT` futures `1h`
- every 4 hours:
  - queue `BTCUSDT` futures `4h`
- only on event trigger:
  - macro summary
  - whale burst summary
  - Hailo local summarizer lane

### Why this cadence

- hourly technicals are cheap and deterministic
- 4h context is enough to suppress noise without burning attention
- LLM work is reserved for event clusters, not every candle

## Recommended Next Build Steps

1. Keep using `binance-quant-control` as the controlled base layer.
2. Add a scheduled research pack that submits workflow tasks instead of relying on chat-driven polling.
3. Add a whale collector that writes structured JSON events, not prose.
4. Add a macro/event veto adapter with explicit event windows and oil-shock penalties.
5. Install `hailo-ollama` only after the structured data lanes are stable.

## Live Pilot Status As Of 2026-04-25

- Backtest path is now executable and connected to strategy YAML.
- Current micro-account pilot lane is `NEARUSDT` futures `4h`, not `BTCUSDT`, because Binance futures minimums make BTC a poor fit for a ~`5 USDT` account.
- The code path from strategy -> analysis -> risk guard -> exchange filters -> order journal is connected.
- Live trading should remain disabled right now because Binance private futures auth is failing with:
  - `status=401`
  - `code=-2015`
  - `Invalid API-key, IP, or permissions for action`
- Until that private auth check passes on the current IP, flipping `BINANCE_LIVE_TRADING_ENABLED=true` would be operationally unsound.
