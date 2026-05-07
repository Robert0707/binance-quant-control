# Strategy Review Automation

This workflow is the scheduled lane for strategy optimization after trades close.

## Purpose

- Review TP / SL / manual-close outcomes after they are written to the live journal.
- Update closed-trade review records for later strategy analysis.
- Avoid scheduled cloud-heavy `submit-analysis` runs that compete for shared API quota.

## Operator Boundaries

- Scheduled automation:
  - `python3 scripts/submit_research_pack.py --scheduled --config config/research-pack.default.yaml`
  - allowed action: `review-closed-trades`
- Manual research:
  - `openclaw-quantctl analyze ...`
  - `openclaw-quantctl submit-analysis ...`
- Execution:
  - `openclaw-quantctl live-readiness ...`
  - `openclaw-quantctl live-pilot --execute`

## Guardrails

- Strategy-only mode blocks scheduled `submit-analysis`.
- Strategy-only mode blocks `--run-now`.
- Strategy-only mode blocks raw analysis profiles in `research-pack.default.yaml`.
- Scheduled review writes an audit JSON under `state/scheduled-research/` so a future session can see whether the lane ran, dry-ran, or was blocked.

## Mutation Scope

- `strategy-optimizer` only writes `config/strategy-hermes-pro.auto.yaml`.
- Allowed auto-tuned fields:
  - `risk.min_convergence`
  - `risk.min_score_long`
  - `risk.max_score_short`
  - `risk.cooldown_hours`
  - `risk.atr_stop_multiple`
  - `risk.min_adx`
  - `risk.trailing_callback_pct`
  - `risk.take_profit_r_multiples`
- Protected fields:
  - `defaults.*`
  - `execution.*`
  - `challenge.*`
  - `risk.max_account_risk_pct`
  - `risk.default_leverage`
  - `risk.max_leverage`
  - `risk.max_notional_pct`
  - `risk.max_daily_trades`
- If the optimizer tries to change a protected field, the run must fail instead of silently rewriting the strategy.

## Default State

- `openclaw-binance-quant-research.timer` should stay disabled by default.
- If an operator intentionally enables it later, the service runs the guarded closed-trade review lane, not raw market-analysis workflows.

## Multi-Asset Review Split

- Closed-trade review records should carry `asset_class`, `route_id`, and `review_lane`.
- Strategy optimizer reports should expose lane breakdowns so BTC, ETH, major alts, and meme/high-beta symbols do not blur together in one undifferentiated review bucket.
- New strategy experiments should be promoted through:
  `route-symbol -> paper-order -> backtest -> review-closed-trades -> strategy-optimizer`
