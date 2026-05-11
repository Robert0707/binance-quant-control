# New Symbol To Trade Pipeline

Status: Hermes-ready fixed workflow. This pipeline is evidence-first and does
not authorize mainnet entries by itself.

## Purpose

Turn any new Binance futures symbol into a machine-readable outcome without
writing symbol-specific Python code. Adding a new coin means running the fixed
workflow command, not editing strategy code, route code, or readiness gates.

- `reject`: data, liquidity, expectancy, robustness, or risk gates failed.
- `research_candidate`: there is a positive-expectancy surface, but promotion or
  forward evidence is not complete.
- `near_ready_market_only`: positive-expectancy evidence exists, but the market
  state gate is still blocking entry.
- `testnet_ready_candidate`: promotion, readiness, position sizing, and
  decision-contract gates all pass for testnet or paper validation.

Mainnet remains a separate operator-approved boundary.

## Invariants

- `mainnet_live_allowed=false` until a separate live approval task changes it.
- `BINANCE_LIVE_TRADING_ENABLED=false` is the expected default.
- Maximum planned account risk per trade stays at `2.5%`.
- HOLD is a valid output and must not be forced into BUY or SELL.
- Win rate is never the only objective. The hard research objectives are PF,
  expectancy, payoff, stop-loss ratio, drawdown, loss streak, and
  walk-forward stability.
- Hailo may triage or veto; it cannot approve order execution by itself.

## Pipeline

1. **One-Command Fixed Workflow**

   ```bash
   openclaw-quantctl new-symbol-workflow --symbols SOLUSDT --compact
   ```

   Low-cost plan/check mode:

   ```bash
   openclaw-quantctl new-symbol-workflow --symbols SOLUSDT --plan-only --compact
   ```

   Focused follow-up when the smoke pass surfaces a research candidate:

   ```bash
   openclaw-quantctl new-symbol-workflow --symbols SOLUSDT --research-depth focused --compact
   ```

   The command always reports `opens_orders=false` and
   `mainnet_live_allowed=false`. Testnet order placement is never automatic; it
   still requires an execution ticket and a separate operator command.

2. **Operator Intent**

   ```bash
   openclaw-quantctl route-intent "把 SOLUSDT 從新幣審核到可交易"
   ```

   Expected intent: `new-symbol-trade-pipeline`.

3. **Symbol And Route Check**

   ```bash
   openclaw-quantctl route-symbol SOLUSDT
   ```

   Stop if the symbol cannot be routed, the route is paper-only, or exchange
   filters would force sizing above the risk ceiling.

4. **Local Watch And Context**

   ```bash
   openclaw-quantctl ai-market-sentinel --symbols SOLUSDT --skip-readiness --compact
   openclaw-quantctl external-context --symbols SOLUSDT --compact
   ```

   This stage collects price state, trend state, spread/liquidity context, news
   risk, and optional whale/context sources. Missing optional external API keys
   are warnings, not trade approvals.

5. **Feature Dataset**

   ```bash
   openclaw-quantctl feature-dataset --symbols SOLUSDT --intervals 30m,1h,4h --limit 5000 --compact
   ```

   Required evidence:

   - replayable feature rows,
   - manifest hash,
   - no lookahead in the feature contract,
   - enough bars for the selected intervals.

6. **Expectancy Research**

   ```bash
   openclaw-quantctl risk-combo-sweep --symbols SOLUSDT --target-side BUY --target-interval 15m --limit 600 --grid-mode fast --min-test-trades 10 --target-profit-factor 1.0 --min-expectancy-r 0.0 --max-stop-loss-ratio 55 --max-configs 8 --max-walk-forward-validations 1 --top-n 5 --skip-news --compact
   openclaw-quantctl risk-combo-sweep --symbols SOLUSDT --target-side SELL --target-interval 15m --limit 600 --grid-mode fast --min-test-trades 10 --target-profit-factor 1.0 --min-expectancy-r 0.0 --max-stop-loss-ratio 55 --max-configs 8 --max-walk-forward-validations 1 --top-n 5 --skip-news --compact
   openclaw-quantctl risk-combo-sweep --symbols SOLUSDT --target-side BUY --target-interval 4h --limit 600 --grid-mode fast --min-test-trades 10 --target-profit-factor 1.0 --min-expectancy-r 0.0 --max-stop-loss-ratio 55 --max-configs 8 --max-walk-forward-validations 1 --top-n 5 --skip-news --compact
   openclaw-quantctl risk-combo-sweep --symbols SOLUSDT --target-side SELL --target-interval 4h --limit 600 --grid-mode fast --min-test-trades 10 --target-profit-factor 1.0 --min-expectancy-r 0.0 --max-stop-loss-ratio 55 --max-configs 8 --max-walk-forward-validations 1 --top-n 5 --skip-news --compact
   openclaw-quantctl risk-combo-sweep --symbols SOLUSDT --target-side BUY --target-interval 1d --limit 600 --grid-mode fast --min-test-trades 10 --target-profit-factor 1.0 --min-expectancy-r 0.0 --max-stop-loss-ratio 55 --max-configs 8 --max-walk-forward-validations 1 --top-n 5 --skip-news --compact
   openclaw-quantctl risk-combo-sweep --symbols SOLUSDT --target-side SELL --target-interval 1d --limit 600 --grid-mode fast --min-test-trades 10 --target-profit-factor 1.0 --min-expectancy-r 0.0 --max-stop-loss-ratio 55 --max-configs 8 --max-walk-forward-validations 1 --top-n 5 --skip-news --compact
   openclaw-quantctl risk-combo-matrix --latest-sweeps 6 --compact
   ```

   These are exactly the research lanes automated by `new-symbol-workflow`.
   They cover BUY/SELL and short/medium/long horizons without adding
   symbol-specific code.

   Promotion requires enough samples and robust train/test/walk-forward evidence.
   A short-sample positive row is only an expansion target.

7. **Research Gate**

   ```bash
   openclaw-quantctl high-win-iteration --compact
   openclaw-quantctl high-win-converge --max-rounds 1 --compact
   ```

   Use plan-only mode unless a bounded research batch is explicitly requested.
   Do not lower gates to create a candidate.

8. **Hermes AI Trader Gate**

   ```bash
   openclaw-quantctl hermes-ai-trader --compact
   openclaw-quantctl ai-readiness-scan --strategy-config config/strategy-live-pilot.yaml --execution-mode testnet_exploration --max-candidates 6 --compact
   ```

   Required evidence:

   - standard signal schema,
   - feature/model registry,
   - portfolio target and open-risk snapshot,
   - structured committee decision,
   - Hailo task allocation with order-execution disabled,
   - readiness denial or approval with reasons.

9. **Decision Contract**

   ```bash
   openclaw-quantctl trade-decision --symbol SOLUSDT --strategy-config config/strategy-live-pilot.yaml --execution-mode testnet_exploration --compact
   openclaw-quantctl decision-audit --compact
   ```

   Every BUY/SELL/EXIT/HOLD artifact must answer:

   - why long now,
   - why short now,
   - why no trade now,
   - where the stop is if wrong,
   - where take-profit is if right,
   - max loss,
   - long-term expected value.

10. **Operator Dashboard**

   ```bash
   openclaw-quantctl operator-dashboard --compact
   ```

   This is the customer-facing summary surface. It must show product readiness,
   decision-artifact audit status, risk-combo matrix state, loss diagnostics,
   and next repair commands.

11. **Testnet Or Paper Only**

    If and only if readiness and decision-contract gates pass:

    ```bash
    openclaw-quantctl live-readiness --symbol SOLUSDT --strategy-config config/strategy-live-pilot.yaml --execution-mode testnet_exploration --compact
    ```

    Testnet execution still requires an explicit operator command. Mainnet is
    out of scope for this pipeline.

## Optimization Hooks

If any layer blocks, optimize that layer only:

- data/liquidity block -> change symbol universe or interval;
- low PF/expectancy -> change strategy family or exit structure;
- low payoff -> change TP/SL shape, not win-rate gates;
- stop-loss ratio high -> improve entry quality or adaptive exit;
- walk-forward negative -> add regime/flow filters or reject;
- portfolio block -> reduce correlation or wait for exposure to clear;
- decision contract invalid -> fix decision output, not trading logic;
- readiness blocked -> fix live gate evidence, not research reports.

## Customer Delivery Definition

The program is customer-presentable when these are true:

- `openclaw-quantctl repository-audit --compact` has no structural blocker.
- `openclaw-quantctl professional-system-audit --compact` explains remaining
  blockers without hidden live behavior.
- `openclaw-quantctl operator-dashboard --compact` gives a compact customer
  status and next action.
- `openclaw-quantctl decision-audit --compact` passes for generated decisions.
- all chosen tests pass.
- the repo is synchronized to Git with no secrets or local runtime state.
