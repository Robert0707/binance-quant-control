# Local Low-Token Schedule

Purpose: keep the Binance quant stack running locally without spending chat/model tokens on routine polling.

## Enabled Timers

- `ai-market-sentinel.timer`
  - Frequency: every 2 minutes.
  - Command: `ai-market-sentinel --skip-readiness --compact`
  - Role: read-only position, trend, quarantine, and expansion-gate watch.
  - Cost profile: low; no readiness scan, no model call, no execution.

- `openclaw-binance-position-guardian.timer`
  - Frequency: every 5 minutes.
  - Role: local position protection and adaptive exit checks.
  - Cost profile: low-medium; no new-entry exploration.

- `openclaw-binance-operator-dashboard.timer`
  - Frequency: every 15 minutes.
  - Role: compact local operator status report.
  - Cost profile: low; local summary only.

- `openclaw-binance-quant-research.timer`
  - Frequency: hourly.
  - Role: scheduled closed-trade review guardrail lane.
  - Cost profile: low; no raw analysis submission.

- `openclaw-binance-strategy-optimizer.timer`
  - Frequency: every 6 hours.
  - Role: optimize whitelisted strategy parameters from closed-trade reviews.
  - Cost profile: medium; local only.

## Disabled Timers

Keep these disabled while a BTC exposure exists or readiness has `allowed_count=0`:

- `openclaw-binance-testnet-explorer.timer`
- `openclaw-binance-autonomy.timer`
- `openclaw-binance-live-lane.timer`

These can trigger heavier candidate scans or execution-lane logic. Re-enable only after:

- open-position management no longer blocks the lane,
- route quarantines are cleared by evidence,
- readiness scan has an approved candidate,
- testnet execution is explicitly desired.

## Status Commands

```bash
systemctl --user list-timers --all --no-pager | rg 'ai-market|binance|quant|NEXT'
systemctl --user status ai-market-sentinel.timer --no-pager
systemctl --user status openclaw-binance-position-guardian.timer --no-pager
```

## Manual Local Checks

```bash
openclaw-quantctl trading-control-status --compact
openclaw-quantctl positions --compact
openclaw-quantctl route-risk-status --compact
openclaw-quantctl ai-market-sentinel --skip-readiness --compact
```

## Heavy Research

Run manually or in a dedicated local terminal, not as high-frequency chat work:

```bash
openclaw-quantctl ai-expectancy-upgrade --universe-limit 20 --limit 8000 --sweep-limit 5000 --max-configs 80 --max-walk-forward-validations 12 --max-readiness-candidates 6 --compact
```
