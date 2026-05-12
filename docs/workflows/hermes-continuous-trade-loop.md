# Hermes Continuous Trade Loop

Status: Binance futures testnet automation only. Mainnet live stays disabled.

## Operator Commands

Start the loop:

```bash
openclaw-quantctl hermes-trade start --compact
```

Run one cycle:

```bash
openclaw-quantctl hermes-trade cycle --compact
```

Run repeated cycles until stopped:

```bash
openclaw-quantctl hermes-trade daemon --max-cycles 0 --compact
```

Stop immediately:

```bash
openclaw-quantctl hermes-trade stop --reason "operator stop" --compact
```

Status:

```bash
openclaw-quantctl hermes-trade status --compact
```

Local GUI:

```bash
openclaw-quantctl trade-console --host 127.0.0.1 --port 8765 --allow-order-actions --compact
```

Open `http://127.0.0.1:8765/`. The console is a thin control surface over the
same operator dashboard, trade session, Hermes cycle, and reduce-only close
paths used by CLI automation. It is intended for both robot control and manual
operator control without creating a separate trading workflow.

## Cycle

1. Review closed trades.
2. Run auto-pause with cooldown-aware loss-streak handling.
3. Run strategy optimizer when its interval is due.
4. Run Hermes AI readiness scan.
5. Use the execution ticket selected by the scanner.
6. Re-run live-readiness preflight for that exact ticket.
7. Execute only Binance futures testnet when all gates still pass.
8. Rescan positions and write a cycle report.

## Safety

- `mainnet_live_allowed` remains false.
- Execution mode must be `testnet_exploration`.
- `hermes-trade stop` disables the loop and writes the kill-switch.
- The loop executes at most one selected ticket per cycle by default.
- Failed execution can stop the loop and set a protective pause.
- GUI close-position actions require explicit confirmation and submit
  reduce-only market closes through the same Binance write gate.
- The GUI never changes `mainnet_live_allowed`; mainnet still requires the
  environment and readiness boundary to be changed outside the console.

## State

- Loop state: `state/hermes-trade-control.json`
- Cycle reports: `state/hermes-trade-loop/*-hermes-trade-cycle.json`
- Readiness reports: `state/hermes-readiness-scan/*-hermes-readiness-scan.json`
