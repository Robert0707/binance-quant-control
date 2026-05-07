# Stability Workflow

This workflow is the default safety gate before strategy or execution changes move forward.

## What it checks

- settings and strategy config validation
- `ruff` static checks on the active control-plane surface:
  - `src/binance_quant_control`
  - `tests`
  - `scripts/run_stability_workflow.py`
- `pytest` regression suite
- `live-readiness` dry-run against the selected strategy profile

## Default command

```bash
cd /home/robert/python/projects/binance-quant-control
./.venv/bin/python -m binance_quant_control.cli stability-workflow \
  --strategy-config config/strategy-stable-risk.yaml
```

## Wrapper script

```bash
cd /home/robert/python/projects/binance-quant-control
./.venv/bin/python scripts/run_stability_workflow.py
```

## Optional doctor step

Include the exchange connectivity and private-auth audit only when you intentionally want network health in the workflow result.

```bash
./.venv/bin/python -m binance_quant_control.cli stability-workflow \
  --strategy-config config/strategy-stable-risk.yaml \
  --include-doctor
```

## Pre-commit

Install local hooks so cheap mistakes are blocked before they enter the workflow:

```bash
cd /home/robert/python/projects/binance-quant-control
./.venv/bin/python -m pre_commit install
```

## Operating guidance

- Use `config/strategy-stable-risk.yaml` as the default live safety profile.
- Treat `live-readiness allowed=false` as a valid protective outcome, not a workflow failure by itself.
- Use this workflow before changing strategy YAML, execution logic, risk checks, or order-management code.
