# Position Monitoring Policy

## Objective

- High-frequency checks should be rule-based and compact.
- Model calls are reserved for ambiguous or materially changed states.

## Monitoring Inputs

- positions --compact
- account --compact
- journal-summary
- closed-trade review summary

## Alert Conditions

- position near stop-loss
- TP1 reached and stop still unprotected
- protective orders missing / duplicated / malformed
- fast PnL deterioration
- symbol structure no longer matches trade thesis

## Silence Conditions

- no open position
- no new closed trade review
- no protective-order drift
- no material state change since last check
