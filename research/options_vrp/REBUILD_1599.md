# REBUILD_1599 -- Independent rebuild of PREREG_1567 v3 (cells 1,599-1,606)

Built from PREREG_1567.md + Amendments 2/2a/3 prose only; did not open cell_1599.py, test_cell_1599.py, cell_1599_cycles.csv/monthly.csv, RESULT_1599.md, or the v1/v2 cell scripts.


## TRAIN selection
Selected cell: delta=0.2, management=B, gate=iv15.


TRAIN cell table (all 8 cells): `rebuild_1599_train_table.csv`.


## VAL (2025-07-07..2026-08-17) for the selected cell

* label: VAL_selected
* n_cycles: 16
* void_rail: 0.6190476190476191
* monthly_mean: 0.0019835897435898342
* monthly_sharpe: 0.07242842611634466
* green_share: 0.7777777777777778
* worst_month: -0.2447876923076915
* max_dd: 0.2447876923076915
* ex_top5_mean: -0.00044273504273495954
* n_months: 9

Buy-and-hold SPY over the same VAL window on B: $1592, max drawdown 6.59%.


**VAL pass bar: FAIL** (mean monthly return on B >= 4%, Sharpe >= 1.0, green >= 60%, worst month >= -B, max DD <= 1.5B, beats SPY B&H per unit drawdown).


## EXTENSION (2013-04-08..2023-12-25), read once for the selected cell

* label: EXTENSION_selected
* n_cycles: 151
* void_rail: 0.6505747126436782
* monthly_mean: 0.014455833333333334
* monthly_sharpe: 0.8715291107332443
* green_share: 0.6041666666666666
* worst_month: -0.27248
* max_dd: 0.3217323076923073
* ex_top5_mean: 0.012299551282051282
* n_months: 96
* years positive/total: 5/7 (bar: >=8/11)

## Method notes / caveats (read as an adversary)

* Monthly P&L is attributed to the calendar month of the EXIT (realization), not entry -- disclosed convention, PREREG does not specify which.
* ATM 45-DTE IV for the gate uses the PUT closest to spot only (no call averaging).
* SPY spot 2013-04-08..2015-12-31 (EXTENSION only) has no Alpaca/Databento equity source; used put-call parity from the same 10:00 options chain -- flagged `used_parity_spot`.
* VOID = entry 10:00 bar lacks a two-sided quote on either leg, per Amendment 3 (rail computed above).
* This is a single independent build, not yet cross-checked trade-by-trade against cell_1599.py's own output (that comparison is a separate step per the CLAUDE.md independent-check protocol, and requires an agent to read both, which this task explicitly forbade).