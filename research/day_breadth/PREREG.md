# PREREG — Find the winners: DAY selection by gapper-universe breadth. Cells 1,406–1,411

Owner 2026-09-23: "Find the winners." Every study this week says the same thing: entry timing and exits do not
separate winners from losers in these gappers, and the R concentrates on a few busy, market-wide days (base-entry
VAL: top 10 % of days = 168–284 % of the R). So the winners are DAYS. This pass tests whether those days are
recognisable at the entry minute from the state of the gapper universe. Frozen before any breadth number exists.

## Breadth universe (knowable at 09:30:00 ET)
`research/hod_pmh_causal/pm_candidates.csv` (gap ≥ 3 % at the open, open $3–50, prior-day volume ≥ 500K), RTH
bars from `bars_sip.db` ∪ `bars_rth.db` (`run_consol.fetch_day_bars_dual`). TRAIN 2025 + VAL 2026-01..05 only.

## Three measures, per trade, at its decision minute d (causal: bars with minute ≤ d only)
* **NG** — number of universe names that day (known at 09:30).
* **BR(d)** — share of that day's universe names (with a bar by d) whose last close ≤ d is above their 09:30 open:
  gappers holding their gap = risk-on small-cap tape.
* **HH(d)** — share of those names whose high of day (through d) was set within the 30 minutes up to d: fresh
  highs across the board = momentum persisting.
Decision minute: base-entry book → `signal_m` (the signal bar's close); HOD-break book → `entry_m − 1`.

## Books (existing trades, unchanged)
* **Base entry, C1** — `research/hod_consol/trades/1400_all.csv` (TRAIN 3,958 / VAL 2,289).
* **HOD-break, live rule B0** — `research/hod_exit_lab/b0_trades.csv` TRAIN + VAL (the dry-run population).

## Cells: keep a trade iff measure ≥ the TRAIN-H1 median of that measure over that book's TRAIN-H1 trades
| cell | measure | book |
|---|---|---|
| 1,406 | NG | base | 1,407 | BR | base | 1,408 | HH | base |
| 1,409 | NG | HOD | 1,410 | BR | HOD | 1,411 | HH | HOD |
The cut is fixed on TRAIN-H1; TRAIN-H2 and VAL are the holdouts.

## Pass bar (all, per cell)
Kept net ≥ +0.10 R on TRAIN-H2 AND VAL; VAL trade-weighted day-clustered t ≥ 2 (kept); dropped mean < 0 on
TRAIN-H2 AND VAL; kept book after the slot rule (first 12/day, 4 concurrent) > 0 on both holdouts with ≥ 3
fills/week on VAL. Report-only: tercile table of each measure (monotone?), and the TRAIN-only WINNER-DAY ANATOMY:
per book, the top-decile days by summed R vs the rest on NG, BR(10:30), HH(10:30), SPY open→10:30 return
(`research/index_orb/cache/SPY_1min.parquet`) and weekday — exploration for the next pass, never scored here.

## Verification of any pass
Causality trace (every measure uses bars ≤ d; the cut uses TRAIN-H1 only), independent rebuild from this prose,
measured-NBBO re-score of the kept book, before anything is reported as a pass.

## Not allowed
Other thresholds or measures; stacking measures; tuning the decision minute; any TEST row. Programme count 1,411.
