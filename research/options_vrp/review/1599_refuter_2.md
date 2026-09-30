# Cells 1,599-1,606 (v3): refuter 2, statistics and risk lens

Checked 2026-09-28 ~05:15 UTC.

## Verdict: there are no numbers to refute. The BLOCKED status is correct.
* `cell_1599_cycles.csv` has only a header (0 rows). `cell_1599_monthly.csv` also has only a header.
  `rebuild_1599_cycles.csv` does not exist. `RESULT_1599.md` says BLOCKED and makes no claims.
* So none of these can be computed yet: the TRAIN selection, the VAL read, the EXTENSION read, the
  TRAIN-to-VAL and TRAIN-to-EXTENSION rank, the neighbour check, year concentration (8 of 11), the spike
  months, the budget assertion on real cycles, max drawdown on B, cap vs naked put, SPY buy-and-hold per
  unit of drawdown, the dollar range at B = $6,500, and the scaling to $1K/month.
  Any value reported for these today would be made up.
* Progress: `fetch_dbn.py` (pid 3984895) is on the EXTENSION definition pulls (2013-12). There is no
  `mondays.parquet` yet and only 4 leg files. Spend is $0.46 of the $150 cap.
  `rebuild_1599.py --stage all --workers 16` (pid 3988487) is on the panel snapshots, 75/133.

## Problems that will distort the statistics once data arrives (fix these before the one-time EXTENSION read)
1. **The 2013 ladder has far fewer weeks.** `fetch_dbn.log` shows 16 of the first 36 extension Mondays
   (44 %) skipped with "no expiry in [38,52] DTE". In 2013 SPY listed only monthly and quarterly expiries at
   45 DTE. The panel (2024+) fills every week. With fewer weeks, fewer spreads are open at once, so less of
   B is used. That mechanically lowers the mean monthly return on B, so the 4 %/month bar does not test
   the same book on the extension as on the panel. Report per year: weeks entered and B-utilisation. Then
   judge the extension per cycle and per unit of B deployed as well as per month.
   The skipped weeks are chosen by the calendar, not by outcome, so they do not bias P&L. They change the
   frequency and the size of the exposure.
2. **The "≥ 8 of 11 years" rule is ill-defined as it stands.** There is no SPY spot price for
   2013-04-08..2016-01-03 (Databento equities start 2018-05, Alpaca 2016-01-04). Delta selection there
   needs a method nobody has registered (put-call parity), or those years drop out. 2013 is also a partial
   year (9 months). The PREREG must be amended before the read to say exactly what happens to 2013-2015:
   a parity spot with a stated rule, or exclusion. It must also give the year denominator (11, 10 or 8),
   so the year-count bar cannot be adjusted after the fact.
3. **The builder and the rebuild handle holidays differently.** The rebuild asks for cbbo-1m on
   holiday Mondays: 2024-02-19, 2024-05-27, 2024-09-02, 2025-01-20, 2025-02-17, 2025-05-26 and 2025-09-01
   all return 422 `symbology_invalid_request`. These are market holidays, not a symbology fault.
   `fetch_dbn.entry_session` moves entry to Tuesday. The rebuild will either drop these weeks or mark
   them VOID with reason `no_chain_or_spot`. Either way that is about 7 of 133 panel weeks (≈ 5 %):
   * the cycle-set Jaccard falls below the 0.99 bar for a reason unrelated to the model;
   * if they are marked VOID, they use about half of the 10 % VOID allowance.
   The rebuild needs the same rule (first trading session of the week) before the comparison is run.
4. The rebuild picks expiries with `nearest_friday_expiry` from the calendar. The builder picks from the
   listed definitions. For 2013-2015 these will differ: the builder skips the week, while the rebuild may
   point at a Friday that was never listed and mark it VOID. Count skipped weeks and VOID weeks
   separately, per sample.
5. Not a statistics issue: `--workers 16` on this 2-CPU node goes against the single-process resource
   rule.

## What the eventual read must include (from the PREREG pass bar and this lens)
* Scoring: mean monthly return on B, monthly Sharpe, share of green months, worst month vs −B, max
  drawdown vs 1.5 B, SPY buy-and-hold on B per unit of drawdown. Compute these separately for PANEL-VAL
  and EXTENSION.
* Rank checks: TRAIN rank vs VAL rank for all 8 cells, the best VAL cell vs the selected cell, and the
  neighbours (the other delta and width for the same M/G). Only the selected cell gets an EXTENSION read.
* Concentration: P&L per year, and the extension with the best 1 and best 3 months removed. Report the
  2015-08, 2018-02, 2020-03 and 2022 months on their own.
* Budget: `assert_budget` on every open cycle. Also, for every day, the worst-case loss summed over all
  open spreads must be ≤ B.
* Scaling: at B = $6,500, give the monthly P10/P50/P90 in dollars and the worst month. Then scale
  linearly to a $1,000/month mean: equity required = 65,000 × 1,000 / mean$. The worst month scales by
  the same factor.
