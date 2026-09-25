# Cell 1,444 — resting entry only on scanner-qualified names: NOT RUNNABLE historically; the cohort it restates is a look-ahead

Judged 2026-09-25 20:05 UTC from the adversarial check of cell 1,438 (`review/1438_check.md`, Opus, independent).

| cohort (1,438 fills, mean net R, day-clustered t) | TRAIN-H2 | VAL |
|---|---|---|
| "tracked" = cache.db holds ≥ 90 % of the minutes before the fill | +0.229 (t 2.8), 39 fills/wk | +0.263 (t 3.8), 45 fills/wk |
| tracked AND full-day range < 10 % | −0.324 | −0.341 |
| live-knowable analogue: range through bar j ≥ 10 % | −0.18 | −0.29 |

* The proxy clears the frozen bar numerically but is not causal: nothing live writes `intraday_bars_1min`; it is written
  only by backtest cache builds that select symbol-days on the FULL-DAY range ≥ 10 % priced on the close. The +0.25 R
  cohort is selected on the day's outcome. `scan_results` is empty for 2025-07..2026-05, so scanner membership at bar j
  cannot be tested from any stored record. The rule as frozen has no historical data. FAIL by the PREREG's consequence:
  HOD-break is closed at every executable entry on this population; the weekend moves to the other populations.
* Cost correction to cell 1,438: the entry half-spread was charged twice (0.10 R). Corrected net ≈ −0.11 R; raw R ≈ 0 —
  the sign of the verdict does not change, and it matches every earlier HOD finding (gross ≈ 0 in every bucket).
* Forward instrument (free, no research cell): the dry run's parity ledger gets a column `scanner_qualified_at_arm`
  (the scanner's qualified set is in-process). 20 sessions of dry data test the scanner-membership hypothesis on
  point-in-time records for the first time. Engineering item for Monday's queue, not a live change.
