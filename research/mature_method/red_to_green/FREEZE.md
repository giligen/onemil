# FREEZE — red-to-green (F6-PDR), mature-method pass

**TEST = 2026-06-01 → 2026-09-04 is SEALED.**

It is opened only for a cell that has already passed **both** claim gates of `PREREG.md` §6 on the
two open splits:

* **G1** — TRAIN mean net R > 0 with `t >= 2` and `>= 5` trades/week;
* **G2** — VAL the same sign **and** `>= 55 %` green weeks.

Nothing else may read it. The reconciliation (`research/fuckup_audit/H/F6_reconcile/REPORT.md`) has
already published TEST figures for this book from three implementations; those may be **quoted as
prior art** and are not a cell being scored here.

## Ledger

| # | date | what was opened | authority |
|---|---|---|---|
| 1 | 2026-09-19 | **cell `P4` (pdr >= 12), scan S1** — TRAIN net +0.0696 R, t 2.09, 17.6 tr/wk (G1 pass); VAL net +0.0303 R, 63.6 % green weeks (G2 pass) | PREREG §6 |
| 2 | 2026-09-19 | **cell `G1c` (B0 + obtainable-only), scan S1** — TRAIN net +0.0664 R, t 2.15, 19.1 tr/wk (G1 pass); VAL net +0.0441 R, 68.2 % green weeks (G2 pass) | PREREG §6 |

Nothing else was read. `score.py` drops every row with `day >= 2026-06-01` before a cell is scored;
only `supp.py` §(d) reads TEST, and only for the two cells above. Result: **both fail on TEST**
(`test_read.csv`) — P4 gross −0.0015 / net −0.0571 R, 35.7 % green weeks, −$2,217; G1c gross −0.0178 /
net −0.0713 R, 28.6 % green weeks, −$2,768. No cell was selected on TEST.
