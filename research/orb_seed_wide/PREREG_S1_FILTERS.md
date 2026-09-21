# PREREG — S1 stratum (gap 3–5 %, open $3–30) loser filters. Cells 1,316–1,318

Committed BEFORE any VAL number is computed. Parent: `PREREG.md` (cells 1,300–1,315), book
`out/runS1_true.csv` (production parameters, static_lock+touchgo exit, cost setting M, R = $375).

## Why
S1 is the only stratum with frequency (6.6 fills/wk on VAL) and it is flat net (VAL +0.04 R,
TRAIN −0.03 R). Owner 2026-09-21: take the frequency and cut the losers. Method: bottom-up
losers (H/METHOD.md) — anatomy on TRAIN only (`s1_anatomy.py`, run before this file, VAL untouched),
filters chosen from TRAIN with a mechanism, then scored once on VAL.

## TRAIN anatomy (210 entered, mean −0.026 R, 46 % stops at −0.32 R, top-5 % = 39 % of gross wins)
Strongest tercile separators (high-vs-low t): gap_pct +3.1, range_return_pct −2.5, spy_gap_pct +2.4,
price_vs_20d_high −2.2 (multiplicity: 23 features scanned, so ~1 expected at |t| > 2 by chance).

## Cells (three, fixed here; no other cut may be scored)
| cell | filter | mechanism |
|---|---|---|
| 1,316 F-A | `gap_pct >= 4.0` | gap size = catalyst conviction; monotone inside S1 (terciles −0.12/−0.04/+0.09) AND across strata (gap ≥ 5 production seed +0.41 R) — the same direction from two independent samples |
| 1,317 F-B | `range_return_pct <= 2.0` | a range that already ran > 2 % from its open is a chased extension; high tercile −0.15 R vs +0.02/+0.05 |
| 1,318 F-AB | both | |

spy_gap and price_vs_20d_high are NOT cells (one expected false positive; they go to a later pass only
if a cell above passes). Cutoffs are the TRAIN tercile edges rounded (4.138 → 4.0, 2.095 → 2.0).

## Pass bar (all, per cell)
1. filtered mean R ≥ +0.15 on TRAIN and on VAL;
2. VAL day-clustered t ≥ 2.0;
3. TRAIN halves (H1/H2) both ≥ 0;
4. ex-top-5 % mean R ≥ 0 on both splits;
5. removed cohort mean R < 0 on BOTH splits (the filter cuts losers, not winners);
6. VAL fills/wk ≥ 3.0 (cadence C5) at 8 shared slots;
7. cadence bar block (`scripts/cadence_bar.py`) printed for TRAIN and VAL — reported, C1–C4 diagnostics.

## Decision rule
Any cell passes 1–6 → next PREREG: live seed widening = production ∪ S3 ∪ S1-filtered, with the
matched control and ≥ 3 R obtainability audit before a dry day. No cell passes → report, then the exit
pass (stops are 46 % of TRAIN trades) under its own PREREG. TEST (≥ 2026-06-01) stays sealed.

## Not allowed
Re-cutting thresholds after seeing VAL; adding a fourth filter; scoring spy_gap / 20d-high; any exit change.
