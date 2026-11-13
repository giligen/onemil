# RESULT 1,765 - pre-open conditioning of the HOD book (PREREG_1765)

**No feature is structure (0 of 5 read; E6 VOID). TEST untouched.** Book: 1,763 HOD trades, R floor 1 %; TRAIN n 1,295
(-0.25 R), VAL n 1,205 (-0.145 R); TRAIN-fixed quintile edges.

Coverage: pre-market 230/230 day files; short interest = FINRA bi-monthly file
(usable_from = settlement + 13 d, median lag 21 d), TRAIN 0.91-0.96 / VAL 0.98-0.99, winner/loser gap <= 2.6 pp; prior close and
ADV20 from cache.db daily_bars (0 of 105,072 rows differ > 1 % from the 1762 daily). **E6 Reg SHO list: not on disk**
(1746_cache/finra = daily short VOLUME files; no network) - VOID.

| Feature | H1 | H2 | VAL top-bottom | VAL t | mono TRAIN | VAL dR/wk vs p95 |
|---|---|---|---|---|---|---|
| pm_dv_ratio | -0.26 | -0.09 | 0.00 | 0.0 | no | +5.3 vs +6.7 |
| pm_bars | -0.03 | -0.04 | +0.45 | 3.7 | no | +8.8 vs +6.7 |
| pm_gap | +0.31 | -0.35 | 0.01 | 0.0 | no | +6.4 vs +7.0 |
| dtc20 | +0.03 | -0.11 | -0.20 | -1.6 | no | +2.3 vs +6.5 |
| si_chg | +0.14 | -0.02 | +0.30 | 2.3 | no | +6.8 vs +6.4 |

Fills kept >= 17/wk everywhere. The two VAL signals fail alike: pm_bars is negative in both TRAIN halves then +0.45 on
VAL (Q5 +0.09 vs Q1 -0.36); si_chg flips sign in H2; neither is monotone on TRAIN; filter gains clear the placebo p95 by only
1-2 R/wk, on a book where any random 60 % drop gains ~5 R/wk. With five features
and several clauses each, one VAL |t| above 3 is expected by chance. Pre-open volume, gap and short interest do not
condition the book.

CHILDREN:
1. E6 on real data: fetch daily FINRA Reg SHO threshold lists (2025-07..2026-05), rerun the yes/no read.
2. pm_bars forward-only check on 2026-06 onward before any rule.
3. Daily short-volume ratio (on disk, 1746) as a full-coverage E5 substitute, same reads.
