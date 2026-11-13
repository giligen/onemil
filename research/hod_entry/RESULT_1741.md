# RESULT 1,741 — option flow before the break as a HOD-break filter: FAIL (primary F1), TEST untouched

Population: 12,135 TRAIN+VAL signals; flow join loss 0; 125 have no B0 net_R (dropped, 12,010 scored; 7,315 TRAIN / 4,695 VAL). Convention: last pre-break bar = entry_m-2.

**Rail (PASS, not VOID):** coverage (>=1 pre-break option print) TRAIN 80.1 %, VAL 81.0 %; chain listed 90 % (not the ~56 % the PREREG expected). Winner/loser gap TRAIN -0.3 pp, VAL +1.6 pp (TRAIN halves +1.7 / -2.9), all <= 5 pp.

**F3 covered-with-print vs not (B0 net R, exploratory):** TRAIN -0.19 vs -0.34 (diff +0.15, day-t 2.2); VAL -0.26 vs -0.51 (diff +0.26, day-t 4.6); TRAIN H1 +0.26 (t 2.4) but H2 +0.02 (t 0.35). Both groups negative; "optionable" is a liquidity proxy, not a profitable subset.

**Terciles (cuts from TRAIN covered; F1 cuts 0.62/0.85), top - bottom spread, day-t, iid-t, ex-top-5 % spread, top fills/wk:**
| F | group | spread | day-t | iid-t | ex5 | top/wk |
|---|---|---|---|---|---|---|
| F1 | TRAIN_H1 | -0.119 | -0.75 | -2.04 | -0.233 | 32.8 |
| F1 | TRAIN_H2 | -0.166 | -2.24 | -2.49 | -0.286 | 40.4 |
| F1 | TRAIN | -0.167 | -1.43 | -3.89 | -0.284 | 37.4 |
| F1 | VAL | -0.087 | -1.23 | -1.56 | -0.206 | 75.1 |
| F2 | TRAIN / VAL | -0.124 / +0.027 | -1.0 / 0.5 | -2.9 / 0.5 | -0.24 / -0.09 | 37 / 63 |
| F4 | TRAIN / VAL | -0.198 / -0.071 | -1.2 / -0.9 | -4.7 / -1.2 | -0.32 / -0.19 | 37 / 75 |
Top-tercile F1 mean net R: TRAIN -0.279, VAL -0.319 (the call-heavy tercile is the WORST, wrong sign vs the hypothesis).

**Permutation (300 within-day shuffles, F1):** observed spread percentile TRAIN 0.0, VAL 32.7 (need >= 95).

**Verdict: FAIL.** Misses spread >= +0.15, day-t >= 2, same-sign halves, top mean >= 0 on VAL, ex-top-5 % >= 0, permutation >= 95 (only fills/wk >= 3 holds). 5 reads (F1 primary + F2, F4, F3, perm) — Bonferroni noted; nothing survives even uncorrected. TEST not read.

Caveats: Alpaca option TRADE bars, not quotes; no OI/GEX; expired-chain listing as returned today; call share is traded volume, not signed by aggressor; rail passes so this is a measured null on this population (covered signals, 2025-01..2026-05), not an availability void.
