# RESULT -- cells 1,481-1,482: buy the retest, not the break

`PREREG_1481.md`. Base = the 9,911 fills of cell 1,438; base net R = cell_1478's standard-cost `outcome_R`. n/n_all = primary-fill / all candidates (never-retest + disagree + zero-risk make up the gap). Paired ΔR = net R' - base net R on the SAME fills. Null = count-matched (1,000 draws, seed 1481) percentile of the observed mean.

| cell | holdout | n | n_all | fill % | mean net R' | t | ex-top5 | fills/wk | paired ΔR | paired t | null %ile | never-retest base R | R' % price | dip $ | delay min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1481 | TRAIN-H2 | 3981 | 4398 | 90.5% | -0.106 | -2.67 | -0.217 | 50.89 | +0.170 | 21.81 | 100.0 | +0.638 | 1.47% | 0.230 | 0.9 |
| 1481 | VAL | 5036 | 5513 | 91.3% | -0.084 | -2.11 | -0.194 | 50.86 | +0.172 | 34.41 | 100.0 | +0.676 | 1.54% | 0.260 | 0.8 |

**1481 pass bar: FAIL**
| 1482 | TRAIN-H2 | 3750 | 4398 | 85.3% | -0.099 | -2.68 | -0.209 | 52.04 | +0.270 | 21.09 | 100.0 | +0.997 | 1.32% | 0.270 | 1.1 |
| 1482 | VAL | 4769 | 5513 | 86.5% | -0.072 | -1.88 | -0.181 | 52.05 | +0.272 | 25.84 | 100.0 | +0.932 | 1.40% | 0.302 | 1.0 |

**1482 pass bar: FAIL**

## Judge (main session, 2026-09-26 17:45 UTC) — FAIL on the level, the mechanism is real

Two implementations (builder `cell_1481.py`, fixed 17:00 to the PREREG's limit-price fill; rebuild `rebuild_1481.py`
from the prose) agree: fill-set Jaccard 0.995 (44 builder-only fills, 0 rebuild-only), 99.96 % of common fills within
0.01 R, entry prices identical, exit minutes 99.5 % identical (`review/1481_compare.md`; the 4 real differences are
same-bar stop/target tie-breaks). Scoring (`RESULT_1481_rebuild_score.md`, independent scorer on the rebuild's fills):
* 1,481 (bid at level − $0.01 for 15 min): filled 91 % of base fills; VAL n 5,016, mean net R′ −0.089 (t −2.2),
  ex-top-5 % −0.20; TRAIN-H2 −0.113 (t −2.9). Paired against the base on the SAME fills: +0.22 R (rebuild) / +0.17 R
  (builder; the 0.05 gap is the base-leg cost convention, the retest leg is identical), ex-top-5 % of the lift +0.15,
  t 22–35 — the retest entry is a robust 0.2 R improvement over buying the ask at the break, and still a losing book
  at the consolidation-low stop. Never-retest cohort (the runners lost by waiting): n 168 / 183, base +0.60 / +0.58 R.
* 1,482 (bid at level − 0.2 % for 30 min, builder only — same code, two parameters): VAL −0.072 R (t −1.9), paired
  +0.27 R, never-retest cohort n larger at +0.93 R.
* 1,486 (1,481 ∧ L3 top tercile, threshold 0.3070): TRAIN-H2 +0.083 (t 1.2), VAL −0.090 (t −1.6). FAIL.
Median R′ 1.5–2.2 % of price (rail clear). Frozen bar (VAL mean ≥ +0.15, t ≥ 2.5): FAIL for 1,481, 1,482 and 1,486.
The dip-depth table of cell 1,491 says where the loss is: base fills whose pullback exceeds 75 bps (30 %) earn −0.46 R,
those under 25 bps −0.02 R. The retest bid fills on all of them; whether the pullback's tape at the fill instant can
separate the two is cell 1,489 (running). Programme count 1,486 on this line.

Addendum (18:20 UTC, from the 1,489 refuter): by bar store, the VAL retest book is −0.14 R on real-SIP rows (83 %) and
+0.15 R on cache-only rows (17 %, the store-identity look-ahead cohort); the paired lift is the same on both.
