# RESULT — cell 1,682: the literature's and the practitioners' exit rules (H37-H58) on the 1,681 harness

PREREG: `PREREG_1682.md` (FROZEN 2026-09-30 18:40 UTC). Book, halves, cost convention, stats and pass bar unchanged
from `PREREG_1681.md` (same n=5,506 HOD-break population, split TRAIN-H2 n=2,349 / VAL n=3,157). 20 new rules
(H37-H56, the E-ids from `research/ideas_web/EXITS_LIT_20260930.md`) implemented into `1681_hypotheses.py`
(harness reuse); synthesis joint H57; H57 stacked on the 1,681 joint H36 as H58. All runs `--lens` under `nice -n
10`, single process. Verified: `1681_reads.csv` carries 58 ids x 2 reads = 116 rows; `1681_per_fill.csv` 319,348
rows.

**Read convention** (unchanged from 1681): "TRAIN" = evaluation population TRAIN-H2 -- the `TRAIN-H2` row for
non-model rules, the `VAL->TRAIN-H2` row for model rules (H41, H43, H45: a statistic fit on VAL, applied to
TRAIN-H2, keeping the lookup out-of-sample). "VAL" = the `VAL` row, or `TRAIN-H2->VAL` for model rules. Green-week
/ P10 baselines: TRAIN 44.4% / -25.45R, VAL 40.9% / -26.90R (TRAIN-H2 baseline differs slightly between model and
non-model rows by construction -- same as 1681, not a bug).

## The 20 rules

| id | rule | dR (TRAIN/VAL) | t_day (TRAIN/VAL) | ex5 (TRAIN/VAL) | green-wk pp (TRAIN/VAL) | P10 delta R (TRAIN/VAL) | verdict |
|---|---|---|---|---|---|---|---|
| H37 | Kaminski-Lo rho1 gate -> breakeven | +0.0031 / -0.0041 | +0.51 / +0.18 | -0.023 / -0.032 | +0.0 / -13.6 | +0.49 / +1.27 | sign flip, tail |
| H38 | confirmed CoC (2/4 signals) | -0.0226 / -0.0143 | -1.00 / +0.08 | -0.119 / -0.112 | -7.4 / -22.7 | +2.87 / +2.73 | negative both, tail |
| H39 | doldrums de-risk @12:30 | +0.0021 / -0.0053 | -0.54 / -0.81 | -0.041 / -0.046 | +0.0 / +0.0 | +2.03 / -3.31 | day-t flips neg |
| H40 | late trim 50% @14:30 | +0.0046 / +0.0054 | +0.44 / +1.37 | -0.019 / -0.019 | +3.7 / +4.5 | +0.74 / -2.51 | **positive both, eligible** |
| H41 | vol-regime trail (ATR tercile) | +0.0026 / -0.0002 | +1.53 / -1.79 | -0.000 / -0.000 | +0.0 / +0.0 | -0.01 / -0.01 | ~zero, no effect |
| H42 | analytic target 2*sig*sqrt(T) | +0.0094 / +0.0146 | +0.17 / +0.02 | -0.095 / -0.089 | +0.0 / +4.5 | -2.43 / -11.88 | +/+ but t~0, tail |
| H43 | Bayesian de-risk (post<0.28) | +0.0189 / +0.0133 | +1.08 / +0.94 | -0.061 / -0.066 | +0.0 / +9.1 | +5.34 / +4.44 | +/+ t<2, tail |
| H44 | chandelier HH-1.5*ATR | -0.0223 / -0.0126 | -0.62 / -0.37 | -0.083 / -0.070 | -3.7 / -13.6 | +1.56 / -0.70 | negative, worse cons. |
| H45 | MAE ceiling (winners P80) | -0.0051 / -0.0381 | +0.23 / -1.51 | -0.043 / -0.081 | -3.7 / -18.2 | +1.21 / -6.51 | negative, worst cons. |
| H46 | 75% out @+1R | -0.0028 / -0.0076 | -0.11 / +0.49 | -0.094 / -0.096 | +3.7 / -9.1 | +6.17 / +4.99 | ~flat, inconsistent |
| H47 | Warrior half+9EMA | -0.0113 / -0.0155 | -0.47 / +0.15 | -0.120 / -0.124 | -3.7 / -13.6 | +6.44 / +5.42 | negative both |
| H48 | SMB give-back (book) | -0.0426 / -0.0114 | -1.14 / +0.22 | -0.117 / -0.086 | -14.8 / -4.5 | +2.02 / -2.35 | negative, worst raw TRAIN |
| H49 | half@+3R, no 2R exit | +0.0250 / +0.0132 | +1.11 / -0.68 | -0.068 / -0.076 | +0.0 / +0.0 | -3.50 / -14.02 | +/+ but tail by construction |
| H50 | swing-low trail (3x5m) | -0.0117 / -0.0117 | -0.43 / -0.14 | -0.102 / -0.098 | -3.7 / -4.5 | -0.32 / +2.92 | negative, consistent |
| H51 | measured-move target | +0.0376 / -0.0080 | +2.52 / -0.72 | -0.069 / -0.113 | +3.7 / +4.5 | -0.08 / -11.66 | TRAIN mirage, VAL reverses |
| H52 | new-high vol fade | -0.0024 / +0.0053 | -0.24 / +1.48 | -0.057 / -0.045 | +0.0 / +0.0 | +5.66 / +4.28 | ~zero, sign flip |
| H53 | SAR+ADX>20 trail | -0.0041 / -0.0046 | -1.08 / +0.09 | -0.098 / -0.099 | +3.7 / -13.6 | -1.34 / -0.89 | ~flat, no edge |
| H54 | power-hour hold @15:00 | +0.0036 / -0.0016 | +1.98 / -1.70 | -0.002 / -0.004 | +0.0 / +0.0 | +0.05 / -0.61 | **+/- but NOT tail-carried, eligible** |
| H55 | 75%@1R + CoC/chandelier | -0.0119 / -0.0118 | -0.47 / +0.36 | -0.122 / -0.121 | -3.7 / -9.1 | +5.88 / +4.71 | negative, inherits both |
| H56 | portfolio hard stop (book) | -0.0190 / -0.0124 | +0.05 / -0.25 | -0.086 / -0.079 | -3.7 / -9.1 | +3.75 / -3.16 | negative, worst raw |

Full per-rule mechanism/saved/forgone/why: `1682_insights.md`. Pattern: **17/20 (85%) have TRAIN ex5 <= -0.02 or a
TRAIN P10 that goes the wrong way** (H40 and H54 pass both; H41's ex5 also technically clears but its P10 fails) --
close to 1681's own 27/35 (77%), same recurring finding: almost every nominally-interesting TRAIN point estimate on
this population is carried by its best 5% of trades. Three rules (H42, H49, H51) initially read as structurally
broken under `run_reshape`'s tighten-only accumulation (their proposed target was routinely ABOVE the unmodified
base target and therefore a silent no-op -- H49 read exactly 0% fired); fixed as bespoke walks that genuinely
replace the target, per PREREG's own "(no exit at 2R)" language for H49. Detail in `1682_insights.md`'s methodology
note.

## Synthesis (fixed before any number; scored on the TRAIN read only, same formula and gates as 1681)

S = dR_TRAIN + 0.5 x (green-week gain, TRAIN, in R-equivalents: 1pp = 0.01R). Hard gate (both must hold on TRAIN):
ex-top-5% dR > -0.02, weekly P10 not worse. Eligible (positive S AND both gates): **H40 (S=+0.0231) and H54
(S=+0.0036) only** -- every other rule fails at least one gate (H41 passes ex5 but fails P10; H43/H46/H49/H51/H52
etc. score positive on raw dR but fail ex5). No third positive-score compatible candidate exists (same shape as
1681's 2-member joint).

H40 and H54 are compatible by construction: different mechanisms (partial-sell vs target-widen) at different, later
clock times (14:30 then 15:00 ET) -- they never compete for the same decision point, so H40 runs first in TIME, not
by a score tie-break. **H57** = this joint, implemented in `1681_hypotheses.py::_h57` (model=False, standard HALVES
reads; the TRAIN-H2 row is context only, never re-used for selection).

## H57's held-out read (VAL, n=3,157) vs the 1681/1682 pass bar (paired dR>=+0.05R, day-clustered t>=2.5, ex5>0,
green-week share >= base+5pp, weekly P10 not worse)

| quantity | held-out (VAL) | bar | pass? |
|---|---|---|---|
| dR | +0.0088 R | >= +0.05 R | **FAIL** |
| day-clustered t | 1.52 | >= 2.5 | **FAIL** |
| ex-top-5% dR | -0.0196 | > 0 | **FAIL** |
| green-week share | 50.0% (base 40.9%, +9.1pp) | >= base+5pp | PASS |
| weekly P10 | -28.91R (base -26.90R) | not worse | **FAIL** |

**H57 FAILS the pass bar (4/5 clauses), same conclusion as 1681's own H36** (which also failed 3/5). fired_share
16.4% on VAL -- a small, late-day overlay; TRAIN-H2 context row: dR +0.0092, day_t 1.25, ex5 -0.0198, green-wk +7.4pp.

## H58: H57 stacked on the 1681 joint H36, read once on VAL

Per PREREG_1682's synthesis clause ("the 1,682 joint is also read stacked on it" -- the 1,681 joint H36 passed
enough of its own bar to be the reference). H36's mechanism (H35's leg, then H16, checked every bar with
precedence) runs first, exactly as `_h36`; H57's two clock snapshots (H40 @14:30, H54 @15:00) then act as an
additional layer on whatever fraction/target H36 leaves open. Implemented `1681_hypotheses.py::_h58`, model=True
(inherits H16's out-of-sample P(+1R,15m) lookup via the SCORINGS reads/swap). Held-out verdict row =
`TRAIN-H2->VAL` (eval population VAL, n=3,157, same convention as H36's own verdict).

| quantity | H58 held-out (VAL) | H36 alone (VAL, for comparison) | bar | H58 pass? |
|---|---|---|---|---|
| dR | +0.0131 R | +0.0086 R | >= +0.05 R | **FAIL** |
| day-clustered t | 2.28 | 2.53 | >= 2.5 | **FAIL** (H36 alone barely passed) |
| ex-top-5% dR | -0.0231 | -0.0061 | > 0 | **FAIL** (worse than H36 alone) |
| green-week share | 50.0% (base 40.9%, +9.1pp) | 45.5% (+4.5pp) | >= base+5pp | PASS (H36 alone narrowly failed) |
| weekly P10 | -26.86R (base -26.90R, +0.04) | -24.30R (+2.60) | not worse | PASS (barely; weaker than H36 alone) |

**H58 FAILS the pass bar (3/5 clauses).** Stacking H57 onto H36 raises the point estimate (+52% relative to H36
alone) and turns the green-week gate from FAIL to PASS, but it also pulls day-clustered t below 2.5 (H36 alone
cleared it) and makes ex5 more negative (more tail-dependent, not less) -- stacking two marginal, mostly-independent
overlays adds variance roughly as fast as it adds mean here. Swap-scoring row (`VAL->TRAIN-H2`, TRAIN-side, context
only): dR +0.0114, day_t 1.01, ex5 -0.0268, green-wk +7.4pp.

## Adequacy review (CLAUDE.md: no closure without one)

This is a null result for H37-H56 individually and for both joints (H57, H58) against the pre-committed pass bar --
not a claim that literature/practitioner exits carry no information on this population. What the test COULD and
COULD NOT rule out: (1) 17/20 rules fail on tail dependence alone, the same shape as 1681's 27/35 -- this
population's TRAIN point estimates are systematically tail-optimistic regardless of the mechanism tried, a property
of THIS TEST (small n per half, day-clustered, thin book) more than of any one rule. (2) Two rules (H40, H54) are
genuinely NOT tail-carried (ex5 close to zero) and worth keeping in view, but both fire on <17% of fills -- too rare
at this sample size to resolve a VAL sign flip (H54) or a P10 that goes the wrong way on VAL (H40); this is a
FREQUENCY problem, not evidence of zero effect (CLAUDE.md: "FREQUENCY, not confidence, is the gating quantity").
(3) H43 (Bayesian de-risk) and H49 (no-2R-exit) both score positive with a sensible, inspectable mechanism (H43's
saved/forgone split is a real behavioural story; H49's forgone=0.0000 on TRAIN is structural, not noise) but fail
ex5 -- these are better candidates for a FUTURE frequency-first or tail-capped re-test than for a flat close. (4) No
number here should be read as "exits from the literature don't work" -- only that, against THIS pass bar, on THIS
5,506-fill population, none of the 20 individually-fixed-threshold rules nor either joint clears it. MDE at VAL
n=3,157 under this book's day-clustered variance is ~0.006-0.009R (1681's own figure, unchanged population/variance
structure) -- the bar's +0.05R threshold is well above the detectable floor, so a true small effect could be
present and still fail on magnitude alone.

**Verdict: no rule and no joint (H57, H58) ships. Dry-run/PAPER gate not met.** Programme count: 20 rules x 2 reads
x 2 lenses + 2 joints x 2 reads = 84 reads this cell; running total on the HOD line now > 5,480 (PREREG's own
estimate).

## Output files

`1681_hypotheses.py` (H37-H58 added, harness reuse), `1682_insights.md` (per-rule mechanism/why), `1681_reads.csv` /
`1681_per_fill.csv` (appended, atomic, ids H37-H58), `1682_day_opens.csv` (H51's day-open cache, derived/research
data, not `data/*.db`), `1681_hypotheses.log` (this run's lines appended -- no separate 1682 log file was created;
the harness's existing logger/file target was reused, consistent with "harness reuse").
