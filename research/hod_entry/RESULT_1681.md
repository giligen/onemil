# RESULT — cell 1,681: HOD mid-flight exit decisions (35 hypotheses + the joint)

PREREG: `PREREG_1681.md` (FROZEN 2026-09-30 17:50 UTC). Book: the 1,663 join, floored r_pct >= 1.5%, n=5,506;
split into TRAIN-H2 (n=2,349) and VAL (n=3,157) by `split`. Base = the live rule. Exits at next-bar open - 6bps;
partials in original-R units. Full per-hypothesis insight prose (what, why, which trades): `1681_insights.md`.
Raw reads: `1681_reads.csv` (70 rows for H1-H35, + 2 for H36). Per-fill rows: `1681_per_fill.csv`.

**Read convention** (verified against `1681_reads.csv`, not assumed): "TRAIN" below = the read whose EVALUATION
population is TRAIN-H2 (n=2,349) -- for non-model hypotheses that is literally the `TRAIN-H2` row; for model
hypotheses (probabilities looked up read-only from 1677/1670, never re-fit) it is the `VAL->TRAIN-H2` row (model
info from VAL, evaluated on TRAIN-H2, so the probability lookup stays out-of-sample for whichever half is being
scored). "VAL" = eval population VAL (n=3,157): the plain `VAL` row, or `TRAIN-H2->VAL` for model hypotheses.
`t` = day-clustered t (iid t is in `1681_reads.csv` and in the prose where it diverges materially from day t).
Green-week / P10 deltas are TRAIN vs VAL baselines of 44.4% / -25.45R and 40.9% / -26.90R respectively (differ
because the two halves cover different weeks).

## The 35 hypotheses (paired dR vs the live rule, whole book, unfired=0)

| id | rule | dR (TRAIN/VAL) | t_day (TRAIN/VAL) | ex5 (TRAIN/VAL) | green-wk pp (TRAIN/VAL) | P10 delta R (TRAIN/VAL) | verdict |
|---|---|---|---|---|---|---|---|
| H1 | exit @30m if mtm<+0.25R | -0.0225 / -0.0052 | -0.68 / 0.11 | -0.091 / -0.073 | -11.1 / +0.0 | +2.29 / -5.74 | null, forgone>saved 1.6x |
| H2 | exit @60m if mtm<+0.5R | -0.0138 / 0.0011 | -1.81 / 0.30 | -0.087 / -0.072 | -7.4 / +0.0 | +2.16 / +0.36 | flat, sign-flips |
| H3 | exit @90m if mtm<+0.75R | -0.0049 / -0.0018 | -1.22 / 0.14 | -0.082 / -0.077 | +3.7 / +9.1 | +0.49 / -3.99 | null, best F1 consistency gain |
| H4 | >20m no new high AND mtm<+0.5R | -0.0242 / -0.0047 | -0.55 / 0.10 | -0.098 / -0.079 | -11.1 / -9.1 | +5.17 / +0.83 | null, worst green-wk |
| H5 | after +1R, no new high 15m | -0.0346 / -0.0166 | -0.78 / -0.35 | -0.130 / -0.111 | -14.8 / -13.6 | +1.59 / +2.54 | negative, worst F1 dR |
| H6 | bearish engulfing 5m >=+0.5R | -0.0081 / -0.0053 | -0.67 / 0.04 | -0.105 / -0.103 | +7.4 / -9.1 | +4.33 / -0.20 | sign-unstable |
| H7 | close<trail-10m-low >=+0.5R | -0.0090 / -0.0150 | -0.77 / -0.55 | -0.103 / -0.106 | +3.7 / -13.6 | +2.09 / -2.69 | negative, VAL green-wk hit |
| H8 | 5m close<day VWAP >=+0.5R | -0.0002 / 0.0005 | -1.05 / 0.51 | -0.002 / -0.002 | +0.0 / +0.0 | +0.00 / -0.38 | non-event, too rare |
| H9 | upper-wick reject >=+1R | 0.0008 / -0.0151 | -0.40 / -0.38 | -0.114 / -0.126 | +3.7 / -13.6 | +4.51 / -1.39 | tail-propped, VAL hit |
| H10 | climax bar (vol>=3x,CLV<=0.5) any profit | 0.0044 / 0.0014 | -0.59 / 0.38 | -0.088 / -0.087 | -7.4 / -9.1 | +5.25 / +1.09 | coin-flip, costs green-wk |
| H11 | trail-15m candle red after +1R | -0.0404 / -0.0167 | -0.93 / -0.12 | -0.138 / -0.113 | -18.5 / -18.2 | +1.21 / +2.71 | worst F2, clips runners |
| H12 | 2 lower 5m highs >=+0.5R | -0.0122 / -0.0115 | -0.37 / 0.20 | -0.110 / -0.109 | +3.7 / -13.6 | +2.46 / +1.17 | negative, overlaps H7/H9 |
| H13 | P(+1R,15m)<0.3 >=+0.5R | -0.0182 / -0.0008 | -1.39 / 0.92 | -0.114 / -0.092 | +0.0 / +0.0 | +2.95 / +3.03 | negative both reads |
| H14 | P(stop)>0.7 >=+0.25R | -0.0205 / -0.0133 | -1.36 / -0.26 | -0.089 / -0.093 | -11.1 / -4.5 | +1.40 / +4.52 | negative, cuts fine trades |
| H15 | H13 AND H14 | -0.0006 / -0.0051 | 0.12 / -1.09 | -0.008 / -0.015 | +0.0 / +4.5 | +0.09 / +0.55 | too sparse, uninformative |
| H16 | k>=60,mtm>=+1R,P<0.3 (1677 ref) | 0.0024 / 0.0041 | 0.04 / 2.02 | -0.009 / -0.005 | +0.0 / +0.0 | +1.26 / +2.06 | small +, tail-leaning, rare |
| H17 | P-weighted partial @+1R | -0.0499 / -0.0469 | -7.17 / -8.21 | -0.074 / -0.067 | -11.1 / -18.2 | -4.63 / -3.89 | worst F3, close it |
| H18 | 50% out @+1R | 0.0002 / -0.0036 | 0.08 / 0.62 | -0.063 / -0.064 | +0.0 / -4.5 | +5.03 / +3.56 | variance cut, dR~0 |
| H19 | 33%@+1R,33%@+2R,trail MFE-1R | 0.0023 / -0.0010 | 0.35 / 0.79 | -0.042 / -0.043 | +3.7 / +0.0 | +3.09 / +2.61 | diluted H18 |
| H20 | 50%@+1R only if P(+1R,15m)<0.5 | 0.0072 / 0.0057 | 1.61 / 1.70 | -0.019 / -0.017 | +3.7 / +4.5 | -0.16 / +1.03 | model-gated, FAILS P10 |
| H21 | 50%@+1R + stop to BE | -0.0128 / -0.0119 | -0.49 / 0.10 | -0.102 / -0.098 | +0.0 / -9.1 | +3.16 / +3.51 | worst F4 ex5, BE shakeout |
| H22 | 25% out @+0.5R | -0.0006 / -0.0036 | 0.23 / 0.81 | -0.032 / -0.032 | +0.0 / -4.5 | +3.63 / +5.29 | widest reach, variance only |
| H23 | target +1.5R if 30m candle red | -0.0070 / -0.0031 | -1.37 / -0.90 | -0.066 / -0.067 | +0.0 / -4.5 | -0.39 / -4.14 | too blunt, P10 worsens |
| H24 | target +3R if P>0.7 @+1R | 0.0024 / 0.0016 | 1.25 / 0.74 | -0.000 / -0.000 | +0.0 / +0.0 | -0.01 / +0.24 | tiny reach, FAILS P10 |
| H25 | stop to BE @+1R | -0.0215 / -0.0119 | -0.54 / -0.31 | -0.082 / -0.069 | -7.4 / -13.6 | +1.43 / +0.12 | classic BE, green-wk craters |
| H26 | stop +0.5R @+1.5R | -0.0065 / -0.0098 | -0.43 / -0.57 | -0.079 / -0.075 | -3.7 / -9.1 | +3.12 / -0.93 | VAL P10 sign-flip |
| H27 | trail=10m candle low @+1R | -0.0326 / -0.0112 | -0.69 / 0.35 | -0.129 / -0.107 | -14.8 / -13.6 | +1.75 / +6.21 | tightest trail, worst green-wk |
| H28 | trail=day VWAP @+1R | 0.0008 / -0.0043 | 0.06 / -0.46 | -0.047 / -0.050 | +0.0 / -4.5 | +1.35 / +0.45 | loosest trail, best risk-adj F5 |
| H29 | vol<0.3x AND mtm<+0.5R @20m | -0.0275 / -0.0118 | -1.01 / 0.28 | -0.100 / -0.087 | -18.5 / -4.5 | +7.23 / -1.15 | broad, taxes middle |
| H30 | red bar vol>=3x >=+0.5R | 0.0103 / -0.0053 | -0.02 / -0.49 | -0.070 / -0.080 | +0.0 / -4.5 | +3.43 / -6.10 | rare, sign-flips |
| H31 | hold @+1R only if up-vol>down-vol | -0.0136 / -0.0153 | -0.46 / 0.25 | -0.118 / -0.119 | -11.1 / -18.2 | +9.90 / +4.77 | best F6 P10, worst Sharpe |
| H32 | H13 AND H7 | -0.0023 / 0.0024 | -1.38 / 0.48 | -0.004 / -0.004 | +0.0 / +4.5 | +0.00 / +0.13 | underpowered AND (14-26 fires) |
| H33 | H18 then H24 on remainder | 0.0002 / -0.0036 | 0.08 / 0.62 | -0.063 / -0.064 | +0.0 / -4.5 | +5.03 / +3.56 | inherits H18 shape |
| H34 | H4 AND H14 | -0.0041 / 0.0016 | -1.79 / 1.62 | -0.005 / -0.001 | +0.0 / +0.0 | -0.37 / +1.21 | underpowered AND, FAILS P10 |
| H35 | H21 AND H8 | 0.0063 / 0.0043 | 1.86 / 1.57 | -0.002 / -0.001 | +0.0 / +0.0 | +0.01 / +0.70 | tiny n, TOP TRAIN SCORER |

## F7 combinations (H32-H35) — headline

All four combinations were run with `--lens` this pass (previously only the non-lens core stats existed). None
beats its own constituents: H32 and H34 (ANDs of two already-weak singles) collapse to 8-26 fires and sign-flip
TRAIN/VAL; H33 (H18 chained into H24) just inherits H18's variance-cut shape with no incremental edge from H24;
H35 (H21 AND H8) is the sweep's single biggest TRAIN point estimate but on 3-4 total fills. Full per-hypothesis
mechanism/decomposition prose: `1681_insights.md` ("## F7 combinations").

## Synthesis (fixed before any number; scored on the TRAIN read only)

S = dR_TRAIN + 0.5 x (green-week gain, TRAIN, in R-equivalents: 1pp = 0.01R). Constraints (hard gate, both must
hold on TRAIN): ex-top-5% dR > -0.02, weekly P10 not worse. **27/35 (77%) fail on ex-top-5% alone or with P10**
-- the single biggest finding of the sweep: almost every nominally-interesting TRAIN point estimate in this
population is carried by its best 5% of trades and would not survive a cap. 3 more (H20, H24, H34) pass ex5 but
fail P10. Eligible (5): 

| id | ex5_TRAIN | P10 delta_TRAIN | S |
|---|---|---|---|
| H35 | -0.002 | +0.01 | **+0.00630** |
| H16 | -0.009 | +1.26 | +0.00241 |
| H8 | -0.002 | +0.00 | -0.00022 |
| H15 | -0.008 | +0.09 | -0.00064 |
| H32 | -0.004 | +0.00 | -0.00233 |

Top scorer = **H35**. Other eligible hypotheses with a positive TRAIN score: **H16** only (H8, H15, H32 score
negative, excluded by the rule). H35 and H16 act on different triggers (H35 = mtm>=+1R AND 5m-close<VWAP -> 50%
out + breakeven remainder; H16 = bar k=60 AND mtm>=+1R AND model P<0.3 -> full exit) -- compatible. No third
positive-score compatible candidate exists, so the joint has 2 members, not 3.

**H36 (the joint)**: at every bar, check H35's leg condition first (precedence = higher TRAIN score); if it has
already fired on this fill, do not check H16 again (the position is already reshaped, not eligible for a second
full-exit action); if H35 has not fired, check H16 and fully exit if it fires. Implemented in
`1681_hypotheses.py::_h36`, registered `model=True` (inherits H16's out-of-sample probability lookup), run once
via `--run H36 --lens`.

## The joint's held-out read (VAL, eval population VAL n=3,157 -- the `TRAIN-H2->VAL` row; the `VAL->TRAIN-H2`
swap-scoring row, n=2,349, is TRAIN-side and is reported below for context ONLY, never as the verdict)

| quantity | held-out (VAL) | swap (TRAIN, context only) |
|---|---|---|
| n / fired | 3,157 / 44 (1.39%) | 2,349 / 45 (1.92%) |
| dR (paired) | **+0.0086 R** | +0.0087 R |
| iid t / day t | 2.60 / **2.53** | 1.87 / 1.23 |
| ex-top-5% dR | **-0.0061** | -0.0109 |
| green-week | 40.9% -> 45.5% (**+4.55 pp**) | 44.4% -> 48.1% (+3.70 pp) |
| weekly P10 | -26.90 -> -24.30 (+2.60 R) | -25.45 -> -24.25 (+1.19 R) |
| weekly Sharpe | -0.139 -> -0.100 | -0.093 -> -0.058 |
| maxdd | -230.81 -> -212.92 | -146.36 -> -118.48 |

**Pass bar (PREREG, held-out read only):**

| clause | required | observed | pass? |
|---|---|---|---|
| paired dR | >= +0.05 R | +0.0086 R | **FAIL** (5.8x short) |
| day-clustered t | >= 2.5 | 2.53 | pass (barely) |
| ex-top-5% dR | > 0 | -0.0061 | **FAIL** |
| green-week share | >= base + 5 pp | +4.55 pp | **FAIL** (just short) |
| weekly P10 | not worse | +2.60 R | pass |

**Verdict: FAIL.** 3 of 5 clauses fail, including the primary magnitude clause by 5.8x. The day-clustered t
clearing 2.5 is a knife-edge pass on an effect that is itself below this read's own MDE (0.0093R vs an observed
0.0086R) and that turns negative when the best 5% of trades are removed -- exactly the tail-redistribution pattern
flagged in `feedback_paired_lift_tail_check.md`. This is not a statistical-vs-economic-significance split to
relitigate; three independent clauses agree it fails.

## Decomposition of the joint (held-out VAL read)

44 fires (1.39% of the book): 21 "saved" (dR>0, mean +1.394R, sum +29.27R) vs 23 "forgone" (dR<0, mean -0.717R,
sum -16.50R) by simple sign-of-dR bucketing on `1681_per_fill.csv` (bucket means match `1681_reads.csv`'s
saved/forgone columns exactly; the bucket sums do not reconstruct the whole-book dR*n exactly under this
project's `decompose()` -- same non-reconciling pattern already present in the F1-F7 per-hypothesis insights, not
specific to H36). Roughly half of what H36 touches is a real save and half gives back a smaller amount -- more
trades saved than forgone by count (48% vs 52%), each forgone trade costing about half of what each saved trade
gains, which is *why* the point estimate is positive -- but on only 44 trades out of 3,157, and losing its sign
under the top-5% cap.

## Adequacy (MDE) and the best single hypothesis

Programme count: 35 hypotheses x 2 reads x 2 lenses + 1 joint = 141 reads, consistent with the PREREG's >5,300
running total on the HOD line. At n=3,157 (VAL) and this population's day-clustered variance, MDE ~ 0.006-0.009R
per hypothesis (0.0093R for H36's held-out read) -- most of the 35 singles' point estimates (median |dR| ~0.01R)
sit inside or barely outside their own MDE, which is *why* day-clustered t rarely exceeds 1.5 anywhere in this
sweep outside H17 (a confirmed-negative, F3's P-weighted partial).

**Per PREREG: a null reports the best single hypothesis.** That is **H35** (H21 AND H8: 50% out + breakeven stop
at +1R, gated additionally on a 5m close below day VWAP) -- TRAIN dR +0.0063R (day t 1.86, ex5 -0.0021, MDE
0.0087R), VAL dR +0.0043R (day t 1.57, ex5 -0.0009, MDE 0.0057R). Its own insight (`1681_insights.md`) is explicit
that this is 3-4 total fills across the whole book: H8's near-never VWAP-close trigger, not H21's 45%-reach
breakeven mechanism, sets the combination's rarity, and the "significant" t-stats are a small-sample artifact, not
a readable signal. **No hypothesis in this sweep, singly or combined, clears the pass bar.** The candle-shape
"story line" (F2), the trained model (F3) and every combination of the two (F7) all read as either broadly
negative, a tail-only artifact, or too rare to score -- consistent with the standing exit-lab finding
(`project_hod_exit_lab_sep2026.md`) that this population carries no mid-flight exit information at the individual-
trade level. HOD stays `dry_run: true`; no `hod_break.yaml` or live-rule change follows from this cell.
