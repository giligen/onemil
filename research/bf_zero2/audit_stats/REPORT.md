# Adversarial statistical audit — F6 "red to green" +2R book (2026-09-16)

Written by the caller from the auditing agent's narrative; the per-question raw outputs and scripts (a0–a7) are in this
directory. The claim reproduces exactly from `f6_2r_book.csv` and an independent re-walk of all 1,676 trades from raw
1-minute bars reproduces it to three decimals, so the simulator is not the problem.

## VERDICT
The effect is distinguishable from the search that found it (search-adjusted p = 0.0009 on the 108-cell grid, ≈0.002 on
the ~1,200-cell program census) — but that is the wrong contest to have won. The searched SETUP has no edge, 100% of the
result comes from an uncounted selection rule that is mostly a tight-stop proxy, and the same signals re-walked under the
entry-fill convention the shipped engine actually uses are NEGATIVE in all three splits.

## 1 Multiplicity — cleared
~1,145 distinct cells before the claim (bf_zero 672, score2 275, score3 108, literature 90). Effective independent tests,
from the eigenvalues of the cells' per-day return correlation matrix: ≈14 inside score3, ≈150 across the program.
Permutation (null imposed by centering each cell within split; 5-day circular blocks; the same draw applied to all cells;
10,000 replicates): P(at least one cell clears t > 2.8 on all three splits) = 0.0010. Observed max-of-min t = 2.86.
A single pre-specified cell clears that bar with probability 1.1e-5, about 600x the normal-theory value — the formula
would have been badly wrong; the bootstrap is the right instrument. Under the null a mean of 13 cells are positive on all
three splits; 25 were observed, so correlation explains most of that count.

## 2 Regime — does not explain the out-of-sample strength
SPY realised volatility FELL into the stronger splits (16.3% → 12.8% → 13.3%); cross-sectional dispersion rose 7–15%;
candidate supply rose 45%. Weekly regressions: the regime block has R² 0.017 and the split dummies survive it at
t 2.0–2.2. The unselected candidate pool shows no such pattern. What does shift is instrument mix: leveraged single-stock
wrappers go from 9.4% to 38.6% of the book and explain ≈30% of VAL's rise and ≈16% of TEST's.

## 3 Bootstrap — sampling noise alone
Pooled +4.0 R/week, 95% interval [+2.5, +5.4], P(true mean ≤ 0) < 0.1%, P(< 2R) 0.4%, P(< 5R) 92%. TRAIN alone has a 29%
chance of a true mean below 2R/week. Week-blocks and day-blocks agree to two decimals. The 10R/week target is outside
every interval.

## 4 Stability
No symbol and no day carries more than 10% of a split's profit (top symbol 6.0 / 9.8 / 5.2%); no single trade exceeds
2.1%. But 900 of 996 TRAIN, 402 of 408 VAL and 272 of 272 TEST trades enter between 09:31 and 09:35, so the "entries until
14:00" rule is vacuous. TRAIN's entire profit is a 4-percentage-point excess in the target hit rate; dropping the best 10%
of trades takes TRAIN to −55% of its profit (an arithmetic consequence of a capped +2R winner, not a tail).

## 5 Selection — the whole effect, and it was never counted
| rule | TRAIN | VAL | TEST |
|---|---|---|---|
| first-come, alphabetical tie-break (the claim) | +0.124 | +0.320 | +0.353 |
| first-come, random tie-break (200 draws) | +0.131 ± 0.014 | +0.298 ± 0.037 | +0.382 ± 0.050 |
| four RANDOM candidates per day (200 draws) | +0.015 ± 0.028 | +0.049 ± 0.046 | −0.020 ± 0.055 |
| last four of the day | +0.023 | −0.053 | −0.039 |
| whole qualifying population | −0.006 | +0.080 | +0.017 |

The alphabetical tie-break is exonerated. The entry-minute ordering is the entire effect; it survives day fixed effects
(t 2.7 / 3.5 / 4.1) but is mostly a proxy for stop distance — median stop distance rises monotonically with entry rank,
and inside stop-size strata the contrast collapses. Stop-size gradient: 1–2% stops +0.32 / +0.49 / +0.47R; 4.5–7% stops
−0.04 / +0.04 / −0.06R.

## 6 Power
Minimum detectable true effect at 80% power: +0.125R per trade (2.3 R/week) TRAIN, +0.209R (3.9) VAL, +0.256R (5.0) TEST.
TRAIN's observed effect is 1.04x its own threshold. These were one underpowered observation and two short ones.

## 7 The finding that decides it — the entry fill
Pass 1 (`research/bf_zero/build_candidates.py`) fills at `level × 1.003` the instant the signal bar's high touches it. The
shipped engine (`trading/hod_break_engine.py`, spec `trading/hod_break.py::entry_fill`, REPORT.md §6) places a capped limit
and fills at the NEXT bar's open, or not at all. One minute after the trigger the price is a median +48 bps above it
(mean +69 to +79, p90 +209 to +255); only 57–60% of signals are obtainable at ≤ +60 bps.

| entry convention | TRAIN | VAL | TEST | weeks green |
|---|---|---|---|---|
| fill at the touch (the claim) | +0.131R, +2.5/wk, t 2.95 | +0.326R, +6.1/wk, t 4.37 | +0.356R, +6.9/wk, t 3.90 | 34/53, 17/22, 12/14 |
| stop order, next bar open | −0.247R, −4.6/wk | −0.160R, −3.0/wk | −0.110R, −2.1/wk | 11/53, 8/22, 6/14 |
| live capped limit, no chase | −0.277R, −5.2/wk, t −7.0 | −0.315R, −5.8/wk, t −4.8 | −0.295R, −5.7/wk, t −3.5 | 11/53, 3/22, 2/14 |

On the ~58% that do fill under the live convention the mean is still −0.112 / −0.066 / −0.157R, so this is adverse
selection, not an artefact of scoring no-fills as zero: the signals obtainable at the level are the ones that did not
continue. Cost stress on the claim's own fill model: +75 bps of extra exit slippage takes TRAIN negative.

## 8 What would change the verdict
Rebuild pass 1 with the live entry convention as the only fill model and let the book choose among candidates that
actually fill. Pre-register "the first four signals of the day" as the rule it now is, and neutralise the stop-size
confound. Price the wrapper drift explicitly. None of this is a reason to trade the book as it stands.
