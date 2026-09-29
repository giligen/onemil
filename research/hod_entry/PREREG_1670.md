# PREREG — cell 1,670: the feature-timing map — where the information about a HOD trade's fate lives, by minute and by family (FROZEN 2026-09-29 19:58 UTC)

Owner, 2026-09-29 19:55 UTC: "maybe you also trained the model on the wrong features, e.g. features during entry vs
features 5 min in or 10 min in, or maybe 60 min in; feature selection is an art." 1,668 used one fixed feature set at
k = 5/10; 1,669 uses the fill bar and minute 1. This cell maps, systematically and once, which feature FAMILY carries
information at which decision minute, and prices the only thing that matters: the cut from that minute onward.

## Population, cost, halves
As 1,668/1,669: the 1,663 join, primary r_pct ≥ 1.5 % (n 5,506), unfloored beside; bars from the completed store (own
day 100 %; SPY days backfilled 2026-09-29 20:00 UTC); cost entry 7 bps, stop 6, target 0, EOD 11, early cut 6.

## Decision minutes and the correct label
k ∈ {0, 1, 2, 5, 10, 30, 60} minutes after the fill bar. At each k the population is the trades STILL OPEN at k
(neither stopped nor targeted nor past EOD). Label: exits at the stop AFTER k. Money quantity: the trade's remaining R
from k onward (base R − mark-to-market R at k) — a cut at k trades that remainder for −6 bps.

## Feature families (each computed only from data available at the close of bar fill+k)
 A arm-time: r_pct, ATR %, level vs open / VWAP-at-arm, cumulative volume / ADV20 at the arm, minutes since open, level
   age, gap %, 5-day and 20-day return, ADV20 (all from 1663/1667 features, prior-session or through-the-level-bar only)
 P path so far: mark-to-market R at k, MFE and MAE in R through k, minutes since the last new high, whether the level
   was re-touched, bars since the last higher low
 V volume and flow: cumulative volume since fill / break-bar volume, volume of the last 3 bars / mean bar volume, progress
   per unit volume, dollar volume since fill / ADV20
 M market: SPY return fill → k, SPY return open → k, SPY 5-day return (from the backfilled SPY bars; VOID if the rail fails)
 S shape and patterns: CLV of the last bar and its mean over the k bars, wick and body shares, red-bar share, the 61
   TA-Lib flags on bars through fill+k (bullish and bearish counts + last-bar flags)
 ALL = A ∪ P ∪ V ∪ M ∪ S. At k = 0, P/V/S use the fill bar only (its close is after the fill instant, so it is known at
 the close of that bar); M uses SPY through the fill bar.

## Reads
R1 the map: for each k × family ∈ {A, P, V, M, S, ALL}: sklearn HistGradientBoostingClassifier (max_iter 200, defaults,
   n_jobs 1), TRAIN-H2 → VAL and the swap, out-of-sample AUC for the label above; within-day label-shuffle placebo AUC
   for ALL at each k. Reported as a 7 × 6 table per scoring with n open at k.
R2 the money: for each k, with ALL: the cut at FIXED τ ∈ {0.5, 0.6, 0.7, 0.8} and the variant "remaining stop ≥ 0.5 R"
   — paired ΔR vs holding on the open-at-k population, iid and day-clustered t, ex-top-5 % ΔR, MDE, share cut, and the
   1,669 decomposition (saved / forgone / cost, break-even precision vs achieved). Both scorings.
R3 permutation importances (n_repeats 5) of ALL at each k on the scoring half: the top 10 features with family tags.

## Pass bar
A (k, τ, variant) cut ships to paper only if paired ΔR ≥ +0.05 R, t ≥ 2.5, ex-top-5 % ΔR > 0 on BOTH out-of-sample
scorings, placebo AUC ≤ 0.55. A pass → independent rebuild from this prose. A null reports the map, the break-even vs
achieved precision at the best k, and the MDE, before any "no lift" sentence.

## Multiplicity
R1: 7 × 6 × 2 = 84 AUC reads (+ 7 placebos); R2: 7 × 4 × 2 × 2 = 112 paired reads. Programme count on the HOD line:
> 2,300 after this cell.

## Not allowed
Fitting τ or k; features from after bar fill+k; any exit-type feature; family or feature changes after seeing the map;
pooled-only numbers; the training-half ΔR as a result. The model is not tuned (defaults) — this cell maps information,
it does not optimise a model.

## Output
`research/hod_entry/RESULT_1670.md` (≤ 160 lines: the map tables first, the money table, importances, verdicts,
adequacy), `1670_map.csv`, `1670_reads.csv`, `1670_per_fill_k.csv` (fill_id × k: open flag, label, mtm R, P(stop) per
family), `1670_timing_map.py`, `1670_timing_map.log`. The agent returns ≤ 150 words.
