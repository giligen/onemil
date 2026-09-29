# PREREG — cell 1,667: every causal arm-time feature, bucketed on the floored HOD book (FROZEN 2026-09-29 18:14 UTC)

Owner, 2026-09-29 18:12 UTC: "you should look at buckets again with the new 1.5% — stock price, distance from avg price,
I don't know, you figure this out." Cells 1,658 (stop distance), 1,663 (ATR %, stop/ATR, time of day, target distance,
price, interactions), 1,665 (relative volume) each cut one axis. This cell cuts EVERY feature knowable at the arm that
can be built from data already on disk, on the same floored book, under one bar, with the multiplicity stated.

## Population and cost
`research/hod_entry/1663_features.csv` (fills_1658 ⋈ causal_arming_causal, status == fill; 9,911 rows; carries r_pct,
split, ATR, stop bucket, level, fill, stop, fill_min). Primary book: r_pct ≥ 1.5 % (n 5,506); unfloored beside it.
Cost: the `net_R` column (measured 7 bps entry, 6 bps stop). Halves: `split` (TRAIN-H2 / VAL). MDE at the primary book:
0.077 R (TRAIN) / 0.066 R (VAL) — a cut must carry ≥ +0.10 R on ≥ a third of the book to be visible at all.

## Features (all computed from bars that CLOSED before the level bar's end, or from prior sessions only)
Daily panel `research/overnight_high/panel_2024_2026.parquet` (prior sessions only for anything "average"):
 F1 gap % = open(d) / close(d−1) − 1                       F2 level extension vs prior close = level / close(d−1) − 1
 F3 level vs SMA20(d−1) = level / SMA20 − 1                 F4 level vs SMA50(d−1)
 F5 return d−1 (close/close)                                F6 return over 5 sessions to d−1
 F7 return over 20 sessions to d−1                          F8 ADV20(d−1) dollar volume (liquidity)
 F9 ATR14(d−1) / close(d−1) is already in 1663_features (skip)  F10 SPY return over 5 sessions to d−1 (panel or daily_bars)
Intraday `research/bf_zero/bars_sip.db` (same-day bars through the level bar; coverage ≈ 81 %, rail applies):
 F11 level vs VWAP to the level bar = level / VWAP − 1     F12 level vs open(d) = level / open − 1
 F13 day range to the level bar / ATR14                    F14 level age = fill_min − level bar minute (minutes)
 F15 dollar volume to the level bar / ADV20 dollar          F16 SPY return open → level bar minute (bars_sip SPY, if present)
From the join itself: F17 n_cross (number of crosses before the fill), F18 stock price (already read in 1,663 as cut f —
 re-read here only in the interaction with F2/F3, not alone).

## Reads (pre-declared, both halves, every t with day-clustered SE and the MDE beside it)
R1 terciles and quintiles of every feature F1–F17 on the primary book: n, mean net R, t, ex-top-5 %, fills/week.
R2 top and bottom quintile vs the rest, paired as a cut of the same book: ΔR, t, ex-top-5 % ΔR.
R3 Spearman ρ of every feature vs net R per half (monotonicity).
R4 the two-way interactions stop bucket (1.5–3 %, ≥ 3 %) × F2, F3, F11, F12 terciles (the "extension" family the
 owner named), and price tercile × F2.
R5 coverage line per feature (fills with the feature / book, winner vs loser missingness). A feature under the rail
 (coverage < 80 % or gap > 5 pp) is reported as VOID, never as a number.

## Pass bar (PREREG_1662 §1,664, unchanged)
A cut ships to paper only if net ≥ +0.05 R AND t ≥ 2.5 in BOTH halves AND ex-top-5 % > 0 in both AND ≥ 3 fills/week at
the live config. A pass triggers an independent rebuild of that feature from prose before any number reaches the
owner. A null states the MDE and the coverage before any "no lift" sentence.

## Multiplicity
17 features × (3 + 5 + 2) + 17 ρ + 5 interactions × 6 ≈ 217 reads. The both-halves rule at t ≥ 2.5 keeps the expected
false passes below 0.01 for the whole sweep; the programme count on the HOD line passes 1,900.

## Not allowed
Adding features after seeing numbers; conditioning on exit type or anything after the level bar; refitting the floor;
pooled-only numbers; same-day ratios that use the fill bar.

## Output
`research/hod_entry/RESULT_1667.md` (≤ 150 lines: coverage table first, then one table per read family with only the
rows whose |t| ≥ 2 in either half plus the full CSV pointer, then the verdict per feature, then the adequacy review),
`1667_features.csv`, `1667_reads.csv` (every read × half), `1667_sweep.py`, `1667_sweep.log`. The agent returns ≤ 150 words.
