# PREREG — cell 1,676: candle shapes, in depth — the seven holes of 1,668–1,671 (FROZEN 2026-09-30 06:50 UTC)

Owner 06:45 UTC: "your research on candle symbols/shapes was not in-depth enough… find all the holes." Holes admitted:
post-entry only; 1-minute bars only; patterns without trend context; no exhaustion/climax bar; no level-rejection or
consolidation structure before the break; no daily candles; failure labels only; a descriptive fire-rate table instead
of a paired lift. This cell closes all seven on the same book and bar. Nothing here is fitted.

## Population, cost, halves
The 1,663 join (fills_1658 ⋈ causal_arming_causal), primary r_pct ≥ 1.5 % (n 5,506), unfloored beside; `split` halves;
the completed bar store (own day 100 %, prior 20 sessions, SPY). Cost: net_R as before. Daily candles from
`research/overnight_high/panel_2024_2026.parquet` (prior sessions and the day's OHLC through the level bar).

## Frames
1-minute bars as stored; 5-, 10- and 15-minute bars built by aggregating the 1-minute bars aligned to 09:30 ET (open
of the first, high/low extrema, close of the last, volume sum). A frame's feature at an instant uses only bars that
CLOSED before that instant (a partial bar is never used). Every feature group and every read below runs on ALL FOUR
frames (owner 07:00 UTC: "candles should also do 5 min, 10 min and 15 min"); the multiplicity count doubles
accordingly and is reported as such.

## Feature groups (each computed at TWO instants: the ARM instant = close of the level bar (pre-entry), and the close
## of bar fill+1 (post-entry); at both frames where meaningful)
 G1 break-bar geometry (1-min and 5-min bar containing/ending at the level bar): CLV, body share, upper-wick share,
    lower-wick share, range/ATR14, volume/mean bar volume, close vs level.
 G2 structure before the break, last 10 and 30 bars in each frame: number of upper wicks that touched the level ±
    5 bps (rejections), count of higher lows, range contraction (range of last 5 / range of prior 10), consolidation
    tightness (std of closes / ATR), share of red bars, net progress per bar, volume trend (slope of bar volume).
 G3 climax / exhaustion: any bar in the last 10 with volume ≥ 3 × mean AND range ≥ 2 × mean AND CLV ≤ 0.5 (flag +
    minutes since); the same on the 5-min frame; "effort vs result" residual (range regressed on volume, last 30 bars).
 G4 context-conditioned patterns: the 61 TA-Lib flags on each frame, each ALSO interacted with the prior move
    (return over the previous 10 bars in ATR units, sign) — reported as pattern × context cells with counts, and as
    features; only patterns firing ≥ 50 times on the book are read individually.
 G5 daily candles: prior day's CLV, body share, upper-wick share, gap %; the day's candle so far at the arm (CLV of the
    day so far, where the level sits in the day's range, the day's range / ATR14); the two prior days' pattern flags
    (TA-Lib on the daily frame with the day-so-far bar as the last bar).
 G6 post-entry shapes (the 1,668 set on the 5-min frame too): CLV, wicks, body, red share over bars fill+1..fill+k,
    k ∈ {5, 15}, at both frames.

 G7 ROLLING post-entry shapes, every minute (amendment 1, 07:05 UTC, before any number; owner: "also run them every
    minute post entry, on the past 5 min / 10 min / …"): at each k ∈ {1, 2, …, 15, 20, 30, 45, 60} minutes after the
    fill bar, the TRAILING candle of width w ∈ {5, 10, 15} minutes ENDING at bar fill+k (open of bar fill+k−w+1,
    high/low extrema, close of bar fill+k, volume sum — unaligned to the clock, so it exists at every k once k ≥ w;
    for k < w the candle spans from the fill bar), and the series of the last 5 non-overlapping trailing candles of
    width w ending at k (for TA-Lib and for the G2 structure counts). Features per (k, w): CLV, body share, upper and
    lower wick shares, range/ATR14, volume/mean bar volume, close vs fill in R, the 61 TA-Lib flags on the trailing
    series, the climax flag (G3) on the trailing series, effort-vs-result residual over the trailing series.
    Reads: R3-style models PER k for the labels "stop after k" and "MFE ≥ +1 R within the next 15 min from k" (a
    rolling success label), groups = G7 alone, G7 + the 1,670 ALL family; both scorings; placebo at k ∈ {1, 5, 15, 30};
    permutation importances at k ∈ {1, 5, 15}. Money read at each k: the cut, the short (1,673 geometry) and the add
    (1,670 R4 geometry) gated by the G7 (+ALL) model at fixed τ ∈ {0.6, 0.7}, paired vs the base, both scorings — the
    same bars and decomposition as 1,669/1,670/1,673. Multiplicity: 19 k × 3 w × ~15 features + 19 × 2 labels × 2
    groups × 2 scorings AUC + 19 × 3 actions × 2 τ × 2 scorings paired reads ≈ 1,100 more reads; stated, and protected
    by the both-scorings rule and the sign line.

## Labels
Failure: stop-out (any) and stop within 10 min. SUCCESS at short horizon: MFE ≥ +1 R within 15 min; target reached.
Money: net_R (the standard paired reads).

## Reads
R1 paired lifts with the bar, pre-entry (ARM instant) — for every G1–G5 feature: terciles/quintiles on the floored
   book, both halves, n / mean net R / iid t / day-clustered t / ex-top-5 % / fills per week / MDE; top-vs-rest paired
   ΔR. Tercile edges from TRAIN only.
R2 pattern × context cells (G4): for each (pattern, context sign) with ≥ 50 fires: mean net R vs the book, both
   halves, t, count — the honest version of the fire-rate table.
R3 models by group and frame: HistGradientBoosting (defaults, max_iter 200, n_jobs 1), TRAIN → VAL and the swap,
   for each label: AUC of G1, G2, G3, G4, G5, G6, ALL-shape, and ALL-shape + the 1,670 ALL (does shape add to path
   and volume?), placebo (within-day shuffle), permutation importances top 10.
R4 the money read of the best pre-entry cut and of a shape-gated entry (trade only fills whose shape-model
   P(success) ≥ τ at the ARM instant, τ ∈ {0.5, 0.6, 0.7} fixed): paired ΔR, t, ex-top-5 %, fills/week, MDE.

## Pass bar (unchanged)
A cut or gate ships to paper only if paired ΔR ≥ +0.05 R, t ≥ 2.5 (day-clustered) in BOTH halves (for R4: both
scorings), ex-top-5 % > 0 in both, ≥ 3 fills/week; pooled day-clustered estimate + sign line reported beside every
read (the 1,674 lesson). A pass → independent rebuild from this prose before the owner sees a number.

## Multiplicity
≈ 40 features × 2 instants × 2 frames × (8 reads) ≈ 1,300 tercile/quintile reads + R2 ≈ 100 + R3 ≈ 64 AUC + R4 ≈ 12.
Stated so the reader knows: at this count, a single-half t of 2.5 is expected ~8 times by chance; the both-halves rule
and the sign line are what protect the read. Programme: > 4,000 on the HOD line.

## Not allowed
Feature or frame changes after seeing numbers; bars that closed after the instant; exit-type features; fitting τ.

## Output
`research/hod_entry/RESULT_1676.md` (≤ 200 lines: R3 table first, then R1 rows with |t| ≥ 2 in either half, R2
cells, R4, verdicts, adequacy), `1676_features.csv`, `1676_reads.csv`, `1676_patterns.csv`, `1676_shapes.py`,
`1676_shapes.log`. The agent returns ≤ 150 words.
