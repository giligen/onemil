# Refuter 2 (statistics lens): PREREG_1562 result

Script: `review/refute2_stats.py`. Rows: `review/refute2_rows.csv`. Recomputed from `cell_1562_fills.csv`,
with a read-only re-walk of the base leg through the builder's own `exit_walk`.

## Verdict
The FAIL stands and gets stronger. The result's secondary claims are refuted: "paired ΔR positive on
every split", "paired ΔR ≥ +0.10 on both splits: yes", "book incl. zero trades ≥ base: yes" and "the
HOD retest learning generalises in direction". They come from comparing numbers in two different units.

## D1: base_r is not an R-multiple (the main defect)
`base_r = pnl_replay/375`. But `population.csv` `pnl` is dollar P&L at $50K notional, not per share
(pnl/pnl_pct = 500), so `shares = _sized_pnl/pnl` = 0.0667 for every row. `_sized_pnl` is P&L on a
FIXED $3,333 notional, so the dollar risk is about 3,333 × range% ≈ $150, not $375. Median
base_r / true-R ratio = 0.414. The retest leg, `retest_r`, is a true per-share R′-multiple. So the paired
ΔR subtracts 0.41 × base from 1.0 × retest.

Paired ΔR with the same units:

| cell | split | builder ΔR | ΔR vs BT true-R | ΔR, same walker, fill @ limit (t_dc, ex-top-5 %) |
|---|---|---|---|---|
| 1562 | TRAIN | +0.355 | +0.045 | −0.121 (t −0.88, −0.225) |
| 1562 | VAL | +0.146 | −0.005 | −0.055 (t −0.55, −0.223) |
| 1562 | POOLED | +0.185 | +0.004 | −0.067 (t −0.78, −0.224) |
| 1563 | TRAIN | +0.445 | +0.135 | −0.010 (t −0.08, −0.112) |
| 1563 | VAL | +0.120 | −0.032 | −0.068 (t −0.64, −0.240) |
| 1563 | POOLED | +0.180 | −0.001 | −0.057 (t −0.64, −0.217) |

The base's own mean with the same walker (skips counted as zero) is +0.271 (t 2.04) pooled, not +0.083.

## D2: the never-retest cohort is the runners, and it sinks the book
Base signals that filled but never retested (n = 27 for 1562, 21 for 1563) earn **+1.66 / +2.10 R**
with the same walker (+1.58 / +1.93 in BT true-R). The builder reported +0.31 / +0.39 in $/375.

Book including the zero trades, fills at the limit: 1562 +0.204, 1563 +0.214. Base: **+0.271**. This
criterion FAILS on both cells (the builder had it passing), in VAL as well (0.159 / 0.146 vs 0.214).

## D3: the paired lift where both legs filled is the mechanical price gap
Both legs filled (n = 187 / 193): ΔR = +0.063 / +0.075. Of that:
- mechanical price gap (same exit price, entry at the limit instead of the chase fill): +0.049 / +0.077
- path: +0.014 / −0.003

The median entry gap is 0.04 to 0.06 R′. The dip-buy adds nothing on the path. The whole gain is the
missing chase slip, and the lost runners more than pay it back.

## D4: tape fills are unobtainable prices
Tape fills are priced at the first print BELOW the limit. The median is 22 bps under the limit (mean
32 bps) for 1562, and 15 / 27 bps for 1563. A resting bid fills at its own price. Repricing those fills
at the limit:
- own mean, 1562 POOLED: +0.319 → +0.243 (t 1.40, ex-top-5 % −0.149, capped +0.001)
- own mean, 1562 VAL: → +0.186 (t 0.95)

## D5: the withdrawal share is tautological
`withdrawn_15min` is "the window's minimum occurred within 15 min". For a 15-minute window this is
always true (1.000). It is not a measure of withdrawal below the level. The "93–100 % vs HOD 87 %"
comparison is meaningless. Fill share (79–88 %) is the only valid number.

## Tails and day concentration (retest own, builder numbers)
- POOLED 1562: ex-top-1 % +0.167, ex-top-5 % −0.102, drop the best 2 days +0.158, capped +0.042.
- The top 3 days = 55.4 R of the 81.0 R sum (68 %). In VAL the top 3 days (55.4 R) are more than the
  whole VAL sum (51.3 R).
- The paired ΔR is tail redistribution: ex-top-5 % is negative on every split except 1563 TRAIN. With
  consistent units it is negative everywhere.

## Other checks
- Fills per week over the span of scored fills: 2.83 / 2.94 pooled; TRAIN 1.8 / 1.9 (only 56 scorable
  signals, all in 2025H1).
- Median R′ as % of price: 3.9–4.0 %. The spread rail passes.
- The walker skips the rest of the tape-fill minute (fill-minute bar low ≤ stop): only 2 rows, a minor
  optimistic bias.

## Cross-programme flag (outside this cell's scope, MUST be triaged)
The same shares defect is in `research/orb_latency_bt/replay.py`: pnl_replay = _sized_pnl +
0.0667 × (entry − fill). The replay fill price therefore never enters P&L: pnl_replay == BT P&L at
every delay (d = 0: 0.0820 vs 0.0820; d = 90: 0.0182 vs 0.0182). Correctly sized (about 330 shares at
the median):

| delay | mean_R as reported | mean_R corrected |
|---|---|---|
| 0 s | 0.082 | 0.102 |
| 30 s | 0.050 | 0.078 |
| 90 s | 0.018 | 0.074 |

Cell 1,426's "zero-latency +0.028 R" and its latency-decay curve are BT-entry-model numbers, not tape
numbers. The "ORB closed as a money book" statement rests on them.
