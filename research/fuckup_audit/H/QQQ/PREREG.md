# H/QQQ — pre-registration (written BEFORE any VAL or TEST read)

Written 2026-09-17, after step 1 (TRAIN anatomy) and before the step-2 filter grid is run.
Splits (declared in the stage brief, fixed): TRAIN 2016-01-01..2022-12-31 · VAL 2023-01-01..2024-06-30 ·
TEST 2024-07-01..2026-09-30. Simulator: `Q/zsim.py`, scenario C (live-convention fill = the next bar's open,
0.5 bp/leg, flat at the 15:59 close), VM = 1.0, semi-hourly checks 10:00..15:30. Anchor reproduced:
Q IS 6.48 bps/traded day t 3.32 · Q OOS 4.57 bps t 1.27 (identical to `Q/REPORT.md`).

Unit = the DAY. The sleeve is flat at every close and every input is trailing, so a **day-level skip is exactly
the removal of that day's return** — no re-simulation needed for day filters. Shape filters (entry count,
side restriction) are re-simulated.

## Candidate filters (each with its mechanism, each computable at 09:31 ET)

| id | rule | mechanism | grid (coarse) |
|---|---|---|---|
| F1 | skip the day if the band half-span `band_w = (UB−LB)/open` at the 10:00 check < c | `sigma` is a 14-day trailing mean of intraday excursion. After a quiet stretch the band is narrow relative to today's real range, so the close crosses it on noise and the VWAP/band stop sits immediately under the entry → whipsaw. TRAIN Q1 (<0.67%) is the only quartile negative in BOTH TRAIN halves. | c ∈ {0.5, 0.6, 0.67, 0.7, 0.8} % |
| F2 | skip the day if `|gap| / band_w` < c | The band is anchored `max(open, prev_close)` / `min(...)`. With a small gap the two anchors coincide: a symmetric band around a market with no overnight impulse, nothing to continue. TRAIN Q1 (<0.20) negative in both halves. | c ∈ {0.15, 0.20, 0.25, 0.30} |
| F3 | one entry per day: take the first crossing only; after its stop do not re-enter (neither side) | Re-entries are by construction entries into a market that has already proven it chops across the band. TRAIN: losing days average 1.59 entries vs 1.37 on winners; the worst 30 days average 2.30; corr(day bps, n_cross) = −0.14. | 2 variants: (a) max 1 entry, (b) max 2 entries |
| F4 | side gate on the 20-day trend: long only if `prev_close > MA20`, short only if `prev_close < MA20` | Cooper–Gutierrez–Hameed (2004 JF): breakout/momentum pays in the direction of market state. | MA20 and MA5 variants |
| F5 | entries only at checks ≤ 12:00 (k ≤ 150) | the published "trade the morning" prior; also a direct test of the prompt's candidate. | k ≤ 150 and k ≤ 210 |
| F6 | skip the day if 20-day realised vol (daily, causal) < c | vol-regime gate: no vol, no breakout. | c ∈ {0.70, 1.00} % |
| F7 | skip if the prior 5 sessions' summed absolute return < c | causal proxy for the weekly finding (high-|move| weeks are green 73%, low 41%). | c ∈ bottom tercile, bottom quartile |

Explicitly REFUTED before the grid by step 1 and therefore NOT run as filters (reported, counted):
"skip after an overnight gap larger than the band" (the largest `gap/band` quartile is the BEST, +10.77 bps)
and "stand down after N consecutive losing days" (TRAIN: after a losing traded day +11.00 bps vs −0.04 after a
winner; lag-1 corr −0.13 — the rule would remove the best days).

## Decision rule (pre-committed)

1. A filter may be adopted only if (a) the vetoed bucket has a negative mean in BOTH TRAIN halves
   (H1 = 2016-01..2019-06, H2 = 2019-07..2022-12), (b) the kept book's mean bps/traded day improves on TRAIN
   and in BOTH halves, and (c) the rule is computable at 09:31 ET.
2. At most 3 filters stacked. The cut is taken from the coarse grid above, never refined.
3. The stack is frozen in writing in this file before VAL is read. VAL passes if: mean bps/traded day improves
   vs the unfiltered VAL book, the vetoed days are negative on VAL, and the filtered VAL book is > 0 with
   ≥ 55% of weeks green.
4. TEST is read ONCE, only if VAL passes, and reported whatever it says: month table, tail tests
   (top 1%/5% of days removed, day capped at +1%), permutation p over every cell this stage looked at, and the
   money line at $60K for QQQ 1× and TQQQ 1×.
5. If VAL fails: the result is "the losing days of this sleeve are not separable by causal day features at this
   power", with the minimum detectable effect stated. TEST is not read.

---

# FROZEN STACK (written 2026-09-17 after the TRAIN grid, BEFORE VAL was read)

TRAIN grid result under the decision rule above:

* **F1 survives at c = 0.8 % only.** At 0.5 / 0.6 / 0.67 / 0.7 the vetoed bucket is positive in TRAIN-H2
  (+9.89 / +5.46 / +0.09 / +0.58) and therefore fails rule 1a. At 0.8 the bucket is −1.34 (H1) / −1.85 (H2).
  The book improves monotonically across the whole grid (6.24 → 7.10 → 7.92 → 8.81 → 9.30 → 10.51).
* **F2 survives at c = 0.20 only** (the pre-registered Q1 boundary): bucket −0.20 (H1) / −1.86 (H2).
  0.15 / 0.25 / 0.30 all leave a positive vetoed bucket.
* **F3 (entry cap) is REJECTED** — max 1 entry/day takes TRAIN from 6.24 to 5.15 and is worse in both
  halves; max 2 → 5.94. The step-1 association "losing days have more entries" is reverse causality: a day
  that chops produces the re-entries, the re-entries do not produce the chop.
* **F4 (MA20/MA5 side gate) REJECTED** — MA20 improves TRAIN (8.26) but is worse in H1 (6.21 vs 7.15);
  MA5 is negative in H2 (−1.82).
* **F5 (entries only before 12:00 / 13:00) REJECTED** — worse in both halves (4.72 / 4.40 vs 6.24).
* **F6 (rv20 floor) and F7 (prior-5-day movement floor) REJECTED on rule 1a** — the vetoed bucket is
  POSITIVE in both halves for every cut (F6: +3.59/−0.02 and +2.24/+1.79; F7: +0.61/+0.76 and +1.13/+4.15).
  They raise the TRAIN mean only by dropping days that were themselves profitable but low-variance.

**The frozen stack is F1(band_w ≥ 0.8 %) AND F2(|gap|/band ≥ 0.20).** Two filters, both day-level, both
computable at 09:31 ET. TRAIN: 11.04 bps/traded day (t 3.55) vs 6.24 (t 2.91); H1 15.69 vs 7.15;
H2 8.02 vs 5.40; SR 1.34 vs 1.10; MDD 4.74 % vs 9.92 %; worst month −371 vs −405 bps; ex-top-5 %-days
+0.32 vs −3.79; capped at +1 %/day +4.35 vs +1.01. 585 of 1,044 TRAIN traded days kept.

VAL is now read ONCE with exactly this stack (plus each filter alone, for the per-filter contribution
required by METHOD step 4). No cut will be changed after VAL.
