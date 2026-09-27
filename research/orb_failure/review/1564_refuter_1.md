# Refuter 1 — cells 1,564–1,566: causality, obtainability, price scale

Verdict: **NOT REFUTED. All three cells still FAIL the frozen VAL bar.** I found several defects. None of them
changes a verdict: every defect either keeps the cell below +0.15 R or makes it worse. Scratch scripts: r1.py, r2.py and r3.py in
the session scratchpad. cache.db was opened read-only.

## Causality — clean
* `classify_one` decides BREAK/FAILURE/SUCCESS on bars with m in [575, 630) only. The 10:30 bar's h/l/c never
  enter the event, the stop (`high_through_decl`) or the VWAP. The entry is `bars.m == 630` open. If there is no
  10:30 bar, the event is excluded (`no_1030_bar`, 75 FAILURE events); it is never filled at a later bar.
* I reimplemented the classification independently from cache.db. The labels agree **100 %** on all 13,316
  candidates (FAILURE 2,014 / SUCCESS 3,448 / INDETERMINATE 1,907 / NO_BREAK 5,947). The exit reason agrees on
  99.2 % of the 1,564 entered rows.
* Minor: cache.db carries premarket bars (13,159 symbol-days). The SSR test (`bars.m < 630`) includes premarket lows.
  The stop uses the regular-session high only. For FAILURE that equals the session high, because the break guarantees it.
* Quotes are taken at or before 10:30:00 ET (causal).

## Candidate pool — a BT reconstruction, not the live nightly file
`orb_features_20260925_2054.csv` is `study_orb_features.py` over `study_orb_broad.load_broad_universe`. That is
cache.db `daily_bars` with gap ≥ 5 % vs the prior close, prior volume ≥ 500K, open $3–30, and intraday bars cached.
Every field is knowable at 09:30 (the official open print). It is **not** the live scanner's output. It includes
delisted names (151 of 6,736 TRAIN candidates have no daily bar after 2026-06-30), so it is not survivor-only.
`read_orb_csv` only protects the "NA" ticker.

## Obtainability
* **Shortability is hindsight.** `borrow_flags.csv` is a snapshot from 2026-09-18. It excludes 896 FAILURE events
  as `not_shortable`, which is more than the 635 that entered; 52 of those symbols are simply missing from the
  snapshot. Sensitivity with every ≥ $5 FAILURE event included, gross at mid: TRAIN +0.023 R (n 806), VAL −0.112 R
  (n 725, t −2.2). The excluded group is not a hidden winner.
* **The half-spread is biased against the cell.** The builder uses the flat 25 bps fallback on 65 % of entered 1,564
  rows (the quote cache was partial at scoring). Using the now-complete quote cache (96–97 % coverage) and no stop slip
  or borrow, the net is TRAIN −0.072, VAL −0.136 R (t −1.8). That is still far below +0.15, and the gross at mid is
  already VAL −0.102. The headline −0.325 overstates the loss by about 0.15 R, but the verdict is unchanged.
* Stops use gap-through at the open (5 of 180 stop exits). A bar touching both stop and target counts as the stop.
  SSR removes only 4 events. Halts show up as missing bars; the next open is used.

## Price scale — clean
* 0 of 12,856 10:30 opens fall outside the same day's daily_bars [low, high]. Intraday bars are raw (no
  `adjustment` set). The events use same-day bars only, so a split cannot fabricate R.
* Glitch rows (|gap_pct| > 200 or |price_vs_20d_high| > 200, 255 candidates) sit on only 3/635 1,564 events,
  1/66 1,565 and 27/2,893 1,566. Removing them gives 1,564 VAL −0.331 (t −4.3) and 1,566 VAL −0.050, so nothing moves.
  They are likely unadjusted reverse splits in daily_bars that create a fake "gap". Those candidates are spurious
  and should be dropped from the pool going forward.

## Defects that change a number but not a verdict
1. **events/week is inflated**. `cell_stats` divides n by (days that had an event / 5), not by the calendar weeks
   in the split. Honest values: 1,564 VAL 246/38 wk ≈ 6.5/wk (reported 12.2) and TRAIN 389/52 ≈ 7.5 (reported 12.6).
   1,565 VAL 23/38 ≈ **0.6/wk (reported 6.05)**, which also fails the ≥ 3/wk rail.
2. **The 1,565 VWAP is not session VWAP.** It starts at 09:35 (the opening range is excluded) and is weighted by
   close×volume. With a proper 09:30 hlc3 VWAP, 1,565 gross is TRAIN +0.273 (n 46) and VAL +0.184 (n 28). Net with real
   quotes and the stop/EOD standards it is TRAIN +0.047 and VAL **−0.138**, at about 0.7 events/week. Still FAIL. This is
   the only place a positive gross appears: a small, cost-dominated sliver, since median R is about 1.2 % of price.
3. The 1,566 held-break long has a gross at mid of VAL +0.066 (t 1.15) at zero cost, below +0.15 before any cost.

## Bottom line
The failure-short signal is correctly causal, point-in-time on its inputs, and price-scale clean. Its gross at the
10:30 open is ≤ 0 on VAL (−0.10 R at mid), whether or not borrow is included. No causality, obtainability or scale
defect moves any cell to the bar.
