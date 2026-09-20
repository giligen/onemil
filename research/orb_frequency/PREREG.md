# ORB LONG — frequency pass (cells 1,274–1,277). PRE-REGISTRATION

Written and committed BEFORE any scoring. Author: research agent, 2026-09-20.

## 1. Question
The honest ORB long book's selection edge is real (+0.15 R TRAIN / +0.44 R VAL over a
matched control, frames13 §2.2). FREQUENCY is the binding constraint on the owner's
$10K/mo goal: ~1–2 fills/week, and a large share of ranked picks never fill.
This pass raises FILL FREQUENCY WITHOUT TOUCHING SELECTION. The composite, the
z-params, the quintile cutoffs, the vetoes, the sizing and the exit spec are FROZEN.

## 2. Baseline (the honest book)
`analysis_results/orb_bplus_book.csv` — the entered-inclusive book produced by
`study_orb_pipeline_static_lock.py`, 8 shared slots, $10K stage sizing, static lock
1.75R→0.5R, touchgo M/D, PDR / range-size / G1-short-history / catalyst vetoes, no refill.
Every increment below is measured as the DELTA on this book with the SAME 8 slots,
the SAME sizing rule and the SAME vetoes. No refill anywhere.

## 3. Mid-orientation finding, declared before scoring (matters for F1a)
Reading `study_orb_features.py:636` → `study_orb.simulate_orb_trade(entry_mode='touch')`:
the BT fills a candidate whenever ANY bar's high exceeds `range_high` inside the 60-min
window, at `range_high x 1.003`, **irrespective of where that bar opened**. There is no
stop-limit cap in the backtest. Consequences, both pre-registered here:
 (a) every `entered=0` / `no_fill` row in the book is, by construction, category (i)
     "range_high never broken within 60 min". Categories (ii)/(iii) do not exist in the
     book's own model.
 (b) the book's FILLED rows include picks whose breakout bar OPENED ABOVE the 30-bps cap.
     Live those are NOT fills at the cap — the live stop-limit at `range_high x 1.003`
     would fill at the opening print (above the limit → no fill) or not at all. The book
     is therefore OPTIMISTIC on those rows.
Therefore F1a is scored against an **honest-30bps baseline** (B30), defined below, not
against the shipped book, and the shipped-book-vs-B30 gap is reported as its own number.

## 4. Definitions
- Breakout bar = the first 1-min bar in [09:35, 10:35) whose high > range_high.
- Cap-30 = `range_high x 1.0030`; Cap-60 = `range_high x 1.0060`.
- Obtainable-fill rule (the engine's convention): a resting BUY fills at the bar's OPEN
  when the open is at/below the cap; a bar that OPENS ABOVE the cap and never trades back
  down to it inside that bar is NOT a fill at the cap. A touch of a level is never a fill.
- **B30 (honest 30-bps baseline)**: the shipped book with every FILLED pick whose breakout
  bar opened above Cap-30 re-classified per the obtainable rule — fill at
  `min(Cap-30, bar_open)` if `bar_low <= Cap-30`, else NO FILL ($0, slot consumed).
- No-fill split, reported first: (i) range_high never broken in the window;
  (ii) broken but the breakout bar opened above Cap-30 (gap-through);
  (iii) broken, opened at/below Cap-30 — a legitimate fill.

## 5. Cells
- **F1a** (cell 1,274): on B30, for a (ii) pick widen the cap to Cap-60 — fill at that
  bar's OPEN if open <= Cap-60, else still no fill. Increment = F1a book − B30 book.
- **F1b** (cell 1,275): on B30, for a (ii) pick, a RE-ARM: a resting limit BUY at
  `range_high` (no premium, passive, zero entry half-spread), live until 10:35, fills on
  a LATER bar (strictly after the gap-through bar) whose LOW <= range_high, at
  `min(range_high, that bar's open)`. Exits then follow the frozen spec from that bar.
  **DECIDING TABLE (the F52 adverse-selection rule):** for the (ii) cohort, compare the
  re-armed fills' realized R against the counterfactual R those same picks would have
  earned at the F1a Cap-60 fill. If re-armed < gap-through-at-Cap-60, the passive limit is
  selecting the broken setups → F1b is DEAD regardless of its own sign.
- **F2-10 / F2-15** (cells 1,276–1,277): a SECOND opening range on the same gap-up
  population, window [09:30, 09:40) resp. [09:30, 09:45). Composite params, z-params,
  quintile cutoffs and vetoes FROZEN at the 5-min-fit values. ONLY the range-derived
  features are recomputed on the new window; they are, exhaustively:
  `range_size_pct, range_total_volume, range_avg_bar_range_pct, range_volume_stddev_pct,
   bars_green_in_range, range_close_position, range_return_pct, last_bar_green,
   range_vwap_distance_pct`. Every non-range feature (gap, prev-day, 20-day, SPY, time)
  is taken UNCHANGED from the 5-min features row for that symbol-day. Entry =
  `range_high_N x 1.003` from 09:40 / 09:45, 60-min expiry, same obtainable-fill rule as
  B30. A symbol-day already taken by the 5-min book is EXCLUDED. Slots shared: 8 total
  across both books, filled in timestamp order (5-min book gets 09:35 priority by clock).

## 6. Splits
TRAIN = 2025-01-01..2025-12-31. VAL = 2026-01-01..2026-05-31. TEST >= 2026-06-01 — SEALED,
not read, not printed, in this pass.

## 7. PASS BAR (pre-committed, per cell, ALL must hold)
1. added trades net >= +0.10 R/trade on TRAIN **and** VAL;
2. day-clustered t >= 2 on VAL for the added trades;
3. ex-top-5% (drop the top 5% of added trades by R) net >= 0 on both splits;
4. TRAIN first half vs second half same-signed;
5. stacked book $ up in BOTH splits;
6. stacked MDD not worse than 1.25x the baseline book's MDD in that split;
7. fills/week up >= 30% vs the baseline.
Anything short of all seven = NO SHIP. The MDE (smallest R/trade the cell's n could have
detected at t=2, day-clustered) is printed beside every null.

## 8. Costs
Measured per-trade NBBO half-spread where the frames16/17 NBBO table covers the
symbol-day; otherwise the book's own slippage convention (30 bps entry / 10 bps exit) is
carried unchanged so the DELTA is cost-neutral by construction. Passive re-arm fills
(F1b) are charged ZERO entry half-spread (they are the resting side) and the normal exit
slippage. Any cost substitution is stated in the report.

## 9. Multiplicity
This pass = 4 cells. Program cumulative count through this pass: 1,277.

## 10. What would make me drop a cell mid-run
Any cell whose added trades cannot be built from data available at or before the
decision bar; any fill that is not inside the filling bar's [low, high]; any composite
recomputation that touches a non-range feature. These are reported, not silently fixed.
