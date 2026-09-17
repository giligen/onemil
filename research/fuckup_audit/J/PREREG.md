# Stage J — the liquid universe (U3), pre-registered 2026-09-17 20:10 UTC, before any bar of it was scanned

## Why
Every book so far was scored on small-cap movers (≥5%-range days, gap/PDR universes) where the honest edge lives in
$5–10 names that carry ~$226 of risk per trade and the spread costs 0.2–0.5R. Stage I: the liquidity cap turns the one
positive book into ~$560/month. Stage E's cost curve: names with > $50M/day quote 16–25 bps — cost ≈ 0.05R on a 2% stop,
and $1,000 of risk is ~0.1% of a 5-minute tape. The liquid universe (U3: 20-day median dollar volume ≥ $5M, open ≥ $5,
1,512,031 symbol-days, 5,571 names, 2025-01-17..2026-09-04) was never scanned. It is the last large untested area with
CAPACITY, and it is also where the short side becomes tradable (borrow is easy).

## Data
Three stores read as ONE tape: `U3/bars_u3/` (parquet, day=…), `E/bars_causal/` (parquet), `research/bf_zero/bars_sip.db`
(same Alpaca SIP source; provenance-checked). Membership is the liquidity rule, known at 09:30 from the daily panel —
no range floor, no gap gate (nothing to guarantee causally; the universe has no end-of-day information in it).
Daily context per (symbol, day) from the Databento panel: prior close, prior-day high/low/range, gap.

## Families (the same definitions as candidates4 / candidates_short, imported not re-written)
Long: F6 red-to-green, F8 N=5/15/30, F14 second break, F11(F6) close-confirmation, F5 K5/X4 (reference), F9 gap-and-go
(G=0.03 here — liquid names gap less), F10 VWAP reclaim. Short: S1 gap-fade (gap ≥ +3%), S2 N=15/30 breakdown, S3
green-to-red, S5 attention fade (the M18 spec; liquid names only). Fill = next bar's open under the 0.6% cap (the engine's
convention; the reconciliation's engine-convention rules apply: signal from 09:31, 14:00 cut on the signal minute, stop =
running low through the signal bar, next bar = the next bar the tape prints). Exits: hold to 15:55 (primary), 2R on a bar
close, partial 50% at +2R + breakeven. R ≥ 0.5% of price here (liquid names move less; declared — the 1% floor would
delete most of the population; report the 1% twin).

## Costs
Contract (c) with the spread from the cost curve's LIQUIDITY bands ("$5–50M/d" and "> $50M/d" rows by hour; the price-band
table is the small-cap population and does not apply) — declared; plus a per-trade NBBO pull for the booked trades of any
cell that clears G1 (Alpaca quotes API, the signal minute), which then REPLACES the band number for that cell.
Borrow for shorts: 0 for names in this universe (ETB assumed — declared; the sensitivity row charges 5 bps locate).

## Book, splits, gates
`run_book(rows, 12, 4)` per family (declared) and, separately, the 4-family pooled book ONLY if a family clears G1 (Stage I:
stacking hurt on small caps). Splits TRAIN 2025-01-17..12-31 / VAL 2026-01..05 / TEST 2026-06..09-04. PLAN §1 gates (G1
TRAIN mean net > 0, t ≥ 2, ≥ 5/wk; G2 VAL mean > 0, t ≥ 1, ≥ 55% weeks green; TEST once). Standing rules: availability
audit of every partial-coverage column; tail tests (ex-1%/5%, +3R cap); permutation p over all cells; the reversed-tape
twin for any model; the phrasing rule.

## Cells
14 families × 3 exits × {R ≥ 0.5%, R ≥ 1%} = 84 declared, + PDR ≥ 8 twin per family (14 × 3) = 42 (the one rule that
replicated on small caps — tested here as a declared twin, not re-derived), + the pooled book (3) if triggered. TEST read
once per G2 survivor. Time-band and price-band breakdowns are DESCRIPTIVE (reported, not selected on).

## Capacity, reported per cell (not assumed)
Participation at $300 / $1,000 / $2,000 risk vs the trailing 5-minute dollar volume; the liquidity-capped twin at $1,000
(drop > 1%, re-book so slots refill) with $/month. A cell that is positive only below $300 of capacity is reported as
"small-cap shaped" and not carried.

## Deliverable
`J/REPORT.md` (one-page summary first: does anything clear G1/G2 on liquid names at real capacity? long vs short? which
exit? the $/month at the capped $1,000 risk with the worst month), the per-trade CSVs of G2 survivors, the cell count,
the smallest visible effect per headline cell, 3 lines in `LOG.md`. Then, for a survivor: the independent rebuild from
prose (H/F6_rebuild's method) BEFORE any engine work.
