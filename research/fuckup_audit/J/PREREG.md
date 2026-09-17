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

---

# ADDENDUM 1-3 — pre-registration amendment, written 2026-09-17 14:40 UTC, BEFORE the scan

Three corrections carried over from the F6-PDR reconciliation (`H/F6_reconcile/REPORT.md`). Written
before any Stage-J number existed: the 19 days built between 14:20 and 14:27 UTC under the un-amended
spec were DELETED, `members_u3.parquet` was rebuilt, and the scan restarted from day 1.

## A1 — universe hygiene: test tickers and names the live engines cannot see

- **Dropped outright from the universe**, never built: every symbol matching `^Z[A-Z]ZZT$` — the
  NASDAQ/NYSE test tickers. In U3 that is **ZAZZT, ZJZZT, ZVZZT = 116 symbol-days of 1,512,031**.
  They print synthetic tapes (ZVZZT 2026-08-18 ran $24 -> $126) and were the entire TEST "profit" of
  the F6 book in the reconciliation. They are not tradeable and nothing may score them.
- **Tagged `univ_flag = 'no_daily_bars'` and excluded from every scored cell**: the **309 symbols /
  29,775 symbol-days (1.97% of U3)** with ZERO rows in `data/cache.db::daily_bars`, the table the live
  engines seed their universe from — a name absent from it can never be traded live. They are TAGGED
  rather than deleted because 309 of 312 are real securities (delisted/acquired names — ANSS, AZPN,
  AMED, ACCD — and preferred classes — ALB-A, BAC-L, C-N), so removing them is itself a **survivorship
  filter in the opposite direction**. The primary book excludes them; the report carries the
  with/without number so the size of that filter is visible rather than assumed.
- `J/universe_exclusions.csv` lists all 312 symbols with their reason. Counts are re-reported in
  `J/REPORT.md`.

## A2 — every family's LEVEL BUFFER is stated, not inherited

`build_candidates.fam_r2g` and `fam_level` take their buffer from the module global `SLIP`, which is the
env var `BFZ_SLIP`, which **every pass-1 builder in this tree pins to 0** (`B/build_candidates4.py:99`).
Stage B/E/G therefore scanned F6 at `prior_close x 1.000` while describing it as the engine's rule. The
declared engine level is `prior_close x 1.003` (`trading/red_to_green.level_for`, `level_buffer=0.003`).

| family | level | buffer |
|---|---|---|
| **F6** | prior close | **x 1.003** (was x1.000 — corrected here, `fam_r2g_buf`) |
| **F11 base=F6** | the same buffered level; signal = first bar CLOSING at/above it | **x 1.003** |
| F8 N=5/15/30 | max(high of the first N bars) | x 1.000 — the opening-range high itself; ORB's 30 bps is a stop-limit FILL offset, expressed here by the 0.6% fill cap, and adding it to the level would double-count |
| F9 G=0.03 | max(high of the first 5 bars) | x 1.000 |
| F5 K=5,X=0.04 | the running high of day at the consolidation | x 1.000 |
| F10 | the prior bar's high | x 1.000 |
| F14 N=15 | the F8-15 level | x 1.000 |
| S1 / S2 N=15/30 | min(low of the first 5 / N bars) | x 1.000 |
| **S3** | prior close | **x 0.997** — already explicit in `G/.../sfam_g2r`, the exact mirror of F6's 1.003. The long and short sides of this tree were INCONSISTENT before this addendum |
| S5 | none — a scheduled 09:35 market order | n/a |

The override is a mirror, not a rewrite: with `buf = 0` `fam_r2g_buf` must reproduce
`build_candidates.fam_r2g` exactly, asserted at import (`_selftest_f6`).

## A3 — BOTH scan rules, declared as twin cells and named on every table

Two rules exist for level-break families and they give opposite signs on small caps:

- **`first`** — the first bar reaching the level IS the signal; if its fill fails the 0.6% cap (or its
  stop is not below the fill) the day is dead. This is what every study in this tree did.
- **`keep`** — keep scanning: later breaks of the SAME level are taken until one produces an acceptable
  fill (`trading/red_to_green.detect`'s `continue`). The stop is recomputed at the later bar by the
  family's own rule (running low through that bar for F6/F11, lowest low from the stop bar for F14, the
  structural opening-range low for F8). **This is the engine-convention rule — the one that could ever
  go live.**

Emitted for **F6, F8 N=5/15/30, F14, F11** (the long level-break families the correction names; the
shorts keep `first` and it is stated on their tables). A `keep` row is written only when it differs from
`first`, so `population(first) = rows with scan == 'first'` and `population(keep) = all rows`. Every
table names the rule.

## Cell count after the addendum

14 family-configs x 3 exits x 2 R floors = 84 · + the PDR>=8 twin per family at the primary floor
(14 x 3) = 42 · + the `keep` scan twin for the 6 twin family-configs at the primary floor (6 x 3) = 18
· **= 144 declared**, + the pooled book (3) only if a family clears G1. TEST read once per G2 survivor.
