# Pre-registration — F2 + A1 (8 cells), written 2026-09-18 BEFORE any return was computed

Panel: `research/multiday/data/` (DATA.md). Splits TRAIN 2016-01→2021-12 / VAL 2022-01→2023-12 /
TEST 2024-01→2026-09 **SEALED** (opened once, only behind `FREEZE.md` naming G2 survivors).

## Universe (both families)
`kind=='common'` ∩ has ≥1 Item-2.02 8-K ∩ RAW close ≥ $5 at the decision close ∩ ADV20$ ≥ $1M
(20-session mean of `vwap × volume` on the ADJUSTED panel, ending at the decision close).
**Deviation declared:** the lit review's `$10M` floor is reported as a robustness ARM, not the primary
floor — a $66K book takes ≤ $3.3K positions, which is 0.33% of a $1M ADV, inside the impact cap. Both
are reported; only the $1M arm is a scored cell.
Features on the ADJUSTED panel; the $5 gate and share counts on RAW. SPY = market benchmark.

## Execution convention (binding; AMENDMENT 2(a), Goyal–Jegadeesh–Wu JFQA 2026)
**Close-to-close, `cls` TIF, both legs.** Opening auctions are illiquid → no MOO anywhere. No quoted
spread on an auction cross. Costs: SEC+TAF **0.4 bps on sells**; impact = **10 bps × (order$ / 1% of
ADV20$)**, both sides, order$ capped at 1% of ADV; borrow **0.3%/yr** on any short leg, gated on
today's `easy_to_borrow` (disclosed as a look-ahead, §7 gap 5). Secondary arm: **5 bps/side flat**.
"Entry at the opening auction" is NOT tested → flagged refuted-by-literature, not a hypothesis here.

## F2 — earnings-announcement-return drift (Chan–Jegadeesh–Lakonishok 1996). 6 cells.
- Event session `S` = first session whose closing auction is strictly after `acceptanceDateTime` (already
  a column; re-verified in the availability audit).
- Signal, known at the close of S: **2-day market-adjusted announcement return**
  `r2 = close_adj[S]/close_adj[S-2] − 1 − (SPY[S]/SPY[S-2] − 1)`.
- Decile: causal trailing-window percentile of `r2` against every event with `event_session < S` in the
  trailing **250 calendar days**. No in-sample or forward ranking.
- **Entry close(S+1), exit close(S+1+H)**, H ∈ {20, 40, 60}. The one-session skip is what makes the fill
  obtainable: a `cls` order must rest before the close it fills, so a signal that needs close(S) cannot
  be executed at close(S). The no-skip variant (entry close(S)) is reported as a NOT-OBTAINABLE robustness.
- Cells: `F2-LS-{20,40,60}` = D10 − D1 overlapping equal-weighted daily portfolios;
  `F2-LO-{20,40,60}` = D10 long-only.

## A1 — earnings-announcement premium (Johnson–So 2018 JAR). 2 cells.
- **Expected** announcement session `Ê` (the actual 8-K date is a look-ahead — the filing IS the event):
  `Ê = snap_to_session(event_session of the same fiscal quarter one year earlier + 364 days)`, built only
  from events with `event_session < Ê − 5` sessions. Diagnostic reported: distribution of `S − Ê`.
- **A1-a**: buy close(Ê−5), sell close(Ê−1)  — pre-announcement run-up only, zero announcement risk.
- **A1-b**: buy close(Ê−5), sell close(Ê)    — "sell AT it"; the expected event is after that close for
  the 56% post-close cohort.
- Robustness (not a cell): sell close(Ê+1) — holds through the event, which Johnson–So predicts gives the
  premium back.
- Long-only is the PUBLISHED form; the "long-leg share" column is therefore 100% by construction and is
  reported as such, with a synthetic L−S (announcers vs non-announcers, matched by day) for comparability.

## Book model (the executable column, both families)
$66K, **20 concurrent slots**, $3.3K/position, entries ranked by signal on the entry day, first-come, no
refill of a skipped name. `trades/week = 5 × 20 / H` when candidates are not binding; the realised number
is reported. The ACADEMIC portfolio (every qualifying event, equal-weighted, overlapping) is the return
estimate; the BOOK is the money estimate. Both reported per cell.

## Gates
G1 TRAIN: t ≥ 2 on the monthly series. G2 VAL: same sign AND ≥ 55% of months positive. TEST once, only
for G2 survivors named in `FREEZE.md`. Every cell also reports: long-leg share of the L−S spread,
break-even cost (bps round trip, and × the honest auction cost), ex-January, trades/week, additivity vs
ORB (resource overlap + return correlation on overlapping months from
`research/fuckup_audit/D1_orb/book_n8_q1on.csv`), tail tests (ex-top-5%, winners capped at 3× the cell's
median win), permutation p across the 8 cells, and the PIT Nasdaq re-run for any survivor.
Cumulative multi-day cell count: K 20 + N2 4 + R_daily 20 + **8 here = 52**.

## Calibration prior (pre-committed)
Chen & Velikov: 204 published anomalies net **~4 bps/month**, strongest ~10 bps before impact. On $66K
that is **$25–65/month**. Anything materially larger here is a **leakage suspect first, a discovery second**.

## Phrasing rule
No cell may be reported as "no edge exists". Only: "no edge detectable in THIS universe, at THIS horizon,
at THIS book size, over THIS window, at THIS cost", with the MDE stated alongside.
