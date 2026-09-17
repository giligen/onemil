# Stage H — bottom-up loser analysis → filters → profit (owner directive 2026-09-17)

Owner: "For each of the near-profitable strategies do a bottom-up analysis on the losers or losing days/weeks and
introduce filters to take it to profit." This document is the ONE method every H sub-stage follows, so that a filter
found on the losers is a rule, not a fit to noise. It is the ORB veto-study method (worst buckets, era-consistent,
mechanism stated) applied per book.

## The books (from stages C/E/G/Q; each gets its own directory H/<book>/)
| book | fill / exit | population file | reference numbers (TRAIN / VAL / TEST net R per trade) |
|---|---|---|---|
| F6 red-to-green, range floor ≥5 | next-open, hold to 15:55, touch stop | `C/pop_c.csv` fam F6 (+ `E/candidates_causal.csv` for the causal-universe twin) | +0.090 (t 2.0) / +0.209 (t 2.8) / +0.096 (t 0.7), dies ex-top-5% |
| F6 with stop −1% | next-open, 2R-close, stop −1% | same | +0.031 / +0.063 / — |
| F14 second break | next-open, hold | `C/pop_c.csv` fam F14 | +0.056 (t 1.3) / +0.050 (t 0.9) / — |
| F8 N=30 opening-range break | next-open, hold | `C/pop_c.csv` | +0.012 / −0.006 / — |
| F11 close-confirmed F6 | next-open, hold | `C/pop_c.csv` | ≈0 / +0.065 (t 2.0) / — |
| S1 gap-fade short (borrowable) | next-open, hold | `G/candidates_short.csv` | +0.026 (t 0.65) / +0.024 / — |
| QQQ noise-band sleeve | close→next-open | `Q/` (zsim.py) | OOS +4.6 bps/day at 0.5 bp/leg, 102% in 5 days |

Costs: contract (c) exactly as `B/score5.py` / `A/acore.py`. Book: `run_book(rows, 12, 4)`. Splits fixed: TRAIN 2025 /
VAL 2026-01..05 / TEST 2026-06..09-11. TEST is read ONCE, for the final filter stack only.

## Step 1 — anatomy of the losers (TRAIN only; VAL untouched until step 4)
On the BOOKED TRAIN trades (12/4) and, alongside, on the whole TRAIN population (so a filter is not a book artefact):
1. **Concentration**: P&L by day and by week; share of loss in the worst 5% / 10% of days; the worst 20 days listed
   with what the market did (SPY/IWM open→close, gap, prior day) and what the book did (n trades, exit mix).
2. **Trade anatomy**: losers vs winners on every causal column in the population file — time band, price band,
   spread/R, r_pct, gap_pct, prev_day_range_pct, range_so_far_pct, dist_open_pct, rv_adv, adv20, consol_bars,
   n_touches, consol_vol_ratio, vwap_dist_pct, cum_dollar_vol, close_confirm, sig-bar shape (close position in range,
   volume vs consolidation), pm_dollar_vol (missing = own bucket — AVAILABILITY AUDIT first, per PLAN §1), news
   presence (D/E news files), asset_class, day-of-week, index state at entry (etf_1min.db), day features
   (`day_features.csv`), the day's signal count so far.
3. **Path anatomy**: for losers, minutes to the stop, MAE/MFE (`mae_pct`, `mfe_r`), whether the stop was a wick (stop
   bar closes back above), whether the loser had +0.5R/+1R on the table; for winners, MAE. This decides whether a
   time stop, a breakeven rule or a confirmation rule is the right SHAPE of filter.
4. **Era consistency inside TRAIN**: split TRAIN into H1 (Jan–Jun 2025) and H2 (Jul–Dec 2025). A bucket is a
   candidate veto only if its mean net R is negative in BOTH halves and its removal raises the book's mean net R in
   both halves. This is the ORB veto rule; it is what kept the ORB vetoes honest.

## Step 2 — candidate filters (each must have a MECHANISM sentence, not just a number)
- At most 3 filters stacked per book. Each filter = a bucket veto (drop trades with feature in the bad range) or a
  shape change (time stop, confirmation) — never a re-tuned parameter of the family itself.
- Every filter value is chosen on TRAIN with a coarse grid (terciles/quartiles or round numbers), never a fine sweep.
- The count of every (feature × cut) looked at is recorded — it is the multiplicity denominator.
- Live-computability: each filter must be computable in the engine at the signal minute (name the source).

## Step 3 — the stack on TRAIN
Report the book before/after each filter: n, trades/week, mean net R, t, weekly R, weeks green, worst week, MDD,
exit mix, and the same ex-top-5% and capped at +3R. Both TRAIN halves shown.

## Step 4 — VAL (read now, once per book)
Apply the frozen stack. Pass = mean net R improves vs the unfiltered VAL book AND the vetoed bucket is negative on VAL
AND VAL mean net R > 0 with ≥ 55% weeks green. Report the per-filter contribution on VAL. No re-tuning after VAL; if it
fails, the book's H result is "the losers are not separable by causal features at this power" with the MDE stated.

## Step 5 — TEST (once, final stack only, after step 4 is written down)
Week-by-week, per-month, tail tests, permutation p over all cells the H stage looked at for that book. Then the money
line: weekly R at 4 slots, $ per week at $100 and $400 risk, worst week, and the honest "what could go wrong" (the
filter's dependence on one period, on one feature's coverage, on the tail).

## Deliverable per book: H/<book>/REPORT.md — step 1 tables, the filters with mechanisms, the stack on TRAIN, VAL,
TEST, the money line, the cell count. Plus the per-trade CSV of the final book (for the independent rebuild) and the
filter rules in prose. Append 3 lines to `research/fuckup_audit/LOG.md`.

## Rules that still apply
PLAN §1 in full (node resources, read-only outside your directory, availability audit, phrasing rule, no TEST before
the stack is frozen in writing). One heavy python at a time per agent, `nice -n 10`, `ulimit -v 1500000`,
`pop_c.csv` via usecols+chunksize, keep_default_na=False.

## Added 2026-09-17 (owner): the live dry-run book
| F5 HOD-break K5/X4 (the live spec, `trading/hod_break.py`) | next-open, 2R-close (live) and hold | `C/pop_c.csv` fam F5 | gross −0.068R at ZERO cost (t −7.9, 25,615 trades); the furthest from profit, but the only book with a parity-audited live engine — a working filter stack here is the shortest path to live. Start from the live spec's exit AND the hold exit; the filters must be computable in `hod_break_engine.py` at the signal minute (it already streams the universe and has news/PM$ hooks available via the ORB helpers). |
