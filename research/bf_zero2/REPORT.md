# bf_zero2 — square one on the honest tape: RESULT (2026-09-16 01:55 UTC)

Owner 9/15: "go back to square 1 now that you have all the data and find the RIGHT params from scratch … 10R+ per week";
9/16: "10R was just a target, data will speak". Pre-registered in DESIGN.md before the scan.

## Data
- 1-min bars: Alpaca SIP only — `research/bf_zero/bars_sip.db` (the +5% causal superset re-fetch, 105K symbol-days, plus
  every universe symbol-day with open ≥ $5 not already in cache.db: 199,157 keys, 66.3M bars, 644 unserved delisted names)
  and `data/cache.db`. Provenance gate before the scan: bars_sip.db 100.0% bar-exact vs a fresh REST call on 50 random
  keys, cache.db 99.7%. The thin EQUS.MINI stores are deleted.
- Universe: `research/bf_zero/universe.csv` — 647,796 point-in-time symbol-days with (high−low)/low ≥ 5%, open ≥ $1,
  ADV20 ≥ 100K. Sub-$5 opens were not fetched (the book's floor is $5 at scoring).
- Pass 1 (`build_candidates2.py`): 420 days, 26 family configurations (F1 flag, F2 micro pullback, F3 opening drive, F4 VWAP
  bounce, F5 HOD break K∈{3,5,8}×X∈{2,4,6%}, F6 red-to-green, F7 pre-market-high break, F8 opening-range break 5/15/30 min,
  F9 gap-and-go, F10 VWAP reclaim), 4.67M candidate rows; pass 2 (`pass2b.py`): relative volume (same-clock own history and
  market profile), wrapper flag, causal sibling cohort → 3.0M rows at price ≥ $5, entry ≤ 14:01.
- Scoring (`score2.py`): the live exit (+2R on a bar CLOSE, stop first, flat 15:55), costs charged per trade (half a
  40-bps spread in, half out on non-target exits; 10 bps stop slip in the fill), the ONE book rule (4 concurrent, 12/day,
  first-come, causal freeing), min R 1% (the live spec), and the CAUSAL FLOOR: the universe's "day range ≥ 5%" is a
  hindsight filter, so F5–F10 entries count only when ≥ 5% above the open (the day is then in the universe whatever
  happens after the entry); F1–F4 imply it by construction.

## The void first run (kept as `score_tables_VOID_no_floor.md`)
Without the causal floor the scorer "found" six candidates at +0.4..+0.9R net, 70% win rate, 14/14 TEST weeks green
(opening-range breaks on quiet names with prev-day range < 5% and range-so-far < 3%). That is the same look-ahead class
as the 9/15 cache population: quiet names that WILL end the day with a ≥ 5% range must move after the entry. The floor
removes 2.1M of the 2.3M F5–F10 rows. Reported so the failure mode is on record.

## RESULT — nothing clears the bar; nothing is even positive beyond noise
TRAIN (2025), 4/12 book, live exit, net of costs, every family configuration:

| family-config | trades/wk | net R/trade | weekly R | weeks green | worst week | WR |
|---|---|---|---|---|---|---|
| F8 ORB 30-min | 21.4 | +0.003 | +0.1 | 47% | −10.0 | 47% |
| F6 red-to-green | 20.8 | +0.007 | +0.1 | 40% | −8.1 | 43% |
| F8 ORB 5-min | 26.2 | −0.012 | −0.3 | 43% | −10.8 | 42% |
| F9 gap-and-go | 17.0 | −0.032 | −0.5 | 53% | −9.9 | 42% |
| F10 VWAP reclaim | 24.0 | −0.022 | −0.5 | 51% | −14.3 | 42% |
| F8 ORB 15-min | 23.4 | −0.029 | −0.7 | 42% | −13.8 | 43% |
| F1 flag P=12% | 29.8 | −0.052 | −1.5 | 47% | −22.5 | 39% |
| F7 pre-market high | 18.2 | −0.139 | −2.5 | 25% | −18.2 | 34% |
| F5 HOD break K=8 X=6% | 36.3 | −0.076 | −2.7 | 36% | −32.5 | 41% |
| … the other 17 (F5 grid, F2, F3, F4, F1 5/8%) | 30–57 | −0.10 … −0.34 | −4 … −19 | ≤ 36% | −24 … −43 | 32–39% |

Two configurations are positive by a hair (+0.003 / +0.007R); their best three-filter versions (275 cells looked at) fail
validation: F6 with prev-range ≥ 10% & ADV < 500K: VAL +1.0R/wk, TEST −0.2; F8-30 with no filter: VAL −1.1, TEST −2.4.
Gross (before costs) population means are +0.05..+0.18R at best (F1 P=12%: +0.18 gross, −0.05 net on the book) — the
spread and the stop slip eat every family. No configuration reaches 10R/week; none reaches 1R/week.

**What this says.** On the whole point-in-time universe of ≥ 5%-range days, on the consolidated tape, net of realistic
costs, no 1-minute breakout or pullback family — bull flag, micro pullback, opening drive, VWAP bounce/reclaim, HOD
break, red-to-green, pre-market high, opening range, gap-and-go — has a tradable edge at 4 concurrent. The 9/13 study's
survivor (F5) and the 9/15 "fixed" book were artifacts of the tape and of the day selection.

**What it cannot say.** The universe is still "days that ended with a ≥ 5% range": strategies whose edge lives on
ORDINARY days of liquid names (the academic opening-range results on large caps, mean-reversion, overnight holds) are not
testable on this store — that needs every symbol-day regardless of range (≈ 1M symbol-days/yr, ~100 GB of 1-min bars,
beyond this node) or a different bar resolution. Sub-$5 names were not fetched. Costs are the spread model, not fills.

## Recommendation
Close the intraday-breakout line of work on this universe. If the owner wants to keep searching, the next honest study
needs (a) a different universe (all symbol-days of a liquid-name list, no range gate) and (b) a different edge family
(overnight/gap statistics, mean reversion, earnings/news events) — pre-registered the same way. The HOD-break engine stays
in dry run only as instrumentation until then; there is nothing to go live with.
