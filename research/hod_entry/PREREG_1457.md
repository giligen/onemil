# PREREG — cells 1,457–1,465: the last round on the resting-order fill book — perfect-foresight ceiling, causal big-day predictors, and the cost side

FROZEN 2026-09-26 05:49 UTC (commit time), before any cell number. Programme count on the HOD line: 1,456 → 1,465. Disclosed as
motivated AFTER cells 1,445–1,456 and the three diagnostics (`review/1445_*.md`); the kill switch below is what makes
this round honest rather than a chase.

## What the diagnostics established (the premises of this round)
1. The profitable cohort of cell 1,438's fills (the 1,928 symbol-days whose bars came from cache.db) is the set some
   backtest builder had selected; 83 % of it is covered by rules that condition on the FULL DAY's range (look-ahead), the
   two causal rules inside it are negative (ORB gap rule −0.42 R) or flat (wrappers +0.04 R), and the uncovered residual
   is worth +0.007 R. The cohort is "the day ended big", not a property visible at the arm bar.
2. Under the corrected cost standard (entry half-spread charged once; measured stop slip), that cohort is +0.23 R on
   TRAIN-H2 and +0.06 R on VAL; gross it is +0.39 R vs −0.07 R for the rest. The gross gap is real; the cost eats it.
3. The cost is structural: median R is 1.6 % of price (p25 1.23 %, p75 2.25 %), so 35 bps of measured stop slip is
   0.21 R on each of the 57 % of fills that stop. Levels are reliable (day high captured within 0.5 % on 99.9 %).

## Base book, cost standard
The 9,911 fills of cell 1,438 (`causal_arming_causal.csv`, status = fill), TRAIN-H2 / VAL, TEST sealed. Cost = the
1,445 standard (`cell_1445.py`: half_entry recovered and charged once; per-trade measured stop slip from cell 1,443's
cache, holdout-mean fallback) with ONE fix: unmeasured EOD exits fall back to the EOD-specific mean (≈ 11 bps), not the
stop mean (35 bps) — `review/1445_cost_reconciliation.md` sized the defect at +0.008 R book-wide. Report the flat 30 bps
variant beside every number.

## Cells
| cell | condition / mechanism | source | role |
|---|---|---|---|
| 1,457 CEILING (look-ahead, report-only) | keep fills whose FULL-DAY range (high − low)/low ≥ 10 % on the PIT daily bar | Databento parquet | the best any predictor of a big day could do. **Kill switch: if 1,457's VAL kept mean net R < +0.15 under the corrected cost, the resting-order book is closed at every filter on this population and cells 1,458–1,462 are reported but cannot pass.** |
| 1,458 | pre-market dollar volume (04:00–09:29 ET, Σ v × c) ≥ $500K | cache.db bars for the 1,928 cache-only days, bars_sip.db otherwise (both hold pre-market bars; UTC timestamps) | causal big-day predictor |
| 1,459 | ATR14 as % of the prior close ≥ 4 % (true range over the 14 prior sessions) | Databento parquet | causal |
| 1,460 | prior session's range (high − low)/low ≥ 5 % | Databento parquet | causal |
| 1,461 | news catalyst: `data/research/orb_news_catalyst_nightly.csv` row with n_articles ≥ 1 on (symbol, day); NaN for any symbol-day outside the universe that file's generator scanned (find the generator under scripts/, state its universe); VOID if computable coverage < 80 % | nightly CSV | causal |
| 1,462 | quoted spread at the fill ≤ 10 bps ∧ 1,458 (the strongest separator of 1,445 joined with the strongest new predictor on TRAIN-H2 — declared now, not chosen later) | as 1,454 + 1,458 | causal joint |
| 1,463 COST | stop-LIMIT exit: on cell 1,443's tape windows (`sip_cache_stopslip/`), re-execute every measured stop as a stop-limit with limit = stop × (1 − 20 bps): filled at the bid at t0 + 250 ms if bid ≥ limit, else at the first print after the limit is breached that is ≥ limit within the minute, else at the minute's last print (the no-fill tail); variant 50 bps. Report mean/median/p90 slip and the book's net R under each | 1,443 cache | execution mechanism; ship bar: mean slip lower by ≥ 10 bps AND the no-fill tail's mean slip ≤ 100 bps on both holdouts |
| 1,464 COST | R floor at 2.5 % of price: stop = min(consolidation low, fill × 0.975), paired re-walk with the 1,440 machinery (`cell_1440.py`), target 2 R from the new R, measured slip re-scaled in R | bars, 1,440 code | ΔR ≥ +0.05 both holdouts, VAL t ≥ 2.5 |
| 1,465 JOINT | the best of 1,458–1,461 on TRAIN-H2 ∧ 1,463's better variant, on the fills | above | one VAL read |

## Pass bar (as PREREG_1445, unchanged)
VAL kept mean net R ≥ +0.15 (corrected cost), day-clustered t ≥ 2.5, ex-top-5 % > 0, ≥ 3 kept fills/week at 12/4,
TRAIN-H2 same sign with t ≥ 1, dropped mean < kept mean on both holdouts. Report n kept/dropped, both means, ΔR, t,
ex-top-5 %, fills/wk, winner-capped (+3 R), count-matched null percentile, flat-30 variant. TEST read once for the single
best passing cell, only if the ceiling cell allows a pass.

## Consequences (pre-committed)
PASS on any of 1,458–1,462 → dry run with the predictor logged per arm, 5 sessions, then $50 real orders under the 9/25
fixes. 1,463 ship bar met → the live StopMonitor exit becomes a stop-limit with that offset (engineering item, its own
rehearsal), regardless of the entry verdict, because it applies to every book. FAIL everywhere, or the kill switch →
the resting-order HOD-break book is closed at every filter and every exit on this population; the owner report states
the ceiling number as the reason and no further cell is opened on it.

## Independent check
Rebuild of every flag from this prose by an agent that has not read the builder's code; ≥ 99 % agreement, kept means
within 0.02 R; refuters on any passing cell; adequacy critic otherwise (including: does the ceiling reproduce the
builders' +0.41/+0.30 MACD-rule cohort?).

## Not allowed
Moving a threshold; choosing 1,465's components on VAL; recomputing outcomes except in 1,464's paired re-walk; reading
TEST for more than one cell; any feature using data after bar j except the declared ceiling.
