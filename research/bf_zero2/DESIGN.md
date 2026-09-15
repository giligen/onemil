# bf_zero2 — square one on the HONEST tape (pre-registered 2026-09-16 00:05 UTC, before any scan)

Owner 9/15 23:50: "go back to square 1 now that you have all the data and find the RIGHT params from scratch that will
take it to 10R+ per week on average". Every number of research/bf_zero is void: 45% of its symbol-days were on a
one-publisher tape (EQUS.MINI) and the rest were the bull-flag scanner's hindsight mover days (REPORT §6b).

## Data (the gate before any number)
- Universe: research/bf_zero/universe.csv — Databento EQUS.SUMMARY point-in-time daily (delisted included),
  647,796 symbol-days, open ≥ $1, day range ≥ 5%, ADV20 ≥ 100K. Live book floor $5 (cost rule) applied at scoring,
  never at the scan. Leveraged wrappers IN (owner 9/14: in for the dry run; the data decides).
- 1-min bars: consolidated SIP ONLY — research/bf_zero/bars_sip.db (Alpaca REST, 04:00–20:00 ET), plus data/cache.db
  intraday_bars_1min (Alpaca SIP, 99.7% bar-exact). The thin stores (bars.db, pit_bars_1min.db, topup.db) are NEVER read.
  Every symbol-day not yet in bars_sip.db is fetched before the scan; symbol-days Alpaca cannot serve (delisted) are
  counted and reported as the survivorship residual, never silently dropped.
- Provenance gate: research/bf_zero/parity_review/tape_provenance_check.py on 50 random keys per store must report
  ≥ 99% bar-exact vs a fresh REST call before pass 1 starts; the run log carries the number.

## Splits (fixed): TRAIN 2025-01-02..2025-12-31 · VAL 2026-01-01..2026-05-31 · TEST 2026-06-01..2026-09-11. TEST read once.
Early-close days excluded on both sides.

## Search space (all of it, no early exits)
- Entry families (from bf_zero, all re-run): F1 flag, F2 micro pullback, F3 opening-drive pullback, F4 VWAP bounce,
  F5 HOD-consolidation break (K ∈ {3,5,8} × X ∈ {0.02,0.04,0.06}), F6 red-to-green, F7 pre-market-high break,
  F8 opening-range break (5/15/30). Plus F9 gap-and-go (first 5-min high break on a gap ≥ 5%) and F10 VWAP reclaim.
- Exits: E1 fixed +2R/−1R (target on close), E2 +3R, E3 partial 50% at +2R + breakeven trail, E4 hold to 15:55.
- Causal filters (values known at the SIGNAL minute only): rv_profile bands, distance from open, gap %, pre-market $
  volume, news presence (Alpaca news, prev 15:00 → signal), range-so-far %, price band, time-of-day band, wrapper flag,
  ADV band, spread proxy (price band + bar count so far). Each as a hypothesis split; a filter is adopted only if the
  excluded bucket is worse on TRAIN AND VAL and TEST agrees in sign.
- Fill model: next bar's open ≤ level × (1 + cap 0.6%), else no trade; stop fills min(stop, open) × 0.999; target on a
  bar CLOSE; costs: half spread in, half out on non-target exits (the §8 model, spread by price band from
  research/bf_zero/spread_study_clean.csv), 10 bps stop slip.
- Book: trading.hod_break.run_book (causal freeing, symbol tie-break), 4 concurrent, 12/day, last entry 14:00.

## Selection rule (pre-committed)
1. TRAIN: net mean R > 0, ≥ 5 trades/week, weekly net R ≥ +10 on the 4-concurrent book.
2. VAL: weekly net R ≥ +7, green weeks ≥ 60%, mean R net > 0.
3. TEST: read once, week by week; reported whatever it says.
4. Multiple comparisons: the number of (family, config, exit, filter) cells looked at is counted and reported; the
   VAL bar is raised by 1 SE of weekly R per 10 cells that passed TRAIN.
5. If nothing passes: the closest miss per family, and the book is closed until new data.

## Deliverables
research/bf_zero2/REPORT.md in the bf_zero format; scan scripts under research/bf_zero2/ reuse bf_zero's pass-1/pass-2
code with the SIP store only (BFZ_SIP_STORE) and the whole-universe seed; every table, failures included.
