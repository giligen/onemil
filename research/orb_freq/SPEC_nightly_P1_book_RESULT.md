# Nightly P1 book - RESULT (2026-10-03)

## What changed
- NEW `trading/orb_pool_defs.py`: the one reader of `orb.yaml universe.addon_pools` (parse, bounds, `pool_matches`).
  `trading/orb_engine.py` now calls it (config parse + membership); no second copy of the thresholds.
- `study_orb_broad.load_broad_universe(bounds=, date_start=)`, `orb_backtest._qualifying_pairs_for_dates(bounds=)`:
  optional pool bounds (None = production, byte-identical). Gap upper bound added to the SQL only for pools.
- `study_orb_features.py`: `ORB_UNIVERSE_POOL` mode (side dir only, refuses the production dir); rows add
  `pool_id`, `move_to_range_high_pct` (engine formula). Production CSV schema untouched.
- `study_orb_pipeline_static_lock.py`: `ORB_BT_POOL` mode: pool gate before scoring (NaN fails closed; other gate keys
  abort), slot budget = N minus slots production consumed pre-veto (`ORB_BT_SLOTS_USED_IN/OUT`), `pool_id` column,
  `_write_markers`. Same ranking, skip_q1, dedup, PDR/G1/range vetoes, no refill.
- `orb_backtest.py`: `build_pool_books()` runs after the production pipeline (fetch pool bars, features, pipeline);
  failures are logged ERROR and never abort the production run. `ORB_CACHE_DB` env = side cache.
- Outputs: `analysis_results/orb_bplus_book_P1.csv` (+ `pool_id`), `analysis_results/orb_bplus_book_markers.csv`
  (date,pool_id,candidates,picks; one row per computed day for BOTH books, picks=0 when empty),
  `analysis_results/orb_bplus_slots_used.csv`, features in `analysis_results/pool_P1/`.
  Marker rows live in their own file (a pseudo-row in the book would break the book readers).
- `scripts/eod_sections.py`: `P1 ranked / P1 picks/fills / P1 P&L / P1 clean sessions n` lines; own counter in
  `logs/orb_p1_parity_state.json`; the production promotion metrics are unchanged. No marker and no row = NO-DATA.

## Acceptance (one-day runs, side cache with the pool bars fetched from Alpaca)
10/1: BT P1 ranked (skip_q1 applied, gate applied) = TDAY Q4 .4191, KD Q4 .3553, APPS Q5 .5262, CRMG Q5 .4788,
ADBG Q3 .2784, DXC Q3 .2355, CNXC Q2 .1855. Engine SCORED (gap<5 lines, 10/1 archive) = the same 7 names with the same
comp/quintile to 4 decimals, plus BRZE Q1 .0455 (logged before skip_q1, dropped by it). Match 7/7, no difference.
Slots: production used 5 pre-veto, so P1 budget 3 (TDAY, KD, APPS); BT PDR veto drops all 3 = 0 picks; the engine log
shows PDR VETO on TDAY, KD, APPS at 13:48:43 UTC. Marker row 10/1 P1 candidates 7 picks 0.
10/2: production used 8 of 8 slots, P1 budget 0; the engine logged no P1 SCORED line. BT P1 picks 0 (marker row).

## Runtime
Extra per night, measured pieces: bar fetch ~15 s (113 pairs over two days), features 44 s, pipeline ~56 s: about
2-3 min on top of 35-49 min. Under the 20 min threshold.

## Caveats
- One-day acceptance used a side cache (pool bars are not in cache.db before tonight's first run, so P1 history
  before 10/1 does not exist; the book accumulates from the first nightly run).
- By mistake an uncaged env-less pipeline run regenerated `orb_bplus_book.csv` / `orb_monthly_static_lock.csv` at 10:33
  from the unchanged 10/2 features CSV (research_run.sh does not pass env through sudo); same inputs and rules as the
  nightly, tonight's run overwrites it.
- Tests: `tests/test_orb_pool_book.py` (12) + CLI test stubs updated; tests/test_orb*.py + test_eod_sections +
  test_eod_report: 1328 passed.
