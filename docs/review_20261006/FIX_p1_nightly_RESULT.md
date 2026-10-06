# FIX: nightly P1 book (bars backfill + failed marker) - 2026-10-06

Specs: `DIAG_RESULT.md` section 2 ("FIX SPEC", "FAILED MARKER SPEC"). Not committed, no service touched,
`data/cache.db` and `analysis_results/orb_bplus_book*.csv` never written (every run = side store + side out dir).

## Code
- `orb_backtest.backfill_features_bars(alpaca, db, features_csv)`: after the pool features build, reads the CSV's
  (symbol, date) pairs (ticker-safe `read_orb_csv`), `_intraday_bars_cached_pairs` -> `fill_intraday_for_pairs` for
  the missing ones only, WARNING for pairs still missing, report-only when no client (`--no-fill`). Runs BEFORE the pipeline.
- `orb_backtest._run_pool_step`: features and pipeline subprocesses; rc != 0 or timeout -> ERROR + `status=failed` marker
  (note = last error line, e.g. the RESIM `0.6364 > 0.02`). `fill_intraday_for_pairs` per-symbol failures now log a WARNING.
- `trading/orb_markers.py` (new, ONE writer): `upsert_marker_rows`, `write_failed_markers`; columns
  `date,pool_id,candidates,picks,status,note`; old rows read as `ok`. `_write_markers` and the no-features direct
  write use it. `ORB_BT_OUT_DIR` env (default `analysis_results/`) = side outputs for rehearsals.
- `scripts/eod_sections`: `bt_marker` returns status/note; a failed marker wins -> `(None, "FAILED: <reason>")` + ERROR;
  P1 lines: `P1 BT: FAILED (<reason>)` first, picks/fills and P&L say FAILED, clean counter neutral.

## Tests (tests/test_orb_pool_book.py +8)
fetch for exactly the missing pairs (incl. ticker NA) then 0 on the second run; no-client report-only; features rc=1,
pipeline rc=1 (RESIM reason in note), pipeline timeout -> failed marker; success adds none; old-format file reads ok and
a later good run replaces a failed row; EOD text has FAILED, no NO-DATA, ERROR logged. 425 passed across the 24 ORB/EOD
test files (incl. all of test_orb_pool_book.py, test_eod_sections.py).

## Acceptance 2026-10-05 (real Alpaca, side store p1_side.db = daily bars only, intraday EMPTY like data/cache.db)
- Confirmed: data/cache.db has 0 intraday rows for MARO 2026-10-02 (read-only).
- Run 1: 27 pairs for 10/5 (9,478 bars, = nightly); features CSV 120 pairs; **99/120 lacked bars (10/1: 19, 10/2: 80) ->
  38,477 bars fetched in 13 s**; pipeline OK in 68 s total; marker `2026-10-05,P1,12,0,ok`.
- Run 2: `all 27 pairs already cached`, `all 120 pairs cached` -> **0 fetched**.
- P1 ranked (BT replica, whole set): GDS BABX DKNG TDAY BBAR STGW OCTV CRCA CONL BMNU PUSA; book = 0 picks
  (PDR veto dropped 3; the replica's `picks` CRCA CONL BMNU use n=999, not the slot budget).
- Outputs: scratchpad `p1_acc/` (accept.log, accept2.log, out/).

## Caveats
Production data/cache.db still lacks the earlier P1 pool-day bars: the first nightly run after deploy does the one-time
~100-pair backfill (~13 s here), then 1 day/night. The service must restart/pick the code up on its next timer run.
