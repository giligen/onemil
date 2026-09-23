# SPEC — ORB production book on 2024H2 (PREREG.md). For a Sonnet implementer, ≤ 40 tool calls

Read `research/orb_2024/PREREG.md` (frozen). Grep/offset reads only; never a file > 300 lines in full. Write only
under `research/orb_2024/`. Do not commit; do not touch `trading/`, `orb.yaml`, `config.yaml`, crontab, `data/cache.db`,
or any existing research cache. No Alpaca fetch between 13:25 and 20:05 UTC.

## Pattern to follow
`research/orb_seed_wide/build_wide_features.py` shows how to run `study_orb_features.main()` against a different
universe by monkeypatching its loaders and setting `ORB_FEATURES_OUT_DIR`. Do the same in
`research/orb_2024/build_features_2024.py`, patching FOUR seams in `study_orb_features` (see `main()` ~line 547):
1. `load_broad_universe` → {date: [symbols]} from `research/day_breadth/y2024/candidates.csv` filtered to the cell's
   gap/price band, with the 09:30–09:35 RTH volume ≥ 15,000 floor computed from `y2024/bars.db`.
2. `load_daily_bars_frame` → the SAME columns/dtypes/sort as the original (read its code, ~line 174): EQUS 2024H2
   daily (`data/research/databento/equs_daily_2024H2.parquet`, symbols mapped '+'→'.WS') UNION `data/cache.db::daily_bars`
   rows for 2024-06-03 .. 2024-06-30 (the 20-day lookbacks of early July), plus SPY daily.
3. `load_spy_intraday` → the SAME columns/format as the original (~line 215) for 2024-06 .. 2024-12, from
   `research/index_orb/cache/SPY_1min.parquet` (tz-aware America/New_York `timestamp`) converted to the original's
   timestamp convention — verify on one day that the SPY 09:30 bar is matched (the loader's own DST note).
4. `Database.get_intraday_bars_bulk` → the same return structure (read it in `persistence/database.py`) served from
   `research/day_breadth/y2024/bars.db` (table bars(symbol, day, t ISO-UTC, o, h, l, c, v)).
Run with `--force-full-regen --start-date 2024-07-02`, one run per cell band, outputs in `research/orb_2024/out_1415/`
and `out_1416/`.

Then run the pipeline per cell exactly like `research/orb_seed_wide/run_quarter_pools.sh`:
`ORB_BT_FEATURES_CSV=<features csv> ORB_BT_BOOK_OUT=research/orb_2024/book_<cell>.csv ORB_CATALYST_VETO=0
python3 study_orb_pipeline_static_lock.py` (grep the pipeline for any date-range constants — e.g. DATE_START/END —
and override them by env if they exclude 2024; report what you changed).

Score (`research/orb_2024/score.py`): entered rows, R = `_sized_pnl` / 375, per cell n fills, fills/week, net R per
fill, day-clustered t, total $, ex-top-5 %, monthly $, no-fill share; the same for `research/orb_seed_wide/out/runB_true.csv`
2025 rows beside cell 1,415. Apply the PREREG verdict (SURVIVES / neither / RED FLAG). Write `research/orb_2024/REPORT.md`.

Reply ≤ 150 words: per cell n fills, net R/fill, t, total $, ex-top-5 %, verdict; and every seam/parity problem found.
