# SPEC — fix the silent bars-source fallback in the ORB pipeline and re-run the 2023–24 books under the LIVE exit rule

Found 2026-09-25 by the adversarial verification (`research/orb_verify/REPORT.md`, lenses C and judge). Sonnet, ≤ 40
tool calls, TDD, do not commit. Grep/offset reads only (`study_orb_pipeline_static_lock.py` is ~1,400 lines).
**Market-hours rule:** every DB-heavy step (the two re-runs) must FINISH before 13:25 UTC; check `date -u` first; run with
`nice -n 19 ionice -c3`. Never touch `data/cache.db` except read-only.

## Defect
`study_orb_pipeline_static_lock.py` (~line 553 and `build_atr14_lookup(db_path='data/cache.db')`) sources minute bars and
daily ATR14 ONLY from `data/cache.db`, which has no bars before 2025-01-02. When a symbol-day has no bars it SILENTLY
keeps the features CSV's legacy `pnl` (the old 2R-target / range-low-stop / time-stop simulator) instead of the live
static-lock + touchgo + ATR-floor + scale-out + 15:45 rule. Both out-of-regime runs show it: `research/orb_2023/run_all.log`
and `research/orb_2024/run_all.log` print "ATR14 available for 0/…" and "0 rows scaled" / "0 resimmed rows", and the
books' `exit_reason` are stop/target/eod instead of the live rule's tag_bb/lock/scale_*. So all 165 fills in
`book_1418.csv`, `book_1419.csv`, `book_1415.csv`, `book_1416.csv` were priced by the wrong exit rule. CLAUDE.md: every
fallback path MUST log ERROR/WARNING; production research code never silently substitutes.

## Fix (code)
1. Env `ORB_BT_BARS_DB` (default `data/cache.db`): the SQLite file for minute bars; env `ORB_BT_DAILY_SOURCE` (default: the
   `daily_bars` table of the same DB) accepting a parquet path with columns (bar_date, symbol, open, high, low, close,
   volume) for the ATR14 lookup. Thread both through every bars/ATR read in the pipeline (grep `cache.db`, `sqlite3`,
   `build_atr14_lookup`, the resim loader).
2. Missing bars for an entered symbol-day → `logger.error(...)` per row (capped at 20 lines + a total), the row is
   EXCLUDED from the book (never legacy P&L), and at the end the pipeline prints `RESIM: n_entered, n_resimmed,
   n_missing_bars, atr14_hits` and EXITS non-zero if `n_missing_bars / n_entered > 0.02` unless
   `ORB_BT_ALLOW_MISSING_BARS=1`.
3. Unit tests (`tests/test_orb_pipeline_bars_source.py`): a temp SQLite with bars for one symbol-day and none for another →
   the missing one is excluded and logged at ERROR, the count line is printed, and the > 2 % rule exits non-zero; the env
   var is honoured (a bars DB at a custom path is read). Also run the existing `tests/test_orb_*` files touched by the
   change — zero failures.

## Re-runs (identical invocations to `research/orb_2023/run_all_2023.sh` / `research/orb_2024/run_all_2024.sh`, plus the
new env vars; ORB_CATALYST_VETO=0)
* 2023-01..2024-06: bars `research/orb_2023/bars.db` (table name: check `research/orb_2023/fetch_minutes.py`); daily
  `research/orb_2023/daily_alpaca.parquet`. Cells 1418 and 1419. Outputs `research/orb_2023/book_1418_liveexit.csv`
  etc. — keep the old files, rename them `*_legacyexit.csv` only in the report text (do not delete anything).
* 2024-07..12: find the bars DB and daily source `research/orb_2024/build_features_2024.py` used (grep `bars.db`,
  `parquet`); cells 1415 and 1416 → `research/orb_2024/book_1415_liveexit.csv` etc.
* Prove parity: the run logs must show `n_resimmed ≈ n_entered` (≥ 98 %), `atr14_hits ≥ 95 %`, and the new books'
  `exit_reason` values must be the live rule's (compare the set with `research/thermo/book_2025_26.csv`). If the 2024H2
  bars source cannot provide ATR14, say so and stop — do not fake it.
* Re-score with the FROZEN scorers unchanged in their rules: `research/orb_2023/score_2023.py` and
  `research/orb_2024/score.py` (point them at the new books via a `--book` / env argument you add, leaving verdict logic
  untouched). Append a section "RERUN 2026-09-25 under the live exit rule (bars-source defect fixed)" to both REPORT.md
  files with the new tables and the frozen verdicts, plus the old numbers beside for the diff.

Return ≤ 150 words: tests, n_resimmed / n_entered per cell, the new R/fill, t and frozen verdict per cell, and anything
that could not be done.
