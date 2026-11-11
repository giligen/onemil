# ORB selection-parity test — RESULT 2026-11-11

Code: `scripts/orb_pick_parity.py` (new; `run_parity`, `compare`, `summary_line`, CLI `--date [--json]`); `bt_ranked` in
`scripts/eod_sections.py` now also returns vetoes/day_symbols/range_size/params_refit_ts (additive);
`scripts/orb_selection_observer.py::run_pick_parity` (audit key `pick_parity`, Telegram only on ERROR);
`scripts/report_common.py::_pick_parity_suffix` (EOD line). Tests: `tests/test_orb_pick_parity_20261111.py` (32),
real 11/3 fixture `tests/fixtures/orb_parity_2026-11-03.log` -> 8/8, 0 unexplained.

Real runs (read-only):
- `ORB pick parity 2026-11-10: matched 2/8 | diffs 5 (range_missing 4, universe_live_only 1) | unexplained 0`
- `ORB pick parity 2026-11-11: BT side unavailable (features CSV has no rows for 2026-11-11) | live final 8 | not counted | unexplained 0`

Pytest (spec set + other bt_ranked/eod users, `--ignore=tests/integration`): 116 + 116 passed, 0 failed.

Cron/ordering finding: the observer cron is `26 13 * * 1-5` (waits to 09:36:30 ET). The features CSV / nightly book are built
after the close, so at observer time the BT side for TODAY does not exist (`bt_ranked` reads the newest features CSV; PIT
params themselves are correct for any day via `params_for_day`). The observer therefore also scores the PREVIOUS session;
a same-day count needs a post-build run (`orb_pick_parity.py --date`, e.g. in the EOD job after the BT) — cron untouched.
`orb_bplus_book.csv` has entered rows and no `params_refit_ts` yet, so the BT top-8 comes from `bt_ranked`.
A live fill outside the final set errors except add-on-pool fills (by design). 11/10 = known sweep failure (4 ranged).
