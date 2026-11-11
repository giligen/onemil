# Nightly ORB selection-parity test: live final top-8 == point-in-time BT top-8, every difference named — spec 2026-11-11

Cell 1,733 (`research/book_review_20261111/RESULT_1733.md` §5; rows `selection_1733.csv`): the existing observer
`scripts/orb_selection_observer.py` compares live "range complete" names to a snapshot-derived CANDIDATE POOL
(`expected_candidates`), so it cannot see that live traded names the BT ranked out, or that the BT entered names
live never submitted. The parity the rulebook requires is at the PICK level. The BT side now scores point-in-time
(`trading/orb_pit_params.py`, book columns `params_source` / `params_refit_ts`, commit eafb557).

Rules: no git (no commit/stash), no service action, no orders, never write orb.yaml / config.yaml / .env /
data/*.db / crontab / logs/* (the observer's own audit files under logs/ are written by the CRON run, not by you —
your test runs write to a temp dir) / analysis_results/*. pytest only on the listed files, `--ignore=tests/integration`,
one run at a time, never piped through head/tail. Never Read a file > 300 lines in full. Journal greps bounded.
This task IS the owner's request ("fix ORB" = selection parity, held by a nightly test); do not pivot. Budget ≤ 40.
Telegram: `logger.error` reaches the owner — only for an UNEXPLAINED difference; everything explained is INFO.

## 1. `scripts/orb_pick_parity.py` (new, importable + CLI `--date YYYY-MM-DD [--json out]`)
- BT side: the day's PIT top-8 from `analysis_results/orb_bplus_book.csv` (entered rows for the date; if the book
  lacks ranks, recompute from the day's rows of the latest `analysis_results/orb_features_*.csv` through the
  pipeline's shared scoring with `params_for_day` — reuse `scripts/eod_sections.py::bt_ranked`, which already does
  this) with each name's composite and quintile.
- Live side: the reconcile's final top-K, submissions and fills from the journal
  (`journalctl -u onemil-trader --since "<date> 13:30" --until "<date> 16:30" --no-pager -o cat`, grep the
  `[ORB PREPLACE]` provisional/final/PARITY lines, `ORB ENTRY SUBMITTED`, `ORB: ... range complete`, the
  `[ORB] final ranges:` line if present) with fallback to `logs/session_archive/<date>*` when the journal has
  rotated (cell 1,733: the journal holds ~a week), plus trades.db (ro) fills for the date.
- Rule: `set(live_final) == set(bt_top8)`; for every shared name `|live_comp − bt_comp| ≤ 1e-4` when the live
  composite is logged.
- Every difference gets exactly one reason from: `universe_live_only | universe_bt_only | veto_input_range |
  veto_input_pdr | param_ts_mismatch | addon_pool | provisional_fill | no_fill | range_missing | time_stop_leak |
  unexplained`. Derive each from the logs/book (e.g. `addon_pool` when the live fill's pool ≠ production;
  `range_missing` when the name never logged "range complete"; `no_fill` when submitted but no fill;
  `param_ts_mismatch` when the book row's `params_refit_ts` ≠ the refit in force that day).
- Output: a dict {date, bt_top8, live_final, live_submitted, live_filled, matched, differences:[{symbol, side,
  reason, detail}], n_unexplained}; a one-line summary "ORB pick parity <date>: matched k/8 | diffs n (reasons…) |
  unexplained m". `n_unexplained > 0` or a live FILL not in `live_final` → `logger.error` (one line, names included);
  else INFO.

## 2. Wiring
- `scripts/orb_selection_observer.py`: after its existing verdict, call the parity for the day and include the
  summary line in its audit json (`pick_parity` key) and its Telegram text only when ERROR.
- `scripts/report_common.selection_parity_line` (used by the EOD hygiene section): append "picks k/8, unexplained m".
- Do not change the cron (owner's call); say in the RESULT which cron runs the observer and that the nightly book
  must exist before it (if the observer runs at 14:36 UTC before the nightly build, the BT side must use the latest
  features csv through `bt_ranked` — verify that path is point-in-time for TODAY, i.e. uses the refit in force today).

## 3. Tests (unit + integration)
`tests/test_orb_pick_parity_20261111.py`: unit — reason derivation for each class on synthetic inputs; the ERROR
path only on `unexplained` or a fill outside the final set (caplog). Integration — a temp book csv + a captured
journal text fixture for 2026-11-03 (take the real lines from `selection_1733.csv` / the journal, bounded) → the
function returns matched 8/8 (or the exact 1,733 result for that day, state which) with every difference named and
zero unexplained; a second fixture with a fill outside the final set → ERROR. Run:
`python3 -m pytest tests/test_orb_pick_parity_20261111.py tests/test_orb_selection_observer*.py
tests/test_report_common*.py tests/test_eod_sections*.py -q --ignore=tests/integration` → 0 failures.
Then a REAL run: `python3 scripts/orb_pick_parity.py --date 2026-11-10` and `--date 2026-11-11` (read-only, prints
only) — paste the two summary lines in the RESULT.

## Output
`docs/orb_selection_parity_test_20261111_RESULT.md` ≤ 200 words: file:line, the two real summary lines, test counts,
the cron/ordering finding. Return ≤ 100 words.
