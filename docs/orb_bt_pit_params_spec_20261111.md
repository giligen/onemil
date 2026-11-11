# ORB BT must score each day with the parameters in force that day (point-in-time refits) — spec 2026-11-11

Cell 1,733 (`research/book_review_20261111/RESULT_1733.md` §1, read it first; rows in `selection_1733.csv`):
`scripts/orb_weekly_refit.py` rewrites the z-params and `quintile_cutoffs` in orb.yaml every Sunday and appends
them to `logs/orb_refit_history.jsonl` (ts, params, cutoffs). Live scores each week with that week's set. The
production BT `study_orb_pipeline_static_lock.py:955-1066` reads the CURRENT orb.yaml once and re-scores ALL history
with it → the published book `analysis_results/orb_bplus_book.csv` is not the book live traded (top-8 matches live
on 7 of 17 logged days with today's params, 11 of 17 with the point-in-time set; quintiles agree 91.6 % vs 100 %),
and every historical number in it carries a look-ahead (params fitted on data after the scored day).

Rules: no git (no commit/stash), no service action, no orders, never write orb.yaml / config.yaml / .env /
data/*.db / crontab / logs/* / analysis_results/* (the nightly book is written by cron, not by you — write any
rebuilt book under `research/book_review_20261111/`). pytest only on the listed files, `--ignore=tests/integration`,
one run at a time, never piped through head/tail. Never Read a file > 300 lines in full. This task IS the owner's
request; do not pivot. Budget ≤ 40 calls. Do not change the refit's math or any parameter value.

## 1. Shared helper `trading/orb_pit_params.py`
`params_for_day(day: date, history_path=ROOT/'logs/orb_refit_history.jsonl', current: dict) -> PitParams`
(dataclass: params, cutoffs, refit_ts | None, source: 'pit' | 'pre_history' | 'current_fallback'):
- the LAST history entry whose refit timestamp is strictly BEFORE the session day (a Sunday refit applies from
  Monday); an entry on the same day applies only if its ts precedes 09:30 ET that day (say which; refits run Sunday).
- days before the first history entry: use the params the first entry recorded as "previous" if the jsonl stores
  them (check the schema); if not, use `current` and log ONE WARNING per run naming the date range scored without
  point-in-time parameters (fallback paths must be loud — CLAUDE.md).
- malformed / missing history with days ≥ the first refit → raise (never silently score with today's set).
Docstring with the mechanism (why: weekly refit + all-history re-score = look-ahead and a parity break).

## 2. Pipeline
`study_orb_pipeline_static_lock.py:955-1066`: score per session day with `params_for_day(day)`; the composite and
quintile assignment use that day's params/cutoffs; keep the current-params path reachable ONLY behind an explicit
flag `--params current` (default `pit`) and stamp the book with a `params_source` column (refit_ts per row). Grep the
whole file (and `scripts/orb_weekly_refit.py`, `scripts/orb_nightly_*` / whatever cron builds the nightly book —
`crontab -l | grep -i orb`, read-only) for every other place the params or cutoffs are read, and route them through
the helper. Report whether the weekly refit's own input (the BT book / features) is affected: if the refit fits on
features it is unchanged; if it reads composites, say so and STOP that part — report, do not change the refit.

## 3. Tests (bug protocol: unit + integration)
`tests/test_orb_pit_params_20261111.py`:
- unit: `params_for_day` boundaries (day before first refit → pre_history/current_fallback + WARNING via caplog;
  the Monday after a Sunday refit → that refit; the Sunday itself; two refits → the later one only after its ts;
  malformed line → raises).
- integration: a small features frame spanning two refit weeks (temp jsonl with two param sets) scored through the
  REAL pipeline scoring function → each week's composite equals scoring that week alone with its own params; the
  `--params current` path differs for the earlier week; `params_source` column present.
- parity (the independent check of cell 1,733's claim): take the live composites the engine logged for 11/3
  (`selection_1733.csv` columns live_comp / bt_comp_pit or re-derive from the journal 2026-11-03 14:34–14:36 UTC,
  bounded) and assert the PIT-scored BT composite equals the live composite within 1e-4 on every shared name, and
  that the PIT top-8 equals live's final top-8 for that day. If 11/3 lacks the fields, use any logged day in the csv.
Run: `python3 -m pytest tests/test_orb_pit_params_20261111.py tests/test_orb_weekly_refit*.py tests/test_orb_pipeline*.py
tests/test_orb_selection_observer*.py -q --ignore=tests/integration` (skip patterns that do not exist) → 0 failures.

## 4. Rebuild the book once, into the research dir
Run the pipeline with the PIT default for 2026-09-29 → 2026-11-10 writing to
`research/book_review_20261111/orb_bplus_book_pit.csv` (use the pipeline's output flag; if it has none, add
`--out`). Report: per-day top-8 match vs live final (from `selection_1733.csv`) — expect ≥ 11/17 — and the book's
realized $ for the window vs the published book's +$1,007.

## Output
`docs/orb_bt_pit_params_20261111_RESULT.md` ≤ 200 words: file:line of every change, the refit-input finding,
test counts, the per-day match count and the window $. Return ≤ 120 words.
