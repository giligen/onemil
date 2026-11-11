# ORB BT point-in-time params - RESULT 2026-11-11

Changes:
- `trading/orb_pit_params.py` (new): `params_for_day`, `load_history`, `PitParams`. Rule: refit applies iff ts < 09:30 ET of the session day. The jsonl stores no "previous" set, so days before the first refit (9/13) fall back to current with one WARNING; missing/malformed history raises.
- `study_orb_pipeline_static_lock.py`: `params_mode` (:627), `score_days` (:641), `assign_quintile_by_day` (:670); main() scoring and quintiles route through them (~:1047-1100). `--params current` (or env ORB_BT_PARAMS) keeps the old path; default `pit`. Book gains `params_source` and `params_refit_ts`. TRAIN-refit/missing-literal modes force `current`.
- `scripts/eod_sections.py` (BT replica, ~:755): uses `params_for_day(day)`.

Refit input: `orb_weekly_refit.py` fits on raw features + yaml threshold only, no composites or book. Unaffected, not changed.

Tests: `tests/test_orb_pit_params_20261111.py` 12 pass (unit, 2-week integration, 11/3 live-composite parity within 1e-4 and 100 % quintile agreement). With refit/pipeline/observer tests: 43 pass.

Rebuild (9/29-11/10, `research/book_review_20261111/orb_bplus_book_pit.csv`): pipeline top-8 equals live final on 9 of 17 days (expected >= 11; misses are universe/dedup and the truncated 10/30, 11/10 days). Window book +$800 (44 picks, 30 filled) vs published +$1,007. 605 older rows are `current_fallback` (pre-9/13, look-ahead remains there).
