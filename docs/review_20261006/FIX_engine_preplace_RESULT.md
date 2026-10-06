# FIX_engine_preplace RESULT (2026-10-06, finished ~04:55 UTC) — GREEN, nothing committed, no service touched

## Changed (trading/orb_engine.py)
- Four veto methods take `record: bool = True`. `record=False` = pure decision (no `_pdr_vetoed_today`, `plan_submitted`,
  `rejected_reason`), logged `[ORB PREPLACE] provisional <VETO> ...`; `record=True` log text byte-identical to before.
- `_preplace_provisional_rank` uses new `_provisional_veto()` (record=False); provisional vetoes -> `self._preplace_vetoed`
  (sym -> reason). DEVIATION: sibling attribute, not `_preplace_state['vetoed']` (that dict is symbol-keyed and iterated at
  `list(keys())` / `.items()` -> a 'vetoed' key would crash reconcile).
- `_reconcile_preplaced` ranks ALL production candidates on final ranges (new `_rank_production_final`: features, phantom-gap
  floor, composite, threshold, quintile, skip_q1, same sort) and takes dedup top-`max_concurrent`. Outside + submitted ->
  cancel + `_pdr_vetoed_today.add` (spec); inside -> kept/replaced as before; submit FAILED -> untouched, re-evaluated by
  the normal pass; filled-but-outside -> WARNING + counted, fill stands. New log `[ORB PREPLACE] reconcile final top-N ...`.
- SPEC ADDITION (needed for DFDV on Monday): reconcile-cancelled names go to new `_preplace_dropped_today`, subtracted in the
  three slot counts (cap check, `_run_pool_selection` budget, provisional budget). Reason: their DB trade row counts as
  'entered', so with JAGX/CRCG submitted OK and cancelled the budget would be 8-2=6 and DFDV (rank 7) is missed AGAIN.
  Both sets reset in `reset_daily`.

## Tests
- NEW `tests/test_orb_preplace_budget_20261006.py` (11): 10/5 fixture (journal comps/quintiles, PDRs from the features CSV,
  real provisional top-8). (a) failed-submit path: picks == [DFDV], `_pdr_vetoed_today` == BT set {SUPV,HOG,BNC,ELPC,GGB,
  EWZS,BCYC} (7); success path: JAGX/CRCG cancelled, still DFDV. (b) provisional pass leaves vetoed set empty, budget 8;
  record=False purity for all four vetoes. (c) outside cancelled / inside kept, no double submit. (d) failed submit re-evaluated.
- Fail-before check (mutation: provisional vetoes recorded again): (a) both variants + (b) FAIL (picks [] , vetoed 6 names).
- `tests/test_orb_pool_book.py::test_no_marker_means_not_computed_not_zero` was STALE since 699fe0b (expected note ""); assertion
  updated to the new "marker: candidates 5, picks 0" note. Not engine-related.
- `pytest tests/test_orb*.py --ignore=tests/integration`: **1284 passed**. Other ORBEngine test files: 301 passed.

## Replay (scripts/orb_open_tick_replay.py --parity; it replays the universe build/admissions, NOT selection)
- 2026-10-05: engine admitted 82, BT admitted 82; engine-only [], BT-only []
- 2026-10-01: engine admitted 35, BT admitted 35; engine-only [], BT-only []
- 2026-10-02: engine admitted 128, BT admitted 128; engine-only [], BT-only []
- DFDV selection is shown by the 10/5 fixture test, not the replay.

## Docs
`docs/orb_preplace_spec_20260928.md` (reconcile section, vetoes limitation) and `research/orb_machine_rules.md` (L7 preplace rule).

## Known gaps (unchanged, not in spec)
- A preplaced name INSIDE the final top-N is not re-vetoed on final data (range-size/catalyst); dedup in the normal pass sees
  only non-preplaced names. Reconcile still latches on the first tick (grace-defer partial field only ever drops more names).
- Service restart needed at the 12:30 boot; no live rehearsal done (no orders/restarts allowed).
