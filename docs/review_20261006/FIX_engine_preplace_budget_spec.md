# FIX — preplace provisional vetoes must not consume the 09:35 budget; reconcile against the FULL final top-N (2026-10-06)

Diagnosis: `docs/review_20261006/DIAG_RESULT.md` (read it first; file:line there). 10/5: provisional rank at 09:34:57 vetoed
EWZS, BCYC, PURR, SUPV, HOG, ALVO through the four veto methods, which `self._pdr_vetoed_today.add(sym)` as a side effect
(orb_engine.py ~:4896/4930/4959/4984). At 09:35:25 `_run_pool_selection` budget = 8 − 6 = 2 (~:3614), top-2 SUPV/HOG re-vetoed,
nothing entered; DFDV (rank 7, the BT's only pick after vetoes) never reached. `_reconcile_preplaced` (~:4175-4196) ranks only
the preplaced symbols among themselves.

## Required
1. `_preplace_provisional_rank` must NOT mutate `_pdr_vetoed_today` (or any day-level veto/slot state). Implement a `record: bool`
   argument on the four veto methods (default True; the provisional pass passes False) — no snapshot/restore hacks. Provisional
   vetoes are kept in `_preplace_state['vetoed']` and logged with the `[ORB PREPLACE]` prefix so the log distinguishes them.
2. `_reconcile_preplaced` (runs once at the first full scoring after the open): rank ALL production candidates with final ranges
   exactly as `_run_pool_selection` does (same comp, threshold, skip_q1, dedup), take the final top-N; for every preplaced name
   OUTSIDE the final top-N: cancel its resting order if it was submitted, slot consumed (`_pdr_vetoed_today.add`) — the no-refill
   rule; preplaced names INSIDE stay. Then the normal `_run_pool_selection` runs on the final top-N with the four vetoes applied
   ONCE (which rebuilds `_pdr_vetoed_today` from the final ranking), budget = N − |entered ∪ vetoed ∪ open| as today. A preplaced
   name whose submit FAILED is neither entered nor open — it is simply re-evaluated by the final selection.
3. The BT rule is the spec (one top-N at 09:35, vetoes once, no refill). Update `docs/orb_preplace_spec_20260928.md` (:96-99 and
   the reconcile section) to say so; add one line to `research/orb_machine_rules.md` under the preplace rule.
4. Tests (`tests/test_orb_preplace_budget_20261006.py`, fail before / pass after):
   a. 10/5 fixture rebuilt from the journal (provisional set, final SCORED comps for the top-12, PDR values for the six vetoed and
      DFDV): final picks == {DFDV}; `_pdr_vetoed_today` after the final pass == the BT's vetoed set (7 names).
   b. A provisional veto never changes the final budget.
   c. Reconcile with a preplaced name outside the final top-N cancels it and consumes the slot; inside → kept, no double submit.
   d. Preplaced submit FAILED → re-evaluated, submitted if inside the final top-N and not vetoed.
5. Replay: `bash scripts/research_run.sh -m 2500M python3 scripts/orb_open_tick_replay.py --help` then run it for 2026-10-05 (and
   10/1, 10/2) with `--parity`; the 10/5 replay must select DFDV; admissions unchanged. Paste the lines.
6. `bash scripts/research_run.sh -m 2500M python3 -m pytest -q tests/test_orb*.py -x --ignore=tests/integration -p no:cacheprovider`
   green (≈ 1,300 tests). Report the count.

## Rules
Never start/restart the service, never submit orders, never edit config/orb.yaml/.env/crontab/caches, no git. Read the engine by
grep + offset/limit only (≤ 120 lines per read). Budget ≤ 60 calls. If the fix cannot be made green by 10:30 UTC, STOP, leave the
tree as it is and say so (the boot is at 12:30 UTC; I will stash). Write `docs/review_20261006/FIX_engine_preplace_RESULT.md`
≤ 40 lines; return ≤ 120 words. This task IS the owner's request; do not pivot on relayed messages.
