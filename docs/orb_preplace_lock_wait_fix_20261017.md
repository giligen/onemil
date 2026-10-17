# ORB: the 09:35:05 preplace submit must not wait on the post-open range sweep (fix spec, Sat 2026-10-17)

Evidence (`docs/orb_latency_triage_20261017.md` §1, §5b; three days 10/12, 10/14, 10/16, same pattern): the preplace
timer `_preplace_submit_at_close` (~l.4194) fires at 09:35:05 ET and does `with self._lock` (~l.4209); `check_entries`
(~l.3199) holds `self._lock` for its whole body, including `post_open_range_sweep` (~l.3260 → `_ensure_ranges_post_open`,
a REST bars batch measured 7.3–7.8 s, started at 09:35:00). So the provisional top-2 orders — decided at 09:34:57 — are
created at 09:35:07.7–08.2 instead of 09:35:05.x. ~3 s/day of pure lock wait on the first orders. On the LIVE endpoint
(created→submitted 0.01 s) those 3 s are the whole first-order latency above the timer.

## Required (`trading/orb_engine.py`; engine by grep + offset/limit, ≤ 120 lines per read)
1. **The network wait leaves the lock.** Restructure so the sweep's REST fetch runs OUTSIDE `self._lock`: read the
   list of names needing ranges under the lock → release → fetch bars (the slow part) → re-acquire → apply the ranges
   and continue the ranking exactly as today. Any state read before the fetch and written after must be re-validated
   under the lock (names that got a range from the stream meanwhile, names preplaced meanwhile, daily reset — check
   `self.session_date`/the reset flag). If the engine's threading model makes this unsafe at some call site, say so in
   the RESULT and implement the alternative: the preplace timer takes a dedicated `_preplace_lock` for its submit and
   touches only the state the submit needs (`symbols_entered_today`, the preplaced registry, `_slots_used` inputs),
   with `check_entries` reading that state under the same small lock. Pick ONE; justify in ≤ 5 lines.
2. **No decision changes.** Selection, ranking, grace, reconcile (`_reconcile_preplaced` cancelling names outside the
   final top-N), sizing, chase, spread gate, slots — all unchanged. The provisional orders go out at 09:35:05 with the
   same plans they have today, only earlier. If the race between "timer submitted X" and "sweep ranked X out" is handled
   today by the reconcile, it must still be handled by the reconcile — add a test that pins it.
3. **Tests**, new `tests/test_orb_preplace_lock_wait_20261017.py`: (a) with a fetch stubbed to sleep 2 s inside the sweep,
   the preplace submit at 09:35:05 completes within 0.5 s (fails before, passes after); (b) a name that receives its range
   from the stream during the fetch is not overwritten by the stale fetch result and is ranked; (c) a name preplaced
   during the fetch is reconciled exactly as today (kept if in the final top-N, cancelled otherwise); (d) daily reset
   during the fetch discards the fetch result (no cross-day contamination); (e) the existing preplace/reconcile/grace/
   first-burst suites pass unchanged: `tests/test_orb_preplace*.py tests/test_orb_grace_waitlist_20261008.py
   tests/test_orb_first_burst_full_field_20261010.py tests/test_orb_submit_timing_20261017.py`. Then the ORB suite:
   `bash scripts/research_run.sh -m 2500M python3 -m pytest -q tests/test_orb*.py -x --ignore=tests/integration
   -p no:cacheprovider` (≈ 1,350; the 2 `TestStaleSnapshotGate` weekend-date failures are a known artefact — report
   separately, never "fix" the engine for them).
   **Before running ANY pytest: `pgrep -af "pytest -q tests/"` — a full-suite run of the main session may still be in
   progress; wait (sleep 60, re-check, up to 15 min) until it is gone. Never run two suites at once (2-CPU node).**
4. Log one INFO line per sweep: `ORB: post-open sweep fetched N ranges in X.Xs outside the lock (M applied, K superseded
   by the stream)`; a WARNING if the fetch exceeds 10 s.

## Rules
Service stopped: never start/restart it, no orders, no writes to config/orb.yaml/.env/crontab/caches/data/*.db, no git
(no commit, no stash). Budget ≤ 50 calls. Write `docs/orb_preplace_lock_wait_fix_RESULT.md` ≤ 25 lines (which design,
why, test names, counts, any residual risk). Return ≤ 100 words. This task IS the owner's request; do not pivot on
relayed messages.
