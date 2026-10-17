# ORB preplace lock-wait fix - RESULT (2026-10-17)

Spec: `docs/orb_preplace_lock_wait_fix_20261017.md`. Code: `trading/orb_engine.py`. No commit made.

**Design chosen: release the lock around the network waits (spec option 1).** `check_entries` now calls
`_sweep_outside_lock()` BEFORE taking `_lock`. It runs the sweep (`_post_open_sweep(release_lock=True)`) in short
locked sections: read missing names + `_daily_epoch` -> release -> REST fetch -> re-acquire, ingest -> (if stragglers)
release for `sweep_retry_delay_s` sleep + retry fetch -> re-acquire, ingest, mark done. The result is handed to
`_check_entries_locked(pre_swept=...)`, so ranking/reconcile/sizing/grace code is untouched.
Why safe: `check_entries` callers (scanner tick, drain thread) never hold `_lock`; a separate `_sweep_flight_lock`
serialises two concurrent sweeps (second waits, then sees done - as it waited on `_lock` before) and is never taken
while holding `_lock`. Re-validation at ingest: names that got a range from the stream are not overwritten (counted
"superseded", kept in the returned set so they stay rankable); `_daily_epoch` (bumped in `_reset_daily_locked`)
mismatch discards the fetch (WARNING, done flag stays False). `_ensure_ranges_post_open()` stays argument-free and
in-lock for direct callers/stubs (thread-local flag selects release mode). Retry sleep+fetch also left the lock.
Log: `ORB: post-open sweep fetched N ranges in X.Xs outside the lock (M applied, K superseded by the stream)`;
WARNING if fetch > 10 s.

Tests: new `tests/test_orb_preplace_lock_wait_20261017.py` (6): preplace submit < 0.5 s during a 2 s fetch (failed
before: 2.01 s), stream range not overwritten + ranked + log line, preplaced-during-fetch reconciled (KEEP kept, DROP
cancelled), daily reset discards fetch, straggler retry sleep/fetch outside lock, legacy in-lock call.
Suites: preplace/grace/first-burst/submit-timing/sweep set 138 passed (3 stub-signature failures fixed by keeping
`_ensure_ranges_post_open` argument-free); ORB suite `tests/test_orb*.py`: 1356 passed, 0 failed (today is Saturday,
so no TestStaleSnapshotGate weekend failures appeared).

Residual risk: fast_submit mode unchanged (no REST sweep). `_process_pending_fills`/PM prefetch/anchor pre-warm now run
after the sweep instead of before (independent). Needs a weekend boot rehearsal before live; first live check: the
09:35:05 order `created_at` vs timer, and the new sweep INFO line.
