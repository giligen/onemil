# ORB/HOD shutdown hygiene — 2026-10-01 20:04-20:10 UTC incident

## Root cause
1. Tick stall (research-load CPU contention, TIMEOUT > 50s) made HOD's 15:55 ET flatten
   fire 9 min late (20:04:08 UTC, after the regular close) with a huge ORB GAP_GATE scan
   backlog still queued; it kept draining (no fix needed — not shutdown-hygiene) to ~20:09:54.
2. HOD's `force_close_all()` submitted 4 unfillable post-close limit sells and returned
   immediately (no internal wait); the per-tick guard would have re-run it forever.
3. At ~20:10:06 ORB's FC SWEEP/`_verify_flat_with_grace` called `get_open_positions()`
   just as the main thread had returned and Python's atexit sequence tore down the
   thread-pool executor backing that call and Telegram's send. Every retry raised
   `RuntimeError: cannot schedule new futures after interpreter shutdown`, treated as a
   transient fault — ERROR/WARNING + a Telegram attempt per hit for ~10s
   (`fc_verify_max_wait_s=10`) — the 24 ERRORs + rate-cap hit, until exit at 20:10:18.
## Fixes
(a) `trading/orb_engine.py`: new `_is_interpreter_shutdown_error()` + `_shutdown_in_progress()`.
    `_verify_flat_with_grace` and FC SWEEP's `get_open_positions()` set `shutdown_requested`
    and log ONE WARNING (never ERROR/Telegram) on this RuntimeError or an existing shutdown;
    the FC-VERIFY retry loop and the final "FC FINAL FAILURE" alert both stop/skip once set.
(b) `notifications/telegram_notifier.py`: `_post_message` catches this RuntimeError ahead of
    the generic handler — ONE WARNING ("interpreter shutting down, message dropped"), not
    ERROR. Any other RuntimeError keeps the old ERROR path.
(c) `trading/hod_break_engine.py`: `force_close_all()` short-circuits once
    `_minute_of_day() >= close_minute` (16:00 ET) — ONE WARNING per session
    (`_post_close_fc_warned`) listing still-open symbols, returns 0, submits nothing.
    In-session mechanics (`< close_minute`, incl. 15:55 flatten) untouched.
(3) `trading/orb_engine.py __init__`: one INFO line after `sizing.rvol_tilt` is read, logged
    enabled or disabled — `ORB RVOL tilt: enabled=<bool> edges=<e0>/<e1> mults=<m0>/<m1>/<m2> applies_to=<list>`.
## Tests
`tests/test_shutdown_hygiene.py` (new, 9 tests): verify-poll + FC-sweep shutdown stop (a),
rvol boot log (3), HOD post-close short-circuit (c), Telegram warning-not-error (b).
Fix (c) exposed a latent real-wall-clock dependency in `force_close_all()` callers that skip
pinning `_minute_of_day` (harmless pre-fix, false failure past 16:00 ET) — pinned to 955 in
`test_hod_break_engine.py::TestForceClose` (8), `test_eod_exit.py` (2);
`test_orb_force_close_chain.py::FakeEngine` needed the two new helpers bound.
Targeted: 200 passed. Full suite (`-x --ignore=tests/integration`): 4898 passed, 6 skipped,
0 failed (807.73s).

## Boot-check grep
`journalctl -u onemil-trader --since -10min | grep "ORB RVOL tilt:"`
