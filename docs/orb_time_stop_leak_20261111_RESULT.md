# ORB "resting entry filled 3.5 h after time stop" (AXTL 9/29) - result 2026-11-11

**Path: (d) ledger clobber, not a leak.** AXTL (trade 382) was submitted 13:35:07 UTC and filled ~13:41 UTC
(`submit_to_fill_ms` 371194; `ORB ATR FLOOR AXTL ... (bound)` + `SCALE OUT ARMED` logged 13:42:00, pid 197526).
Seven restarts that day (pids 237564 ... 360967, logs/session_archive/2026-09-29.log) each re-ran `_confirm_fill`;
the last (18:09:21) overwrote `filled_at`. Already fixed 9/30 (`already_confirmed` guard, orb_engine.py ~6712).
Ledger scan (trades.db, all 'orb' rows since 5/18): AXTL is the ONLY row with filled_at > submit + 60 min.

**Hardening added (rail):** `ORBEngine._adopted_order_submit_ts` (orb_engine.py ~6837) - broker `submitted_at`/
`created_at` is the restart clock (earlier of broker/DB wins); `sync_positions` State B logs WARNING naming
order and age when adopted past `time_stop_minutes`; first `check_entries` tick cancels (counters incl.).
Deadline = per-order submit + `time_stop_minutes`.

**Tests:** tests/test_orb_time_stop_leak_20261111.py 6 pass (4 unit, 2 integration). Listed set: 162 pass,
15 FAIL in tests/test_orb_preplace_* (late_fill, at_close, budget, lock_wait, cancel_db_status), pre-existing:
still fail with my sync change reverted (wall-clock suspected; not investigated).
