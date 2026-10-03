# FIX A result (2026-10-03)
- A1: `_fetch_today_open_bars` computes remaining per chunk; <5 s stops (WARNING now also counts failed symbols);
  call gets `timeout_s=min(remaining-1,20)`, `retries=0`. `AlpacaClient.get_1min_bars_range_multi` gained optional
  `timeout_s`/`retries` (default None = unchanged for other callers). Failed chunk -> misses, 60 s retry.
- A2: new `ORBEngine._open_at_0930_from_frame` takes only the row stamped 09:30 ET (tz-aware/naive, column/index); else miss.
- A3: `_orb_tick` order is now check_exits -> build (if due) -> check_entries; docstring comment explains.
- A4: admission loop collects exceptions; ONE WARNING `ORB admission: n symbol(s) skipped on exception (first: ...)`,
  ERROR once if every candidate failed. `_get_rvol_tilt_mult` keeps fail-open x1.0, WARNING once per day.
- A5: `build_universe` guarded by `_build_in_progress` (cleared in finally); overlap skipped with one INFO; body moved
  to `_build_universe_locked`.
## Tests
- New `tests/test_orb_fix_a_20261003.py`: 12 passed.
- Pre-existing `tests/test_orb_gap_input_parity_20261002.py`: mock lambda updated to accept **kw; fixtures made
  wall-clock independent (bars stamped 09:30 ET, fixed 09:35:30 now) - they were failing before 09:31 ET and with A2.
- Suite `tests/test_orb_gap_input_parity_20261002.py tests/test_orb*.py tests/test_prestage_approach_intake.py
  --ignore=tests/integration`: 1275 passed, 3 failed (all in the gap-parity file, fixed after); re-run of
  gap-parity + fix_a + stall files: 34 passed. Full suite not re-run after the fixture fix.
## Replay parity
`PARITY 2026-10-02: engine admitted 128, BT admitted 128 ... 09:30 REST calls 0, stale injected []` (no diffs).
