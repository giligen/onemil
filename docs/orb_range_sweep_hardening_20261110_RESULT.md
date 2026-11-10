# ORB range-sweep hardening — RESULT 2026-11-10

Spec: `docs/orb_range_sweep_hardening_spec_20261110.md`.

**Structure chosen (§1): RE-ENTRANT, not a sleep loop.** The sweep runs on the check_entries tick/drain thread
(`_sweep_outside_lock`), so the first call keeps today's pass + one sleep/retry; if names are still rangeless before
`entry.sweep_deadline_et` (default 09:40:00) it returns without setting `done`, and each later tick runs one pass over
ONLY the leftovers, paced >= `sweep_retry_delay_s` apart. Final outcome via `_finalize_sweep`: INFO "filled n/n",
WARNING "gave up ... short_window/no_bars/no_open_bar" (8 names each), one ERROR/day when < 50 % of >= 10 filled.

**Changes** (`trading/orb_engine.py`): config + state (~481, ~947, ~8342 reset), `_post_open_sweep` (~1740-1990),
`_finalize_sweep`, `_classify_rangeless`. `scripts/orb_selection_observer.py`: `classify_drop` (end 09:34:59),
`send_verdict` (clean = log only). `scripts/report_common.py`: `selection_parity_line`; wired into
`scripts/eod_report.py::hygiene_section` (one line).

**Tests**: new `tests/test_orb_range_sweep_hardening_20261110.py` (12). Three old tests edited to the new structure
(none deleted): `test_orb_post_open_sweep.py` idempotent/empty-bars (clock past the deadline, pacing sleep),
`test_orb_preplace_lock_wait_20261017.py` (done stays False while a name is rangeless). Run: 219 passed, 0 failed.

**Not done / notes**: `eod_report.py` touched (one line + import) since §2's hygiene line lives there. Observer main()
not exercised end-to-end (functions extracted and unit-tested). Takes effect at the next trader boot.
