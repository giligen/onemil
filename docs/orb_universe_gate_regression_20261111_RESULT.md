# ORB universe gate regression (2a55c82) - result

Root cause: trading/orb_engine.py `_recheck_gap_admission` (scoring + provisional re-checks, ~L1640) re-derived the
gap from `providers['prev_day_bar']['close']` and `range_data.range_open`, instead of judging `feats['gap_pct']`.
Fixtures with stubbed `_compute_features` (gap 12 %, no prev_day_bar, range_open 0) were rejected. The 09:30 filter
path (`_production_universe_cfg_admit`) was equivalent; unchanged.

Fix: new `admit_gap_value()` in trading/orb_universe_gate.py (same comparison/reasons as `admit`); the re-check now
takes `feats` and calls it. Thresholds and the SKIPPED WARNING unchanged.

Tests: +4 regression tests (unit + engine integration) in tests/test_orb_universe_gate_20261111.py.
Run: 244 pass, 1 fail: test_orb_time_stop_leak_20261111::test_missing_broker_ts_falls_back_to_db (uncommitted
time-stop rail, `_adopted_order_submit_ts` L6837: MagicMock `submitted_at` is truthy and passes the hasattr
checks -> min(MagicMock, datetime) TypeError). Not touched, per instruction.
