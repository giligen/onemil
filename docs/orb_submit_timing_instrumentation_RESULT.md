# ORB submit-timing instrumentation: RESULT (2026-10-17)
Logging only. Tripwire threshold (10 s) and action unchanged; selection, sizing, chase untouched. No git, no service touch.
Files: `trading/orb_submit_timing.py` (new, one writer/parser module), `trading/orb_engine.py`,
`data_sources/alpaca_client.py`, `scripts/eod_sections.py`, `tests/test_orb_submit_timing_20261017.py` (14 tests).
New log lines:
* `[ORB] SUBMIT TIMING sym= decision_ts=HH:MM:SS.mmm quote_ms= size_ms= http_ms= decision_to_http_start_s= broker_submitted_at= broker_minus_decision_s= path=production|addon|preplace|reconcile_resubmit`
* `[ORB] SUBMIT BURST n=<k> span_s=<x.xx> serial=<yes|no>` (per `_run_pool_selection` call, preplace submit, reconcile)
* `LATENCY TRIPWIRE` / `first order submit` lines and the Telegram text now carry `clock=broker|engine`.
Mechanics:
* The wrapper responses (`submit_stop_bracket_order`, `submit_bracket_order`) now include `submitted_at` / `created_at`
  (ISO, from the SDK order; None if absent). No extra API call.
* decision_ts = the `t_rank` stamp the SUBMIT LATENCY line uses (preplace: `_t_rank` at 09:35:00; reconcile: resubmit time).
  size_ms = spread/tilt lookup + plan build + buying-power check (NA for preplace/reconcile). Tags travel as plan attributes
  (`timing_path`, `timing_decision_ts`, `timing_size_ms`), so `_submit_entry(plan)` call sites keep their signature.
* Tripwire clock = EARLIEST broker `submitted_at` today when any exists, else engine (`clock=engine`). Missing broker
  timestamp: `broker_submitted_at=NA` plus one WARNING per ET day. Dry-run orders have no broker clock, so they stay engine.
* EOD: `ORB submit latency: broker median/max (n) | engine median/max (n)` added to the ORB parity block
  (`orb_parity_lines`, after the P&L line) via `timing_summary_line`. The old number had no EOD line before; it is now `engine`.
  The 21:58 UTC archive cron already keeps `[ORB` lines, so the new lines reach the archive.
Notes: path `chase_replace` does not exist as a `_submit_entry` caller (only the preplace-reconcile REPLACE resubmits),
so it is covered by `reconcile_resubmit`. Negative broker delay is possible for preplace (09:34:57 provisional).
Tests: `pytest tests/test_orb*.py tests/test_eod_sections.py tests/test_eod_report.py tests/test_alpaca_client.py
--ignore=tests/integration`: 1514 passed, 0 failed (the TestStaleSnapshotGate weekend artefact did not trigger).
Not done: real-API probe and weekend boot rehearsal (service untouched); verify Monday that `submitted_at` is populated.
