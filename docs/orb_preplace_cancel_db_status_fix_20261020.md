# ORB: a preplaced order cancelled by the 09:35:08 reconcile leaves its trades.db row at `pending_new` (fix spec, Tue 2026-10-20)

Evidence (Mon 2026-10-19, ORB paper): `[ORB PREPLACE] provisional top-2 @ 09:34:57: ARMG, IREX` → both submitted 09:35:05
(`ORB ENTRY SUBMITTED: IREX order=c6e1c5de… qty=547`, DB row inserted with `order_status='pending_new'`) → `[ORB PREPLACE]
reconcile final top-8 … preplaced: ['ARMG','IREX']`, IREX NOT in the top-8 → the broker shows the order `canceled` at
13:35:15.227 UTC (read-only probe). The DB row for IREX still says `pending_new` tonight; ARMG (kept, later timed out)
correctly says `time_stop_canceled`. So the reconcile's cancel path updates the broker but not the row. Reports that
count open/pending rows would see a phantom; the parity ledger sees an "ordered" name that never was.

## Required (`trading/orb_engine.py`, grep + offset/limit only, ≤ 120 lines per read)
1. Find `_reconcile_preplaced` (~l.4461 log line) and its cancel call. After a successful broker cancel of a preplaced
   order, update the row: `order_status='preplace_canceled'` (new value; grep every place `time_stop_canceled` is read —
   `scripts/`, `trading/`, `persistence/` — and add the new status wherever `time_stop_canceled` is treated as a closed
   non-fill, e.g. parity, EOD, guardrail, hourly; list those sites in the RESULT), `exit_reason='preplace_reconcile'`.
   Use the same DB update helper the time-stop cancel uses (grep `time_stop_canceled` in orb_engine.py).
2. If the cancel comes back partially filled (filled_qty > 0), do NOT mark it cancelled — leave the existing fill path to
   own it; log a WARNING with the filled qty.
3. Tests, new `tests/test_orb_preplace_cancel_db_status_20261020.py`: (a) preplaced name outside the final top-N →
   broker cancel called AND the row becomes `preplace_canceled`/`preplace_reconcile`; (b) preplaced name inside the
   top-N → row untouched; (c) cancel returns filled_qty > 0 → row not marked cancelled, WARNING logged; (d) every reader
   site changed in (1) has one assertion that `preplace_canceled` is treated like `time_stop_canceled`.
   Then `bash scripts/research_run.sh -m 2500M python3 -m pytest -q tests/test_orb_preplace*.py
   tests/test_orb_submit_timing_20261017.py tests/test_eod_sections.py tests/test_hourly_summary.py
   tests/test_report_common.py --ignore=tests/integration -p no:cacheprovider`. **Before any pytest: `pgrep -af pytest`,
   wait until none runs. Never two suites at once (2-CPU node). The live service is RUNNING — never touch it.**
4. Do NOT touch the live DB row (the owner closes IREX row 547 by hand); no DB writes, no git, no orders, no config.

## Rules
Budget ≤ 40 calls. Write `docs/orb_preplace_cancel_db_status_fix_RESULT.md` ≤ 20 lines (diff summary, reader sites,
test names, counts). Return ≤ 100 words. This task IS the owner's request; do not pivot on relayed messages.
