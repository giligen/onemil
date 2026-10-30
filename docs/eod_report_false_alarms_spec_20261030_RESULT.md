# RESULT — EOD / green-check false alarms (spec 2026-10-30)

## Files changed (report side only; `trading/` and `main.py` untouched)
* `scripts/report_common.py` — new `unfilled_bt_filled_symbols(rows, bt_filled_syms)`; `green_verdict` fill-parity uses it. A
  (symbol, day) is FILLED if ANY ORB row has `fill_price` or status in (`filled`,`closed`); only symbols with rows and no
  filled row are flagged (7/30 IREX shape still flagged).
* `scripts/eod_report.py` — `unit_installed` (`systemctl cat`), `in_trader_window` (weekday 12:35-19:55 UTC),
  `service_status_flags`, `_utc_now`; `hygiene_section` uses them. Trader inactive outside the window prints
  `onemil-trader=exited-at-close (expected)`; macd-wave without a unit file prints `onemil-macd-wave=not-installed`, never NOT ACTIVE.
* `scripts/eod_sections.py` (DEVIATION, see below) — `parse_orb_log` now returns `preplace_ranked` (provisional top-N +
  reconcile-final lists) and `submitted` (`ORB ENTRY SUBMITTED`); `ENGINE_LINE_PATTERNS` keeps `ENTRY SUBMITTED`;
  `orb_parity_lines` counts those names as engine-ranked/ordered; `why_not_ordered` returns "" for a submitted symbol and the caller skips it.
* `tests/test_eod_report_false_alarms_20261030.py` — 18 tests (fill-parity 5 unit + 1 sqlite integration through the real `load_live_rows` -> `green_verdict`; selection parity 4; hygiene 8).

## Deviation
The spec places the selection-parity logic in `report_common.py` ~l.655-720; that range is `decision_parity` (composite drift). The
`ORB ranked: engine N vs BT M ... BT-only` and `not scored by the engine` lines live in `scripts/eod_sections.py`, which is imported
only by `eod_report.py` and tests (not by the trader). The fix is there.

## Dry runs (`--dry-run --no-telegram`; code-confirmed: dry-run skips `streak_update` and `reconcile_freeze` returns before set/clear; `ramp_freeze.get` is read-only)
* 2026-10-29: GREEN, "BT parity clean (3 BT picks)", orb -$163. Freeze line: `would CLEAR` (the live freeze, since 10/22, reason CORD/IREZ fill-parity, is owned by the 10/29 check).
* 2026-10-27: GREEN (5 BT picks, orb +$1,859). Freeze `unchanged` (not raised by 10/27's check). NOTE: the live freeze is still set; clearing it is a real (non-dry) re-run of 10/29 or `orb_ramp_check.py --clear-freeze orb`, left to the caller.

## SAIQ (10/23 freeze reason "BT picks never ordered live: ['SAIQ']")
Journald no longer holds 10/23 (starts 10/28), so the requested `journalctl` returned nothing; read `logs/session_archive/2026-10-23.log` instead.
SAIQ gap 27.65 %, `ORB SCORED: SAIQ comp=0.2031 Q3` at 13:35:08 UTC (production pool); the only `ORB ENTRY SUBMITTED` that day in the archive is SK;
no SAIQ row in `trades` for 10/23. So it was SCORED (not a preplace blind spot) and never ordered: a real skip. The archive is grep-filtered, so
the exact veto/selection reason is not recoverable from it (no SAIQ veto line kept).

## Defect 4 proposal (not implemented; `trading/orb_engine.py` frozen)
The engine's `ORB DAILY REPORT ... P&L=$-228` fires at 19:45:08 UTC, the same second as the AXTQ force-close (+$65), so it sums the closed rows before the force-close exit is booked
(Telegram -$228 vs ledger -$163). Fix: emit the daily report only after the force-close exits are booked (await the close-all fills / run the report one tick
after the 19:45 sweep, or after `exit_pending_verification` clears), and compute P&L from the DB's closed rows (`exit_price IS NOT NULL`) rather than an in-memory running sum.
Rehearse over a weekend (boot on the exact ExecStart) before the 12:30 boot of a live day.

## Tests / suite
New file 18/18 pass. Full suite (`tests -x --ignore=tests/integration`): 5384 passed, 6 skipped, 0 failed (645 s).
