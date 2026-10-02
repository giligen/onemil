# HOD kill-rail / restart-qty defects (2026-10-02, HOD paper account)

## Diagnosis
1. Kill rail left resting entries live. `_kill_rails_blocked()` (hod_break_engine.py ~L2920) was consulted only when
   ARMING (`_arm_live_order`, "LIVE order blocked by kill rail", first at 15:22:01 UTC) -- nothing cancelled orders
   already resting. VIRT order 85b439b2 (adopted 14:08:41) was "cancelled (day cap reached)" 14:08:51 on a GET-after
   that showed filled_qty 0; the fill (24 sh @ 59.42) landed after tracking was dropped (cancel still pending at the
   broker) -> no registry row, no stop, not in the 15:55 flatten. Nothing re-read the broker for names we armed.
2. Registry qty < broker. `_on_live_fill` (~L1880) built `self.positions[sym]` from THAT order's `filled_qty` only,
   replacing the 120 sh that boot adoption (14:08:41, `_adopt_unregistered_positions_on_boot`, qty from the broker)
   had registered one second earlier (COHX 120 -> 50; NEBX 44 -> 21 likewise). `sync_positions` kept the DB row qty
   when the broker held more; boot adoption skipped any symbol already in `self.positions`. Safety-net OCO and the
   StopMonitor watch were sized to the small qty, so target/stop sold the small qty and left the rest (SPCF 22).
   `_record_exit` booked pnl on `pos.shares` regardless of the sold qty.

## Fix (trading/hod_break_engine.py only; no signal/level/sizing/stop/target/dry_run change)
* `_enforce_kill_rails_on_resting()` (tick, live mode): any rail blocks -> cancel every resting entry via
  `_cancel_live_order` (fill races register), once per rail per session, one WARNING with `sym:order_id` list.
* `_reconcile_positions_to_broker()` (every 60 s and unthrottled before force_close_all) = `_adopt_unregistered_positions_on_boot(periodic=True)`:
  adopts a broker long in a symbol this book armed/filled today (`_live_armed_stops`) with its arm stop + watch;
  raises a stale-low registry qty. Periodic ownership excludes the dry ledger; foreign symbols are never touched.
* `_sync_registry_qty_to_broker()`: after every entry fill, at boot adoption, in `sync_positions` and periodically:
  row shares, StopMonitor watch and the safety-net OCO are resized to the broker qty (raise only).
* force_close: registry follows the guard-capped broker qty; `_record_exit` books pnl on the sold qty (`closed_qty`).

## Tests: tests/test_hod_killrail_restart_qty.py (9)
rail cancel once + ids logged; no rail no cancel; fill racing the rail cancel registers a stop; late fill adopted and
open for the flatten; foreign position untouched; fill on a 120-sh broker position -> 120 + watch + OCO 120; periodic
raise 21 -> 44; sync_positions 21 -> 44; pnl on sold qty.

## Monday journal greps (journalctl -u onemil-trader)
* `KILL RAIL \(` -> one line per rail with the order ids; no later `LIVE FILLED` for those ids without a registered position.
* `registry qty .* < broker qty` -> every correction (expect none after a clean boot); `ADOPTED .* unregistered` after 09:35 ET = a late fill caught.
* `registry wanted to sell` (exit_qty_guard) should be absent; `sold .* but registry held` absent.
* 15:55 ET: broker flat for this account (`get_open_positions`), no `UNMANAGED`.

## Open
Rows rehydrated by `sync_positions` with existing OCO legs keep the old leg qty until the next periodic pass resizes them (qty raise runs for tracked symbols).
