# Track B review - HOD paper dry-run (2026-10-03)
Scope read: trading/hod_break_engine.py (rails, reconcile, adopt, sync, _on_live_fill, force_close_all, tick loop),
trading/exit_qty_guard.py. tests/test_hod_killrail_restart_qty.py: 9 passed. Not read in depth: stop_monitor routing, hod_dry_ledger.py.

## F1 (HIGH, loses the day) - transient positions-API error at 15:55 drops the position unsold
- trading/exit_qty_guard.py:67-78 get_signed_broker_qty returns 0 on ANY exception ("fail closed").
- trading/hod_break_engine.py:2874-2880 force_close_all feeds that 0 to resolve_broker_capped_sell_qty -> None (exit_qty_guard.py:~100),
  then `self.positions.pop(sym)` and DB order_status='exit_pending_verification'. No sell is submitted; _flattened can become True.
- Trigger Monday: one 429/timeout on get_open_positions during the flatten pass. Position carries overnight, no stop watch.
- Partial mitigation: the 60 s reconcile may re-adopt it (entered_today), but adopt then inserts a NEW row (existing row no longer open).
- Fix: make get_signed_broker_qty return None on error; in force_close_all treat None as "retry next tick", keep the position registered.

## F2 (HIGH, owner's manual shares sold) - qty sync/clamp uses the TOTAL broker qty of the symbol
- hod_break_engine.py:1539-1548 _sync_registry_qty_to_broker (called every 60 s from :1422 loop line ~1466, and after each fill :1980)
  raises registry qty, StopMonitor watch and the OCO to broker_qty; force_close_all :2874-2880 sells "the broker's qty, not the registry's".
- If the owner holds/buys the same symbol on the shared paper account, the book will resize its stop/OCO to and flatten the owner's shares.
  Likewise :1469-1472 adoption treats any broker long in a symbol in _live_armed_stops/entered_today as ours (even after our own exit).
- Trigger: owner manual trade in a gapper this book armed. Violates feedback_owner_manual_trades_untouchable.
- Fix: track our own cumulative filled-minus-sold qty per symbol; clamp sell/sync to min(ours, broker), never raise above our fills unless
  the extra is traced to our own order ids.

## F3 (MEDIUM) - kill-rail sweep is once per rail even if the cancel failed
- hod_break_engine.py:1610-1618: rail added to _rail_cancel_done BEFORE the cancels. _cancel_live_order (:~1700) leaves the order ARMED
  when the pre-cancel GET fails. That order is never retried -> may fill after the daily kill, with a stop only via periodic adopt (<=60 s late).
- Fix: only add the rail to the done-set when every live_order is cleared; else retry next 10 s pass.

## F4 (MEDIUM) - adoption race after our own exit
- :1469-1472 periodic mode: symbol in _live_armed_stops/entered_today and not in self.positions (popped after exit) and broker positions
  list lagging the sell fill -> phantom ADOPTED row + new watch + flatten sell attempt (guard then skips at 0). Noise/duplicate rows, not loss.
- Fix: skip symbols whose Position closed within the last ~120 s or whose DB row closed today with same qty.

## F5 (LOW/MEDIUM) - force_close_all holds self._lock across several REST calls per position (:2830-2920)
  Ticks/fills block while N positions x (get_positions + order submit) run; get_signed_broker_qty lists ALL positions per symbol (N+1 calls).
  Fix: fetch positions once per pass; do REST outside the lock.

## F6 (LOW) - _on_live_fill (:1887+) rebuilds self.positions[sym] from cumulative filled_qty on every partial fill, resetting closed_qty/
  close_order_id; masked by the :1980 sync but a partial fill arriving after a partial exit re-inflates qty. Fix: update existing Position in place.

## Checked OK
Reconcile loop exceptions: _reconcile_positions_to_broker wraps in try, tick wrapped too; API error only logs ERROR and retries in 60 s (loop does not die).
qty int(float()) handles Alpaca string qty. Caps (:1645-1650) count fills (_entered_today_count/_open_position_count), resting only vs max_resting.
Kill rails: idempotent per rail (but see F3). Post-close flatten skip (:2827) is deliberate and logged.
