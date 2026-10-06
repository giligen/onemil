# FIX B (HOD paper dry-run) - result, 2026-10-06
Spec: FIX_B_hod_spec.md. Review: B_hod.md. Scope kept: hod_break_engine.py, exit_qty_guard.py, tests (+ two None-guards below).
Suite: `tests/test_hod_killrail_restart_qty.py tests/test_hod*.py tests/test_exit_qty_guard.py tests/test_shutdown_hygiene.py
tests/test_orb_pool_exit_integration.py tests/test_orb_add_on.py tests/test_orb_scale_out_monitor.py` = 438 passed, 1 skipped.

## B1 (HIGH) positions-API error at the flatten
- trading/exit_qty_guard.py:69 `get_signed_broker_qty` -> Optional[int]: None on API error (WARNING, throttled 1/symbol/min),
  0 only when the symbol is genuinely absent.
- hod_break_engine.py:~2978 `force_close_all`: None -> position stays registered, `_fc_unknown`, ONE WARNING/symbol/minute,
  no pop, no exit_pending_verification; the tick loop already re-runs the flatten every tick while any position is open.
- hod_break_engine.py:~2932 the post-close short-circuit is now ONE ERROR naming every still-open symbol (was WARNING).
  tests/test_shutdown_hygiene.py updated to assert the ERROR.
- Other callers of get_signed_broker_qty kept correct under None: hod failure-short (~2440, cover qty), orb_engine.py:2552
  (add-on leg resize skips, WARNING), stop_monitor.py:1865 (scale leg aborts, WARNING, retried next trigger).
- Tests: raises-once-then-succeeds -> exactly one sell of 40; warning throttle; one ERROR after the close; guard + 2 caller tests.

## B2 (HIGH) our quantity, never the broker total
- exit_qty_guard.py:91 `get_our_buy_fill_qty`: sum of filled BUY qty on client ids with prefix `hod-` since ET midnight,
  orders API (`trading_client.get_orders`, status ALL, per symbol); None on error.
- hod_break_engine.py:1576-1604 `_our_buy_fills_today` / `_our_qty` (fills minus registry closed_qty; unreadable or zero fills
  fall back to the REGISTRY qty with a WARNING, never the broker total); `_warn_foreign` = "foreign shares present".
- `_sync_registry_qty_to_broker` (1606) raises to min(ours, broker); adoption (1482-1500) adopts min(ours, broker), skips
  as foreign when our fills are readable and zero, ERROR + skip when unreadable and nothing bounds it; flatten (2984) sells
  min(broker, ours). Test: broker 100 / our fills 40 -> stop/watch/row resized to 40, flatten sells 40.
- Existing adoption/sync tests now feed our fills (tests/hod_fills_helper.py `set_our_fills`).

## B3 (MEDIUM) kill-rail cancel retried
- `_enforce_kill_rails_on_resting` (1677): rail added to `_rail_cancel_done` only when no resting entry order is left; each
  failed pass = WARNING, ERROR on the 5th (keeps retrying), existing 10 s throttle unchanged. Test: get_order raises once.

## F4-F6
- F4 DONE (one-liner + guard): `_record_exit` stamps `_recent_close_ts`; periodic adoption skips a symbol closed < 120 s ago.
- F5 LEFT: fetching positions once per pass and doing REST outside `self._lock` changes the lock scope of a 100-line loop
  (and a pre-fetched list goes stale after `_settle_exit_legs`); not a one-line fix. Fills lookup has a 5 s TTL cache.
- F6 LEFT: `_on_live_fill` rebuilding Position resets closed_qty; an in-place update also needs the OCO sized to
  (cumulative fill - closed_qty), a logic change beyond one line. Masked by the sync, which now cannot over-inflate (B2).

## Notes
- Fills are bought-only (OCO legs and replaced orders lose our prefix); exits are netted via registry closed_qty.
- Not exercised against the real API (no real-API run allowed in this task): the orders-API call shape
  (`GetOrdersRequest(status=ALL, symbols=[sym], after=..., limit=500)`) must be probed on paper before Tuesday's boot.
