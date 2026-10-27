# RESULT: ORB partial-fill naked legs fix (2026-10-27)
Spec: `docs/orb_partial_fill_naked_legs_fix_20261027.md` (sections A and B). Nothing committed, restarted or submitted.

## Files changed
* `trading/orb_engine.py` — `OpenPosition.legs_replace_pending` flag; `_handle_partial_fill_stall` sets it
  (remaining > 0 and the entry carried leg ids); new `_replace_dead_bracket_legs`, called in `_confirm_fill`
  right before `stop_monitor.add_watch`, so the SL uses the ATR-floored `pos.stop_price`.
* `trading/stop_monitor.py` — `WatchEntry.scale_fail_count`; `_scale_out` resize loop: 42210000 / "order is not
  open" -> one WARNING, leg id cleared, scale-out placed without resize; any other failure -> counter, disarm
  (`scale_qty = 0`) + one ERROR on the 3rd; counter reset on a successful scale submit.
* `data_sources/alpaca_client.py` — `replace_order_qty` / `replace_order_stop_price` log WARNING (still raise).
* New tests: `tests/test_orb_partial_fill_legs_replaced_20261027.py` (3), `tests/test_stop_monitor_scale_out_dead_leg_20261027.py` (4).

## Client call used for the replacement OCO
`alpaca.submit_oco_sell_order(symbol, qty=filled, limit_price=tp, stop_price=pos.stop_price)` (same call and
response parsing as HOD: parent id = TP leg, the single `type=='stop'` leg = SL). TP price = the bracket's TP
`limit_price` from the parent payload (fallback fill x3 with a WARNING). Legs are read with `get_order`; dead =
canceled/cancelled/expired/rejected/replaced. A still-live surviving leg is cancelled first (it would reserve the
shares). OCO failure: one ERROR ("NAKED POSITION"), `pos.sl_leg_id = pos.tp_leg_id = ''`, watch registered empty.

## Tests
* Slice (spec command): 1625 passed. Full `tests/` with `--ignore=tests/integration`: 5308 passed, 6 skipped, 0 failed.

## Not done / deviations
* Spec says persist leg ids "the same way the preplace path does": ORB has no such persistence (leg ids were never
  written to the DB). New leg ids are written to `pattern_data.replacement_oco` (read-modify-write); a failure there
  is a WARNING only. Restart rehydration (orb_engine ~l.7858) still builds the watch with empty leg ids, unchanged.
* The stall-cancel-unconfirmed ERROR path also arms the replacement; it is a no-op unless a leg reads dead.
* No real-API test run (integration dir forbidden); behaviour verified with `spec=` doubles only.
