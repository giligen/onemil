# ORB partial-fill stall fix - result (2026-10-07)

Spec: `docs/orb_partial_fill_stall_fix_20261007.md`. Not deployed (service untouched; live process keeps old code until the next boot).

## Real qty key
`AlpacaClient.get_order` (REST) carries `qty`. The first status source in `_process_pending_fills`,
`OrderStreamWatcher.get_status` (`trading/order_stream.py` `_order_to_dict`, ~:339-354), carries **no `qty` key at all**
(keys: id, client_order_id, symbol, status, filled_avg_price, filled_qty, submitted_at, filled_at, updated_at, event,
reject_reason). `order_status.get('qty', 0)` was 0 -> `_remaining = 0` -> no cancel, log said `170/0 sh`.

## Changes (`trading/orb_engine.py`)
* :5551-5552 requested qty = `pos.shares` (the engine's submitted size); log line now `filled X/<pos.shares>`.
* :5662 `_handle_partial_fill_stall` (new): remainder = `max(pos.shares - filled, 0)`; cancel the parent once (retry next
  tick only if the call raised); re-fetch; confirm the fill qty only when the re-fetch is canceled/expired/filled/
  done_for_day (2026-07-04 rule: confirmed qty = freshest payload). Re-fetch still live -> keep polling; unconfirmed
  after `partial_fill_stall_seconds_max` from the first cancel attempt -> ERROR `stall cancel NOT confirmed`, then confirm
  the observed qty so held shares get a stop (sync_positions orphan-detect is the backstop). remainder 0 -> no cancel.
* :5641 `_warn_payload_qty_mismatch`: WARNING when payload qty exists and differs from `pos.shares`.
* :5615 `elif pos.stall_cancel_sent_at` routes pending_cancel/accepted ticks into the same handler (timeout escalation).
* :292-293 `OpenPosition.stall_cancel_sent_at`, `stall_cancel_acked` (defaults None/False).

## Tests
* New `tests/test_orb_partial_fill_stall_20261007.py` (9): 5 failed before the change, all pass after (no-qty stream
  payload cancels 30 once with the parent id; payload with qty; mismatch WARNING; filled==requested no cancel; 170->200
  between poll and ack confirms 200; unconfirmed cancel waits then ERRORs; raised cancel retried).
* `tests/test_orb_partial_stall_timeout.py`: 2 tests updated to the confirm-the-cancel contract (re-fetch shows canceled;
  cancel-raises case confirms only after the stall window, with ERROR).
* `tests/test_orb*.py --ignore=tests/integration`: **1301 passed**, 0 failed.
