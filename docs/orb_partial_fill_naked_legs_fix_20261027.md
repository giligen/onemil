# ORB partial-fill stall cancel kills the bracket legs → StopMonitor loops on a dead leg (spec, Fable 2026-10-27)

## Incident (ORB paper account PA3Y, LABX, 2026-10-27)
09:35:05 ET preplace stop-bracket buy 622 @ 8.01/8.03, TP 24.09, SL 7.23 (parent bdb74ee6, SL leg b6a75322, TP leg
6444c9c7). 09:38 partial 423/622. 09:39:00 `partial-fill stall — cancelling remaining 199 sh + accepting observed qty as
final` → `cancel_order(parent)` (orb_engine.py ~l.6141). **Alpaca cancels the whole bracket with the parent: the broker
shows ALL THREE orders `canceled` and the 423-share position has had NO broker exit since 09:39.** The engine never
re-placed legs: `ORB FILL: LABX @ 8.03 x423 — StopMonitor watching (stop=$7.76 …)` and the watch kept the dead leg ids.
12:07 ET price touched the scale level 8.77 → `_scale_out` → `replace_order_qty(b6a75322)` → `42210000 order is not
open` → WARNING "will retry on next trigger" → **retried on every quote tick: 1,316 ERROR lines by 12:48 ET, all to
Telegram** (alpaca_client.py l.1906 logs ERROR before raising). The scale-out never happened. The stop-trigger path is
safe (it reads the leg and falls back to cancel-and-place), so the exit is the in-engine stop / 15:55 force_close.
Live money: none (paper), but the same code runs the live ORB path.

## Fix (mechanism, both halves; no knob changes; scope = exactly this)
### A. `trading/orb_engine.py` partial-fill stall: cancel the parent, then PROTECT the filled shares
After `cancel_order(pos.order_id)` is acked and the observed qty is accepted as final (the existing path that logs
`ORB FILL: … x{filled}`), if the entry order was a bracket (pos has `sl_leg_id`/`tp_leg_id` or `legs=2` was logged),
read both legs via REST; any leg that is not open (status in canceled/expired/rejected/replaced) is dead. Submit ONE
fresh OCO for the FILLED qty — `alpaca.submit_oco_sell_order(...)` if the client has it (HOD uses a replacement OCO:
grep `hod_break_engine.py` for how it builds the OCO after a fill and reuse that client call), else a plain stop-sell
(SL) + limit-sell (TP) pair with `OrderClass.OCO`. Prices: the SL at the stop the engine just set for the position
(`stop=$7.76` in the FILL line, i.e. `pos.stop_price` after the ATR floor), the TP at the bracket's TP price. Update
`pos.sl_leg_id` / `pos.tp_leg_id`, persist them to `pattern_data` the same way the preplace path does, and re-register
the StopMonitor watch with the NEW leg ids (find the call that registered the watch with the old ids and pass the new
ones; `StopMonitor` must not keep a dead id). Log INFO `ORB: {sym} bracket legs cancelled with the parent — replaced
with OCO sl={id[:8]} tp={id[:8]} for {filled} sh`. If the OCO submit fails: ERROR (one line, this IS a naked
position) and leave the watch with empty leg ids so StopMonitor uses its cancel-and-place path.

### B. `trading/stop_monitor.py` scale-out: a dead leg is not retried every tick
In the `_scale_out` resize loop (l.1896–1925): when `replace_order_qty` raises with `42210000` / "order is not open",
(1) log ONE WARNING `… leg {id[:8]} is not open — dropping the dead leg id, placing the scale-out without a resize`,
(2) clear that leg id on the watch (`setattr(watch, attr, '')`) so no later cycle touches it, (3) continue with the
scale-out limit sell for `scale_qty` (the runner's protection is the in-engine stop + cancel-and-place; with a dead SL
leg there is nothing to resize). Any OTHER replace failure keeps today's behaviour (rollback, return False) BUT must be
rate-limited: a per-watch `scale_fail_count`; after 3 consecutive failures the scale-out is disarmed for that watch
(`watch.scale_qty = 0`) with one ERROR line naming the symbol and the reason. Never more than 3 attempts per watch.
Also `data_sources/alpaca_client.py` l.1906 `replace_order_qty`: log the failure at WARNING (the caller decides the
severity — it raises anyway); same for `replace_order_stop_price` if it has the same ERROR-then-raise shape.

## Tests (TDD; unit + integration-style with real `StopMonitor`/`ORBEngine` instances and `spec=` doubles)
`tests/test_orb_partial_fill_legs_replaced_20261027.py`: stall-cancel on a bracket with a partial fill → legs read as
canceled → one OCO submitted for the filled qty at the position's stop/TP → watch registered with the new ids; OCO
submit failure → ERROR once, watch ids empty. `tests/test_stop_monitor_scale_out_dead_leg_20261027.py`: 42210000 on
resize → one WARNING, leg id cleared, scale-out limit sell submitted; a second tick does not call replace again; a
generic failure ×3 → disarmed with one ERROR; `replace_order_qty` failure logs WARNING not ERROR.

## Validation
`python3 -m pytest tests/test_orb_partial_fill_legs_replaced_20261027.py tests/test_stop_monitor_scale_out_dead_leg_20261027.py tests/test_stop_monitor*.py tests/test_orb_*.py tests/test_alpaca_client.py -q -x --no-header -p no:randomly --ignore=tests/integration`
green, then the full `tests/` (same flags, `--ignore=tests/integration`), zero failures. Write
`docs/orb_partial_fill_naked_legs_fix_20261027_RESULT.md` (≤ 40 lines). Do NOT commit, restart, or submit orders.
