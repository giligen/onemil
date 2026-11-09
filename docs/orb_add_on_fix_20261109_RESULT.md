# ORB add-on fix 2026-11-09 — RESULT

Changes, all in `trading/orb_engine.py` (`orb_add_on.py` untouched):
1. ET clock: `_execute_add_on` now uses `self._et_now()` for the 15:00 cutoff. The `now_utc + timedelta(hours=_et_offset_hours)` arithmetic is deleted.
2. Act on the add's result: new `_settle_add_order`. It polls `_await_terminal` for 10 s, then calls `cancel_order` and polls for 5 s more. It books `filled_qty` and `filled_avg_price` from the broker, never the limit. If nothing filled it logs INFO and changes nothing (`pos.shares`, legs, watch and `add_count` stay as they were). If the order is still non-terminal after the cancel wait it logs ERROR. Partial fills book only the filled qty.
3. `_resize_exit_legs_for_add`: `broker_qty < new_qty` is now ERROR. The new `_leg_qty` reads the leg via `get_order` and skips `replace_order_qty` with an INFO line when the leg already equals the target (no 42210000 ERROR).
4. Ledger row (needed a decision): `_persist_add_on_state` already writes `trades.shares` = combined qty plus `pattern_data.add_on.add_avg_px` = combined average. `fill_price` and `entry_price` deliberately stay at the base entry. P&L reads `pos.combined_avg_price`, which is rehydrated from `add_avg_px`. No change needed.

Tests:
- New `tests/test_orb_add_on_fix_20261109.py`, 10 tests: ET clock (3 cases), source guard, delayed fill, never fills, partial fill, legs already at target, broker short of target.
- `tests/test_orb_add_on.py`: the `mock_alpaca` fixture now returns a `get_order` result (the add is booked from the broker, not the submit response).
- The 6 specified files: 109 passed, 0 failed.
