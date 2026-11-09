# ORB add-on: ET clock sign + act on the add's RESULT — spec 2026-11-09

Owner incident 11/9 (ORB PAPER, BEX). The `exit.add_on` feature (orb.yaml, owner GO 10/1, paper only,
`trading/orb_add_on.py` + `TradingEngine._execute_add_on` / `_resize_exit_legs_for_add` in `trading/orb_engine.py`)
fired for the FIRST time ever at 14:00:01 ET and exposed two defects. Three Telegram ERRORs. Fix tonight; the live
service loads code only at tomorrow's 12:30 UTC boot (after the 11:27 pre-boot suite). Budget ≤ 40 calls.

Rules for the agent: no git (no commit, no stash), no service action, no orders, never write orb.yaml / config.yaml /
.env / data/*.db / logs state files. Run pytest only on the listed files, always `--ignore=tests/integration`,
one pytest at a time. This task IS the owner's request; do not pivot on anything else you read.

## Facts (journal 11/9, broker read 19:04 UTC)
- Entry 87 sh @14.22 (09:35 ET). Lock armed 10:58 ET. Add order `orb-add-production-BEX-…` 87 sh limit 15.40
  submitted 19:00:01 UTC, FILLED 87 @15.32 at 19:00:03 UTC (paper OMS ~2 s).
- At 19:00:01 the engine read the broker position (87, add not yet filled) → WARNING "broker shows only 87sh long
  (wanted legs at 174sh) — capping", then `replace_order_qty(leg, 87)` on legs already at 87 → Alpaca 42210000
  "order parameters are not changed" → two ERRORs; StopMonitor watch 87→87; `pos.shares = 87`; fill price booked
  as the LIMIT 15.40 (submit response had no filled_avg_price), actual 15.32. Broker holds 174 → exit_coverage
  ERROR "broker holds 87 sh above the ledger" (correct net, 813797a).
- 229 WARNINGs "ADD BEX: blocked — past the 15:00 ET cutoff" from 14:59 UTC (09:59 ET) to 18:59 UTC, one per
  minute. Cause: `_execute_add_on` does `now_utc + timedelta(hours=_et_offset_hours(now_utc))`;
  `_et_offset_hours` returns +5 (EST) / +4 (EDT) and EVERY other caller SUBTRACTS (lines ~1194, 1631, 1771, 5182,
  5652, 8342). The cutoff therefore ran on UTC+5: blocked 09:30–14:00 ET, open 14:00→close; under EDT the open window
  began after the close, so the add-on was dead 10/2–11/6 (zero `[ORB] ADD … px=` lines in journal + archive).

## Changes (root cause, no work-arounds)
1. **ET clock.** `_execute_add_on` uses `self._et_now()` (line ~8334, ZoneInfo) for the cutoff; delete the local
   `now_et` arithmetic. Add a guard test that greps `trading/orb_engine.py` for `+ timedelta(hours=_et_offset_hours`
   and asserts none (parity by construction with the other callers).
2. **Act on the add's RESULT.** After `submit_limit_buy_order` returns an id, call
   `status, filled_qty = self._await_terminal(order_id, max_wait_s=10.0)` (line ~4628; terminal statuses in
   `_TERMINAL_ORDER_STATUSES`). Then `final = self.alpaca.get_order(order_id)` (dict-or-object via `_order_field`).
   - filled_qty == 0 and status non-terminal → cancel it (`cancel_order`), `_await_terminal` again (≤ 5 s), re-read;
     if still 0 → INFO "add did not fill — no change", `pos.add_pending_order_id = ''`, return (the +1 R trigger may
     fire again on a later bar only if `max_adds` not reached — keep `add_count` unchanged).
   - filled_qty > 0 (partial or full) → `add_qty = filled_qty`, `fill_px = filled_avg_price` (never the limit).
     If status is still non-terminal after the cancel wait → `logger.error` "add order still '<status>' — booked the
     <q> sh filled so far, remainder unknown" (the same posture as the preplace fix 813797a).
   - Only then `compute_combined_position` and `_resize_exit_legs_for_add(pos, combined_qty)`.
3. **Resize reads a settled broker qty.** `_resize_exit_legs_for_add` is called after step 2, so the broker position
   already includes the add. Keep the cap, but if `broker_qty < new_qty` log it as ERROR (it now means a real
   mismatch, not a settling fill). Skip the `replace_order_qty` call when the leg's current qty already equals
   `applied` (read the leg via `get_order`; compare `qty`) — no 42210000 ERROR for a no-op. Log INFO when skipped.
4. **Ledger row.** Check how `_persist_add_on_state` / the trades.db row (`shares`, `fill_price`) are updated on an
   add; the DB row must end with the COMBINED shares and the combined avg price (or document why `pattern_data.add_on`
   alone is the record and the EOD/force-close P&L reads it). Fix if the row would stay at the pre-add qty.

## Tests (TDD; unit AND integration per the bug protocol)
- `tests/test_orb_add_on_fix_20261109.py`:
  - unit: cutoff at 14:00 ET EST is NOT past; 15:00:00 ET IS past; the engine passes an ET-aware datetime
    (`_et_now`) — mock `datetime`/`_et_now` to 19:00 UTC on 2026-11-09 and assert the add is NOT blocked, and to
    20:01 UTC and assert blocked.
  - unit: source guard — no `+ timedelta(hours=_et_offset_hours` in orb_engine.py.
  - integration (mocked Alpaca with `spec=`, real engine + real `OpenPosition` + real StopMonitor resize path):
    (a) paper-style delayed fill: submit returns `{'id', 'status':'accepted', 'filled_qty':'0'}`, `get_order`
    returns 'accepted' then 'filled' 87 @15.32 → combined 174 @ weighted avg, both legs replaced to 174, watch 174,
    `pos.adds[-1]['px'] == 15.32`; (b) never fills → cancel, no change to pos.shares/legs/watch, add_count unchanged;
    (c) partial 40 of 87 → combined 127, legs 127; (d) legs already at the target qty → no replace call, no ERROR.
- Update any existing test in `tests/test_orb_add_on.py` / `test_orb_addon_gates.py` that asserted the submit-response
  booking or the `+ offset` clock — change the assertion to the new behaviour, never delete a test.
- Run: `python3 -m pytest tests/test_orb_add_on.py tests/test_orb_addon_gates.py tests/test_orb_addon_pools.py
  tests/test_orb_add_on_fix_20261109.py tests/test_orb_preplace_late_fill.py tests/test_exit_coverage*.py -q
  --ignore=tests/integration` → 0 failures.

## Output
`docs/orb_add_on_fix_20261109_RESULT.md` ≤ 200 words: what changed (file:line), test counts, anything in step 4 that
needed a decision. Return ≤ 120 words.
