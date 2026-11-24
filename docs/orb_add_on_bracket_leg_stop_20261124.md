# ORB add-on: a bracket leg's qty can never be replaced — place a dedicated stop for the added shares (2026-11-24)

## The incident (verified read-only: journal 15:26 UTC, ORB paper account GET, logs/session_archive)
* RAMZ 11/24: entry bracket 216 sh filled 14:37:19 @ 23.07 (parent ec8e4d60; legs TP e388daee 216 @ 69.21 `new`,
  SL f25ac28e 216 @ 20.76 `held`, order_class `bracket`). StopMonitor watch stop 22.13 (lock stop).
* 15:26:06 the add-on filled 216 @ 24.0075 (orb-add-production-RAMZ-…). `_resize_exit_legs_for_add(pos, 432)`
  (trading/orb_engine.py ~2905-2950) called `replace_order_qty` on BOTH legs → Alpaca 422
  `{"code":42210000,"message":"qty cannot be changed for advanced orders"}` → two ERRORs
  "resize sl_leg_id/tp_leg_id FAILED … safety leg still covers only the pre-add qty"; `resize_watch_qty(RAMZ, 432)` ran.
* 15:26:17 exit coverage saw NAKED 216/432 → placed stop-sell 380d28ac 216 @ 22.13 (ERROR by design,
  trading/exit_coverage.py ~212). Broker now: 432 long, SL leg 216 @ 20.76 held, coverage stop 216 @ 22.13, TP 216 @ 69.21.
  Position covered; ~11 s of the add qty naked; three ERRORs to Telegram for a structural, expected refusal.
* Same three lines on 2026-11-09 BEX 19:00 and 2026-11-13 DAMD 14:45 (session_archive). Alpaca never allows a qty change on
  a leg of an advanced order (bracket / oco / oto); the replace path is dead code for every bracket position, i.e. for every
  ORB add-on. The 11/13 D2 fix made coverage's stop count and register; D4 (lost leg ids) is still open.
* Safe to add a second stop: `_execute_stop_exit` bulk-cancels ALL open orders of the symbol before its exit sell
  (stop_monitor.py ~4248 docstring), and `exit_coverage.covered_qty` counts every open sell — a coverage stop already lives
  beside the bracket today.

## The fix (TDD: failing tests first). trading/orb_engine.py only; no selection, sizing, stop-level or StopMonitor change.
1. Replace `_leg_qty(leg_id)` by `_leg_info(leg_id) -> Optional[Dict]` returning `{'qty': int, 'order_class': str,
   'stop_price': float|None}` from ONE `get_order` (keep `_leg_qty` as a thin wrapper if other code/tests call it — grep).
2. In `_resize_exit_legs_for_add`, per leg, after the "already covers applied" skip: if `order_class.lower()` is in
   `ADVANCED_ORDER_CLASSES = frozenset({'bracket', 'oco', 'oto'})` → NEVER call `replace_order_qty`:
   * **SL leg**, `delta = applied - leg_qty > 0`: place ONE plain stop-sell for `delta` shares via
     `self.alpaca.submit_stop_sell_order(...)` (grep its signature in data_sources/alpaca_client.py; exit coverage calls it
     through `place_stop_fn` in stop_monitor.py ~1053 — same call), `client_order_id=f"orb-addstop-{sym}-{int(time.time()*1000)}"[:48]`,
     stop price = the StopMonitor watch's current `stop_price` when `self.stop_monitor.get_watch_snapshot(sym)` gives one > 0
     (the lock stop, the level coverage would use), else the leg's own `stop_price`. Log INFO
     `[ORB] ADD {sym}: bracket SL leg {id[:8]} keeps {leg_qty}sh (Alpaca 42210000: an advanced-order leg's qty cannot change)
     — add stop {new_id[:8]} placed for {delta}sh @ ${px:.2f}`. Append `new_id` to the NEW `OpenPosition.add_stop_leg_ids:
     List[str]` (field default list; persisted/rehydrated in `_persist_add_on_state` and wherever `adds` is rehydrated — grep
     `add_count` in the rehydrate path). Placement failure → ONE ERROR
     `[ORB] ADD {sym}: add stop for {delta}sh could not be placed ({e}) — add shares NAKED until exit coverage repairs`.
   * **TP leg**: no replace; INFO `[ORB] ADD {sym}: bracket TP leg {id[:8]} keeps {leg_qty}sh — the StopMonitor exit cancels
     every open order of the symbol and sells the combined qty`.
   * `delta < 0` (the ADD-BE shrink path calls this with `remaining`): cancel the position's `add_stop_leg_ids` first
     (`self.alpaca.cancel_order`, each logged INFO, ids removed from the list; a failed cancel = WARNING, id kept); then if
     the bracket leg still exceeds `applied`, INFO "leg exceeds the remaining qty — the exit path's symbol-wide cancel handles it"
     (no replace attempt on an advanced leg).
3. A NON-advanced leg (a plain stop that became `sl_leg_id` through exit coverage, or a simple order in tests) keeps today's
   `replace_order_qty` path unchanged.
4. The `_exit_qty_guard` cap, the `broker_qty < new_qty` ERROR and `resize_watch_qty` stay exactly as they are.

## Tests — `tests/test_orb_add_on_bracket_leg_stop_20261124.py` (fixtures from tests/test_orb_add_on_fix_20261109.py and
tests/test_orb_add_on.py — import/reuse, do not copy; mocks `spec=AlpacaClient`)
1. RAMZ replay: pos 216 sh, sl_leg f25ac28e (get_order → qty 216, order_class 'bracket', stop_price 20.76), tp_leg e388daee
   (216, 'bracket', limit 69.21), broker qty 432, watch snapshot stop_price 22.13 → `_resize_exit_legs_for_add(pos, 432)`
   returns 432; `replace_order_qty` NOT called; `submit_stop_sell_order` called once with (RAMZ, 216, 22.13); `pos.add_stop_leg_ids
   == [<id>]`; caplog has NO ERROR record and the two INFO lines.
2. No watch snapshot / stop_price 0 → the leg's 20.76 is used.
3. `submit_stop_sell_order` raises → exactly one ERROR containing "NAKED until exit coverage"; return value still 432 (the watch
   must track the real qty), `add_stop_leg_ids` unchanged.
4. Simple (order_class 'simple') sl leg → `replace_order_qty` called as today (pinned); new id adopted.
5. Leg already at `applied` → skip INFO, no replace, no stop (pinned from the 11/09 fix).
6. ADD-BE shrink: pos with `add_stop_leg_ids=['x']`, `_resize_exit_legs_for_add(pos, 216)` → `cancel_order('x')` called, list
   empty, no replace on the bracket leg, no ERROR.
7. Persist round-trip: `_persist_add_on_state` writes `add_stop_leg_ids`; the rehydrate path reads it back (use the real temp
   `Database` the 11/09 test file uses).
8. Integration (real `StopMonitor(api_key='k', api_secret='s', alpaca_client=<mock spec=AlpacaClient>)` + real engine, as
   tests/test_orb_scaleout_coverage_conflict_20261113.py builds them): after test 1's resize, `exit_coverage.check` with the
   mocked open sells (bracket SL 216 + add stop 216) reports `naked_qty == 0` for 432 sh — no coverage ERROR would fire.

## Run (these files only; the main session runs the full suite post-close — NEVER start it yourself)
`python3 -m pytest tests/test_orb_add_on_bracket_leg_stop_20261124.py tests/test_orb_add_on.py tests/test_orb_add_on_fix_20261109.py
tests/test_orb_addon_gates.py tests/test_orb_addon_pools.py tests/test_orb_scaleout_coverage_conflict_20261113.py -q -p no:cacheprovider
--ignore=tests/integration` → all green.

## Rules
Sonnet, ≤ 60 calls. The LIVE service is running: no `git commit`/`git stash`, no service restart, no orders, no writes to
`data/*.db`, `logs/*`, `orb.yaml`, `config.yaml`, `.env`. Files you may edit: `trading/orb_engine.py` (`_leg_info`,
`_resize_exit_legs_for_add`, the `OpenPosition` field, the persist/rehydrate of that field only), the new test file, and a
"Result" section appended to THIS file. Docstrings on every new function; every fallback logs WARNING or ERROR. Return ≤ 150
words: files changed, test counts (before/after), the `submit_stop_sell_order` signature you used, anything you could not do,
verified facts separated from guesses.

## Result (2026-11-24)
* Files: `trading/orb_engine.py` (`ADVANCED_ORDER_CLASSES`, `_add_stop_ids_from_pdata`, `OpenPosition.add_stop_leg_ids`,
  `_leg_info` (+ `_leg_qty` thin wrapper), `_handle_advanced_leg`, `_cancel_add_stops`, `_resize_exit_legs_for_add`,
  persist + rehydrate of the field), new `tests/test_orb_add_on_bracket_leg_stop_20261124.py` (8 tests), this section.
* Tests: new file 0 -> 8 (failed first, 8/8 red); Run set 97 -> 105 passed, 0 failed.
* Signature used: `AlpacaClient.submit_stop_sell_order(symbol, qty, stop_price)` — it has NO `client_order_id`
  parameter, so the spec's `orb-addstop-...` id was NOT applied (verified by reading the signature).
* Not done: no live/paper probe, no full suite, no commit (per rules). Shrink cancel is triggered when `applied < pos.shares`.
* Verified: tests green. Guess: real Alpaca `get_order` returns `order_class` as enum/str (handled via `.value`/rsplit).

## Review (main session, 2026-11-24 16:0x UTC)
Diff read: `_leg_info` (one GET, enum-or-string `order_class`), `_handle_advanced_leg` (SL delta → one plain stop at the watch
stop, else the leg's; TP left alone; shrink → INFO), `_cancel_add_stops` on any shrink, `add_stop_leg_ids` persisted and
rehydrated. No change to the exit-qty cap, the broker-qty ERROR or `resize_watch_qty`. `submit_stop_sell_order` carries no
client_order_id (same as exit coverage's stop) — accepted. Touched set 115 green in the main session. Full suite + commit at
the post-close one-shot; live at the next 12:30 UTC boot. Pre-existing and untouched: a scale-out while a second stop holds
shares (coverage stop today, add stop tomorrow) still depends on the 11/13 D2 "shrink" path, and D4 stays open.

## Follow-up found at the post-close check (2026-11-24 21:2x UTC): FC DB SYNC priced add-on rows off the base fill
* All four add-on positions were force-closed 15:45 ET at market (RAMZ 432 @ 24.02, NVDQ 1022 @ 9.93, MUD 776 @ 13.1782,
  MUZ 302 @ 11.51; every bracket leg and coverage stop cancelled cleanly). `_sync_db_after_fc` booked each row's P&L from
  `fill_price` (the BASE fill — D3 11/13 keeps it so for rehydrate) instead of the blended `pattern_data.add_on.add_avg_px`:
  rows 882-885 carry +411.70 / +214.62 / +243.45 / +145.72 = +$1,015.49 where the blended basis gives +208.58 / −21.93 /
  +89.76 / +51.72 = +$328.13 (overstated by $687.36). The StopMonitor exit path (`_handle_exit_event` ~7196) already uses
  `combined_avg_price`; only the FC sync path was wrong. The DB rows are left as they are — a four-row correction is the
  owner's call (the ORB ramp gate and the EOD report read these numbers).
* Fix (main session, inline): `ORBEngine._add_on_pdata(row)` reads the add_on block (JSON or dict, WARNING + {} when
  unreadable); the FC sync uses `add_avg_px` when present (else `fill_price`) and composes `add_be_pnl` once.
  Tests `tests/test_orb_fc_sync_addon_basis_20261124.py` (6: RAMZ replay, add-be composition, no-add row unchanged,
  unreadable pattern_data, scaled row pinned, helper). Touched set 91 green; full suite before the commit.
