# HOD: the 60-s broker reconcile adopts our OWN resting-order fill before the fill poll registers it (2026-11-20)

## The incident (verified read-only: journal, data/trades.db row 853, HOD paper account GET)
* 14:37:01 UTC HOD rested the NBIZ entry (buy stop-limit 18 sh, stop 75.21 / limit 75.31, order eb5ba207,
  coid `hod-rest-NBIZ-11-20-…`). It FILLED 14:41:07 @ 75.30.
* 14:41:08 — ONE second later — `_reconcile_positions_to_broker` (every `BROKER_RECONCILE_S` 60 s, trading/hod_break_engine.py
  ~1761) → `_adopt_unregistered_positions_on_boot(periodic=True)` (~1529) saw a broker long of 18 NBIZ with no registry
  Position. `ours_today` was true because `cand.live_order is not None` (our resting order, still tracked) → it ADOPTED
  the position: DB row 853 (`order_status='filled'`, `adopted_on_boot`), `self.positions['NBIZ']`, StopMonitor watch
  with `tp_leg_id='' , sl_leg_id=''` and the WARNING "ADOPTED 18 sh unregistered broker position on boot … never
  registered by a prior process" (wrong on every word: not boot, not a prior process). `cand.live_order` stayed set and
  `cand.live_filled` stayed False.
* 14:41:48 the StopMonitor exit-coverage cycle saw a 40-s-old watch with no stop leg → NAKED 18/18 → placed stop-sell
  58096ff1 @ 72.50 (ERROR by design, trading/exit_coverage.py ~212).
* 14:42:07 the engine finally processed the fill (through `_cancel_live_order` → `_on_live_fill`, ~1899: "already carries a
  fill … registering") → `_place_safety_net` → OCO and stop-only both rejected `40310000 insufficient qty available
  (held_for_orders 18, related_orders [58096ff1])` — the coverage stop holds the shares → 5 ERROR lines + Telegram
  "ERROR SL NBIZ — UNPROTECTED POSITION 18 sh" (FALSE: 58096ff1 protected it) → the watch was re-added without legs.
* 14:49:41 the stop triggered; cancel-and-place limit; partial 6/18 → emergency market 12 → 18 @ 72.00 blended, −$59.34
  (−1.18 R). The 14:49 PARTIAL ERRORs are the normal emergency path — not part of this fix.
* Not new: the same "ADOPTED … on boot" mid-session on 2026-11-17 15:30:14 (AXTQ 37 sh) and 2026-10-22 16:04:33 (SNDQ 108 sh)
  (logs/session_archive). Precedents: 2026-10-02 VIRT (a fill after a cancel/rail — the case periodic adoption exists for:
  there `cand.live_order` is already None) and docs/hod_live_incident_20260929.md.
* Why the poll lost the race: `_poll_live_fills` reads the order-stream snapshot (~2005) and its REST fallback runs at most
  every 5 s (~2042); both ran in the tick that started at 14:41:07 before the fill was visible, and the reconcile in the
  SAME tick (after them, ~619) saw the broker position. Any fill landing inside that window is adopted instead of registered.

## The fix (TDD: failing tests first). trading/hod_break_engine.py + one StopMonitor accessor. No strategy, sizing, cap or
constant change other than the two new ones named below.
### Part 1 — a broker long whose resting order of ours is still tracked is THAT order's fill: register it, never adopt it
In `_adopt_unregistered_positions_on_boot`, periodic mode only (`periodic=True`), BEFORE the `bound`/`_our_qty` logic, when
`cand is not None and cand.live_order is not None and not cand.live_filled`:
1. `st = self.alpaca.get_order(lo['order_id'])` (one GET; the REST fallback does the same call). If
   `int(st.get('filled_qty') or 0) > int(lo.get('booked_qty') or 0)`: log WARNING
   `{tag} {sym}: broker reconcile caught the fill of resting order {order_id} ({fq} sh) before the fill poll — registering
   through _on_live_fill, not adopting` and call `self._on_live_fill(cand, lo, st, status)` (the ONE registration path:
   safety net, DB row, Position, watch WITH leg ids, parity + dry ledger rows). `continue`.
2. If the GET shows no new fill (status new/accepted/partially filled below booked), or the GET raises: do not adopt yet.
   Record `self._adopt_deferred_since.setdefault(sym, time.time())` (new dict, init in `__init__`, cleared for `sym` on
   `_on_live_fill` and on adoption) and log a throttled INFO (`_warn_throttled` is WARNING-only — add an `info` path or
   use `logger.info` guarded by the same 60-s key logic) `{sym}: broker long {qty} sh while resting order {order_id} is
   still tracked ({status}) — waiting for the fill poll ({age:.0f}s)`. `continue`.
3. Backstop: when `time.time() - self._adopt_deferred_since[sym] > ADOPT_PENDING_ORDER_GRACE_S` (new class constant,
   180.0 s = three reconcile passes) fall through to today's adoption, with the WARNING text changed to say
   `… adopted after {age:.0f}s with resting order {order_id} still unresolved` so the log never again says "on boot" for a
   periodic adoption. Also change the periodic adoption's normal WARNING/Telegram wording from "on boot … by a prior
   process" to "on reconcile" when `periodic` is True (the boot wording stays for boot).
Boot mode (`periodic=False`) is unchanged: `_reconcile_live_orders_on_boot` runs first there and the existing tests pin it.
### Part 2 — a safety net that cannot be placed because an exit-coverage stop already holds the shares is not "UNPROTECTED"
* `StopMonitor.get_watch_snapshot` (trading/stop_monitor.py ~2900): add `'sl_leg_id'` and `'tp_leg_id'` to the returned dict.
* `_place_safety_net` (~2265): after the second failure and before the ERROR + `_notify`, when the error text contains
  `40310000` or `insufficient qty` and `self.stop_monitor is not None`: `snap = self.stop_monitor.get_watch_snapshot(sym)`;
  `cov = str((snap or {}).get('sl_leg_id') or '')`; if `cov`, `st = self.alpaca.get_order(cov)` and if its status is one of
  `new/accepted/pending_new/held` and its type is `stop`/`stop_limit`: log WARNING `{sym}: safety net not placed — the
  shares are held by exit-coverage stop {cov[:8]} @ {stop_price}; adopting it as the SL leg` and `return ('', cov)`
  (no ERROR, no "UNPROTECTED" Telegram). Any GET failure or a non-live/non-stop order → today's ERROR path unchanged.
### Tests — `tests/test_hod_reconcile_adopt_race_20261120.py`
Fixtures: `real_db`, `mock_alpaca`, `mock_sm` and `_seed_row` from `tests/test_hod_break_engine_boot_defects.py`, `hod_cfg`
from `tests.test_hod_break_engine import cfg`, `set_our_fills` from `tests/hod_fills_helper.py`. Build `cand.live_order`
with the real keys the engine uses (grep `'coid':` in trading/hod_break_engine.py where the dict is created: order_id, coid,
qty, stop, limit, level, trigger, booked_qty, tp_leg_id, sl_leg_id, trade_id). Mock `submit_oco_sell_order` to return a
parent id with one stop leg so `_on_live_fill`'s safety net succeeds.
1. NBIZ replay: broker long 18 @ 75.30, `get_order(eb5ba207)` → `{'status':'filled','filled_qty':18,'filled_avg_price':75.30}`
   → `eng._adopt_unregistered_positions_on_boot(periodic=True)` → `cand.live_filled is True`, `cand.live_order is None`,
   `eng.positions['NBIZ'].sl_leg_id` is the OCO stop leg id, exactly ONE DB row (status filled, shares 18, fill_price 75.30),
   `mock_sm.add_watch` called once with a non-empty `sl_leg_id`, a WARNING containing "registering through _on_live_fill",
   NO record containing "ADOPTED".
2. Same but `get_order` → `{'status':'new','filled_qty':0}` → nothing adopted (no DB row, no Position, no add_watch), an
   INFO "waiting for the fill poll"; then set `eng._adopt_deferred_since['NBIZ'] -= 200` and run again → adopted once (ONE row),
   WARNING contains "adopted after" and "still unresolved", not "on boot".
3. `get_order` raises → same deferral as test 2 (no adoption on the first pass).
4. VIRT-style pin: `cand.live_order is None`, `sym in eng._live_armed_stops`, broker long → adopted as today, WARNING says
   "on reconcile" (not "on boot").
5. Boot mode pin: `periodic=False` with `cand.live_order` set → adopts as today (mirror
   `test_adopts_as_filled_and_links_trade_id_into_live_order`; the existing test must still pass).
6. Idempotence: after test 1, a `_poll_live_fills_rest_fallback` pass returning the same filled status books nothing new
   (one row, one Position, add_watch not called again).
7. Part 2: `submit_oco_sell_order` and `submit_stop_sell_order` both raise `AlpacaAPIError('… 40310000 insufficient qty …')`,
   `mock_sm.get_watch_snapshot.return_value = {'sl_leg_id': '58096ff1', 'tp_leg_id': ''}`, `get_order('58096ff1')` →
   `{'status':'accepted','type':'stop','stop_price':72.5}` → `_place_safety_net` returns `('', '58096ff1')`, one WARNING with
   "exit-coverage stop", NO ERROR record, `_notify` NOT called with "UNPROTECTED". Variant: `get_order` → status `canceled`
   → today's ERROR + notify (pinned). Variant: no watch (`None`) → ERROR path.
8. Snapshot: a real `StopMonitor(api_key='k', api_secret='s', alpaca_client=MagicMock(spec=AlpacaClient))` with one
   `add_watch(... sl_leg_id='sl-1', tp_leg_id='tp-1')` → `get_watch_snapshot` carries both ids.

## Run (these files only; the main session runs the full suite post-close — NEVER start it yourself)
`python3 -m pytest tests/test_hod_reconcile_adopt_race_20261120.py tests/test_hod_break_engine_boot_defects.py
tests/test_hod_killrail_restart_qty.py tests/test_hod_review_b_fixes.py tests/test_hod_oco_persistence.py
tests/test_hod_live_resting.py tests/test_hod_break_engine.py -q -p no:cacheprovider --ignore=tests/integration` → all green.

## Rules
Sonnet, ≤ 60 calls. The LIVE service is running: no `git commit`/`git stash`, no service restart, no orders, no writes to
`data/*.db`, `logs/*`, `config.yaml`, `orb.yaml`, `.env`. Files you may edit: `trading/hod_break_engine.py`,
`trading/stop_monitor.py` (the snapshot dict only), the new test file, and a "Result" section appended to THIS file. Do not
change `BROKER_RECONCILE_S`, `REST_FALLBACK_INTERVAL_S`, caps, kill rails, sizing or the boot path. Docstrings on every new
function; every fallback logs WARNING or ERROR. Return ≤ 150 words: files changed, test counts (before/after), anything
you could not do, verified facts separated from guesses.

## Result (2026-11-20)
Implemented Part 1 (`_defer_to_tracked_resting_order`, `ADOPT_PENDING_ORDER_GRACE_S = 180`, `_adopt_deferred_since`, periodic wording "on reconcile" / "adopted after Ns ... still unresolved") and Part 2 (`_adopt_exit_coverage_stop` in `_place_safety_net`; `get_watch_snapshot` carries `sl_leg_id`/`tp_leg_id`). New `tests/test_hod_reconcile_adopt_race_20261120.py` = 10 tests (8 spec items, item 7 with 2 variants). Run-section files: 200 passed. Full suite not run (main session).

## Review (main session, 2026-11-20 15:3x UTC)
Diff read: deferral helper `_defer_to_tracked_resting_order` (GET once → `_on_live_fill` on a new fill; INFO deferral; 180-s
backstop) and `_adopt_exit_coverage_stop` (40310000 + live coverage stop on the watch → adopt as SL leg) match the spec; the
periodic WARNING no longer says "on boot". Touched HOD/ORB/StopMonitor files green in the main session. Commit after the
post-close full suite (one-shot 21:14 UTC); live at Monday's 12:30 UTC boot. Row 853 (NBIZ −$59.34 stop-out) is correct.
