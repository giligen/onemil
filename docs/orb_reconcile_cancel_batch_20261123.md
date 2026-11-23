# ORB preplace reconcile: not-in-final cancels are awaited one by one (5 x 15 s), the 09:35 tick stalled 92 s and paged "Engine tick TIMEOUT" (2026-11-23)

## The incident (verified read-only: journal 14:35-14:37 UTC, data/trades.db rows 861-865, ORB paper account GET)
* 09:35:24 ET `_reconcile_preplaced` (trading/orb_engine.py ~4696): final top-8 = AI, CGAU, MTA, NFXL, QUBT, BOTZ, ABTC, EGO;
  preplaced = BMNU, DRAL, RGTI, AEVA, RDW — none in the final set. The not-in-final branch (~4790) called
  `_cancel_preplaced_order` per name; each one `_cancel_symbol_open_orders` then `_await_terminal(order_id)` with the FULL
  15 s budget. The paper OMS held every cancel in `pending_cancel` for 15-20 s today, so the waits ran back to back:
  WARNING "not terminal after 15.0s" at 14:35:43 (BMNU, status new), 14:36:02 (DRAL), 14:36:21 (RGTI), 14:36:39 (AEVA);
  PARITY line at 14:36:56 (`n_cancelled=5 n_added=0`). 92 s for a reconcile the REPLACE pre-pass (~4744-4761) bounds to
  15 s total. The scanner's `_orb_tick` (scanner/realtime_scanner.py 814) waited on the engine lock behind it and
  `ENGINE_TICK_TIMEOUT_S` fired: ERROR "Engine tick TIMEOUT (>50s) — one engine stalled" (14:36:03) -> Telegram.
* Outcome: all five provisional orders are `canceled`, filled 0 (broker canceled_at 14:35:46 / 14:36:05 / 14:36:22 ...), rows
  861-865 `preplace_canceled / preplace_reconcile`, no positions, no open orders. No money lost: the eight final names were
  all vetoed anyway (PDR veto x7, G1 veto QUBT) — `n_added=0` by rule. The cost is the false page and 90 s of blocked tick.
* Precedent: 2026-10-27 CRDU (docs in `_await_terminal`'s docstring) introduced the shared-deadline pre-pass for REPLACE
  candidates only; the not-in-final branch kept the serial wait. "not terminal after" lines: 10/21-10/23 (6), 11/11, 11/13, 11/16.

## The fix (TDD: failing tests first). No selection, sizing, veto or order-content change; the row marking, late-fill handling
## and all counters stay exactly as today.
### Part 1 — one shared cancel pass for every provisional order that must go (trading/orb_engine.py)
In the pre-pass, ALSO fire `_cancel_symbol_open_orders(sym)` and append `(sym, order_id)` to `_to_wait` for every preplaced
symbol that will hit the not-in-final branch: `st.get('submitted')`, `pos is not None and pos.order_id` (not already filled),
and (`sym not in final_top_syms` or `final_rh is None`). Compute `final_rh` the same way the main loop does. Then the single
`_await_terminal` loop with the shared `_RECONCILE_WAIT_S` budget covers them too (`late_results[sym]`).
`_cancel_preplaced_order(self, sym, pos, awaited: Optional[Tuple[str, int]] = None) -> bool`: when `awaited` is given, use it
as `(status, filled_qty)` and do NOT call `_cancel_symbol_open_orders` / `_await_terminal` again; when None (any other caller),
behave exactly as today. The not-in-final branch passes `late_results.get(sym)`. Everything after (filled_qty > 0 -> False,
row marking `preplace_reconcile`, `del open_positions`, the dropped/vetoed sets, `n_cancelled`) unchanged.
Log ONE INFO after the shared wait: `[ORB PREPLACE] cancel pass: N orders, M terminal within Xs, K still pending`
(K > 0 -> WARNING instead).
### Part 2 — the scanner never stacks ORB ticks (scanner/realtime_scanner.py)
`self._orb_tick_lock = threading.Lock()` (init where the scanner builds its engines); `_orb_tick` acquires it non-blocking:
when the previous tick still holds it, log WARNING `ORB tick skipped — previous tick still running (started Ns ago)` and
return; otherwise run the body in try/finally and release. Record the start time for the message.

## Tests — `tests/test_orb_reconcile_cancel_batch_20261123.py` (fixtures from `tests/test_orb_preplace_late_fill.py` —
`eng`, `_no_sleep`; the reconcile harness from `tests/test_orb_preplace_cancel_db_status_20261020.py` — reuse, do not copy)
1. Replay: 5 preplaced names, none in the final top-8, `get_order` returns `pending_cancel` for 3 polls then `canceled` ->
   `_cancel_symbol_open_orders` called once per name BEFORE any `get_order`; total `time.sleep` calls <= one 15-s budget
   (count the patched sleeps: < 5 x n_polls); all 5 rows marked `preplace_canceled/preplace_reconcile`; `n_cancelled == 5`;
   one INFO containing "cancel pass: 5 orders".
2. One of the five fills during the wait (filled_qty 100) -> that one returns False from `_cancel_preplaced_order`, row NOT
   marked, the other four marked; `n_cancelled == 4` (pinned from `test_d_cancel_path_late_partial_returns_false_row_unmarked`).
3. Never terminal -> after the budget, the row is still marked canceled with filled_qty 0 (today's behaviour), one WARNING
   "still pending".
4. A REPLACE candidate and a not-in-final name together -> both are in the same `_to_wait`; `_await_terminal` is called once
   per order and the summed `max_wait_s` passed never exceeds `_RECONCILE_WAIT_S` + 0.5.
5. `_cancel_preplaced_order(sym, pos)` with `awaited=None` still cancels and awaits itself (pinned for any other caller).
6. Scanner: a `RealtimeScanner`-like stub (or the real class with engines mocked as the existing scanner tests do — look at
   `tests/test_realtime_scanner*.py`) whose `orb_engine.check_exits` blocks on an Event -> a second `_orb_tick` returns at
   once with the WARNING "tick skipped"; after the Event is set, a third tick runs normally.
7. Integration: real `Database` temp file + the reconcile harness -> after the reconcile, `get_trade_by_order_id` shows
   `preplace_canceled` for every provisional id (as `tests/test_orb_preplace_cancel_db_status_20261020.py` does).

## Run (these files only; the main session runs the full suite post-close — NEVER start it yourself)
`python3 -m pytest tests/test_orb_reconcile_cancel_batch_20261123.py tests/test_orb_preplace_late_fill.py
tests/test_orb_preplace_cancel_db_status_20261020.py tests/test_orb_preplace_replaced_row_20261120.py
tests/test_orb_preplace_lock_wait_20261017.py tests/test_orb_preplace_budget_20261006.py tests/test_realtime_scanner*.py
-q -p no:cacheprovider --ignore=tests/integration` -> all green (skip a glob that matches nothing).

## Rules
Sonnet, <= 60 calls. The LIVE service is running during trading hours: no `git commit`/`git stash`, no service restart, no orders,
no writes to `data/*.db`, `logs/*`, `orb.yaml`, `config.yaml`, `.env`. Files you may edit: `trading/orb_engine.py` (the pre-pass,
the not-in-final branch, `_cancel_preplaced_order` only), `scanner/realtime_scanner.py` (`_orb_tick` + the lock init only), the
new test file, and a "Result" section appended to THIS file. Do not change `_RECONCILE_WAIT_S`, `_await_terminal`,
`ENGINE_TICK_TIMEOUT_S`, vetoes, sizing or the REPLACED branch. Docstrings on every new/changed function; every fallback logs
WARNING or ERROR. Return <= 150 words: files changed, test counts red-before / green-after, anything you could not do,
verified facts separated from guesses.

## Result (2026-11-23)
* Implemented Part 1 (shared cancel pass in the pre-pass; `_cancel_preplaced_order(..., awaited=None)`; one `cancel pass: N orders, M terminal within Xs, K still pending` line, WARNING when K > 0) and Part 2 (`_orb_tick_lock` non-blocking in `_orb_tick`, WARNING `ORB tick skipped`, lock + `_orb_tick_started_at` init in `RealtimeScanner.__init__`).
* New `tests/test_orb_reconcile_cancel_batch_20261123.py`: 7 tests. Red before the change: 5 failed / 2 passed (tests 2 and 7 pin unchanged behaviour). Spec run command: 49 passed after.
* Test 4 uses a fake monotonic clock (each await burns its full wait) because the summed per-call `max_wait_s` is a remaining-budget, not a slice; the assertion is that the budget shrinks across calls.
* Not run: full suite, boot rehearsal. Live service untouched; nothing committed.

## Review (main session, 2026-11-23 14:5x UTC)
Diff read: the not-in-final names join `_to_wait` with the same predicate the branch uses (disjoint from the REPLACE set by
construction), `_cancel_preplaced_order(awaited=...)` skips the second cancel/await, one cancel-pass summary line, and the
scanner's non-blocking `_orb_tick_lock`. Touched files 49 green in the main session. Commit after the post-close full suite
(one-shot 21:16 UTC); live at Tuesday's 12:30 UTC boot. Today's five cancels and rows 861-865 were already correct.
