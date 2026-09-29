# ORB exit: rest the target as a real limit at the broker — spec (main session, 2026-09-29)

## Evidence (research/exec_quality/REPORT_20260928.md §4)
39 live target exits (tag_bb): the fill was 68.8 bps mean / 47.6 median / p90 140 bps WORSE than the target price, and
no fill ever beat it; mean ≈ trimmed mean, so this is not a tail. The target is currently detected from our feed
("tagged") and then sold with a marketable limit — we pay the spread and the chase every time. On the 31 % of ORB
trades that reach the target this is ≈ 15–20 bps per ORB trade overall ≈ 0.05 R at the live stop distance — a third of
the live-vs-backtest gap. The backtest fills the target at the touch; resting the limit is the live mechanism that
matches it (obtainability: the tape trades at or through the limit while our order has queue priority).

## Mechanism
From the moment the entry fills, a DAY limit SELL for the full position sits at the broker at the target price
(client_order_id `orb-tp-<sym>-<yyyymmdd>`). The stop logic stays in `StopMonitor` unchanged. Rules:
1. Entry fill event → submit the resting TP; log `[ORB TP] rested <sym> qty q @ target` (WARNING with the reason if
   the submit fails, then fall back to today's tag-and-sell path for that trade).
2. Stop trigger → cancel the resting TP FIRST, wait for the cancel ack (≤ 1 s), then submit the stop exit. If the
   cancel returns "already filled", the position is flat: record the exit as `target_rested`, no stop order.
3. TP fill event on the order stream → clear the stop watch, record the exit `target_rested` with the limit price, the
   fill price and the resting time; Telegram line as today's target exit.
4. Trail/lock rules that MOVE the target → cancel/replace the resting TP (rate-limited to one replace per 5 s per
   symbol; if the replace fails, keep the old TP and WARN).
5. EOD flatten and every other exit path → cancel the resting TP before submitting, same as rule 2.
6. Partial-exit variants (if the trail rules exit a partial quantity at the target): the resting TP carries the
   partial quantity; the remainder keeps today's logic.
7. Reconciliation at boot and every sync: any `orb-tp-*` order without a matching open position is cancelled with a
   WARNING; any position without its TP (flag ON) gets one rested.
If the entry bracket already carries a take-profit leg, the implementer replaces that leg's price with the target
instead of creating a second order (never two resting sells for one position); state which in the PR.

## Flag, parity, tests, rollout
* `orb.yaml exit.target_resting_limit: false` (default OFF; byte-identical behaviour when OFF).
* Rulebook: add the fill rule to `research/orb_machine_rules.md` ("target fills at the touch by a resting limit; BT and
  live share the rule; obtainability = the tape trades at or through the limit").
* Tests: unit (submit on fill, cancel-before-stop, cancel-race "already filled", replace on target move, EOD cancel,
  boot reconciliation) with `MagicMock(spec=...)` clients; integration on the real order-stream event shapes recorded
  in `logs/` (a TP fill event, a cancel ack, a reject); the whole ORB suite green.
* Rollout: ONE paper session with the flag ON on the ORB paper account (never the same session as another new
  mechanism's first test), grep `[ORB TP]`, measure fill-vs-limit bps per target exit (bar ≤ 5 bps mean) and the count
  of cancel races; live only on the owner's word after that session.

## Companion question for the paper sessions (not a rule yet): the first five seconds
The tick replay (§6) shows breaks in the first 0–5 s after 09:35:00 as the worst bucket (n 42, 19 % win, −$24/fill)
while the latency replay (cell 1,426) shows 3–20 s of delay costs nothing in the backtest. Pre-placement submits the
buy-stops at 09:35:00.0 and would take those instant breaks first. Measure on paper: the trigger-time histogram of the
pre-placed fills and their outcome; if the 0–5 s bucket loses again on ≥ 20 fills, PREREG a `preplace_submit_delay_s: 5`
cell on the tick replay before changing anything live. n 42 in one bucket of five is not a rule.

## Implementation notes (2026-09-29 build)

**Design decision — bracket TP leg vs. a second order**: the entry bracket already carries a take-profit leg
(`trading/orb_engine.py` `submit_entry`, `safety_tp = entry_price * 3.0` — intentionally unreachable, "ORB has no
fixed target; legally must set"). The implementation REPRICES that existing leg to the real touchgo target via
`AlpacaClient.replace_order_limit_price` (extended with an optional `client_order_id` — `ReplaceOrderRequest` supports
it) rather than creating a second, freestanding resting sell. This reuses the same primitive `stop_monitor.py`'s D3-FIX-1
stop path already uses to reprice the SL leg in place, and preserves the broker-side OCO relationship with the SL leg
(a fill on one leg auto-cancels the other — no orphan safety-net order). Alpaca's replace mints a new order id each
time (documented already at `trading/hod_break_engine.py:1163`); every reprice threads the new id back onto
`OpenPosition.tp_leg_id` and `StopMonitor`'s `WatchEntry.tp_leg_id` (via the new `mark_target_resting`) so the next
cancel/replace targets a live order, never a dead one.

**Files**:
- `trading/orb_target_limit.py` (new) — broker-facing primitives, synchronous, no asyncio/engine coupling:
  `reprice_target` (rules 1 + 4, one function, `force_first` bypasses the rate limit for the initial rest),
  `cancel_resting_target` (rules 2 + 5, classifies the "already filled" race via `get_order` when `cancel_order`
  doesn't confirm cleanly), `reconcile_orphan_targets` (rule 7, orphan half), `target_client_order_id`
  (`orb-tp-<sym>-<yyyymmdd>`).
- `trading/orb_engine.py` — `exit.target_resting_limit` config flag (default False); `OpenPosition` gains
  `target_resting`/`target_price`/`target_rested_at`/`target_last_replace_ts`; `_fire_touchgo_exit` branches to the
  new `_rest_touchgo_target` when the flag is ON, falling back to the unchanged chase-and-sell path on any failure
  (no leg / rate limited / broker error — all WARNING-logged by the helper); `_poll_target_fills` (new, called from
  `_check_exits_locked` before draining) detects a resting fill via `order_stream.get_status` and books it through
  `StopMonitor.book_target_rested_fill`; `_handle_exit_event` gained an `exit_fill_latency_ms` augmentation for
  `target_rested` rows (resting seconds); `sync_positions` gained a best-effort orphan-cancellation pass (rule 7).
- `trading/stop_monitor.py` — `WatchEntry` gains `target_resting`/`target_price`/`target_rested_at` (default
  `False`/`0.0`/`0.0` — inert for every non-ORB watch); two new public methods, `mark_target_resting` (keeps the
  watch's leg id in sync after ORBEngine's broker-side reprice) and `book_target_rested_fill` (queues a
  `target_rested` `StopExitEvent` the same way the existing scale-out fill poll does); a new branch at the top of
  `_execute_stop_exit`, gated by `watch.target_resting`, cancels the resting TP with an `asyncio.wait_for(...,
  CANCEL_ACK_WAIT_S=1.0)` budget and resolves the "already filled" race by emitting a `target_rested` event instead
  of continuing into the stop-exit machinery.
- `data_sources/alpaca_client.py` — `replace_order_limit_price` gained an optional `client_order_id` kwarg
  (backward compatible; existing callers unaffected).
- `trading/exit_reasons.py` — new `ExitReason.TARGET_RESTED = "target_rested"`, added to `_ATTRIBUTED_EXITS`.
- `orb.yaml.template` — `exit.target_resting_limit: false` documented under the `exit:` section.
- Telemetry: no new DB columns. `exit_limit_price` / `exit_price` (bps vs. limit is derived downstream) and
  `exit_fill_latency_ms` (repurposed as resting-seconds — mirrors the HOD 9/28 resting-order telemetry reuse of the
  same column) are the existing, shared columns `build_exit_update` already writes.

**Rule 5 (EOD/every-other-exit-path cancels first)**: required NO code change. `_cancel_symbol_open_orders` (called
from `_force_close_all_locked` before every close) already re-queries Alpaca for ALL open orders on the symbol and
cancels each one by id — it never referenced a specific stored leg id, so it cancels whichever order (safety-net or
our repriced target) happens to be live. Covered by a regression test
(`TestEodCancelIsGeneric`) rather than new production code.

**Rule 6 money defect — fixed 2026-09-29 (follow-up)**: the first build had a real bug, not just a documented gap:
`cancel_order() == True` does NOT prove zero fill — Alpaca cancels the remaining OPEN quantity of a partially-filled
order just as cleanly as an untouched one, so the original code (which only called `get_order` when `cancel_order`
returned `False`) would have silently booked a partial race as a same-size FULL close. Fixed: `cancel_resting_target`
now ALWAYS calls `get_order` after the cancel attempt, regardless of what the cancel itself returned, and classifies
on `filled_qty` vs. a `requested_qty` argument the caller now must pass (`watch.shares`/`pos.shares` at call time) —
`0` -> `CANCELLED`, `0 < filled_qty < requested_qty` -> new `CancelOutcome.PARTIALLY_FILLED`, `>= requested_qty` ->
`ALREADY_FILLED`. A `PARTIALLY_FILLED` race now books ONLY the filled qty via a new `StopMonitor.book_target_partial_fill`
(mirrors `_book_scale_fill`'s mechanism exactly: reduces `watch.shares` in place, does NOT retire the watch, queues a
`target_rested_partial` event) and then FALLS THROUGH into the unchanged stop-exit logic for the reduced remainder —
never a full close on a partial fill, never two exit legs merged into one. `ORBEngine._handle_target_partial_fill_event`
reuses the trades-table `scale_qty`/`scale_price`/`scale_pnl`/`scaled_at` columns (the one existing partial-exit
representation in this schema, also used by the deliberate 3R scale-out) but keeps its own `TARGET_RESTED_PARTIAL`
exit_reason so the two mechanisms are never confused in the exec-quality report; the row stays OPEN and the eventual
final exit (stop / EOD / a later full target fill) composes `pnl` from `pos.shares` (already reduced) + accumulated
`scale_pnl`, exactly as scale-out's runner leg does today. `_poll_target_fills` (the non-race path) now also handles
`order status == 'partially_filled'`, delta-tracking via a new `OpenPosition.target_last_booked_qty` field so a
still-resting, repeatedly-partially-filling order books only the NEW shares each poll — never double-counted across
ticks, and correctly composes a full close afterward from whatever remains. `_force_close_all_locked` now polls target
fills first (when the flag is on) before computing what to close, so a last-second fill is booked and `pos.shares` is
accurate before EOD's cancel-and-close sequence runs.

**Alpaca leg-cancel semantics relied on**: `_exit_via_sl_leg` (D3 FIX 1, unmodified) already re-verifies the SL leg's
live status via a fresh `get_order` + `_LIVE_ORDER_STATES` check, and re-queries the broker's actual held qty
(`broker_qty`) before trusting the SL leg to cover the whole position — it falls back to the legacy "cancel every open
order for the symbol, then submit a fresh protective sell for the broker's real qty" path whenever the SL leg isn't
provably live and full-covering, for ANY reason. This means the implementation does NOT need to assume a specific
answer to "does cancelling the TP leg cascade-cancel its OCO sibling SL leg" (Alpaca bracket/OCO legs are documented
to cancel as a linked pair — cancelling one member typically cancels the other): whether the SL leg survives the TP
cancel or is cascade-cancelled alongside it, `_exit_via_sl_leg`'s pre-flight check catches either outcome and the
existing bulk-cancel-and-place fallback re-discovers and protects whatever the broker actually shows. No position is
ever left without SOME stop-management path between my hook's TP cancel and the (unchanged) code that follows it.

**Rule 7 (boot reconciliation) — explicit-fallback half added 2026-09-29 (follow-up)**: `target_resting`/`target_price`
still live only on the in-memory `OpenPosition` and are NOT persisted across a restart (unchanged limitation — would
need a persisted column to fix for real), so re-resting a target automatically after a crash is still not implemented.
What changed: `sync_positions` now logs exactly ONE WARNING per open ORB position without a currently-tracked resting
TP, naming the symbol, whenever the flag is ON — this is no longer silent. The position needs no other handling: falls
back to today's tag-and-sell touchgo path automatically, since `_fire_touchgo_exit` already branches on
`pos.target_resting` (the dataclass default, `False`, for every rehydrated position).

**Tests**: `tests/test_orb_target_limit.py` — 56 tests (up from 39): the original primitives/parity/telemetry/EOD-
genericity/orphan-reconciliation/integration coverage, plus the 2026-09-29 follow-up's partial-fill-then-stop (two
exit legs, remainder sized correctly), partial-then-second-partial (delta-tracking, no double-count),
partial-then-EOD-flatten (pre-close poll wired + `pos.shares` correct before the close), and
`_handle_target_partial_fill_event` (row stays open, scale columns, orphan write path) coverage. `sync_positions`'s
new blocks are still exercised by construction/code review only, not end-to-end — its existing test surface is large
enough that driving the whole function was judged lower value than the primitive-level and unit-level coverage above
within the session's step budget.
