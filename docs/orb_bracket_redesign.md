# ORB Bracket + StopMonitor Re-Architecture — Design Doc

**Status**: Design only — DO NOT IMPLEMENT WITHOUT REVIEW.
**Author**: B3 design after 2026-05-19 incident investigation.
**Scope**: ORB entry-bracket + StopMonitor exit interaction. BF unaffected.

---

## 1. Problem statement

The 2026-05-19 incident exposed two related issues with ORB's bracket-order +
StopMonitor design:

1. **Alpaca-side bracket SL is at `safety_sl = entry × 0.90`** (10% below entry),
   while StopMonitor manages the **real stop at `range_low`** (typically 5–10%
   below entry). The two levels DON'T align — if StopMonitor's exit fails for
   any reason, the bracket SL fires at the wider safety level, leaking `(range_low − safety_sl) × shares` per stop-out.
   * Today's LMRI: real stop $6.86, safety_sl $6.56 → potential $284 leak on
     a stopped trade.

2. **Auxiliary cleanup paths must remember to cancel brackets first** before
   calling `close_position`, or Alpaca rejects with `insufficient qty
   available, held_for_orders=qty`. Today, BF's `_close_orphan_positions` did
   not — fixed in commit `f0fd573` (B2). Other code paths (manual
   intervention, future strategies) carry the same risk.

A **deeper architectural question**: should ORB use the bracket pattern at
all, given StopMonitor already manages the real stop client-side?

---

## 2. Current architecture (today's code, post-`f0fd573`)

### Entry placement
`orb_engine._submit_entry` → `alpaca.submit_stop_bracket_order` with:
* Entry: stop-limit BUY at `range_high` / `range_high × 1.003`
* SL leg: SELL STOP at `entry_price × (1 − safety_sl_pct)` (= entry × 0.90)
* TP leg: SELL LIMIT at `entry_price × 3.0` (unreachable filler — Alpaca's
  bracket order class requires a TP)

On submit, Alpaca returns 3 order IDs (parent, SL leg, TP leg).
* Parent transitions: `pending_new` → `accepted` → `filled`
* Legs sit in `HELD` status until parent fills; then both legs become active.
  TP is `NEW` (live limit); SL is `HELD` (armed stop — fires when price
  drops to stop_price).

### Position tracking
On entry fill, `ORBEngine._process_pending_fills` calls
`stop_monitor.add_watch(symbol, stop=range_low, lock_arm_at_r=1.5,
lock_stop_r=1.0, sl_leg_id=..., tp_leg_id=...)`.

`WatchEntry` (stop_monitor.py:43) stores:
* `stop_price` — the real stop = `range_low`
* `sl_leg_id` / `tp_leg_id` — Alpaca order IDs of the bracket children
* `lock_arm_at_r` / `lock_stop_r` — static-lock parameters

### Exit paths (all StopMonitor-driven)
**A. Autonomous stop trigger** (`_execute_stop_exit`): price drops to
`stop_price` (range_low or locked +1R). Steps:
1. Cancel `tp_leg_id` + `sl_leg_id`
2. Submit marketable limit SELL at bid (or bid − offset)
3. Poll for fill; fall back to `close_position` if not filled in N seconds
4. Update DB, emit `StopExitEvent`

**B. Force exit** (`force_exit` — touchgo, force-close):
Same as autonomous trigger, but driven by an external decision (engine call)
instead of price. Whitelist of valid reasons (`tag_bb`, `tag_b1`).

**C. Lock arm** (price touches +1.5R): client-side stop_price updates in
WatchEntry. Bracket SL on Alpaca is NOT updated — only the client-side stop
tracker. **The Alpaca-side SL stays at safety_sl forever.**

### What today's 2026-05-19 cascade revealed
* Path A worked: WAY/CORD/PURR all exited cleanly via the touchgo→force_exit
  path which correctly cancelled brackets first.
* LMRI: no touchgo (close in top half of breakout bar), Rule D didn't fire,
  position sitting open. Real stop $6.86, safety_sl $6.56. If price drops to
  $6.86, StopMonitor's exit attempt would compete with the bracket TP's
  `held_for_orders` claim — **cancel-then-sell would still work** (per
  `_execute_stop_exit`). So position IS protected at $6.86 via Path A.
* BF's `_close_orphan_positions` skipped the bracket-cancel step → failed.
  (Fixed B2.)
* If StopMonitor process dies AND the bracket SL is at $6.56 not $6.86,
  position protected only at the wider safety net. **Alignment gap.**

---

## 3. Design goals

In priority order:

1. **Alignment**: Alpaca-side fallback SL should match the strategy's real
   stop (range_low). The 30-cent gap on LMRI is unnecessary.
2. **Atomicity**: SL placement should be atomic with entry — no window where
   the position exists without a stop on Alpaca's side.
3. **Single source of truth**: as few places as possible to update when the
   stop changes (lock arm, etc.).
4. **Backwards-compatible cleanup**: any auxiliary path that flattens a
   position must continue to work via the existing
   `_cancel_open_orders_for_symbol` → `close_position` pattern.
5. **BT-LIVE parity**: BT fills the position on `bar_high >= range_high`,
   exits on `bar_low <= stop`. Any new design must produce equivalent fills.

---

## 4. Options considered

### Option A (current): bracket with safety_sl + StopMonitor manages real stop client-side
Status: today's code. Works but **misaligned** (gap of `range_low − safety_sl`
per stopped trade).

### Option B (recommended): bracket with SL at `range_low`, TP unreachable, StopMonitor uses `replace_order_stop_price` for trail/lock
Move the bracket SL to range_low. StopMonitor's role becomes: track price,
on lock-arm call `replace_order_stop_price(sl_leg_id, new_stop)`. Autonomous
stop trigger and force-exit paths unchanged (still cancel + sell).

**Pros**:
* Aligned: Alpaca-side fallback = real stop. Zero leak.
* Atomic: SL placed with entry.
* Single source: lock_arm reaches Alpaca via one replace call.
* Minimal code change: only `orb_planner` SL computation + StopMonitor lock-arm
  logic.

**Cons**:
* `replace_order_stop_price` returns a NEW order ID. WatchEntry.sl_leg_id
  must be updated after each replace.
* OCO behavior of the bracket after replace_order: **unverified**. If the
  parent-child relationship breaks, TP doesn't auto-cancel when SL fires →
  orphan unreachable limit sell. **Research item.**
* Today's A2 manual replace proved A2-style replace WORKS while in HELD status,
  but the OCO question wasn't tested.

### Option C: bracket with stop_limit SL (instead of stop) at range_low
Same as Option B but the SL leg is `stop_limit` instead of `stop`. Limit
caps slippage on the safety-net fill (`stop_limit_price = range_low − 0.5 × spread`).

**Pros over B**: slippage control if the bracket SL ever fires.
**Cons over B**: more complex; needs spread estimate at entry time.

### Option D: drop bracket entirely; use simple stop-limit entry + post-fill SL
Match the BF pattern. Submit a separate SELL STOP after entry fills.
StopMonitor manages updates via `replace_order_stop_price`.

**Pros**: simplest; uniform with BF.
**Cons**: race window between entry fill and SL submission (~10–500 ms).
If process dies in that window, position is **naked**. Today's bracket
design explicitly prevents this. ORB's strict-risk profile makes this
race window a real cost.

### Option E: hybrid — bracket for atomic protection, swap to plain SL after StopMonitor takes over
Submit bracket on entry. Once StopMonitor.add_watch is confirmed, cancel the
bracket SL+TP and submit a single plain SL. From there, manage as Option D.

**Pros**: atomic + clean post-handoff
**Cons**: most complex; cancel+resubmit window has a ~ms race; more code paths.

---

## 5. Recommended design — Option B (with B+ stop_limit SL upgrade later)

**Primary fix**: move bracket SL from `entry × 0.90` to `range_low`. Keep
unreachable TP for bracket-class compliance. StopMonitor uses
`replace_order_stop_price` for trail/lock; autonomous and force-exit paths
unchanged.

### Code changes

1. **`trading/orb_planner.py`** — change SL computation from
   `entry × (1 − safety_sl_pct)` to `range_low`. Add `safety_sl_pct` as a
   deprecated config (kept for one release for rollback).

2. **`trading/orb_engine.py:_submit_entry`** — pass new SL to
   `submit_stop_bracket_order`. After fill, when StopMonitor would have
   armed the static lock, call `alpaca.replace_order_stop_price(sl_leg_id, locked_stop)`
   instead of updating only `WatchEntry.stop_price`.

3. **`trading/stop_monitor.py`** — on `_arm_static_lock`:
   * Call `replace_order_stop_price(watch.sl_leg_id, locked_stop)` — get new order ID
   * Update `watch.sl_leg_id` to the new ID (atomic under `_watch_lock`)
   * Update `watch.stop_price` to `locked_stop`
   * Handle replace failure: log ERROR, fall back to current client-side-only
     tracking (better than crashing)

4. **`persistence/database.py`** — Migration 14: add `bracket_sl_order_id`,
   `bracket_tp_order_id` columns to `trades` table for restart-safe
   re-attachment. Already partially there (sl_leg_id / tp_leg_id) — verify.

5. **`orb.yaml`** — new section:
   ```yaml
   bracket:
     sl_at_range_low:
       enabled: false        # phased rollout: default off → on after BT validation
   ```

6. **Tests**:
   * `tests/test_orb_bracket_sl_at_range_low.py`:
     - submit_entry with flag=true puts SL at range_low
     - lock_arm replaces SL via replace_order_stop_price
     - autonomous stop trigger still cancels both legs
     - force_exit still cancels both legs
     - replace_order failure → log ERROR + fall back gracefully
   * `tests/test_orb_bracket_sl_parity.py` — BT side: same code path (BT
     uses bar-based fills regardless of where SL is set; should be byte-identical).

### What does NOT change
* BF code — completely untouched.
* StopMonitor's autonomous stop-trigger path (`_execute_stop_exit`).
* StopMonitor's force_exit path.
* `_cancel_open_orders_for_symbol` (B2 fix) — still needed for orphan cleanup.
* MACD wave — untouched.
* The unreachable TP (kept for bracket-class compliance).

---

## 6. Risks & unknowns ⚠

### R1 — `replace_order_stop_price` and OCO behavior **[BLOCKER until verified]**
The OCO pair behavior between an SL leg and TP leg: when SL fires, TP auto-cancels.
After `replace_order_stop_price` creates a NEW order ID for the SL, does the
new SL still participate in the OCO with the original TP? Or does the
parent-child link break, leaving the TP orphaned?

**Verification plan**:
1. On paper account: submit bracket order, wait for entry fill.
2. Call `replace_order_stop_price` on the SL leg → confirm new order ID.
3. Force-trigger the SL (replace stop price to slightly above current ask) →
   wait for fill.
4. Check: is the TP cancelled automatically?

If TP is NOT auto-cancelled, Option B requires manual TP cancellation after
SL fires — same pattern as today's force_exit (cancel both before sell), but
applied to the autonomous trigger too. Manageable, but adds code.

### R2 — `replace_order` race condition
Between calling `replace_order_stop_price` and getting the new order ID,
price could trigger the OLD SL. Alpaca's replace is atomic from their side
(reject the old, create the new in one transaction), but the client receives
the new ID only after the transaction. If price triggers during this window,
the OLD SL fires first.

**Verdict**: acceptable. Trigger during a sub-second window is a known
brokerage racing pattern. The fill happens at the OLD stop price, which is
*better* than the locked level (lower stop = more loss but earlier protection).

### R3 — Bracket SL at `range_low` may be tighter than today's 10% safety
If range_low is, say, 8% below entry, the bracket SL fires more often than
the 10% safety_sl. In theory it should NEVER fire if StopMonitor's
client-side exit fires first — but if StopMonitor is dead, the position
exits at range_low instead of safety_sl. **This is the desired behavior**
(closer to strategy intent), but worth flagging.

### R4 — BT-LIVE divergence on lock-arm update
BT models a static lock by checking `bar_high >= entry + 1.5R` → set
in-memory lock_stop = entry + 1R → trigger when `bar_low <= lock_stop`.
This is bar-based, no API.
LIVE Option B: replace_order_stop_price at the moment +1.5R is touched →
Alpaca-side SL armed at new price. Different mechanism, same outcome
(stop at entry + 1R). **Verify with side-by-side BT.**

### R5 — Concurrent replace calls
If StopMonitor handles trail+lock and both want to replace the SL at nearly
the same time (e.g., +1.5R touched simultaneously with new trail update),
race condition possible. **Mitigation**: hold `_watch_lock` during the
replace+update sequence. Acceptable performance overhead.

### R6 — Restart recovery
On restart, `sync_positions` must re-attach to the existing bracket SL.
Today's WatchEntry persists sl_leg_id via DB. Need to verify the DB
schema actually stores it and `sync_positions` rehydrates correctly.

**Verification**: query `sqlite> SELECT id, symbol, bracket_sl_order_id FROM trades WHERE strategy='orb' AND order_status='filled';` — should show non-NULL bracket_sl_order_id for filled positions.

If the DB doesn't store it: Migration 14 adds the columns + backfill from
Alpaca on restart for in-flight positions.

---

## 7. Implementation phasing

### Phase 0 — Research (no code; ~2 days)
* **R1 verification on paper account** — submit bracket, replace SL,
  trigger, confirm TP auto-cancel. If TP not auto-cancelled, add cancel
  logic to autonomous-trigger path.
* **R6 DB schema audit** — confirm sl_leg_id/tp_leg_id columns exist.

### Phase 1 — Behind config flag (default OFF; ~1 week)
* Code changes 1–4 (planner, engine, stop_monitor, DB migration).
* Tests (orb_bracket_sl_at_range_low + parity).
* Ship with `bracket.sl_at_range_low.enabled: false`. Today's path stays
  default.

### Phase 2 — BT validation (~2 days)
* Run side-by-side BT: today's design vs Option B. Should be functionally
  identical on bar-based exits. Confirm <$100 difference on 16-month BT.

### Phase 3 — Live single-day flag-on (~1 day observation)
* Flip flag to `true` on a Friday after-close. Monitor Monday's session.
* Watch for: replace failures, OCO weirdness, slippage anomalies, BT-LIVE
  divergence on stop fills.

### Phase 4 — Default-on (~2 weeks observation)
* If Monday clean, flip default to `true` in `orb.yaml.template`.
* Watch for 2 weeks. No additional changes.

### Phase 5 — Cleanup (~1 day, after 1 quarter)
* Remove the flag and the old code path.
* Remove `safety_sl_pct` config.

**Total elapsed**: ~6 weeks from start to cleanup. Pace dictated by
observation windows, not coding time.

---

## 8. Open questions for the user

1. Do we want stop-limit SL (Option C) or plain stop SL (Option B)? Tradeoff
   is slippage cap (limit) vs simplicity (stop). For ORB's $3–30 stocks,
   slippage from a market sell is typically 5–15 bps. Probably fine; B is
   simpler.

2. Should we keep `safety_sl_pct` as a *deeper* fallback (e.g., 15%)? If
   Alpaca's replace_order fails AND StopMonitor's client-side stop fails,
   we want SOMETHING. Today's safety_sl serves this role. Could keep it as
   a "double-deep" net below range_low. **Marginal value**; adds complexity.

3. Do we want to backport Option B to MACD wave? It also uses brackets.
   Probably yes for consistency, but separate ticket.

4. Migration of in-flight positions: at the moment we flip the flag, any
   open ORB positions still have SL at safety_sl. Should we:
   * (a) Leave them — they'll exit naturally; new entries use new design.
   * (b) Sweep them: cancel each safety_sl + submit replacement at range_low.
   (a) is safer; (b) is cleaner. Recommend (a).

5. Pace: phases 1–5 take ~6 weeks. Is that acceptable, or do we accelerate?

---

## 9. Files this change would touch

* `trading/orb_planner.py` — SL computation
* `trading/orb_engine.py` — bracket submission, fill handling
* `trading/stop_monitor.py` — lock_arm replace_order pattern
* `persistence/database.py` — Migration 14 (if columns missing)
* `orb.yaml` + `orb.yaml.template` — new flag
* `config.py` — new config property `orb_bracket_sl_at_range_low_cfg`
* `tests/test_orb_bracket_sl_at_range_low.py` — new (~10 tests)
* `tests/test_orb_engine_replace_order_path.py` — new (~5 tests)
* `tests/test_orb_bracket_sl_parity.py` — new (BT-LIVE parity, ~3 tests)
* `docs/orb_bracket_redesign.md` — this file
* `README.md` — add a section once shipped
* `CLAUDE.md` — feature-flag note once shipped

**Estimated code volume**: ~400 LOC changed/added, ~600 LOC tests.

---

## 10. NOT in scope

* MACD wave bracket usage (separate ticket).
* BF — no bracket use; B2 fix is sufficient.
* `safety_sl_pct` removal — kept for one release as deprecated.
* OTO order class — declined in favor of bracket+unreachable-TP for the
  simpler migration path.
