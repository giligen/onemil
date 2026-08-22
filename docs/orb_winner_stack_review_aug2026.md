# ORB Winner Stack (ATR floor + scale-out) — Adversarial Design Review
**Reviewer role: hostile production engineer + hostile market. 2026-08-22.**
**Target: docs/orb_winner_stack_design_aug2026.md (pre-build).**

## VERDICT: CONDITIONAL-GO
The physics freeze is mostly faithful and the validated numbers check out
(base book $10,715.68/81 ✓; C-point $9,092/15 green/MDD −700/worst −309 ✓;
B-point $11,004/13 ✓). But the design misdescribes one frozen semantic, leaves
the live partial-exit state machine underspecified in exactly the places that
produce orphaned runners / double-counted P&L, and its restart + rollback
stories contradict the current rehydration code. Six P0s below are all fixable
in the design before a line of implementation is written. No NO-GO-grade flaw
in the validated economics themselves was found.

---

## P0 — must fix before live

### P0-1. Design §1.3 states the same-bar ordering BACKWARDS vs the frozen code
Design: "a bar that hits BOTH the stop and the scale level counts as stopped
first". The validated code does the opposite. `resim_exit.variant_scale` and
`phaseB_frontier.variant_scale_sz1` both gate the stop check with
`l[i] <= stop AND h[i] < scale_px` — on a bar that touches both, the branch
falls through to `h[i] >= scale_px` → the **scale FILLS at +3R**, and only the
runner stops (and since +3R > 1.75R the same bar arms the lock, so the runner
exits at +0.5R, not the floor). That is the *optimistic* convention, not the
pessimistic one.
- **Empirical**: exactly 1 of 81 book trades has a both-hit bar — **NCNA
  2025-08-21**: code convention → scale_sz1 +$158; design-prose convention →
  stopped ≈ +$50. A ~$108 delta in a single month obliterates gate 3.2's
  <$5/month tolerance. NCNA is the mandatory golden for gate 3.3.
- **Required**: rewrite §1.3 to the code's actual rule; the gate-3.3
  "pessimistic same-bar" golden must encode the CODE's behavior (scale fills,
  runner lock-stops same bar), not the prose. Anyone implementing from the doc
  today builds a book that fails gate 3.2 — or worse, "fixes" the harness.
- **Live deviation note (direction)**: live tick sequencing is
  whichever-trades-first. On stop-then-scale sequences live exits the whole
  position where BT banks 40% at +3R → live ≤ BT on those bars. Rare
  (one 1-min bar spanning stop→+3R) but must be documented as a known
  optimistic edge of the frozen model, with the EoD dive attributing any such
  live trade.

### P0-2. The scale partial must NOT flow through the terminal exit-event path
`orb_engine._handle_exit_event` (orb_engine.py:3303) **pops the position and
writes a closed exit_update for EVERY drained event**. The design specifies the
StopMonitor side (`partial_exit(...)` emitting an event) but never the engine
drain-path branch. As designed, the first scale fill closes the DB row
(`order_status='closed'`, pnl = (scale_px−entry)×full shares) and orphans the
60% runner — no engine tracking, no force-close coverage, only the (resized)
safety leg.
**Required in design**: a distinct partial event type/flag + an explicit
`_handle_exit_event` branch: update scale columns, reduce `pos.shares`, do NOT
pop, do NOT write exit_price/order_status, do NOT touch daily_pnl (per the
realized-at-close convention). Gate 3.4's "DB two-part row lifecycle" test must
assert the row is still OPEN after the scale fill.

### P0-3. P&L writers: enumerate or double-count
`_handle_exit_event` computes `pnl = (exit_price − entry) × pos.shares`
(orb_engine.py:3312). With a filled scale, this is wrong in one of two ways
depending on whether pos.shares was reduced (scale_pnl dropped) or not (scale
leg double-counted at the runner's exit price). ALL pnl writers must be made
scale-aware and tested:
1. `orb_engine._handle_exit_event` (3312) — runner-qty pnl + stored scale_pnl.
2. `orb_engine._sync_db_after_fc` (4792) — force-close reconstruction uses
   `(exit−entry)×qty` from the close order; must add scale_pnl.
3. The orphan reconciler / `_sync_closed_positions` path.
4. Report fallbacks: `report_common._row_pnl` and `cumulative_orb_since`
   recompute `(exit_price − fill) × filled_qty` when `pnl` is NULL — on a
   scaled row that silently drops scale_pnl. Kill rails
   (`orb_engine._realized_orb_pnl`: `SUM(pnl) WHERE pnl IS NOT NULL`) are
   consistent with realized-at-close ONLY if `pnl` is guaranteed non-NULL on
   every close path.
**Rules to freeze in the design**: (a) `pnl` is written non-NULL, combined, at
final close on every path; (b) `filled_qty` stays the ENTRY qty forever (it
feeds fill-parity and _row_pnl); (c) the SQL fallbacks gain
`+ COALESCE(scale_pnl,0)` anyway (defense in depth); (d) daily_pnl/kill rails
get scale_pnl exactly once, at final close (already the design's stated
convention — now it has a mechanism).

### P0-4. Restart mid-scale re-inflates the runner and reverts the floored stop
`sync_positions` rehydration (orb_engine.py:4060-4090) restores
`shares = trades.shares` and `stop_price = trades.stop_loss_price`:
- Scale filled → restart → watch and pos come back with **FULL shares** →
  final pnl on full shares (P0-3 double-count realized), plus an oversell
  attempt that only the broker-qty snapshot in `_execute_stop_exit` papers
  over.
- The design routes the floored stop to `real_stop_loss_price` + the watch
  only; `stop_loss_price` (written from `plan.stop_price` at 4637) keeps
  range_low → **any restart silently reverts the stop to range_low**, live
  drifting from the book the flags were validated on.
**Required**: floored stop becomes `plan.stop_price` (single source feeding
stop_loss_price, the watch, and rehydration); rehydration subtracts
`scale_qty` when `scaled_at` is set and restores scale_done; gate 3.4's
restart test must assert shares AND stop AND scale_done — and it must do so
via a real kill-9/reboot drill, not just a unit test (pre-deploy rehearsal
protocol). Note the same rehydrate block still defaults lock params to
1.5/1.0 (stale) and loses lock_armed — pre-existing, but do not add scale
state to a path that already forgets state without fixing the pattern.

### P0-5. Stop-vs-scale collision windows are undefined (and one is blocking)
1. **_exit_in_progress starvation**: the existing partial pattern
   (`execute_partial_exit`) holds `_exit_in_progress` through a 30s
   sleep-polling fill wait; `_execute_stop_exit`, `force_exit`, and touchgo all
   NO-OP (not queue) while it is set. A collapse during the scale fill-wait
   means the software stop and touchgo cannot fire for up to 30s, with the
   safety leg already resized to 60%. Required: short fill budget for a
   marketable +3R limit (seconds), an immediate stop re-check after the flag
   clears, and the wait must NOT run on the WS event loop (the existing
   pattern is documented "called from main thread"; the design calls it from
   tick/poll paths — blocking the loop freezes tick processing for ALL
   symbols).
2. **Leg release failure ordering**: design says release/resize legs first,
   retry next cycle on failure — but does not define the compensating action
   when the resize SUCCEEDS and the scale submit FAILS (runner+scale shares
   covered by a 60% leg, no scale order). Required: restore leg qty on submit
   failure, WARNING both ways.
3. **replace_order_qty returns a NEW order id**
   (alpaca_client.py:1712-1734 — Alpaca replace = cancel+create). The
   existing exhaustion code discards it, leaving `watch.sl_leg_id` stale; a
   later `_execute_stop_exit` best-effort cancel then targets a dead id. Do
   not copy this bug into the scale path: update sl_leg_id from the return.
4. **Partial fill of the resting scale order at stop time**:
   `_execute_stop_exit`'s bulk-cancel kills the resting scale limit — but if
   it partially filled first, those shares are booked nowhere (the stop event
   only carries the stop sell). Required: after bulk-cancel, query the scale
   order's filled_qty/avg price and book it as scale_qty/scale_pnl before the
   final row is written. (15:45 force-close: `_cancel_symbol_open_orders` +
   broker-qty close already sequence correctly; `_sync_db_after_fc` is covered
   by P0-3.2.)
5. Do NOT implement the scale by replacing the 3×entry safety-TP leg: it is
   OCO with the SL leg — a TP-side fill cancels the SL and strands the runner
   legless.

### P0-6. ATR availability is a function of CACHE COVERAGE, not the frozen rule
Two distinct problems:
1. **Boundary rule mismatch (latent, book-inert)**: the frontier C-point
   (`phaseB_frontier.atr14`) fetches `LIMIT 15` prior bars, returns None only
   under 14, and takes `tr.iloc[-14:].mean()` with pandas skipna — at exactly
   14 prior bars it floors from a **13-TR mean**, where design §1.1 (and the
   B-point script) require ≥15 bars → fail-open. Empirical: **0 of 81** book
   trades sit on this boundary, so gate 3.2's target is unaffected — but the
   shared module must freeze ONE rule (§1.1's ≥15/fail-open) and golden-test
   the 14-bar boundary.
2. **Cache-gap parity hole (real, 6/81 trades)**: SMST 2025-01-07 (5 prior
   cached bars), ARQQ 2025-01-16 (11), BTQ 2025-10-03 (5), RGTZ 2025-10-16
   (5), FJET 2025-12-24 (4), PS 2026-05-01 (2) were left UNFLOORED in the
   validated book because **cache.db daily_bars has gaps** — SMST (2x-MSTR
   wrapper, listed Aug'24) and ARQQ (listed 2021) had ample real history.
   Live's 40-day Alpaca fetch WILL find ≥15 bars and floor them → live
   deviates from the validated book on ~7% of trades, in an unvalidated
   direction, and the nightly BT (same gappy cache) never flags it because
   green_verdict compares symbols only, not stops.
   **Required**: (a) backfill daily_bars for all book symbols before the
   Monday regen (and keep the nightly refresh covering active ORB symbols);
   (b) add a floored-stop drift check to the EoD dive: recorded
   `real_stop_loss_price` vs the BT-recomputed floor per trade — HARD flag on
   mismatch, same spirit as the pm_mult recompute gate.
Also freeze the rollout ordering (attack #6): the Monday pre-boot regen runs on
prod cache where vendor bar revisions can move bars. **Gate the flip**: regen
must reproduce the gate-2 book (same picks, monthly totals within tolerance);
if regen FAILS or drifts, flags revert to false before boot — never boot with
flags on against the stale base reference book.

---

## P1 — before scale-up

### P1-1. Scale evaluation must honor the entry-bar skip + touchgo window
BT's scale search starts at bar 1 and touchgo prefire pre-empts any scale.
Live: the scale check must sit AFTER the `skip_exits_until_ts` gate (same as
arm/stop) and must not race touchgo — a bar-0 tick at +3R firing a partial
while Rule M fires force_exit at bar close leaves force_exit no-op'd by
`_exit_in_progress` and the touchgo cut silently lost (force_exit does not
retry). Simplest faithful rule: do not evaluate scale until the touchgo
bars (0/1) are resolved — matches the frozen loop structure.

### P1-2. Scale fill realism: BT touch-fills, a live limit does not
The frozen model fills the scale whenever a bar HIGH touches the level, at
level×(1−10bps) — a price a limit sell at level can never print, and a bare
touch frequently doesn't fill live (queue). Direction: live scale fill rate <
BT; an unfilled scale reverts that trade to base behavior — reintroducing
exactly the monthly variance the owner paid −$81/mo to remove. Required:
EoD attribution line (scale-armed / submitted / filled / touch-no-fill with
prices) + weekly live-vs-BT scale-fill-rate compare; revisit after ~20
scale-armed trades. Any limit-price sweetener (e.g. level −5bps) is a
semantics change → owner sign-off + frozen-params note, not a quiet tweak.

### P1-3. Degenerate-ATR clamp (attack #2)
floor = entry − 0.25×ATR is strictly below entry only when ATR > 0; a
halt-shell tape with 14 H==L days gives ATR≈0 → stop == entry → guaranteed
instant stop-out after the skip window. Study data: min bound-floor distance
below entry across the book = 1.57% (median 3.24%), so the book never came
near the degenerate zone — the clamp costs nothing.
**Required**: reject the floor when `floored_stop > entry×(1−ε)` (ε = 0.5%
suggested) → fall back to range_low + WARNING. Also state which entry anchors
the floor live: the ACTUAL fill price (not planned rh×1.003), and document it
as a deliberate BT deviation (same class as the touchgo fill-vs-market-bar
decision).

### P1-4. DB shape: two parallel partial-exit column families
`partial_exit_{price,shares,pnl,reason,exited_at}` already exist (bull-flag
exhaustion, Migration 4) with an ExitReason (`EXHAUSTION_PARTIAL`) and
documented semantics. The design adds `scale_qty/scale_price/scale_pnl/
scaled_at` — a second, ORB-only convention every report consumer must now
know about. Either reuse the existing columns with
`partial_exit_reason='scale_out'`, or state in the design WHY a separate
family is needed (e.g. exhaustion coexistence) — and in either case point
`_row_pnl`/`cumulative_orb_since`/attribution at ONE convention with tests.

### P1-5. Green-check / vocabulary drill for the first scaled trade
Runner keeps its exit_reason ✓, but any path that writes the scale event's
reason into `exit_reason` (reconciler, FC verify) must find it in
`trading/exit_reasons` or day 1 of a scaled trade REDs the streak on
"unattributed exits". Add to the Sunday rehearsal: replay a scaled trade's
DB row through `green_verdict` + `sizing_attribution` + `_row_pnl` and require
green. Note green_verdict's BT-parity compares SYMBOLS only → the mid-month
book swap is parity-safe for selection (verified: selection untouched by exit
flags; days 1-5 of the validation month have ZERO orb rows in trades.db, so
the swap-point bookkeeping claim holds), but any MTD live-vs-book P&L tracker
must switch reference books at the same commit.

### P1-6. Composition debt vs the frozen contract (STAG/MID)
The frozen contract (orb_bplus_frozen_params_aug2026.yaml §4-5) mandates STAG
and MID as part of the B+ program; neither is in the live engine, the BT
pipeline, nor the resim harness today — the C-point was validated on
static-lock+touchgo physics only, and the book CSV confirms it (exit reasons:
stop/eod/tag_bb/tag_b1/lock exclusively). When STAG/MID ship, the winner-stack
expectation set ($455/mo, 15/20 green) is STALE: SZ1's raised floor can fire
before STAG's 20-min check; MID's BE-raise interacts with the floored stop and
the runner. The amendment must state ship ORDER and that the next exit-rule
ship re-derives the combined book through the same harness. Do not let two
"validated separately" amendments compose unvalidated — that is how the
accidental-rules class returns.

### P1-7. Rollback with an open scaled position is NOT zero-state as written
Flag-off + restart rehydrates through the code in P0-4: shares re-inflate and
scale columns stop being read if the reader is flag-gated. Required: the
scale-aware rehydration and pnl composition are UNCONDITIONAL (data-driven by
`scaled_at IS NOT NULL`), never gated on `exit.scale_out.enabled`; the flag
gates only NEW scale arming. Then the design's "rollback = flip + restart"
claim becomes true. State it.

### P1-8. Carry the study's own caveat into the contract amendment
phaseB_regime_atr.md's verdict on SZ1-alone: "NO-GO standalone… classic
overfit shape (TRAIN 6/6, VAL/OOS flat), flips exactly one red, zero monster
cost." The owner bought the C-point knowingly (−$81/mo for +3 greens) — the
amendment should quote the caveat so the next reader doesn't mistake the floor
for alpha: it is a tail-shaping device whose measurable value appears only in
worst months; most months it must do nothing, and "it did nothing again" is
the EXPECTED monitoring reading.

---

## Empirical checks (81-trade B+ book, C-point composition; read-only
per-trade resim against cache.db, reproducing the frontier walk exactly)
- **C-point reproduction: $9,092 — exact match** to _frontier_monthly.json
  (validates that the frozen semantics extracted here ARE the validated ones).
- Reason mix under C: stop 25, scale 20, tag_bb 19, lock 8, eod 7, tag_b1 2
  (20/81 trades scale).
- Floor binds on 30/81 trades; aggregate P&L delta on bound trades vs base
  = −$1,681 within the C composition (the floor is a tail device, not alpha —
  consistent with the study's own NO-GO-standalone read).
- Both-hit (stop AND +3R same 1-min bar): 1 trade (NCNA 2025-08-21) — the
  P0-1 golden.
- Arm(1.75R)+stop same bar: 4 trades (the frozen arm-before-stop convention
  is load-bearing; do not "fix" it).
- Scale bar also touching the stop: 1 (same NCNA bar — scale fills, runner
  lock-stops same bar).
- Bar-0 low ≤ floored stop: 3 trades (vs 2 for range_low; touchgo covers 0)
  — live matches BT ONLY because of the `skip_exits_until_ts` entry-bar gate;
  the scale/floor code must sit behind the same gate and never weaken it.
- ATR 14-vs-15-bar boundary: 0 trades affected (P0-6.1 latent, not
  book-material).
- Trades with <15 cached daily bars (unfloored in the validated book):
  6 — SMST, ARQQ, BTQ, RGTZ, FJET, PS (P0-6.2).

## Rehearsal additions (beyond design §4)
1. **Collapse-during-scale drill**: replayed sequence — scale submits, price
   crosses the stop before fill, partial times out/cancels → assert stop
   re-fires immediately after `_exit_in_progress` clears and the row books
   any partial scale fill.
2. **kill-9 mid-scale drill** (paper): scale filled → hard kill → boot →
   assert watch shares = runner qty, stop = floored stop, scale_done=True,
   final pnl = scale + runner on close.
3. **Green-check replay**: run daily_green_check --dry-run on the drill day;
   require green with a scaled row present.
4. **Regen-failure drill**: simulate Monday regen failure → confirm the
   documented flag-revert path leaves the system booting against the OLD book
   with flags OFF (never mixed).

## Cleared (attacked, held)
- Frontier composition = "two rules composed" holds structurally
  (variant_scale_sz1: floored stop from bar 1, runner keeps the floored
  stop — §1.3's "ORIGINAL stop/lock levels" must be read as the FLOORED
  initial stop under composition; say so explicitly), except the ATR
  boundary rule (P0-6) and the same-bar prose (P0-1).
- Design's headline numbers all reproduce from the artifacts (base
  $10,715.68/81; C $9,092/15/−700/−309; B $11,004/13/−330/−966).
- Sizing untouched by the floor ✓ (study kept _rp_position fixed; realized
  risk shrinks when the floor binds — matches §1.2).
- Live stop-first tick ordering is CONSERVATIVE vs the frozen model (never
  better than BT on both-hit bars) — validation claim survives in the safe
  direction, with P0-1's attribution note.
- Entry-bar skip parity exists live (`skip_exits_until_ts`) — the floored
  stop does NOT create new entry-bar stop-outs vs BT as long as the scale
  check sits behind the same gate (P1-1).
- 40-calendar-day daily-bars fetch ≈ 26-27 trading bars ≥ 15 needed —
  window OK; note `_get_feature_context` discards raw bars, so the ATR needs
  the bar list stashed in ctx (plumbing, called out in the design already).
- Kill rails read `SUM(pnl) WHERE pnl IS NOT NULL` — consistent with
  realized-at-close given P0-3's non-NULL rule.
- Force-close cancels the resting scale limit via the existing bulk-cancel ✓
  (residual partial-fill race handled by P0-5.4).
- Cross-strategy add_watch collision (8/22 prestage change): scale updates
  mutate the existing watch in place, no re-add → unaffected.
