# D3_exec — what shipped

Built 2026-09-17 from `REPORT.md`, in the report's ship order. **Nothing was
deployed.** The live engine keeps running the old code until the owner's
deploy word; the weekend rehearsal protocol
(`feedback_predeploy_rehearsal_protocol`) applies. No service was restarted,
no order submitted, and the only config flipped ON is the one that has no
plausible way to be worse (FIX 1, with a kill switch).

Branch `fix/spy-regime-shared-helper`, six commits, one per fix, pushed.

> **One caveat that is not optional to read.** `onemil-trader` runs FROM this
> working tree and auto-restarts on failure. The process currently in memory
> still holds the old code, but **any restart — an operator's, or a crash —
> now loads the new code**, and `config.yaml` (gitignored) already carries
> `prefer_sl_leg_exit: true` and the dormant `exit_ladder` block. Bull flag
> and ORB are both paused (`trading.enabled: false`, `orb.yaml
> strategy.enabled: false`), so nothing new can open a position; the exposure
> is limited to an exit on a position that is already open. If the owner
> wants the old exit path to survive an unplanned restart before the deploy
> word, the one-line hold is `prefer_sl_leg_exit: false` in `config.yaml`
> (the ladder is already off, and FIX 6/7/3/5 have no flag).

---

## 0. Status per fix

| fix | state | default | rollback |
|---|---|---|---|
| 6 — split `exit_branch` from `exit_reason` | **shipped** | on (it is a new column + a bug fix) | `git revert` the commit; the column stays and goes NULL |
| 7 — `unknown_exit` writes no price / no P&L | **shipped** | on | `git revert` the commit |
| 1 — replace the broker SL leg, never cancel it | **shipped** | **on** | `trading.self_managed_stops.prefer_sl_leg_exit: false` + restart |
| 3 — a partial fill is a partial success | **shipped** | on | `git revert` the commit |
| 2 — the exit ladder | **shipped, DORMANT** | **off** | `exit_ladder.enabled: false` (already) |
| 5 — no midpoint on an urgent exit | **shipped** | on | `git revert` the commit |
| 4 — validate the trigger print | **NOT BUILT** | — | — |
| 8 — terminal-state sweep | **NOT BUILT** | — | — |

FIX 4 and FIX 8 were out of the time box and are left out cleanly — no
half-built code, no skipped tests. Both are still specified in `REPORT.md`
§4 with their code locations and tests. FIX 4 is tail insurance worth ≈ $0
in the census (RCAT's stop was hit legitimately one bar later); FIX 8 is
four historic rows and a post-close job. Neither blocks the others.

The **daily sweep** inside FIX 7 is a documented STUB
(`trading/unknown_exit.sweep_unknown_exits`) that logs `NOT IMPLEMENTED` at
WARNING, writes nothing, and returns `[]`. It is the half of FIX 7 that
writes P&L, so it needs its own evidence and its own review.

---

## 1. FIX 6 — the branch is not the reason (`be62c8f`)

`_execute_stop_exit` overwrote `exit_reason` with `stop_loss_market_fallback`
for ANY watch that escalated. The bucket therefore held FJET 2026-06-12 — a
**trailing-stop exit on a winner, +$122.84** — SMCX (another trail), and
HCAI 2026-09-11 (an ignition EOD `stage_force_flat`), alongside three real
stops.

- `trading/exit_reasons.py`: new `ExitBranch` enum — `limit`,
  `market_fallback`, `sl_leg`, `sl_leg_race`, `last_resort` — plus
  `is_known_branch`. NULL means "pre-split row", which is legitimate and
  must not be alerted on.
- `persistence/database.py`: **Migration 15**, PRAGMA-guarded
  `ALTER TABLE trades ADD COLUMN exit_branch VARCHAR(24)`. Idempotent,
  matching the shape of migrations 2-14. On the live WAL DB an ADD COLUMN
  with no default is an O(1) header rewrite: it touches no page of row data
  and blocks no reader. **No backfill. Historic strings are untouched.**
- `StopExitEvent.exit_branch` + `build_exit_update` — the single funnel all
  five engines (BF, ORB, MACD wave, ignition, prestage) already drain
  through, so the branch reaches the DB by construction.
- `confirmed` now keys on the BRANCH. With the reason preserved,
  `exit_reason != 'stop_loss_unconfirmed'` would have reported a last-resort
  `trail_stop` as a confirmed fill.
- Two branches still SET the reason, deliberately: `sl_leg_race` (the
  broker's leg, not our order, executed the sale — a different exit, not a
  different route) and `last_resort` (`stop_loss_unconfirmed` is the string
  `orphan_reconciler.STALE_EXIT_REASONS` and `scripts/report_common` read to
  know a row needs follow-up).
- Retired as writers: `STOP_LOSS_MARKET_FALLBACK` (from StopMonitor — still
  written by `trading_engine._recover_exit_from_order_history`, which
  classifies from order history and genuinely cannot tell what fired) and
  `STOP_LOSS_FALLBACK`. Both keep their enum members and their historic rows.
- Fixed in passing: the limit-submit-failure path wrote a CONFIRMED exit at
  `trigger_price` when its poll timed out — a fabricated fill price and P&L.
  It is now `last_resort`.

## 2. FIX 7 — `unknown_exit` books nothing (`9823a7d`)

Four rows carry `exit_price == fill_price` and `pnl == 0.00` exactly —
BDMD 2026-03-20, NPT / FBYD / SVRN 2026-03-30 — hiding between −$1,684 and
−$6,800 of realized loss. NPT alone: entry $5.49, 16:52 close $4.96, 9,090
shares = **−$4,818 booked as $0.00**.

- `trading/unknown_exit.py` — `build_unknown_exit_update()` is the ONE spec.
  No `exit_price`, no `exited_at`, no `pnl`; `order_status =
  exit_pending_verification`. A last-known price may be recorded as
  `exit_trigger_price` for forensics; it never becomes the exit price.
- `trading_engine._sync_closed_positions` uses it. The order-history
  recovery ahead of it (GLXG 2026-06-11) is untouched and still heals the
  row into a real P&L when the sell shows up — there is a regression test.
- The row stays open **by design**: the orphan reconciler owns it, it stays
  in `get_open_trades`, and the daily green check HARD-fails while it is
  stuck. That is the alarm the fake $0 silenced.
- Because the sync now revisits it every cycle, the Telegram pages **once**
  per trade id (`_unknown_exit_alerted`) and logs thereafter.

## 3. FIX 1 — the broker SL leg IS the exit (`c03ab75`)

The bulk-cancel killed every open order for the symbol — the bracket SL leg
included — *before* submitting our own limit. Measured naked window across
the eleven events: **23.4 / 38.6 / 50.6 / 52.6 / 55.5 / 68.6 s**, on a
position being liquidated precisely because it is falling.

- `StopMonitor._exit_via_sl_leg`: when the watch has a LIVE SL leg covering
  the whole position, `replace_order_stop_price` moves its stop down to the
  marketable level (`bid − max(tick, 30% of spread)` — the same number the
  limit would have used) and the broker leg does the exit. The TP sibling is
  left alone; Alpaca's OCO retires it on the fill. Nothing is cancelled
  before the position is flat.
- Alpaca's replace mints a **NEW order id**; `watch.sl_leg_id` follows it.
  Polling the id you replaced is how the HOD-break engine lost sight of
  target fills.
- Every fall-through to the legacy cancel-and-place path is LOGGED with its
  reason: no `sl_leg_id`, leg not live, position re-query failed, broker
  flat, leg covers less than the position (it would strand the rest),
  replace rejected (the original leg is untouched and still protecting — the
  safe failure).
- If the repriced stop does not elect inside the poll budget we escalate
  exactly as before. **Protection lapses there, not at the trigger.**
- `_emit_stop_exit_event` is now the single emit site for both paths, so the
  `confirmed` rule and the telemetry payload cannot drift apart.

## 4. FIX 3 — a partial fill is a partial success (`1e5e876`)

`_poll_order_fill` refused to count `partially_filled`. In 9 of 11 events
the order was larger than the displayed bid (46.9× EEIQ, 14.2× RBNE, 7.1×
IRE, 4.2× EHGO), so the order that cleaned out the bid and stalled was
judged a TOTAL failure and the WHOLE quantity was market-ordered into the
hole the first slice had just made — and the DB booked the market order's
price against the full pre-cancel `shares` (≈ $67 of pessimism on EHGO).

- `_poll_order_fill` returns `(filled_qty, avg_price, status)`.
- `_poll_fill_price` keeps the old yes/no contract for the five call sites
  that ask about somebody else's order.
- The limit path books the partial, reads the order's FINAL state after the
  cancel (so a limit that completed during the cancel is not recorded as a
  `market_fallback` that never happened), then works only the remainder
  through `_verify_fill_qty` — reused, not reimplemented. The event carries
  the qty-weighted BLEND.
- A remainder that cannot be proven gone books only what is certain and goes
  to `exit_pending_verification`.
- `_verify_fill_qty` gained a `client` parameter: it used `self._alpaca`
  unconditionally, which from the stop-exit path would close an ORB
  remainder through the bull-flag account — a wrong-ACCOUNT order.

## 5. FIX 2 — the exit ladder (`c6b98ca`) — **DEFAULT OFF**

Slice to the displayed bid, price each rung at `bid − max(tick,
cross_factor × spread)`, re-price via `replace_order_limit_price` as the bid
moves (a replace keeps the order's place in line), escalate to
`close_position` only after `max_rounds` or `hard_deadline_s`. The event
carries the blend of every rung plus any escalated remainder.

Config `trading.self_managed_stops.exit_ladder` (config.yaml +
config.yaml.template + `Config.exit_ladder_cfg`):

```yaml
exit_ladder:
  enabled: false          # <- DORMANT
  slice_to_bid_size: true
  min_slice: 100
  cross_factor: 0.25
  reprice_after_s: 2
  max_rounds: 3
  hard_deadline_s: 10
```

The pricing and slicing rules are pure static helpers (`ladder_limit_price`,
`ladder_slice_qty`) so they are testable without a broker, and
`ladder_limit_price` can never price above the bid.

Also fixed here: `_escalate_to_market_close` no longer cancels or
race-checks an EMPTY stale order id — the ladder retires each rung as it
goes, and re-reading a completed rung would report that rung's 200-share
fill as the whole exit's price.

## 6. FIX 5 — no midpoint on an urgent exit (`42f4afe`)

EEIQ 2026-03-26 was priced by the tight tier: bid 7.67 / ask 7.72 → limit
**$7.70, three cents ABOVE the bid**. It sat 38.5 s and 9,375 shares were
market-closed at 7.6552. `compute_limit_price_from_quote(..., urgent=True)`
is now the default and rests AT the bid; the OFI and size escape hatches keep
their precedence. `urgent=False` still reaches the legacy tiers — gated, not
deleted — and an AST source-guard test fails if any production file asks for
them.

---

## 7. Tests

| file | tests | what |
|---|---|---|
| `tests/test_exit_reasons.py` | 46 (was 16) | branch catalog, escalation-tag mapping, payload, migration idempotency + historic-row immutability, FJET/HCAI replayed through the real `_execute_stop_exit` |
| `tests/test_stop_exit_unconfirmed.py` | 28 (was 13) | FIX 7 payload, the NPT row against a real `Database`, the old fabricated flat documented, the sweep stub's refusal to write |
| `tests/test_stop_exit_limit_buffer.py` | 47 (was 27) | FIX 1 decision logic + every fall-through + the rollback flag; FIX 3 tuple contract, blend, completed-during-cancel, unresolvable remainder |
| `tests/test_stop_exit_ladder.py` | 22 (new) | FIX 2 slicing, pricing, reprice, escalate-after-deadline, blend, no-bid bail-out, **4 rollback-contract tests** |
| `tests/test_spread_pricing.py` | 37 (was 28) | FIX 5 urgent default, the EEIQ quote, every spread tier at-or-below the bid, AST guard |
| `tests/integration/test_stop_exit_execution.py` | 18 (new) | NPT through the real `_sync_closed_positions`; the RBNE bracket lifecycle with a falling tape; the EHGO partial; RBNE thin-book ladder |
| `tests/fakes/fake_alpaca_broker.py` | — | grew `replace_order_stop_price` (shared `_replace`, Alpaca new-id semantics) and opt-in `allow_close_position` |

**Whole suite: 3,886 passed, 10 skipped, 0 failed** (`nice -n 10 python3 -m
pytest -q`, 5m37s). Note: the suite needs `ulimit -v 6000000`, not
1800000 — at 1.8 GB of address space `test_ignition_engine.py` cannot start
its worker threads (`RuntimeError: can't start new thread`). That is an
environment limit, not a code failure; the file passes 40/40 unconstrained.

---

## 8. Deploy steps (for when the owner gives the word)

Nothing below has been run.

1. **Rehearse on the weekend node first** (`feedback_predeploy_rehearsal_protocol`):
   boot `main.py` with the new code, confirm `Migration 15: added
   exit_branch column to trades` appears once in the log and not again on the
   second boot, and confirm `StopMonitor STARTED`.
2. Verify the config reads:
   ```bash
   python3 -c "from config import Config; c=Config(); print(c.prefer_sl_leg_exit, c.exit_ladder_cfg)"
   # -> True {'enabled': False, ...}
   ```
3. `git pull` on the trading node, then `sudo systemctl restart onemil-trader`.
4. The migration runs itself at the first `Database()` construction. It is
   idempotent and safe on the live WAL DB; no downtime, no backfill.
5. **Leave the ladder off.** Flip `exit_ladder.enabled: true` only after at
   least 10 real exits have been observed under FIX 1 + FIX 3 and
   `slip_vs_bid_bps` has been read.
6. BF and ORB are currently PAUSED (`trading.enabled: false`,
   `orb.yaml strategy.enabled: false`), so the first exits this code sees
   will be whatever the owner resumes first. That is a feature: the fixes
   land before the volume does.

### Rollback

| symptom | action |
|---|---|
| SL-leg replace misbehaves at the broker | `trading.self_managed_stops.prefer_sl_leg_exit: false` + restart. Zero state — the next exit uses the pre-2026-09-17 cancel-and-place path. |
| ladder misbehaves | it is already off; if it was turned on, `exit_ladder.enabled: false` + restart. |
| anything else | `git revert` the offending commit and restart. The `exit_branch` column can stay — it simply stops being written. |

There is no state to unwind in any of these: every change is inside one
exit's lifetime.

---

## 9. Live metrics to watch

```bash
journalctl -u onemil-trader | grep -E "EXIT BRANCH|SL LEG REPLACED|EXIT LADDER|PARTIAL"
```

1. **`slip_vs_bid_bps`** — logged on the `EXIT BRANCH` line of every exit.
   `(fill − bid_at_pricing) / bid × 1e4`. Today's 11 events: median **−58
   bps**, worst **−229** (RBNE). **Target median ≥ −25.** This is the single
   number the work is judged on.
2. **`naked_window_ms`** — SL-leg-cancel → confirmed flat. Today up to
   68,596 ms. On a bracketed exit that takes the `sl_leg` branch it is **0**
   by construction (nothing is cancelled before the fill). A non-zero value
   on an `sl_leg` row is an alert. The escalation path still opens a window
   — that is the honest limit of FIX 1.
3. **`exit_branch` histogram** —
   ```sql
   SELECT exit_branch, count(*), round(sum(pnl),2) FROM trades
   WHERE exited_at >= date('now','-7 day') GROUP BY exit_branch;
   ```
   `limit` must dominate; `market_fallback` must fall; `last_resort` should
   be zero. A NULL branch on a StopMonitor-era row means a writer was
   missed. Until this histogram has data, remember that
   `stop_loss_market_fallback` was meaningless.
4. **Zero-P&L exits** — `SELECT count(*) FROM trades WHERE pnl = 0.0 AND
   exit_price = fill_price`. Any NEW row is a FIX 7 regression.
5. **Rows stuck in `exit_pending_verification`** — the green check already
   HARD-fails on these. Expect the count to be able to RISE now: that is FIX
   7 working, not breaking. Each one is a real unattributed exit that used to
   be invisible.
6. **`qty_over_bid_size`** at pricing time — logged on the `EXIT LADDER`
   line when the ladder runs. It predicted 9 of the 11 events; above 3× is
   where the money went.

---

## 10. What this does NOT fix

* The **poll-mode StopMonitor** (paper nodes only) is untouched.
* FIX 4 is not built: `_on_trade` still has no SIP condition filter and no
  NBBO sanity check, so a single aberrant print can still trigger an exit
  (RCAT 2026-06-29, 9.11 against a 9.78/9.79 quote).
* FIX 8 is not built: a row written at submit and never given a terminal
  state is still silent.
* The **ladder is dormant**, so the +$797 decidable counterfactual is not
  being collected yet. FIX 1 + FIX 3 + FIX 5 are what is live.
* None of the counterfactual dollars are a forecast. They are a census of
  what happened to 11 events on 1-minute bars, with obtainability checked;
  `REPORT.md` §4 states the caveats and marks EHGO's +$285 undecidable.
