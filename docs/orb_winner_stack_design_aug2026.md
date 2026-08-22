# ORB "Winner Stack" — ATR Stop-Floor + Scale-Out — Design
**Status: DESIGN (2026-08-22, owner-ordered: "design go deep on 1 and 2, implement after design, add test, validate roll out"). Ships as TWO independent flags; combined = the validated C-point.**

## 0. What ships and what it buys (validated, research/stability/)
- **SZ1 ATR stop-floor (k=0.25)**: initial stop = max(range_low, entry − 0.25×ATR14).
  Alone: $536→$550/mo, greens 12→13, MDD −1,185→−966, worst −407→−330, zero
  monster cost (phaseB_regime_atr.md).
- **Scale-out 40% @ +3.0R**: sell 40% at entry+3R (range-R), runner keeps the
  static lock unchanged. Stacked with SZ1 = the frontier C-point:
  **$455/mo avg, 15/20 green, MDD −700, worst −309** (_frontier_monthly.json;
  monthly table shown to owner 8/16). Owner accepted the −$81/mo average cost
  for +3 green months on 8/22.

## 1. FROZEN SEMANTICS (verbatim from the validated harness — deviations are bugs)
1. **ATR14** (phaseB_regime_atr.py:47-62): TR = max(H−L, |H−prevC|, |L−prevC|)
   on DAILY bars; ATR14 = simple rolling mean of 14 TRs; the value used on
   trading day T is the ATR **ending T-1** (shift 1 — no lookahead). If fewer
   than 15 daily bars exist → ATR unavailable → **floor not applied** (stop =
   range_low, log WARNING with reason) — fail-open to legacy, mirroring the
   study which had NaN ATR rows unfloored.
2. **Floor applies to the PROTECTIVE STOP ONLY**:
   - stop_initial = max(range_low, entry − 0.25×ATR14)
   - **Sizing UNCHANGED**: shares = risk_usd / (range-based stop pct, min 1%)
     exactly as today — when the floor binds, realized $risk/trade drops below
     $375; we do NOT resize up (study kept book sizing fixed).
   - **Lock machinery UNCHANGED**: R = range_high − range_low for arm (1.75R)
     and lock (+0.5R) levels. Touchgo Rule D's R likewise unchanged.
3. **Scale-out** (resim_exit.py::variant_scale): level = entry + 3.0×R
   (range-R); sell 40% of filled shares (floor to int; if scale qty < 1 →
   no scale, all-runner); fill assumption = limit at level, 10bps slip;
   pessimistic same-bar ordering (a bar that hits BOTH the stop and the scale
   level counts as stopped first unless the lock armed above the stop earlier
   or that same bar); runner (60%) continues with the ORIGINAL stop/lock
   levels; touchgo (tag_bb/tag_b1) prefires exit the WHOLE position (scale
   never happens on touchgo trades); 15:45 force-close covers the runner.
4. **Both flags independent**: atr floor can run without scale and vice versa;
   the combined semantics are literally the two rules composed (validated
   combination is the C-point book).

## 1b. P0 RESOLUTIONS (v2 — docs/orb_winner_stack_review_aug2026.md is binding)
- **P0-1 same-bar ordering CORRECTED**: the frozen harness FILLS THE SCALE on a
  bar hitting both stop and +3R (stop check is gated `low<=stop AND
  high<scale_px`). §1.3's earlier "stopped first" prose was BACKWARDS. NCNA
  2025-08-21 is the mandatory golden (+$158 under correct semantics). Live tick
  ordering can only be conservative vs this (<= BT on stop-first sequences) —
  documented, tracked via P1 telemetry.
- **P0-2/P0-3 single-writer rule for the two-part exit**: `pnl` is written
  non-NULL exactly once, at FINAL close, as combined scale_pnl + runner leg, on
  EVERY path — `_handle_exit_event` (must NOT pop/close the row on a scale
  partial: new event type routes to a scale-fill branch that updates
  scale_* columns + watch shares only), `_sync_db_after_fc`, the orphan
  reconciler, and all SQL fallbacks (`_row_pnl`, `cumulative_orb_since`,
  `_realized_orb_pnl`) audited with tests. `filled_qty` stays ENTRY qty.
- **P0-4 restart mid-scale**: the floored stop is written to BOTH
  `stop_loss_price` (plan) and `real_stop_loss_price`; rehydration subtracts
  `scale_qty` from shares and restores scale_done from the scale_* columns.
  Kill -9 mid-scale drill mandatory.
- **P0-5 collision protocol**: the resting scale order is cancelled BEFORE any
  stop/touchgo/force exit executes (cancel-confirm-or-adopt: cancel reject =>
  poll => a filled scale books its columns first, then the full exit proceeds
  on runner qty); scale submission must NOT hold `_exit_in_progress` through a
  blocking fill-wait (async poll like entry orders); `replace_order_qty`'s NEW
  order id captured into sl_leg_id; leg-resize-ok/submit-fail compensates by
  restoring leg qty; NEVER convert the OCO safety-TP leg into the scale order
  (independent limit sell after leg resize).
- **P0-6 ATR data parity**: backfill daily_bars history for symbols with
  cache gaps (the 6 book trades unfloored only by missing data: SMST, ARQQ,
  BTQ, RGTZ, FJET, PS — after backfill the regenerated reference book floors
  them, so gate-2's target is the REGENERATED book, recomputed at validation
  time, not the stale artifact); EoD dive gains a floored-stop drift check
  (live floored_stop vs BT recompute per trade); Monday flip is GATED on the
  regen succeeding and reproducing validation gate 2 — regen failure => flags
  revert, ship postponed. ATR boundary frozen as the harness behaves
  (rolling(14).mean() over TRs, shift(1); "insufficient history" = whatever
  yields NaN there) with a boundary golden at exactly 14/15 bars.

## 2. Live implementation
- **Shared module `trading/orb_winner_stack.py`** (parity by construction —
  imported by BOTH `trading/orb_engine.py` and `study_orb_pipeline_static_lock.py`):
  - `atr14_t1(daily_bars) -> Optional[float]` — the exact study formula.
  - `floored_stop(range_low, entry, atr14, k) -> (stop, floored_bool)`
  - `scale_levels(entry, range_size, frac, level_r) -> (scale_px, scale_qty_fn)`
- **Engine, entry path**: ORB already fetches per-symbol daily history for
  features (rv20/20d stats — verify window ≥ 15 bars; extend fetch window if
  needed). At submission: compute stop via floored_stop; use it for
  `real_stop_loss_price` (DB) + StopMonitor watch stop. Sizing code path
  untouched. Log per entry: `[ORB] ATR FLOOR SYM: range_low=X atr14=Y
  floored_stop=Z (bound|unbound)`.
- **Engine/StopMonitor, scale path**: extend WatchEntry with
  `scale_at_px / scale_frac / scale_done`. StopMonitor tick+poll paths: when
  bar/quote high ≥ scale_at_px and not scale_done and lock/stop hasn't fired
  first (existing precedence code order — stop checks already run before this
  new check within a cycle, matching the pessimistic model), submit a LIMIT
  SELL for scale_qty at scale_px through the same order-submission path
  force_exit uses (partial variant: new `partial_exit(symbol, qty, limit_px,
  reason='scale_out')`); on fill: reduce watch shares to the runner qty,
  mark scale_done, persist. **Safety-leg interaction**: ORB entries carry
  broker-side safety legs holding qty — the partial sell must release/resize
  legs first (reuse the orphan-reconciler's release-held-shares pattern);
  runner keeps a resized safety leg. If leg-release fails → skip scale this
  cycle, WARNING, retry next cycle (never strand shares).
- **DB model for the two-part exit**: SAME row. New columns
  `scale_qty INTEGER, scale_price REAL, scale_pnl REAL, scaled_at TEXT`
  (nullable — absent = no scale). Final `pnl` = scale_pnl + runner leg pnl
  (computed at final exit as today from exit_price × runner shares + stored
  scale_pnl). `exit_reason` stays the runner's reason. Report layer
  (`_row_pnl`, green check, sizing attribution, weekly) audited for the new
  columns; a scaled-but-still-open position's realized scale_pnl counts
  toward kill rails ONLY at final close (same as today's realized-only
  convention — document).
- **Config (orb.yaml + template)**:
  `exit.atr_stop_floor: {enabled, k: 0.25}` and
  `exit.scale_out: {enabled, frac: 0.40, level_r: 3.0}`; env kills
  `ORB_ATR_FLOOR=0`, `ORB_SCALE_OUT=0`. Frozen-params amendment note appended
  to research/orb_bplus_frozen_params_aug2026.yaml (owner order 8/22 +
  study citations) — constants may be LOOSENED never tightened.
- **BT pipeline**: implements both flags via the SAME shared module inside
  its exit walk; `load_bt_config` picks the flags up from orb.yaml → the
  nightly book regenerates with flags ON automatically at rollout.

## 3. Validation gates (all must pass before rollout)
1. **Byte-identity OFF**: flags off → pipeline reproduces the current book
   EXACTLY ($10,715.68 / 81 trades) and engine behavior unchanged (existing
   parity tests still green).
2. **Study reproduction ON**: flags on → pipeline book matches the frontier
   C-point (total ≈ $9,093, 15/20 green, per the monthly table) within
   rounding; SZ1-only matches B-point ($11,004 / 13 green). Tolerance: <$5
   per month absolute (fill-model identical, so it should be near-exact).
3. **Parity tests**: shared-module goldens (ATR values vs study CSV rows;
   floored stops on 10 known trades; scale fills on known paths incl. the
   pessimistic same-bar case, touchgo-prefire case, tiny-qty no-scale case).
4. **Live-mechanics tests**: partial_exit leg-release success/failure/retry,
   watch shares reduction, DB two-part row lifecycle, kill-rail accounting,
   restart resume mid-scale (watch rehydrates with scale_done state from DB).
5. Full suite zero failures.

## 4. Rollout
- Build + tests tonight (Sat). Sunday 10:06 rehearsal EXTENDED: boot with
  flags ON in a config copy (enabled stays false in live orb.yaml until GO),
  verify init lines + one replayed scale drill.
- **Monday pre-boot (with the GO-session window)**: flip both flags in
  orb.yaml, regenerate analysis_results/orb_bplus_book.csv (pipeline, flags
  on) so the nightly parity gate compares against the NEW reference book;
  Telegram the new expectation set ($455/mo, 15/20 green, worst −309).
  Validation-month bookkeeping: days 1-5 ran on the base book (all zero-trade
  — conveniently identical under both books); days 6-30 run on the C-book.
- Rollback: each flag independently to false + restart (zero state; an
  open scaled position's runner is still just a watch with a stop — safe).
- Monitor: `journalctl -u onemil-trader | grep -E "ATR FLOOR|SCALE OUT"`;
  EOD dive gains both lines; green check parity covers via the regenerated
  book automatically.
