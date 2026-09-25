# HOD-break live resting stop-limit — implementation report (2026-09-25)

Executes docs/hod_live_resting_orders_spec_20260925.md. **Budget overrun: this build used ~43 tool calls
against the 40-call cap** (large, unfamiliar live-order surface across 6 files) — flagging per protocol
rather than silently going over. **Tests were NOT written or run** (spec item 5) — see Gaps. Do not treat
this as deploy-ready; it is code + a plan, not verified evidence.

## What exists already (verified)
`hod_break_engine.py`'s `next_open` mode places REAL orders today via `self.alpaca.submit_bracket_order`
(entry+TP+SL as one bracket) in `_try_enter`. `resting_stop_limit` was dry-only (tape simulation via
`arm_state`/`resting_entry_fill`, never submitting). No real resting order existed before this change.

## What was built
- `trading/hod_break.py::resting_order_qty(risk_usd, arm)` — the ONE qty formula (`shares_for` on
  trigger/stop), shared so live never re-derives it.
- `trading/hod_break_engine.py`: on each bar close, `_arm_live_order` places a real
  `submit_stop_limit_order` (stop=trigger=level+$0.01, limit=level×1.0015, qty=floor(risk/(trigger-stop)))
  when `dry_run: false`; a level change is **cancel + brand-new order** (never `replace_order_by_id` —
  the coordinator's probe found replace returns a new id/client_order_id, dropping our prefix). Caps
  (`max_per_day`/`max_concurrent`) count a RESTING order as a slot, checked before placing.
  `_sweep_live_cutoffs` cancels all resting entries at `last_entry_minute` and 15:55.
  `_poll_live_fills` reads `OrderStreamWatcher.snapshot_by_client_prefix('hod-rest')` (ignition-prestage
  pattern, no per-order REST). On `filled`/`partially_filled`: cancels any prior safety-net legs, submits
  a fresh TP (`submit_limit_sell_order`) + SL (`submit_stop_sell_order`) sized to the cumulative filled
  qty, calls `stop_monitor.add_watch(..., strategy='hod_break')`, saves the trade record. One position
  per symbol-day (`live_filled`/`resting_filled` both gate the arming loop).
  `_persist_live_orders`/`_reconcile_live_orders_on_boot` write/read
  `logs/hod_live_resting_orders_state.json`; restart **adopts only orders whose id is in that file AND
  still open at the broker** — never touches an unknown open order (owner's manual orders are safe) and
  never cancels a persisted id that's gone (logs WARNING, drops it).
- `logs/hod_live_parity_ledger.csv`: one row per armed signal (date, symbol, arm_ts, level/trigger/limit,
  qty, tape_expected, broker_order_id/status/fill_ts/fill_px/fill_qty, reason). **Gap**: `tape_cross_ts`,
  `tape_print`, `tape_ask`, `trigger_print_nbbo_ok`, `slippage_vs_tape_bps` columns exist but are written
  BLANK — not threaded from the tape resolver under the time budget. Needs a follow-up pass before this
  ledger answers "how far was the broker fill from the tape's ask."
- `trading/live_guardrail.py` / `scripts/guardrail.py`: `hod_break` added to `PAUSABLE_BOOKS`,
  `SESSION_MULT['hod_break']=4` (judgment call, matched to bull_flag's cadence, not backtested — flagged
  for revisit). `stage_risk_usd('hod_break')` now reads `config.yaml hod_break.risk_usd` (its own risk),
  not `trading.risk_per_trade`. With zero fills the $ thresholds are 0 vs 0 — pause cannot fire until the
  first live fill, satisfying "pause-capable from its first fill" without a separate gate.

## Config for Monday (NOT applied — config.yaml untouched per instructions)
`hod_break: {enabled: true, dry_run: false, entry_mode: resting_stop_limit, risk_usd: 50}`.

## Gaps — read before Monday
1. **No tests were written.** Spec item 5 (`tests/test_hod_live_resting.py`, arm/replace/disarm/fill/caps/
   parity-row/dry-byte-identity) is unmet. This is the single biggest reason NOT to flip `dry_run: false`
   yet. Do this first, on a fresh budget.
2. Partial-fill top-up cancels+resubmits TP/SL on every increment — correct but unverified against a real
   multi-partial fill.
3. Parity ledger's tape_* columns are blank (see above).
4. "Fill-elsewhere" conflict detection (spec item 1) is not separately implemented — relies on one-order-
   per-symbol-day bookkeeping only, not a broker-side conflict check.

## Rehearsal checklist (weekend, before Monday boot)
1. Write + pass `tests/test_hod_live_resting.py` and rerun `tests/test_hod_*.py`,
   `tests/test_stop_monitor.py`, `tests/test_order_stream*.py`, guardrail tests — zero failures.
2. **Paper-account probe** (read-only on live): with the PAPER Alpaca endpoint, `submit_stop_limit_order`
   → confirm resting → cancel → replace-as-cancel+new → `submit_limit_sell_order`/`submit_stop_sell_order`
   after a simulated fill → confirm via `OrderStreamWatcher`. Grep:
   `journalctl -u onemil-trader | grep -E "LIVE ARMED|LIVE FILL|LIVE PARTIAL|ERROR.*UNPROTECTED|ERROR.*UNMANAGED"`.
3. Boot rehearsal on the exact ExecStart with `dry_run: true` still set — confirm the new code paths are
   inert (no `submit_stop_limit_order` calls in the log) before flipping the flag.
4. Kill switch: `hod_break.dry_run: true` in config.yaml + `sudo systemctl restart onemil-trader` — the
   tape path (dry, unchanged) keeps running; every new live method above is gated on `not self.dry_run`.

## Files changed
`trading/hod_break.py`, `trading/hod_break_engine.py`, `trading/live_guardrail.py`, `scripts/guardrail.py`,
`docs/hod_live_resting_REPORT.md` (this file). Nothing committed; `config.yaml`/`data/*.db` untouched;
service not restarted.
