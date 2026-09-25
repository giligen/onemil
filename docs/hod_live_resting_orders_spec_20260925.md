# SPEC — HOD-break resting stop-limit: REAL orders at $50 risk with a per-signal expected-vs-actual ledger

Owner 2026-09-25 17:05 UTC: "if you need real live one with small orders... $50 $100 whatever I approve as well if this
helps to close the gap between dry-run/theory and live — and expose any issues". Target: Monday 2026-09-28 boot, after
a weekend boot rehearsal on the exact ExecStart. Zero orders until then.

## What exists (build on it, do not fork)
`hod_break.entry_mode: resting_stop_limit` (dry): `trading/hod_break.py::arm_state` / `resting_entry_fill` (ONE fill
rule), `trading/hod_break_engine.py::_evaluate_resting` (arms at bar j close, subscribes the print watch),
`_on_trade_print` (first print ≥ trigger → tape fill decision), `logs/hod_dry_entry_ledger.csv`,
`trading/stop_monitor.py` print-watch pool, `trading/live_guardrail.py` (auto-pause), the HOD engine's existing
kill rails (`kills=-600/-1500`, `per_day=12`, concurrency 4) and its live order path for `next_open` (grep
`hod_break_engine.py` for the submit / OrderExecutor calls used when `dry_run` is false — verify it exists and what it
places; if the engine has never placed a real order, say so in the report and build the minimum path below).

## The live path (Sonnet; ≤ 40 tool calls; TDD; do not commit; do not start/restart any service; no DB writes)
1. When `dry_run: false` AND `entry_mode: resting_stop_limit`: on ARM, place a real BUY STOP-LIMIT at Alpaca via the
   existing executor/client (stop = trigger = level + $0.01, limit = level × 1.0015, qty = floor(risk / (trigger −
   stop_loss)), time-in-force day, extended hours off); cancel/replace at the next bar close if level/limit changed or
   the arm was lost; cancel on fill-elsewhere / after the first fill of the day (one position per symbol-day);
   cancel ALL resting HOD orders at `last_entry_minute` and at 15:55. Respect `per_day` and the 4-concurrent cap
   BEFORE placing (a resting order counts as a slot only once filled? NO — count it as a slot while resting, so the
   fill can never exceed the cap; document this choice; it matches the backtest's first-12/day 4-concurrent rule
   only approximately — report the difference).
2. Fill detection via `OrderStreamWatcher` (trading/order_stream.py): on fill, register the position with the
   StopMonitor exactly as the `next_open` live path would (stop = consolidation low, target = fill + target_r × R,
   flat at 15:55), with the HOD tag so `scripts/hod_dry_ledger.py` / EOD attribution see it as `hod_break`.
3. The EXPECTED-vs-ACTUAL ledger (`logs/hod_live_parity_ledger.csv`), one row per armed signal per day: symbol, arm_ts,
   level, trigger, limit, qty, tape_cross_ts, tape_print, tape_ask, tape_expected (FILL/NO_FILL/NO_CROSS),
   broker_order_id, broker_status (filled/cancelled/expired/rejected), broker_fill_ts, broker_fill_px,
   broker_fill_qty, slippage_vs_tape_bps = (broker_fill_px − tape_ask) / tape_ask × 1e4, and reason codes. The tape
   side is the dry logic already running; both are written for every arm whether or not the broker filled.
4. Risk: `hod_break.risk: 50` for the first live week (config, not code); the live guardrail's per-session and
   trailing rules apply to `hod_break` from its first fill (today it is report-only — make it pause-capable like the
   others, same thresholds scaled to the $50 risk).
5. Tests (`tests/test_hod_live_resting.py`, `MagicMock(spec=...)`, fixtures in conftest): arm places one stop-limit
   with the right prices/qty; re-arm with a new level replaces it; disarm cancels; a fill registers the position and
   cancels nothing else; the second cross the same day places nothing; caps respected while resting; the parity
   ledger row is complete for FILL, NO_FILL and NO_CROSS cases; `dry_run: true` places nothing (byte-identical to
   today). Run `tests/test_hod_*.py`, `tests/test_stop_monitor.py`, `tests/test_order_stream*.py`, the guardrail
   tests — zero failures.
6. `docs/hod_live_resting_REPORT.md`: files, config for Monday (`dry_run: false`, `risk: 50`, `entry_mode`), the
   rehearsal checklist (weekend boot; a paper-account probe of stop-limit placement/cancel/replace with the real
   alpaca-py client against the PAPER endpoint — read-only on live), the grep lines, the kill switch (`dry_run: true`
   + restart), and what remains unknowable until Monday. Return ≤ 150 words.
