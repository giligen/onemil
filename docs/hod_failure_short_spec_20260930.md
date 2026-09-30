# HOD-break failure-short — order mechanics spec (2026-09-30)

Paper trading only, behind `config.yaml hod_break.failure_short.*` (ALL default OFF/safe). This
file covers ORDER MECHANICS only. The signal (probability of a failed breakout) is a separate,
pluggable module (`trading/hod_failure_features.py`, owned elsewhere) handed in as `fs_signal_fn`.

## Rule
Our own HOD-break long fills. At the close of the bar AFTER the fill (fill+1), evaluate
`signal_fn(symbol, bars_through_fill_plus_1, arm_context) -> float | None`. If `P >= tau` and every
rail passes, submit a marketable SELL of 2x the long's shares at the close of bar
fill+`entry_bar_offset` (default fill+2) — closes the long, opens a short of the long's own size —
then place the short's protective stop-buy (day's high-so-far + $0.01) and target buy-limit (the
long's own stop level), with an EOD cover at 15:55 ET.

## Config keys (`hod_break.failure_short`, config.py `hod_break_cfg` whitelist)
`enabled` (false), `telemetry_only` (true), `tau` (0.6), `risk_usd` (150), `max_concurrent` (3),
`max_per_day` (8), `day_kill_r` (-5.0), `require_etb` (true), `entry_bar_offset` (2),
`allow_fresh_short` (false), `ledger_path` (`logs/hod_failure_short_ledger.csv`).

## Order flow (`trading/hod_break_engine.py`, `_fs_*` methods)
1. `_fs_register_long` (hooked at the end of `_on_live_fill`, non-partial fills only): arms tracking
   keyed by the fill's minute-of-day.
2. `_fs_advance` / `_fs_evaluate_signal` (close of fill+1): calls `signal_fn`, checks rails, writes
   one ledger row per evaluation (both modes). `telemetry_only` stops here — zero orders.
3. `_fs_submit` (close of fill+`entry_bar_offset`): `submit_market_sell_order(symbol, 2x shares)`,
   `_record_exit(long_pos, reason='failure_reversal')`, `db.save_trade` for the NEW short row
   (`side='sell'`, `strategy='hod_break'`, `pattern_data.mechanism='failure_short'`), then
   `submit_stop_limit_order(side='buy', ...)` (protective) and `submit_limit_buy_order(...)`
   (target), sized to the broker's signed qty via `exit_qty_guard.get_signed_broker_qty`.
4. `_fs_poll_short_exits` / `_fs_check_eod_cover`: reconcile the two resting legs each tick with
   SHORT-signed pnl (`entry - exit`, never the long formula); EOD cancels both legs and covers.

## Rails (`trading/hod_failure_short.py::rails_reason`, pure, unit-tested)
In order: shortable -> easy-to-borrow (if `require_etb`) -> SSR proxy (price >= 90% of prior
close; unknown prior close fails CLOSED) -> `max_concurrent` -> `max_per_day` -> `day_kill_r`
(day-realized R across failure-short trades).

## Known deviation from the literal spec text (evidence-based, flagged not silent)
The spec text says "StopMonitor watch" for the short's stop-buy. `trading/stop_monitor.py` is
grep-verified SELL-only (every exit fires `submit_stop_sell_order` / `submit_limit_sell_order` /
`close_position`) and its `add_watch` cross-detection is long-oriented (fires on price falling
to/through `stop_price`). A short's protective stop sits ABOVE the entry price, so registering it
the same way would read as "already through the stop" the instant it arms. The short's real
protection is instead the two broker-resting orders in step 3 above, tracked in the engine's own
`_fs_shorts` dict — StopMonitor is deliberately NOT armed for the short leg.

## Telemetry
`logs/hod_failure_short_ledger.csv` (configurable path): one row per evaluation — date, ts,
symbol, fill_bar, p, tau, rails_reason, passed, would_be_qty/stop/target, telemetry_only.

## Rollback
`config.yaml hod_break.failure_short.enabled: false` (already the default). No restart-required
state; the overlay reads its config fresh from `cfg` at engine construction only.

## Not done in this pass
`_confirm_fill` (the `next_open` entry-mode long-fill path) is not wired — only `_on_live_fill`
(the live `resting_stop_limit` config) arms the overlay. `allow_fresh_short=true` (short 1x when
the long already stopped before the signal bar) has pure sizing (`fresh_short_qty`) but no engine
wiring — out of scope for v1 ("first version trades only reversals of our own fills").
