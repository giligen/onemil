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

## Signal wiring (2026-09-30 follow-up): `trading/hod_failure_signal.py`
`build_signal_fn(cfg)` loads `hod_break.failure_short.model_path` (default
`research/hod_entry/models/ff10_k1_val.joblib`, the VAL-half model) and
`.../ff10_k1_features.json` (152 ordered columns) ONCE at boot, and returns the
`signal_fn(symbol, bars, arm_context)` closure `main.py` passes to `HodBreakEngine(fs_signal_fn=...)`
— built only when `failure_short.enabled` (both telemetry_only sub-modes still evaluate/log; the
model load is skipped entirely, not just gated, when the overlay is off). Every column is scored
via `trading.hod_failure_features.k1_features` — the SAME function
`research/hod_entry/1675_forward.py` (the research scorer) calls — fed the engine's own bars
(`hod_break_engine.py::_fs_evaluate_signal`, canonical `{'o','h','l','c','v','minarr'}` dict, index
0 = first RTH bar of the day, through the signal bar), the arm context (level/stop/fill/target),
`adv20` (engine's ADV map) and `atr14` (`data/cache.db daily_bars`, read-only, 15 prior sessions,
ported verbatim from `1675_forward.py::atr14_causal`, memoized per symbol-day for the process
lifetime). F11-F15 are ported from `research/hod_entry/1667_sweep.py`'s formulas (level-vs-VWAP,
level-vs-open, day-range/ATR, level age, dollar-volume-vs-normal) computed from the engine's own
bars instead of `bars_sip.db`; **F8** (F15's denominator) isn't available live and is mapped to the
engine's `adv20` — a disclosed approximation, not a parity bug (F15 is excluded from the parity
test for this reason). `cS5_spyret_1` is structurally always NaN (no SPY bar feed wired) — logged
once at boot, not per evaluation. A NaN storm (> 20/152 columns) returns `None` and logs ERROR once
per session day.

## OCO cover for the short (2026-09-30 follow-up)
The short's two exits are submitted as ONE `submit_oco_buy_order` (new, mirrors
`submit_oco_sell_order` exactly: `order_class='oco'`, `side=BUY`, `take_profit`=the target limit,
`stop_loss`=the day-high stop; parent order id = the limit leg, `legs` carries the stop leg) —
never two independently-resting buy orders on the same short qty, the same class of bug the 9/25
VECO incident fixed on the sell side. If the OCO submit itself raises, `_fs_submit` falls back to
the two independent orders (`submit_stop_limit_order` side='buy' + `submit_limit_buy_order`), and
`_fs_poll_short_exits` cancels the sibling leg the instant either one shows `status == 'filled'`
(this cancel-on-fill logic is unconditional — a no-op against an already broker-cancelled OCO
sibling, the real safety net when the fallback pair is used). `_fs_shorts[symbol]['is_oco']` records
which path was taken, for telemetry only.
