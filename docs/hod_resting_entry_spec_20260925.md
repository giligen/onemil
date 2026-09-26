# SPEC — HOD-break resting stop-limit entry in the LIVE engine (dry run first), 2026-09-25

## Why
Cells 1,423/1,427 (`research/hod_entry/REPORT_1427.md`): a buy-stop-limit resting at the HOD level + $0.01 with a
15 bps limit, filled at the NBBO ask at the first cross inside the break bar, earns +0.29 / +0.24 / +0.33 R net per
fill on 2025H2 / VAL / sealed TEST (two independent builds, two tapes), ~30 % fill rate, ~34 fills/week after slots;
the unfilled breaks average −0.57 R. Cell 1,438 (`PREREG_1438.md`) restates the rule so it uses only information
available when a live order fills (armed at the prior bar's close). The live engine must implement THAT rule, in dry
mode first (zero orders), so 10 sessions of live fills/no-fills can be compared with the backtest.

## The rule (one spec, backtest and live — `trading/hod_break.py` is the shared module; research reads it too)
* At each closed 1-minute RTH bar j of a universe symbol (the existing HOD-break universe and `HodBreakParams`):
  ARMED for bar j+1 iff `consolidation_low(l, h, j)` is not None, level = HOD through j, level ≥ open × (1 +
  min_dist_open_pct/100), `rv_profile(cumv[j], adv20, m[j])` ∈ [rv_lo, rv_hi), m[j+1] ≤ last_entry_minute, and no
  fill yet today for the symbol.
* While armed: a buy-stop-limit rests at trigger = level + $0.01, limit = level × 1.0015 (new param
  `entry_limit_pct: 0.0015` beside the existing `cap`); it is cancelled/re-placed at the next bar close if the
  conditions change (level moves, arming lost) and cancelled after a fill.
* Fill = the ask at the first print ≥ trigger if ask ≤ limit; else the order does not fill (no chase). Stop =
  consolidation low; target = fill + target_r × (fill − stop); flat at 15:55; the existing walker/exit logic.
* Config: `hod_break.entry_mode: next_open | resting_stop_limit` (default `next_open` = today's behaviour,
  byte-identical); `dry_run` stays true.
* DRY MODE must be informative: on every arm log `[HOD DRY] ARMED {sym} level {L} trigger {T} limit {Lim} stop {S}`;
  on the first print ≥ trigger (from the live trade/quote stream the engine already consumes) log
  `[HOD DRY] CROSS {sym} at {HH:MM:SS.mmm} print {p} ask {a} -> FILL {a} | NO FILL (ask > limit)` and append a row to
  `logs/hod_dry_entry_ledger.csv` (date, symbol, arm_ts, cross_ts, level, trigger, limit, ask, filled, fill_px, stop,
  target); the dry walker then tracks the position to its exit as today, so `scripts/hod_dry_ledger.py` shows the
  book. If the engine only has 1-minute bars and not a print stream for a symbol, log WARNING and use the bar's high ≥
  trigger with the minute's first quote — and count those rows separately (they are not tape-accurate).

## Rules (Sonnet; ≤ 40 tool calls; TDD; do not commit; do not start/restart any service; no DB writes)
* Tests (`tests/test_hod_resting_entry.py`, fixtures in `tests/conftest.py`, `MagicMock(spec=...)`): arming at the
  boundary of each condition; re-arm after a no-cross bar; one fill per symbol-day; fill at ask ≤ limit, no fill
  above; the dry ledger row; `entry_mode: next_open` byte-identical (parity on a recorded day); and a PARITY test
  that `trading/hod_break.py`'s arming function and the research rule in `research/hod_entry/causal_arming.py` (if
  present; else the PREREG_1438 prose) agree on a synthetic day. Run `tests/test_hod_*` — zero failures.
* Write `docs/hod_resting_entry_REPORT.md`: files, config keys, the grep lines for the dry sessions, the rehearsal plan
  (weekend boot with `entry_mode: resting_stop_limit`, `dry_run: true`), and the daily comparison the main session
  runs (live fill rate and fill mean vs TEST's 27.8 % / +0.33 R). Return ≤ 150 words.

## Forward-instrument counterfactuals (2026-09-26, `hod_break.log_counterfactuals`, default OFF)
Research closed the resting-order HOD-break book as money (2026-09-26, 23 cells 1,445-1,467); the dry run stays open
purely to keep collecting point-in-time evidence. Three columns, zero effect on any gate/size/order:
1. `scanner_qualified_at_arm` (0/1/blank) — was the symbol already in the live scanner's `_qualified_symbols` when
   this book armed it. Wired via an `is_qualified: Callable[[str], bool]` constructor kwarg on `HodBreakEngine`
   (`trading/hod_break_engine.py`); `main.py` passes `lambda s: trading_engine is not None and s in
   trading_engine._qualified_symbols` — a late-binding closure since `trading_engine` may not exist yet when the HOD
   engine is constructed. `is_qualified=None` while the flag is on logs one WARNING at boot and leaves the column blank.
2. `cf_floor_stop_px` — `min(consolidation-low stop, fill × 0.975)`, computed at every fill (dry AND live).
3. `cf_floor_stop_hit` (0/1) and `cf_stoplimit_exit_px` — from a `CFWatch` (module `trading/hod_break_engine.py`)
   that rides the SAME print-watch subscription used to arm/resolve the resting order, kept open past a DRY fill
   until that fill's own recorded exit (its ACTUAL target or ACTUAL stop) is reached. `cf_floor_stop_hit` = any
   print at or below `cf_floor_stop_px` before that exit. The stop-limit counterfactual arms
   `actual_stop × (1 − 0.0020)` the instant a print first touches the actual stop; `cf_stoplimit_exit_px` is the
   first SUBSEQUENT print at or above that limit within 60s, else the last print seen at the 60s deadline (the
   no-fill tail, swept once per bar close — `_sweep_cf_watch_timeouts`, so resolution is bounded to roughly ±1 min
   of the true deadline). LIVE fills only get column 2 — a live fill already has a real broker bracket managing its
   exit, so no print-watch is opened for it.

Items 1-2 are appended to BOTH `hod_dry_entry_ledger.csv` and `hod_live_parity_ledger.csv` (blank when the flag is
off). Item 3 goes to a separate file, `hod_break.cf_ledger_path` (default `logs/hod_dry_counterfactuals.csv`):
date, symbol, fill_ts, fill_px, actual_stop, actual_target, cf_floor_stop_px, cf_floor_stop_hit, cf_stoplimit_px,
cf_stoplimit_exit_px, exit_px, exit_reason. Backward compatibility: an existing ledger file whose header predates
the two shared columns is NEVER rewritten — its rows keep the OLD shape and one WARNING fires per path (not per
row); only a brand-new file gets the new header. Tests: `tests/test_hod_counterfactuals.py` (flag off, is_qualified
wiring, `cf_floor_stop_px` formula, header backward-compatibility, and an integration test driving a dry fill
through synthetic prints to both a target exit and a stop/stop-limit exit).
