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
