# F6-PDR live engine — specification (2026-09-17, Fable; implementation after the independent rebuild agrees)

Goal: run the declared F6-PDR book live through the existing HOD-break engine (`trading/hod_break_engine.py`), which
already streams the universe, drains bars on a thread, places capped-limit brackets, tracks legs, flats at 15:55 and
has the parity audit + EOD check. F6 is the same shape as the HOD-break — a level break with a next-open fill — with a
different level, a different precondition and no consolidation rule. ONE spec module, shared by the backtest and the
engine (`trading/red_to_green.py`), the same way `trading/hod_break.py` is.

## The rule (`RedToGreenParams`, all defaults = the declared book)
| field | value | live source | BT source |
|---|---|---|---|
| precondition | 09:30 open < prior close | the first RTH bar of the stream / REST backfill; prior close from `daily_bars` | universe.csv / Databento |
| prior-day range | (high − low)/low of the prior day ≥ 8% | `daily_bars` (the ORB PDR helper `trading/orb_pdr_veto.py` computes it — reuse) | same |
| level | prior close × 1.003 | `daily_bars` | same |
| range floor | (running high − running low)/running low ≥ 5% on bars STRICTLY before the signal bar | the engine's per-symbol bar array | same |
| signal | first closed bar with high ≥ level, minute ≤ 14:00 | bar drain thread (seconds after the close) | same |
| fill | capped limit at level × 1.006 submitted after the signal bar closes; fill ≈ next bar's open; ask above cap → no chase | `submit_bracket_order` (as HOD) | next open ≤ cap |
| stop | lowest low from 09:30 through the signal bar; R ≥ 1% of price | SL leg at the stop | min(stop, open) × 0.999 |
| exits | `exit_mode: hold | target2r | partial` — hold: no TP leg, SL leg only, flat 15:55; target2r: TP leg at entry + 2R (re-anchored to the real fill as HOD does); partial: TP leg for half the shares at +2R, then SL leg moved to the fill price for the remainder | bracket legs; `_anchor_target_to_fill` | close-fill at target |
| floors | price ≥ 5 (the BT population); the HOD engine's `min_price 20` / spread ≤ 15% of R gates are NOT part of this book — they must be configurable per book and OFF for F6 unless the sizing study says the $5–20 band cannot carry the risk | config | — |
| book | 12/day, 4 concurrent, one trade per symbol-day, first-come by signal minute | DB-derived caps (as HOD) | `run_book(rows, 12, 4)` |
| kill rails | daily −6R, weekly −12R of the configured risk (from the book's worst week −10.7R on TRAIN) | realized P&L from the DB (fail closed) | — |

## Engine delta (minimal)
1. `trading/red_to_green.py`: `RedToGreenParams`, `first_signal(o, h, l, c, m, prior_close, p)` (precondition, floor, level
   break), `entry_fill` (reuse HOD's), `run_book` (reuse HOD's). Tests: `tests/test_red_to_green.py` — the spec against
   hand-built tapes, and a parity test that replays the rebuild agent's per-trade CSV for 20 symbol-days through the
   spec (same signal minute, level, stop).
2. `hod_break_engine.py`: a `book` parameter (`hod_break` | `red_to_green`) selecting the signal function and the exit
   mode; the universe stream is shared (names with prior close ≥ `universe_min_prev_close`; for F6 the stream must also
   include names that open BELOW prior close — the HOD stream keys on prev close ≥ 17, fine; the F6 candidate set is
   "open < prior close AND PDR ≥ 8", known at 09:30 from `daily_bars`, so the engine can subscribe ONLY those names —
   a few hundred a day, lighter than HOD's 3,600).
3. Log lines: `[R2G] gates …` at boot; `[R2G DRY] WOULD BUY sym level stop R pdr floor` (or `[R2G] BUY`); `[R2G] FILLED`;
   `[R2G] EXIT … reason`; `[R2G] FLAT 15:55`. The EOD check (`scripts/hod_break_eod_check.py`) gets a `--book r2g`
   mode that re-runs the spec on the day's REST bars and diffs.
4. Config block `red_to_green:` in config.yaml (+ template): `enabled: false`, `dry_run: true`, `exit_mode: partial`,
   `risk_usd: 100`, `min_price: 5`, `max_per_day: 12`, `max_concurrent: 4`, `daily_kill_r: 6`, `weekly_kill_r: 12`,
   `pdr_min_pct: 8`, `range_floor_pct: 5`, `level_buffer: 0.003`, `cap: 0.006`, `last_entry_minute: 840`.
5. Dry run for 5 sessions → the EOD check's parity number must be 0 misses and 0 spec mismatches; then live at
   `risk_usd 100` on the owner's word; ramp by the BF/ORB rule (advance on positive realized stage P&L over ≥ 15
   sessions and ≥ 8 trades, never while stage P&L < 0).

## What must be true before the first live order
- The independent rebuild (H/F6_rebuild) reproduces the trade set and the monthly R of `H/F6/f6_pdr_book.md` on
  (day, symbol), for the hold and 2R exits, and reports the partial exit's numbers.
- Sizing (H/F6_sizing): the risk level at which ≥ 90% of trades stay under 1% of the 5-minute dollar volume — the
  ramp ceiling.
- The engine's dry run shows the same signals as the spec on the day's REST tape (the HOD miss audit, `--book r2g`).
