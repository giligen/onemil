# PREREG — ORB edge as a function of order latency, replayed on tick data. Cell 1,426

Frozen 2026-09-25 14:55 UTC before any tick exists. Programme count 1,425 → 1,426. Owner GO ("test it on a BT with
increased latency"); data spend cap $15.

## Why
Live ORB lost −$5,281 on 123 fills (5/19–9/23) while the same-month backtest was positive. The backtest assumes the
buy-stop is resting at 09:35:00; live's first submit was 26–49 s late and the chase guard skipped the fast movers (BIAF
9/22). The question the owner needs answered with data, not guesses: how much of the backtest survives at 1, 2, 3, 5,
10, 20, 30, 60 s of latency — and would May–Sep 2026 have been positive at the latency the pre-warm fix can reach?

## Population
Every BT fill (`entered == 1`) of the live-config books: `research/thermo/book_2025_26.csv` (2025-01..2026-09-23),
`research/orb_2023/book_1418_liveexit.csv`, `research/orb_2024/book_1415_liveexit.csv`. Periods reported separately:
2023-24 (untouched), 2025H1 (in-sample), 2025H2–2026-09 (tuned-on), and May–Sep 2026 alone. Also the BT picks that
did NOT fill (`entered == 0`) are counted but not replayed (a resting order that never triggered stays unfilled).

## Tick data
Databento XNAS.ITCH `trades` + `mbp-1`, `stype_in='raw_symbol'`, window 09:34:50–09:40:00 ET per (symbol, date), one
fetch per window (reuse `research/hod_ofi/pipeline.py`'s client/blackout/manifest helpers; raw parquet under
`research/orb_latency_bt/raw/`). Cost gate first: `metadata.get_cost` on 30 random windows, extrapolate, print; STOP
if > $15. Availability rail: a fill is usable iff ≥ 1 trade print and a two-sided quote exist in its window; usable ≥
80 % of fills and winner/loser missingness gap ≤ 5 pp, else VOID.

## The engine's entry rule (extract it from `trading/orb_engine.py` and cite line numbers before simulating)
Buy-stop trigger = the range high + the planner's buffer; limit = trigger + the planner's limit offset; the chase guard
skips when `ask + rebump_buffer ($0.02 default, `_buy_stop_guard_cfg`) > limit` ("ENTRY SKIPPED — breakout extended
past limit"). Use the exact constants the engine uses; if the BT book's trigger/limit columns exist, use those.

## Replay, per fill and per delay d ∈ {0, 1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90} s
T0 = 09:35:00 ET + d (the instant the order rests). t* = the first XNAS print ≥ trigger in [09:35:00, 09:40:00).
* no t* → not filled (P&L 0, counted as skipped-by-market);
* t* ≥ T0 → the resting stop-limit fills at the prevailing ask at t* (strictly prior mbp-1 record) if ask ≤ limit,
  else skipped;
* t* < T0 → at T0 the engine sees the breakout already under way: fill at ask(T0) if ask(T0) + rebump_buffer ≤ limit,
  else SKIPPED (never chased).
Exit: the BT's own exit price for that fill (the path after entry is unchanged), so the fill's P&L = BT `_sized_pnl`
+ shares × (BT entry − replay fill). Skipped fills contribute 0. Cost: the BT's exit slippage unchanged; the replay's
entry already pays the ask (no extra entry slippage).

## Report (report-only cell; the decision number is pre-committed)
For each d and period: n filled, n skipped-by-guard, total $, mean R per fill (R = P&L / 375), day-clustered t,
ex-top-5 % — as a table and a curve. **Decision number:** the largest d at which BOTH the out-of-sample book
(2023-24 + 2025H2–2026-09) mean R ≥ 0 AND May–Sep 2026 total $ ≥ 0. Lenses: (i) d = 0 vs the BT's own 30 bps entry
model (does the tick fill agree with the BT?); (ii) XNAS-only trigger bias (share of fills whose first XNAS print ≥
trigger is > 2 s after 09:35:00 while the BT's minute bar shows the breakout in the first minute); (iii) the fills
whose t* is within 1 s of 09:35:00 (gap-through cases) listed separately.

## Consequence (pre-committed)
The engineering latency target becomes the decision number minus 2 s (safety); if the decision number is < 3 s, ORB
cannot be made positive by latency engineering on this stack and the owner is told so with the curve.

## Not allowed
Changing the delay grid, the entry rule constants, the exit rule or the decision number after any tick exists.
