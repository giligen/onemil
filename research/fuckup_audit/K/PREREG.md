# Stage K — multi-day holds, pre-registered 2026-09-17 (before any run)

## Why
The cost curve (`lit_review_2026/cost_curve.md`) says a viable book needs spread/R < 0.05, i.e. R measured in whole
percents. Every intraday universe failed on that arithmetic (J: median R 1.24%, cost 0.14R). A 2–10 day hold on liquid
names has R of 5–15% and a spread of 0.1–0.5% → spread/R ≈ 0.02. Capacity is in the DAILY dollar volume, not a 5-minute
tape. The program's own daily-panel tests (`lit_review_2026/daily_queue.md`, `daily_addons.md`: M29 overnight
continuation, M41 new-high momentum) were TRAIN/VAL-positive at t 3.5–5 and TEST-negative on a 14-week window whose
power for a 5-day-hold book is near zero. The review (§2) ranks a multi-day book as the one direction pointed at and
never run properly.

## Data (all on disk)
Databento point-in-time daily panel `data/research/databento/equs_daily_2025_2026.parquet` (delisted included, raw
prices — the price-scale check vs `daily_bars` on 200 keys is step 0), `research/lit_review_2026/daily_panel.parquet`
(the panel the earlier daily tests used — read `build_daily_panel.py` for its fields and its causal conventions), SPY/IWM
daily. Universe: 20-day median dollar volume ≥ $10M, price ≥ $5, common stock only (asset-class map), test tickers
excluded (`^Z[A-Z]ZZT$`). Fills: next day's OPEN after the signal day's close (market-on-open; cost = half the
liquidity-band spread + 5 bps), exits at the close of day N or at a stop on a daily close; no intraday data used.

## Families (declared; each with its mechanism)
- K1 post-earnings-style gap continuation WITHOUT an earnings calendar: gap ≥ +8% on ≥ 3× ADV, close in the top third
  of the day's range → buy next open, hold 5 days, stop = the gap day's low on a close (the "day-2 continuation" of
  ORB's PDR rule, at the daily scale).
- K2 52-week-high breakout on volume: close at a new 252-day high with volume ≥ 2× ADV, hold 10 days, stop 7% on a
  close (M41's shape with the volume leg).
- K3 short-term reversal, long side only: 5-day return in the bottom decile of the universe, close above the day's
  open (a first up-close), hold 3 days (the strongest documented daily effect; long-only version).
- K4 overnight continuation, top decile of the trailing 20-day mean overnight return, hold 1 day close-to-close (M29
  re-run with the causal universe and a proper power statement).
- K5 pullback in an uptrend: 50-day high within 10 days, 3-day decline ≥ 5%, close > 20-day SMA → buy next open, hold
  5 days, stop 5% on a close.
Each family: long only; sizing = equal $ per position; book = 10 positions max, one per name, first-come by signal
strength as declared per family (K1 gap size, K2 volume ratio, K3 reversal depth, K4 the mean, K5 pullback depth).

## Cells
5 families × 2 holds (the declared hold and half of it) × 2 books (10 / 20 positions) = 20. Splits TRAIN 2025 / VAL
2026-01..05 / TEST 2026-06..09 — with the power statement per split (a 5-day book has ~50 independent observations in
TEST; report the MDE). Gates: PLAN §1 (G1 t ≥ 2 on TRAIN, G2 VAL sign + ≥ 55% weeks green, TEST once); tail tests;
permutation p across the 20 cells; the availability audit for every column; capacity = position size at 1% of the
20-day median dollar volume (report the $ per position and the book's $/month at that size).

## Deliverable
`K/REPORT.md` (one page first: does any multi-day family clear G1/G2? the $/month at capacity with worst month;
which is the smallest visible effect), the per-trade CSVs of survivors, the cell count, 3 lines in `LOG.md`; for a
survivor: the independent rebuild from prose before any engine work (there is no engine for daily orders yet —
`submit_market_on_open` would be new).
