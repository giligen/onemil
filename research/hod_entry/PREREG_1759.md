# PREREG 1,759 — F3b leveraged-ETF rebalance flow into the close on the FULL SPY / QQQ minute history (child of 1,758)

FROZEN 2026-11-13 19:30 UTC before any number. Programme cell count: 1,758 → 1,759. Rules: write only
`research/hod_entry/1759_*`, `RESULT_1759.md`; no git/service/orders/network; data on disk:
`research/index_orb/cache/SPY_1min.parquet`, `QQQ_1min.parquet` (timestamp,o,h,l,c,v; 2016-01-04 → 2026-05-29);
`ulimit -v 2500000; nice -n 19`; budget ≤ 40 calls. Splits: TRAIN 2016–2020 (halves 2016–18 / 2019–20), VAL
2021–2023 once, TEST 2024 → sealed (drop at load). Parent: 1,758 F3 VOID (8 VAL days). Prior: Cheng–Madhavan
2009 — leveraged/inverse ETFs rebalance at the close in the direction of the day's move; the effect is larger
when leveraged AUM is large and on big-move days; post-2012 persistence is the open question.

## Signal and trade (pre-picked primary: SPY, k = 1.0 %, 15:00 → 15:55)
r = close_15:00 / open_09:30 − 1 (and the secondary r' = close_15:00 / prev_close − 1, the overnight-inclusive
move — the rebalance need is on the DAY's return including the gap; report both, primary = r'). If |r'| ≥ k,
k ∈ {0.5 %, 1.0 %, 1.5 %, 2.0 %}, trade WITH the sign from the 15:00 close to the 15:55 close (cost 1 bp per leg).
Reads: by k; by sign; SPY and QQQ; by year; the |r'| < 0.5 % comparator (should be ≈ 0); 15:30 → 15:55 variant
(the flow is concentrated in the last 30 min). Numbers: n days/yr, mean bp, t (days iid), hit rate, ex-top-5 %,
worst day, year signs, TRAIN halves, VAL once, MDE.

## Kill (primary = SPY, r', k 1.0 %, 15:00 → 15:55)
VAL mean ≤ +5 bp or t < 2; TRAIN halves opposite; the comparator not below the trade; fewer than 30 VAL days.
Survive → TEST once → independent rebuild → paper sleeve (SPY/QQQ, no borrow issue: long or short an ETF).

## Output
`1759_score.py`, `1759_grid.csv`, `RESULT_1759.md` ≤ 350 words ending with "CHILDREN:" (≥ 2). Return ≤ 120 words:
VAL primary n / mean / t / hit rate, by sign, 15:30 variant, QQQ, comparator, year signs, KILL/SURVIVE, children.
