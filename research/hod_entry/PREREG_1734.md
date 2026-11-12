# PREREG 1,734 — does the HOD-break book depend on the market regime? (owner 2026-11-12: "HOD seems to do well recently, can it be it works better in a bearish market? Test on historical data")

FROZEN 2026-11-12 before any number. One cell. Rules: no git (no commit/stash), no service action, no orders,
read-only on `data/cache.db` / `data/trades.db` / every existing CSV; write ONLY `research/hod_entry/1734_*`.
Never Read a file > 300 lines in full. Budget ≤ 45 calls. This task IS the owner's request; do not pivot.

## Data (all on disk, no fetch)
* **Base book** = `research/hod_entry/1681_per_fill.csv`, columns `read, fill_id, date, symbol, base_R`; drop
  duplicates on `fill_id` (the file has one row per hypothesis × fill; `base_R` is the live-rule NET R of the fill,
  identical across rows). Expect 5,506 fills, 2025-07-01..2026-05-29, reads `TRAIN-H2` (2,349) and `VAL` (3,157).
  State in the RESULT what `base_R` is, from `PREREG_1681.md` / `PREREG_1670.md` (one grep each), so the owner reads
  the right number.
* **Forward book** = `data/trades.db` `trades` where `strategy='hod_break' and account='paper' and trade_date >=
  '2026-09-25' and pnl is not null`; R = `pnl / total_risk`; expect ~311 fills. Reported as a THIRD read, never a pass
  criterion (MDE 0.45 R at n 311).
* **Regime** = SPY from `data/cache.db` `daily_bars` (`symbol='SPY'`, `bar_date`, `close`), 2024-06-03..2026-11-11.
  Every feature is computed at the PRIOR session's close (causal — nothing from the fill day):
  1. `spy_ret20` = close[t−1] / close[t−21] − 1 (20 sessions), terciles cut on the WHOLE base book's day set
     (one cut per day, then applied to fills; the forward book uses the SAME cut points — frozen, not re-cut);
  2. `spy_below_50` = close[t−1] < mean(close[t−50..t−1]) (binary);
  3. `spy_ret1` = close[t−1]/close[t−2] − 1, terciles as in 1 (yesterday's direction).
  Day-of-fill SPY move is NOT a feature (it is not knowable at the arm; the owner's "bearish market" means the trend).

## Scoring (per feature, per read: TRAIN-H2, VAL, FORWARD, and POOLED TRAIN+VAL)
Per bucket: n fills, n days, mean net R/fill, day-clustered t (cluster = date), iid t, ex-top-5 % mean (drop the
top 5 % of fills by R inside the bucket), winner-capped mean (R capped at +3), green-day share, sum R.
Difference test: bottom tercile (bearish) − top tercile (bullish) ΔR with day-clustered SE, per read.
**Placebo:** 500 shuffles of the day→tercile label (permute across days, keep fills within a day together) → the
percentile of the observed Δ (bottom − top) under the null, per read.
**Frequency:** fills per session by bucket (the regime may change FREQUENCY, not per-fill edge — the ORB finding).

## Pass bar (pre-committed; anything else = FAIL, report as such)
The hypothesis "HOD works better in a bearish market" PASSES only if, for `spy_ret20` (the primary feature):
bottom tercile mean ≥ +0.10 R/fill with day-t ≥ 2.0 in BOTH TRAIN-H2 and VAL, ex-top-5 % ≥ 0 in both, placebo
percentile ≥ 95 in both, and the forward read has the same sign. `spy_below_50` and `spy_ret1` are secondary
(reported, cannot carry a pass on their own; if one passes the same bar it is flagged "secondary, needs its own
PREREG"). A pass means SIZE-BY-REGIME is a candidate for the paper instrument, NOT a live change.

## Output
`research/hod_entry/1734_score.py` (one script, runs end-to-end, prints every table), `1734_buckets.csv` (one row per
feature × read × bucket with the numbers above), `1734_fills.csv` (fill_id/date/symbol/base_R + the three features —
the rebuild compares on this), `RESULT_1734.md` ≤ 300 words: the tables, the pass/fail verdict against the bar,
the three caveats (cost model of base_R, n per bucket, 11 months of regime = few independent regime episodes —
count the number of distinct tercile RUNS, i.e. regime episodes, per read). Return ≤ 120 words: verdict + the
bottom/top tercile R per read.

## Independent rebuild (second agent, has not read 1734_score.py)
From this prose only: recompute `1734_fills.csv` features and the per-bucket mean R / n for `spy_ret20` on
TRAIN-H2 and VAL; compare row by row on `fill_id` (feature values exact, bucket labels identical) and bucket means
to 0.001 R; write `REBUILD_1734.md`. Any mismatch > 0 rows = the number does not reach the owner.
