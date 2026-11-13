# PREREG 1,765 — CONDITIONING the honest gapper book on what is known before the open: Reg SHO threshold (E6), short-interest / days-to-cover (E5), pre-market dollar volume (F6)

FROZEN 2026-11-13 20:55 UTC before any number. Programme cell count: 1,764 → 1,765. Rules: write only
`research/hod_entry/1765_*`, `RESULT_1765.md`; no git/service/orders/network; data on disk; `ulimit -v 2500000;
nice -n 19`, peak RSS < 1 GB; budget ≤ 50 calls. TEST sealed. Binding lesson 1,764: a filter is evidence only if
its count-matched placebo (random drop of the same size) shows no gain — report the placebo beside every filter.

## Data
Outcomes: the regular HOD-break net R per symbol-day on the honest population from `1763_score.py`'s comparator
(import). Pre-market bars: `1762_cache/premarket/<day>.parquet` (04:00–09:29; from 1,763). Daily bars / ADV20:
`1762_cache/daily.parquet` and `cell_1428_population.csv::adv20`. FINRA Reg SHO threshold lists and short-interest
files: `research/signal_search/1746_cache/finra/` (from cells 1,746/1,748 — inspect what is there; threshold list
daily, short interest bi-monthly; say which files and dates; coverage per split ≥ 80 % or that read is VOID).

## Features (all known before 09:30 on day t)
F6 pm_dollar_vol / ADV20 (dollar), pm_bars_with_trades (count), pm_high vs prev_close (the pre-market gap);
E6 on_threshold_list (t−1), days_on_list; E5 short_interest / float — float is NOT on disk: use SI / ADV20 shares
(days-to-cover) from the latest settlement date before t (say the lag), and the change vs the prior report.

## Reads (TRAIN halves, TRAIN, VAL once)
Per feature: quintiles (or yes/no) → n, mean net R, day-clustered t of top − bottom (or yes − no), ex-top-5 %,
fast-loser share, monotonicity; the filter read (keep the top two quintiles / drop the flag) with ΔR/wk, fills/wk
kept, and the count-matched placebo ΔR/wk (mean of 200 random drops of the same size).

## Kill (per feature; a feature is "structure" only if ALL hold)
Top − bottom (or yes − no) same sign on both TRAIN halves and VAL, day-t ≥ 2 on VAL; monotone on TRAIN;
filter ΔR/wk on VAL above the placebo's 95th percentile; ≥ 3 fills/wk kept. Survivor → TEST once → independent
rebuild → dry-run config / owner-list column.

## Output
`1765_score.py`, `1765_features.parquet`, `1765_buckets.csv`, `RESULT_1765.md` ≤ 350 words ending with
"CHILDREN:" (≥ 2). Return ≤ 120 words: coverage per source, per feature the VAL top − bottom / t / monotone /
filter ΔR vs placebo p95, structure YES/NO, children.
