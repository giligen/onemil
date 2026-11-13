# PREREG 1,756 — the GAP FADE as a FILTER on the long books (book J3): is gap ≥ 10 % the loser bucket of the HOD / ORB longs?

FROZEN 2026-11-13 19:50 UTC before any number. Programme cell count: 1,755 → 1,756. Rules: write only
`research/hod_entry/1756_*`, `RESULT_1756.md`; no git/service/orders/network; data on disk only (< 100 MB);
budget ≤ 40 calls. Parent: 1,754 (gap ≥ 10 % names fade −185 bp open-to-close, every year). Owner's ask 11/13:
"filter on HOD and not buy gappers with more than 10 %".

## Data
HOD: `research/hod_entry/1735_per_signal.csv` (B0 net_R per arm, split) joined to `research/hod_exit_lab/
signals.parquet` (entry_m, price, r_pct) and the gap: open_t / close_{t−1} − 1 from the daily bars on disk
(`research/signal_search/1748_cache` / `1746_cache`; report the join coverage; arms without a gap are a bucket of
their own). ORB: the entered-inclusive features CSV via `trading/orb_csv.read_orb_csv` (gap_pct is a column;
net R per fill; 2025-01 → 2026-09 — the ORB book has no TRAIN/VAL split: report by half-year).

## Reads (all on the LIVE rule's outcome, nothing re-simulated)
Net R by gap bucket (< 5 % / 5–10 / 10–20 / ≥ 20 / no gap) × break hour (09:30–10:00 / 10:00–11:00 / 11:00–13:00 /
13:00–close): n, mean net R, day-clustered t, ex-top-5 %; TRAIN halves, TRAIN, VAL (one read); the same for the
ORB book by half-year. The filter read: the book WITHOUT gap ≥ 10 % arms vs WITH, paired on days (ΔR/wk, fills/wk
lost, green-week share). Placebo: the same split by a random bucket of prior-day RANGE instead of gap.

## Kill (the filter is adopted only if ALL hold; else the fade and the break are different populations/times)
gap ≥ 10 % bucket below the rest on BOTH TRAIN halves and on VAL (day-t of the difference ≥ 2 on VAL); the
filtered book's ΔR/wk > 0 on VAL with ≥ 3 fills/wk kept; ex-top-5 % of the filtered book ≥ the unfiltered;
placebo split shows no such effect. Adopt → dry-run config change spec (`hod_break.max_gap_pct`), ORB the same
(`orb.yaml` filter) — each with a parity test, owner's word before any live config.

## Output
`1756_score.py`, `1756_buckets.csv`, `1756_filter.csv`, `RESULT_1756.md` ≤ 350 words ending with "CHILDREN:" (≥ 2).
Return ≤ 120 words: join coverage, the gap-bucket table on VAL (mean R per bucket), the break-hour interaction,
the filtered-vs-unfiltered ΔR/wk, ORB by half-year, ADOPT or REJECT.
