# PREREG — Order-flow imbalance at the HOD break (the owner's 9/13 idea). Cells 1,393–1,395. Spend-gated

Why now: bars carry no information about the path after an HOD break (35 cells, `REPORT.md`). The quote/trade
tape at the break minute is the only untested information channel. Unit cost probed 2026-09-22 with the
Databento API: **XNAS.ITCH MBP-1 ≈ $0.0045 per symbol per 10-minute window** → 12,135 signals × 10 min ≈ **$55**,
20-minute windows ≈ $110. EQUS.MINI quote schemas are NOT used (CLAUDE.md). Owner approval of the spend is the
only gate; nothing below runs before it.

## Data
XNAS.ITCH `mbp-1` (Nasdaq top-of-book, single venue — the live engine can compute the same from its feed) for each
HOD-break signal (`features.csv`): window = [signal minute − 5 min, signal minute + 5 min]. Also `trades` for the
same window (≈ $0.0004/symbol-window) for the trade-sign fallback. Fetch once into
`research/hod_ofi/mbp1.parquet`; availability rail: a signal is usable only if ≥ 60 % of its window seconds have a
quote; report the share and the winner/loser missingness gap (≤ 5 pp else VOID).

## Features (all from the 5 minutes BEFORE the break bar closes; nothing after)
* **OFI_5** — Cont-Kukanov-Stoikov L1 order-flow imbalance summed over the 5 minutes before the signal, normalized
  by the mean displayed depth over the window (depth-normalized, dimensionless).
* **OFI_1** — the same over the last 60 seconds.
* **TSI_5** — trade-sign imbalance (Lee-Ready against the prevailing quote) over 5 minutes, in shares / total.
* **spread_bps_at_break** — the quoted spread at the break, in bps (also the honest cost for that trade).

## Cells (each ONE cut on the B0 book from `research/hod_exit_lab/b0_trades.csv`, paired kept-vs-dropped)
| cell | keep iff |
|---|---|
| 1,393 F1 | OFI_5 ≥ the TRAIN-H1 median (buyers lifting into the break) |
| 1,394 F2 | OFI_1 ≥ the TRAIN-H1 median (the last minute confirms) |
| 1,395 F3 | TSI_5 ≥ the TRAIN-H1 median |
Report-only: the monotone decile table of B0 net R by each feature on TRAIN-H2 and VAL (the PREREG's "monotone
TRAIN→VAL" requirement is scored on the deciles, not only the median cut); the same features for the D1 placebo
minute (a random minute of the same name-day) to show the feature is about the BREAK, not the name.

## Pass bar (all)
Kept-cohort net R ≥ +0.10 above the dropped cohort on TRAIN-H2 and VAL (TRAIN-H1 sets the cut), VAL day-clustered
t ≥ 2, kept cohort ≥ 3 fills/week at first-12/day 4-concurrent, deciles monotone (Spearman ≥ 0.6) on both
holdouts, placebo-minute feature shows no such lift, ex-top-5 % ≥ 0. Verification: three refuter lenses (window
causality — the window must end before the signal bar CLOSES; depth normalization; missingness) + independent
rebuild from this prose. Pass → the dry run computes the same feature live from its quote stream and tags each
signal; 10 sessions side by side before any gating. Programme count 1,395.
