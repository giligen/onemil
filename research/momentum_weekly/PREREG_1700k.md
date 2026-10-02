# PREREG — cell 1,700k: diversification caps on the momentum sleeve (FROZEN 2026-10-02 15:25 UTC)

From 1,700j's anatomy: four of the five deepest drawdowns are THEME clusters reversing while SPY barely moves
(2021 China ADRs + speculative names, SPY −9 %; 2025-10 quantum/nuclear, SPY −2 %; 2026-06 optical/semis), 62 % of the
loss from the held names' own declines. Stops, vol targets and sizing cut exposure after the fact and cost 6–11 points
of CAGR per 10 points of drawdown (0/14 pass). The untested lever is what the book holds: cap the cluster.
No sector map exists on disk, so clusters come from returns (point-in-time, data through the prior Friday).

## Cells (reference = the reconciled book of 1,700j on the daily basis: 27.2 % / −44.5 % / $508K)
* K1 correlation cap: walk the ranked list top-down; skip a name whose trailing 63-day daily-return correlation with
  any already-selected name exceeds 0.70; continue until 20 names.
* K2 as K1 at 0.60.
* K3 cluster cap: hierarchical clustering (average linkage, 1 − correlation, 63 days) of the top 100 ranked names
  cut at 0.5; at most 4 names per cluster; fill to 20 by rank.
* K4 as K3 with at most 3 per cluster.
* K5 beta cap: skip names with trailing 252-day beta to SPY above 2.0.
* K6 K1 + K5.
* K7 ADR/foreign exclusion: names whose asset name contains (word-boundary) ADR|ADS|Depositary|Holdings? Ltd|Limited|
  N\.V\.|S\.A\.|plc are excluded (the 2021 episode's China ADRs) — a universe rule, reported with its 2020 cost.
* K8 K3 + T1 (15 % trailing name stop, the best single repair of 1,700j).

## Reads and pass rule
As 1,700j (CAGR, max DD on the daily basis, the five episodes' depths, worst year, years beating SPY, rolling-5-year
share, end $ from $50K, turnover, cost drag) plus the average pairwise correlation of holdings. Pass = max DD better
by ≥ 10 points for ≤ 5 points of CAGR, rolling-5y ≥ 90 %, ≥ 3 of 5 episodes shallower by ≥ 25 %. Nothing outside the
grid after seeing numbers; 8 cells, stated.

## Output
`1700k_caps.py` (reuse 1700j_frontier.py), `1700k_cells.csv`, `RESULT_1700k.md` (≤ 70 lines). Agent returns ≤ 150 words.
