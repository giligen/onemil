# RESULT — cell 1,674: the bar's power error, pooled reads, joint book, capped-day selection

PREREG: `research/hod_entry/PREREG_1674.md` (FROZEN 2026-09-30). Owner ask: "find the errors and oversights
in your research, there's money there." Population: 1667_features.csv, the 5,506-row floored primary book
(fills_1658 ⋈ causal_arming_causal, r_pct >= 1.5%), verified identical to 1663/1665 on (date,symbol) before
scoring (0 dup keys, 0 net_R/r_pct mismatches across all 5,506 rows). Script: `1674_pooled.py`. Log:
`1674_pooled.log`. Reads: `1674_reads.csv`. Joint per-fill ledger: `1674_joint_per_fill.csv`.

## Oversight confirmed, but does not recover money
The original per-half bar (net >= +0.05 R AND t >= 2.5 in BOTH TRAIN-H2 and VAL) was a real statistical
error: per-half MDE is 0.061-0.13 R here, so it silently demanded power the data didn't have. The pooled,
day-clustered re-read below is the correct test. It does NOT find the money the owner expects: every
candidate's true pooled lift is either near-zero in magnitude or, for the one candidate with real
significance (MOC), too small to matter at this book's economics.

## Part A — pooled reads (both halves together), day-clustered SE, sign-agreement, MDE
ΔR = pooled mean of the kept book minus the pooled mean of its own floored-base reference population
(same day-clustering method, `delta_vs_base_day_clustered_t`); MDE = 2.80 x SD / sqrt(n) at the reference
book's own SD. TRAIN/VAL columns are the same ΔR construction restricted to each half.

| candidate | n | pooled ΔR | day t | ex-top-5% | fills/wk | MDE | TRAIN ΔR (t) | VAL ΔR (t) | sign(T,V) | agree |
|---|---|---|---|---|---|---|---|---|---|---|
| drop top RVOL_A(20) tercile | 3,679 | -0.0012 | -0.07 | -0.115 | 77.6 | 0.062 | -0.0098 (-0.33) | +0.0095 (+0.47) | -,+ | N |
| drop top RVOL_A(5) tercile | 3,718 | +0.0016 | 0.10 | -0.121 | 78.4 | 0.061 | +0.0033 (0.13) | -0.0004 (-0.02) | +,- | N |
| drop top RVOL_B tercile | 3,651 | +0.0034 | 0.20 | -0.122 | 77.0 | 0.062 | +0.0031 (0.11) | +0.0038 (0.21) | +,+ | Y |
| drop top F15 tercile (dv/level/ADV20) | 3,599 | -0.0041 | -0.21 | -0.114 | 75.9 | 0.062 | -0.0102 (-0.35) | +0.0035 (0.16) | -,+ | N |
| drop top F11 tercile (level vs VWAP) | 3,562 | +0.0041 | 0.25 | -0.131 | 75.1 | 0.063 | +0.0062 (0.26) | +0.0016 (0.07) | +,+ | Y |
| keep >=3% stops only | 1,003 | +0.0229 | 0.45 | -0.054 | 21.2 | 0.118 | -0.0754 (-1.05) | +0.1280 (1.78) | -,+ | N |
| keep 11:00-12:30 only | 869 | -0.0575 | -1.05 | -0.148 | 18.3 | 0.127 | +0.0157 (0.19) | -0.1429 (-2.11) | +,- | N |
| MOC exit* | 7,927 | +0.0095 | 25.00 | -0.101 | 110.1 | 0.041 | +0.0100 (19.2) | +0.0087 (18.2) | +,+ | Y |

\* MOC read on **1660's own B0 population** (floored on its own r_pct>=1.5%), NOT row-joined to the primary
5,506 book — see the data-integrity finding below. Its huge t comes from near-zero day-to-day noise in the
bid-vs-moc delta itself (a mechanical repricing, not idiosyncratic P&L), not from a large or newly-discovered
effect; +0.01 R is real and reliable but an order of magnitude under the +0.05 R pass bar.

Only 2/8 candidates sign-agree (RVOL_B, F11), both with |t| < 0.3 and pooled |ΔR| < 0.005 R — genuinely flat,
not underpowered (their own MDE is only 0.06 R at this n, so a true +0.05 R lift would very likely have
shown). stop_ge_3pct and time_1100_1230 have the largest pooled magnitudes but sign-DISAGREE between halves
and are underpowered (MDE 0.12-0.13 R > the effect itself) — genuinely inconclusive, not evidence of an edge.

## Data-integrity finding (the actual "oversight" with teeth this cycle)
`1660_per_fill.csv` cell=='B0' matches the primary book on (date,symbol) for only 4,111/5,506 rows (74.7%,
below the 80% availability rail). Worse: for MATCHED keys, 1660's own net_R_bid does **not** reproduce the
primary book's net_R for the same (date,symbol) — median entry-price gap $0.14 (mean $0.39, max $13.28),
median |r_pct| gap 0.27pp, and **374/4,111 matched rows have opposite-signed net_R** (e.g. AAOI 2025-08-12:
primary net_R = +1.92 R, 1660's net_R_bid = -0.47 R). (date,symbol) is not a same-fill key across these two
pipelines — 1660's B0 is a differently-constructed population. The MOC candidate is therefore read entirely
within 1660's own book (above); when it "enters" the joint-book rule (below) it cannot be applied to the
primary book's per-fill ledger — there is no validated join key on disk to do so. This is logged as a
WARNING in `1674_pooled.log` and is a real open item for a future cell (build the missing fill_id-level
bridge, or regenerate the MOC column directly inside the 1663/1667 pipeline).

## Part B — joint book (selected on TRAIN, read once on VAL)
Rule: TRAIN ΔR > 0 AND TRAIN day-clustered t >= 1.0 AND VAL sign agrees. Only **moc_exit** enters (TRAIN
ΔR +0.0100, t 19.2, VAL sign +); every filter-type cut is rejected (best TRAIN t was F11 at 0.26). Because
moc_exit cannot be applied to the primary per-fill book (above), the joint book as actually computable =
the floored base with **zero filters applied**:

VAL: n 3,157, mean **-0.0295 R**, day t **-2.08**, ex-top-5% -0.134, fills/wk 150.3, MDE 0.066. TRAIN-H2
in-sample caveat: -0.0210 R. Leave-one-out table is empty (no filter-type cut entered). **Fails the pass bar**
(negative mean, not just insignificant).

## Part C — capped-day selection (12/day cap vs ~30-46/day in the floored book; n=2,575 kept in every order)

| order | pooled mean | day t | TRAIN mean | VAL mean | ex-top-5% | fills/wk |
|---|---|---|---|---|---|---|
| time order (baseline) | -0.0376 | -1.03 | -0.0688 | -0.0027 | -0.143 | 54.3 |
| largest r_pct first | -0.0321 | -0.92 | -0.0490 | -0.0132 | -0.137 | 54.3 |
| lowest RVOL_A(20) first | -0.0413 | -1.18 | -0.0535 | -0.0277 | -0.147 | 54.3 |
| lowest F15 first | -0.0431 | -1.22 | -0.0567 | -0.0277 | -0.149 | 54.3 |
| joint-book members first | -0.0376 | -1.03 | -0.0688 | -0.0027 | -0.143 | 54.3 |
(joint-members-first == time-order exactly: no filter-type cut is in the joint book, so there is no
differentiated priority group to sort on.)

Best order = **largest r_pct first**: pooled -0.0321 R (t -0.92) vs time-order's -0.0376 R (t -1.03) — a
+0.0055 R improvement, still negative and not significant. **No capped-selection order passes.**

## Verdict
Nothing passes. The per-half bar's power error is real and worth fixing on principle, but the pooled,
correctly-clustered re-read shows the underlying cuts are genuinely flat (RVOL_B, F11: |ΔR|<0.005, well
powered) or inconclusive on too little data (stop/time-window cuts), not merely "victims of the bar." MOC
is the one candidate with real statistical teeth (t 19-25) but at ~+0.01 R it is uneconomic and, this cycle,
structurally unmergeable into the primary book. The joint book and every capped-selection order are
negative. No independent rebuild is triggered (nothing cleared the pass bar to report to the owner).

## Files
`research/hod_entry/1674_pooled.py`, `1674_reads.csv` (14 rows), `1674_joint_per_fill.csv` (3,157 VAL rows,
unfiltered floored book), `1674_pooled.log`, this file.
