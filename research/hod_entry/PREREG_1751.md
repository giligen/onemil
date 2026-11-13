# PREREG 1,751 — confirmation entry: enter the HOD break only after it has HELD, rule learned from winners vs losers

FROZEN 2026-11-13 18:50 UTC before any number. Programme cell count: 1,750 → 1,751. Owner's ask 11/13: "test 3, be
data driven on winners vs losers". Rules: write only `research/hod_entry/1751_*` and `RESULT_1751.md`; no git, no
service, no orders, no network; start under `ulimit -v 2500000; nice -n 19`; budget ≤ 60 calls. This task IS the
owner's request; do not pivot.

## Why this is not 1,742
1,742 measured hold_k / retest_depth EX-POST to the fill (a path that never retests has by construction already gone
up: "mechanical"). 1,751 conditions only on what is observable at minute N AFTER the break and scores the return
earned FROM minute N onward. The question: does "held N minutes without retesting the level" predict the FUTURE path,
or is the never-retest premium entirely the part already earned before N?

## Data (on disk, read-only)
`research/hod_exit_lab/paths.parquet` (day,symbol,m,hhmm,o,h,l,c,vwap,spy_close,minutes_since_entry — 1-min bars
per arm), `signals.parquet` (day,symbol,entry_m,entry,stop,target,R,r_pct,price,half_spread,spread_mean,split,
half,wk), `b0_trades.csv` (B0 net_R per arm — the live rule's outcome, the comparator). The level = B0 `entry`
(the break price), the original stop distance R0 = `R` (price units), costs: half_spread + 2 bp per leg, stop slip
as PREREG_1740. TRAIN/VAL from `split`; TEST sealed (never read; if the files contain TEST rows, drop them first
and say how many).

## Stage A — anatomy on TRAIN only (data-driven; VAL untouched until the rule is frozen in the RESULT file)
Per arm and per N ∈ {3, 5, 10, 15, 20, 30} minutes after entry_m, compute the STATE at the close of bar N (all
causal at N): held = min(l) over bars 1..N > level; dist_N = (c_N − level)/R0; maxdd_N = (min(l) − level)/R0;
above_vwap_N = c_N > vwap_N; ret_SPY_N; vol_ratio_N = volume? (not in paths — skip; say so); bars_since_open at N.
Forward outcome from c_N: fwd_R = (exit − c_N)/R0 under the LIVE exit rule restarted at N (stop at the original
stop price, target at the original target, else close) AND fwd_close_R = (close_15:55 − c_N)/R0 (no stop).
Winners vs losers: split arms by fwd_R sign; report the distribution of each state variable in winners vs losers
(deciles, AUC, day-clustered t of the mean difference) per N. Then the ONE-rule search: for each N and each single
threshold on held / dist_N / maxdd_N / above_vwap_N, the TRAIN mean fwd_R GROSS and NET (entry at c_N + half_spread
+ 2 bp; stop at original stop → new risk per share = c_N − stop; report both R0 units and new-risk units), n/wk,
ex-top-5 %, both TRAIN halves same sign. Rank by TRAIN-H2 net (H1 learns, H2 confirms) — pick the best cell with
≥ 3 fills/wk, record the FULL table (every cell, so multiplicity is visible).

## Stage B — frozen rule on VAL (one read)
The picked rule, unchanged, on VAL: n/wk, net mean R (R0 units and new-risk units), day-clustered t, ex-top-5 %,
winner-capped +3 R, weekly P10, green-week share vs count-matched null, year signs, vs B0 on the SAME arms and vs B0
on all arms, MDE at 80 % power. Placebo: the same rule with "held" replaced by "not held" (should be worse, else
the rule is the population). Cost sensitivity: ×1.5.

## Kill (pre-committed — any one kills)
VAL net ≤ 0 (R0 units) or day-t < 2; TRAIN halves opposite sign; VAL ex-top-5 % ≤ 0; < 3 fills/wk; placebo not
below the rule on VAL; the lift over B0 on the same arms ≤ +0.10 R. Survive → TEST once → independent rebuild →
live dry-run entry mode spec (`entry_mode: confirmation`, N minutes, same levels from bars_sip.db).

## Output
`1751_score.py`, `1751_anatomy.csv` (winners vs losers per N), `1751_grid.csv` (every rule cell, TRAIN),
`1751_val.csv`, `RESULT_1751.md` ≤ 350 words: the winners-vs-losers reading first (which state variables separate
at which N, and whether the separation is the already-earned part), then the rule, then VAL, KILL/SURVIVE.
Return ≤ 120 words.
