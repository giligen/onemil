# PREREG 1,743 — variance-ratio / first-passage diagnostic on the stored HOD paths (wide list G1, Lo–MacKinlay)

FROZEN 2026-11-13 05:55 UTC before any number. Programme count on the HOD line: 1,742 → 1,743. Rules: no git /
service / orders; read-only everywhere; no DB (paths.parquet only, pyarrow with selected columns, `nice -n 19`,
scorer as a script file run in the background with log polling); write only `research/hod_entry/1743_*` and
`RESULT_1743.md`; never Read a file > 300 lines in full; budget ≤ 50 calls. This task IS the owner's request
(the wide-list top item after seven geometry cells failed); do not pivot. TEST (split == 'TEST') is never read.

## Question
Seven cells (1,735–1,742) show every entry and exit geometry on this population losing net. A stop/target design
earns only if post-entry returns trend (VR > 1); on a driftless or mean-reverting path the stop is hit first with
probability ≥ k/(k+m) and the design is bounded below zero after cost. This cell MEASURES that, so the next decision
is about the population, not another geometry.

## Data
`research/hod_exit_lab/paths.parquet` (day, symbol, m, c, vwap, minutes_since_entry) for signals in
`signals.parquet` with split in {TRAIN, VAL} (TRAIN 7,390 / VAL 4,745); `b0_trades.csv` net_R for the subgroup
tables; signals columns R, r_pct, half_spread, entry_m for the subgroups.

## Statistics (per split; TRAIN also H1/H2)
1. Variance ratio on post-entry log returns of `c`, pooled across signals: VR(q) = Var(q-min return) /
   (q · Var(1-min return)) for q ∈ {2, 5, 10, 15, 30, 60}, computed over minutes_since_entry 1..120 and separately
   over 1..15 and 16..120; the heteroskedasticity-robust Lo–MacKinlay z and a signal-level block bootstrap (1,000
   draws, resample signals) 95 % CI. Also the same VR on the 60 minutes BEFORE entry (minutes_since_entry < 0, if
   present) as the pre-break reference.
2. First passage: for every signal, does the path hit −1 R (close ≤ entry − R) before +2 R (close ≥ entry + 2 R)
   within the session? Report P(stop first), P(target first), P(neither), against the driftless bound 2/3, with the
   signal-level bootstrap CI; repeat for (−1 R, +1 R) and (−0.5 R, +1 R). No cost here — this is the gross geometry.
3. Drift: mean log return per minute post-entry for minutes 1..15, 16..60, 61..120 (bps/min) with signal-clustered
   SE; and the mean return from entry to 15:55 (gross, bps and in R).
4. Subgroups (exploratory, 9 reads): terciles of half_spread, r_pct, entry minute-of-day; VR(10) over 1..120 and
   P(target first) per tercile, bootstrap CI; B0 net R of the tercile beside it.

## Pre-committed reading (no money claim either way)
* VR(10) and VR(30) over 1..120 with the 95 % CI entirely below 1.0 on TRAIN and VAL AND P(target first) for
  (−1 R, +2 R) ≤ 1/3 → the population is mean-reverting after the break: no stop-based long design can be net positive
  here; the programme stops testing geometry on this population and moves to population/cost.
* CI entirely above 1.0 on both → trend exists; the loss is cost; the queue-timing cell (next) is the right lever.
* CI straddles 1 → undecided; report the MDE in VR units.
* A subgroup with VR(10) CI above 1 on BOTH splits and P(target first) > 1/3 on both becomes a PREREG candidate
  (named in RESULT, not scored for money here).

## Output
`1743_score.py`, `1743_vr.csv`, `1743_passage.csv`, `1743_subgroups.csv`, `RESULT_1743.md` ≤ 350 words (tables, the
pre-committed reading that applies, caveats: close-to-close returns on 1-min bars carry bid-ask bounce that biases
VR(2) down — report VR from `vwap` as well). Return ≤ 120 words: VR(10)/VR(30) with CIs per split, P(stop/target
first) for (−1, +2), drift by window, which reading applies, any subgroup candidate.
