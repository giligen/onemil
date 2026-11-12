# PREREG 1,742 — research around the retest bid (the withdrawn 1,737 survivor): anatomy on TRUE levels + two causal variants

FROZEN 2026-11-12 23:20 UTC before any number. Programme count on the HOD line: 1,741 → 1,742. Rules as PREREG_1740
(no git / service / orders; read-only everywhere; write only `research/hod_entry/1742_*` and `RESULT_1742.md`;
`walker.fetch_day_bars` on a read-only `bars_sip.db` URI, `walker.wait_for_db_window` honoured; `nice -n 19`;
background script + log polling; never Read a file > 300 lines in full; budget ≤ 60 calls). This task IS the
owner's request ("do a research around the survivor to find a winning strat"); do not pivot.

## What we know (RESULT_1740)
On true SIP levels (`1740_arms.csv`, 12,135 arms TRAIN/VAL) the retest bid at the level loses (−0.09/−0.18 R
strict) and the selection runs the wrong way: arms that come back to the level have B0 −0.31/−0.42 R, arms that
never retest have B0 +0.68/+0.73 R. The retest is INFORMATION; the bid is on the wrong side of it. A confirmation
entry k bars after the break (hold, then buy) was already tested: cell 1,487 −0.13 R — do not repeat it.

## Part 1 — anatomy of the strict M1 fills (ranked hypotheses, NO pass/fail, no claims)
For every arm compute from the bars: C (first close ≥ level at/after entry_m), hold_k = bars from C+1 until the
first low < level (∞ if never), retest_depth = (level − min low over the 3 bars from the touch)/level, chase =
(B0 `entry` − level)/level, half_spread (signals.parquet), C minute-of-day, bars_since_open at C, prior 5-bar range
before the break /level. For the M1 strict fills (conventions of PREREG_1740) report mean net R and n by quintile of
each feature on TRAIN and VAL separately, the top−bottom day-clustered t, and the same for B0 net R on all arms.
Rank the features by |t| on VAL with the TRAIN sign agreeing; present as hypotheses for a later PREREG.

## Part 2 — two pre-named causal variants (scored, pass bar below)
V1 hold-gated retest bid (primary, k = 5): bid at `level` only from bar C+k+1 if min low over C+1..C+k ≥ level
(the first pullback after a hold); cancel on a close ≥ level × 1.02 or at 15:30; fill on the first bar with low <
level (strict) and the 10 bp variant; stop / target / costs exactly PREREG_1740. Exploratory: k = 3, 10.
V2 retest-and-reclaim: after the first bar at/after C+1 with low < level (the touch), buy at the CLOSE of the first
later bar with close ≥ level, within 15 bars of the touch (else no trade); stop = PREREG_1740 stop; target
fill + 2 R; entry cost half_spread + 2 bp; exits as PREREG_1740.
For both: n, fill share, mean net R, day-clustered t, iid t, ex-top-5 %, winner-capped, stop share, green-week share,
weekly P10, fills/week; capped book (8/day, 4 concurrent, first-come); filled vs unfilled cohorts' B0 net R;
placebo 300 draws (same geometry from a random minute after the gate). Splits TRAIN H1, H2, TRAIN, VAL; TEST sealed.

## Pass bar (pre-committed; primary = V1 k=5 strict; V2 and k=3/10 are exploratory, 4 reads noted)
Mean net R ≥ +0.10 with day-t ≥ 2.0 on TRAIN AND VAL, both TRAIN halves same sign, 10 bp ≥ +0.05 on both,
ex-top-5 % ≥ 0, placebo ≥ 95 on VAL, ≥ 3 fills/week. Pass → TEST read ONCE, then an independent rebuild from this
prose before the owner sees it. Fail → stop, TEST untouched; the anatomy ranking still goes into RESULT_1742.

## Output
`1742_score.py`, `1742_features.csv` (one row per arm), `1742_orders.csv`, `1742_summary.csv`, `RESULT_1742.md`
≤ 400 words (anatomy table top 6 features, V1/V2 tables, capped book, PASS/FAIL, caveats). Return ≤ 120 words:
top 3 anatomy features (feature, VAL spread, t), V1 k=5 and V2 mean R / day-t / n, PASS/FAIL.
