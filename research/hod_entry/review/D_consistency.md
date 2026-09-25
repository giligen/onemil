# LENS D — Consistency, tail, multiplicity review of cell 1,427's E1 resting-stop-limit claim

Adversarial pass on `research/hod_entry/sip_rebuild_val.csv` (TRAIN-H2 + VAL, quote-history run, Step 2c —
this IS the file the claim's numbers come from: 1,165 + 1,443 = 2,608 fills matches the claim exactly) and
`sip_rebuild_test.csv` (TEST, 972 fills, matches exactly). All numbers below recomputed fresh from these two
files (3,580 total fills) + `docs/cadence_bar.md` / `scripts/cadence_bar.py`. Nothing reused from REPORT_1427
except as a cross-check. **Verdict: the point estimates replicate exactly, but the claim fails multiplicity
badly and fails the cadence bar's shallow-reds criterion; TEST's edge is heavily one-week-carried.**

## 1. Monthly consistency — genuinely strong

Every fill-month from 2025-07 to 2026-09 (15 months, pooling TRAIN-H2+VAL+TEST) has **mean net R > 0**:
15/15 = 100%. Range +0.110R (2026-03) to +0.539R (2026-08). No losing month exists in this population.
This is the one dimension where the claim is not just defensible but *better* than advertised — a
15-for-15 monthly record is rare and worth flagging as real signal, not an artifact of the split boundaries
(TRAIN-H2/VAL/TEST all individually positive per-month too).

## 2. Weekly P&L — green share is real, but the shallow-reds bound fails

Pooled across all fills, at $100/trade risk, grouping by the CSV's own `wk` label (62 weeks with >=1 fill):
P10 = +$11.2, median = +$1,218.7, min = **-$1,716.5**, max = +$12,110.7. 56/62 weeks green (90.3%), 53/59
green excluding 3 flat weeks (89.8%).

Ran `scripts/cadence_bar.py` (owner's own scorer) on the fill population split by its fixed date ranges:

| Split | C1 gap | C2 bleed | **C3 reds** | C4 green | C5 freq | C7 power |
|---|---|---|---|---|---|---|
| TRAIN (2025) | pass (median 1.0wk) | pass | **FAIL** — min -8.84R, MDD 8.84R (cap 8R), 26wk underwater (cap 6wk) | pass (84% vs null 50%) | pass (22.0/wk) | FAIL (17 cycles, bootstrap UB 8.0wk) |
| VAL (2026-01..05) | pass (median 1.0wk) | pass | **FAIL** — min -17.17R, MDD 17.17R (cap 8R) | pass (95% vs null 51%) | pass (65.6/wk) | pass |

Both splits fail C3 (shallow reds) on a single catastrophic week each: **2026-03-21/27 (VAL): 72 fills,
-17.17R**, and a TRAIN-side bad week at -8.84R. Traced the VAL week trade-by-trade: it is not one outlier —
6+ individual stop-outs that week lose **-1.5R to -2.09R each** (NBR -2.09, ISSC -1.76, YSS -1.54, NKTR
-1.53, HYMC -1.53, CRDU -1.52), i.e. the nominal 1R stop is routinely blown through by 50-100%+ in a bad
week. This corroborates LENS B's finding independently (no size check / real stop slippage) from the P&L
side: **max observed net_R = +1.99 (hard-capped by the fixed 2R target), min observed net_R = -3.31** — the
loss tail is NOT capped the way the win tail is, and that asymmetry is exactly what produces the C3 failures.

**Caveat on this cadence-bar run**: it used the RAW (unslotted) fill population — every row where
`status=='fill'` — not the live-config, slot-capped population (REPORT_1427's own "fills/wk after slots"
column shows ~30-37/wk after 4-concurrent slotting, vs the ~22-66/wk raw rate used here). `docs/cadence_bar.md`
is explicit that a full-timeline walk of a book that runs live at 4 slots is not evidence of anything. I did
not have budget to re-run `run_consol.simulate_slots` to get the exact slotted weekly series, so the true
live-config C3/C5 result is **unknown, not passing** — this review cannot certify it either way, and it should
be run before any launch decision.

## 3. Tail concentration

Top-5% of fills (n=179 of 3,580) = 35.3% of total R; top-1% (n=36) = 7.1%. Winner-capping at +2R or +3R
changes NOTHING (mean stays 0.2785, 0 trades capped) because the exit spec hard-caps the win side at 2R by
construction — so the classic "lottery ticket" pattern (a few 5R+ trades carrying the book) genuinely does
not apply here. But **week-level** concentration is real on TEST specifically: one week, 2026-08-01/07 (144
fills, win rate 100/144 = 69.4% vs the TEST-wide 52.4%), contributes **+121.1R of the TEST period's +320.7R
total — 37.8% of the entire sealed TEST result from one week's 6% of TEST fill-days**. TEST is 12/14 green
weeks (86%), but the PASS verdict leans heavily on this one anomalously-high-win-rate week.

## 4. Count-matched random-entry null

Used `b0_net_R` (every signal's row, fill or not — the same B0 baseline execution population that
`research/hod_exit_lab/b0_trades.csv` and REPORT_1427's own "no-fill cohort B0" column are built from;
confirmed in `sip_rebuild.py` — `b0_net_R=r.net_R` sourced from `research/hod_exit_lab/paths.parquet`, B0's
path rules) as the draw population (n=11,641, mean -0.2951). Drew n=3,580 (seed 42, no replacement, 200
draws) from this population 200 times:

- Null distribution of the 200 draw-means: mean **-0.2946**, P5 -0.3258, P50 -0.2945, P95 -0.2631.
- Observed E1 fill-mean: **+0.2785**. Rank: **200/200** — every single null draw is below the observed
  mean by a wide margin (~0.57R).

This null is legitimate but answers a narrower question than "does the edge exist": it shows the rv-gated
signal + resting-fill selection adds real value **over a blind draw from the already-detected HOD-break
population**, not that HOD-break signals in general beat the market. Given the population is a subset chosen
by the strategy's OWN admission filters (rv, price, etc.), this comparison cannot rule out that both the
admission filter and the fill selection share the same in-sample bias — it rules out "fill selection is
noise," not "the whole pipeline is noise."

## 5. Multiplicity — the decisive finding

Cell IDs found live in this repo's history (`git log --all`, filenames): 1289, 1328 (ORB union rung), 1355,
1412, 1417, 1423, 1426 (ORB latency replay), **1427** (this claim), 1438 (causal-arming amendment). The
counter is confirmed GLOBAL/shared across books — 1,328 and 1,426 are explicitly ORB cells interleaved with
HOD cells — so by cell 1,438 at least **1,438 research cells** have been scored across the whole programme,
exactly the number CLAUDE.md's own multiplicity rule points to. Separately, HOD-break-specific cells
documented in CLAUDE.md alone: 12 (causal-filter, closed 9/18) + 35 (exit lab 9/22) + 4 (rank-refuted,
1,351-1,354) + the ~8 entry/exit variants inside REPORT_1427 itself (E1 15bps / limit 5bps / limit 30bps /
ask+1tick, x2 methodology passes) = a conservative floor of **~60 HOD-line cells** even before counting
unlogged exploratory runs.

Bonferroni-style bar (two-sided, family-wise alpha 0.05, t-distribution, df = trading days per split):

| N tests | t_crit needed (TEST, df=68) | TEST t=2.12 | t_crit (TRAIN-H2, df=120) | t=3.45 | t_crit (VAL, df=102) | t=3.53 |
|---|---|---|---|---|---|---|
| 12 (HOD causal-filter cells alone) | 2.97 | **FAIL** | 2.92 | PASS | 2.93 | PASS |
| 60 (conservative HOD-line floor) | ~3.47 | **FAIL** | ~3.45 | borderline PASS | ~3.46 | PASS |
| 1,438 (global programme count, CLAUDE.md's own multiplicity rule) | 4.44 | **FAIL** | 4.30 | **FAIL** | 4.33 | **FAIL** |

**TEST's t=2.12 fails a Bonferroni bar at the smallest plausible N tried (12) — its raw two-sided p-value is
0.038, which survives at most ~1.3 independent tests at family-wise alpha 0.05.** TRAIN-H2 and VAL survive a
HOD-line-scoped multiplicity count (N~12-60) but fail decisively at the full-programme count CLAUDE.md itself
says to use (N=1,438). The PASS verdict in REPORT_1427's Step 3 table did not apply any multiplicity
correction — its bar (t >= 2) is a single-test bar, not a multiple-comparisons bar, and CLAUDE.md section 7
requires the latter.

## Bottom line for LENS D

Monthly consistency (15/15 positive months) and the count-matched null (200/200) are real, honest positives
that argue the entry-timing/fill-selection mechanism is doing something, not pure noise-mining. But three
things should stop an unconditional "lock and launch": (1) the cadence bar's own shallow-reds criterion (C3)
FAILS on both TRAIN and VAL from stop-slippage clusters that independently corroborate LENS B's no-size-check
finding; (2) TEST's PASS is ~38% carried by a single anomalous week; (3) at the multiplicity scale CLAUDE.md's
own rule prescribes (1,438 cells program-wide), none of the three splits' t-stats clear a Bonferroni bar, and
even TEST alone fails at the smallest defensible cell count. None of this proves the edge is fake — but it is
not yet a "locked, no-excuses" result either.
