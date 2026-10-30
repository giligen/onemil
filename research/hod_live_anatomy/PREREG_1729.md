# PREREG — cell 1,729: is the TIGHT-R cohort the profitable part of HOD? (from cell 1,728, 2026-10-30)

## Where it comes from (honest provenance)
1,728 found no live-vs-BT gap (the old +0.2 R was cell 1,438's banded cost) and a parity defect: the research twin
(`research/hod_entry/causal_arming.py` l.81) rejects arms with (trigger − stop)/trigger < `min_r_pct` (1.5 %) and
trigger < `min_price`; the live engine has NO such gate (`trading/hod_break.py`), so 82 of the 231 live fills have
R < 1.5 % of price. Those 82 are the only net-positive cohort of the live book: R < 0.75 % n 29 **+0.105 R**,
0.75–1.5 % n 53 **+0.090 R**, versus 1.5–3 % n 102 −0.066 and ≥ 3 % n 47 −0.078 (Fable, from `fills_1721.csv`).
1,728 also noted that the 9,911-fill population of cell 1,438 (which includes the 0.75–1.5 % bucket) sits ≈ +0.04 R
above the 5,506-fill population that excludes it. So the hypothesis was SEEN on live and partly on the BT; TRAIN-H2/VAL
are therefore a replication, not a discovery, and the sealed TEST split is the real confirmation. Multiplicity: this
is the 13th cut on the live book tonight; Bonferroni applies to the live number, not to a sealed-TEST read.

## Mechanism (stated before the number)
A tight consolidation under the high (stop close to the trigger) is a coiled, low-noise base; the breakout's gross in R is
large relative to the spread cost in R precisely because R is small in price terms — the opposite of the "R must exceed
the spread" prior, which assumed the gross in price terms is fixed. If true, the min_r_pct gate has been removing the
book's edge in the backtests since it was introduced.

## Data
Cell 1,438's 9,911-fill population (locate its per-fill file via `research/hod_entry/cell_1438*.py` / `REPORT_1438.md`;
if only the 5,506 + the 0.75–1.5 % bucket are recoverable, say so) with its TRAIN-H2 / VAL / TEST splits; minute paths or
`bars_sip.db` for exits as in 1,681/1,727; cost = measured (entry slip as observed ≈ 8 bps, exit 12 bps marketable).
Live: `fills_1721.csv`.

## Cells (all frozen)
C1 cohort R_pct < 1.5 % vs ≥ 1.5 % — net R, n, day-clustered t, ex-top-5 %, winner-capped (+2 R), fills/week; per split.
C2 the same at R_pct < 0.75 % (sub-cohort) — report only.
C3 the complement rule as a BOOK: keep only R_pct < 1.5 % — weekly P&L series, green-week share vs a count-matched
random null, max drawdown, at $50 and $150 R; TRAIN-H2 halves by alternating weeks.
**TEST is read ONCE, for C1 and C3 only, and only if C1 passes on BOTH TRAIN-H2 and VAL.**

## Pass bar
C1 kept (R_pct < 1.5 %) net ≥ +0.05 R/trade on TRAIN-H2 AND VAL AND TEST, day-clustered t ≥ 2 pooled, ex-top-5 % > 0
on each split, both TRAIN halves same sign, ≥ 3 fills/week at the live rate (scale by the live/BT ratio), and the
live 82 fills ≥ 0 (they are +0.09; report their CI). Also the complement must be ≤ 0 on each split (else it is not a
cohort but noise). PASS → PREREG the live rule change (R_pct ≤ 1.5 % as an explicit gate in BOTH the twin and the
engine, the twin's current min gate inverted) with a weekend rehearsal; FAIL → the parity defect is fixed the other way
(the twin's gate goes live) and the cohort is reported as noise with its CI.
