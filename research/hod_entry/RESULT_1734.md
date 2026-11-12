# RESULT 1,734 — HOD-break book by SPY regime (PREREG_1734, one cell, run 2026-11-12)

**Verdict: FAIL** on the primary feature `spy_ret20` (and on both secondaries). Not a regime edge; do not size by regime.

`base_R` = live-rule NET R per fill from the 1,663 join, floored r_pct >= 1.5 % (PREREG_1681 line 6-7); costs per
PREREG_1670 line 10: entry 7 bps, stop 6, EOD 11, early cut 6. 5,506 fills / 229 days; forward 306 paper fills
(307 with pnl; one dropped for total_risk <= 0). Cuts (frozen, day-level): ret20 [+0.67 %, +2.83 %], ret1 [-0.14 %, +0.40 %].

Bottom (bearish) vs top (bullish) tercile of `spy_ret20`, mean net R/fill (n, day-t, ex-top-5 %, placebo pct):

| Read | Bottom | Top | Delta (day-t) |
|---|---|---|---|
| TRAIN-H2 | +0.183 (601, t 1.48, ex5 +0.086, plc 96.2) | -0.096 (741) | +0.278 (1.89) |
| VAL | +0.003 (1,504, t 0.04, ex5 -0.101, plc 62.4) | -0.032 (1,205) | +0.035 (0.33) |
| POOLED | +0.055 (2,105, t 0.78) | -0.056 (1,946) | +0.111 (1.27, plc 90.6) |
| FORWARD | -0.030 (185, t -0.26) | empty (no forward day had ret20 > +2.83 %) | n/a |

Pass bar (bottom >= +0.10, day-t >= 2.0, ex-top-5 % >= 0, placebo >= 95 in BOTH TRAIN-H2 and VAL, forward same sign):
TRAIN-H2 meets mean, ex-top-5 % and placebo but not t (1.48); VAL fails all four; forward is negative. FAIL.
Secondaries: `spy_below_50` bottom = below-50d: TRAIN-H2 -0.085, VAL +0.026, forward -0.110; `spy_ret1` bottom: +0.009,
-0.071, -0.199. Neither passes. Frequency: bottom-ret20 days carry more fills per session in TRAIN-H2 (24.0 vs 17.2) but
not in VAL (28.9 vs 35.4); no consistent frequency effect.

Caveats. (1) base_R is a modelled net R (fixed bps costs, not per-trade NBBO), forward is realised paper P&L / risk.
(2) Day-clustered SE is large: bucket n is 185-1,500 fills but only 21-52 days; the forward read has MDE ~0.45 R.
(3) Few regime episodes: bottom-ret20 runs = 8 (TRAIN-H2), 5 (VAL), 4 (forward), 13 pooled; top = 8 / 2 / 0 — the
sample is a handful of independent regimes. TRAIN-H2's +0.18 does not replicate in VAL, so the bearish-tape story is
at best a TRAIN-H2 regime artefact. Independent rebuild (REBUILD_1734.md) still owed before any number reaches the owner.

Files: `1734_score.py`, `1734_buckets.csv`, `1734_fills.csv`.
