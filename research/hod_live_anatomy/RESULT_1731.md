# RESULT — cell 1,731: thin breaks, a CAUSAL break-volume filter (Opus, 2026-10-31) — **FAIL on all three measures**

PREREG: `PREREG_1731.md` (frozen). Script `cell_1731.py` (reads `cache/opens_1730.parquet`, writes resumable
`cache/vol_1731_bars.parquet` / `cache/vol_1731_tape.parquet`), outputs `vol_1731.csv` (V1), `filter_1731.csv` (V2),
`live_1731.csv` (V3), log `cell_1731_out.txt`. bars_sip.db and the sip caches were read only.

## Book and data
* **Primary book = 1,730's population B1 at the live cap**: the first armed crossing per symbol-day, filled when the
  trigger-bar open is ≤ +15 bps above the level (`d_bps ≤ 15`), exits by the live rule, cost 12 bps on non-target
  exits. n 7,190 (TRAIN-H2 3,113 over 27 wk / VAL 4,077 over 22 wk). Net +0.039 / +0.016 R/trade. These levels are
  optimistic because the entry model is about 4 bps cheaper than the NBBO ask (1,730 deviation 5). The deltas are paired,
  so that bias drops out of them. Population A (the 5,417 recovered 1,663 BT fills) was scored the same way, report-only.
* **Tape** = the 1,438 SIP caches (`sip_cache/c1438_{day}.pkl.gz`, key sym|m_lo|m_hi; 1,427's `{day}.pkl.gz` as a
  fallback, used twice). Crossing print = the first trade in minute T at ≥ trigger (level + 1 c). The tape and the
  bars agree exactly: tape-minute volume / bar v_T has median 1.000 and IQR 1.000–1.000. The median cross comes
  15.3 s into T.

## Availability rail
| | B1 book | A |
|---|---|---|
| tape found | **84.0 %** (winners 84.4 % / losers 83.8 %, gap 0.6 pp) | 100 % |
| cross in the first 5 s of T (M1 undefined, by PREREG) | 1,547 (21.5 %) | 1,410 (26.0 %) |
| M1 defined | 62.5 % (missing: winners 36.0 % / losers 38.4 %, gap 2.5 pp) | 74.0 % (gap 3.0 pp) |

The rail passes on tape coverage (≥ 80 %, gap ≤ 5 pp), so the tape cell is read and not VOID. M1 is defined on only
62.5 % because the PREREG excludes first-5-s crosses; a strict reading of the rail on the defined share would VOID the
tape cell, and the verdict is FAIL either way. Caveat: the 16 % of rows with no tape are worse on average
(+0.010 / −0.027 R vs +0.047 / +0.051 for tape-ok rows), even though their win-rate gap is only 0.6 pp.

## V1 — quintiles (cuts fixed on TRAIN-H2), B1 book, net R/trade
| measure | TRAIN-H2 Q1 … Q5 | VAL Q1 … Q5 | Q1 worst? |
|---|---|---|---|
| M1 pre-cross rate (tape) | +0.027, +0.077, +0.096, −0.024, +0.060 | **−0.072**, +0.042, +0.081, +0.155, +0.035 | VAL only |
| M2 last-bar ratio | +0.071, +0.045, +0.062, +0.035, −0.017 | +0.025, −0.015, −0.003, +0.106, −0.026 | no (TRAIN Q1 is the BEST) |
| M3 cum. volume / ADV20 | +0.007, +0.053, +0.057, +0.025, +0.055 | +0.032, +0.045, −0.017, +0.030, −0.016 | TRAIN only |
| VT (1,730's leaking v_T ratio, ref.) | +0.080, −0.015, −0.009, +0.002, +0.139 | −0.034, −0.035, +0.058, +0.069, +0.023 | VAL only |

Sign agreement of Q1 vs Q5 across splits: M1 has Q1 < Q5 in both splits, but the pattern is not monotone (TRAIN Q4 is
the worst quintile, VAL Q4 the best). M2 has Q1 > Q5 in both splits, which is the reverse of the mechanism. M3
disagrees between splits. Day-clustered |t| per quintile is ≤ 2.2 everywhere. Ex-top-5 % per quintile is in
`vol_1731.csv`.

## V2 — drop the thinnest TRAIN-H2 quintile (undefined M1 kept), paired vs the unfiltered book
| measure | split | dropped | Δ net R (t) | halves A / B | R/wk all → kept | ex-5 % Δ | MDE80 |
|---|---|---|---|---|---|---|---|
| M1 | TRAIN-H2 | 12.2 % | +0.002 (0.18) | −0.001 / +0.004 | 4.53 → 4.15 | +0.002 | 0.027 |
| M1 | VAL | 12.7 % | +0.013 (1.57) | +0.015 / +0.011 | 2.97 → 4.64 | +0.014 | 0.023 |
| M2 | TRAIN-H2 | 20.0 % | −0.008 (−0.68) | −0.005 / −0.010 | 4.53 → 2.89 | −0.008 | 0.033 |
| M2 | VAL | 22.0 % | −0.003 (−0.23) | −0.011 / +0.008 | 2.97 → 1.93 | −0.002 | 0.033 |
| M3 | TRAIN-H2 | 20.0 % | +0.008 (0.54) | +0.007 / +0.008 | 4.53 → 4.36 | +0.008 | 0.042 |
| M3 | VAL | 25.8 % | −0.006 (−0.31) | −0.021 / +0.013 | 2.97 → 1.44 | −0.006 | 0.050 |

Kept fills/week are 92–162, far above 3. Restricted to rows where M1 is defined, the M1 delta is +0.005 (t 0.32) on
TRAIN-H2 and +0.030 (t 2.20) on VAL. That is VAL only, and it is below the Bonferroni-3 bar of |t| ≥ 2.39.
Winner-capped deltas equal the means to within 0.002. Population A (report-only) also fails every measure:
M1 0.000 / +0.013, M2 +0.011 / +0.021 (t 0.66 / 1.50), M3 +0.018 / −0.013.

## V3 — live sign check (ledger twin; it cannot pass on its own)
* Ledger definitions (`trading/hod_break.py` l.207–238): `cumvol_over_adv20` = the volume through the arm bar ÷ adv20,
  the same definition as **M3**. `arm_bar_vol_x` = v_j ÷ mean(v over bars 0..j, the day so far), which is **not M2**
  (M2 divides by the mean of the last 10 bars). It was used through its BT twin M2L, with the cut taken from the BT
  M2L quintiles. M2 has no ledger twin, and M1 has no live tape (none was fetched).
* The feature columns are filled only from 10/19, the day the engine began writing them (0 % before, 100 % after), so
  98 of the 231 R-valid live fills are usable.
* M3, cut 0.128: kept 49 at −0.066 R, dropped 49 at −0.027 R, kept − dropped **−0.039**, CI [−0.55, +0.48]. Half of
  the live fills fall below the BT Q1 cut.
* M2L, cut 0.488: kept 72 at −0.013 R, dropped 26 at −0.140 R, difference **+0.128**, CI [−0.50, +0.74]. The BT M2L Q1
  is not the worst quintile (+0.013 / +0.040), so the live sign does not agree with the BT.

## Verdict: FAIL on M1, M2 and M3 — no engine filter, nothing to spec
Pre-committed consequence: report the MDE per measure and whether the 1,730 read was look-ahead.
* **MDE** (80 % power, paired delta, full book): M1 0.027 / 0.023 R (0.044 / 0.039 on M1-defined rows), M2 0.033 /
  0.033, M3 0.042 / 0.050 (TRAIN-H2 / VAL). The test could see the +0.03 R bar on M1 and M2. No delta reached it, and
  M2's point estimate is negative.
* **Was the 1,730 read look-ahead?** By the PREREG rule ("M1/M2 flat while v_T is not"), **no**: v_T is also flat on the
  filled book (TRAIN-H2 Q1 +0.080). The 1,730 thin-break read (Q1 −0.095 / −0.086) was a **composition artefact**. It
  was taken on all B1 crossings, including the 15 % that open more than 15 bps through the level, which the live cap
  never fills. Those make up 30 % of v_T-Q1 (11 % elsewhere) and lose −0.37 R (TRAIN-H2) / −0.15 R (VAL) in v_T-Q1.
  Among fills at the live cap, break volume (causal or not) carries no reliable information.
* Report-only: a cross in the first 5 s of T loses −0.056 R on VAL against +0.051 for the rest, but earns +0.041 vs
  +0.047 on TRAIN-H2. The splits disagree, so this is not ranked.

## Ranked next hypotheses (each needs its own PREREG)
1. **Retrace-fill adverse selection** (1,730 #2). This is the only untested entry lever with a mechanism. The 1,438 tape
   now has measured 84 % coverage of the book and an exact match to the bars.
2. **Spread-scaled limit / what is paid** (1,730 #3). Gross at the level is flat across slip buckets, so the cost side is
   the lever.
3. **M1 as a forward instrument, not a filter.** Log the pre-cross rate in the dry ledger: the engine sees trade prints
   through the StopMonitor. The VAL-only covered-row delta (+0.030, t 2.2) would then get an out-of-sample read at zero
   cost. It is not a test claim.

## Deviations
1. Primary book = B1 at d ≤ 15, because the PREREG named "the population as in 1,730" without choosing between A and
   B1. A is reported alongside and fails as well.
2. The M1 denominator is v_cons10 / 60 using the bars present (1,730's v_cons10). No-trade minutes are not zero-filled,
   so on thin names the per-second consolidation rate is slightly overstated.
3. In V2, rows with M1 undefined (first-5-s crosses, no tape) are kept in the book. The M1-defined-only scope is
   reported alongside.
4. V3 uses M2L ↔ `arm_bar_vol_x`, because M2 has no ledger twin. 98 of 231 live fills are usable, since the features
   exist only from 10/19.
5. ADV20 comes from `research/bf_zero/universe.csv`, the arming's source. Live adv20 may come from a different source.

## Multiplicity
HOD live-book cells 1,721–1,731. This cell tests 3 measures with 1 filter each, so the Bonferroni bar is |t| ≥ 2.39.
The best full-book |t| is 1.57 (M1, VAL).
