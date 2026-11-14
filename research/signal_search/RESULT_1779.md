# RESULT 1,779 GAPSHORT frequency frontier (marginal rows per rung, VAL 2026-01..05, cost 40.8 bp stack)

| rung | names/wk added (med) | names/day after (med) | VAL net bp | day-t | ex-top-5% | worst-5% | wkP10 comb (R) | placebo p95 (i / ii) | PASS |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 14.0 | 8 | +3.0 | 0.05 | -104 | -19.1% | -0.86 | +14 / +11 | FAIL |
| R2 | 9.0 | 7 | -48.8 | -0.83 | -162 | -18.4% | -0.79 | n/a (VAL<=0) | FAIL |
| R3 | 7.0 | 7 | -103.6 | -1.92 | -181 | -16.3% | -0.73 | n/a (VAL<=0) | FAIL |
| R4 | 7.0 | 6 | -39.0 | -0.54 | -127 | -18.5% | -0.69 | n/a (VAL<=0) | FAIL |
| R5 | 6.5 | 6 | -81.7 | -1.27 | -164 | -17.9% | -0.91 | n/a (VAL<=0) | FAIL |
| R6 | 3.0 | 6 | -104.8 | -1.37 | -191 | -16.4% | -0.94 | n/a (VAL<=0) | FAIL |

Pass bar: net >= +50 bp, t >= 2, ex-top-5% > 0, worst-5% >= -12%, both TRAIN halves same sign, beats both p95s, >= 3 names/wk. No rung passes (every rung fails net, ex-top-5% and worst-5%).

## TRAIN reads (marginal rows)
| rung | TRAIN net | n | H1 net | H2 net | same sign |
|---|---|---|---|---|---|
| R1 | +65.7 | 359 | +113.4 | +16.6 | yes |
| R2 | +78.0 | 263 | +132.2 | +35.3 | yes |
| R3 | +24.8 | 165 | +45.0 | +8.0 | yes |
| R4 | +19.6 | 184 | -20.7 | +62.6 | no |
| R5 | -52.8 | 152 | +62.6 | -118.2 | no |
| R6 | -50.3 | 83 | -69.8 | -40.4 | yes |

## Details (VAL)
| rung | rows | valid | coverage | miss gap pp | names/wk p10 | rung wkP10 R | combined wk mean R |
|---|---|---|---|---|---|---|---|
| R1 | 303 | 297 | 98% | 2.5 | 9.2 | -0.86 | 0.66 |
| R2 | 207 | 190 | 92% | 3.2 | 4.0 | -0.56 | 0.57 |
| R3 | 156 | 155 | 99% | 1.2 | 1.1 | -0.46 | 0.51 |
| R4 | 156 | 154 | 99% | 0.1 | 4.0 | -0.76 | 0.60 |
| R5 | 140 | 139 | 99% | 0.7 | 3.0 | -0.51 | 0.55 |
| R6 | 83 | 83 | 100% | 0.0 | 1.0 | -0.33 | 0.57 |

R5 secondary (cover at measured ask1100, rows with it): n=42 mean -92.7 bp; R6 n=23 mean -125.1 bp.

## Notes / caveats
* Base book = 1776R book (1,087 rows, reproduced exactly). L1 tercile cutoffs follow the 1776R convention (ALL TRAIN population rows, not top-10 rows): T3 >= 7,224,163 (matches 1776R); T2 >= 945,387. Top-10 TRAIN-only T3 cutoff would be 16,970,246 (not used).
* Quote source: quotes.jsonl first, else 1774 bid/ask1030 (agreement on 1,085 overlapping rows: 100 % exact); ~656 missing rows fetched (appended to quotes.jsonl). Rows with no/invalid quote are dropped (coverage column).
* Placebo run only for rungs with VAL mean > 0 (R1 only). Placebo (i) pool = a 600-row sample of same-day non-base non-rung quoted VAL rows (not the full population); (ii) = same symbol +-5 trading days. 200 draws.
* Rungs R5/R6 enter at bid - 10 bp, cover at the close; rows lacking `close` are invalid. R5/R6 VAL means are negative before placebos.
* Gap rank is computed on the 1776R universe (population CSV + daily_bars gap, gap <= 200 %), identical to the base book definition.

CHILDREN: (1) is the R1 +3 bp mean a placebo-level result, i.e. is any L1 gating adding information beyond gap rank; (2) short side of the 10:30 extended survivors is negative, so test the continuation (long/hold-through) mirror only under its own PREREG with GROSS edge first; (3) frequency from other signal families (non-gap) rather than widening gap/L1 gates.
