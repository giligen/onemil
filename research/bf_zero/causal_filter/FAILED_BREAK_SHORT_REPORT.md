# FAILED_BREAK_SHORT_REPORT — PREREG_FAILED_BREAK_SHORT.md (cells 1357-1358)

Population (TRAIN+VAL, TEST untouched): 12135. Shortable-proxy excluded: 43.72%. SSR-proxy excluded: 0.38%. Kept for simulation: 6797.


## Cell 1357

No-trade reasons (share of kept population): no_failure 20.1%, r_too_small 11.2%

NBBO coverage on real trades: 100.0%

| split    |    n |   tpw |   meanR_gross |   meanR |    sd |   t_iid |   t_clust |   WR |    ex5 |   cap5 |   share_stop |   share_target |   share_eod |   green |
|:---------|-----:|------:|--------------:|--------:|------:|--------:|----------:|-----:|-------:|-------:|-------------:|---------------:|------------:|--------:|
| TRAIN    | 2903 | 54.77 |         0.022 |  -0.33  | 2.673 |   -6.66 |     -1.65 | 21.2 | -0.801 | -0.511 |        0.772 |              0 |       0.228 |    0.26 |
| VAL      | 1767 | 76.83 |        -0.055 |  -0.478 | 2.601 |   -7.73 |     -5.21 | 17.1 | -0.956 | -0.656 |        0.809 |              0 |       0.191 |    0.13 |
| TRAIN-H1 | 1712 | 32.3  |        -0.008 |  -0.354 | 2.407 |   -6.08 |     -1.09 | 23.2 | -0.77  | -0.488 |        0.749 |              0 |       0.251 |    0.09 |
| TRAIN-H2 | 1191 | 22.47 |         0.064 |  -0.297 | 3.017 |   -3.39 |     -2.27 | 18.3 | -0.836 | -0.544 |        0.804 |              0 |       0.196 |    0.17 |


Placebo means:
| split   | placebo                       |    n |   meanR |
|:--------|:------------------------------|-----:|--------:|
| TRAIN   | D1 (time-shuffle, same sym)   | 1797 |  -0.765 |
| TRAIN   | D3 (symbol-shuffle, same day) | 1501 |  -0.654 |
| VAL     | D1 (time-shuffle, same sym)   | 1076 |  -0.933 |
| VAL     | D3 (symbol-shuffle, same day) |  992 |  -0.891 |

VAL week-by-week P&L at R=$100:
wk
2025-12-27/2026-01-02    -1209.0
2026-01-03/2026-01-09    -4016.0
2026-01-10/2026-01-16     -401.0
2026-01-17/2026-01-23      264.0
2026-01-24/2026-01-30     3481.0
2026-01-31/2026-02-06    -3415.0
2026-02-07/2026-02-13    -8101.0
2026-02-14/2026-02-20    -3921.0
2026-02-21/2026-02-27    -3464.0
2026-02-28/2026-03-06    -3770.0
2026-03-07/2026-03-13     -414.0
2026-03-14/2026-03-20    -4306.0
2026-03-21/2026-03-27    -5332.0
2026-03-28/2026-04-03   -12240.0
2026-04-04/2026-04-10     -649.0
2026-04-11/2026-04-17    -6597.0
2026-04-18/2026-04-24     2188.0
2026-04-25/2026-05-01    -5280.0
2026-05-02/2026-05-08    -6523.0
2026-05-09/2026-05-15    -5549.0
2026-05-16/2026-05-22   -10148.0
2026-05-23/2026-05-29    -5080.0

Cadence bar (VAL):
```
CADENCE BAR  (HOD-S-1357, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median 12.0 wk  P90 12.0 wk        [fail]   gaps: [12]
C2 bleed     P90 -522.09 R     cycles net>0 0% [fail]
C3 reds      P10 -79.51 R  min -122.40 R  MDD 844.83 R   under-water max 22 wk   [fail]
C4 green     14%  null 50%                   [fail]
C5 fills/wk  80.32                             [pass]
C6 tail      C6 not audited
C7 power     cycles 1   bootstrap P90-gap 75% UB 11.0 wk    [fail]
diagnostics  ex-top-5% -879.64 R   capped -891.52 R   top-5 share -4.1%   weekly P&L histogram: [-122.4, -101.5, -81.0, -66.0, -65.2, -55.5, -53.3, -52.8, -50.8, -43.1, -40.2, -39.2, -37.7, -34.6, -34.1, -12.1, -6.5, -4.1, -4.0, 2.6, 21.9, 34.8]

```

### Pass-bar criteria — cell 1357: **FAIL**

- 1. net>=+0.10R TRAIN & VAL, VAL t>=2: FAIL
- 2. both TRAIN halves > 0: FAIL
- 3. ex-top-5% > 0 both splits: FAIL
- 4. beats D1 & D3 by >=0.10R on VAL: PASS
- 5. >=3 trades/week VAL: PASS
- VAL D1 mean -0.933 R, VAL D3 mean -0.891 R

## Cell 1358

No-trade reasons (share of kept population): no_failure 20.1%, r_too_small 11.2%

NBBO coverage on real trades: 100.0%

| split    |    n |   tpw |   meanR_gross |   meanR |    sd |   t_iid |   t_clust |   WR |    ex5 |   cap5 |   share_stop |   share_target |   share_eod |   green |
|:---------|-----:|------:|--------------:|--------:|------:|--------:|----------:|-----:|-------:|-------:|-------------:|---------------:|------------:|--------:|
| TRAIN    | 2903 | 54.77 |         0.057 |  -0.299 | 1.469 |  -10.97 |     -2.14 | 35.8 | -0.418 | -0.299 |        0.633 |          0.338 |       0.029 |    0.15 |
| VAL      | 1767 | 76.83 |        -0.007 |  -0.434 | 1.463 |  -12.48 |     -8.84 | 32.8 | -0.56  | -0.434 |        0.658 |          0.323 |       0.019 |    0    |
| TRAIN-H1 | 1712 | 32.3  |         0.081 |  -0.269 | 1.456 |   -7.66 |     -1.15 | 36.9 | -0.386 | -0.269 |        0.62  |          0.341 |       0.039 |    0.06 |
| TRAIN-H2 | 1191 | 22.47 |         0.023 |  -0.342 | 1.487 |   -7.93 |     -6.19 | 34.1 | -0.463 | -0.342 |        0.651 |          0.333 |       0.016 |    0.09 |


Placebo means:
| split   | placebo                       |    n |   meanR |
|:--------|:------------------------------|-----:|--------:|
| TRAIN   | D1 (time-shuffle, same sym)   | 1797 |  -0.641 |
| TRAIN   | D3 (symbol-shuffle, same day) | 1501 |  -0.617 |
| VAL     | D1 (time-shuffle, same sym)   | 1076 |  -0.756 |
| VAL     | D3 (symbol-shuffle, same day) |  992 |  -0.668 |

VAL week-by-week P&L at R=$100:
wk
2025-12-27/2026-01-02    -773.0
2026-01-03/2026-01-09   -4849.0
2026-01-10/2026-01-16     -70.0
2026-01-17/2026-01-23   -2649.0
2026-01-24/2026-01-30   -1382.0
2026-01-31/2026-02-06   -3663.0
2026-02-07/2026-02-13   -4115.0
2026-02-14/2026-02-20   -2778.0
2026-02-21/2026-02-27   -3430.0
2026-02-28/2026-03-06   -5182.0
2026-03-07/2026-03-13   -1909.0
2026-03-14/2026-03-20   -4642.0
2026-03-21/2026-03-27   -2449.0
2026-03-28/2026-04-03   -9159.0
2026-04-04/2026-04-10    -619.0
2026-04-11/2026-04-17   -5325.0
2026-04-18/2026-04-24   -1528.0
2026-04-25/2026-05-01   -2693.0
2026-05-02/2026-05-08   -5191.0
2026-05-09/2026-05-15   -5714.0
2026-05-16/2026-05-22   -5962.0
2026-05-23/2026-05-29   -2686.0

Cadence bar (VAL):
```
CADENCE BAR  (HOD-S-1358, VAL, live config: N/A slots, N/A, R = $N/A)
C1 gap       median N/A wk  P90 N/A wk        [fail]   gaps: []
C2 bleed     P90 N/A R     cycles net>0 0% [fail]
C3 reds      P10 -56.75 R  min -91.59 R  MDD 767.67 R   under-water max 22 wk   [fail]
C4 green     0%  null 50%                   [fail]
C5 fills/wk  80.32                             [pass]
C6 tail      C6 not audited
C7 power     cycles 0   bootstrap P90-gap 75% UB N/A wk    [fail]
diagnostics  ex-top-5% -766.97 R   capped -767.67 R   top-5 share 0.1%   weekly P&L histogram: [-91.6, -59.6, -57.1, -53.3, -51.9, -51.8, -48.5, -46.4, -41.2, -36.6, -34.3, -27.8, -26.9, -26.9, -26.5, -24.5, -19.1, -15.3, -13.8, -7.7, -6.2, -0.7]

```

### Pass-bar criteria — cell 1358: **FAIL**

- 1. net>=+0.10R TRAIN & VAL, VAL t>=2: FAIL
- 2. both TRAIN halves > 0: FAIL
- 3. ex-top-5% > 0 both splits: FAIL
- 4. beats D1 & D3 by >=0.10R on VAL: PASS
- 5. >=3 trades/week VAL: PASS
- VAL D1 mean -0.756 R, VAL D3 mean -0.668 R

## Overall: cell 1357 FAIL, cell 1358 FAIL

Runtime: 0.7 min.
## Verdict (main session, 2026-09-22, COMPLETE data after the backfill) — cells 1,357–1,358 REFUTED
Bar coverage 100 % on trades (3.93 M bars appended by `backfill_bars_sip.py`; the pre-backfill run is kept as
`FAILED_BREAK_SHORT_REPORT_void.md`). Population 12,135; walked 2,903 TRAIN / 1,767 VAL after the shortable
(43.7 % excluded) and SSR (0.4 %) proxies.

| cell | split | n | gross R | net R | t_clust | WR | ex-top-5 % | halves |
|---|---|---|---|---|---|---|---|---|
| 1,357 hold to 15:55 | TRAIN | 2,903 | +0.022 | −0.330 | −1.65 | 21.2 % | −0.801 | −0.354 / −0.297 |
| 1,357 | VAL | 1,767 | −0.055 | −0.478 | −5.21 | 17.1 % | −0.956 | |
| 1,358 cover at +2 R | TRAIN | 2,903 | +0.057 | −0.299 | −2.14 | 35.8 % | −0.418 | −0.269 / −0.342 |
| 1,358 | VAL | 1,767 | −0.007 | −0.434 | −8.84 | 32.8 % | −0.560 | |

**Gross is zero; the entire loss is cost.** Net − gross is −0.35 to −0.43 R per trade, because the PREREG's own
minimum R (≥ 0.3 % of price) is the same order as the round-trip spread on these names: measured NBBO half-spread
both legs + 2 bp slippage ≈ 0.3–0.4 % of price ≈ 1 R. Criterion 4 passes (the rule beats D1 −0.93 / D3 −0.89 on
VAL) but that only says it picks the least-bad shorts inside a population that drifts up.
**Adequacy:** VAL SE ≈ 0.09 R, so a gross edge ≥ +0.18 R is excluded; the observed gross is 0.00 ± 0.09.
Refuted — not re-opened by an R-floor variant, because a zero gross edge stays zero when the sample is trimmed.

**The transferable law (third book to die of it):** on this universe a rule whose R is ≤ ~0.5 % of price cannot
clear the spread. In-play ORB (R = 0.38 % of price), index ORB (R = 0.1–0.2 %), this short book (R ≥ 0.3 %) all
died the same way. The live HOD long book runs R ≈ 2 % of price (cost ≈ 0.15 R) and the production ORB book
R ≈ 5 % — those are the only two structures on this programme where cost is not the binding constraint.
Programme count 1,358.
