# 1 — the honest p-value: bootstrap null of t, not the normal table

H0 imposed by centering each split's trade R at zero; days resampled in 5-day circular blocks, 20,000 reps.
split    obs t   normal p  bootstrap p  null t 95%  null t 99%   ratio
TRAIN     2.92    0.00177      0.00190        1.68        2.36     1.1x
VAL       4.32    0.00001      0.00000        1.76        2.49     0.0x
TEST      3.83    0.00006      0.00010        1.82        2.55     1.6x
  ("ratio" = how many times larger the honest p is than the normal-table p. > 1 means the t-test
   overstates the evidence, because the R distribution is skewed and the days are dependent.)


# 2 — is "first four of the day" a signal, or a proxy for a tight stop?

## within-day contrast: mean R of the day's first four MINUS mean R of the same day's others
   (day fixed effects — every market-wide regime, volatility and selection-count effect is differenced out)
split   days     diff      se       t
TRAIN    249   +0.130   0.048    2.68
VAL      102   +0.297   0.085    3.49
TEST      68   +0.405   0.100    4.07

## the same contrast INSIDE R-size strata (R as % of price) — does it survive the tight-stop control?
  TRAIN  1-2% +0.146 (t +0.8, n172) | 2-3% +0.250 (t +1.7, n168) | 3-4.5% +0.170 (t +1.8, n247) | 4.5-7% -0.027 (t -0.4, n316) | 7%+ -0.118 (t -1.2, n93)
  VAL    1-2% -0.054 (t -0.2, n88) | 2-3% +0.018 (t +0.1, n74) | 3-4.5% +0.278 (t +2.0, n122) | 4.5-7% +0.060 (t +0.5, n98) | 7%+ +0.235 (t +1.0, n26)
  TEST   1-2% +0.269 (t +0.8, n50) | 2-3% +0.618 (t +2.5, n42) | 3-4.5% +0.201 (t +1.2, n86) | 4.5-7% -0.099 (t -0.7, n75)

## population mean R by R-size, ALL candidates (the mechanical gradient the selection rides)
  TRAIN  (1.0, 2.0] +0.321 (n388) | (2.0, 3.0] +0.060 (n433) | (3.0, 4.5] +0.061 (n1423) | (4.5, 7.0] -0.036 (n7202) | (7.0, 1000000000.0] -0.016 (n3877)
  VAL    (1.0, 2.0] +0.494 (n218) | (2.0, 3.0] +0.302 (n260) | (3.0, 4.5] +0.169 (n878) | (4.5, 7.0] +0.036 (n3959) | (7.0, 1000000000.0] +0.053 (n1899)
  TEST   (1.0, 2.0] +0.465 (n186) | (2.0, 3.0] +0.443 (n254) | (3.0, 4.5] +0.199 (n651) | (4.5, 7.0] -0.062 (n2607) | (7.0, 1000000000.0] -0.049 (n1641)


# 3 — how much of the TRAIN -> VAL -> TEST rise is the changing wrapper mix?

split   wrapper share  wrapper R   other R  overall   TRAIN-mix counterfactual
TRAIN            9.3%      0.219     0.120    0.130                      0.130
VAL             24.8%      0.604     0.229    0.322                      0.264
TEST            38.6%      0.424     0.303    0.350                      0.315
  (the last column re-weights each split to TRAIN's wrapper share: what is left is the genuine change)


# 4 — cost stress in bps of PRICE (the book is concentrated in tight-R trades)

extra slippage charged on every non-target exit (stops and 15:55 closes), on top of the 40 bps already charged.
 extra bps              TRAIN                VAL               TEST
         0   +0.130R  +2.4/wk   +0.322R  +6.0/wk   +0.350R  +6.8/wk
        10   +0.110R  +2.1/wk   +0.302R  +5.6/wk   +0.332R  +6.5/wk
        20   +0.090R  +1.7/wk   +0.283R  +5.2/wk   +0.314R  +6.1/wk
        30   +0.070R  +1.3/wk   +0.264R  +4.9/wk   +0.296R  +5.8/wk
        50   +0.030R  +0.6/wk   +0.225R  +4.2/wk   +0.261R  +5.1/wk
        75   -0.020R  -0.4/wk   +0.176R  +3.3/wk   +0.216R  +4.2/wk
       100   -0.070R  -1.3/wk   +0.128R  +2.4/wk   +0.171R  +3.3/wk

median R as % of price in the book: TRAIN 4.00% VAL 3.60% TEST 3.79%
  -> 25 bps of extra stop slippage costs about 0.07R on the median trade.


# 5 — how thin is the margin? target->stop conversions that zero out a split

split      n  targets  hit rate  total R  R per conversion  conversions to zero  = pct-points of hit rate
TRAIN    996      294     29.5%    129.2              3.20                 40.3                      4.0%
VAL      408      156     38.2%    131.3              3.27                 40.2                      9.9%
TEST     272      103     37.9%     95.2              3.25                 29.3                     10.8%