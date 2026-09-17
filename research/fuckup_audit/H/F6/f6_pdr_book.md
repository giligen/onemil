# F6-PDR — declared book, TEST read once (2026-09-17)

F6-PDR — the owner's book, declared 2026-09-17 15:10 UTC BEFORE this run, TEST read ONCE here.

Definition (all causal at the signal bar): F6 red-to-green (opens below prev close; signal = first 1-min bar whose high
reaches prev close x 1.003); prev_day_range_pct >= 8 (ORB's shipped day-2-continuation veto, NOT derived from these
losers — H/F6/REPORT.md found it the only rule that replicated on VAL); range_so_far_pct >= 5 on bars before the signal
(baked into pop_c's F6 rows); price >= 5; R >= 1% of price; entry <= 14:01; fill = next bar's open <= level x 1.006;
stop = lowest low before entry (touch); exits: HOLD to 15:55 (primary) and 2R-on-close (secondary); contract (c) as
B/score5.py; run_book(12, 4). Cells read here: 2 exits x {PDR>=8, all} = 4, TEST once for both PDR cells.
Outputs: H/F6/f6_pdr_book.md, f6_pdr_trades_<exit>.csv.

## exit hold | PDR>=8
split    n  tpw  meanR  gross    t   WR  stopP  wkR  green  worst    ex5   cap3   mdd
TRAIN 1122 21.2  0.078  0.106 1.81 44.7   29.2 1.65   0.57  -10.7 -0.161 -0.005 -24.3
  VAL  519 23.6  0.251  0.281 3.28 48.4   29.5 5.91   0.82   -9.3 -0.036  0.114 -17.5
 TEST  376 26.9  0.104  0.138 0.74 41.5   33.5 2.79   0.43  -13.3 -0.223 -0.061 -26.9

monthly R: 2025-01 +3.4 (83) | 2025-02 +10.1 (79) | 2025-03 +11.1 (93) | 2025-04 -0.6 (105) | 2025-05 +1.5 (97) | 2025-06 -5.3 (92) | 2025-07 +25.9 (94) | 2025-08 +12.7 (89) | 2025-09 +7.1 (86) | 2025-10 +4.8 (113) | 2025-11 +16.0 (91) | 2025-12 +0.7 (100) | 2026-01 +19.1 (102) | 2026-02 +3.1 (94) | 2026-03 +35.5 (105) | 2026-04 +50.6 (108) | 2026-05 +21.8 (110) | 2026-06 -28.9 (125) | 2026-07 -4.7 (121) | 2026-08 +69.3 (111) | 2026-09 +3.4 (19)
months green 17/21

## exit hold | all
split    n  tpw  meanR  gross     t   WR  stopP   wkR  green  worst    ex5   cap3   mdd
TRAIN 1245 23.5  0.051  0.084  1.33 44.3   26.4  1.20   0.58  -12.1 -0.168 -0.021 -32.6
  VAL  536 24.4  0.170  0.204  2.56 47.9   28.4  4.14   0.73   -8.1 -0.076  0.074 -14.7
 TEST  379 27.1 -0.089 -0.050 -1.44 37.2   33.8 -2.42   0.29  -14.4 -0.286 -0.125 -37.9

## exit 2r | PDR>=8
split    n  tpw  meanR  gross    t   WR  stopP  wkR  green  worst    ex5  cap3   mdd
TRAIN 1217 23.0  0.058  0.088 2.00 47.8   26.3 1.33   0.58   -9.5 -0.044 0.058 -20.0
  VAL  593 27.0  0.110  0.142 2.56 48.7   26.6 2.97   0.68  -10.5  0.011 0.110 -20.2
 TEST  420 30.0  0.003  0.038 0.07 44.3   29.5 0.10   0.43  -12.5 -0.100 0.003 -12.9

monthly R: 2025-01 -12.5 (88) | 2025-02 +4.7 (83) | 2025-03 -3.6 (100) | 2025-04 +8.3 (114) | 2025-05 +8.5 (104) | 2025-06 -11.0 (100) | 2025-07 +23.9 (104) | 2025-08 +11.0 (97) | 2025-09 +10.7 (89) | 2025-10 +2.2 (124) | 2025-11 +13.1 (102) | 2025-12 +15.3 (112) | 2026-01 +8.8 (115) | 2026-02 +9.2 (106) | 2026-03 +20.2 (118) | 2026-04 +23.9 (127) | 2026-05 +3.3 (127) | 2026-06 -18.8 (134) | 2026-07 +4.8 (138) | 2026-08 +13.0 (126) | 2026-09 +2.5 (22)
months green 17/21

## exit 2r | all
split    n  tpw  meanR  gross     t   WR  stopP   wkR  green  worst    ex5   cap3   mdd
TRAIN 1350 25.5  0.030  0.064  1.14 46.5   24.4  0.76   0.57  -11.3 -0.074  0.030 -23.8
  VAL  607 27.6  0.076  0.111  1.85 48.8   25.7  2.08   0.68   -9.2 -0.026  0.076 -14.5
 TEST  421 30.1 -0.046 -0.005 -0.91 40.6   30.2 -1.37   0.43  -13.6 -0.156 -0.046 -17.3
