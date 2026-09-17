# R10 - the B -> E convention ladder

R10 - the B -> E ladder in ONE tape walk. Four configs, each differing from the previous by exactly one
convention, so the step that moves the book is visible:

  L0  = B's rule set, on E's data plumbing (panel prior day, cache-first bars)   [level 1.003, floor/runlo,
        stop incl, price on entry, day-open >= 5, 14:01 on the FILL bar, first break THEN floor]
  L1  = L0 without B's day-level `universe open >= 5` prefilter
  L2  = L1 with the engine's cut: 14:00 on the SIGNAL bar (detect returns None past it)
  L3  = L2 with the engine's SCAN: bars failing the floor are skipped, not fatal   == implementation E


pre-book candidates: {'L0_B_rules': 7606, 'L1_no_dayopen5': 7743, 'L2_engine_cut': 7755, 'L3_engine_scan_E': 13541}

## exit hold
                    n              meanR                     t              totR              green             worst            
split            TEST TRAIN  VAL    TEST   TRAIN     VAL  TEST TRAIN   VAL  TEST TRAIN    VAL  TEST TRAIN   VAL  TEST TRAIN   VAL
variant                                                                                                                          
L0_B_rules        366  1114  531 -0.0421  0.0623  0.2065 -0.64  1.50  2.51 -15.4  69.4  109.7  0.43  0.57  0.77 -11.3 -10.0 -12.9
L1_no_dayopen5    366  1125  534 -0.0461  0.0668  0.2158 -0.70  1.62  2.64 -16.9  75.1  115.3  0.43  0.58  0.77 -11.1 -10.0 -12.9
L2_engine_cut     366  1126  535 -0.0461  0.0667  0.2154 -0.70  1.62  2.64 -16.9  75.1  115.2  0.43  0.58  0.77 -11.1 -10.0 -12.9
L3_engine_scan_E  427  1500  664 -0.1017 -0.0268 -0.0117 -1.58 -0.64 -0.17 -43.4 -40.2   -7.8  0.43  0.51  0.59 -18.1 -20.8 -26.7

## exit r2
                    n              meanR                     t              totR             green             worst            
split            TEST TRAIN  VAL    TEST   TRAIN     VAL  TEST TRAIN   VAL  TEST TRAIN   VAL  TEST TRAIN   VAL  TEST TRAIN   VAL
variant                                                                                                                         
L0_B_rules        408  1206  600 -0.0221  0.0725  0.0762 -0.43  2.45  1.79  -9.0  87.5  45.7  0.36  0.60  0.77  -7.3  -9.7 -11.5
L1_no_dayopen5    410  1218  606 -0.0242  0.0762  0.0886 -0.47  2.59  2.08  -9.9  92.8  53.7  0.36  0.62  0.82  -7.3  -9.4 -11.5
L2_engine_cut     410  1219  607 -0.0242  0.0761  0.0884 -0.47  2.59  2.08  -9.9  92.8  53.7  0.36  0.62  0.82  -7.3  -9.4 -11.5
L3_engine_scan_E  491  1713  761 -0.0979 -0.0389 -0.0606 -1.99 -1.43 -1.46 -48.1 -66.6 -46.1  0.43  0.43  0.55 -16.3 -14.8 -25.5

## exit partial
                    n              meanR                     t              totR             green             worst            
split            TEST TRAIN  VAL    TEST   TRAIN     VAL  TEST TRAIN   VAL  TEST TRAIN   VAL  TEST TRAIN   VAL  TEST TRAIN   VAL
variant                                                                                                                         
L0_B_rules        369  1128  541 -0.0165  0.0780  0.1146 -0.28  2.30  2.04  -6.1  88.0  62.0  0.43  0.57  0.73  -7.5  -9.7 -12.2
L1_no_dayopen5    369  1139  545 -0.0204  0.0815  0.1282 -0.35  2.41  2.29  -7.5  92.9  69.9  0.43  0.58  0.77  -7.5  -9.4 -12.2
L2_engine_cut     369  1140  546 -0.0204  0.0815  0.1279 -0.35  2.41  2.29  -7.5  92.9  69.9  0.43  0.58  0.77  -7.5  -9.4 -12.2
L3_engine_scan_E  434  1532  679 -0.0805 -0.0381 -0.0501 -1.44 -1.18 -0.97 -35.0 -58.4 -34.0  0.43  0.45  0.55 -17.6 -15.0 -24.4

