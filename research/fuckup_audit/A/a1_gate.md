# A1 — the re-gate under the corrected contract

## contract (c) corrected

G1 (TRAIN mean>0, t>=2, >=5 tpw): **0 of 52 pass**
old score4 gate (TRAIN >= +10R/week): 0 of 52 pass

TRAIN cells with a POSITIVE mean net R (9 of 52), ranked by t:

                   key exit  corr_n  corr_tpw  corr_meanR  corr_se  corr_t  corr_WR  corr_wkR  corr_green  corr_worst
        F1 {"P": 0.12} hold     833      15.7       0.266   0.1700    1.56     18.7      4.18        0.43       -22.9
        F2 {"P": 0.12} hold     834      15.7       0.427   0.2984    1.43     14.4      6.72        0.47       -23.6
                 F6 {} hold    1242      23.4       0.051   0.0380    1.34     44.4      1.20        0.57       -12.1
                 F6 {}   2r    1342      25.3       0.028   0.0259    1.06     46.5      0.70        0.57       -11.3
F5 {"K": 8, "X": 0.06} hold    1602      30.2       0.028   0.0494    0.57     38.5      0.85        0.49       -21.0
          F8 {"N": 30} hold    1124      21.2       0.012   0.0235    0.49     48.6      0.25        0.57        -9.6
F5 {"K": 8, "X": 0.04} hold    1862      35.1       0.011   0.0482    0.22     35.3      0.38        0.42       -19.1
          F8 {"N": 30}   2r    1148      21.7       0.005   0.0215    0.22     49.0      0.10        0.57        -9.6
          F8 {"N": 15} hold    1244      23.5       0.004   0.0281    0.13     44.9      0.09        0.47       -13.9

**Nothing clears G1, so G2 is not evaluated and TEST is not read for selection.**

## contract (d) corrected + live liquidity gate

G1 (TRAIN mean>0, t>=2, >=5 tpw): **1 of 52 pass**
old score4 gate (TRAIN >= +10R/week): 0 of 52 pass

TRAIN cells with a POSITIVE mean net R (10 of 52), ranked by t:

                   key exit  gate_n  gate_tpw  gate_meanR  gate_se  gate_t  gate_WR  gate_wkR  gate_green  gate_worst
        F1 {"P": 0.12} hold     509       9.6       0.426   0.2107    2.02     23.0      4.09        0.53       -15.1
                 F6 {} hold    1194      22.5       0.069   0.0353    1.95     45.6      1.55        0.64       -11.7
F5 {"K": 8, "X": 0.06} hold    1500      28.3       0.109   0.0598    1.82     40.4      3.09        0.55       -21.3
        F2 {"P": 0.12} hold     363       6.8       0.656   0.4883    1.34     17.6      4.49        0.38       -13.5
                 F6 {}   2r    1280      24.2       0.034   0.0252    1.34     46.6      0.82        0.58       -11.7
        F2 {"P": 0.08} hold     495       9.3       0.334   0.3516    0.95     17.6      3.12        0.38       -15.6
        F2 {"P": 0.05} hold     557      10.5       0.248   0.3137    0.79     19.7      2.60        0.38       -19.3
          F8 {"N": 30} hold    1125      21.2       0.012   0.0236    0.52     48.6      0.26        0.57        -9.6
          F8 {"N": 15} hold    1245      23.5       0.010   0.0285    0.34     45.0      0.23        0.49       -14.3
          F8 {"N": 30}   2r    1149      21.7       0.005   0.0215    0.25     49.0      0.12        0.57        -9.6

G1 survivors on VAL:

           key exit split  gross_n  gross_tpw  gross_meanR  gross_se  gross_t  gross_WR  gross_wkR  gross_green  gross_worst  s4_n  s4_tpw  s4_meanR  s4_se  s4_t  s4_WR  s4_wkR  s4_green  s4_worst  corr_n  corr_tpw  corr_meanR  corr_se  corr_t  corr_WR  corr_wkR  corr_green  corr_worst  corrp_n  corrp_tpw  corrp_meanR  corrp_se  corrp_t  corrp_WR  corrp_wkR  corrp_green  corrp_worst  mix_stop  mix_target  mix_eod  gate_n  gate_tpw  gate_meanR  gate_se  gate_t  gate_WR  gate_wkR  gate_green  gate_worst  gatep_n  gatep_tpw  gatep_meanR  gatep_se  gatep_t  gatep_WR  gatep_wkR  gatep_green  gatep_worst
F1 {"P": 0.12} hold   VAL      390       17.7       -0.192    0.1414    -1.36      15.6       -3.4         0.36        -22.7   390    17.7     -0.71  0.144 -4.93   14.9  -12.58      0.14     -34.7     390      17.7      -0.279    0.142   -1.96     15.4     -4.94        0.32       -24.7      390       17.7       -0.279     0.142    -1.96      15.4      -4.94         0.32        -24.7     0.838         0.0    0.162     207       9.4      -0.255   0.1698    -1.5     17.4      -2.4        0.18       -13.3      207        9.4       -0.255    0.1698     -1.5      17.4       -2.4         0.18        -13.3

## closest miss per family on TRAIN, contract (c), with its minimum detectable effect (MDE = 2.8 x SE)

fam                     key exit  corr_n  corr_tpw  corr_meanR  corr_se  MDE_R  corr_t  corr_WR  corr_wkR  corr_green
 F1          F1 {"P": 0.12} hold     833      15.7       0.266   0.1700  0.476    1.56     18.7      4.18        0.43
 F2          F2 {"P": 0.12} hold     834      15.7       0.427   0.2984  0.836    1.43     14.4      6.72        0.47
 F6                   F6 {} hold    1242      23.4       0.051   0.0380  0.106    1.34     44.4      1.20        0.57
 F5  F5 {"K": 8, "X": 0.06} hold    1602      30.2       0.028   0.0494  0.138    0.57     38.5      0.85        0.49
 F8            F8 {"N": 30} hold    1124      21.2       0.012   0.0235  0.066    0.49     48.6      0.25        0.57
 F3 F3 {"M": 15, "P": 0.05} hold    1794      33.8      -0.070   0.0946  0.265   -0.74     16.9     -2.36        0.38
 F9          F9 {"G": 0.05} hold    1141      21.5      -0.041   0.0430  0.120   -0.95     37.8     -0.88        0.42
 F7                   F7 {} hold    1343      25.3      -0.177   0.0506  0.142   -3.50     32.0     -4.49        0.19
 F4                   F4 {} hold    1982      37.4      -0.260   0.0736  0.206   -3.54     16.9     -9.73        0.28
