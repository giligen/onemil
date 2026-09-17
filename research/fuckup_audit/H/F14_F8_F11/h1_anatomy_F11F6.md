# Stage H step 1 — loser anatomy, book `F11F6`, exit `hold`, TRAIN only

population (TRAIN, scoreable, floor on): **5,995** signals · booked 12/4: **1,239** trades
  ·  book stats: `{'n': 1239, 'tpw': 23.4, 'meanR': -0.0115, 'gross': 0.0216, 'se': 0.0332, 't': -0.35, 'mde': 0.093, 'WR': 43.5, 'stopP': 26.2, 'wkR': -0.27, 'wkSE': 0.96, 'green': 0.43, 'worst': -12.8, 'mdd': -50.4, 'ex5': -0.1926, 'ex1': -0.0791, 'cap3': -0.0557}`

## 0. Availability audit (before any feature is used)

| feature | split | coverage | 09:30-09:45 | 09:45-10:00 | 10:00-11:00 | 11:00-13:00 | 13:00-14:01 |
|---|---|---:|---:|---:|---:|---:|---:|
| pm_dollar_vol | TRAIN | 0.3813 | 0.311 | 0.242 | 0.372 | 0.453 | 0.462 |
| pm_dollar_vol | VAL | 0.4035 | 0.248 | 0.261 | 0.382 | 0.484 | 0.574 |
| pm_dollar_vol | TEST | 0.3893 | 0.237 | 0.303 | 0.390 | 0.483 | 0.507 |
| news_pre | TRAIN | 0.9203 | 0.710 | 0.855 | 0.964 | 0.960 | 0.961 |
| news_pre | VAL | 0.9313 | 0.814 | 0.854 | 0.971 | 0.954 | 0.970 |
| news_pre | TEST | 0.9174 | 0.758 | 0.851 | 0.973 | 0.968 | 0.952 |
| spy_at_entry | TRAIN | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| spy_at_entry | VAL | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| spy_at_entry | TEST | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| prev_day_range_pct | TRAIN | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| prev_day_range_pct | VAL | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| prev_day_range_pct | TEST | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| adv20 | TRAIN | 0.9780 | 0.961 | 0.978 | 0.983 | 0.977 | 0.982 |
| adv20 | VAL | 0.9930 | 0.989 | 0.992 | 0.992 | 0.996 | 0.995 |
| adv20 | TEST | 0.9919 | 0.992 | 0.985 | 0.995 | 0.993 | 0.991 |
| spread_cc_bps | TRAIN | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| spread_cc_bps | VAL | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| spread_cc_bps | TEST | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Outcome correlation of the missingness (population mean net R, TRAIN / VAL / TEST):

| feature | missing | present |
|---|---|---|
| pm_dollar_vol | +0.086 (n 3709) / +0.154 (n 1962) / +0.010 (n 1509) | -0.188 / -0.122 / -0.172 |
| news_pre | -0.251 (n 478) / -0.251 (n 226) / -0.360 (n 204) | +0.001 / +0.064 / -0.033 |

## 1.1 Concentration

- booked trading days: **249**, green 120 / red 129 (48% green); total **-14.3 R**
- sum of the losing days: **-271.4 R**; sum of the winning days **+257.1 R**
- worst 5% of days (12) carry **-60.8 R** = -22% of all day-losses
- worst 10% of days (24) carry **-103.1 R** = -38% of all day-losses
- best 5% of days carry **+91.0 R**; the book without BOTH tails = **-44.5 R**
- weeks: 53 booked of 53; green 43%; worst week -12.8 R; best week +20.5 R

### The 20 worst booked days, with market context

             n      R  stops  eods  tgts                       syms  spy_gap  spy_co  iwm_co  qqq_co  spy_vol20  spy_prev_ret regime
day                                                                                                                                 
2025-05-30   7 -6.075      5     2     0     KG,LICN,NNE,NTAP,QVCGP   -0.190   0.078  -0.034  -0.064     17.339         0.395      A
2025-11-21  10 -6.022      6     4     0     AGX,ANAB,CIFR,CRDU,IDN    0.386   0.608   2.453   0.443     14.134        -1.524     C2
2025-05-01   7 -5.546      3     4     0   ALTS,FTRE,MEDP,MORN,SSII    1.051  -0.339   0.184  -0.358     54.053         0.040      B
2025-01-06  10 -5.441      6     4     0  BCAX,ELTX,EWCZ,FBIOP,LXEH    0.730  -0.153  -0.549   0.099     14.885         1.250     C2
2025-03-10   7 -5.355      4     3     0     CLBK,EBS,GDS,HIMS,PNFP   -1.446  -1.235  -1.312  -2.217     15.884         0.560     C1
2025-06-13   7 -5.278      4     3     0   ARAI,ARQT,CURI,NAMS,ROBN   -0.870  -0.251  -0.453  -0.136     12.086         0.397      A
2025-01-27   8 -5.063      4     4     0    ATRA,BIOA,KGEI,MBX,ORKA   -2.165   0.767  -0.093   0.628     14.122        -0.292      A
2025-10-06   8 -4.786      4     4     0    CHOW,CMPR,CRCD,FBL,KGEI    0.360  -0.001  -0.371  -0.122      5.669        -0.001      A
2025-09-04   7 -4.594      3     4     0     CHPT,GCTK,NA,NAOV,OPAD    0.106   0.729   1.008   0.787      9.538         0.542      A
2025-07-07   7 -4.223      3     4     0     ARCX,FLWS,GDXU,IONZ,NA   -0.317  -0.430  -0.830  -0.269     10.271         0.788      A
2025-10-30   7 -4.212      4     3     0       AGH,ALSN,CWK,EL,LUMN   -0.508  -0.595  -0.171  -0.968     14.011         0.048      A
2025-07-28   6 -4.205      2     4     0    ARAI,CYCC,HTO,IMCR,KZIA    0.060  -0.085  -0.533   0.070      6.565         0.422      A
2025-05-02   6 -4.110      2     4     0    KWR,NTGR,OILD,PCTY,SMCZ    1.121   0.359   1.084   0.471     54.065         0.709      B
2025-04-01   7 -4.108      3     4     0      AEYE,EVCM,MBX,NCT,PHH   -0.347   0.631   0.261   1.156     20.492         0.671     C1
2025-10-09   7 -3.682      3     4     0      BGMS,BTQ,FOFO,NCTY,QH    0.062  -0.352  -0.479  -0.128      6.209         0.596      A
2025-03-12   5 -3.477      3     2     0    DNTH,NFBK,UVIX,UVXY,ZIM    1.124  -0.587  -0.960  -0.467     17.593        -0.831     C1
2025-09-24   7 -3.441      4     3     0     AIR,ARBB,BINI,GDC,PLTS    0.196  -0.513  -0.943  -0.580      7.113        -0.544      A
2025-06-26   6 -3.439      3     3     0     HCHL,JBIO,LEU,PATK,YSG    0.308   0.473   1.212   0.528     10.302         0.056      A
2025-06-05   6 -3.432      3     3     0    DXF,ORKA,RAPP,SNDK,TNXP    0.285  -0.766  -0.081  -1.017     16.063        -0.027      A
2025-09-08   6 -3.391      2     4     0   ANTE,CCUP,INHD,NUVL,SNBR    0.213   0.032  -0.109   0.086      9.691        -0.290      A

context of ALL booked days for comparison: spy_co mean +0.043 · iwm_co mean +0.045

| bucket | n days | mean day R | mean spy_co | mean iwm_co |
|---|---:|---:|---:|---:|
| worst 20 | 20 | -4.49 | -0.081 | -0.036 |
| all red | 129 | -2.10 | -0.129 | -0.173 |
| all green | 120 | +2.14 | +0.227 | +0.279 |
| all | 249 | -0.06 | +0.043 | +0.045 |

Day-direction split of the BOOKED trades (the market number is the day close-to-open, NOT causal at entry — reported as a diagnostic, never as a filter):

| SPY close-open | n | mean net R |
|---|---:|---:|
| < -0.5% | 258 | -0.1311 |
| -0.5..0% | 348 | -0.1531 |
| 0..+0.5% | 334 | +0.0520 |
| > +0.5% | 299 | +0.1854 |

## 1.2 Trade anatomy — winners vs losers (booked TRAIN) and the whole TRAIN population

| feature | booked losers (mean) | booked winners | pop losers | pop winners |
|---|---:|---:|---:|---:|
| next_entry_m | 608 | 613.4 | 652.2 | 670.3 |
| minutes_since_open | 36.35 | 41.65 | 80.65 | 98.44 |
| price | 30.95 | 26.58 | 39.77 | 35.12 |
| spread_cc_bps | 40.26 | 39.67 | 36.31 | 35.37 |
| spread_over_r | 0.08548 | 0.07313 | 0.0614 | 0.05705 |
| next_r_pct | 6.316 | 6.793 | 6.984 | 7.264 |
| gap_pct | -3.503 | -3.748 | -3.751 | -3.846 |
| prev_day_range_pct | 14.7 | 15.22 | 11.49 | 12.99 |
| range_so_far_pct | 7.265 | 7.824 | 7.337 | 7.764 |
| dist_open_pct | 3.724 | 4.375 | 3.979 | 4.188 |
| rv_adv | 0.1927 | 0.1988 | 0.3284 | 0.3652 |
| adv20 | 5.863e+06 | 5.182e+06 | 5.681e+06 | 6.462e+06 |
| consol_bars | 26.77 | 29.73 | 63.92 | 78.5 |
| n_touches | 0.5743 | 0.7347 | 1.17 | 1.341 |
| consol_vol_ratio | 1.016 | 1.047 | 1.369 | 1.376 |
| vwap_dist_pct | 2.883 | 3.321 | 3.672 | 3.769 |
| cum_dollar_vol | 2.603e+07 | 1.521e+07 | 5.883e+07 | 8.159e+07 |
| sig_close_pos | 0.8438 | 0.8449 | 0.8606 | 0.8441 |
| sig_body_pct | 0.8934 | 0.8067 | 0.6588 | 0.5895 |
| sig_dollar | 1.023e+06 | 6.42e+05 | 1.155e+06 | 1.209e+06 |
| pm_dollar_vol | 7.363e+06 | 2.077e+06 | 9.632e+06 | 1.212e+07 |
| sig_seq_day | 4.476 | 3.742 | 129.3 | 40.86 |
| spy_at_entry | -0.01398 | -0.01648 | 1.114 | 0.4309 |
| iwm_at_entry | -0.02061 | -0.01866 | 1.608 | 0.6701 |
| spy_gap | -0.02343 | 0.01603 | -0.9646 | -0.531 |
| spy_co | -0.02123 | 0.1186 | 0.6716 | 0.8656 |
| iwm_co | -0.05653 | 0.1771 | 0.7733 | 1.081 |
| spy_vol20 | 16.15 | 16.43 | 19.55 | 18.49 |
| spy_vs_sma20 | 0.5654 | 0.396 | -1.725 | -1.244 |

## 1.3 Path anatomy

- exit mix (booked TRAIN): eod 914, stop 325
- stops: **325** (26.2%); mean minutes entry->stop **97** (median 66)
- **wick stops** (the touch stop fired but the bar closed back above the stop level): 178 = 14.4% of booked trades, 54.8% of stops
- of the stopped trades, share that had **+0.5R** on the table first: 30.8%; **+1R**: 13.8%; **+1.5R**: 8.3%
- winners: mean MAE 1.77% of price, median 1.28%; losers mean MAE 4.69%
- MFE of the booked book: mean 0.71 R; losers 0.40 R; winners 1.11 R

| minutes held | n | mean net R | stop share |
|---|---:|---:|---:|
| 0-15 | 62 | -1.2240 | 100% |
| 15-60 | 93 | -1.1077 | 100% |
| 60-150 | 105 | -0.8443 | 79% |
| 150+ | 979 | +0.2587 | 9% |

| MFE bucket (R) | n | mean net R |
|---|---:|---:|
| < 0.5 | 674 | -0.4797 |
| 0.5-1 | 260 | -0.0753 |
| 1-2 | 169 | +0.5087 |
| 2+ | 136 | +1.7840 |

## 1.4 Era consistency inside TRAIN (H1 = Jan-Jun 2025, H2 = Jul-Dec 2025)

Every (feature x bucket) cell is in `h1_buckets_F11F6.csv` (135 cells). A bucket is a candidate veto only if its POPULATION mean net R is negative in BOTH halves.

**37 of 135 cells are negative in both halves**; 37 of them also have >= 150 population rows and >= 20 booked trades:

           feature                    bucket  pop_n   pop_R  pop_H1_n  pop_H1_R  pop_H2_n  pop_H2_R  bk_n    bk_R  bk_H1_R  bk_H2_R  bk_share  era_neg
          news_pre                       nan    478 -0.2507       262   -0.2436       216   -0.2592   245 -0.3126  -0.2838  -0.3480     0.198     True
     pm_dollar_vol        (1601852.955, inf]    572 -0.2379       210   -0.2339       362   -0.2403    94 -0.3080  -0.3496  -0.2743     0.076     True
            iwm_co               [-0.5, 0.0)    972 -0.2116       390   -0.1896       582   -0.2264   293 -0.1225   0.0025  -0.2279     0.236     True
     pm_dollar_vol   (31851.885, 213474.015]    571 -0.2028       337   -0.2101       234   -0.1923   103 -0.2372  -0.2129  -0.2584     0.083     True
               dow                         0   1772 -0.1684      1413   -0.1992       359   -0.0468   247 -0.1548  -0.1116  -0.1997     0.199     True
     pm_dollar_vol (213474.015, 1601852.955]    571 -0.1659       300   -0.1817       271   -0.1483    91 -0.0864  -0.0627  -0.1085     0.073     True
     pm_dollar_vol         (-inf, 31851.885]    572 -0.1471       390   -0.1703       182   -0.0973   134 -0.1535  -0.0834  -0.2642     0.108     True
prev_day_range_pct                [0.0, 5.0)   1130 -0.1363       688   -0.1837       442   -0.0625   284 -0.1895  -0.1875  -0.1919     0.229     True
     sig_close_pos               [0.25, 0.5)    204 -0.0901       109   -0.0223        95   -0.1678    46 -0.0825   0.0958  -0.2459     0.037     True
            spy_co               [-0.5, 0.0)   1230 -0.0886       548   -0.0445       682   -0.1241   348 -0.1531  -0.1039  -0.1862     0.281     True
            spy_co     [-1000000000.0, -0.5)    916 -0.0879       523   -0.0797       393   -0.0988   258 -0.1311  -0.1301  -0.1323     0.208     True
            iwm_co     [-1000000000.0, -0.5)   1063 -0.0810       543   -0.0524       520   -0.1109   346 -0.1414  -0.0966  -0.1823     0.279     True
            rv_adv            (0.113, 0.232]   1466 -0.0796       870   -0.1268       596   -0.0108   271 -0.0469  -0.0633  -0.0317     0.219     True
      spy_at_entry       [0.5, 1000000000.0)   1845 -0.0796      1523   -0.0843       322   -0.0576    48  0.1431   0.1482   0.1133     0.039     True
             price      [50.0, 1000000000.0)   1075 -0.0759       583   -0.0957       492   -0.0525   150 -0.1333  -0.1756  -0.1088     0.121     True
     vwap_dist_pct                [0.0, 1.0)    153 -0.0668        78   -0.0529        75   -0.0813   106 -0.0947  -0.0341  -0.1508     0.086     True
      iwm_at_entry       [0.5, 1000000000.0)   2751 -0.0548      1738   -0.0864      1013   -0.0005   130 -0.0294  -0.0522  -0.0003     0.105     True
    cum_dollar_vol (924731.898, 4887938.715]   1499 -0.0538       888   -0.0679       611   -0.0332   285  0.0047   0.0458  -0.0307     0.230     True
        sig_dollar      (18329.15, 87816.08]   1499 -0.0533       865   -0.0880       634   -0.0058   325 -0.0687  -0.0670  -0.0705     0.262     True
     spread_over_r             (0.0685, inf]   1499 -0.0518       809   -0.0272       690   -0.0807   572 -0.0277   0.0762  -0.1339     0.462     True
             adv20   (1111252.9, 3669710.45]   1465 -0.0513       862   -0.0750       603   -0.0175   250 -0.0010   0.1023  -0.0964     0.202     True
      sig_body_pct              (0.824, inf]   1499 -0.0511       933   -0.0750       566   -0.0118   426 -0.0670  -0.0769  -0.0583     0.344     True
      iwm_at_entry               [-0.5, 0.0)   1013 -0.0464       484   -0.0071       529   -0.0824   437 -0.0956  -0.0329  -0.1570     0.353     True
      spy_vs_sma20                [0.0, 2.0)   2186 -0.0460       681   -0.0410      1505   -0.0483   667 -0.0601  -0.0056  -0.0851     0.538     True
      sig_body_pct            (0.101, 0.397]   1499 -0.0385       784   -0.0512       715   -0.0245   201 -0.0228  -0.0132  -0.0308     0.162     True
  consol_vol_ratio              (1.518, inf]   1499 -0.0372       898   -0.0607       601   -0.0020   236 -0.0923  -0.2300   0.0259     0.190     True
         n_touches                       nan   3046 -0.0369      1741   -0.0556      1305   -0.0120   800 -0.0496   0.0155  -0.1061     0.646     True
     dist_open_pct             (1.831, 3.52]   1499 -0.0360       868   -0.0442       631   -0.0247   254  0.0627   0.2613  -0.1423     0.205     True
  consol_vol_ratio            (0.401, 0.815]   1499 -0.0303       793   -0.0497       706   -0.0084   290  0.0447   0.0978  -0.0063     0.234     True
       asset_class                     stock   4478 -0.0292      2662   -0.0427      1816   -0.0094   935 -0.0264   0.0298  -0.0827     0.755     True
  range_so_far_pct                [5.0, 8.0)   4393 -0.0292      2464   -0.0440      1929   -0.0103   932 -0.0156   0.0133  -0.0422     0.752     True
             adv20   (402398.425, 1111252.9]   1466 -0.0276       865   -0.0396       601   -0.0105   317 -0.0296  -0.0088  -0.0476     0.256     True
        next_r_pct            (5.475, 6.405]   1499 -0.0270       829   -0.0449       670   -0.0049   258 -0.0237   0.0089  -0.0548     0.208     True
     sig_close_pos               [0.5, 0.75)    679 -0.0212       382   -0.0292       297   -0.0109   118 -0.0237  -0.0194  -0.0276     0.095     True
            regime                         A   3296 -0.0126       847   -0.0218      2449   -0.0094   865 -0.0052   0.0659  -0.0372     0.698     True
prev_day_range_pct               [8.0, 15.0)   2108 -0.0102      1154   -0.0058       954   -0.0157   335  0.0438   0.1376  -0.0270     0.270     True
               dow                         4   1219 -0.0102       514   -0.0178       705   -0.0046   244 -0.0208  -0.0490   0.0079     0.197     True
