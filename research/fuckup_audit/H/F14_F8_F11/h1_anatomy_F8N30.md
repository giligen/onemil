# Stage H step 1 — loser anatomy, book `F8N30`, exit `hold`, TRAIN only

population (TRAIN, scoreable, floor on): **25,495** signals · booked 12/4: **1,124** trades
  ·  book stats: `{'n': 1124, 'tpw': 21.2, 'meanR': 0.0079, 'gross': 0.0318, 'se': 0.0235, 't': 0.34, 'mde': 0.066, 'WR': 48.4, 'stopP': 15.1, 'wkR': 0.17, 'wkSE': 0.62, 'green': 0.57, 'worst': -9.7, 'mdd': -40.5, 'ex5': -0.1045, 'ex1': -0.0289, 'cap3': 0.0006}`

## 0. Availability audit (before any feature is used)

| feature | split | coverage | 09:30-09:45 | 09:45-10:00 | 10:00-11:00 | 11:00-13:00 | 13:00-14:01 |
|---|---|---:|---:|---:|---:|---:|---:|
| pm_dollar_vol | TRAIN | 0.3538 | nan | nan | 0.347 | 0.365 | 0.361 |
| pm_dollar_vol | VAL | 0.3710 | nan | nan | 0.357 | 0.375 | 0.467 |
| pm_dollar_vol | TEST | 0.4149 | nan | nan | 0.412 | 0.415 | 0.441 |
| news_pre | TRAIN | 1.0000 | nan | nan | 1.000 | 1.000 | 1.000 |
| news_pre | VAL | 1.0000 | nan | nan | 1.000 | 1.000 | 1.000 |
| news_pre | TEST | 1.0000 | nan | nan | 1.000 | 1.000 | 1.000 |
| spy_at_entry | TRAIN | 1.0000 | nan | nan | 1.000 | 1.000 | 1.000 |
| spy_at_entry | VAL | 1.0000 | nan | nan | 1.000 | 1.000 | 1.000 |
| spy_at_entry | TEST | 1.0000 | nan | nan | 1.000 | 1.000 | 1.000 |
| prev_day_range_pct | TRAIN | 0.9907 | nan | nan | 0.991 | 0.990 | 0.993 |
| prev_day_range_pct | VAL | 0.9985 | nan | nan | 0.998 | 0.998 | 0.999 |
| prev_day_range_pct | TEST | 0.9984 | nan | nan | 0.998 | 0.998 | 1.000 |
| adv20 | TRAIN | 0.9690 | nan | nan | 0.971 | 0.966 | 0.963 |
| adv20 | VAL | 0.9918 | nan | nan | 0.993 | 0.989 | 0.990 |
| adv20 | TEST | 0.9919 | nan | nan | 0.993 | 0.989 | 0.990 |
| spread_cc_bps | TRAIN | 1.0000 | nan | nan | 1.000 | 1.000 | 1.000 |
| spread_cc_bps | VAL | 1.0000 | nan | nan | 1.000 | 1.000 | 1.000 |
| spread_cc_bps | TEST | 1.0000 | nan | nan | 1.000 | 1.000 | 1.000 |

Outcome correlation of the missingness (population mean net R, TRAIN / VAL / TEST):

| feature | missing | present |
|---|---|---|
| pm_dollar_vol | +0.137 (n 16474) / +0.111 (n 10312) / +0.049 (n 7115) | -0.182 / -0.154 / -0.181 |
| news_pre | +nan (n 0) / +nan (n 0) / +nan (n 0) | +0.024 / +0.012 / -0.046 |

## 1.1 Concentration

- booked trading days: **250**, green 126 / red 124 (50% green); total **+8.9 R**
- sum of the losing days: **-178.7 R**; sum of the winning days **+187.7 R**
- worst 5% of days (12) carry **-44.9 R** = -25% of all day-losses
- worst 10% of days (25) carry **-77.8 R** = -44% of all day-losses
- best 5% of days carry **+62.8 R**; the book without BOTH tails = **-9.0 R**
- weeks: 53 booked of 53; green 57%; worst week -9.7 R; best week +12.7 R

### The 20 worst booked days, with market context

             n      R  stops  eods  tgts                      syms  spy_gap  spy_co  iwm_co  qqq_co  spy_vol20  spy_prev_ret regime
day                                                                                                                                
2025-10-10  10 -5.242      6     4     0   AEVA,BITF,BKKT,BTDR,EDZ    0.145  -2.843  -3.239  -3.582      5.864        -0.290      A
2025-11-20   7 -5.079      5     2     0    ADUR,AGCC,BKSY,BTDR,CD    1.551  -3.029  -3.427  -4.243     13.363         0.386     C2
2025-10-27   7 -3.975      3     4     0  AMDL,BAIG,CDTX,COGT,CORD    0.809   0.368  -0.588   0.572     13.530         0.817      A
2025-11-28   6 -3.872      3     3     0  ANVS,BMNG,BMNZ,MAGH,SVRA    0.174   0.372   0.298   0.510     15.363         0.690      A
2025-02-26   6 -3.827      2     4     0    ACMR,AS,ASPI,BITX,CONL    0.284  -0.233  -0.176  -0.188     10.855        -0.497     C1
2025-10-22   7 -3.691      4     3     0  ABAT,AVAH,BKKT,CBIO,CREV    0.106  -0.625  -1.093  -0.873     13.309        -0.001      A
2025-12-17   8 -3.569      5     3     0  BITU,BITX,BMNU,BMNZ,CLPT    0.150  -1.249  -1.250  -2.063     11.541        -0.273      A
2025-03-18   5 -3.328      2     3     0   FNGD,IBTA,NVD,NVDL,UVIX   -0.414  -0.669  -0.314  -1.016     19.940         0.771     C1
2025-11-04   5 -3.248      2     3     0   ACDC,AEHR,BLOX,BTDR,THH   -1.058  -0.129  -0.244  -0.647     14.685         0.188      A
2025-02-14   6 -3.053      2     4     0    ALTS,CAE,HLF,INLF,INOD    0.034  -0.039  -0.572   0.400     11.162         1.056      A
2025-03-04   4 -3.004      3     1     0        FAZ,GDXD,JDST,TSDD   -0.695  -0.492   0.238   0.277     14.115        -1.752     C1
2025-01-31   4 -2.969      2     2     0        CLS,EOSE,PLTU,POET    0.407  -0.935  -0.992  -0.879     13.849         0.537      A
2025-03-14   6 -2.855      4     2     0   MSTU,MSTZ,NNE,SMST,TSLQ    0.851   1.205   1.323   1.252     18.139        -1.333     C1
2025-11-17   4 -2.827      2     2     0         ACVA,BW,CNTA,GPRE   -0.332  -0.602  -1.565  -0.435     13.155        -0.016      A
2025-09-29   5 -2.809      2     3     0  ARQT,AXTI,BMNU,FLNC,HIMZ    0.384  -0.102  -0.502  -0.063      7.654         0.573      A
2025-07-18   6 -2.752      2     4     0  BKSY,BOSC,CONL,CRCL,JOBY    0.201  -0.273  -1.314  -0.330      8.115         0.612      A
2025-08-01   6 -2.698      2     4     0  SMST,TSDD,TSLL,TSLQ,TSLT   -0.914  -0.731  -0.601  -0.886      6.758        -0.375      A
2025-05-09   7 -2.602      3     4     0  BMBL,COEP,COIN,CONI,CONL    0.251  -0.378  -0.362  -0.457     26.371         0.697      B
2025-01-14   5 -2.508      2     3     0    AG,BTDR,MSTZ,PTLO,SMST    0.511  -0.371   0.246  -0.719     15.905         0.155     C2
2025-06-27   5 -2.467      2     3     0   BYRN,DNA,HIMS,HIMZ,INMB    0.165   0.331  -0.264   0.152     10.183         0.782      A

context of ALL booked days for comparison: spy_co mean +0.039 · iwm_co mean +0.041

| bucket | n days | mean day R | mean spy_co | mean iwm_co |
|---|---:|---:|---:|---:|
| worst 20 | 20 | -3.32 | -0.521 | -0.720 |
| all red | 124 | -1.44 | -0.113 | -0.194 |
| all green | 126 | +1.49 | +0.189 | +0.273 |
| all | 250 | +0.04 | +0.039 | +0.041 |

Day-direction split of the BOOKED trades (the market number is the day close-to-open, NOT causal at entry — reported as a diagnostic, never as a filter):

| SPY close-open | n | mean net R |
|---|---:|---:|
| < -0.5% | 244 | -0.1079 |
| -0.5..0% | 303 | -0.0886 |
| 0..+0.5% | 309 | +0.0152 |
| > +0.5% | 268 | +0.2143 |

## 1.2 Trade anatomy — winners vs losers (booked TRAIN) and the whole TRAIN population

| feature | booked losers (mean) | booked winners | pop losers | pop winners |
|---|---:|---:|---:|---:|
| next_entry_m | 613.6 | 614.8 | 661.8 | 666 |
| minutes_since_open | 42.49 | 43.66 | 90.34 | 94.57 |
| price | 35.84 | 31.4 | 33.44 | 32.67 |
| spread_cc_bps | 42.4 | 41.45 | 33.92 | 33.66 |
| spread_over_r | 0.0659 | 0.06382 | 0.05473 | 0.05449 |
| next_r_pct | 7.233 | 7.326 | 7.085 | 6.967 |
| gap_pct | 1.282 | 0.8831 | 6.437 | 8.037 |
| prev_day_range_pct | 11.12 | 10.41 | 11.49 | 11.59 |
| range_so_far_pct | 8.028 | 8.103 | 7.982 | 7.797 |
| dist_open_pct | 5.888 | 6.27 | 5.143 | 5.063 |
| rv_adv | 1.044 | 0.4964 | 0.5827 | 0.5334 |
| adv20 | 8.451e+06 | 8.595e+06 | 5.74e+06 | 5.349e+06 |
| consol_bars | 9.236 | 8.996 | 37.16 | 39.68 |
| n_touches | 1.234 | 1.18 | 1.554 | 1.54 |
| consol_vol_ratio | 2.381 | 2.741 | 3.066 | 3.035 |
| vwap_dist_pct | 3.311 | 3.222 | 3.275 | 3.203 |
| cum_dollar_vol | 7.917e+07 | 6.443e+07 | 5.772e+07 | 5.481e+07 |
| sig_close_pos | 0.5846 | 0.6018 | 0.6502 | 0.6613 |
| sig_body_pct | 0.3176 | 0.2674 | 0.3752 | 0.3879 |
| sig_dollar | 2.467e+06 | 2.318e+06 | 1.305e+06 | 1.221e+06 |
| pm_dollar_vol | 1.625e+07 | 1.215e+07 | 7.921e+06 | 6.278e+06 |
| sig_seq_day | 10.34 | 11.36 | 95.64 | 94.07 |
| spy_at_entry | -0.004693 | -0.03282 | 0.2339 | 0.3191 |
| iwm_at_entry | -0.01992 | -0.04758 | 0.3849 | 0.4678 |
| spy_gap | 0.03564 | 0.04307 | -0.1257 | -0.1308 |
| spy_co | -0.1009 | 0.127 | 0.1283 | 0.7229 |
| iwm_co | -0.1691 | 0.1604 | 0.2094 | 0.8946 |
| spy_vol20 | 15.81 | 16.52 | 16.43 | 17.43 |
| spy_vs_sma20 | 0.4016 | 0.5335 | 0.03732 | -0.3657 |

## 1.3 Path anatomy

- exit mix (booked TRAIN): eod 954, stop 170
- stops: **170** (15.1%); mean minutes entry->stop **144** (median 120)
- **wick stops** (the touch stop fired but the bar closed back above the stop level): 79 = 7.0% of booked trades, 46.5% of stops
- of the stopped trades, share that had **+0.5R** on the table first: 13.5%; **+1R**: 5.9%; **+1.5R**: 1.8%
- winners: mean MAE 1.90% of price, median 1.41%; losers mean MAE 5.04%
- MFE of the booked book: mean 0.57 R; losers 0.30 R; winners 0.87 R

| minutes held | n | mean net R | stop share |
|---|---:|---:|---:|
| 0-15 | 4 | -1.2268 | 100% |
| 15-60 | 37 | -1.0650 | 100% |
| 60-150 | 72 | -0.8641 | 78% |
| 150+ | 1011 | +0.1142 | 7% |

| MFE bucket (R) | n | mean net R |
|---|---:|---:|
| < 0.5 | 630 | -0.3911 |
| 0.5-1 | 311 | +0.2222 |
| 1-2 | 140 | +0.6717 |
| 2+ | 43 | +2.1433 |

## 1.4 Era consistency inside TRAIN (H1 = Jan-Jun 2025, H2 = Jul-Dec 2025)

Every (feature x bucket) cell is in `h1_buckets_F8N30.csv` (137 cells). A bucket is a candidate veto only if its POPULATION mean net R is negative in BOTH halves.

**26 of 137 cells are negative in both halves**; 26 of them also have >= 150 population rows and >= 20 booked trades:

           feature                    bucket  pop_n   pop_R  pop_H1_n  pop_H1_R  pop_H2_n  pop_H2_R  bk_n    bk_R  bk_H1_R  bk_H2_R  bk_share  era_neg
     pm_dollar_vol        (1583584.293, inf]   2255 -0.2615       908   -0.2239      1347   -0.2868   145 -0.3159  -0.2022  -0.4163     0.129     True
     pm_dollar_vol (182751.271, 1583584.293]   2255 -0.1846       962   -0.1766      1293   -0.1905   121 -0.1259  -0.0370  -0.2363     0.108     True
            spy_co     [-1000000000.0, -0.5)   4341 -0.1833      2371   -0.1740      1970   -0.1944   244 -0.1079  -0.0175  -0.2127     0.217     True
     pm_dollar_vol    (24244.47, 182751.271]   2255 -0.1614      1128   -0.1607      1127   -0.1622    86 -0.1189  -0.0818  -0.1496     0.077     True
            iwm_co     [-1000000000.0, -0.5)   5474 -0.1547      2607   -0.1596      2867   -0.1503   330 -0.1132  -0.0568  -0.1590     0.294     True
            iwm_co               [-0.5, 0.0)   5141 -0.1415      2059   -0.1461      3082   -0.1384   255 -0.1043  -0.1613  -0.0552     0.227     True
     pm_dollar_vol          (-inf, 24244.47]   2256 -0.1217      1197   -0.1235      1059   -0.1197    35 -0.0745  -0.0839  -0.0634     0.031     True
  range_so_far_pct      [20.0, 1000000000.0)    440 -0.0597       212   -0.1012       228   -0.0211    20  0.2011  -0.0370   0.4920     0.018     True
            spy_co               [-0.5, 0.0)   6075 -0.0520      2316   -0.0767      3759   -0.0368   303 -0.0886  -0.2147  -0.0001     0.270     True
           spy_gap                [0.0, 0.3)   6883 -0.0380      2558   -0.0228      4325   -0.0470   379 -0.0597  -0.0122  -0.0885     0.337     True
prev_day_range_pct                [0.0, 5.0)   5053 -0.0324      2290   -0.0601      2763   -0.0095   271  0.0101   0.0490  -0.0313     0.241     True
           gap_pct               [3.0, 10.0)   3587 -0.0306      1558   -0.0268      2029   -0.0335   227 -0.0297   0.0133  -0.0694     0.202     True
      spy_vs_sma20       [2.0, 1000000000.0)   3822 -0.0279      2366   -0.0148      1456   -0.0490   185  0.0172  -0.0029   0.0542     0.165     True
      iwm_at_entry               [-0.5, 0.0)   5569 -0.0259      2691   -0.0292      2878   -0.0229   422 -0.0171  -0.0330  -0.0003     0.375     True
         spy_vol20            (-inf, 11.981]   8532 -0.0228      1278   -0.1042      7254   -0.0085   418  0.0017  -0.0408   0.0093     0.372     True
            rv_adv                       nan    791 -0.0214       617   -0.0261       174   -0.0047    27  0.0224   0.0982  -0.5839     0.024     True
             adv20                       nan    791 -0.0214       617   -0.0261       174   -0.0047    27  0.0224   0.0982  -0.5839     0.024     True
       sig_seq_day                [1.0, 2.0)    216 -0.0197       108   -0.0055       108   -0.0339   216 -0.0197  -0.0055  -0.0339     0.192     True
      spy_at_entry                [0.0, 0.5)  11459 -0.0176      4467   -0.0153      6992   -0.0190   511 -0.0320   0.0013  -0.0591     0.455     True
      spy_vs_sma20                [0.0, 2.0)  12012 -0.0168      3376   -0.0229      8636   -0.0144   616 -0.0310  -0.0650  -0.0154     0.548     True
         spy_vol20          (11.981, 15.031]   8520 -0.0147      3027   -0.0362      5493   -0.0028   346 -0.0176   0.0007  -0.0306     0.308     True
prev_day_range_pct                [5.0, 8.0)   6192 -0.0128      2911   -0.0039      3281   -0.0207   277 -0.0090   0.0266  -0.0354     0.246     True
            regime                         A  16963 -0.0113      4471   -0.0300     12492   -0.0046   791 -0.0012   0.0011  -0.0022     0.704     True
               dow                         4   5311 -0.0081      2348   -0.0167      2963   -0.0013   234 -0.0729  -0.0556  -0.0888     0.208     True
      sig_body_pct              (0.0, 0.272]   5166 -0.0080      2314   -0.0065      2852   -0.0093   220  0.0226   0.0506  -0.0060     0.196     True
       consol_bars              (13.0, 52.0]   6326 -0.0073      2775   -0.0133      3551   -0.0027    35  0.2114   0.5734  -0.0935     0.031     True
