# Stage H step 1 — loser anatomy, book `F14`, exit `hold`, TRAIN only

population (TRAIN, scoreable, floor on): **1,492** signals · booked 12/4: **768** trades
  ·  book stats: `{'n': 768, 'tpw': 14.5, 'meanR': 0.0564, 'gross': 0.0836, 'se': 0.0428, 't': 1.32, 'mde': 0.12, 'WR': 45.2, 'stopP': 21.2, 'wkR': 0.82, 'wkSE': 0.83, 'green': 0.45, 'worst': -8.1, 'mdd': -31.3, 'ex5': -0.1189, 'ex1': -0.0129, 'cap3': 0.0125}`

## 0. Availability audit (before any feature is used)

| feature | split | coverage | 09:30-09:45 | 09:45-10:00 | 10:00-11:00 | 11:00-13:00 | 13:00-14:01 |
|---|---|---:|---:|---:|---:|---:|---:|
| pm_dollar_vol | TRAIN | 0.4665 | nan | nan | 0.427 | 0.449 | 0.495 |
| pm_dollar_vol | VAL | 0.4547 | nan | nan | 0.439 | 0.439 | 0.485 |
| pm_dollar_vol | TEST | 0.4619 | nan | nan | 0.333 | 0.409 | 0.586 |
| news_pre | TRAIN | 0.7768 | nan | nan | 0.878 | 0.806 | 0.721 |
| news_pre | VAL | 0.8040 | nan | nan | 0.902 | 0.831 | 0.723 |
| news_pre | TEST | 0.8501 | nan | nan | 0.905 | 0.862 | 0.814 |
| spy_at_entry | TRAIN | 1.0000 | nan | nan | 1.000 | 1.000 | 1.000 |
| spy_at_entry | VAL | 1.0000 | nan | nan | 1.000 | 1.000 | 1.000 |
| spy_at_entry | TEST | 1.0000 | nan | nan | 1.000 | 1.000 | 1.000 |
| prev_day_range_pct | TRAIN | 0.9940 | nan | nan | 0.992 | 0.990 | 0.998 |
| prev_day_range_pct | VAL | 0.9984 | nan | nan | 1.000 | 1.000 | 0.995 |
| prev_day_range_pct | TEST | 0.9951 | nan | nan | 0.976 | 0.996 | 1.000 |
| adv20 | TRAIN | 0.9725 | nan | nan | 0.962 | 0.975 | 0.971 |
| adv20 | VAL | 0.9918 | nan | nan | 1.000 | 1.000 | 0.976 |
| adv20 | TEST | 0.9877 | nan | nan | 0.976 | 0.987 | 0.993 |
| spread_cc_bps | TRAIN | 1.0000 | nan | nan | 1.000 | 1.000 | 1.000 |
| spread_cc_bps | VAL | 1.0000 | nan | nan | 1.000 | 1.000 | 1.000 |
| spread_cc_bps | TEST | 1.0000 | nan | nan | 1.000 | 1.000 | 1.000 |

Outcome correlation of the missingness (population mean net R, TRAIN / VAL / TEST):

| feature | missing | present |
|---|---|---|
| pm_dollar_vol | +0.380 (n 796) / +0.116 (n 331) / +0.022 (n 219) | -0.096 / -0.111 / -0.136 |
| news_pre | -0.010 (n 333) / -0.003 (n 119) / -0.338 (n 61) | +0.206 / +0.016 / -0.000 |

## 1.1 Concentration

- booked trading days: **231**, green 110 / red 121 (48% green); total **+43.3 R**
- sum of the losing days: **-168.6 R**; sum of the winning days **+212.0 R**
- worst 5% of days (11) carry **-41.5 R** = -25% of all day-losses
- worst 10% of days (23) carry **-73.4 R** = -44% of all day-losses
- best 5% of days carry **+88.3 R**; the book without BOTH tails = **-3.4 R**
- weeks: 53 booked of 53; green 45%; worst week -8.1 R; best week +27.4 R

### The 20 worst booked days, with market context

            n      R  stops  eods  tgts                      syms  spy_gap  spy_co  iwm_co  qqq_co  spy_vol20  spy_prev_ret regime
day                                                                                                                               
2025-04-14  5 -4.947      4     1     0   DXF,MSTZ,UVIX,UVXY,VIXY    1.893  -0.906  -0.682  -1.503     52.228         1.784      B
2025-10-03  7 -4.822      5     2     0   JFB,MSTU,MSTX,NEBX,NUKK    0.115  -0.116   0.196  -0.549      5.872         0.115      A
2025-02-27  5 -4.546      4     1     0  ALLT,DOCN,LQDA,RCAT,UBRL    0.389  -1.977  -1.507  -3.447     10.332         0.050     C1
2025-05-21  5 -3.868      4     1     0   NVDU,PCT,QMCO,RYET,XNET   -0.744  -0.948  -1.655  -0.626     15.142        -0.336      A
2025-01-02  4 -3.612      3     1     0        CLSK,CREV,HIT,PRTH    0.565  -0.806  -0.803  -0.791     14.282        -0.364     C2
2025-04-10  8 -3.516      4     4     0   GEV,PESI,PLTU,PTIR,RAPP   -2.998  -1.426  -1.523  -1.627     50.224        10.502      B
2025-06-03  4 -3.367      2     2     0       ALAR,CURI,NVCT,RDTL   -0.062   0.633   1.245   0.714     16.785         0.563      A
2025-02-05  3 -3.295      3     0     0              INKT,NNE,WGS   -0.189   0.596   0.588   0.964     13.705         0.671      A
2025-12-30  4 -3.276      1     3     0      AMCI,DJTWW,LUNR,RKLX   -0.059  -0.063  -0.776  -0.066      8.326        -0.356      A
2025-01-07  5 -3.229      1     4     0  AAOI,CADL,CRNC,IRBT,JOBY    0.346  -1.471  -1.268  -1.981     15.091         0.576      A
2025-01-14  5 -3.001      2     3     0   IONQ,IRBT,MSW,PTLE,SVIX    0.511  -0.371   0.246  -0.719     15.905         0.155     C2
2025-04-07  6 -2.905      3     3     0  EXOD,HAYW,KOLD,LFCR,PSFE   -3.184   3.105   3.000   3.678     32.066        -5.854      B
2025-10-10  6 -2.868      5     1     0  FEAM,NBIL,NBIS,NBTX,NEBX    0.145  -2.843  -3.239  -3.582      5.864        -0.290      A
2025-06-27  3 -2.801      2     1     0             HCHL,NTLA,SDM    0.165   0.331  -0.264   0.152     10.183         0.782      A
2025-03-25  4 -2.788      2     2     0         DAO,QUBT,RADX,RDW    0.213   0.028  -0.474   0.421     21.246         1.791     C1
2025-08-11  5 -2.781      3     2     0   ACB,ETON,MSTU,MSTX,SMST    0.044  -0.242  -0.226  -0.320     10.360         0.780      A
2025-11-10  6 -2.708      3     3     0    ARMP,BW,FORD,FTRE,MIND    0.934   0.621  -0.400   0.696     11.993         0.098      A
2025-12-05  5 -2.698      1     4     0  CCUP,CRCA,MSTZ,RGNT,SMST    0.158   0.032  -0.284   0.176     14.374         0.073      A
2025-09-19  4 -2.642      1     3     0        ALAB,BMNR,SQFT,TGS    0.011   0.207  -1.055   0.338      8.748         0.467      A
2025-07-11  5 -2.460      2     3     0  NNNN,OKLO,REPL,SONN,TRNR   -0.492   0.141  -0.659   0.163     10.181         0.282      A

context of ALL booked days for comparison: spy_co mean +0.028 · iwm_co mean +0.044

| bucket | n days | mean day R | mean spy_co | mean iwm_co |
|---|---:|---:|---:|---:|
| worst 20 | 20 | -3.31 | -0.274 | -0.477 |
| all red | 121 | -1.39 | -0.088 | -0.127 |
| all green | 110 | +1.93 | +0.156 | +0.231 |
| all | 231 | +0.19 | +0.028 | +0.044 |

Day-direction split of the BOOKED trades (the market number is the day close-to-open, NOT causal at entry — reported as a diagnostic, never as a filter):

| SPY close-open | n | mean net R |
|---|---:|---:|
| < -0.5% | 180 | -0.0130 |
| -0.5..0% | 187 | +0.0019 |
| 0..+0.5% | 220 | +0.0662 |
| > +0.5% | 181 | +0.1698 |

## 1.2 Trade anatomy — winners vs losers (booked TRAIN) and the whole TRAIN population

| feature | booked losers (mean) | booked winners | pop losers | pop winners |
|---|---:|---:|---:|---:|
| next_entry_m | 722.3 | 726.6 | 746.4 | 756.4 |
| minutes_since_open | 150.8 | 155.1 | 175 | 185 |
| price | 27.32 | 24 | 34.24 | 31.35 |
| spread_cc_bps | 29.18 | 28.88 | 29.78 | 29.75 |
| spread_over_r | 0.06959 | 0.07134 | 0.06886 | 0.06737 |
| next_r_pct | 5.134 | 4.907 | 5.219 | 5.185 |
| gap_pct | 6.387 | 0.3434 | 6.199 | -0.007863 |
| prev_day_range_pct | 13.67 | 11.92 | 12.66 | 12.22 |
| range_so_far_pct | 7.542 | 7.062 | 7.515 | 7.109 |
| dist_open_pct | 2.405 | 2.067 | 2.285 | 2.189 |
| rv_adv | 0.9955 | 0.6397 | 0.8461 | 0.6646 |
| adv20 | 7.462e+06 | 8.256e+06 | 8.434e+06 | 7.909e+06 |
| consol_bars | 52.37 | 57.85 | 67.34 | 79.66 |
| n_touches | 3.862 | 4.112 | 4.461 | 4.461 |
| consol_vol_ratio | 3.525 | 3.079 | 2.882 | 3.413 |
| vwap_dist_pct | 1.976 | 1.953 | 2.007 | 2.12 |
| cum_dollar_vol | 8.965e+07 | 8.036e+07 | 1.31e+08 | 1.441e+08 |
| sig_close_pos | 0.652 | 0.6384 | 0.661 | 0.6835 |
| sig_body_pct | 0.3999 | 0.3589 | 0.3724 | 0.4976 |
| sig_dollar | 8.587e+05 | 9.706e+05 | 9.734e+05 | 1.487e+06 |
| pm_dollar_vol | 6.914e+06 | 1.586e+07 | 8.969e+06 | 2.453e+07 |
| sig_seq_day | 1.8 | 1.813 | 8.991 | 17.79 |
| spy_at_entry | -0.03913 | -0.001308 | 0.05512 | 0.606 |
| iwm_at_entry | -0.06217 | 0.03685 | 0.1306 | 0.6889 |
| spy_gap | 0.05484 | 0.01336 | -0.2167 | -0.2051 |
| spy_co | -0.07825 | 0.1475 | -0.0699 | 1.478 |
| iwm_co | -0.1154 | 0.2264 | 0.04625 | 1.455 |
| spy_vol20 | 16.13 | 16.72 | 17.68 | 19.56 |
| spy_vs_sma20 | 0.4434 | 0.1302 | -0.6227 | -1.974 |

## 1.3 Path anatomy

- exit mix (booked TRAIN): eod 605, stop 163
- stops: **163** (21.2%); mean minutes entry->stop **100** (median 95)
- **wick stops** (the touch stop fired but the bar closed back above the stop level): 163 = 21.2% of booked trades, 100.0% of stops
- of the stopped trades, share that had **+0.5R** on the table first: 25.2%; **+1R**: 8.6%; **+1.5R**: 3.7%
- winners: mean MAE 1.26% of price, median 0.97%; losers mean MAE 3.78%
- MFE of the booked book: mean 0.72 R; losers 0.37 R; winners 1.15 R

| minutes held | n | mean net R | stop share |
|---|---:|---:|---:|
| 0-15 | 21 | -1.1571 | 100% |
| 15-60 | 33 | -1.0764 | 100% |
| 60-150 | 168 | -0.3120 | 42% |
| 150+ | 546 | +0.2849 | 7% |

| MFE bucket (R) | n | mean net R |
|---|---:|---:|
| < 0.5 | 383 | -0.5071 |
| 0.5-1 | 165 | -0.0350 |
| 1-2 | 153 | +0.6090 |
| 2+ | 67 | +2.2408 |

## 1.4 Era consistency inside TRAIN (H1 = Jan-Jun 2025, H2 = Jul-Dec 2025)

Every (feature x bucket) cell is in `h1_buckets_F14.csv` (138 cells). A bucket is a candidate veto only if its POPULATION mean net R is negative in BOTH halves.

**16 of 138 cells are negative in both halves**; 9 of them also have >= 150 population rows and >= 20 booked trades:

      feature                    bucket  pop_n   pop_R  pop_H1_n  pop_H1_R  pop_H2_n  pop_H2_R  bk_n    bk_R  bk_H1_R  bk_H2_R  bk_share  era_neg
       iwm_co     [-1000000000.0, -0.5)    395 -0.1496       217   -0.1418       178   -0.1591   213 -0.1202  -0.1195  -0.1208     0.277     True
pm_dollar_vol (493205.797, 4121918.368]    174 -0.1185        80   -0.0854        94   -0.1466    81 -0.1495  -0.2868  -0.0502     0.105     True
pm_dollar_vol        (4121918.368, inf]    174 -0.1129        94   -0.0274        80   -0.2133    91 -0.2525  -0.1267  -0.3992     0.118     True
pm_dollar_vol         (-inf, 59885.014]    174 -0.1033       102   -0.1323        72   -0.0623    97 -0.1406  -0.1914  -0.0843     0.126     True
       spy_co     [-1000000000.0, -0.5)    392 -0.1018       242   -0.1195       150   -0.0731   180 -0.0130  -0.0679   0.0556     0.234     True
 spy_at_entry     [-1000000000.0, -0.5)    285 -0.0668       160   -0.0767       125   -0.0541   136  0.0487   0.0401   0.0580     0.177     True
pm_dollar_vol   (59885.014, 493205.797]    174 -0.0511        86   -0.0263        88   -0.0754    89 -0.1178  -0.1304  -0.1097     0.116     True
 spy_vs_sma20                [0.0, 2.0)    546 -0.0499       121   -0.1134       425   -0.0319   393 -0.0629  -0.1208  -0.0402     0.512     True
dist_open_pct             (-inf, 0.806]    374 -0.0356       204   -0.0432       170   -0.0264   194 -0.1132  -0.2119  -0.0204     0.253     True
