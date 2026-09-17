# Stage E — STEP 0: verification and the availability audit

_generated 2026-09-17 00:30:00 — nothing below this line is a trading result._

## 1. Build verification

| item | value |
|---|---|
| days in `build_causal_state.json` | 410 |
| data rows in `candidates_causal.csv` | 583,769 |
| columns in header | 88 |

last builder log line: `00:20:09 353/353 2026-09-04 keys 389 rows+=1159 miss=5 total 506,195 | 21.0 min, 3.6 s/day, ETA 0 min`

## 2. Survivorship residual (keys with no usable tape)

`coverage_causal_missing.csv` rows: **1,510**  |  `bars_causal_index.csv` rows: 124,875

| index `src` of the missing key | keys |
|---|---:|
| `badsym` | 702 |
| `alpaca` | 661 |
| `none` | 76 |
| `not-in-index (served from bars_sip.db)` | 71 |

index keys with `n_bars == 0` (Alpaca served nothing): **1,067** of 124,875 = 0.85%

## 3. The scoreable population

signal rows read: **583,769**  |  scoreable (entry >= $5, 570 <= entry_m <= 841, r_pct >= 1%): next **314,288**, rest **346,177**

### fill `next` — rows per family x split
split                     TEST  TRAIN    VAL
key                                         
F13 {"K": 5, "X": 0.04}   3495   8418   4906
F6 {}                     5493  13621   7894
F8 {"N": 15}             18771  46905  26923
F8 {"N": 30}             15937  40370  23704
F8 {"N": 5}              20238  48668  28945

### fill `rest` — rows per family x split
split          TEST  TRAIN    VAL
key                              
F6 {}          7234  17423  10246
F8 {"N": 15}  21167  52943  30482
F8 {"N": 30}  17495  44931  26122
F8 {"N": 5}   24534  58874  34726

## 4. AVAILABILITY AUDIT (PLAN §1 standing rule) — fill `next`, exit hold, per-trade net R

Coverage of every column a Stage-E cell reads, on the scoreable population:

| column | non-null | coverage |
|---|---:|---:|
| `pm_dollar_vol` | 279,749 | 89.01% |
| `has_news` | 314,288 | 100.00% |
| `prev_day_range_pct` | 314,278 | 100.00% |
| `adv20` | 314,288 | 100.00% |
| `gap_pct` | 314,278 | 100.00% |
| `spread_cc_bps` | 314,288 | 100.00% |
| `range_so_far_pct` | 314,288 | 100.00% |

### `pm_dollar_vol` — missing on 10.99% of scoreable rows

**missing rate per split x time band**

split        TEST  TRAIN   VAL
band                          
09:30-09:35   5.0    8.7   7.1
09:35-10:00   5.8    8.9   6.9
10:00-11:00   9.6   13.3  11.1
11:00-13:00  13.5   18.0  15.4
13:00-14:01  17.1   19.4  13.8

**mean net R (hold), missing vs present, per split**

         mean                   size               
split    TEST   TRAIN     VAL   TEST   TRAIN    VAL
_m                                                 
0     -0.1137 -0.0405 -0.0401  58394  138260  83095
1     -0.1663 -0.0128 -0.0888   5540   19722   9277

**mean net R (hold), missing vs present, per time band (all splits)**

               mean            size       
_m                0       1       0      1
band                                      
09:30-09:35 -0.1020 -0.0959    5640    447
09:35-10:00 -0.0893 -0.1395  119777   9895
10:00-11:00 -0.0552 -0.0725  101952  13743
11:00-13:00 -0.0003 -0.0221   40510   7925
13:00-14:01  0.1128  0.2360   11870   2529

`has_news`: 100% coverage — no missingness table needed.

### `prev_day_range_pct` — missing on 0.00% of scoreable rows

**missing rate per split x time band**

split        TEST  TRAIN  VAL
band                         
09:30-09:35   0.0    0.0  0.0
09:35-10:00   0.0    0.0  0.0
10:00-11:00   0.0    0.0  0.0
11:00-13:00   0.0    0.0  0.0
13:00-14:01   0.0    0.0  0.0

**mean net R (hold), missing vs present, per split**

         mean                  size               
split    TEST   TRAIN    VAL   TEST   TRAIN    VAL
_m                                                
0     -0.1182 -0.0370 -0.045  63933  157974  92371
1     -1.1891  0.0337 -1.532      1       8      1

**mean net R (hold), missing vs present, per time band (all splits)**

               mean              size     
_m                0       1         0    1
band                                      
09:30-09:35 -0.1016     NaN    6087.0  NaN
09:35-10:00 -0.0931 -0.2441  129669.0  3.0
10:00-11:00 -0.0572 -0.5816  115693.0  2.0
11:00-13:00 -0.0038 -0.5088   48432.0  3.0
13:00-14:01  0.1344  0.4852   14397.0  2.0

`adv20`: 100% coverage — no missingness table needed.

### `gap_pct` — missing on 0.00% of scoreable rows

**missing rate per split x time band**

split        TEST  TRAIN  VAL
band                         
09:30-09:35   0.0    0.0  0.0
09:35-10:00   0.0    0.0  0.0
10:00-11:00   0.0    0.0  0.0
11:00-13:00   0.0    0.0  0.0
13:00-14:01   0.0    0.0  0.0

**mean net R (hold), missing vs present, per split**

         mean                  size               
split    TEST   TRAIN    VAL   TEST   TRAIN    VAL
_m                                                
0     -0.1182 -0.0370 -0.045  63933  157974  92371
1     -1.1891  0.0337 -1.532      1       8      1

**mean net R (hold), missing vs present, per time band (all splits)**

               mean              size     
_m                0       1         0    1
band                                      
09:30-09:35 -0.1016     NaN    6087.0  NaN
09:35-10:00 -0.0931 -0.2441  129669.0  3.0
10:00-11:00 -0.0572 -0.5816  115693.0  2.0
11:00-13:00 -0.0038 -0.5088   48432.0  3.0
13:00-14:01  0.1344  0.4852   14397.0  2.0

## 5. `pm_dollar_vol` provenance — the D1 question

A key whose bars came from the Stage-E parquet store was fetched 04:00-15:59, so a missing/zero premarket value is a real "no premarket trades". A key served from `bars_sip.db` carries whatever window that store holds; if that store is RTH-only, "pm missing" would mean "this key was already in the >=5%-range fetch" — a D1-style availability leak.

**pm missing rate by provenance**

                mean                   size              
split           TEST   TRAIN     VAL   TEST  TRAIN    VAL
pm_src_store                                             
0             0.1019  0.1355  0.1118  25259  59821  35442
1             0.0767  0.1183  0.0934  38675  98161  56930

**mean net R (hold) by provenance x pm-missing**

                           mean                   size              
split                      TEST   TRAIN     VAL   TEST  TRAIN    VAL
pm_src_store pm_missing                                             
0            0          -0.2072 -0.1889 -0.1434  22685  51715  31480
             1          -0.1443 -0.0730 -0.0589   2574   8106   3962
1            0          -0.0542  0.0483  0.0229  35709  86545  51615
             1          -0.1854  0.0292 -0.1111   2966  11616   5315

## 6. The bucket shares the 60 cells will restrict to

(`pm_dollar_vol` NaN is treated as BELOW the cut — the pre-registered rule needs the leg to be positively established, exactly as the live ORB gate does.)

split       TEST  TRAIN    VAL
bucket                        
combo       4.71   4.98   4.80
neither    74.55  72.76  73.59
news_only   8.07  11.30   9.58
pm_only    12.68  10.95  12.02

split       TEST   TRAIN    VAL
bucket                         
combo       3009    7874   4436
neither    47660  114955  67978
news_only   5161   17855   8853
pm_only     8104   17298  11105

**mean net R (hold) per bucket per split, ALL families pooled, per trade (not booked)**

             mean                   size               
split        TEST   TRAIN     VAL   TEST   TRAIN    VAL
bucket                                                 
combo     -0.1569 -0.1192 -0.0581   3009    7874   4436
neither   -0.1209 -0.0276 -0.0435  47660  114955  67978
news_only -0.0756  0.0083 -0.0496   5161   17855   8853
pm_only   -0.1151 -0.1090 -0.0453   8104   17298  11105
