## 1. Signals, fills and the SSR exclusion

                   key  signals  filled fill_rate  scoreable  ssr_signals ssr_pct  scoreable_after_ssr
                 S1 {}     6210    5460     87.9%       4939          415    6.7%                 4620
          S2 {"N": 15}    45826   43031     93.9%      39630         5824   12.7%                34670
          S2 {"N": 30}    40908   39120     95.6%      35627         5131   12.5%                31227
                 S3 {}    20961   17179     82.0%      14278         3111   14.8%                12343
S4 {"K": 5, "X": 0.04}    68044   58193     85.5%      45006         8577   12.6%                39121
                 S5 {}     2822    2822    100.0%       1637         1156   41.0%                  979

`filled` = the next-open (S5: the 09:35 open) fill passed the no-chase cap and left a stop above the entry; `scoreable` adds entry >= $10, entry_m <= 14:01 and r_pct >= 1.0.

## 2. What the cost contract costs, in R (booked TRAIN trades, UA, hold / S5 1030)

                   key exit    n   gross     net   cost  med_spread_bps  med_r_pct  stopP
                 S1 {} hold  903  0.0155 -0.0546 0.0701            46.0       3.14   31.5
          S2 {"N": 15} hold 1196  0.0119 -0.0568 0.0687            45.3       3.11   28.6
          S2 {"N": 30} hold 1110 -0.0364 -0.0886 0.0522            39.2       3.65   21.3
                 S3 {} hold 1393 -0.0527 -0.1574 0.1047            45.3       3.64   44.4
S4 {"K": 5, "X": 0.04} hold 1394  0.0143 -0.0958 0.1101            46.0       1.89   47.1
                 S5 {} 1030  519  0.0556 -0.0639 0.1195            51.6       2.23   39.7

## 3. The ETF leak in `asset_class == stock`

- 38 of 2,610 distinct G symbols carry a fund/ETF token in the Alpaca asset name (1.5%); 40 symbols have no name row (they cannot be audited and are LEFT IN).
- they are **0.5% of the scoreable rows** (767 of 142,561); by family: S1 {} 0.7%, S2 {"N": 15} 0.6%, S2 {"N": 30} 0.7%, S3 {} 0.5%, S4 {"K": 5, "X": 0.04} 0.3%, S5 {} 0.4%
- regex used: `\bETF\b|\bETN\b|\bFund\b|\bIndex\b|\bSPDR\b|iShares|Vanguard|Invesco|WisdomTree|ProShares|Direxion|Global X|First Trust|VanEck|Amplify|Roundhill|YieldMax|Defiance|Simplify|Xtrackers|GraniteShares|Innovator|Pacer|ALPS |Franklin .*ETF|Schwab .*ETF|\bUnit Trust\b|\bClosed[- ]End\b|\bPortfolio\b`

### The 24 primary cells re-scored with those names removed (POST-HOC, +24 cells)

                   key uni exit  TRAIN_n  TRAIN_tpw  TRAIN_meanR  TRAIN_grossR  TRAIN_t  VAL_n  VAL_meanR  VAL_t
                 S5 {}  UB 1030      134        2.6       0.0444        0.1076     0.22     97     0.2589   1.43
                 S1 {}  UB hold      478        9.4       0.0261        0.0570     0.65    256     0.0333   0.53
                 S1 {}  UB   2R      484        9.5       0.0231        0.0550     0.61    258     0.0180   0.33
                 S3 {}  UB hold      660       12.9      -0.0036        0.0189    -0.14    363     0.0016   0.04
                 S3 {}  UB   2R      664       13.0      -0.0069        0.0158    -0.28    368    -0.0033  -0.09
          S2 {"N": 15}  UB hold      968       19.0      -0.0214        0.0052    -0.89    464    -0.0928  -2.76
          S2 {"N": 15}  UB   2R      976       19.1      -0.0273       -0.0004    -1.19    467    -0.0984  -3.07
          S2 {"N": 30}  UB hold      960       18.8      -0.0296       -0.0072    -1.43    448    -0.1080  -3.49
          S2 {"N": 30}  UB   2R      967       19.0      -0.0303       -0.0077    -1.51    452    -0.1127  -3.81
                 S1 {}  UA   2R      930       18.2      -0.0343        0.0389    -1.06    449    -0.1542  -3.42
                 S1 {}  UA hold      901       17.7      -0.0513        0.0184    -1.47    434    -0.1725  -3.67
          S2 {"N": 15}  UA hold     1197       23.5      -0.0535        0.0149    -1.84    542    -0.1614  -3.89
          S2 {"N": 15}  UA   2R     1264       24.8      -0.0606        0.0106    -2.28    568    -0.1686  -4.56
                 S5 {}  UA 1030      518       10.2      -0.0644        0.0551    -0.90    240     0.0235   0.24
                 S5 {}  UB  eod      134        2.6      -0.0650        0.0024    -0.30     97     0.2177   0.94
S4 {"K": 5, "X": 0.04}  UB hold     1439       28.2      -0.0659        0.0316    -1.56    705    -0.2207  -4.07
                 S5 {}  UA  eod      518       10.2      -0.0778        0.0512    -0.85    240    -0.0634  -0.51
          S2 {"N": 30}  UA hold     1108       21.7      -0.0823       -0.0309    -3.25    482    -0.1213  -3.30
          S2 {"N": 30}  UA   2R     1141       22.4      -0.0869       -0.0342    -3.69    490    -0.1279  -3.67
S4 {"K": 5, "X": 0.04}  UA hold     1394       27.3      -0.0965        0.0135    -2.50    641    -0.1892  -3.40
S4 {"K": 5, "X": 0.04}  UB   2R     1578       30.9      -0.1305       -0.0248    -4.14    769    -0.2344  -5.43
S4 {"K": 5, "X": 0.04}  UA   2R     1512       29.6      -0.1343       -0.0150    -4.38    697    -0.1550  -3.41
                 S3 {}  UA   2R     1574       30.9      -0.1533       -0.0457    -5.49    777    -0.1338  -3.29
                 S3 {}  UA hold     1389       27.2      -0.1560       -0.0513    -3.99    665    -0.0664  -1.00

**G1 without the fund names: 0 of 24**

## 4. S5 attention minus control, same machinery, in R

uni exit  atte_TRAIN  atte_VAL  cont_TRAIN  cont_VAL  diff_TRAIN  diff_VAL
 UA 1030     -0.0639    0.0184     -0.2373   -0.1852      0.1734    0.2036
 UA  eod     -0.0772   -0.0682     -0.2506   -0.3593      0.1734    0.2911
 UB 1030      0.0444    0.2450      0.0204   -0.2803      0.0240    0.5253
 UB  eod     -0.0650    0.2042      0.1357   -0.3874     -0.2007    0.5916
