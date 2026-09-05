# ORB entry-signal study — REPORT (auto-appended by run_study.py; protocol in DESIGN.md)

## Baseline (production stack, flags off)

P&L +6,085 | MDD -818 | red months 6/21 | worst -236 | picks 130 (fills 73, 0.301/day) | eras T +2,442 / O1 +1,135 / O2 +2,509

OOS: P&L +3,644 | MDD -818 | red months 6/15 | worst -236 | picks 98 (fills 56, 0.326/day) | eras T +2,442 / O1 +1,135 / O2 +2,509

Identity vs production book: **PASS** (130 vs 130 picks, $6,085 vs $6,085)


## C5 — range/ATR14 tier READ

L1 (candidate level, entered rows):

| tier | era | n | mean_pnl_pct | wr% |
|---|---|---|---|---|
| narrow | TRAIN | 620 | -0.504 | 23.1 |
| narrow | OOS1 | 565 | -0.07 | 30.8 |
| narrow | OOS2 | 1232 | 0.443 | 32.9 |
| normal | TRAIN | 498 | -0.177 | 20.7 |
| normal | OOS1 | 536 | -0.343 | 21.1 |
| normal | OOS2 | 1102 | 0.252 | 25.2 |
| wide | TRAIN | 704 | -0.026 | 18.0 |
| wide | OOS1 | 830 | -0.771 | 11.3 |
| wide | OOS2 | 1075 | -0.501 | 14.7 |
| unknown | TRAIN | 8 | -4.366 | 12.5 |
| unknown | OOS1 | 62 | 1.575 | 40.3 |
| unknown | OOS2 | 72 | 0.313 | 38.9 |

L2 (baseline book by tier):

| tier | era | picks | sized_pnl |
|---|---|---|---|
| narrow | TRAIN | 17 | 1560 |
| narrow | OOS1 | 24 | 918 |
| narrow | OOS2 | 39 | 2955 |
| normal | TRAIN | 11 | 1003 |
| normal | OOS1 | 8 | -109 |
| normal | OOS2 | 14 | -399 |
| wide | TRAIN | 3 | -121 |
| wide | OOS1 | 0 | 0 |
| wide | OOS2 | 0 | 0 |
| unknown | TRAIN | 1 | 0 |
| unknown | OOS1 | 5 | 326 |
| unknown | OOS2 | 8 | -48 |

Tiers negative in BOTH OOS eras: ['normal'] → veto test queued


## Singles


### C1 data: rvol_open5 coverage 97.1% of candidates; median 2.28; entered-row quintile read:

| rvol_q | n | mean_pnl_pct | wr% | rvol_range |
|---|---|---|---|---|
| q1 | 1421 | 0.245 | 29.6 | 0.00-1.03 |
| q2 | 1420 | -0.029 | 25.6 | 1.03-1.74 |
| q3 | 1421 | -0.225 | 22.3 | 1.74-2.93 |
| q4 | 1420 | -0.134 | 21.5 | 2.93-6.02 |
| q5 | 1421 | -0.45 | 13.2 | 6.02-7976.97 |


TRAIN grid for `ORB_EXP_RVOL_VETO` (base TRAIN P&L +2,442 | MDD -226 | red months 0/6 | worst +18 | picks 32 (fills 17, 0.256/day) | eras T +2,442 / O1 +1,135 / O2 +2,509):

| t | train_pnl | train_mdd | mdd_ok | picks |
|---|---|---|---|---|
| 0.5 | 1830 | -296 | True | 26 |
| 1.0 | 1722 | -221 | True | 24 |
| 1.5 | -436 | -310 | True | 19 |
| 2.0 | -180 | -191 | True | 16 |

chosen t = 0.5 (grid edge — reported, not extended)


### C1a rvol veto < 0.5 (TRAIN-fit)

env: `{'ORB_EXP_RVOL_VETO': '0.5'}`

L1 — rvol_open5 >= 0.5 (kept) vs < 0.5 (dropped):

| era | n_keep | n_drop | mean_keep | mean_drop | diff | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| ALL | 6917 | 387 | -0.175 | 0.761 | -0.936 | -1.883 | -0.142 |
| TRAIN | 1749 | 81 | -0.31 | 1.087 | -1.396 | -3.513 | 0.408 |
| OOS1 | 1909 | 84 | -0.392 | -0.206 | -0.186 | -1.336 | 0.908 |
| OOS2 | 3259 | 222 | 0.025 | 1.009 | -0.984 | -2.311 | 0.152 |

L2 OOS baseline: P&L +3,644 | MDD -818 | red months 6/15 | worst -236 | picks 98 (fills 56, 0.326/day) | eras T +2,442 / O1 +1,135 / O2 +2,509

L2 OOS variant:  P&L +2,285 | MDD -824 | red months 8/15 | worst -290 | picks 88 (fills 51, 0.292/day) | eras T +1,830 / O1 +831 / O2 +1,454

ALL-window variant: P&L +4,115 | MDD -824 | red months 8/21 | worst -290 | picks 114 (fills 64, 0.264/day) | eras T +1,830 / O1 +831 / O2 +1,454

checks: {'1_pnl_+5%': False, '2_mdd': True, '3_neg_months': False, '4_eras': False, '5_giants_kept>=8': True, '6_picks_not_halved': True, '7_L1_effect': False} | giants kept 8/10

**VERDICT: REJECT**


### C1b rank by rvol desc, quintile/composite tie-break

env: `{'ORB_EXP_RVOL_RANK': '1'}`

L1 — above vs below median rvol (2.24) — rank-form proxy:

| era | n_keep | n_drop | mean_keep | mean_drop | diff | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| ALL | 3753 | 3551 | -0.318 | 0.079 | -0.397 | -0.633 | -0.153 |
| TRAIN | 907 | 923 | -0.273 | -0.224 | -0.049 | -0.483 | 0.395 |
| OOS1 | 1179 | 814 | -0.685 | 0.052 | -0.737 | -1.202 | -0.259 |
| OOS2 | 1667 | 1814 | -0.082 | 0.245 | -0.327 | -0.706 | 0.04 |

L2 OOS baseline: P&L +3,644 | MDD -818 | red months 6/15 | worst -236 | picks 98 (fills 56, 0.326/day) | eras T +2,442 / O1 +1,135 / O2 +2,509

L2 OOS variant:  P&L +5,187 | MDD -972 | red months 5/15 | worst -400 | picks 83 (fills 45, 0.27/day) | eras T +807 / O1 +1,388 / O2 +3,800

ALL-window variant: P&L +5,995 | MDD -972 | red months 6/21 | worst -400 | picks 109 (fills 59, 0.251/day) | eras T +807 / O1 +1,388 / O2 +3,800

checks: {'1_pnl_+5%': True, '2_mdd': False, '3_neg_months': True, '4_eras': True, '5_giants_kept>=8': False, '6_picks_not_halved': True, '7_L1_effect': False} | giants kept 3/10

**VERDICT: REJECT**


### C2_green_pre 

env: `{'ORB_EXP_RCP_GATE': 'pre', 'ORB_EXP_RCP_FORM': 'green'}`

L1 — green: kept vs dropped:

| era | n_keep | n_drop | mean_keep | mean_drop | diff | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| ALL | 4440 | 2864 | -0.071 | -0.209 | 0.139 | -0.098 | 0.38 |
| TRAIN | 1101 | 729 | -0.228 | -0.278 | 0.05 | -0.42 | 0.498 |
| OOS1 | 1158 | 835 | -0.304 | -0.496 | 0.192 | -0.239 | 0.644 |
| OOS2 | 2181 | 1300 | 0.133 | 0.013 | 0.12 | -0.267 | 0.47 |

L2 OOS baseline: P&L +3,644 | MDD -818 | red months 6/15 | worst -236 | picks 98 (fills 56, 0.326/day) | eras T +2,442 / O1 +1,135 / O2 +2,509

L2 OOS variant:  P&L +7,183 | MDD -828 | red months 6/15 | worst -414 | picks 94 (fills 62, 0.308/day) | eras T +1,979 / O1 +2,041 / O2 +5,142

ALL-window variant: P&L +9,163 | MDD -828 | red months 7/21 | worst -414 | picks 116 (fills 76, 0.269/day) | eras T +1,979 / O1 +2,041 / O2 +5,142

checks: {'1_pnl_+5%': True, '2_mdd': True, '3_neg_months': True, '4_eras': True, '5_giants_kept>=8': True, '6_picks_not_halved': True, '7_L1_effect': False} | giants kept 8/10

**VERDICT: REJECT**


### C2_green_post 

env: `{'ORB_EXP_RCP_GATE': 'post', 'ORB_EXP_RCP_FORM': 'green'}`

L1 — green: kept vs dropped:

| era | n_keep | n_drop | mean_keep | mean_drop | diff | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| ALL | 4440 | 2864 | -0.071 | -0.209 | 0.139 | -0.098 | 0.38 |
| TRAIN | 1101 | 729 | -0.228 | -0.278 | 0.05 | -0.42 | 0.498 |
| OOS1 | 1158 | 835 | -0.304 | -0.496 | 0.192 | -0.239 | 0.644 |
| OOS2 | 2181 | 1300 | 0.133 | 0.013 | 0.12 | -0.267 | 0.47 |

L2 OOS baseline: P&L +3,644 | MDD -818 | red months 6/15 | worst -236 | picks 98 (fills 56, 0.326/day) | eras T +2,442 / O1 +1,135 / O2 +2,509

L2 OOS variant:  P&L +3,220 | MDD -564 | red months 6/15 | worst -178 | picks 68 (fills 41, 0.228/day) | eras T +2,021 / O1 +1,419 / O2 +1,801

ALL-window variant: P&L +5,241 | MDD -564 | red months 7/21 | worst -178 | picks 88 (fills 54, 0.204/day) | eras T +2,021 / O1 +1,419 / O2 +1,801

checks: {'1_pnl_+5%': False, '2_mdd': True, '3_neg_months': True, '4_eras': False, '5_giants_kept>=8': True, '6_picks_not_halved': True, '7_L1_effect': False} | giants kept 8/10

**VERDICT: REJECT**


### C2_upper_pre 

env: `{'ORB_EXP_RCP_GATE': 'pre', 'ORB_EXP_RCP_FORM': 'upper'}`

L1 — upper: kept vs dropped:

| era | n_keep | n_drop | mean_keep | mean_drop | diff | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| ALL | 4548 | 2756 | -0.068 | -0.22 | 0.152 | -0.082 | 0.388 |
| TRAIN | 1135 | 695 | -0.2 | -0.327 | 0.127 | -0.321 | 0.599 |
| OOS1 | 1215 | 778 | -0.278 | -0.551 | 0.273 | -0.173 | 0.714 |
| OOS2 | 2198 | 1283 | 0.117 | 0.039 | 0.078 | -0.276 | 0.452 |

L2 OOS baseline: P&L +3,644 | MDD -818 | red months 6/15 | worst -236 | picks 98 (fills 56, 0.326/day) | eras T +2,442 / O1 +1,135 / O2 +2,509

L2 OOS variant:  P&L +5,407 | MDD -1,098 | red months 7/15 | worst -479 | picks 95 (fills 62, 0.315/day) | eras T +2,621 / O1 +1,151 / O2 +4,256

ALL-window variant: P&L +8,028 | MDD -1,098 | red months 7/21 | worst -479 | picks 122 (fills 80, 0.282/day) | eras T +2,621 / O1 +1,151 / O2 +4,256

checks: {'1_pnl_+5%': True, '2_mdd': False, '3_neg_months': False, '4_eras': True, '5_giants_kept>=8': True, '6_picks_not_halved': True, '7_L1_effect': False} | giants kept 9/10

**VERDICT: REJECT**


### C2_upper_post 

env: `{'ORB_EXP_RCP_GATE': 'post', 'ORB_EXP_RCP_FORM': 'upper'}`

L1 — upper: kept vs dropped:

| era | n_keep | n_drop | mean_keep | mean_drop | diff | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| ALL | 4548 | 2756 | -0.068 | -0.22 | 0.152 | -0.082 | 0.388 |
| TRAIN | 1135 | 695 | -0.2 | -0.327 | 0.127 | -0.321 | 0.599 |
| OOS1 | 1215 | 778 | -0.278 | -0.551 | 0.273 | -0.173 | 0.714 |
| OOS2 | 2198 | 1283 | 0.117 | 0.039 | 0.078 | -0.276 | 0.452 |

L2 OOS baseline: P&L +3,644 | MDD -818 | red months 6/15 | worst -236 | picks 98 (fills 56, 0.326/day) | eras T +2,442 / O1 +1,135 / O2 +2,509

L2 OOS variant:  P&L +3,163 | MDD -779 | red months 7/14 | worst -236 | picks 78 (fills 49, 0.261/day) | eras T +2,484 / O1 +1,281 / O2 +1,882

ALL-window variant: P&L +5,647 | MDD -779 | red months 7/20 | worst -236 | picks 102 (fills 65, 0.237/day) | eras T +2,484 / O1 +1,281 / O2 +1,882

checks: {'1_pnl_+5%': False, '2_mdd': True, '3_neg_months': False, '4_eras': False, '5_giants_kept>=8': True, '6_picks_not_halved': True, '7_L1_effect': False} | giants kept 9/10

**VERDICT: REJECT**


### C3 

env: `{'ORB_EXP_MID_KILL': '1'}`

L1 — rows where the exit changed (mid_kill fired): variant − baseline pnl:

| era | n_fired | mean_delta_pnl | sum_delta | ci_lo | ci_hi |
|---|---|---|---|---|---|
| ALL | 782 | -46.2 | -36132 | -176.7 | 75.0 |
| TRAIN | 215 | 80.9 | 17401 | -186.8 | 297.7 |
| OOS1 | 173 | -12.6 | -2174 | -312.8 | 232.4 |
| OOS2 | 394 | -130.4 | -51358 | -313.0 | 30.0 |

L2 OOS baseline: P&L +3,644 | MDD -818 | red months 6/15 | worst -236 | picks 98 (fills 56, 0.326/day) | eras T +2,442 / O1 +1,135 / O2 +2,509

L2 OOS variant:  P&L +3,881 | MDD -730 | red months 6/15 | worst -221 | picks 98 (fills 56, 0.326/day) | eras T +2,528 / O1 +1,256 / O2 +2,625

ALL-window variant: P&L +6,408 | MDD -730 | red months 6/21 | worst -221 | picks 130 (fills 73, 0.301/day) | eras T +2,528 / O1 +1,256 / O2 +2,625

checks: {'1_pnl_+5%': True, '2_mdd': True, '3_neg_months': True, '4_eras': True, '5_giants_kept>=8': True, '6_picks_not_halved': True, '7_L1_effect': False} | giants kept 10/10

**VERDICT: REJECT**


### C4 

env: `{'ORB_EXP_REARM': '1'}`

L1 — rows where a re-arm fired: variant − baseline pnl:

| era | n_fired | mean_delta_pnl | sum_delta | ci_lo | ci_hi |
|---|---|---|---|---|---|
| ALL | 1687 | -126.5 | -213470 | -238.1 | -12.6 |
| TRAIN | 340 | -105.0 | -35690 | -326.9 | 125.1 |
| OOS1 | 446 | -124.3 | -55444 | -316.5 | 91.9 |
| OOS2 | 901 | -135.8 | -122336 | -287.3 | 23.0 |

L2 OOS baseline: P&L +3,644 | MDD -818 | red months 6/15 | worst -236 | picks 98 (fills 56, 0.326/day) | eras T +2,442 / O1 +1,135 / O2 +2,509

L2 OOS variant:  P&L +3,674 | MDD -687 | red months 5/15 | worst -469 | picks 98 (fills 56, 0.326/day) | eras T +2,510 / O1 +841 / O2 +2,834

ALL-window variant: P&L +6,184 | MDD -687 | red months 6/21 | worst -469 | picks 130 (fills 73, 0.301/day) | eras T +2,510 / O1 +841 / O2 +2,834

checks: {'1_pnl_+5%': False, '2_mdd': True, '3_neg_months': True, '4_eras': False, '5_giants_kept>=8': True, '6_picks_not_halved': True, '7_L1_effect': False} | giants kept 10/10

**VERDICT: REJECT**


TRAIN grid for `ORB_EXP_RVOL_MAX` (base TRAIN P&L +2,442 | MDD -226 | red months 0/6 | worst +18 | picks 32 (fills 17, 0.256/day) | eras T +2,442 / O1 +1,135 / O2 +2,509):

| t | train_pnl | train_mdd | mdd_ok | picks |
|---|---|---|---|---|
| 2.0 | 2864 | -326 | True | 32 |
| 3.0 | 2737 | -326 | True | 32 |
| 4.0 | 2392 | -387 | False | 34 |
| 6.0 | 2213 | -387 | False | 33 |

chosen t = 2.0 (grid edge — reported, not extended)


### C1c POST-HOC high-rvol veto > 2.0 (TRAIN-fit)

env: `{'ORB_EXP_RVOL_MAX': '2.0'}`

L1 — rvol_open5 <= 2.0 (kept) vs > 2.0 (dropped) — POST-HOC:

| era | n_keep | n_drop | mean_keep | mean_drop | diff | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| ALL | 3453 | 3851 | 0.073 | -0.303 | 0.376 | 0.131 | 0.618 |
| TRAIN | 869 | 961 | -0.138 | -0.348 | 0.21 | -0.244 | 0.67 |
| OOS1 | 834 | 1159 | 0.025 | -0.679 | 0.704 | 0.229 | 1.185 |
| OOS2 | 1750 | 1731 | 0.201 | -0.025 | 0.226 | -0.141 | 0.608 |

L2 OOS baseline: P&L +3,644 | MDD -818 | red months 6/15 | worst -236 | picks 98 (fills 56, 0.326/day) | eras T +2,442 / O1 +1,135 / O2 +2,509

L2 OOS variant:  P&L +3,595 | MDD -555 | red months 5/15 | worst -318 | picks 125 (fills 75, 0.41/day) | eras T +2,864 / O1 +1,282 / O2 +2,313

ALL-window variant: P&L +6,459 | MDD -555 | red months 5/21 | worst -318 | picks 157 (fills 92, 0.363/day) | eras T +2,864 / O1 +1,282 / O2 +2,313

checks: {'1_pnl_+5%': False, '2_mdd': True, '3_neg_months': True, '4_eras': True, '5_giants_kept>=8': True, '6_picks_not_halved': True, '7_L1_effect': True} | giants kept 9/10

**VERDICT: PARK**


**C1c is POST-HOC: relabelled EXPLORATORY-PARK. A promising exploratory result needs its own pre-registration and a fresh era before it can be proposed.**


## Pairs

survivors (PROPOSE): [] | L1-yes PARK: []

(no eligible pairs)


## Summary

baseline OOS: P&L +3,644 | MDD -818 | red months 6/15 | worst -236 | picks 98 (fills 56, 0.326/day) | eras T +2,442 / O1 +1,135 / O2 +2,509

| variant | verdict | oos_pnl | mdd | neg_months | picks | L1 | env |
|---|---|---|---|---|---|---|---|
| C1a | REJECT | 2285 | -824 | 8 | 88 | False | {'ORB_EXP_RVOL_VETO': '0.5'} |
| C1b | REJECT | 5187 | -972 | 5 | 83 | False | {'ORB_EXP_RVOL_RANK': '1'} |
| C2_green_pre | REJECT | 7183 | -828 | 6 | 94 | False | {'ORB_EXP_RCP_GATE': 'pre', 'ORB_EXP_RCP_FORM': 'green'} |
| C2_green_post | REJECT | 3220 | -564 | 6 | 68 | False | {'ORB_EXP_RCP_GATE': 'post', 'ORB_EXP_RCP_FORM': 'green'} |
| C2_upper_pre | REJECT | 5407 | -1098 | 7 | 95 | False | {'ORB_EXP_RCP_GATE': 'pre', 'ORB_EXP_RCP_FORM': 'upper'} |
| C2_upper_post | REJECT | 3163 | -779 | 7 | 78 | False | {'ORB_EXP_RCP_GATE': 'post', 'ORB_EXP_RCP_FORM': 'upper'} |
| C3 | REJECT | 3881 | -730 | 6 | 98 | False | {'ORB_EXP_MID_KILL': '1'} |
| C4 | REJECT | 3674 | -687 | 5 | 98 | False | {'ORB_EXP_REARM': '1'} |
| C1c | EXPLORATORY-PARK | 3595 | -555 | 5 | 125 | True | {'ORB_EXP_RVOL_MAX': '2.0'} |

### Proposals

- PROPOSE (ship-candidates, to shadow live first): none
- PARK (signal exists at L1, no book-level lift yet): none
- everything else: REJECT

Decisions are joint: nothing ships from this file. Reproduce any row: `ORB_BT_FEATURES_CSV=/home/ec2-user/onemil/research/orb_signal_study/features_base.csv ORB_BT_SIDECAR_CSV=/home/ec2-user/onemil/research/orb_signal_study/sidecar_rvol.csv,/home/ec2-user/onemil/research/orb_signal_study/sidecar_ratr.csv <env> python3 study_orb_pipeline_static_lock.py`


## Interpretation (written after the rules ran; the rules decided, this explains)

**Headline: none of the five candidates from the sweep improves the ORB book under the pre-registered rules. The one published driver we did not have (Zarattini's opening relative volume) is INVERTED on our universe.**

1. **C1 — opening RVOL is inverted here.** Candidate level, entered rows, by RVOL quintile: q1 (rvol < 1.03) +0.25% mean / 30% WR → q5 (rvol > 6) −0.45% / 13% WR, monotone. Every low-RVOL veto lost money on TRAIN; the rank form (C1b) picked the crowd and lost 7 of the 10 giants. Reading: our universe gate (gap ≥ 5%, prev-day volume ≥ 500K, catalyst) already *is* the stocks-in-play filter (median rvol 2.28); inside it, the extreme-volume names are the crowded fades — the same texture as the July news-crowding result. The post-hoc opposite rule (C1c, veto rvol > 2.0) shows a real candidate-level effect (pooled CI +0.13…+0.62, sign holds in 2 of 3 eras) and a better drawdown (−$555 vs −$818) at flat P&L; it is EXPLORATORY-PARK: it needs its own pre-registration and a fresh era before it can be proposed, because it was generated by this data.
2. **C2 — range-candle direction: book lifts without candidate-level support.** `green_pre` nearly doubles OOS P&L (+$3,539) at flat MDD, but the candidate-level difference is +0.14% with a CI that spans zero, and the composite already weights `range_close_position`/`last_bar_green`. Under the rules that is REJECT: a thin-book lift (56 fills) with no L1 backing is exactly the noise the L1 gate exists to catch. It is the first thing to re-test when the live sample grows.
3. **C3 — midpoint kill: no edge.** Changes 3 of 15 OOS months by small amounts; the fired rows' delta is not distinguishable from zero. Touchgo Rule D already covers the first-bar reversal; the NQ tick-data result does not transfer to small-cap gappers at 1-min granularity.
4. **C4 — one re-arm: the only clean near-miss.** Flat P&L (+$30), MDD −$687 vs −$818, one fewer red month, all 10 giants kept, but it fails the +5% P&L bar and the L1 gate (few fired rows). Cheap to keep as a shadow-only counter in live (would-re-arm events) so the sample grows for free.
5. **C5 — range/ATR tiers:** the composite already excludes wide ranges (zero wide picks in OOS; wide candidates have 11–18% WR). The normal tier was negative in both OOS eras in the book, but the pre-registered veto test did not clear the bar.

**Proposals (joint decision; nothing ships from this file):**
- Do NOT add any of the sweep's entry filters to ORB. The universe gate + composite + vetoes already capture what they measure; the remaining lifts are thin-book noise.
- Instrument two things live at zero cost: (a) log `rvol_open5` per candidate in ORB SCORED so the C1c hypothesis accumulates an out-of-sample era; (b) log would-re-arm events (C4) after tag exits.
- Re-run this study, same protocol, when the live book has one more quarter; C2 green-pre and C1c are the two rows to watch.
- The WhatsApp rule, when you have it verbatim, maps onto this table; if it is one of these, it is answered.
