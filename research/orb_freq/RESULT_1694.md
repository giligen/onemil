# RESULT 1,694 -- where ORB's money actually is (run started 2026-10-01T15:02:03.764485)

PREREG: research/orb_freq/PREREG_1694.md (FROZEN). Owner 10/1 15:25 UTC: "unacceptable. act as a true researcher, find the money." Written incrementally per part.

Interpretation notes (stated, not hidden): A2 keeps A1's lock-as-live stop management at a higher target (a parametric sweep, not a silent mechanism switch); A6 is a pre-lock reference anchor, not a claim that it equals today's live rule -- A3 (no target, ride with the live lock) is the actual production exit and every pass-bar/paired-delta baseline below. Part B's bin EDGES are fixed (1,693's TRAIN-2025-cutpoint tercile boundaries / fixed gap-size and price bins) for both directions; 'direction' flips which window's mean R RANKS the bins into bottom/mid/top. Part C2's gate reconstruction applies PDR/G1/range-size directly via their own live/BT-shared helper functions in the pipeline's own order; candidates clearing all three but still absent from the book are bucketed 'rank_not_selected' (score/Q1/slot/dedup, not individually decomposed -- stated, not assumed) per the PREREG's own fallback clause.

## Part A -- let the runners run (baseline = A3, no target, ride the live lock to 15:45)

| exit | window | n | fills/wk | mean R | day_t | ex_top5 | cap>=2R | cap>=3R | cap>=5R | paired dR | dR day_t | cont. share/dR | giveback share/dR | wk P10 R | strong-gap med/p90 wk | $/yr@375 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1_target3R_lockLive | TRAIN2025 | 212 | 4.08 | +0.300 | 2.3 | +0.154 | 0.21 | 0.00 | 0.00 | +0.030 | 0.6 | 0.00/+0.000 | 1.00/+0.030 | -1.52 | 9.0/14.400000000000002 | $+23,882 |
| A1_target3R_lockLive | VAL2026 | 259 | 6.97 | +0.172 | 1.8 | +0.024 | 0.17 | 0.00 | 0.00 | -0.146 | -1.8 | 0.00/+0.000 | 1.00/-0.146 | -3.40 | 5.0/6.200000000000001 | $+23,383 |
| A2_target4R_lockLive | TRAIN2025 | 212 | 4.08 | +0.338 | 2.3 | +0.139 | 0.17 | 0.13 | 0.00 | +0.068 | 1.2 | 0.00/+0.000 | 1.00/+0.068 | -1.56 | 4.5/12.0 | $+26,889 |
| A2_target4R_lockLive | VAL2026 | 259 | 6.97 | +0.152 | 1.7 | -0.050 | 0.12 | 0.07 | 0.00 | -0.166 | -2.0 | 0.00/+0.000 | 1.00/-0.166 | -3.26 | 5.0/6.200000000000001 | $+20,647 |
| A4_half2R_restNoTarget | TRAIN2025 | 212 | 4.08 | +0.287 | 2.1 | +0.089 | 0.13 | 0.04 | 0.00 | +0.016 | 0.6 | 0.00/+0.000 | 1.00/+0.016 | -1.48 | 10.0/14.0 | $+22,794 |
| A4_half2R_restNoTarget | VAL2026 | 259 | 6.97 | +0.227 | 2.1 | -0.018 | 0.12 | 0.06 | 0.02 | -0.091 | -2.0 | 0.00/+0.000 | 1.00/-0.091 | -3.45 | 3.5/6.0 | $+30,887 |
| A5_target3R_BEat2R | TRAIN2025 | 212 | 4.08 | +0.353 | 2.6 | +0.209 | 0.24 | 0.00 | 0.00 | +0.083 | 1.2 | 0.12/+0.354 | 0.88/+0.045 | -2.02 | 5.0/13.4 | $+28,084 |
| A5_target3R_BEat2R | VAL2026 | 259 | 6.97 | +0.158 | 1.6 | +0.008 | 0.18 | 0.00 | 0.00 | -0.161 | -1.9 | 0.09/-0.114 | 0.91/-0.165 | -3.74 | 3.0/6.0 | $+21,416 |
| A6_target2R_plain_ref | TRAIN2025 | 212 | 4.08 | +0.291 | 2.4 | +0.198 | 0.00 | 0.00 | 0.00 | +0.021 | 0.4 | 0.06/+0.841 | 0.94/-0.033 | -1.10 | 15.0/15.0 | $+23,142 |
| A6_target2R_plain_ref | VAL2026 | 259 | 6.97 | +0.112 | 1.2 | +0.013 | 0.00 | 0.00 | 0.00 | -0.206 | -2.1 | 0.04/-0.299 | 0.96/-0.202 | -3.23 | 6.0/7.7 | $+15,231 |

SHIPS (both TRAIN2025+VAL2026 clear paired dR>=+0.05R, day_t>=2.0, >=3R capture rises): none

## Part B -- who gets the risk (tilt vs flat production sizing)

Ordering agreement (pooled Spearman of bin ranks, TRAIN-select vs VAL-select) = -0.250 (ROBUST bar: >= +0.60)

| variant | direction | test window | n | EV/risk tilted | EV/risk prod | EV/risk gain | $/yr tilted | $/yr prod | ex_top5 tilted | wk P10/unit risk | max DD $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1_gap_size | dirA_selTRAIN_testVAL | VAL2026 | 259 | +0.307 | +0.318 | -3.6% | $+46,829 | $+43,242 | -0.118 | -0.493 | $7,496 |
| B1_gap_size | dirB_selVAL_testTRAIN | TRAIN2025 | 211 | +0.266 | +0.265 | +0.5% | $+19,495 | $+20,930 | -0.077 | -0.819 | $6,435 |
| B1_price | dirA_selTRAIN_testVAL | VAL2026 | 255 | +0.300 | +0.324 | -7.6% | $+38,613 | $+43,396 | -0.084 | -0.501 | $5,508 |
| B1_price | dirB_selVAL_testTRAIN | TRAIN2025 | 211 | +0.258 | +0.264 | -2.5% | $+24,659 | $+20,922 | -0.077 | -0.843 | $8,839 |
| B1_prior_day_volume | dirA_selTRAIN_testVAL | VAL2026 | 259 | +0.234 | +0.318 | -26.4% | $+31,252 | $+43,242 | -0.117 | -0.519 | $6,135 |
| B1_prior_day_volume | dirB_selVAL_testTRAIN | TRAIN2025 | 212 | +0.243 | +0.270 | -10.2% | $+19,342 | $+21,490 | -0.015 | -0.819 | $6,859 |
| B1_range5m_pct | dirA_selTRAIN_testVAL | VAL2026 | 259 | +0.345 | +0.318 | +8.4% | $+48,124 | $+43,242 | -0.107 | -0.541 | $5,198 |
| B1_range5m_pct | dirB_selVAL_testTRAIN | TRAIN2025 | 212 | +0.292 | +0.270 | +8.2% | $+23,198 | $+21,490 | -0.045 | -0.848 | $8,555 |
| B1_rvol_0935 | dirA_selTRAIN_testVAL | VAL2026 | 252 | +0.335 | +0.279 | +20.1% | $+43,578 | $+36,864 | -0.130 | -0.569 | $5,906 |
| B1_rvol_0935 | dirB_selVAL_testTRAIN | TRAIN2025 | 204 | +0.322 | +0.285 | +12.7% | $+24,620 | $+21,837 | -0.021 | -0.753 | $6,400 |
| B1_premkt_dollar_vol | dirA_selTRAIN_testVAL | VAL2026 | 259 | +0.250 | +0.318 | -21.5% | $+30,012 | $+43,242 | -0.107 | -0.551 | $6,216 |
| B1_premkt_dollar_vol | dirB_selVAL_testTRAIN | TRAIN2025 | 212 | +0.240 | +0.270 | -11.1% | $+19,069 | $+21,490 | -0.107 | -0.780 | $9,598 |
| B2_additive | dirA_selTRAIN_testVAL | VAL2026 | 259 | +0.217 | +0.318 | -31.6% | $+26,544 | $+43,242 | -0.143 | -0.486 | $5,929 |
| B3_additive_capped | dirA_selTRAIN_testVAL | VAL2026 | 259 | +0.217 | +0.318 | -31.6% | $+26,544 | $+43,242 | -0.143 | -0.486 | $5,929 |
| B2_additive | dirB_selVAL_testTRAIN | TRAIN2025 | 212 | +0.270 | +0.270 | +0.1% | $+19,730 | $+21,490 | -0.030 | -0.779 | $6,504 |
| B3_additive_capped | dirB_selVAL_testTRAIN | TRAIN2025 | 212 | +0.270 | +0.270 | +0.1% | $+19,730 | $+21,490 | -0.030 | -0.779 | $6,504 |

Spearman ordering agreement across directions = -0.250

## Part C1 -- runner anatomy (>=2R / >=3R by MFE, path-based) and the pre-entry classifier

- >= 2R runners: n=182/478 (38.1%); mean entry_minute 581 vs rest 583; mean minutes_to_peak 194 vs rest 105; mean gap_pct 11.47 vs 8.00; mean range5m_pct 4.17 vs 4.81
- >= 3R runners: n=107/478 (22.4%); mean entry_minute 581 vs rest 582; mean minutes_to_peak 194 vs rest 123; mean gap_pct 10.46 vs 9.00; mean range5m_pct 4.13 vs 4.69

| classifier direction | AUC | AUC placebo | top-decile mean R | top-decile runner share | all mean R | all runner share |
|---|---|---|---|---|---|---|
| TRAIN2025_to_VAL2026 | 0.580 | 0.525 | +0.956 | 0.35 | +0.318 | 0.34 |
| VAL2026_to_TRAIN2025 | 0.528 | 0.557 | -0.324 | 0.55 | +0.270 | 0.44 |

## Part C2 -- the missed runners (by veto)

Taken (entered production fills) >=3R runner share (A3/MFE path): 0.224

| veto | n candidates | n w/ counterfactual | bar coverage % | mean counterfactual R | runner share >=3R | ex_top5 | $ left/yr @375 (full size) |
|---|---|---|---|---|---|---|---|
| G1_veto | 648 | 82 | 12.7 | +0.550 | 0.098 | +0.342 | $+9,888 |
| PDR_veto | 4323 | 575 | 13.3 | +0.072 | 0.042 | -0.136 | $+9,066 |
| no_fill | 5666 | 5666 | 100.0 | +0.000 | 0.000 | +0.000 | $+0 |
| range_size_veto | 113 | 19 | 16.8 | +0.095 | 0.053 | -0.343 | $+398 |
| rank_not_selected | 1937 | 316 | 16.3 | +0.284 | 0.063 | +0.077 | $+19,682 |

At half size: $ left/yr halves for every row above; runner share (R-based, scale-invariant) is unchanged.
NOT reconstructable from this data: the spread gate (needs real-time NBBO, not in OHLCV bars) and the exact individual split of score/Q1/slot-cap/dedup within 'rank_not_selected' (needs the per-day ranking pass re-run; out of this cell's budget) -- stated per PREREG's own fallback clause, not silently assumed.


## Files
`1694_reads.csv` (Part A, 10 rows), `1694_runners.csv` (per-fill table, 478 rows), `1694_missed.csv` (12687 rows), this file, `1694_money.py`, `1694_money.log`.
