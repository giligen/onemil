# RESULT 1,693 -- every ORB sub-pool x its own exit (run started 2026-10-01T13:57:22.074369)

PREREG: research/orb_freq/PREREG_1693.md (FROZEN, Amendment 1). Written incrementally.

## Seeds NOT built (declared per PREREG's own escape clause)
Both original missing seeds (2-3% gap band x F1/F3/F4/F5/F6; F2 premarket $ volume for bands B/C) require NEW minute bars via the bars_sip.db appender for a candidate universe that exists in no CSV on disk -- cells 1684/1685/1689a each needed 5-6 pipeline files to build their own universe from scratch. Infeasible inside this cell's call budget alongside the mandatory 12-exit grid, and there is no owner GO for a new data pull. Everything else below runs on existing data.

## Validation (reconstructed opening-range % vs the book's OWN range_size_pct ground truth)
n=478/483 reconstructed (coverage 99.0%, 5 dropped), corr(my opening-range %, book's range_size_pct)=0.9984, mean diff=+0.152pp (my range measured to ENTRY, book's to range_high -- a small, structural, expected offset, not noise)

## General pools (18) -- best exit per direction, classification
- **idea1** [regime_specific] dirA: none select | dirB: E5_scale50_1R (VAL +0.189R t1.5 -> TRAIN +0.206R n55) | mean OOS-2024H2 R across exits: +0.228 (n_fills reconstructed=166, dropped=268)
- **idea2** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +nan (n_fills reconstructed=5, dropped=8)
- **idea10** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +0.162 (n_fills reconstructed=662, dropped=0)
- **idea11** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +0.309 (n_fills reconstructed=575, dropped=0)
- **AF6** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +0.095 (n_fills reconstructed=38, dropped=0)
- **BF1** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: -0.052 (n_fills reconstructed=8, dropped=11)
- **BF3** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +0.331 (n_fills reconstructed=88, dropped=160)
- **BF4** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +0.118 (n_fills reconstructed=4, dropped=4)
- **BF5** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +0.271 (n_fills reconstructed=29, dropped=42)
- **BF6** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +2.202 (n_fills reconstructed=33, dropped=0)
- **CF1** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: -0.439 (n_fills reconstructed=7, dropped=21)
- **CF3** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: -0.392 (n_fills reconstructed=80, dropped=120)
- **CF4** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: -0.869 (n_fills reconstructed=3, dropped=1)
- **CF5** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: -0.606 (n_fills reconstructed=20, dropped=33)
- **CF6** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: -1.024 (n_fills reconstructed=24, dropped=0)
- **19PRIME** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: -0.044 (n_fills reconstructed=265, dropped=0)
- **19** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +0.047 (n_fills reconstructed=43, dropped=0)
- **20** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: -0.500 (n_fills reconstructed=14, dropped=0)

## Production (478 fills reconstructed) -- admission-feature slices (18 of 18 computed; premkt $ vol coverage=100.0%)
- **gap_size_5-7%** [regime_specific] dirA: E3_target1_5R (TRAIN +0.333R t2.0 -> VAL +0.091R n118) | dirB: E1_production (VAL +0.302R t1.5 -> TRAIN +0.505R n102) | mean OOS-2024H2 R across exits: +nan (n=224)
- **gap_size_7-10%** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +nan (n=160)
- **gap_size_>=10%** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +nan (n=93)
- **price_$3-10** [robust] dirA: E4_target3R (TRAIN +0.360R t1.6 -> VAL +0.183R n109) | dirB: E4_target3R (VAL +0.183R t1.7 -> TRAIN +0.360R n117) | mean OOS-2024H2 R across exits: +nan (n=229)
- **price_$10-20** [regime_specific] dirA: E4_target3R (TRAIN +0.504R t2.1 -> VAL +0.319R n90) | dirB: none select | mean OOS-2024H2 R across exits: +nan (n=157)
- **price_$20-30** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +nan (n=87)
- **prior_day_volume_low** [robust] dirA: E6_scale50_1_5R (TRAIN +0.416R t2.6 -> VAL +0.398R n108) | dirB: E3_target1_5R (VAL +0.307R t2.0 -> TRAIN +0.617R n71) | mean OOS-2024H2 R across exits: +nan (n=183)
- **prior_day_volume_mid** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +nan (n=142)
- **prior_day_volume_high** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +nan (n=153)
- **range5m_pct_low** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +nan (n=173)
- **range5m_pct_mid** [robust] dirA: E1_production (TRAIN +0.244R t1.7 -> VAL +0.669R n75) | dirB: E4_target3R (VAL +0.300R t1.8 -> TRAIN +0.342R n70) | mean OOS-2024H2 R across exits: +nan (n=147)
- **range5m_pct_high** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +nan (n=158)
- **rvol_0935_low** [robust] dirA: E6_scale50_1_5R (TRAIN +0.317R t1.7 -> VAL +0.392R n94) | dirB: E3_target1_5R (VAL +0.246R t1.7 -> TRAIN +0.434R n68) | mean OOS-2024H2 R across exits: +nan (n=164)
- **rvol_0935_mid** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +nan (n=152)
- **rvol_0935_high** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +nan (n=146)
- **premkt_dollar_vol_low** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +nan (n=128)
- **premkt_dollar_vol_mid** [regime_specific] dirA: E3_target1_5R (TRAIN +0.281R t1.6 -> VAL +0.190R n115) | dirB: none select | mean OOS-2024H2 R across exits: +nan (n=187)
- **premkt_dollar_vol_high** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +nan (n=163)

Slices promoted (ROBUST both directions -> replaces production's exit for that slice only): {'price_$3-10': 'E4_target3R', 'prior_day_volume_low': 'E6_scale50_1_5R', 'range5m_pct_mid': 'E6_scale50_1_5R', 'rvol_0935_low': 'E6_scale50_1_5R'}

## Robust pairs (both directions confirm) -- general pools
- price_$3-10 x E3_target1_5R: TRAIN +0.280R (t1.7, n117) / VAL +0.155R (t1.9, n109) / OOS2024H2 +nanR (n0)
- price_$3-10 x E4_target3R: TRAIN +0.360R (t1.6, n117) / VAL +0.183R (t1.7, n109) / OOS2024H2 +nanR (n0)
- prior_day_volume_low x E3_target1_5R: TRAIN +0.617R (t3.8, n71) / VAL +0.307R (t2.0, n108) / OOS2024H2 +nanR (n0)
- prior_day_volume_low x E4_target3R: TRAIN +0.500R (t2.2, n71) / VAL +0.346R (t1.7, n108) / OOS2024H2 +nanR (n0)
- prior_day_volume_low x E5_scale50_1R: TRAIN +0.269R (t1.8, n71) / VAL +0.333R (t1.8, n108) / OOS2024H2 +nanR (n0)
- prior_day_volume_low x E6_scale50_1_5R: TRAIN +0.416R (t2.6, n71) / VAL +0.398R (t2.0, n108) / OOS2024H2 +nanR (n0)
- range5m_pct_mid x E3_target1_5R: TRAIN +0.326R (t2.9, n70) / VAL +0.239R (t1.9, n75) / OOS2024H2 +nanR (n0)
- range5m_pct_mid x E4_target3R: TRAIN +0.342R (t2.3, n70) / VAL +0.300R (t1.8, n75) / OOS2024H2 +nanR (n0)
- range5m_pct_mid x E5_scale50_1R: TRAIN +0.193R (t2.0, n70) / VAL +0.376R (t2.0, n75) / OOS2024H2 +nanR (n0)
- range5m_pct_mid x E6_scale50_1_5R: TRAIN +0.285R (t2.4, n70) / VAL +0.454R (t2.2, n75) / OOS2024H2 +nanR (n0)
- rvol_0935_low x E3_target1_5R: TRAIN +0.434R (t2.6, n68) / VAL +0.246R (t1.7, n94) / OOS2024H2 +nanR (n0)
- rvol_0935_low x E6_scale50_1_5R: TRAIN +0.317R (t1.7, n68) / VAL +0.392R (t2.1, n94) / OOS2024H2 +nanR (n0)

## Regime-specific pairs (one direction only; reported, not shipped) -- general pools
- idea1 x E2_target1R: TRAIN +0.095R (t0.1, n55) / VAL +0.177R (t1.8, n71) / OOS2024H2 +0.204R (n40)
- idea1 x E3_target1_5R: TRAIN +0.171R (t0.3, n55) / VAL +0.253R (t1.7, n71) / OOS2024H2 +0.155R (n40)
- idea1 x E5_scale50_1R: TRAIN +0.206R (t0.6, n55) / VAL +0.189R (t1.5, n71) / OOS2024H2 +0.212R (n40)
- gap_size_5-7% x E1_production: TRAIN +0.505R (t1.6, n102) / VAL +0.302R (t1.5, n118) / OOS2024H2 +nanR (n0)
- gap_size_5-7% x E3_target1_5R: TRAIN +0.333R (t2.0, n102) / VAL +0.091R (t1.1, n118) / OOS2024H2 +nanR (n0)
- gap_size_5-7% x E6_scale50_1_5R: TRAIN +0.419R (t1.9, n102) / VAL +0.196R (t1.6, n118) / OOS2024H2 +nanR (n0)
- gap_size_5-7% x E12_powerHour: TRAIN +0.505R (t1.6, n102) / VAL +0.302R (t1.5, n118) / OOS2024H2 +nanR (n0)
- price_$3-10 x E2_target1R: TRAIN +0.139R (t0.7, n117) / VAL +0.096R (t1.8, n109) / OOS2024H2 +nanR (n0)
- price_$3-10 x E5_scale50_1R: TRAIN +0.188R (t0.9, n117) / VAL +0.279R (t2.1, n109) / OOS2024H2 +nanR (n0)
- price_$3-10 x E6_scale50_1_5R: TRAIN +0.258R (t1.3, n117) / VAL +0.309R (t2.1, n109) / OOS2024H2 +nanR (n0)
- price_$10-20 x E3_target1_5R: TRAIN +0.346R (t2.2, n65) / VAL +0.162R (t0.8, n90) / OOS2024H2 +nanR (n0)
- price_$10-20 x E4_target3R: TRAIN +0.504R (t2.1, n65) / VAL +0.319R (t1.2, n90) / OOS2024H2 +nanR (n0)
- price_$10-20 x E6_scale50_1_5R: TRAIN +0.329R (t1.7, n65) / VAL +0.259R (t1.1, n90) / OOS2024H2 +nanR (n0)
- prior_day_volume_low x E2_target1R: TRAIN +0.324R (t2.4, n71) / VAL +0.177R (t1.3, n108) / OOS2024H2 +nanR (n0)
- prior_day_volume_low x E9_trail_MFE1R: TRAIN +0.354R (t2.3, n71) / VAL +0.148R (t0.8, n108) / OOS2024H2 +nanR (n0)
- range5m_pct_mid x E1_production: TRAIN +0.244R (t1.7, n70) / VAL +0.669R (t2.0, n75) / OOS2024H2 +nanR (n0)
- range5m_pct_mid x E2_target1R: TRAIN +0.143R (t2.0, n70) / VAL +0.084R (t1.1, n75) / OOS2024H2 +nanR (n0)
- range5m_pct_mid x E7_BE_lock_1R: TRAIN +0.293R (t1.7, n70) / VAL +0.488R (t1.7, n75) / OOS2024H2 +nanR (n0)
- range5m_pct_mid x E12_powerHour: TRAIN +0.244R (t1.7, n70) / VAL +0.669R (t2.0, n75) / OOS2024H2 +nanR (n0)
- rvol_0935_low x E4_target3R: TRAIN +0.382R (t1.4, n68) / VAL +0.339R (t2.0, n94) / OOS2024H2 +nanR (n0)
- rvol_0935_low x E5_scale50_1R: TRAIN +0.182R (t1.0, n68) / VAL +0.318R (t1.8, n94) / OOS2024H2 +nanR (n0)
- premkt_dollar_vol_mid x E3_target1_5R: TRAIN +0.281R (t1.6, n70) / VAL +0.190R (t0.8, n115) / OOS2024H2 +nanR (n0)
## Union (2025-01-01..2026-09-18, production + robust pairs; fixed $375/fill unless noted)
- production ALONE: n=471, 5.28 fills/wk, mean +0.249 R/fill, weekly P10 -1.91 R ($-715), worst week $-1995, max drawdown 12.030956614243607 R, strong-week gap median/p90=7.0/17.400000000000002 wk, green 0.6086956521739131 vs null 0.49898894982531505, worst day 2026-07-20 ($-2683)
- UNION (production+robust): n=471, 5.28 fills/wk, mean +0.249 R/fill, weekly P10 -1.91 R ($-715 at $375/fill flat), worst week $-1995, max drawdown 12.030956614243607 R, strong-week gap median/p90=7.0/17.400000000000002 wk, green 0.6086956521739131 vs null 0.49898894982531505, worst day 2026-07-20 ($-2683)
- weekly P10 PER FILL: production -0.362 R/fill-week vs union -0.362 R/fill-week
- at a FIXED total weekly risk budget W=$1978 (= production's own weekly risk): union's per-fill risk recalibrates to $375/fill, weekly $ P10 = $-715 vs production's $-715
- see 1693_union.csv for the robust+regime variant and the per-window (TRAIN/VAL/OOS2024H2) breakdown

## Union with per-slice exits (Amendment 1: a robust slice's own exit, production's E1 elsewhere)
Precedence on overlap = higher TRAIN2025 day-clustered t wins (applied last); 220 fills belong to >=2 robust slices. Overlaps: price_$3-10 & prior_day_volume_low: 93 fills; price_$3-10 & range5m_pct_mid: 74 fills; price_$3-10 & rvol_0935_low: 95 fills; prior_day_volume_low & range5m_pct_mid: 53 fills; prior_day_volume_low & rvol_0935_low: 124 fills; range5m_pct_mid & rvol_0935_low: 50 fills.
Direction-A picks (highest VAL2026 mean_R among robust exits): {price_$3-10: E4_target3R, prior_day_volume_low: E6_scale50_1_5R, range5m_pct_mid: E6_scale50_1_5R, rvol_0935_low: E6_scale50_1_5R}
Direction-B picks (highest TRAIN2025 mean_R among robust exits): {price_$3-10: E4_target3R, prior_day_volume_low: E3_target1_5R, range5m_pct_mid: E4_target3R, rvol_0935_low: E3_target1_5R}

- **TRAIN2025**: production +0.270R (n212) | union-dirA +0.326R (n212, ΔR +0.056 iid-t 1.17 day-t 0.85 ex5 -0.035) | union-dirB +0.376R (n212, ΔR +0.106 iid-t 1.43 day-t 0.85 ex5 -0.025) | weekly P10 R prod/dirA/dirB = -2.25/-1.59/-1.10 | max DD 18.1/12.0/9.0 R | strong-wk gap median 4.0/10.0/10.0 wk | $/yr@$375 +21504/+25924/+29931
- **VAL2026**: production +0.318R (n259) | union-dirA +0.186R (n259, ΔR -0.132 iid-t -2.10 day-t -1.97 ex5 -0.177) | union-dirB +0.109R (n259, ΔR -0.209 iid-t -2.04 day-t -1.95 ex5 -0.310) | weekly P10 R prod/dirA/dirB = -3.26/-2.24/-1.93 | max DD 10.8/7.6/5.6 R | strong-wk gap median 3.0/6.0/7.0 wk | $/yr@$375 +43224/+25341/+14783
- **OOS2024H2**: production +nanR (n0) | union-dirA +nanR (n0, ΔR +nan iid-t nan day-t nan ex5 +nan) | union-dirB +nanR (n0, ΔR +nan iid-t nan day-t nan ex5 +nan) | weekly P10 R prod/dirA/dirB = +nan/+nan/+nan | max DD 0.0/0.0/0.0 R | strong-wk gap median nan/nan/nan wk | $/yr@$375 +nan/+nan/+nan
- **WHOLE_2025_2026**: production +0.297R (n471) | union-dirA +0.249R (n471, ΔR -0.047 iid-t -1.16 day-t -1.44 ex5 -0.118) | union-dirB +0.229R (n471, ΔR -0.067 iid-t -1.02 day-t -1.32 ex5 -0.193) | weekly P10 R prod/dirA/dirB = -2.47/-1.91/-1.58 | max DD 19.6/12.0/9.0 R | strong-wk gap median 3.5/7.0/7.5 wk | $/yr@$375 +30560/+25681/+23615
