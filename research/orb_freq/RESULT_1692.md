# RESULT_1692 -- reversed protocol: SELECT 2026-01-01..2026-09-26, TEST 2025 (vs ORIGINAL TRAIN25->VAL26)

Bar-walk coverage for (b)/(c) P1 exit variants: 394/394 (100.0%). SELECT2026 truncates at the 2026-09-18 data cutoff (8 trading days short of 09-26, disclosed in PREREG_1692.md). Pool "original" TRAIN25/VAL26 numbers marked `fresh` below were never previously published (RESULT_1684/1685 scored full in-regime only) -- computed here, same method.

## Side-by-side (REVERSED protocol first, then ORIGINAL direction, then class)
| Candidate | grp | n/wk/R/t_sel26 | pass26 | n/wk/R/iid/dc_test25 | exTop5/MDE/grn_test25 | R/t_train25(orig) | R/t_val26(orig) | class |
|---|---|---|---|---|---|---|---|---|
| P1_plain [context] | baseline | 216/5.68/+0.051/1.45 | - | 178/3.49/+0.041/1.14/0.18 | -0.032/0.10/24 | +0.041/0.18 | +0.076/1.69 | context_only |
| b_live_rule [context] | baseline | 216/5.68/+0.227/1.22 | - | 178/3.49/+0.080/0.69/-0.22 | -0.169/0.32/23 | +0.080/-0.22 | +0.372/1.36 | context_only |
| a_F1 | p1_variant | 12/1.50/+0.198/0.86 | N | 18/1.20/+0.009/0.09/0.31 | -0.076/0.28/5 | +0.009/0.31 | +0.198/0.86 | fails_both |
| a_F3 | p1_variant | 170/4.59/+0.050/1.58 | N | 130/2.71/+0.009/0.21/-1.13 | -0.067/0.12/20 | +0.009/-1.13 | +0.093/2.07 | fails_both |
| a_F4 | p1_variant | 1/1.00/-0.146/nan | N | 5/1.25/-0.134/-1.73/-1.19 | -0.208/0.22/1 | -0.134/-1.19 | -0.146/nan | fails_both |
| a_F5 | p1_variant | 34/1.79/+0.057/0.49 | N | 29/1.16/-0.012/-0.16/-0.05 | -0.093/0.21/6 | -0.012/-0.05 | +0.096/0.92 | fails_both |
| a_F6 | p1_variant | 18/1.29/+0.010/-0.12 | N | 17/1.13/-0.067/-0.89/-0.83 | -0.120/0.21/3 | -0.067/-0.83 | +0.011/-0.13 | fails_both |
| b_scale50_1R | p1_variant | 216/5.68/+0.265/2.70 | Y | 178/3.49/+0.091/1.08/0.15 | -0.057/0.24/27 | +0.091/0.15 | +0.374/2.68 | regime_specific |
| b_noexit2R_half3R_trail1R | p1_variant | 216/5.68/+0.094/0.56 | N | 178/3.49/+0.066/0.59/-0.06 | -0.118/0.31/23 | +0.066/-0.06 | +0.162/0.45 | fails_both |
| c_cost_sizing | p1_variant | 216/5.68/+0.051/1.45 | N | 178/3.49/+0.041/1.14/0.18 | -0.032/0.10/24 | +0.041/0.18 | +0.076/1.69 | fails_both |
| idea1 (fresh-orig) | pool | 216/5.68/+0.051/1.45 | N | 178/3.49/+0.041/1.14/0.18 | -0.032/0.10/24 | +0.041/0.18 | +0.076/1.69 | fails_both |
| idea2 (fresh-orig) | pool | 11/2.20/-0.073/-1.15 | N | 2/1.00/+0.846/1.70/1.70 | +0.349/1.39/2 | +0.846/1.70 | -0.017/0.01 | fails_both |
| idea10 (fresh-orig) | pool | 322/8.94/+0.035/0.62 | N | 265/5.10/-0.103/-4.35/-3.98 | -0.165/0.07/14 | -0.103/-3.98 | +0.015/0.69 | fails_both |
| idea11 (fresh-orig) | pool | 229/6.19/+0.041/0.84 | N | 258/5.06/-0.029/-1.14/-0.29 | -0.092/0.07/19 | -0.029/-0.29 | +0.037/0.70 | fails_both |
| AF6 (fresh-orig) | pool | 5/1.25/+0.035/0.19 | N | 23/1.28/+0.156/1.31/1.24 | +0.019/0.33/7 | +0.156/1.24 | +0.100/0.47 | fails_both |
| BF1 (fresh-orig) | pool | 3/1.00/-0.221/-4.60 | N | 13/1.18/-0.036/-0.26/0.04 | -0.160/0.39/2 | -0.036/0.04 | -0.228/-2.77 | fails_both |
| BF3 (fresh-orig) | pool | 116/3.22/-0.005/-0.22 | N | 106/2.36/-0.041/-0.95/-1.80 | -0.134/0.12/11 | -0.041/-1.80 | +0.050/0.57 | fails_both |
| BF4 (fresh-orig) | pool | 1/1.00/-0.146/nan | N | 4/1.00/-0.208/-7.17/-7.17 | -0.231/0.08/0 | -0.208/-7.17 | -0.146/nan | fails_both |
| BF5 (fresh-orig) | pool | 26/1.37/+0.031/0.59 | N | 33/1.43/-0.007/-0.11/-0.63 | -0.056/0.17/7 | -0.007/-0.63 | +0.024/0.43 | fails_both |
| BF6 (fresh-orig) | pool | 16/1.14/-0.158/-1.57 | N | 14/1.40/-0.172/-2.54/-2.03 | -0.228/0.19/1 | -0.172/-2.03 | -0.151/-1.39 | fails_both |
| CF1 (fresh-orig) | pool | 17/1.55/+0.017/0.24 | N | 8/1.14/-0.028/-0.38/-0.66 | -0.056/0.21/3 | -0.028/-0.66 | +0.056/0.59 | fails_both |
| CF3 (fresh-orig) | pool | 114/3.35/-0.010/0.42 | N | 67/1.86/+0.097/1.46/1.04 | +0.015/0.19/18 | +0.097/1.04 | +0.020/0.47 | fails_both |
| CF4 (fresh-orig) | pool | 0/nan/nan/nan | N | 2/1.00/-0.016/-0.09/-0.09 | -0.194/0.50/1 | -0.016/-0.09 | nan/nan | fails_both |
| CF5 (fresh-orig) | pool | 24/1.85/+0.021/-0.26 | N | 19/1.19/+0.059/0.51/0.55 | -0.001/0.32/5 | +0.059/0.55 | +0.007/-0.35 | fails_both |
| CF6 (fresh-orig) | pool | 11/1.10/+0.102/0.76 | N | 12/1.00/-0.083/-0.90/-0.90 | -0.161/0.26/3 | -0.083/-0.90 | +0.159/0.83 | fails_both |
| production (fresh-orig) | production | 267/7.42/+0.105/2.59 | Y | 215/4.39/+0.108/2.82/2.29 | +0.019/0.11/23 | +0.108/2.29 | +0.137/2.58 | robust |

Columns: sel26 = n/fills-wk/meanR/dc_t on SELECT2026. test25 = n/fills-wk/meanR/iid_t/dc_t on TEST2025. tail25 = exTop5/MDE/greenWeeks on TEST2025. orig = existing-file (P1 family) or freshly-computed-here (pools/production) meanR/dc_t on the ORIGINAL TRAIN2025 and VAL2026 windows.

## Classification counts (24 candidates; P1_plain/b_live_rule shown as context, excluded): {'fails_both': 22, 'regime_specific': 1, 'robust': 1}

## Regime-specific candidates -- exploration-tier line + forward read needed
- **b_scale50_1R**: SELECT26 meanR=+0.265 dc_t=2.70 (n=216) but TEST25 meanR=+0.091 dc_t=0.15 (n=178). Exploration-tier line: positive point estimate on the latest regime (2026) + mechanism stated in PREREG_1684/1685/1690 + bounded downside at minimum size = eligible to run live at minimum size per `feedback_live_exploration_tier`, NOT as a proven edge. Forward read needed: >=40 live fills forward from today (2026-10-01), out-of-both-samples, before any ramp -- same bar as the ORB ramp-advance rule.

## Robust candidates (pass SELECT26 AND TEST25): ['production']

## Caveats (read before relaying): multiplicity continues the programme ledger (24 cands x 2 windows re-cut on populations already scored twice in cells 1684/1685/1690 -- "robust" here means "survives a second cut of the SAME book," not a fresh out-of-sample population). No placebo/green-week-null decomposition run (budget). Causality/fill-realism/price-scale inherited unchanged from PREREG_1684/1685/1690, not re-verified here. **Not independent evidence, do not triple-count**: `idea1` (pool) IS `P1_plain` -- same filter (pool=='idea1', window=='in_regime') on the same source file, confirmed byte-identical here. `c_cost_sizing` is also byte-identical to `P1_plain` in BOTH directions and in the pre-existing 1690_reads.csv (verified) -- the cost-sizing gate (range_pct<0.75%) fires on 0/394 fills in this population (matches commit 8916fef's prior finding), so it is algebraically P1_plain, not a distinct test. Of the 22 fails_both, these two are the same population as the P1_plain/b_live_rule context rows, not independent fails.

Files: research/orb_freq/PREREG_1692.md, 1692_reads.csv, 1692_reverse.py, 1692_reverse.log.
