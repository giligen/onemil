# RESULT_1667 -- every causal arm-time feature, bucketed on the 1.5% floored book

Population: 1663_features.csv, n=5506 (TRAIN-H2=2349, VAL=3157). Book SD net_R: TRAIN-H2=1.3412, VAL=1.3314. MDE at full n: TRAIN-H2=0.078 R, VAL=0.066 R (PREREG stated 0.077/0.066 R -- sanity check).

## R5 -- coverage (availability rail: VOID if <80% or winner/loser gap>5pp)
| feature | coverage% | winner cov% | loser cov% | gap pp | VOID |
|---|---|---|---|---|---|
| F1 | 100.0 | 100.0 | 100.0 | 0.0 | no |
| F2 | 100.0 | 100.0 | 100.0 | 0.0 | no |
| F3 | 96.8 | 96.9 | 96.7 | 0.2 | no |
| F4 | 91.2 | 91.8 | 90.8 | 1.1 | no |
| F5 | 100.0 | 100.0 | 100.0 | 0.0 | no |
| F6 | 99.7 | 99.6 | 99.7 | 0.1 | no |
| F7 | 96.6 | 96.8 | 96.6 | 0.2 | no |
| F8 | 98.7 | 98.7 | 98.8 | 0.1 | no |
| F9 | 98.1 | 98.0 | 98.1 | 0.1 | no |
| F10 | 100.0 | 100.0 | 100.0 | 0.0 | no |
| F11 | 81.4 | 77.2 | 84.0 | 6.8 | YES |
| F12 | 81.4 | 77.2 | 84.0 | 6.8 | YES |
| F13 | 80.2 | 75.9 | 82.9 | 6.9 | YES |
| F14 | 81.4 | 77.2 | 84.0 | 6.8 | YES |
| F15 | 80.7 | 76.3 | 83.4 | 7.1 | YES |
| F16 | 0.0 | 0.0 | 0.0 | 0.0 | YES |
| F17 | 100.0 | 100.0 | 100.0 | 0.0 | no |
| F18 | 100.0 | 100.0 | 100.0 | 0.0 | no |

VOID (not reported as numbers below): F11, F12, F13, F14, F15, F16

## R1_tercile (rows with |t|>=2 in either half; 15/33 shown)
| feature | bucket | TRAIN-H2 n/mean/iid_t/day_t/ex5 | VAL n/mean/iid_t/day_t/ex5 |
|---|---|---|---|
| F1 | B2 | n=836 m=-0.008 it=-0.18 dt=-1.52 ex5=-0.112 | n=1000 m=-0.107 it=-2.63 dt=-3.19 ex5=-0.215 |
| F10 | B2 | n=883 m=-0.091 it=-2.08 dt=-1.19 ex5=-0.201 | n=947 m=-0.078 it=-1.86 dt=-2.49 ex5=-0.187 |
| F17 | B1 | n=997 m=-0.674 it=-23.85 dt=-16.88 ex5=-0.812 | n=1322 m=-0.665 it=-27.79 dt=-20.46 ex5=-0.804 |
| F17 | B2 | n=586 m=-0.070 it=-1.34 dt=-1.74 ex5=-0.179 | n=788 m=-0.145 it=-3.25 dt=-2.55 ex5=-0.258 |
| F17 | B3 | n=766 m=0.866 it=17.34 dt=11.53 ex5=0.807 | n=1047 m=0.860 it=20.21 dt=12.54 ex5=0.801 |
| F2 | B2 | n=823 m=0.035 it=0.75 dt=-1.58 ex5=-0.068 | n=1012 m=-0.121 it=-3.03 dt=-2.86 ex5=-0.231 |
| F3 | B1 | n=749 m=-0.101 it=-2.14 dt=-1.32 ex5=-0.212 | n=1027 m=-0.006 it=-0.15 dt=-2.49 ex5=-0.111 |
| F3 | B2 | n=748 m=0.044 it=0.88 dt=-0.04 ex5=-0.059 | n=1028 m=-0.087 it=-2.17 dt=-3.21 ex5=-0.196 |
| F4 | B1 | n=731 m=-0.032 it=-0.66 dt=-1.23 ex5=-0.138 | n=942 m=-0.012 it=-0.28 dt=-2.04 ex5=-0.118 |
| F4 | B2 | n=669 m=0.060 it=1.15 dt=0.07 ex5=-0.042 | n=1004 m=-0.076 it=-1.88 dt=-2.36 ex5=-0.184 |
| F5 | B1 | n=780 m=-0.017 it=-0.35 dt=-1.01 ex5=-0.121 | n=1055 m=-0.109 it=-2.74 dt=-3.44 ex5=-0.219 |
| F5 | B2 | n=820 m=-0.069 it=-1.54 dt=-2.92 ex5=-0.176 | n=1015 m=-0.078 it=-1.92 dt=-2.75 ex5=-0.186 |
| F6 | B2 | n=772 m=-0.030 it=-0.64 dt=-0.57 ex5=-0.136 | n=1057 m=-0.059 it=-1.49 dt=-2.23 ex5=-0.166 |
| F7 | B1 | n=760 m=-0.069 it=-1.45 dt=-1.17 ex5=-0.175 | n=1014 m=-0.038 it=-0.94 dt=-2.53 ex5=-0.144 |
| F9 | B1 | n=771 m=0.007 it=0.14 dt=-0.28 ex5=-0.097 | n=1029 m=-0.161 it=-4.28 dt=-3.36 ex5=-0.273 |

## R1_quintile (rows with |t|>=2 in either half; 23/55 shown)
| feature | bucket | TRAIN-H2 n/mean/iid_t/day_t/ex5 | VAL n/mean/iid_t/day_t/ex5 |
|---|---|---|---|
| F1 | B1 | n=458 m=-0.099 it=-1.59 dt=-2.57 ex5=-0.208 | n=644 m=0.046 it=0.85 dt=-1.26 ex5=-0.057 |
| F1 | B2 | n=429 m=0.056 it=0.87 dt=-0.68 ex5=-0.047 | n=672 m=-0.057 it=-1.12 dt=-2.17 ex5=-0.164 |
| F1 | B4 | n=507 m=-0.005 it=-0.08 dt=-1.02 ex5=-0.111 | n=594 m=-0.071 it=-1.33 dt=-2.21 ex5=-0.179 |
| F10 | B3 | n=587 m=-0.125 it=-2.37 dt=-1.81 ex5=-0.238 | n=508 m=-0.202 it=-3.65 dt=-2.77 ex5=-0.319 |
| F10 | B4 | n=560 m=0.047 it=0.82 dt=0.24 ex5=-0.053 | n=555 m=-0.247 it=-4.66 dt=-3.34 ex5=-0.364 |
| F17 | B1 | n=550 m=-0.831 it=-28.44 dt=-17.84 ex5=-0.964 | n=711 m=-0.779 it=-28.37 dt=-21.62 ex5=-0.914 |
| F17 | B2 | n=447 m=-0.481 it=-9.55 dt=-5.95 ex5=-0.613 | n=611 m=-0.532 it=-13.28 dt=-10.66 ex5=-0.665 |
| F17 | B3 | n=479 m=-0.144 it=-2.57 dt=-2.85 ex5=-0.255 | n=628 m=-0.222 it=-4.56 dt=-3.57 ex5=-0.339 |
| F17 | B4 | n=450 m=0.394 it=5.95 dt=4.83 ex5=0.310 | n=630 m=0.370 it=6.67 dt=4.38 ex5=0.285 |
| F17 | B5 | n=423 m=1.216 it=19.83 dt=14.14 ex5=1.175 | n=577 m=1.198 it=22.85 dt=15.37 ex5=1.158 |
| F2 | B1 | n=463 m=-0.082 it=-1.35 dt=-2.17 ex5=-0.194 | n=639 m=0.027 it=0.50 dt=-0.73 ex5=-0.076 |
| F2 | B3 | n=498 m=-0.003 it=-0.05 dt=-1.00 ex5=-0.107 | n=603 m=-0.148 it=-2.92 dt=-2.17 ex5=-0.262 |
| F2 | B4 | n=501 m=0.036 it=0.59 dt=0.66 ex5=-0.069 | n=600 m=-0.037 it=-0.67 dt=-2.21 ex5=-0.142 |
| F3 | B3 | n=441 m=-0.039 it=-0.62 dt=-0.79 ex5=-0.149 | n=624 m=-0.082 it=-1.58 dt=-2.55 ex5=-0.192 |
| F4 | B1 | n=437 m=0.058 it=0.90 dt=-0.02 ex5=-0.043 | n=567 m=-0.006 it=-0.11 dt=-2.02 ex5=-0.112 |
| F4 | B3 | n=385 m=0.068 it=0.99 dt=0.45 ex5=-0.035 | n=618 m=-0.096 it=-1.91 dt=-2.63 ex5=-0.205 |
| F4 | B4 | n=427 m=-0.051 it=-0.79 dt=-2.01 ex5=-0.160 | n=577 m=-0.039 it=-0.70 dt=-0.40 ex5=-0.145 |
| F5 | B1 | n=463 m=-0.056 it=-0.91 dt=-1.52 ex5=-0.166 | n=639 m=-0.069 it=-1.31 dt=-2.33 ex5=-0.176 |
| F5 | B2 | n=480 m=-0.065 it=-1.09 dt=-1.57 ex5=-0.171 | n=621 m=-0.156 it=-3.11 dt=-2.92 ex5=-0.271 |
| F5 | B4 | n=477 m=0.011 it=0.18 dt=-1.04 ex5=-0.092 | n=624 m=-0.020 it=-0.37 dt=-2.65 ex5=-0.127 |
| F6 | B3 | n=465 m=0.007 it=0.11 dt=0.00 ex5=-0.099 | n=632 m=-0.082 it=-1.59 dt=-2.08 ex5=-0.190 |
| F9 | B1 | n=466 m=0.010 it=0.17 dt=0.02 ex5=-0.096 | n=614 m=-0.124 it=-2.55 dt=-2.33 ex5=-0.234 |
| F9 | B2 | n=457 m=0.028 it=0.44 dt=-0.98 ex5=-0.075 | n=623 m=-0.131 it=-2.63 dt=-3.85 ex5=-0.244 |

## R2 (rows with |t|>=2 in either half; 4/22 shown)
| feature | bucket | TRAIN-H2 n/mean/iid_t/day_t/ex5 | VAL n/mean/iid_t/day_t/ex5 |
|---|---|---|---|
| F10 | bottom_vs_rest | n=524 m=0.289 it=4.26 dt=nan ex5=0.302 | n=583 m=-0.007 it=-0.12 dt=nan ex5=-0.010 |
| F10 | top_vs_rest | n=381 m=-0.155 it=-2.12 dt=nan ex5=-0.169 | n=697 m=0.082 it=1.41 dt=nan ex5=0.086 |
| F17 | bottom_vs_rest | n=550 m=-1.057 it=-24.05 dt=-14.88 ex5=-1.099 | n=711 m=-0.967 it=-24.64 dt=-17.32 ex5=-1.008 |
| F17 | top_vs_rest | n=423 m=1.509 it=22.48 dt=16.45 ex5=1.587 | n=577 m=1.503 it=26.18 dt=18.29 ex5=1.581 |

## R3_spearman (rows with |t|>=2 in either half; 6/11 shown)
| feature | bucket | TRAIN-H2 n/mean/iid_t/day_t/ex5 | VAL n/mean/iid_t/day_t/ex5 |
|---|---|---|---|
| F10 | fill_level | n=2349 m=-0.046 it=-2.23 dt=nan ex5=nan | n=3157 m=-0.028 it=-1.59 dt=nan ex5=nan |
| F17 | fill_level | n=2349 m=0.453 it=24.60 dt=nan ex5=nan | n=3157 m=0.448 it=28.19 dt=nan ex5=nan |
| F5 | fill_level | n=2349 m=0.023 it=1.10 dt=nan ex5=nan | n=3157 m=0.052 it=2.92 dt=nan ex5=nan |
| F7 | fill_level | n=2261 m=0.030 it=1.43 dt=nan ex5=nan | n=3060 m=0.037 it=2.04 dt=nan ex5=nan |
| F8 | fill_level | n=2318 m=0.078 it=3.75 dt=nan ex5=nan | n=3119 m=0.109 it=6.10 dt=nan ex5=nan |
| F9 | fill_level | n=2304 m=0.020 it=0.95 dt=nan ex5=nan | n=3096 m=0.085 it=4.75 dt=nan ex5=nan |

## R3_spearman_dayclust (rows with |t|>=2 in either half; 2/11 shown)
| feature | bucket | TRAIN-H2 n/mean/iid_t/day_t/ex5 | VAL n/mean/iid_t/day_t/ex5 |
|---|---|---|---|
| F17 | day_level | n=127 m=0.680 it=10.36 dt=nan ex5=nan | n=102 m=0.739 it=10.98 dt=nan ex5=nan |
| F8 | day_level | n=127 m=-0.108 it=-1.22 dt=nan ex5=nan | n=102 m=0.262 it=2.72 dt=nan ex5=nan |

## R4_stopbucket_x_feature (rows with |t|>=2 in either half; 4/12 shown)
| feature | bucket | TRAIN-H2 n/mean/iid_t/day_t/ex5 | VAL n/mean/iid_t/day_t/ex5 |
|---|---|---|---|
| stopbucket x F2 | 1.5-3% x B2 | n=702 m=0.035 it=0.68 dt=-0.79 ex5=-0.069 | n=869 m=-0.135 it=-3.10 dt=-2.87 ex5=-0.246 |
| stopbucket x F2 | 1.5-3% x B3 | n=600 m=-0.035 it=-0.62 dt=-0.37 ex5=-0.140 | n=824 m=-0.047 it=-0.98 dt=-2.33 ex5=-0.154 |
| stopbucket x F3 | 1.5-3% x B1 | n=614 m=-0.079 it=-1.48 dt=-1.17 ex5=-0.187 | n=862 m=-0.035 it=-0.77 dt=-3.01 ex5=-0.141 |
| stopbucket x F3 | 1.5-3% x B2 | n=649 m=0.044 it=0.83 dt=0.05 ex5=-0.058 | n=873 m=-0.103 it=-2.33 dt=-2.81 ex5=-0.211 |

## R4_price_x_F2 (rows with |t|>=2 in either half; 2/6 shown)
| feature | bucket | TRAIN-H2 n/mean/iid_t/day_t/ex5 | VAL n/mean/iid_t/day_t/ex5 |
|---|---|---|---|
| F18 x F2 | priceB1 x F2B2 | n=267 m=-0.082 it=-1.01 dt=-2.05 ex5=-0.195 | n=341 m=-0.099 it=-1.40 dt=-1.26 ex5=-0.213 |
| F18 x F2 | priceB3 x F2B2 | n=245 m=0.000 it=0.01 dt=-0.54 ex5=-0.109 | n=360 m=-0.183 it=-2.85 dt=-2.43 ex5=-0.296 |

## Verdict per feature
Pass bar (PREREG_1662 S1664): net>=+0.05R AND t>=2.5 in BOTH halves AND ex-top-5%>0 in both AND >=3 fills/week at that cut.
PASSES (4): R1_tercile:F17:B3; R1_quintile:F17:B4; R1_quintile:F17:B5; R2:F17:top_vs_rest
Every pass requires an independent reimplementation from prose before reaching the owner (CLAUDE.md #1) -- NOT done in this cell.

## Adequacy review
- 17 features x (3 tercile + 5 quintile + 2 R2) + 17 Spearman + 5 interactions x 6 = ~217 reads pre-declared (PREREG multiplicity); 300 reads actually produced.
- VOID features (availability rail): F11, F12, F13, F14, F15, F16.
- Scope: primary (floored, r_pct>=1.5%) book only, per the Reads section (R1 says "on the primary book"); the unfloored book named in Population is not re-swept here -- it was already covered by cell 1663.
- MDE at full-n book matches the PREREG-stated 0.077/0.066 R (see header) -- SD/formula check passes.
- t=2.0 was used as the reporting threshold (per Output: "|t|>=2 in either half") and t=2.5 in BOTH halves as the pass bar (per S1664), so several rows above may appear here without passing.
- This is a null-population line (HOD-break, 1,438+ cells to date per CLAUDE_HISTORY.md); a lone pass among ~217 reads at the both-halves t>=2.5 bar is within the <=0.01 expected false-positive rate stated in the PREREG and must still clear independent rebuild before being called a finding.

Full read table: /home/ec2-user/onemil/research/hod_entry/1667_reads.csv. Full feature table: /home/ec2-user/onemil/research/hod_entry/1667_features.csv.

## Judge note (main session, 2026-09-29 19:00 UTC) — F17 is VOID, a full-day look-ahead
`n_cross` in causal_arming_causal.csv is `len(cands)` with `cands = armed_crossing_bars(bars, ...)` over the WHOLE
symbol-day (`research/hod_entry/causal_arming.py:268-269`): it counts every armed crossing bar of the day, including
the crosses AFTER the fill. A stock that crosses its level many times is one that kept making new highs — the outcome.
The t of 17–28 on a book whose MDE is 0.07 R is the leak's signature, not a finding. Every F17 read (R1, R2, R3 and any
interaction using it) is VOID; no rebuild is warranted. F11–F16 are VOID under the availability rail (missingness gap
6.8–7.1 pp) and will be re-run on the same PREREG once the bars backfill (running, `research/bf_zero/
backfill_bars_sip_20260929.log`) closes the store's own-day gap. Daily-panel features F1–F10 and price interactions:
nothing clears in both halves. Verdict for the synthesis (1,664): NOTHING SHIPS from this cell.
