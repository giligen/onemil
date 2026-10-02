# RESULT 1,689c -- quiet-window in-regime read, ORB sub-pools 24/25/26/30 (cache.db
# default leg, VOID BY DESIGN in RESULT_1689b.md); 21/27/28 reused unchanged from 1689b.
# Generated 2026-10-01T23:13:33Z by queue_1689c.sh. See PREREG_1689c.md for the frozen
# pool definitions, pass bar, and paper-candidacy decision rule.

production in_regime  n=482
production out_regime n=59

--- POOL 21 / in_regime (n=75, reused(1689b), path=/home/ec2-user/onemil/research/orb_freq/subpools_1689b/21_in_regime_true.csv) ---
21/in_regime/2025: n=32 fills/wk=1.78 meanR=+0.013 iid_t=0.14 dc_t=0.44 exTop5=-0.104 MDE=0.265 wkP10=-0.54R worstWk=-0.98R $+155
21/in_regime/2026: n=43 fills/wk=2.15 meanR=-0.012 iid_t=-0.21 dc_t=-0.81 exTop5=-0.079 MDE=0.166 wkP10=-0.59R worstWk=-0.61R $-198
21/in_regime/FULL: n=75 fills/wk=1.97 meanR=-0.002 iid_t=-0.03 dc_t=-0.23 exTop5=-0.078 MDE=0.147 wkP10=-0.58R worstWk=-0.98R $-42
  raw_overlap_with_prod=0.0% added_after_excl=73 (frequency gain = 73 fills)
21/in_regime/UNION: n=555 fills/wk=6.38 meanR=+0.092 iid_t=3.68 dc_t=3.10 exTop5=-0.004 MDE=0.070 wkP10=-0.90R worstWk=-2.54R $+19,149
21/in_regime/UNION [2025-01-01..2026-09-26]: weeks=91 strong-weeks=3 C1 gap median=12.0 wk P90=20.8 wk [fail]  C4 green=62% null=50% [pass]  weeklyP10=-0.87R worst=-2.54R
in_regime/prod-alone [2025-01-01..2026-09-26]: weeks=91 strong-weeks=3 C1 gap median=12.0 wk P90=20.8 wk [fail]  C4 green=65% null=50% [pass]  weeklyP10=-0.90R worst=-2.39R

--- POOL 21 / out_regime (n=18, reused(1689b), path=/home/ec2-user/onemil/research/orb_freq/subpools_1689b/21_out_regime_true.csv) ---
21/out_regime/FULL: n=18 fills/wk=1.80 meanR=-0.069 iid_t=-0.96 dc_t=-1.97 exTop5=-0.136 MDE=0.201 wkP10=-0.42R worstWk=-0.62R $-464
  raw_overlap_with_prod=0.0% added_after_excl=18 (frequency gain = 18 fills)
21/out_regime/UNION: n=77 fills/wk=3.35 meanR=+0.008 iid_t=0.11 dc_t=0.16 exTop5=-0.106 MDE=0.186 wkP10=-0.77R worstWk=-1.12R $+217
21/out_regime/UNION [2024-07-01..2024-12-31]: weeks=27 strong-weeks=0 C1 gap median=None wk P90=None wk [fail]  C4 green=42% null=49% [fail]  weeklyP10=-0.72R worst=-1.12R
out_regime/prod-alone [2024-07-01..2024-12-31]: weeks=27 strong-weeks=0 C1 gap median=None wk P90=None wk [fail]  C4 green=62% null=50% [pass]  weeklyP10=-0.51R worst=-0.78R

--- POOL 24 / in_regime (n=262, NEW(1689c), path=/home/ec2-user/onemil/research/orb_freq/subpools_1689c/24_in_regime_true.csv) ---
24/in_regime/2025: n=122 fills/wk=2.65 meanR=+0.061 iid_t=1.44 dc_t=0.12 exTop5=-0.011 MDE=0.120 wkP10=-0.71R worstWk=-1.01R $+2,808
24/in_regime/2026: n=140 fills/wk=3.68 meanR=+0.015 iid_t=0.40 dc_t=0.72 exTop5=-0.046 MDE=0.106 wkP10=-1.00R worstWk=-1.66R $+788
24/in_regime/FULL: n=262 fills/wk=3.12 meanR=+0.037 iid_t=1.29 dc_t=0.61 exTop5=-0.030 MDE=0.079 wkP10=-0.79R worstWk=-1.66R $+3,596
  raw_overlap_with_prod=0.0% added_after_excl=259 (frequency gain = 259 fills)
24/in_regime/UNION: n=741 fills/wk=8.23 meanR=+0.082 iid_t=4.00 dc_t=3.17 exTop5=-0.005 MDE=0.058 wkP10=-1.19R worstWk=-3.42R $+22,869
24/in_regime/UNION [2025-01-01..2026-09-26]: weeks=91 strong-weeks=2 C1 gap median=24 wk P90=24 wk [fail]  C4 green=67% null=50% [pass]  weeklyP10=-1.18R worst=-3.42R
in_regime/prod-alone [2025-01-01..2026-09-26]: weeks=91 strong-weeks=3 C1 gap median=12.0 wk P90=20.8 wk [fail]  C4 green=65% null=50% [pass]  weeklyP10=-0.90R worst=-2.39R

--- POOL 24 / out_regime (n=20, reused(1689b), path=/home/ec2-user/onemil/research/orb_freq/subpools_1689b/24_out_regime_true.csv) ---
24/out_regime/FULL: n=20 fills/wk=1.67 meanR=+0.158 iid_t=1.45 dc_t=1.39 exTop5=+0.109 MDE=0.305 wkP10=-0.35R worstWk=-0.50R $+1,186
  raw_overlap_with_prod=0.0% added_after_excl=20 (frequency gain = 20 fills)
24/out_regime/UNION: n=79 fills/wk=3.29 meanR=+0.063 iid_t=0.92 dc_t=0.87 exTop5=-0.044 MDE=0.191 wkP10=-0.57R worstWk=-0.97R $+1,867
24/out_regime/UNION [2024-07-01..2024-12-31]: weeks=27 strong-weeks=0 C1 gap median=None wk P90=None wk [fail]  C4 green=64% null=50% [pass]  weeklyP10=-0.55R worst=-0.97R
out_regime/prod-alone [2024-07-01..2024-12-31]: weeks=27 strong-weeks=0 C1 gap median=None wk P90=None wk [fail]  C4 green=62% null=50% [pass]  weeklyP10=-0.51R worst=-0.78R

--- POOL 25 / in_regime (n=58, NEW(1689c), path=/home/ec2-user/onemil/research/orb_freq/subpools_1689c/25_in_regime_true.csv) ---
25/in_regime/2025: n=16 fills/wk=1.14 meanR=-0.016 iid_t=-0.09 dc_t=0.05 exTop5=-0.187 MDE=0.498 wkP10=-0.65R worstWk=-0.81R $-96
25/in_regime/2026: n=42 fills/wk=2.00 meanR=+0.056 iid_t=0.73 dc_t=1.44 exTop5=-0.040 MDE=0.213 wkP10=-0.60R worstWk=-1.73R $+875
25/in_regime/FULL: n=58 fills/wk=1.66 meanR=+0.036 iid_t=0.49 dc_t=0.98 exTop5=-0.062 MDE=0.205 wkP10=-0.71R worstWk=-1.73R $+779
  raw_overlap_with_prod=0.0% added_after_excl=58 (frequency gain = 58 fills)
25/in_regime/UNION: n=540 fills/wk=6.21 meanR=+0.098 iid_t=3.79 dc_t=3.37 exTop5=+0.001 MDE=0.072 wkP10=-0.97R worstWk=-2.39R $+19,850
25/in_regime/UNION [2025-01-01..2026-09-26]: weeks=91 strong-weeks=3 C1 gap median=12.0 wk P90=20.8 wk [fail]  C4 green=62% null=50% [pass]  weeklyP10=-0.94R worst=-2.39R
in_regime/prod-alone [2025-01-01..2026-09-26]: weeks=91 strong-weeks=3 C1 gap median=12.0 wk P90=20.8 wk [fail]  C4 green=65% null=50% [pass]  weeklyP10=-0.90R worst=-2.39R

--- POOL 25 / out_regime (n=6, reused(1689b), path=/home/ec2-user/onemil/research/orb_freq/subpools_1689b/25_out_regime_true.csv) ---
25/out_regime/FULL: n=6 fills/wk=1.00 meanR=+0.084 iid_t=0.46 dc_t=0.46 exTop5=-0.091 MDE=0.514 wkP10=-0.22R worstWk=-0.29R $+188
  raw_overlap_with_prod=0.0% added_after_excl=6 (frequency gain = 6 fills)
25/out_regime/UNION: n=65 fills/wk=2.83 meanR=+0.036 iid_t=0.46 dc_t=0.36 exTop5=-0.098 MDE=0.217 wkP10=-0.60R worstWk=-0.78R $+869
25/out_regime/UNION [2024-07-01..2024-12-31]: weeks=27 strong-weeks=0 C1 gap median=None wk P90=None wk [fail]  C4 green=55% null=50% [fail]  weeklyP10=-0.59R worst=-0.78R
out_regime/prod-alone [2024-07-01..2024-12-31]: weeks=27 strong-weeks=0 C1 gap median=None wk P90=None wk [fail]  C4 green=62% null=50% [pass]  weeklyP10=-0.51R worst=-0.78R

--- POOL 26 / in_regime (n=0, NEW(1689c), path=/home/ec2-user/onemil/research/orb_freq/subpools_1689c/26_in_regime_true.csv) ---
26/in_regime: n=0 (no fills)

--- POOL 26 / out_regime (n=0, reused(1689b), path=/home/ec2-user/onemil/research/orb_freq/subpools_1689b/26_out_regime_true.csv) ---
26/out_regime: n=0 (no fills)

--- POOL 27 / in_regime (n=260, reused(1689b), path=/home/ec2-user/onemil/research/orb_freq/subpools_1689b/27_in_regime_true.csv) ---
27/in_regime/2025: n=141 fills/wk=3.07 meanR=+0.002 iid_t=0.06 dc_t=-0.65 exTop5=-0.084 MDE=0.117 wkP10=-0.91R worstWk=-1.13R $+128
27/in_regime/2026: n=119 fills/wk=3.31 meanR=+0.021 iid_t=0.52 dc_t=0.85 exTop5=-0.047 MDE=0.115 wkP10=-0.67R worstWk=-1.57R $+952
27/in_regime/FULL: n=260 fills/wk=3.21 meanR=+0.011 iid_t=0.38 dc_t=-0.02 exTop5=-0.063 MDE=0.083 wkP10=-0.83R worstWk=-1.57R $+1,080
  raw_overlap_with_prod=0.0% added_after_excl=259 (frequency gain = 259 fills)
27/in_regime/UNION: n=741 fills/wk=8.14 meanR=+0.073 iid_t=3.53 dc_t=2.63 exTop5=-0.019 MDE=0.058 wkP10=-1.12R worstWk=-3.02R $+20,346
27/in_regime/UNION [2025-01-01..2026-09-26]: weeks=91 strong-weeks=2 C1 gap median=23 wk P90=23 wk [fail]  C4 green=59% null=50% [fail]  weeklyP10=-1.12R worst=-3.02R
in_regime/prod-alone [2025-01-01..2026-09-26]: weeks=91 strong-weeks=3 C1 gap median=12.0 wk P90=20.8 wk [fail]  C4 green=65% null=50% [pass]  weeklyP10=-0.90R worst=-2.39R

--- POOL 27 / out_regime (n=42, reused(1689b), path=/home/ec2-user/onemil/research/orb_freq/subpools_1689b/27_out_regime_true.csv) ---
27/out_regime/FULL: n=42 fills/wk=2.10 meanR=-0.018 iid_t=-0.29 dc_t=-0.70 exTop5=-0.101 MDE=0.176 wkP10=-0.84R worstWk=-1.01R $-286
  raw_overlap_with_prod=0.0% added_after_excl=42 (frequency gain = 42 fills)
27/out_regime/UNION: n=101 fills/wk=4.04 meanR=+0.010 iid_t=0.19 dc_t=-0.01 exTop5=-0.102 MDE=0.155 wkP10=-1.02R worstWk=-1.71R $+394
27/out_regime/UNION [2024-07-01..2024-12-31]: weeks=27 strong-weeks=0 C1 gap median=None wk P90=None wk [fail]  C4 green=47% null=50% [fail]  weeklyP10=-0.93R worst=-1.71R
out_regime/prod-alone [2024-07-01..2024-12-31]: weeks=27 strong-weeks=0 C1 gap median=None wk P90=None wk [fail]  C4 green=62% null=50% [pass]  weeklyP10=-0.51R worst=-0.78R

--- POOL 28 / in_regime (n=156, reused(1689b), path=/home/ec2-user/onemil/research/orb_freq/subpools_1689b/28_in_regime_true.csv) ---
28/in_regime/2025: n=68 fills/wk=1.94 meanR=-0.013 iid_t=-0.24 dc_t=-0.55 exTop5=-0.097 MDE=0.150 wkP10=-0.53R worstWk=-1.22R $-330
28/in_regime/2026: n=88 fills/wk=3.14 meanR=-0.009 iid_t=-0.19 dc_t=-0.23 exTop5=-0.090 MDE=0.139 wkP10=-0.82R worstWk=-1.96R $-303
28/in_regime/FULL: n=156 fills/wk=2.48 meanR=-0.011 iid_t=-0.30 dc_t=-0.52 exTop5=-0.087 MDE=0.102 wkP10=-0.66R worstWk=-1.96R $-634
  raw_overlap_with_prod=0.0% added_after_excl=156 (frequency gain = 156 fills)
28/in_regime/UNION: n=638 fills/wk=7.25 meanR=+0.077 iid_t=3.39 dc_t=2.69 exTop5=-0.017 MDE=0.064 wkP10=-1.04R worstWk=-2.94R $+18,438
28/in_regime/UNION [2025-01-01..2026-09-26]: weeks=91 strong-weeks=3 C1 gap median=12.0 wk P90=20.8 wk [fail]  C4 green=54% null=50% [fail]  weeklyP10=-1.01R worst=-2.94R
in_regime/prod-alone [2025-01-01..2026-09-26]: weeks=91 strong-weeks=3 C1 gap median=12.0 wk P90=20.8 wk [fail]  C4 green=65% null=50% [pass]  weeklyP10=-0.90R worst=-2.39R

--- POOL 28 / out_regime (n=25, reused(1689b), path=/home/ec2-user/onemil/research/orb_freq/subpools_1689b/28_out_regime_true.csv) ---
28/out_regime/FULL: n=25 fills/wk=2.27 meanR=-0.099 iid_t=-1.85 dc_t=-1.17 exTop5=-0.158 MDE=0.149 wkP10=-0.62R worstWk=-0.74R $-925
  raw_overlap_with_prod=0.0% added_after_excl=25 (frequency gain = 25 fills)
28/out_regime/UNION: n=84 fills/wk=3.65 meanR=-0.008 iid_t=-0.13 dc_t=0.13 exTop5=-0.123 MDE=0.171 wkP10=-0.92R worstWk=-1.23R $-244
28/out_regime/UNION [2024-07-01..2024-12-31]: weeks=27 strong-weeks=0 C1 gap median=None wk P90=None wk [fail]  C4 green=38% null=50% [fail]  weeklyP10=-0.83R worst=-1.23R
out_regime/prod-alone [2024-07-01..2024-12-31]: weeks=27 strong-weeks=0 C1 gap median=None wk P90=None wk [fail]  C4 green=62% null=50% [pass]  weeklyP10=-0.51R worst=-0.78R

--- POOL 30 / in_regime (n=51, NEW(1689c), path=/home/ec2-user/onemil/research/orb_freq/subpools_1689c/30_in_regime_true.csv) ---
30/in_regime/2025: n=23 fills/wk=1.35 meanR=+0.036 iid_t=0.36 dc_t=-0.41 exTop5=-0.089 MDE=0.280 wkP10=-0.39R worstWk=-0.45R $+311
30/in_regime/2026: n=28 fills/wk=1.27 meanR=-0.043 iid_t=-0.52 dc_t=-0.14 exTop5=-0.131 MDE=0.229 wkP10=-0.44R worstWk=-0.87R $-448
30/in_regime/FULL: n=51 fills/wk=1.31 meanR=-0.007 iid_t=-0.11 dc_t=-0.35 exTop5=-0.089 MDE=0.177 wkP10=-0.44R worstWk=-0.87R $-138
  raw_overlap_with_prod=0.0% added_after_excl=51 (frequency gain = 51 fills)
30/in_regime/UNION: n=533 fills/wk=6.27 meanR=+0.095 iid_t=3.68 dc_t=3.21 exTop5=-0.001 MDE=0.072 wkP10=-0.94R worstWk=-2.39R $+18,934
30/in_regime/UNION [2025-01-01..2026-09-26]: weeks=91 strong-weeks=2 C1 gap median=23 wk P90=23 wk [fail]  C4 green=63% null=50% [pass]  weeklyP10=-0.90R worst=-2.39R
in_regime/prod-alone [2025-01-01..2026-09-26]: weeks=91 strong-weeks=3 C1 gap median=12.0 wk P90=20.8 wk [fail]  C4 green=65% null=50% [pass]  weeklyP10=-0.90R worst=-2.39R

--- POOL 30 / out_regime (n=6, reused(1689b), path=/home/ec2-user/onemil/research/orb_freq/subpools_1689b/30_out_regime_true.csv) ---
30/out_regime/FULL: n=6 fills/wk=1.00 meanR=+0.259 iid_t=1.18 dc_t=1.18 exTop5=+0.094 MDE=0.616 wkP10=-0.21R worstWk=-0.30R $+583
  raw_overlap_with_prod=0.0% added_after_excl=6 (frequency gain = 6 fills)
30/out_regime/UNION: n=65 fills/wk=2.50 meanR=+0.052 iid_t=0.66 dc_t=0.58 exTop5=-0.081 MDE=0.220 wkP10=-0.51R worstWk=-0.78R $+1,264
30/out_regime/UNION [2024-07-01..2024-12-31]: weeks=27 strong-weeks=0 C1 gap median=None wk P90=None wk [fail]  C4 green=70% null=49% [pass]  weeklyP10=-0.51R worst=-0.78R
out_regime/prod-alone [2024-07-01..2024-12-31]: weeks=27 strong-weeks=0 C1 gap median=None wk P90=None wk [fail]  C4 green=62% null=50% [pass]  weeklyP10=-0.51R worst=-0.78R

wrote /home/ec2-user/onemil/research/orb_freq/1689c_pool_books.csv rows=979
wrote /home/ec2-user/onemil/research/orb_freq/1689c_reads.csv rows=14

=== CELL 1689c PASS BAR (own meanR>=+0.05 & dc_t>=2.0 in-regime, >=0 out-of-regime, exTop5>0) -- identical to 1689b's bar, PREREG_1689c.md ===
21: in n=75 meanR=-0.002 dc_t=-0.23 exTop5=-0.078 | out n=18 meanR=-0.069 -> FAIL
24: in n=262 meanR=+0.037 dc_t=0.61 exTop5=-0.030 | out n=20 meanR=+0.158 -> FAIL
25: in n=58 meanR=+0.036 dc_t=0.98 exTop5=-0.062 | out n=6 meanR=+0.084 -> FAIL
26: in n=0 meanR=+nan dc_t=nan exTop5=+nan | out n=0 meanR=+nan -> FAIL
27: in n=260 meanR=+0.011 dc_t=-0.02 exTop5=-0.063 | out n=42 meanR=-0.018 -> FAIL
28: in n=156 meanR=-0.011 dc_t=-0.52 exTop5=-0.087 | out n=25 meanR=-0.099 -> FAIL
30: in n=51 meanR=-0.007 dc_t=-0.35 exTop5=-0.089 | out n=6 meanR=+0.259 -> FAIL

=== PAPER-CANDIDACY RULE (PREREG_1689c.md: meanR>=+0.10 BOTH halves 2025/2026, >=2 fills/wk, exTop5>=0) ===
21: 2025 meanR=+0.013 n=32 (1.8/wk) | 2026 meanR=-0.012 n=43 (2.1/wk) -> research-only
24: 2025 meanR=+0.061 n=122 (2.7/wk) | 2026 meanR=+0.015 n=140 (3.7/wk) -> research-only
25: 2025 meanR=-0.016 n=16 (1.1/wk) | 2026 meanR=+0.056 n=42 (2.0/wk) -> research-only
26: n=0 -> NOT a candidate
27: 2025 meanR=+0.002 n=141 (3.1/wk) | 2026 meanR=+0.021 n=119 (3.3/wk) -> research-only
28: 2025 meanR=-0.013 n=68 (1.9/wk) | 2026 meanR=-0.009 n=88 (3.1/wk) -> research-only
30: 2025 meanR=+0.036 n=23 (1.4/wk) | 2026 meanR=-0.043 n=28 (1.3/wk) -> research-only
