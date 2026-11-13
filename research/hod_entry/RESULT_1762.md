# RESULT 1,762: approach entry on the honest population (PIT gappers, R >= 1 %). KILL (TEST sealed)

Rule frozen 20:31 UTC before VAL. 3,819 cells; 18 net > 0 in both TRAIN halves, 0 with t >= 2.

**Coverage.** 7,487 PIT gapper symbol-days, 230 days. Minutes 99.7 % (41 % from bars_sip.db, rest fetched; up-minus-down-day gap 0.03 pp). Daily MAs 95.3 %, valid. Windows use the real ET offset.

**R floor** (min3, K20): removes 38 % of APPR_0.5/1.0, 54-65 % of APPR2; bar stop 70-87 %; pct stop 99 %.

**TRAIN anatomy** (min3, K20, floored). Gross is flat: APPR_1.0 -0.003 R (net -0.229, P(break) 0.64), APPR_0.5 -0.030, causal BREAK +0.019 (net -0.173).
- Winner AUC 0.46-0.55; best pc_ma50, ma20 slope, gap 0.55 (t about 2).
- P(break): dist_open 0.57 (t 5.3), prev_range 0.57 (t 4.7).
- Volume and time add nothing to wins.

**Rule.** APPR_1.0, bar stop, K10, dist_open >= +9.1 %. TRAIN 4.9/wk, +0.203 R (t 1.41), H1 -0.008 / H2 +0.356, ex-top-5 % +0.103.

**VAL (22 wk):** 116 trades (5.3/wk), net -0.182 R, day-t -1.13, gross +0.100, ex-top-5 % -0.300, P(break) 0.67 (TRAIN 0.74), placebo -0.168, cost x1.5 -0.322, green weeks 27 % (null 26 %), weekly P10 -4.64 R, MDE 0.45 R.

**Comparators** (same stop, K, floor, unfiltered): BREAK n 940 -0.249 R (t -4.9, gross +0.001); HOLD10 -0.346; RETEST -0.456; unfiltered APPR_1.0 -0.366.

**KILL:** net <= 0, day-t < 2, halves opposite, ex-top-5 % <= 0, placebo >= rule. The 1,752 gross came from break-conditioned arm days; on the honest population it is zero.

**Caveats.** Cost is a symbol-median or price-band half-spread, not NBBO (44 % of rule rows). 3,819 cells join the programme count.

CHILDREN:
1. dist_open and prev_range predict P(break) at AUC 0.57 but not net; test a break-conditional entry on the top P(break) decile with NBBO cost.
2. Rerun BREAK here with the stop-limit exit 20 bp below the stop (gross +0.00 R, cost-bound).
3. Score the first 30 minutes on these bars; the 1,752 time effect moved only P(break).
