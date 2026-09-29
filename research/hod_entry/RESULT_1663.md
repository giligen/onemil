# RESULT_1663 -- the cuts I should have proposed (1,438 population, 1.5% stop floor)

PREREG: research/hod_entry/PREREG_1662.md (frozen). Population: research/hod_entry/fills_1658.csv
(9,911 fills, measured cost entry 7 / stop 6 / target 0 / EOD 11 bps) inner-joined 1:1 on (date,symbol)
to causal_arming_causal.csv status==fill (9,911/9,911 matched) for entry/stop/level/fill_min/exit_m/exit_price/why.
Half labels = fills_1658 `split` (TRAIN-H2 n=4,398; VAL n=5,513), matches cell 1,658 exactly (spot-checked).

## Floor and features
>=1.5% stop floor (bucket_r in {1.5-3%,>=3%}): 5506 of 9,911 fills (4,503 + 1,003).
ATR14: **research/overnight_high/panel_2024_2026.parquet** daily OHLC (16,263 symbols, 2024-07-01..2026-09-04).
bars_sip.db has no `daily` table (only `bars`+`fetch_log`); a `bars` query did not return in 120s (likely
intraday SIP granularity, locked/slow per the system-in-dev note) -- not used, parquet has full date/symbol coverage.
TR/ATR14 computed per symbol from daily OHLC; asof-merged using the **last panel bar strictly before the fill date**
(prior-close information only, causal). Coverage: 98.7% (129/9,911 missing, mostly thin recent listings);
winner/loser missingness gap = 0.12pp (rail: <=5pp, pass). atr14_pct = atr14/entry_price*100.
Arm time = fill_min (minutes since midnight) on the causal fill file, per instruction. Target = entry + 2*(entry-stop);
target_pct = 2*r_pct exactly (fixed 2R target by construction). Price bucket uses entry price; no fills <$15 (min $19.98).
Sanity check: unsliced bucket_r x half net_R reproduces cell 1,658 exactly: 1.5-3% TRAIN -0.009/VAL -0.068,
>=3% TRAIN -0.076/VAL +0.138 (commit cc7cee1: -0.01/-0.07, -0.08/+0.14). Pipeline verified.

Terciles (a,b,d) computed on the POOLED floored population, same edges applied to both halves.
MDE = 2.8 x day-clustered SE (~80% power, two-sided 5%). t/MDE = NaN where a half has <2 distinct trade days or zero variance.

## All 30 pre-declared reads (n / mean net R / day-clustered t), TRAIN-H2 vs VAL vs POOLED

| cut | bucket | n TR | net TR | t TR | n VAL | net VAL | t VAL | net POOL | fpw TR | fpw VAL | ex5 TR | ex5 VAL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) ATR14% of price, terciles | T1(low) | 771 | 0.007 | -0.3 | 1029 | -0.161 | -3.4 | -0.089 | 29.5 | 49.0 | -0.097 | -0.273 |
| (a) ATR14% of price, terciles | T2(mid) | 779 | -0.011 | 0.6 | 1021 | 0.041 | -0.8 | 0.018 | 29.8 | 48.6 | -0.115 | -0.062 |
| (a) ATR14% of price, terciles | T3(high) | 754 | -0.060 | -1.6 | 1046 | 0.025 | -0.6 | -0.011 | 28.8 | 49.8 | -0.167 | -0.079 |
| (b) stop%/ATR14% (tight-for-name vs wide), terciles | T3(high) | 793 | -0.007 | -0.9 | 1007 | -0.064 | -1.4 | -0.039 | 30.3 | 48.0 | -0.112 | -0.172 |
| (b) stop%/ATR14% (tight-for-name vs wide), terciles | T1(low) | 777 | -0.042 | -1.3 | 1023 | -0.030 | -2.1 | -0.035 | 29.7 | 48.7 | -0.148 | -0.136 |
| (b) stop%/ATR14% (tight-for-name vs wide), terciles | T2(mid) | 734 | -0.014 | 0.2 | 1066 | -0.003 | -1.1 | -0.008 | 28.1 | 50.8 | -0.119 | -0.108 |
| (c) arm time of day | 09:35-10:00 | 767 | -0.085 | -1.9 | 1118 | 0.053 | -1.3 | -0.003 | 29.3 | 53.2 | -0.195 | -0.047 |
| (c) arm time of day | 10:00-11:00 | 994 | 0.012 | -1.2 | 1347 | -0.054 | -1.8 | -0.026 | 38.0 | 64.1 | -0.091 | -0.161 |
| (c) arm time of day | 11:00-12:30 | 413 | 0.028 | -0.4 | 456 | -0.105 | -3.0 | -0.042 | 15.8 | 21.7 | -0.075 | -0.214 |
| (c) arm time of day | 12:30-14:00 | 175 | -0.045 | -0.9 | 236 | -0.137 | -1.3 | -0.098 | 6.7 | 11.2 | -0.153 | -0.249 |
| (d) target distance (2R %of price), terciles | T3(high) | 767 | -0.044 | -1.7 | 1069 | 0.080 | -0.4 | 0.028 | 29.3 | 50.9 | -0.152 | -0.021 |
| (d) target distance (2R %of price), terciles | T2(mid) | 795 | 0.017 | -0.1 | 1040 | -0.023 | -0.7 | -0.006 | 30.4 | 49.5 | -0.086 | -0.126 |
| (d) target distance (2R %of price), terciles | T1(low) | 787 | -0.036 | -1.9 | 1048 | -0.148 | -3.4 | -0.100 | 30.1 | 49.9 | -0.142 | -0.258 |
| (f) price bucket | 30-100 | 1270 | 0.008 | -0.9 | 1546 | -0.026 | -1.8 | -0.011 | 48.6 | 73.6 | -0.095 | -0.132 |
| (f) price bucket | 15-30 | 767 | -0.049 | -2.3 | 1022 | -0.038 | -1.2 | -0.042 | 29.3 | 48.7 | -0.156 | -0.145 |
| (f) price bucket | >=100 | 312 | -0.073 | -0.5 | 589 | -0.024 | -1.5 | -0.041 | 11.9 | 28.0 | -0.183 | -0.130 |
| (e) exit type within stop-distance bucket [NOT CAUSAL - see note] | 1.5-3% | eod | 279 | 0.339 | 7.0 | 377 | 0.372 | 7.5 | 0.358 | 10.7 | 18.0 | 0.269 | 0.305 |
| (e) exit type within stop-distance bucket [NOT CAUSAL - see note] | 1.5-3% | stop | 1092 | -1.093 | -354.7 | 1509 | -1.089 | -545.1 | -1.091 | 41.8 | 71.9 | -1.095 | -1.091 |
| (e) exit type within stop-distance bucket [NOT CAUSAL - see note] | 1.5-3% | target | 559 | 1.934 | 2552.0 | 687 | 1.935 | 3333.9 | 1.935 | 21.4 | 32.7 | 1.933 | 1.934 |
| (e) exit type within stop-distance bucket [NOT CAUSAL - see note] | >=3% | stop | 230 | -1.048 | -369.4 | 272 | -1.057 | -229.7 | -1.053 | 8.8 | 13.0 | -1.049 | -1.058 |
| (e) exit type within stop-distance bucket [NOT CAUSAL - see note] | >=3% | eod | 95 | 0.262 | 4.1 | 151 | 0.345 | 4.7 | 0.313 | 3.6 | 7.2 | 0.189 | 0.268 |
| (e) exit type within stop-distance bucket [NOT CAUSAL - see note] | >=3% | target | 94 | 1.963 | 6333.0 | 161 | 1.962 | 5748.3 | 1.962 | 3.6 | 7.7 | 1.962 | 1.962 |
| (g) stop-distance x time-of-day (2-cut interaction) | 1.5-3% | 09:35-10:00 | 545 | -0.087 | -1.1 | 803 | 0.007 | -2.0 | -0.031 | 20.8 | 38.2 | -0.198 | -0.098 |
| (g) stop-distance x time-of-day (2-cut interaction) | 1.5-3% | 10:00-11:00 | 851 | 0.026 | -0.9 | 1147 | -0.081 | -1.9 | -0.036 | 32.6 | 54.6 | -0.077 | -0.190 |
| (g) stop-distance x time-of-day (2-cut interaction) | 1.5-3% | 11:00-12:30 | 380 | 0.030 | -0.2 | 413 | -0.117 | -2.9 | -0.047 | 14.5 | 19.7 | -0.071 | -0.228 |
| (g) stop-distance x time-of-day (2-cut interaction) | >=3% | 10:00-11:00 | 143 | -0.071 | -1.2 | 200 | 0.103 | 0.9 | 0.031 | 5.5 | 9.5 | -0.192 | 0.005 |
| (g) stop-distance x time-of-day (2-cut interaction) | 1.5-3% | 12:30-14:00 | 154 | -0.022 | -0.8 | 210 | -0.179 | -2.0 | -0.113 | 5.9 | 10.0 | -0.130 | -0.296 |
| (g) stop-distance x time-of-day (2-cut interaction) | >=3% | 09:35-10:00 | 222 | -0.079 | -0.7 | 315 | 0.172 | 0.4 | 0.068 | 8.5 | 15.0 | -0.196 | 0.076 |
| (g) stop-distance x time-of-day (2-cut interaction) | >=3% | 12:30-14:00 | 21 | -0.208 | -0.7 | 26 | 0.197 | 0.9 | 0.016 | 0.8 | 1.2 | -0.419 | 0.049 |
| (g) stop-distance x time-of-day (2-cut interaction) | >=3% | 11:00-12:30 | 33 | 0.011 | -0.2 | 43 | 0.012 | 0.0 | 0.012 | 1.3 | 2.0 | -0.115 | -0.134 |

## Synthesis (cell 1,664 input): net>=+0.05R AND t>=2.5 on BOTH halves AND ex-top-5%>0 both AND fills/wk>=3 both

**4 of 30 reads numerically clear** -- all 4 are `(e) exit type within stop bucket`: {1.5-3%,>=3%} x {target,eod}.

**Excluded from candidacy, not just flagged: cut (e) fails the causality trace.** `exit_type` (stop/target/EOD) is the
TRADE OUTCOME, not knowable at entry -- conditioning on "hit target" is conditioning on winning, exactly the
population-filter-look-ahead trap (an ex-ante rule cannot select "fills that will hit target"). The near-zero within-bucket
variance drives the absurd |t| (up to ~6,300): net_R is ~constant at +1.96 (target, = fixed 2R minus cost) or ~-1.05 (stop,
= full stop minus cost) by construction. Diagnostic only, per PREREG's own framing ("exit type share and net R" is
attribution, not a filter). **0 of the 24 causally-valid reads (a,b,c,d,f,g) clear the bar.**

### Three most promising among the causally-valid cuts (none clear; closest / most sign-consistent)
1. `g: >=3% stop x 11:00-12:30` -- the ONLY non-(e) combo with **same-signed net R both halves** (TRAIN +0.011 n=33, VAL +0.012 n=43) but t~0 (-0.19/+0.03), fpw 1.3/2.0 (<3 floor), magnitude far under +0.05R. Not tradeable, just not-negative.
2. `d: target2R T2(mid)` -- smallest both-halves gap (TRAIN +0.017 t-0.10, VAL -0.023 t-0.71): flat, not positive.
3. `c: arm 11:00-12:30` / `g: 1.5-3% x 11:00-12:30` -- best TRAIN net (+0.028/+0.030) but VAL reverses sign (-0.105 t-3.01 / -0.117 t-2.92) -- fails the same-sign-both-halves check (CLAUDE.md item 7); a TRAIN-only artifact.
4. Price and ATR14 buckets: no bucket exceeds |net| 0.07R on both halves; `f: price $15-30` is the one **consistently negative**
   combo (TRAIN -0.049 t-2.30, VAL -0.038 t-1.17) -- a candidate EXCLUSION filter, not an inclusion rule (not scored against the bar as stated).

## Multiplicity
30 distinct (cut,bucket) reads executed (PREREG estimated <=28 = 7x4; actual breakdown a3 b3 c4 d3 e6 f3 g8 = 30),
each read on TRAIN-H2/VAL/POOLED as one paired-halves test, not 3 separate reads. Feeds cell 1,664 alongside 1,662's re-read count.

## Bottom line
No causally-valid cut or 2-cut interaction on the floored (>=1.5%) 1,438 population clears the pre-declared bar on both
halves. This is consistent with cell 1,658's own halves-disagree finding for the unsliced buckets. The only "passing"
reads are tautological (conditioned on the trade outcome) and cannot be implemented as an ex-ante rule.

Files: research/hod_entry/1663_features.csv (5,506-row floored population with all computed features);
research/hod_entry/RESULT_1663.md (this file). Intermediate CSVs in the session scratchpad only.