# RESULT 1,778 - long gap-and-go survivors at 10:30 (p1030 >= open), MOC exit

Verdict: **FAIL** (primary ALL/MOC/VAL: n 194, net -14.6 bp, day-t -0.23; bar needs >= +50 bp and t >= 2). Frequency table below is the deliverable.

## 1. FREQUENCY (executable population = quote-fetched top-10/day subset with ask1030, nq>=1; zero-filled over all population days/weeks of the split)

| cut | split | n | names/day med | names/day mean | days with >=1 | names/wk p10 | names/wk med | wks>=3 | wks>=5 | weeks |
|---|---|---|---|---|---|---|---|---|---|---|
| ALL | TRAIN | 243 | 2.0 | 1.90 | 105/128 | 5.0 | 8.0 | 100% | 96% | 27 |
| ALL | VAL | 194 | 2.0 | 1.92 | 88/101 | 5.0 | 8.5 | 95% | 91% | 22 |
| L1_T1 | TRAIN | 81 | 0.0 | 0.63 | 59/128 | 1.0 | 3.0 | 56% | 19% | 27 |
| L1_T1 | VAL | 51 | 0.0 | 0.50 | 38/101 | 0.0 | 2.0 | 36% | 14% | 22 |
| L1_T2 | TRAIN | 80 | 0.0 | 0.62 | 53/128 | 1.0 | 2.0 | 48% | 19% | 27 |
| L1_T2 | VAL | 72 | 1.0 | 0.71 | 51/101 | 1.0 | 3.0 | 64% | 18% | 22 |
| L1_T3 | TRAIN | 81 | 0.0 | 0.63 | 59/128 | 1.0 | 3.0 | 59% | 19% | 27 |
| L1_T3 | VAL | 70 | 1.0 | 0.69 | 55/101 | 1.0 | 3.0 | 68% | 23% | 22 |
| HOLD5 | TRAIN | 84 | 0.0 | 0.66 | 55/128 | 0.6 | 3.0 | 52% | 22% | 27 |
| HOLD5 | VAL | 78 | 1.0 | 0.77 | 56/101 | 1.0 | 4.0 | 68% | 32% | 22 |
| TIER_3-5 | TRAIN | 75 | 0.0 | 0.59 | 53/128 | 0.6 | 3.0 | 56% | 22% | 27 |
| TIER_3-5 | VAL | 58 | 0.0 | 0.57 | 42/101 | 0.1 | 2.0 | 45% | 23% | 22 |
| TIER_5-20 | TRAIN | 152 | 1.0 | 1.19 | 89/128 | 3.0 | 5.0 | 100% | 63% | 27 |
| TIER_5-20 | VAL | 122 | 1.0 | 1.21 | 73/101 | 2.0 | 6.0 | 82% | 64% | 22 |
| TIER_20-30 | TRAIN | 16 | 0.0 | 0.12 | 14/128 | 0.0 | 0.0 | 7% | 0% | 27 |
| TIER_20-30 | VAL | 14 | 0.0 | 0.14 | 14/101 | 0.0 | 0.0 | 5% | 0% | 22 |
| ABOVE_VWAP | TRAIN | 181 | 1.0 | 1.41 | 89/128 | 3.0 | 7.0 | 93% | 74% | 27 |
| ABOVE_VWAP | VAL | 129 | 1.0 | 1.28 | 71/101 | 2.0 | 5.5 | 82% | 64% | 22 |

ALL survivors: median 8-8.5 names/week, 91-96 % of weeks >= 5, so the >= 3-5/wk frequency target is met by the unfiltered cut; sub-cuts reach a >= 3/wk median only for TIER_5-20, ABOVE_VWAP, HOLD5 (VAL), L1_T2/T3 (VAL, borderline) - see table.

## 2. Reads (MOC primary; net = gross - 4 bp fees; gross = close/(ask1030*1.001)-1; R = $3,000 x 5 % = $150)

| cut | split | n | net bp | median | win | day-t | ex-top5 | worst-5% | wk P10 R | green wk | TR H1/H2 | pl(i) p95 | pl(ii) p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | TRAIN | 243 | -36.0 | -14 | 49% | -0.69 | -127 | -2283 | -4.65 | 41% | -39/-34 | -18 | -168 |
| ALL | VAL | 194 | -14.6 | -68 | 45% | -0.23 | -139 | -1422 | -4.93 | 45% | - | -95 | -91 |
| L1_T1 | TRAIN | 81 | -18.1 | -14 | 48% | -0.44 | -78 | -868 | -1.83 | 48% | +54/-76 | -38 | -290 |
| L1_T1 | VAL | 51 | -25.1 | -38 | 45% | -0.18 | -163 | -1464 | -2.36 | 33% | - | -73 | +5 |
| L1_T2 | TRAIN | 80 | +112.6 | +35 | 55% | +1.96 | +36 | -774 | -1.43 | 68% | +65/+156 | -23 | -251 |
| L1_T2 | VAL | 72 | +43.0 | -51 | 49% | +0.39 | -108 | -1302 | -1.93 | 60% | - | -98 | -325 |
| L1_T3 | TRAIN | 81 | -160.1 | -88 | 43% | -1.18 | -282 | -3577 | -5.58 | 48% | -161/-159 | +143 | +28 |
| L1_T3 | VAL | 70 | -51.2 | -74 | 41% | -0.67 | -150 | -1373 | -2.84 | 33% | - | -22 | +204 |
| HOLD5 | TRAIN | 84 | -211.8 | -105 | 45% | -1.71 | -316 | -3853 | -6.15 | 46% | -436/-35 | +2 | -132 |
| HOLD5 | VAL | 78 | -15.8 | -69 | 45% | -0.16 | -142 | -1572 | -2.69 | 40% | - | -39 | -144 |
| TIER_3-5 | TRAIN | 75 | -74.2 | -60 | 44% | -0.52 | -174 | -3375 | -4.34 | 50% | -80/-69 | +42 | +32 |
| TIER_3-5 | VAL | 58 | -174.9 | -268 | 33% | -1.36 | -327 | -1772 | -4.36 | 32% | - | -120 | +51 |
| TIER_5-20 | TRAIN | 152 | -7.3 | +23 | 51% | -0.15 | -90 | -1721 | -2.50 | 33% | +40/-46 | -30 | -298 |
| TIER_5-20 | VAL | 122 | +69.7 | +4 | 52% | +0.98 | -55 | -1166 | -2.91 | 57% | - | -47 | -198 |
| TIER_20-30 | TRAIN | 16 | -129.9 | -97 | 44% | -1.04 | -170 | -1849 | -1.61 | 50% | -154/-90 | +483 | +73 |
| TIER_20-30 | VAL | 14 | -85.6 | -135 | 36% | -1.28 | -126 | -377 | -0.74 | 22% | - | +138 | +155 |
| ABOVE_VWAP | TRAIN | 181 | -42.4 | -36 | 48% | -0.67 | -135 | -2333 | -3.92 | 48% | -90/-3 | +2 | -244 |
| ABOVE_VWAP | VAL | 129 | -51.0 | -70 | 42% | -0.71 | -166 | -1346 | -3.11 | 36% | - | -89 | +131 |

## 3. Stop variant (stop = p1030 - 5 %, fill stop - 35 bp, else MOC; bars coverage 100 %)

| cut | split | n | net bp | day-t | ex-top5 | worst-5% | wk P10 R |
|---|---|---|---|---|---|---|---|
| ALL | TRAIN | 243 | -55.8 | -2.01 | -131 | -650 | -4.01 |
| ALL | VAL | 194 | -20.0 | -0.37 | -144 | -668 | -4.64 |
| L1_T1 | TRAIN | 81 | -44.3 | -1.06 | -103 | -617 | -2.01 |
| L1_T1 | VAL | 51 | -17.6 | -0.15 | -155 | -708 | -2.23 |
| L1_T2 | TRAIN | 80 | +25.3 | +0.54 | -42 | -585 | -1.74 |
| L1_T2 | VAL | 72 | +46.5 | +0.45 | -104 | -609 | -1.80 |
| L1_T3 | TRAIN | 81 | -134.2 | -2.26 | -236 | -608 | -3.21 |
| L1_T3 | VAL | 70 | -77.7 | -1.19 | -178 | -576 | -3.02 |
| HOLD5 | TRAIN | 84 | -117.2 | -2.18 | -193 | -737 | -2.91 |
| HOLD5 | VAL | 78 | -37.3 | -0.42 | -164 | -619 | -3.37 |
| TIER_3-5 | TRAIN | 75 | -72.7 | -1.08 | -164 | -617 | -2.99 |
| TIER_3-5 | VAL | 58 | -115.0 | -1.07 | -264 | -702 | -4.70 |
| TIER_5-20 | TRAIN | 152 | -47.7 | -1.22 | -114 | -666 | -3.72 |
| TIER_5-20 | VAL | 122 | +38.4 | +0.54 | -88 | -652 | -2.79 |
| TIER_20-30 | TRAIN | 16 | -54.0 | -0.81 | -89 | -647 | -1.30 |
| TIER_20-30 | VAL | 14 | -135.0 | -1.65 | -179 | -554 | -1.10 |
| ABOVE_VWAP | TRAIN | 181 | -54.0 | -1.71 | -133 | -666 | -3.37 |
| ABOVE_VWAP | VAL | 129 | -41.0 | -0.67 | -156 | -642 | -4.40 |

## 4. Pass bar on primary (ALL, MOC, VAL)

- net >= +50 bp: FAIL (-14.6). day-t >= 2: FAIL (-0.23). ex-top-5 % > 0: FAIL (-139). worst-5 % >= -12 %: FAIL (-1,422 bp). weekly P10 >= -2 R: FAIL (-4.9).
- TRAIN halves same sign: both negative (-39/-34) PASS. Beats placebo p95s: pl(i) p95 -95, pl(ii) p95 -91, ALL beats both but both p95s are negative, i.e. the survivors are no worse than a bad pool, not an edge.
- names/week median >= 3: PASS (8.5). Overall: FAIL. No sub-cut qualifies (ALL line is < 0, so the sub-cut clause is not met).

## 5. Caveats

- Population = 1,774 events with a measured ask1030 and nq1030 >= 1 and gap <= 200 %: 1,172 of 2,269 rows (TRAIN 677, VAL 495); survivors 437 (TRAIN 243, VAL 194). This is the top-10/day quote-fetched subset, NOT the whole gapper population.
- Bars: all 1,172 events join to `1762_cache/bars` (100 %), VWAP and stop variants are NOT void; bars_sip.db was never needed. VWAP = typical price (h+l+c)/3 over 09:30-10:29 ET. L1 coverage 99.8 %.
- Tier '1-5' in the parquet is labelled TIER_3-5 here (population is $3-30). L1 tercile cutoffs from TRAIN survivors: 1.55 M / 12.3 M $.
- Weekly P10 is over weeks having >= 1 name; <= 10 names/day seeded sample (1773 weekly_R). Day-t clustered per 1773 tclust. Tails are extreme (worst-5 % about -14 %): +/- 1 name moves the mean by tens of bp.
- Placebo (i) draws from the same-day non-survivors (which have lower gross by construction); (ii) same symbol within +/- 5 calendar days, 1 draw/event. 21 cuts x 2 exits x 2 splits were scored, so isolated positives (below) are multiplicity-exposed.
- 10 bp chase slip is on top of the measured ask; stop slip 35 bp per cell 1,443; the stop variant is worse than MOC in TRAIN because the gap names often dip 5 % then recover.

CHILDREN:
1. L1_T2 (middle pre-market $vol tercile) +113 bp TRAIN (t 1.96) and +43 bp VAL (t 0.39) is the only same-signed cut, but L1_T1/T3 are negative (non-monotone), so it reads as noise or a nonlinear liquidity sweet spot; test on a fresh window / PIT universe before any claim.
2. TIER_5-20 VAL +70 bp (t 0.98) but TRAIN -7 bp: regime or one-window; check whether the survivor edge exists only when SPY is up on the day (cond. on market direction at 10:30).
3. HOLD5 (>= 5 % above the open) is the worst TRAIN cut (-212 bp, worst-5 % -3,853): names extended at 10:30 revert into the close; a SHORT of HOLD5 survivors with the 1776R cost stack is the mirror to test, under its own PREREG.
