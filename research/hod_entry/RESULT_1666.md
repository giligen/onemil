# RESULT 1,666 -- independent rebuild of the no-withdrawal pyramid (cell 1,488) at measured cost

Built from PREREG_1666.md prose alone; no 1487/1488/1662 file opened.

## Coverage
- Joined population (fills_1658 x causal fill-status, inner join on day+symbol): 9911 rows
- fills_1658 rows with no matching causal fill row: 0
- Duplicate (day,symbol) keys dropped: causal=0, fills_1658=0
- Simulated to a determinate exit (both books): 7983/9911 (80.5%); excluded for insufficient bar coverage: 1928 (19.5%)
- Coverage by half: TRAIN-H2 79.6% (n=4398), VAL 81.3% (n=5513) -- gap 1.7pp. By r_pct bucket: <1.5% 79.4%, >=1.5% 81.4% -- gap 2.0pp. Both gaps clear the <=5pp availability-rail bar; TRAIN-H2's raw coverage sits marginally under the >=80% floor, VAL clears it -- flagged, not fatal, since the gap between groups (the thing that could bias a comparison) is small.

## Own-book (pyramid) R, by half
Primary population: r_pct >= 1.5%. Unfloored (all r_pct) reported beside each half.

| half | pop | n | mean R | iid t | day-clust t | ex-top5% | fills/wk | MDE |
|---|---|---|---|---|---|---|---|---|
| TRAIN-H2 | >=1.5% | 1898 | -0.136 | -4.63 | -4.07 | -0.250 | 70.3 | 0.082 |
| TRAIN-H2 | unfloored | 3499 | -0.204 | -9.50 | -8.14 | -0.322 | 129.6 | 0.060 |
| VAL | >=1.5% | 2586 | -0.108 | -4.30 | -3.97 | -0.220 | 117.5 | 0.070 |
| VAL | unfloored | 4484 | -0.171 | -9.02 | -6.86 | -0.287 | 203.8 | 0.053 |

## Paired base (1x, standard rule, same fills), by half
| half | pop | n | mean R | iid t | day-clust t | ex-top5% |
|---|---|---|---|---|---|---|
| TRAIN-H2 | >=1.5% | 1898 | -0.092 | -3.04 | -3.10 | -0.205 |
| TRAIN-H2 | unfloored | 3499 | -0.162 | -7.23 | -6.67 | -0.278 |
| VAL | >=1.5% | 2586 | -0.064 | -2.44 | -2.71 | -0.174 |
| VAL | unfloored | 4484 | -0.124 | -6.25 | -5.22 | -0.238 |

## Paired delta_R (pyramid - base), by half -- the pass-bar quantity
| half | pop | n | mean dR | iid t | day-clust t | ex-top5% dR | MDE |
|---|---|---|---|---|---|---|---|
| TRAIN-H2 | >=1.5% | 1898 | -0.043 | -5.80 | -4.85 | -0.059 | 0.021 |
| TRAIN-H2 | unfloored | 3499 | -0.042 | -7.50 | -5.36 | -0.061 | 0.016 |
| VAL | >=1.5% | 2586 | -0.044 | -6.70 | -5.09 | -0.064 | 0.018 |
| VAL | unfloored | 4484 | -0.047 | -9.15 | -7.80 | -0.069 | 0.014 |

## Add mechanics, exposure, worst day (primary population, r_pct >= 1.5%)
| half | add share | worst day (sum pyr_R) |
|---|---|---|
| TRAIN-H2 | 0.079 | -21.699 |
| VAL | 0.084 | -32.001 |

Every add multiplies the live position from 1/3 to full size (3x the base leg); no per-trade dollar notional is in this data, so exposure is reported as this fixed multiplier, not a dollar figure.

## Pass-bar verdict
- TRAIN-H2: dR mean=-0.043, iid t=-5.80, day-clust t=-4.85, ex-top5%=-0.059, fills/wk=70.3. 1487-bar FAIL; 1662-bar FAIL
- VAL: dR mean=-0.044, iid t=-6.70, day-clust t=-5.09, ex-top5%=-0.064, fills/wk=117.5. VAL t=-6.70 (>=2.5? False), VAL book mean=-0.108 (>=0.10? False) 1487-bar FAIL; 1662-bar FAIL

**Combined: PREREG_1487 bar FAILS on both halves; PREREG_1662 synthesis bar FAILS on both halves.**

Row-by-row agreement against cell 1,662's per-fill file (withdrawal flag, sign of pyr_R, >=95% required before this reaches the owner) is a third-party step outside this build's scope -- this agent never opened that file, per the independence rule.

## Adequacy note
MDE on paired dR at n: VAL=0.018 R, TRAIN-H2=0.021 R (two-sided, 80% power, observed SD). Compare against the +0.05 R pass-bar threshold: an MDE close to or above 0.05 R means a null here is underpowered, not evidence of no effect -- only a point estimate clearly above the MDE (or a pass) should be read as a finding.

Programme count on the HOD line: >= 1,700 (this cell = 1,666; per PREREG_1666 pass bar note).
