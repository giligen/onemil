# HOD-break — the bottom-up loser anatomy — REPORT (2026-09-19)

The method of `research/fuckup_audit/H/METHOD.md` (loser anatomy -> filters -> profit), done on ORB
and BF and **never on HOD-break**, applied here. Owner 9/19: *"think deep, be creative, what else can
separate here? Also look at the shape of the losing days — are these DAYS we don't want to trade?
WEEKS? or specific TRADES we want to avoid. We will not leave it till you find it."*

Part 1 (the anatomy) is descriptive — no cell, no gate, no selection. Part 2's 20 cells were declared
in `PREREG.md`, **committed `06e2327` before any of them was scored.** Artifacts: `anat.py` ->
`part1.txt` - `walk.py` -> `path.csv` (8,046 signals re-walked) - `path_anat.py` -> `part1b.txt` -
`walk2.py` -> `exits.csv` (7 exit variants per signal) - `cells.py` -> `part2.txt`, `cells.csv`,
`nulls.csv` - `supp.py` -> `supp.txt`. One python process at a time, `nice -n 15`,
`ulimit -v 1500000`; `cache.db`, `bars_sip.db` opened **read-only**. No config, `orb.yaml`, systemd
unit, cron, order or cache was written. The dry run was not touched.

---

## VERDICT — **STAY DRY**

*The anatomy answers the owner's question cleanly and the answer is **TRADES, not days and not
weeks**: red weeks pass a runs test in all four base x split (z -0.13 ... +0.93), day-level dispersion
is symmetric (the worst 10 % of days carry 36-47 % of the loss and the best 10 % carry 43-55 % of the
gain), and the loser signature is a path — **MFE +0.45 R at minute 6, then a bleed to a stop at
minute 28**, against winners' +2.10 R at minute 65. That path separates by **0.77-0.88 R of gross,
same sign in both years and both bases** — the largest era-consistent separation found anywhere in
this programme.*

***And it cannot be monetised.*** *Cutting the trade at fill + 10 minutes when it has not made
+0.25 R saves **+0.012 R (TRAIN) / +0.016 R (VAL)** a fired trade, because by minute 10 the median
fired trade is already at **-0.50 R** — the rule fires 0.46 R into a 1 R loss and then frees a slot
that the book fills with a worse trade (mean hold 97 -> 73 min, 31.8 -> 34.5 trades a week, the added
trades net -$357 / -$429). **0 of 18 cell-rows pass the claim bar; 0 pass the live-exploration bar;
TEST was never opened.** The best cell in the pass (P14, level at/above the 20-day high) improves
green weeks on BOTH splits, 41.5 -> 45.3 % and 43.5 -> 47.8 %, and improves dollars from -$14,835 ->
-$102 and -$4,128 -> -$721 — and it sits **BELOW its own count-matched null mean on TRAIN**
(45.3 % vs 46.7 %), i.e. its week shape is entirely what cutting pick count from 31.8 to 10.5 a week
does arithmetically. MDE 0.051-0.088 R on the population, 0.157-0.180 R on P14's subset, against the
+0.2151 R the book must clear.*

---

# PART 1 — THE ANATOMY

Reproduction is exact before anything is described: `B0` = 1,688 / -0.088 R / 41.5 % green /
-$14,835 (TRAIN) and 820 / -0.050 / 43.5 % / -$4,128 (VAL); `B2` = 1,622 / -0.107 / 32.1 % /
-$17,346 and 706 / +0.013 / 43.5 % / +$893 — the printed digits of `hod_break` §6 and
`hod_filter_stack` §2. The independent path walk reproduces every exit: **why match 1.000000, exit
minute match 1.000000, max |d rr| 4.4e-16** on 4,941 (B0) and 7,027 (B2) rows.

## 1. Concentration — where the money actually sits

Share of the total **negative** mass carried by the worst q of units, and of the total **positive**
mass carried by the best q. A perfectly even book reads 5 / 10 / 20 %.

| base | split | unit | n | worst 5 % | **worst 10 %** | worst 20 % | best 5 % | **best 10 %** | best 20 % |
|---|---|---|---|---|---|---|---|---|---|
| B0 | TRAIN | DAY | 242 | 20.3 % | **35.7 %** | 59.4 % | 31.9 % | **52.7 %** | 80.4 % |
| B0 | TRAIN | WEEK | 52 | 21.2 % | **32.1 %** | 55.5 % | 25.1 % | 40.9 % | 70.1 % |
| B0 | TRAIN | SYMBOL | 464 | 31.1 % | 45.3 % | 65.5 % | 29.3 % | 45.4 % | 71.5 % |
| B0 | VAL | DAY | 102 | 25.3 % | **43.7 %** | 70.2 % | 25.1 % | **43.4 %** | 71.7 % |
| B0 | VAL | WEEK | 22 | 18.3 % | **36.3 %** | 55.4 % | 32.8 % | 53.8 % | 78.2 % |
| B0 | VAL | SYMBOL | 344 | 27.7 % | 41.6 % | 59.5 % | 36.5 % | 53.6 % | 78.5 % |
| B2 | TRAIN | DAY | 242 | 22.4 % | 39.0 % | 63.0 % | 34.4 % | 55.5 % | 81.0 % |
| B2 | VAL | DAY | 102 | 28.6 % | 47.0 % | 71.6 % | 23.9 % | 41.3 % | 68.6 % |

**Per trade**: worst 5 % carry 9.3-11.1 %, worst 10 % 18.1-21.9 %, worst 20 % 35.1-42.4 %.

The brief's own decision rule — *"if the worst 10 % of days carry > 50 % of losses, the answer is a
day gate"* — **is not met**: 35.7 % / 43.7 %. And the gain side concentrates just as hard
(52.7 % / 43.4 %). Day-level dispersion in this book is **symmetric**: it is variance, not a
separable class of bad days. Weeks concentrate less than days, symbols less than either, and the
trade level is close to even. Nothing here nominates a gate on its own; §2-§4 decide.

**By hour of entry** (net $, mean gross R) the two years disagree completely — TRAIN loses in the
morning and makes money after 11:00, VAL does the exact opposite:

| | 09:30-45 | 09:45-10:00 | 10:00-11:00 | 11:00-12:00 | 12:00-14:00 |
|---|---|---|---|---|---|
| B0 TRAIN | -$1,728 (-0.08) | -$7,078 (-0.10) | -$9,492 (-0.08) | **+$2,193 (+0.14)** | +$1,270 (+0.12) |
| B0 VAL | **+$1,331 (+0.19)** | **+$2,717 (+0.20)** | -$3,252 (-0.04) | -$3,139 (-0.24) | -$1,784 (-0.13) |

This is `hod_filter_stack` §4c's entry-minute reversal seen in dollars. There is no hour to avoid.

## 2. Losing DAYS — their shape

Traded days split into worst / mid / best 20 % by net $.

| | B0 TRAIN worst20 % | mid | best20 % | B0 VAL worst20 % | mid | best20 % |
|---|---|---|---|---|---|---|
| net $ total | -30,559 | -13,681 | +29,406 | -14,068 | -1,467 | +11,407 |
| net $ / day | -637 | -94 | +613 | -703 | -24 | +570 |
| **trades / day** | **8.29** | 6.18 | 8.08 | **9.40** | 7.66 | 7.85 |
| gross R / trade | -0.74 | -0.13 | +0.87 | -0.71 | +0.03 | +0.86 |
| **stop fraction** | **0.85** | 0.60 | **0.31** | **0.83** | 0.50 | **0.31** |
| median hold (min) | 36 | 55 | 55 | 35 | 71 | 48 |
| SPY 09:30->10:00 (median) | +0.022 % | +0.015 % | +0.047 % | **-0.043 %** | +0.049 % | **+0.223 %** |
| breadth by 10:00 | 101 | 89 | 94.5 | **89** | 140 | **124.5** |

Losing days ARE "everything fails together" days: 85 % of the day's trades stop, win rate 12.8 %
(TRAIN) / 14.4 % (VAL) against 64.4 % / 66.2 % on the best days, and they die **fast** (median 36 min
vs 55). The within-day clustering is real and measured, not asserted: the **win-rate intraclass
correlation is +0.0735 (TRAIN) / +0.0517 (VAL)** across 233 / 101 days with >= 3 trades — outcomes
inside a day are positively correlated beyond binomial. **This is the same fact the concurrent
`hod_preopen_regime` pass reached from the other end** — its day-clustered standard error roughly
halves the t of every day-level gate in the programme (its `D-c` +2.49/+2.81 becomes +2.00/+2.37).
Measured here as a correlation, measured there as an inflated t: HOD signals arrive in day-bursts.
Every cell in this pass is trade-level, so the correction does not apply to them; it applies to any
future day gate. *Within* the worst days the ICC is ~ 0
(-0.043 / -0.000), i.e. once you know the day is bad, the individual trades are exchangeable.

But the failures are **not** one 30-minute market pullback killing everything: worst-20 % days spread
their entries over a median of 4.0-4.5 distinct 30-minute buckets, their worst bucket holds a median
0.47-0.53 of the day's loss, and the largest single 30-minute **exit** bucket holds only 33-39 % of
the day's stops (against 59-61 % on the best days, which have only 2.7-3.0 stops to cluster). Losing
days are broad, not synchronised.

The two fields that DO mark a losing day — SPY's own 09:30->10:00 return and breadth by 10:00 — are
exactly the pair `hod_filter_stack` §7 found, and they fail the same causality trace (43 % of signals
fire before 10:00; its causal forms C1/C2 read +0.028 R on TRAIN and change sign). The causal
day-gate family is the concurrent `hod_preopen_regime` pass's 162 declared cells and is **not
duplicated here** (PREREG §2, declared-not-run).

## 3. Losing WEEKS — noise, not a regime

| base / split | weeks | green | red | runs | E[runs] | z | red runs | weekly-$ lag-1 ac | week-SIGN lag-1 ac |
|---|---|---|---|---|---|---|---|---|---|
| B0 TRAIN | 53 | 22 | 31 | 27 | 26.7 | **+0.08** | 7,4,4,3,2,2,2,1x7 | -0.118 | -0.024 |
| B0 VAL | 23 | 10 | 13 | 12 | 12.3 | **-0.13** | 6,2,2,1,1,1 | -0.433 | -0.017 |
| B2 TRAIN | 53 | 17 | 36 | 27 | 24.1 | **+0.93** | 5,4,4,4,3,3,3,2,2,2,1x4 | +0.097 | -0.136 |
| B2 VAL | 23 | 10 | 13 | 14 | 12.3 | **+0.74** | 4,3,2,1,1,1,1 | -0.100 | -0.203 |

**Red weeks do not come in runs.** The runs statistic sits within +-1 z of its independence
expectation in all four; the sign autocorrelation is negative in all four (a red week is, if
anything, slightly MORE likely to be followed by a green one); the 7-week and 6-week streaks the
reports quote are exactly what 31-of-53 and 13-of-23 coin flips produce. Worst 5 % of weeks carry
16-22 % of red-week dollars, worst 10 % 32-36 %.

**There is no week to sit out, and a "stop after N red weeks" rule would be fitting a coin.**

## 4. Losing TRADES — the path, and this is where the loss lives

Booked trades, winners = ended > 0 R.

| | B0 TRAIN L | B0 TRAIN W | B0 VAL L | B0 VAL W | B2 TRAIN L | B2 VAL L |
|---|---|---|---|---|---|---|
| **MFE (R), median** | **+0.452** | +2.104 | **+0.377** | +2.057 | +0.404 | +0.304 |
| **minutes to MFE** | **6** | 65 | **4** | 75 | 6 | 4 |
| MAE (R), median | -1.064 | -0.371 | -1.067 | -0.452 | -1.048 | -1.043 |
| hold, median min | 32 | 101 | 28 | 133 | 45 | 47 |
| **fill-bar close position** | **0.451** | 0.577 | **0.428** | 0.625 | 0.417 | 0.465 |
| breakout-bar close position | 0.760 | 0.750 | 0.746 | 0.720 | 0.700 | 0.649 |

MFE ladder (median R), losers | winners — B0 TRAIN: +1 min 0.09 | 0.13 - +3 0.18 | 0.28 -
+5 0.22 | 0.41 - **+10 0.28 | 0.66** - +15 0.34 | 0.82 - +30 0.41 | 1.26.
MAE ladder: +5 -0.37 | -0.20 - **+10 -0.55 | -0.24** - +30 -0.98 | -0.30.

Stops are 52-59 % of trades, **44-45 % of them wicks** (the stop bar closed back above the stop),
median MFE before the stop +0.25...+0.43 R reached at minute 2-5, median time to stop 25-33 min.

**This is ORB's `LIVE_LOSERS.md` signature in a different book, and it separates harder:**

| split | `MFE@10min < 0.25 R` n | gross | WR | vs | `>= 0.25 R` n | gross | WR |
|---|---|---|---|---|---|---|---|
| B0 TRAIN | 601 | **-0.524** | 20.3 % | | 1,087 | **+0.248** | 46.0 % |
| B0 VAL | 323 | **-0.521** | 21.4 % | | 497 | **+0.364** | 52.5 % |
| B2 TRAIN | 625 | -0.470 | 23.4 % | | 997 | +0.231 | 48.4 % |

0.77-0.88 R of separation, same sign in both years and both bases — bigger than anything in the 580
cells. Two more from the same family, also era-consistent:

* **fill-bar close in the bottom half of its own range**: -0.166 / -0.193 vs +0.112 / +0.227.
* **fill-bar low >= 0.25 R below entry**: -0.297 / -0.302 vs +0.056 / +0.099.

And one **falsification**: ORB's Rule M input — the **breakout** bar's close position — is
**reversed** here. Bottom-half breakout bars are the BETTER side on both splits (+0.042 / +0.080 vs
-0.047 / +0.005). ORB's Rule M does not transfer, and the anatomy said so before the cell ran.

## 5. The level itself

Pre-book signal set, gross R, kept-minus-rejected separation:

| field | B0 TRAIN | B0 VAL | B2 TRAIN | B2 VAL | reading |
|---|---|---|---|---|---|
| `touch_n >= 5` (level tested >= 5x before the break) | -0.127 | -0.095 | -0.065 | -0.175 | **worse in all four** |
| `consol_bars >= 8` (the tight zone is OLD) | -0.130 | -0.226 | -0.084 | -0.052 | **worse in all four** |
| breakout-bar vol >= HOD-bar vol | +0.111 | +0.040 | +0.043 | +0.000 | decays |
| level >= prior-day high | +0.010 | -0.032 | -0.098 | +0.051 | nothing |
| level >= prior-5-day high | +0.045 | +0.003 | +0.011 | +0.019 | nothing |
| **level >= 20-day high** | **+0.066** | **+0.078** | +0.021 | +0.055 | same sign in all four |
| level within 5 c of a round dollar | +0.001 | -0.036 | +0.002 | +0.055 | nothing |

Distribution: `touch_n` median 2 (p90 5-7); `consol_bars` median 9-21 (p90 55-71); 64 % / 59 % of
levels are above the prior day's high, 35 % above the 5-day high.

**The brief's candidate (d) is inverted by the data.** It asked for *consolidation duration >= N* and
*tests-before-break >= 2*; both are the WRONG side. A HOD that has been sat on for 8+ bars and tested
5+ times is the worse break, in every base and every split — the same direction as `hod_break` §5's
finding that the shipped `K5 / 4 %` proximity rule is net-negative (-0.123 R, t -2.91).

## 6. The stock's context

| field | B0 TRAIN | B0 VAL | B2 TRAIN | B2 VAL |
|---|---|---|---|---|
| RE-break (`n_break > 0`) vs first break | **+0.113** | -0.002 | **+0.236** | +0.067 |
| leveraged / inverse wrapper vs common | **+0.066** | **+0.124** | +0.044 | **+0.149** |
| `dist_open / ATR14` quintiles | non-monotone, TRAIN Q4 +0.077 | VAL Q4 **-0.149** | flat | flat |
| level above session VWAP | **degenerate — 100 % of signals** (the HOD is always above VWAP) | | | |

* **(c) ATR-normalised admission is ruled out**: the ladder is non-monotone and its top quintile
  changes sign between the years. The fixed +5 % floor is not the problem.
* The **wrapper** split is a binary the 580-cell tercile screen skipped as "degenerate" and it has the
  same sign in all four cells (+0.044 ... +0.149) — but the kept side's gross is +0.025 ... +0.107,
  nowhere near the +0.215 R the book needs, and wrappers are already 44 % of the signals.
* **`coh_by_t` was "degenerate" because it is 10.6 % covered and 94 % of the covered rows are 0** —
  there is no same-morning anchor cohort to confirm against in this population. **(f)
  family / sector confirmation is ruled out**, not by a null but by the field not existing.

## The Part 1 sentence

> **The losses live in the first ten minutes of the trade — not in a day, not in a week, not in a
> symbol.** Days cluster (ICC +0.05...+0.07) but their dispersion is symmetric and their marker is not
> causal; weeks are a coin; the loser is a trade that goes +0.45 R in six minutes, rolls over, and is
> stopped at minute 28.

---

# PART 2 — THE CELLS

20 declared in `PREREG.md` (committed `06e2327` **before scoring**). `$` at the live $100 risk.
**Reference rows reproduce the two prior reports to the dollar.**

| cell | base | TRAIN n | /wk | gross | net | t | **green %** | rs | **$** | VAL n | /wk | gross | net | **green %** | rs | **$** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **B0 shipped (ref)** | B0 | 1,688 | 31.8 | -0.027 | -0.088 | -2.64 | **41.5** | 7 | **-14,835** | 820 | 35.7 | +0.016 | -0.050 | **43.5** | 6 | **-4,128** |
| **B2 shipped (ref)** | B2 | 1,622 | 30.6 | -0.039 | -0.107 | -3.42 | 32.1 | 5 | -17,346 | 706 | 30.7 | +0.083 | +0.013 | 43.5 | 4 | +893 |
| P1 T10/0.25 | B0 | 1,826 | 34.5 | -0.017 | -0.080 | -2.76 | 34.0 | 6 | -14,596 | 942 | 41.0 | +0.009 | -0.059 | 30.4 | 5 | -5,541 |
| P2 T10/0.25 | B2 | 2,008 | 37.9 | -0.031 | -0.101 | -3.94 | 30.2 | 8 | -20,189 | 942 | 41.0 | +0.052 | -0.021 | 34.8 | 5 | -1,973 |
| P3 T5/0.25 | B0 | 1,925 | 36.3 | -0.025 | -0.090 | -3.49 | 28.3 | 7 | -17,290 | 1,006 | 43.7 | +0.042 | -0.029 | 39.1 | 4 | -2,874 |
| P4 Rule D (ORB verbatim) | B0 | 1,688 | 31.8 | -0.022 | -0.083 | -2.49 | 41.5 | 7 | -13,969 | 821 | 35.7 | +0.016 | -0.050 | 43.5 | 6 | -4,124 |
| P5 Rule D | B2 | 1,622 | 30.6 | -0.036 | -0.104 | -3.33 | 34.0 | 5 | -16,874 | 707 | 30.7 | +0.085 | +0.015 | 43.5 | 4 | +1,067 |
| P6 fill-bar shape exit | B0 | 1,965 | 37.1 | -0.046 | -0.112 | -4.88 | 28.3 | 7 | -22,071 | 1,013 | 44.0 | +0.048 | -0.023 | 39.1 | 5 | -2,282 |
| P7 fill-bar shape exit | B2 | 2,166 | 40.9 | -0.007 | -0.081 | -3.91 | 28.3 | 5 | -17,536 | 1,020 | 44.3 | +0.050 | -0.028 | 43.5 | 3 | -2,852 |
| P8 BE at +0.5 R | B0 | 1,904 | 35.9 | -0.025 | -0.091 | -3.78 | 34.0 | 7 | -17,291 | 978 | 42.5 | -0.054 | -0.126 | 26.1 | 5 | **-12,317** |
| P9 T10 + BE | B0 | 1,982 | 37.4 | -0.028 | -0.095 | -4.60 | 34.0 | 9 | -18,919 | 1,054 | 45.8 | -0.053 | -0.126 | 21.7 | 9 | **-13,318** |
| P10 Rule M veto (falsification) | B0 | 1,377 | 26.0 | -0.050 | -0.111 | -3.02 | 35.8 | 6 | -15,238 | 718 | 31.2 | +0.027 | -0.037 | 39.1 | 6 | -2,681 |
| P11 `touch_n < 5` | B0 | 1,444 | 27.2 | -0.010 | -0.070 | -1.95 | 41.5 | 7 | -10,115 | 728 | 31.7 | +0.060 | -0.005 | 30.4 | 6 | -366 |
| P12 `touch_n < 5` | B2 | 1,496 | 28.2 | -0.031 | -0.099 | -3.07 | 28.3 | 8 | -14,828 | 681 | 29.6 | +0.103 | +0.034 | **56.5** | 2 | **+2,296** |
| P13 `consol_bars < 8` | B0 | 297 | **5.6** | -0.038 | -0.104 | -1.39 | 35.8 | 4 | -3,099 | 219 | **9.5** | **+0.207** | +0.136 | 52.2 | 5 | **+2,977** |
| **P14 level >= 20-day high** | B0 | 558 | 10.5 | **+0.057** | -0.002 | -0.03 | **45.3** | 5 | **-102** | 377 | 16.4 | +0.046 | -0.019 | **47.8** | 3 | **-721** |
| P15 first break only | B0 | 1,468 | 27.7 | -0.021 | -0.080 | -2.26 | 41.5 | 5 | -11,817 | 738 | 32.1 | +0.024 | -0.040 | 30.4 | 6 | -2,929 |
| P15c re-breaks only (compl.) | B0 | 537 | 10.1 | -0.038 | -0.105 | -1.76 | 37.7 | 5 | -5,632 | 371 | 16.1 | -0.023 | -0.094 | 43.5 | 4 | -3,489 |
| P16 kill after 3 stops | B0 | 1,414 | 26.7 | -0.029 | -0.090 | -2.50 | 34.0 | 7 | -12,777 | 681 | 29.6 | +0.045 | -0.020 | 34.8 | 6 | -1,359 |
| P17 kill after 2 stops | B0 | 1,200 | 22.6 | -0.032 | -0.092 | -2.36 | 35.8 | 7 | -11,093 | 597 | 26.0 | +0.063 | -0.002 | 30.4 | 6 | -134 |
| **P18 = P14 stack** | B0 | identical to P14 — the selector's pick | | | | | | | | | | | | | | |
| **P19 = P14 on B2** | B2 | 774 | 14.6 | -0.022 | -0.091 | -2.04 | 32.1 | **12** | -7,031 | 446 | 19.4 | +0.009 | -0.062 | 26.1 | 4 | -2,755 |

### The selector (PREREG §3, no criterion added after the fact)

Eligible (>= 10 trades/week on both splits), ranked by `min(green % TRAIN, green % VAL)`:
**P14 45.3 %** > P4 41.5 % > P10 35.8 % > P16 34.0 % > P5 34.0 % > P11/P17/P15/P1 30.4 % > P2 30.2 %
> P12/P3/P7/P6 28.3 % > P8 26.1 % > P9 21.7 %. **P14 selected.** P13 (`consol_bars < 8`, the other
cell with positive VAL dollars) fails the frequency floor at 5.6 trades a week on TRAIN.

### 1. Why the strongest separation in the programme buys nothing (`supp.txt` S1/S2)

The `MFE@10min < 0.25 R` split is worth 0.77-0.88 R of gross. Run as a rule, it is worth **+0.012 R
(TRAIN) / +0.016 R (VAL)** a fired trade:

| | fires on | gross under the shipped exit | gross at the fill+10 close | **saved** | median R when it fires |
|---|---|---|---|---|---|
| B0 TRAIN | 1,117 of 3,166 (35 %) | -0.554 | -0.542 | **+0.012** | **-0.500** |
| B0 VAL | 665 of 1,775 (37 %) | -0.536 | -0.521 | **+0.016** | **-0.463** |

**The bleed is already booked by minute 10.** The stop sits at -1.0 R and the rule fires 0.46-0.48 R
into it. It is a *classifier of trades that have already lost*, not a lever. And it then frees a slot:
mean hold 97 -> 73 min (TRAIN), 115 -> 82 (VAL), the book grows 31.8 -> 34.5 trades a week, and the
355 trades it adds that the shipped book never took are **-$357 (TRAIN) / -$429 (VAL)** while the 95
it drops were **+$998 / +$257**. Green weeks fall 41.5 -> 34.0 and 43.5 -> 30.4.

**This is the difference from ORB, stated precisely.** ORB's losers reached MFE 0.35 R at minute 3
and bled for **38 minutes** to the stop — 35 minutes of room to save. HOD's losers reach MFE 0.45 R
at minute 6 and are at **-0.50 R by minute 10**, stopped at minute 28. Same shape, a third of the
runway, and a stop that is 1 R away instead of ORB's wider range-low. The ORB time stop does not
transfer, and the reason is measurable, not a hunch.

Pre-book, no slots — every exit variant over the whole signal set (mean gross R):

| base / split | base | t10 | t5 | ruleD | shape | be | t10+be |
|---|---|---|---|---|---|---|---|
| B0 TRAIN | +0.007 | +0.011 | -0.006 | +0.011 | +0.007 | **-0.032** | -0.028 |
| B0 VAL | +0.001 | +0.007 | +0.006 | +0.004 | +0.015 | **-0.059** | -0.047 |
| B2 TRAIN | -0.002 | +0.002 | -0.001 | -0.000 | +0.006 | -0.009 | -0.003 |
| B2 VAL | +0.015 | -0.002 | +0.002 | +0.017 | **+0.026** | **-0.034** | -0.039 |

Every exit rule in the declared set moves the book by less than +-0.03 R before slots, and the
breakeven move — the one the anatomy nominated hardest, since losers DO go +0.45 R favourable first —
is the **worst** thing in the table. It converts winners into scratches faster than it saves losers:
VAL -$12,317 against the shipped -$4,128. `green_weeks`' verdict on this book ("the shipped exit is
rank 1 of 12; the average loss is -1.03 R in every cell") is confirmed by an independent route.

### 2. Rule D and Rule M — ORB's touchgo, measured on HOD for the first time

* **Rule D (ORB thresholds verbatim, 0.75 R revert -> exit at -0.5 R) is very nearly inert**: it fires
  on 3.5 % of B0 fills and moves TRAIN from -$14,835 to -$13,969 and VAL from -$4,128 to -$4,124,
  with green weeks unchanged at 41.5 / 43.5. On B2 it is the best-looking exit in the pass
  (+$1,067 VAL, TRAIN green 32.1 -> 34.0) and still -$16,874 on TRAIN. The reason it is inert is
  structural and was known from the anatomy: HOD fills at the **open of the bar after the break**,
  so a bar that reverts 0.75 R inside that same minute is rare, and the spec's walk would have
  stopped most of those trades at the next bar anyway.
* **Rule M is refuted, in the direction the anatomy predicted.** Vetoing signals whose breakout bar
  closed in the bottom half of its range costs money on both splits: -$14,835 -> -$15,238 (TRAIN) and
  green weeks 41.5 -> 35.8 / 43.5 -> 39.1. On HOD the weak-looking breakout bar is the better trade.
  **Do not port ORB's Rule M to HOD-break.**

### 3. The kill switch — it works on dollars and it costs week shape

P16 (stop after 3 consecutive closed stops in a session) cuts the book 31.8 -> 26.7 trades a week and
the loss -$14,835 -> -$12,777 (TRAIN), -$4,128 -> -$1,359 (VAL); P17 (after 2) -> -$11,093 / -$134.
Both *worsen* the primary metric (green weeks 41.5 -> 34.0 / 35.8, 43.5 -> 34.8 / 30.4). This is the
same failure mode `hod_break` §7's `C7` showed: a veto on a bad bucket must improve dollars
arithmetically and buys no week shape. It is the honest confirmation of the ICC finding — losing days
ARE clusterable in-day — and it is still not a lever, because what the switch removes is the tail of
a day already lost, not the day's first three trades.

### 4. P14, the best cell, against its own null

| cell | split | observed green % | null mean | [p5, p95] | outside? |
|---|---|---|---|---|---|
| **P18 = P14** | TRAIN | **45.3** | **46.7** | [39.6, 54.7] | inside — and **below the null MEAN** |
| P18 = P14 | VAL | 47.8 | 44.6 | [34.8, 56.5] | inside |
| P19 = P14 on B2 | TRAIN | 32.1 | 37.2 | [30.2, 45.3] | inside |
| P19 = P14 on B2 | VAL | 26.1 | 38.6 | [30.2, 47.8] | **below** |
| B0 shipped | TRAIN / VAL | 41.5 / 43.5 | 34.7 / 38.7 | [28.3, 41.5] / [30.4, 47.8] | inside (at TRAIN p95) |
| B2 shipped | TRAIN / VAL | 32.1 / 43.5 | 30.8 / 49.5 | [24.5, 37.7] / [39.1, 60.9] | inside |

P14 improves green weeks on both splits and improves dollars on both splits (-$14,835 -> -$102,
-$4,128 -> -$721) and it is **still the same answer this programme has now reached four independent
times**: at 10.5 trades a week instead of 31.8, a book of this mean and variance produces 46.7 %
green weeks *by arithmetic*, and P14 delivers 45.3 %. It does not transfer to the corrected base
(B2: 32.1 % / 26.1 % green, -$7,031 / -$2,755, a 12-week red streak). Its TRAIN gross of **+0.057 R**
is the best in the pass and a quarter of the +0.2151 R the book must clear.

**Ex-tail** (reported, never a rejection reason): P18 TRAIN net -0.002 -> ex-top-1 % -0.023 ->
ex-top-5 % -0.107; VAL -0.019 -> -0.041 -> -0.126. No edge to concentrate.

---

## BOTH BARS

**Claim bar G1 — 0 of 18 cell-rows pass** (TRAIN net R > 0, t >= 2.0, >= 10 trades/week, TRAIN gross
>= +0.25 R). The best TRAIN net R in the pass is **-0.0018** (P14 / B0) at **t -0.03**; the best TRAIN
gross is **+0.0572**. G2 was never evaluated and, per PREREG §3, **TEST was never opened**
(`FREEZE.md`; no TEST-dated bar was read by either walk).

**Live-exploration bar — 0 cells.** It requires positive dollars on BOTH splits at >= 10 trades/week.
The two cells with positive VAL dollars are `P12` (`touch_n < 5` on B2, +$2,296 at 56.5 % green — and
-$14,828 at 28.3 % green on TRAIN) and `P13` (`consol_bars < 8`, +$2,977 at 52.2 % green — and
-$3,099 on TRAIN at **5.6 trades a week**, below the frequency floor).

**MDE — this is a powered rejection, not an underpowered null.** Per trade, 80 % power:
**0.066 R** (B0 TRAIN pre-book, n 3,166), 0.088 R (B0 VAL), **0.051 R** (B2 TRAIN, n 4,575),
0.071 R (B2 VAL); on the selected subsets 0.157 / 0.180 R (P14), 0.073 / 0.100 R (P11),
0.143 / 0.253 R (P13). The book needs **+0.2151 R**. On the green-week share the resolution is
**+-19.0 pp** over 53 TRAIN weeks and **+-28.9 pp** over 23 VAL weeks — which is why no cell is
separable from `B0` on the primary metric, and that limit is part of the answer.

**Multiplicity.** 20 declared decision cells (18 scored rows + the 2 stack rows, one of which is
identical to P14) x 2 splits, plus 1 reported complement (`P15c`) and Part 1's ~20 descriptive tables
(not cells). **Programme cumulative: 764 (through `hod_preopen_regime`, its own LOG line) + 20 =
784.** Expected largest |t| under a pure null over 20 cells x 2 splits ~ 2.6-2.8.
Moot in the favourable direction: **no cell has a positive TRAIN t at all** — the maximum is -0.03.

**Declared and NOT run, with the anatomy line that ruled each out** (PREREG §2):

| candidate | ruled out by |
|---|---|
| **(c) ATR-normalised admission** `dist_open >= k*ATR` | Part 1 §6: quintiles non-monotone; VAL top quintile -0.149 vs TRAIN +0.077 |
| **(f) family / sector confirmation** | Part 1 §6: `coh_by_t` 10.6 % covered, 94 % of covered rows = 0 — the cohort does not exist in this population |
| **(g) day gate on the losing-day signature** | Part 1 §2: the signature (SPY 09:30->10:00, breadth by 10:00) is not known before 10:00 and 43 % of signals fire earlier; `hod_filter_stack` §7 already scored the causal forms (C1 +0.028 R TRAIN, C2 sign-flipping); the causal day-gate family is `hod_preopen_regime`'s 162 cells — not duplicated |
| **week gate** | Part 1 §3: runs test z -0.13 ... +0.93 in all four base x split; red weeks are a coin |

---

## VERDICT — **STAY DRY.** No `HodBreakParams` change, no touchgo diff.

`config.yaml hod_break` stays exactly as the owner set it (`enabled: true, dry_run: true`). There is
no SHIP-TO-DRY diff to write: the dry run already runs the shipped rule, and not one of the 20 cells
earns a change to it. The MDE above is the honest statement of what this pass could have seen.

**What this pass adds that the 580 cells did not**, and that should not be re-run:

1. The owner's three-way question is **answered**: not days (symmetric dispersion, non-causal
   marker), not weeks (a coin, by a runs test), but **trades** — and the trade-level signature is the
   largest era-consistent separation in the programme *and* the least monetisable, for a reason that
   is now measured to the hundredth of an R (Part 2 §1).
2. **ORB's touchgo is settled on HOD**: Rule M is refuted (wrong sign), Rule D is structurally inert
   (HOD fills at the next bar's open), the 10-minute time stop saves +0.012 R. Do not port them.
3. **Two level-quality rules point the OPPOSITE way to the brief's hypothesis** — a HOD that is old
   (`consol_bars >= 8`) and much-tested (`touch_n >= 5`) is the worse break, in all four base x split.
   That is the same direction as `hod_break` §5's wrong-side consolidation finding, reached
   independently, and it is the one thing in this pass worth carrying into any future
   pre-registration.
4. **The breakeven move is the worst idea the anatomy suggested** and would have been shipped on the
   strength of "losers go +0.45 R favourable first". It costs -$8,189 on VAL against the shipped book.
   That is what Part 2 exists for.

**What would change the verdict** — unchanged from the two prior passes, with one addition:

1. A causal, era-consistent **GROSS** separator worth +0.25 R. The best in this pass is +0.057 R.
2. **The stop, not the exit after it.** Every cell here kept the consolidation-low stop and a ~40 %
   win rate against a 2:1 payoff; the only exit that changes the loss distribution (BE) makes it
   worse. `hod_break`'s "what would change this verdict" named re-shaping the STOP and it remains
   untested.
3. **New**: the level-quality direction — *fresh, untested* HODs (`consol_bars < 8` reads +0.207 R
   gross on VAL, +$2,977, 52.2 % green) is the only positive-dollar direction in this pass, and it is
   dead on arrival at **5.6 trades a week**. A pre-registration that widened the candidate stream
   enough to make a fresh-HOD rule trade >= 10 times a week — i.e. a different admission rule, not a
   different filter on this one — is the only live direction the anatomy produced.

**Recommended action: NONE.**
