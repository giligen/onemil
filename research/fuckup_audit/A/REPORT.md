# Stage A — the corrected cost contract, the re-gate, time bands, intraday market state

Executed 2026-09-16 per `research/fuckup_audit/PLAN.md` §4 row A. Diagnostic stage: it always completes, it claims no
book. Everything written by this stage is under `research/fuckup_audit/A/`; everything outside it was read only.

---

## A4 — one page: what Stage A found

**The corrected cost contract is worth +0.412 R per trade on average across the 52 cells, and it does not produce a
book.** Under the measured signal-minute NBBO spread of this population (22-76 bps by price band x time of day, from
`lit_review_2026/cost_curve.csv`) instead of score4's banded 190/120/80/60/50 bps, plus the 0.25 entry coefficient the
live fill data supports, **9 of the 52 family-config x exit cells turn positive on TRAIN** (score4: 0 of 52). None of
the nine clears G1, because none has t >= 2.0:

| TRAIN, contract (c) | mean net R | SE | MDE (2.8 SE) | t | trades/wk | wkR | weeks green |
|---|---|---|---|---|---|---|---|
| F2 {"P":0.12} hold | +0.427 | 0.298 | 0.836 | 1.43 | 15.7 | +6.7 | 0.47 |
| F1 {"P":0.12} hold | +0.266 | 0.170 | 0.476 | 1.56 | 15.7 | +4.2 | 0.43 |
| F6 {} hold | +0.051 | 0.038 | 0.106 | 1.34 | 23.4 | +1.2 | 0.57 |
| F6 {} 2r | +0.028 | 0.026 | 0.073 | 1.06 | 25.3 | +0.7 | 0.57 |
| F8 {"N":30} hold | +0.012 | 0.024 | 0.066 | 0.49 | 21.2 | +0.3 | 0.57 |

The two big means (F1/F2 at P=0.12, hold-to-close) are lottery tickets by construction: WR 14-19%, 80-85% of exits are
stops, and the whole mean sits in a handful of runners — they are reported, not proposed. The two cells with a
believable shape (F6 red-to-green, F8 30-min opening-range, WR 44-49%, worst week -10 to -12R) are +0.01 to +0.05 R,
i.e. **inside their own MDE**: at these n the test could not have seen an effect smaller than 0.07-0.11 R, and the
point estimates are smaller than that.

**Time bands (A2) produce the stage's only G1 passes, and they do not replicate.** Restricting F6 to entries >= 10:30
gives TRAIN +0.048 R, t 2.08, 16.9 trades/week (hold) and +0.049 R, t 2.25, 17.1/week (2r) — both clear G1 — and
`F1 {"P":0.12}` hold >= 10:30 gives +0.838 R, t 2.01 at exactly 5.0 trades/week. On VAL the same three cells are
+0.016 (t 0.49), +0.017 (t 0.53) and -0.104 (t -0.38): **0 of 3 clear G2**, so TEST was not read for selection.

**Intraday market state (A3) does not carry.** Over 104 buckets (entries >= 10:00, IWM/SPY open->entry-minute return in
TRAIN-cut terciles and by sign, plus an approximate breadth-so-far), TRAIN->VAL sign agreement is **0.51** — a coin
flip (brd 0.54, iwm 0.52, spy 0.48). The single direction that does repeat is the one `probe_days.md` already found:
for F6, SPY-positive-so-far buckets are +0.082 R (t 2.15) TRAIN and +0.200 R (t 2.25) VAL while SPY-negative buckets are
-0.017 and -0.027. It fails the pre-registered adoption rule by a hair on TRAIN (kept bucket improves +0.046 R vs the
+0.05 R required) and it halves the trade count.

**Where the correction's money comes from** (mean over the 52 TRAIN cells): the spread TABLE +0.357 R, the entry
coefficient 1.0 -> 0.25 +0.066 R, charging the +2R target exit 0.875 x half -0.010 R. In other words H7 was right about
the size of the defect and right about which part of it mattered: **it was the 3.8x-too-wide band table, not the
double-charged entry crossing.**

**Power / phrasing.** In THIS universe (the point-in-time >= 5%-range day population with the causal floor), at THIS
1-minute horizon, over 2025-01-02 -> 2026-09-11, under THIS book (12/day, 4 concurrent) and THIS corrected cost
contract, **no family-config x exit and no time band or intraday-state bucket showed an edge that survives TRAIN
significance and VAL replication**. The smallest per-trade effect the headline tests could have seen at 80% power is
**0.066 R (F8 N=30), 0.106 R (F6), 0.476 R (F1 P=0.12)**; at the book level that is roughly 1.4 / 2.5 / 7.5 R per week.
Effects below those sizes are invisible here and are not excluded. The largest positive t over 52 cells is 1.56, BELOW
the ~2.7 that 52 independent nulls would be expected to produce, so no permutation test was needed (and none was run:
nothing cleared G2, which is what the pre-registration conditioned it on).

**What goes to Stage C/D.** (1) The corrected contract (c) replaces score4's for every later stage — it is the more
accurate model whether or not it flips a cell; (c') when the engine's take-profit leg genuinely rests. (2) Base
families: **F6 {} and F8 {"N":30}** (positive gross AND positive corrected net on TRAIN, WR 44-49%, the smallest worst
weeks in the grid), with `F8 {"N":15}` as the third. F5 stays dropped; F1/F2 P=0.12 are carried only as a tail-risk
reference. (3) Time window: **entries >= 10:00** for F6/F8 (the >= 10:30 cut is where F6's TRAIN t goes above 2 and it
does not hold on VAL — carry the window, do not treat it as validated). (4) The intraday index-return sign at the entry
minute goes to Stage D as a FEATURE, never as a standalone day filter, per PLAN §3 H3.

**Caveat that limits every number above**: the corrected spread is a per-cell MEDIAN (25 cells, 50-120 sampled quotes
each) applied to every candidate in the cell, so within-cell dispersion is uncharged, and the "live liquidity gate" (d)
is therefore not really a liquidity filter — with spread constant inside a cell, `spread/R <= 0.15` reduces to a
minimum-R floor of 1.50% ($5-10 after 13:00) to 5.07% ($200+ 10-11am). Contract (d)'s gains are the H7 stop-width
mechanism a second time, not a liquidity effect, and they are labelled as such.


---

## 0. PRE-REGISTRATION (written before any run of this stage)

### 0.1 What will be computed

**A0 — the corrected cost contract (H7).** Re-score `research/bf_zero2/candidates3.csv` on score4's population
(`price >= 5`, `entry_m <= 841`, `r_pct >= 1`, and `range_so_far_pct >= 5` for F5-F10), for EVERY family-config x exit
(`hold`, `2r`), under score4's own book — `from trading.hod_break import run_book`,
`rows = (day, entry_m, exit_m, symbol, net, wk)`, `run_book(rows, 12, 4)` — reporting side by side:

| contract | entry charge | stop exit | 15:55 exit | +2R target exit | spread source |
|---|---|---|---|---|---|
| (a) GROSS | 0 | 0 | 0 | 0 | — |
| (b) score4 | `half` | `0.875*half` | `0.412*half` | 0 | `spread_pct` column: 1.90/1.20/0.80/0.60/0.50 % by price band |
| (c) CORRECTED | `0.25*half` | `0.875*half` | `0.412*half` | `0.875*half` | `lit_review_2026/cost_curve.csv` median NBBO spread of THIS population, per (price band x time-of-day band) |
| (c') CORRECTED, resting TP | `0.25*half` | `0.875*half` | `0.412*half` | 0 | as (c) |
| (d) = (c) + live gate | as (c) | as (c) | as (c) | as (c) | as (c), plus drop every candidate with `spread_pct / r_pct > 0.15` BEFORE the book |
| (d') = (c') + live gate | as (c') | as (c') | as (c') | 0 | as (c'), same gate |

`half = 0.5 * spread_pct / max(r_pct, 0.05)` in R units, exactly score4's definition; only the spread table and the
multipliers change. The 0.25 entry coefficient is `probe_costs.md`'s measurement: against the quote at the MOMENT of the
fill our entries pay a median 0.0 bps (46% at or below the mid), and the next-bar-open fill already contains the drift
(`bf_zero/REPORT.md` §8: the last ask of the signal minute is +5.9 bps ABOVE the next bar's open), so charging a full
half spread on top double-counts the crossing. 0.25 is the conservative quartile, not the median.
Target = `0.875*half` in (c) even though the live HOD bracket rests its take-profit leg, because the sim's target fills
on a bar CLOSE; (c') is the engine-accurate version.

Splits: TRAIN 2025-01-02..2025-12-31, VAL 2026-01-01..2026-05-31, TEST 2026-06-01..2026-09-11. **A0 reports all three
because A0 selects nothing** — it is a restatement of an already-published table (`bf_zero2/score4_tables.md` printed
TEST for all 52 cells on 9/16, so TEST is already burned for this exact grid). Selection happens in A1 and reads TEST
only after G2 is frozen in writing.

Per cell: n, trades/week, gross mean R, score4 net, corrected net (c), corrected net (c'), corrected+gate net (d),
t of the corrected net, WR, weekly R, % weeks green, worst week, exit mix.

**A1 — the re-gate (H10).** PLAN §1's gate applied to contract (c):
- **G1 (TRAIN):** mean net R > 0 AND t >= 2.0 on the booked trades AND >= 5 trades/week.
- **G2 (VAL):** mean net R > 0 AND t >= 1.0 AND >= 55% of weeks green. The bar is raised by 1 SE of weekly R for every
  10 cells that passed G1.
- **G3 (TEST):** read ONCE, only for G2 survivors, reported whatever it says, week by week.
- Economic bar, reported and not gated: >= 3R/week at 4 slots.
- Every G2 survivor also gets: tail test (top 1% and top 5% of booked trades removed; winners capped at +3R), a
  per-month table, and a permutation p — day labels shuffled within the split 500x, the null taken as the MAX weekly R
  over ALL cells of the stage, so the p is search-adjusted.

**A2 — time bands (H5).** Base families `F8 {"N":5}`, `F8 {"N":15}`, `F8 {"N":30}`, `F6 {}`, `F1 {"P":0.12}` (F5 is
dropped as a base family — gross-negative at zero cost, PLAN §3 — but `F5 {"K":5,"X":0.04}` is carried as a reference
row), contract (c), the book re-run with entries restricted to `entry_m >= 600` (10:00), `entry_m >= 630` (10:30), and
`entry_m < 600` (the 09:30-10:00 window only). Same columns as A0. TRAIN and VAL; TEST not read here.

**A3 — intraday market state (H3+H5).** Entries `entry_m >= 600` only. Booked trades split by
(i) the IWM return from its 09:30 open to the trade's ENTRY minute, (ii) the same for SPY (`etf_1min.db`), in terciles
(cut on TRAIN, applied to VAL) and by sign; and (iii) breadth-so-far — the share of THAT day's scoring-population
candidates that signalled at an EARLIER minute and had `dist_open_pct > 0`. **(iii) is an approximation and is labelled
as one**: it is breadth inside the study's own candidate population, not the market's, and each candidate's
`dist_open_pct` is measured at its own entry minute, not at the split minute. Report the TRAIN->VAL sign agreement and
the cell count. Nothing here is adopted in Stage A; per PLAN §3 H3 the decision rule for a day/state filter is: the
excluded bucket must be negative on TRAIN AND VAL, the kept bucket's mean net R must improve by >= 0.05R, and the
bucket must be defined by data available at 09:30 (or at the entry minute for an intraday index return).

**A4 — one-page summary at the top of this report**, with the power (SE of mean R, and the minimum detectable effect
`MDE = 2.8*SE` for 80% power at alpha 0.05 two-sided) beside every headline number, and the PLAN §1 phrasing rule:
never "no edge exists"; always "not detectable in THIS universe / window / book / cost, smallest effect visible X".

### 0.2 Cells I will look at (declared in advance)

| block | cells |
|---|---|
| A0 | 26 family-configs x 2 exits x 6 contracts (a, b, c, c', d, d') x 3 splits — 52 decision-relevant cells (contract (c), TRAIN) |
| A1 | the same 52 under G1; G2 only for G1 survivors; TEST only for G2 survivors |
| A2 | 6 keys x 2 exits x 3 time windows = 36 |
| A3 | 4 keys x 2 exits x (2 indices x [3 terciles + 2 signs] + 3 breadth terciles) = 4 x 2 x 13 = 104 bucket-cells |
| total declared | 52 + 36 + 104 = 192 new cells, on top of the program's 52 + probe_stops' ~100 + probe_days' 156 |

### 0.3 What would make me say the corrected contract changes the verdict

A family-config x exit whose corrected (c) TRAIN mean net R is > 0 with t >= 2 and >= 5 trades/week. If none exists, the
reported conclusion is the closest miss per family with its own MDE, and the corrected contract is still adopted for
Stages B-E (it is the more accurate cost model whether or not it flips anything).

(results appended below after the runs)

---

## A0 — every family-config x exit under the six contracts

population 446,012 rows (parity with score4: identical) | 26 keys | weeks TRAIN 53 / VAL 22 / TEST 14 | exit mix of the population (2r): stop 46.0%, eod 34.2%, target 19.9%

**Parity anchor (CLAUDE.md independent-check rule, coding-error half):** re-deriving contract (b) reproduces `bf_zero2/score4_results.csv` cell for cell — 52 of 52 merged, max |delta n| = 0, max |delta mean R| = 0.000, max |delta weekly R| = 0.05 (score4 rounds weekly R to 1 dp, this run to 2). `A/a0_parity.csv`. This catches coding errors only; the specification caveats are in §Caveats.

corrected spread table, median signal-minute NBBO in bps of price (rows = price band, cols = time band):

```
hb       09:30-09:35  09:35-10:00  10:00-11:00  11:00-13:00  13:00+
pb                                                                 
$10-20          43.8         42.2         28.8         21.7    28.6
$20-50          51.6         46.0         45.3         32.2    34.9
$200+           54.3         74.3         76.1         64.0    68.2
$5-10           35.8         34.9         29.8         26.6    22.5
$50-200         66.8         63.4         39.2         30.3    36.4

n quotes per cell:
hb       09:30-09:35  09:35-10:00  10:00-11:00  11:00-13:00  13:00+
pb                                                                 
$10-20           118          116          114          113     115
$20-50           118          120          115          114     119
$200+             50           77           79           78      79
$5-10            117          116          113          110     112
$50-200           80           79           80           78      79
```

score4 charged, for the same names: 190 / 120 / 80 / 60 / 50 bps at $5-10 / $10-20 / $20-50 / $50-100 / $100+, at every hour.

### TRAIN
```
                    key exit  corr_n  corr_tpw  gross_meanR  s4_meanR  corr_meanR  corrp_meanR  gate_meanR  gatep_meanR  corr_se  corr_t  corr_WR  corr_wkR  corr_green  corr_worst  gate_n  gate_tpw  gate_wkR  mix_stop  mix_target  mix_eod
         F2 {"P": 0.12} hold     834      15.7        0.524    -0.162       0.427        0.427       0.656        0.656   0.2984    1.43     14.4      6.72        0.47       -23.6     363       6.8      4.49     0.845       0.000    0.155
         F1 {"P": 0.12} hold     833      15.7        0.344    -0.205       0.266        0.266       0.426        0.426   0.1700    1.56     18.7      4.18        0.43       -22.9     509       9.6      4.09     0.797       0.000    0.203
                  F6 {} hold    1242      23.4        0.084    -0.111       0.051        0.051       0.069        0.069   0.0380    1.34     44.4      1.20        0.57       -12.1    1194      22.5      1.55     0.262       0.000    0.738
 F5 {"K": 8, "X": 0.06} hold    1602      30.2        0.082    -0.257       0.028        0.028       0.109        0.109   0.0494    0.57     38.5      0.85        0.49       -21.0    1500      28.3      3.09     0.466       0.000    0.534
                  F6 {}   2r    1342      25.3        0.061    -0.121       0.028        0.033       0.034        0.037   0.0259    1.06     46.5      0.70        0.57       -11.3    1280      24.2      0.82     0.241       0.107    0.651
           F8 {"N": 30} hold    1124      21.2        0.032    -0.106       0.012        0.012       0.012        0.012   0.0235    0.49     48.6      0.25        0.57        -9.6    1125      21.2      0.26     0.151       0.000    0.849
 F5 {"K": 8, "X": 0.04} hold    1862      35.1        0.084    -0.342       0.011        0.011      -0.004       -0.004   0.0482    0.22     35.3      0.38        0.42       -19.1    1773      33.5     -0.15     0.557       0.000    0.443
           F8 {"N": 30}   2r    1148      21.7        0.025    -0.112       0.005        0.006       0.005        0.006   0.0215    0.22     49.0      0.10        0.57        -9.6    1149      21.7      0.12     0.150       0.036    0.814
           F8 {"N": 15} hold    1244      23.5        0.030    -0.126       0.004        0.004       0.010        0.010   0.0281    0.13     44.9      0.09        0.47       -13.9    1245      23.5      0.23     0.233       0.000    0.767
           F8 {"N": 15}   2r    1326      25.0        0.023    -0.133      -0.004       -0.001      -0.003       -0.001   0.0244   -0.16     45.8     -0.10        0.47       -12.6    1323      25.0     -0.07     0.227       0.080    0.693
         F2 {"P": 0.08} hold    1861      35.1        0.095    -0.668      -0.025       -0.025       0.334        0.334   0.1287   -0.19     14.0     -0.86        0.36       -44.7     495       9.3      3.12     0.850       0.000    0.150
         F2 {"P": 0.05} hold    2601      49.1        0.112    -0.724      -0.029       -0.029       0.248        0.248   0.0989   -0.29     16.1     -1.43        0.38       -60.0     557      10.5      2.60     0.830       0.000    0.170
            F8 {"N": 5} hold    1427      26.9       -0.002    -0.200      -0.034       -0.034      -0.024       -0.024   0.0341   -1.00     40.4     -0.92        0.49       -18.6    1409      26.6     -0.63     0.362       0.000    0.638
         F9 {"G": 0.05} hold    1141      21.5       -0.011    -0.184      -0.041       -0.041      -0.050       -0.050   0.0430   -0.95     37.8     -0.88        0.42       -12.9    1126      21.2     -1.06     0.413       0.000    0.587
            F8 {"N": 5}   2r    1589      30.0       -0.022    -0.217      -0.056       -0.051      -0.045       -0.042   0.0259   -2.15     42.2     -1.67        0.40       -14.0    1559      29.4     -1.34     0.346       0.118    0.536
 F5 {"K": 8, "X": 0.06}   2r    1941      36.6        0.000    -0.347      -0.060       -0.050      -0.042       -0.034   0.0275   -2.20     40.7     -2.21        0.40       -20.7    1769      33.4     -1.39     0.457       0.181    0.362
         F1 {"P": 0.08} hold    2014      38.0        0.039    -0.637      -0.063       -0.063      -0.004       -0.004   0.0926   -0.68     17.1     -2.39        0.45       -48.6     908      17.1     -0.07     0.815       0.000    0.185
 F5 {"K": 5, "X": 0.06} hold    1879      35.5        0.000    -0.398      -0.063       -0.063      -0.057       -0.057   0.0502   -1.25     31.8     -2.23        0.36       -26.0    1710      32.3     -1.84     0.564       0.000    0.436
 F5 {"K": 5, "X": 0.04} hold    2139      40.4        0.019    -0.475      -0.063       -0.063      -0.098       -0.098   0.0504   -1.24     29.4     -2.52        0.42       -38.5    1897      35.8     -3.50     0.635       0.000    0.365
F3 {"M": 15, "P": 0.05} hold    1794      33.8        0.066    -0.655      -0.070       -0.070      -0.219       -0.219   0.0946   -0.74     16.9     -2.36        0.38       -45.3     519       9.8     -2.14     0.820       0.000    0.180
         F9 {"G": 0.05}   2r    1191      22.5       -0.052    -0.218      -0.083       -0.079      -0.088       -0.085   0.0308   -2.70     39.5     -1.87        0.32       -12.5    1170      22.1     -1.94     0.397       0.127    0.476
 F5 {"K": 3, "X": 0.06} hold    1989      37.5       -0.030    -0.492      -0.105       -0.105      -0.043       -0.043   0.0497   -2.12     29.2     -3.95        0.32       -31.6    1713      32.3     -1.38     0.619       0.000    0.381
F3 {"M": 30, "P": 0.05} hold    2033      38.4        0.028    -0.710      -0.109       -0.109      -0.326       -0.326   0.0862   -1.26     16.5     -4.18        0.38       -53.6     603      11.4     -3.71     0.824       0.000    0.176
 F5 {"K": 3, "X": 0.04} hold    2248      42.4       -0.024    -0.568      -0.116       -0.116      -0.034       -0.034   0.0546   -2.12     26.6     -4.90        0.34       -40.5    1837      34.7     -1.19     0.688       0.000    0.312
 F5 {"K": 8, "X": 0.04}   2r    2350      44.3       -0.037    -0.465      -0.117       -0.101      -0.086       -0.076   0.0266   -4.39     38.2     -5.18        0.23       -26.7    2194      41.4     -3.57     0.533       0.223    0.243
 F5 {"K": 5, "X": 0.02} hold    2418      45.6       -0.000    -0.700      -0.126       -0.126      -0.030       -0.030   0.0523   -2.41     25.3     -5.75        0.34       -41.3    1249      23.6     -0.70     0.719       0.000    0.281
 F5 {"K": 5, "X": 0.06}   2r    2339      44.1       -0.071    -0.467      -0.143       -0.128      -0.102       -0.094   0.0266   -5.36     36.4     -6.29        0.21       -28.7    2055      38.8     -3.97     0.540       0.223    0.237
 F3 {"M": 15, "P": 0.1} hold     854      16.1       -0.038    -0.694      -0.153       -0.153      -0.110       -0.110   0.1423   -1.08     13.7     -2.47        0.38       -29.4     271       5.1     -0.56     0.852       0.000    0.148
 F5 {"K": 8, "X": 0.02} hold    2291      43.2       -0.037    -0.708      -0.156       -0.156      -0.035       -0.035   0.0462   -3.38     27.2     -6.76        0.38       -50.7    1452      27.4     -0.96     0.682       0.000    0.318
 F5 {"K": 5, "X": 0.04}   2r    2688      50.7       -0.083    -0.555      -0.172       -0.153      -0.123       -0.111   0.0259   -6.62     35.0     -8.71        0.21       -44.9    2321      43.8     -5.38     0.590       0.249    0.161
                  F7 {} hold    1343      25.3       -0.132    -0.406      -0.177       -0.177      -0.140       -0.140   0.0506   -3.50     32.0     -4.49        0.19       -23.3    1228      23.2     -3.24     0.399       0.000    0.601
 F5 {"K": 3, "X": 0.04}   2r    2789      52.6       -0.086    -0.598      -0.188       -0.165      -0.119       -0.106   0.0265   -7.10     34.9     -9.88        0.21       -31.5    2183      41.2     -4.92     0.617       0.276    0.106
 F5 {"K": 3, "X": 0.06}   2r    2493      47.0       -0.103    -0.551      -0.188       -0.171      -0.096       -0.085   0.0267   -7.04     35.2     -8.85        0.17       -26.5    2067      39.0     -3.73     0.578       0.238    0.183
         F1 {"P": 0.05} hold    2649      50.0       -0.076    -0.851      -0.205       -0.205      -0.054       -0.054   0.0671   -3.05     16.3    -10.25        0.23       -49.2    1087      20.5     -1.11     0.823       0.000    0.177
         F1 {"P": 0.12}   2r     847      16.0       -0.127    -0.612      -0.210       -0.190      -0.090       -0.076   0.0517   -4.07     33.1     -3.36        0.21       -14.1     510       9.6     -0.87     0.656       0.313    0.031
 F5 {"K": 3, "X": 0.02} hold    2516      47.5       -0.091    -0.821      -0.223       -0.223      -0.032       -0.032   0.0530   -4.20     21.1    -10.56        0.30       -51.6     782      14.8     -0.47     0.770       0.000    0.230
 F5 {"K": 5, "X": 0.02}   2r    2935      55.4       -0.098    -0.726      -0.234       -0.202      -0.169       -0.154   0.0263   -8.91     33.7    -12.95        0.15       -42.0    1339      25.3     -4.28     0.642       0.292    0.065
 F5 {"K": 8, "X": 0.02}   2r    2806      52.9       -0.109    -0.729      -0.237       -0.209      -0.112       -0.098   0.0259   -9.15     33.8    -12.53        0.13       -36.7    1651      31.2     -3.49     0.626       0.262    0.112
                  F7 {}   2r    1448      27.3       -0.194    -0.456      -0.241       -0.236      -0.223       -0.221   0.0251   -9.62     33.8     -6.59        0.11       -20.7    1297      24.5     -5.45     0.380       0.092    0.528
 F3 {"M": 30, "P": 0.1} hold    1170      22.1       -0.127    -0.817      -0.248       -0.248      -0.273       -0.273   0.1137   -2.18     12.7     -5.47        0.30       -43.4     350       6.6     -1.80     0.862       0.000    0.138
                  F4 {} hold    1982      37.4       -0.167    -0.838      -0.260       -0.260      -0.116       -0.116   0.0736   -3.54     16.9     -9.73        0.28       -42.3     957      18.1     -2.09     0.807       0.000    0.193
         F2 {"P": 0.12}   2r     847      16.0       -0.167    -0.763      -0.270       -0.243      -0.048       -0.032   0.0567   -4.76     34.1     -4.31        0.30       -15.2     366       6.9     -0.33     0.652       0.335    0.013
 F3 {"M": 30, "P": 0.1}   2r    1243      23.5       -0.144    -0.750      -0.271       -0.239      -0.264       -0.252   0.0419   -6.48     31.9     -6.37        0.17       -25.4     350       6.6     -1.74     0.677       0.310    0.013
         F1 {"P": 0.08}   2r    2178      41.1       -0.163    -0.765      -0.272       -0.246      -0.169       -0.154   0.0320   -8.50     32.1    -11.16        0.11       -35.5     919      17.3     -2.93     0.671       0.305    0.024
 F3 {"M": 15, "P": 0.1}   2r     890      16.8       -0.156    -0.731      -0.279       -0.248      -0.275       -0.263   0.0496   -5.62     31.7     -4.68        0.19       -23.0     271       5.1     -1.40     0.680       0.306    0.015
 F5 {"K": 3, "X": 0.02}   2r    2937      55.4       -0.138    -0.793      -0.282       -0.248      -0.186       -0.170   0.0266  -10.58     32.4    -15.60        0.15       -50.0     810      15.3     -2.84     0.665       0.295    0.040
         F2 {"P": 0.08}   2r    1948      36.8       -0.162    -0.832      -0.289       -0.256      -0.160       -0.144   0.0354   -8.16     32.9    -10.62        0.13       -35.3     503       9.5     -1.51     0.667       0.325    0.008
                  F4 {}   2r    2105      39.7       -0.196    -0.805      -0.296       -0.274      -0.222       -0.212   0.0309   -9.59     30.5    -11.77        0.11       -30.4     969      18.3     -4.06     0.675       0.276    0.048
         F1 {"P": 0.05}   2r    2944      55.5       -0.161    -0.846      -0.301       -0.267      -0.181       -0.166   0.0270  -11.15     31.5    -16.70        0.06       -40.2    1123      21.2     -3.84     0.679       0.302    0.019
F3 {"M": 15, "P": 0.05}   2r    2110      39.8       -0.164    -0.807      -0.311       -0.275      -0.207       -0.194   0.0320   -9.70     31.6    -12.37        0.15       -44.6     522       9.8     -2.04     0.680       0.303    0.017
         F2 {"P": 0.05}   2r    2823      53.3       -0.159    -0.903      -0.313       -0.275      -0.257       -0.243   0.0282  -11.12     32.0    -16.67        0.11       -49.3     565      10.7     -2.74     0.677       0.313    0.010
F3 {"M": 30, "P": 0.05}   2r    2450      46.2       -0.172    -0.832      -0.321       -0.286      -0.222       -0.210   0.0297  -10.83     31.2    -14.84        0.11       -49.2     610      11.5     -2.56     0.684       0.302    0.015
```

### VAL
```
                    key exit  corr_n  corr_tpw  gross_meanR  s4_meanR  corr_meanR  corrp_meanR  gate_meanR  gatep_meanR  corr_se  corr_t  corr_WR  corr_wkR  corr_green  corr_worst  gate_n  gate_tpw  gate_wkR  mix_stop  mix_target  mix_eod
                  F6 {} hold     536      24.4        0.197     0.001       0.163        0.163       0.145        0.145   0.0663    2.46     47.9      3.97        0.68        -8.1     520      23.6      3.42     0.284       0.000    0.716
 F5 {"K": 3, "X": 0.06} hold     791      36.0        0.207    -0.209       0.130        0.130       0.001        0.001   0.0803    1.62     35.7      4.68        0.64       -18.9     738      33.5      0.04     0.571       0.000    0.429
 F5 {"K": 3, "X": 0.04} hold     862      39.2        0.185    -0.291       0.092        0.092       0.082        0.082   0.0835    1.10     34.1      3.59        0.55       -40.2     789      35.9      2.94     0.621       0.000    0.379
 F5 {"K": 5, "X": 0.06} hold     751      34.1        0.135    -0.231       0.071        0.071       0.031        0.031   0.0922    0.77     37.3      2.43        0.50       -14.2     685      31.1      0.95     0.530       0.000    0.470
                  F6 {}   2r     607      27.6        0.106    -0.072       0.071        0.076       0.089        0.093   0.0407    1.73     48.8      1.95        0.64        -9.2     580      26.4      2.34     0.257       0.125    0.618
         F2 {"P": 0.08} hold     827      37.6        0.178    -0.541       0.048        0.048       0.241        0.241   0.1444    0.33     16.3      1.80        0.41       -53.8     201       9.1      2.20     0.826       0.000    0.174
            F8 {"N": 5} hold     600      27.3        0.060    -0.115       0.025        0.025       0.026        0.026   0.0582    0.44     42.5      0.69        0.50       -11.7     587      26.7      0.71     0.363       0.000    0.637
         F2 {"P": 0.12} hold     376      17.1        0.131    -0.496       0.024        0.024       0.453        0.453   0.2484    0.10     12.8      0.42        0.41       -26.6     141       6.4      2.90     0.867       0.000    0.133
           F8 {"N": 30} hold     456      20.7        0.014    -0.106      -0.006       -0.006      -0.008       -0.008   0.0336   -0.19     48.7     -0.13        0.45        -6.8     457      20.8     -0.16     0.134       0.000    0.866
           F8 {"N": 30}   2r     463      21.0        0.015    -0.105      -0.006       -0.005      -0.007       -0.006   0.0324   -0.17     49.0     -0.12        0.45        -6.8     464      21.1     -0.14     0.132       0.026    0.842
            F8 {"N": 5}   2r     673      30.6        0.024    -0.151      -0.012       -0.008      -0.010       -0.007   0.0403   -0.31     43.8     -0.38        0.50       -12.8     653      29.7     -0.31     0.331       0.122    0.547
         F9 {"G": 0.05}   2r     559      25.4        0.020    -0.135      -0.015       -0.011      -0.003        0.001   0.0447   -0.33     45.4     -0.38        0.59       -16.0     546      24.8     -0.08     0.361       0.120    0.519
 F5 {"K": 8, "X": 0.06} hold     662      30.1        0.041    -0.275      -0.017       -0.017      -0.012       -0.012   0.0674   -0.25     37.8     -0.51        0.50       -18.1     639      29.0     -0.36     0.468       0.000    0.532
         F9 {"G": 0.05} hold     527      24.0        0.005    -0.157      -0.029       -0.029      -0.007       -0.007   0.0526   -0.54     43.5     -0.68        0.55       -18.5     516      23.5     -0.17     0.385       0.000    0.615
 F5 {"K": 5, "X": 0.04} hold     819      37.2        0.039    -0.403      -0.047       -0.047      -0.065       -0.065   0.0726   -0.65     33.1     -1.75        0.41       -33.4     749      34.0     -2.20     0.603       0.000    0.397
 F5 {"K": 3, "X": 0.06}   2r    1074      48.8        0.041    -0.362      -0.049       -0.025      -0.107       -0.096   0.0421   -1.16     40.1     -2.39        0.32       -31.5     927      42.1     -4.51     0.544       0.282    0.174
 F5 {"K": 8, "X": 0.04}   2r     936      42.5        0.011    -0.379      -0.069       -0.053      -0.089       -0.079   0.0417   -1.65     40.2     -2.94        0.55       -34.1     879      40.0     -3.57     0.501       0.221    0.278
F3 {"M": 15, "P": 0.05} hold     778      35.4        0.066    -0.594      -0.076       -0.076       0.258        0.258   0.1103   -0.69     19.8     -2.70        0.36       -37.0     222      10.1      2.61     0.788       0.000    0.212
 F5 {"K": 5, "X": 0.04}   2r    1073      48.8        0.016    -0.404      -0.078       -0.056      -0.078       -0.066   0.0415   -1.88     40.1     -3.82        0.41       -24.3     969      44.0     -3.45     0.544       0.267    0.189
         F2 {"P": 0.05} hold    1093      49.7        0.076    -0.690      -0.079       -0.079       0.280        0.280   0.1105   -0.72     17.8     -3.93        0.41       -49.5     262      11.9      3.33     0.809       0.000    0.191
           F8 {"N": 15} hold     503      22.9       -0.051    -0.191      -0.079       -0.079      -0.079       -0.079   0.0381   -2.08     42.3     -1.81        0.41       -11.4     500      22.7     -1.80     0.239       0.000    0.761
 F5 {"K": 8, "X": 0.04} hold     739      33.6       -0.006    -0.397      -0.079       -0.079      -0.105       -0.105   0.0608   -1.30     35.0     -2.65        0.45       -27.3     723      32.9     -3.45     0.536       0.000    0.464
           F8 {"N": 15}   2r     517      23.5       -0.052    -0.190      -0.081       -0.079      -0.083       -0.082   0.0355   -2.28     42.7     -1.90        0.41       -11.4     513      23.3     -1.93     0.232       0.041    0.727
 F5 {"K": 5, "X": 0.06}   2r     956      43.5       -0.010    -0.379      -0.083       -0.067      -0.045       -0.036   0.0419   -1.98     40.1     -3.61        0.32       -17.7     834      37.9     -1.70     0.507       0.218    0.275
 F5 {"K": 3, "X": 0.04}   2r    1147      52.1        0.021    -0.423      -0.084       -0.058      -0.106       -0.092   0.0418   -2.01     39.2     -4.39        0.41       -28.2     986      44.8     -4.73     0.581       0.303    0.117
 F5 {"K": 8, "X": 0.06}   2r     808      36.7       -0.040    -0.370      -0.106       -0.094      -0.051       -0.043   0.0422   -2.50     39.1     -3.88        0.41       -19.8     754      34.3     -1.74     0.473       0.181    0.347
 F5 {"K": 5, "X": 0.02} hold    1001      45.5        0.013    -0.637      -0.120       -0.120      -0.059       -0.059   0.0787   -1.53     24.6     -5.46        0.36       -35.1     624      28.4     -1.69     0.724       0.000    0.276
F3 {"M": 30, "P": 0.05} hold     876      39.8        0.022    -0.660      -0.121       -0.121       0.229        0.229   0.0995   -1.22     19.5     -4.83        0.36       -40.6     245      11.1      2.55     0.787       0.000    0.213
 F5 {"K": 3, "X": 0.02} hold    1037      47.1       -0.007    -0.686      -0.144       -0.144      -0.289       -0.289   0.0815   -1.76     22.7     -6.77        0.27       -44.1     453      20.6     -5.96     0.755       0.000    0.245
         F1 {"P": 0.05} hold    1074      48.8       -0.005    -0.711      -0.145       -0.145      -0.136       -0.136   0.0908   -1.59     19.3     -7.06        0.27       -49.0     518      23.5     -3.21     0.791       0.000    0.209
                  F4 {} hold     857      39.0       -0.076    -0.733      -0.179       -0.179      -0.373       -0.373   0.0978   -1.83     20.0     -6.96        0.32       -29.7     363      16.5     -6.16     0.785       0.000    0.215
 F5 {"K": 8, "X": 0.02}   2r    1154      52.5       -0.051    -0.613      -0.188       -0.157      -0.101       -0.086   0.0410   -4.59     36.4     -9.87        0.18       -31.9     855      38.9     -3.93     0.604       0.278    0.118
 F5 {"K": 8, "X": 0.02} hold     921      41.9       -0.072    -0.690      -0.201       -0.201      -0.060       -0.060   0.0657   -3.06     28.3     -8.43        0.36       -41.1     717      32.6     -1.94     0.672       0.000    0.328
         F1 {"P": 0.08} hold     896      40.7       -0.096    -0.718      -0.204       -0.204      -0.253       -0.253   0.0954   -2.14     17.9     -8.31        0.32       -53.5     402      18.3     -4.63     0.809       0.000    0.191
 F5 {"K": 5, "X": 0.02}   2r    1195      54.3       -0.090    -0.678      -0.236       -0.202      -0.160       -0.144   0.0411   -5.75     33.8    -12.83        0.18       -35.5     728      33.1     -5.29     0.647       0.294    0.059
F3 {"M": 30, "P": 0.05}   2r    1075      48.9       -0.092    -0.704      -0.250       -0.210      -0.248       -0.236   0.0449   -5.58     33.6    -12.24        0.14       -28.8     246      11.2     -2.78     0.657       0.319    0.024
 F5 {"K": 3, "X": 0.02}   2r    1217      55.3       -0.116    -0.716      -0.270       -0.232      -0.233       -0.218   0.0415   -6.51     33.2    -14.94        0.23       -45.5     493      22.4     -5.22     0.661       0.306    0.032
F3 {"M": 15, "P": 0.05}   2r     953      43.3       -0.117    -0.709      -0.275       -0.235      -0.292       -0.280   0.0480   -5.73     33.1    -11.91        0.14       -31.9     222      10.1     -2.95     0.664       0.315    0.021
                  F4 {}   2r     945      43.0       -0.165    -0.756      -0.276       -0.250      -0.205       -0.196   0.0469   -5.90     32.0    -11.87        0.09       -24.3     366      16.6     -3.42     0.667       0.285    0.049
         F1 {"P": 0.12} hold     390      17.7       -0.192    -0.710      -0.279       -0.279      -0.255       -0.255   0.1420   -1.96     15.4     -4.94        0.32       -24.7     207       9.4     -2.40     0.838       0.000    0.162
         F1 {"P": 0.12}   2r     398      18.1       -0.193    -0.656      -0.284       -0.263      -0.245       -0.233   0.0730   -3.90     30.4     -5.15        0.23       -16.3     209       9.5     -2.32     0.688       0.291    0.020
         F1 {"P": 0.08}   2r     984      44.7       -0.172    -0.730      -0.288       -0.260      -0.319       -0.306   0.0466   -6.17     31.2    -12.87        0.18       -34.0     409      18.6     -5.92     0.681       0.300    0.019
                  F7 {}   2r     654      29.7       -0.270    -0.512      -0.326       -0.323      -0.241       -0.239   0.0354   -9.22     31.3     -9.70        0.09       -19.1     562      25.5     -6.16     0.393       0.064    0.543
         F1 {"P": 0.05}   2r    1221      55.5       -0.179    -0.823      -0.333       -0.297      -0.290       -0.277   0.0417   -7.99     30.8    -18.46        0.00       -43.0     538      24.5     -7.10     0.685       0.298    0.017
         F2 {"P": 0.08}   2r     897      40.8       -0.198    -0.844      -0.338       -0.305      -0.208       -0.193   0.0498   -6.78     31.0    -13.77        0.09       -31.9     201       9.1     -1.90     0.687       0.304    0.009
                  F7 {} hold     624      28.4       -0.288    -0.537      -0.342       -0.342      -0.261       -0.261   0.0418   -8.20     29.2     -9.71        0.18       -22.0     540      24.5     -6.40     0.415       0.000    0.585
         F2 {"P": 0.12}   2r     380      17.3       -0.237    -0.800      -0.348       -0.324      -0.142       -0.128   0.0774   -4.49     30.0     -6.01        0.14       -16.3     141       6.4     -0.91     0.697       0.295    0.008
 F3 {"M": 30, "P": 0.1}   2r     634      28.8       -0.215    -0.782      -0.359       -0.327      -0.305       -0.293   0.0590   -6.08     30.1    -10.35        0.27       -35.2     135       6.1     -1.87     0.694       0.293    0.013
         F2 {"P": 0.05}   2r    1197      54.4       -0.217    -0.913      -0.386       -0.346      -0.241       -0.227   0.0426   -9.06     30.4    -21.02        0.00       -41.1     262      11.9     -2.87     0.693       0.297    0.010
 F3 {"M": 15, "P": 0.1}   2r     469      21.3       -0.291    -0.838      -0.427       -0.399      -0.339       -0.327   0.0670   -6.38     27.9     -9.11        0.18       -35.4     114       5.2     -1.76     0.716       0.271    0.013
 F3 {"M": 30, "P": 0.1} hold     587      26.7       -0.351    -0.985      -0.487       -0.487      -0.429       -0.429   0.1103   -4.41     13.1    -12.98        0.23       -38.0     135       6.1     -2.63     0.860       0.000    0.140
 F3 {"M": 15, "P": 0.1} hold     453      20.6       -0.459    -1.066      -0.589       -0.589      -0.366       -0.366   0.1151   -5.12     12.4    -12.14        0.05       -32.3     114       5.2     -1.89     0.868       0.000    0.132
```

### TEST
*(TEST is shown because A0 selects nothing and `score4_tables.md` already published TEST for all 52 of these cells on 9/16. It is not used to choose anything in Stage A.)*

```
                    key exit  corr_n  corr_tpw  gross_meanR  s4_meanR  corr_meanR  corrp_meanR  gate_meanR  gatep_meanR  corr_se  corr_t  corr_WR  corr_wkR  corr_green  corr_worst  gate_n  gate_tpw  gate_wkR  mix_stop  mix_target  mix_eod
 F3 {"M": 15, "P": 0.1} hold     293      20.9        0.551    -0.085       0.428        0.428       0.164        0.164   0.3005    1.42     17.1      8.96        0.57       -16.4      92       6.6      1.08     0.829       0.000    0.171
 F3 {"M": 30, "P": 0.1} hold     392      28.0        0.300    -0.364       0.174        0.174      -0.105       -0.105   0.2353    0.74     14.3      4.88        0.50       -23.9     117       8.4     -0.88     0.857       0.000    0.143
 F5 {"K": 3, "X": 0.06} hold     553      39.5        0.135    -0.291       0.059        0.059      -0.043       -0.043   0.2017    0.29     31.6      2.32        0.43       -16.4     514      36.7     -1.57     0.588       0.000    0.412
           F8 {"N": 15}   2r     326      23.3        0.026    -0.106      -0.000        0.001      -0.001        0.000   0.0444   -0.01     44.5     -0.01        0.57       -14.2     327      23.4     -0.03     0.163       0.043    0.794
 F5 {"K": 8, "X": 0.06}   2r     528      37.7        0.050    -0.277      -0.012       -0.001      -0.023       -0.015   0.0532   -0.23     42.4     -0.45        0.50       -23.0     487      34.8     -0.78     0.430       0.193    0.377
 F5 {"K": 3, "X": 0.04} hold     620      44.3        0.079    -0.430      -0.015       -0.015       0.185        0.185   0.1868   -0.08     26.3     -0.66        0.43       -32.1     523      37.4      6.92     0.679       0.000    0.321
           F8 {"N": 15} hold     318      22.7        0.001    -0.133      -0.025       -0.025      -0.029       -0.029   0.0430   -0.58     44.7     -0.57        0.50       -14.2     318      22.7     -0.65     0.167       0.000    0.833
 F5 {"K": 8, "X": 0.06} hold     425      30.4        0.015    -0.296      -0.038       -0.038      -0.017       -0.017   0.0654   -0.58     40.0     -1.15        0.50       -17.4     415      29.6     -0.49     0.435       0.000    0.565
           F8 {"N": 30} hold     296      21.1       -0.021    -0.144      -0.041       -0.041      -0.039       -0.039   0.0372   -1.10     45.9     -0.86        0.29        -7.2     296      21.1     -0.83     0.111       0.000    0.889
         F2 {"P": 0.08} hold     559      39.9        0.087    -0.686      -0.042       -0.042       0.156        0.156   0.1940   -0.21     15.2     -1.66        0.29       -36.3     137       9.8      1.52     0.837       0.000    0.163
                  F6 {}   2r     419      29.9       -0.008    -0.197      -0.048       -0.041      -0.067       -0.063   0.0501   -0.96     40.6     -1.44        0.43       -13.6     389      27.8     -1.85     0.301       0.131    0.568
           F8 {"N": 30}   2r     299      21.4       -0.029    -0.154      -0.049       -0.049      -0.048       -0.048   0.0362   -1.37     45.5     -1.06        0.29        -7.2     299      21.4     -1.02     0.114       0.010    0.876
 F5 {"K": 8, "X": 0.04} hold     469      33.5        0.009    -0.387      -0.065       -0.065      -0.006       -0.006   0.0719   -0.91     36.7     -2.18        0.50       -26.2     460      32.9     -0.21     0.516       0.000    0.484
 F5 {"K": 8, "X": 0.04}   2r     619      44.2        0.013    -0.378      -0.070       -0.052      -0.042       -0.032   0.0518   -1.34     40.4     -3.07        0.36       -25.2     557      39.8     -1.69     0.493       0.231    0.276
                  F6 {} hold     378      27.0       -0.043    -0.254      -0.082       -0.082      -0.057       -0.057   0.0623   -1.32     37.6     -2.22        0.29       -14.4     358      25.6     -1.45     0.333       0.000    0.667
F3 {"M": 15, "P": 0.05} hold     511      36.5        0.045    -0.655      -0.094       -0.094       0.026        0.026   0.1538   -0.61     19.2     -3.44        0.36       -29.5     186      13.3      0.34     0.800       0.000    0.200
            F8 {"N": 5}   2r     452      32.3       -0.060    -0.251      -0.097       -0.092      -0.059       -0.055   0.0483   -2.01     42.0     -3.12        0.29       -18.9     437      31.2     -1.83     0.365       0.104    0.531
                  F7 {} hold     403      28.8       -0.043    -0.287      -0.097       -0.097      -0.113       -0.113   0.1271   -0.76     35.7     -2.78        0.36       -22.3     353      25.2     -2.85     0.387       0.000    0.613
F3 {"M": 30, "P": 0.05} hold     551      39.4        0.040    -0.664      -0.099       -0.099      -0.032       -0.032   0.1462   -0.68     19.4     -3.91        0.43       -32.2     208      14.9     -0.48     0.800       0.000    0.200
         F9 {"G": 0.05}   2r     365      26.1       -0.074    -0.234      -0.107       -0.103      -0.125       -0.122   0.0564   -1.90     39.7     -2.80        0.29       -11.2     364      26.0     -3.25     0.408       0.126    0.466
         F9 {"G": 0.05} hold     347      24.8       -0.077    -0.244      -0.109       -0.109      -0.128       -0.128   0.0674   -1.61     37.5     -2.69        0.36       -11.1     347      24.8     -3.18     0.435       0.000    0.565
 F5 {"K": 5, "X": 0.06}   2r     616      44.0       -0.064    -0.438      -0.139       -0.124      -0.089       -0.080   0.0509   -2.72     38.5     -6.09        0.29       -27.7     515      36.8     -3.26     0.529       0.206    0.265
            F8 {"N": 5} hold     408      29.1       -0.115    -0.305      -0.150       -0.150      -0.090       -0.090   0.0520   -2.88     40.7     -4.37        0.29       -20.0     396      28.3     -2.54     0.392       0.000    0.608
 F5 {"K": 5, "X": 0.06} hold     487      34.8       -0.098    -0.453      -0.162       -0.162      -0.066       -0.066   0.0663   -2.45     35.1     -5.65        0.36       -22.9     437      31.2     -2.06     0.548       0.000    0.452
 F5 {"K": 5, "X": 0.04}   2r     718      51.3       -0.077    -0.527      -0.168       -0.150      -0.062       -0.050   0.0494   -3.40     37.5     -8.60        0.29       -25.1     626      44.7     -2.76     0.571       0.227    0.202
                  F7 {}   2r     437      31.2       -0.134    -0.361      -0.190       -0.184      -0.149       -0.146   0.0515   -3.69     39.4     -5.93        0.29       -22.7     374      26.7     -3.98     0.362       0.098    0.540
 F5 {"K": 3, "X": 0.04}   2r     784      56.0       -0.100    -0.578      -0.205       -0.181      -0.127       -0.114   0.0497   -4.13     33.7    -11.50        0.29       -33.3     683      48.8     -6.22     0.619       0.279    0.102
 F5 {"K": 3, "X": 0.02} hold     683      48.8       -0.075    -0.762      -0.208       -0.208      -0.157       -0.157   0.0840   -2.48     24.3    -10.16        0.14       -35.1     311      22.2     -3.49     0.729       0.000    0.271
 F5 {"K": 5, "X": 0.04} hold     559      39.9       -0.129    -0.588      -0.210       -0.210      -0.056       -0.056   0.0659   -3.19     32.6     -8.40        0.21       -23.0     496      35.4     -1.98     0.610       0.000    0.390
 F5 {"K": 3, "X": 0.02}   2r     815      58.2       -0.064    -0.660      -0.214       -0.177      -0.174       -0.158   0.0506   -4.23     35.0    -12.47        0.07       -39.2     351      25.1     -4.37     0.636       0.306    0.059
 F5 {"K": 3, "X": 0.06}   2r     708      50.6       -0.129    -0.554      -0.217       -0.198      -0.205       -0.196   0.0487   -4.45     34.5    -10.96        0.00       -31.1     625      44.6     -9.15     0.578       0.227    0.195
         F2 {"P": 0.05} hold     676      48.3       -0.072    -0.877      -0.225       -0.225      -0.104       -0.104   0.1183   -1.90     17.3    -10.86        0.43       -43.5     165      11.8     -1.23     0.820       0.000    0.180
F3 {"M": 30, "P": 0.05}   2r     720      51.4       -0.073    -0.706      -0.231       -0.190      -0.142       -0.129   0.0553   -4.18     34.7    -11.88        0.07       -28.6     209      14.9     -2.12     0.649       0.325    0.026
         F1 {"P": 0.08}   2r     661      47.2       -0.122    -0.705      -0.236       -0.207      -0.125       -0.110   0.0591   -4.00     33.6    -11.16        0.21       -31.4     296      21.1     -2.64     0.657       0.324    0.020
 F5 {"K": 5, "X": 0.02} hold     634      45.3       -0.109    -0.746      -0.239       -0.239      -0.194       -0.194   0.0808   -2.95     25.7    -10.81        0.29       -36.0     442      31.6     -6.14     0.700       0.000    0.300
 F3 {"M": 15, "P": 0.1}   2r     327      23.4       -0.106    -0.660      -0.242       -0.205      -0.185       -0.173   0.0864   -2.80     33.9     -5.65        0.21       -20.1      92       6.6     -1.22     0.661       0.333    0.006
F3 {"M": 15, "P": 0.05}   2r     648      46.3       -0.088    -0.716      -0.244       -0.204      -0.101       -0.086   0.0592   -4.12     34.6    -11.29        0.21       -27.3     187      13.4     -1.35     0.651       0.324    0.025
         F2 {"P": 0.05}   2r     779      55.6       -0.092    -0.806      -0.260       -0.215      -0.187       -0.172   0.0555   -4.69     34.7    -14.48        0.21       -52.5     166      11.9     -2.21     0.650       0.341    0.009
         F1 {"P": 0.12}   2r     300      21.4       -0.180    -0.679      -0.266       -0.244      -0.246       -0.235   0.0851   -3.12     31.0     -5.69        0.14       -17.3     170      12.1     -2.99     0.677       0.300    0.023
         F1 {"P": 0.05}   2r     812      58.0       -0.118    -0.761      -0.270       -0.231      -0.161       -0.147   0.0523   -5.16     33.7    -15.66        0.00       -36.1     397      28.4     -4.57     0.661       0.315    0.023
 F5 {"K": 8, "X": 0.02}   2r     774      55.3       -0.137    -0.717      -0.271       -0.243      -0.216       -0.203   0.0494   -5.49     32.6    -15.01        0.14       -35.7     558      39.9     -8.62     0.632       0.260    0.109
 F5 {"K": 8, "X": 0.02} hold     616      44.0       -0.153    -0.776      -0.280       -0.280      -0.283       -0.283   0.0732   -3.82     27.9    -12.30        0.14       -31.9     475      33.9     -9.62     0.679       0.000    0.321
 F5 {"K": 5, "X": 0.02}   2r     792      56.6       -0.160    -0.749      -0.301       -0.271      -0.261       -0.247   0.0496   -6.07     31.4    -17.03        0.00       -38.3     518      37.0     -9.64     0.650       0.269    0.081
                  F4 {}   2r     649      46.4       -0.214    -0.803      -0.324       -0.300      -0.141       -0.130   0.0577   -5.61     31.4    -15.00        0.07       -31.9     278      19.9     -2.80     0.672       0.284    0.045
                  F4 {} hold     577      41.2       -0.229    -0.875      -0.330       -0.330       0.031        0.031   0.1210   -2.73     18.0    -13.60        0.21       -41.9     268      19.1      0.58     0.790       0.000    0.210
 F3 {"M": 30, "P": 0.1}   2r     454      32.4       -0.194    -0.782      -0.333       -0.298      -0.250       -0.239   0.0703   -4.73     30.6    -10.78        0.14       -29.2     117       8.4     -2.09     0.694       0.302    0.004
         F1 {"P": 0.05} hold     727      51.9       -0.204    -0.947      -0.340       -0.340      -0.243       -0.243   0.0964   -3.53     18.6    -17.67        0.21       -49.4     373      26.6     -6.47     0.802       0.000    0.198
         F1 {"P": 0.08} hold     599      42.8       -0.239    -0.899      -0.345       -0.345      -0.261       -0.261   0.1229   -2.81     16.7    -14.77        0.21       -68.9     290      20.7     -5.42     0.818       0.000    0.182
         F2 {"P": 0.08}   2r     617      44.1       -0.232    -0.923      -0.369       -0.336      -0.225       -0.210   0.0619   -5.96     31.1    -16.26        0.00       -30.2     137       9.8     -2.20     0.684       0.306    0.010
         F2 {"P": 0.12}   2r     276      19.7       -0.284    -0.896      -0.394       -0.365      -0.513       -0.503   0.0938   -4.20     30.4     -7.76        0.14       -16.5     104       7.4     -3.81     0.688       0.297    0.014
         F2 {"P": 0.12} hold     270      19.3       -0.320    -1.014      -0.423       -0.423      -0.536       -0.536   0.1962   -2.15     13.0     -8.15        0.21       -28.2     104       7.4     -3.98     0.856       0.000    0.144
         F1 {"P": 0.12} hold     287      20.5       -0.374    -0.942      -0.457       -0.457      -0.340       -0.340   0.1459   -3.13     13.2     -9.36        0.14       -23.2     169      12.1     -4.10     0.850       0.000    0.150
```


---

## A1 — the re-gate

## contract (c) corrected

G1 (TRAIN mean>0, t>=2, >=5 tpw): **0 of 52 pass**
old score4 gate (TRAIN >= +10R/week): 0 of 52 pass

TRAIN cells with a POSITIVE mean net R (9 of 52), ranked by t:

                   key exit  corr_n  corr_tpw  corr_meanR  corr_se  corr_t  corr_WR  corr_wkR  corr_green  corr_worst
        F1 {"P": 0.12} hold     833      15.7       0.266   0.1700    1.56     18.7      4.18        0.43       -22.9
        F2 {"P": 0.12} hold     834      15.7       0.427   0.2984    1.43     14.4      6.72        0.47       -23.6
                 F6 {} hold    1242      23.4       0.051   0.0380    1.34     44.4      1.20        0.57       -12.1
                 F6 {}   2r    1342      25.3       0.028   0.0259    1.06     46.5      0.70        0.57       -11.3
F5 {"K": 8, "X": 0.06} hold    1602      30.2       0.028   0.0494    0.57     38.5      0.85        0.49       -21.0
          F8 {"N": 30} hold    1124      21.2       0.012   0.0235    0.49     48.6      0.25        0.57        -9.6
F5 {"K": 8, "X": 0.04} hold    1862      35.1       0.011   0.0482    0.22     35.3      0.38        0.42       -19.1
          F8 {"N": 30}   2r    1148      21.7       0.005   0.0215    0.22     49.0      0.10        0.57        -9.6
          F8 {"N": 15} hold    1244      23.5       0.004   0.0281    0.13     44.9      0.09        0.47       -13.9

**Nothing clears G1, so G2 is not evaluated and TEST is not read for selection.**

## contract (d) corrected + live liquidity gate

G1 (TRAIN mean>0, t>=2, >=5 tpw): **1 of 52 pass**
old score4 gate (TRAIN >= +10R/week): 0 of 52 pass

TRAIN cells with a POSITIVE mean net R (10 of 52), ranked by t:

                   key exit  gate_n  gate_tpw  gate_meanR  gate_se  gate_t  gate_WR  gate_wkR  gate_green  gate_worst
        F1 {"P": 0.12} hold     509       9.6       0.426   0.2107    2.02     23.0      4.09        0.53       -15.1
                 F6 {} hold    1194      22.5       0.069   0.0353    1.95     45.6      1.55        0.64       -11.7
F5 {"K": 8, "X": 0.06} hold    1500      28.3       0.109   0.0598    1.82     40.4      3.09        0.55       -21.3
        F2 {"P": 0.12} hold     363       6.8       0.656   0.4883    1.34     17.6      4.49        0.38       -13.5
                 F6 {}   2r    1280      24.2       0.034   0.0252    1.34     46.6      0.82        0.58       -11.7
        F2 {"P": 0.08} hold     495       9.3       0.334   0.3516    0.95     17.6      3.12        0.38       -15.6
        F2 {"P": 0.05} hold     557      10.5       0.248   0.3137    0.79     19.7      2.60        0.38       -19.3
          F8 {"N": 30} hold    1125      21.2       0.012   0.0236    0.52     48.6      0.26        0.57        -9.6
          F8 {"N": 15} hold    1245      23.5       0.010   0.0285    0.34     45.0      0.23        0.49       -14.3
          F8 {"N": 30}   2r    1149      21.7       0.005   0.0215    0.25     49.0      0.12        0.57        -9.6

G1 survivors on VAL:

           key exit split  gross_n  gross_tpw  gross_meanR  gross_se  gross_t  gross_WR  gross_wkR  gross_green  gross_worst  s4_n  s4_tpw  s4_meanR  s4_se  s4_t  s4_WR  s4_wkR  s4_green  s4_worst  corr_n  corr_tpw  corr_meanR  corr_se  corr_t  corr_WR  corr_wkR  corr_green  corr_worst  corrp_n  corrp_tpw  corrp_meanR  corrp_se  corrp_t  corrp_WR  corrp_wkR  corrp_green  corrp_worst  mix_stop  mix_target  mix_eod  gate_n  gate_tpw  gate_meanR  gate_se  gate_t  gate_WR  gate_wkR  gate_green  gate_worst  gatep_n  gatep_tpw  gatep_meanR  gatep_se  gatep_t  gatep_WR  gatep_wkR  gatep_green  gatep_worst
F1 {"P": 0.12} hold   VAL      390       17.7       -0.192    0.1414    -1.36      15.6       -3.4         0.36        -22.7   390    17.7     -0.71  0.144 -4.93   14.9  -12.58      0.14     -34.7     390      17.7      -0.279    0.142   -1.96     15.4     -4.94        0.32       -24.7      390       17.7       -0.279     0.142    -1.96      15.4      -4.94         0.32        -24.7     0.838         0.0    0.162     207       9.4      -0.255   0.1698    -1.5     17.4      -2.4        0.18       -13.3      207        9.4       -0.255    0.1698     -1.5      17.4       -2.4         0.18        -13.3

## closest miss per family on TRAIN, contract (c), with its minimum detectable effect (MDE = 2.8 x SE)

fam                     key exit  corr_n  corr_tpw  corr_meanR  corr_se  MDE_R  corr_t  corr_WR  corr_wkR  corr_green
 F1          F1 {"P": 0.12} hold     833      15.7       0.266   0.1700  0.476    1.56     18.7      4.18        0.43
 F2          F2 {"P": 0.12} hold     834      15.7       0.427   0.2984  0.836    1.43     14.4      6.72        0.47
 F6                   F6 {} hold    1242      23.4       0.051   0.0380  0.106    1.34     44.4      1.20        0.57
 F5  F5 {"K": 8, "X": 0.06} hold    1602      30.2       0.028   0.0494  0.138    0.57     38.5      0.85        0.49
 F8            F8 {"N": 30} hold    1124      21.2       0.012   0.0235  0.066    0.49     48.6      0.25        0.57
 F3 F3 {"M": 15, "P": 0.05} hold    1794      33.8      -0.070   0.0946  0.265   -0.74     16.9     -2.36        0.38
 F9          F9 {"G": 0.05} hold    1141      21.5      -0.041   0.0430  0.120   -0.95     37.8     -0.88        0.42
 F7                   F7 {} hold    1343      25.3      -0.177   0.0506  0.142   -3.50     32.0     -4.49        0.19
 F4                   F4 {} hold    1982      37.4      -0.260   0.0736  0.206   -3.54     16.9     -9.73        0.28


---

## A2 — time bands (contract (c), book re-run inside each window)

The >= 10:30 window for the two keys involved in Stage A's only G1 passes. G1 passes: `F6 {}` hold (t 2.08), `F6 {}` 2r (t 2.25) and `F1 {"P":0.12}` hold (t 2.01 at exactly 5.0 trades/week). `F1` 2r is shown for contrast. Every one of them fails G2 on VAL:

```
           key exit  window split  corr_n  corr_tpw  corr_meanR  corr_se  corr_t  corr_WR  corr_wkR  corr_green  corr_worst
F1 {"P": 0.12} hold >=10:30 TRAIN     267       5.0       0.838   0.4169    2.01     23.2      4.22        0.53       -13.1
F1 {"P": 0.12} hold >=10:30   VAL     128       5.8      -0.104   0.2762   -0.38     18.8     -0.60        0.36       -10.3
F1 {"P": 0.12}   2r >=10:30 TRAIN     269       5.1      -0.141   0.0953   -1.48     36.4     -0.72        0.47        -9.5
F1 {"P": 0.12}   2r >=10:30   VAL     129       5.9      -0.214   0.1326   -1.61     32.6     -1.25        0.27        -7.1
         F6 {} hold >=10:30 TRAIN     895      16.9       0.048   0.0232    2.08     48.5      0.82        0.60        -7.4
         F6 {} hold >=10:30   VAL     406      18.5       0.016   0.0321    0.49     48.3      0.29        0.45        -6.8
         F6 {}   2r >=10:30 TRAIN     904      17.1       0.049   0.0217    2.25     49.1      0.83        0.58        -5.9
         F6 {}   2r >=10:30   VAL     410      18.6       0.017   0.0314    0.53     48.5      0.31        0.45        -6.8
```

Full A2 grid (6 keys x 2 exits x 4 windows x 2 splits):

### TRAIN
```
                   key exit      window  corr_n  corr_tpw  gross_meanR  corr_meanR  corr_se  corr_t  corr_WR  corr_wkR  corr_green  corr_worst  gate_meanR  mix_stop  mix_target  mix_eod
        F1 {"P": 0.12}   2r 09:30-10:00     390       7.4       -0.108      -0.208   0.0747   -2.79     32.3     -1.53        0.36        -8.4      -0.008     0.672       0.321    0.008
        F1 {"P": 0.12}   2r     >=10:00     457       8.6       -0.144      -0.212   0.0715   -2.97     33.7     -1.83        0.30       -13.2      -0.133     0.643       0.306    0.050
        F1 {"P": 0.12}   2r     >=10:30     269       5.1       -0.077      -0.141   0.0953   -1.48     36.4     -0.72        0.47        -9.5      -0.040     0.613       0.323    0.063
        F1 {"P": 0.12}   2r         ALL     847      16.0       -0.127      -0.210   0.0517   -4.07     33.1     -3.36        0.21       -14.1      -0.090     0.656       0.313    0.031
        F1 {"P": 0.12} hold 09:30-10:00     387       7.3        0.003      -0.092   0.1662   -0.55     16.0     -0.67        0.30       -15.1       0.229     0.829       0.000    0.171
        F1 {"P": 0.12} hold     >=10:00     454       8.6        0.600       0.536   0.2777    1.93     20.7      4.59        0.51       -14.7       0.530     0.773       0.000    0.227
        F1 {"P": 0.12} hold     >=10:30     267       5.0        0.896       0.838   0.4169    2.01     23.2      4.22        0.53       -13.1       0.774     0.745       0.000    0.255
        F1 {"P": 0.12} hold         ALL     833      15.7        0.344       0.266   0.1700    1.56     18.7      4.18        0.43       -22.9       0.426     0.797       0.000    0.203
F5 {"K": 5, "X": 0.04}   2r 09:30-10:00    1390      26.2       -0.046      -0.137   0.0357   -3.83     36.0     -3.59        0.30       -22.4      -0.137     0.573       0.253    0.175
F5 {"K": 5, "X": 0.04}   2r     >=10:00    2660      50.2       -0.141      -0.230   0.0262   -8.79     33.2    -11.54        0.13       -28.0      -0.145     0.617       0.243    0.140
F5 {"K": 5, "X": 0.04}   2r     >=10:30    2273      42.9       -0.116      -0.202   0.0284   -7.10     34.4     -8.65        0.17       -29.9      -0.145     0.597       0.246    0.157
F5 {"K": 5, "X": 0.04}   2r         ALL    2688      50.7       -0.083      -0.172   0.0259   -6.62     35.0     -8.71        0.21       -44.9      -0.123     0.590       0.249    0.161
F5 {"K": 5, "X": 0.04} hold 09:30-10:00    1296      24.5        0.052      -0.030   0.0586   -0.52     30.1     -0.74        0.43       -23.2      -0.049     0.623       0.000    0.377
F5 {"K": 5, "X": 0.04} hold     >=10:00    2207      41.6       -0.085      -0.168   0.0514   -3.27     26.4     -6.99        0.25       -31.6      -0.165     0.677       0.000    0.323
F5 {"K": 5, "X": 0.04} hold     >=10:30    1890      35.7       -0.102      -0.182   0.0461   -3.94     28.1     -6.48        0.32       -26.3      -0.145     0.654       0.000    0.346
F5 {"K": 5, "X": 0.04} hold         ALL    2139      40.4        0.019      -0.063   0.0504   -1.24     29.4     -2.52        0.42       -38.5      -0.098     0.635       0.000    0.365
                 F6 {}   2r 09:30-10:00     847      16.0        0.060       0.018   0.0362    0.50     45.8      0.29        0.57       -11.0       0.026     0.312       0.142    0.547
                 F6 {}   2r     >=10:00    1044      19.7        0.046       0.027   0.0237    1.12     46.1      0.52        0.49        -6.7       0.025     0.134       0.051    0.815
                 F6 {}   2r     >=10:30     904      17.1        0.066       0.049   0.0217    2.25     49.1      0.83        0.58        -5.9       0.048     0.075       0.033    0.892
                 F6 {}   2r         ALL    1342      25.3        0.061       0.028   0.0259    1.06     46.5      0.70        0.57       -11.3       0.034     0.241       0.107    0.651
                 F6 {} hold 09:30-10:00     830      15.7        0.088       0.048   0.0525    0.92     43.4      0.76        0.53       -12.1       0.068     0.330       0.000    0.670
                 F6 {} hold     >=10:00    1018      19.2        0.055       0.036   0.0275    1.31     45.3      0.69        0.55       -10.0       0.033     0.139       0.000    0.861
                 F6 {} hold     >=10:30     895      16.9        0.065       0.048   0.0232    2.08     48.5      0.82        0.60        -7.4       0.048     0.077       0.000    0.923
                 F6 {} hold         ALL    1242      23.4        0.084       0.051   0.0380    1.34     44.4      1.20        0.57       -12.1       0.069     0.262       0.000    0.738
          F8 {"N": 15}   2r 09:30-10:00    1011      19.1        0.020      -0.008   0.0283   -0.29     45.5     -0.16        0.45       -12.2      -0.006     0.240       0.079    0.681
          F8 {"N": 15}   2r     >=10:00    1296      24.5       -0.011      -0.035   0.0238   -1.47     44.4     -0.86        0.45       -12.4      -0.034     0.228       0.070    0.702
          F8 {"N": 15}   2r     >=10:30    1194      22.5        0.002      -0.021   0.0239   -0.88     43.7     -0.47        0.45       -11.0      -0.017     0.190       0.064    0.745
          F8 {"N": 15}   2r         ALL    1326      25.0        0.023      -0.004   0.0244   -0.16     45.8     -0.10        0.47       -12.6      -0.003     0.227       0.080    0.693
          F8 {"N": 15} hold 09:30-10:00    1010      19.1        0.024      -0.002   0.0307   -0.08     44.7     -0.05        0.45       -12.2       0.004     0.244       0.000    0.756
          F8 {"N": 15} hold     >=10:00    1222      23.1       -0.008      -0.032   0.0271   -1.17     43.5     -0.73        0.45       -13.8      -0.025     0.232       0.000    0.768
          F8 {"N": 15} hold     >=10:30    1133      21.4        0.008      -0.015   0.0281   -0.52     43.2     -0.31        0.42       -11.0      -0.006     0.192       0.000    0.808
          F8 {"N": 15} hold         ALL    1244      23.5        0.030       0.004   0.0281    0.13     44.9      0.09        0.47       -13.9       0.010     0.233       0.000    0.767
          F8 {"N": 30}   2r 09:30-10:00       0       NaN          NaN         NaN      NaN     NaN      NaN       NaN         NaN         NaN         NaN       NaN         NaN      NaN
          F8 {"N": 30}   2r     >=10:00    1148      21.7        0.025       0.005   0.0215    0.22     49.0      0.10        0.57        -9.6       0.005     0.150       0.036    0.814
          F8 {"N": 30}   2r     >=10:30    1114      21.0        0.025       0.005   0.0220    0.24     45.1      0.11        0.45        -7.6       0.005     0.134       0.044    0.822
          F8 {"N": 30}   2r         ALL    1148      21.7        0.025       0.005   0.0215    0.22     49.0      0.10        0.57        -9.6       0.005     0.150       0.036    0.814
          F8 {"N": 30} hold 09:30-10:00       0       NaN          NaN         NaN      NaN     NaN      NaN       NaN         NaN         NaN         NaN       NaN         NaN      NaN
          F8 {"N": 30} hold     >=10:00    1124      21.2        0.032       0.012   0.0235    0.49     48.6      0.25        0.57        -9.6       0.012     0.151       0.000    0.849
          F8 {"N": 30} hold     >=10:30    1084      20.5        0.039       0.020   0.0250    0.80     45.0      0.41        0.51        -7.6       0.020     0.133       0.000    0.867
          F8 {"N": 30} hold         ALL    1124      21.2        0.032       0.012   0.0235    0.49     48.6      0.25        0.57        -9.6       0.012     0.151       0.000    0.849
           F8 {"N": 5}   2r 09:30-10:00    1131      21.3       -0.024      -0.058   0.0310   -1.86     42.2     -1.23        0.43       -11.8      -0.046     0.368       0.119    0.513
           F8 {"N": 5}   2r     >=10:00    1495      28.2       -0.046      -0.081   0.0272   -2.99     41.8     -2.29        0.38       -14.4      -0.041     0.334       0.110    0.555
           F8 {"N": 5}   2r     >=10:30    1371      25.9       -0.080      -0.116   0.0271   -4.30     40.2     -3.01        0.34       -20.6      -0.084     0.327       0.093    0.580
           F8 {"N": 5}   2r         ALL    1589      30.0       -0.022      -0.056   0.0259   -2.15     42.2     -1.67        0.40       -14.0      -0.045     0.346       0.118    0.536
           F8 {"N": 5} hold 09:30-10:00    1110      20.9       -0.001      -0.033   0.0385   -0.85     40.9     -0.69        0.45       -14.5      -0.021     0.380       0.000    0.620
           F8 {"N": 5} hold     >=10:00    1366      25.8        0.006      -0.028   0.0436   -0.64     39.8     -0.72        0.47       -13.5      -0.010     0.350       0.000    0.650
           F8 {"N": 5} hold     >=10:30    1280      24.2       -0.068      -0.103   0.0352   -2.92     38.9     -2.48        0.38       -20.6      -0.082     0.340       0.000    0.660
           F8 {"N": 5} hold         ALL    1427      26.9       -0.002      -0.034   0.0341   -1.00     40.4     -0.92        0.49       -18.6      -0.024     0.362       0.000    0.638
```

### VAL
```
                   key exit      window  corr_n  corr_tpw  gross_meanR  corr_meanR  corr_se  corr_t  corr_WR  corr_wkR  corr_green  corr_worst  gate_meanR  mix_stop  mix_target  mix_eod
        F1 {"P": 0.12}   2r 09:30-10:00     214       9.7       -0.270      -0.379   0.0955   -3.97     27.6     -3.69        0.18       -10.8      -0.466     0.724       0.262    0.014
        F1 {"P": 0.12}   2r     >=10:00     184       8.4       -0.104      -0.174   0.1118   -1.56     33.7     -1.46        0.36       -12.7      -0.104     0.647       0.326    0.027
        F1 {"P": 0.12}   2r     >=10:30     129       5.9       -0.143      -0.214   0.1326   -1.61     32.6     -1.25        0.27        -7.1      -0.146     0.651       0.310    0.039
        F1 {"P": 0.12}   2r         ALL     398      18.1       -0.193      -0.284   0.0730   -3.90     30.4     -5.15        0.23       -16.3      -0.245     0.688       0.291    0.020
        F1 {"P": 0.12} hold 09:30-10:00     212       9.6       -0.144      -0.248   0.1951   -1.27     15.1     -2.39        0.32       -16.5      -0.338     0.849       0.000    0.151
        F1 {"P": 0.12} hold     >=10:00     182       8.3       -0.243      -0.310   0.2038   -1.52     15.9     -2.57        0.36       -15.8      -0.202     0.824       0.000    0.176
        F1 {"P": 0.12} hold     >=10:30     128       5.8       -0.037      -0.104   0.2762   -0.38     18.8     -0.60        0.36       -10.3      -0.025     0.789       0.000    0.211
        F1 {"P": 0.12} hold         ALL     390      17.7       -0.192      -0.279   0.1420   -1.96     15.4     -4.94        0.32       -24.7      -0.255     0.838       0.000    0.162
F5 {"K": 5, "X": 0.04}   2r 09:30-10:00     572      26.0        0.087      -0.009   0.0562   -0.16     42.5     -0.23        0.45       -15.3       0.007     0.516       0.273    0.212
F5 {"K": 5, "X": 0.04}   2r     >=10:00    1098      49.9        0.007      -0.088   0.0423   -2.07     38.3     -4.37        0.36       -18.1      -0.093     0.579       0.293    0.128
F5 {"K": 5, "X": 0.04}   2r     >=10:30    1010      45.9        0.004      -0.092   0.0444   -2.07     38.4     -4.22        0.27       -20.8      -0.115     0.565       0.294    0.141
F5 {"K": 5, "X": 0.04}   2r         ALL    1073      48.8        0.016      -0.078   0.0415   -1.88     40.1     -3.82        0.41       -24.3      -0.078     0.544       0.267    0.189
F5 {"K": 5, "X": 0.04} hold 09:30-10:00     526      23.9        0.158       0.073   0.0963    0.76     35.6      1.76        0.45       -19.1       0.009     0.576       0.000    0.424
F5 {"K": 5, "X": 0.04} hold     >=10:00     843      38.3        0.039      -0.046   0.0722   -0.64     31.1     -1.78        0.41       -29.3      -0.105     0.633       0.000    0.367
F5 {"K": 5, "X": 0.04} hold     >=10:30     774      35.2        0.056      -0.032   0.0739   -0.43     32.3     -1.11        0.50       -21.4      -0.099     0.612       0.000    0.388
F5 {"K": 5, "X": 0.04} hold         ALL     819      37.2        0.039      -0.047   0.0726   -0.65     33.1     -1.75        0.41       -33.4      -0.065     0.603       0.000    0.397
                 F6 {}   2r 09:30-10:00     410      18.6        0.138       0.095   0.0552    1.73     48.3      1.78        0.68        -9.9       0.108     0.317       0.171    0.512
                 F6 {}   2r     >=10:00     448      20.4        0.089       0.068   0.0349    1.94     50.2      1.38        0.59        -4.3       0.064     0.121       0.049    0.830
                 F6 {}   2r     >=10:30     410      18.6        0.036       0.017   0.0314    0.53     48.5      0.31        0.45        -6.8       0.012     0.095       0.024    0.880
                 F6 {}   2r         ALL     607      27.6        0.106       0.071   0.0407    1.73     48.8      1.95        0.64        -9.2       0.089     0.257       0.125    0.618
                 F6 {} hold 09:30-10:00     394      17.9        0.239       0.200   0.0865    2.31     47.0      3.58        0.77       -10.0       0.164     0.332       0.000    0.668
                 F6 {} hold     >=10:00     436      19.8        0.122       0.101   0.0550    1.84     50.7      2.00        0.45        -4.3       0.092     0.126       0.000    0.874
                 F6 {} hold     >=10:30     406      18.5        0.035       0.016   0.0321    0.49     48.3      0.29        0.45        -6.8       0.010     0.096       0.000    0.904
                 F6 {} hold         ALL     536      24.4        0.197       0.163   0.0663    2.46     47.9      3.97        0.68        -8.1       0.145     0.284       0.000    0.716
          F8 {"N": 15}   2r 09:30-10:00     412      18.7       -0.034      -0.064   0.0413   -1.55     43.9     -1.20        0.45        -9.7      -0.063     0.243       0.046    0.711
          F8 {"N": 15}   2r     >=10:00     481      21.9        0.063       0.038   0.0353    1.09     47.6      0.84        0.64        -6.0       0.041     0.166       0.040    0.794
          F8 {"N": 15}   2r     >=10:30     484      22.0        0.005      -0.020   0.0366   -0.54     44.4     -0.43        0.50        -6.0      -0.015     0.178       0.048    0.775
          F8 {"N": 15}   2r         ALL     517      23.5       -0.052      -0.081   0.0355   -2.28     42.7     -1.90        0.41       -11.4      -0.083     0.232       0.041    0.727
          F8 {"N": 15} hold 09:30-10:00     411      18.7       -0.031      -0.060   0.0433   -1.39     43.8     -1.12        0.50        -9.7      -0.055     0.246       0.000    0.754
          F8 {"N": 15} hold     >=10:00     468      21.3        0.068       0.044   0.0389    1.13     47.4      0.94        0.64        -6.0       0.047     0.169       0.000    0.831
          F8 {"N": 15} hold     >=10:30     470      21.4        0.013      -0.011   0.0412   -0.27     43.6     -0.24        0.45        -6.3      -0.002     0.183       0.000    0.817
          F8 {"N": 15} hold         ALL     503      22.9       -0.051      -0.079   0.0381   -2.08     42.3     -1.81        0.41       -11.4      -0.079     0.239       0.000    0.761
          F8 {"N": 30}   2r 09:30-10:00       0       NaN          NaN         NaN      NaN     NaN      NaN       NaN         NaN         NaN         NaN       NaN         NaN      NaN
          F8 {"N": 30}   2r     >=10:00     463      21.0        0.015      -0.006   0.0324   -0.17     49.0     -0.12        0.45        -6.8      -0.007     0.132       0.026    0.842
          F8 {"N": 30}   2r     >=10:30     453      20.6        0.011      -0.010   0.0323   -0.31     46.1     -0.20        0.41        -5.0      -0.011     0.119       0.022    0.859
          F8 {"N": 30}   2r         ALL     463      21.0        0.015      -0.006   0.0324   -0.17     49.0     -0.12        0.45        -6.8      -0.007     0.132       0.026    0.842
          F8 {"N": 30} hold 09:30-10:00       0       NaN          NaN         NaN      NaN     NaN      NaN       NaN         NaN         NaN         NaN       NaN         NaN      NaN
          F8 {"N": 30} hold     >=10:00     456      20.7        0.014      -0.006   0.0336   -0.19     48.7     -0.13        0.45        -6.8      -0.008     0.134       0.000    0.866
          F8 {"N": 30} hold     >=10:30     446      20.3        0.037       0.016   0.0408    0.38     46.4      0.32        0.41        -5.0       0.014     0.121       0.000    0.879
          F8 {"N": 30} hold         ALL     456      20.7        0.014      -0.006   0.0336   -0.19     48.7     -0.13        0.45        -6.8      -0.008     0.134       0.000    0.866
           F8 {"N": 5}   2r 09:30-10:00     467      21.2        0.054       0.018   0.0500    0.35     44.8      0.38        0.50       -10.7       0.008     0.353       0.137    0.510
           F8 {"N": 5}   2r     >=10:00     583      26.5        0.104       0.068   0.0431    1.59     48.9      1.81        0.59       -10.7       0.076     0.283       0.132    0.585
           F8 {"N": 5}   2r     >=10:30     552      25.1        0.024      -0.012   0.0428   -0.29     46.2     -0.31        0.50       -10.1       0.019     0.272       0.114    0.614
           F8 {"N": 5}   2r         ALL     673      30.6        0.024      -0.012   0.0403   -0.31     43.8     -0.38        0.50       -12.8      -0.010     0.331       0.122    0.547
           F8 {"N": 5} hold 09:30-10:00     463      21.0        0.103       0.070   0.0696    1.00     43.8      1.46        0.59       -11.3       0.057     0.363       0.000    0.637
           F8 {"N": 5} hold     >=10:00     533      24.2        0.125       0.091   0.0584    1.57     46.0      2.22        0.55        -7.7       0.108     0.304       0.000    0.696
           F8 {"N": 5} hold     >=10:30     506      23.0        0.047       0.013   0.0534    0.24     44.7      0.30        0.45        -9.9       0.029     0.285       0.000    0.715
           F8 {"N": 5} hold         ALL     600      27.3        0.060       0.025   0.0582    0.44     42.5      0.69        0.50       -11.7       0.026     0.363       0.000    0.637
```


---

## A3 — intraday market state at the entry minute (entries >= 10:00)

104 buckets, TRAIN->VAL sign agreement **0.51**. By feature:

```
      mean  size
feat            
brd   0.54    24
iwm   0.52    40
spy   0.48    40
```

```
           key exit feat  bucket  n_TRAIN  meanR_TRAIN  t_TRAIN  n_VAL  meanR_VAL  t_VAL  sign_agree
F1 {"P": 0.12}   2r  brd  T1 low    155.0       -0.207    -1.73   42.0     -0.358  -1.47        True
F1 {"P": 0.12}   2r  brd  T2 mid    150.0        0.006     0.05   50.0     -0.131  -0.63       False
F1 {"P": 0.12}   2r  brd T3 high    152.0       -0.433    -3.28   92.0     -0.113  -0.71        True
F1 {"P": 0.12}   2r  iwm  T1 low    152.0       -0.109    -0.87   62.0     -0.313  -1.66        True
F1 {"P": 0.12}   2r  iwm  T2 mid    152.0       -0.329    -2.93   52.0     -0.055  -0.25        True
F1 {"P": 0.12}   2r  iwm T3 high    153.0       -0.199    -1.49   70.0     -0.140  -0.76        True
F1 {"P": 0.12}   2r  iwm  sign +    241.0       -0.252    -2.53   96.0     -0.192  -1.26        True
F1 {"P": 0.12}   2r  iwm  sign -    216.0       -0.167    -1.63   88.0     -0.155  -0.94        True
F1 {"P": 0.12}   2r  spy  T1 low    152.0       -0.047    -0.39   58.0     -0.329  -1.69        True
F1 {"P": 0.12}   2r  spy  T2 mid    152.0       -0.382    -3.26   56.0     -0.380  -2.00        True
F1 {"P": 0.12}   2r  spy T3 high    153.0       -0.207    -1.57   70.0      0.119   0.63       False
F1 {"P": 0.12}   2r  spy  sign +    244.0       -0.286    -2.91  110.0     -0.062  -0.42        True
F1 {"P": 0.12}   2r  spy  sign -    213.0       -0.128    -1.23   74.0     -0.341  -2.01        True
F1 {"P": 0.12} hold  brd  T1 low    154.0        0.483     1.02   42.0     -0.293  -0.54       False
F1 {"P": 0.12} hold  brd  T2 mid    149.0        1.447     2.26   50.0     -0.283  -0.78       False
F1 {"P": 0.12} hold  brd T3 high    151.0       -0.308    -1.27   90.0     -0.334  -1.28        True
F1 {"P": 0.12} hold  iwm  T1 low    151.0        0.792     1.76   61.0     -0.184  -0.43       False
F1 {"P": 0.12} hold  iwm  T2 mid    151.0        0.548     1.04   52.0     -0.731  -2.86       False
F1 {"P": 0.12} hold  iwm T3 high    152.0        0.271     0.59   69.0     -0.105  -0.32       False
F1 {"P": 0.12} hold  iwm  sign +    239.0        0.617     1.41   94.0     -0.230  -0.86       False
F1 {"P": 0.12} hold  iwm  sign -    215.0        0.446     1.36   88.0     -0.396  -1.28       False
F1 {"P": 0.12} hold  spy  T1 low    151.0        0.324     1.01   58.0     -0.289  -0.65       False
F1 {"P": 0.12} hold  spy  T2 mid    151.0        0.111     0.31   54.0     -0.796  -4.08       False
F1 {"P": 0.12} hold  spy T3 high    152.0        1.169     1.74   70.0      0.047   0.13        True
F1 {"P": 0.12} hold  spy  sign +    242.0        0.628     1.42  108.0     -0.268  -1.13       False
F1 {"P": 0.12} hold  spy  sign -    212.0        0.432     1.36   74.0     -0.372  -1.02       False
         F6 {}   2r  brd  T1 low    348.0        0.036     0.83   84.0      0.071   0.87        True
         F6 {}   2r  brd  T2 mid    348.0        0.044     1.10  124.0      0.093   1.42        True
         F6 {}   2r  brd T3 high    348.0        0.000     0.00  240.0      0.054   1.12       False
         F6 {}   2r  iwm  T1 low    348.0       -0.001    -0.04  166.0      0.147   2.40       False
         F6 {}   2r  iwm  T2 mid    348.0       -0.003    -0.07  124.0     -0.069  -1.10        True
         F6 {}   2r  iwm T3 high    348.0        0.084     2.00  158.0      0.091   1.64        True
         F6 {}   2r  iwm  sign +    543.0        0.071     2.08  231.0      0.041   0.90        True
         F6 {}   2r  iwm  sign -    501.0       -0.022    -0.69  217.0      0.097   1.80       False
         F6 {}   2r  spy  T1 low    348.0        0.029     0.72  135.0      0.014   0.24        True
         F6 {}   2r  spy  T2 mid    348.0       -0.016    -0.39  151.0      0.115   1.81       False
         F6 {}   2r  spy T3 high    348.0        0.066     1.63  162.0      0.068   1.19        True
         F6 {}   2r  spy  sign +    565.0        0.060     1.82  253.0      0.110   2.32        True
         F6 {}   2r  spy  sign -    479.0       -0.013    -0.38  195.0      0.014   0.26       False
         F6 {} hold  brd  T1 low    340.0        0.054     1.01   82.0      0.067   0.83        True
         F6 {} hold  brd  T2 mid    339.0        0.031     0.73  121.0      0.048   0.76        True
         F6 {} hold  brd T3 high    339.0        0.024     0.51  233.0      0.141   1.50        True
         F6 {} hold  iwm  T1 low    339.0       -0.023    -0.56  162.0      0.125   1.93       False
         F6 {} hold  iwm  T2 mid    339.0        0.024     0.45  120.0     -0.031  -0.41       False
         F6 {} hold  iwm T3 high    340.0        0.107     2.26  154.0      0.179   1.41        True
         F6 {} hold  iwm  sign +    525.0        0.094     2.40  223.0      0.125   1.34        True
         F6 {} hold  iwm  sign -    493.0       -0.026    -0.68  213.0      0.076   1.36       False
         F6 {} hold  spy  T1 low    339.0        0.015     0.34  131.0     -0.017  -0.29       False
         F6 {} hold  spy  T2 mid    340.0       -0.012    -0.24  146.0      0.252   1.78       False
         F6 {} hold  spy T3 high    339.0        0.106     2.20  159.0      0.060   1.06        True
         F6 {} hold  spy  sign +    547.0        0.082     2.15  246.0      0.200   2.25        True
         F6 {} hold  spy  sign -    471.0       -0.017    -0.43  190.0     -0.027  -0.55        True
  F8 {"N": 30}   2r  brd  T1 low    383.0       -0.007    -0.20   98.0     -0.042  -0.70        True
  F8 {"N": 30}   2r  brd  T2 mid    382.0       -0.003    -0.09  142.0      0.051   0.80       False
  F8 {"N": 30}   2r  brd T3 high    383.0        0.025     0.67  223.0     -0.026  -0.56       False
  F8 {"N": 30}   2r  iwm  T1 low    382.0        0.002     0.06  188.0     -0.006  -0.11       False
  F8 {"N": 30}   2r  iwm  T2 mid    383.0       -0.002    -0.06  116.0     -0.087  -1.33        True
  F8 {"N": 30}   2r  iwm T3 high    382.0        0.017     0.47  159.0      0.054   0.93        True
  F8 {"N": 30}   2r  iwm  sign +    542.0        0.008     0.26  226.0      0.022   0.46        True
  F8 {"N": 30}   2r  iwm  sign -    605.0        0.004     0.12  237.0     -0.032  -0.75       False
  F8 {"N": 30}   2r  spy  T1 low    383.0        0.076     1.98  161.0     -0.152  -2.82       False
  F8 {"N": 30}   2r  spy  T2 mid    385.0       -0.105    -3.13  125.0      0.050   0.81       False
  F8 {"N": 30}   2r  spy T3 high    380.0        0.044     1.12  177.0      0.088   1.68        True
  F8 {"N": 30}   2r  spy  sign +    589.0        0.014     0.46  265.0      0.075   1.77        True
  F8 {"N": 30}   2r  spy  sign -    559.0       -0.005    -0.16  198.0     -0.113  -2.28        True
  F8 {"N": 30} hold  brd  T1 low    375.0       -0.006    -0.16   98.0     -0.042  -0.70        True
  F8 {"N": 30} hold  brd  T2 mid    374.0        0.006     0.14  137.0      0.062   0.88        True
  F8 {"N": 30} hold  brd T3 high    375.0        0.035     0.84  221.0     -0.033  -0.70       False
  F8 {"N": 30} hold  iwm  T1 low    375.0        0.021     0.52  181.0     -0.015  -0.29       False
  F8 {"N": 30} hold  iwm  T2 mid    374.0       -0.005    -0.13  120.0     -0.071  -1.10        True
  F8 {"N": 30} hold  iwm T3 high    374.0        0.021     0.51  155.0      0.053   0.86        True
  F8 {"N": 30} hold  iwm  sign +    524.0        0.006     0.18  221.0      0.026   0.50        True
  F8 {"N": 30} hold  iwm  sign -    599.0        0.018     0.56  235.0     -0.036  -0.84       False
  F8 {"N": 30} hold  spy  T1 low    376.0        0.102     2.33  156.0     -0.160  -2.92       False
  F8 {"N": 30} hold  spy  T2 mid    375.0       -0.125    -3.89  129.0      0.065   1.04       False
  F8 {"N": 30} hold  spy T3 high    373.0        0.057     1.29  171.0      0.080   1.44        True
  F8 {"N": 30} hold  spy  sign +    574.0        0.009     0.26  259.0      0.072   1.61        True
  F8 {"N": 30} hold  spy  sign -    550.0        0.015     0.43  197.0     -0.109  -2.17       False
   F8 {"N": 5}   2r  brd  T1 low    499.0       -0.105    -2.19  131.0      0.068   0.73       False
   F8 {"N": 5}   2r  brd  T2 mid    498.0       -0.097    -2.04  171.0      0.030   0.37       False
   F8 {"N": 5}   2r  brd T3 high    498.0       -0.042    -0.91  281.0      0.092   1.51       False
   F8 {"N": 5}   2r  iwm  T1 low    498.0       -0.129    -2.60  227.0      0.031   0.44       False
   F8 {"N": 5}   2r  iwm  T2 mid    498.0       -0.121    -2.71  173.0      0.075   0.91       False
   F8 {"N": 5}   2r  iwm T3 high    499.0        0.006     0.14  183.0      0.109   1.48        True
   F8 {"N": 5}   2r  iwm  sign +    737.0       -0.020    -0.52  263.0      0.139   2.16       False
   F8 {"N": 5}   2r  iwm  sign -    758.0       -0.141    -3.65  320.0      0.010   0.18       False
   F8 {"N": 5}   2r  spy  T1 low    498.0       -0.129    -2.60  205.0     -0.077  -1.04        True
   F8 {"N": 5}   2r  spy  T2 mid    498.0       -0.059    -1.29  167.0      0.089   1.15       False
   F8 {"N": 5}   2r  spy T3 high    499.0       -0.056    -1.22  211.0      0.194   2.71       False
   F8 {"N": 5}   2r  spy  sign +    777.0       -0.025    -0.69  312.0      0.170   2.95       False
   F8 {"N": 5}   2r  spy  sign -    718.0       -0.142    -3.52  271.0     -0.048  -0.75        True
   F8 {"N": 5} hold  brd  T1 low    456.0       -0.023    -0.26  123.0      0.234   1.48       False
   F8 {"N": 5} hold  brd  T2 mid    455.0       -0.063    -1.07  158.0     -0.073  -0.89        True
   F8 {"N": 5} hold  brd T3 high    455.0        0.002     0.02  252.0      0.125   1.54        True
   F8 {"N": 5} hold  iwm  T1 low    455.0       -0.134    -2.12  210.0     -0.047  -0.58        True
   F8 {"N": 5} hold  iwm  T2 mid    455.0        0.006     0.07  159.0      0.220   1.71        True
   F8 {"N": 5} hold  iwm T3 high    456.0        0.044     0.67  164.0      0.144   1.48        True
   F8 {"N": 5} hold  iwm  sign +    670.0        0.039     0.67  238.0      0.205   2.46        True
   F8 {"N": 5} hold  iwm  sign -    696.0       -0.093    -1.44  295.0      0.000   0.00       False
   F8 {"N": 5} hold  spy  T1 low    455.0        0.035     0.37  193.0     -0.074  -0.68       False
   F8 {"N": 5} hold  spy  T2 mid    455.0       -0.104    -1.91  153.0      0.082   0.86       False
   F8 {"N": 5} hold  spy T3 high    456.0       -0.015    -0.20  187.0      0.270   2.87       False
   F8 {"N": 5} hold  spy  sign +    709.0       -0.004    -0.07  280.0      0.222   2.98       False
   F8 {"N": 5} hold  spy  sign -    657.0       -0.054    -0.78  253.0     -0.053  -0.59        True
```


---

## A4 support — decomposition of the correction

                    key exit    n  gross     s4  step1_spread_table  step2_entry_025  step3_target_charge  corrected  total_delta
         F2 {"P": 0.12} hold  834  0.524 -0.162               0.520            0.069                0.000      0.427        0.589
         F1 {"P": 0.12} hold  833  0.344 -0.205               0.415            0.056                0.000      0.266        0.471
                  F6 {} hold 1242  0.084 -0.111               0.132            0.029                0.000      0.051        0.162
 F5 {"K": 8, "X": 0.06} hold 1602  0.082 -0.257               0.240            0.045                0.000      0.028        0.285
                  F6 {}   2r 1342  0.061 -0.121               0.125            0.029               -0.005      0.028        0.148
           F8 {"N": 30} hold 1124  0.032 -0.106               0.097            0.021                0.000      0.012        0.118
 F5 {"K": 8, "X": 0.04} hold 1862  0.084 -0.342               0.293            0.059                0.000      0.011        0.352
           F8 {"N": 30}   2r 1148  0.025 -0.112               0.096            0.021               -0.001      0.005        0.116
           F8 {"N": 15} hold 1244  0.030 -0.126               0.105            0.025                0.000      0.004        0.130
           F8 {"N": 15}   2r 1326  0.023 -0.133               0.107            0.025               -0.002     -0.004        0.129
         F2 {"P": 0.08} hold 1861  0.095 -0.668               0.559            0.085                0.000     -0.025        0.643
         F2 {"P": 0.05} hold 2601  0.112 -0.724               0.594            0.101                0.000     -0.029        0.695
            F8 {"N": 5} hold 1427 -0.002 -0.200               0.138            0.028                0.000     -0.034        0.166
         F9 {"G": 0.05} hold 1141 -0.011 -0.184               0.118            0.026                0.000     -0.041        0.143
            F8 {"N": 5}   2r 1589 -0.022 -0.217               0.137            0.028               -0.004     -0.056        0.161
 F5 {"K": 8, "X": 0.06}   2r 1941  0.000 -0.347               0.251            0.047               -0.011     -0.060        0.287
         F1 {"P": 0.08} hold 2014  0.039 -0.637               0.501            0.073                0.000     -0.063        0.574
 F5 {"K": 5, "X": 0.06} hold 1879  0.000 -0.398               0.284            0.051                0.000     -0.063        0.335
 F5 {"K": 5, "X": 0.04} hold 2139  0.019 -0.475               0.350            0.063                0.000     -0.063        0.413
F3 {"M": 15, "P": 0.05} hold 1794  0.066 -0.655               0.488            0.097                0.000     -0.070        0.585
         F9 {"G": 0.05}   2r 1191 -0.052 -0.218               0.113            0.026               -0.004     -0.083        0.135
 F5 {"K": 3, "X": 0.06} hold 1989 -0.030 -0.492               0.328            0.058                0.000     -0.105        0.386
F3 {"M": 30, "P": 0.05} hold 2033  0.028 -0.710               0.503            0.098                0.000     -0.109        0.601
 F5 {"K": 3, "X": 0.04} hold 2248 -0.024 -0.568               0.383            0.070                0.000     -0.116        0.453
 F5 {"K": 8, "X": 0.04}   2r 2350 -0.037 -0.465               0.304            0.059               -0.015     -0.117        0.348
 F5 {"K": 5, "X": 0.02} hold 2418 -0.000 -0.700               0.480            0.095                0.000     -0.126        0.575
 F5 {"K": 5, "X": 0.06}   2r 2339 -0.071 -0.467               0.287            0.052               -0.014     -0.143        0.324
 F3 {"M": 15, "P": 0.1} hold  854 -0.038 -0.694               0.459            0.082                0.000     -0.153        0.541
 F5 {"K": 8, "X": 0.02} hold 2291 -0.037 -0.708               0.460            0.091                0.000     -0.156        0.551
 F5 {"K": 5, "X": 0.04}   2r 2688 -0.083 -0.555               0.340            0.063               -0.019     -0.172        0.384
                  F7 {} hold 1343 -0.132 -0.406               0.190            0.039                0.000     -0.177        0.229
 F5 {"K": 3, "X": 0.04}   2r 2789 -0.086 -0.598               0.363            0.070               -0.023     -0.188        0.410
 F5 {"K": 3, "X": 0.06}   2r 2493 -0.103 -0.551               0.320            0.060               -0.017     -0.188        0.363
         F1 {"P": 0.05} hold 2649 -0.076 -0.851               0.553            0.093                0.000     -0.205        0.646
         F1 {"P": 0.12}   2r  847 -0.127 -0.612               0.366            0.056               -0.020     -0.210        0.402
 F5 {"K": 3, "X": 0.02} hold 2516 -0.091 -0.821               0.502            0.097                0.000     -0.223        0.599
 F5 {"K": 5, "X": 0.02}   2r 2935 -0.098 -0.726               0.431            0.093               -0.032     -0.234        0.492
 F5 {"K": 8, "X": 0.02}   2r 2806 -0.109 -0.729               0.431            0.089               -0.028     -0.237        0.492
                  F7 {}   2r 1448 -0.194 -0.456               0.182            0.038               -0.005     -0.241        0.215
 F3 {"M": 30, "P": 0.1} hold 1170 -0.127 -0.817               0.484            0.085                0.000     -0.248        0.569
                  F4 {} hold 1982 -0.167 -0.838               0.511            0.067                0.000     -0.260        0.578
         F2 {"P": 0.12}   2r  847 -0.167 -0.763               0.451            0.069               -0.026     -0.270        0.493
 F3 {"M": 30, "P": 0.1}   2r 1243 -0.144 -0.750               0.426            0.085               -0.032     -0.271        0.478
         F1 {"P": 0.08}   2r 2178 -0.163 -0.765               0.447            0.073               -0.026     -0.272        0.494
 F3 {"M": 15, "P": 0.1}   2r  890 -0.156 -0.731               0.400            0.082               -0.030     -0.279        0.452
 F5 {"K": 3, "X": 0.02}   2r 2937 -0.138 -0.793               0.448            0.097               -0.033     -0.282        0.512
         F2 {"P": 0.08}   2r 1948 -0.162 -0.832               0.491            0.085               -0.033     -0.289        0.543
                  F4 {}   2r 2105 -0.196 -0.805               0.464            0.067               -0.023     -0.296        0.509
         F1 {"P": 0.05}   2r 2944 -0.161 -0.846               0.485            0.094               -0.033     -0.301        0.545
F3 {"M": 15, "P": 0.05}   2r 2110 -0.164 -0.807               0.433            0.098               -0.035     -0.311        0.496
         F2 {"P": 0.05}   2r 2823 -0.159 -0.903               0.524            0.103               -0.038     -0.313        0.589
F3 {"M": 30, "P": 0.05}   2r 2450 -0.172 -0.832               0.447            0.099               -0.035     -0.321        0.511

means over the 52 cells: {'step1_spread_table': 0.357, 'step2_entry_025': 0.066, 'step3_target_charge': -0.01, 'total_delta': 0.412}

max TRAIN t over the 52 cells under (c): 1.56  (E[max] of 52 independent N(0,1) draws ~ 2.7)
median |t|: 2.56 | cells with mean net R > 0: 9 of 52


---

## Caveats — what these numbers cannot say

1. **The corrected spread is a cell median, not a per-trade quote.** 25 (price band x time band) cells, 50-120 sampled
   signal-minute NBBO medians each, from `build_cost_curve.py`'s stratified sample of THIS candidate pool (its own
   filter was `price >= 5, r_pct >= 0.5`, without the F5-F10 range floor — a slightly wider pool than score4's).
   Within-cell dispersion is uncharged. PLAN §3 H7 Stage B (per-trade NBBO for every booked trade of a surviving cell)
   is the fix and is unchanged by this stage.
2. **The 0.25 entry coefficient is an inference, not a measurement on this population.** It comes from 191 live entries
   in `trades.db` (median 0.0 bps vs the fill-moment quote, 46% at or below the mid) in names that passed live liquidity
   gates the research universe never applied. If the untraded remainder of this universe crosses a full half spread,
   contract (b)'s entry charge is right for those names and every positive cell above disappears. Contract (c) is
   therefore an UPPER bound on what the account keeps, and (b) a lower bound; both are reported side by side for that
   reason.
3. **Everything here inherits the fill convention under audit.** `candidates3.csv` only contains signals whose next-bar
   open came back under level x 1.006, i.e. the population is conditioned on cheap fills (probe_stops §3). H2's resting
   fill (+30% trades) cannot be tested on this file; Stage B rebuilds pass 1 with both fills.
4. **No stop/exit variation.** Stage A re-scores the two exits that exist in `candidates3.csv` (hold, +2R on a close).
   H1's stop x exit cross needs the re-walk in Stage B.
5. **Early-close days are not excluded** (score4 did not exclude them, and this stage reproduces score4's population
   exactly so the difference is attributable to the cost contract alone).
6. **Breadth-so-far is an approximation** and is described as one in the pre-registration: it is breadth inside the
   study's own candidate pool, deduped on (day, symbol, entry_m) across all 26 family-configs, and each candidate's
   `dist_open_pct` is its value at its own entry minute.
7. **TEST.** A0 reprints TEST because `score4_tables.md` already published it for these 52 cells; nothing in Stage A
   selects on it. No cell reached G2, so the pre-registered TEST read (G3), the tail tests, the per-month tables and the
   permutation p were not triggered.

## Cell count

| block | cells looked at |
|---|---|
| A0 | 52 family-config x exit cells x 3 splits x 6 contracts = 936 numbers; **52 decision-relevant** (contract (c), TRAIN) |
| A1 | the same 52 under G1 for contract (c), and 52 again for the secondary contract (d) = 104 gate evaluations; 1 VAL read (the single (d) G1 survivor) |
| A2 | 6 keys x 2 exits x 4 windows x 2 splits = 96 (of which 12 duplicate A0's ALL-window cells, so 36 new window cells per split) |
| A3 | 104 buckets x 2 splits = 208 |
| Stage A total | **~1,240 numbers, 192 pre-declared decision cells** |
| program to date | 52 (score4) + ~100 (probe_stops) + 156 (probe_days) + 192 (Stage A) = **~500 cells** |

## Scripts (all under `research/fuckup_audit/A/`, each run `nice -n 10` with `ulimit -v 1300000`)

| script | what | runtime |
|---|---|---|
| `PREREG.md` | the pre-registration block, frozen before the first run and pasted verbatim into §0 | - |
| `extract_pop_a.py` | chunked scan of the 678 MB `candidates3.csv` -> `pop_a.csv` (446,012 rows = score4's population exactly) | 20 s |
| `acore.py` | ONE definition of the population, the six cost contracts and the book (`run_book(rows, 12, 4)`) | - |
| `a0_score.py` | A0: 26 keys x 2 exits x 3 splits under all six contracts + the score4 parity anchor -> `a0_cells.csv`, `a0_parity.csv`, `a0_tables.md` | 3 min |
| `a1_gate.py` | A1: G1/G2 under (c) and (d), closest miss per family with its MDE -> `a1_gate.md` | 5 s |
| `a2_bands.py` | A2: time bands -> `a2_cells.csv`, `a2_tables.md` | 1 min |
| `a3_state.py` | A3: IWM/SPY open->entry-minute return and breadth-so-far buckets -> `a3_cells.csv`, `a3_tables.md` | 2 min |
| `a4_decomp.py` | A4: decomposition of the score4 -> corrected move -> `a4_decomp.csv/.md` | 1 min |
| `assemble_report.py` | builds this REPORT.md from the pre-registration + the outputs | 5 s |

