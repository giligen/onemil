# Stage D0 — the feature-selection PILOT (H4)

Executed 2026-09-16 per `research/fuckup_audit/PLAN.md` §3 H4 and §4 row D. **Pilot**: it runs the whole H4 machinery
on the features ALREADY on disk, so that Stage D1 can re-run the identical pipeline on `candidates4.csv` (both fills,
signal-bar OHLCV, level history) the moment Stage B finishes. Everything written is under `research/fuckup_audit/D/`;
everything outside it was read only (`data/cache.db`, `etf_1min.db` via `file:...?mode=ro`). No config, service or
order was touched.

---

## A. One page — what D0 found

**A walk-forward feature model does NOT select a positive book out of these raw-flat families. Selection is worth
+0.020 / +0.004 / −0.021 R per trade against the unselected book on the three out-of-sample periods — i.e. zero, with
the sign wandering.** 18 pre-registered cells (3 families x 3 models x 2 selection rules), one fixed hyper-parameter
set, monthly walk-forward refits from 2025-10 to 2026-09, 33 causal features including a fresh 100%-coverage Alpaca
news pull.

| gate | result |
|---|---|
| **G1 (TRAINPRED 2025-10..12)** | **0 of 18** cells pass. Largest t is **1.16** (`F8 {"N":30}` regressor S1, +0.054 R). Because nothing passed G1, G2's bar was not raised. |
| **G2 (VAL 2026-01..05)** | **5 of 18** clear the arithmetic: `F6` reg S2 (+0.086, t 1.37), `F6` clf S2 (+0.123, t 1.64), `F8 N=15` reg S1 (+0.085, t 1.84), `F8 N=15` reg S2 (**+0.151, t 2.42, 64% weeks green, +2.75 R/week**), `F8 N=15` clf S1 (+0.067, t 1.43). |
| **the transparent baseline** | **0 of its 6 cells clears G2.** PLAN §3 H4 makes this a hard ship requirement — a black box the rule cannot approximate is not deployable in this engine. **No ship candidate exists.** |
| **Nagel reversed tape** | 2 of the 5 survivors **FAIL on VAL**: `F8 N=15` reg S1 makes **+0.065 (t 1.42)** and clf S1 **+0.090 (t 2.22)** with the sign of every target flipped. On TEST the reversed tape is positive for **every** F8 cell — up to **+0.187 R, t 4.39, 14 of 14 weeks green** (`F8 N=15` clf S1 rev). The model's ranking is *inverted* out of sample. |
| **tail** | **All five VAL survivors go negative once the top 5% of their booked trades are removed** (−0.034 to −0.073 R). |
| **G3 (TEST, read once after the rule was frozen in `PREREG.md` §0.9)** | **All five are negative**: `F6` reg S2 −0.138 (t −2.79, 21% weeks green), `F6` clf S2 −0.099, `F8 N=15` reg S1 −0.095 (t −2.20), `F8 N=15` reg S2 −0.020, `F8 N=15` clf S1 −0.068. Only **5 of 18** cells are positive on TEST at all (16 of 18 on VAL). |
| **decile calibration** | monotone on VAL (rho +0.38..+0.88 for the two HGB models) and **anti-monotone on TRAINPRED** (rho −0.37..−0.59); the rho sign flips between two consecutive out-of-sample periods for **7 of the 9** family x model pairs. |
| **search-adjusted permutation p on VAL** | **0.034** for the max weekly R over the 18 cells — but this permutation holds the fitted models FIXED and only randomises outcomes, so it charges the *selection* search and not the *model-fitting* search. The one full-pipeline shuffled-target draw (which does charge the fit) reached max weekly R **1.77** and max t **3.36** on VAL against the real tape's **2.75** and **2.42**. |

**Two findings that are worth more than the null itself.**

1. **The single best VAL number in the whole stage is the unselected book.** `F6 {}` FCFS on VAL is +0.101 R, t 1.84 —
   *higher* than four of the five "selected" survivors. Every F6 selection cell on VAL is inside ±0.05 R of doing
   nothing. The apparent F6 edge on VAL is a property of the January-2026 tape, not of any feature: `F6` reg S2 earns
   **+26.9 R of its +31.4 R VAL total in 2026-01 alone**, and `F6` clf S2 **+30.3 R of +35.4 R**.
2. **The reversed-tape gate earns its place.** It is the only test in the battery that flagged `F8 N=15` S1 *before*
   TEST was opened, and TEST then confirmed it in the harshest possible way (the flipped tape is a 14-of-14-green,
   t 4.39 "book"). PLAN's adoption of Nagel (2025) was correct: on ~30K rows with a 33-feature booster, what the model
   learns from a short window is the recent sign of the population, and it keeps building the same object when the
   tape is reversed.

**Power / phrasing (PLAN §1).** In THIS universe (the point-in-time >=5%-range day population with the causal floor),
for THESE three families, at entries >= 10:00, under THIS book (12/day, 4 concurrent), THIS corrected cost contract
and THIS 33-feature causal set, **no walk-forward feature model produced a selection that survived TRAINPRED, VAL,
the reversed-tape gate, the tail test and TEST together**. The smallest per-trade effect the VAL tests could have seen
at 80% power is **0.088 to 0.209 R** depending on the cell (MDE column), i.e. roughly **1.7 to 3.2 R per week** at
4 slots; on TEST the MDEs run **0.105 to 0.511 R** (the F6 cells' TEST SE is inflated by a single outlier week). Effects below those sizes are invisible here and are NOT excluded. What IS
excluded, at this power, is the size of edge the two live books' selection stacks historically carried (+0.2 to +0.4 R
per trade, `probe_design.md` item 4) — an effect that large would have been visible in every period and is not there.

**What D1 must change (the pilot's real output).** The machinery works and is reusable as written; the inputs are the
weak part. D1 on `candidates4.csv` gets: (i) the resting fill as a second target column — 30% more signals, and
`candidates3` is conditioned on cheap fills so the *fastest continuations are missing from D0's population entirely*;
(ii) the signal-bar OHLCV and the level's pre-signal history (prior touches, minutes in consolidation), which are the
features the ORB rulebook says carry the edge and which D0 simply does not have; (iii) **premarket dollar volume** —
the ORB news gate is `PM$ > cut AND has_news`, and D0 could only test the news half; alone it is worth nothing here
(permutation importance ~0, mean rank 16-19 of 33, univariate |delta| <= 0.04 R and sign-flipping by period);
(iv) prev-day range % and range-size %, the two ORB vetoes; (v) per-trade NBBO instead of a band median.
Until those exist, the honest statement is that **D0 tested whether the day-context + tape-shape features on disk can
rank these candidates, and they cannot**.
# Stage D0 — the feature-selection PILOT

Executed 2026-09-16 per `research/fuckup_audit/PLAN.md` §3 H4 and §4 row D. This is the **pilot**: it runs the H4
machinery on the features that are ALREADY on disk (`candidates3.csv` + `day_features.csv` + `etf_1min.db` + a fresh
Alpaca news pull), so that Stage D1 can re-run the identical pipeline on `candidates4.csv` (both fills, signal-bar
OHLCV, level history) the moment Stage B finishes. Everything written by this stage is under
`research/fuckup_audit/D/`; everything outside it was read only (`data/cache.db` and `etf_1min.db` via
`file:...?mode=ro`).

---

## 0. PRE-REGISTRATION (written before any model was fitted)

### 0.1 Candidate table

Stage A's population (`A/acore.py`, reused verbatim for the cost contract and the book):
`price >= 5`, `entry_m <= 841`, `r_pct >= 1`, `range_so_far_pct >= 5` on bars strictly before the signal (the causal
universe floor), **plus** Stage A's adopted window `entry_m >= 600` (10:00 ET) and Stage A's adopted base families:

| key | mechanism | TRAIN mean net R, contract (c), `>=10:00`, hold (Stage A `A/a2_cells.csv`) |
|---|---|---|
| `F6 {}` | red-to-green: opened below the prior close, first cross back above it; stop = the day's low so far | **+0.036** (t 1.31, 19.2/wk) |
| `F8 {"N": 30}` | break of the 09:30-10:00 opening-range high; stop = that range's low | **+0.012** (t 0.49, 21.2/wk) |
| `F8 {"N": 15}` | break of the 09:30-09:45 opening-range high; stop = that range's low | **-0.032** (t -1.17, 23.1/wk) |

**Exit, chosen now, one per family: `hold` (flat at 15:55) for all three.** Reason, from Stage A's `>=10:00` TRAIN
rows under contract (c): F6 hold +0.036 vs 2r +0.027; F8 N=30 hold +0.012 vs 2r +0.005; F8 N=15 hold -0.032 vs 2r
-0.035. `hold` is weakly better in all three, and it is ONE exit spec across the three families, which keeps the cell
count at the declared number instead of doubling it. The `2r` exit is not scored in D0.

Rows: 90,376 (F6 8,533 · F8 N=15 27,793 · F8 N=30 54,050) over 420 days, 61,902 distinct symbol-days.

**Target** = net R under Stage A's adopted contract (c), computed with `A/acore.py`'s own constants:
`half_c = 0.5 * spread_pct_c / max(r_pct, 0.05)` with `spread_pct_c` the measured median signal-minute NBBO of THIS
population per (price band x time band) from `research/lit_review_2026/cost_curve.csv`;
`net = rr_hold - 0.25*half_c - half_c*{stop: 0.875, eod: 0.412, target: 0.875}[why]`.
Secondary target for the classifier: `win = (net > 0)`.

**Splits** (PLAN §1, fixed): TRAIN 2025-01-02..2025-12-31 · VAL 2026-01-01..2026-05-31 · TEST 2026-06-01..2026-09-11.

**Book**: `trading.hod_break.run_book(rows, 12, 4)` — unchanged. A selection rule is a FILTER applied to a day's
candidates before `run_book`; the book itself is never touched.

### 0.2 Features — the exact list, each with its live source

All 33 are causal at the **signal minute** `sig_m` (the bar whose high reached the level; the engine decides at its
close and the fill is the next bar's open). Three columns of `candidates3.csv` — `price`, `r_pct`, `dist_open_pct` —
are built from the FILL price and are therefore known one minute late; they are **not** used as features. Causal twins
are rebuilt from `level` and `stop`.

| # | feature | definition at `sig_m` | live source |
|---|---|---|---|
| 1 | `f_price_sig` | `level` | the running level the engine is watching |
| 2 | `f_log_price` | `log10(level)` | same |
| 3 | `f_r_pct_sig` | `(level - stop)/level * 100` | `level` and the consolidation/range low the engine computes |
| 4 | `f_dist_open_sig` | `(level / open_0930 - 1) * 100` | scanner's TRUE 09:30 open (`realtime_scanner`) |
| 5 | `f_range_so_far` | `(running high - running low)/open_0930 * 100`, bars strictly before `sig_m` | engine's own bar array |
| 6 | `f_rv_adv` | cumulative RTH volume through `sig_m` / ADV20 | engine cum volume + `load_adv20_from_daily_bars` |
| 7 | `f_gap_pct` | `(open_0930 / prev_close - 1) * 100` | scanner open + `daily_bars` prev close |
| 8 | `f_log_adv20` | `log10(ADV20)` | `daily_bars` |
| 9 | `f_mins_since_open` | `sig_m - 570` | the clock |
| 10 | `f_dow` | weekday 0-4 | the calendar |
| 11 | `f_spread_cc_bps` | measured median NBBO spread, bps, for (price band of `level` x time band of `sig_m`) | live: the real NBBO at the signal minute (Stage B makes it per-trade); here the `cost_curve.csv` table |
| 12 | `f_sp_over_r` | `f_spread_cc_bps/1e4 / (f_r_pct_sig/100)` — the live `spread <= 15% of R` gate as a continuous feature | same |
| 13 | `f_pb` | price-band index 0-4 | the quote |
| 14-18 | `f_spy_gap`, `f_spy_prev_ret`, `f_spy_vs_sma5`, `f_spy_vs_sma20`, `f_spy_vol20` | SPY day context, all known at 09:30 | `data/cache.db daily_bars` (the live BF regime path already fetches these) |
| 19-23 | `f_iwm_gap`, `f_iwm_prev_ret`, `f_iwm_vs_sma5`, `f_iwm_vs_sma20`, `f_iwm_vol20` | IWM day context, known at 09:30 | same |
| 24 | `f_regime` | `trading/regime_helpers.build_regime_lookup` A/B/C1/C2 -> 0/1/2/3 | the live BF classifier, already in production |
| 25 | `f_iwm_ret` | IWM close of bar `sig_m` / IWM 09:30 open - 1, % | an IWM 1-min bar subscription |
| 26 | `f_iwm_sign` | `1{f_iwm_ret >= 0}` | same |
| 27 | `f_spy_ret` | SPY close of bar `sig_m` / SPY 09:30 open - 1, % | a SPY 1-min bar subscription |
| 28 | `f_spy_sign` | `1{f_spy_ret >= 0}` | same |
| 29 | `f_breadth` | **APPROXIMATION, labelled as one** — share of that day's scoring-population candidates that signalled at an EARLIER minute with `dist_open_pct > 0` (A3's definition, >= 5 earlier candidates required). It is breadth inside the study's own candidate population, not the market's | live analogue: the scanner's count of watched names above their 09:30 open |
| 30 | `f_n_pop_before` | count of the day's scoring-population candidates that signalled strictly earlier | the engine's own signal counter |
| 31 | `f_n_fam_before` | count of THIS family's candidates that signalled strictly earlier that day | the engine's own signal counter |
| 32 | `f_news_pre` | Alpaca articles tagging the symbol in [prev calendar day 15:00 ET, 09:30 ET) | `AlpacaClient.get_premarket_news_multi` — the ORB book's live call |
| 33 | `f_news_intraday` | Alpaca articles tagging the symbol in [09:30 ET, `sig_m`) — strictly before the signal minute | the same call re-issued intraday |

Features 32-33 are included **only if >= 80% of the table is covered** by the time the model is trained; otherwise the
coverage is reported and they are left to D1. Missing values are NOT imputed: `HistGradientBoosting*` handles NaN
natively, and the transparent baseline treats a NaN as "no decile".

**Causality assertions in code** (`d0_features.py`): `sig_m < entry_m` on every row; the window rule `entry_m >= 600`;
every index / breadth / news column indexed at `sig_m` or earlier; the entry-derived columns are excluded from the
feature list by construction (they do not carry the `f_` prefix).

### 0.3 Model — ONE config, no tuning

```
HistGradientBoostingRegressor  (target: net R)
HistGradientBoostingClassifier (target: net R > 0)
max_depth=4, max_iter=200, learning_rate=0.05, min_samples_leaf=200, l2_regularization=1.0, random_state=0
```
**Walk-forward**: first training window 2025-01-02..2025-09-30 -> predict 2025-10; then a monthly refit on ALL data up
to the month before, through 2026-09. Months before 2025-10 receive no prediction and are reported as such. Each family
is fitted separately (three models, never pooled).

**TRANSPARENT baseline** (the thing that would actually be shipped, and PLAN §3 H4's ship gate): on TRAIN rows only,
each numeric feature is cut into deciles (edges from TRAIN); the decile means of net R are computed; monotonicity is
`rho` = Spearman(decile index, decile mean) and the effect is `spread = mean(top 3 deciles) - mean(bottom 3 deciles)`.
The **3 features with the largest |spread| among those with |rho| >= 0.6** are chosen, their direction fixed to
`sign(rho)`, and the score is the sum of their decile ranks (`d` if rho > 0 else `9 - d`). Frozen once; never refit.

> **Amendment made before any model was fitted (logged, not retro-fitted):** the baseline's deciles are fitted on the
> FIRST TRAINING WINDOW ONLY (2025-01-02..2025-09-30), not on all of TRAIN. Fitting on all of TRAIN would make the
> baseline in-sample on the TRAINPRED months 2025-10..12 while the walk-forward models are out-of-sample there, and the
> two would not be comparable. With this amendment every number in the report is out-of-sample for all three models.
> Features with fewer than 10 distinct values on that window (`f_dow`, `f_pb`, `f_regime`, `f_iwm_sign`, `f_spy_sign`)
> cannot form deciles and are therefore ineligible for the baseline; they remain inputs to the two HGB models. The
> S2 threshold for the baseline is likewise the median score over that same window.

### 0.4 Selection rules and books

| rule | definition | live-implementable? |
|---|---|---|
| **S1 top-12** | per day, rank that family's candidates by the predicted value, keep the top 12, then `run_book(12, 4)` | **NO** — ranking a whole day needs the day's later candidates. Declared as a DIAGNOSTIC ceiling, not a ship rule |
| **S2 threshold** | keep rows with predicted net R > 0 (regressor) / P(win) > 0.5 (classifier) / baseline score >= the TRAIN median score, then `run_book(12, 4)` | **YES** — a per-row gate followed by the first-come book, exactly what the engine does |
| control **FCFS** | no selection: `run_book(12, 4)` on every candidate (Stage A's number) | yes |
| control **RAND12** | per day, 12 candidates drawn uniformly without replacement, then `run_book(12, 4)`; mean over 20 seeds | — |

### 0.5 Evaluation

Per evaluation period — **TRAINPRED** = the predicted TRAIN months 2025-10..12 (13 weeks), **VAL** = 2026-01..05
(22 weeks), **TEST** = 2026-06..2026-09-11 (14 weeks) — report n, trades/week, mean net R, SE, t, MDE (2.8*SE), WR,
weekly R, % weeks green, worst week, exit mix. Plus: decile calibration of the predictions on VAL (Spearman rho of
decile index vs realised mean; must be monotone); permutation importance on VAL and its rank stability across the
monthly refits; the tail test (top 1% and top 5% of booked trades removed, winners capped at +3R); a per-month table.

**Gates (PLAN §1)**: G1 on TRAINPRED (mean > 0, t >= 2.0, >= 5 trades/week) — **a WEAK gate here, declared as such**:
only 3 months / 13 weeks of the TRAIN span are predicted at all, so G1's power is roughly a third of Stage A's. The
real bar is **G2 on VAL** (mean > 0, t >= 1.0, >= 55% weeks green; bar raised by 1 SE of weekly R per 10 cells passing
G1) and **G3 on TEST, read ONCE after the selection rule is frozen in writing and VAL has been evaluated**. Economic
bar, reported not gated: >= 3R/week at 4 slots.

**The Nagel reversed-tape gate (mandatory, PLAN §3 H4)**: the identical pipeline is re-run with the sign of every
target flipped (`net -> -net`, `win -> 1 - win`) and the selected book is scored on that flipped tape. If the selected
book is also "profitable" there (mean > 0 AND t > 1 on VAL), the model is fitting population momentum rather than a
feature edge and **the result FAILS**, whatever the real-tape number says.
**Shuffled-target null**: targets permuted WITHIN each day, pipeline re-run, the selected rows scored on the TRUE
targets. Reported as the null distribution of the selection machinery.

### 0.6 Cells declared in advance

| block | cells |
|---|---|
| main | 3 families x 3 models (HGB regressor, HGB classifier, transparent baseline) x 2 selections (S1, S2) = **18** |
| controls | FCFS (3) and RAND12 (3) — not search cells; they are the benchmark the 18 are measured against |
| gate runs | reversed tape 18, shuffled tape 18 — diagnostics, not candidates |
| per-cell reads | each of the 18 is read on TRAINPRED and VAL; TEST is read only after the VAL table is written and the rule is frozen |
| **declared total** | **18 main + 6 control + 36 gate = 60**, on top of the program's 52 (score4) + Stage A's 192 + probe_stops' ~100 + probe_days' 156 |

### 0.7 What would make me say a feature model selects a positive book

A family x model x selection cell whose **VAL** mean net R > 0 with t >= 1.0 and >= 55% of weeks green, whose VAL
decile calibration is monotone, whose **reversed-tape twin is NOT profitable**, and — for a ship candidate — the
**transparent baseline** clears the same bar (PLAN §3 H4's explicit decision rule: a black box the rule cannot
approximate is not deployable in this engine). If none exists, the report gives the closest miss per family with its
own MDE, and PLAN §1's phrasing rule applies: never "no edge exists", always "no edge detectable in THIS universe /
horizon / book / window / cost, and the smallest effect this test could have seen was X".

### 0.8 Parity anchor (independent-check rule, coding-error half)

Before any model was fitted, the unselected book of this table was compared cell for cell with Stage A's
`A/a2_cells.csv` (`>=10:00`, `hold`, contract (c)): **6 of 6 cells, max |delta n| = 0, max |delta mean R| = 0.0004**
(Stage A rounds to 3 dp). `D/d0_parity.csv`. Weeks per split: TRAIN 53 / VAL 22 / TEST 14 — identical to Stage A.

---

## 0.9 RULE FROZEN — written after TRAINPRED and VAL were evaluated, BEFORE TEST was read

VAL (`D/cells_val.csv`) and TRAINPRED (`D/cells_trainpred.csv`) are on disk. TEST has not been opened.

* **G1 (TRAINPRED): 0 of 18 cells pass.** Largest t is 1.16 (`F8 {"N":30}` regressor S1, mean +0.054). Because no cell
  passed G1, G2's bar is NOT raised (the pre-registered raise is 1 SE of weekly R per 10 G1 passers).
* **G2 (VAL): 5 of 18 cells clear the arithmetic** — `F6 {}` reg S2, `F6 {}` clf S2, `F8 {"N":15}` reg S1,
  `F8 {"N":15}` reg S2, `F8 {"N":15}` clf S1.
* **No ship candidate exists under §0.7**, and this is frozen now: (a) the **transparent baseline clears G2 in 0 of its
  6 cells**, which §0.7 makes a hard requirement; (b) two of the five survivors — `F8 {"N":15}` reg S1 and
  `F8 {"N":15}` clf S1 — **FAIL the Nagel reversed-tape gate** (their reversed twins make +0.065 t 1.42 and +0.090
  t 2.22 on VAL); (c) every one of the five has a **negative mean once the top 5% of its booked trades are removed**;
  (d) the decile calibration of the same models is **anti-monotone on TRAINPRED** (rho −0.37 to −0.59) and monotone on
  VAL, i.e. it does not hold its sign across two consecutive out-of-sample periods.
* **TEST is therefore read ONCE now for completeness and reported whatever it says. It cannot rescue a candidate and no
  candidate will be proposed from it.** The frozen selection rules are exactly those in §0.4 with the models of §0.3;
  nothing is re-fitted, re-thresholded or re-chosen after this line.
## 1. The 18 declared cells + the two controls, real tape

`meanR` = mean net R per booked trade under Stage A contract (c); `mde` = 2.8*SE, the smallest per-trade effect the
cell could have seen at 80% power; `wkR` = mean weekly R at 4 slots; `green` = share of weeks positive;
`worst` = worst week in R. RAND12 `se` is the sd across its 20 seeds, not a t-test SE.

### TRAINPRED (predicted TRAIN months 2025-10..12, 14 weeks)

```
         key  model sel   n  tpw   meanR     se     t    mde   WR   wkR  green  worst
       F6 {}   FCFS   - 272 19.4  0.0018 0.0511  0.04 0.1431 44.5  0.04   0.50   -5.9
       F6 {} RAND12   - 269 19.2  0.0079 0.0214   NaN    NaN 45.8  0.15   0.53   -6.0
       F6 {}    reg  S1 266 19.0  0.0322 0.0503  0.64 0.1408 46.6  0.61   0.64   -5.6
       F6 {}    reg  S2 184 13.1  0.0606 0.0628  0.97 0.1758 46.2  0.80   0.57   -5.6
       F6 {}    clf  S1 265 18.9 -0.0028 0.0480 -0.06 0.1344 46.8 -0.05   0.50   -6.2
       F6 {}    clf  S2 143 10.2 -0.0347 0.0606 -0.57 0.1698 45.5 -0.35   0.50   -6.7
       F6 {}   base  S1 268 19.1 -0.0304 0.0465 -0.65 0.1302 45.5 -0.58   0.50   -6.9
       F6 {}   base  S2 272 19.4 -0.0027 0.0505 -0.05 0.1413 44.9 -0.05   0.50   -7.0
F8 {"N": 30}   FCFS   - 306 21.9 -0.0758 0.0457 -1.66 0.1280 45.1 -1.66   0.29   -9.6
F8 {"N": 30} RAND12   - 288 20.6 -0.0138 0.0387   NaN    NaN 45.4 -0.29   0.47   -7.9
F8 {"N": 30}    reg  S1 272 19.4  0.0537 0.0463  1.16 0.1297 48.2  1.04   0.64   -4.6
F8 {"N": 30}    reg  S2 242 17.3 -0.0212 0.0485 -0.44 0.1358 45.9 -0.37   0.43   -9.1
F8 {"N": 30}    clf  S1 277 19.8  0.0011 0.0500  0.02 0.1399 45.5  0.02   0.36   -6.3
F8 {"N": 30}    clf  S2 177 12.6 -0.0607 0.0517 -1.17 0.1449 45.8 -0.77   0.21   -6.7
F8 {"N": 30}   base  S1 268 19.1 -0.0724 0.0497 -1.46 0.1391 41.0 -1.39   0.36   -8.1
F8 {"N": 30}   base  S2 165 11.8 -0.1107 0.0560 -1.98 0.1569 41.8 -1.30   0.50   -9.6
F8 {"N": 15}   FCFS   - 325 23.2 -0.0515 0.0560 -0.92 0.1568 40.3 -1.20   0.36  -13.8
F8 {"N": 15} RAND12   - 307 22.0 -0.0651 0.0548   NaN    NaN 41.9 -1.44   0.42  -10.0
F8 {"N": 15}    reg  S1 300 21.4  0.0238 0.0581  0.41 0.1627 47.3  0.51   0.64   -5.1
F8 {"N": 15}    reg  S2 293 20.9 -0.1110 0.0522 -2.13 0.1461 39.2 -2.32   0.21  -11.9
F8 {"N": 15}    clf  S1 293 20.9 -0.0032 0.0507 -0.06 0.1421 46.1 -0.07   0.36  -11.4
F8 {"N": 15}    clf  S2 171 12.2  0.0393 0.0686  0.57 0.1922 48.5  0.48   0.43   -2.1
F8 {"N": 15}   base  S1 287 20.5 -0.0499 0.0466 -1.07 0.1305 43.6 -1.02   0.57   -8.5
F8 {"N": 15}   base  S2 306 21.9 -0.1091 0.0491 -2.22 0.1376 38.2 -2.39   0.29   -7.7
```

### VAL (2026-01..05, 22 weeks)

```
         key  model sel   n  tpw   meanR     se     t    mde   WR   wkR  green  worst
       F6 {}   FCFS   - 436 19.8  0.1011 0.0550  1.84 0.1541 50.7  2.00   0.45   -4.3
       F6 {} RAND12   - 433 19.7  0.0531 0.0229   NaN    NaN 49.0  1.05   0.49   -4.7
       F6 {}    reg  S1 427 19.4  0.1022 0.0565  1.81 0.1582 49.6  1.98   0.50   -4.4
       F6 {}    reg  S2 367 16.7  0.0857 0.0625  1.37 0.1749 47.4  1.43   0.55   -4.6
       F6 {}    clf  S1 427 19.4  0.0760 0.0555  1.37 0.1554 49.4  1.48   0.50   -4.9
       F6 {}    clf  S2 289 13.1  0.1227 0.0747  1.64 0.2093 51.6  1.61   0.55   -3.9
       F6 {}   base  S1 434 19.7  0.0466 0.0355  1.31 0.0994 49.1  0.92   0.50   -5.2
       F6 {}   base  S2 436 19.8  0.0470 0.0373  1.26 0.1043 49.8  0.93   0.50   -4.2
F8 {"N": 30}   FCFS   - 456 20.7 -0.0064 0.0336 -0.19 0.0940 48.7 -0.13   0.45   -6.8
F8 {"N": 30} RAND12   - 438 19.9  0.0339 0.0486   NaN    NaN 49.0  0.68   0.53   -7.0
F8 {"N": 30}    reg  S1 432 19.6  0.0101 0.0319  0.32 0.0892 46.3  0.20   0.64   -5.4
F8 {"N": 30}    reg  S2 377 17.1 -0.0051 0.0362 -0.14 0.1014 48.3 -0.09   0.64   -7.9
F8 {"N": 30}    clf  S1 420 19.1  0.0276 0.0313  0.88 0.0877 49.3  0.53   0.64   -4.5
F8 {"N": 30}    clf  S2 267 12.1  0.0313 0.0406  0.77 0.1138 49.8  0.38   0.59   -3.9
F8 {"N": 30}   base  S1 424 19.3 -0.0363 0.0341 -1.07 0.0955 48.1 -0.70   0.36   -7.7
F8 {"N": 30}   base  S2 226 10.3  0.0021 0.0510  0.04 0.1428 49.6  0.02   0.45   -4.9
F8 {"N": 15}   FCFS   - 468 21.3  0.0441 0.0389  1.13 0.1089 47.4  0.94   0.64   -6.0
F8 {"N": 15} RAND12   - 462 21.0  0.0153 0.0404   NaN    NaN 47.4  0.32   0.51   -8.2
F8 {"N": 15}    reg  S1 452 20.5  0.0852 0.0463  1.84 0.1296 49.8  1.75   0.55   -5.5
F8 {"N": 15}    reg  S2 401 18.2  0.1508 0.0622  2.42 0.1742 50.1  2.75   0.64   -4.6
F8 {"N": 15}    clf  S1 441 20.0  0.0673 0.0472  1.43 0.1322 49.4  1.35   0.59   -8.2
F8 {"N": 15}    clf  S2 224 10.2  0.0706 0.0513  1.38 0.1435 50.0  0.72   0.50   -3.0
F8 {"N": 15}   base  S1 442 20.1  0.0141 0.0465  0.30 0.1301 46.8  0.28   0.45   -9.8
F8 {"N": 15}   base  S2 453 20.6  0.0150 0.0352  0.42 0.0986 47.7  0.31   0.45   -6.0
```

### TEST (2026-06..2026-09-11, 14 weeks) — READ ONCE

```
         key  model sel   n  tpw   meanR     se     t    mde   WR   wkR  green  worst
       F6 {}   FCFS   - 299 21.4  0.1068 0.1655  0.65 0.4635 45.5  2.28   0.29  -10.7
       F6 {} RAND12   - 297 21.3  0.0571 0.0914   NaN    NaN 42.2  1.21   0.30   -8.3
       F6 {}    reg  S1 296 21.1  0.1512 0.1825  0.83 0.5109 42.2  3.20   0.29   -9.4
       F6 {}    reg  S2 230 16.4 -0.1378 0.0495 -2.79 0.1385 43.5 -2.26   0.21   -7.7
       F6 {}    clf  S1 298 21.3  0.0973 0.1670  0.58 0.4676 43.3  2.07   0.43  -11.3
       F6 {}    clf  S2 141 10.1 -0.0992 0.0656 -1.51 0.1837 44.0 -1.00   0.36   -7.2
       F6 {}   base  S1 302 21.6  0.0593 0.1612  0.37 0.4512 43.4  1.28   0.14  -10.2
       F6 {}   base  S2 297 21.2  0.1359 0.1674  0.81 0.4686 45.5  2.88   0.50   -9.4
F8 {"N": 30}   FCFS   - 296 21.1 -0.0408 0.0372 -1.10 0.1041 45.9 -0.86   0.29   -7.2
F8 {"N": 30} RAND12   - 295 21.1 -0.0531 0.0348   NaN    NaN 44.4 -1.12   0.35   -6.8
F8 {"N": 30}    reg  S1 301 21.5 -0.0260 0.0448 -0.58 0.1255 42.2 -0.56   0.50   -8.4
F8 {"N": 30}    reg  S2 251 17.9 -0.0479 0.0433 -1.10 0.1213 42.2 -0.86   0.43   -6.6
F8 {"N": 30}    clf  S1 281 20.1 -0.0140 0.0376 -0.37 0.1054 44.1 -0.28   0.43   -5.3
F8 {"N": 30}    clf  S2 124  8.9 -0.0053 0.0547 -0.10 0.1531 47.6 -0.05   0.57   -5.9
F8 {"N": 30}   base  S1 286 20.4 -0.0960 0.0420 -2.29 0.1175 40.6 -1.96   0.29   -5.3
F8 {"N": 30}   base  S2 180 12.9  0.0024 0.0505  0.05 0.1414 50.0  0.03   0.64   -6.0
F8 {"N": 15}   FCFS   - 322 23.0 -0.0609 0.0434 -1.40 0.1215 44.1 -1.40   0.36   -6.4
F8 {"N": 15} RAND12   - 313 22.4 -0.0938 0.0341   NaN    NaN 41.6 -2.10   0.28   -8.9
F8 {"N": 15}    reg  S1 315 22.5 -0.0946 0.0431 -2.20 0.1206 42.2 -2.13   0.14   -9.5
F8 {"N": 15}    reg  S2 284 20.3 -0.0198 0.0504 -0.39 0.1410 46.1 -0.40   0.43  -10.2
F8 {"N": 15}    clf  S1 301 21.5 -0.0675 0.0423 -1.60 0.1183 41.5 -1.45   0.36   -9.8
F8 {"N": 15}    clf  S2 190 13.6 -0.0431 0.0527 -0.82 0.1476 46.3 -0.58   0.43   -4.9
F8 {"N": 15}   base  S1 298 21.3 -0.0448 0.0457 -0.98 0.1279 44.0 -0.95   0.36   -6.5
F8 {"N": 15}   base  S2 317 22.6 -0.0904 0.0453 -2.00 0.1267 41.6 -2.05   0.07   -6.4
```

## 2. Selection minus the unselected book (FCFS), per period

```
                        meanR_trainpred  meanR_val  meanR_test  t_trainpred  t_val  t_test  vs_fcfs_trainpred  vs_fcfs_val  vs_fcfs_test
key          model sel                                                                                                                  
F6 {}        base  S1           -0.0304     0.0466      0.0593        -0.65   1.31    0.37            -0.0322      -0.0545       -0.0475
                   S2           -0.0027     0.0470      0.1359        -0.05   1.26    0.81            -0.0045      -0.0541        0.0291
             clf   S1           -0.0028     0.0760      0.0973        -0.06   1.37    0.58            -0.0046      -0.0251       -0.0095
                   S2           -0.0347     0.1227     -0.0992        -0.57   1.64   -1.51            -0.0365       0.0216       -0.2060
             reg   S1            0.0322     0.1022      0.1512         0.64   1.81    0.83             0.0304       0.0011        0.0444
                   S2            0.0606     0.0857     -0.1378         0.97   1.37   -2.79             0.0588      -0.0154       -0.2446
F8 {"N": 15} base  S1           -0.0499     0.0141     -0.0448        -1.07   0.30   -0.98             0.0016      -0.0300        0.0161
                   S2           -0.1091     0.0150     -0.0904        -2.22   0.42   -2.00            -0.0576      -0.0291       -0.0295
             clf   S1           -0.0032     0.0673     -0.0675        -0.06   1.43   -1.60             0.0483       0.0232       -0.0066
                   S2            0.0393     0.0706     -0.0431         0.57   1.38   -0.82             0.0908       0.0265        0.0178
             reg   S1            0.0238     0.0852     -0.0946         0.41   1.84   -2.20             0.0753       0.0411       -0.0337
                   S2           -0.1110     0.1508     -0.0198        -2.13   2.42   -0.39            -0.0595       0.1067        0.0411
F8 {"N": 30} base  S1           -0.0724    -0.0363     -0.0960        -1.46  -1.07   -2.29             0.0034      -0.0299       -0.0552
                   S2           -0.1107     0.0021      0.0024        -1.98   0.04    0.05            -0.0349       0.0085        0.0432
             clf   S1            0.0011     0.0276     -0.0140         0.02   0.88   -0.37             0.0769       0.0340        0.0268
                   S2           -0.0607     0.0313     -0.0053        -1.17   0.77   -0.10             0.0151       0.0377        0.0355
             reg   S1            0.0537     0.0101     -0.0260         1.16   0.32   -0.58             0.1295       0.0165        0.0148
                   S2           -0.0212    -0.0051     -0.0479        -0.44  -0.14   -1.10             0.0546       0.0013       -0.0071
```

mean of (selected - FCFS) over the 18 cells: test -0.0206, trainpred +0.0197, val +0.0044

share of the 18 cells with mean net R > 0: test 0.28, trainpred 0.33, val 0.89

## 3. The Nagel reversed-tape gate and the shuffled-target null

Same pipeline, sign of every target flipped (`rev`) / targets permuted within day and the selected rows scored on
the TRUE outcome (`shuf`). A cell FAILS the gate if its reversed twin is profitable on VAL (mean > 0 AND t > 1).

### TRAINPRED (predicted TRAIN months 2025-10..12, 14 weeks)
```
         key model sel tape   n   meanR     t   wkR  green
       F6 {}   reg  S1  rev 266  0.0040  0.08  0.08   0.50
       F6 {}   reg  S2  rev 176 -0.0037 -0.06 -0.05   0.57
       F6 {}   reg  S1 shuf 269  0.0015  0.03  0.03   0.50
       F6 {}   reg  S2 shuf 189  0.0002  0.00  0.00   0.43
       F6 {}   clf  S1  rev 266 -0.0091 -0.18 -0.17   0.43
       F6 {}   clf  S2  rev 217 -0.0447 -0.79 -0.69   0.50
       F6 {}   clf  S1 shuf 265 -0.0250 -0.53 -0.47   0.64
       F6 {}   clf  S2 shuf 134  0.0122  0.20  0.12   0.50
       F6 {}  base  S1  rev 270  0.0106  0.21  0.20   0.43
       F6 {}  base  S2  rev  64  0.0107  0.11  0.05   0.43
       F6 {}  base  S1 shuf 261  0.0008  0.02  0.02   0.57
       F6 {}  base  S2 shuf 234  0.0447  0.98  0.75   0.57
F8 {"N": 30}   reg  S1  rev 295 -0.0059 -0.10 -0.13   0.71
F8 {"N": 30}   reg  S2  rev 296  0.1032  2.13  2.18   0.57
F8 {"N": 30}   reg  S1 shuf 290 -0.1131 -2.38 -2.34   0.21
F8 {"N": 30}   reg  S2 shuf 205 -0.1110 -2.39 -1.63   0.21
F8 {"N": 30}   clf  S1  rev 291  0.0938  1.99  1.95   0.64
F8 {"N": 30}   clf  S2  rev 296  0.0620  1.29  1.31   0.57
F8 {"N": 30}   clf  S1 shuf 288 -0.1028 -2.32 -2.11   0.21
F8 {"N": 30}   clf  S2 shuf 175 -0.0614 -1.15 -0.77   0.36
F8 {"N": 30}  base  S1  rev 273  0.0213  0.41  0.41   0.57
F8 {"N": 30}  base  S2  rev 192 -0.0418 -0.59 -0.57   0.50
F8 {"N": 30}  base  S1 shuf 265 -0.1643 -3.76 -3.11   0.07
F8 {"N": 30}  base  S2 shuf 148 -0.1449 -2.52 -1.53   0.36
F8 {"N": 15}   reg  S1  rev 314  0.1113  1.84  2.50   0.64
F8 {"N": 15}   reg  S2  rev 287  0.0402  0.68  0.82   0.57
F8 {"N": 15}   reg  S1 shuf 292  0.1277  1.55  2.66   0.57
F8 {"N": 15}   reg  S2 shuf 201 -0.0855 -1.30 -1.23   0.36
F8 {"N": 15}   clf  S1  rev 308  0.1010  1.80  2.22   0.57
F8 {"N": 15}   clf  S2  rev 310  0.0588  1.01  1.30   0.64
F8 {"N": 15}   clf  S1 shuf 297 -0.0218 -0.39 -0.46   0.43
F8 {"N": 15}   clf  S2 shuf 133 -0.1476 -1.68 -1.40   0.29
F8 {"N": 15}  base  S1  rev 316 -0.0127 -0.18 -0.29   0.57
F8 {"N": 15}  base  S2  rev 299 -0.0365 -0.48 -0.78   0.43
F8 {"N": 15}  base  S1 shuf 286 -0.0371 -0.74 -0.76   0.43
F8 {"N": 15}  base  S2 shuf 297 -0.0889 -1.70 -1.89   0.29
```

### VAL (2026-01..05, 22 weeks)
```
         key model sel tape   n   meanR     t   wkR  green
       F6 {}   reg  S1  rev 428 -0.0124 -0.38 -0.24   0.59
       F6 {}   reg  S2  rev 319 -0.0845 -2.09 -1.23   0.36
       F6 {}   reg  S1 shuf 434  0.0694  1.28  1.37   0.55
       F6 {}   reg  S2 shuf 369  0.1053  1.72  1.77   0.50
       F6 {}   clf  S1  rev 430 -0.0503 -1.45 -0.98   0.45
       F6 {}   clf  S2  rev 373 -0.0773 -2.02 -1.31   0.32
       F6 {}   clf  S1 shuf 430  0.0195  0.55  0.38   0.41
       F6 {}   clf  S2 shuf 285  0.0192  0.45  0.25   0.50
       F6 {}  base  S1  rev 434 -0.1055 -1.91 -2.08   0.55
       F6 {}  base  S2  rev 159 -0.1513 -1.22 -1.09   0.55
       F6 {}  base  S1 shuf 422  0.0114  0.35  0.22   0.50
       F6 {}  base  S2 shuf 404  0.0747  2.40  1.37   0.64
F8 {"N": 30}   reg  S1  rev 449 -0.0138 -0.36 -0.28   0.36
F8 {"N": 30}   reg  S2  rev 425  0.0078  0.22  0.15   0.45
F8 {"N": 30}   reg  S1 shuf 427  0.0698  1.96  1.35   0.68
F8 {"N": 30}   reg  S2 shuf 359  0.0489  1.19  0.80   0.64
F8 {"N": 30}   clf  S1  rev 445  0.0708  1.64  1.43   0.68
F8 {"N": 30}   clf  S2  rev 442  0.0264  0.77  0.53   0.50
F8 {"N": 30}   clf  S1 shuf 441 -0.0524 -1.53 -1.05   0.36
F8 {"N": 30}   clf  S2 shuf 219  0.1670  3.36  1.66   0.68
F8 {"N": 30}  base  S1  rev 424 -0.0764 -2.25 -1.47   0.32
F8 {"N": 30}  base  S2  rev 302  0.0004  0.01  0.01   0.55
F8 {"N": 30}  base  S1 shuf 420 -0.1146 -3.61 -2.19   0.27
F8 {"N": 30}  base  S2 shuf 204  0.0416  0.79  0.39   0.64
F8 {"N": 15}   reg  S1  rev 484  0.0651  1.42  1.43   0.59
F8 {"N": 15}   reg  S2  rev 462 -0.0200 -0.49 -0.42   0.45
F8 {"N": 15}   reg  S1 shuf 446 -0.0318 -0.89 -0.65   0.45
F8 {"N": 15}   reg  S2 shuf 306  0.0327  0.67  0.46   0.55
F8 {"N": 15}   clf  S1  rev 472  0.0898  2.22  1.93   0.68
F8 {"N": 15}   clf  S2  rev 449 -0.0664 -1.65 -1.36   0.32
F8 {"N": 15}   clf  S1 shuf 454  0.0263  0.63  0.54   0.59
F8 {"N": 15}   clf  S2 shuf 185  0.1124  1.64  0.95   0.64
F8 {"N": 15}  base  S1  rev 472 -0.0625 -1.43 -1.34   0.32
F8 {"N": 15}  base  S2  rev 450 -0.0516 -1.15 -1.06   0.36
F8 {"N": 15}  base  S1 shuf 446  0.0186  0.41  0.38   0.50
F8 {"N": 15}  base  S2 shuf 448  0.0277  0.77  0.56   0.50
```

### TEST (2026-06..2026-09-11, 14 weeks) — READ ONCE
```
         key model sel tape   n   meanR     t   wkR  green
       F6 {}   reg  S1  rev 295 -0.0911 -0.54 -1.92   0.71
       F6 {}   reg  S2  rev 212 -0.3058 -1.22 -4.63   0.57
       F6 {}   reg  S1 shuf 299  0.0481  0.29  1.03   0.21
       F6 {}   reg  S2 shuf 202 -0.2436 -4.81 -3.51   0.00
       F6 {}   clf  S1  rev 297 -0.1301 -0.72 -2.76   0.71
       F6 {}   clf  S2  rev 254 -0.1743 -0.90 -3.16   0.64
       F6 {}   clf  S1 shuf 297  0.0621  0.37  1.32   0.29
       F6 {}   clf  S2 shuf 122 -0.1191 -2.13 -1.04   0.29
       F6 {}  base  S1  rev 299 -0.1223 -0.68 -2.61   0.71
       F6 {}  base  S2  rev  90  0.2105  3.53  1.35   0.79
       F6 {}  base  S1 shuf 289 -0.1585 -4.24 -3.27   0.00
       F6 {}  base  S2 shuf 259 -0.0032 -0.08 -0.06   0.50
F8 {"N": 30}   reg  S1  rev 299  0.1368  3.63  2.92   0.86
F8 {"N": 30}   reg  S2  rev 281  0.0751  1.77  1.51   0.71
F8 {"N": 30}   reg  S1 shuf 293 -0.0366 -0.87 -0.77   0.36
F8 {"N": 30}   reg  S2 shuf 204  0.0231  0.53  0.34   0.57
F8 {"N": 30}   clf  S1  rev 294  0.1100  2.63  2.31   0.79
F8 {"N": 30}   clf  S2  rev 295  0.0620  1.66  1.31   0.71
F8 {"N": 30}   clf  S1 shuf 287 -0.0851 -2.36 -1.75   0.14
F8 {"N": 30}   clf  S2 shuf 119  0.0403  0.66  0.34   0.50
F8 {"N": 30}  base  S1  rev 291  0.0485  1.16  1.01   0.57
F8 {"N": 30}  base  S2  rev 163  0.0568  1.20  0.66   0.57
F8 {"N": 30}  base  S1 shuf 279 -0.1773 -5.13 -3.53   0.14
F8 {"N": 30}  base  S2 shuf 158 -0.0094 -0.18 -0.11   0.50
F8 {"N": 15}   reg  S1  rev 310  0.1494  3.35  3.31   0.93
F8 {"N": 15}   reg  S2  rev 324  0.1264  2.84  2.93   0.86
F8 {"N": 15}   reg  S1 shuf 293 -0.0897 -2.05 -1.88   0.36
F8 {"N": 15}   reg  S2 shuf 209 -0.0742 -1.43 -1.11   0.36
F8 {"N": 15}   clf  S1  rev 320  0.1868  4.39  4.27   1.00
F8 {"N": 15}   clf  S2  rev 315  0.0503  1.14  1.13   0.57
F8 {"N": 15}   clf  S1 shuf 305 -0.1852 -4.46 -4.03   0.21
F8 {"N": 15}   clf  S2 shuf  87  0.0097  0.05  0.06   0.21
F8 {"N": 15}  base  S1  rev 325  0.1233  2.57  2.86   0.71
F8 {"N": 15}  base  S2  rev 317  0.1013  2.23  2.29   0.86
F8 {"N": 15}  base  S1 shuf 302 -0.0719 -1.60 -1.55   0.21
F8 {"N": 15}  base  S2 shuf 315 -0.0934 -2.05 -2.10   0.14
```

## 4. Decile calibration of the real-tape prediction (whole candidate pool, not the book)

### TRAINPRED (predicted TRAIN months 2025-10..12, 14 weeks)
```
decile                  0      1      2      3      4      5      6      7      8      9
key          model                                                                      
F6 {}        base  -0.148  0.004  0.009 -0.048 -0.063  0.216  0.114  0.156 -0.047  0.095
             clf    0.163  0.209 -0.009  0.015  0.028  0.017  0.076  0.017  0.051 -0.039
             reg    0.025  0.127  0.122  0.012  0.099  0.075 -0.029  0.021 -0.028  0.103
F8 {"N": 15} base  -0.064  0.005 -0.018 -0.047  0.004  0.010  0.013 -0.018 -0.004  0.016
             clf    0.030  0.029  0.006  0.016 -0.011 -0.102  0.035 -0.049  0.006 -0.034
             reg    0.020 -0.001 -0.021 -0.009  0.026 -0.017 -0.023 -0.002 -0.020 -0.026
F8 {"N": 30} base   0.187 -0.123 -0.060  0.152  0.067 -0.052 -0.076 -0.119  0.022 -0.000
             clf    0.100  0.015  0.017  0.021 -0.086  0.013 -0.068 -0.041 -0.072  0.071
             reg   -0.027  0.075  0.085  0.006 -0.044 -0.025 -0.067 -0.048 -0.048  0.062

Spearman rho(decile, realised mean net R):
key           model
F6 {}         base     0.515
              clf     -0.418
              reg     -0.370
F8 {"N": 15}  base     0.515
              clf     -0.467
              reg     -0.588
F8 {"N": 30}  base    -0.152
              clf     -0.382
              reg     -0.430
```

### VAL (2026-01..05, 22 weeks)
```
decile                  0      1      2      3      4      5      6      7      8      9
key          model                                                                      
F6 {}        base   0.052  0.051  0.026  0.017  0.054 -0.015  0.078 -0.005  0.010  0.095
             clf   -0.080  0.020  0.147  0.008 -0.001 -0.004  0.048  0.056  0.063  0.133
             reg   -0.060  0.005  0.037  0.101  0.011 -0.045 -0.025  0.054  0.089  0.224
F8 {"N": 15} base   0.002  0.024  0.019  0.022 -0.017 -0.022 -0.030 -0.049  0.066  0.043
             clf   -0.178 -0.010  0.006  0.004  0.007  0.086  0.062  0.042  0.052  0.026
             reg   -0.154 -0.026  0.027 -0.016 -0.012  0.003  0.030  0.133  0.068  0.043
F8 {"N": 30} base  -0.002 -0.003  0.068 -0.001 -0.056  0.016  0.009 -0.017  0.088  0.024
             clf   -0.112 -0.055 -0.011  0.051  0.041  0.042 -0.000  0.034  0.021  0.115
             reg   -0.078 -0.054 -0.030  0.065  0.067  0.031  0.029  0.049  0.023  0.024

Spearman rho(decile, realised mean net R):
key           model
F6 {}         base    -0.006
              clf      0.503
              reg      0.564
F8 {"N": 15}  base     0.055
              clf      0.758
              reg      0.879
F8 {"N": 30}  base     0.358
              clf      0.624
              reg      0.382
```

### TEST (2026-06..2026-09-11, 14 weeks) — READ ONCE
```
decile                  0      1      2      3      4      5      6      7      8      9
key          model                                                                      
F6 {}        base  -0.040 -0.205 -0.169 -0.113  0.547  0.088 -0.064 -0.120 -0.026  0.044
             clf   -0.147 -0.006  0.126 -0.030  0.004 -0.149  0.062  0.178 -0.008 -0.260
             reg   -0.018 -0.063 -0.084  0.299  0.067 -0.108 -0.114 -0.023 -0.093 -0.095
F8 {"N": 15} base  -0.204 -0.147 -0.199 -0.088 -0.073 -0.033  0.006  0.040  0.037 -0.049
             clf   -0.098 -0.042 -0.039 -0.056 -0.128 -0.100 -0.074 -0.069  0.050 -0.059
             reg   -0.099 -0.017 -0.048 -0.039 -0.124 -0.112 -0.110 -0.060 -0.010  0.003
F8 {"N": 30} base  -0.175 -0.034 -0.092 -0.016  0.048 -0.058 -0.167 -0.065  0.051    NaN
             clf    0.014 -0.048 -0.013 -0.049 -0.080 -0.108 -0.067 -0.023 -0.034 -0.050
             reg   -0.003 -0.063 -0.068 -0.082 -0.063 -0.040 -0.013 -0.018  0.005 -0.114

Spearman rho(decile, realised mean net R):
key           model
F6 {}         base     0.406
              clf     -0.079
              reg     -0.503
F8 {"N": 15}  base     0.855
              clf      0.079
              reg      0.273
F8 {"N": 30}  base     0.367
              clf     -0.358
              reg      0.018
```

## 5. Tail dependence of the booked cells (real tape)

`cut1` / `cut5` = mean net R with the top 1% / top 5% of the cell's booked trades removed; `cap3` = winners capped at +3R.

### TRAINPRED (predicted TRAIN months 2025-10..12, 14 weeks)
```
         key model sel   n   meanR  cut1_meanR  cut5_meanR  cap3_meanR   wkR  cut5_wkR  cap3_wkR
       F6 {}  FCFS   - 272  0.0018     -0.0280     -0.1095      0.0014  0.04     -2.02      0.03
       F6 {}   reg  S1 266  0.0322     -0.0005     -0.0819      0.0317  0.61     -1.47      0.60
       F6 {}   reg  S2 184  0.0606      0.0289     -0.0593      0.0601  0.80     -0.74      0.79
       F6 {}   clf  S1 265 -0.0028     -0.0332     -0.1104     -0.0032 -0.05     -1.98     -0.06
       F6 {}   clf  S2 143 -0.0347     -0.0661     -0.1399     -0.0347 -0.35     -1.35     -0.35
       F6 {}  base  S1 268 -0.0304     -0.0573     -0.1346     -0.0304 -0.58     -2.44     -0.58
       F6 {}  base  S2 272 -0.0027     -0.0326     -0.1131     -0.0031 -0.05     -2.09     -0.06
F8 {"N": 30}  FCFS   - 306 -0.0758     -0.1228     -0.1922     -0.0835 -1.66     -3.98     -1.83
F8 {"N": 30}   reg  S1 272  0.0537      0.0216     -0.0592      0.0535  1.04     -1.09      1.04
F8 {"N": 30}   reg  S2 242 -0.0212     -0.0608     -0.1361     -0.0237 -0.37     -2.23     -0.41
F8 {"N": 30}   clf  S1 277  0.0011     -0.0452     -0.1303     -0.0122  0.02     -2.45     -0.24
F8 {"N": 30}   clf  S2 177 -0.0607     -0.0910     -0.1539     -0.0607 -0.77     -1.85     -0.77
F8 {"N": 30}  base  S1 268 -0.0724     -0.1151     -0.1944     -0.0802 -1.39     -3.53     -1.54
F8 {"N": 30}  base  S2 165 -0.1107     -0.1427     -0.2096     -0.1107 -1.30     -2.34     -1.30
F8 {"N": 15}  FCFS   - 325 -0.0515     -0.1010     -0.2064     -0.0633 -1.20     -4.54     -1.47
F8 {"N": 15}   reg  S1 300  0.0238     -0.0248     -0.1327      0.0037  0.51     -2.70      0.08
F8 {"N": 15}   reg  S2 293 -0.1110     -0.1467     -0.2467     -0.1150 -2.32     -4.90     -2.41
F8 {"N": 15}   clf  S1 293 -0.0032     -0.0391     -0.1296     -0.0094 -0.07     -2.57     -0.20
F8 {"N": 15}   clf  S2 171  0.0393     -0.0064     -0.0992      0.0285  0.48     -1.15      0.35
F8 {"N": 15}  base  S1 287 -0.0499     -0.0786     -0.1565     -0.0508 -1.02     -3.04     -1.04
F8 {"N": 15}  base  S2 306 -0.1091     -0.1507     -0.2376     -0.1107 -2.39     -4.92     -2.42
```

### VAL (2026-01..05, 22 weeks)
```
         key model sel   n   meanR  cut1_meanR  cut5_meanR  cap3_meanR   wkR  cut5_wkR  cap3_wkR
       F6 {}  FCFS   - 436  0.1011      0.0262     -0.0465      0.0588  2.00     -0.87      1.17
       F6 {}   reg  S1 427  0.1022      0.0254     -0.0477      0.0591  1.98     -0.88      1.15
       F6 {}   reg  S2 367  0.0857      0.0140     -0.0667      0.0446  1.43     -1.05      0.74
       F6 {}   clf  S1 427  0.0760     -0.0007     -0.0734      0.0329  1.48     -1.35      0.64
       F6 {}   clf  S2 289  0.1227      0.0420     -0.0336      0.0705  1.61     -0.42      0.93
       F6 {}  base  S1 434  0.0466      0.0096     -0.0592      0.0390  0.92     -1.11      0.77
       F6 {}  base  S2 436  0.0470      0.0073     -0.0642      0.0386  0.93     -1.21      0.77
F8 {"N": 30}  FCFS   - 456 -0.0064     -0.0349     -0.1028     -0.0076 -0.13     -2.02     -0.16
F8 {"N": 30}   reg  S1 432  0.0101     -0.0142     -0.0752      0.0101  0.20     -1.40      0.20
F8 {"N": 30}   reg  S2 377 -0.0051     -0.0354     -0.0950     -0.0091 -0.09     -1.55     -0.16
F8 {"N": 30}   clf  S1 420  0.0276      0.0013     -0.0527      0.0265  0.53     -0.95      0.51
F8 {"N": 30}   clf  S2 267  0.0313      0.0039     -0.0583      0.0310  0.38     -0.67      0.38
F8 {"N": 30}  base  S1 424 -0.0363     -0.0647     -0.1298     -0.0365 -0.70     -2.37     -0.70
F8 {"N": 30}  base  S2 226  0.0021     -0.0355     -0.1117     -0.0004  0.02     -1.09     -0.00
F8 {"N": 15}  FCFS   - 468  0.0441      0.0025     -0.0735      0.0337  0.94     -1.48      0.72
F8 {"N": 15}   reg  S1 452  0.0852      0.0329     -0.0604      0.0636  1.75     -1.18      1.31
F8 {"N": 15}   reg  S2 401  0.1508      0.0509     -0.0356      0.0852  2.75     -0.61      1.55
F8 {"N": 15}   clf  S1 441  0.0673      0.0046     -0.0721      0.0381  1.35     -1.37      0.76
F8 {"N": 15}   clf  S2 224  0.0706      0.0368     -0.0330      0.0706  0.72     -0.32      0.72
F8 {"N": 15}  base  S1 442  0.0141     -0.0478     -0.1225     -0.0157  0.28     -2.33     -0.32
F8 {"N": 15}  base  S2 453  0.0150     -0.0141     -0.0842      0.0139  0.31     -1.65      0.29
```

### TEST (2026-06..2026-09-11, 14 weeks) — READ ONCE
```
         key model sel   n   meanR  cut1_meanR  cut5_meanR  cap3_meanR   wkR  cut5_wkR  cap3_wkR
       F6 {}  FCFS   - 299  0.1068     -0.0983     -0.1713     -0.0672  2.28     -3.47     -1.43
       F6 {}   reg  S1 296  0.1512     -0.1152     -0.2073     -0.0900  3.20     -4.16     -1.90
       F6 {}   reg  S2 230 -0.1378     -0.1818     -0.2474     -0.1437 -2.26     -3.85     -2.36
       F6 {}   clf  S1 298  0.0973     -0.1132     -0.1973     -0.0853  2.07     -3.99     -1.82
       F6 {}   clf  S2 141 -0.0992     -0.1537     -0.2157     -0.1162 -1.00     -2.05     -1.17
       F6 {}  base  S1 302  0.0593     -0.1356     -0.2036     -0.0967  1.28     -4.16     -2.09
       F6 {}  base  S2 297  0.1359     -0.0750     -0.1573     -0.0474  2.88     -3.17     -1.01
F8 {"N": 30}  FCFS   - 296 -0.0408     -0.0647     -0.1186     -0.0411 -0.86     -2.38     -0.87
F8 {"N": 30}   reg  S1 301 -0.0260     -0.0704     -0.1388     -0.0347 -0.56     -2.83     -0.75
F8 {"N": 30}   reg  S2 251 -0.0479     -0.0824     -0.1379     -0.0559 -0.86     -2.34     -1.00
F8 {"N": 30}   clf  S1 281 -0.0140     -0.0378     -0.0961     -0.0140 -0.28     -1.83     -0.28
F8 {"N": 30}   clf  S2 124 -0.0053     -0.0284     -0.0805     -0.0053 -0.05     -0.67     -0.05
F8 {"N": 30}  base  S1 286 -0.0960     -0.1298     -0.2015     -0.0980 -1.96     -3.90     -2.00
F8 {"N": 30}  base  S2 180  0.0024     -0.0236     -0.0763      0.0018  0.03     -0.93      0.02
F8 {"N": 15}  FCFS   - 322 -0.0609     -0.0938     -0.1598     -0.0610 -1.40     -3.48     -1.40
F8 {"N": 15}   reg  S1 315 -0.0946     -0.1288     -0.1918     -0.0952 -2.13     -4.10     -2.14
F8 {"N": 15}   reg  S2 284 -0.0198     -0.0578     -0.1410     -0.0270 -0.40     -2.71     -0.55
F8 {"N": 15}   clf  S1 301 -0.0675     -0.1016     -0.1655     -0.0686 -1.45     -3.37     -1.47
F8 {"N": 15}   clf  S2 190 -0.0431     -0.0697     -0.1406     -0.0432 -0.58     -1.81     -0.59
F8 {"N": 15}  base  S1 298 -0.0448     -0.0845     -0.1530     -0.0539 -0.95     -3.09     -1.15
F8 {"N": 15}  base  S2 317 -0.0904     -0.1332     -0.1984     -0.0961 -2.05     -4.26     -2.18
```

## 6. Permutation importance on VAL and its stability across the monthly refits

### F6 {} — top 12 by mean VAL permutation importance (of 33 features)
```
                   mean_imp  std_imp  mean_rank  std_rank
feat                                                     
f_iwm_prev_ret       0.0101   0.0261    14.7500   14.7817
f_mins_since_open    0.0051   0.0051    10.0000    8.6520
f_rv_adv             0.0046   0.0132    12.8750   11.5812
f_dist_open_sig      0.0043   0.0039    11.7500    8.7137
f_iwm_gap            0.0029   0.0131    15.5000   13.5857
f_spy_vol20          0.0017   0.0021    17.7500   11.1066
f_range_so_far       0.0017   0.0048    16.3750    8.7004
f_spy_vs_sma20       0.0011   0.0044    23.2500    8.9243
f_dow                0.0010   0.0148    12.7500   10.7271
f_regime             0.0007   0.0035    16.8750    6.0282
f_iwm_vol20          0.0007   0.0015    14.5000   10.9805
f_spread_cc_bps      0.0003   0.0005    14.6875    3.5146
```

### F8 {"N": 30} — top 12 by mean VAL permutation importance (of 33 features)
```
                   mean_imp  std_imp  mean_rank  std_rank
feat                                                     
f_iwm_prev_ret       0.0162   0.0228     10.125   13.1305
f_iwm_gap            0.0043   0.0140     18.625   14.4018
f_spy_gap            0.0027   0.0022      8.000   10.1419
f_breadth            0.0024   0.0096     13.750   14.0178
f_mins_since_open    0.0020   0.0035     13.500   10.8891
f_iwm_vol20          0.0017   0.0057     18.875   11.7891
f_spy_vs_sma20       0.0009   0.0008     13.875    9.2186
f_iwm_vs_sma20       0.0008   0.0065      8.625    9.8262
f_r_pct_sig          0.0005   0.0012     16.000    9.0079
f_dist_open_sig      0.0001   0.0009     16.750   10.3060
f_sp_over_r          0.0001   0.0008     14.125    5.7925
f_news_intraday      0.0000   0.0001     16.875    4.0245
```

### F8 {"N": 15} — top 12 by mean VAL permutation importance (of 33 features)
```
                   mean_imp  std_imp  mean_rank  std_rank
feat                                                     
f_iwm_prev_ret       0.0071   0.0123     13.125   14.1970
f_iwm_vol20          0.0064   0.0114     14.250   12.9035
f_breadth            0.0035   0.0025      8.500    8.9602
f_rv_adv             0.0024   0.0047      7.625   10.3501
f_sp_over_r          0.0020   0.0009      7.000    4.7809
f_spy_gap            0.0017   0.0049     17.125   11.8254
f_r_pct_sig          0.0012   0.0023     17.750    9.4529
f_iwm_ret            0.0008   0.0009     15.875    7.2395
f_log_adv20          0.0007   0.0025     18.625   10.5957
f_mins_since_open    0.0006   0.0025     15.750    9.9821
f_spy_prev_ret       0.0006   0.0025     16.250    9.6917
f_iwm_vs_sma5        0.0006   0.0029     18.250   11.8533
```

## 7. The transparent baseline that was actually chosen

```
         key            feat    rho  spread  n_dec
       F6 {}     f_sp_over_r -0.673 -0.0901   10.0
       F6 {}       f_spy_gap  0.767  0.2252    9.0
       F6 {}       f_iwm_gap  0.617  0.2004    9.0
F8 {"N": 30}  f_spy_prev_ret -0.709 -0.2372   10.0
F8 {"N": 30}  f_iwm_prev_ret -0.673 -0.2370   10.0
F8 {"N": 30}       f_breadth  0.891  0.1932   10.0
F8 {"N": 15} f_dist_open_sig  0.758  0.0942   10.0
F8 {"N": 15}       f_breadth  0.915  0.2626   10.0

the same fit on the reversed tape (it must, and does, flip every direction):
         key            feat    rho  spread
       F6 {}     f_sp_over_r  0.673  0.0901
       F6 {}       f_spy_gap -0.767 -0.2252
       F6 {}       f_iwm_gap -0.617 -0.2004
F8 {"N": 30}  f_spy_prev_ret  0.709  0.2372
F8 {"N": 30}  f_iwm_prev_ret  0.673  0.2370
F8 {"N": 30}       f_breadth -0.891 -0.1932
F8 {"N": 15} f_dist_open_sig -0.758 -0.0942
F8 {"N": 15}       f_breadth -0.915 -0.2626
```

## 8. Per-month net R of the five cells that cleared G2 on VAL

```
F6 {}  reg  S2
          n    sumR  meanR
month                     
2025-10  79   0.459  0.006
2025-11  37   1.905  0.051
2025-12  68   8.793  0.129
2026-01  62  26.909  0.434
2026-02  73  -0.352 -0.005
2026-03  86  -4.284 -0.050
2026-04  71   8.746  0.123
2026-05  75   0.424  0.006
2026-06  90 -14.811 -0.165
2026-07  70  -8.395 -0.120
2026-08  55  -6.596 -0.120
2026-09  15  -1.883 -0.126

F6 {}  clf  S2
          n    sumR  meanR
month                     
2025-10  58  -2.113 -0.036
2025-11  30  -0.771 -0.026
2025-12  55  -2.080 -0.038
2026-01  57  30.349  0.532
2026-02  60   3.975  0.066
2026-03  64   1.813  0.028
2026-04  52   2.341  0.045
2026-05  56  -3.029 -0.054
2026-06  70 -15.107 -0.216
2026-07  28   5.424  0.194
2026-08  39  -6.892 -0.177
2026-09   4   2.583  0.646

F8 {"N": 15}  reg  S1
           n    sumR  meanR
month                      
2025-10  112  -3.259 -0.029
2025-11   90  10.657  0.118
2025-12   98  -0.259 -0.003
2026-01   87  28.271  0.325
2026-02   84  -1.261 -0.015
2026-03   99 -10.398 -0.105
2026-04   96  12.781  0.133
2026-05   86   9.123  0.106
2026-06   96  -3.290 -0.034
2026-07  104 -20.218 -0.194
2026-08   98  -6.647 -0.068
2026-09   17   0.349  0.021

F8 {"N": 15}  reg  S2
           n    sumR  meanR
month                      
2025-10  108 -10.672 -0.099
2025-11   80  -5.568 -0.070
2025-12  105 -16.288 -0.155
2026-01   61  22.045  0.361
2026-02   80   4.271  0.053
2026-03   90 -11.663 -0.130
2026-04   79  26.389  0.334
2026-05   91  19.419  0.213
2026-06   93  -3.976 -0.043
2026-07   91 -21.121 -0.232
2026-08   86  18.405  0.214
2026-09   14   1.070  0.076

F8 {"N": 15}  clf  S1
           n    sumR  meanR
month                      
2025-10  110  -9.516 -0.087
2025-11   85   4.798  0.056
2025-12   98   3.785  0.039
2026-01   86   6.379  0.074
2026-02   84  -1.608 -0.019
2026-03   96  -6.006 -0.063
2026-04   92  17.538  0.191
2026-05   83  13.373  0.161
2026-06   94 -18.839 -0.200
2026-07   98 -13.750 -0.140
2026-08   93  10.020  0.108
2026-09   16   2.254  0.141

```

## 9. Search-adjusted permutation p on VAL

```
 obs_max_wkR     p  ndraw  null_mean  null_p95
       2.748 0.034    500      1.317     2.601
```

---

## 10. What it means

**1. The raw-flat families are still raw-flat after selection.** Stage A left F6 at +0.036 R and F8 N=30 at +0.012 R
on TRAIN with entries >= 10:00, both inside their own MDE. D0 asked whether a model can find, inside those flat
populations, a subset that is not flat. Over 18 pre-registered cells, three out-of-sample periods and 33 causal
features, the mean gain of selection over doing nothing is **+0.020 R (TRAINPRED), +0.004 R (VAL), −0.021 R (TEST)**.
That is not a small positive edge; it is noise with a mean of zero.

**2. The one cell that looked like a book on VAL is the one the gates were built for.** `F8 {"N":15}` regressor S2 is
+0.151 R, t 2.42, 64% of weeks green, +2.75 R/week on VAL — a cell that under the program's OLD gate (TRAIN >= +10R/wk)
would never have been looked at, and under PLAN §1's gate reads as a candidate. It then: lost −0.111 R (t −2.13) on the
three predicted TRAIN months that came BEFORE VAL; went negative when the top 5% of its trades were removed
(−0.036 R); and printed −0.020 R on TEST. Its own model's decile calibration was anti-monotone (rho −0.59) on
TRAINPRED, monotone (+0.88) on VAL and half-monotone (+0.27) on TEST. One period in three is what a coin does.

**3. The reversed-tape gate is now load-bearing, not ceremonial.** Two of the five VAL survivors are "profitable" with
every target's sign flipped, and on TEST *every* F8 cell is profitable on the flipped tape (up to t 4.39, 14/14 weeks
green) while being negative on the real one. That is the Nagel (2025) signature exactly: a flexible learner on a short
window rebuilds the population's own recent direction and calls it a prediction. Any future stage that fits a model on
this data without running the flipped-tape twin is not measuring what it thinks it is measuring. **This gate should be
promoted from "H4's gate" to a standing requirement for every model-based claim in this program.**

**4. The features on disk carry the day, not the trade.** The only feature with any consistent permutation importance
is `f_iwm_prev_ret` — a DAY-level variable, the previous day's IWM return — and its mean importance rank across refits
is 10 to 15 of 33 with a standard deviation of 13 to 15, i.e. it is not stably ranked either. The transparent baseline,
which is forced to be legible, picked `f_spy_gap` / `f_iwm_gap` / `f_sp_over_r` for F6 and
`f_spy_prev_ret` / `f_iwm_prev_ret` / `f_breadth` for F8 N=30 — again day context, plus the cost ratio, plus the
breadth approximation. This is the same conclusion `probe_days.md` reached by a different route (day direction
separates the book, but nothing knowable at 09:30 reproduces it) and Stage A's A3 reached by a third (TRAIN->VAL sign
agreement 0.51 over 104 buckets). Three independent attempts, one answer: **the day-context features on disk are a
description of the outcome, not a predictor of it.**

**5. News presence, measured causally for the first time in this program, is worth nothing HERE — and that is not a
contradiction of the ORB result.** 61,902 symbol-days pulled fresh (100% fetch success, 17.8% have an article in the
[prev 15:00 ET, 09:30 ET) window). Permutation importance ~0.000 with mean rank 16-19 of 33 in all three families; the
univariate difference between has-news and no-news candidates is <= 0.04 R and flips sign between TRAIN, VAL and TEST.
The live ORB rule is **not** "news"; it is `PM$ volume > $5.82M AND has_news AND common stock` — a conjunction whose
dollar-volume leg does not exist in `candidates3.csv`. D0 tested one leg of a two-leg rule on a different population.
D1, on `candidates4.csv` with premarket dollar volume, is where that rule can actually be tested.

**6. What would have to be true for a feature book to exist here.** Given the MDEs (0.09-0.21 R per trade on VAL), a
selection stack worth the +0.2 to +0.4 R that the BF and ORB stacks historically carried would have shown up in every
period at t > 3. It did not. So either (a) the edge lives in features D0 does not have (the resting-fill signals, the
level's history, PM$ volume, per-trade spread, prev-day range) — which is exactly what Stage B is building; or (b) the
edge lives in a universe D0 cannot see (H6, the causal 09:30 universe, Stage E); or (c) it is not in this shape of
trade at all. D0 cannot distinguish (a) from (b) from (c), and it does not claim to.

## 11. Caveats that limit every number above

* **The population is conditioned on cheap fills.** `candidates3.csv` only contains signals whose next-bar open came
  back under the 0.6% cap; `probe_stops.md` §3 measured that the resting model books **+30% more trades**, and the
  extra ones are the fastest continuations. D0's candidate set is missing them by construction.
* **17.2% of rows have `entry_m − sig_m > 1`** (a missing minute on the tape); 177 rows have `sig_m < 599` even though
  `entry_m >= 600`. Their features are still causal (read at `sig_m`), but their fill is a stale one. Stage B's
  signal-bar OHLCV will let D1 drop or flag them.
* **The corrected spread is a per-(price band x hour band) MEDIAN** applied to every candidate (Stage A's own caveat),
  so `f_spread_cc_bps` and `f_sp_over_r` carry no within-cell dispersion — as features they are close to a
  price x time interaction, not a liquidity measurement.
* **`f_breadth` is an approximation** (breadth inside the study's own candidate population, each candidate's
  `dist_open_pct` measured at its own entry minute) and is labelled as one everywhere it appears.
* **S1 (top-12 per day) is not live-implementable** — ranking a day requires the day's later candidates. It is reported
  as a diagnostic ceiling. Every ship-relevant statement above rests on S2, which is a per-row gate plus the first-come
  book, exactly what the engine does.
* **TEST was read once**, after `PREREG.md` §0.9 froze the rule in writing. Nothing was re-fitted, re-thresholded or
  re-chosen afterwards. The TEST numbers appear here because the pre-registration says to report them whatever they
  say, not because a candidate was being rescued.
* The permutation p of 0.034 is conditional on the fitted models and therefore **understates** the search; the
  full-pipeline shuffled-target draw is the honest null and it reaches 65% of the observed statistic in one draw.

## 12. Cell count

| block | cells |
|---|---|
| D0 main | 3 families x 3 models x 2 selection rules = **18** |
| D0 controls | FCFS 3, RAND12 3 (20 seeds each) = 6 |
| D0 gate runs | reversed tape 18, shuffled tape 18 = 36 |
| D0 reads | each cell read on TRAINPRED and VAL; TEST read once after the freeze |
| **D0 declared total** | **60** |
| program to date | score4 52 + Stage A 192 + `probe_stops` ~100 + `probe_days` 156 + **D0 60** = **~560** |

The 18 main cells were declared in `PREREG.md` §0.6 before any model was fitted; the only amendment (the baseline's
fit window) was written into `PREREG.md` before the first fit and is flagged there.

## 13. Files, and how to re-run

| file | what |
|---|---|
| `D/PREREG.md` | the pre-registration (§0.1-0.8) and the frozen rule (§0.9, written before TEST was read) |
| `D/d0_table.py` -> `D/table.csv` | the candidate table: chunked extraction from `candidates3.csv`, 90,376 rows |
| `D/d0_news.py` -> `D/news_presence.csv` | the causal news pull. **COMPLETE**: 420/420 days, 61,902 symbol-days, `fetch_ok` 100%, 17.8% with premarket news. Resumable — it keeps `D/news_state.json` and appends per day, so re-running it continues where it stopped; delete both files to rebuild from scratch. ~28 min wall clock at `ulimit -v 1000000` (600000 is too low for the pandas import). |
| `D/d0_features.py` -> `D/feat.csv` | the 33 causal features + the contract-(c) target; the causality assertions live here |
| `D/d0_parity.py` -> `D/d0_parity.csv` | the parity anchor against Stage A's `a2_cells.csv` (6/6 cells, max abs delta n = 0) |
| `D/d0_model.py` -> `D/preds.csv`, `D/importance.csv`, `D/baseline_univariate.csv` | the walk-forward fits on the three tapes (~11 min) |
| `D/d0_eval.py trainpred val` / `d0_eval.py test` -> `D/cells_*.csv`, `D/calib_*.csv`, `D/booked_*.csv` | the books and the statistics; TEST is a separate invocation on purpose |
| `D/d0_perm.py` -> `D/perm_val*.csv` | the search-adjusted permutation on VAL |
| `D/d0_tables.py` -> `D/results.md` | every table in this report |
| `D/booked_{trainpred,val,test}.csv` | the per-trade selected books of all 18 cells, real tape |

**To re-run D0 end to end on `candidates4.csv` (Stage D1):** point `d0_table.py::SRC` at the new file, add the new
columns to `USE` and to `FEATS` in `d0_features.py` (both fills become two target columns), and run
`d0_table -> d0_features -> d0_parity -> d0_model -> d0_eval trainpred val -> d0_perm -> (freeze) -> d0_eval test`.
Nothing else changes; the gates, the tapes and the cell accounting are already written.
