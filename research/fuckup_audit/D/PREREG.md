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
